"""Orchestration: raw image -> complete analysis.

Chains the four services into one flow:
    preprocess -> classify -> GradCAM (top-N present findings) -> ground each
    present finding in literature -> assemble.

This is the backend spine the /analyze endpoint and the demo UI both call. It
holds no model logic of its own — it wires proven services together.

Heatmaps are generated only for the top-N present findings (by probability),
not all present findings: on a typical image many low-threshold findings flag,
and generating a heatmap for each is slow and unfocused. N is configurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.classifier import ChestXrayClassifier, Classifier
from app.services.gradcam import GradCAM, render_overlay_base64
from app.services.literature import Article, CachedPubMedRetriever, LiteratureRetriever
from app.services.preprocessing import ChestXrayPreprocessor, Preprocessor
from app.services.tracking import log_inference


@dataclass
class FindingAnalysis:
    """One present finding, with its optional heatmap and literature."""
    name: str
    probability: float
    threshold: float
    heatmap_base64: str | None = None      # PNG overlay, base64 (top-N only)
    articles: list[Article] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """The complete analysis for one uploaded image."""
    model_name: str
    findings: list[FindingAnalysis]         # present findings, prob-desc
    disclaimer: str

    @property
    def num_present(self) -> int:
        return len(self.findings)


class Orchestrator:
    """Wires preprocessing, classification, explainability, and literature."""

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        classifier: Classifier | None = None,
        explainer: GradCAM | None = None,
        retriever: LiteratureRetriever | None = None,
    ) -> None:
        self.preprocessor = preprocessor or ChestXrayPreprocessor()
        self.classifier = classifier or ChestXrayClassifier()
        self.explainer = explainer or GradCAM()
        self.retriever = retriever or CachedPubMedRetriever()

    def analyze(
        self,
        image_path: str,
        heatmap_top_n: int = 3,
        literature_top_n: int = 3,
        articles_per_finding: int = 2,
        image_id: str | None = None,
    ) -> AnalysisResult:
        # 1. preprocess -> tensor
        tensor = self.preprocessor.process(image_path)

        # 2. classify -> structured findings
        classification = self.classifier.predict(tensor)

        # Log this inference run to MLflow for provenance (Phase 2 tracking).
        # Wrapped so a tracking failure never breaks the analysis (as with
        # literature); tracking is observability, not core function.
        try:
            log_inference(classification, image_id=image_id or "orchestrator")
        except Exception:
            pass
        present = classification.present  # prob-desc already

        # 3. + 4. per present finding: heatmap and literature for the top-N
        # (the highest-probability findings). Lower-ranked findings are still
        # reported, but without the expensive heatmap/PubMed work — focused and
        # fast for the demo.
        analyses: list[FindingAnalysis] = []
        for rank, f in enumerate(present):
            heatmap_b64 = None
            if rank < heatmap_top_n:
                hm = self.explainer.heatmap(tensor, f.name)
                heatmap_b64 = render_overlay_base64(tensor, hm)

            articles = []
            if rank < literature_top_n:
                articles = self.retriever.retrieve(f.name, max_results=articles_per_finding)

            analyses.append(
                FindingAnalysis(
                    name=f.name,
                    probability=f.probability,
                    threshold=f.threshold,
                    heatmap_base64=heatmap_b64,
                    articles=articles,
                )
            )

        return AnalysisResult(
            model_name=classification.model_name,
            findings=analyses,
            disclaimer=(
                "MedVision AI is a research/educational prototype and is NOT a "
                "diagnostic tool. Outputs are not a substitute for evaluation by "
                "a qualified clinician."
            ),
        )
