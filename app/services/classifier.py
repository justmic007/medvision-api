"""Classification: model-ready array -> structured findings.

Wraps the exported TorchXRayVision DenseNet (run via onnxruntime, no torch)
behind a pluggable interface (D-11) and turns its output into a structured,
per-finding result: a confidence score + present/absent flag.

Decision rule (D-07, corrected): the model's output is *operating-point
normalized* — TorchXRayVision rescales each pathology's raw probability so that
that pathology's published operating point maps to exactly 0.5. So a finding is
"present" when its normalized confidence is at or above 0.5, uniformly across
all 18 pathologies. (The earlier code compared the normalized confidence against
the *raw* operating point, two different scales, which over-flagged findings the
model did not actually reach the operating point for.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.services._onnx_model import OnnxXrayModel

# The model output is normalized so each pathology's operating point is 0.5;
# present == confidence >= this, the same boundary for every pathology.
OPERATING_POINT = 0.5


@dataclass
class Finding:
    """One pathology result."""
    name: str
    probability: float          # operating-point-normalized confidence in [0, 1]
    threshold: float            # the present/absent boundary (OPERATING_POINT)
    present: bool


@dataclass
class ClassificationResult:
    """The full structured output for one image."""
    findings: list[Finding]
    model_name: str
    # Feature map from the forward pass that produced these findings, kept for
    # downstream explainability (GradCAM). Not serialized to the API.
    activations: np.ndarray | None = field(default=None, repr=False)

    @property
    def present(self) -> list[Finding]:
        """Only the findings flagged present, highest confidence first."""
        return sorted(
            (f for f in self.findings if f.present),
            key=lambda f: -f.probability,
        )


class Classifier(ABC):
    """Turns a model-ready array into a ClassificationResult."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> ClassificationResult:
        ...


class ChestXrayClassifier(Classifier):
    """Chest X-ray classifier backed by the exported DenseNet (onnxruntime)."""

    def __init__(
        self,
        weights: str = "densenet121-res224-all",
        model: OnnxXrayModel | None = None,
    ) -> None:
        self._model = model if model is not None else OnnxXrayModel(weights_name=weights)
        self.model_name = weights
        self.pathologies: list[str] = list(self._model.pathologies)
        self._activations: np.ndarray | None = None

    @property
    def model(self) -> OnnxXrayModel:
        return self._model

    def predict(self, image: np.ndarray) -> ClassificationResult:
        probs, featmap = self._model.run(image)   # (1, 18), (1, 1024, 7, 7)
        probs = probs[0]                           # (18,)
        self._activations = featmap

        findings = [
            Finding(
                name=name,
                probability=float(prob),
                threshold=OPERATING_POINT,
                present=bool(prob >= OPERATING_POINT),
            )
            for name, prob in zip(self.pathologies, probs.tolist())
        ]

        return ClassificationResult(
            findings=findings,
            model_name=self.model_name,
            activations=self._activations,
        )
