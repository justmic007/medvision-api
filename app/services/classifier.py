"""Classification: model-ready array -> structured findings.

Wraps the exported TorchXRayVision DenseNet (run via onnxruntime, no torch)
behind a pluggable interface (D-11) and turns its output into a structured,
per-finding result: probability + present/absent flag decided by the model's
*published* operating points (D-07), never hand-tuned.

Activation exposure (Phase 3 groundwork): the ONNX graph returns the conv
feature map alongside the probabilities, so the activations that produced a
prediction are available for explainability (GradCAM) without a second pass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.services._onnx_model import OnnxXrayModel


@dataclass
class Finding:
    """One pathology result."""
    name: str
    probability: float
    threshold: float
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
        """Only the findings flagged present, highest probability first."""
        return sorted(
            (f for f in self.findings if f.present),
            key=lambda f: -f.probability,
        )


class Classifier(ABC):
    """Turns a model-ready array into a ClassificationResult.

    The contract: given a (1, 1, H, W) array from a Preprocessor, return a
    ClassificationResult. Implementations own their model and label set.
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> ClassificationResult:
        ...


class ChestXrayClassifier(Classifier):
    """Chest X-ray classifier backed by the exported DenseNet (onnxruntime).

    Uses the model's published per-pathology operating points (`op_threshs`) to
    flag findings present/absent (D-07). Keeps the feature map for GradCAM.
    """

    def __init__(
        self,
        weights: str = "densenet121-res224-all",
        model: OnnxXrayModel | None = None,
    ) -> None:
        # Reuse a shared model instance when passed (see Orchestrator), so the
        # classifier and GradCAM don't each load the graph + weights.
        self._model = model if model is not None else OnnxXrayModel(weights_name=weights)
        self.model_name = weights
        self.pathologies: list[str] = list(self._model.pathologies)
        self._thresholds: np.ndarray = self._model.op_threshs
        self._activations: np.ndarray | None = None

    @property
    def model(self) -> OnnxXrayModel:
        # The shared ONNX model, reused by GradCAM to avoid a second load.
        return self._model

    def predict(self, image: np.ndarray) -> ClassificationResult:
        probs, featmap = self._model.run(image)   # (1, 18), (1, 1024, 7, 7)
        probs = probs[0]                           # (18,)
        self._activations = featmap

        findings = [
            Finding(
                name=name,
                probability=float(prob),
                threshold=float(thr),
                present=bool(prob > thr),
            )
            for name, prob, thr in zip(
                self.pathologies, probs.tolist(), self._thresholds.tolist()
            )
        ]

        return ClassificationResult(
            findings=findings,
            model_name=self.model_name,
            activations=self._activations,
        )
