"""Classification: model-ready tensor -> structured findings.

Wraps the TorchXRayVision DenseNet behind a pluggable interface (D-11) and turns
its raw output into a structured, per-finding result: probability + present/absent
flag decided by the model's *published* operating points (D-07), never
hand-tuned.

Activation exposure (Phase 3 groundwork): the classifier retains the feature
activations from each forward pass. GradCAM (Phase 3) must hook the same
convolutional features that produced the prediction; exposing them here means
Phase 3 does not have to re-run inference. See DECISIONS.md (Phase 2 note).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torchxrayvision as xrv


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
    # Activations from the forward pass that produced these findings, kept for
    # downstream explainability (GradCAM, Phase 3). Not serialized to the API.
    activations: torch.Tensor | None = field(default=None, repr=False)

    @property
    def present(self) -> list[Finding]:
        """Only the findings flagged present, highest probability first."""
        return sorted(
            (f for f in self.findings if f.present),
            key=lambda f: -f.probability,
        )


class Classifier(ABC):
    """Turns a model-ready tensor into a ClassificationResult.

    The contract: given a (1, 1, H, W) tensor from a Preprocessor, return a
    ClassificationResult. Implementations own their model and label set.
    """

    @abstractmethod
    def predict(self, tensor: torch.Tensor) -> ClassificationResult:
        ...


class ChestXrayClassifier(Classifier):
    """Chest X-ray classifier backed by the TorchXRayVision DenseNet.

    Uses the model's published per-pathology operating points (`op_threshs`) to
    flag findings present/absent (D-07). Captures feature activations for GradCAM.
    """

    def __init__(self, weights: str = "densenet121-res224-all") -> None:
        self._model = xrv.models.DenseNet(weights=weights)
        self._model.eval()
        self.model_name = weights
        self.pathologies: list[str] = list(self._model.pathologies)
        self._thresholds: torch.Tensor = self._model.op_threshs
        self._activations: torch.Tensor | None = None
        self._register_activation_hook()

    @property
    def model(self):
        # The underlying DenseNet, shared with GradCAM to avoid loading a
        # second copy of the weights (halves model memory).
        return self._model

    def _register_activation_hook(self) -> None:
        """Capture the output of the convolutional feature block on each pass.

        `features` is the DenseNet backbone (verified via named_children()); its
        output is the spatial feature map GradCAM needs.
        """
        def hook(_module, _inputs, output):
            self._activations = output.detach()

        self._model.features.register_forward_hook(hook)

    def predict(self, tensor: torch.Tensor) -> ClassificationResult:
        with torch.no_grad():
            probs = self._model(tensor)[0]  # (18,), probabilities in [0, 1]

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
