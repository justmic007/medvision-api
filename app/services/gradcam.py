"""Explainability: CAM heatmaps over the X-ray.

Answers "where did the model look?" for a chosen finding. Given a preprocessed
image and a target pathology, it produces a heatmap (values in [0, 1], sized to
the input image) highlighting the regions that drove that finding.

Why CAM, not gradient GradCAM: the DenseNet head is global-average-pool -> a
single linear classifier. For that architecture GradCAM reduces *exactly* to
classic CAM — the gradient-weighted channel importances equal the classifier's
own weights for the target class. So the heatmap is computed directly from the
conv feature map and the classifier weights, with no backprop and no torch:

  1. Take the (1024, 7, 7) feature map from the forward pass.
  2. Weight each channel by that finding's classifier weight; sum -> 7x7.
  3. ReLU (keep positive contributions), upsample to input size, normalize.

This is numerically the same explanation the torch GradCAM produced (verified:
heatmap correlation 1.0, identical peak). The overlay is rendered with
matplotlib, exactly as the original — only the inference engine changed (D-11).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from app.services._onnx_model import OnnxXrayModel

TARGET_SIZE = 224


class Explainer(ABC):
    """Produces a heatmap for a target finding on a preprocessed image."""

    @abstractmethod
    def heatmap(self, image: np.ndarray, finding: str) -> np.ndarray:
        ...


class GradCAM(Explainer):
    """CAM for the exported DenseNet, computed from the ONNX feature map."""

    def __init__(
        self,
        weights: str = "densenet121-res224-all",
        model: OnnxXrayModel | None = None,
    ) -> None:
        # Reuse the shared model when passed (the classifier's), so we don't
        # load a second copy of the graph and weights.
        self._model = model if model is not None else OnnxXrayModel(weights_name=weights)
        self.pathologies: list[str] = list(self._model.pathologies)
        self._weights = weights

    def _finding_index(self, finding: str) -> int:
        if finding not in self.pathologies:
            raise ValueError(
                f"Unknown finding {finding!r}. "
                f"Must be one of the model's {len(self.pathologies)} pathologies."
            )
        return self.pathologies.index(finding)

    def heatmap(self, image: np.ndarray, finding: str) -> np.ndarray:
        """Return a (H, W) float heatmap in [0, 1] for `finding` on `image`.

        `image` is a (1, 1, H, W) preprocessed array from a Preprocessor.
        """
        idx = self._finding_index(finding)
        _probs, featmap = self._model.run(image)   # (1, 1024, 7, 7)

        act = featmap[0]                            # (1024, 7, 7)
        w = self._model.weight[idx]                 # (1024,)
        cam = np.tensordot(w, act, axes=([0], [0]))  # (7, 7)
        cam = np.maximum(cam, 0.0)                   # keep positive contributions

        # Upsample 7x7 -> input size via bilinear interpolation.
        size = image.shape[-2:]                      # (H, W)
        cam_im = Image.fromarray(cam.astype(np.float32), mode="F")
        cam_im = cam_im.resize((size[1], size[0]), Image.BILINEAR)
        cam = np.array(cam_im, dtype=np.float32)     # writable copy (PIL array is read-only)

        # Normalize to [0, 1] (guard against a flat map).
        cam -= cam.min()
        peak = cam.max()
        if peak > 0:
            cam /= peak
        return cam


def render_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    out_path: str,
    alpha: float = 0.4,
) -> str:
    """Render `heatmap` as a colored overlay on the X-ray and save a PNG.

    `image` is the (1, 1, H, W) preprocessed array; `heatmap` is the (H, W) map
    from GradCAM.heatmap(). Saves to out_path and returns it.

    Kept as a standalone function (not on the class) so rendering is decoupled
    from the CAM computation. matplotlib is imported lazily (only at render
    time) so it never costs memory on a request that doesn't draw a heatmap.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend; no display needed
    import matplotlib.pyplot as plt

    # The X-ray as a 2D grayscale array in display range.
    img = np.asarray(image)[0, 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    return out_path


def render_overlay_base64(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> str:
    """Render the overlay to an in-memory PNG and return base64 (no file).

    Same rendering as render_overlay(), base64-encoded for a JSON response or an
    <img src="data:..."> — used by the orchestrator / /analyze endpoint.
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.asarray(image)[0, 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
