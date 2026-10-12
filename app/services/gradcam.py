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

This is numerically the same explanation the torch GradCAM produced, now
torch-free. (D-11) explainability remains defined by an interface; this is the
chest X-ray implementation.
"""
from __future__ import annotations

import base64
import io
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
        cam = np.array(cam_im, dtype=np.float32)  # writable copy (PIL array is read-only)

        # Normalize to [0, 1] (guard against a flat map).
        cam -= cam.min()
        peak = cam.max()
        if peak > 0:
            cam /= peak
        return cam


def _colorize_jet(heatmap: np.ndarray) -> np.ndarray:
    """Map a [0, 1] heatmap to RGB uint8 via matplotlib's exact 'jet' LUT
    (precomputed in _jet_lut.py) -- same colors as matplotlib, no import."""
    from app.services._jet_lut import JET_LUT

    idx = np.clip((np.asarray(heatmap, dtype=np.float32) * 255.0).round(), 0, 255).astype(np.int64)
    return JET_LUT[idx]


def _overlay_png_bytes(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    size: int = 512,
) -> bytes:
    """Alpha-blend a jet-colored heatmap over the grayscale X-ray; return PNG
    bytes. NumPy + Pillow only. Both layers are bilinear-upsampled to `size`."""
    img = np.asarray(image)[0, 0].astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    img = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)

    gray_im = Image.fromarray((img * 255.0).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    heat = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    heat_im = Image.fromarray((heat * 255.0).astype(np.uint8)).resize((size, size), Image.BILINEAR)

    gray_up = np.asarray(gray_im, dtype=np.float32)
    heat_up = np.asarray(heat_im, dtype=np.float32) / 255.0

    base_rgb = np.stack([gray_up, gray_up, gray_up], axis=-1)
    heat_rgb = _colorize_jet(heat_up).astype(np.float32)
    blended = np.clip((1.0 - alpha) * base_rgb + alpha * heat_rgb, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def render_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    out_path: str,
    alpha: float = 0.4,
) -> str:
    """Render `heatmap` as a colored overlay on the X-ray and save a PNG.

    `image` is the (1, 1, H, W) preprocessed array; `heatmap` is the (H, W) map
    from GradCAM.heatmap(). Saves to out_path and returns it."""
    with open(out_path, "wb") as f:
        f.write(_overlay_png_bytes(image, heatmap, alpha))
    return out_path


def render_overlay_base64(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> str:
    """Render the overlay to an in-memory PNG and return base64 (no file)."""
    return base64.b64encode(_overlay_png_bytes(image, heatmap, alpha)).decode("ascii")
