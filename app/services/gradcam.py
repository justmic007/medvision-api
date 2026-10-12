"""Explainability: GradCAM heatmaps over the X-ray.

Answers "where did the model look?" for a chosen finding. Given a preprocessed
image and a target pathology, it produces a heatmap (values in [0, 1], sized to
the input image) highlighting the regions that drove that finding.

How it works:
  1. Forward pass with gradients enabled; capture the `features` activations
     (1, 1024, 7, 7) and retain their gradient.
  2. Backpropagate from the target finding's score.
  3. Weight each of the 1024 channels by its mean gradient (channel importance),
     sum into a 7x7 map, ReLU (keep positive contributions only), normalize.
  4. Upsample the 7x7 map to the input size -> the heatmap.

Gradient capture uses `Tensor.retain_grad()` on the forward-captured activations
rather than a backward hook: torchxrayvision's model applies an in-place ReLU in
its `features2` step, which collides with backward hooks on `features`. Reading
`.grad` off a retained tensor sidesteps that entirely (verified on torch 2.2.2).

Design (D-11): explainability is defined by an interface; the chest X-ray
implementation is one concrete case. GradCAM manages its own gradient-enabled
forward pass, independent of the classifier's no_grad inference path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

TARGET_SIZE = 224


class Explainer(ABC):
    """Produces a heatmap for a target finding on a preprocessed image."""

    @abstractmethod
    def heatmap(self, tensor: torch.Tensor, finding: str) -> np.ndarray:
        ...


class GradCAM(Explainer):
    """GradCAM for the TorchXRayVision DenseNet.

    Hooks the `features` block's output, backpropagates a chosen finding, and
    builds a normalized heatmap sized to the input image.
    """

    def __init__(
        self,
        weights: str = "densenet121-res224-all",
        model=None,
    ) -> None:
        # Reuse an existing DenseNet when passed (the classifier's), so we
        # don't load a second full copy of the model into memory.
        self._model = model if model is not None else xrv.models.DenseNet(weights=weights)
        self._model.eval()
        self.pathologies: list[str] = list(self._model.pathologies)
        self._weights = weights

    def _finding_index(self, finding: str) -> int:
        if finding not in self.pathologies:
            raise ValueError(
                f"Unknown finding {finding!r}. "
                f"Must be one of the model's {len(self.pathologies)} pathologies."
            )
        return self.pathologies.index(finding)

    def heatmap(self, tensor: torch.Tensor, finding: str) -> np.ndarray:
        """Return a (H, W) float heatmap in [0, 1] for `finding` on `tensor`.

        `tensor` is a (1, 1, H, W) preprocessed image from a Preprocessor.
        """
        idx = self._finding_index(finding)

        captured: dict[str, torch.Tensor] = {}

        def fwd_hook(_m, _i, output):
            output.retain_grad()
            captured["act"] = output

        handle = self._model.features.register_forward_hook(fwd_hook)
        try:
            inp = tensor.clone().requires_grad_(True)
            score = self._model(inp)[0][idx]
            self._model.zero_grad()
            score.backward()

            act = captured["act"]          # (1, 1024, 7, 7)
            grad = act.grad                # (1, 1024, 7, 7)

            # Channel importance = mean gradient per channel -> (1, 1024, 1, 1)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            # Weighted sum over channels -> (1, 1, 7, 7)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)              # keep positive contributions

            # Upsample to input size, drop batch/channel dims -> (H, W)
            size = tensor.shape[-2:]
            cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()

            # Normalize to [0, 1] (guard against a flat map)
            cam -= cam.min()
            peak = cam.max()
            if peak > 0:
                cam /= peak
            return cam
        finally:
            handle.remove()


def _colorize_jet(heatmap: np.ndarray) -> np.ndarray:
    """Map a [0, 1] heatmap to RGB uint8 via matplotlib's exact 'jet' LUT
    (precomputed in _jet_lut.py) -- same colors as matplotlib, no import."""
    from app.services._jet_lut import JET_LUT

    idx = np.clip((np.asarray(heatmap, dtype=np.float32) * 255.0).round(), 0, 255).astype(np.int64)
    return JET_LUT[idx]


def _overlay_png_bytes(
    tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    size: int = 512,
) -> bytes:
    """Alpha-blend a jet-colored heatmap over the grayscale X-ray and return PNG
    bytes. NumPy + Pillow only (no matplotlib). Both layers are bilinear-upsampled
    to `size` so the result is as smooth as matplotlib's render. The heatmap VALUES
    are unchanged (straight from GradCAM); only the drawing method changed."""
    import io

    from PIL import Image

    img = tensor[0, 0].detach().cpu().numpy().astype(np.float32)
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
    tensor: torch.Tensor,
    heatmap: np.ndarray,
    out_path: str,
    alpha: float = 0.4,
) -> str:
    """Render `heatmap` as a colored overlay on the X-ray and save a PNG.

    `tensor` is the (1, 1, H, W) preprocessed image; `heatmap` is the (H, W) map
    from GradCAM.heatmap(). Saves to out_path and returns it. NumPy + Pillow with
    matplotlib's exact jet colormap (no matplotlib import) to keep memory low."""
    with open(out_path, "wb") as f:
        f.write(_overlay_png_bytes(tensor, heatmap, alpha))
    return out_path


def render_overlay_base64(
    tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> str:
    """Render the overlay to an in-memory PNG and return base64 (no file).

    Same output as render_overlay(), base64-encoded for an <img src="data:...">
    or JSON response. NumPy + Pillow, matplotlib's exact jet colors, no matplotlib."""
    import base64

    return base64.b64encode(_overlay_png_bytes(tensor, heatmap, alpha)).decode("ascii")
