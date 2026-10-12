"""Shared ONNX model: the exported TorchXRayVision DenseNet, run with
onnxruntime instead of torch.

Why this exists: `import torch` alone costs ~291 MB RSS, which pushed the
server over Render's 512 MB free limit before inference even ran. The model is
exported once (scripts/export_onnx.py) to a portable ONNX graph and run here
with onnxruntime, whose whole footprint — runtime + this model — is a fraction
of that. Predictions are identical to the torch model (verified at export:
max abs diff ~2e-7).

The graph has TWO outputs:
  probs    (1, 18)        — op-normalized probabilities, exactly as the torch
                            model returned them (sigmoid + op_norm baked in).
  featmap  (1, 1024, 7, 7) — the post-ReLU conv feature map, for CAM.

One instance is shared by the classifier and GradCAM (D-11 seam preserved) so
only a single copy of the graph and weights sits in memory.
"""
from __future__ import annotations

import os

import numpy as np

_ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_assets"))
ONNX_PATH = os.path.join(_ASSETS, "densenet_xrv.onnx")
NPZ_PATH = os.path.join(_ASSETS, "densenet_xrv.npz")


class OnnxXrayModel:
    """Loads the exported DenseNet once and runs it on a (1,1,224,224) array.

    Also carries the classifier weights and operating-point thresholds that the
    classifier and GradCAM need (loaded from the companion .npz).
    """

    def __init__(
        self,
        onnx_path: str = ONNX_PATH,
        npz_path: str = NPZ_PATH,
        weights_name: str = "densenet121-res224-all",
    ) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        # Single-threaded CPU: on a small shared instance extra threads don't
        # speed up one image and each adds memory. Mirrors the old
        # torch.set_num_threads(1).
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name

        data = np.load(npz_path, allow_pickle=True)
        self.weight: np.ndarray = data["weight"].astype(np.float32)        # (18, 1024)
        self.bias: np.ndarray = data["bias"].astype(np.float32)            # (18,)
        self.op_threshs: np.ndarray = data["op_threshs"].astype(np.float32)  # (18,)
        self.pathologies: list[str] = [str(p) for p in data["pathologies"].tolist()]
        self.weights_name = weights_name

    def run(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run one forward pass.

        `image` is a (1, 1, 224, 224) float array from the Preprocessor.
        Returns (probs (1, 18), featmap (1, 1024, 7, 7)).
        """
        x = np.ascontiguousarray(image, dtype=np.float32)
        probs, featmap = self.session.run(None, {self._input_name: x})
        return probs, featmap
