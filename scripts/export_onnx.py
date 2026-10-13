import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

WEIGHTS = "densenet121-res224-all"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "ml_assets")
OPSET = 13


class ExportWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.features = m.features
        self.classifier = m.classifier
        self.apply_sigmoid = bool(getattr(m, "apply_sigmoid", False))
        ot = getattr(m, "op_threshs", None)
        self.has_op = ot is not None
        if self.has_op:
            self.register_buffer("op_threshs", ot.detach().clone().float())

    def _op_norm(self, out, thr):
        thr = thr.to(out.dtype)
        below = out / (thr * 2.0)
        above = 1.0 - ((1.0 - out) / ((1.0 - thr) * 2.0))
        return torch.where(out < thr, below, above)

    def forward(self, x):
        feats = self.features(x)
        A = torch.relu(feats)
        pooled = torch.flatten(F.adaptive_avg_pool2d(A, (1, 1)), 1)
        out = self.classifier(pooled)
        if self.apply_sigmoid:
            out = torch.sigmoid(out)
        if self.has_op:
            out = torch.sigmoid(out)
            out = self._op_norm(out, self.op_threshs)
        return out, A


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    m = xrv.models.DenseNet(weights=WEIGHTS).eval()
    print("pathologies:", list(m.pathologies))
    print("apply_sigmoid:", getattr(m, "apply_sigmoid", False))
    print("op_threshs set:", getattr(m, "op_threshs", None) is not None)

    wrapper = ExportWrapper(m).eval()
    x = torch.randn(1, 1, 224, 224)

    with torch.no_grad():
        ref_probs = m(x)
        w_probs, w_A = wrapper(x)

    probs_diff = (ref_probs - w_probs).abs().max().item()
    print("\n[wrapper vs original torch] max abs prob diff: %.3e" % probs_diff)

    onnx_path = os.path.join(OUT_DIR, "densenet_xrv.onnx")
    torch.onnx.export(
        wrapper, x, onnx_path,
        input_names=["input"], output_names=["probs", "featmap"],
        dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"},
                      "featmap": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True,
    )
    print("wrote", onnx_path, "(%.1f MB)" % (os.path.getsize(onnx_path) / 1e6))

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    o_probs, o_A = sess.run(None, {"input": x.numpy().astype(np.float32)})

    onnx_probs_diff = float(np.abs(o_probs - w_probs.numpy()).max())
    onnx_A_diff = float(np.abs(o_A - w_A.numpy()).max())
    print("[onnxruntime vs torch]   max abs prob diff: %.3e" % onnx_probs_diff)
    print("[onnxruntime vs torch]   max abs featmap diff: %.3e" % onnx_A_diff)
    print("featmap shape:", o_A.shape)

    W = m.classifier.weight.detach().numpy().astype(np.float32)
    b = m.classifier.bias.detach().numpy().astype(np.float32)
    npz_path = os.path.join(OUT_DIR, "densenet_xrv.npz")
    np.savez(npz_path, weight=W, bias=b,
             pathologies=np.array(list(m.pathologies)))
    print("wrote", npz_path, "(classifier W", W.shape, ")")

    top = int(o_probs[0].argmax())
    cam = np.tensordot(W[top], o_A[0], axes=([0], [0]))
    cam = np.maximum(cam, 0)
    print("\nCAM sanity (top class '%s'): shape %s, min %.3f, max %.3f" %
          (m.pathologies[top], cam.shape, cam.min(), cam.max()))

    meta = {"pathologies": list(m.pathologies),
            "apply_sigmoid": bool(getattr(m, "apply_sigmoid", False)),
            "input_shape": [1, 1, 224, 224], "opset": OPSET}
    with open(os.path.join(OUT_DIR, "densenet_xrv_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDONE. If both prob diffs are < 1e-4, ONNX == torch and we proceed.")


if __name__ == "__main__":
    main()
