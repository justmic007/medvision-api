"""Image preprocessing: raw scan -> model-ready array.

Turns a chest X-ray (DICOM or PNG/JPG, per D-06) into the exact array the
exported DenseNet expects: shape (1, 1, 224, 224), pixel values in roughly
[-1024, 1024] (xrv's training convention).

Torch-free: this is a faithful NumPy/scikit-image reimplementation of the
torchxrayvision utilities it used to call (xrv.utils.load_image /
read_xray_dcm / normalize, xrv.datasets.XRayCenterCrop / XRayResizer). Verified
bit-identical to the xrv pipeline across grayscale, RGB and square inputs
(max abs diff 0.0), so the model sees exactly the same input as before.

Design (D-11): a model is fed by a *pluggable* preprocessor. `Preprocessor` is
the interface; `ChestXrayPreprocessor` is the one concrete implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import skimage.io
import skimage.transform

# The pretrained `*-res224-*` models expect 224x224 input.
TARGET_SIZE = 224


def _normalize(img: np.ndarray, maxval: int) -> np.ndarray:
    """Scale an image to roughly [-1024, 1024] (xrv.utils.normalize)."""
    if img.max() > maxval:
        raise ValueError(
            "max image value (%s) higher than expected bound (%s)."
            % (img.max(), maxval)
        )
    return (2 * (img.astype(np.float32) / maxval) - 1.0) * 1024


def _read_xray_dcm(path: str, voi_lut: bool = False, fix_monochrome: bool = True) -> np.ndarray:
    """Read a DICOM to a 2D array scaled to [-1024, 1024] (xrv.utils.read_xray_dcm)."""
    import pydicom

    ds = pydicom.dcmread(path, force=True)
    if ds.PhotometricInterpretation not in ("MONOCHROME1", "MONOCHROME2"):
        raise NotImplementedError(
            "PhotometricInterpretation `%s` is not yet supported."
            % ds.PhotometricInterpretation
        )
    max_possible = 2 ** ds.BitsStored - 1
    data = ds.pixel_array
    if voi_lut:
        try:
            from pydicom.pixels import apply_voi_lut
        except ImportError:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        data = apply_voi_lut(data, ds, index=0)
    if fix_monochrome and ds.PhotometricInterpretation == "MONOCHROME1":
        data = max_possible - data
    return _normalize(data, max_possible)


def _load_image(fname: str) -> np.ndarray:
    """Load PNG/JPG (or magic-byte DICOM) to a normalized (1, H, W) float array.

    Faithful to xrv.utils.load_image, including the first-channel (not mean)
    selection for color images and the DICM magic-byte sniff.
    """
    with open(fname, "rb") as f:
        header = f.read(132)
    if len(header) >= 132 and header[128:132] == b"DICM":
        return _read_xray_dcm(fname)[None, ...]

    img = skimage.io.imread(fname)
    img = _normalize(img, 255)
    if img.ndim > 2:
        img = img[:, :, 0]
    return img[None, :, :]


class Preprocessor(ABC):
    """Turns a raw image file into a model-ready (1, 1, H, W) float array."""

    @abstractmethod
    def process(self, path: str | Path) -> np.ndarray:
        ...


class ChestXrayPreprocessor(Preprocessor):
    """Chest X-ray preprocessor for the exported DenseNet.

    Accepts DICOM (.dcm) or PNG/JPG. Both are normalized to xrv's ~[-1024, 1024]
    convention, center-cropped to a square, resized to 224, shaped (1, 1, 224, 224).
    """

    def __init__(self, size: int = TARGET_SIZE) -> None:
        self._size = size

    def _load(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".dcm":
            img = _read_xray_dcm(str(path))
        else:
            img = _load_image(str(path))
        # Defensive: guarantee channel-first single-channel shape (1, H, W).
        if img.ndim == 2:
            img = img[None, ...]
        elif img.ndim == 3 and img.shape[0] != 1:
            img = img.mean(axis=0, keepdims=True)
        return img

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        _, y, x = img.shape
        c = int(np.min([y, x]))
        sx = x // 2 - (c // 2)
        sy = y // 2 - (c // 2)
        return img[:, sy:sy + c, sx:sx + c]

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return skimage.transform.resize(
            img, (1, self._size, self._size), mode="constant", preserve_range=True
        ).astype(np.float32)

    def process(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = self._load(path)          # (1, H, W), ~[-1024, 1024]
        img = self._center_crop(img)    # (1, S, S)
        img = self._resize(img)         # (1, 224, 224)
        return img[None, ...].astype(np.float32)  # (1, 1, 224, 224)
