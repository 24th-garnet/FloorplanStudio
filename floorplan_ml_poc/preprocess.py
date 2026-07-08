"""Letterbox + ImageNet normalization (matches floorplan-to-3d data.py)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_INPUT_SIZE

LetterboxMeta = Dict[str, float | int]


def compute_letterbox(
    src_h: int, src_w: int, dst_size: int = MODEL_INPUT_SIZE
) -> LetterboxMeta:
    scale = min(dst_size / src_w, dst_size / src_h)
    inner_w = max(1, int(round(src_w * scale)))
    inner_h = max(1, int(round(src_h * scale)))
    pad_left = (dst_size - inner_w) // 2
    pad_top = (dst_size - inner_h) // 2
    return {
        "input_size": dst_size,
        "scale": scale,
        "inner_width": inner_w,
        "inner_height": inner_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }


def letterbox_rgb(image: np.ndarray, dst_size: int = MODEL_INPUT_SIZE) -> Tuple[np.ndarray, LetterboxMeta]:
    """RGB uint8 (H,W,3) -> letterboxed RGB uint8 (dst_size,dst_size,3)."""
    src_h, src_w = image.shape[:2]
    meta = compute_letterbox(src_h, src_w, dst_size)
    inner_w = int(meta["inner_width"])
    inner_h = int(meta["inner_height"])
    pad_left = int(meta["pad_left"])
    pad_top = int(meta["pad_top"])

    resized = _resize_rgb(image, inner_w, inner_h)
    canvas = np.zeros((dst_size, dst_size, 3), dtype=np.uint8)
    mean_rgb = tuple(int(round(v * 255)) for v in IMAGENET_MEAN)
    canvas[:, :] = mean_rgb
    canvas[pad_top : pad_top + inner_h, pad_left : pad_left + inner_w] = resized
    return canvas, meta


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> float32 NCHW normalized tensor."""
    arr = image.astype(np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return np.transpose(arr, (2, 0, 1))


def prepare_model_input(image: np.ndarray) -> Tuple[np.ndarray, LetterboxMeta]:
    """RGB uint8 render image -> (1,3,512,512) float32 + letterbox meta."""
    boxed, meta = letterbox_rgb(image)
    tensor = normalize_rgb(boxed)
    return tensor[np.newaxis, ...].astype(np.float32), meta


def prepare_stretch_input(
    image: np.ndarray, dst_size: int = MODEL_INPUT_SIZE
) -> Tuple[np.ndarray, Dict[str, int]]:
    """RGB uint8 -> stretch-resized (dst_size,dst_size) tensor (MitUNet-style)."""
    src_h, src_w = image.shape[:2]
    resized = _resize_rgb(image, dst_size, dst_size)
    tensor = normalize_rgb(resized)
    meta = {
        "preprocess": "stretch_imagenet",
        "input_size": dst_size,
        "src_width": src_w,
        "src_height": src_h,
    }
    return tensor[np.newaxis, ...].astype(np.float32), meta


def undo_stretch_mask(mask: np.ndarray, meta: Dict[str, int]) -> np.ndarray:
    out_w = int(meta["src_width"])
    out_h = int(meta["src_height"])
    return _resize_nearest(mask, out_w, out_h)


def undo_letterbox_mask(mask: np.ndarray, meta: LetterboxMeta, out_h: int, out_w: int) -> np.ndarray:
    """Model-size class mask -> render-resolution binary/label mask."""
    pad_left = int(meta["pad_left"])
    pad_top = int(meta["pad_top"])
    inner_h = int(meta["inner_height"])
    inner_w = int(meta["inner_width"])
    cropped = mask[pad_top : pad_top + inner_h, pad_left : pad_left + inner_w]
    return _resize_nearest(cropped, out_w, out_h)


def _resize_rgb(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    try:
        from PIL import Image

        pil = Image.fromarray(image)
        return np.array(pil.resize((out_w, out_h), Image.Resampling.BILINEAR))
    except ImportError:
        # Nearest-neighbor fallback without Pillow.
        src_h, src_w = image.shape[:2]
        ys = (np.arange(out_h) * src_h / out_h).astype(int)
        xs = (np.arange(out_w) * src_w / out_w).astype(int)
        return image[np.ix_(ys, xs)]


def _resize_nearest(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    try:
        from PIL import Image

        pil = Image.fromarray(mask.astype(np.uint8))
        return np.array(pil.resize((out_w, out_h), Image.Resampling.NEAREST))
    except ImportError:
        src_h, src_w = mask.shape[:2]
        ys = (np.arange(out_h) * src_h / out_h).astype(int)
        xs = (np.arange(out_w) * src_w / out_w).astype(int)
        return mask[np.ix_(ys, xs)]
