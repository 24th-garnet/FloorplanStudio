"""ONNX Runtime inference with per-model preprocess/decode."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .constants import DEFAULT_ONNX_PATH
from .labels import WALL_CLASS_ID
from .masks import ml_wall_mask_from_logits
from .model_specs import load_model_spec
from .preprocess import (
    prepare_model_input,
    prepare_stretch_input,
    undo_letterbox_mask,
    undo_stretch_mask,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _wall_mask_from_binary_logits(
    logits: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    if logits.ndim == 4:
        logits = logits[0]
    channel = logits[0] if logits.shape[0] == 1 else logits
    prob = _sigmoid(channel.astype(np.float32))
    return (prob > threshold).astype(np.uint8) * 255


def _decode_wall_mask_model(
    logits: np.ndarray,
    spec: Dict,
) -> np.ndarray:
    family = spec.get("family", "multiclass_unet")
    if family == "binary_unet":
        threshold = float(spec.get("wall_threshold", 0.5))
        return _wall_mask_from_binary_logits(logits, threshold=threshold)
    return ml_wall_mask_from_logits(logits)


def _prepare_tensor(
    render_rgb: np.ndarray,
    spec: Dict,
) -> Tuple[np.ndarray, Dict]:
    preprocess = spec.get("preprocess", "letterbox_imagenet")
    if preprocess == "stretch_imagenet":
        return prepare_stretch_input(render_rgb, int(spec.get("input_size", 512)))
    return prepare_model_input(render_rgb)


def _map_mask_to_render(
    wall_mask_model: np.ndarray,
    preprocess_meta: Dict,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    if preprocess_meta.get("preprocess") == "stretch_imagenet":
        return undo_stretch_mask(wall_mask_model, preprocess_meta)
    return undo_letterbox_mask(wall_mask_model, preprocess_meta, out_h, out_w)


def run_onnx_inference(
    render_rgb: np.ndarray,
    model_path: Path | None = None,
    warmup_runs: int = 1,
    bench_runs: int = 5,
    model_spec: Dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run segmentation on a render image.

    Returns:
        wall_mask_render: uint8 (H,W) at render resolution
        class_mask_model: uint8 (512,512) model-space mask
        stats: timing and model info
    """
    onnx_path = Path(model_path or DEFAULT_ONNX_PATH)
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. "
            "Run: python scripts/floorplan_ml_poc/setup_compare_models.py"
        )

    spec = model_spec or load_model_spec(onnx_path)

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. pip install onnxruntime pillow"
        ) from exc

    tensor, preprocess_meta = _prepare_tensor(render_rgb, spec)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    for _ in range(max(0, warmup_runs)):
        sess.run([output_name], {input_name: tensor})

    t0 = time.perf_counter()
    for _ in range(max(1, bench_runs)):
        outputs = sess.run([output_name], {input_name: tensor})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(1, bench_runs)

    logits = outputs[0]
    wall_mask_model = _decode_wall_mask_model(logits, spec)
    h, w = render_rgb.shape[:2]
    wall_mask_render = _map_mask_to_render(wall_mask_model, preprocess_meta, h, w)

    class_mask = wall_mask_model
    if spec.get("family", "multiclass_unet") == "multiclass_unet":
        if logits.ndim == 4:
            class_mask = np.argmax(logits[0], axis=0).astype(np.uint8)
        else:
            class_mask = np.argmax(logits, axis=0).astype(np.uint8)

    stats = {
        "inference_ms_cpu": round(elapsed_ms, 2),
        "model_input_shape": list(tensor.shape),
        "wall_class_id": int(spec.get("wall_class_id", WALL_CLASS_ID)),
        "model_id": spec.get("id", onnx_path.stem),
        "model_family": spec.get("family"),
        "preprocess": spec.get("preprocess"),
        "onnx_path": str(onnx_path),
    }
    return wall_mask_render, class_mask, stats
