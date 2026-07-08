"""Per-model ONNX metadata (preprocess + wall mask decoding)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import DEFAULT_ONNX_PATH, MODEL_INPUT_SIZE
from .labels import WALL_CLASS_ID

MODELS_DIR = DEFAULT_ONNX_PATH.parent

DEFAULT_MULTICLASS_SPEC: Dict[str, Any] = {
    "family": "multiclass_unet",
    "preprocess": "letterbox_imagenet",
    "input_size": MODEL_INPUT_SIZE,
    "num_classes": 4,
    "wall_class_id": WALL_CLASS_ID,
}


def meta_path_for_onnx(onnx_path: Path) -> Path:
    return Path(onnx_path).with_suffix(".meta.json")


def load_model_spec(onnx_path: Path) -> Dict[str, Any]:
    onnx_path = Path(onnx_path)
    spec: Dict[str, Any] = {
        "id": onnx_path.stem,
        "label": onnx_path.stem,
        "filename": onnx_path.name,
        "path": str(onnx_path.resolve()),
        **DEFAULT_MULTICLASS_SPEC,
    }
    meta_path = meta_path_for_onnx(onnx_path)
    if meta_path.is_file():
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        spec.update(loaded)
    spec["id"] = onnx_path.stem
    spec["path"] = str(onnx_path.resolve())
    return spec


def save_model_spec(onnx_path: Path, spec: Dict[str, Any]) -> Path:
    onnx_path = Path(onnx_path)
    out = meta_path_for_onnx(onnx_path)
    payload = dict(spec)
    payload["id"] = onnx_path.stem
    payload["filename"] = onnx_path.name
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def list_model_specs(models_dir: Path | None = None) -> List[Dict[str, Any]]:
    root = Path(models_dir or MODELS_DIR)
    if not root.is_dir():
        return []
    specs: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.onnx")):
        specs.append(load_model_spec(path))
    return specs
