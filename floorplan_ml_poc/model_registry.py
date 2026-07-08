"""Discover ONNX models and metadata for ML wall segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .constants import DEFAULT_ONNX_PATH
from .model_specs import list_model_specs, load_model_spec

MODELS_DIR = DEFAULT_ONNX_PATH.parent


def list_onnx_models(models_dir: Path | None = None) -> List[Dict]:
    """Return sorted list of model descriptors (includes .meta.json when present)."""
    return list_model_specs(models_dir)


def resolve_model_path(model_id: str | None = None, models_dir: Path | None = None) -> Path:
    if model_id:
        path = (Path(models_dir or MODELS_DIR) / f"{model_id}.onnx").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        return path
    if DEFAULT_ONNX_PATH.is_file():
        return DEFAULT_ONNX_PATH
    models = list_onnx_models(models_dir)
    if not models:
        raise FileNotFoundError(
            f"No ONNX models in {models_dir or MODELS_DIR}. "
            "Run: python scripts/floorplan_ml_poc/setup_compare_models.py"
        )
    return Path(models[0]["path"])


def resolve_model_spec(model_id: str, models_dir: Path | None = None) -> Dict:
    return load_model_spec(resolve_model_path(model_id, models_dir=models_dir))
