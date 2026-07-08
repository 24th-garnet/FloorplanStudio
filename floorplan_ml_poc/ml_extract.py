"""Run ML inference and extract wall polygons for 3D extrusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .constants import DEFAULT_ONNX_PATH
from .dxf_render import render_positions_to_image
from .inference import run_onnx_inference
from .mask_to_walls import mask_to_wall_extract
from .preprocess import compute_letterbox

WALL_ML_EXTRACT_FILE = "wall_ml_extract.json"


def _average_y(positions) -> float:
    if not positions:
        return 0.0
    total = 0.0
    count = 0
    for i in range(1, len(positions), 3):
        total += float(positions[i])
        count += 1
    return total / max(count, 1)


def run_ml_wall_extract(
    session_dir: Path,
    model_path: Optional[Path] = None,
    *,
    save: bool = True,
) -> Dict:
    session_dir = Path(session_dir)
    geom_path = session_dir / "dxf_geometry.json"
    if not geom_path.is_file():
        raise FileNotFoundError(f"DXF geometry not found: {geom_path}")

    geometry = json.loads(geom_path.read_text(encoding="utf-8"))
    positions = geometry.get("positions") or []
    if not positions:
        raise ValueError("DXF geometry is empty")

    render = render_positions_to_image(
        positions,
        bounds=geometry.get("bounds"),
        unit_scale_to_meters=float(geometry.get("unit_scale_to_meters") or 1.0),
    )
    render_rgb = render["image"]
    render_meta = render["meta"]
    render_meta["letterbox"] = compute_letterbox(
        int(render_meta["render_height_px"]),
        int(render_meta["render_width_px"]),
    )

    ml_mask, _, infer_stats = run_onnx_inference(
        render_rgb, model_path=model_path or DEFAULT_ONNX_PATH
    )
    y_m = _average_y(positions)
    result = mask_to_wall_extract(ml_mask, render_meta, y_m=y_m)
    result["inference_ms_cpu"] = infer_stats.get("inference_ms_cpu")
    result["onnx_path"] = infer_stats.get("onnx_path")

    if save:
        out_path = session_dir / WALL_ML_EXTRACT_FILE
        out_path.write_text(json.dumps(result), encoding="utf-8")
    return result
