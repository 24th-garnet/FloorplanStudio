"""End-to-end ML PoC: DXF render -> ONNX -> compare with heuristic walls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from wall_extract import extract_walls_from_positions

from .constants import DEFAULT_ONNX_PATH, ML_POC_SUBDIR
from .dxf_render import render_positions_to_image
from .inference import run_onnx_inference
from .mask_to_walls import mask_to_wall_extract
from .ml_extract import WALL_ML_EXTRACT_FILE
from .masks import (
    compute_binary_metrics,
    heuristic_wall_mask,
    make_diff_image,
    make_overlay_compare,
)
from .preprocess import compute_letterbox


def _save_png(path: Path, rgb_or_gray: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pillow is required to save PNG outputs") from exc
    arr = rgb_or_gray
    if arr.ndim == 2:
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(arr.astype(np.uint8)).save(path)


def run_ml_poc_compare(
    session_dir: Path,
    model_path: Optional[Path] = None,
    *,
    run_wall_extract_if_missing: bool = True,
) -> Dict:
    session_dir = Path(session_dir)
    geom_path = session_dir / "dxf_geometry.json"
    if not geom_path.is_file():
        raise FileNotFoundError(f"DXF geometry not found: {geom_path}")

    geometry = json.loads(geom_path.read_text(encoding="utf-8"))
    positions = geometry.get("positions") or []
    if not positions:
        raise ValueError("DXF geometry is empty")

    extract_path = session_dir / "wall_extract.json"
    if not extract_path.is_file():
        if not run_wall_extract_if_missing:
            raise FileNotFoundError("wall_extract.json missing; run wall extraction first")
        extract = extract_walls_from_positions(positions)
        extract_path.write_text(json.dumps(extract), encoding="utf-8")
    else:
        extract = json.loads(extract_path.read_text(encoding="utf-8"))

    render = render_positions_to_image(
        positions,
        bounds=geometry.get("bounds"),
        unit_scale_to_meters=float(geometry.get("unit_scale_to_meters") or 1.0),
    )
    render_rgb = render["image"]
    render_meta = render["meta"]
    letterbox = compute_letterbox(
        int(render_meta["render_height_px"]),
        int(render_meta["render_width_px"]),
    )
    render_meta["letterbox"] = letterbox

    ml_mask, _, infer_stats = run_onnx_inference(
        render_rgb, model_path=model_path or DEFAULT_ONNX_PATH
    )
    y_m = sum(float(positions[i]) for i in range(1, len(positions), 3)) / max(
        len(positions) // 3, 1
    )
    ml_extract = mask_to_wall_extract(ml_mask, render_meta, y_m=y_m)
    ml_extract["inference_ms_cpu"] = infer_stats.get("inference_ms_cpu")
    (session_dir / WALL_ML_EXTRACT_FILE).write_text(
        json.dumps(ml_extract), encoding="utf-8"
    )
    heur_mask = heuristic_wall_mask(extract.get("walls") or [], render_meta)
    metrics = compute_binary_metrics(ml_mask, heur_mask)

    overlay = make_overlay_compare(render_rgb, ml_mask, heur_mask)
    diff = make_diff_image(ml_mask, heur_mask)

    out_dir = session_dir / ML_POC_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_png(out_dir / "render.png", render_rgb)
    _save_png(out_dir / "ml_wall_mask.png", ml_mask)
    _save_png(out_dir / "heuristic_wall_mask.png", heur_mask)
    _save_png(out_dir / "overlay_compare.png", overlay)
    _save_png(out_dir / "diff.png", diff)
    (out_dir / "render_meta.json").write_text(
        json.dumps(render_meta, indent=2), encoding="utf-8"
    )

    report = {
        "session_id": session_dir.name,
        "heuristic_wall_count": int(extract.get("wall_count") or len(extract.get("walls") or [])),
        "ml_wall_count": int(ml_extract.get("wall_count") or 0),
        **metrics,
        **infer_stats,
        "output_dir": str(out_dir),
        "files": [
            "render.png",
            "render_meta.json",
            "ml_wall_mask.png",
            "heuristic_wall_mask.png",
            "overlay_compare.png",
            "diff.png",
            "compare_report.json",
        ],
    }
    (out_dir / "compare_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
