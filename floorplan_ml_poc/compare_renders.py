"""Compare ML inference: DXF PIL raster vs DXF→SVG→cairosvg raster."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from wall_extract import extract_walls_from_positions

from .constants import DEFAULT_ONNX_PATH, ML_POC_SUBDIR
from .dxf_render import render_positions_to_image
from .inference import run_onnx_inference
from .masks import compute_binary_metrics, heuristic_wall_mask, make_diff_image
from .preprocess import compute_letterbox
from .svg_build import export_native_dxf_svg, positions_to_svg
from .svg_render import render_svg_to_rgb

RENDER_COMPARE_SUBDIR = "render_compare"


def _save_png(path: Path, rgb_or_gray: np.ndarray) -> None:
    from PIL import Image

    if rgb_or_gray.ndim == 2:
        Image.fromarray(rgb_or_gray).save(path)
    else:
        Image.fromarray(rgb_or_gray.astype(np.uint8)).save(path)


def make_render_compare_overlay(
    render_dxf: np.ndarray,
    ml_dxf: np.ndarray,
    render_svg: np.ndarray,
    ml_svg: np.ndarray,
    heuristic: np.ndarray,
) -> np.ndarray:
    from .masks import _tint_mask

    panels = [
        render_dxf,
        _tint_mask(render_dxf, ml_dxf, (255, 122, 0)),
        render_svg,
        _tint_mask(render_svg, ml_svg, (255, 122, 0)),
        _tint_mask(render_dxf, heuristic, (0, 120, 255)),
    ]
    return np.concatenate(panels, axis=1)


def run_render_path_compare(
    session_dir: Path,
    model_path: Optional[Path] = None,
    *,
    run_wall_extract_if_missing: bool = True,
    include_native_dxf_svg: bool = True,
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
            raise FileNotFoundError("wall_extract.json missing")
        extract = extract_walls_from_positions(positions)
        extract_path.write_text(json.dumps(extract), encoding="utf-8")
    else:
        extract = json.loads(extract_path.read_text(encoding="utf-8"))

    unit_scale = float(geometry.get("unit_scale_to_meters") or 1.0)
    bounds = geometry.get("bounds")

    # Path A: direct PIL raster (current PoC).
    dxf_render = render_positions_to_image(
        positions, bounds=bounds, unit_scale_to_meters=unit_scale
    )
    render_dxf = dxf_render["image"]
    meta_dxf = dxf_render["meta"]
    meta_dxf["letterbox"] = compute_letterbox(
        int(meta_dxf["render_height_px"]), int(meta_dxf["render_width_px"])
    )
    meta_dxf["render_path"] = "dxf_pil"

    # Path B: positions -> SVG -> cairosvg (CubiCasa-style renderer).
    svg_xml, meta_svg = positions_to_svg(positions, bounds=bounds)
    w, h = int(meta_svg["render_width_px"]), int(meta_svg["render_height_px"])
    render_svg = render_svg_to_rgb(svg_xml, w, h)
    meta_svg["letterbox"] = compute_letterbox(h, w)
    meta_svg["render_path"] = "dxf_svg_cairosvg"
    meta_svg["unit_scale_to_meters"] = unit_scale

    ml_mask_dxf, _, stats_dxf = run_onnx_inference(render_dxf, model_path=model_path)
    ml_mask_svg, _, stats_svg = run_onnx_inference(render_svg, model_path=model_path)

    heur_mask = heuristic_wall_mask(extract.get("walls") or [], meta_dxf)

    metrics_dxf = compute_binary_metrics(ml_mask_dxf, heur_mask)
    metrics_svg = compute_binary_metrics(ml_mask_svg, heur_mask)
    metrics_cross = compute_binary_metrics(ml_mask_dxf, ml_mask_svg)

    out_dir = session_dir / ML_POC_SUBDIR / RENDER_COMPARE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "model.svg").write_text(svg_xml, encoding="utf-8")
    _save_png(out_dir / "render_dxf.png", render_dxf)
    _save_png(out_dir / "render_svg.png", render_svg)
    _save_png(out_dir / "ml_wall_mask_dxf.png", ml_mask_dxf)
    _save_png(out_dir / "ml_wall_mask_svg.png", ml_mask_svg)
    _save_png(out_dir / "heuristic_wall_mask.png", heur_mask)
    _save_png(
        out_dir / "overlay_compare.png",
        make_render_compare_overlay(
            render_dxf, ml_mask_dxf, render_svg, ml_mask_svg, heur_mask
        ),
    )
    _save_png(
        out_dir / "diff_dxf_vs_svg.png",
        make_diff_image(ml_mask_dxf, ml_mask_svg),
    )
    (out_dir / "render_meta_dxf.json").write_text(
        json.dumps(meta_dxf, indent=2), encoding="utf-8"
    )
    (out_dir / "render_meta_svg.json").write_text(
        json.dumps(meta_svg, indent=2), encoding="utf-8"
    )

    native_ok = False
    if include_native_dxf_svg:
        dxf_file = session_dir / "original.dxf"
        if dxf_file.is_file():
            native_ok = export_native_dxf_svg(str(dxf_file), str(out_dir / "native_ezdxf.svg"))

    report = {
        "session_id": session_dir.name,
        "heuristic_wall_count": int(extract.get("wall_count") or 0),
        "render_paths": {
            "dxf_pil": {
                **metrics_dxf,
                "inference_ms_cpu": stats_dxf.get("inference_ms_cpu"),
            },
            "dxf_svg_cairosvg": {
                **metrics_svg,
                "inference_ms_cpu": stats_svg.get("inference_ms_cpu"),
            },
        },
        "ml_dxf_vs_ml_svg": {
            "wall_iou": metrics_cross["wall_iou"],
            "wall_precision": metrics_cross["wall_precision"],
            "wall_recall": metrics_cross["wall_recall"],
            "ml_dxf_pixels": metrics_cross["ml_wall_pixels"],
            "ml_svg_pixels": metrics_cross["heuristic_wall_pixels"],
        },
        "native_ezdxf_svg_exported": native_ok,
        "output_dir": str(out_dir),
        "panel_labels": ["dxf_render", "ml_on_dxf", "svg_render", "ml_on_svg", "heuristic"],
        "files": [
            "model.svg",
            "native_ezdxf.svg",
            "render_dxf.png",
            "render_svg.png",
            "ml_wall_mask_dxf.png",
            "ml_wall_mask_svg.png",
            "heuristic_wall_mask.png",
            "overlay_compare.png",
            "diff_dxf_vs_svg.png",
            "render_meta_dxf.json",
            "render_meta_svg.json",
            "compare_renders_report.json",
        ],
    }
    (out_dir / "compare_renders_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
