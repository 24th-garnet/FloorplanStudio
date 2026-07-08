"""Batch extrude compare: all methods × all ONNX models → 3D meshes."""

from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from wall_extract import extract_walls_from_positions
from wall_extract_refine import refine_wall_extract, structural_wall_extract
from wall_extrude import DEFAULT_WALL_HEIGHT_M, extrude_walls_from_extract

from .constants import ML_POC_SUBDIR
from .dxf_render import render_positions_to_image
from .inference import run_onnx_inference
from .mask_to_walls import mask_to_wall_extract
from .ml_extract import _average_y
from .model_registry import list_onnx_models, resolve_model_path
from .preprocess import compute_letterbox
from .svg_build import positions_to_svg
from .svg_render import render_svg_to_rgb

EXTRUDE_COMPARE_SUBDIR = "extrude_compare"
REFINE_COMPARE_REPORT_FILE = "refine_compare_report.json"

METHOD_HEURISTIC = "heuristic"
METHOD_HEURISTIC_REFINED = "heuristic_refined"
METHOD_HEURISTIC_STRUCTURAL = "heuristic_structural"
METHOD_DXF_PIL = "dxf_pil"
METHOD_DXF_SVG = "dxf_svg_cairosvg"

HEURISTIC_METHODS = (
    METHOD_HEURISTIC,
    METHOD_HEURISTIC_REFINED,
    METHOD_HEURISTIC_STRUCTURAL,
)
REFINE_COMPARE_METHODS = HEURISTIC_METHODS
ML_RENDER_METHODS = (METHOD_DXF_PIL, METHOD_DXF_SVG)

METHOD_LABELS = {
    METHOD_HEURISTIC: "平行ペア",
    METHOD_HEURISTIC_REFINED: "平行ペア(精修)",
    METHOD_HEURISTIC_STRUCTURAL: "平行ペア(構造)",
    METHOD_DXF_PIL: "DXF直接ラスタ",
    METHOD_DXF_SVG: "SVG+cairosvg",
}

# Backward-compatible API keys from the first implementation.
LEGACY_SOURCE_ALIASES = {
    "ml_dxf": METHOD_DXF_PIL,
    "ml_svg": METHOD_DXF_SVG,
}


def variant_id(method: str, model_id: str | None = None) -> str:
    if method in HEURISTIC_METHODS:
        return method
    if not model_id:
        raise ValueError(f"model_id required for method {method}")
    return f"{method}__{model_id}"


def parse_variant_id(vid: str) -> Tuple[str, Optional[str]]:
    if vid in HEURISTIC_METHODS:
        return vid, None
    if "__" not in vid:
        raise ValueError(f"unknown variant id: {vid}")
    method, model_id = vid.split("__", 1)
    return method, model_id


def _variant_color(index: int, total: int) -> str:
    if index == 0:
        return "#6699cc"
    hue = ((index - 1) * 0.61803398875) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.52, 0.62)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _mesh_stats(extrude: Dict) -> Dict:
    positions = extrude.get("positions") or []
    indices = extrude.get("indices") or []
    return {
        "vertex_count": len(positions) // 3,
        "index_count": len(indices),
        "triangle_count": len(indices) // 3,
    }


def _variant_label(method: str, model_id: str | None, model_label: str | None = None) -> str:
    if method in HEURISTIC_METHODS:
        return METHOD_LABELS[method]
    model_part = model_label or model_id or "?"
    return f"ML · {METHOD_LABELS.get(method, method)} · {model_part}"


def _build_render_inputs(geometry: Dict) -> Dict[str, Dict]:
    positions = geometry.get("positions") or []
    bounds = geometry.get("bounds")
    unit_scale = float(geometry.get("unit_scale_to_meters") or 1.0)

    dxf_render = render_positions_to_image(
        positions, bounds=bounds, unit_scale_to_meters=unit_scale
    )
    meta_dxf = dxf_render["meta"]
    meta_dxf["letterbox"] = compute_letterbox(
        int(meta_dxf["render_height_px"]), int(meta_dxf["render_width_px"])
    )
    meta_dxf["render_path"] = METHOD_DXF_PIL

    svg_xml, meta_svg = positions_to_svg(positions, bounds=bounds)
    w, h = int(meta_svg["render_width_px"]), int(meta_svg["render_height_px"])
    render_svg = render_svg_to_rgb(svg_xml, w, h)
    meta_svg["letterbox"] = compute_letterbox(h, w)
    meta_svg["render_path"] = METHOD_DXF_SVG
    meta_svg["unit_scale_to_meters"] = unit_scale

    return {
        METHOD_DXF_PIL: {"image": dxf_render["image"], "meta": meta_dxf},
        METHOD_DXF_SVG: {"image": render_svg, "meta": meta_svg},
    }


def _ensure_heuristic_extract(session_dir: Path, positions: List[float]) -> Dict:
    path = session_dir / "wall_extract.json"
    if not path.is_file():
        extract = extract_walls_from_positions(positions)
        path.write_text(json.dumps(extract), encoding="utf-8")
    else:
        extract = json.loads(path.read_text(encoding="utf-8"))
    extract.setdefault("source", "heuristic")
    extract["method"] = METHOD_HEURISTIC
    return extract


def _ensure_heuristic_refined_extract(
    session_dir: Path,
    positions: List[float],
    heuristic_extract: Dict | None = None,
) -> Dict:
    base = heuristic_extract or _ensure_heuristic_extract(session_dir, positions)
    refined = refine_wall_extract(base)
    refined["method"] = METHOD_HEURISTIC_REFINED
    return refined


def _ensure_heuristic_structural_extract(
    session_dir: Path,
    positions: List[float],
    heuristic_extract: Dict | None = None,
) -> Dict:
    base = heuristic_extract or _ensure_heuristic_extract(session_dir, positions)
    structural = structural_wall_extract(base)
    structural["method"] = METHOD_HEURISTIC_STRUCTURAL
    return structural


def _build_ml_extract(
    mask: np.ndarray,
    render_meta: Dict,
    y_m: float,
    method: str,
    model_id: str,
) -> Dict:
    extract = mask_to_wall_extract(mask, render_meta, y_m=y_m)
    extract["source"] = "ml"
    extract["method"] = method
    extract["model_id"] = model_id
    extract["compare_source"] = variant_id(method, model_id)
    return extract


def _extrude_variant(
    vid: str,
    method: str,
    model_id: str | None,
    extract: Dict,
    height_m: float,
    color: str,
    model_label: str | None = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    if not extract.get("walls"):
        return None, "no walls in extract"
    try:
        extrude = extrude_walls_from_extract(extract, height_m=height_m)
    except Exception as exc:
        return None, str(exc)
    if extrude.get("mesh_wall_count", 0) <= 0:
        return None, "no wall mesh generated"
    extrude["variant_id"] = vid
    extrude["compare_source"] = vid
    extrude["method"] = method
    extrude["model_id"] = model_id
    extrude["label"] = _variant_label(method, model_id, model_label)
    extrude["color"] = color
    return extrude, None


def _save_variant_artifacts(
    out_dir: Path,
    vid: str,
    extract: Dict,
    extrude: Optional[Dict],
) -> None:
    safe = vid.replace("/", "_")
    (out_dir / f"{safe}_extract.json").write_text(json.dumps(extract), encoding="utf-8")
    extrude_path = out_dir / f"{safe}_extrude.json"
    if extrude is not None:
        extrude_path.write_text(json.dumps(extrude), encoding="utf-8")
    elif extrude_path.is_file():
        extrude_path.unlink()


def run_extrude_compare(
    session_dir: Path,
    height_m: float = DEFAULT_WALL_HEIGHT_M,
    *,
    run_render_compare_if_missing: bool = True,  # noqa: ARG001 — kept for API compat
    model_ids: Optional[List[str]] = None,
) -> Dict:
    del run_render_compare_if_missing

    session_dir = Path(session_dir)
    height_m = float(height_m)
    if height_m <= 0:
        raise ValueError("wall height must be positive")

    geom_path = session_dir / "dxf_geometry.json"
    if not geom_path.is_file():
        raise FileNotFoundError(f"DXF geometry not found: {geom_path}")
    geometry = json.loads(geom_path.read_text(encoding="utf-8"))
    positions = geometry.get("positions") or []
    if not positions:
        raise ValueError("DXF geometry is empty")

    all_models = list_onnx_models()
    if not all_models:
        raise FileNotFoundError(
            "No ONNX models found. Run: python scripts/floorplan_ml_poc/export_onnx.py"
        )
    if model_ids:
        selected = [m for m in all_models if m["id"] in model_ids]
        if not selected:
            raise ValueError(f"none of the requested models found: {model_ids}")
    else:
        selected = all_models

    y_m = _average_y(positions)
    render_inputs = _build_render_inputs(geometry)
    heuristic_extract = _ensure_heuristic_extract(session_dir, positions)

    out_dir = session_dir / ML_POC_SUBDIR / EXTRUDE_COMPARE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    planned: List[Tuple[str, str, Optional[str]]] = [
        (METHOD_HEURISTIC, METHOD_HEURISTIC, None),
        (METHOD_HEURISTIC_REFINED, METHOD_HEURISTIC_REFINED, None),
        (METHOD_HEURISTIC_STRUCTURAL, METHOD_HEURISTIC_STRUCTURAL, None),
    ]
    for model in selected:
        for method in ML_RENDER_METHODS:
            planned.append((variant_id(method, model["id"]), method, model["id"]))

    variants: List[Dict] = []
    sources: Dict[str, Dict] = {}
    matrix: Dict[str, Dict[str, Dict]] = {
        METHOD_HEURISTIC: {},
        METHOD_HEURISTIC_REFINED: {},
        METHOD_HEURISTIC_STRUCTURAL: {},
    }
    for method in ML_RENDER_METHODS:
        matrix[method] = {}

    default_model_id = selected[0]["id"]
    model_by_id = {m["id"]: m for m in selected}

    for index, (vid, method, model_id) in enumerate(planned):
        color = _variant_color(index, len(planned))
        model_meta = model_by_id.get(model_id or "")
        model_label = model_meta.get("label") if model_meta else None
        entry: Dict = {
            "id": vid,
            "method": method,
            "model_id": model_id,
            "label": _variant_label(method, model_id, model_label),
            "color": color,
        }

        if method == METHOD_HEURISTIC:
            extract = heuristic_extract
            inference_ms = None
        elif method == METHOD_HEURISTIC_REFINED:
            extract = _ensure_heuristic_refined_extract(
                session_dir, positions, heuristic_extract
            )
            inference_ms = None
        elif method == METHOD_HEURISTIC_STRUCTURAL:
            extract = _ensure_heuristic_structural_extract(
                session_dir, positions, heuristic_extract
            )
            inference_ms = None
        else:
            model_path = resolve_model_path(model_id)
            render_rgb = render_inputs[method]["image"]
            render_meta = render_inputs[method]["meta"]
            mask, _, stats = run_onnx_inference(render_rgb, model_path=model_path)
            inference_ms = stats.get("inference_ms_cpu")
            extract = _build_ml_extract(mask, render_meta, y_m, method, model_id or "")

        extrude, error = _extrude_variant(
            vid, method, model_id, extract, height_m, color, model_label
        )
        _save_variant_artifacts(out_dir, vid, extract, extrude)

        entry.update(
            {
                "wall_count": int(extract.get("wall_count") or 0),
                "available": extrude is not None,
                "error": error,
                "inference_ms_cpu": inference_ms,
            }
        )
        if extrude is not None:
            entry.update(
                {
                    "mesh_wall_count": int(extrude.get("mesh_wall_count") or 0),
                    "height_m": extrude.get("height_m"),
                    **_mesh_stats(extrude),
                }
            )

        variants.append(entry)
        if method in HEURISTIC_METHODS:
            matrix[method]["_"] = entry
            sources[method] = entry
        else:
            matrix[method][model_id or ""] = entry
            legacy_key = {METHOD_DXF_PIL: "ml_dxf", METHOD_DXF_SVG: "ml_svg"}.get(method)
            if legacy_key and model_id == default_model_id:
                sources[legacy_key] = entry

    legacy_sources = {
        "ml_dxf": variant_id(METHOD_DXF_PIL, default_model_id),
        "ml_svg": variant_id(METHOD_DXF_SVG, default_model_id),
    }

    report = {
        "session_id": session_dir.name,
        "height_m": round(height_m, 4),
        "models": selected,
        "methods": [
            {"id": m, "label": METHOD_LABELS[m]}
            for m in (*HEURISTIC_METHODS, *ML_RENDER_METHODS)
        ],
        "variants": variants,
        "matrix": matrix,
        "sources": sources,
        "legacy_sources": legacy_sources,
        "variant_count": len(variants),
        "available_count": sum(1 for v in variants if v.get("available")),
        "output_dir": str(out_dir),
    }
    (out_dir / "extrude_compare_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def run_refine_wall_compare(
    session_dir: Path,
    height_m: float = DEFAULT_WALL_HEIGHT_M,
) -> Dict:
    session_dir = Path(session_dir)
    height_m = float(height_m)
    if height_m <= 0:
        raise ValueError("wall height must be positive")

    geom_path = session_dir / "dxf_geometry.json"
    if not geom_path.is_file():
        raise FileNotFoundError(f"DXF geometry not found: {geom_path}")
    geometry = json.loads(geom_path.read_text(encoding="utf-8"))
    positions = geometry.get("positions") or []
    if not positions:
        raise ValueError("DXF geometry is empty")

    heuristic_extract = _ensure_heuristic_extract(session_dir, positions)
    out_dir = session_dir / ML_POC_SUBDIR / EXTRUDE_COMPARE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    planned: List[Tuple[str, str, Optional[str]]] = [
        (METHOD_HEURISTIC, METHOD_HEURISTIC, None),
        (METHOD_HEURISTIC_REFINED, METHOD_HEURISTIC_REFINED, None),
        (METHOD_HEURISTIC_STRUCTURAL, METHOD_HEURISTIC_STRUCTURAL, None),
    ]

    variants: List[Dict] = []
    sources: Dict[str, Dict] = {}
    matrix: Dict[str, Dict[str, Dict]] = {
        METHOD_HEURISTIC: {},
        METHOD_HEURISTIC_REFINED: {},
        METHOD_HEURISTIC_STRUCTURAL: {},
    }

    for index, (vid, method, model_id) in enumerate(planned):
        color = _variant_color(index, len(planned))
        entry: Dict = {
            "id": vid,
            "method": method,
            "model_id": model_id,
            "label": _variant_label(method, model_id),
            "color": color,
        }

        if method == METHOD_HEURISTIC:
            extract = heuristic_extract
        elif method == METHOD_HEURISTIC_REFINED:
            extract = _ensure_heuristic_refined_extract(
                session_dir, positions, heuristic_extract
            )
        else:
            extract = _ensure_heuristic_structural_extract(
                session_dir, positions, heuristic_extract
            )

        extrude, error = _extrude_variant(
            vid, method, model_id, extract, height_m, color
        )
        _save_variant_artifacts(out_dir, vid, extract, extrude)

        entry.update(
            {
                "wall_count": int(extract.get("wall_count") or 0),
                "available": extrude is not None,
                "error": error,
                "inference_ms_cpu": None,
            }
        )
        if extrude is not None:
            entry.update(
                {
                    "mesh_wall_count": int(extrude.get("mesh_wall_count") or 0),
                    "height_m": extrude.get("height_m"),
                    **_mesh_stats(extrude),
                }
            )
        if extract.get("refine_stats"):
            entry["refine_stats"] = extract["refine_stats"]

        variants.append(entry)
        matrix[method]["_"] = entry
        sources[method] = entry

    report = {
        "session_id": session_dir.name,
        "compare_kind": "refine",
        "height_m": round(height_m, 4),
        "models": [],
        "methods": [
            {"id": m, "label": METHOD_LABELS[m]}
            for m in REFINE_COMPARE_METHODS
        ],
        "variants": variants,
        "matrix": matrix,
        "sources": sources,
        "legacy_sources": {},
        "variant_count": len(variants),
        "available_count": sum(1 for v in variants if v.get("available")),
        "output_dir": str(out_dir),
    }
    (out_dir / REFINE_COMPARE_REPORT_FILE).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _resolve_variant_key(session_dir: Path, key: str) -> str:
    if key in HEURISTIC_METHODS or "__" in key:
        return key
    report_path = (
        Path(session_dir) / ML_POC_SUBDIR / EXTRUDE_COMPARE_SUBDIR / "extrude_compare_report.json"
    )
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        legacy = report.get("legacy_sources") or {}
        if key in legacy:
            return legacy[key]
    method = LEGACY_SOURCE_ALIASES.get(key)
    if method:
        models = list_onnx_models()
        if models:
            return variant_id(method, models[0]["id"])
    return key


def _variant_artifact_path(session_dir: Path, key: str, suffix: str) -> Path:
    vid = _resolve_variant_key(session_dir, key)
    safe = vid.replace("/", "_")
    return Path(session_dir) / ML_POC_SUBDIR / EXTRUDE_COMPARE_SUBDIR / f"{safe}_{suffix}.json"


def load_extrude_compare_mesh(session_dir: Path, compare_source: str) -> Dict:
    path = _variant_artifact_path(session_dir, compare_source, "extrude")
    if not path.is_file():
        raise FileNotFoundError(f"no extrusion for variant: {compare_source}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_extrude_compare_extract(session_dir: Path, compare_source: str) -> Dict:
    path = _variant_artifact_path(session_dir, compare_source, "extract")
    if not path.is_file():
        raise FileNotFoundError(f"no extract for variant: {compare_source}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_variant_ids_from_report(session_dir: Path) -> List[str]:
    ids: List[str] = []
    out_dir = Path(session_dir) / ML_POC_SUBDIR / EXTRUDE_COMPARE_SUBDIR
    for report_name in ("extrude_compare_report.json", REFINE_COMPARE_REPORT_FILE):
        report_path = out_dir / report_name
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        ids.extend(v["id"] for v in report.get("variants") or [])
    return ids
