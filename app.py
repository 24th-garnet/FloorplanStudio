import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from dxf_parser import parse_dxf_file
from wall_extract import extract_walls_from_positions
from wall_extrude import DEFAULT_WALL_HEIGHT_M, extrude_walls_from_extract
from roomplan_export import build_export_file, get_export_format_spec, list_export_formats
from floorplan_ml_poc.compare import run_ml_poc_compare
from floorplan_ml_poc.compare_renders import RENDER_COMPARE_SUBDIR, run_render_path_compare
from floorplan_ml_poc.constants import DEFAULT_ONNX_PATH, ML_POC_SUBDIR
from floorplan_ml_poc.extrude_compare import (
    EXTRUDE_COMPARE_SUBDIR,
    METHOD_HEURISTIC_STRUCTURAL,
    REFINE_COMPARE_REPORT_FILE,
    load_extrude_compare_extract,
    load_extrude_compare_mesh,
    list_variant_ids_from_report,
    run_extrude_compare,
    run_refine_wall_compare,
)
from floorplan_ml_poc.ml_extract import WALL_ML_EXTRACT_FILE, run_ml_wall_extract
from floorplan_ml_poc.model_registry import list_onnx_models

try:
    from pxr import Usd
except Exception:  # pragma: no cover - optional dependency fallback
    Usd = None

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SESSION_DIR = BASE_DIR / "sessions"
REPLACEMENTS_DIR = BASE_DIR / "assets" / "replacements"
TEXTURE_DIR = BASE_DIR / "texture"
REPLACEMENT_STATE_FILE = "replacement_state.json"
OBJECT_TAGS_STATE_FILE = "object_tags_state.json"
LAYER_STATE_FILE = "layer_state.json"
DXF_GEOMETRY_FILE = "dxf_geometry.json"
DXF_LAYER_STATE_FILE = "dxf_layer_state.json"
WALL_EXTRACT_FILE = "wall_extract.json"
WALL_EXTRUDE_FILE = "wall_extrude.json"
SESSION_META_FILE = "session_meta.json"
INPUT_SOURCE_SPECS: Dict[str, Dict] = {
    "roomplan_usdz": {
        "label": "RoomPlan",
        "description": "Apple RoomPlan API が出力する USDA ベースの USDZ",
        "extensions": [".usdz"],
        "kind": "usdz",
    },
    "scaniverse_usdz": {
        "label": "Scaniverse",
        "description": "Scaniverse アプリからエクスポートした USDZ",
        "extensions": [".usdz"],
        "kind": "usdz",
    },
    "polycam_dxf": {
        "label": "PolyCAM",
        "description": "Polycam からエクスポートした DXF（.xdf 拡張子も受け付け）",
        "extensions": [".dxf", ".xdf"],
        "kind": "dxf",
    },
}
USDZ_INPUT_SOURCES = frozenset(
    source_id for source_id, spec in INPUT_SOURCE_SPECS.items() if spec["kind"] == "usdz"
)
DXF_INPUT_SOURCES = frozenset(
    source_id for source_id, spec in INPUT_SOURCE_SPECS.items() if spec["kind"] == "dxf"
)
ML_POC_ALLOWED_FILES = frozenset(
    {
        "render.png",
        "render_meta.json",
        "ml_wall_mask.png",
        "heuristic_wall_mask.png",
        "overlay_compare.png",
        "diff.png",
        "compare_report.json",
    }
)
ML_RENDER_COMPARE_ALLOWED_FILES = frozenset(
    {
        "overlay_compare.png",
        "diff_dxf_vs_svg.png",
        "render_dxf.png",
        "render_svg.png",
        "ml_wall_mask_dxf.png",
        "ml_wall_mask_svg.png",
        "heuristic_wall_mask.png",
        "compare_renders_report.json",
        "render_meta_dxf.json",
        "render_meta_svg.json",
        "model.svg",
        "native_ezdxf.svg",
    }
)
LAYER_DEFINITIONS: Dict[str, Dict] = {
    "floor": {"label": "Floor", "z_order": 0, "default_visible": True, "default_locked": True},
    "wall": {"label": "Wall", "z_order": 10, "default_visible": True, "default_locked": True},
    "door": {"label": "Door", "z_order": 20, "default_visible": True, "default_locked": False},
    "window": {"label": "Window", "z_order": 20, "default_visible": True, "default_locked": False},
    "opening": {"label": "Opening", "z_order": 20, "default_visible": True, "default_locked": False},
    "chair": {"label": "Chair", "z_order": 30, "default_visible": True, "default_locked": False},
    "table": {"label": "Table", "z_order": 31, "default_visible": True, "default_locked": False},
    "storage": {"label": "Storage", "z_order": 32, "default_visible": True, "default_locked": False},
    "other": {"label": "Other", "z_order": 99, "default_visible": True, "default_locked": False},
}
LAYER_NAME_PREFIXES: List[Tuple[str, str]] = [
    ("floor", "floor"),
    ("wall", "wall"),
    ("door", "door"),
    ("window", "window"),
    ("opening", "opening"),
    ("chair", "chair"),
    ("table", "table"),
    ("storage", "storage"),
]
LAYER_TO_ROOMPLAN_CATEGORY: Dict[str, str] = {
    "floor": "Floor",
    "wall": "Wall",
    "door": "Door",
    "window": "Window",
    "opening": "Opening",
    "chair": "Chair",
    "table": "Table",
    "storage": "Storage",
    "other": "Object",
}
REPLACEMENT_ASSET_MAP = {
    "chair": "Chair.usdz",
    "table": "table.usdz",
    "storage": "storage.usdz",
}
TEXTURE_ASSET_MAP = {
    "floor": "floor.png",
    "wall": "wall.png",
}
UPLOAD_DIR.mkdir(exist_ok=True)
SESSION_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True

CATEGORY_RE = re.compile(r'string\s+Category\s*=\s*"([^"]+)"')
UUID_RE = re.compile(r'string\s+UUID\s*=\s*"([^"]+)"')
XFORM_NAME_RE = re.compile(r'def\s+Xform\s+"([^"]+)"')
POINTS_RE = re.compile(r'point3f\[\]\s+points\s*=\s*\[(.*?)\]', re.DOTALL)


def safe_session_path(session_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9\-]{36}", session_id):
        raise ValueError("invalid session id")
    return SESSION_DIR / session_id


def unpack_usdz(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dst)


def repack_usdz(src_dir: Path, out_path: Path) -> None:
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
        files = sorted([p for p in src_dir.rglob("*") if p.is_file()])
        files.sort(
            key=lambda p: (
                0 if p.parent == src_dir and p.suffix.lower() in [".usda", ".usd"] else 1,
                str(p),
            )
        )
        for p in files:
            zf.write(p, p.relative_to(src_dir).as_posix())


# RoomPlan may use `prepend references = [ @a@, @b@, ... ]` on one Xform. THREE.USDZLoader
# expects a single `@path@` and its naive USDA parser turns arrays into `prepend references = "["`,
# which crashes in findMeshGeometry (parts[1] undefined). Expand each array into one child Xform
# per reference with a single prepend line.
_USD_REF_PATH_RE = re.compile(r"@[^@]+\.usda@")
_PREPEND_REF_ARRAY_BLOCK_RE = re.compile(
    r"(?ms)^(?P<idef>\s*)def Xform \"(?P<name>[^\"]+)\" \(\s*\n"
    r"(?P<ikind>\s*)kind = \"group\"\s*\n"
    r"(?P<iprep>\s*)prepend references = \[\s*\n"
    r"(?P<refs>(?:[ \t]*@[^@\n]+@\s*,?\s*\n)+)"
    r"[ \t]*\]\s*\n"
    r"(?P<iclose>\s*)\)\s*\n"
    r"(?P<iob>\s*)\{\s*\n"
    r"(?P<ice>\s*)\}\s*"
)


def expand_prepend_reference_arrays_for_usdz_loader(text: str) -> str:
    def repl(m: re.Match) -> str:
        idef = m.group("idef")
        name = m.group("name")
        ref_block = m.group("refs")
        paths = _USD_REF_PATH_RE.findall(ref_block)
        if not paths:
            return m.group(0)
        inner_indent = idef + "    "
        chunks = []
        for i, pth in enumerate(paths):
            child = f"{name}__r{i}"
            chunks.append(
                f'{inner_indent}def Xform "{child}" (\n'
                f'{inner_indent}    kind = "group"\n'
                f"{inner_indent}    prepend references = {pth}\n"
                f"{inner_indent})\n"
                f"{inner_indent}{{\n"
                f"{inner_indent}}}\n"
            )
        body = "".join(chunks)
        return (
            f'{idef}def Xform "{name}" (\n'
            f'{idef}    kind = "group"\n'
            f"{idef})\n"
            f"{idef}{{\n"
            f"{body}"
            f"{idef}}}\n"
        )

    return _PREPEND_REF_ARRAY_BLOCK_RE.sub(repl, text)


def normalize_extract_usda_for_usdz_loader(extract_dir: Path) -> None:
    for p in extract_dir.rglob("*.usda"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "prepend references = [" not in text:
            continue
        new = expand_prepend_reference_arrays_for_usdz_loader(text)
        if new != text:
            p.write_text(new, encoding="utf-8")


def _rewrite_usd_references_to_usda(text: str) -> str:
    """
    Rewrite common reference forms from .usdc/.usd to .usda inside USDA text.
    This helps downstream tools (and our own regex-based parser) follow references
    after we materialize .usda files from .usdc.
    """

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        lowered = inner.lower()
        if lowered.endswith(".usdc"):
            inner = inner[:-5] + "usda"
        elif lowered.endswith(".usd"):
            inner = inner[:-3] + "usda"
        return f"@{inner}@"

    return re.sub(r"@([^@]+?\.(?:usd|usdc))@", repl, text, flags=re.IGNORECASE)


def materialize_usda_from_usdc_in_extract(extract_dir: Path) -> Dict[str, object]:
    """
    Ensure extract_dir contains readable .usda layers even when the USDZ only includes .usdc.
    Strategy:
    - Convert every *.usdc found under extract_dir into a sibling *.usda (same relative path).
    - Rewrite any @*.usdc@/@*.usd@ references in all *.usda to point to the *.usda we created.
    Uses usdcat when available, otherwise pxr.Usd if installed.
    """
    result: Dict[str, object] = {"converted": 0, "failed": 0, "rewritten_usda_files": 0}
    usdc_files = sorted([p for p in extract_dir.rglob("*.usdc") if p.is_file()])
    if not usdc_files:
        return result

    usdcat_bin = shutil.which("usdcat")

    for usdc_path in usdc_files:
        usda_path = usdc_path.with_suffix(".usda")
        if usda_path.exists():
            continue
        try:
            if usdcat_bin:
                proc = subprocess.run(
                    [usdcat_bin, str(usdc_path), "-o", str(usda_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0 or not usda_path.exists():
                    stderr = (proc.stderr or "").strip()
                    raise RuntimeError(stderr or "usdcat failed")
            else:
                if Usd is None:
                    raise RuntimeError(
                        "USD conversion tool is unavailable. Install Pixar USD tools (usdcat) "
                        "or Python package 'usd-core'."
                    )
                stage = Usd.Stage.Open(str(usdc_path))
                if stage is None:
                    raise RuntimeError("Usd.Stage.Open failed")
                usda_path.write_text(stage.GetRootLayer().ExportToString(), encoding="utf-8")
            result["converted"] = int(result["converted"]) + 1
        except Exception:
            result["failed"] = int(result["failed"]) + 1

    # Rewrite references in any existing/new USDA layers.
    rewritten = 0
    for p in extract_dir.rglob("*.usda"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            new = _rewrite_usd_references_to_usda(text)
            if new != text:
                p.write_text(new, encoding="utf-8")
                rewritten += 1
        except OSError:
            continue
    result["rewritten_usda_files"] = rewritten
    return result


def find_balanced_matrix_span(text: str) -> Optional[Tuple[int, int, str]]:
    key = "matrix4d xformOp:transform"
    key_pos = text.find(key)
    if key_pos < 0:
        return None
    eq_pos = text.find("=", key_pos)
    if eq_pos < 0:
        return None
    start = text.find("(", eq_pos)
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return start, i + 1, text[start : i + 1]
    return None


def parse_matrix(text: str) -> Optional[np.ndarray]:
    span = find_balanced_matrix_span(text)
    if not span:
        return None
    _, _, matrix_text = span
    nums = [
        float(x)
        for x in re.findall(
            r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
            matrix_text,
        )
    ]
    if len(nums) != 16:
        return None
    return np.array(nums, dtype=float).reshape(4, 4)


def replace_matrix_text(text: str, mat: np.ndarray) -> str:
    span = find_balanced_matrix_span(text)
    if not span:
        raise ValueError("matrix4d xformOp:transform not found")
    start, end, _ = span
    return text[:start] + format_matrix(mat) + text[end:]


def parse_points(text: str) -> Optional[np.ndarray]:
    m = POINTS_RE.search(text)
    if not m:
        return None
    nums = [
        float(x)
        for x in re.findall(
            r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
            m.group(1),
        )
    ]
    if len(nums) < 3 or len(nums) % 3 != 0:
        return None
    return np.array(nums, dtype=float).reshape(-1, 3)


def replace_points_text(text: str, pts: np.ndarray) -> str:
    m = POINTS_RE.search(text)
    if not m:
        raise ValueError("point3f[] points not found")
    body = ", ".join(f"({p[0]:.6g}, {p[1]:.6g}, {p[2]:.6g})" for p in pts)
    replacement = f"point3f[] points = [{body}]"
    return text[: m.start()] + replacement + text[m.end() :]


def load_replacement_state(session_path: Path) -> Dict:
    state_path = session_path / REPLACEMENT_STATE_FILE
    if not state_path.exists():
        return {"objects": {}}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        objects = data.get("objects")
        if isinstance(objects, dict):
            return {"objects": objects}
    except (OSError, json.JSONDecodeError):
        pass
    return {"objects": {}}


def save_replacement_state(session_path: Path, state: Dict) -> None:
    state_path = session_path / REPLACEMENT_STATE_FILE
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_object_tags_state(session_path: Path) -> Dict:
    state_path = session_path / OBJECT_TAGS_STATE_FILE
    if not state_path.exists():
        return {"objects": {}}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        objects = data.get("objects")
        if isinstance(objects, dict):
            return {"objects": objects}
    except (OSError, json.JSONDecodeError):
        pass
    return {"objects": {}}


def save_object_tags_state(session_path: Path, state: Dict) -> None:
    state_path = session_path / OBJECT_TAGS_STATE_FILE
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_object_tag_list(tags) -> List[str]:
    if not isinstance(tags, list):
        return []
    out: List[str] = []
    for raw in tags:
        tag = str(raw or "").strip().lower()
        if tag not in LAYER_DEFINITIONS or tag in out:
            continue
        out.append(tag)
    return out


def roomplan_category_for_layer(layer_id: str) -> str:
    return LAYER_TO_ROOMPLAN_CATEGORY.get(str(layer_id or "").strip().lower(), "Object")


def replace_category_text(text: str, category: Optional[str]) -> str:
    cat_value = str(category or "").strip()
    if not cat_value:
        return re.sub(r"\n[ \t]*string Category = \"[^\"]*\"[ \t]*", "\n", text)
    match = CATEGORY_RE.search(text)
    if match:
        return text[: match.start(1)] + cat_value + text[match.end(1) :]
    custom_data_re = re.compile(r"(customData\s*=\s*\{\s*\n)")
    cm = custom_data_re.search(text)
    if cm:
        insert = f'        string Category = "{cat_value}"\n'
        return text[: cm.end()] + insert + text[cm.end() :]
    return text


def effective_tags_for_object(inferred_layer: str, override: Optional[Dict]) -> List[str]:
    if isinstance(override, dict) and "tags" in override:
        return normalize_object_tag_list(override.get("tags"))
    layer_id = str(inferred_layer or "other").strip().lower()
    if layer_id in LAYER_DEFINITIONS:
        return [layer_id]
    return ["other"]


def apply_object_tag_overrides(objects: List[Dict], session_path: Path) -> None:
    state = load_object_tags_state(session_path)
    state_objects = state.get("objects", {})
    if not isinstance(state_objects, dict):
        state_objects = {}
    for obj in objects:
        inferred_layer = str(
            obj.get("inferred_layer")
            or infer_layer_from_object(
                str(obj.get("name") or ""),
                str(obj.get("path") or obj.get("id") or ""),
                str(obj.get("category") or ""),
            )
        ).strip().lower()
        if inferred_layer not in LAYER_DEFINITIONS:
            inferred_layer = "other"
        obj["inferred_layer"] = inferred_layer
        obj["inferred_tags"] = [inferred_layer]

        override = state_objects.get(obj.get("id"))
        tags = effective_tags_for_object(inferred_layer, override if isinstance(override, dict) else None)
        obj["tags"] = tags
        obj["tags_overridden"] = isinstance(override, dict) and "tags" in override
        primary = tags[0] if tags else "other"
        obj["layer"] = primary
        if tags:
            obj["category"] = roomplan_category_for_layer(primary)


def _normalize_object_base_name(name: str) -> str:
    text = str(name or "").strip()
    if "(proxy)" in text.lower() and " - " in text:
        text = text.split(" - ")[-1].strip()
    return text


def object_rel_path(obj: Dict) -> str:
    return str(obj.get("path") or obj.get("id") or "").replace("\\", "/").lstrip("/")


def is_assets_object_path(rel_path: str) -> bool:
    rel = str(rel_path or "").replace("\\", "/").lstrip("/")
    return rel.startswith("assets/")


def is_assets_object(obj: Dict) -> bool:
    return is_assets_object_path(object_rel_path(obj))


def infer_layer_from_object(name: str, path: str, category: str) -> str:
    base = _normalize_object_base_name(name)
    lower = base.lower()
    for prefix, layer_id in LAYER_NAME_PREFIXES:
        if lower.startswith(prefix):
            return layer_id

    path_lower = str(path or "").lower()
    for token, layer_id in (
        ("/floors/", "floor"),
        ("/walls/", "wall"),
        ("/doors/", "door"),
        ("/windows/", "window"),
        ("/openings/", "opening"),
        ("/chair/", "chair"),
        ("/table/", "table"),
        ("/storage/", "storage"),
    ):
        if token in path_lower:
            return layer_id

    cat = str(category or "").strip().lower()
    cat_map = {
        "floor": "floor",
        "floors": "floor",
        "wall": "wall",
        "walls": "wall",
        "door": "door",
        "doors": "door",
        "window": "window",
        "windows": "window",
        "opening": "opening",
        "openings": "opening",
        "chair": "chair",
        "table": "table",
        "storage": "storage",
    }
    return cat_map.get(cat, "other")


def load_layer_state(session_path: Path) -> Dict[str, Dict]:
    state_path = session_path / LAYER_STATE_FILE
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        layers = data.get("layers")
        if isinstance(layers, dict):
            return layers
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_layer_state(session_path: Path, state: Dict[str, Dict]) -> None:
    state_path = session_path / LAYER_STATE_FILE
    state_path.write_text(json.dumps({"layers": state}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dxf_layer_state(session_path: Path) -> Dict[str, Dict]:
    state_path = session_path / DXF_LAYER_STATE_FILE
    if not state_path.is_file():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        layers = data.get("layers")
        if isinstance(layers, dict):
            return layers
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_dxf_layer_state(session_path: Path, state: Dict[str, Dict]) -> None:
    state_path = session_path / DXF_LAYER_STATE_FILE
    state_path.write_text(json.dumps({"layers": state}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dxf_layers_payload(session_path: Path, geometry: Dict) -> List[Dict]:
    layers = geometry.get("layers") or []
    if not layers:
        return []
    overrides = load_dxf_layer_state(session_path)
    payload: List[Dict] = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_id = str(layer.get("id") or "").strip()
        if not layer_id:
            continue
        override = overrides.get(layer_id, {}) if isinstance(overrides.get(layer_id), dict) else {}
        payload.append(
            {
                "id": layer_id,
                "label": str(layer.get("label") or layer_id),
                "segment_count": int(layer.get("segment_count") or 0),
                "visible": bool(override.get("visible", True)),
            }
        )
    return payload


def merge_dxf_layers_with_geometry(session_path: Path, geometry: Dict) -> List[Dict]:
    layer_geom = {
        str(layer.get("id") or ""): layer
        for layer in (geometry.get("layers") or [])
        if isinstance(layer, dict) and layer.get("id")
    }
    merged: List[Dict] = []
    for meta in build_dxf_layers_payload(session_path, geometry):
        geom = layer_geom.get(meta["id"]) or {}
        positions = geom.get("positions") or []
        merged.append({**meta, "positions": positions})
    return merged


def build_layers_payload(session_path: Path, objects: List[Dict]) -> List[Dict]:
    counts: Dict[str, int] = {}
    for obj in objects:
        if not is_assets_object(obj):
            continue
        tags = normalize_object_tag_list(obj.get("tags"))
        if not tags:
            layer_id = str(obj.get("layer") or "other").strip().lower()
            tags = [layer_id if layer_id in LAYER_DEFINITIONS else "other"]
        for layer_id in tags:
            counts[layer_id] = counts.get(layer_id, 0) + 1

    overrides = load_layer_state(session_path)
    payload: List[Dict] = []
    for layer_id, meta in sorted(LAYER_DEFINITIONS.items(), key=lambda item: item[1]["z_order"]):
        if layer_id == "other":
            continue
        object_count = counts.get(layer_id, 0)
        if object_count <= 0:
            continue
        override = overrides.get(layer_id, {}) if isinstance(overrides.get(layer_id), dict) else {}
        payload.append(
            {
                "id": layer_id,
                "label": meta["label"],
                "z_order": meta["z_order"],
                "visible": bool(override.get("visible", True)),
                "locked": bool(override.get("locked", meta["default_locked"])),
                "object_count": object_count,
            }
        )
    return payload


def is_floor_or_wall_object(obj: Dict) -> bool:
    category = str(obj.get("category") or "").strip().lower()
    if category in ("floor", "floors", "wall", "walls"):
        return True
    name = str(obj.get("name") or "").strip().lower()
    return name in ("floor", "wall")


def object_has_tag(obj: Dict, tag_id: str) -> bool:
    tag = str(tag_id or "").strip().lower()
    if tag not in LAYER_DEFINITIONS:
        return False
    tags = normalize_object_tag_list(obj.get("tags"))
    if not tags:
        layer = str(obj.get("layer") or "").strip().lower()
        tags = [layer] if layer in LAYER_DEFINITIONS else []
    return tag in tags


def filter_objects_by_tag(objects: List[Dict], tag_id: str) -> List[Dict]:
    return [o for o in objects if isinstance(o, dict) and object_has_tag(o, tag_id)]


def apply_proxy_replacement_overrides(objects: List[Dict], session_path: Path) -> None:
    state = load_replacement_state(session_path)
    state_objects = state.get("objects", {})
    if not isinstance(state_objects, dict):
        return
    for obj in objects:
        obj_id = obj.get("id")
        if not isinstance(obj_id, str):
            continue
        entry = state_objects.get(obj_id)
        if not isinstance(entry, dict):
            continue
        texture_key = str(entry.get("texture_asset_key") or "").strip().lower()
        if texture_key in TEXTURE_ASSET_MAP and is_floor_or_wall_object(obj):
            obj["texture_asset_key"] = texture_key


def apply_replacement_display_metadata(objects: List[Dict], session_path: Path) -> None:
    state = load_replacement_state(session_path)
    state_objects = state.get("objects", {})
    if not isinstance(state_objects, dict):
        return
    for obj in objects:
        obj_id = obj.get("id")
        if not isinstance(obj_id, str):
            continue
        entry = state_objects.get(obj_id)
        if not isinstance(entry, dict):
            continue
        asset_key = str(entry.get("asset_key") or "").strip().lower()
        if asset_key in REPLACEMENT_ASSET_MAP:
            obj["replacement_asset_key"] = asset_key


def load_replacement_template_text(asset_name: str) -> str:
    asset_path = REPLACEMENTS_DIR / asset_name
    if not asset_path.exists():
        raise FileNotFoundError(f"replacement asset not found: {asset_name}")
    with zipfile.ZipFile(asset_path, "r") as zf:
        members = zf.namelist()
        usda_members = sorted([n for n in members if n.lower().endswith(".usda")])
        if usda_members:
            raw = zf.read(usda_members[0])
            return raw.decode("utf-8", errors="ignore")

        usdc_members = sorted([n for n in members if n.lower().endswith(".usdc")])
        if not usdc_members:
            raise ValueError(f"replacement asset has no usda/usdc file: {asset_name}")
        with tempfile.TemporaryDirectory(prefix="topviewer-usd-") as td:
            tmp_dir = Path(td)
            usdc_in = tmp_dir / "in.usdc"
            usda_out = tmp_dir / "out.usda"
            usdc_in.write_bytes(zf.read(usdc_members[0]))
            usdcat_bin = shutil.which("usdcat")
            if usdcat_bin:
                proc = subprocess.run(
                    [usdcat_bin, str(usdc_in), "-o", str(usda_out)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode == 0 and usda_out.exists():
                    return usda_out.read_text(encoding="utf-8", errors="ignore")
                stderr = (proc.stderr or "").strip()
                raise RuntimeError(f"usdcat conversion failed for {asset_name}: {stderr or 'unknown error'}")

            if Usd is None:
                raise RuntimeError(
                    "USD conversion tool is unavailable. Install Pixar USD tools (usdcat) "
                    "or Python package 'usd-core'."
                )
            stage = Usd.Stage.Open(str(usdc_in))
            if stage is None:
                raise RuntimeError(f"failed to open usdc replacement asset: {asset_name}")
            return stage.GetRootLayer().ExportToString()


def build_replacement_usda(target_text: str, template_text: str) -> str:
    target_matrix = parse_matrix(target_text)
    if target_matrix is None:
        raise ValueError("matrix4d xformOp:transform not found")
    target_points = parse_points(target_text)
    if target_points is None:
        raise ValueError("point3f[] points not found on target object")
    template_points = parse_points(template_text)
    if template_points is None:
        raise ValueError("point3f[] points not found in replacement asset")

    # Resize replacement geometry to target BB size and center before swapping.
    target_min = target_points.min(axis=0)
    target_max = target_points.max(axis=0)
    target_dims = np.maximum(np.abs(target_max - target_min), 1e-6)
    target_center = (target_min + target_max) / 2.0

    template_min = template_points.min(axis=0)
    template_max = template_points.max(axis=0)
    template_dims = np.maximum(np.abs(template_max - template_min), 1e-6)
    template_center = (template_min + template_max) / 2.0

    scale = target_dims / template_dims
    resized_template_points = (template_points - template_center) * scale + target_center

    # Keep target transform/material/meta and swap only geometry points payload.
    replaced = replace_points_text(target_text, resized_template_points)
    replaced = replace_matrix_text(replaced, target_matrix)
    return replaced


def replacement_backup_path(session_path: Path, obj_id: str) -> Path:
    safe = obj_id.replace("/", "__").replace("\\", "__")
    backup_dir = session_path / "replacement_backups"
    backup_dir.mkdir(exist_ok=True)
    return backup_dir / f"{safe}.usda"


def preserve_usda_original(session_path: Path, obj_id: str) -> None:
    usda_path = session_path / "extract" / obj_id
    if not usda_path.is_file():
        return
    backup_path = replacement_backup_path(session_path, obj_id)
    if backup_path.is_file():
        usda_path.write_text(backup_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return
    backup_path.write_text(usda_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")


def apply_usdz_object_replacement(session_path: Path, obj_id: str, asset_key: str) -> None:
    if asset_key not in REPLACEMENT_ASSET_MAP:
        raise ValueError(f"invalid replacement asset: {asset_key}")
    usda_path = session_path / "extract" / obj_id
    if not usda_path.is_file():
        raise FileNotFoundError(f"object file not found: {obj_id}")
    preserve_usda_original(session_path, obj_id)


def restore_usdz_object_replacement(session_path: Path, obj_id: str) -> bool:
    backup_path = replacement_backup_path(session_path, obj_id)
    usda_path = session_path / "extract" / obj_id
    if not backup_path.is_file() or not usda_path.is_file():
        return False
    usda_path.write_text(backup_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    backup_path.unlink(missing_ok=True)
    return True


def repack_session_usdz(session_path: Path) -> str:
    out = session_path / "current.usdz"
    repack_usdz(session_path / "extract", out)
    return f"/api/usdz/{session_path.name}/current.usdz"


def parse_bbox(text: str) -> Tuple[List[float], List[float]]:
    pts = parse_points(text)
    if pts is None:
        return [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]

    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    dims = np.maximum(np.abs(pmax - pmin), 0.05)
    center = (pmin + pmax) / 2.0
    return [float(v) for v in dims], [float(v) for v in center]


def transform_local_point(mat: np.ndarray, local: List[float]) -> List[float]:
    p = np.array([local[0], local[1], local[2], 1.0], dtype=float)
    world = p @ mat
    return [float(world[0]), float(world[1]), float(world[2])]


def matrix_translation(mat: np.ndarray) -> List[float]:
    return [float(mat[3, 0]), float(mat[3, 1]), float(mat[3, 2])]


def matrix_to_yaw(mat: np.ndarray) -> float:
    return float(math.degrees(math.atan2(mat[0, 2], mat[0, 0])))


def matrix_to_quaternion_xyzw(mat: np.ndarray) -> List[float]:
    rot_col = mat[0:3, 0:3].T
    trace = float(rot_col[0, 0] + rot_col[1, 1] + rot_col[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot_col[2, 1] - rot_col[1, 2]) / s
        y = (rot_col[0, 2] - rot_col[2, 0]) / s
        z = (rot_col[1, 0] - rot_col[0, 1]) / s
    elif rot_col[0, 0] > rot_col[1, 1] and rot_col[0, 0] > rot_col[2, 2]:
        s = math.sqrt(1.0 + rot_col[0, 0] - rot_col[1, 1] - rot_col[2, 2]) * 2.0
        w = (rot_col[2, 1] - rot_col[1, 2]) / s
        x = 0.25 * s
        y = (rot_col[0, 1] + rot_col[1, 0]) / s
        z = (rot_col[0, 2] + rot_col[2, 0]) / s
    elif rot_col[1, 1] > rot_col[2, 2]:
        s = math.sqrt(1.0 + rot_col[1, 1] - rot_col[0, 0] - rot_col[2, 2]) * 2.0
        w = (rot_col[0, 2] - rot_col[2, 0]) / s
        x = (rot_col[0, 1] + rot_col[1, 0]) / s
        y = 0.25 * s
        z = (rot_col[1, 2] + rot_col[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot_col[2, 2] - rot_col[0, 0] - rot_col[1, 1]) * 2.0
        w = (rot_col[1, 0] - rot_col[0, 1]) / s
        x = (rot_col[0, 2] + rot_col[2, 0]) / s
        y = (rot_col[1, 2] + rot_col[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    q /= norm
    return [float(v) for v in q]


def quaternion_xyzw_to_row_rotation(quat_xyzw: List[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    n = x * x + y * y + z * z + w * w
    if n <= 1e-12:
        return np.eye(3, dtype=float)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    rot_col = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )
    return rot_col.T


def get_basis_scales(mat: np.ndarray) -> Tuple[float, float, float]:
    sx = float(np.linalg.norm(mat[0, 0:3])) or 1.0
    sy = float(np.linalg.norm(mat[1, 0:3])) or 1.0
    sz = float(np.linalg.norm(mat[2, 0:3])) or 1.0
    return sx, sy, sz


def update_matrix_from_position_quaternion(
    old: np.ndarray, matrix_position: List[float], quaternion_xyzw: Optional[List[float]]
) -> np.ndarray:
    new = old.copy()
    scales = get_basis_scales(old)
    if quaternion_xyzw is not None:
        rot_row = quaternion_xyzw_to_row_rotation(quaternion_xyzw)
        new[0, 0:3] = rot_row[0, :] * scales[0]
        new[1, 0:3] = rot_row[1, :] * scales[1]
        new[2, 0:3] = rot_row[2, :] * scales[2]
    new[3, 0:3] = np.array(matrix_position, dtype=float)
    new[3, 3] = 1.0
    return new


def format_matrix(mat: np.ndarray) -> str:
    rows = []
    for r in range(4):
        rows.append("(" + ", ".join(f"{mat[r, c]:.12g}" for c in range(4)) + ")")
    return "( " + ", ".join(rows) + " )"


def infer_category_from_path(path: Path) -> str:
    parts = path.parts
    for name in ["Chair", "Table", "Storage", "Floors", "Walls", "Doors", "Windows", "Openings"]:
        if name in parts:
            if name.endswith("s"):
                return name[:-1]
            return name
    return "Object"


def find_objects(extract_dir: Path) -> List[Dict]:
    objects = []
    for p in sorted(extract_dir.rglob("*.usda")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        mat = parse_matrix(text)
        # Some USDZ exports (e.g. scan meshes) store geometry but no matrix4d xformOp:transform.
        # In that case, fall back to treating the mesh as one object in identity transform space
        # so we can still generate a top-down floorplan from points/bounds.
        has_points = parse_points(text) is not None
        if mat is None and not has_points:
            continue
        if mat is None:
            mat = np.eye(4, dtype=float)

        rel = p.relative_to(extract_dir).as_posix()
        if not is_assets_object_path(rel):
            continue
        cat_match = CATEGORY_RE.search(text)
        uid_match = UUID_RE.search(text)
        name_match = XFORM_NAME_RE.search(text)

        category = cat_match.group(1) if cat_match else infer_category_from_path(p)
        name = name_match.group(1) if name_match else p.stem
        uuid_value = uid_match.group(1) if uid_match else ""
        dimensions, local_bbox_center = parse_bbox(text)
        matrix_pos = matrix_translation(mat)
        display_pos = transform_local_point(mat, local_bbox_center)
        yaw = matrix_to_yaw(mat)
        quaternion_xyzw = matrix_to_quaternion_xyzw(mat)

        inferred_layer = infer_layer_from_object(name, rel, category)
        obj = {
            "id": rel,
            "name": name,
            "category": category,
            "inferred_layer": inferred_layer,
            "layer": inferred_layer,
            "uuid": uuid_value,
            "path": rel,
            "matrix_position": matrix_pos,
            "display_position": display_pos,
            "position": matrix_pos,
            "local_bbox_center": local_bbox_center,
            "yaw_deg": yaw,
            "quaternion_xyzw": quaternion_xyzw,
            "dimensions": dimensions,
        }
        obj.update(compute_world_basis_fields(obj))
        objects.append(obj)
    return objects


def write_manifest(session_path: Path) -> List[Dict]:
    normalize_extract_usda_for_usdz_loader(session_path / "extract")
    objects = find_objects(session_path / "extract")
    apply_object_tag_overrides(objects, session_path)
    apply_proxy_replacement_overrides(objects, session_path)
    apply_replacement_display_metadata(objects, session_path)
    basis = choose_common_projection_basis(objects)
    for obj in objects:
        obj.update(compute_common_plane_rectangle_fields(obj, basis))
    manifest = {
        "objects": objects,
        "world_planar_basis": {
            "origin": [float(v) for v in basis["origin"]],
            "normal": [float(v) for v in basis["normal"]],
            "u_axis": [float(v) for v in basis["u_axis"]],
            "v_axis": [float(v) for v in basis["v_axis"]],
        },
    }
    (session_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return objects


def floorplan_payload(objects: List[Dict], basis: Optional[Dict] = None) -> Dict:
    items: List[Dict] = []
    min_u, max_u = float("inf"), float("-inf")
    min_v, max_v = float("inf"), float("-inf")

    def bump_bounds(points: List[Tuple[float, float]]) -> None:
        nonlocal min_u, max_u, min_v, max_v
        for u, v in points:
            min_u = min(min_u, float(u))
            max_u = max(max_u, float(u))
            min_v = min(min_v, float(v))
            max_v = max(max_v, float(v))

    for o in objects:
        footprint = object_footprint_projected_2d(o)
        if len(footprint) < 3:
            continue
        corners = [[float(p[0]), float(p[1])] for p in footprint]
        bump_bounds(footprint)

        rect = o.get("world_planar_rect") if isinstance(o.get("world_planar_rect"), dict) else {}
        long_axis = rect.get("long_axis") if isinstance(rect.get("long_axis"), dict) else {}
        short_axis = rect.get("short_axis") if isinstance(rect.get("short_axis"), dict) else {}
        center = rect.get("center")
        if isinstance(center, list) and len(center) >= 2:
            center_uv = [float(center[0]), float(center[1])]
        else:
            center_uv = [
                sum(c[0] for c in corners) / len(corners),
                sum(c[1] for c in corners) / len(corners),
            ]

        dims = o.get("dimensions") or [0.5, 0.5, 0.5]
        long_m = float(long_axis.get("length") or max(abs(float(dims[0])), 0.05))
        short_m = float(short_axis.get("length") or max(abs(float(dims[2])), 0.05))
        if short_m > long_m:
            long_m, short_m = short_m, long_m
        angle_deg = float(long_axis.get("angle_deg") if long_axis.get("angle_deg") is not None else (o.get("yaw_deg") or 0.0))

        items.append(
            {
                "id": o.get("id"),
                "name": o.get("name") or o.get("id"),
                "category": o.get("category"),
                "corners": corners,
                "center": center_uv,
                "long_m": long_m,
                "short_m": short_m,
                "long_angle_deg": angle_deg,
                "dimensions": [max(abs(float(d)), 0.05) for d in dims[:3]],
            }
        )

    if not items:
        min_u, max_u, min_v, max_v = -1.0, 1.0, -1.0, 1.0

    bounds = {
        "min_u": min_u,
        "max_u": max_u,
        "min_v": min_v,
        "max_v": max_v,
        # backward-compatible aliases (planar u/v mapped to legacy x/z keys)
        "min_x": min_u,
        "max_x": max_u,
        "min_z": min_v,
        "max_z": max_v,
    }
    payload: Dict = {
        "objects": items,
        "bounds": bounds,
        "coordinate_space": "world_planar",
    }
    if basis:
        payload["basis"] = basis
    return payload


def convex_hull_2d(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) <= 1:
        return points
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def world_z_to_plan_v(z: float) -> float:
    """Map world Z to floor-plan v (top-down from +Y uses v = -Z)."""
    return -float(z)


def world_xz_to_plan_uv(x: float, z: float) -> Tuple[float, float]:
    return float(x), world_z_to_plan_v(z)


def world_xz_poly_to_plan_uv(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [world_xz_to_plan_uv(x, z) for x, z in points]


def object_footprint_xz(obj: Dict) -> List[Tuple[float, float]]:
    dims = obj.get("dimensions") or [0.5, 0.5, 0.5]
    hx = max(abs(float(dims[0])), 0.05) * 0.5
    hy = max(abs(float(dims[1])), 0.05) * 0.5
    hz = max(abs(float(dims[2])), 0.05) * 0.5
    pos = obj.get("matrix_position") or obj.get("position") or [0.0, 0.0, 0.0]
    px, py, pz = [float(v) for v in pos]

    q = obj.get("quaternion_xyzw") or [0.0, 0.0, 0.0, 1.0]
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n <= 1e-12:
        rot = np.eye(3, dtype=float)
    else:
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xzv, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        rot = np.array(
            [
                [1.0 - (yy + zz), xy - wz, xzv + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xzv - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dtype=float,
        )

    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * hx, sy * hy, sz * hz], dtype=float)
                world = local @ rot + np.array([px, py, pz], dtype=float)
                corners.append((float(world[0]), float(world[2])))

    hull = convex_hull_2d(corners)
    if len(hull) >= 3:
        return hull
    min_x = min(p[0] for p in corners)
    max_x = max(p[0] for p in corners)
    min_z = min(p[1] for p in corners)
    max_z = max(p[1] for p in corners)
    eps = 1e-3
    return [(min_x - eps, min_z - eps), (max_x + eps, min_z - eps), (max_x + eps, max_z + eps), (min_x - eps, max_z + eps)]


def object_centerline_xz(obj: Dict) -> Tuple[np.ndarray, np.ndarray]:
    rect = obj.get("world_planar_rect")
    if (
        isinstance(rect, dict)
        and isinstance(rect.get("long_axis"), dict)
        and isinstance(rect["long_axis"].get("a"), list)
        and len(rect["long_axis"]["a"]) >= 2
        and isinstance(rect["long_axis"].get("b"), list)
        and len(rect["long_axis"]["b"]) >= 2
    ):
        a = rect["long_axis"]["a"]
        b = rect["long_axis"]["b"]
        return np.array([float(a[0]), float(a[1])], dtype=float), np.array([float(b[0]), float(b[1])], dtype=float)

    wb = obj.get("world_basis_long_axis_xz")
    if (
        isinstance(wb, dict)
        and isinstance(wb.get("a"), list)
        and len(wb.get("a")) >= 2
        and isinstance(wb.get("b"), list)
        and len(wb.get("b")) >= 2
    ):
        a = wb["a"]
        b = wb["b"]
        return (
            np.array(world_xz_to_plan_uv(float(a[0]), float(a[1])), dtype=float),
            np.array(world_xz_to_plan_uv(float(b[0]), float(b[1])), dtype=float),
        )

    center, direction, edge_len = longest_axis_from_footprint_xz(obj)
    half_len = max(edge_len * 0.5, 0.025)
    a = center - direction * half_len
    b = center + direction * half_len
    return (
        np.array(world_xz_to_plan_uv(float(a[0]), float(a[1])), dtype=float),
        np.array(world_xz_to_plan_uv(float(b[0]), float(b[1])), dtype=float),
    )


def longest_axis_from_footprint_xz(obj: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    poly = object_footprint_xz(obj)
    if len(poly) < 2:
        pos = obj.get("matrix_position") or obj.get("position") or [0.0, 0.0, 0.0]
        center = np.array([float(pos[0]), float(pos[2])], dtype=float)
        return center, np.array([1.0, 0.0], dtype=float), 0.05

    points = [np.array(p, dtype=float) for p in poly]
    center = np.mean(np.array(points), axis=0)
    best_len = -1.0
    best_dir = np.array([1.0, 0.0], dtype=float)
    n = len(points)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        edge = b - a
        seg_len = float(np.linalg.norm(edge))
        if seg_len <= 1e-9:
            continue
        if seg_len > best_len:
            best_len = seg_len
            best_dir = edge / seg_len

    if best_len <= 0:
        best_len = 0.05
    return center.astype(float), best_dir.astype(float), float(best_len)


def compute_world_basis_fields(obj: Dict) -> Dict:
    center, direction, edge_len = longest_axis_from_footprint_xz(obj)
    half_len = max(edge_len * 0.5, 0.025)
    a = center - direction * half_len
    b = center + direction * half_len
    angle_deg = float(math.degrees(math.atan2(direction[1], direction[0])))
    pos = obj.get("matrix_position") or obj.get("position") or [0.0, 0.0, 0.0]
    return {
        "world_basis_position": [float(pos[0]), float(pos[1]), float(pos[2])],
        "world_basis_footprint_xz": [[float(x), float(z)] for (x, z) in object_footprint_xz(obj)],
        "world_basis_long_axis_xz": {
            "a": [float(a[0]), float(a[1])],
            "b": [float(b[0]), float(b[1])],
            "dir": [float(direction[0]), float(direction[1])],
            "length": float(edge_len),
            "angle_deg": angle_deg,
        },
    }


def local_points_to_world(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    rot = mat[0:3, 0:3]
    trans = mat[3, 0:3]
    return points @ rot + trans


def world_points_to_local(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    rot = mat[0:3, 0:3]
    trans = mat[3, 0:3]
    rot_inv = np.linalg.inv(rot)
    return (points - trans) @ rot_inv


def compute_floor_top_world_y(objects: List[Dict]) -> Optional[float]:
    ys: List[float] = []
    for obj in objects:
        if not is_floor_object(obj):
            continue
        for corner in object_world_corners_xyz(obj):
            ys.append(float(corner[1]))
    if not ys:
        return None
    return float(max(ys))


def stretch_mesh_bottom_to_floor_y(text: str, floor_top_y: float) -> Tuple[bool, str]:
    """Extend bottom vertices in world +Y until the mesh bottom meets floor_top_y. Matrix unchanged."""
    mat = parse_matrix(text)
    pts = parse_points(text)
    if mat is None or pts is None or len(pts) == 0:
        return False, text

    world = local_points_to_world(pts, mat)
    bottom_y = float(np.min(world[:, 1]))
    gap = float(floor_top_y) - bottom_y
    if abs(gap) < 1e-4:
        return False, text

    span = float(np.max(world[:, 1]) - bottom_y)
    tol = max(1e-4, span * 0.08)
    mask = world[:, 1] <= bottom_y + tol
    if not np.any(mask):
        return False, text

    world = world.copy()
    world[mask, 1] += gap
    new_pts = world_points_to_local(world, mat)
    return True, replace_points_text(text, new_pts)


def object_world_corners_xyz(obj: Dict) -> List[np.ndarray]:
    dims = obj.get("dimensions") or [0.5, 0.5, 0.5]
    hx = max(abs(float(dims[0])), 0.05) * 0.5
    hy = max(abs(float(dims[1])), 0.05) * 0.5
    hz = max(abs(float(dims[2])), 0.05) * 0.5
    pos = obj.get("matrix_position") or obj.get("position") or [0.0, 0.0, 0.0]
    t = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=float)
    rot = quaternion_xyzw_to_row_rotation(obj.get("quaternion_xyzw") or [0.0, 0.0, 0.0, 1.0])
    corners: List[np.ndarray] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * hx, sy * hy, sz * hz], dtype=float)
                world = local @ rot + t
                corners.append(world.astype(float))
    return corners


def _safe_unit3(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return (v / n).astype(float)


def _safe_unit2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return (v / n).astype(float)


def choose_common_projection_basis(objects: List[Dict]) -> Dict[str, np.ndarray]:
    floors = [o for o in objects if is_floor_object(o)]
    target = None
    best_area = -1.0
    for f in floors:
        area = polygon_area_abs_2d(object_footprint_xz(f))
        if area > best_area:
            best_area = area
            target = f

    normal = np.array([0.0, 1.0, 0.0], dtype=float)
    if target is not None:
        rot = quaternion_xyzw_to_row_rotation(target.get("quaternion_xyzw") or [0.0, 0.0, 0.0, 1.0])
        candidates = [
            np.array([1.0, 0.0, 0.0], dtype=float) @ rot,
            np.array([0.0, 1.0, 0.0], dtype=float) @ rot,
            np.array([0.0, 0.0, 1.0], dtype=float) @ rot,
        ]
        best_idx = int(np.argmax([abs(float(c[1])) for c in candidates]))
        normal = candidates[best_idx]
        if float(normal[1]) < 0:
            normal = -normal
        normal = _safe_unit3(normal)

    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, normal))) >= 0.95:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
    u_axis = ref - float(np.dot(ref, normal)) * normal
    u_axis = _safe_unit3(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis = _safe_unit3(v_axis)
    return {
        "origin": np.array([0.0, 0.0, 0.0], dtype=float),
        "normal": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
    }


def polygon_area_abs_2d(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(s) * 0.5


def compute_common_plane_rectangle_fields(obj: Dict, basis: Dict[str, np.ndarray]) -> Dict:
    corners = object_world_corners_xyz(obj)
    u_axis = basis["u_axis"]
    v_axis = basis["v_axis"]
    origin = basis["origin"]

    pts = []
    for p in corners:
        rel = p - origin
        u = float(np.dot(rel, u_axis))
        v = float(np.dot(rel, v_axis))
        pts.append(np.array([u, v], dtype=float))

    center = np.mean(np.array(pts), axis=0)
    mat = np.array(pts) - center
    cov = mat.T @ mat
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    d_long = _safe_unit2(eigvecs[:, order[0]])
    d_short = _safe_unit2(np.array([-d_long[1], d_long[0]], dtype=float))

    proj_long = [float(np.dot(p - center, d_long)) for p in pts]
    proj_short = [float(np.dot(p - center, d_short)) for p in pts]
    long_len = max(max(proj_long) - min(proj_long), 0.05)
    short_len = max(max(proj_short) - min(proj_short), 0.05)
    if short_len > long_len:
        long_len, short_len = short_len, long_len
        d_long, d_short = d_short, d_long
        proj_long, proj_short = proj_short, proj_long

    hl = long_len * 0.5
    hs = short_len * 0.5
    rect = [
        center + d_long * hl + d_short * hs,
        center - d_long * hl + d_short * hs,
        center - d_long * hl - d_short * hs,
        center + d_long * hl - d_short * hs,
    ]
    a = center - d_long * hl
    b = center + d_long * hl
    angle = float(math.degrees(math.atan2(float(d_long[1]), float(d_long[0]))))
    return {
        "world_planar_rect": {
            "center": [float(center[0]), float(center[1])],
            "corners": [[float(p[0]), float(p[1])] for p in rect],
            "long_axis": {
                "a": [float(a[0]), float(a[1])],
                "b": [float(b[0]), float(b[1])],
                "dir": [float(d_long[0]), float(d_long[1])],
                "length": float(long_len),
                "angle_deg": angle,
            },
            "short_axis": {
                "dir": [float(d_short[0]), float(d_short[1])],
                "length": float(short_len),
            },
        }
    }


def object_footprint_projected_2d(obj: Dict) -> List[Tuple[float, float]]:
    rect = obj.get("world_planar_rect")
    if isinstance(rect, dict) and isinstance(rect.get("corners"), list) and len(rect["corners"]) >= 3:
        out = []
        for p in rect["corners"]:
            if isinstance(p, list) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
        if len(out) >= 3:
            return out
    return world_xz_poly_to_plan_uv(object_footprint_xz(obj))


def shift_2d_to_world_xyz(shift_vec: np.ndarray, projection_basis: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if not projection_basis:
        return np.array([float(shift_vec[0]), 0.0, float(shift_vec[1])], dtype=float)
    u_axis = projection_basis["u_axis"]
    v_axis = projection_basis["v_axis"]
    return u_axis * float(shift_vec[0]) + v_axis * float(shift_vec[1])


def closest_point_on_segment_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return a.copy(), 0.0
    t = float(((p - a) @ ab) / denom)
    t = max(0.0, min(1.0, t))
    return a + ab * t, t


def segment_intersection_2d(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> Optional[np.ndarray]:
    r = a1 - a0
    s = b1 - b0
    rxs = float(r[0] * s[1] - r[1] * s[0])
    qp = b0 - a0
    qpxr = float(qp[0] * r[1] - qp[1] * r[0])
    if abs(rxs) <= 1e-12 and abs(qpxr) <= 1e-12:
        return None
    if abs(rxs) <= 1e-12:
        return None
    t = float((qp[0] * s[1] - qp[1] * s[0]) / rxs)
    u = float((qp[0] * r[1] - qp[1] * r[0]) / rxs)
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return a0 + t * r
    return None


def segment_to_segment_distance_2d(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    inter = segment_intersection_2d(a0, a1, b0, b1)
    if inter is not None:
        return 0.0, inter, inter
    candidates: List[Tuple[float, np.ndarray, np.ndarray]] = []
    p_on_b0, _ = closest_point_on_segment_2d(a0, b0, b1)
    candidates.append((float(np.linalg.norm(a0 - p_on_b0)), a0, p_on_b0))
    p_on_b1, _ = closest_point_on_segment_2d(a1, b0, b1)
    candidates.append((float(np.linalg.norm(a1 - p_on_b1)), a1, p_on_b1))
    p_on_a0, _ = closest_point_on_segment_2d(b0, a0, a1)
    candidates.append((float(np.linalg.norm(b0 - p_on_a0)), p_on_a0, b0))
    p_on_a1, _ = closest_point_on_segment_2d(b1, a0, a1)
    candidates.append((float(np.linalg.norm(b1 - p_on_a1)), p_on_a1, b1))
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    return best[0], best[1], best[2]


def build_floor_edges_with_hash(floors: List[Dict]) -> Tuple[List[Dict], Dict[Tuple[int, int], List[int]], float]:
    edges: List[Dict] = []
    lengths = []
    for f in floors:
        poly = object_footprint_projected_2d(f)
        if len(poly) < 2:
            continue
        for i in range(len(poly)):
            a = np.array(poly[i], dtype=float)
            b = np.array(poly[(i + 1) % len(poly)], dtype=float)
            seg_len = float(np.linalg.norm(b - a))
            if seg_len <= 1e-6:
                continue
            lengths.append(seg_len)
            edges.append(
                {
                    "a": a,
                    "b": b,
                    "min_x": min(float(a[0]), float(b[0])),
                    "max_x": max(float(a[0]), float(b[0])),
                    "min_z": min(float(a[1]), float(b[1])),
                    "max_z": max(float(a[1]), float(b[1])),
                }
            )
    if not edges:
        return [], {}, 1.0
    cell = max(float(np.median(lengths)), 0.5)
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, e in enumerate(edges):
        cx0 = int(math.floor(e["min_x"] / cell))
        cx1 = int(math.floor(e["max_x"] / cell))
        cz0 = int(math.floor(e["min_z"] / cell))
        cz1 = int(math.floor(e["max_z"] / cell))
        for ix in range(cx0, cx1 + 1):
            for iz in range(cz0, cz1 + 1):
                grid.setdefault((ix, iz), []).append(idx)
    return edges, grid, cell


# Align: treat as “already on edge” when parallel-line snap translation length is below this.
ALIGN_EXACT_MATCH_ABS_TOL = 1e-12


def _cross2d_scalar(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _unit2(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return None
    return np.asarray(v, dtype=float) / n


# |cross(u,v)| for unit vectors ≈ |sin(angle)|; parallel ⇒ ~0.
PARALLEL_CROSS_TOL = 0.02


def _translation_point_to_line_2d(
    p: np.ndarray, line_origin: np.ndarray, line_dir_unit: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Shortest translation (XZ) that moves point p onto the infinite line through line_origin along line_dir_unit."""
    n = np.array([-line_dir_unit[1], line_dir_unit[0]], dtype=float)
    s = float(np.dot(p - line_origin, n))
    t_vec = -s * n
    return t_vec, float(np.linalg.norm(t_vec))


def best_floor_edge_min_translation_parallel_wall(
    seg_a: np.ndarray, seg_b: np.ndarray, edges: List[Dict]
) -> Optional[Tuple[np.ndarray, float, Dict]]:
    """
    Among floor hull edges whose direction is parallel to the wall centerline (no rotation),
    choose the edge for which translating the wall by t puts the wall centerline onto that edge's
    infinite line, minimizing ||t||.
    Returns (translation_xz, magnitude, chosen_edge) or None if no sufficiently parallel edge exists.
    """
    u_w = _unit2(seg_b - seg_a)
    if u_w is None:
        return None
    best_mag = float("inf")
    best_t: Optional[np.ndarray] = None
    best_edge: Optional[Dict] = None
    for e in edges:
        ev = e["b"] - e["a"]
        u_f = _unit2(ev)
        if u_f is None:
            continue
        if abs(_cross2d_scalar(u_w, u_f)) > PARALLEL_CROSS_TOL:
            continue
        t_vec, mag = _translation_point_to_line_2d(seg_a, e["a"], u_f)
        if mag < best_mag:
            best_mag = mag
            best_t = t_vec
            best_edge = e
    if best_t is None or best_edge is None:
        return None
    return best_t, best_mag, best_edge


def _starts_with_token(value: str, token: str) -> bool:
    s = (value or "").strip().lower()
    t = token.lower()
    if not s:
        return False
    if s == t:
        return True
    return s.startswith(t)


def is_wall_object(obj: Dict) -> bool:
    category = str(obj.get("category", "")).strip().lower()
    if category == "wall":
        return True
    name = str(obj.get("name", ""))
    path = str(obj.get("path", ""))
    obj_id = str(obj.get("id", ""))
    return _starts_with_token(name, "wall") or _starts_with_token(path.split("/")[-1], "wall") or _starts_with_token(obj_id.split("/")[-1], "wall")


def is_floor_object(obj: Dict) -> bool:
    category = str(obj.get("category", "")).strip().lower()
    if category == "floor":
        return True
    name = str(obj.get("name", ""))
    path = str(obj.get("path", ""))
    obj_id = str(obj.get("id", ""))
    return _starts_with_token(name, "floor") or _starts_with_token(path.split("/")[-1], "floor") or _starts_with_token(obj_id.split("/")[-1], "floor")


def describe_align_result(
    moved: int,
    floor_count: int,
    wall_count: int,
    floor_edge_count: int,
    skip: Dict[str, int],
) -> str:
    """Human-readable reason when moved == 0 (Japanese)."""
    if moved > 0:
        return ""
    if floor_count == 0:
        return (
            "床オブジェクトが 0 件です（一覧に Floor0 のように Category または名前が floor で始まるものが必要です）。"
            " 壁だけあっても寄せはできません。"
        )
    if wall_count == 0:
        return "壁オブジェクトが 0 件です。"
    if floor_edge_count == 0:
        return (
            "床の外形から辺を作れませんでした（床メッシュの頂点が読めない、輪郭が潰れている等）。"
            " Floor0 がサイドバーにあってもこの状態になることがあります。"
        )
    sa = int(skip.get("already_aligned", 0))
    npf = int(skip.get("no_parallel_floor_edge", 0))
    mf = int(skip.get("missing_file", 0))
    nm = int(skip.get("no_matrix", 0))
    wid = int(skip.get("invalid_id", 0))
    if sa == wall_count and wall_count > 0:
        return (
            "壁は検出されていますが、壁中心線と平行な床の辺へ載せるために必要な並進がすべて "
            f"{ALIGN_EXACT_MATCH_ABS_TOL:g} 以下です（既に載っていると判定）。寄せの値変更は行いません。"
        )
    parts = []
    if npf > 0:
        parts.append(
            f"壁中心線と平行な床の輪郭辺がなく（回転なしでは同一無限直線に載せられない）、スキップ: {npf} 件。"
            f" 平行判定は |sin(角)|≤{PARALLEL_CROSS_TOL:g}（XZ 上の方向ベクトル）。"
        )
    if mf > 0:
        parts.append(f"USDA パス欠落: {mf} 件。")
    if nm > 0:
        parts.append(f"変換行列の読み取り失敗: {nm} 件。")
    if wid > 0:
        parts.append(f"オブジェクト id 不正: {wid} 件。")
    if parts:
        return "（診断）" + " ".join(parts)
    return (
        "移動の理由を特定できませんでした。"
        f"（壁 {wall_count} 件・床辺 {floor_edge_count} ・skip already_aligned={sa} no_parallel_floor_edge={npf}）"
    )


@app.route("/")
def suite():
    return render_template("suite.html")


@app.route("/editor")
def editor():
    return render_template("suite.html")


@app.route("/simple")
def simple_editor():
    return render_template("suite.html")


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


def _file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_input_source(source: str, *, expected_kind: str) -> Optional[str]:
    spec = INPUT_SOURCE_SPECS.get(source)
    if not spec:
        return f"不明な入力ソース: {source}"
    if spec["kind"] != expected_kind:
        return f"入力ソース「{spec['label']}」はこの形式のアップロードに対応していません"
    return None


def validate_filename_for_source(filename: str, source: str) -> Optional[str]:
    spec = INPUT_SOURCE_SPECS.get(source)
    if not spec:
        return "不明な入力ソースです"
    ext = _file_extension(filename)
    if ext not in spec["extensions"]:
        allowed = ", ".join(spec["extensions"])
        return f"{spec['label']} 用ファイル ({allowed}) を選択してください"
    return None


@app.get("/api/input-sources")
def list_input_sources():
    return jsonify(
        {
            "sources": [
                {
                    "id": source_id,
                    "label": spec["label"],
                    "description": spec["description"],
                    "extensions": spec["extensions"],
                    "kind": spec["kind"],
                }
                for source_id, spec in INPUT_SOURCE_SPECS.items()
            ]
        }
    )


@app.post("/api/upload")
def upload():
    source = str(request.form.get("source") or "").strip()
    source_err = validate_input_source(source, expected_kind="usdz")
    if source_err:
        return jsonify({"error": source_err}), 400

    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "ファイルを選択してください"}), 400

    name_err = validate_filename_for_source(f.filename, source)
    if name_err:
        return jsonify({"error": name_err}), 400

    sid = str(uuid.uuid4())
    sp = SESSION_DIR / sid
    sp.mkdir(parents=True)
    original = sp / "original.usdz"
    f.save(original)
    extract = sp / "extract"
    unpack_usdz(original, extract)
    materialize_usda_from_usdc_in_extract(extract)
    objects = write_manifest(sp)
    out = sp / "current.usdz"
    repack_usdz(extract, out)
    layers = build_layers_payload(sp, objects)
    spec = INPUT_SOURCE_SPECS[source]
    (sp / SESSION_META_FILE).write_text(
        json.dumps(
            {
                "kind": "usdz",
                "input_source": source,
                "input_source_label": spec["label"],
                "original_filename": f.filename,
            }
        ),
        encoding="utf-8",
    )
    return jsonify(
        {
            "session_id": sid,
            "input_source": source,
            "input_source_label": spec["label"],
            "objects": objects,
            "layers": layers,
            "usdz_url": f"/api/usdz/{sid}/current.usdz",
        }
    )


@app.post("/api/upload-dxf")
def upload_dxf():
    source = str(request.form.get("source") or "").strip()
    source_err = validate_input_source(source, expected_kind="dxf")
    if source_err:
        return jsonify({"error": source_err}), 400

    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "ファイルを選択してください"}), 400

    name_err = validate_filename_for_source(f.filename, source)
    if name_err:
        return jsonify({"error": name_err}), 400

    sid = str(uuid.uuid4())
    sp = SESSION_DIR / sid
    sp.mkdir(parents=True)
    original = sp / "original.dxf"
    f.save(original)

    try:
        geometry = parse_dxf_file(original)
    except ValueError as exc:
        shutil.rmtree(sp, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        shutil.rmtree(sp, ignore_errors=True)
        return jsonify({"error": f"DXF parse failed: {exc}"}), 500

    (sp / DXF_GEOMETRY_FILE).write_text(json.dumps(geometry), encoding="utf-8")
    spec = INPUT_SOURCE_SPECS[source]
    (sp / SESSION_META_FILE).write_text(
        json.dumps(
            {
                "kind": "dxf",
                "input_source": source,
                "input_source_label": spec["label"],
                "filename": f.filename,
            }
        ),
        encoding="utf-8",
    )

    dxf_layers = merge_dxf_layers_with_geometry(sp, geometry)

    return jsonify(
        {
            "session_id": sid,
            "kind": "dxf",
            "input_source": source,
            "input_source_label": spec["label"],
            "dxf": {
                "segment_count": geometry["segment_count"],
                "layer_count": geometry.get("layer_count", len(dxf_layers)),
                "bounds": geometry["bounds"],
                "entity_counts": geometry.get("entity_counts", {}),
                "source_filename": geometry.get("source_filename"),
                "unit_scale_to_meters": geometry.get("unit_scale_to_meters"),
            },
            "positions": geometry["positions"],
            "layers": dxf_layers,
        }
    )


@app.post("/api/extract-walls/<session_id>")
def extract_walls(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    meta_path = sp / SESSION_META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "dxf":
            return jsonify({"error": "wall extraction requires a DXF session"}), 400
    geom_path = sp / DXF_GEOMETRY_FILE
    if not geom_path.is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    geometry = json.loads(geom_path.read_text(encoding="utf-8"))
    positions = geometry.get("positions") or []
    if not positions:
        return jsonify({"error": "DXF geometry is empty"}), 400
    try:
        result = extract_walls_from_positions(positions)
    except Exception as exc:
        return jsonify({"error": f"wall extraction failed: {exc}"}), 500
    (sp / WALL_EXTRACT_FILE).write_text(json.dumps(result), encoding="utf-8")
    return jsonify({"session_id": session_id, **result})


@app.post("/api/wall-extrusion/<session_id>")
def run_wall_extrusion(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    meta_path = sp / SESSION_META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "dxf":
            return jsonify({"error": "wall extrusion requires a DXF session"}), 400
    geom_path = sp / DXF_GEOMETRY_FILE
    if not geom_path.is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    geometry = json.loads(geom_path.read_text(encoding="utf-8"))
    positions = geometry.get("positions") or []
    if not positions:
        return jsonify({"error": "DXF geometry is empty"}), 400
    body = request.get_json(silent=True) or {}
    height_m = body.get("height_m", DEFAULT_WALL_HEIGHT_M)
    try:
        height_m = float(height_m)
    except (TypeError, ValueError):
        return jsonify({"error": "height_m must be a number"}), 400
    try:
        extract = extract_walls_from_positions(positions)
    except Exception as exc:
        return jsonify({"error": f"wall extraction failed: {exc}"}), 500
    (sp / WALL_EXTRACT_FILE).write_text(json.dumps(extract), encoding="utf-8")
    if not extract.get("walls"):
        return jsonify({"error": "no walls detected from parallel lines"}), 400
    try:
        extrude = extrude_walls_from_extract(extract, height_m=height_m)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"wall extrusion failed: {exc}"}), 500
    if extrude["mesh_wall_count"] <= 0:
        return jsonify({"error": "no wall mesh generated"}), 400
    (sp / WALL_EXTRUDE_FILE).write_text(json.dumps(extrude), encoding="utf-8")
    return jsonify(
        {
            "session_id": session_id,
            **extrude,
            "source": "heuristic",
            "wall_count": extract.get("wall_count", 0),
            "highlight_positions": extract.get("highlight_positions", []),
        }
    )


@app.post("/api/extract-walls-ml/<session_id>")
def extract_walls_ml(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    meta_path = sp / SESSION_META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "dxf":
            return jsonify({"error": "ML wall extraction requires a DXF session"}), 400
    if not (sp / DXF_GEOMETRY_FILE).is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    if not DEFAULT_ONNX_PATH.is_file():
        return jsonify(
            {
                "error": (
                    "ONNX model not found. Run: "
                    "python scripts/floorplan_ml_poc/export_onnx.py"
                )
            }
        ), 503
    try:
        result = run_ml_wall_extract(sp)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"ML wall extraction failed: {exc}"}), 500
    return jsonify({"session_id": session_id, **result})


@app.get("/api/wall-extract-ml/<session_id>")
def get_wall_extract_ml(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    path = sp / WALL_ML_EXTRACT_FILE
    if not path.is_file():
        return jsonify({"error": "no ML wall extraction for this session"}), 404
    result = json.loads(path.read_text(encoding="utf-8"))
    return jsonify({"session_id": session_id, **result})


@app.post("/api/extrude-walls/<session_id>")
def extrude_walls(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    meta_path = sp / SESSION_META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "dxf":
            return jsonify({"error": "wall extrusion requires a DXF session"}), 400
    body = request.get_json(silent=True) or {}
    height_m = body.get("height_m", DEFAULT_WALL_HEIGHT_M)
    source = body.get("source", "ml")
    try:
        height_m = float(height_m)
    except (TypeError, ValueError):
        return jsonify({"error": "height_m must be a number"}), 400

    if source == "ml":
        if not DEFAULT_ONNX_PATH.is_file():
            return jsonify(
                {
                    "error": (
                        "ONNX model not found. Run: "
                        "python scripts/floorplan_ml_poc/export_onnx.py"
                    )
                }
            ), 503
        try:
            extract = run_ml_wall_extract(sp)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            return jsonify({"error": f"ML wall extraction failed: {exc}"}), 500
    else:
        extract_path = sp / WALL_EXTRACT_FILE
        if not extract_path.is_file():
            return jsonify({"error": "run wall extraction first"}), 400
        extract = json.loads(extract_path.read_text(encoding="utf-8"))

    if not extract.get("walls"):
        return jsonify({"error": "no walls to extrude"}), 400

    try:
        result = extrude_walls_from_extract(extract, height_m=height_m)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"wall extrusion failed: {exc}"}), 500

    if result["mesh_wall_count"] <= 0:
        return jsonify({"error": "no wall mesh generated; re-run ML wall extraction"}), 400

    (sp / WALL_EXTRUDE_FILE).write_text(json.dumps(result), encoding="utf-8")
    return jsonify({"session_id": session_id, **result})


@app.get("/api/wall-extrude/<session_id>")
def get_wall_extrude(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    path = sp / WALL_EXTRUDE_FILE
    if not path.is_file():
        return jsonify({"error": "no wall extrusion for this session"}), 404
    result = json.loads(path.read_text(encoding="utf-8"))
    return jsonify({"session_id": session_id, **result})


@app.get("/api/wall-extract/<session_id>")
def get_wall_extract(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    path = sp / WALL_EXTRACT_FILE
    if not path.is_file():
        return jsonify({"error": "no wall extraction for this session"}), 404
    result = json.loads(path.read_text(encoding="utf-8"))
    return jsonify({"session_id": session_id, **result})


@app.post("/api/ml-poc/<session_id>")
def run_ml_poc(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    meta_path = sp / SESSION_META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "dxf":
            return jsonify({"error": "ML PoC requires a DXF session"}), 400
    if not (sp / DXF_GEOMETRY_FILE).is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    if not DEFAULT_ONNX_PATH.is_file():
        return jsonify(
            {
                "error": (
                    "ONNX model not found. Run: "
                    "pip install -r scripts/floorplan_ml_poc/requirements-poc.txt && "
                    "python scripts/floorplan_ml_poc/export_onnx.py"
                )
            }
        ), 503
    try:
        report = run_ml_poc_compare(sp)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"ML PoC failed: {exc}"}), 500
    return jsonify({"session_id": session_id, **report})


@app.get("/api/ml-poc/<session_id>")
def get_ml_poc_report(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    path = sp / ML_POC_SUBDIR / "compare_report.json"
    if not path.is_file():
        return jsonify({"error": "no ML PoC results for this session"}), 404
    report = json.loads(path.read_text(encoding="utf-8"))
    return jsonify({"session_id": session_id, **report})


@app.get("/api/ml-poc/<session_id>/<path:filename>")
def get_ml_poc_file(session_id: str, filename: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    if filename not in ML_POC_ALLOWED_FILES:
        return jsonify({"error": "file not allowed"}), 400
    path = sp / ML_POC_SUBDIR / filename
    if not path.is_file():
        return jsonify({"error": "file not found"}), 404
    if filename.endswith(".json"):
        return send_file(path, mimetype="application/json")
    return send_file(path, mimetype="image/png")


def _ml_poc_onnx_missing_response():
    return jsonify(
        {
            "error": (
                "ONNX model not found. Run: "
                "pip install -r scripts/floorplan_ml_poc/requirements-poc.txt && "
                "python scripts/floorplan_ml_poc/setup_compare_models.py"
            )
        }
    ), 503


@app.post("/api/ml-poc-render-compare/<session_id>")
def run_ml_poc_render_compare(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    meta_path = sp / SESSION_META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "dxf":
            return jsonify({"error": "render compare requires a DXF session"}), 400
    if not (sp / DXF_GEOMETRY_FILE).is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    if not DEFAULT_ONNX_PATH.is_file():
        return _ml_poc_onnx_missing_response()
    try:
        report = run_render_path_compare(sp)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"render compare failed: {exc}"}), 500
    return jsonify({"session_id": session_id, **report})


@app.get("/api/ml-poc-render-compare/<session_id>")
def get_ml_poc_render_compare_report(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    path = sp / ML_POC_SUBDIR / RENDER_COMPARE_SUBDIR / "compare_renders_report.json"
    if not path.is_file():
        return jsonify({"error": "no render compare results for this session"}), 404
    report = json.loads(path.read_text(encoding="utf-8"))
    return jsonify({"session_id": session_id, **report})


@app.get("/api/ml-poc-render-compare/<session_id>/<path:filename>")
def get_ml_poc_render_compare_file(session_id: str, filename: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    if filename not in ML_RENDER_COMPARE_ALLOWED_FILES:
        return jsonify({"error": "file not allowed"}), 400
    path = sp / ML_POC_SUBDIR / RENDER_COMPARE_SUBDIR / filename
    if not path.is_file():
        return jsonify({"error": "file not found"}), 404
    if filename.endswith(".json"):
        return send_file(path, mimetype="application/json")
    if filename.endswith(".svg"):
        return send_file(path, mimetype="image/svg+xml")
    return send_file(path, mimetype="image/png")


@app.get("/api/ml-models")
def get_ml_models():
    models = list_onnx_models()
    if not models and not DEFAULT_ONNX_PATH.is_file():
        return jsonify(
            {
                "models": [],
                "error": (
                    "No ONNX models found. Run: "
                    "python scripts/floorplan_ml_poc/export_onnx.py"
                ),
            }
        ), 503
    return jsonify({"models": models, "default_model_id": models[0]["id"] if models else None})


def _extrude_compare_variant_allowed(sp: Path, source: str) -> bool:
    allowed = set(list_variant_ids_from_report(sp))
    allowed.update({"heuristic", "heuristic_refined", "heuristic_structural", "ml_dxf", "ml_svg"})
    legacy_path = sp / ML_POC_SUBDIR / EXTRUDE_COMPARE_SUBDIR / "extrude_compare_report.json"
    if legacy_path.is_file():
        report = json.loads(legacy_path.read_text(encoding="utf-8"))
        allowed.update((report.get("legacy_sources") or {}).keys())
        allowed.update((report.get("legacy_sources") or {}).values())
    return source in allowed


@app.post("/api/extrude-compare/<session_id>")
def run_extrude_compare_api(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    meta_path = sp / SESSION_META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "dxf":
            return jsonify({"error": "extrude compare requires a DXF session"}), 400
    if not (sp / DXF_GEOMETRY_FILE).is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    if not DEFAULT_ONNX_PATH.is_file() and not list_onnx_models():
        return _ml_poc_onnx_missing_response()
    body = request.get_json(silent=True) or {}
    height_m = body.get("height_m", DEFAULT_WALL_HEIGHT_M)
    model_ids = body.get("model_ids")
    if model_ids is not None and not isinstance(model_ids, list):
        return jsonify({"error": "model_ids must be a list of model id strings"}), 400
    try:
        height_m = float(height_m)
    except (TypeError, ValueError):
        return jsonify({"error": "height_m must be a number"}), 400
    try:
        report = run_extrude_compare(sp, height_m=height_m, model_ids=model_ids)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"extrude compare failed: {exc}"}), 500
    return jsonify({"session_id": session_id, **report})


@app.get("/api/extrude-compare/<session_id>")
def get_extrude_compare_report(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    path = sp / ML_POC_SUBDIR / EXTRUDE_COMPARE_SUBDIR / "extrude_compare_report.json"
    if not path.is_file():
        return jsonify({"error": "no extrude compare results for this session"}), 404
    report = json.loads(path.read_text(encoding="utf-8"))
    return jsonify({"session_id": session_id, **report})


@app.get("/api/extrude-compare/<session_id>/<source>")
def get_extrude_compare_mesh(session_id: str, source: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    if not _extrude_compare_variant_allowed(sp, source):
        return jsonify({"error": "unknown compare variant"}), 400
    try:
        result = load_extrude_compare_mesh(sp, source)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"session_id": session_id, **result})


@app.get("/api/extrude-compare/<session_id>/<source>/extract")
def get_extrude_compare_extract(session_id: str, source: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    if not _extrude_compare_variant_allowed(sp, source):
        return jsonify({"error": "unknown compare variant"}), 400
    try:
        result = load_extrude_compare_extract(sp, source)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"session_id": session_id, **result})


@app.post("/api/wall-refine-compare/<session_id>")
def run_wall_refine_compare_api(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    meta_path = sp / SESSION_META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "dxf":
            return jsonify({"error": "wall refine compare requires a DXF session"}), 400
    if not (sp / DXF_GEOMETRY_FILE).is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    body = request.get_json(silent=True) or {}
    height_m = body.get("height_m", DEFAULT_WALL_HEIGHT_M)
    try:
        height_m = float(height_m)
    except (TypeError, ValueError):
        return jsonify({"error": "height_m must be a number"}), 400
    try:
        report = run_refine_wall_compare(sp, height_m=height_m)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"wall refine compare failed: {exc}"}), 500
    return jsonify({"session_id": session_id, **report})


@app.get("/api/wall-refine-compare/<session_id>")
def get_wall_refine_compare_report(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    path = sp / ML_POC_SUBDIR / EXTRUDE_COMPARE_SUBDIR / REFINE_COMPARE_REPORT_FILE
    if not path.is_file():
        return jsonify({"error": "no wall refine compare results for this session"}), 404
    report = json.loads(path.read_text(encoding="utf-8"))
    variant_ids = [v.get("id") for v in report.get("variants") or []]
    if METHOD_HEURISTIC_STRUCTURAL not in variant_ids:
        if not (sp / DXF_GEOMETRY_FILE).is_file():
            return jsonify({"error": "no DXF geometry for this session"}), 404
        try:
            report = run_refine_wall_compare(sp, height_m=report.get("height_m", DEFAULT_WALL_HEIGHT_M))
        except Exception as exc:
            return jsonify({"error": f"wall refine compare upgrade failed: {exc}"}), 500
        return jsonify({"session_id": session_id, **report})
    return jsonify({"session_id": session_id, **report})


@app.get("/api/dxf-geometry/<session_id>")
def get_dxf_geometry(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    path = sp / DXF_GEOMETRY_FILE
    if not path.is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    geometry = json.loads(path.read_text(encoding="utf-8"))
    return jsonify(
        {
            "session_id": session_id,
            "kind": "dxf",
            "positions": geometry["positions"],
            "layers": merge_dxf_layers_with_geometry(sp, geometry),
            "dxf": {
                "segment_count": geometry["segment_count"],
                "layer_count": geometry.get("layer_count", len(geometry.get("layers") or [])),
                "bounds": geometry["bounds"],
                "entity_counts": geometry.get("entity_counts", {}),
                "source_filename": geometry.get("source_filename"),
                "unit_scale_to_meters": geometry.get("unit_scale_to_meters"),
            },
        }
    )


@app.get("/api/dxf-layers/<session_id>")
def get_dxf_layers(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    geom_path = sp / DXF_GEOMETRY_FILE
    if not geom_path.is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    geometry = json.loads(geom_path.read_text(encoding="utf-8"))
    return jsonify(
        {
            "session_id": session_id,
            "layers": build_dxf_layers_payload(sp, geometry),
        }
    )


@app.post("/api/dxf-layers/<session_id>")
def set_dxf_layer_visibility(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    geom_path = sp / DXF_GEOMETRY_FILE
    if not geom_path.is_file():
        return jsonify({"error": "no DXF geometry for this session"}), 404
    geometry = json.loads(geom_path.read_text(encoding="utf-8"))
    layer_ids = {
        str(layer.get("id") or "")
        for layer in (geometry.get("layers") or [])
        if isinstance(layer, dict) and layer.get("id")
    }
    if not layer_ids:
        return jsonify({"error": "DXF has no layer metadata; re-import the file"}), 400

    body = request.get_json(silent=True) or {}
    layer_id = str(body.get("layer_id") or "").strip()
    if layer_id not in layer_ids:
        return jsonify({"error": f"unknown DXF layer: {layer_id}"}), 400
    if "visible" not in body:
        return jsonify({"error": "visible is required"}), 400

    state = load_dxf_layer_state(sp)
    entry = state.get(layer_id, {}) if isinstance(state.get(layer_id), dict) else {}
    entry["visible"] = bool(body.get("visible"))
    state[layer_id] = entry
    save_dxf_layer_state(sp, state)
    return jsonify({"ok": True, "layers": build_dxf_layers_payload(sp, geometry)})


@app.get("/api/objects/<session_id>")
def get_objects(session_id: str):
    sp = safe_session_path(session_id)
    objects = write_manifest(sp)
    return jsonify({"objects": objects, "layers": build_layers_payload(sp, objects)})


@app.get("/api/layers/<session_id>")
def get_layers(session_id: str):
    sp = safe_session_path(session_id)
    objects = write_manifest(sp)
    return jsonify({"layers": build_layers_payload(sp, objects)})


@app.get("/api/tag-definitions")
def get_tag_definitions():
    return jsonify(
        {
            "tags": [
                {"id": layer_id, "label": meta["label"]}
                for layer_id, meta in sorted(LAYER_DEFINITIONS.items(), key=lambda item: item[1]["z_order"])
            ]
        }
    )


@app.post("/api/object-tags/<session_id>")
def update_object_tags(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    data = request.get_json(force=True)
    obj_path = data.get("id")
    action = str(data.get("action") or "set").strip().lower()
    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400

    objects_now = write_manifest(sp)
    target = next((o for o in objects_now if isinstance(o, dict) and o.get("id") == obj_path), None)
    if not target:
        return jsonify({"error": "object not found"}), 404

    state = load_object_tags_state(sp)
    state_objects = state.setdefault("objects", {})
    entry = state_objects.get(obj_path)
    if not isinstance(entry, dict):
        entry = {}
    inferred_layer = str(target.get("inferred_layer") or "other").strip().lower()
    current_tags = effective_tags_for_object(inferred_layer, entry if "tags" in entry else None)

    if action == "reset":
        state_objects.pop(obj_path, None)
    elif action == "add":
        tag = str(data.get("tag") or "").strip().lower()
        if tag not in LAYER_DEFINITIONS:
            return jsonify({"error": "invalid tag"}), 400
        if tag not in current_tags:
            current_tags.append(tag)
        entry["tags"] = current_tags
        state_objects[obj_path] = entry
    elif action == "remove":
        tag = str(data.get("tag") or "").strip().lower()
        if tag not in LAYER_DEFINITIONS:
            return jsonify({"error": "invalid tag"}), 400
        entry["tags"] = [t for t in current_tags if t != tag]
        state_objects[obj_path] = entry
    elif action == "set":
        entry["tags"] = normalize_object_tag_list(data.get("tags"))
        state_objects[obj_path] = entry
    else:
        return jsonify({"error": "invalid action"}), 400

    save_object_tags_state(sp, state)

    usda_path = sp / "extract" / obj_path
    if usda_path.exists() and usda_path.is_file():
        text = usda_path.read_text(encoding="utf-8", errors="ignore")
        override = state_objects.get(obj_path)
        tags_after = effective_tags_for_object(inferred_layer, override if isinstance(override, dict) else None)
        primary = tags_after[0] if tags_after else None
        category = roomplan_category_for_layer(primary) if primary else None
        usda_path.write_text(replace_category_text(text, category), encoding="utf-8")

    objects = write_manifest(sp)
    out = sp / "current.usdz"
    repack_usdz(sp / "extract", out)
    return jsonify(
        {
            "ok": True,
            "objects": objects,
            "layers": build_layers_payload(sp, objects),
            "usdz_url": f"/api/usdz/{session_id}/current.usdz",
        }
    )


@app.post("/api/layers/<session_id>")
def update_layer(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    layer_id = str(data.get("layer_id") or data.get("id") or "").strip().lower()
    if layer_id not in LAYER_DEFINITIONS:
        return jsonify({"error": "invalid layer id"}), 400

    state = load_layer_state(sp)
    entry = state.get(layer_id)
    if not isinstance(entry, dict):
        entry = {}
        state[layer_id] = entry

    if "visible" in data:
        entry["visible"] = bool(data.get("visible"))
    if "locked" in data:
        entry["locked"] = bool(data.get("locked"))

    if not entry:
        state.pop(layer_id, None)
    save_layer_state(sp, state)

    objects = write_manifest(sp)
    return jsonify({"ok": True, "layers": build_layers_payload(sp, objects), "objects": objects})


@app.get("/api/floorplan/<session_id>")
def get_floorplan(session_id: str):
    sp = safe_session_path(session_id)
    objects = write_manifest(sp)
    basis = None
    manifest_path = sp / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            basis = manifest.get("world_planar_basis")
        except Exception:
            basis = None
    return jsonify(floorplan_payload(objects, basis))


@app.get("/api/texture-assets")
def get_texture_assets():
    assets = []
    for key, filename in TEXTURE_ASSET_MAP.items():
        p = TEXTURE_DIR / filename
        if p.exists():
            stat = p.stat()
            assets.append(
                {
                    "key": key,
                    "label": key.capitalize(),
                    "filename": filename,
                    "mtime": int(stat.st_mtime),
                    "size": stat.st_size,
                }
            )
    return jsonify({"assets": assets})


@app.get("/api/texture-asset/<asset_key>")
def get_texture_asset(asset_key: str):
    key = str(asset_key or "").strip().lower()
    filename = TEXTURE_ASSET_MAP.get(key)
    if not filename:
        return jsonify({"error": "invalid texture asset"}), 400
    path = TEXTURE_DIR / filename
    if not path.exists():
        return jsonify({"error": "texture asset file not found"}), 404
    stat = path.stat()
    response = send_file(path, mimetype="image/png", as_attachment=False, download_name=filename)
    response.headers["Cache-Control"] = "no-cache"
    response.headers["ETag"] = f'"{int(stat.st_mtime)}-{stat.st_size}"'
    return response


@app.get("/api/replacement-assets")
def get_replacement_assets():
    assets = []
    for key, filename in REPLACEMENT_ASSET_MAP.items():
        p = REPLACEMENTS_DIR / filename
        if p.exists():
            assets.append({"key": key, "label": key.capitalize(), "filename": filename})
    return jsonify({"assets": assets})


@app.get("/api/replacement-asset/<asset_key>")
def get_replacement_asset(asset_key: str):
    key = str(asset_key or "").strip().lower()
    usdz_name = REPLACEMENT_ASSET_MAP.get(key)
    if not usdz_name:
        return jsonify({"error": "invalid replacement asset"}), 400
    prefer_usdz = str(request.args.get("format") or "").strip().lower() == "usdz"
    glb_name = f"{Path(usdz_name).stem}.glb"
    glb_path = REPLACEMENTS_DIR / glb_name
    usdz_path = REPLACEMENTS_DIR / usdz_name
    if not prefer_usdz and glb_path.exists():
        return send_file(glb_path, mimetype="model/gltf-binary", as_attachment=False, download_name=glb_name)
    if usdz_path.exists():
        return send_file(usdz_path, mimetype="model/vnd.usdz+zip", as_attachment=False, download_name=usdz_name)
    if glb_path.exists():
        return send_file(glb_path, mimetype="model/gltf-binary", as_attachment=False, download_name=glb_name)
    return jsonify({"error": "replacement asset file not found"}), 404


@app.post("/api/object/<session_id>")
def update_object(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    obj_path = data.get("id")
    position = data.get("position")
    yaw_deg_raw = data.get("yaw_deg")
    yaw_deg = float(yaw_deg_raw) if yaw_deg_raw is not None else 0.0
    quaternion_xyzw = data.get("quaternion_xyzw")

    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400
    if not isinstance(position, list) or len(position) != 3:
        return jsonify({"error": "position must be [x,y,z]"}), 400

    if quaternion_xyzw is not None:
        if not isinstance(quaternion_xyzw, list) or len(quaternion_xyzw) != 4:
            return jsonify({"error": "quaternion_xyzw must be [x,y,z,w]"}), 400
        try:
            quaternion_xyzw = [float(v) for v in quaternion_xyzw]
        except (TypeError, ValueError):
            return jsonify({"error": "invalid quaternion values"}), 400

    usda_path = sp / "extract" / obj_path
    if not usda_path.exists():
        return jsonify({"error": "object file not found"}), 404

    text = usda_path.read_text(encoding="utf-8", errors="ignore")
    old = parse_matrix(text)
    if old is None:
        return jsonify({"error": "matrix4d xformOp:transform not found"}), 400

    if quaternion_xyzw is None:
        a = math.radians(yaw_deg)
        quaternion_xyzw = [0.0, math.sin(a / 2.0), 0.0, math.cos(a / 2.0)]

    new = update_matrix_from_position_quaternion(
        old=old,
        matrix_position=[float(v) for v in position],
        quaternion_xyzw=quaternion_xyzw,
    )
    usda_path.write_text(replace_matrix_text(text, new), encoding="utf-8")
    objects = write_manifest(sp)
    out = sp / "current.usdz"
    repack_usdz(sp / "extract", out)
    return jsonify({"ok": True, "objects": objects, "usdz_url": f"/api/usdz/{session_id}/current.usdz"})


@app.post("/api/replace-object/<session_id>")
def replace_object(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    obj_path = data.get("id")
    asset_key = str(data.get("asset_key") or "").strip().lower()
    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400
    if asset_key not in REPLACEMENT_ASSET_MAP:
        return jsonify({"error": "invalid replacement asset"}), 400

    objects_now = write_manifest(sp)
    target = next((o for o in objects_now if isinstance(o, dict) and o.get("id") == obj_path), None)
    if not target:
        return jsonify({"error": "object not found"}), 404
    if is_floor_or_wall_object(target):
        return jsonify({"error": "object replacement is not supported for floor or wall"}), 400

    try:
        apply_usdz_object_replacement(sp, obj_path, asset_key)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"replacement failed: {exc}"}), 500

    state = load_replacement_state(sp)
    state_objects = state.setdefault("objects", {})
    if obj_path not in state_objects:
        state_objects[obj_path] = {}
    state_objects[obj_path]["asset_key"] = asset_key
    save_replacement_state(sp, state)
    objects = write_manifest(sp)
    usdz_url = repack_session_usdz(sp)
    return jsonify(
        {
            "ok": True,
            "objects": objects,
            "asset_key": asset_key,
            "usdz_url": usdz_url,
            "updated_count": 1,
        }
    )


@app.post("/api/apply-texture/<session_id>")
def apply_texture(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    obj_path = data.get("id")
    texture_key = str(data.get("texture_key") or data.get("asset_key") or "").strip().lower()
    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400
    if texture_key not in TEXTURE_ASSET_MAP:
        return jsonify({"error": "invalid texture asset"}), 400

    objects_now = write_manifest(sp)
    target = next((o for o in objects_now if isinstance(o, dict) and o.get("id") == obj_path), None)
    if not target:
        return jsonify({"error": "object not found"}), 404
    if not is_floor_or_wall_object(target):
        return jsonify({"error": "texture can only be applied to floor or wall objects"}), 400

    state = load_replacement_state(sp)
    state_objects = state.setdefault("objects", {})
    if obj_path not in state_objects:
        state_objects[obj_path] = {}
    state_objects[obj_path]["texture_asset_key"] = texture_key
    save_replacement_state(sp, state)
    objects = write_manifest(sp)
    return jsonify({"ok": True, "objects": objects, "texture_key": texture_key})


@app.post("/api/remove-texture/<session_id>")
def remove_texture(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    obj_path = data.get("id")
    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400

    state = load_replacement_state(sp)
    state_objects = state.get("objects", {})
    entry = state_objects.get(obj_path)
    if not isinstance(entry, dict) or not entry.get("texture_asset_key"):
        return jsonify({"error": "texture state not found"}), 404
    entry.pop("texture_asset_key", None)
    if not entry:
        state_objects.pop(obj_path, None)
    save_replacement_state(sp, state)
    objects = write_manifest(sp)
    return jsonify({"ok": True, "objects": objects})


@app.post("/api/unreplace-object/<session_id>")
def unreplace_object(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    obj_path = data.get("id")
    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400

    state = load_replacement_state(sp)
    state_objects = state.get("objects", {})
    entry = state_objects.get(obj_path)
    if not isinstance(entry, dict) or not entry.get("asset_key"):
        return jsonify({"error": "replacement state not found"}), 404
    if not restore_usdz_object_replacement(sp, obj_path):
        return jsonify({"error": "original geometry backup not found"}), 404
    state_objects.pop(obj_path, None)
    save_replacement_state(sp, state)

    objects = write_manifest(sp)
    usdz_url = repack_session_usdz(sp)
    return jsonify({"ok": True, "objects": objects, "usdz_url": usdz_url, "removed_count": 1})


@app.post("/api/apply-texture-by-tag/<session_id>")
def apply_texture_by_tag(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    data = request.get_json(force=True)
    tag_id = str(data.get("tag") or data.get("tag_id") or "").strip().lower()
    texture_key = str(data.get("texture_key") or data.get("asset_key") or "").strip().lower()
    if tag_id not in LAYER_DEFINITIONS:
        return jsonify({"error": "invalid tag"}), 400
    if texture_key not in TEXTURE_ASSET_MAP:
        return jsonify({"error": "invalid texture asset"}), 400

    objects_now = write_manifest(sp)
    matched = filter_objects_by_tag(objects_now, tag_id)
    if not matched:
        return jsonify({"error": "no objects with this tag"}), 404

    state = load_replacement_state(sp)
    state_objects = state.setdefault("objects", {})
    updated = 0
    skipped = 0
    for obj in matched:
        obj_id = obj.get("id")
        if not isinstance(obj_id, str):
            skipped += 1
            continue
        if not is_floor_or_wall_object(obj):
            skipped += 1
            continue
        if obj_id not in state_objects:
            state_objects[obj_id] = {}
        state_objects[obj_id]["texture_asset_key"] = texture_key
        updated += 1
    save_replacement_state(sp, state)
    objects = write_manifest(sp)
    return jsonify(
        {
            "ok": True,
            "objects": objects,
            "layers": build_layers_payload(sp, objects),
            "tag": tag_id,
            "texture_key": texture_key,
            "updated_count": updated,
            "skipped_count": skipped,
            "matched_count": len(matched),
        }
    )


@app.post("/api/remove-texture-by-tag/<session_id>")
def remove_texture_by_tag(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    data = request.get_json(force=True)
    tag_id = str(data.get("tag") or data.get("tag_id") or "").strip().lower()
    if tag_id not in LAYER_DEFINITIONS:
        return jsonify({"error": "invalid tag"}), 400

    objects_now = write_manifest(sp)
    matched = filter_objects_by_tag(objects_now, tag_id)
    if not matched:
        return jsonify({"error": "no objects with this tag"}), 404

    state = load_replacement_state(sp)
    state_objects = state.get("objects", {})
    removed = 0
    for obj in matched:
        obj_id = obj.get("id")
        if not isinstance(obj_id, str):
            continue
        entry = state_objects.get(obj_id)
        if not isinstance(entry, dict) or not entry.get("texture_asset_key"):
            continue
        entry.pop("texture_asset_key", None)
        if not entry:
            state_objects.pop(obj_id, None)
        removed += 1
    save_replacement_state(sp, state)
    objects = write_manifest(sp)
    return jsonify(
        {
            "ok": True,
            "objects": objects,
            "layers": build_layers_payload(sp, objects),
            "tag": tag_id,
            "removed_count": removed,
            "matched_count": len(matched),
        }
    )


@app.post("/api/replace-object-by-tag/<session_id>")
def replace_object_by_tag(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    data = request.get_json(force=True)
    tag_id = str(data.get("tag") or data.get("tag_id") or "").strip().lower()
    asset_key = str(data.get("asset_key") or "").strip().lower()
    if tag_id not in LAYER_DEFINITIONS:
        return jsonify({"error": "invalid tag"}), 400
    if asset_key not in REPLACEMENT_ASSET_MAP:
        return jsonify({"error": "invalid replacement asset"}), 400

    objects_now = write_manifest(sp)
    matched = filter_objects_by_tag(objects_now, tag_id)
    if not matched:
        return jsonify({"error": "no objects with this tag"}), 404

    state = load_replacement_state(sp)
    state_objects = state.setdefault("objects", {})
    updated = 0
    skipped = 0
    for obj in matched:
        obj_id = obj.get("id")
        if not isinstance(obj_id, str):
            skipped += 1
            continue
        if is_floor_or_wall_object(obj):
            skipped += 1
            continue
        try:
            apply_usdz_object_replacement(sp, obj_id, asset_key)
        except Exception:
            skipped += 1
            continue
        if obj_id not in state_objects:
            state_objects[obj_id] = {}
        state_objects[obj_id]["asset_key"] = asset_key
        updated += 1
    if updated <= 0:
        return jsonify({"error": "no objects could be replaced"}), 400
    save_replacement_state(sp, state)
    objects = write_manifest(sp)
    usdz_url = repack_session_usdz(sp)
    return jsonify(
        {
            "ok": True,
            "objects": objects,
            "layers": build_layers_payload(sp, objects),
            "tag": tag_id,
            "asset_key": asset_key,
            "updated_count": updated,
            "skipped_count": skipped,
            "matched_count": len(matched),
            "usdz_url": usdz_url,
        }
    )


@app.post("/api/unreplace-object-by-tag/<session_id>")
def unreplace_object_by_tag(session_id: str):
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    data = request.get_json(force=True)
    tag_id = str(data.get("tag") or data.get("tag_id") or "").strip().lower()
    if tag_id not in LAYER_DEFINITIONS:
        return jsonify({"error": "invalid tag"}), 400

    objects_now = write_manifest(sp)
    matched = filter_objects_by_tag(objects_now, tag_id)
    if not matched:
        return jsonify({"error": "no objects with this tag"}), 404

    state = load_replacement_state(sp)
    state_objects = state.get("objects", {})
    removed = 0
    skipped = 0
    for obj in matched:
        obj_id = obj.get("id")
        if not isinstance(obj_id, str):
            skipped += 1
            continue
        entry = state_objects.get(obj_id)
        if not isinstance(entry, dict) or not entry.get("asset_key"):
            skipped += 1
            continue
        if not restore_usdz_object_replacement(sp, obj_id):
            skipped += 1
            continue
        state_objects.pop(obj_id, None)
        removed += 1
    if removed <= 0:
        return jsonify({"error": "no replaced objects to restore for this tag"}), 404
    save_replacement_state(sp, state)
    objects = write_manifest(sp)
    usdz_url = repack_session_usdz(sp)
    return jsonify(
        {
            "ok": True,
            "objects": objects,
            "layers": build_layers_payload(sp, objects),
            "tag": tag_id,
            "removed_count": removed,
            "skipped_count": skipped,
            "matched_count": len(matched),
            "usdz_url": usdz_url,
        }
    )


@app.post("/api/resize-object/<session_id>")
def resize_object(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    obj_path = data.get("id")
    dimensions = data.get("dimensions")
    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400
    if not isinstance(dimensions, list) or len(dimensions) != 3:
        return jsonify({"error": "dimensions must be [x,y,z]"}), 400
    try:
        target_dims = np.array([max(abs(float(v)), 0.05) for v in dimensions], dtype=float)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid dimensions values"}), 400

    usda_path = sp / "extract" / obj_path
    if not usda_path.exists() or usda_path.is_dir():
        return jsonify({"error": "object file not found"}), 404

    text = usda_path.read_text(encoding="utf-8", errors="ignore")
    pts = parse_points(text)
    if pts is None or len(pts) == 0:
        return jsonify({"error": "points not found"}), 400

    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    curr_dims = np.maximum(np.abs(pmax - pmin), 0.05)
    center = (pmin + pmax) * 0.5
    scale = target_dims / curr_dims
    new_pts = (pts - center) * scale + center
    usda_path.write_text(replace_points_text(text, new_pts), encoding="utf-8")

    objects = write_manifest(sp)
    out = sp / "current.usdz"
    repack_usdz(sp / "extract", out)
    return jsonify({"ok": True, "objects": objects, "usdz_url": f"/api/usdz/{session_id}/current.usdz"})


@app.post("/api/delete-object/<session_id>")
def delete_object(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    obj_path = data.get("id")
    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400

    objects = write_manifest(sp)
    target = next((o for o in objects if o.get("id") == obj_path), None)
    if target and is_floor_object(target):
        return jsonify({"error": "floor objects cannot be deleted"}), 400

    usda_path = sp / "extract" / obj_path
    if not usda_path.exists():
        return jsonify({"error": "object file not found"}), 404
    if usda_path.is_dir():
        return jsonify({"error": "invalid object path"}), 400

    deleted_text = usda_path.read_text(encoding="utf-8", errors="ignore")
    usda_path.unlink()
    state = load_replacement_state(sp)
    state.get("objects", {}).pop(obj_path, None)
    save_replacement_state(sp, state)
    # Clean up empty parent directories (up to extract/)
    try:
        parent = usda_path.parent
        extract_root = (sp / "extract").resolve()
        while parent.exists() and parent.is_dir() and parent.resolve() != extract_root:
            if any(parent.iterdir()):
                break
            parent.rmdir()
            parent = parent.parent
    except OSError:
        pass

    objects = write_manifest(sp)
    out = sp / "current.usdz"
    repack_usdz(sp / "extract", out)
    return jsonify(
        {
            "ok": True,
            "objects": objects,
            "usdz_url": f"/api/usdz/{session_id}/current.usdz",
            "deleted": {"id": obj_path, "usda_text": deleted_text},
        }
    )


@app.post("/api/restore-object/<session_id>")
def restore_object(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    obj_path = data.get("id")
    usda_text = data.get("usda_text")
    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400
    if not isinstance(usda_text, str) or not usda_text.strip():
        return jsonify({"error": "usda_text is required"}), 400

    # Prevent restoring floors (matches delete constraint).
    objects = write_manifest(sp)
    existing = next((o for o in objects if o.get("id") == obj_path), None)
    if existing and is_floor_object(existing):
        return jsonify({"error": "floor objects cannot be restored via this endpoint"}), 400

    usda_path = sp / "extract" / obj_path
    if usda_path.exists():
        return jsonify({"error": "object already exists"}), 409
    usda_path.parent.mkdir(parents=True, exist_ok=True)
    usda_path.write_text(usda_text, encoding="utf-8")

    objects = write_manifest(sp)
    out = sp / "current.usdz"
    repack_usdz(sp / "extract", out)
    return jsonify({"ok": True, "objects": objects, "usdz_url": f"/api/usdz/{session_id}/current.usdz"})


@app.get("/api/usdz/<session_id>/<name>")
def get_usdz(session_id: str, name: str):
    sp = safe_session_path(session_id)
    if name not in ["original.usdz", "current.usdz"]:
        return jsonify({"error": "invalid file"}), 400
    p = sp / name
    if not p.exists():
        p = sp / "original.usdz"
    return send_file(p, mimetype="model/vnd.usdz+zip", as_attachment=False, download_name=name)


def _safe_extract_relpath(rel: str) -> Optional[Path]:
    if not isinstance(rel, str):
        return None
    rel = rel.strip()
    if not rel or rel.startswith("/") or "\\" in rel or "\x00" in rel:
        return None
    if ".." in rel.split("/"):
        return None
    return Path(rel)


@app.get("/api/session-assets/<session_id>")
def get_session_assets(session_id: str):
    sp = safe_session_path(session_id)
    extract = sp / "extract"
    if not extract.exists() or not extract.is_dir():
        return jsonify({"error": "extract not found"}), 404
    exts = {".png", ".jpg", ".jpeg", ".webp", ".ktx2", ".ktx", ".tga", ".exr"}
    assets: List[str] = []
    for p in extract.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        try:
            assets.append(p.relative_to(extract).as_posix())
        except ValueError:
            continue
    assets.sort()
    return jsonify({"assets": assets})


@app.get("/api/session-asset/<session_id>")
def get_session_asset(session_id: str):
    sp = safe_session_path(session_id)
    rel = request.args.get("path", "")
    rel_path = _safe_extract_relpath(rel)
    if rel_path is None:
        return jsonify({"error": "invalid path"}), 400
    extract = sp / "extract"
    full = (extract / rel_path).resolve()
    try:
        extract_root = extract.resolve()
    except OSError:
        return jsonify({"error": "extract not found"}), 404
    if not str(full).startswith(str(extract_root) + os.sep) and full != extract_root:
        return jsonify({"error": "invalid path"}), 400
    if not full.exists() or not full.is_file():
        return jsonify({"error": "file not found"}), 404
    # Let send_file infer; provide no-cache to help debugging.
    resp = send_file(full, as_attachment=False, download_name=full.name)
    resp.headers["Cache-Control"] = "no-cache"
    return resp


def ensure_current_usdz(sp: Path) -> Path:
    extract = sp / "extract"
    if not extract.is_dir():
        raise ValueError("Session extract folder not found")
    normalize_extract_usda_for_usdz_loader(extract)
    out = sp / "current.usdz"
    repack_usdz(extract, out)
    return out


def _session_is_usdz(sp: Path) -> bool:
    meta_path = sp / SESSION_META_FILE
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return str(meta.get("kind") or "usdz") == "usdz"
        except json.JSONDecodeError:
            pass
    return (sp / "original.usdz").is_file() or (sp / "extract").is_dir()


@app.get("/api/export-formats")
def export_formats():
    return jsonify({"formats": list_export_formats()})


@app.get("/api/export/<session_id>/<format_id>")
def export_session(session_id: str, format_id: str):
    if not re.fullmatch(r"[a-z0-9\-]+", format_id or ""):
        return jsonify({"error": "invalid export format"}), 400
    try:
        sp = safe_session_path(session_id)
    except ValueError:
        return jsonify({"error": "invalid session id"}), 400
    if not _session_is_usdz(sp):
        return jsonify({"error": "Export is only available for USDZ (RoomPlan) sessions"}), 400
    if not get_export_format_spec(format_id):
        return jsonify({"error": "unknown export format"}), 404
    try:
        path, download_name, mime_type = build_export_file(
            sp,
            format_id,
            repack_usdz_fn=repack_usdz,
            ensure_current_usdz_fn=ensure_current_usdz,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Export failed: {exc}"}), 500
    return send_file(
        path,
        mimetype=mime_type,
        as_attachment=True,
        download_name=download_name,
    )


@app.get("/api/download/<session_id>")
def download(session_id: str):
    """Legacy alias for USDZ export."""
    return export_session(session_id, "usdz")


@app.post("/api/restore-object-geometry/<session_id>")
def restore_object_geometry(session_id: str):
    sp = safe_session_path(session_id)
    data = request.get_json(force=True)
    obj_path = data.get("id")
    usda_text = data.get("usda_text")
    if not isinstance(obj_path, str) or ".." in obj_path or obj_path.startswith("/"):
        return jsonify({"error": "invalid object id"}), 400
    if not isinstance(usda_text, str) or not usda_text.strip():
        return jsonify({"error": "usda_text is required"}), 400

    usda_path = sp / "extract" / obj_path
    if not usda_path.exists() or usda_path.is_dir():
        return jsonify({"error": "object file not found"}), 404

    usda_path.write_text(usda_text, encoding="utf-8")
    objects = write_manifest(sp)
    out = sp / "current.usdz"
    repack_usdz(sp / "extract", out)
    return jsonify({"ok": True, "objects": objects, "usdz_url": f"/api/usdz/{session_id}/current.usdz"})


@app.post("/api/snap-bottoms-to-floor/<session_id>")
def snap_bottoms_to_floor(session_id: str):
    sp = safe_session_path(session_id)
    objects = write_manifest(sp)
    floor_top_y = compute_floor_top_world_y(objects)
    if floor_top_y is None:
        return jsonify({"error": "床オブジェクトが見つかりません"}), 400

    moved = 0
    moved_ids: List[str] = []
    undo_items: List[Dict] = []
    skip: Dict[str, int] = {}

    for obj in objects:
        if is_floor_object(obj):
            continue
        obj_id = obj.get("id")
        if not isinstance(obj_id, str):
            skip["invalid_id"] = skip.get("invalid_id", 0) + 1
            continue
        usda_path = sp / "extract" / obj_id
        if not usda_path.exists() or usda_path.is_dir():
            skip["missing_file"] = skip.get("missing_file", 0) + 1
            continue
        text = usda_path.read_text(encoding="utf-8", errors="ignore")
        changed, new_text = stretch_mesh_bottom_to_floor_y(text, floor_top_y)
        if not changed:
            skip["already_seated"] = skip.get("already_seated", 0) + 1
            continue
        undo_items.append({"id": obj_id, "usda_text": text})
        usda_path.write_text(new_text, encoding="utf-8")
        moved += 1
        moved_ids.append(obj_id)

    objects = write_manifest(sp)
    out = sp / "current.usdz"
    repack_usdz(sp / "extract", out)
    return jsonify(
        {
            "ok": True,
            "moved": moved,
            "moved_ids": moved_ids,
            "floor_top_y": floor_top_y,
            "undo_items": undo_items,
            "skip_counts": skip,
            "objects": objects,
            "usdz_url": f"/api/usdz/{session_id}/current.usdz",
        }
    )


@app.post("/api/align-walls/<session_id>")
def align_walls_to_floor(session_id: str):
    sp = safe_session_path(session_id)
    objects = write_manifest(sp)
    floors = [o for o in objects if is_floor_object(o)]
    walls = [o for o in objects if is_wall_object(o)]
    floor_count = len(floors)
    wall_count = len(walls)
    if not floors or not walls:
        hint = describe_align_result(
            0,
            floor_count,
            wall_count,
            0,
            {
                "already_aligned": 0,
                "no_parallel_floor_edge": 0,
                "missing_file": 0,
                "no_matrix": 0,
                "invalid_id": 0,
            },
        )
        return jsonify(
            {
                "ok": True,
                "moved": 0,
                "moved_ids": [],
                "total_shift": 0.0,
                "max_shift": 0.0,
                "floor_count": floor_count,
                "wall_count": wall_count,
                "floor_edge_count": 0,
                "align_skip_counts": {},
                "align_hint": hint,
                "align_exact_match_tol": ALIGN_EXACT_MATCH_ABS_TOL,
                "objects": objects,
                "usdz_url": f"/api/usdz/{session_id}/current.usdz",
            }
        )

    edges, _, _ = build_floor_edges_with_hash(floors)
    floor_edge_count = len(edges)
    moved = 0
    moved_ids: List[str] = []
    total_shift = 0.0
    max_shift = 0.0
    skip: Dict[str, int] = {
        "already_aligned": 0,
        "no_parallel_floor_edge": 0,
        "missing_file": 0,
        "no_matrix": 0,
        "invalid_id": 0,
    }
    projection_basis = choose_common_projection_basis(objects)

    for wall in walls:
        wall_id = wall.get("id")
        if not isinstance(wall_id, str):
            skip["invalid_id"] += 1
            continue
        seg_a, seg_b = object_centerline_xz(wall)
        choice = best_floor_edge_min_translation_parallel_wall(seg_a, seg_b, edges)
        if choice is None:
            skip["no_parallel_floor_edge"] += 1
            continue
        shift_vec, shift_len, _chosen_e = choice
        if shift_len <= ALIGN_EXACT_MATCH_ABS_TOL:
            skip["already_aligned"] += 1
            continue
        base_pos = wall.get("matrix_position") or wall.get("position") or [0.0, 0.0, 0.0]
        shift_world = shift_2d_to_world_xyz(shift_vec, projection_basis)
        new_pos = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
        new_pos[0] += float(shift_world[0])
        new_pos[1] += float(shift_world[1])
        new_pos[2] += float(shift_world[2])

        usda_path = sp / "extract" / wall_id
        if not usda_path.exists():
            skip["missing_file"] += 1
            continue
        text = usda_path.read_text(encoding="utf-8", errors="ignore")
        old = parse_matrix(text)
        if old is None:
            skip["no_matrix"] += 1
            continue

        quat = wall.get("quaternion_xyzw")
        new_mat = update_matrix_from_position_quaternion(old=old, matrix_position=new_pos, quaternion_xyzw=quat)
        usda_path.write_text(replace_matrix_text(text, new_mat), encoding="utf-8")
        moved += 1
        moved_ids.append(wall_id)
        total_shift += shift_len
        max_shift = max(max_shift, shift_len)

    objects = write_manifest(sp)
    out = sp / "current.usdz"
    repack_usdz(sp / "extract", out)
    align_hint = describe_align_result(moved, floor_count, wall_count, floor_edge_count, skip)
    return jsonify(
        {
            "ok": True,
            "moved": moved,
            "moved_ids": moved_ids,
            "total_shift": total_shift,
            "max_shift": max_shift,
            "floor_count": floor_count,
            "wall_count": wall_count,
            "floor_edge_count": floor_edge_count,
            "align_skip_counts": skip,
            "align_hint": align_hint,
            "align_exact_match_tol": ALIGN_EXACT_MATCH_ABS_TOL,
            "objects": objects,
            "usdz_url": f"/api/usdz/{session_id}/current.usdz",
        }
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8100"))
    debug = os.getenv("FLASK_DEBUG", "").strip() in ("1", "true", "True")
    app.run(host=host, port=port, debug=debug)
