"""Export edited RoomPlan USDZ sessions to common 3D interchange formats."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    from pxr import Usd, UsdUtils
except Exception:  # pragma: no cover
    Usd = None
    UsdUtils = None

ExportResult = Tuple[Path, str, str]  # path, download_name, mimetype

EXPORT_FORMAT_SPECS: List[Dict] = [
    {
        "id": "usdz",
        "extension": "usdz",
        "label": "USDZ",
        "description": "編集済み RoomPlan パッケージ（複数 USDA レイヤー保持・再インポート向け）",
        "mime_type": "model/vnd.usdz+zip",
        "filename": "edited_room.usdz",
        "kind": "native",
    },
    {
        "id": "usdz-arkit",
        "extension": "usdz",
        "label": "USDZ (ARKit / Quick Look)",
        "description": "Apple AR 向けにフラット化した USDZ（variant 等は失われる場合あり）",
        "mime_type": "model/vnd.usdz+zip",
        "filename": "edited_room_arkit.usdz",
        "kind": "arkit",
        "requires_usd": True,
    },
    {
        "id": "usda",
        "extension": "usda",
        "label": "USDA",
        "description": "単一ファイル・ASCII USD（合成をフラット化）",
        "mime_type": "model/vnd.usda+text",
        "filename": "edited_room_flat.usda",
        "kind": "flat",
        "requires_usd": True,
    },
    {
        "id": "usdc",
        "extension": "usdc",
        "label": "USDC",
        "description": "単一ファイル・バイナリ USD（合成をフラット化）",
        "mime_type": "model/vnd.usdc+usd",
        "filename": "edited_room_flat.usdc",
        "kind": "flat",
        "requires_usd": True,
    },
    {
        "id": "usd",
        "extension": "usd",
        "label": "USD",
        "description": "単一ファイル・バイナリ USD（.usdc と同内容、拡張子 .usd）",
        "mime_type": "model/vnd.usd+usd",
        "filename": "edited_room_flat.usd",
        "kind": "flat",
        "flat_suffix": ".usd",
        "requires_usd": True,
    },
    {
        "id": "usd-package",
        "extension": "zip",
        "label": "USD パッケージ (ZIP)",
        "description": "extract フォルダ一式（room_mesh.usda と assets/ を含む ZIP）",
        "mime_type": "application/zip",
        "filename": "edited_room_usd_package.zip",
        "kind": "package",
    },
    {
        "id": "glb",
        "extension": "glb",
        "label": "GLB",
        "description": "glTF バイナリ（USDZ → usd2gltf）",
        "mime_type": "model/gltf-binary",
        "filename": "edited_room.glb",
        "kind": "glb",
        "requires_usd2gltf": True,
    },
    {
        "id": "gltf",
        "extension": "gltf",
        "label": "glTF",
        "description": "glTF JSON（USDZ → usd2gltf）",
        "mime_type": "model/gltf+json",
        "filename": "edited_room.gltf",
        "kind": "gltf",
        "requires_usd2gltf": True,
    },
    {
        "id": "obj",
        "extension": "obj",
        "label": "OBJ",
        "description": "Wavefront OBJ（メッシュ統合・マテリアルは簡略化）",
        "mime_type": "text/plain",
        "filename": "edited_room.obj",
        "kind": "mesh",
        "mesh_extension": "obj",
        "requires_usd2gltf": True,
        "requires_trimesh": True,
    },
    {
        "id": "dxf",
        "extension": "dxf",
        "label": "DXF",
        "description": "",
        "mime_type": "application/dxf",
        "filename": "edited_room.dxf",
        "kind": "dxf",
        "requires_usd2gltf": True,
        "requires_trimesh": True,
        "requires_ezdxf": True,
    },
    {
        "id": "fbx",
        "extension": "fbx",
        "label": "FBX",
        "description": "Autodesk FBX（Blender 経由・要 blender コマンド）",
        "mime_type": "application/octet-stream",
        "filename": "edited_room.fbx",
        "kind": "fbx",
        "requires_usd2gltf": True,
        "requires_blender": True,
    },
    {
        "id": "stl",
        "extension": "stl",
        "label": "STL",
        "description": "3D プリント向けメッシュ（メッシュ統合）",
        "mime_type": "model/stl",
        "filename": "edited_room.stl",
        "kind": "mesh",
        "mesh_extension": "stl",
        "requires_usd2gltf": True,
        "requires_trimesh": True,
    },
]


def _session_root_usda(extract_dir: Path) -> Path:
    room_mesh = extract_dir / "room_mesh.usda"
    if room_mesh.is_file():
        return room_mesh
    candidates = sorted(extract_dir.glob("*.usda"))
    if not candidates:
        raise ValueError("No root .usda layer found in session extract")
    return candidates[0]


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def _package_available(name: str) -> bool:
    if name == "usd2gltf":
        return _resolve_usd2gltf_bin() is not None or importlib.util.find_spec("usd2gltf") is not None
    return importlib.util.find_spec(name) is not None


def _usd2gltf_unavailable_reason() -> str:
    return "pip install usd2gltf（または usd2gltf を PATH に追加）"


def _resolve_usd2gltf_bin() -> Optional[str]:
    found = shutil.which("usd2gltf")
    if found:
        return found
    exe_parent = Path(sys.executable).resolve().parent
    for candidate in (exe_parent / "usd2gltf", exe_parent / "usd2gltf.exe"):
        if candidate.is_file():
            return str(candidate)
    return None


def list_export_formats() -> List[Dict]:
    formats: List[Dict] = []
    for spec in EXPORT_FORMAT_SPECS:
        entry = {
            "id": spec["id"],
            "extension": spec["extension"],
            "label": spec["label"],
            "description": spec["description"],
            "mime_type": spec["mime_type"],
            "filename": spec["filename"],
            "available": True,
            "unavailable_reason": None,
        }
        if spec.get("requires_usd") and Usd is None:
            entry["available"] = False
            entry["unavailable_reason"] = "Pixar USD (usd-core) が未インストールです"
        elif spec.get("requires_usd2gltf") and not _package_available("usd2gltf"):
            entry["available"] = False
            entry["unavailable_reason"] = _usd2gltf_unavailable_reason()
        elif spec.get("requires_trimesh") and not _package_available("trimesh"):
            entry["available"] = False
            entry["unavailable_reason"] = "pip install trimesh"
        elif spec.get("requires_ezdxf") and not _package_available("ezdxf"):
            entry["available"] = False
            entry["unavailable_reason"] = "pip install ezdxf"
        elif spec.get("requires_blender") and not _tool_available("blender"):
            entry["available"] = False
            entry["unavailable_reason"] = "Blender が PATH にありません（FBX 出力用）"
        formats.append(entry)
    return formats


def get_export_format_spec(format_id: str) -> Optional[Dict]:
    for spec in EXPORT_FORMAT_SPECS:
        if spec["id"] == format_id:
            return spec
    return None


def _run_usd2gltf(input_path: Path, output_path: Path) -> None:
    usd2gltf_bin = _resolve_usd2gltf_bin()
    if usd2gltf_bin:
        proc = subprocess.run(
            [usd2gltf_bin, "-i", str(input_path), "-o", str(output_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not output_path.is_file():
            stderr = (proc.stderr or proc.stdout or "").strip()
            raise ValueError(stderr or "usd2gltf conversion failed")
        return

    if not _package_available("usd2gltf"):
        raise ValueError(_usd2gltf_unavailable_reason())

    try:
        from usd2gltf.converter import Converter
    except Exception as exc:  # pragma: no cover
        raise ValueError(_usd2gltf_unavailable_reason()) from exc

    factory = Converter()
    stage = factory.load_usd(str(input_path))
    if not stage:
        raise ValueError(f"Failed to open USD stage: {input_path.name}")
    factory.process(stage, str(output_path))
    if not output_path.is_file():
        raise ValueError("usd2gltf conversion failed")


def _export_mesh_with_trimesh(glb_path: Path, out_path: Path, mesh_extension: str) -> None:
    import trimesh

    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh is None:
        raise ValueError("No mesh geometry to export")
    mesh.export(out_path, file_type=mesh_extension)


def _export_dxf_with_trimesh(glb_path: Path, out_path: Path) -> None:
    import trimesh

    try:
        import ezdxf
    except Exception as exc:  # pragma: no cover
        raise ValueError("ezdxf is required for DXF export") from exc

    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh is None:
        raise ValueError("No mesh geometry to export")

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    if not hasattr(mesh, "faces") or not hasattr(mesh, "vertices"):
        raise ValueError("Unsupported mesh type for DXF export")

    faces = mesh.faces
    verts = mesh.vertices
    max_faces = 200_000
    if len(faces) > max_faces:
        raise ValueError(f"DXF export face limit exceeded: {len(faces)} > {max_faces}")

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    for f in faces:
        i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
        v0 = verts[i0]
        v1 = verts[i1]
        v2 = verts[i2]
        # 3DFACE needs 4 points; repeat last.
        msp.add_3dface(
            [(float(v0[0]), float(v0[1]), float(v0[2])),
             (float(v1[0]), float(v1[1]), float(v1[2])),
             (float(v2[0]), float(v2[1]), float(v2[2])),
             (float(v2[0]), float(v2[1]), float(v2[2]))]
        )
    doc.saveas(out_path)


def _export_fbx_with_blender(glb_path: Path, out_path: Path) -> None:
    blender = shutil.which("blender")
    if not blender:
        raise ValueError("Blender is required for FBX export")

    script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath={str(glb_path)!r})
bpy.ops.export_scene.fbx(filepath={str(out_path)!r}, use_selection=False)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        script_path = Path(handle.name)
        handle.write(script)

    try:
        proc = subprocess.run(
            [blender, "--background", "--python", str(script_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out_path.is_file():
            stderr = (proc.stderr or proc.stdout or "").strip()
            raise ValueError(stderr or "Blender FBX export failed")
    finally:
        script_path.unlink(missing_ok=True)


def _export_via_usd2gltf(
    usdz_path: Path,
    out_path: Path,
    spec: Dict,
) -> None:
    kind = spec["kind"]
    if kind == "glb":
        _run_usd2gltf(usdz_path, out_path)
        return
    if kind == "gltf":
        _run_usd2gltf(usdz_path, out_path)
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        glb_path = tmp_dir / "intermediate.glb"
        _run_usd2gltf(usdz_path, glb_path)

        if kind == "mesh":
            _export_mesh_with_trimesh(glb_path, out_path, spec["mesh_extension"])
            return
        if kind == "dxf":
            _export_dxf_with_trimesh(glb_path, out_path)
            return
        if kind == "fbx":
            _export_fbx_with_blender(glb_path, out_path)
            return

    raise ValueError(f"Unsupported usd2gltf export kind: {kind}")


def build_export_file(
    session_path: Path,
    format_id: str,
    *,
    repack_usdz_fn: Callable[[Path, Path], None],
    ensure_current_usdz_fn: Callable[[Path], Path],
) -> ExportResult:
    spec = get_export_format_spec(format_id)
    if not spec:
        raise ValueError(f"Unknown export format: {format_id}")

    formats = {f["id"]: f for f in list_export_formats()}
    meta = formats.get(format_id)
    if not meta or not meta["available"]:
        reason = (meta or {}).get("unavailable_reason") or "Export format is not available"
        raise ValueError(reason)

    extract = session_path / "extract"
    if not extract.is_dir():
        raise ValueError("Session extract folder not found")

    exports_dir = session_path / "exports"
    exports_dir.mkdir(exist_ok=True)
    out_path = exports_dir / spec["filename"]
    kind = spec["kind"]

    if kind == "native":
        ensure_current_usdz_fn(session_path)
        current = session_path / "current.usdz"
        if not current.is_file():
            repack_usdz_fn(extract, current)
        shutil.copy2(current, out_path)
        return out_path, spec["filename"], spec["mime_type"]

    if kind == "arkit":
        if UsdUtils is None:
            raise ValueError("Pixar USD is not installed")
        root = _session_root_usda(extract)
        UsdUtils.CreateNewARKitUsdzPackage(str(root), str(out_path))
        return out_path, spec["filename"], spec["mime_type"]

    if kind == "flat":
        if Usd is None:
            raise ValueError("Pixar USD is not installed")
        root = _session_root_usda(extract)
        stage = Usd.Stage.Open(str(root))
        if not stage:
            raise ValueError(f"Failed to open USD stage: {root.name}")
        suffix = spec.get("flat_suffix", f".{spec['extension']}")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if not stage.Export(str(tmp_path)):
                raise ValueError("USD stage export failed")
            shutil.copy2(tmp_path, out_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return out_path, spec["filename"], spec["mime_type"]

    if kind == "package":
        if out_path.exists():
            out_path.unlink()
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(extract.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(extract).as_posix())
        return out_path, spec["filename"], spec["mime_type"]

    if spec.get("requires_usd2gltf"):
        ensure_current_usdz_fn(session_path)
        usdz_path = session_path / "current.usdz"
        if out_path.exists():
            out_path.unlink()
        _export_via_usd2gltf(usdz_path, out_path, spec)
        return out_path, spec["filename"], spec["mime_type"]

    raise ValueError(f"Unsupported export kind: {kind}")
