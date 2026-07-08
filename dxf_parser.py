"""Parse DXF drawings into line segments for the Three.js viewer."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import ezdxf
except ImportError:  # pragma: no cover
    ezdxf = None

Vec3 = Tuple[float, float, float]
Segment = Tuple[Vec3, Vec3]

INSUNITS_TO_METERS = {
    0: 1.0,
    1: 0.0254,
    2: 0.3048,
    3: 1609.344,
    4: 0.001,
    5: 0.01,
    6: 1.0,
}

MAX_SEGMENTS = 750_000


def _vec3_from_ezdxf(v) -> Vec3:
    # R12 and newer DXF may expose coordinates as Vec3 or as vertex entities.
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z") and not hasattr(v, "dxf"):
        return (float(v.x), float(v.y), float(v.z))
    loc = v.dxf.location
    return (float(loc.x), float(loc.y), float(loc.z))


def _segment_key(a: Vec3, b: Vec3) -> Tuple[Vec3, Vec3]:
    return (a, b) if a <= b else (b, a)


def _arc_segments(center: Vec3, radius: float, start_deg: float, end_deg: float, z: float, steps: int = 16) -> Iterable[Segment]:
    if radius <= 0:
        return
    start = math.radians(start_deg)
    end = math.radians(end_deg)
    if end < start:
        end += math.tau
    span = end - start
    if span <= 1e-9:
        return
    steps = max(4, min(steps, int(math.ceil(span / (math.pi / 18)))))
    cx, cy, _ = center
    prev = None
    for i in range(steps + 1):
        t = start + span * (i / steps)
        pt = (cx + radius * math.cos(t), cy + radius * math.sin(t), z)
        if prev is not None:
            yield (prev, pt)
        prev = pt


def _polyline_segments(points: List[Vec3], closed: bool) -> Iterable[Segment]:
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        yield (points[i], points[i + 1])
    if closed:
        yield (points[-1], points[0])


def _polyface_segments(vertices) -> Iterable[Segment]:
    coords: List[Vec3] = []

    for vtx in vertices:
        dxf = vtx.dxf
        vtx0 = getattr(dxf, "vtx0", None)
        if vtx0 is not None:
            refs = [getattr(dxf, f"vtx{i}", None) for i in range(4)]
            refs = [r for r in refs if r is not None and r != 0]
            if len(refs) < 2:
                continue
            pts = []
            for ref in refs:
                idx = abs(int(ref)) - 1
                if 0 <= idx < len(coords):
                    pts.append(coords[idx])
            for i in range(len(pts)):
                yield (pts[i], pts[(i + 1) % len(pts)])
            continue

        loc = dxf.location
        if abs(loc.x) + abs(loc.y) + abs(loc.z) < 1e-9:
            continue
        coords.append(_vec3_from_ezdxf(vtx))


def _polyline_vertex_points(vertices) -> List[Vec3]:
    pts: List[Vec3] = []
    for vtx in vertices:
        dxf = vtx.dxf
        if getattr(dxf, "vtx0", None) is not None:
            continue
        loc = dxf.location
        if abs(loc.x) + abs(loc.y) + abs(loc.z) < 1e-9:
            continue
        pts.append(_vec3_from_ezdxf(vtx))
    return pts


def _polymesh_grid_segments(vertices, m_count: int, n_count: int) -> Iterable[Segment]:
    coords: List[Vec3] = []
    for vtx in vertices:
        dxf = vtx.dxf
        if getattr(dxf, "vtx0", None) is not None:
            continue
        loc = dxf.location
        if abs(loc.x) + abs(loc.y) + abs(loc.z) < 1e-9:
            continue
        coords.append(_vec3_from_ezdxf(vtx))

    if m_count < 2 or n_count < 1 or len(coords) < 2:
        return

    def at(i: int, j: int) -> Optional[Vec3]:
        idx = i * n_count + j
        if 0 <= idx < len(coords):
            return coords[idx]
        return None

    for i in range(m_count):
        for j in range(n_count):
            a = at(i, j)
            if a is None:
                continue
            b = at(i + 1, j)
            if b is not None:
                yield (a, b)
            c = at(i, j + 1)
            if c is not None:
                yield (a, c)


def _entity_segments(entity) -> Iterable[Segment]:
    dxftype = entity.dxftype()
    if dxftype == "LINE":
        yield (_vec3_from_ezdxf(entity.dxf.start), _vec3_from_ezdxf(entity.dxf.end))
        return

    if dxftype == "3DFACE":
        # 3D face defined by up to 4 vertices; vtx2/vtx3 can duplicate to make triangles.
        pts: List[Vec3] = []
        for attr in ("vtx0", "vtx1", "vtx2", "vtx3"):
            v = getattr(entity.dxf, attr, None)
            if v is None:
                continue
            pts.append((float(v.x), float(v.y), float(v.z)))
        # Remove duplicate trailing points.
        compact: List[Vec3] = []
        for p in pts:
            if not compact or compact[-1] != p:
                compact.append(p)
        while len(compact) >= 2 and compact[-1] == compact[-2]:
            compact.pop()
        # Some exporters repeat vtx2=vtx3 for triangles.
        if len(compact) >= 4 and compact[2] == compact[3]:
            compact = compact[:3]
        yield from _polyline_segments(compact, closed=True)
        return

    if dxftype == "LWPOLYLINE":
        pts = []
        elev = float(getattr(entity.dxf, "elevation", 0.0) or 0.0)
        for v in entity.get_points("xy"):
            pts.append((float(v[0]), float(v[1]), elev))
        yield from _polyline_segments(pts, bool(entity.closed))
        return

    if dxftype == "POLYLINE":
        verts = list(entity.vertices)
        segments: List[Segment] = []
        segments.extend(_polyface_segments(verts))
        m_count = int(getattr(entity.dxf, "m_count", 0) or 0)
        n_count = int(getattr(entity.dxf, "n_count", 0) or 0)
        if m_count > 1 and n_count >= 1:
            segments.extend(_polymesh_grid_segments(verts, m_count, n_count))
        if not segments:
            has_face_refs = any(getattr(v.dxf, "vtx0", None) is not None for v in verts)
            if not has_face_refs:
                flags = int(getattr(entity.dxf, "flags", 0) or 0)
                pts = _polyline_vertex_points(verts)
                segments.extend(_polyline_segments(pts, bool(flags & 1)))
        yield from segments
        return

    if dxftype == "ARC":
        center = _vec3_from_ezdxf(entity.dxf.center)
        z = center[2]
        yield from _arc_segments(
            center,
            float(entity.dxf.radius),
            float(entity.dxf.start_angle),
            float(entity.dxf.end_angle),
            z,
        )
        return

    if dxftype == "CIRCLE":
        center = _vec3_from_ezdxf(entity.dxf.center)
        z = center[2]
        yield from _arc_segments(center, float(entity.dxf.radius), 0.0, 360.0, z, steps=48)
        return


def _expand_entities(entity) -> Iterable:
    if entity.dxftype() == "INSERT":
        try:
            for virtual in entity.virtual_entities():
                yield from _expand_entities(virtual)
        except Exception:
            return
        return
    yield entity


def _entity_layer_name(entity) -> str:
    layer = getattr(entity.dxf, "layer", None)
    name = str(layer).strip() if layer is not None else ""
    return name or "0"


def _flatten_segments_to_positions(
    segments: Iterable[Segment],
    to_viewer,
    bounds: Optional[Dict[str, List[float]]] = None,
) -> List[float]:
    flat: List[float] = []
    vmin = bounds["min"] if bounds else [math.inf, math.inf, math.inf]
    vmax = bounds["max"] if bounds else [-math.inf, -math.inf, -math.inf]
    for a, b in segments:
        va = to_viewer(a)
        vb = to_viewer(b)
        flat.extend([*va, *vb])
        if bounds is not None:
            for pt in (va, vb):
                for i in range(3):
                    vmin[i] = min(vmin[i], pt[i])
                    vmax[i] = max(vmax[i], pt[i])
    if bounds is not None:
        bounds["min"] = vmin
        bounds["max"] = vmax
    return flat


def parse_dxf_file(path: Path) -> Dict:
    if ezdxf is None:
        raise RuntimeError("ezdxf is not installed")

    doc = ezdxf.readfile(str(path))
    insunits = int(doc.header.get("$INSUNITS", 0) or 0)
    unit_scale = INSUNITS_TO_METERS.get(insunits, 1.0)

    raw_segments: Set[Tuple[Vec3, Vec3]] = set()
    layer_segments: Dict[str, Set[Tuple[Vec3, Vec3]]] = defaultdict(set)
    entity_counts: Counter = Counter()
    skipped = 0
    total_segments = 0

    for entity in doc.modelspace():
        entity_counts[entity.dxftype()] += 1
        try:
            for draw_entity in _expand_entities(entity):
                layer_name = _entity_layer_name(draw_entity)
                for seg in _entity_segments(draw_entity):
                    key = _segment_key(seg[0], seg[1])
                    if key in layer_segments[layer_name]:
                        continue
                    layer_segments[layer_name].add(key)
                    if key not in raw_segments:
                        raw_segments.add(key)
                        total_segments += 1
                        if total_segments >= MAX_SEGMENTS:
                            raise ValueError(f"DXF exceeds segment limit ({MAX_SEGMENTS})")
        except ValueError:
            raise
        except Exception:
            skipped += 1

    if not raw_segments:
        raise ValueError("No drawable geometry found in DXF")

    all_pts = [p for seg in raw_segments for p in seg]
    raw_extent = 0.0
    for i in range(3):
        lo = min(p[i] for p in all_pts)
        hi = max(p[i] for p in all_pts)
        raw_extent = max(raw_extent, hi - lo)
    if raw_extent > 500.0 and unit_scale >= 1.0:
        unit_scale = 0.001
    cx = sum(p[0] for p in all_pts) / len(all_pts)
    cy = sum(p[1] for p in all_pts) / len(all_pts)
    cz = sum(p[2] for p in all_pts) / len(all_pts)
    center = (cx, cy, cz)

    def to_viewer(pt: Vec3) -> Vec3:
        x = (pt[0] - cx) * unit_scale
        y = (pt[1] - cy) * unit_scale
        z = (pt[2] - cz) * unit_scale
        return (x, z, y)

    bounds = {"min": [math.inf, math.inf, math.inf], "max": [-math.inf, -math.inf, -math.inf]}
    flat = _flatten_segments_to_positions(raw_segments, to_viewer, bounds)

    layers_out: List[Dict] = []
    for layer_name in sorted(layer_segments.keys(), key=lambda name: (-len(layer_segments[name]), name)):
        layer_flat = _flatten_segments_to_positions(layer_segments[layer_name], to_viewer)
        if not layer_flat:
            continue
        layers_out.append(
            {
                "id": layer_name,
                "label": layer_name,
                "segment_count": len(layer_segments[layer_name]),
                "positions": layer_flat,
            }
        )

    return {
        "positions": flat,
        "segment_count": len(raw_segments),
        "layer_count": len(layers_out),
        "layers": layers_out,
        "bounds": bounds,
        "center_dxf": [cx, cy, cz],
        "insunits": insunits,
        "unit_scale_to_meters": unit_scale,
        "entity_counts": dict(entity_counts.most_common(20)),
        "skipped_entities": skipped,
        "source_filename": path.name,
    }
