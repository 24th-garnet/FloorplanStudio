"""Extrude detected wall footprints into 3D mesh prisms."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Segment = Tuple[Vec3, Vec3]

DEFAULT_WALL_HEIGHT_M = 2.4


def _dist2_xz(a: Vec3, b: Vec3) -> float:
    dx, dz = b[0] - a[0], b[2] - a[2]
    return dx * dx + dz * dz


def footprint_corners(seg_a: Segment, seg_b: Segment) -> List[Vec3]:
    a0, a1 = seg_a
    b0, b1 = seg_b
    straight = _dist2_xz(a0, b0) + _dist2_xz(a1, b1)
    crossed = _dist2_xz(a0, b1) + _dist2_xz(a1, b0)
    if straight <= crossed:
        return [a0, a1, b1, b0]
    return [a0, a1, b0, b1]


def _parse_segment(raw: Sequence) -> Segment:
    a = raw[0]
    b = raw[1]
    return (
        (float(a[0]), float(a[1]), float(a[2])),
        (float(b[0]), float(b[1]), float(b[2])),
    )


def extrude_quad(quad: List[Vec3], height_m: float) -> Tuple[List[float], List[int]]:
    y0 = sum(p[1] for p in quad) / len(quad)
    y1 = y0 + height_m
    positions: List[float] = []
    for p in quad:
        positions.extend([p[0], y0, p[2]])
    for p in quad:
        positions.extend([p[0], y1, p[2]])

    # Bottom 0-3, top 4-7.
    indices = [
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6,
        3, 0, 4, 3, 4, 7,
    ]
    return positions, indices


def extrude_walls_from_extract(
    extract: Dict,
    height_m: float = DEFAULT_WALL_HEIGHT_M,
) -> Dict:
    if extract.get("source") == "ml":
        return extrude_walls_from_ml_extract(extract, height_m=height_m)
    height = float(height_m)
    if height <= 0:
        raise ValueError("wall height must be positive")

    positions: List[float] = []
    indices: List[int] = []
    vertex_offset = 0
    mesh_wall_count = 0

    for wall in extract.get("walls") or []:
        raw_segments = wall.get("segments")
        if not raw_segments or len(raw_segments) != 2:
            continue
        seg_a = _parse_segment(raw_segments[0])
        seg_b = _parse_segment(raw_segments[1])
        if _seg_len_xz(seg_a) < 1e-6 or _seg_len_xz(seg_b) < 1e-6:
            continue
        quad = footprint_corners(seg_a, seg_b)
        wall_positions, wall_indices = extrude_quad(quad, height)
        positions.extend(wall_positions)
        indices.extend(i + vertex_offset for i in wall_indices)
        vertex_offset += 8
        mesh_wall_count += 1

    return {
        "wall_count": int(extract.get("wall_count") or mesh_wall_count),
        "mesh_wall_count": mesh_wall_count,
        "height_m": round(height, 4),
        "source": extract.get("source", "heuristic"),
        "positions": positions,
        "indices": indices,
    }


def extrude_polygon_prism(
    polygon_xz: List[Sequence[float]],
    y_base: float,
    height_m: float,
) -> Tuple[List[float], List[int]]:
    """Extrude a simple polygon footprint (fan triangulation) in the XZ plane."""
    n = len(polygon_xz)
    if n < 3:
        return [], []
    y_top = y_base + height_m
    positions: List[float] = []
    for x, z in polygon_xz:
        positions.extend([float(x), y_base, float(z)])
    for x, z in polygon_xz:
        positions.extend([float(x), y_top, float(z)])

    indices: List[int] = []
    for i in range(1, n - 1):
        indices.extend([0, i + 1, i])
    for i in range(1, n - 1):
        indices.extend([n, n + i, n + i + 1])
    for i in range(n):
        j = (i + 1) % n
        indices.extend([i, j, n + j, i, n + j, n + i])
    return positions, indices


def extrude_walls_from_ml_extract(
    extract: Dict,
    height_m: float = DEFAULT_WALL_HEIGHT_M,
) -> Dict:
    height = float(height_m)
    if height <= 0:
        raise ValueError("wall height must be positive")

    positions: List[float] = []
    indices: List[int] = []
    vertex_offset = 0
    mesh_wall_count = 0

    for wall in extract.get("walls") or []:
        polygon_xz = wall.get("polygon_xz") or []
        if len(polygon_xz) < 3:
            continue
        y_base = float(wall.get("y_m", 0.0))
        wall_positions, wall_indices = extrude_polygon_prism(polygon_xz, y_base, height)
        if not wall_positions:
            continue
        positions.extend(wall_positions)
        indices.extend(i + vertex_offset for i in wall_indices)
        vertex_offset += len(wall_positions) // 3
        mesh_wall_count += 1

    return {
        "wall_count": int(extract.get("wall_count") or mesh_wall_count),
        "mesh_wall_count": mesh_wall_count,
        "height_m": round(height, 4),
        "source": "ml",
        "positions": positions,
        "indices": indices,
    }


def _seg_len_xz(seg: Segment) -> float:
    a, b = seg
    return math.hypot(b[0] - a[0], b[2] - a[2])
