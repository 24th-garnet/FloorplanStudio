"""Detect walls as close parallel line-segment pairs in the plan (viewer XZ plane)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Segment = Tuple[Vec3, Vec3]
Point2 = Tuple[float, float]

WALL_DETECT_DEFAULTS: Dict[str, float] = {
    "parallel_angle_tol_deg": 3.0,
    "wall_thickness_min_m": 0.06,
    "wall_thickness_max_m": 0.25,
    "min_wall_length_m": 0.8,
    "min_overlap_m": 0.5,
    "min_segment_length_m": 0.15,
    "max_endpoint_gap_m": 0.15,
}


def positions_to_segments(positions: Sequence[float]) -> List[Segment]:
    segments: List[Segment] = []
    for i in range(0, len(positions) - 5, 6):
        a: Vec3 = (float(positions[i]), float(positions[i + 1]), float(positions[i + 2]))
        b: Vec3 = (float(positions[i + 3]), float(positions[i + 4]), float(positions[i + 5]))
        segments.append((a, b))
    return segments


def _xz(p: Vec3) -> Point2:
    return (p[0], p[2])


def _seg_len_xz(seg: Segment) -> float:
    a, b = seg
    return math.hypot(b[0] - a[0], b[2] - a[2])


def _seg_dir_xz(seg: Segment) -> Optional[Point2]:
    a, b = seg
    dx, dz = b[0] - a[0], b[2] - a[2]
    length = math.hypot(dx, dz)
    if length < 1e-9:
        return None
    return (dx / length, dz / length)


def _angle_between_dirs(d1: Point2, d2: Point2) -> float:
    dot = abs(d1[0] * d2[0] + d1[1] * d2[1])
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(math.acos(dot))


def _perp_distance_xz(seg1: Segment, seg2: Segment) -> Optional[float]:
    direction = _seg_dir_xz(seg1)
    if direction is None:
        return None
    a1, _ = seg1
    a2, b2 = seg2
    mx = (a2[0] + b2[0]) * 0.5
    mz = (a2[2] + b2[2]) * 0.5
    vx, vz = mx - a1[0], mz - a1[2]
    return abs(-direction[1] * vx + direction[0] * vz)


def _project_scalar(p: Point2, origin: Point2, direction: Point2) -> float:
    return (p[0] - origin[0]) * direction[0] + (p[1] - origin[1]) * direction[1]


def _interval_overlap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    return max(0.0, min(max_a, max_b) - max(min_a, min_b))


def _projection_interval(seg: Segment, origin: Point2, direction: Point2) -> Tuple[float, float]:
    pa, pb = _xz(seg[0]), _xz(seg[1])
    sa = _project_scalar(pa, origin, direction)
    sb = _project_scalar(pb, origin, direction)
    return (min(sa, sb), max(sa, sb))


def _endpoint_gap_xz(seg1: Segment, seg2: Segment, direction: Point2) -> float:
    origin = _xz(seg1[0])
    i1 = _projection_interval(seg1, origin, direction)
    i2 = _projection_interval(seg2, origin, direction)
    if i1[1] < i2[0]:
        return i2[0] - i1[1]
    if i2[1] < i1[0]:
        return i1[0] - i2[1]
    return 0.0


def _segment_key(seg: Segment) -> Tuple[Vec3, Vec3]:
    return seg if seg[0] <= seg[1] else (seg[1], seg[0])


def extract_walls_from_positions(
    positions: Sequence[float],
    params: Optional[Dict[str, float]] = None,
) -> Dict:
    cfg = {**WALL_DETECT_DEFAULTS, **(params or {})}
    segments = positions_to_segments(positions)
    min_seg_len = float(cfg["min_segment_length_m"])
    filtered: List[Segment] = []
    for seg in segments:
        if _seg_len_xz(seg) >= min_seg_len:
            filtered.append(seg)

    angle_tol = float(cfg["parallel_angle_tol_deg"])
    thick_min = float(cfg["wall_thickness_min_m"])
    thick_max = float(cfg["wall_thickness_max_m"])
    min_wall_len = float(cfg["min_wall_length_m"])
    min_overlap = float(cfg["min_overlap_m"])
    max_endpoint_gap = float(cfg["max_endpoint_gap_m"])

    candidates: List[Dict] = []
    n = len(filtered)
    for i in range(n):
        for j in range(i + 1, n):
            seg_a = filtered[i]
            seg_b = filtered[j]
            dir_a = _seg_dir_xz(seg_a)
            dir_b = _seg_dir_xz(seg_b)
            if dir_a is None or dir_b is None:
                continue
            if _angle_between_dirs(dir_a, dir_b) > angle_tol:
                continue
            thickness = _perp_distance_xz(seg_a, seg_b)
            if thickness is None or thickness < thick_min or thickness > thick_max:
                continue
            length_a = _seg_len_xz(seg_a)
            length_b = _seg_len_xz(seg_b)
            shorter = min(length_a, length_b)
            if shorter < min_wall_len:
                continue
            direction = dir_a
            origin = _xz(seg_a[0])
            ia = _projection_interval(seg_a, origin, direction)
            ib = _projection_interval(seg_b, origin, direction)
            overlap = _interval_overlap(ia[0], ia[1], ib[0], ib[1])
            if overlap < min_overlap:
                continue
            if _endpoint_gap_xz(seg_a, seg_b, direction) > max_endpoint_gap:
                continue
            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "overlap": overlap,
                    "thickness_m": thickness,
                    "length_m": overlap,
                    "seg_a": seg_a,
                    "seg_b": seg_b,
                }
            )

    candidates.sort(key=lambda c: c["overlap"], reverse=True)
    used: set[int] = set()
    walls: List[Dict] = []
    highlight: List[float] = []

    for cand in candidates:
        i, j = cand["i"], cand["j"]
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        seg_a: Segment = cand["seg_a"]
        seg_b: Segment = cand["seg_b"]
        wall_id = len(walls)
        walls.append(
            {
                "id": wall_id,
                "thickness_m": round(cand["thickness_m"], 4),
                "length_m": round(cand["length_m"], 4),
                "segments": [
                    [list(seg_a[0]), list(seg_a[1])],
                    [list(seg_b[0]), list(seg_b[1])],
                ],
            }
        )
        for seg in (seg_a, seg_b):
            highlight.extend([*seg[0], *seg[1]])

    return {
        "wall_count": len(walls),
        "segment_count": len(segments),
        "filtered_segment_count": len(filtered),
        "params": cfg,
        "walls": walls,
        "highlight_positions": highlight,
    }
