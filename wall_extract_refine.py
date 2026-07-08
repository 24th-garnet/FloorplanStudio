"""Commercial-style post-processing for parallel-pair wall extraction."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from wall_extract import WALL_DETECT_DEFAULTS, extract_walls_from_positions

Point2 = Tuple[float, float]
Wall = Dict

REFINE_DEFAULTS: Dict[str, float] = {
    "orthogonal_snap_deg": 3.0,
    "gap_close_m": 0.03,
    "merge_gap_m": 0.03,
    "thickness_tol_m": 0.025,
    "collinear_dist_tol_m": 0.05,
    "min_wall_length_m": 0.5,
    "min_segment_length_m": 0.001,
    "structural_merge_lateral_m": 0.12,
    "structural_merge_angle_deg": 3.0,
    "structural_merge_min_overlap_m": 0.4,
    "structural_merge_gap_m": 0.15,
}


def _dist2(a: Point2, b: Point2) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _lerp_y(seg_a: Sequence, seg_b: Sequence) -> float:
    ys = [float(seg_a[0][1]), float(seg_a[1][1]), float(seg_b[0][1]), float(seg_b[1][1])]
    return sum(ys) / len(ys)


def _wall_midline(wall: Wall) -> Tuple[Point2, Point2, float, float]:
    sa, sb = wall["segments"]
    a0: Point2 = (float(sa[0][0]), float(sa[0][2]))
    a1: Point2 = (float(sa[1][0]), float(sa[1][2]))
    b0: Point2 = (float(sb[0][0]), float(sb[0][2]))
    b1: Point2 = (float(sb[1][0]), float(sb[1][2]))
    straight = _dist2(a0, b0) + _dist2(a1, b1)
    crossed = _dist2(a0, b1) + _dist2(a1, b0)
    if straight <= crossed:
        m0 = ((a0[0] + b0[0]) * 0.5, (a0[1] + b0[1]) * 0.5)
        m1 = ((a1[0] + b1[0]) * 0.5, (a1[1] + b1[1]) * 0.5)
        off_a, off_b = a0, b0
    else:
        m0 = ((a0[0] + b1[0]) * 0.5, (a0[1] + b1[1]) * 0.5)
        m1 = ((a1[0] + b0[0]) * 0.5, (a1[1] + b0[1]) * 0.5)
        off_a, off_b = a0, b0
    thickness = float(wall.get("thickness_m") or _dist2(off_a, off_b))
    y_m = _lerp_y(sa, sb)
    return m0, m1, thickness, y_m


def _direction(m0: Point2, m1: Point2) -> Optional[Point2]:
    dx, dz = m1[0] - m0[0], m1[1] - m0[1]
    length = math.hypot(dx, dz)
    if length < 1e-9:
        return None
    return (dx / length, dz / length)


def _snap_angle_deg(angle_deg: float, tol_deg: float) -> float:
    candidates = (0.0, 90.0, 180.0, -90.0, -180.0)
    best = angle_deg
    best_delta = 1e9
    for cand in candidates:
        delta = abs(((angle_deg - cand + 180.0) % 360.0) - 180.0)
        if delta < best_delta:
            best_delta = delta
            best = cand
    if best_delta <= tol_deg:
        return best
    return angle_deg


def _snap_midline(m0: Point2, m1: Point2, tol_deg: float) -> Tuple[Point2, Point2]:
    direction = _direction(m0, m1)
    if direction is None:
        return m0, m1
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    snapped = _snap_angle_deg(angle, tol_deg)
    if snapped == angle:
        return m0, m1
    rad = math.radians(snapped)
    ux, uz = math.cos(rad), math.sin(rad)
    center = ((m0[0] + m1[0]) * 0.5, (m0[1] + m1[1]) * 0.5)
    half = _dist2(m0, m1) * 0.5
    return (
        (center[0] - ux * half, center[1] - uz * half),
        (center[0] + ux * half, center[1] + uz * half),
    )


def _project_scalar(p: Point2, origin: Point2, direction: Point2) -> float:
    return (p[0] - origin[0]) * direction[0] + (p[1] - origin[1]) * direction[1]


def _normal(direction: Point2) -> Point2:
    return (-direction[1], direction[0])


def _midline_to_wall(m0: Point2, m1: Point2, thickness: float, y_m: float) -> Wall:
    direction = _direction(m0, m1)
    if direction is None:
        raise ValueError("degenerate wall midline")
    nx, nz = _normal(direction)
    half = thickness * 0.5
    a0 = [m0[0] + nx * half, y_m, m0[1] + nz * half]
    a1 = [m1[0] + nx * half, y_m, m1[1] + nz * half]
    b0 = [m0[0] - nx * half, y_m, m0[1] - nz * half]
    b1 = [m1[0] - nx * half, y_m, m1[1] - nz * half]
    length_m = _dist2(m0, m1)
    return {
        "thickness_m": round(thickness, 4),
        "length_m": round(length_m, 4),
        "segments": [[a0, a1], [b0, b1]],
    }


def _intervals_merge(intervals: List[Tuple[float, float]], gap_m: float) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap_m:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _quantize_direction(direction: Point2, tol_deg: float) -> Point2:
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    snapped = _snap_angle_deg(angle, tol_deg)
    rad = math.radians(snapped)
    return (math.cos(rad), math.sin(rad))


def _merge_collinear_midlines(
    items: List[Tuple[Point2, Point2, float, float]],
    cfg: Dict[str, float],
) -> List[Tuple[Point2, Point2, float, float]]:
    tol_deg = float(cfg["orthogonal_snap_deg"])
    gap_m = float(cfg["merge_gap_m"])
    thick_tol = float(cfg["thickness_tol_m"])
    collinear_tol = float(cfg["collinear_dist_tol_m"])

    buckets: Dict[Tuple[int, int], List[Tuple[Point2, Point2, float, float]]] = {}
    for m0, m1, thickness, y_m in items:
        direction = _quantize_direction(_direction(m0, m1) or (1.0, 0.0), tol_deg)
        angle_key = int(round(math.degrees(math.atan2(direction[1], direction[0]))))
        thick_key = int(round(thickness / thick_tol))
        buckets.setdefault((angle_key, thick_key), []).append((m0, m1, thickness, y_m))

    merged: List[Tuple[Point2, Point2, float, float]] = []
    for (_angle_key, _thick_key), group in buckets.items():
        if not group:
            continue
        ref_thickness = sum(item[2] for item in group) / len(group)
        ref_y = sum(item[3] for item in group) / len(group)
        direction = _quantize_direction(_direction(group[0][0], group[0][1]) or (1.0, 0.0), tol_deg)
        normal = _normal(direction)

        chains: List[List[Tuple[Point2, Point2, float, float]]] = []
        for m0, m1, thickness, y_m in group:
            center = ((m0[0] + m1[0]) * 0.5, (m0[1] + m1[1]) * 0.5)
            offset = _project_scalar(center, (0.0, 0.0), normal)
            placed = False
            for chain in chains:
                chain_offset = _project_scalar(
                    ((chain[0][0][0] + chain[0][1][0]) * 0.5, (chain[0][0][1] + chain[0][1][1]) * 0.5),
                    (0.0, 0.0),
                    normal,
                )
                if abs(offset - chain_offset) <= collinear_tol:
                    chain.append((m0, m1, thickness, y_m))
                    placed = True
                    break
            if not placed:
                chains.append([(m0, m1, thickness, y_m)])

        for chain in chains:
            origin = chain[0][0]
            intervals: List[Tuple[float, float]] = []
            for m0, m1, _t, _y in chain:
                s0 = _project_scalar(m0, origin, direction)
                s1 = _project_scalar(m1, origin, direction)
                intervals.append((min(s0, s1), max(s0, s1)))
            for start, end in _intervals_merge(intervals, gap_m):
                p0 = (
                    origin[0] + direction[0] * start,
                    origin[1] + direction[1] * start,
                )
                p1 = (
                    origin[0] + direction[0] * end,
                    origin[1] + direction[1] * end,
                )
                merged.append((p0, p1, ref_thickness, ref_y))
    return merged


def _close_endpoint_gaps(
    items: List[Tuple[Point2, Point2, float, float]],
    gap_m: float,
) -> List[Tuple[Point2, Point2, float, float]]:
    if len(items) < 2:
        return items
    endpoints: List[Tuple[int, int, Point2]] = []
    for idx, (m0, m1, _t, _y) in enumerate(items):
        endpoints.append((idx, 0, m0))
        endpoints.append((idx, 1, m1))

    adjusted = [list(item) for item in items]
    used: set[Tuple[int, int]] = set()
    for i, (wi, end_i, pi) in enumerate(endpoints):
        if (wi, end_i) in used:
            continue
        best = None
        best_dist = gap_m
        for j, (wj, end_j, pj) in enumerate(endpoints):
            if i == j or wi == wj:
                continue
            if (wj, end_j) in used:
                continue
            dist = _dist2(pi, pj)
            if dist <= best_dist:
                best_dist = dist
                best = (wj, end_j, pj)
        if best is None:
            continue
        wj, end_j, pj = best
        meet = ((pi[0] + pj[0]) * 0.5, (pi[1] + pj[1]) * 0.5)
        mi = list(adjusted[wi])
        mj = list(adjusted[wj])
        if end_i == 0:
            mi[0] = meet
        else:
            mi[1] = meet
        if end_j == 0:
            mj[0] = meet
        else:
            mj[1] = meet
        adjusted[wi] = mi
        adjusted[wj] = mj
        used.add((wi, end_i))
        used.add((wj, end_j))
    return [(m0, m1, t, y) for m0, m1, t, y in adjusted]


def _angle_between_dirs_deg(d1: Point2, d2: Point2) -> float:
    dot = abs(d1[0] * d2[0] + d1[1] * d2[1])
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(math.acos(dot))


def _interval_overlap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    return max(0.0, min(max_a, max_b) - max(min_a, min_b))


def _interval_gap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    if a[1] < b[0]:
        return b[0] - a[1]
    if b[1] < a[0]:
        return a[0] - b[1]
    return 0.0


def _midline_interval(
    m0: Point2,
    m1: Point2,
    origin: Point2,
    direction: Point2,
) -> Tuple[float, float]:
    s0 = _project_scalar(m0, origin, direction)
    s1 = _project_scalar(m1, origin, direction)
    return (min(s0, s1), max(s0, s1))


def _can_structural_merge(
    item_a: Tuple[Point2, Point2, float, float],
    item_b: Tuple[Point2, Point2, float, float],
    cfg: Dict[str, float],
) -> bool:
    m0a, m1a, _, _ = item_a
    m0b, m1b, _, _ = item_b
    da = _direction(m0a, m1a)
    db = _direction(m0b, m1b)
    if da is None or db is None:
        return False
    angle_tol = float(cfg["structural_merge_angle_deg"])
    if _angle_between_dirs_deg(da, db) > angle_tol:
        return False

    direction = _quantize_direction(da, angle_tol)
    normal = _normal(direction)
    ca = ((m0a[0] + m1a[0]) * 0.5, (m0a[1] + m1a[1]) * 0.5)
    cb = ((m0b[0] + m1b[0]) * 0.5, (m0b[1] + m1b[1]) * 0.5)
    lateral = abs(_project_scalar(ca, (0.0, 0.0), normal) - _project_scalar(cb, (0.0, 0.0), normal))
    if lateral > float(cfg["structural_merge_lateral_m"]):
        return False

    origin = m0a
    ia = _midline_interval(m0a, m1a, origin, direction)
    ib = _midline_interval(m0b, m1b, origin, direction)
    min_overlap = float(cfg["structural_merge_min_overlap_m"])
    if _interval_overlap(ia[0], ia[1], ib[0], ib[1]) >= min_overlap:
        return True
    gap_tol = float(cfg["structural_merge_gap_m"])
    return _interval_gap(ia, ib) <= gap_tol


def _collapse_structural_cluster(
    cluster: List[Tuple[Point2, Point2, float, float]],
    cfg: Dict[str, float],
) -> List[Tuple[Point2, Point2, float, float]]:
    if not cluster:
        return []
    if len(cluster) == 1:
        return cluster

    longest = max(cluster, key=lambda item: _dist2(item[0], item[1]))
    direction = _quantize_direction(
        _direction(longest[0], longest[1]) or (1.0, 0.0),
        float(cfg["structural_merge_angle_deg"]),
    )
    normal = _normal(direction)
    ref_point = longest[0]

    intervals: List[Tuple[float, float]] = []
    offsets: List[float] = []
    thicknesses: List[float] = []
    ys: List[float] = []
    for m0, m1, thickness, y_m in cluster:
        center = ((m0[0] + m1[0]) * 0.5, (m0[1] + m1[1]) * 0.5)
        offsets.append(_project_scalar(center, ref_point, normal))
        intervals.append(_midline_interval(m0, m1, ref_point, direction))
        thicknesses.append(thickness)
        ys.append(y_m)

    avg_offset = sum(offsets) / len(offsets)
    avg_thickness = sum(thicknesses) / len(thicknesses)
    avg_y = sum(ys) / len(ys)
    gap_m = float(cfg["structural_merge_gap_m"])
    min_len = float(cfg["min_wall_length_m"])

    merged: List[Tuple[Point2, Point2, float, float]] = []
    for start, end in _intervals_merge(intervals, gap_m):
        if end - start < min_len:
            continue
        p0 = (
            ref_point[0] + direction[0] * start + normal[0] * avg_offset,
            ref_point[1] + direction[1] * start + normal[1] * avg_offset,
        )
        p1 = (
            ref_point[0] + direction[0] * end + normal[0] * avg_offset,
            ref_point[1] + direction[1] * end + normal[1] * avg_offset,
        )
        merged.append((p0, p1, avg_thickness, avg_y))
    return merged


def _merge_structural_parallel_midlines(
    items: List[Tuple[Point2, Point2, float, float]],
    cfg: Dict[str, float],
) -> List[Tuple[Point2, Point2, float, float]]:
    n = len(items)
    if n <= 1:
        return items

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _can_structural_merge(items[i], items[j], cfg):
                union(i, j)

    groups: Dict[int, List[Tuple[Point2, Point2, float, float]]] = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)

    merged: List[Tuple[Point2, Point2, float, float]] = []
    for group in groups.values():
        merged.extend(_collapse_structural_cluster(group, cfg))
    return merged


def _midlines_to_walls(
    midlines: List[Tuple[Point2, Point2, float, float]],
    cfg: Dict[str, float],
) -> Tuple[List[Wall], List[float]]:
    min_len = float(cfg["min_wall_length_m"])
    walls: List[Wall] = []
    highlight: List[float] = []
    for wall_id, (m0, m1, thickness, y_m) in enumerate(midlines):
        if _dist2(m0, m1) < min_len:
            continue
        wall = _midline_to_wall(m0, m1, thickness, y_m)
        wall["id"] = wall_id
        walls.append(wall)
        for seg in wall["segments"]:
            highlight.extend([*seg[0], *seg[1]])
    return walls, highlight


def refine_wall_extract(extract: Dict, params: Optional[Dict[str, float]] = None) -> Dict:
    cfg = {**REFINE_DEFAULTS, **(params or {})}
    min_len = float(cfg["min_wall_length_m"])
    tol_deg = float(cfg["orthogonal_snap_deg"])
    gap_close = float(cfg["gap_close_m"])

    midlines: List[Tuple[Point2, Point2, float, float]] = []
    for wall in extract.get("walls") or []:
        m0, m1, thickness, y_m = _wall_midline(wall)
        m0, m1 = _snap_midline(m0, m1, tol_deg)
        if _dist2(m0, m1) >= min_len * 0.5:
            midlines.append((m0, m1, thickness, y_m))

    midlines = _merge_collinear_midlines(midlines, cfg)
    midlines = _close_endpoint_gaps(midlines, gap_close)

    walls, highlight = _midlines_to_walls(midlines, cfg)

    return {
        **extract,
        "source": "heuristic_refined",
        "wall_count": len(walls),
        "walls": walls,
        "highlight_positions": highlight,
        "refine_params": cfg,
        "refine_stats": {
            "input_wall_count": int(extract.get("wall_count") or 0),
            "output_wall_count": len(walls),
        },
    }


def structural_wall_extract(extract: Dict, params: Optional[Dict[str, float]] = None) -> Dict:
    refined = refine_wall_extract(extract, params)
    cfg = {**REFINE_DEFAULTS, **(params or {})}
    input_count = int(refined.get("wall_count") or 0)

    midlines: List[Tuple[Point2, Point2, float, float]] = []
    for wall in refined.get("walls") or []:
        m0, m1, thickness, y_m = _wall_midline(wall)
        if _dist2(m0, m1) >= float(cfg["min_wall_length_m"]) * 0.5:
            midlines.append((m0, m1, thickness, y_m))

    midlines = _merge_structural_parallel_midlines(midlines, cfg)
    walls, highlight = _midlines_to_walls(midlines, cfg)
    base_stats = dict(refined.get("refine_stats") or {})

    return {
        **refined,
        "source": "heuristic_structural",
        "method": "heuristic_structural",
        "wall_count": len(walls),
        "walls": walls,
        "highlight_positions": highlight,
        "refine_params": cfg,
        "refine_stats": {
            **base_stats,
            "structural_input_wall_count": input_count,
            "structural_output_wall_count": len(walls),
        },
    }


def extract_walls_refined_from_positions(
    positions: Sequence[float],
    params: Optional[Dict[str, float]] = None,
) -> Dict:
    detect_params = {**WALL_DETECT_DEFAULTS, **(params or {})}
    base = extract_walls_from_positions(positions, detect_params)
    return refine_wall_extract(base, params)


def extract_walls_structural_from_positions(
    positions: Sequence[float],
    params: Optional[Dict[str, float]] = None,
) -> Dict:
    detect_params = {**WALL_DETECT_DEFAULTS, **(params or {})}
    base = extract_walls_from_positions(positions, detect_params)
    return structural_wall_extract(base, params)
