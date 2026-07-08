"""Convert pixel coordinates using render_meta bbox (viewer XZ plane)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Point2 = Tuple[float, float]


def pixel_to_xz(
    px: float,
    py: float,
    bbox_xz: Sequence[float],
    width: int,
    height: int,
) -> Point2:
    xmin, zmin, xmax, zmax = bbox_xz
    span_x = max(float(xmax) - float(xmin), 1e-9)
    span_z = max(float(zmax) - float(zmin), 1e-9)
    if width <= 1:
        x = float(xmin)
    else:
        x = float(xmin) + float(px) / (width - 1) * span_x
    if height <= 1:
        z = float(zmin)
    else:
        z = float(zmin) + float(py) / (height - 1) * span_z
    return (x, z)


def polygon_pixels_to_xz(
    ring: Sequence[Sequence[float]],
    render_meta: Dict,
) -> List[List[float]]:
    w = int(render_meta["render_width_px"])
    h = int(render_meta["render_height_px"])
    bbox = render_meta["bbox_xz_m"]
    out: List[List[float]] = []
    for pt in ring:
        x, z = pixel_to_xz(float(pt[0]), float(pt[1]), bbox, w, h)
        out.append([round(x, 5), round(z, 5)])
    return out


def polygon_area_m2(polygon_xz: Sequence[Sequence[float]]) -> float:
    if len(polygon_xz) < 3:
        return 0.0
    area = 0.0
    n = len(polygon_xz)
    for i in range(n):
        x0, z0 = polygon_xz[i]
        x1, z1 = polygon_xz[(i + 1) % n]
        area += float(x0) * float(z1) - float(x1) * float(z0)
    return abs(area) * 0.5
