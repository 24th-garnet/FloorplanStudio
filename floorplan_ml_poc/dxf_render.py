"""Rasterize DXF line geometry (viewer XZ plane) to a plan image."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .constants import (
    RENDER_BG_RGB,
    RENDER_LINE_RGB,
    RENDER_LINE_WIDTH_PX,
    RENDER_LONG_EDGE_PX,
    RENDER_PADDING_RATIO,
)

RenderResult = Dict


def _bounds_from_positions(positions: Sequence[float]) -> Tuple[float, float, float, float]:
    min_x = min_z = math.inf
    max_x = max_z = -math.inf
    for i in range(0, len(positions) - 2, 3):
        x = float(positions[i])
        z = float(positions[i + 2])
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_z = min(min_z, z)
        max_z = max(max_z, z)
    return min_x, min_z, max_x, max_z


def _padded_bbox(
    min_x: float, min_z: float, max_x: float, max_z: float, padding_ratio: float
) -> Tuple[float, float, float, float]:
    span_x = max(max_x - min_x, 0.01)
    span_z = max(max_z - min_z, 0.01)
    pad_x = span_x * padding_ratio
    pad_z = span_z * padding_ratio
    return min_x - pad_x, min_z - pad_z, max_x + pad_x, max_z + pad_z


def _render_size(span_x: float, span_z: float, long_edge: int) -> Tuple[int, int]:
    if span_x >= span_z:
        w = long_edge
        h = max(1, int(round(long_edge * span_z / span_x)))
    else:
        h = long_edge
        w = max(1, int(round(long_edge * span_x / span_z)))
    return w, h


def xz_to_pixel(
    x: float,
    z: float,
    bbox_xz: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[float, float]:
    xmin, zmin, xmax, zmax = bbox_xz
    span_x = max(xmax - xmin, 1e-9)
    span_z = max(zmax - zmin, 1e-9)
    px = (x - xmin) / span_x * (width - 1)
    py = (z - zmin) / span_z * (height - 1)
    return px, py


def render_positions_to_image(
    positions: Sequence[float],
    bounds: Dict | None = None,
    long_edge_px: int = RENDER_LONG_EDGE_PX,
    padding_ratio: float = RENDER_PADDING_RATIO,
    line_width_px: float = RENDER_LINE_WIDTH_PX,
    unit_scale_to_meters: float = 1.0,
) -> RenderResult:
    if bounds and bounds.get("min") and bounds.get("max"):
        min_x, _, min_z = bounds["min"]
        max_x, _, max_z = bounds["max"]
    else:
        min_x, min_z, max_x, max_z = _bounds_from_positions(positions)

    bbox = _padded_bbox(min_x, min_z, max_x, max_z, padding_ratio)
    xmin, zmin, xmax, zmax = bbox
    width, height = _render_size(xmax - xmin, zmax - zmin, long_edge_px)

    image = np.full((height, width, 3), RENDER_BG_RGB, dtype=np.uint8)
    _draw_segments(image, positions, bbox, line_width_px)

    meta = {
        "render_width_px": width,
        "render_height_px": height,
        "bbox_xz_m": [xmin, zmin, xmax, zmax],
        "padding_ratio": padding_ratio,
        "line_width_px": line_width_px,
        "long_edge_px": long_edge_px,
        "unit_scale_to_meters": unit_scale_to_meters,
    }
    return {"image": image, "meta": meta}


def _draw_segments(
    image: np.ndarray,
    positions: Sequence[float],
    bbox: Tuple[float, float, float, float],
    line_width_px: float,
) -> None:
    h, w = image.shape[:2]
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        width = max(1, int(round(line_width_px)))
        for i in range(0, len(positions) - 5, 6):
            x0, z0 = float(positions[i]), float(positions[i + 2])
            x1, z1 = float(positions[i + 3]), float(positions[i + 5])
            p0 = xz_to_pixel(x0, z0, bbox, w, h)
            p1 = xz_to_pixel(x1, z1, bbox, w, h)
            draw.line([p0, p1], fill=RENDER_LINE_RGB, width=width)
        image[:] = np.array(pil)
        return
    except ImportError:
        pass

    # Bresenham-style fallback: 1px lines only.
    for i in range(0, len(positions) - 5, 6):
        x0, z0 = float(positions[i]), float(positions[i + 2])
        x1, z1 = float(positions[i + 3]), float(positions[i + 5])
        px0, py0 = xz_to_pixel(x0, z0, bbox, w, h)
        px1, py1 = xz_to_pixel(x1, z1, bbox, w, h)
        _draw_line_naive(image, int(px0), int(py0), int(px1), int(py1))


def _draw_line_naive(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    h, w = img.shape[:2]
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for t in range(steps + 1):
        x = int(round(x0 + (x1 - x0) * t / steps))
        y = int(round(y0 + (y1 - y0) * t / steps))
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = RENDER_LINE_RGB
