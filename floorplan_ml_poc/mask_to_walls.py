"""Extract wall polygons from an ML segmentation mask (render resolution)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .coords import polygon_area_m2, polygon_pixels_to_xz
from .labels import WALL_CLASS_ID

CLOSING_KERNEL_PX = 3
APPROX_EPSILON_PX = 2.0
MIN_POLYGON_AREA_PX = 80.0
MIN_POLYGON_AREA_M2 = 0.04


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless is required for ML wall extraction. "
            "pip install opencv-python-headless"
        ) from exc


def wall_polygons_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """Binary wall mask (H,W) -> list of outer contours as (N,2) float arrays in pixel coords."""
    cv2 = _require_cv2()
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSING_KERNEL_PX, CLOSING_KERNEL_PX))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    polygons: List[np.ndarray] = []
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1:
            continue
        if cv2.contourArea(contours[i]) < MIN_POLYGON_AREA_PX:
            continue
        simplified = cv2.approxPolyDP(contours[i], APPROX_EPSILON_PX, closed=True)
        ring = simplified.reshape(-1, 2).astype(float)
        if len(ring) >= 3:
            polygons.append(ring)
    return polygons


def mask_to_wall_extract(
    mask: np.ndarray,
    render_meta: Dict,
    y_m: float = 0.0,
) -> Dict:
    """Build wall extract payload from ML wall mask at render resolution."""
    pixel_polys = wall_polygons_from_mask(mask)
    walls: List[Dict] = []
    highlight: List[float] = []

    for ring in pixel_polys:
        polygon_xz = polygon_pixels_to_xz(ring.tolist(), render_meta)
        area_m2 = polygon_area_m2(polygon_xz)
        if area_m2 < MIN_POLYGON_AREA_M2:
            continue
        wall_id = len(walls)
        walls.append(
            {
                "id": wall_id,
                "polygon_xz": polygon_xz,
                "area_m2": round(area_m2, 4),
                "y_m": round(float(y_m), 4),
            }
        )
        for i in range(len(polygon_xz)):
            x0, z0 = polygon_xz[i]
            x1, z1 = polygon_xz[(i + 1) % len(polygon_xz)]
            highlight.extend([x0, y_m, z0, x1, y_m, z1])

    return {
        "source": "ml",
        "wall_class_id": WALL_CLASS_ID,
        "wall_count": len(walls),
        "walls": walls,
        "highlight_positions": highlight,
        "render_meta": render_meta,
    }
