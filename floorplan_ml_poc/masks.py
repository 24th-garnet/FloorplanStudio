"""Mask generation, metrics, and comparison overlays."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .dxf_render import xz_to_pixel


def heuristic_wall_mask(
    walls: List[Dict],
    render_meta: Dict,
) -> np.ndarray:
    """Rasterize parallel-pair walls as filled bands at render resolution."""
    h = int(render_meta["render_height_px"])
    w = int(render_meta["render_width_px"])
    bbox = tuple(render_meta["bbox_xz_m"])
    mask = np.zeros((h, w), dtype=np.uint8)

    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(mask)
        draw = ImageDraw.Draw(pil)
        for wall in walls:
            segs = wall.get("segments") or []
            if len(segs) < 2:
                continue
            poly = _wall_band_polygon(segs[0], segs[1], bbox, w, h)
            if poly:
                draw.polygon(poly, fill=255)
        return np.array(pil, dtype=np.uint8)
    except ImportError:
        for wall in walls:
            segs = wall.get("segments") or []
            if len(segs) < 2:
                continue
            poly = _wall_band_polygon(segs[0], segs[1], bbox, w, h)
            if not poly:
                continue
            _fill_polygon_naive(mask, poly)
        return mask


def ml_wall_mask_from_logits(logits: np.ndarray) -> np.ndarray:
    """Logits (4,H,W) or (1,4,H,W) -> binary wall mask at model resolution."""
    if logits.ndim == 4:
        logits = logits[0]
    pred = np.argmax(logits, axis=0)
    from .labels import WALL_CLASS_ID

    return (pred == WALL_CLASS_ID).astype(np.uint8) * 255


def compute_binary_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    p = pred > 0
    t = target > 0
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    pred_sum = p.sum()
    target_sum = t.sum()
    iou = float(inter / union) if union else 0.0
    precision = float(inter / pred_sum) if pred_sum else 0.0
    recall = float(inter / target_sum) if target_sum else 0.0
    return {
        "wall_iou": round(iou, 4),
        "wall_precision": round(precision, 4),
        "wall_recall": round(recall, 4),
        "ml_wall_pixels": int(pred_sum),
        "heuristic_wall_pixels": int(target_sum),
    }


def make_overlay_compare(
    render_rgb: np.ndarray,
    ml_mask: np.ndarray,
    heuristic_mask: np.ndarray,
) -> np.ndarray:
    """Three-column panel: input | ML wall (orange) | heuristic wall (blue)."""
    h, w = render_rgb.shape[:2]
    ml_panel = _tint_mask(render_rgb, ml_mask, (255, 122, 0))
    heur_panel = _tint_mask(render_rgb, heuristic_mask, (0, 120, 255))
    return np.concatenate([render_rgb, ml_panel, heur_panel], axis=1)


def make_diff_image(ml_mask: np.ndarray, heuristic_mask: np.ndarray) -> np.ndarray:
    """RGB diff: FP=red (ML only), FN=blue (heuristic only), TP=gray."""
    h, w = ml_mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ml = ml_mask > 0
    he = heuristic_mask > 0
    tp = np.logical_and(ml, he)
    fp = np.logical_and(ml, np.logical_not(he))
    fn = np.logical_and(he, np.logical_not(ml))
    out[tp] = (180, 180, 180)
    out[fp] = (220, 60, 60)
    out[fn] = (60, 100, 220)
    return out


def _tint_mask(base: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = base.copy()
    active = mask > 0
    tint = np.array(color, dtype=np.float32)
    blended = out[active].astype(np.float32) * 0.45 + tint * 0.55
    out[active] = blended.astype(np.uint8)
    return out


def _wall_band_polygon(
    seg_a: Sequence,
    seg_b: Sequence,
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> List[Tuple[float, float]]:
    a0 = (float(seg_a[0][0]), float(seg_a[0][2]))
    a1 = (float(seg_a[1][0]), float(seg_a[1][2]))
    b0 = (float(seg_b[0][0]), float(seg_b[0][2]))
    b1 = (float(seg_b[1][0]), float(seg_b[1][2]))
    p_a0 = xz_to_pixel(a0[0], a0[1], bbox, width, height)
    p_a1 = xz_to_pixel(a1[0], a1[1], bbox, width, height)
    p_b0 = xz_to_pixel(b0[0], b0[1], bbox, width, height)
    p_b1 = xz_to_pixel(b1[0], b1[1], bbox, width, height)
    return [p_a0, p_a1, p_b1, p_b0]


def _fill_polygon_naive(mask: np.ndarray, poly: List[Tuple[float, float]]) -> None:
    h, w = mask.shape
    ys = [p[1] for p in poly]
    y_min = max(0, int(math.floor(min(ys))))
    y_max = min(h - 1, int(math.ceil(max(ys))))
    for y in range(y_min, y_max + 1):
        xs: List[float] = []
        n = len(poly)
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            if y0 == y1:
                continue
            if (y >= min(y0, y1)) and (y < max(y0, y1)):
                xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            x_start = max(0, int(math.floor(xs[i])))
            x_end = min(w - 1, int(math.ceil(xs[i + 1])))
            mask[y, x_start : x_end + 1] = 255
