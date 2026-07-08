"""Render SVG with cairosvg (matches floorplan-to-3d background convention)."""

from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np

from .constants import IMAGENET_MEAN

# floorplan-to-3d svg_render.py uses ImageNet mean as PNG background.
_IMAGENET_MEAN_RGB = "rgb(124, 116, 104)"


def _require_cairosvg():
    try:
        import cairosvg

        return cairosvg
    except ImportError as exc:
        raise RuntimeError(
            "cairosvg is required for SVG rendering. pip install cairosvg"
        ) from exc


def render_svg_to_rgb(
    svg: str | bytes,
    width: int,
    height: int,
    *,
    background: str = _IMAGENET_MEAN_RGB,
) -> np.ndarray:
    """SVG string/bytes -> RGB uint8 (height, width, 3)."""
    cairosvg = _require_cairosvg()
    if isinstance(svg, str):
        svg = svg.encode("utf-8")
    png_bytes = cairosvg.svg2png(
        bytestring=svg,
        output_width=int(width),
        output_height=int(height),
        background_color=background,
    )
    try:
        from PIL import Image

        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except ImportError as exc:
        raise RuntimeError("pillow is required to decode cairosvg output") from exc


def render_svg_file_to_rgb(path: str, width: int, height: int) -> np.ndarray:
    cairosvg = _require_cairosvg()
    png_bytes = cairosvg.svg2png(
        url=path,
        output_width=int(width),
        output_height=int(height),
        background_color=_IMAGENET_MEAN_RGB,
    )
    from PIL import Image

    return np.array(Image.open(BytesIO(png_bytes)).convert("RGB"), dtype=np.uint8)
