"""Build a minimal SVG from DXF line geometry (viewer XZ plane)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .dxf_render import _bounds_from_positions, _padded_bbox, _render_size


def positions_to_svg(
    positions: Sequence[float],
    bounds: Dict | None = None,
    padding_ratio: float = 0.08,
    long_edge_px: int = 2048,
    stroke_width_px: float = 1.5,
) -> Tuple[str, Dict]:
    """Return (svg_xml, render_meta) aligned with dxf_render.py."""
    if bounds and bounds.get("min") and bounds.get("max"):
        min_x, _, min_z = bounds["min"]
        max_x, _, max_z = bounds["max"]
    else:
        min_x, min_z, max_x, max_z = _bounds_from_positions(positions)

    bbox = _padded_bbox(min_x, min_z, max_x, max_z, padding_ratio)
    xmin, zmin, xmax, zmax = bbox
    width, height = _render_size(xmax - xmin, zmax - zmin, long_edge_px)

    view_w = max(xmax - xmin, 1e-9)
    view_h = max(zmax - zmin, 1e-9)

    def to_px(x: float, z: float) -> Tuple[float, float]:
        px = (x - xmin) / view_w * (width - 1)
        py = (z - zmin) / view_h * (height - 1)
        return px, py

    lines: List[str] = []
    for i in range(0, len(positions) - 5, 6):
        x0, z0 = float(positions[i]), float(positions[i + 2])
        x1, z1 = float(positions[i + 3]), float(positions[i + 5])
        px0, py0 = to_px(x0, z0)
        px1, py1 = to_px(x1, z1)
        lines.append(
            f'<line x1="{px0:.3f}" y1="{py0:.3f}" x2="{px1:.3f}" y2="{py1:.3f}" />'
        )

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n'
        f'  <g stroke="#000000" fill="none" stroke-width="{stroke_width_px:.3f}">\n'
    )
    svg += "\n".join(f"    {line}" for line in lines)
    svg += "\n  </g>\n</svg>\n"

    meta = {
        "render_width_px": width,
        "render_height_px": height,
        "bbox_xz_m": [xmin, zmin, xmax, zmax],
        "padding_ratio": padding_ratio,
        "line_width_px": stroke_width_px,
        "long_edge_px": long_edge_px,
        "svg_source": "viewer_positions",
    }
    return svg, meta


def export_native_dxf_svg(dxf_path: str, out_path: str) -> bool:
    """Optional: ezdxf native SVG export (skips text/hatch). Returns False on failure."""
    try:
        import ezdxf
        from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
    except ImportError:
        return False

    skip = frozenset({"MTEXT", "TEXT", "DIMENSION", "HATCH", "IMAGE", "ATTDEF", "ATTRIB"})

    def _filter(e):
        return e.dxftype() not in skip

    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        ctx = RenderContext(doc)
        backend = svg.SVGBackend()
        cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE)
        frontend = Frontend(ctx, backend, config=cfg)
        frontend.draw_layout(msp, filter_func=_filter)
        page = layout.Page(0, 0, layout.Units.mm, margins=layout.Margins.all(0))
        out = backend.get_string(page)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
        return True
    except Exception:
        return False
