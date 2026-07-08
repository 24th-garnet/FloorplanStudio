#!/usr/bin/env python3
"""CLI: compare ML results for DXF PIL raster vs DXF→SVG→cairosvg."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from floorplan_ml_poc.compare_renders import run_render_path_compare  # noqa: E402
from floorplan_ml_poc.constants import DEFAULT_ONNX_PATH  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Compare DXF raster vs SVG+cairosvg ML inference")
    p.add_argument(
        "session_id",
        nargs="?",
        default="7770ec3f-a258-46aa-b302-c634d244570d",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_ONNX_PATH)
    p.add_argument("--sessions-dir", type=Path, default=ROOT / "sessions")
    args = p.parse_args()

    session_dir = args.sessions_dir / args.session_id
    if not session_dir.is_dir():
        print(f"Session not found: {session_dir}", file=sys.stderr)
        sys.exit(1)

    report = run_render_path_compare(session_dir, model_path=args.model)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
