#!/usr/bin/env python3
"""Download weights and export all local floorplan ONNX models for batch compare."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    py = sys.executable
    _run([py, str(SCRIPTS / "download_models.py")])
    _run([py, str(SCRIPTS / "export_onnx.py")])
    _run([py, str(SCRIPTS / "export_mitunet_onnx.py")])

    from floorplan_ml_poc.model_registry import list_onnx_models

    models = list_onnx_models()
    print("\nONNX models ready:")
    for m in models:
        print(f"  - {m['id']}: {m.get('label', m['id'])} ({m.get('family')})")


if __name__ == "__main__":
    main()
