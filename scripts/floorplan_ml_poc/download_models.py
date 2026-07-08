#!/usr/bin/env python3
"""Download third-party floorplan model weights for local ONNX export."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from floorplan_ml_poc.constants import (  # noqa: E402
    DEFAULT_ONNX_PATH,
    DEFAULT_WEIGHTS_PATH,
    HF_WEIGHTS_URL,
)

MODELS_DIR = DEFAULT_ONNX_PATH.parent
WEIGHTS_DIR = MODELS_DIR / "weights"

MITUNET_WEIGHTS = {
    "mitunet-cubicasa5k": {
        "url": (
            "https://media.githubusercontent.com/media/aliasstudio/mitunet/"
            "master/experiments/models/MitUNet_cubicasa-5k_a62_mit_b4_tversky_7606_20E.pth"
        ),
        "filename": "MitUNet_cubicasa-5k_a62_mit_b4_tversky_7606_20E.pth",
        "label": "MitUNet CubiCasa5k",
    },
    "mitunet-finetune": {
        "url": (
            "https://media.githubusercontent.com/media/aliasstudio/mitunet/"
            "master/experiments/models/mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth"
        ),
        "filename": "mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth",
        "label": "MitUNet Fine-tune",
    },
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1024:
        print(f"skip (exists): {dest.name}")
        return
    print(f"download: {dest.name}")
    urllib.request.urlretrieve(url, dest)


def download_yytsi_weights(dest: Path = DEFAULT_WEIGHTS_PATH) -> Path:
    _download(HF_WEIGHTS_URL, dest)
    return dest


def download_mitunet_weights(weights_dir: Path = WEIGHTS_DIR) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for model_id, meta in MITUNET_WEIGHTS.items():
        path = weights_dir / meta["filename"]
        _download(meta["url"], path)
        out[model_id] = path
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Download floorplan model weights")
    p.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    args = p.parse_args()
    download_yytsi_weights()
    download_mitunet_weights(args.weights_dir)
    print(f"weights ready in {args.weights_dir}")


if __name__ == "__main__":
    main()
