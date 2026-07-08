#!/usr/bin/env python3
"""Export Yytsi/floorplan-to-3d UNet weights to ONNX."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from floorplan_ml_poc.constants import (  # noqa: E402
    DEFAULT_ONNX_PATH,
    DEFAULT_WEIGHTS_PATH,
    HF_WEIGHTS_URL,
    MODEL_INPUT_SIZE,
)
from floorplan_ml_poc.labels import NUM_CLASSES  # noqa: E402
from floorplan_ml_poc.model_specs import save_model_spec  # noqa: E402


def _download_weights(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return
    print(f"Downloading weights -> {dest}")
    urllib.request.urlretrieve(HF_WEIGHTS_URL, dest)


def _build_model():
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )


def _load_weights(model, weights_path: Path):
    from safetensors.torch import load_file

    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def export_onnx(weights_path: Path, onnx_path: Path, opset: int = 17) -> None:
    import torch

    weights_path = Path(weights_path)
    onnx_path = Path(onnx_path)
    if not weights_path.is_file():
        _download_weights(weights_path)

    model = _load_weights(_build_model(), weights_path)
    dummy = torch.randn(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes=None,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"ONNX saved: {onnx_path}")
    print(f"PyTorch vs ONNX max abs diff: {max_diff:.6e}")
    if max_diff >= 1e-3:
        print("WARNING: numerical diff is larger than expected", file=sys.stderr)

    save_model_spec(
        onnx_path,
        {
            "label": "Yytsi floorplan-walls",
            "family": "multiclass_unet",
            "preprocess": "letterbox_imagenet",
            "input_size": MODEL_INPUT_SIZE,
            "num_classes": NUM_CLASSES,
            "wall_class_id": 1,
            "source_repo": "https://huggingface.co/Yytsi/floorplan-to-3d-walls",
            "source_weights": weights_path.name,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export floorplan UNet to ONNX")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_ONNX_PATH)
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()
    export_onnx(args.weights, args.out, opset=args.opset)


if __name__ == "__main__":
    main()
