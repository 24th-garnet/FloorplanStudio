#!/usr/bin/env python3
"""Export MitUNet wall segmentation checkpoints to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from floorplan_ml_poc.constants import DEFAULT_ONNX_PATH, MODEL_INPUT_SIZE  # noqa: E402
from floorplan_ml_poc.model_specs import save_model_spec  # noqa: E402
from scripts.floorplan_ml_poc.download_models import MITUNET_WEIGHTS  # noqa: E402


def _build_mitunet():
    import segmentation_models_pytorch as smp

    aux = smp.Segformer(encoder_name="mit_b4", encoder_weights=None, in_channels=3, classes=1)
    model = smp.Unet(
        encoder_name="mit_b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
    )
    model.encoder = aux.encoder
    model.eval()
    return model


def export_mitunet(weights_path: Path, onnx_path: Path, *, model_id: str, label: str) -> None:
    import onnxruntime as ort
    import torch

    weights_path = Path(weights_path)
    onnx_path = Path(onnx_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    model = _build_mitunet()
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

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
        opset_version=18,
        dynamic_axes=None,
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"ONNX saved: {onnx_path}")
    print(f"PyTorch vs ONNX max abs diff: {max_diff:.6e}")

    save_model_spec(
        onnx_path,
        {
            "label": label,
            "family": "binary_unet",
            "preprocess": "stretch_imagenet",
            "input_size": MODEL_INPUT_SIZE,
            "num_classes": 1,
            "wall_threshold": 0.5,
            "source_repo": "https://github.com/aliasstudio/mitunet",
            "source_weights": weights_path.name,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export MitUNet checkpoints to ONNX")
    p.add_argument("--weights-dir", type=Path, default=DEFAULT_ONNX_PATH.parent / "weights")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_ONNX_PATH.parent)
    p.add_argument(
        "--only",
        choices=sorted(MITUNET_WEIGHTS.keys()),
        action="append",
        help="export a single model id (repeatable)",
    )
    args = p.parse_args()

    selected = args.only or list(MITUNET_WEIGHTS.keys())
    for model_id in selected:
        meta = MITUNET_WEIGHTS[model_id]
        weights = args.weights_dir / meta["filename"]
        onnx_path = args.out_dir / f"{model_id}.onnx"
        export_mitunet(weights, onnx_path, model_id=model_id, label=meta["label"])


if __name__ == "__main__":
    main()
