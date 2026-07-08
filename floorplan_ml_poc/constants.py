from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ONNX_PATH = BASE_DIR / "scripts" / "floorplan_ml_poc" / "models" / "floorplan-walls.onnx"
DEFAULT_WEIGHTS_PATH = BASE_DIR / "scripts" / "floorplan_ml_poc" / "models" / "best.safetensors"
ML_POC_SUBDIR = "ml_poc"

RENDER_LONG_EDGE_PX = 2048
RENDER_PADDING_RATIO = 0.08
RENDER_LINE_WIDTH_PX = 1.5
RENDER_BG_RGB = (255, 255, 255)
RENDER_LINE_RGB = (0, 0, 0)

MODEL_INPUT_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HF_WEIGHTS_URL = (
    "https://huggingface.co/Yytsi/floorplan-to-3d-walls/resolve/main/best.safetensors"
)
