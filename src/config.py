import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ON_KAGGLE = Path("/kaggle/input").exists()
ON_COLAB = bool(os.environ.get("COLAB_RELEASE_TAG"))
KAGGLE_DATA_DIR = Path("/kaggle/input/fairface")
COLAB_DATA_DIR = Path("/content/drive/MyDrive/fairface")
COLAB_OUT_DIR = Path("/content/drive/MyDrive/fairface-artifacts")
DEFAULT_DATA_DIR = (
    KAGGLE_DATA_DIR
    if ON_KAGGLE
    else COLAB_DATA_DIR
    if ON_COLAB
    else ROOT_DIR / "dataset"
)
DEFAULT_OUT_DIR = (
    Path("/kaggle/working")
    if ON_KAGGLE
    else COLAB_OUT_DIR
    if ON_COLAB
    else ROOT_DIR / "artifacts"
)
DATA_DIR = Path(os.environ.get("FAIRFACE_DATA_DIR", DEFAULT_DATA_DIR))
OUT_DIR = Path(os.environ.get("FAIRFACE_OUT_DIR", DEFAULT_OUT_DIR))
SPLIT_DIR = Path(os.environ.get("FAIRFACE_SPLIT_DIR", OUT_DIR / "splits"))
NUM_WORKERS = int(
    os.environ.get(
        "FAIRFACE_NUM_WORKERS",
        (min(os.cpu_count() or 1, 4) if ON_KAGGLE or ON_COLAB else 0),
    )
)
IMG_SIZE = 224
BATCH = 32
SEED = 42
CLASSES = ["Male", "Female"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
