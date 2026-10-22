import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ON_KAGGLE = Path("/kaggle/input").exists()
KAGGLE_DATA_DIR = Path("/kaggle/input/fairface")
DATA_DIR = KAGGLE_DATA_DIR if ON_KAGGLE else ROOT_DIR / "dataset"
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else ROOT_DIR / "artifacts"
SPLIT_DIR = OUT_DIR / "splits" if ON_KAGGLE else ROOT_DIR / "splits"
NUM_WORKERS = (os.cpu_count() or 1) if ON_KAGGLE else 0
IMG_SIZE = 224
BATCH = 32
SEED = 42
CLASSES = ["Male", "Female"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
