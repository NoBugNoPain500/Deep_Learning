from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .config import (
    BATCH,
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_WORKERS,
    SEED,
)
from .utils import seed_worker


def _find_csv(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if path.exists():
        return path
    matches = list(data_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Khong tim thay {name} trong DATA_DIR={data_dir}")
    return matches[0]


def _image_path(data_dir: Path, relative: str | Path) -> Path:
    direct = data_dir / relative
    if direct.exists():
        return direct
    return data_dir / "fairface-img-margin025-trainval" / relative


def _add_label(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["label"] = result["gender"].map({"Male": 0, "Female": 1})
    return result


def load_splits(
    data_dir: str | Path, split_dir: str | Path, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir, split_dir = Path(data_dir), Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        split_dir / name for name in ("train_split.csv", "val_split.csv", "test.csv")
    ]
    if all(path.exists() for path in paths):
        frames = [pd.read_csv(path) for path in paths]
    else:
        train = pd.read_csv(_find_csv(data_dir, "fairface_label_train.csv"))
        test = pd.read_csv(_find_csv(data_dir, "fairface_label_val.csv"))
        required = {"file", "age", "gender", "race"}
        if not required.issubset(train.columns) or not required.issubset(test.columns):
            raise ValueError("CSV phai co cac cot file, age, gender, race")
        stratify = train["gender"].astype(str) + "_" + train["race"].astype(str)
        train, val = train_test_split(
            train, test_size=0.1, random_state=seed, stratify=stratify
        )
        frames = [train, val, test]
        for frame, path in zip(frames, paths):
            frame.to_csv(path, index=False)
    for frame in frames:
        if not set(frame["gender"].dropna()).issubset(set(CLASSES)):
            raise AssertionError("gender chi duoc chua Male/Female")
        if not {"file", "age", "gender", "race"}.issubset(frame.columns):
            raise ValueError("Split thieu cot bat buoc")
    if (
        set(frames[0].file) & set(frames[1].file)
        or set(frames[0].file) & set(frames[2].file)
        or set(frames[1].file) & set(frames[2].file)
    ):
        raise AssertionError("Cac split bi trung file")
    for frame in frames:
        for relative in frame["file"].head(5):
            image_path = _image_path(data_dir, relative)
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Thieu anh {image_path}; hay kiem tra DATA_DIR"
                )
    return tuple(_add_label(frame) for frame in frames)  # type: ignore[return-value]


class FairFaceGender(Dataset[tuple[torch.Tensor, int]]):
    classes = CLASSES

    def __init__(self, df: pd.DataFrame, root: str | Path, transform: Any) -> None:
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        image_path = _image_path(self.root, row["file"])
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        return self.transform(image), int(row["label"])


def get_transforms(train: bool, augment: bool = True) -> transforms.Compose:
    operations: list[Any] = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train and augment:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, hue=0.05),
            ]
        )
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transforms.Compose(operations)


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = (
        df["label"]
        .value_counts()
        .reindex(range(len(CLASSES)), fill_value=1)
        .astype(float)
    )
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(weights.to_numpy(), dtype=torch.float32)


def _sample(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df
    return df.groupby("label", group_keys=False).sample(
        n=max(1, limit // len(CLASSES)), random_state=seed
    )


def make_loaders(
    cfg: Any, smoke: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def value(name: str, default: Any = None) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(name, default)
        return getattr(cfg, name, default)

    data_dir = Path(value("data_dir"))
    split_dir = Path(value("split_dir"))
    train_df, val_df, test_df = load_splits(data_dir, split_dir, value("seed", SEED))
    if smoke:
        train_df, val_df, test_df = (
            _sample(df, limit, SEED)
            for df, limit in zip((train_df, val_df, test_df), (48, 16, 16))
        )
    batch = 8 if smoke else value("batch", BATCH)
    workers = 0 if smoke else value("num_workers", NUM_WORKERS)
    augment = value("augment", True)
    datasets = (
        FairFaceGender(train_df, data_dir, get_transforms(True, augment)),
        FairFaceGender(val_df, data_dir, get_transforms(False)),
        FairFaceGender(test_df, data_dir, get_transforms(False)),
    )
    generator = torch.Generator().manual_seed(value("seed", SEED))
    common = {
        "batch_size": batch,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(datasets[0], shuffle=True, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
    )
