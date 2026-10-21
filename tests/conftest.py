from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture()
def fake_data(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "fairface"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    rows = []
    for index in range(40):
        split = "train" if index < 32 else "val"
        relative = f"{split}/{index}.jpg"
        Image.fromarray(np.full((224, 224, 3), index, dtype=np.uint8)).save(
            root / relative
        )
        rows.append(
            {
                "file": relative,
                "age": "20-29",
                "gender": "Male" if index % 2 == 0 else "Female",
                "race": ["White", "Black", "East Asian", "Indian"][index % 4],
                "service_test": "train",
            }
        )
    frame = pd.DataFrame(rows)
    frame.iloc[:32].to_csv(root / "fairface_label_train.csv", index=False)
    frame.iloc[32:].to_csv(root / "fairface_label_val.csv", index=False)
    return root, tmp_path / "splits"
