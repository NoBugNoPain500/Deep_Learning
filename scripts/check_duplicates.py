import argparse
import hashlib
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=Path("duplicates.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    train = pd.read_csv(next(args.data_dir.rglob("fairface_label_train.csv")))
    test = pd.read_csv(next(args.data_dir.rglob("fairface_label_val.csv")))
    if args.limit:
        train, test = train.head(args.limit), test.head(args.limit)
    hashes = {}
    for row in train.itertuples():
        path = args.data_dir / row.file
        if path.exists():
            hashes[hashlib.md5(path.read_bytes()).hexdigest()] = row.file
    duplicates = [
        {"train_file": value, "test_file": row.file}
        for row in test.itertuples()
        if (path := args.data_dir / row.file).exists()
        and hashlib.md5(path.read_bytes()).hexdigest() in hashes
        for value in [hashes[hashlib.md5(path.read_bytes()).hexdigest()]]
    ]
    pd.DataFrame(duplicates).to_csv(args.out, index=False)
    print(f"exact duplicate pairs: {len(duplicates)}")


if __name__ == "__main__":
    main()
