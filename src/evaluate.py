import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import config
from .data import load_splits, make_loaders
from .experiments import EXPERIMENTS
from .model import build_model, count_params
from .utils import save_json


def predict(
    model: torch.nn.Module, loader: object, df: pd.DataFrame, device: torch.device
) -> pd.DataFrame:
    rows, offset = [], 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            probabilities = model(images.to(device)).softmax(1).cpu()
            for label, probability in zip(labels.tolist(), probabilities.tolist()):
                row = df.iloc[offset]
                rows.append(
                    {
                        "file": row["file"],
                        "y_true": label,
                        "y_pred": int(probability.index(max(probability))),
                        "prob_male": probability[0],
                        "prob_female": probability[1],
                        "race": row["race"],
                        "age": row["age"],
                    }
                )
                offset += 1
    return pd.DataFrame(rows)


def compute_metrics(preds: pd.DataFrame) -> dict:
    report = classification_report(
        preds.y_true, preds.y_pred, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(preds.y_true, preds.y_pred),
        "precision_male": report.get("0", {}).get("precision", 0),
        "recall_male": report.get("0", {}).get("recall", 0),
        "f1_male": report.get("0", {}).get("f1-score", 0),
        "precision_female": report.get("1", {}).get("precision", 0),
        "recall_female": report.get("1", {}).get("recall", 0),
        "f1_female": report.get("1", {}).get("f1-score", 0),
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0),
        "confusion_matrix": confusion_matrix(
            preds.y_true, preds.y_pred, labels=[0, 1]
        ).tolist(),
    }


def measure_latency(model: torch.nn.Module, device: torch.device, n: int = 50) -> float:
    model.eval()
    sample = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
    with torch.no_grad():
        for _ in range(min(5, n)):
            model(sample)
        started = time.perf_counter()
        for _ in range(n):
            model(sample)
    return (time.perf_counter() - started) / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=EXPERIMENTS, default="e1_main")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=config.OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=config.SPLIT_DIR)
    args = parser.parse_args()
    output = args.out_dir / args.exp
    metrics_path = output / (
        "metrics.json" if args.split == "test" else "val_metrics.json"
    )
    if args.split == "test" and metrics_path.exists() and not args.force:
        raise SystemExit("Test metrics da ton tai; dung --force de chay lai.")
    exp = EXPERIMENTS[args.exp]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(exp.arch, pretrained=False).to(device)
    state = torch.load(output / "best.pth", map_location=device)
    model.load_state_dict(state["model"])
    cfg = SimpleNamespace(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        seed=config.SEED,
        batch=config.BATCH,
        num_workers=config.NUM_WORKERS,
        augment=False,
    )
    loaders = make_loaders(cfg, smoke=args.smoke)
    frames = load_splits(args.data_dir, args.split_dir, config.SEED)
    index = 1 if args.split == "val" else 2
    sampled_frame = getattr(loaders[index].dataset, "df", frames[index])
    preds = predict(model, loaders[index], sampled_frame, device)
    preds.to_csv(output / f"{args.split}_preds.csv", index=False)
    metrics = compute_metrics(preds)
    metrics.update(
        {
            "n_params": count_params(model)[0],
            "latency_s_per_img": measure_latency(
                model, device, n=5 if args.smoke else 50
            ),
        }
    )
    save_json(metrics, metrics_path)


if __name__ == "__main__":
    main()
