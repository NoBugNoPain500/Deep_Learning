import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def group_accuracy(preds: pd.DataFrame, by: str = "race") -> pd.DataFrame:
    if by not in {"race", "age"}:
        raise ValueError("by phai la race hoac age")
    result = (
        preds.assign(correct=preds.y_true == preds.y_pred)
        .groupby(by)
        .agg(accuracy=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    return result


def plot_history(history_json: str | Path, path: str | Path) -> None:
    history = pd.DataFrame(json.loads(Path(history_json).read_text(encoding="utf-8")))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="val")
    axes[1].plot(history.epoch, history.train_acc, label="train")
    axes[1].plot(history.epoch, history.val_acc, label="val")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_confusion(cm: list[list[int]], classes: list[str], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def compare_experiments(artifacts_dir: str | Path) -> pd.DataFrame:
    rows = []
    for directory in Path(artifacts_dir).iterdir():
        metrics, cfg = directory / "metrics.json", directory / "config.json"
        if metrics.exists() and cfg.exists():
            metric_data, config_data = (
                json.loads(metrics.read_text()),
                json.loads(cfg.read_text()),
            )
            rows.append(
                {
                    "experiment": directory.name,
                    "accuracy": metric_data.get("accuracy"),
                    "macro_f1": metric_data.get("macro_f1"),
                    "n_params": metric_data.get(
                        "n_params", config_data.get("n_params")
                    ),
                    "latency_s_per_img": metric_data.get("latency_s_per_img"),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(Path(artifacts_dir) / "comparison.csv", index=False)
    result.to_markdown(Path(artifacts_dir) / "comparison.md", index=False)
    return result


def top_errors(preds: pd.DataFrame, k: int = 24) -> pd.DataFrame:
    errors = preds[preds.y_true != preds.y_pred].copy()
    errors["confidence"] = errors[["prob_male", "prob_female"]].max(axis=1)
    return errors.sort_values("confidence", ascending=False).head(k)
