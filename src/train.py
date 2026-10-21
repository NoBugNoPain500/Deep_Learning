import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from . import config
from .data import class_weights, load_splits, make_loaders
from .experiments import EXPERIMENTS, Experiment
from .model import build_model, count_params, freeze_bn, param_groups, set_trainable
from .utils import get_logger, save_json, set_seed

LOGGER = get_logger(__name__)


def run_epoch(
    model: nn.Module,
    loader: Any,
    criterion: Any,
    device: torch.device,
    optimizer: Any = None,
    scaler: Any = None,
    amp: bool = True,
) -> dict:
    training = optimizer is not None
    model.train(training)
    predictions, targets, losses = [], [], []
    use_amp = amp and device.type == "cuda"
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        losses.append(loss.item() * len(labels))
        predictions.extend(logits.detach().argmax(1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return {
        "loss": sum(losses) / max(1, len(targets)),
        "acc": accuracy_score(targets, predictions),
        "f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def fit(
    model: nn.Module,
    loaders: tuple[Any, Any, Any],
    optimizer: Any,
    scheduler: Any,
    exp: Experiment,
    out_dir: str | Path,
    tag: str,
    resume: bool = False,
    criterion: Any = None,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    criterion = criterion or nn.CrossEntropyLoss()
    checkpoint = out_dir / "last.pth"
    history: list[dict] = []
    start, best, stale = 0, float("-inf"), 0
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start, best, history = (
            state["epoch"] + 1,
            state["best_metric"],
            state.get("history", []),
        )
    for epoch in range(start, exp.stage2_epochs):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model, loaders[0], criterion, device, optimizer=optimizer, amp=True
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, loaders[1], criterion, device, amp=False)
        if scheduler is not None:
            scheduler.step()
        metric = (
            val_metrics["acc"]
            if exp.selection_metric == "val_acc"
            else val_metrics["f1"]
        )
        record = {
            "epoch": epoch + 1,
            "stage": tag,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "best_metric": max(best, metric),
                "history": history,
            },
            checkpoint,
        )
        if metric > best:
            best, stale = metric, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best,
                    "history": history,
                },
                out_dir / f"best_{tag}.pth",
            )
        else:
            stale += 1
        LOGGER.info(
            "ep=%s stage=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f lr=%s time=%.1fs",
            epoch + 1,
            tag,
            train_metrics["loss"],
            train_metrics["acc"],
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["f1"],
            optimizer.param_groups[0]["lr"],
            record["seconds"],
        )
        if stale >= exp.patience:
            break
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=EXPERIMENTS, default="e1_main")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=config.OUT_DIR)
    parser.add_argument("--weights-path")
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=config.SPLIT_DIR)
    args = parser.parse_args()
    set_seed(config.SEED)
    exp = EXPERIMENTS[args.exp]
    if args.epochs_override is not None or args.smoke:
        epochs = args.epochs_override or 1
        exp = Experiment(
            **{**exp.to_dict(), "stage1_epochs": epochs, "stage2_epochs": epochs}
        )
    output = args.out_dir / exp.name
    model = build_model(
        exp.arch, exp.pretrained and not args.smoke, weights_path=args.weights_path
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    cfg = SimpleNamespace(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        seed=config.SEED,
        batch=config.BATCH,
        num_workers=config.NUM_WORKERS,
        augment=exp.augment,
    )
    loaders = make_loaders(cfg, smoke=args.smoke)
    train_df, _, _ = load_splits(args.data_dir, args.split_dir, config.SEED)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_df).to(device) if exp.use_class_weights else None
    )
    history: list[dict] = []
    if exp.stage1_epochs:
        set_trainable(model, "head")
        if exp.freeze_bn_stage1:
            freeze_bn(model)
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=exp.stage1_lr
        )
        stage1 = Experiment(**{**exp.to_dict(), "stage2_epochs": exp.stage1_epochs})
        history.extend(
            fit(
                model,
                loaders,
                optimizer,
                None,
                stage1,
                output,
                "stage1",
                args.resume,
                criterion,
            )
        )
        best_stage1 = output / "best_stage1.pth"
        if best_stage1.exists():
            model.load_state_dict(torch.load(best_stage1, map_location=device)["model"])
    if exp.stage2_epochs:
        optimizer = torch.optim.Adam(
            param_groups(model, exp.unfreeze, exp.lr_backbone, exp.lr_head)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=exp.stage2_epochs
        )
        history.extend(
            fit(
                model,
                loaders,
                optimizer,
                scheduler,
                exp,
                output,
                "stage2",
                args.resume,
                criterion,
            )
        )
    final = output / "best.pth"
    best_file = output / ("best_stage2.pth" if exp.stage2_epochs else "best_stage1.pth")
    if best_file.exists():
        final.write_bytes(best_file.read_bytes())
    save_json(history, output / "history.json")
    save_json(
        {
            "experiment": exp.to_dict(),
            "seed": config.SEED,
            "n_params": count_params(model),
        },
        output / "config.json",
    )


if __name__ == "__main__":
    main()
