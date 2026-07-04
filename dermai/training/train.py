from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dermai.data.ham10000 import (
    HAM10000Dataset,
    get_train_transform,
    get_val_transform,
    normalize_metadata,
)
from dermai.labels import HAM10000_LABELS, LABEL_TO_INDEX
from dermai.models.efficientnet import create_efficientnet_b0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on HAM10000.")
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--image-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/efficientnet_b0"), type=Path)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--val-size", default=0.2, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download/use ImageNet pretrained weights.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def choose_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_class_weights(train_df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    label_indices = train_df["dx"].map(LABEL_TO_INDEX).to_numpy()
    counts = np.bincount(label_indices, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    progress = tqdm(dataloader, leave=False, desc="train" if is_train else "val")
    for images, labels, _ in progress:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_targets.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = total_loss / max(len(dataloader.dataset), 1)
    return mean_loss, np.array(all_targets), np.array(all_preds)


def compute_metrics(targets: np.ndarray, preds: np.ndarray, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    recalls = recall_score(targets, preds, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "class_recall": {
            class_name: float(recalls[index]) for index, class_name in enumerate(class_names)
        },
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    image_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": "efficientnet_b0",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": HAM10000_LABELS,
            "label_to_index": LABEL_TO_INDEX,
            "image_size": image_size,
            "metrics": metrics,
        },
        path,
    )


def append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = normalize_metadata(pd.read_csv(args.metadata_csv))
    counts = metadata["dx"].value_counts()
    stratify = metadata["dx"] if (counts >= 2).all() else None
    train_df, val_df = train_test_split(
        metadata,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify,
    )

    train_dataset = HAM10000Dataset(
        metadata_df=train_df,
        image_dirs=args.image_dir,
        transform=get_train_transform(args.image_size),
    )
    val_dataset = HAM10000Dataset(
        metadata_df=val_df,
        image_dirs=args.image_dir,
        transform=get_val_transform(args.image_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Using device: {device}")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    print("Class counts:")
    print(train_df["dx"].value_counts().sort_index().to_string())

    model = create_efficientnet_b0(
        num_classes=len(HAM10000_LABELS),
        pretrained=not args.no_pretrained,
    ).to(device)
    class_weights = make_class_weights(train_df, len(HAM10000_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_targets, train_preds = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_targets, val_preds = run_epoch(model, val_loader, criterion, device)
        train_metrics = compute_metrics(train_targets, train_preds, HAM10000_LABELS)
        val_metrics = compute_metrics(val_targets, val_preds, HAM10000_LABELS)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            **{
                f"val_recall_{name}": value
                for name, value in val_metrics["class_recall"].items()
            },
        }
        append_history(args.output_dir / "history.csv", row)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            {"train": train_metrics, "validation": val_metrics, "val_loss": val_loss},
            args.image_size,
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                {"train": train_metrics, "validation": val_metrics, "val_loss": val_loss},
                args.image_size,
            )


if __name__ == "__main__":
    main()
