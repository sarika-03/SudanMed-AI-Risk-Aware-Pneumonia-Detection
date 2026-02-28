from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import quantize_dynamic
from tqdm.auto import tqdm

from models.model import PneumoniaModel
from utils import (
    EarlyStopping,
    compute_binary_classification_metrics,
    get_dataloaders,
    get_model_size_mb,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    setup_logging,
)


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    loop = tqdm(dataloader, desc="train" if is_train else "val", leave=False)
    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

        loop.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_binary_classification_metrics(y_true=y_true, y_pred=y_pred)
    metrics["loss"] = float(epoch_loss)

    return epoch_loss, metrics


def quantize_and_save_model(
    model: nn.Module,
    save_dir: Path,
    base_filename: str = "best_model",
) -> Tuple[Path, float, float]:
    save_dir.mkdir(parents=True, exist_ok=True)

    float_model_path = save_dir / f"{base_filename}_float_state_dict.pth"
    torch.save(model.state_dict(), float_model_path)
    float_size_mb = get_model_size_mb(float_model_path)

    model_cpu = model.to("cpu").eval()
    quantized_model = quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model_path = save_dir / f"{base_filename}_quantized_state_dict.pth"
    torch.save(quantized_model.state_dict(), quantized_model_path)
    quantized_size_mb = get_model_size_mb(quantized_model_path)

    return quantized_model_path, float_size_mb, quantized_size_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset root directory")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["custom_cnn", "mobilenet_v2"],
        default="custom_cnn",
        help="Model architecture",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for MobileNetV2",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    logger = setup_logging(output_dir / "logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataloaders = get_dataloaders(
        dataset_root=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    model = PneumoniaModel(
        model_type=args.model_type,
        pretrained=args.pretrained,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4)
    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best_model.pth"

    for epoch in range(args.epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.epochs)

        _, train_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_loss, val_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        scheduler.step(val_loss)

        logger.info(
            "Train | loss: %.4f acc: %.4f precision: %.4f recall: %.4f f1: %.4f",
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1"],
        )
        logger.info(
            "Val   | loss: %.4f acc: %.4f precision: %.4f recall: %.4f f1: %.4f",
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                model_type=args.model_type,
                output_path=best_checkpoint_path,
            )
            logger.info("Saved new best model checkpoint to %s", best_checkpoint_path)

        if early_stopping.step(val_loss):
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    load_checkpoint(best_checkpoint_path, model=model, optimizer=None, map_location=device)
    quantized_path, float_size_mb, quantized_size_mb = quantize_and_save_model(
        model=model,
        save_dir=checkpoint_dir,
        base_filename="best_model",
    )

    logger.info("Model size before quantization: %.2f MB", float_size_mb)
    logger.info("Model size after quantization: %.2f MB", quantized_size_mb)
    logger.info("Quantized model saved to %s", quantized_path)


if __name__ == "__main__":
    main()
