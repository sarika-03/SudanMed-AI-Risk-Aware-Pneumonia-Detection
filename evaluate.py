from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from models.model import PneumoniaModel
from utils import compute_binary_classification_metrics, get_dataloaders, load_checkpoint


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="evaluate", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_binary_classification_metrics(y_true=y_true, y_pred=y_pred)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pneumonia model")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["custom_cnn", "mobilenet_v2"],
        default="custom_cnn",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaModel(model_type=args.model_type, pretrained=False).to(device)

    checkpoint = load_checkpoint(Path(args.checkpoint), model=model, map_location=device)
    print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'N/A')}")

    dataloaders = get_dataloaders(
        dataset_root=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    metrics = evaluate(model=model, dataloader=dataloaders["val"], device=device)

    print("Validation Metrics:")
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
