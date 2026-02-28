from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from gradcam import GradCAM, save_gradcam_visualization
from models.model import PneumoniaModel
from utils import CLASS_NAMES, load_checkpoint, load_image_for_inference


def enable_mc_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def predict_single_image_mc_dropout(
    model: PneumoniaModel,
    image_tensor: torch.Tensor,
    device: torch.device,
    passes: int = 20,
) -> tuple[int, float, float, torch.Tensor]:
    enable_mc_dropout(model)

    mc_probs: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(image_tensor.to(device))
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            mc_probs.append(probs)

    stacked_probs = torch.stack(mc_probs, dim=0)
    mean_probs = stacked_probs.mean(dim=0)
    std_probs = stacked_probs.std(dim=0)

    confidence, pred_idx = torch.max(mean_probs, dim=0)
    uncertainty = float(std_probs[1].item())

    return int(pred_idx.item()), float(confidence.item()), uncertainty, mean_probs


def stratify_risk(pneumonia_probability: float) -> tuple[str, str]:
    if pneumonia_probability > 0.75:
        return (
            "High Risk",
            "High pneumonia likelihood. Prioritize urgent clinical review and confirmatory assessment.",
        )
    if pneumonia_probability >= 0.50:
        return (
            "Moderate Risk",
            "Moderate pneumonia likelihood. Recommend prompt clinician review and close monitoring.",
        )
    return (
        "Low Risk",
        "Lower pneumonia likelihood. Continue standard clinical evaluation and monitor symptoms.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image inference with Grad-CAM")
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["custom_cnn", "mobilenet_v2"],
        default="custom_cnn",
    )
    parser.add_argument("--output", type=str, default="outputs/gradcam_inference.png")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    try:
        original_image, image_tensor = load_image_for_inference(
            image_path=image_path,
            image_size=args.image_size,
        )
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return
    except Exception as err:
        print(f"Failed to load image: {err}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaModel(model_type=args.model_type, pretrained=False).to(device)

    try:
        load_checkpoint(checkpoint_path=checkpoint_path, model=model, map_location=device)
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return
    except Exception as err:
        print(f"Failed to load checkpoint: {err}")
        return

    pred_idx, confidence, uncertainty, mean_probs = predict_single_image_mc_dropout(
        model=model,
        image_tensor=image_tensor,
        device=device,
        passes=20,
    )
    predicted_class = CLASS_NAMES[pred_idx]
    pneumonia_probability = float(mean_probs[1].item())
    risk_level, advisory = stratify_risk(pneumonia_probability)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence score (mean probability): {confidence:.4f}")
    print(f"Uncertainty score (std): {uncertainty * 100:.2f}%")
    print(f"Risk level: {risk_level}")
    print(f"Advisory: {advisory}")

    gradcam = GradCAM(model=model, target_layer=model.get_target_layer(), device=device)
    heatmap = gradcam.generate(input_tensor=image_tensor, class_idx=pred_idx)
    gradcam.remove_hooks()

    save_path = save_gradcam_visualization(
        image=original_image,
        heatmap=heatmap,
        save_path=output_path,
        alpha=0.45,
    )
    print(f"Grad-CAM visualization saved to: {save_path}")

    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(plt.imread(save_path))
        plt.axis("off")
        plt.title(f"Prediction: {predicted_class} ({confidence:.2%})")
        plt.show()
    except Exception as err:
        print(f"Could not display visualization window: {err}")


if __name__ == "__main__":
    main()
