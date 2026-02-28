from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM implementation for image classification models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    image_np = np.array(image).astype(np.float32) / 255.0

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        image.size, resample=Image.BILINEAR
    )
    heatmap_resized = np.array(heatmap_img).astype(np.float32) / 255.0

    colored_heatmap = cm.get_cmap("jet")(heatmap_resized)[:, :, :3]
    overlay = (1 - alpha) * image_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    return Image.fromarray((overlay * 255).astype(np.uint8))


def save_gradcam_visualization(
    image: Image.Image,
    heatmap: np.ndarray,
    save_path: Path,
    alpha: float = 0.4,
) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap_on_image(image=image, heatmap=heatmap, alpha=alpha)
    overlay.save(save_path)
    return save_path
