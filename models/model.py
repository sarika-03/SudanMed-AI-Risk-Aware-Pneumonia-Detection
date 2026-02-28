from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


ModelType = Literal["custom_cnn", "mobilenet_v2"]


class CustomCNN(nn.Module):
    """Lightweight CNN for binary pneumonia classification."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class PneumoniaModel(nn.Module):
    """Wrapper model supporting custom CNN and MobileNetV2 transfer learning."""

    def __init__(
        self,
        model_type: ModelType = "custom_cnn",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.model_type = model_type

        if model_type == "custom_cnn":
            self.model = CustomCNN(num_classes=num_classes, dropout=dropout)
        elif model_type == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            nn.init.xavier_uniform_(backbone.classifier[1].weight)
            nn.init.zeros_(backbone.classifier[1].bias)
            self.model = backbone
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_target_layer(self) -> nn.Module:
        """Returns a sensible target layer for Grad-CAM."""
        if self.model_type == "custom_cnn":
            return self.model.features[-4]
        if self.model_type == "mobilenet_v2":
            return self.model.features[-1]
        raise ValueError(f"Unsupported model_type: {self.model_type}")
