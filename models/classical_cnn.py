import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=1)


def _make_sequence(value: Iterable[int] | int, length: int) -> Sequence[int]:
    if isinstance(value, Iterable) and not isinstance(value, (int, float)):
        return list(value)
    return [int(value)] * length


class ClassicalCNN(nn.Module):
    """Compact CNN baseline mirroring QCNN dataset setups."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        in_channels: int = 1,
        conv_channels: Sequence[int] | int = (16, 32),
        kernel_size: Sequence[int] | int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")

        conv_channels = _make_sequence(conv_channels, 2 if not isinstance(conv_channels, Iterable) else len(conv_channels))  # type: ignore[arg-type]
        kernel_size = _make_sequence(kernel_size, len(conv_channels))

        layers: list[nn.Module] = []
        current_channels = in_channels
        current_size = input_size

        for idx, (out_channels, k) in enumerate(zip(conv_channels, kernel_size)):
            padding = k // 2  # keep size for odd kernels
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=k, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            if current_size >= 4:
                layers.append(nn.MaxPool2d(kernel_size=2))
                current_size = math.floor(current_size / 2)

            current_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(Flatten())

        feature_dim = current_channels
        classifier: list[nn.Module] = []
        if dropout > 0.0:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(feature_dim, num_classes))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
