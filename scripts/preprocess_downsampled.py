#!/usr/bin/env python3
"""
Generate downsampled (4x4) versions of MNIST and FashionMNIST suitable for
lightweight QCNN experiments. The output tensors keep the full image
information in 16 features per sample.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import AvgPool2d
from torchvision import datasets, transforms


DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
}

OUTPUT_ROOTS = {
    4: Path("data/processed_4x4"),
    8: Path("data/processed_8x8"),
}


def downsample_split(dataset_key: str, train: bool) -> None:
    DatasetCls = DATASETS[dataset_key]
    split_name = "train" if train else "test"

    transform = transforms.Compose([transforms.ToTensor()])
    ds = DatasetCls(root="data/raw", train=train, download=True, transform=transform)

    pool4 = AvgPool2d(kernel_size=7, stride=7)  # 28 -> 4

    features = {size: [] for size in OUTPUT_ROOTS}
    labels = []

    for img, label in ds:
        tensor = img.unsqueeze(0)  # (1,1,28,28)

        down4 = pool4(tensor).squeeze(0)
        features[4].append(down4.flatten())

        down8 = F.interpolate(tensor.float(), size=(8, 8), mode="bilinear", align_corners=False).squeeze(0)
        features[8].append(down8.flatten())

        labels.append(int(label))

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    for size, feats in features.items():
        features_tensor = torch.stack(feats, dim=0).to(torch.float32)
        out_dir = OUTPUT_ROOTS[size] / dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(features_tensor, out_dir / f"{split_name}_features.pt")
        torch.save(labels_tensor, out_dir / f"{split_name}_labels.pt")


def main() -> None:
    for key in DATASETS:
        for train in (True, False):
            downsample_split(key, train)


if __name__ == "__main__":
    main()
