#!/usr/bin/env python3
"""
Generate downsampled (4x4) versions of MNIST and FashionMNIST suitable for
lightweight QCNN experiments. The output tensors keep the full image
information in 16 features per sample.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import AvgPool2d
from torchvision import datasets, transforms


DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
}

OUTPUT_ROOT = Path("data/processed_4x4")


def downsample_split(dataset_key: str, train: bool) -> None:
    DatasetCls = DATASETS[dataset_key]
    split_name = "train" if train else "test"

    transform = transforms.Compose([transforms.ToTensor()])
    ds = DatasetCls(root="data/raw", train=train, download=True, transform=transform)

    pool = AvgPool2d(kernel_size=7, stride=7)  # 28 -> 4
    features = []
    labels = []

    for img, label in ds:
        pooled = pool(img.unsqueeze(0)).squeeze(0)  # (1,1,4,4) -> (4,4)
        features.append(pooled.flatten())
        labels.append(int(label))

    features_tensor = torch.stack(features).to(torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    out_dir = OUTPUT_ROOT / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features_tensor, out_dir / f"{split_name}_features.pt")
    torch.save(labels_tensor, out_dir / f"{split_name}_labels.pt")


def main() -> None:
    for key in DATASETS:
        for train in (True, False):
            downsample_split(key, train)


if __name__ == "__main__":
    main()
