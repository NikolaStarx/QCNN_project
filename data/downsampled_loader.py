from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset


DATA_ROOT = Path("data/processed_4x4")


def _load_features(dataset_name: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_key = dataset_name.lower()
    if "fashion" in dataset_key:
        folder = "fashion_mnist"
    elif "mnist" in dataset_key:
        folder = "mnist"
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    split = "train" if train else "test"
    feature_path = DATA_ROOT / folder / f"{split}_features.pt"
    label_path = DATA_ROOT / folder / f"{split}_labels.pt"
    if not feature_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Downsampled tensors not found for {dataset_name} split={split}. "
            "Run scripts/preprocess_downsampled.py first."
        )

    features = torch.load(feature_path)
    labels = torch.load(label_path)
    return features, labels


def _normalise_amplitude(vectors: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(vectors, dim=1, keepdim=True)
    norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    return vectors / norms


def get_downsampled_dataloader(config: dict, *, train: bool) -> DataLoader:
    data_cfg = config["data"]
    encoding = data_cfg["encoding"]

    features, labels = _load_features(data_cfg["dataset"], train)

    if "label_subset" in data_cfg:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        subset_vals = torch.tensor(data_cfg["label_subset"], dtype=torch.long)
        for val in subset_vals:
            mask |= labels == val
        features = features[mask]
        labels = labels[mask]

        # Re-map labels to 0..num_classes-1 for cross-entropy
        unique_vals = subset_vals.tolist()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        labels = torch.tensor([mapping[int(l)] for l in labels], dtype=torch.long)

    if encoding == "amplitude":
        features = _normalise_amplitude(features)

    dataset = TensorDataset(features, labels)

    num_samples_key = f"num_{'train' if train else 'test'}_samples"
    if num_samples_key in data_cfg:
        num_samples = data_cfg[num_samples_key]
        if num_samples and num_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
    )
