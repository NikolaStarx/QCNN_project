#!/usr/bin/env python3
"""
Report pairwise L2 distances between class means on the 4x4 downsampled datasets.

Usage:
    python scripts/analyse_downsampled_l2.py --dataset mnist
    python scripts/analyse_downsampled_l2.py --dataset fashion --top-k 5 --bottom-k 5
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Sequence

import torch


DATA_ROOT = Path("data/processed_4x4")

DATASETS = {
    "mnist": {
        "folder": "mnist",
        "labels": [str(i) for i in range(10)],
    },
    "fashion": {
        "folder": "fashion_mnist",
        "labels": [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    },
}


def load_split(dataset_key: str, train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    entry = DATASETS[dataset_key]
    split = "train" if train else "test"
    feat_path = DATA_ROOT / entry["folder"] / f"{split}_features.pt"
    label_path = DATA_ROOT / entry["folder"] / f"{split}_labels.pt"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found. Run scripts/preprocess_downsampled.py first."
        )
    features = torch.load(feat_path)
    labels = torch.load(label_path)
    return features, labels


def compute_pairwise_l2(features: torch.Tensor, labels: torch.Tensor) -> list[tuple[float, int, int]]:
    unique_labels = sorted({int(l) for l in labels.tolist()})
    class_means = {}
    for label in unique_labels:
        mask = labels == label
        class_means[label] = features[mask].mean(dim=0)

    pairs = []
    for a, b in itertools.combinations(unique_labels, 2):
        dist = torch.dist(class_means[a], class_means[b]).item()
        pairs.append((dist, a, b))
    pairs.sort(reverse=True)
    return pairs


def render_pairs(pairs: Sequence[tuple[float, int, int]], label_names: Sequence[str], *, top_k: int, bottom_k: int) -> None:
    print(f"\nTop {top_k} pairwise distances:")
    for dist, a, b in pairs[:top_k]:
        print(f"{a} ({label_names[a]}) vs {b} ({label_names[b]}): {dist:.3f}")

    if bottom_k > 0:
        print(f"\nBottom {bottom_k} pairwise distances:")
        for dist, a, b in pairs[-bottom_k:]:
            print(f"{a} ({label_names[a]}) vs {b} ({label_names[b]}): {dist:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse 4x4 downsampled dataset class separability.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
        help="Dataset to analyse (mnist or fashion).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of largest distances to report.")
    parser.add_argument("--bottom-k", type=int, default=10, help="Number of smallest distances to report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, labels = load_split(args.dataset)
    pairs = compute_pairwise_l2(features, labels)
    render_pairs(pairs, DATASETS[args.dataset]["labels"], top_k=args.top_k, bottom_k=args.bottom_k)


if __name__ == "__main__":
    main()
