#!/usr/bin/env python3
"""Evaluate classical CNN checkpoints produced by train_cnn.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
import yaml
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.downsampled_loader import get_downsampled_dataloader  # noqa: E402
from models.classical_cnn import ClassicalCNN  # noqa: E402
from train_optimized import get_dataloader  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def fetch_dataloader(config: dict, train: bool) -> torch.utils.data.DataLoader:
    data_cfg = config.get("data", {})
    if data_cfg.get("downsampled", False):
        return get_downsampled_dataloader(config, train=train)
    return get_dataloader(config, train=train)


def build_eval_loader(config: dict) -> DataLoader:
    base_loader = fetch_dataloader(config, train=False)
    dataset = base_loader.dataset
    batch_size = config['data']['batch_size']
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def infer_input_shape(data_cfg: dict, sample: torch.Tensor) -> Tuple[int, int, int]:
    if sample.dim() == 4:
        return sample.shape[1], sample.shape[2], sample.shape[3]
    if sample.dim() == 2:
        num_features = data_cfg.get("num_features")
        size = int(data_cfg.get("downsample_size", 0))
        if size and size * size == sample.shape[1]:
            return 1, size, size
        if num_features and int(num_features) == sample.shape[1]:
            side = int(int(num_features) ** 0.5)
            return 1, side, side
        raise ValueError("Cannot infer spatial dimensions for flattened input.")
    raise ValueError(f"Unexpected sample shape: {sample.shape}")


def build_model(config: dict, data_loader) -> ClassicalCNN:
    data_cfg = config["data"]
    training_cfg = config.get("training", {})
    sample = next(iter(data_loader))[0]
    if not isinstance(sample, torch.Tensor):
        sample = sample[0]
    in_channels, height, width = infer_input_shape(data_cfg, sample)
    conv_channels = (32, 64) if max(height, width) >= 8 else (16, 32)
    model = ClassicalCNN(
        input_size=height,
        num_classes=data_cfg["num_classes"],
        in_channels=in_channels,
        conv_channels=conv_channels,
        dropout=training_cfg.get("dropout", 0.0),
    )
    return model


def evaluate(model: torch.nn.Module, loader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            if data.dim() == 2:
                side = int((data.shape[1]) ** 0.5)
                data = data.view(data.shape[0], 1, side, side)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total if total else 0.0


def iter_targets(prefixes):
    targets = []
    prefix_paths = [Path(p) for p in prefixes]
    for cfg_path in sorted(Path("configs").rglob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        training = cfg.get("training") or {}
        ckpt_dir = training.get("checkpoint_dir")
        if not ckpt_dir:
            continue
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.exists():
            continue
        if any(str(ckpt_path).startswith(str(prefix)) for prefix in prefix_paths):
            targets.append((cfg_path, cfg, ckpt_path))
    return targets


def main(args):
    targets = iter_targets(args.prefix)
    print(f"Discovered {len(targets)} configs for CNN evaluation.")

    output_path = Path(args.output)
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except Exception:
            existing = []
    else:
        existing = []
    results = []

    for cfg_path, config, ckpt_dir in targets:
        print(f"Evaluating {cfg_path} -> {ckpt_dir}")
        loader = build_eval_loader(config)
        model = build_model(config, loader).to(torch.device("cpu"))
        for ckpt in sorted(ckpt_dir.glob("*.pt")):
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state["model_state"])
            acc = evaluate(model, loader)
            entry = {
                "config": str(cfg_path),
                "checkpoint": ckpt.as_posix(),
                "epoch": state.get("epoch"),
                "recorded_accuracy": state.get("accuracy"),
                "recorded_loss": state.get("loss"),
                "evaluated_test_accuracy": acc,
            }
            results.append(entry)
            print(f"  {ckpt.name}: {acc:.2f}%")

    existing.extend(results)
    output_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"✅ Saved {len(results)} new CNN entries; total {len(existing)} in {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN checkpoints.")
    parser.add_argument("--prefix", nargs="+", required=True, help="Checkpoint prefixes (e.g., checkpoints_cnn)")
    parser.add_argument("--output", type=str, default="AI_logs/cnn_eval_results.pt", help="Torch serialized output path")
    main(parser.parse_args())
