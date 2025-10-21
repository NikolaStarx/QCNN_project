#!/usr/bin/env python3
"""
Classical CNN baseline training script mirroring QCNN configs.

This script reuses the downsampled data pipeline and checkpointing scheme from
the quantum training flows while replacing the quantum circuit with a small CNN.
It supports the same configuration files used by noise experiments (e.g. angle
configs with downsample_size 4 or 8) and introduces optional Gaussian noise
injection to emulate depolarising noise levels.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from data.downsampled_loader import get_downsampled_dataloader
from models.classical_cnn import ClassicalCNN
from train_optimized import get_dataloader, load_checkpoint_if_available


def fetch_dataloader(config: dict, train: bool):
    data_cfg = config.get("data", {})
    if data_cfg.get("downsampled", False):
        return get_downsampled_dataloader(config, train=train)
    return get_dataloader(config, train=train)


def resolve_input_shape(data_cfg: dict, sample: torch.Tensor) -> tuple[int, int]:
    """Infer spatial dimensions for reshaping 1D feature vectors."""
    if sample.dim() == 4:
        return sample.shape[-2], sample.shape[-1]

    if sample.dim() == 2:
        if "downsample_size" in data_cfg:
            size = int(data_cfg["downsample_size"])
            return size, size
        num_features = data_cfg.get("num_features")
        if num_features:
            size = int(math.isqrt(num_features))
            if size * size == num_features:
                return size, size
        raise ValueError("Unable to infer spatial dimensions for the provided data configuration.")

    raise ValueError(f"Unexpected sample shape: {sample.shape}")


def compute_noise_std(env_cfg: dict) -> float:
    if not env_cfg.get("add_noise", False):
        return 0.0
    noise_cfg = env_cfg.get("noise", {})
    p1 = float(noise_cfg.get("depolarizing_p1", 0.0))
    p2 = float(noise_cfg.get("depolarizing_p2", 0.0))
    base = max(p1, p2)
    return base * 5  # heuristic scaling to produce noticeable Gaussian noise


def maybe_add_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        return tensor
    noise = torch.randn_like(tensor) * std
    return tensor + noise


def train(config_path: str, args):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_cfg = config["environment"]
    data_cfg = config["data"]
    training_cfg = dict(config.get("training", {}))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🌍 Using device: {device}")

    train_loader = fetch_dataloader(config, train=True)
    val_loader: Optional[torch.utils.data.DataLoader] = None
    if args.eval:
        val_loader = fetch_dataloader(config, train=False)
        print(f"✅ Validation data loaded with {len(val_loader.dataset)} samples.")

    print(f"✅ Training data loaded with {len(train_loader.dataset)} samples.")

    example_batch = next(iter(train_loader))[0]
    height, width = resolve_input_shape(data_cfg, example_batch)
    in_channels = example_batch.shape[1] if example_batch.dim() == 4 else 1

    model = ClassicalCNN(
        input_size=height,
        num_classes=data_cfg["num_classes"],
        in_channels=in_channels,
        conv_channels=(32, 64) if max(height, width) >= 8 else (16, 32),
        dropout=training_cfg.get("dropout", 0.0),
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    print("✅ Model, optimizer, and loss function initialized.")

    ckpt_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints_cnn"))
    ckpt_prefix = training_cfg.get("checkpoint_prefix", "cnn_baseline")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        training_cfg["resume_from_last"] = True

    best_acc = float("-inf")
    start_epoch, best_acc = load_checkpoint_if_available(model, optimizer, ckpt_dir, device, training_cfg, best_acc)
    total_epochs = int(training_cfg["epochs"])

    noise_std = compute_noise_std(env_cfg)
    if noise_std > 0:
        print(f"🌫️  Injecting Gaussian noise with std={noise_std:.4f} per batch.")

    def reshape_batch(batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() == 4:
            return batch
        return batch.view(batch.shape[0], in_channels, height, width)

    print("\n--- [ Training Started ] ---")
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            data = reshape_batch(data).to(device)
            target = target.to(device)
            data = maybe_add_noise(data, noise_std)

            optimizer.zero_grad()
            logits = model(data)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

            if args.log_interval and batch_idx % args.log_interval == 0:
                avg_loss = total_loss / batch_idx
                acc = 100.0 * correct / total
                print(f"Epoch {epoch + 1}/{total_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss {avg_loss:.4f} | Acc {acc:.2f}%")

        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        print(f"Epoch [{epoch + 1}/{total_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")

        eval_acc = train_acc
        if val_loader is not None:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for data, target in val_loader:
                    data = reshape_batch(data).to(device)
                    target = target.to(device)
                    logits = model(data)
                    pred = logits.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                    val_total += data.size(0)
            eval_acc = 100.0 * val_correct / max(1, val_total)
            print(f"  🔎 Validation Accuracy: {eval_acc:.2f}%")

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "accuracy": eval_acc,
            "loss": avg_loss,
            "config": config,
        }

        torch.save(state, ckpt_dir / "last.pt")
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(state, ckpt_dir / "best.pt")

        save_interval = int(training_cfg.get("save_interval", 0))
        save_start_epoch = int(training_cfg.get("save_start_epoch", 1))
        if save_interval > 0 and (epoch + 1) >= save_start_epoch:
            if ((epoch + 1 - save_start_epoch) % save_interval) == 0:
                torch.save(state, ckpt_dir / f"{ckpt_prefix}_epoch_{epoch + 1}.pt")

    print("--- [ Training Finished ] ---")


def parse_args():
    parser = argparse.ArgumentParser(description="Classical CNN baseline training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available.")
    parser.add_argument("--log-interval", type=int, default=0, help="Iterations between logging metrics.")
    parser.add_argument("--eval", action="store_true", help="Evaluate on the test split after each epoch.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args)
