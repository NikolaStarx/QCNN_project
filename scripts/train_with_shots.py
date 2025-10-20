#!/usr/bin/env python3
"""
Profiling-friendly QCNN training entry point.

This script mirrors the behaviour of `train_optimized.py` while exposing
extra instrumentation knobs (notably the ability to override the Aer
estimator's shot count and to emit periodic batch timing logs). It keeps
the quantum circuit architecture intact, preserving the physical meaning
of the hybrid QCNN experiments.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import Estimator as AerEstimator

from encoders.angle import build_angle_encoder_circuit
from encoders.hybrid import build_hybrid_encoder_circuit
from models.qcnn_optimized import QCNNOptimized
from train_optimized import get_dataloader, load_checkpoint_if_available


def build_estimator(env_config: dict, shots: Optional[int]) -> AerEstimator:
    """Construct an Aer estimator with optional noise and shot override."""
    backend_options: dict[str, object] = {}
    backend = env_config.get("backend", "").upper()

    if backend == "GPU" and torch.cuda.is_available():
        backend_options["device"] = "GPU"
        print("🚀 Using qiskit-aer GPU backend.")
    else:
        if backend == "GPU":
            print("⚠️  GPU backend requested but CUDA unavailable; falling back to CPU.")
        backend_options["device"] = "CPU"

    if env_config.get("add_noise", False):
        noise_cfg = env_config.get("noise", {})
        noise_model = NoiseModel()
        p1 = float(noise_cfg.get("depolarizing_p1", 0.0))
        p2 = float(noise_cfg.get("depolarizing_p2", 0.0))
        if p1 > 0:
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "rz", "h"])
        if p2 > 0:
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        backend_options["noise_model"] = noise_model
        print(f"🔥 Injecting depolarizing noise (p1={p1}, p2={p2}).")

    if shots is not None:
        backend_options["shots"] = int(shots)
        print(f"🎯 Overriding estimator shots: {shots}")

    return AerEstimator(backend_options=backend_options)


def build_encoder_fn(encoding: str):
    encoder_map = {
        "angle": build_angle_encoder_circuit,
        "hybrid": build_hybrid_encoder_circuit,
    }
    encoder_fn = encoder_map.get(encoding)
    if encoder_fn is None and encoding != "amplitude":
        raise ValueError(f"Unsupported encoding '{encoding}' for profiling runner.")
    return encoder_fn


def train(config_path: str, *, resume: bool, shots: Optional[int], log_interval: Optional[int]):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_config = config["environment"]
    data_config = config["data"]
    training_cfg = dict(config.get("training", {}))  # shallow copy

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🌍 Using device: {device}")

    estimator = build_estimator(env_config, shots)
    print("\nLoading data...")
    train_loader = get_dataloader(config, train=True)
    print(f"✅ Training data loaded with {len(train_loader.dataset)} samples.")

    encoder_fn = build_encoder_fn(data_config["encoding"])
    num_features = None
    if data_config["encoding"] != "amplitude":
        num_features = data_config.get("num_features", data_config["num_qubits"])

    model = QCNNOptimized(
        num_qubits=data_config["num_qubits"],
        num_classes=data_config["num_classes"],
        estimator=estimator,
        encoding=data_config["encoding"],
        num_features=num_features,
        encoder_fn=encoder_fn,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    print("✅ Model, optimizer, and loss function initialized.")

    save_start_epoch = int(training_cfg.get("save_start_epoch", 1))
    save_interval = int(training_cfg.get("save_interval", 0))
    ckpt_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_prefix = training_cfg.get("checkpoint_prefix", "qcnn")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        training_cfg["resume_from_last"] = True

    best_acc = float("-inf")
    start_epoch, best_acc = load_checkpoint_if_available(model, optimizer, ckpt_dir, device, training_cfg, best_acc)
    total_epochs = int(training_cfg["epochs"])

    print("\n--- [ Training Started ] ---")
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            batch_start = time.time()

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)

            if log_interval and batch_idx % log_interval == 0:
                batch_time = time.time() - batch_start
                avg_batch_time = (time.time() - epoch_start) / batch_idx
                print(
                    f"[Epoch {epoch + 1}] Batch {batch_idx}/{len(train_loader)} "
                    f"batch_time={batch_time:.2f}s avg_batch_time={avg_batch_time:.2f}s"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{total_epochs}] - Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}% | epoch_time={epoch_time/60:.2f} min"
        )

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "accuracy": accuracy,
            "loss": avg_loss,
            "config": config,
        }

        torch.save(state, ckpt_dir / "last.pt")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(state, ckpt_dir / "best.pt")

        if save_interval > 0 and (epoch + 1) >= save_start_epoch:
            if ((epoch + 1 - save_start_epoch) % save_interval) == 0:
                torch.save(state, ckpt_dir / f"{ckpt_prefix}_epoch_{epoch + 1}.pt")

    print("--- [ Training Finished ] ---")


def parse_args():
    parser = argparse.ArgumentParser(description="QCNN training with configurable shots for profiling.")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument("--shots", type=int, default=None, help="Override estimator shots (leave unset to use Aer default).")
    parser.add_argument("--log-interval", type=int, default=None, help="Print batch timing every N batches.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if available.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, resume=args.resume, shots=args.shots, log_interval=args.log_interval)
