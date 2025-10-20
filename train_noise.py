#!/usr/bin/env python3
"""
Accelerated QCNN training loop for noisy simulations.

This script keeps the original QCNN architecture and data processing pipeline
intact while introducing estimator shot scheduling and profiling hooks that
reduce wall-clock time in noisy Aer simulations. It is compatible with all
existing configs used by `train_optimized.py` and can be invoked as:

    python train_noise.py --config path/to/config.yaml

Key differences vs. train_optimized.py
--------------------------------------
1. Shot scheduling: you can specify a lower `--min-shots` for early epochs and
   gradually ramp to a higher `--max-shots` (default 1024, matching Aer defaults)
   to retain physically meaningful accuracy at later stages.
2. Optional periodic evaluation with high-shot estimator for trustworthy metrics.
3. Batch-level timing logs to empirically validate the speedups.

All enhancements stay within the spirit of the QCNN approach—no circuit changes,
no approximation shortcuts beyond tuneable shot counts.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import yaml
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import Estimator as AerEstimator

from encoders.angle import build_angle_encoder_circuit
from encoders.hybrid import build_hybrid_encoder_circuit
from models.qcnn_optimized import QCNNOptimized
from train_optimized import get_dataloader, load_checkpoint_if_available


def build_noise_model(env_config: dict) -> Optional[NoiseModel]:
    if not env_config.get("add_noise", False):
        return None

    noise_cfg = env_config.get("noise", {})
    noise_model = NoiseModel()
    p1 = float(noise_cfg.get("depolarizing_p1", 0.0))
    p2 = float(noise_cfg.get("depolarizing_p2", 0.0))
    if p1 > 0:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "rz", "h"])
    if p2 > 0:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return noise_model


def build_estimator(env_config: dict, *, shots: Optional[int], noise_model: Optional[NoiseModel]) -> AerEstimator:
    backend_options: dict[str, object] = {}
    backend = env_config.get("backend", "CPU").upper()

    if backend == "GPU" and torch.cuda.is_available():
        backend_options["device"] = "GPU"
        print("🚀 Using qiskit-aer GPU backend.")
    else:
        if backend == "GPU":
            print("⚠️  GPU backend requested but CUDA unavailable; falling back to CPU.")
        backend_options["device"] = "CPU"

    if noise_model is not None:
        backend_options["noise_model"] = noise_model
        print("🔥 Noise model attached to estimator.")

    if shots is not None:
        backend_options["shots"] = int(shots)
        print(f"🎯 Estimator shots set to {shots}.")

    return AerEstimator(backend_options=backend_options)


def resolve_encoder_fn(encoding: str):
    encoder_map = {
        "angle": build_angle_encoder_circuit,
        "hybrid": build_hybrid_encoder_circuit,
    }
    fn = encoder_map.get(encoding)
    if fn is None and encoding != "amplitude":
        raise ValueError(f"Unsupported encoding '{encoding}'.")
    return fn


def linear_schedule(epoch_idx: int, total_epochs: int, min_shots: int, max_shots: int) -> int:
    if total_epochs <= 1 or max_shots <= min_shots:
        return min_shots
    alpha = epoch_idx / (total_epochs - 1)
    return int(round(min_shots + alpha * (max_shots - min_shots)))


def exponential_schedule(epoch_idx: int, total_epochs: int, min_shots: int, max_shots: int, gamma: float) -> int:
    if total_epochs <= 1 or max_shots <= min_shots:
        return min_shots
    gamma = max(1.0, gamma)
    alpha = (gamma ** epoch_idx - 1) / (gamma ** (total_epochs - 1) - 1)
    return int(round(min_shots + alpha * (max_shots - min_shots)))


def constant_schedule(_: int, __: int, min_shots: int, ___: int) -> int:
    return min_shots


def compute_shots_for_epoch(
    schedule: str,
    epoch_idx: int,
    total_epochs: int,
    min_shots: int,
    max_shots: int,
    gamma: float,
) -> int:
    if schedule == "constant":
        return constant_schedule(epoch_idx, total_epochs, min_shots, max_shots)
    if schedule == "linear":
        return linear_schedule(epoch_idx, total_epochs, min_shots, max_shots)
    if schedule == "exponential":
        return exponential_schedule(epoch_idx, total_epochs, min_shots, max_shots, gamma)
    raise ValueError(f"Unknown schedule '{schedule}'.")


@contextlib.contextmanager
def timer() -> Iterable[None]:
    start = time.time()
    yield
    end = time.time()
    print(f"⏱️  elapsed {end - start:.2f}s")


def evaluate_model(
    model: QCNNOptimized,
    data_loader,
    device: torch.device,
    env_config: dict,
    noise_model: Optional[NoiseModel],
    shots: int,
    max_batches: Optional[int] = None,
) -> tuple[float, float]:
    """Optional evaluation in inference mode with a high-shot estimator."""
    estimator = build_estimator(env_config, shots=shots, noise_model=noise_model)
    model.estimator = estimator
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader, start=1):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            total_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)

            if max_batches and batch_idx >= max_batches:
                break

    avg_loss = total_loss / max(1, min(len(data_loader), max_batches or len(data_loader)))
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy


def train(config_path: str, args):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_config = config["environment"]
    data_config = config["data"]
    training_cfg = dict(config.get("training", {}))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🌍 Using device: {device}")

    noise_model = build_noise_model(env_config)
    if noise_model is not None:
        print("🧪 Depolarizing noise parameters respected for physical fidelity.")

    train_loader = get_dataloader(config, train=True)
    print(f"✅ Training data loaded with {len(train_loader.dataset)} samples.")

    encoder_fn = resolve_encoder_fn(data_config["encoding"])
    num_features = None
    if data_config["encoding"] != "amplitude":
        num_features = data_config.get("num_features", data_config["num_qubits"])

    # Initial estimator uses min shots.
    current_shots = args.min_shots
    estimator = build_estimator(env_config, shots=current_shots, noise_model=noise_model)

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

    if args.resume:
        training_cfg["resume_from_last"] = True

    best_acc = float("-inf")
    start_epoch, best_acc = load_checkpoint_if_available(model, optimizer, ckpt_dir, device, training_cfg, best_acc)
    total_epochs = int(training_cfg["epochs"])

    print("\n--- [ Training Started ] ---")
    for epoch_idx in range(start_epoch, total_epochs):
        epoch_start = time.time()

        current_shots = compute_shots_for_epoch(
            args.schedule,
            epoch_idx,
            total_epochs,
            args.min_shots,
            args.max_shots,
            args.schedule_gamma,
        )
        print(f"\n🎬 Epoch {epoch_idx + 1}/{total_epochs} | training shots = {current_shots}")
        model.estimator = build_estimator(env_config, shots=current_shots, noise_model=noise_model)

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

            if args.log_interval and batch_idx % args.log_interval == 0:
                batch_time = time.time() - batch_start
                avg_batch_time = (time.time() - epoch_start) / batch_idx
                print(
                    f"[Epoch {epoch_idx + 1}] Batch {batch_idx}/{len(train_loader)} "
                    f"batch_time={batch_time:.2f}s avg_batch_time={avg_batch_time:.2f}s"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch_idx + 1}/{total_epochs}] - Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}% | epoch_time={epoch_time/60:.2f} min"
        )

        if args.eval_max_shots and args.max_shots > current_shots:
            eval_batches = args.eval_max_batches if args.eval_max_batches > 0 else None
            print(f"🔍 Evaluating with {args.max_shots} shots on {eval_batches or 'all'} batches.")
            eval_loss, eval_acc = evaluate_model(
                model,
                train_loader,
                device,
                env_config,
                noise_model,
                shots=args.max_shots,
                max_batches=eval_batches,
            )
            print(
                f"    Eval (shots={args.max_shots}) - Loss: {eval_loss:.4f}, "
                f"Accuracy: {eval_acc:.2f}%"
            )

        state = {
            "epoch": epoch_idx + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "accuracy": accuracy,
            "loss": avg_loss,
            "config": config,
            "training_shots": current_shots,
        }

        torch.save(state, ckpt_dir / "last.pt")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(state, ckpt_dir / "best.pt")

        if save_interval > 0 and (epoch_idx + 1) >= save_start_epoch:
            if ((epoch_idx + 1 - save_start_epoch) % save_interval) == 0:
                torch.save(state, ckpt_dir / f"{ckpt_prefix}_epoch_{epoch_idx + 1}.pt")

    print("--- [ Training Finished ] ---")


def parse_args():
    parser = argparse.ArgumentParser(description="Accelerated QCNN training for noisy simulations.")
    parser.add_argument("--config", required=True, help="Path to YAML config compatible with train_optimized.py.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if available.")
    parser.add_argument("--log-interval", type=int, default=None, help="Print batch timing every N batches.")

    parser.add_argument("--min-shots", type=int, default=256, help="Minimum shots used at the start of training.")
    parser.add_argument("--max-shots", type=int, default=1024, help="Maximum shots used by the end of training.")
    parser.add_argument(
        "--schedule",
        choices=["constant", "linear", "exponential"],
        default="linear",
        help="Shot scheduling strategy across epochs.",
    )
    parser.add_argument("--schedule-gamma", type=float, default=2.0, help="Growth factor for exponential schedule.")

    parser.add_argument(
        "--eval-max-shots",
        action="store_true",
        help="If set, evaluate each epoch with max shots to log high-fidelity metrics.",
    )
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=0,
        help="Limit number of batches during max-shot evaluation (0 = use all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_args = parse_args()
    train(train_args.config, train_args)
