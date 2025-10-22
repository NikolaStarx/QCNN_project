#!/usr/bin/env python3

"""
Deep QCNN training script mirroring train_optimized.py but with configurable
convolutional depth for angle/hybrid encodings on small patches.
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from torchvision import datasets, transforms
import numpy as np

from models.qcnn_deep import QCNNDeep
from encoders.angle import build_angle_encoder_circuit
from encoders.hybrid import build_hybrid_encoder_circuit
from train_optimized import load_checkpoint_if_available


def get_dataloader(config: dict, train: bool):
    data_config = config["data"]
    encoding = data_config["encoding"]
    dataset_name_from_config = data_config["dataset"]

    dataset_name_lower = dataset_name_from_config.lower()
    if "fashion" in dataset_name_lower:
        folder_name, DatasetClass = "fashion_mnist", datasets.FashionMNIST
    elif "mnist" in dataset_name_lower:
        folder_name, DatasetClass = "mnist", datasets.MNIST
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name_from_config}")

    if encoding == "amplitude":
        processed_path = Path("data/processed") / folder_name / encoding
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data not found at {processed_path}.")
        all_data = torch.load(processed_path / f"{'train' if train else 'test'}_images.pt")
        all_targets = torch.load(processed_path / f"{'train' if train else 'test'}_labels.pt")

    elif encoding in ["angle", "hybrid"]:
        raw_path_base = Path("data/raw")
        num_features = data_config.get("num_features", data_config["num_qubits"])
        patch_size = int(np.sqrt(num_features))
        if patch_size * patch_size != num_features:
            raise ValueError(f"For '{encoding}', num_features must be a perfect square.")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:, :patch_size, :patch_size]),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])

        temp_dataset = DatasetClass(root=raw_path_base, train=train, download=True, transform=transform)
        loader = DataLoader(temp_dataset, batch_size=len(temp_dataset))
        all_data, all_targets = next(iter(loader))
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    if data_config.get("num_classes", 10) == 2:
        idx = (all_targets == 0) | (all_targets == 1)
        all_data = all_data[idx]
        all_targets = all_targets[idx]

    final_dataset = TensorDataset(all_data, all_targets)

    num_samples = data_config.get(f"num_{'train' if train else 'test'}_samples")
    if num_samples and num_samples < len(final_dataset):
        indices = torch.randperm(len(final_dataset))[:num_samples]
        final_dataset = Subset(final_dataset, indices)

    return DataLoader(final_dataset, batch_size=data_config["batch_size"], shuffle=True)


def main(config_path: str, log_interval: int = 0, resume: bool = False):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("📝 Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    env_config, data_config = config["environment"], config["data"]
    encoding = data_config["encoding"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🌍 Using device: {device}")

    backend_options = {}
    if env_config["backend"] == "GPU" and torch.cuda.is_available():
        backend_options["device"] = "GPU"
        print("🚀 Configuring for qiskit-aer GPU backend.")
    else:
        print("⚙️  Configuring for qiskit-aer CPU backend.")

    if env_config.get("add_noise", False):
        print("🔥 Injecting noise into the simulation.")
        noise_model = NoiseModel()
        noise_cfg = env_config.get("noise", {})
        p1 = noise_cfg.get("depolarizing_p1", 0.0)
        p2 = noise_cfg.get("depolarizing_p2", 0.0)
        if p1 > 0:
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "rz", "h"])
        if p2 > 0:
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        backend_options["noise_model"] = noise_model

    estimator = AerEstimator(backend_options=backend_options)

    print("\nLoading data...")
    train_loader = get_dataloader(config, train=True)
    print(f"✅ Training data loaded with {len(train_loader.dataset)} samples.")

    encoder_fn_map = {
        "angle": build_angle_encoder_circuit,
        "hybrid": build_hybrid_encoder_circuit,
    }
    encoder_fn = encoder_fn_map.get(encoding)
    num_features = None
    if encoding != "amplitude":
        num_features = data_config.get("num_features", data_config["num_qubits"])

    model_cfg = config.get("model", {})
    conv_depth = int(model_cfg.get("conv_depth", 2))

    model = QCNNDeep(
        num_qubits=data_config["num_qubits"],
        num_classes=data_config["num_classes"],
        estimator=estimator,
        encoding=encoding,
        num_features=num_features,
        encoder_fn=encoder_fn,
        conv_depth=conv_depth,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    loss_fn = nn.CrossEntropyLoss()
    print("✅ Model, optimizer, and loss function initialized.")

    training_cfg = dict(config.get("training", {}))
    save_start_epoch = int(training_cfg.get("save_start_epoch", 1))
    save_interval = int(training_cfg.get("save_interval", 0))
    ckpt_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints_deep"))
    ckpt_prefix = training_cfg.get("checkpoint_prefix", "qcnn_deep")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        training_cfg["resume_from_last"] = True

    best_acc = float("-inf")
    start_epoch, best_acc = load_checkpoint_if_available(model, optimizer, ckpt_dir, device, training_cfg, best_acc)
    total_epochs = int(config["training"]["epochs"])

    print("\n--- [ Training Started ] ---")
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(device), target.to(device)
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
                avg_loss = total_loss / batch_idx
                running_acc = 100.0 * correct_predictions / total_samples if total_samples else 0.0
                print(f"Epoch {epoch + 1}/{total_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss {avg_loss:.4f} | Acc {running_acc:.2f}%")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_samples if total_samples else 0.0
        print(f"Epoch [{epoch + 1}/{total_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep QCNN Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--log-interval", type=int, default=0, help="Iterations between logging metrics.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if available.")
    args = parser.parse_args()
    main(args.config, log_interval=args.log_interval, resume=args.resume)
