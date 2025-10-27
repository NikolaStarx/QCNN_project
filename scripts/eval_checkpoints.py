#!/usr/bin/env python3
"""
Batch evaluation tool for QCNN checkpoints.

Given one or more checkpoint directory prefixes, this script:
  1. Finds all YAML configs whose checkpoint_dir matches those prefixes.
  2. Builds the appropriate QCNN model (deep vs optimized) according to the config.
  3. Runs inference on the configured test split to obtain fresh accuracies.
  4. Stores the consolidated metrics as JSON for later reporting.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
from pathlib import Path
from typing import Any, Iterable

import sys

import torch
import yaml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import Estimator as AerEstimator
from torch.utils.data import DataLoader

from data.downsampled_loader import get_downsampled_dataloader
from encoders.angle import build_angle_encoder_circuit
from encoders.hybrid import build_hybrid_encoder_circuit
from models.qcnn_deep import QCNNDeep
from models.qcnn_optimized import QCNNOptimized
from train_optimized import get_dataloader

def fetch_dataloader(config: dict, *, train: bool) -> DataLoader:
    data_cfg = config.get("data", {})
    if data_cfg.get("downsampled", False):
        return get_downsampled_dataloader(config, train=train)
    return get_dataloader(config, train=train)

def build_noise_model(env_config: dict) -> NoiseModel | None:
    if not env_config.get("add_noise", False):
        return None
    noise_model = NoiseModel()
    noise_cfg = env_config.get("noise", {})
    p1 = float(noise_cfg.get("depolarizing_p1", 0.0))
    p2 = float(noise_cfg.get("depolarizing_p2", 0.0))
    if p1 > 0:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "rz", "h"])
    if p2 > 0:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return noise_model

def build_estimator(env_config: dict, noise_model: NoiseModel | None) -> AerEstimator:
    backend_options: dict[str, Any] = {}
    backend_options["device"] = "GPU" if env_config.get("backend", "CPU").upper() == "GPU" and torch.cuda.is_available() else "CPU"
    if noise_model is not None:
        backend_options["noise_model"] = noise_model
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

def is_deep_config(config: dict) -> bool:
    model_cfg = config.get("model") or {}
    return model_cfg.get("conv_depth") is not None

def build_model(config: dict) -> torch.nn.Module:
    data_cfg = config["data"]
    env_cfg = copy.deepcopy(config["environment"])
    env_cfg["backend"] = "CPU"

    encoding = data_cfg["encoding"]
    encoder_fn = resolve_encoder_fn(encoding)
    num_features = None
    if encoding != "amplitude":
        num_features = data_cfg.get("num_features", data_cfg["num_qubits"])
        if encoding == "angle":
            scale = float(data_cfg.get("angle_scale", 1.0))
            encoder_fn = functools.partial(encoder_fn, scale=scale)
        elif encoding == "hybrid":
            scale = float(data_cfg.get("hybrid_scale", 1.0))
            encoder_fn = functools.partial(encoder_fn, scale=scale)

    noise_model = build_noise_model(env_cfg)
    estimator = build_estimator(env_cfg, noise_model)

    if is_deep_config(config):
        conv_depth = int(config.get("model", {}).get("conv_depth", 2))
        return QCNNDeep(
            num_qubits=data_cfg["num_qubits"],
            num_classes=data_cfg["num_classes"],
            estimator=estimator,
            encoding=encoding,
            num_features=num_features,
            encoder_fn=encoder_fn,
            conv_depth=conv_depth,
        )

    return QCNNOptimized(
        num_qubits=data_cfg["num_qubits"],
        num_classes=data_cfg["num_classes"],
        estimator=estimator,
        encoding=encoding,
        num_features=num_features,
        encoder_fn=encoder_fn,
    )

def evaluate(model: torch.nn.Module, loader: DataLoader, max_samples: int | None) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            if max_samples and total >= max_samples:
                break
            if max_samples and len(data) > max_samples - total:
                keep = max_samples - total
                if keep <= 0:
                    break
                data = data[:keep]
                target = target[:keep]
            output = model(data)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total if total else 0.0

def iter_targets(prefixes: Iterable[str]) -> list[tuple[Path, dict, Path]]:
    targets: list[tuple[Path, dict, Path]] = []
    prefix_paths = tuple(Path(p) for p in prefixes)
    for cfg_path in sorted(Path("configs").rglob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        training_cfg = cfg.get("training") or {}
        ckpt_dir = training_cfg.get("checkpoint_dir")
        if not ckpt_dir:
            continue
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.exists():
            continue
        if any(str(ckpt_path).startswith(str(prefix)) for prefix in prefix_paths):
            targets.append((cfg_path, cfg, ckpt_path))
    return targets

def main(args: argparse.Namespace) -> None:
    targets = iter_targets(args.prefix)
    print(f"Discovered {len(targets)} configs matching prefixes: {args.prefix}")

    output_path = Path(args.output)
    resource_cache: dict[str, tuple] = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except Exception:
            existing = []
    else:
        existing = []
    done_checkpoints = {entry.get("checkpoint") for entry in existing if isinstance(entry, dict)}

    results: list[dict[str, Any]] = []
    new_count = 0
    for idx, (cfg_path, config, ckpt_dir) in enumerate(targets, 1):
        print(f"[{idx}/{len(targets)}] {cfg_path} -> {ckpt_dir}")
        for ckpt in sorted(ckpt_dir.glob("*.pt")):
            if str(ckpt) in done_checkpoints:
                continue
            try:
                state = torch.load(ckpt, map_location="cpu")
            except Exception as exc:
                print(f"    !! failed loading {ckpt.name}: {exc}")
                continue
            ckpt_config = state.get("config") or config
            key = json.dumps(ckpt_config, sort_keys=True)
            if key not in resource_cache:
                try:
                    loader = fetch_dataloader(ckpt_config, train=False)
                except Exception as exc:
                    print(f"  !! dataloader error: {exc}")
                    continue
                try:
                    model = build_model(ckpt_config)
                except Exception as exc:
                    print(f"  !! model build error: {exc}")
                    continue
                resource_cache[key] = (loader, model)
            else:
                loader, model = resource_cache[key]
            try:
                model.load_state_dict(state["model_state"])
                eval_acc = evaluate(model, loader, args.max_samples if args.max_samples else None)
                entry = {
                    "config": str(cfg_path),
                    "checkpoint": str(ckpt),
                    "epoch": state.get("epoch"),
                    "recorded_accuracy": state.get("accuracy"),
                    "recorded_loss": state.get("loss"),
                    "evaluated_test_accuracy": eval_acc,
                }
                results.append(entry)
                done_checkpoints.add(str(ckpt))
                new_count += 1
                print(f"    {ckpt.name}: {eval_acc:.2f}%")
                if args.max_checkpoints and new_count >= args.max_checkpoints:
                    break
            except Exception as exc:
                print(f"    !! failed on {ckpt.name}: {exc}")
        if args.max_checkpoints and new_count >= args.max_checkpoints:
            break

    if results:
        existing.extend(results)
        output_path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
    print(f"✅ Saved {len(results)} new entries; total {len(existing)} in {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QCNN checkpoints.")
    parser.add_argument(
        "--prefix",
        nargs="+",
        required=True,
        help="Checkpoint directory prefixes to include (e.g., checkpoints_noise26).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="AI_logs/checkpoint_eval_results.json",
        help="Output JSON path for accuracy records.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=0,
        help="Limit the number of new checkpoints evaluated per run (0 = unlimited).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap the number of test samples per evaluation (0 = use full set).",
    )
    main(parser.parse_args())
