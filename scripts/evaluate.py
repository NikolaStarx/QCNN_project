"""
Evaluate a trained QCNN checkpoint on the test split and report accuracy.

Usage:
  python scripts/evaluate.py --config configs/mnist_angle_colab.yaml \
                             --checkpoint checkpoints/mnist_angle/best.pt

This script mirrors train.py's environment setup (GPU/CPU and optional noise)
to ensure evaluation matches training conditions.
"""

import argparse
from pathlib import Path
import yaml
import torch

from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from models.qcnn import QCNNAmplitude, QCNNGeneral
from encoders.angle import build_angle_encoder_circuit
from encoders.hybrid import build_hybrid_encoder_circuit
from train import get_dataloader


def build_estimator(env_cfg: dict) -> AerEstimator:
    backend_options = {}
    if env_cfg.get('backend') == 'GPU' and torch.cuda.is_available():
        backend_options["device"] = "GPU"
    if env_cfg.get('add_noise', False):
        nm = NoiseModel()
        p1 = env_cfg.get('noise', {}).get('depolarizing_p1', 0.0)
        p2 = env_cfg.get('noise', {}).get('depolarizing_p2', 0.0)
        if p1 > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['ry', 'rz', 'h'])
        if p2 > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
        backend_options["noise_model"] = nm
    return AerEstimator(backend_options=backend_options)


def build_model(config: dict, estimator: AerEstimator, device: torch.device):
    data_cfg = config['data']
    encoding = data_cfg['encoding']
    if encoding == 'amplitude':
        model = QCNNAmplitude(
            num_qubits=data_cfg['num_qubits'],
            num_classes=data_cfg['num_classes'],
            estimator=estimator,
        )
    else:
        encoder_fn = {
            'angle': build_angle_encoder_circuit,
            'hybrid': build_hybrid_encoder_circuit,
        }[encoding]
        num_input_features = data_cfg.get('num_features', data_cfg['num_qubits'])
        model = QCNNGeneral(
            num_qubits=data_cfg['num_qubits'],
            encoder_fn=encoder_fn,
            num_input_features=num_input_features,
            num_classes=data_cfg['num_classes'],
            estimator=estimator,
        )
    return model.to(device)


def evaluate(config_path: Path, ckpt_path: Path) -> float:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    estimator = build_estimator(config['environment'])
    model = build_model(config, estimator, device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    test_loader = get_dataloader(config, train=False)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / max(1, total)
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc


def main():
    ap = argparse.ArgumentParser(description="Evaluate QCNN checkpoint on test set")
    ap.add_argument('--config', required=True, help='Path to YAML config used for training/eval')
    ap.add_argument('--checkpoint', required=True, help='Path to checkpoint .pt file (best.pt/last.pt/etc.)')
    args = ap.parse_args()

    evaluate(Path(args.config), Path(args.checkpoint))


if __name__ == '__main__':
    main()

