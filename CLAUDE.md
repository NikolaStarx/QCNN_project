# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Quantum Convolutional Neural Network (QCNN) implementation using PyTorch and Qiskit 1.x for quantum machine learning on image classification tasks. The project features a hybrid classical-quantum architecture with custom gradient computation via parameter-shift rule.

## Common Commands

### Data Preprocessing
```bash
# Preprocess MNIST for amplitude encoding (required before training with amplitude)
python scripts/preprocess.py mnist amplitude --num_qubits 10

# Preprocess Fashion-MNIST
python scripts/preprocess.py fashion_mnist amplitude --num_qubits 10
```

### Training
```bash
# Train with a specific config
python train.py --config configs/mnist_amplitude.yaml

# Other encoding types (no preprocessing needed for angle/hybrid)
python train.py --config configs/mnist_angle.yaml
python train.py --config configs/mnist_hybrid.yaml
```

### Evaluation
```bash
# Evaluate a trained checkpoint
python scripts/evaluate.py --config configs/mnist_angle.yaml --checkpoint checkpoints/mnist_angle/best.pt
```

### Testing
```bash
# Run tests (if available)
python -m pytest tests/
```

## Architecture Overview

### Core Components

**models/qcnn.py** - The main QCNN model with custom quantum gradient computation:
- `QuantumFunction`: Custom `autograd.Function` implementing parameter-shift rule for quantum circuit gradients
- `QCNN`: Main model class that combines quantum circuit with classical output layer
- `create_qcnn_ansatz()`: Builds the quantum circuit with convolution and pooling layers
- Key implementation detail: Uses `Initialize` gate for amplitude encoding and SparsePauliOp for measurements
- **Device handling**: All quantum operations use `.cpu()` conversion to ensure CPU/GPU compatibility with Qiskit-Aer

**train.py** - Unified training script:
- `get_dataloader()`: Single function handles all three encoding types (amplitude, angle, hybrid)
- Automatic dataset detection and loading (MNIST vs Fashion-MNIST)
- **Critical naming convention**: Always uses lowercase folder names (`mnist`, `fashion_mnist`) for data paths
- Supports CPU/GPU backends, noise simulation, and configurable checkpointing

**encoders/** - Quantum encoding schemes:
- `amplitude.py`: True amplitude encoding using `qc.initialize()` with `normalize=False` (data is pre-normalized)
- `angle.py`: Angle encoding - maps each pixel to a rotation angle on a single qubit
- `hybrid.py`: Layered encoding combining phase (RZ) and angle (RY) rotations with entanglement

### Data Flow

1. **Amplitude encoding**: Raw data → `scripts/preprocess.py` → normalized tensors → `data/processed/{dataset}/{encoding}/` → training
2. **Angle/Hybrid encoding**: Raw data → on-the-fly transforms in `get_dataloader()` → training (no preprocessing needed)

### Key Architectural Patterns

**Quantum-Classical Hybrid**:
- Quantum circuit computes expectation value of Z-operator on final qubit
- Output passes through classical linear layer for multi-class classification
- Custom backward pass computes gradients via parameter-shift rule (shifts each parameter by ±π/2)

**Device Management**:
- PyTorch tensors can be on CPU or CUDA
- Qiskit circuits always run on CPU (or GPU via qiskit-aer-gpu)
- All tensor-to-numpy conversions use `.cpu()` to prevent device mismatch

**Noise Simulation**:
- Configurable depolarizing error rates for single-qubit and two-qubit gates
- Applied via Qiskit's `NoiseModel` in the Aer Estimator

## Configuration System

All experiments use YAML configs in `configs/`:
- `*_colab.yaml`: Optimized for Google Colab with GPU support and checkpoint management
- Standard configs: For local CPU/GPU development
- Naming convention: `{dataset}_{encoding}[_variant].yaml`

Key config sections:
- `data.encoding`: `amplitude`, `angle`, or `hybrid`
- `data.num_qubits`: Circuit size (must be ≥10 for amplitude encoding of 28×28 images)
- `data.num_features`: Input dimension for angle/hybrid (must be perfect square for angle, multiple of features_per_layer for hybrid)
- `environment.backend`: `CPU` or `GPU`
- `environment.add_noise`: Enable noise simulation
- `training.checkpoint_dir`: Where to save checkpoints (creates `last.pt`, `best.pt`, and periodic saves)

## Important Implementation Details

### Critical Fixes in Current Codebase

1. **Dataset path naming** (train.py:24-28): Always converts dataset names to lowercase to match torchvision's download structure
2. **Amplitude encoding normalization** (encoders/amplitude.py:24): Uses `normalize=False` because preprocessing already L2-normalizes data
3. **Gradient computation** (models/qcnn.py:64-94): Backward pass must reconstruct circuits for each parameter shift - no circuit reuse

### Checkpoint System

Three types automatically saved during training:
- `last.pt`: After every epoch
- `best.pt`: When training accuracy improves
- `{prefix}_epoch_{N}.pt`: Periodic saves based on `save_interval` config

Each checkpoint contains: `model_state`, `optimizer_state`, `epoch`, `accuracy`, `loss`, `config`

### Common Pitfalls

- **Memory**: Amplitude encoding requires 2^num_qubits dimensional vectors (10 qubits = 1024 dimensions)
- **Preprocessing requirement**: Amplitude encoding MUST run `scripts/preprocess.py` first; angle/hybrid do not
- **Feature dimensions**: Angle encoding needs `num_features` = perfect square; hybrid needs multiple of (num_qubits/2 * 4)
- **Device compatibility**: Never pass CUDA tensors directly to Qiskit - always use `.cpu().numpy()`

## Dependencies

Core: `qiskit`, `qiskit-machine-learning`, `torch`, `torchvision`, `numpy`, `pyyaml`, `kagglehub`, `tqdm`

Optional: `qiskit-aer-gpu` for GPU-accelerated quantum simulation (requires CUDA)
