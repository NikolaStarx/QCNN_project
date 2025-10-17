# Quantum Convolutional Neural Network (QCNN) Project

A PyTorch-based implementation of Quantum Convolutional Neural Networks with Qiskit 1.x compatibility. This project explores quantum machine learning through hybrid classical-quantum neural networks for image classification tasks.

## 🌟 Features

- **Qiskit 1.x Compatible**: Updated to work with the latest Qiskit primitives API
- **Multiple Encoding Schemes**: Support for amplitude, angle, and hybrid encoding
- **Unified CPU/GPU Support**: Seamless compatibility across CPU and GPU backends with automatic tensor device handling
- **Flexible Architecture**: Configurable QCNN models via YAML configuration files
- **Advanced Checkpoint Management**: Configurable periodic and best model saving with customizable intervals
- **Noise Simulation**: Built-in noise models for realistic quantum simulations with configurable depolarizing error rates
- **Modular Design**: Clean separation of encoders, models, and utilities
- **Memory-Efficient Data Loading**: Unified data loading pipeline with in-memory tensor operations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/NikolaStarx/QCNN_project.git
cd QCNN_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Preprocess the data (this will automatically download MNIST):
```bash
python scripts/preprocess.py --dataset mnist --encoding amplitude
```

**Note**: Large dataset files are not included in the repository due to GitHub's size limits. The preprocessing script will automatically download the required data.

4. Train the model:
```bash
python train.py --config configs/mnist_amplitude.yaml
```

## 📁 Project Structure

```
QCNN_project/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── data/                      # Dataset directory
│   ├── raw/                   # Raw downloaded data
│   │   ├── mnist/
│   │   └── fashion_mnist/
│   └── processed/             # Preprocessed data
│       └── mnist/
│           └── amplitude/
│               ├── train_images.pt
│               └── test_images.pt
│
├── encoders/                  # Quantum encoding modules
│   ├── angle.py               # Angle encoding implementation
│   └── amplitude.py           # Amplitude encoding implementation
│
├── models/                    # Model definitions
│   └── qcnn.py                # QCNN model implementation (CPU/GPU unified)
│
├── train.py                   # Training script
│
├── utils/                     # Utility functions
│   └── data_utils.py          # Data loading and preprocessing
│
├── configs/                   # Configuration files
│   ├── mnist_amplitude.yaml   # MNIST amplitude encoding config
│   └── mnist_angle.yaml       # MNIST angle encoding config
│
├── logs/                      # Training logs
├── checkpoints/               # Model checkpoints
└── scripts/                   # Helper scripts
    └── preprocess.py          # Data preprocessing script
```

## 🔧 Configuration

The project uses YAML configuration files to define experiments. See `configs/` directory for examples.

Key configuration sections:
- `data`: Dataset and encoding parameters
  - `encoding`: Choose from `amplitude`, `angle`, or `hybrid`
  - `num_qubits`: Number of qubits in the quantum circuit
  - `num_features`: Input feature dimension (for angle/hybrid encoding)
  - `batch_size`, `num_train_samples`, `num_test_samples`: Data loading parameters
- `environment`: Backend and noise settings
  - `backend`: `CPU` or `GPU` (requires qiskit-aer-gpu)
  - `add_noise`: Enable/disable noise simulation
  - `noise.depolarizing_p1`: Single-qubit gate error rate
  - `noise.depolarizing_p2`: Two-qubit gate error rate
- `training`: Optimization and checkpoint parameters
  - `epochs`, `lr`: Training hyperparameters
  - `checkpoint_dir`: Directory to save model checkpoints
  - `checkpoint_prefix`: Prefix for checkpoint filenames
  - `save_start_epoch`: Start saving periodic checkpoints from this epoch (1-based)
  - `save_interval`: Save checkpoint every N epochs (<=0 disables periodic saving)

## 🧪 Experiments

### MNIST Classification

Run MNIST classification with amplitude encoding:
```bash
python train.py --config configs/mnist_amplitude.yaml
```

Run with angle encoding:
```bash
python train.py --config configs/mnist_angle.yaml
```

## 🏗️ Model Architecture

The QCNN implementation has been upgraded with a unified CPU/GPU compatible architecture:

### Key Improvements in `models/qcnn.py`
- **Device-Agnostic Design**: Automatic tensor device conversion (`.cpu()`) ensures compatibility across CPU and CUDA backends
- **Two Model Classes**:
  - `QCNNAmplitude`: Optimized for amplitude encoding with state preparation via `Initialize` gate
  - `QCNNGeneral`: Flexible model supporting angle, hybrid, and other parameterized encodings
- **Parameter-Shift Rule Gradient**: Custom `autograd.Function` implementation for computing quantum gradients
- **Modular Circuit Building**: Reusable `create_qcnn_ansatz()` function generates convolution and pooling layers

### Training Script Enhancements (`train.py`)
- **Unified Data Pipeline**: Single `get_dataloader()` function handles all encoding types
- **Smart Dataset Loading**: Automatic mapping between config names and torchvision dataset classes
- **Memory Optimization**: In-memory tensor operations for efficient data handling
- **Flexible Sampling**: Configurable train/test sample counts with random sampling

## 📊 Results

The model achieves competitive performance on MNIST classification tasks while demonstrating the potential of quantum machine learning approaches.

## 🛠️ Development

### Testing
```bash
python -m pytest tests/
```

### Code Style
This project follows PEP 8 style guidelines.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{qcnn_project,
  title={Quantum Convolutional Neural Networks with Qiskit},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/QCNN_project}
}
```

## 🔗 References

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- Quantum Convolutional Neural Networks (Cong et al., 2019)

## Colab Usage

For a Colab-friendly workflow, open and run the notebook:

- `notebooks/qcnn.ipynb` — includes:
  - Drive mount and project path detection
  - Dependency installation with `qiskit-aer-gpu` preference (fallback to CPU)
  - Training via a selected YAML config
  - An evaluation cell to load a checkpoint and report test accuracy

Tips:
- Set `CONFIG_PATH` in the notebook to one of the Colab configs, e.g. `configs/mnist_angle_colab.yaml`.
- Ensure `environment.backend: GPU` in the config when Aer GPU is available.

## Evaluate a Checkpoint (Script)

Use the evaluation script to run batch inference on the test set and print accuracy:

```
python scripts/evaluate.py \
  --config configs/mnist_angle_colab.yaml \
  --checkpoint checkpoints/mnist_angle/best.pt
```

Notes:
- The script mirrors `train.py` environment setup (GPU/CPU and optional noise) to match training conditions.
- Checkpoints are saved under the directory in `training.checkpoint_dir` (defaults to `checkpoints/`).
- With the Colab configs, each experiment uses its own subdirectory, e.g. `checkpoints/mnist_angle/`.

## 💾 Checkpoints

The training loop automatically saves model checkpoints:

### Checkpoint Types
- **`last.pt`**: Always saved after each epoch, contains the most recent model state
- **`best.pt`**: Saved whenever training accuracy improves, preserves the best performing model
- **Periodic checkpoints**: `{checkpoint_prefix}_epoch_{N}.pt` files saved at regular intervals

### Configuration
Control checkpoint behavior via `training` section in your config YAML:
```yaml
training:
  checkpoint_dir: "checkpoints/mnist_angle"  # Output directory
  checkpoint_prefix: "qcnn"                   # Filename prefix for periodic saves
  save_start_epoch: 5                         # Start periodic saving from epoch 5 (1-based)
  save_interval: 10                           # Save every 10 epochs (<=0 disables)
```

### Checkpoint Content
Each checkpoint file contains:
- `model_state`: Model parameters
- `optimizer_state`: Optimizer state for resuming training
- `epoch`: Training epoch number
- `accuracy`: Training accuracy at save time
- `loss`: Training loss at save time
- `config`: Full configuration used for training
