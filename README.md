# Quantum Convolutional Neural Network (QCNN) Project

A PyTorch-based implementation of Quantum Convolutional Neural Networks with Qiskit 1.x compatibility. This project explores quantum machine learning through hybrid classical-quantum neural networks for image classification tasks.

## 🌟 Features

- **Qiskit 1.x Compatible**: Updated to work with the latest Qiskit primitives API
- **Multiple Encoding Schemes**: Support for amplitude and angle encoding
- **Flexible Architecture**: Configurable QCNN models via YAML configuration files
- **GPU Acceleration**: Support for both CPU and GPU backends
- **Noise Simulation**: Built-in noise models for realistic quantum simulations
- **Modular Design**: Clean separation of encoders, models, and utilities

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
│   └── qcnn.py                # QCNN model implementation
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
- `environment`: Backend and noise settings
- `training`: Optimization parameters

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

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

## Checkpoints

The training loop saves:
- `last.pt` — last epoch state
- `best.pt` — best training accuracy in this example
- Periodic files — controlled by:
  - `training.save_start_epoch` (1-based start)
  - `training.save_interval` (epochs between saves; <=0 disables)
