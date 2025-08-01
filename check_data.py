import torch
from pathlib import Path

# 检查预处理数据的维度
processed_path = Path("data/processed/mnist/amplitude")
images = torch.load(processed_path / "train_images.pt")
labels = torch.load(processed_path / "train_labels.pt")

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Image data type: {images.dtype}")
print(f"Labels data type: {labels.dtype}")
print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
print(f"Unique labels: {torch.unique(labels)}")

# 检查一个样本
sample = images[0]
print(f"Single image shape: {sample.shape}")
print(f"Single image size: {sample.numel()}")

# 为了amplitude编码，我们需要2^n = image_size
import math
required_qubits = math.ceil(math.log2(sample.numel()))
print(f"Required qubits for amplitude encoding: {required_qubits}")
