# file: scripts/preprocess.py

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

def preprocess_for_amplitude(dataset, num_qubits):
    """
    Transforms a dataset for amplitude encoding.
    """
    target_dim = 2**num_qubits
    if target_dim < 28*28:
        raise ValueError("Not enough qubits for amplitude encoding.")

    processed_images = []
    # 使用 tqdm 显示进度条
    for img, _ in tqdm(dataset, desc="Processing images"):
        # 1. 展平
        img_flat = torch.flatten(img)
        # 2. 填充
        img_padded = F.pad(img_flat, (0, target_dim - img_flat.shape[0]), 'constant', 0)
        # 3. L2 归一化
        norm = torch.linalg.norm(img_padded)
        img_normalized = img_padded / norm if norm > 0 else img_padded
        processed_images.append(img_normalized)
    
    # 将列表堆叠成一个大的张量
    # return torch.stack(processed_images), torch.tensor(dataset.targets)
    # file: scripts/preprocess.py (line 31)
    return torch.stack(processed_images), torch.tensor(dataset.targets, dtype=torch.long).clone().detach()

def main(args):
    """
    Main preprocessing function.
    """
    print(f"Starting preprocessing for dataset: '{args.dataset}' with encoding: '{args.encoding}'")
    
    # 定义输入和输出路径
    raw_path = Path("data/raw") / args.dataset
    processed_path = Path("data/processed") / args.dataset / args.encoding
    processed_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Raw data source: {raw_path}")
    print(f"Processed data destination: {processed_path}")

    # 加载原始数据集
    # 注意：这里的 transform 只是简单的 ToTensor()
    raw_train_dataset = datasets.MNIST(root=raw_path.parent, train=True, download=False, transform=transforms.ToTensor())
    raw_test_dataset = datasets.MNIST(root=raw_path.parent, train=False, download=False, transform=transforms.ToTensor())

    if args.encoding == "amplitude":
        print("\nProcessing training data for amplitude encoding...")
        train_images, train_labels = preprocess_for_amplitude(raw_train_dataset, args.num_qubits)
        
        print("\nProcessing test data for amplitude encoding...")
        test_images, test_labels = preprocess_for_amplitude(raw_test_dataset, args.num_qubits)
    else:
        raise NotImplementedError(f"Preprocessing for '{args.encoding}' is not implemented.")

    # 保存处理后的张量
    print("\nSaving processed tensors...")
    torch.save(train_images, processed_path / "train_images.pt")
    torch.save(train_labels, processed_path / "train_labels.pt")
    torch.save(test_images, processed_path / "test_images.pt")
    torch.save(test_labels, processed_path / "test_labels.pt")
    
    print(f"✅ Preprocessing complete. Files saved in {processed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets for QCNN.")
    parser.add_argument("dataset", type=str, choices=['mnist', 'fashion_mnist'], help="Dataset to preprocess.")
    parser.add_argument("encoding", type=str, choices=['amplitude', 'angle'], help="Target encoding type.")
    parser.add_argument("--num_qubits", type=int, default=10, help="Number of qubits for the model (determines padding for amplitude encoding).")
    
    args = parser.parse_args()
    main(args)