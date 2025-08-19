# file: scripts/preprocess.py (FINAL, UNIFIED NAMING)

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

def preprocess_for_amplitude(dataset, num_qubits):
    target_dim = 2**num_qubits
    if target_dim < 28*28: raise ValueError("Not enough qubits for amplitude encoding.")
    processed_images = []
    for img, _ in tqdm(dataset, desc="Processing images"):
        img_flat = torch.flatten(img)
        img_padded = F.pad(img_flat, (0, target_dim - img_flat.shape[0]), 'constant', 0)
        norm = torch.linalg.norm(img_padded)
        img_normalized = img_padded / norm if norm > 0 else img_padded
        processed_images.append(img_normalized)
    return torch.stack(processed_images), torch.tensor(dataset.targets, dtype=torch.long).clone().detach()

def main(args):
    print(f"Starting preprocessing for dataset: '{args.dataset}' with encoding: '{args.encoding}'")
    raw_path_base = Path("data/raw")
    
    # --- THE FIX: Unify the naming convention ---
    dataset_name_clean = args.dataset.lower().replace('-', '_')
    processed_path = Path("data/processed") / dataset_name_clean / args.encoding
    # --------------------------------------------

    processed_path.mkdir(parents=True, exist_ok=True)
    print(f"Base raw data path: {raw_path_base}")
    print(f"Processed data destination: {processed_path}")

    if dataset_name_clean == 'mnist':
        DatasetClass = datasets.MNIST
    elif dataset_name_clean == 'fashion_mnist':
        DatasetClass = datasets.FashionMNIST
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    raw_train_dataset = DatasetClass(root=raw_path_base, train=True, download=True, transform=transforms.ToTensor())
    raw_test_dataset = DatasetClass(root=raw_path_base, train=False, download=True, transform=transforms.ToTensor())

    if args.encoding == "amplitude":
        print("\nProcessing training data for amplitude encoding...")
        train_images, train_labels = preprocess_for_amplitude(raw_train_dataset, args.num_qubits)
        print("\nProcessing test data for amplitude encoding...")
        test_images, test_labels = preprocess_for_amplitude(raw_test_dataset, args.num_qubits)
    else:
        raise NotImplementedError(f"Preprocessing for '{args.encoding}' is not implemented.")

    print("\nSaving processed tensors...")
    torch.save(train_images, processed_path / "train_images.pt")
    torch.save(train_labels, processed_path / "train_labels.pt")
    torch.save(test_images, processed_path / "test_images.pt")
    torch.save(test_labels, processed_path / "test_labels.pt")
    print(f"✅ Preprocessing complete. Files saved in {processed_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets for QCNN.")
    parser.add_argument("dataset", type=str, choices=['mnist', 'fashion_mnist'], help="Dataset to preprocess.")
    parser.add_argument("encoding", type=str, choices=['amplitude'], help="Target encoding type.")
    parser.add_argument("--num_qubits", type=int, default=10, help="Number of qubits for the model.")
    args = parser.parse_args()
    main(args)