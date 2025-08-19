# file: train.py (FINAL, UNIFIED, CORRECTED VERSION)

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

from models.qcnn import QCNNAmplitude, QCNNGeneral
from encoders.angle import build_angle_encoder_circuit
from encoders.hybrid import build_hybrid_encoder_circuit

def get_dataloader(config: dict, train: bool):
    data_config = config['data']
    encoding = data_config['encoding']
    dataset_name_from_config = data_config['dataset']

    # --- Step 1: Correctly map config name to folder name and Dataset class ---
    dataset_name_lower = dataset_name_from_config.lower()
    if 'fashion' in dataset_name_lower:
        folder_name, DatasetClass = 'fashion_mnist', datasets.FashionMNIST
    elif 'mnist' in dataset_name_lower:
        folder_name, DatasetClass = 'mnist', datasets.MNIST
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name_from_config}")

    # --- Step 2: Load the full dataset into memory as tensors ---
    if encoding == "amplitude":
        processed_path = Path("data/processed") / folder_name / encoding
        if not processed_path.exists(): raise FileNotFoundError(f"Processed data not found at {processed_path}.")
        all_data = torch.load(processed_path / f"{'train' if train else 'test'}_images.pt")
        all_targets = torch.load(processed_path / f"{'train' if train else 'test'}_labels.pt")
    
    elif encoding in ["angle", "hybrid"]:
        raw_path_base = Path("data/raw")
        num_features = data_config.get('num_features', data_config['num_qubits'])
        patch_size = int(np.sqrt(num_features))
        if patch_size * patch_size != num_features: raise ValueError(f"For '{encoding}', num_features must be a perfect square.")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:, :patch_size, :patch_size]),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])
        
        temp_dataset = DatasetClass(root=raw_path_base, train=train, download=True, transform=transform)
        # Unify by loading the entire torchvision dataset into memory
        loader = DataLoader(temp_dataset, batch_size=len(temp_dataset))
        all_data, all_targets = next(iter(loader))
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    # --- Step 3: UNIFIED filtering on the tensors ---
    if data_config['num_classes'] == 2:
        idx = (all_targets == 0) | (all_targets == 1)
        all_data = all_data[idx]
        all_targets = all_targets[idx]

    # --- Step 4: Create a clean TensorDataset ---
    final_dataset = TensorDataset(all_data, all_targets)

    # --- Step 5: UNIFIED sampling ---
    num_samples = data_config.get(f"num_{'train' if train else 'test'}_samples")
    if num_samples and num_samples < len(final_dataset):
        indices = torch.randperm(len(final_dataset))[:num_samples]
        final_dataset = Subset(final_dataset, indices)
        
    return DataLoader(final_dataset, batch_size=data_config['batch_size'], shuffle=True)


def main(config_path: str):
    # (The main function is unchanged and correct)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("📝 Configuration loaded:"); print(yaml.dump(config, default_flow_style=False))
    env_config, data_config, encoding = config['environment'], config['data'], config['data']['encoding']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(f"🌍 Using device: {device}")
    backend_options = {}
    if env_config['backend'] == 'GPU' and torch.cuda.is_available():
        backend_options["device"] = "GPU"; print("🚀 Configuring for qiskit-aer GPU backend.")
    else:
        print("⚙️  Configuring for qiskit-aer CPU backend.")
    if env_config.get('add_noise', False):
        print("🔥 Injecting noise into the simulation.")
        noise_model = NoiseModel()
        p1 = env_config.get('noise', {}).get('depolarizing_p1', 0.0); p2 = env_config.get('noise', {}).get('depolarizing_p2', 0.0)
        if p1 > 0: noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['ry', 'rz', 'h'])
        if p2 > 0: noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
        backend_options["noise_model"] = noise_model
    estimator = AerEstimator(backend_options=backend_options)
    print("\nLoading data..."); train_loader = get_dataloader(config, train=True); print(f"✅ Training data loaded with {len(train_loader.dataset)} samples.")
    print("\nInitializing model...")
    if encoding == 'amplitude':
        model = QCNNAmplitude(num_qubits=data_config['num_qubits'], num_classes=data_config['num_classes'], estimator=estimator)
    elif encoding in ['angle', 'hybrid']:
        encoder_fn = {'angle': build_angle_encoder_circuit, 'hybrid': build_hybrid_encoder_circuit}[encoding]
        num_input_features = data_config.get('num_features', data_config['num_qubits'])
        model = QCNNGeneral(num_qubits=data_config['num_qubits'], encoder_fn=encoder_fn, num_input_features=num_input_features, num_classes=data_config['num_classes'], estimator=estimator)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr']); loss_fn = nn.CrossEntropyLoss(); print("✅ Model, optimizer, and loss function initialized.")
    print("\n--- [ Training Started ] ---")
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss, correct_predictions, total_samples = 0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(); output = model(data); loss = loss_fn(output, target); loss.backward(); optimizer.step()
            total_loss += loss.item(); pred = output.argmax(dim=1, keepdim=True); correct_predictions += pred.eq(target.view_as(pred)).sum().item(); total_samples += len(data)
        avg_loss = total_loss / len(train_loader); accuracy = 100. * correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print("--- [ Training Finished ] ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QCNN Training Script"); parser.add_argument('--config', type=str, required=True, help="Path to YAML config."); args = parser.parse_args()
    main(args.config)