# file: train.py (FINAL, QISKIT 1.x COMPATIBLE VERSION)

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

# +++ THE FIX: Import the correct V2 Estimator from qiskit_aer +++
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from torchvision import datasets, transforms
from models.qcnn import QCNNAmplitude

def get_dataloader(config: dict, train: bool):
    """
    Loads pre-processed data for amplitude encoding.
    """
    data_config = config['data']
    encoding = data_config['encoding']
    
    if encoding == "amplitude":
        processed_path = Path("data/processed") / data_config['dataset'] / encoding
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data not found at {processed_path}. Please run preprocess.py first.")
        
        image_file = processed_path / f"{'train' if train else 'test'}_images.pt"
        label_file = processed_path / f"{'train' if train else 'test'}_labels.pt"
        
        images = torch.load(image_file)
        labels = torch.load(label_file)
        
        full_dataset = TensorDataset(images, labels)
    else:
        raise ValueError(f"This train.py is currently configured for amplitude encoding only.")

    num_samples = data_config.get(f"num_{'train' if train else 'test'}_samples")
    targets = full_dataset.tensors[1]

    if data_config['num_classes'] == 2:
        idx = (targets == 0) | (targets == 1)
        full_dataset = TensorDataset(full_dataset.tensors[0][idx], full_dataset.tensors[1][idx])

    if num_samples and num_samples < len(full_dataset):
        indices = torch.randperm(len(full_dataset))[:num_samples]
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset
        
    return DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=True)


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("📝 Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    env_config = config['environment']
    data_config = config['data']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🌍 Using device: {device}")

    # --- THE FIX: Use the new API to configure the AerEstimator ---
    backend_options = {}
    
    if env_config['backend'] == 'GPU' and torch.cuda.is_available():
        backend_options["device"] = "GPU"
        print("🚀 Configuring for qiskit-aer GPU backend.")
    else:
        print("⚙️  Configuring for qiskit-aer CPU backend.")

    if env_config.get('add_noise', False):
        print("🔥 Injecting noise into the simulation.")
        noise_model = NoiseModel()
        p1 = env_config.get('noise', {}).get('depolarizing_p1', 0.0)
        p2 = env_config.get('noise', {}).get('depolarizing_p2', 0.0)
        
        if p1 > 0:
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['ry', 'rz', 'h'])
        if p2 > 0:
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
        
        backend_options["noise_model"] = noise_model

    # Instantiate the correct estimator with the options dictionary
    estimator = AerEstimator(backend_options=backend_options)
    # -----------------------------------------------------------------

    print("\nLoading data...")
    train_loader = get_dataloader(config, train=True)
    print(f"✅ Training data loaded with {len(train_loader.dataset)} samples.")

    print("\nInitializing model...")
    model = QCNNAmplitude(
        num_qubits=data_config['num_qubits'],
        num_classes=data_config['num_classes'],
        estimator=estimator
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    loss_fn = nn.CrossEntropyLoss()
    print("✅ Model, optimizer, and loss function initialized.")

    print("\n--- [ Training Started ] ---")
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss, correct_predictions, total_samples = 0, 0, 0
        for data, target in train_loader:
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
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print("--- [ Training Finished ] ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QCNN Training Script")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()
    main(args.config)