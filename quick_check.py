#!/usr/bin/env python3
# quick_check.py - 快速环境验证

import sys
print("Python version:", sys.version)

try:
    import qiskit_machine_learning
    print("✅ qiskit-machine-learning available")
    
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.connectors import TorchConnector
    print("✅ Key components imported successfully")
    
    import torch
    import yaml
    print("✅ All dependencies ready!")
    print("\n🚀 Ready to run QCNN training!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
