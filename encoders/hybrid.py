# file: encoders/hybrid.py (FINAL, CORRECTED LOGIC)

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector

def build_hybrid_encoder_circuit(qc: QuantumCircuit, params: ParameterVector):
    """
    Applies a layered hybrid encoding scheme.
    For N features and M qubits, it applies N/M features per layer across all qubits.
    
    Example: 16 features, 4 qubits.
    - Divides 4 qubits into 2 pairs (tiles): (0,1) and (2,3).
    - Requires 4 features per tile. Total 8 features per layer.
    - Applies 2 layers to use all 16 features.
    """
    num_qubits = qc.num_qubits
    num_params = len(params)

    if num_qubits % 2 != 0:
        raise ValueError("Hybrid encoder requires an even number of qubits.")

    num_tiles = num_qubits // 2
    features_per_tile = 4
    features_per_layer = num_tiles * features_per_tile

    if num_params % features_per_layer != 0:
        raise ValueError(f"Number of features ({num_params}) must be a multiple of "
                         f"features per layer ({features_per_layer}).")
    
    num_layers = num_params // features_per_layer
    
    param_idx = 0
    for _ in range(num_layers):
        # Apply encoding to each tile
        for tile_idx in range(num_tiles):
            q_idx1 = tile_idx * 2
            q_idx2 = tile_idx * 2 + 1
            
            # Phase encoding
            qc.rz(params[param_idx + 0] * 2 * np.pi, q_idx1)
            qc.rz(params[param_idx + 1] * 2 * np.pi, q_idx2)
            
            # Angle encoding
            qc.ry(params[param_idx + 2] * np.pi, q_idx1)
            qc.ry(params[param_idx + 3] * np.pi, q_idx2)
            
            param_idx += features_per_tile

        # Entangle between tiles after each layer
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        if num_qubits > 1: # Circular entanglement
             qc.cx(num_qubits - 1, 0)
        
    qc.barrier()