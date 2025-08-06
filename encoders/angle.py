# file: encoders/angle.py

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector

def build_angle_encoder_circuit(qc: QuantumCircuit, params: ParameterVector):
    """
    Applies angle encoding to the quantum circuit.
    Each feature in the input data is encoded as a rotation on a single qubit.

    Args:
        qc (QuantumCircuit): The QuantumCircuit object to add gates to.
        params (ParameterVector): A vector of parameters representing input data pixels.
                                  Assumes values are in [0, 1].
    """
    num_qubits = qc.num_qubits
    if len(params) != num_qubits:
        raise ValueError(
            f"Angle encoding requires len(params) == num_qubits, "
            f"but got len={len(params)} and num_qubits={num_qubits}."
        )

    for i in range(num_qubits):
        # Maps input feature `params[i]` (assumed in [0, 1]) to a rotation angle in [0, pi].
        qc.ry(params[i] * np.pi, i)
    
    qc.barrier()