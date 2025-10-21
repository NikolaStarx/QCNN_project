# file: encoders/angle.py

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector

def build_angle_encoder_circuit(qc: QuantumCircuit, params: ParameterVector, *, scale: float = 1.0):
    """
    Applies angle encoding to the quantum circuit.
    Each feature in the input data is encoded as a rotation on a single qubit.

    Args:
        qc (QuantumCircuit): The QuantumCircuit object to add gates to.
        params (ParameterVector): A vector of parameters representing input data pixels.
                                  Assumes values are in [0, 1].
    """
    num_qubits = qc.num_qubits
    num_params = len(params)

    if num_params % num_qubits != 0:
        raise ValueError(
            f"Angle encoding requires len(params) to be a multiple of num_qubits, "
            f"but got len={num_params} and num_qubits={num_qubits}."
        )
    scale = float(scale)

    for offset in range(0, num_params, num_qubits):
        for i in range(num_qubits):
            qc.ry(params[offset + i] * np.pi * scale, i)
        qc.barrier()
