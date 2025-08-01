# file: encoders/amplitude.py (FIXED VERSION)

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.exceptions import QiskitError

def build_amplitude_encoder_circuit(qc: QuantumCircuit, params: ParameterVector):
    """
    Applies true amplitude encoding using qc.initialize.
    This encodes the entire input vector into the quantum state's amplitudes.
    
    This version explicitly disables the internal normalization of the gate,
    as the data is assumed to be pre-normalized.
    """
    num_qubits = qc.num_qubits
    required_length = 2**num_qubits

    try:
        # The params vector is a symbolic placeholder.
        # The concrete numerical data will be bound by TorchConnector during training.
        # We MUST disable normalization here because a symbolic vector cannot be normalized.
        # Our preprocessing script already handles the numerical normalization.
        
        # --- THE FIX IS HERE ---
        qc.initialize(params, qc.qubits, normalize=False)
        # -----------------------

    except QiskitError as e:
        print(f"Error during qc.initialize: {e}")
        print("Ensure input data is L2-normalized and has length "
              f"{required_length} for {num_qubits} qubits.")
        raise
        
    qc.barrier() # Add a barrier for visual separation in circuit diagrams.