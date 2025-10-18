import torch
import torch.nn as nn
from torch.autograd import Function
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import Initialize
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def conv_layer(qc: QuantumCircuit, params, active_qubits):
    """Two-qubit convolution applied pairwise across the active register."""
    for i in range(0, len(active_qubits) - 1, 2):
        q1, q2 = active_qubits[i], active_qubits[i + 1]
        qc.ry(params[0], q1)
        qc.ry(params[1], q2)
        qc.cx(q1, q2)


def pooling_layer(qc: QuantumCircuit, active_qubits):
    """Entangling pooling layer that halves the number of active qubits."""
    new_active = []
    for i in range(0, len(active_qubits) - 1, 2):
        control, target = active_qubits[i], active_qubits[i + 1]
        qc.cx(control, target)
        new_active.append(target)
    return new_active


def create_qcnn_ansatz(num_qubits: int):
    """Construct the canonical QCNN ansatz used across the project."""
    num_layers = num_qubits.bit_length() - 1
    num_weights = 2 * num_layers
    weights_params = ParameterVector("θ", num_weights)
    ansatz = QuantumCircuit(num_qubits, name="QCNN Ansatz")
    active_qubits = list(range(num_qubits))
    weight_idx = 0

    while len(active_qubits) > 1:
        layer_weights = [weights_params[i] for i in range(weight_idx, weight_idx + 2)]
        conv_layer(ansatz, layer_weights, active_qubits)
        ansatz.barrier()
        active_qubits = pooling_layer(ansatz, active_qubits)
        ansatz.barrier()
        weight_idx += 2

    return ansatz, weights_params, active_qubits[0]


class QuantumFunctionOptimized(Function):
    """Custom autograd Function implementing batched parameter-shift for the QCNN."""

    @staticmethod
    def forward(ctx, input_data, weights, qcnn_ansatz, final_qubit_idx, estimator, encoding, encoder_circuit=None):
        device = input_data.device
        batch_size = input_data.shape[0]

        weights_np = weights.detach().cpu().numpy()
        input_data_np = input_data.detach().cpu().numpy()

        num_qubits = qcnn_ansatz.num_qubits
        # Measure Z on the surviving qubit, pad with I elsewhere.
        pauli_string = "I" * (num_qubits - 1 - final_qubit_idx) + "Z" + "I" * final_qubit_idx
        observable = SparsePauliOp(pauli_string)

        base_circuits = []
        if encoding == "amplitude":
            for sample in input_data_np:
                init_gate = Initialize(sample, normalize=True)
                qc = QuantumCircuit(num_qubits)
                qc.append(init_gate, qc.qubits)
                qc.compose(qcnn_ansatz, inplace=True)
                base_circuits.append(qc)
        else:
            if encoder_circuit is None:
                raise ValueError("encoder_circuit must be provided for non-amplitude encodings.")
            for sample in input_data_np:
                bound_encoder = encoder_circuit.assign_parameters(sample)
                base_circuits.append(bound_encoder.compose(qcnn_ansatz))

        parameter_values = [weights_np] * batch_size
        job = estimator.run(base_circuits, [observable] * batch_size, parameter_values=parameter_values)
        expectations = job.result().values

        ctx.save_for_backward(weights)
        ctx.estimator = estimator
        ctx.base_circuits = base_circuits
        ctx.observable = observable
        ctx.final_qubit_idx = final_qubit_idx
        ctx.qcnn_ansatz = qcnn_ansatz

        return torch.tensor(expectations, dtype=torch.float32, device=device).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        estimator = ctx.estimator
        base_circuits = ctx.base_circuits
        observable = ctx.observable

        batch_size = len(base_circuits)
        weights_np = weights.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy().reshape(batch_size)

        shift = np.pi / 2
        num_params = len(weights_np)

        all_circuits = []
        all_parameter_values = []
        for param_idx in range(num_params):
            weights_plus = weights_np.copy()
            weights_minus = weights_np.copy()
            weights_plus[param_idx] += shift
            weights_minus[param_idx] -= shift

            for circuit in base_circuits:
                all_circuits.append(circuit)
                all_parameter_values.append(weights_plus)
                all_circuits.append(circuit)
                all_parameter_values.append(weights_minus)

        observables = [observable] * len(all_circuits)
        job = estimator.run(all_circuits, observables, parameter_values=all_parameter_values)
        results = np.array(job.result().values, dtype=np.float64).reshape(num_params, batch_size, 2)

        # Parameter-shift rule: f(θ+) - f(θ-), rescaled and weighted by downstream gradients.
        shifted_diffs = 0.5 * (results[:, :, 0] - results[:, :, 1])
        grad_weights_np = np.sum(shifted_diffs * grad_output_np[None, :], axis=1)
        grad_weights = torch.tensor(grad_weights_np, dtype=weights.dtype, device=weights.device)

        return None, grad_weights, None, None, None, None, None


class QCNNOptimized(nn.Module):
    """Optimized QCNN model sharing the canonical circuit while batching estimator calls."""

    def __init__(self, num_qubits, num_classes, estimator: Estimator, encoding, encoder_fn=None, num_features=None):
        super().__init__()
        if estimator is None:
            raise ValueError("An Estimator instance is required for QCNNOptimized.")

        self.estimator = estimator
        self.encoding = encoding
        self.qcnn_ansatz, _, self.final_qubit_idx = create_qcnn_ansatz(num_qubits)
        self.q_weights = nn.Parameter(torch.randn(self.qcnn_ansatz.num_parameters))

        self.encoder_circuit = None
        if encoding != "amplitude":
            if encoder_fn is None or num_features is None:
                raise ValueError("encoder_fn and num_features must be provided for non-amplitude encodings.")
            self.input_params = ParameterVector("x", num_features)
            self.encoder_circuit = QuantumCircuit(num_qubits, name="Encoder")
            encoder_fn(self.encoder_circuit, self.input_params)

        self.classical_head = nn.Linear(1, num_classes)

    def forward(self, x):
        quantum_expectation = QuantumFunctionOptimized.apply(
            x,
            self.q_weights,
            self.qcnn_ansatz,
            self.final_qubit_idx,
            self.estimator,
            self.encoding,
            self.encoder_circuit,
        )
        return self.classical_head(quantum_expectation)
