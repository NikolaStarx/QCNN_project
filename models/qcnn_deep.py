import math
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import SparsePauliOp


def _pairwise(iterable: Sequence[int]) -> Sequence[tuple[int, int]]:
    return [(iterable[i], iterable[i + 1]) for i in range(0, len(iterable) - 1, 2)]


def create_qcnn_deep_ansatz(num_qubits: int, conv_depth: int = 2) -> tuple[QuantumCircuit, ParameterVector, int]:
    """Create a deeper QCNN ansatz with configurable convolutional depth."""
    if conv_depth < 1:
        raise ValueError("conv_depth must be >= 1.")

    ansatz = QuantumCircuit(num_qubits, name="QCNN Deep Ansatz")
    active_qubits = list(range(num_qubits))

    # Each depth adds 5 parameters per qubit pair (4 single-qubit + 1 entangling rotation).
    total_params = 0
    layer_specs: list[tuple[Sequence[tuple[int, int]], int]] = []
    tmp_active = active_qubits.copy()
    while len(tmp_active) > 1:
        pairs = _pairwise(tmp_active)
        params_per_layer = conv_depth * len(pairs) * 5
        layer_specs.append((pairs, params_per_layer))
        total_params += params_per_layer
        tmp_active = [pair[1] for pair in pairs]

    params = ParameterVector("θ", total_params)
    param_idx = 0
    current_active = active_qubits

    for pairs, params_per_layer in layer_specs:
        ansatz.barrier()
        for _ in range(conv_depth):
            for q1, q2 in pairs:
                ansatz.ry(params[param_idx], q1)
                param_idx += 1
                ansatz.rz(params[param_idx], q1)
                param_idx += 1
                ansatz.ry(params[param_idx], q2)
                param_idx += 1
                ansatz.rz(params[param_idx], q2)
                param_idx += 1
                ansatz.cx(q1, q2)
                ansatz.rzz(params[param_idx], q1, q2)
                param_idx += 1
        ansatz.barrier()

        # Pooling
        new_active = []
        for q1, q2 in pairs:
            ansatz.cx(q1, q2)
            new_active.append(q2)
        current_active = new_active

    final_qubit = current_active[0]
    return ansatz, params, final_qubit


class QuantumFunctionDeep(Function):
    @staticmethod
    def forward(ctx, input_data, weights, qcnn_ansatz, final_qubit_idx, estimator, encoding, encoder_circuit=None):
        ctx.qcnn_ansatz = qcnn_ansatz
        ctx.final_qubit_idx = final_qubit_idx
        ctx.input_data = input_data
        ctx.estimator = estimator
        ctx.encoding = encoding
        ctx.encoder_circuit = encoder_circuit

        weights_np = weights.detach().cpu().numpy()
        input_np = input_data.detach().cpu().numpy()
        circuits = []

        if encoding == "amplitude":
            for sample in input_np:
                qc = QuantumCircuit(qcnn_ansatz.num_qubits)
                qc.append(Initialize(sample, normalize=True), qc.qubits)
                circuits.append(qc.compose(qcnn_ansatz))
        else:
            if encoder_circuit is None:
                raise ValueError("encoder_circuit must be provided for non-amplitude encodings.")
            for sample in input_np:
                bound = encoder_circuit.assign_parameters(sample)
                circuits.append(bound.compose(qcnn_ansatz))

        observable = SparsePauliOp("I" * (qcnn_ansatz.num_qubits - 1 - final_qubit_idx) + "Z" + "I" * final_qubit_idx)
        job = estimator.run(circuits, [observable] * len(circuits), parameter_values=[weights_np] * len(circuits))
        ctx.save_for_backward(weights)

        values = job.result().values
        return torch.tensor(values, dtype=torch.float32, device=input_data.device).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output):
        (weights,) = ctx.saved_tensors
        weights_np = weights.detach().cpu().numpy()
        input_np = ctx.input_data.detach().cpu().numpy()
        grad_output_np = grad_output.squeeze().cpu().numpy()
        shift = np.pi / 2

        if ctx.encoding == "amplitude":
            base_circuits = []
            for sample in input_np:
                qc = QuantumCircuit(ctx.qcnn_ansatz.num_qubits)
                qc.append(Initialize(sample, normalize=True), qc.qubits)
                base_circuits.append(qc.compose(ctx.qcnn_ansatz))
        else:
            base_circuits = []
            for sample in input_np:
                bound = ctx.encoder_circuit.assign_parameters(sample)
                base_circuits.append(bound.compose(ctx.qcnn_ansatz))

        observable = SparsePauliOp("I" * (ctx.qcnn_ansatz.num_qubits - 1 - ctx.final_qubit_idx) + "Z" + "I" * ctx.final_qubit_idx)
        num_params = len(weights_np)
        batch_size = len(base_circuits)

        all_circuits = []
        all_params = []
        for param_idx in range(num_params):
            plus = weights_np.copy()
            minus = weights_np.copy()
            plus[param_idx] += shift
            minus[param_idx] -= shift
            for circuit in base_circuits:
                all_circuits.extend([circuit, circuit])
                all_params.extend([plus, minus])

        job = ctx.estimator.run(all_circuits, [observable] * len(all_circuits), parameter_values=all_params)
        results = np.array(job.result().values).reshape(num_params, batch_size, 2)
        shifted = 0.5 * (results[:, :, 0] - results[:, :, 1])
        grad_weights_np = np.sum(shifted * grad_output_np[None, :], axis=1)
        grad_weights = torch.tensor(grad_weights_np, dtype=weights.dtype, device=weights.device)
        return None, grad_weights, None, None, None, None, None


class QCNNDeep(nn.Module):
    def __init__(self, num_qubits, num_classes, estimator, encoding, encoder_fn=None, num_features=None, conv_depth: int = 2):
        super().__init__()
        if estimator is None:
            raise ValueError("Estimator must be provided.")

        self.estimator = estimator
        self.encoding = encoding

        self.qcnn_ansatz, _, self.final_qubit_idx = create_qcnn_deep_ansatz(num_qubits, conv_depth=conv_depth)
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
        expectation = QuantumFunctionDeep.apply(
            x,
            self.q_weights,
            self.qcnn_ansatz,
            self.final_qubit_idx,
            self.estimator,
            self.encoding,
            self.encoder_circuit,
        )
        return self.classical_head(expectation)
