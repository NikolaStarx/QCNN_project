# file: models/qcnn_optimized.py
# Optimized version with batched gradient computation

import torch
import torch.nn as nn
from torch.autograd import Function
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from qiskit.primitives import Estimator
import numpy as np
from qiskit.quantum_info import SparsePauliOp

# ------------------- QCNN Circuit Building Logic (Shared) -------------------
def conv_layer(qc: QuantumCircuit, params, active_qubits):
    for i in range(0, len(active_qubits) - 1, 2):
        q1, q2 = active_qubits[i], active_qubits[i+1]
        qc.ry(params[0], q1)
        qc.ry(params[1], q2)
        qc.cx(q1, q2)

def pooling_layer(qc: QuantumCircuit, active_qubits):
    new_active_qubits = []
    for i in range(0, len(active_qubits) - 1, 2):
        q_control, q_target = active_qubits[i], active_qubits[i+1]
        qc.cx(q_control, q_target)
        new_active_qubits.append(q_target)
    return new_active_qubits

def create_qcnn_ansatz(num_qubits: int):
    from qiskit.circuit import ParameterVector
    num_layers = num_qubits.bit_length() - 1
    num_weights = 2 * num_layers
    weights_params = ParameterVector('θ', num_weights)
    ansatz = QuantumCircuit(num_qubits, name='QCNN Ansatz')
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

# =================================================================================
# Optimized QuantumFunction with batched gradient computation
# =================================================================================
class QuantumFunctionOptimized(Function):
    @staticmethod
    def forward(ctx, input_data: torch.Tensor, weights: torch.Tensor, qcnn_ansatz, final_qubit_idx, estimator):
        ctx.qcnn_ansatz = qcnn_ansatz
        ctx.final_qubit_idx = final_qubit_idx
        ctx.input_data = input_data
        ctx.estimator = estimator

        weights_np = weights.detach().cpu().numpy()
        input_data_np = input_data.detach().cpu().numpy()

        circuits = []
        for x_np in input_data_np:
            init_gate = Initialize(x_np, normalize=True)
            qc = QuantumCircuit(qcnn_ansatz.num_qubits)
            qc.append(init_gate, qc.qubits)
            full_circuit = qc.compose(qcnn_ansatz)
            circuits.append(full_circuit)

        num_qubits = qcnn_ansatz.num_qubits
        pauli_string = "I" * (num_qubits - 1 - final_qubit_idx) + "Z" + "I" * final_qubit_idx
        observable = SparsePauliOp(pauli_string)

        job = estimator.run(circuits, [observable] * len(circuits), [weights_np] * len(circuits))
        result_np = job.result().values

        ctx.save_for_backward(weights)

        result_tensor = torch.tensor(result_np, dtype=torch.float32, device=input_data.device)
        return result_tensor.unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        qcnn_ansatz = ctx.qcnn_ansatz
        input_data = ctx.input_data
        estimator = ctx.estimator

        weights_np = weights.detach().cpu().numpy()
        input_data_np = input_data.detach().cpu().numpy()
        grad_output_np = grad_output.squeeze().cpu().numpy()

        shift = np.pi / 2
        num_params = len(weights)
        batch_size = len(input_data)

        # Build all circuits for all parameter shifts at once
        all_circuits = []
        all_observables = []
        all_weights = []

        num_qubits = qcnn_ansatz.num_qubits
        pauli_string = "I" * (num_qubits - 1 - ctx.final_qubit_idx) + "Z" + "I" * ctx.final_qubit_idx
        observable = SparsePauliOp(pauli_string)

        for i in range(num_params):
            weights_plus = weights_np.copy()
            weights_minus = weights_np.copy()
            weights_plus[i] += shift
            weights_minus[i] -= shift

            for x_np in input_data_np:
                init_gate = Initialize(x_np, normalize=True)
                qc = QuantumCircuit(qcnn_ansatz.num_qubits)
                qc.append(init_gate, qc.qubits)
                full_circuit = qc.compose(qcnn_ansatz)

                # Add both plus and minus circuits
                all_circuits.append(full_circuit)
                all_circuits.append(full_circuit)
                all_observables.append(observable)
                all_observables.append(observable)
                all_weights.append(weights_plus)
                all_weights.append(weights_minus)

        # Run all circuits in one batch
        job = estimator.run(all_circuits, all_observables, all_weights)
        results = job.result().values

        # Parse results and compute gradients
        grad_weights = torch.zeros_like(weights)
        idx = 0
        for i in range(num_params):
            plus_results = []
            minus_results = []
            for _ in range(batch_size):
                plus_results.append(results[idx])
                minus_results.append(results[idx + 1])
                idx += 2

            gradient_per_sample = 0.5 * (np.array(plus_results) - np.array(minus_results))
            grad_weights[i] = torch.tensor(np.sum(grad_output_np * gradient_per_sample), device=weights.device)

        return None, grad_weights, None, None, None

# =================================================================================
# Optimized QCNN model
# =================================================================================
class QCNNOptimized(nn.Module):
    def __init__(self, num_qubits: int, num_classes: int, estimator: Estimator, **kwargs):
        super().__init__()
        if estimator is None:
            raise ValueError("An Estimator must be provided.")
        self.estimator = estimator
        self.qcnn_ansatz, _, self.final_qubit_idx = create_qcnn_ansatz(num_qubits)
        self.q_weights = nn.Parameter(torch.randn(self.qcnn_ansatz.num_parameters))
        self.classical_head = nn.Linear(1, num_classes)

    def forward(self, x):
        return self.classical_head(
            QuantumFunctionOptimized.apply(
                x, self.q_weights, self.qcnn_ansatz, self.final_qubit_idx, self.estimator
            )
        )
