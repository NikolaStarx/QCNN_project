# file: models/qcnn.py (FINAL, CORRECTED TRI-ENCODING VERSION)

import torch
import torch.nn as nn
from torch.autograd import Function
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import Initialize

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
# SECTION 1: FOR AMPLITUDE ENCODING (Unchanged)
# =================================================================================
class QuantumFunctionAmplitude(Function):
    @staticmethod
    def forward(ctx, input_data: torch.Tensor, weights: torch.Tensor, qcnn_ansatz, final_qubit_idx, estimator):
        ctx.qcnn_ansatz, ctx.final_qubit_idx, ctx.input_data, ctx.estimator = qcnn_ansatz, final_qubit_idx, input_data, estimator
        circuits, weight_values = [], [weights.detach().numpy()] * len(input_data)
        for x in input_data:
            init_gate = Initialize(x.detach().numpy(), normalize=True)
            qc = QuantumCircuit(qcnn_ansatz.num_qubits); qc.append(init_gate, qc.qubits); qc.compose(qcnn_ansatz, inplace=True)
            circuits.append(qc)
        num_qubits = qcnn_ansatz.num_qubits
        pauli_string = "I" * (num_qubits - 1 - final_qubit_idx) + "Z" + "I" * final_qubit_idx
        observable = SparsePauliOp(pauli_string)
        job = estimator.run(circuits, [observable] * len(circuits), weight_values)
        ctx.save_for_backward(weights)
        return torch.tensor(job.result().values, dtype=torch.float32).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        qcnn_ansatz, input_data, estimator = ctx.qcnn_ansatz, ctx.input_data, ctx.estimator
        shift, grad_weights = np.pi / 2, torch.zeros_like(weights)
        for i in range(len(weights)):
            weights_plus, weights_minus = weights.detach().numpy().copy(), weights.detach().numpy().copy()
            weights_plus[i] += shift; weights_minus[i] -= shift
            circuits_plus, circuits_minus = [], []
            for x in input_data:
                init_gate = Initialize(x.detach().numpy(), normalize=True)
                qc_plus = QuantumCircuit(qcnn_ansatz.num_qubits); qc_plus.append(init_gate, qc_plus.qubits); qc_plus.compose(qcnn_ansatz, inplace=True)
                circuits_plus.append(qc_plus)
                qc_minus = QuantumCircuit(qcnn_ansatz.num_qubits); qc_minus.append(init_gate, qc_minus.qubits); qc_minus.compose(qcnn_ansatz, inplace=True)
                circuits_minus.append(qc_minus)
            num_qubits = qcnn_ansatz.num_qubits
            pauli_string = "I" * (num_qubits - 1 - ctx.final_qubit_idx) + "Z" + "I" * ctx.final_qubit_idx
            observable = SparsePauliOp(pauli_string)
            job_plus = estimator.run(circuits_plus, [observable] * len(input_data), [weights_plus] * len(input_data))
            job_minus = estimator.run(circuits_minus, [observable] * len(input_data), [weights_minus] * len(input_data))
            gradient_per_sample = 0.5 * (job_plus.result().values - job_minus.result().values)
            grad_weights[i] = torch.sum(grad_output.squeeze() * torch.tensor(gradient_per_sample, dtype=torch.float32))
        return None, grad_weights, None, None, None

class QCNNAmplitude(nn.Module):
    def __init__(self, num_qubits: int, num_classes: int = 2, estimator: Estimator = None):
        super().__init__()
        self.num_qubits = num_qubits
        if estimator is None: raise ValueError("An Estimator must be provided.")
        self.estimator = estimator
        self.qcnn_ansatz, self.q_weights_params, self.final_qubit_idx = create_qcnn_ansatz(num_qubits)
        self.q_weights = nn.Parameter(torch.randn(len(self.q_weights_params)))
        self.classical_head = nn.Linear(1, num_classes)
    def forward(self, x):
        return self.classical_head(QuantumFunctionAmplitude.apply(x, self.q_weights, self.qcnn_ansatz, self.final_qubit_idx, self.estimator))

# =================================================================================
# SECTION 2: FOR GENERAL ENCODINGS (Angle, Hybrid, etc.) (Unchanged)
# =================================================================================
class QuantumFunctionGeneral(Function):
    @staticmethod
    def forward(ctx, input_data: torch.Tensor, weights: torch.Tensor, encoder_circuit, qcnn_ansatz, final_qubit_idx, estimator):
        ctx.encoder_circuit, ctx.qcnn_ansatz, ctx.final_qubit_idx, ctx.input_data, ctx.estimator = \
            encoder_circuit, qcnn_ansatz, final_qubit_idx, input_data, estimator
        circuits, input_values, weight_values = [], [x.detach().numpy() for x in input_data], [weights.detach().numpy()] * len(input_data)
        for x_val in input_values:
            bound_encoder = encoder_circuit.assign_parameters(x_val)
            full_circuit = bound_encoder.compose(qcnn_ansatz)
            circuits.append(full_circuit)
        num_qubits = qcnn_ansatz.num_qubits
        pauli_string = "I" * (num_qubits - 1 - final_qubit_idx) + "Z" + "I" * final_qubit_idx
        observable = SparsePauliOp(pauli_string)
        job = estimator.run(circuits, [observable] * len(circuits), parameter_values=weight_values)
        ctx.save_for_backward(weights)
        return torch.tensor(job.result().values, dtype=torch.float32).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        encoder_circuit, qcnn_ansatz, input_data, estimator = \
            ctx.encoder_circuit, ctx.qcnn_ansatz, ctx.input_data, ctx.estimator
        shift, grad_weights = np.pi / 2, torch.zeros_like(weights)
        for i in range(len(weights)):
            weights_plus, weights_minus = weights.detach().numpy().copy(), weights.detach().numpy().copy()
            weights_plus[i] += shift; weights_minus[i] -= shift
            circuits, input_values = [], [x.detach().numpy() for x in input_data]
            for x_val in input_values:
                bound_encoder = encoder_circuit.assign_parameters(x_val)
                full_circuit = bound_encoder.compose(qcnn_ansatz)
                circuits.append(full_circuit)
            num_qubits = qcnn_ansatz.num_qubits
            pauli_string = "I" * (num_qubits - 1 - ctx.final_qubit_idx) + "Z" + "I" * ctx.final_qubit_idx
            observable = SparsePauliOp(pauli_string)
            job_plus = estimator.run(circuits, [observable] * len(input_data), parameter_values=[weights_plus] * len(input_data))
            job_minus = estimator.run(circuits, [observable] * len(input_data), parameter_values=[weights_minus] * len(input_data))
            gradient_per_sample = 0.5 * (job_plus.result().values - job_minus.result().values)
            grad_weights[i] = torch.sum(grad_output.squeeze() * torch.tensor(gradient_per_sample, dtype=torch.float32))
        return None, grad_weights, None, None, None, None

class QCNNGeneral(nn.Module):
    def __init__(self, num_qubits: int, encoder_fn, num_input_features: int, num_classes: int = 2, estimator: Estimator = None):
        super().__init__()
        self.num_qubits = num_qubits
        if estimator is None: raise ValueError("An Estimator must be provided.")
        self.estimator = estimator

        # --- THE FIX IS HERE ---
        # The number of input parameters must match the number of features.
        self.input_params = ParameterVector('x', num_input_features)
        # -----------------------
        
        self.encoder_circuit = QuantumCircuit(num_qubits, name='Encoder')
        encoder_fn(self.encoder_circuit, self.input_params)
        
        self.qcnn_ansatz, self.q_weights_params, self.final_qubit_idx = create_qcnn_ansatz(num_qubits)
        self.q_weights = nn.Parameter(torch.randn(len(self.q_weights_params)))
        self.classical_head = nn.Linear(1, num_classes)

    def forward(self, x):
        q_out = QuantumFunctionGeneral.apply(x, self.q_weights, self.encoder_circuit, self.qcnn_ansatz, self.final_qubit_idx, self.estimator)
        return self.classical_head(q_out)