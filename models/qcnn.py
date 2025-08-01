# file: models/qcnn.py

import torch
import torch.nn as nn
from torch.autograd import Function
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import Initialize

# ------------------- QCNN Circuit Building Logic -------------------
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

def create_qcnn_circuit(num_qubits: int):
    num_layers = num_qubits.bit_length() - 1
    num_weights = 2 * num_layers
    weights = ParameterVector('θ', num_weights)
    qc = QuantumCircuit(num_qubits)
    active_qubits = list(range(num_qubits))
    weight_idx = 0
    while len(active_qubits) > 1:
        layer_weights = [weights[i] for i in range(weight_idx, weight_idx + 2)]
        conv_layer(qc, layer_weights, active_qubits)
        qc.barrier()
        active_qubits = pooling_layer(qc, active_qubits)
        qc.barrier()
        weight_idx += 2
    return qc, weights, active_qubits[0]

# ------------------- Custom Autograd Function for QCNN -------------------
class QuantumFunction(Function):
    @staticmethod
    def forward(ctx, input_data: torch.Tensor, weights: torch.Tensor, qcnn_ansatz, final_qubit_idx, estimator):
        ctx.qcnn_ansatz = qcnn_ansatz
        ctx.final_qubit_idx = final_qubit_idx
        ctx.input_data = input_data
        ctx.estimator = estimator
        
        circuits = []
        weight_values = [weights.detach().numpy()] * len(input_data)
        
        for x in input_data:
            x_np = x.detach().numpy()
            init_gate = Initialize(x_np, normalize=True)
            
            qc = QuantumCircuit(qcnn_ansatz.num_qubits)
            qc.append(init_gate, qc.qubits)
            qc.compose(qcnn_ansatz, inplace=True)
            circuits.append(qc)

        num_qubits = qcnn_ansatz.num_qubits
        pauli_string = "I" * (num_qubits - 1 - final_qubit_idx) + "Z" + "I" * final_qubit_idx
        observable = SparsePauliOp(pauli_string)

        job = estimator.run(circuits, [observable] * len(circuits), weight_values)
        result = job.result().values
        
        ctx.save_for_backward(weights)
        return torch.tensor(result, dtype=torch.float32).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        qcnn_ansatz = ctx.qcnn_ansatz
        input_data = ctx.input_data
        estimator = ctx.estimator
        
        shift = np.pi / 2
        grad_weights = torch.zeros_like(weights)

        for i in range(len(weights)):
            weights_plus = weights.detach().numpy().copy(); weights_plus[i] += shift
            weights_minus = weights.detach().numpy().copy(); weights_minus[i] -= shift
            
            circuits_plus, circuits_minus = [], []
            
            for x in input_data:
                x_np = x.detach().numpy()
                init_gate = Initialize(x_np, normalize=True)

                qc_plus = QuantumCircuit(qcnn_ansatz.num_qubits)
                qc_plus.append(init_gate, qc_plus.qubits)
                qc_plus.compose(qcnn_ansatz, inplace=True)
                circuits_plus.append(qc_plus)
                
                qc_minus = QuantumCircuit(qcnn_ansatz.num_qubits)
                qc_minus.append(init_gate, qc_minus.qubits)
                qc_minus.compose(qcnn_ansatz, inplace=True)
                circuits_minus.append(qc_minus)

            num_qubits = qcnn_ansatz.num_qubits
            pauli_string = "I" * (num_qubits - 1 - ctx.final_qubit_idx) + "Z" + "I" * ctx.final_qubit_idx
            observable = SparsePauliOp(pauli_string)

            job_plus = estimator.run(circuits_plus, [observable] * len(input_data), [weights_plus] * len(input_data))
            exp_val_plus = job_plus.result().values
            job_minus = estimator.run(circuits_minus, [observable] * len(input_data), [weights_minus] * len(input_data))
            exp_val_minus = job_minus.result().values
            
            gradient_per_sample = 0.5 * (exp_val_plus - exp_val_minus)
            grad_weights[i] = torch.sum(grad_output.squeeze() * torch.tensor(gradient_per_sample, dtype=torch.float32))

        return None, grad_weights, None, None, None

# ------------------- The Main QCNN Model -------------------
class QCNNAmplitude(nn.Module):
    def __init__(self, num_qubits: int, num_classes: int = 2, estimator: Estimator = None):
        super().__init__()
        self.num_qubits = num_qubits
        self.qcnn_ansatz, self.q_weights_params, self.final_qubit_idx = create_qcnn_circuit(num_qubits)
        self.q_weights = nn.Parameter(torch.randn(len(self.q_weights_params)))
        self.classical_head = nn.Linear(1, num_classes)
        
        if estimator is None:
            raise ValueError("An Estimator instance must be provided to the model.")
        self.estimator = estimator

    def forward(self, x):
        q_out = QuantumFunction.apply(x, self.q_weights, self.qcnn_ansatz, self.final_qubit_idx, self.estimator)
        return self.classical_head(q_out)