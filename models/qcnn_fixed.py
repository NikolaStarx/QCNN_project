# models/qcnn_fixed.py - 修复版本

import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp

# ------------------- Helper Functions for QCNN Layers -------------------

def conv_layer(qc: QuantumCircuit, params, active_qubits):
    """
    Applies a translationally-invariant convolutional layer.
    """
    for i in range(0, len(active_qubits) - 1, 2):
        q1 = active_qubits[i]
        q2 = active_qubits[i+1]
        qc.ry(params[0], q1)
        qc.ry(params[1], q2)
        qc.cx(q1, q2)

def pooling_layer(qc: QuantumCircuit, active_qubits):
    """
    Applies a pooling layer, reducing the number of qubits by half.
    """
    new_active_qubits = []
    for i in range(0, len(active_qubits) - 1, 2):
        q_control = active_qubits[i]
        q_target = active_qubits[i+1]
        qc.cx(q_control, q_target)
        new_active_qubits.append(q_target)
    return new_active_qubits

# ------------------- Main QCNN Class -------------------

class QCNN(nn.Module):
    """
    A PyTorch-compatible QCNN model based on Cong et al. (2019).
    """
    def __init__(self, num_qubits: int, encoder_fn, num_classes: int = 2):
        super().__init__()
        
        if not num_qubits > 0 or (num_qubits & (num_qubits - 1)) != 0:
            raise ValueError("Number of qubits must be a power of 2.")

        self.num_qubits = num_qubits
        self.encoder_fn = encoder_fn
        self.num_classes = num_classes
        
        # 构建量子电路和神经网络
        self._build_quantum_network()
        
        # 经典头部，将量子输出映射到类别分数
        self.classical_head = nn.Linear(1, self.num_classes)

    def _build_quantum_network(self):
        """
        构建量子神经网络
        """
        # 参数定义
        input_params = ParameterVector('x', self.num_qubits)
        
        # 计算需要的权重参数数量
        num_layers = self.num_qubits.bit_length() - 1
        num_weights = 2 * num_layers
        weight_params = ParameterVector('θ', num_weights)
        
        # 构建量子电路
        qc = QuantumCircuit(self.num_qubits)
        
        # 1. 编码层
        self.encoder_fn(qc, input_params)
        qc.barrier()
        
        # 2. 分层卷积和池化层
        active_qubits = list(range(self.num_qubits))
        weight_idx = 0
        while len(active_qubits) > 1:
            layer_weights = [weight_params[i] for i in range(weight_idx, weight_idx + 2)]
            
            # 卷积层
            conv_layer(qc, layer_weights, active_qubits)
            qc.barrier()
            
            # 池化层
            active_qubits = pooling_layer(qc, active_qubits)
            qc.barrier()
            
            weight_idx += 2
        
        # 3. 定义观测算子（最终qubit的Z算子）
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1.0)])
        
        # 4. 创建SamplerQNN
        sampler = AerSimulator()
        
        self.qnn = SamplerQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            observables=observable,
            sampler=sampler
        )
        
        # 5. 创建TorchConnector
        self.torch_connector = TorchConnector(self.qnn)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        """
        # 量子神经网络的输出
        q_out = self.torch_connector(x)
        
        # 通过经典头部进行分类
        class_scores = self.classical_head(q_out)
        return class_scores
