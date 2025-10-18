# file: test_gpu.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

print("--- [ Step 1: Script Started ] ---")

try:
    # 1. 创建一个明确使用 GPU 的 Aer 模拟器
    # 如果这一步失败，说明 qiskit_aer_gpu 根本没有被正确安装
    print("--- [ Step 2: Creating AerSimulator(device='GPU') ] ---")
    gpu_simulator = AerSimulator(device='GPU')
    print("--- [ Step 3: Simulator created successfully! ] ---")

    # 2. 创建一个简单的量子电路（贝尔态）
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    
    print("\n--- [ Step 4: Running a simple circuit on the GPU... ] ---")
    print("This is the step where it might hang.")
    
    # 3. 在 GPU 上运行电路
    # 我们只运行一次（shots=1），以最快速度得到结果
    result = gpu_simulator.run(circuit, shots=1).result()
    counts = result.get_counts(0)
    
    print("\n--- [ Step 5: GPU execution finished! ] ---")
    
    # 4. 打印结果
    print("\n✅ SUCCESS! qiskit-aer-gpu is working correctly.")
    print("Measurement result:", counts)

except Exception as e:
    print("\n❌ FAILURE! An error occurred.")
    print("Error details:", e)
