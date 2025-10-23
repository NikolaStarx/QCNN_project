## 5 评估指标

### 1.**准确率 / F1**（测试集）
### 2.**资源开销**：qubit 数、两比特门数CNOT、线路深度L
### 3. **训练效率**:收敛 epoch
### 4. **优化稳定性**：梯度范数 $\lVert\nabla\theta\rVert_2$，爆/消失比例；
### 5. **鲁棒性**：噪声下准确率下降 

### 6. **如何选择编码方式，适合什么不同任务** 




## 5. Evaluation Metrics

To provide a comprehensive comparison, we evaluate the models based on the following criteria:

1.  **Classification Performance:** We use **Accuracy** and **F1-Score** on the held-out test set as the primary indicators of the model's generalization capability.
2.  **Resource Cost:** We analyze the hardware requirements of each model, specifically:
    *   **Qubit Count:** The number of qubits required by the encoder.
    *   **Gate Count:** The number of two-qubit CNOT gates in the variational ansatz, a key driver of noise in NISQ devices.
    *   **Circuit Depth:** The length of the longest path of gates in the circuit, which correlates with susceptibility to decoherence.
3.  **Training Efficiency:** We compare the **number of epochs** required for the training loss to converge, providing an estimate of training speed.
4.  **Optimization Stability:** (Optional) The L2-norm of the gradient vector ($\lVert\nabla\theta\rVert_2$) can be monitored during training to detect potential issues like barren plateaus (vanishing gradients) or exploding gradients.
5.  **Robustness:** We explicitly measure the **degradation in test accuracy** when the depolarizing noise model is applied, quantifying the resilience of each encoding scheme.



## References





1.  Cong, I., Choi, S., & Lukin, M. D. (2019). Quantum convolutional neural networks. *Nature Physics*, 15(12), 1273–1278.
2.  Henderson, M., Shakya, S., Pradhan, S., & Cook, T. (2020). Quanvolutional neural networks: powering image recognition with quantum circuits. *Quantum Machine Intelligence*, 2(1), 2.
3.  Hu, S., Li, X., Ruan, B., & Liu, Z. (2025). An Amplitude-Encoding-Based Classical-Quantum Transfer Learning framework: Outperforming Classical Methods in Image Recognition. *arXiv preprint arXiv:2502.20184*.
4.  Bosco, D. L., Portelli, B., & Serra, G. (2024). Integrated Encoding and Quantization to Enhance Quanvolutional Neural Networks. *arXiv preprint arXiv:2410.05777*.
