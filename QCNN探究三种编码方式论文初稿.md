# Angle、Amplitude 与 Hybrid 编码下的量子卷积神经网络在 MNIST 分类任务中/其他类型数据集(fasion-mnist,quick draw) 的模拟比较


## 1. Introduction

The relentless progress of classical computing, famously described by Moore's Law, is encountering fundamental physical limits. In the quest for new computational paradigms, quantum computing has emerged as a revolutionary approach. Unlike classical bits that exist as either 0 or 1, a quantum bit (qubit) can exist in a **superposition** of both states simultaneously. Harnessing this principle, along with **entanglement**—a unique quantum correlation—allows a quantum computer to explore an exponentially large computational space using a modest number of qubits. The fusion of these two fields has given rise to Quantum Machine Learning (QML), a frontier that seeks to leverage this quantum power to solve problems intractable for even the most powerful classical supercomputers. However, bridging the classical and quantum worlds presents a critical challenge: **data encoding**, the process of translating classical data like images into the language of quantum states. This is not a mere technicality; the choice of encoding scheme profoundly impacts the performance, resource requirements, and ultimate scalability of any QML model.

Among the various QML architectures, the Quantum Convolutional Neural Network (QCNN), introduced by Cong et al. [1], has garnered significant attention. Drawing inspiration from the multiscale entanglement renormalization ansatz (MERA), the QCNN architecture is particularly well-suited for hierarchical feature extraction, mirroring the function of classical CNNs. Its primary advantage lies in its parameter efficiency, requiring only $\mathcal{O}(\log N)$ trainable parameters for an $N$-qubit system. This logarithmic scaling makes the QCNN an especially attractive candidate for implementation on current Noisy Intermediate-Scale Quantum (NISQ) devices, which are constrained by limited qubit counts and high error rates.

However, the performance of a QCNN is fundamentally determined by its initial data encoding strategy. A spectrum of methods has been proposed, each embodying a different trade-off. For instance, **Angle encoding** offers simplicity and hardware amenability by mapping pixel values to the rotation angles of individual qubits. In contrast, **Amplitude encoding** provides exponential efficiency in qubit usage by embedding an entire image vector into the amplitudes of a quantum state, but at the cost of a potentially deep and complex state preparation circuit. More recently, **Hybrid encoding** schemes have emerged to strike a balance between these extremes. Despite the availability of these methods, the literature lacks a systematic, side-by-side comparison of how they affect a QCNN's performance, resource cost, and trainability within a unified framework.

This paper directly addresses this gap by conducting a comprehensive, simulation-based comparison of Angle, Amplitude, and Hybrid encoding for QCNNs applied to image classification tasks. Using the **MNIST** and **Fashion-MNIST** datasets, we evaluate these methods within a consistent, end-to-end QML workflow built with **PyTorch** and **Qiskit**. Our primary contributions are:

1.  **Unified Framework:** We develop and implement a modular, open-source QML framework that integrates QCNN models with all three encoding schemes, ensuring a fair and reproducible comparison.
2.  **Comparative Analysis:** We conduct a detailed evaluation on binary classification tasks, assessing each encoding strategy based on key metrics including accuracy, resource cost, training convergence, and robustness under simulated depolarizing noise.
3.  **Practical Guidelines:** Based on our empirical results, we distill the trade-offs inherent to each encoding method and propose practical guidelines for selecting an appropriate strategy for QCNN-based image classification on near-term quantum hardware.

## 2. Literature Review

This section reviews the foundational concepts and prior work that underpin our study, focusing on the QCNN architecture and the data encoding techniques central to its application.

#### 2.1. Quantum Convolutional Neural Network (QCNN)

The QCNN architecture was first introduced by Cong, Choi, and Lukin [1] as a quantum analogue to classical CNNs, designed for identifying features in quantum data. The model's structure is inspired by tensor networks, specifically the Multiscale Entanglement Renormalization Ansatz (MERA), which provides a hierarchical framework for feature extraction.

A QCNN consists of alternating layers of convolutional and pooling operations.
*   **Convolutional layers** apply parameterized two-qubit unitary gates to neighboring qubits. This operation creates entanglement and extracts local features. Critically, the variational parameters of these unitaries are shared across the layer, analogous to the shared weights of a classical convolutional filter.
*   **Pooling layers** reduce the system's dimensionality. This is typically achieved by applying a controlled operation between two qubits and then discarding one of them (e.g., by measurement and reset), effectively pooling information from multiple qubits into a smaller subset.

This alternating structure systematically reduces the number of qubits while creating increasingly global correlations, enabling the QCNN to analyze data at multiple scales. Its logarithmic parameter count makes it a highly efficient variational quantum algorithm, well-suited for the constraints of NISQ-era devices.

#### 2.2. Data Encoding Methods

The performance of a QML model on classical data is heavily dependent on the chosen feature map. This choice represents a critical trade-off between qubit resources, circuit depth, and the amount of information encoded. Our study focuses on three prominent methods.

**Angle Encoding:** Also known as rotation encoding, this is one of the most direct methods for embedding classical data. As utilized in "Quanvolutional" networks by Henderson et al. [2], each classical feature $x_j$ (e.g., a normalized pixel value) is mapped to the rotation angle of a single-qubit gate. For a feature vector $x$, the encoding can be expressed as:
$$
|\psi(x)\rangle = \bigotimes_{j=1}^{n} R_Y(\pi x_j)|0\rangle
$$
For a patch of $k \times k$ pixels, this requires $n=k^2$ qubits. This local approach is robust and simple to implement, requiring only shallow circuit depth per feature, making it highly compatible with near-term hardware.

**Amplitude Encoding:** This technique offers exponential compression by embedding a $d$-dimensional normalized classical vector $x$ into the amplitudes of a quantum state using just $n = \lceil\log_2(d)\rceil$ qubits. As analyzed by Hu et al. [3] and Chen et al., the resulting state is:
$$
|x\rangle = \frac{1}{\|x\|} \sum_{i=0}^{d-1} x_i |i\rangle
$$
where $\|x\|$ is the L2 norm of the vector. While exceptionally efficient in qubit count, its primary drawback is the state preparation cost. Generating an arbitrary state with this method can require a circuit depth that scales polynomially with the vector size $d$, posing a significant challenge for NISQ devices susceptible to decoherence and gate errors.

**Hybrid Encoding:** To mediate the trade-offs between Angle and Amplitude encoding, sophisticated schemes that combine multiple encoding primitives have been proposed. These methods often integrate the encoding and processing steps. For instance, Biswas details a strategy that combines Angle and Phase encoding ($R_Y$ and $R_Z$ gates) to store more information on each qubit. Similarly, Bosco et al. [4] introduced a scheme where data parameterizes multi-qubit entangling gates. Our work implements a similar "integrated" strategy where features from an image patch are used to parameterize a sequence of single-qubit rotations and two-qubit entangling gates, thereby embedding the data while simultaneously creating entanglement. This offers a balanced compromise between qubit efficiency and circuit depth.

**Despite the individual merits of these techniques, a practical, side-by-side comparison of their impact on a QCNN's classification performance, training dynamics, and resource costs within a unified experimental setup remains an open area of investigation. This study directly addresses that gap.**

## 3. Methodology

This section details the experimental framework, model architecture, and training procedures used to compare the different QCNN encoding schemes. Our implementation leverages PyTorch for classical optimization and data handling, and Qiskit for quantum circuit simulation. The resulting hybrid workflow is designed to be general-purpose, applicable to various image classification tasks beyond the initial binary case.

### 3.1. System Architecture

A primary challenge in building our hybrid model was integrating Qiskit's quantum circuit construction with PyTorch's automatic differentiation engine. While high-level abstractions like Qiskit's `TorchConnector` are convenient, they were found to be incompatible with operations essential to our research, particularly the `Initialize` instruction required for true amplitude encoding. This is because `Initialize` expects concrete numerical data during circuit construction, whereas a PyTorch-compatible workflow requires a circuit defined with symbolic `ParameterVector` placeholders.

To overcome this fundamental conflict, we engineered a more granular bridge using a custom **`torch.autograd.Function`** class. This approach provides complete control over the quantum-classical interface, implemented in our `models/qcnn.py` module.

#### Implementation via Custom Autograd Function

Our solution involves two main components: a `forward` pass for execution and a `backward` pass for gradient computation.

*   **Forward Pass**: For each data batch, the `forward` function dynamically constructs and executes concrete quantum circuits. As shown in the code snippet below, for amplitude encoding, we instantiate an `Initialize` gate directly with the numerical batch data (`x_np`). The circuits are then run on a **Qiskit Aer `Estimator` primitive** to compute the final expectation values, which are returned as a standard PyTorch tensor.

    ```python
    # In models/qcnn.py: QuantumFunctionAmplitude.forward
    
    for x in input_data:
        # Create a concrete Initialize gate with numerical data
        init_gate = Initialize(x.detach().numpy(), normalize=True)
        
        qc = QuantumCircuit(qcnn_ansatz.num_qubits)
        qc.append(init_gate, qc.qubits)
        qc.compose(qcnn_ansatz, inplace=True)
        circuits.append(qc)
    
    # ... run circuits with Estimator ...
    ```

*   **Backward Pass**: To enable gradient-based optimization, the `backward` pass implements the **parameter-shift rule**. It systematically shifts each variational parameter $\theta_i$ by $\pm \frac{\pi}{2}$, re-runs the simulation to calculate the corresponding expectation values $\langle E \rangle_{\pm}$, and computes the exact analytic gradient as $\frac{\partial \langle E \rangle}{\partial \theta_i} = \frac{1}{2} (\langle E \rangle_{+} - \langle E \rangle_{-})$. These gradients are then seamlessly passed back to PyTorch's `autograd` engine.

    ```python
    # In models/qcnn.py: QuantumFunctionAmplitude.backward
    
    for i in range(len(weights)):
        # Shift the i-th weight
        weights_plus = weights.detach().numpy().copy(); weights_plus[i] += np.pi / 2
        weights_minus = weights.detach().numpy().copy(); weights_minus[i] -= np.pi / 2
        
        # ... re-run circuits with shifted weights ...
        
        exp_val_plus = job_plus.result().values
        exp_val_minus = job_minus.result().values
        
        # Compute gradient via parameter-shift rule
        gradient_per_sample = 0.5 * (exp_val_plus - exp_val_minus)
        
        # ... apply chain rule with grad_output ...
    ```

This architecture ensures that all three encoding schemes can be implemented within a unified, end-to-end differentiable framework. All simulations were conducted with the Qiskit Aer simulator, which supports both ideal and noisy simulations, as well as GPU acceleration.

### 3.2. Datasets and Preprocessing

To evaluate our framework across different levels of complexity, we utilize two standard computer vision datasets: **MNIST** and **Fashion-MNIST**. Preprocessing is tailored to the specific requirements of each encoding scheme.

1.  **Amplitude Encoding**: The full $28 \times 28$ image is flattened into a 784-dimensional vector. To match the state vector dimension of a 10-qubit system ($2^{10}=1024$), the vector is padded with zeros to a length of 1024. Finally, the entire vector is L2-normalized. This process is performed **offline** using a dedicated script, and the resulting tensors are saved to disk for efficient loading during training.

2.  **Angle Encoding**: An input image is processed **on-the-fly** by the data loader. It is downsampled to a smaller patch (e.g., $8 \times 8$ for 64 qubits), flattened, and the pixel values (already normalized to $[0, 1]$) are used directly. Each pixel value $p$ parameterizes a single-qubit rotation $R_Y(\pi \cdot p)$.

3.  **Hybrid Encoding**: Also processed **on-the-fly**, a patch (e.g., $4 \times 4$ for 16 features) is extracted. The 16 pixel values are used to parameterize a sequence of single-qubit ($R_Y, R_Z$) and two-qubit ($CX$) gates within a fixed encoder circuit structure, designed to encode 4 features per pair of qubits over two layers.

### 3.3. QCNN Architecture

Our network follows the hierarchical design of Cong et al. [1], in which convolutional and pooling operations alternate to progressively reduce the number of active qubits while expanding the effective receptive field. In each convolutional stage, a translationally invariant two-qubit unitary is applied to adjacent pairs of active qubits; specifically, two single-qubit $R_Y$ rotations precede a controlled-$X$ operation, and the two rotation angles are shared across all pairs within the same layer, analogous to weight sharing in classical CNNs. Pooling is implemented by applying a $CX$ from the control to the target qubit of each pair and then discarding the control qubits, effectively halving the active register for the subsequent stage. For an initial register of $N$ qubits (with $N$ a power of two), this convolution–pooling motif is repeated $\log_2(N)$ times until a single qubit remains. The readout is given by the expectation value of the Pauli-$Z$ operator on that remaining qubit, yielding a one-dimensional feature that is fed to a classical linear layer whose output dimension matches the number of target classes.

### 3.4. QCNN Architecture

Training proceeds end-to-end using the Adam optimizer with a cross-entropy objective, which we adopt for both binary and multi-class classification. To assess robustness under realistic noise, we configure the Qiskit Aer Estimator with a depolarizing noise model, applying a depolarizing channel of probability $p_1$ to single-qubit gates and probability $p_2$ to two-qubit entangling operations (notably $CX$). All experimental settings—including dataset and encoding choice, qubit and feature counts, batch size, optimizer hyperparameters, and noise parameters—are specified in YAML configuration files to ensure reproducibility and transparent ablations.

## 4. Experiments

To empirically evaluate the performance of the three encoding schemes, we designed a series of simulation experiments. We begin with a foundational binary classification task on MNIST to establish a baseline, followed by more complex multi-class and cross-dataset tasks to assess the scalability and transferability of our framework.

### 4.1. Experiment 1: Binary Classification on MNIST

Our initial experiment focuses on a well-defined binary task: distinguishing between handwritten digits '0' and '1'. This serves to validate our framework and provide a clear, initial comparison of the three encoding schemes. The key hyperparameters for this experiment are detailed in Table 1.

| Parameter | Amplitude Encoding | Angle Encoding | Hybrid Encoding |
| :--- | :--- | :--- | :--- |
| **Input Features** | 1024 (padded from 784) | 64 (from 8x8 patch) | 16 (from 4x4 patch) |
| **Qubits** | 10 | 64 | 4 |
| **Trainable Params** | 18 | 10 | 4 |
| **Training Samples**| 5000 | 5000 | 5000 |
| **Epochs** | 10 | 10 | 10 |

**Table 1:** Hyperparameter settings for the MNIST binary classification experiment.

The results, including learning curves and final test set performance under both ideal and noisy conditions, are presented in .

*(Placeholder for binary classification results, tables, and figures. This is where the output from our current Colab notebook will go.)*

### 4.2. Experiment 2: Multi-class Classification on Full MNIST

To assess the scalability of each encoding scheme, we extend our analysis to the full 10-class MNIST dataset. The models are trained to distinguish all digits from '0' to '9'. The QCNN architecture remains the same, with the only change being the output dimension of the final classical linear layer, which is set to 10.

*(Placeholder for a discussion on the expected challenges, such as increased training time for the Angle encoding due to the large number of qubits, and potential convergence difficulties. Placeholder tables and figures for 10-class results will be added here.)*

### 4.3. Experiment 3: Transferability Study on Fashion-MNIST

To evaluate the generalizability of our findings beyond the MNIST dataset, we apply the same framework to Fashion-MNIST. We conduct a binary classification task on this dataset (e.g., 'T-shirt' vs. 'Trouser'), which is known to be more challenging than MNIST digit classification. This experiment tests whether the relative performance of the encoding schemes holds across different data distributions.

*(Placeholder for Fashion-MNIST experimental setup, results, and a comparative discussion against the MNIST findings.)*
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
