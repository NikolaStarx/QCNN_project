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

