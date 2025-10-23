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

