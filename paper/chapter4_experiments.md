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
