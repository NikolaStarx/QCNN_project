# Noise Configuration File Generation Plan

**Date:** 2025-10-18
**Task:** Generate 18 noise-injected configuration files from the source directory `configs/full_scale/` into the target directory `configs/config_noise/`.

---

## 1. Task Summary

The user requests the following:

1. Copy all configuration files from `configs/full_scale/` to `configs/config_noise/`.
2. For each source configuration file, create **three noise variations**: low, medium, and high noise.
3. The final count should be **18 configuration files** in `configs/config_noise/`.

This implies that there are **6 base configuration files** in `configs/full_scale/` (6 × 3 = 18).

---

## 2. Noise Parameter Selection Rationale

### 2.1 Background: Depolarizing Noise in Quantum Simulations

Quantum noise is a critical factor in simulating realistic quantum hardware. In the Qiskit framework, **depolarizing noise** is a standard noise model that simulates the effect of decoherence and gate errors:

- **Single-qubit gate error** (`depolarizing_p1`): Affects operations like `ry`, `rz`, and `h`.
- **Two-qubit gate error** (`depolarizing_p2`): Affects entangling gates like `cx` (CNOT).

These error rates are typically expressed as probabilities. Real near-term quantum devices (NISQ devices) have reported gate fidelities in the following ranges:

- Single-qubit gates: **99.9% to 99.99%** fidelity → error rates of **0.001 to 0.0001** (0.1% to 0.01%)
- Two-qubit gates: **99% to 99.9%** fidelity → error rates of **0.01 to 0.001** (1% to 0.1%)

Source: IBM Quantum, Google Quantum AI, and Rigetti hardware benchmarks (as of 2023-2024).

### 2.2 Proposed Noise Levels

Based on realistic quantum hardware characteristics and the need for a clear experimental gradient, I propose the following noise levels:

| Noise Level | `depolarizing_p1` (single-qubit) | `depolarizing_p2` (two-qubit) | Description |
|-------------|----------------------------------|-------------------------------|-------------|
| **Low**     | 0.0005                           | 0.005                         | Optimistic, high-fidelity gates (close to best current hardware) |
| **Medium**  | 0.001                            | 0.01                          | Realistic near-term NISQ device performance |
| **High**    | 0.005                            | 0.05                          | Pessimistic, high-noise scenario (early-generation or degraded hardware) |

**Rationale:**

- **Low noise** represents **optimistic** performance, close to the best-reported single-qubit and two-qubit gate fidelities from leading quantum hardware providers.
- **Medium noise** is a **realistic** middle ground for current NISQ devices.
- **High noise** simulates **early-generation or degraded hardware**, useful for testing the robustness of the QCNN model under challenging conditions.

The two-qubit error rate is consistently **10x higher** than the single-qubit error rate, which is a standard assumption in the literature (two-qubit gates are inherently more error-prone due to the complexity of entangling operations).

---

## 3. Operational Procedure

### 3.1 Pre-Execution Verification

1. **Verify source directory exists:** Check if `configs/full_scale/` exists and contains exactly 6 `.yaml` files.
   - If the directory does not exist or does not contain the expected number of files, I will report this to the user for clarification.

### 3.2 File Generation Workflow

For each source configuration file in `configs/full_scale/`:

1. **Read the source YAML file** and parse it into a Python dictionary.
2. **Modify the noise section** in the dictionary:
   - Set `environment.add_noise` to `true`.
   - Set `environment.noise.depolarizing_p1` and `environment.noise.depolarizing_p2` according to the noise level (low, medium, or high).
3. **Generate a new filename** with the naming convention:
   ```
   <original_name>_noise_<level>.yaml
   ```
   Example: If the source file is `mnist_amplitude.yaml`, the three output files will be:
   - `mnist_amplitude_noise_low.yaml`
   - `mnist_amplitude_noise_mid.yaml`
   - `mnist_amplitude_noise_high.yaml`
4. **Write the modified YAML** to the new file in `configs/config_noise/`.
5. **Repeat** for all source files.

### 3.3 Final Verification

After generation, I will:

1. List all files in `configs/config_noise/` to confirm that exactly 18 files were created.
2. Optionally display a sample file to verify the noise values are correctly set.

---

## 4. Expected Output

Assuming the source files in `configs/full_scale/` are named as follows (example):

1. `mnist_amplitude.yaml`
2. `mnist_angle.yaml`
3. `mnist_hybrid.yaml`
4. `fashion_amplitude.yaml`
5. `fashion_angle.yaml`
6. `fashion_hybrid.yaml`

The final output in `configs/config_noise/` will be:

1. `mnist_amplitude_noise_low.yaml`
2. `mnist_amplitude_noise_mid.yaml`
3. `mnist_amplitude_noise_high.yaml`
4. `mnist_angle_noise_low.yaml`
5. `mnist_angle_noise_mid.yaml`
6. `mnist_angle_noise_high.yaml`
7. `mnist_hybrid_noise_low.yaml`
8. `mnist_hybrid_noise_mid.yaml`
9. `mnist_hybrid_noise_high.yaml`
10. `fashion_amplitude_noise_low.yaml`
11. `fashion_amplitude_noise_mid.yaml`
12. `fashion_amplitude_noise_high.yaml`
13. `fashion_angle_noise_low.yaml`
14. `fashion_angle_noise_mid.yaml`
15. `fashion_angle_noise_high.yaml`
16. `fashion_hybrid_noise_low.yaml`
17. `fashion_hybrid_noise_mid.yaml`
18. `fashion_hybrid_noise_high.yaml`

**Total: 18 files.**

---

## 5. Next Steps

1. **User approval required:** Please review the proposed noise values in Section 2.2 and the operational procedure in Section 3.
2. **Upon approval:** I will execute the file generation script.
3. **Post-execution:** I will verify the output and report the results.

---

## 6. Notes

- If the source directory `configs/full_scale/` does not exist or contains a different number of files, I will halt and request clarification.
- The `experiment_name` field in each YAML file will also be updated to reflect the noise level (e.g., `mnist_amplitude_noise_low`).
- The generated files will preserve all other settings from the source files (e.g., batch size, number of epochs, learning rate).
