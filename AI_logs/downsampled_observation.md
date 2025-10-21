# Low-Resolution Encoding Observation

## Setup
- Dataset: MNIST digits, downsampled to 4×4 (16 features) via average pooling.
- Binary task: digits `6` vs `7` (largest class-mean L2 distance under 4×4 pooling).
- Encoders evaluated (QCNN with depolarizing noise):
  - Amplitude (4 qubits, direct amplitude normalization).
  - Angle (16 qubits, single-layer Ry encoding).
  - Hybrid (8 qubits, single-layer hybrid tiling).
- Shot schedule: constant 64–128; epochs: 10; optimizer: Adam (lr=0.008).

## Empirical Findings
- **Amplitude encoding reaches ≈60–65% accuracy** on downsampled MNIST `6 vs 7` even with 256 samples, indicating the 16-dimensional amplitude vector preserves salient global structure (relative brightness distribution).
- **Angle & Hybrid encodings remain near chance (~50%)** under identical settings. Parameter-shift gradients show non-zero norms, so the issue is not vanishing gradients but insufficient discriminative signal after rotation-based encodings in such low resolution.
- Increasing epochs or shots does not materially improve Angle/Hybrid accuracy; switching to other class pairs (e.g., `0 vs 1`) restores separability, suggesting sensitivity to feature amplitude rather than training instability.

## Interpretation
- At extremely low feature counts, amplitude encoding retains global intensity patterns by embedding the entire vector into a normalized state. The QCNN can exploit aggregate probability differences.
- Angle/hybrid encodings depend on per-qubit rotations; when pixel averages are small (typical in 4×4 pooling), rotations stay near identity. Subsequent pooling and depolarizing noise wash out subtle phase differences, leaving too little signal.
- **Research implication**: In strongly resource-constrained regimes (few qubits, coarse features), amplitude encoding can outperform rotation-based schemes because it preserves relative intensity ratios without relying on large rotation angles. This can be highlighted as a guidance point for hardware-limited QCNN deployments.

## Next Steps
- Validate on additional binary pairs (e.g., `0 vs 1`, `1 vs 6`) to confirm the pattern.
- Repeat on FashionMNIST (`Coat vs Sandal`) to see if amplitude keeps its advantage under low-res pooling.
- Explore mild upsampling (e.g., 6×6) to find the resolution threshold where angle/hybrid recover.
