# Profiling Configurations

This directory collects supplemental YAML configs used for performance diagnostics while keeping the production configurations untouched.

- `mnist_hybrid_noise_mid_baseline.yaml` mirrors the original noisy MNIST hybrid setup but saves checkpoints under `checkpoints/profiling/...` to avoid clashes. Use it to run the exact workload with alternative training scripts (e.g. `train_with_shots.py`) that expose additional controls such as custom shot counts.
- `mnist_hybrid_noiseless_reference.yaml` disables the injected depolarizing noise to provide a physically meaningful noise-free baseline for comparison. The circuit structure, dataset size, and QCNN hyperparameters remain unchanged.
- `mnist_hybrid_noise_mid_mini.yaml` keeps the same noise model but reduces the sample count for quick throughput experiments. This is useful to validate optimisation ideas before committing to the full 15 000-sample run.

All configs write checkpoints into the dedicated `checkpoints/profiling/` subtree. This prevents interference with active long-running experiments.

## Suggested usage

1. **Measure noisy workload with lower shots**  
   ```bash
   python scripts/train_with_shots.py \
     --config configs/profiling/mnist_hybrid_noise_mid_baseline.yaml \
     --shots 512 --log-interval 1
   ```
   This keeps the noise model intact while reducing the sampling budget, allowing you to observe how batch latency scales with shot count.

2. **Establish a noiseless reference**  
   ```bash
   python scripts/train_with_shots.py \
     --config configs/profiling/mnist_hybrid_noiseless_reference.yaml \
     --log-interval 1
   ```
   Compare the reported batch/epoch timings with the noisy run to isolate the simulator’s noise overhead.

3. **Rapid throughput checks**  
   ```bash
   python scripts/train_with_shots.py \
     --config configs/profiling/mnist_hybrid_noise_mid_mini.yaml \
     --shots 256 --log-interval 1
   ```
   Use this for quick iterations; once satisfied, revert to the baseline config to retain physical fidelity.

All three workflows preserve the QCNN architecture and data processing pipeline, ensuring optimisation experiments remain faithful to the underlying physics.

## Reference timings (single batch, noisy MNIST hybrid)

| Runner & settings | Effective shots | Wall time (s) | Notes |
|-------------------|-----------------|---------------|-------|
| `train_optimized.py` | ~1024 (Aer default) | 173 | Baseline script |
| `train_noise.py` (`--schedule constant --min-shots 1024`) | 1024 | 173 | Matches baseline behaviour |
| `train_noise.py` (default linear schedule, min 256 → max 1024) | 256 | **51** | 3.4× faster than baseline |
| `train_noise.py` (`--schedule constant --min-shots 512`) | 512 | 99 | Mid-point trade-off |

Measurements taken with `configs/profiling/mnist_hybrid_noise_mid_batch1.yaml` (single-batch workload) to highlight per-batch latency. Larger datasets scale roughly linearly with the chosen shot count.
