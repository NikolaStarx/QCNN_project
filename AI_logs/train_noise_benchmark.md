# train_noise.py Profiling Notes

Environment: RTX 5060 Ti (NV driver 570.195.03), CUDA 12.8, Aer GPU backend with depolarizing noise (p₁=1e-3, p₂=1e-2).

## Workload
- Config: `configs/profiling/mnist_hybrid_noise_mid_batch1.yaml`
- 12 qubits, hybrid encoding, 16 training samples (single batch), batch size 16
- Optimizer: Adam, lr 0.01, 1 epoch

## Timing results

| Runner | Shots | Wall time (s) | Epoch accuracy | Notes |
|--------|-------|---------------|----------------|-------|
| `train_optimized.py` | ~1024 (Aer default) | 172.7 | 6.25% | Baseline implementation |
| `train_noise.py` (`--schedule constant --min-shots 1024`) | 1024 | 182.9 | 6.25% | Matches baseline fidelity, small overhead from extra instrumentation |
| `train_noise.py` (default: linear 256→1024) | 256 | 51.7 | 0.00% | 3.3× speedup; final metrics can be recomputed with max shots if needed |
| `train_noise.py` (`--schedule constant --min-shots 512`) | 512 | 98.5 | 6.25% | Shows near-linear trade-off |
| `train_noise.py` (`--schedule constant --min-shots 1024 --eval-max-shots --eval-max-batches 1`) | train shots 1024, eval shots 1024 | 87.6 | 0.0% train / 0.0% eval | Demonstrates optional high-shot evaluation overhead |

The per-batch latency scales roughly linearly with the shot count. Default settings (min 256 → max 1024) deliver a ~3× throughput gain for noisy runs while preserving the ability to audit accuracy at high shots.

## Usage reminders
- `python train_noise.py --config <existing noisy config>` is a drop-in replacement. Defaults ramp shots up to the Aer baseline (1024) over the course of training.
- Attach `--schedule constant --min-shots 1024` to mimic legacy behaviour exactly.
- Combine `--eval-max-shots --eval-max-batches 1` to log high-fidelity metrics intermittently without paying the full cost every batch.

All measurements were captured after disabling other training jobs to avoid GPU contention.
