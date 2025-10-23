# Config & Checkpoint Catalog (noise series inventory)

This note consolidates every configuration suite in `configs/` and the corresponding checkpoint trees so we can trace why each batch of training ran and what data it produced. Numbers below are taken from the YAML definitions and the metadata saved inside the `.pt` checkpoints (state dicts include the originating config). All paths are workspace-relative.

## Config Suites

### `configs/cnn_baselines`
- 10 angle-encoding CNN baselines on MNIST digits 6 vs 7 (4×4 and 8×8 patches).
- Fixed setup: 16 qubits/features, batch 8, 256/64 samples, 10 epochs, lr 0.008.
- Noise sweep: no-noise plus p₁/p₂ ∈ {5e‑5/5e‑4, 1.5e‑4/1.5e‑3, 4e‑4/4e‑3}.
- Saves under `checkpoints_cnn/...`.

### `configs/config_noise`
- 18 configs covering MNIST + Fashion-MNIST, amplitude/angle/hybrid encodings (10 each).
- Full 10-class tasks (no `label_subset`), large sample counts (12k–30k train, 3k–5k test), batch sizes {8,16,32}.
- Uniform lr 0.01, 10 epochs, GPU backend with three depolarising levels: (5e‑4/5e‑3), (1e‑3/1e‑2), (5e‑3/5e‑2).
- Checkpoints land in the top-level `checkpoints/` tree (`fashion_*`, `mnist_*` folders).

### `configs/full_scale`
- 6 configs (MNIST + Fashion-MNIST, three encodings) for full 10-class experiments.
- High sample regime matching `config_noise` but noiseless baseline (GPU, `add_noise: false`).
- Used to populate `checkpoints/fashion_*` and `checkpoints/mnist_*` (`best.pt` vs noisy counterparts).

### `configs/lightweight`
- 6 quick sanity configs (MNIST + Fashion, three encodings) with tiny sample counts (64–128 train, ≤64 test), epochs 3, lr 0.05.
- Mostly CPU, no noise; intended for smoke tests and performance checks.
- Writes to `checkpoints/lightweight/...`.

### `configs/noise_light`
- 18 configs (Fashion + MNIST, three encodings) for 6‑epoch “light” noise sweeps with moderate dataset subsampling (6k or 8k train, 1.2k/1.6k test).
- Noise grid: (3e‑4/3e‑3), (8e‑4/8e‑3), (2.5e‑3/2.5e‑2). Batch sizes {8,12,16}.
- Results expected in `checkpoints/lightweight` (older runs) or still pending—no dedicated checkpoint tree in repo yet.

### `configs/noise_2`
- 18 MNIST/Fashion configs, amplitude + angle + hybrid (even split) for binary 0 vs 1 classification.
- Samples: 3k or 4k train, 600–800 test; batches {6,8,12}; epochs 10; lr 0.01.
- Noise levels: (2e‑4/2e‑3), (6e‑4/6e‑3), (1.5e‑3/1.5e‑2).
- Checkpoints stored in `checkpoints_noise2/...` (contains amplitude 73–86 % best acc, hybrid up to 88 % for low noise).

### `configs/noise_2.1`
- 18 configs continuing the 0 vs 1 study with shorter runs (4 epochs) and smaller sample sizes (192 train / 48 test) to benchmark speed.
- Batch sizes: amplitude=16, others=8; lr 0.008.
- Same noise triplet as `noise_2.1`: (5e‑5/5e‑4), (2e‑4/2e‑3), (6e‑4/6e‑3).
- Outputs are in `checkpoints_noise21/...`; companion stress-test batch size variants live in `checkpoints_noise21_test/...` (batch 24 angle runs).

### `configs/noise_2.2`
- 18 configs targeting digits 4 vs 5 (Fashion) and 6 vs 7 (MNIST) with larger batches (64) and 4096/1024 samples.
- Encodings: amplitude (4 qubits), angle (16), hybrid (8); lr 0.008; epochs 10.
- Noise: (1e‑4/1e‑3), (3e‑4/3e‑3), (7e‑4/7e‑3).
- Checkpoints should sit under a yet-to-be-sync’d tree (not present—experiments likely pending).

### `configs/noise_2.3`, `noise_2.3.1`, `noise_2.3.2`, `noise_2.3.3`
- Each folder contains 18 configs (Fashion + MNIST, all encodings) for digits 4 vs 5 and 6 vs 7 with 256/64 samples, batch {8,16}, epochs 10, lr 0.008.
- Noise grid identical across these 2.3* suites: (5e‑5/5e‑4), (1.5e‑4/1.5e‑3), (4e‑4/4e‑3); differences are in checkpoint destinations:
  - `noise_2.3` → `checkpoints_noise23/...` (high-accuracy amplitude runs up to 97 %).
  - `noise_2.3.1` → `checkpoints_noise23_1/...` (adds Fashion amplitude sweeps and hybrid runs ~60 %).
  - `noise_2.3.2` → expected (not yet committed; configs exist for future runs).
  - `noise_2.3.3` → also pending; use same templates with new checkpoint prefixes.

### `configs/noise_2.4`, `noise_2.4.1`, `noise_2.4.2`, `noise_2.4.3`
- 18 configs per folder, digits 4 vs 5 / 6 vs 7 with downsampled 8×8 patches (amplitude uses 6 qubits, angle 16, hybrid 8; `num_features` = 64 when applicable).
- 256/64 samples, batch {8,16}, epochs 10, lr 0.008, noise triplet identical to 2.3-series.
- `noise_2.4.1` results appear in `checkpoints_noise23_4_1/...` (MNIST-only sweeps).
- Plain `noise_2.4`/`.2`/`.3` configs point to `checkpoints_noise23_4/...` (currently absent—runs likely pending or archived elsewhere).

### `configs/noise_2.5`, `noise_2.5.1`, `noise_2.5.2`, `noise_2.5.3`
- 18 configs each, Fashion (0 vs 7) and MNIST (0 vs 1) after more aggressive downsampling (amplitude: 6 qubits, hybrid: 8 qubits @64 features, angle: 64 qubits).
- 256/64 samples, batch {8,16}, epochs 10, lr 0.008; same three noise levels as previous suites.
- Principal checkpoints are in `checkpoints_noise25_1/...` (contains amplitude up to ~74 %, hybrid 36–57 %, angle ~59 %).
- Additional variants (`noise_2.5.2`/`.3`) share dirs but tweak experiment names for repeated sweeps.

### `configs/noise_2.5.4`
- 6 hybrid-only configs (MNIST 0 vs 1, Fashion 0 vs 7) with increased sample counts (512/128) and lower lr 0.006.
- Noise: three standard levels; batch 8, epochs 10.
- No dedicated checkpoints committed yet.

### `configs/noise_2.6`
- 8 configs focusing on MNIST digits 6 vs 7 with deep QCNN (`conv_depth: 3`) on downsampled 4×4 patches.
- Hybrid (8 qubits) and angle (16 qubits) variants, batch 8, 256/64 samples, epochs 10, lr 0.008.
- Noise: low/mid/high plus noiseless. Hybrid results are in `checkpoints_noise26/mnist_hybrid/...`; angle runs still in progress (only `noise_none` directory stub so far).

### `configs/noise_2.6.3`
- Copy of the noise_2.6 suite with intensity amplification (`hybrid_scale` / `angle_scale` = 3.0) to test whether boosting the 4×4 patch helps training.
- Uses dedicated checkpoint paths under `checkpoints_noise263/...` (`*_noise263_*` prefixes) so scaled runs remain separate from the baseline noise_2.6 results.

### `configs/profiling`
- 4 hybrid MNIST configs for performance diagnostics: full 15k/3k dataset (noiseless & noisy) plus mini/micro batches (1024 samples and 16-sample single-epoch run).
- All use 12 qubits, 144 features, batch 16, lr 0.01.
- Outputs reside under `checkpoints/profiling/...`.

### Standalone quick configs
- `configs/fashion_{amplitude,angle,hybrid}.yaml`: Fashion-MNIST binary (default pairs) miniature runs on CPU, 256/128 samples, 5 epochs.
- `configs/mnist_{amplitude,angle,hybrid}.yaml`: MNIST binary baselines (some with depolarising noise).
- `configs/mnist_amplitude_{fast,quick}.yaml`: reduced-sample smoke tests (batch 4/2 respectively) with GPU/no-noise for the fast variant.
- `configs/mnist_angle_fast.yaml`: angle encoding 8×8 patch benchmark (num_features = 64).
- `configs/mnist_amplitude_clean.yaml`: placeholder (empty) reserved for a noise-free amplitude config.
- Checkpoints for these reside in the root `checkpoints/` tree (`best.pt`, `mnist_*`, `fashion_*`, and `lightweight/*` directories).

## Checkpoint Suites

### `checkpoints/`
- Legacy and quick-run experiments driven by the standalone configs and `full_scale`/`lightweight`.
- Contains subfolders `fashion_*`, `mnist_*`, and `lightweight/*`, plus `best.pt`/`last.pt` for early amplitude trials.
- Accuracy ranges: full-scale amplitude ~28–29 % on Fashion 10-class, lightweight sweeps reach 67 % (MNIST amplitude) / 79 % (Fashion amplitude) in 3 epochs.

### `checkpoints_cnn/`
- Classical CNN baselines corresponding to `configs/cnn_baselines` (4×4 and 8×8 patches).
- Each run logs 10 epochs with exceptionally high accuracy (≈95–100 %) even under injected depolarising noise, providing classical upper bounds.

### `checkpoints_noise2/`
- Binary 0 vs 1 study (MNIST-heavy) covering three encodings and noise levels.
- Best results: hybrid up to 88 %, amplitude low-noise ~86 %, angle remains near 53 %.
- Fashion hybrid run (noise_high) capped at ~52 %—useful contrast for dataset difficulty.

### `checkpoints_noise21/` & `checkpoints_noise21_test/`
- Short 4-epoch follow-ups from `noise_2.1`, focusing on MNIST angle encoding with reduced (and test) batch sizes.
- Accuracies hover mid-40 % (reflecting the tiny dataset), with `*_test` variant using batch 24 to stress batching logic.

### `checkpoints_noise23/`
- First large rescan for digits 4/5 & 6/7 (same configs as `noise_2.3`).
- Amplitude runs achieve 95–97 % accuracy across noise settings; hybrid peaks ≈61 %, angle stays ~51–56 %.

### `checkpoints_noise23_1/`
- Extended sweep including Fashion amplitude/high-noise runs and additional hybrid trials.
- Fashion amplitude reaches ~95 %, MNIST amplitude ~95 % (low noise); hybrid and angle sweep between 43–66 %.

### `checkpoints_noise23_4_1/`
- Downsampled 8×8 MNIST experiments (configs `noise_2.4.1`).
- Amplitude accuracy tops out at 85 % (noise_mid); hybrid sits 53–58 %; angle (only low-noise) plateaued at 50 %.

### `checkpoints_noise25_1/`
- Extensive mixed dataset study for configs `noise_2.5.*` (0 vs 1, 0 vs 7).
- Amplitude reaches 69–74 % (low/mid noise), hybrid mostly 45–57 %, angle ≈54–59 % depending on noise.
- Contains 147 checkpoints across 14 runs—primary source for moderate-noise binary comparisons.

### `checkpoints_noise26/`
- Current noise_2.6 hybrid deep-QCNN experiments (digits 6 vs 7, conv depth 3).
- All four noise levels completed; best accuracy so far ≈54 % (noiseless). Angle counterparts still missing.

### `checkpoints_cnn` vs quantum checkpoints
- Use the CNN results as classical baselines when reporting quantum performance under equivalent noise schedules (matching noise suffixes).

### Other directories
- `checkpoints_noise23_2`, `checkpoints_noise25`, etc., are not present—if configs point to them, the corresponding jobs have not been committed yet.

## Mapping Cheat Sheet
- `configs/cnn_baselines` → `checkpoints_cnn/...`
- `configs/full_scale`, `configs/config_noise`, standalone quick configs → subfolders inside `checkpoints/`
- `configs/lightweight` → `checkpoints/lightweight/...` (under `checkpoints/`)
- `configs/noise_light` → intended for lightweight sweeps; reuse `checkpoints/lightweight` (no dedicated runs yet)
- `configs/noise_2*` (2, 2.1, 2.3.x, 2.4.x, 2.5.x, 2.6) → matching `checkpoints_noise*` trees (suffix stripped dots → underscore)
- `configs/profiling` → `checkpoints/profiling/...`
- Any config whose `checkpoint_dir` is absent indicates pending/uncommitted training (see especially `noise_2.2`, `noise_2.3.2`, `noise_2.3.3`, `noise_2.4`, `noise_2.4.2`, `noise_2.4.3`, `noise_2.5.2`, `noise_2.5.3`, `noise_2.5.4`, and the blank `mnist_amplitude_clean.yaml`).

Keep this file updated after new training so the paper-writing phase can target complete datasets and quickly identify gaps.
