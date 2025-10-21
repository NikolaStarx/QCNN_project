# Noise Configuration Training Commands

## Preprocessing

```bash
python scripts/preprocess_downsampled.py
```

## CNN Baselines

```bash
python train_cnn.py --config configs/cnn_baselines/mnist_angle_4x4.yaml --log-interval 4 --eval
python train_cnn.py --config configs/cnn_baselines/mnist_angle_8x8.yaml --log-interval 4 --eval
```

## CNN Noise Simulation

Gaussian perturbations are injected according to the `environment.classical_noise` section and saved under
dedicated directories so prior experiments remain untouched.

```bash
python train_cnn.py --config configs/cnn_noise/mnist_angle_4x4_noise.yaml --log-interval 4 --eval
python train_cnn.py --config configs/cnn_noise/mnist_angle_8x8_noise.yaml --log-interval 4 --eval
```

## Noise Light Suite

```bash
python train_noise.py --config configs/noise_light/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_light/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_light/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_light/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_light/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_light/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_light/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_light/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_light/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_light/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_light/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_light/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_light/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_light/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_light/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_light/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_light/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_light/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2 Suite

```bash
python train_noise.py --config configs/noise_2/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.2 Suite

```bash
python train_noise.py --config configs/noise_2.2/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.2/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.3 Suite

```bash
python train_noise.py --config configs/noise_2.3/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.3.1 Suite

```bash
python train_noise.py --config configs/noise_2.3.1/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.1/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.3.2 Suite

```bash
python train_noise.py --config configs/noise_2.3.2/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.2/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.3.3 Suite

```bash
python train_noise.py --config configs/noise_2.3.3/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.3.3/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.4 Suite

```bash
python train_noise.py --config configs/noise_2.4/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.4.1 Suite

```bash
python train_noise.py --config configs/noise_2.4.1/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.1/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.4.2 Suite

```bash
python train_noise.py --config configs/noise_2.4.2/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.2/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.4.3 Suite

```bash
python train_noise.py --config configs/noise_2.4.3/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.4.3/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.5 Suite

```bash
python train_noise.py --config configs/noise_2.5/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.5.1 Suite

```bash
python train_noise.py --config configs/noise_2.5.1/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.1/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.5.2 Suite

```bash
python train_noise.py --config configs/noise_2.5.2/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.2/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Noise 2.5.3 Suite

```bash
python train_noise.py --config configs/noise_2.5.3/fashion_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/fashion_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/fashion_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/fashion_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/fashion_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/fashion_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/fashion_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/fashion_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/fashion_hybrid_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/mnist_amplitude_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/mnist_amplitude_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/mnist_amplitude_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/mnist_angle_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/mnist_angle_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/mnist_angle_noise_mid.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/mnist_hybrid_noise_high.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/mnist_hybrid_noise_low.yaml --log-interval 4
python train_noise.py --config configs/noise_2.5.3/mnist_hybrid_noise_mid.yaml --log-interval 4
```

## Classical CNN Baselines

```bash
# 4x4 downsampled MNIST angle baseline
python train_cnn.py --config configs/cnn_baselines/mnist_angle_4x4.yaml --log-interval 2 --eval

# 8x8 downsampled MNIST angle baseline
python train_cnn.py --config configs/cnn_baselines/mnist_angle_8x8.yaml --log-interval 2 --eval

# 4x4 downsampled MNIST angle with higher noise
python train_cnn.py --config configs/cnn_baselines/mnist_angle_4x4_noise.yaml --log-interval 2 --eval

# 8x8 downsampled MNIST angle with higher noise
python train_cnn.py --config configs/cnn_baselines/mnist_angle_8x8_noise.yaml --log-interval 2 --eval
```
