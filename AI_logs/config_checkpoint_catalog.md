# Config ↔ Checkpoint Catalog (Structured)

每个条目列出了配置文件的关键超参数及其输出 checkpoint 目录，便于追踪所有实验。

### configs/cnn_baselines
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/cnn_baselines/mnist_angle_4x4.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | none | `checkpoints_cnn/noise23/mnist_angle_4x4` |
| `configs/cnn_baselines/mnist_angle_4x4_noise.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_cnn/noise23/mnist_angle_4x4_noise` |
| `configs/cnn_baselines/mnist_angle_4x4_noise_high.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_cnn/noise23/mnist_angle_4x4_high` |
| `configs/cnn_baselines/mnist_angle_4x4_noise_low.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_cnn/noise23/mnist_angle_4x4_low` |
| `configs/cnn_baselines/mnist_angle_4x4_noise_mid.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_cnn/noise23/mnist_angle_4x4_mid` |
| `configs/cnn_baselines/mnist_angle_8x8.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | none | `checkpoints_cnn/noise24/mnist_angle_8x8` |
| `configs/cnn_baselines/mnist_angle_8x8_noise.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_cnn/noise24/mnist_angle_8x8_noise` |
| `configs/cnn_baselines/mnist_angle_8x8_noise_high.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_cnn/noise24/mnist_angle_8x8_high` |
| `configs/cnn_baselines/mnist_angle_8x8_noise_low.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_cnn/noise24/mnist_angle_8x8_low` |
| `configs/cnn_baselines/mnist_angle_8x8_noise_mid.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_cnn/noise24/mnist_angle_8x8_mid` |

### configs/config_noise
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/config_noise/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 10 | - | - | 30000 / 5000 | 32 | 10 | p1=0.005, p2=0.05 | `checkpoints/fashion_amplitude/noise_high` |
| `configs/config_noise/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 10 | - | - | 30000 / 5000 | 32 | 10 | p1=0.0005, p2=0.005 | `checkpoints/fashion_amplitude/noise_low` |
| `configs/config_noise/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 10 | - | - | 30000 / 5000 | 32 | 10 | p1=0.001, p2=0.01 | `checkpoints/fashion_amplitude/noise_mid` |
| `configs/config_noise/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 16 | - | 12000 / 3000 | 8 | 10 | p1=0.005, p2=0.05 | `checkpoints/fashion_angle/noise_high` |
| `configs/config_noise/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 16 | - | 12000 / 3000 | 8 | 10 | p1=0.0005, p2=0.005 | `checkpoints/fashion_angle/noise_low` |
| `configs/config_noise/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 16 | - | 12000 / 3000 | 8 | 10 | p1=0.001, p2=0.01 | `checkpoints/fashion_angle/noise_mid` |
| `configs/config_noise/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | p1=0.005, p2=0.05 | `checkpoints/fashion_hybrid/noise_high` |
| `configs/config_noise/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | p1=0.0005, p2=0.005 | `checkpoints/fashion_hybrid/noise_low` |
| `configs/config_noise/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | p1=0.001, p2=0.01 | `checkpoints/fashion_hybrid/noise_mid` |
| `configs/config_noise/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 10 | - | - | 30000 / 5000 | 32 | 10 | p1=0.005, p2=0.05 | `checkpoints/mnist_amplitude/noise_high` |
| `configs/config_noise/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 10 | - | - | 30000 / 5000 | 32 | 10 | p1=0.0005, p2=0.005 | `checkpoints/mnist_amplitude/noise_low` |
| `configs/config_noise/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 10 | - | - | 30000 / 5000 | 32 | 10 | p1=0.001, p2=0.01 | `checkpoints/mnist_amplitude/noise_mid` |
| `configs/config_noise/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | - | 12000 / 3000 | 8 | 10 | p1=0.005, p2=0.05 | `checkpoints/mnist_angle/noise_high` |
| `configs/config_noise/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | - | 12000 / 3000 | 8 | 10 | p1=0.0005, p2=0.005 | `checkpoints/mnist_angle/noise_low` |
| `configs/config_noise/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | - | 12000 / 3000 | 8 | 10 | p1=0.001, p2=0.01 | `checkpoints/mnist_angle/noise_mid` |
| `configs/config_noise/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | p1=0.005, p2=0.05 | `checkpoints/mnist_hybrid/noise_high` |
| `configs/config_noise/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | p1=0.0005, p2=0.005 | `checkpoints/mnist_hybrid/noise_low` |
| `configs/config_noise/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | p1=0.001, p2=0.01 | `checkpoints/mnist_hybrid/noise_mid` |

### configs/full_scale
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/full_scale/fashion_amplitude_full.yaml` | FashionMNIST | amplitude | 10 | - | - | 30000 / 5000 | 32 | 10 | none | `checkpoints/fashion_amplitude` |
| `configs/full_scale/fashion_angle_full.yaml` | FashionMNIST | angle | 16 | 16 | - | 12000 / 3000 | 8 | 10 | none | `checkpoints/fashion_angle` |
| `configs/full_scale/fashion_hybrid_full.yaml` | FashionMNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | none | `checkpoints/fashion_hybrid` |
| `configs/full_scale/mnist_amplitude_full.yaml` | MNIST | amplitude | 10 | - | - | 30000 / 5000 | 32 | 10 | none | `checkpoints/mnist_amplitude` |
| `configs/full_scale/mnist_angle_full.yaml` | MNIST | angle | 16 | 16 | - | 12000 / 3000 | 8 | 10 | none | `checkpoints/mnist_angle` |
| `configs/full_scale/mnist_hybrid_full.yaml` | MNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | none | `checkpoints/mnist_hybrid` |

### configs/lightweight
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/lightweight/fashion_amplitude.yaml` | FashionMNIST | amplitude | 10 | - | - | 64 / 32 | 4 | 3 | none | `checkpoints/lightweight/fashion_amplitude` |
| `configs/lightweight/fashion_angle.yaml` | FashionMNIST | angle | 16 | 16 | - | 128 / 64 | 4 | 3 | none | `checkpoints/lightweight/fashion_angle` |
| `configs/lightweight/fashion_hybrid.yaml` | FashionMNIST | hybrid | 4 | 16 | - | 128 / 64 | 8 | 3 | none | `checkpoints/lightweight/fashion_hybrid` |
| `configs/lightweight/mnist_amplitude.yaml` | MNIST | amplitude | 10 | - | - | 64 / 32 | 4 | 3 | none | `checkpoints/lightweight/mnist_amplitude` |
| `configs/lightweight/mnist_angle.yaml` | MNIST | angle | 16 | 16 | - | 128 / 64 | 4 | 3 | none | `checkpoints/lightweight/mnist_angle` |
| `configs/lightweight/mnist_hybrid.yaml` | MNIST | hybrid | 4 | 16 | - | 128 / 64 | 8 | 3 | none | `checkpoints/lightweight/mnist_hybrid` |

### configs/noise_2
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 10 | - | 0,1 | 4000 / 800 | 12 | 10 | p1=0.0015, p2=0.015 | `checkpoints_noise2/fashion_amplitude/noise_high` |
| `configs/noise_2/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 10 | - | 0,1 | 4000 / 800 | 12 | 10 | p1=0.0002, p2=0.002 | `checkpoints_noise2/fashion_amplitude/noise_low` |
| `configs/noise_2/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 10 | - | 0,1 | 4000 / 800 | 12 | 10 | p1=0.0006, p2=0.006 | `checkpoints_noise2/fashion_amplitude/noise_mid` |
| `configs/noise_2/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 16 | 0,1 | 3000 / 600 | 6 | 10 | p1=0.0015, p2=0.015 | `checkpoints_noise2/fashion_angle/noise_high` |
| `configs/noise_2/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 16 | 0,1 | 3000 / 600 | 6 | 10 | p1=0.0002, p2=0.002 | `checkpoints_noise2/fashion_angle/noise_low` |
| `configs/noise_2/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 16 | 0,1 | 3000 / 600 | 6 | 10 | p1=0.0006, p2=0.006 | `checkpoints_noise2/fashion_angle/noise_mid` |
| `configs/noise_2/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 12 | 144 | 0,1 | 3000 / 600 | 8 | 10 | p1=0.0015, p2=0.015 | `checkpoints_noise2/fashion_hybrid/noise_high` |
| `configs/noise_2/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 12 | 144 | 0,1 | 3000 / 600 | 8 | 10 | p1=0.0002, p2=0.002 | `checkpoints_noise2/fashion_hybrid/noise_low` |
| `configs/noise_2/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 12 | 144 | 0,1 | 3000 / 600 | 8 | 10 | p1=0.0006, p2=0.006 | `checkpoints_noise2/fashion_hybrid/noise_mid` |
| `configs/noise_2/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 10 | - | 0,1 | 4000 / 800 | 12 | 10 | p1=0.0015, p2=0.015 | `checkpoints_noise2/mnist_amplitude/noise_high` |
| `configs/noise_2/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 10 | - | 0,1 | 4000 / 800 | 12 | 10 | p1=0.0002, p2=0.002 | `checkpoints_noise2/mnist_amplitude/noise_low` |
| `configs/noise_2/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 10 | - | 0,1 | 4000 / 800 | 12 | 10 | p1=0.0006, p2=0.006 | `checkpoints_noise2/mnist_amplitude/noise_mid` |
| `configs/noise_2/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | 0,1 | 3000 / 600 | 6 | 10 | p1=0.0015, p2=0.015 | `checkpoints_noise2/mnist_angle/noise_high` |
| `configs/noise_2/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | 0,1 | 3000 / 600 | 6 | 10 | p1=0.0002, p2=0.002 | `checkpoints_noise2/mnist_angle/noise_low` |
| `configs/noise_2/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | 0,1 | 3000 / 600 | 6 | 10 | p1=0.0006, p2=0.006 | `checkpoints_noise2/mnist_angle/noise_mid` |
| `configs/noise_2/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 12 | 144 | 0,1 | 3000 / 600 | 8 | 10 | p1=0.0015, p2=0.015 | `checkpoints_noise2/mnist_hybrid/noise_high` |
| `configs/noise_2/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 12 | 144 | 0,1 | 3000 / 600 | 8 | 10 | p1=0.0002, p2=0.002 | `checkpoints_noise2/mnist_hybrid/noise_low` |
| `configs/noise_2/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 12 | 144 | 0,1 | 3000 / 600 | 8 | 10 | p1=0.0006, p2=0.006 | `checkpoints_noise2/mnist_hybrid/noise_mid` |

### configs/noise_2.1
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.1/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 10 | - | 0,1 | 256 / 64 | 16 | 4 | p1=0.0006, p2=0.006 | `checkpoints_noise21/fashion_amplitude/noise_high` |
| `configs/noise_2.1/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 10 | - | 0,1 | 256 / 64 | 16 | 4 | p1=5e-05, p2=0.0005 | `checkpoints_noise21/fashion_amplitude/noise_low` |
| `configs/noise_2.1/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 10 | - | 0,1 | 256 / 64 | 16 | 4 | p1=0.0002, p2=0.002 | `checkpoints_noise21/fashion_amplitude/noise_mid` |
| `configs/noise_2.1/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 16 | 0,1 | 192 / 48 | 8 | 4 | p1=0.0006, p2=0.006 | `checkpoints_noise21/fashion_angle/noise_high` |
| `configs/noise_2.1/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 16 | 0,1 | 192 / 48 | 8 | 4 | p1=5e-05, p2=0.0005 | `checkpoints_noise21/fashion_angle/noise_low` |
| `configs/noise_2.1/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 16 | 0,1 | 192 / 48 | 8 | 4 | p1=0.0002, p2=0.002 | `checkpoints_noise21/fashion_angle/noise_mid` |
| `configs/noise_2.1/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 12 | 144 | 0,1 | 192 / 48 | 8 | 4 | p1=0.0006, p2=0.006 | `checkpoints_noise21/fashion_hybrid/noise_high` |
| `configs/noise_2.1/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 12 | 144 | 0,1 | 192 / 48 | 8 | 4 | p1=5e-05, p2=0.0005 | `checkpoints_noise21/fashion_hybrid/noise_low` |
| `configs/noise_2.1/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 12 | 144 | 0,1 | 192 / 48 | 8 | 4 | p1=0.0002, p2=0.002 | `checkpoints_noise21/fashion_hybrid/noise_mid` |
| `configs/noise_2.1/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 10 | - | 0,1 | 256 / 64 | 16 | 4 | p1=0.0006, p2=0.006 | `checkpoints_noise21/mnist_amplitude/noise_high` |
| `configs/noise_2.1/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 10 | - | 0,1 | 256 / 64 | 16 | 4 | p1=5e-05, p2=0.0005 | `checkpoints_noise21/mnist_amplitude/noise_low` |
| `configs/noise_2.1/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 10 | - | 0,1 | 256 / 64 | 16 | 4 | p1=0.0002, p2=0.002 | `checkpoints_noise21/mnist_amplitude/noise_mid` |
| `configs/noise_2.1/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | 0,1 | 192 / 48 | 8 | 4 | p1=0.0006, p2=0.006 | `checkpoints_noise21/mnist_angle/noise_high` |
| `configs/noise_2.1/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | 0,1 | 192 / 48 | 8 | 4 | p1=5e-05, p2=0.0005 | `checkpoints_noise21/mnist_angle/noise_low` |
| `configs/noise_2.1/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | 0,1 | 192 / 48 | 8 | 4 | p1=0.0002, p2=0.002 | `checkpoints_noise21/mnist_angle/noise_mid` |
| `configs/noise_2.1/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 12 | 144 | 0,1 | 192 / 48 | 8 | 4 | p1=0.0006, p2=0.006 | `checkpoints_noise21/mnist_hybrid/noise_high` |
| `configs/noise_2.1/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 12 | 144 | 0,1 | 192 / 48 | 8 | 4 | p1=5e-05, p2=0.0005 | `checkpoints_noise21/mnist_hybrid/noise_low` |
| `configs/noise_2.1/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 12 | 144 | 0,1 | 192 / 48 | 8 | 4 | p1=0.0002, p2=0.002 | `checkpoints_noise21/mnist_hybrid/noise_mid` |

### configs/noise_2.2
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.2/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 4096 / 1024 | 64 | 10 | p1=0.0007, p2=0.007 | `checkpoints_noise22/fashion_amplitude/noise_high` |
| `configs/noise_2.2/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 4096 / 1024 | 64 | 10 | p1=0.0001, p2=0.001 | `checkpoints_noise22/fashion_amplitude/noise_low` |
| `configs/noise_2.2/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 4096 / 1024 | 64 | 10 | p1=0.0003, p2=0.003 | `checkpoints_noise22/fashion_amplitude/noise_mid` |
| `configs/noise_2.2/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 4096 / 1024 | 64 | 10 | p1=0.0007, p2=0.007 | `checkpoints_noise22/fashion_angle/noise_high` |
| `configs/noise_2.2/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 4096 / 1024 | 64 | 10 | p1=0.0001, p2=0.001 | `checkpoints_noise22/fashion_angle/noise_low` |
| `configs/noise_2.2/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 4096 / 1024 | 64 | 10 | p1=0.0003, p2=0.003 | `checkpoints_noise22/fashion_angle/noise_mid` |
| `configs/noise_2.2/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 4096 / 1024 | 64 | 10 | p1=0.0007, p2=0.007 | `checkpoints_noise22/fashion_hybrid/noise_high` |
| `configs/noise_2.2/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 4096 / 1024 | 64 | 10 | p1=0.0001, p2=0.001 | `checkpoints_noise22/fashion_hybrid/noise_low` |
| `configs/noise_2.2/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 4096 / 1024 | 64 | 10 | p1=0.0003, p2=0.003 | `checkpoints_noise22/fashion_hybrid/noise_mid` |
| `configs/noise_2.2/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 4 | - | 6,7 | 4096 / 1024 | 64 | 10 | p1=0.0007, p2=0.007 | `checkpoints_noise22/mnist_amplitude/noise_high` |
| `configs/noise_2.2/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 4 | - | 6,7 | 4096 / 1024 | 64 | 10 | p1=0.0001, p2=0.001 | `checkpoints_noise22/mnist_amplitude/noise_low` |
| `configs/noise_2.2/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 4 | - | 6,7 | 4096 / 1024 | 64 | 10 | p1=0.0003, p2=0.003 | `checkpoints_noise22/mnist_amplitude/noise_mid` |
| `configs/noise_2.2/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | 6,7 | 4096 / 1024 | 64 | 10 | p1=0.0007, p2=0.007 | `checkpoints_noise22/mnist_angle/noise_high` |
| `configs/noise_2.2/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | 6,7 | 4096 / 1024 | 64 | 10 | p1=0.0001, p2=0.001 | `checkpoints_noise22/mnist_angle/noise_low` |
| `configs/noise_2.2/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | 6,7 | 4096 / 1024 | 64 | 10 | p1=0.0003, p2=0.003 | `checkpoints_noise22/mnist_angle/noise_mid` |
| `configs/noise_2.2/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 4096 / 1024 | 64 | 10 | p1=0.0007, p2=0.007 | `checkpoints_noise22/mnist_hybrid/noise_high` |
| `configs/noise_2.2/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 4096 / 1024 | 64 | 10 | p1=0.0001, p2=0.001 | `checkpoints_noise22/mnist_hybrid/noise_low` |
| `configs/noise_2.2/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 4096 / 1024 | 64 | 10 | p1=0.0003, p2=0.003 | `checkpoints_noise22/mnist_hybrid/noise_mid` |

### configs/noise_2.3
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.3/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23/fashion_amplitude/noise_high` |
| `configs/noise_2.3/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23/fashion_amplitude/noise_low` |
| `configs/noise_2.3/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23/fashion_amplitude/noise_mid` |
| `configs/noise_2.3/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23/fashion_angle/noise_high` |
| `configs/noise_2.3/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23/fashion_angle/noise_low` |
| `configs/noise_2.3/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23/fashion_angle/noise_mid` |
| `configs/noise_2.3/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23/fashion_hybrid/noise_high` |
| `configs/noise_2.3/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23/fashion_hybrid/noise_low` |
| `configs/noise_2.3/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23/fashion_hybrid/noise_mid` |
| `configs/noise_2.3/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23/mnist_amplitude/noise_high` |
| `configs/noise_2.3/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23/mnist_amplitude/noise_low` |
| `configs/noise_2.3/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23/mnist_amplitude/noise_mid` |
| `configs/noise_2.3/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23/mnist_angle/noise_high` |
| `configs/noise_2.3/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23/mnist_angle/noise_low` |
| `configs/noise_2.3/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23/mnist_angle/noise_mid` |
| `configs/noise_2.3/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23/mnist_hybrid/noise_high` |
| `configs/noise_2.3/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23/mnist_hybrid/noise_low` |
| `configs/noise_2.3/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23/mnist_hybrid/noise_mid` |

### configs/noise_2.3.1
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.3.1/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_1/fashion_amplitude/noise_high` |
| `configs/noise_2.3.1/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_1/fashion_amplitude/noise_low` |
| `configs/noise_2.3.1/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_1/fashion_amplitude/noise_mid` |
| `configs/noise_2.3.1/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_1/fashion_angle/noise_high` |
| `configs/noise_2.3.1/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_1/fashion_angle/noise_low` |
| `configs/noise_2.3.1/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_1/fashion_angle/noise_mid` |
| `configs/noise_2.3.1/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_1/fashion_hybrid/noise_high` |
| `configs/noise_2.3.1/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_1/fashion_hybrid/noise_low` |
| `configs/noise_2.3.1/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_1/fashion_hybrid/noise_mid` |
| `configs/noise_2.3.1/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_1/mnist_amplitude/noise_high` |
| `configs/noise_2.3.1/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_1/mnist_amplitude/noise_low` |
| `configs/noise_2.3.1/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_1/mnist_amplitude/noise_mid` |
| `configs/noise_2.3.1/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_1/mnist_angle/noise_high` |
| `configs/noise_2.3.1/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_1/mnist_angle/noise_low` |
| `configs/noise_2.3.1/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_1/mnist_angle/noise_mid` |
| `configs/noise_2.3.1/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_1/mnist_hybrid/noise_high` |
| `configs/noise_2.3.1/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_1/mnist_hybrid/noise_low` |
| `configs/noise_2.3.1/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_1/mnist_hybrid/noise_mid` |

### configs/noise_2.3.2
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.3.2/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_2/fashion_amplitude/noise_high` |
| `configs/noise_2.3.2/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_2/fashion_amplitude/noise_low` |
| `configs/noise_2.3.2/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_2/fashion_amplitude/noise_mid` |
| `configs/noise_2.3.2/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_2/fashion_angle/noise_high` |
| `configs/noise_2.3.2/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_2/fashion_angle/noise_low` |
| `configs/noise_2.3.2/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_2/fashion_angle/noise_mid` |
| `configs/noise_2.3.2/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_2/fashion_hybrid/noise_high` |
| `configs/noise_2.3.2/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_2/fashion_hybrid/noise_low` |
| `configs/noise_2.3.2/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_2/fashion_hybrid/noise_mid` |
| `configs/noise_2.3.2/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_2/mnist_amplitude/noise_high` |
| `configs/noise_2.3.2/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_2/mnist_amplitude/noise_low` |
| `configs/noise_2.3.2/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_2/mnist_amplitude/noise_mid` |
| `configs/noise_2.3.2/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_2/mnist_angle/noise_high` |
| `configs/noise_2.3.2/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_2/mnist_angle/noise_low` |
| `configs/noise_2.3.2/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_2/mnist_angle/noise_mid` |
| `configs/noise_2.3.2/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_2/mnist_hybrid/noise_high` |
| `configs/noise_2.3.2/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_2/mnist_hybrid/noise_low` |
| `configs/noise_2.3.2/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_2/mnist_hybrid/noise_mid` |

### configs/noise_2.3.3
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.3.3/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_3/fashion_amplitude/noise_high` |
| `configs/noise_2.3.3/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_3/fashion_amplitude/noise_low` |
| `configs/noise_2.3.3/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 4 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_3/fashion_amplitude/noise_mid` |
| `configs/noise_2.3.3/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_3/fashion_angle/noise_high` |
| `configs/noise_2.3.3/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_3/fashion_angle/noise_low` |
| `configs/noise_2.3.3/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_3/fashion_angle/noise_mid` |
| `configs/noise_2.3.3/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_3/fashion_hybrid/noise_high` |
| `configs/noise_2.3.3/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_3/fashion_hybrid/noise_low` |
| `configs/noise_2.3.3/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 16 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_3/fashion_hybrid/noise_mid` |
| `configs/noise_2.3.3/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_3/mnist_amplitude/noise_high` |
| `configs/noise_2.3.3/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_3/mnist_amplitude/noise_low` |
| `configs/noise_2.3.3/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 4 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_3/mnist_amplitude/noise_mid` |
| `configs/noise_2.3.3/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_3/mnist_angle/noise_high` |
| `configs/noise_2.3.3/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_3/mnist_angle/noise_low` |
| `configs/noise_2.3.3/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_3/mnist_angle/noise_mid` |
| `configs/noise_2.3.3/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_3/mnist_hybrid/noise_high` |
| `configs/noise_2.3.3/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_3/mnist_hybrid/noise_low` |
| `configs/noise_2.3.3/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_3/mnist_hybrid/noise_mid` |

### configs/noise_2.4
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.4/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4/fashion_amplitude/noise_high` |
| `configs/noise_2.4/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4/fashion_amplitude/noise_low` |
| `configs/noise_2.4/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4/fashion_amplitude/noise_mid` |
| `configs/noise_2.4/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4/fashion_angle/noise_high` |
| `configs/noise_2.4/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4/fashion_angle/noise_low` |
| `configs/noise_2.4/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4/fashion_angle/noise_mid` |
| `configs/noise_2.4/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4/fashion_hybrid/noise_high` |
| `configs/noise_2.4/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4/fashion_hybrid/noise_low` |
| `configs/noise_2.4/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4/fashion_hybrid/noise_mid` |
| `configs/noise_2.4/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4/mnist_amplitude/noise_high` |
| `configs/noise_2.4/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4/mnist_amplitude/noise_low` |
| `configs/noise_2.4/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4/mnist_amplitude/noise_mid` |
| `configs/noise_2.4/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4/mnist_angle/noise_high` |
| `configs/noise_2.4/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4/mnist_angle/noise_low` |
| `configs/noise_2.4/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4/mnist_angle/noise_mid` |
| `configs/noise_2.4/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4/mnist_hybrid/noise_high` |
| `configs/noise_2.4/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4/mnist_hybrid/noise_low` |
| `configs/noise_2.4/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4/mnist_hybrid/noise_mid` |

### configs/noise_2.4.1
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.4.1/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_1/fashion_amplitude/noise_high` |
| `configs/noise_2.4.1/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_1/fashion_amplitude/noise_low` |
| `configs/noise_2.4.1/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_1/fashion_amplitude/noise_mid` |
| `configs/noise_2.4.1/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_1/fashion_angle/noise_high` |
| `configs/noise_2.4.1/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_1/fashion_angle/noise_low` |
| `configs/noise_2.4.1/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_1/fashion_angle/noise_mid` |
| `configs/noise_2.4.1/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_1/fashion_hybrid/noise_high` |
| `configs/noise_2.4.1/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_1/fashion_hybrid/noise_low` |
| `configs/noise_2.4.1/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_1/fashion_hybrid/noise_mid` |
| `configs/noise_2.4.1/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_1/mnist_amplitude/noise_high` |
| `configs/noise_2.4.1/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_1/mnist_amplitude/noise_low` |
| `configs/noise_2.4.1/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_1/mnist_amplitude/noise_mid` |
| `configs/noise_2.4.1/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_1/mnist_angle/noise_high` |
| `configs/noise_2.4.1/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_1/mnist_angle/noise_low` |
| `configs/noise_2.4.1/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_1/mnist_angle/noise_mid` |
| `configs/noise_2.4.1/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_1/mnist_hybrid/noise_high` |
| `configs/noise_2.4.1/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_1/mnist_hybrid/noise_low` |
| `configs/noise_2.4.1/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_1/mnist_hybrid/noise_mid` |

### configs/noise_2.4.2
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.4.2/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_2/fashion_amplitude/noise_high` |
| `configs/noise_2.4.2/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_2/fashion_amplitude/noise_low` |
| `configs/noise_2.4.2/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_2/fashion_amplitude/noise_mid` |
| `configs/noise_2.4.2/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_2/fashion_angle/noise_high` |
| `configs/noise_2.4.2/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_2/fashion_angle/noise_low` |
| `configs/noise_2.4.2/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_2/fashion_angle/noise_mid` |
| `configs/noise_2.4.2/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_2/fashion_hybrid/noise_high` |
| `configs/noise_2.4.2/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_2/fashion_hybrid/noise_low` |
| `configs/noise_2.4.2/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_2/fashion_hybrid/noise_mid` |
| `configs/noise_2.4.2/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_2/mnist_amplitude/noise_high` |
| `configs/noise_2.4.2/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_2/mnist_amplitude/noise_low` |
| `configs/noise_2.4.2/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_2/mnist_amplitude/noise_mid` |
| `configs/noise_2.4.2/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_2/mnist_angle/noise_high` |
| `configs/noise_2.4.2/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_2/mnist_angle/noise_low` |
| `configs/noise_2.4.2/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_2/mnist_angle/noise_mid` |
| `configs/noise_2.4.2/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_2/mnist_hybrid/noise_high` |
| `configs/noise_2.4.2/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_2/mnist_hybrid/noise_low` |
| `configs/noise_2.4.2/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_2/mnist_hybrid/noise_mid` |

### configs/noise_2.4.3
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.4.3/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_3/fashion_amplitude/noise_high` |
| `configs/noise_2.4.3/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_3/fashion_amplitude/noise_low` |
| `configs/noise_2.4.3/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 6 | - | 4,5 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_3/fashion_amplitude/noise_mid` |
| `configs/noise_2.4.3/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_3/fashion_angle/noise_high` |
| `configs/noise_2.4.3/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_3/fashion_angle/noise_low` |
| `configs/noise_2.4.3/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_3/fashion_angle/noise_mid` |
| `configs/noise_2.4.3/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_3/fashion_hybrid/noise_high` |
| `configs/noise_2.4.3/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_3/fashion_hybrid/noise_low` |
| `configs/noise_2.4.3/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 64 | 4,5 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_3/fashion_hybrid/noise_mid` |
| `configs/noise_2.4.3/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_3/mnist_amplitude/noise_high` |
| `configs/noise_2.4.3/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_3/mnist_amplitude/noise_low` |
| `configs/noise_2.4.3/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 6 | - | 6,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_3/mnist_amplitude/noise_mid` |
| `configs/noise_2.4.3/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_3/mnist_angle/noise_high` |
| `configs/noise_2.4.3/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_3/mnist_angle/noise_low` |
| `configs/noise_2.4.3/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_3/mnist_angle/noise_mid` |
| `configs/noise_2.4.3/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise23_4_3/mnist_hybrid/noise_high` |
| `configs/noise_2.4.3/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise23_4_3/mnist_hybrid/noise_low` |
| `configs/noise_2.4.3/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 64 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise23_4_3/mnist_hybrid/noise_mid` |

### configs/noise_2.5
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.5/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25/fashion_amplitude/noise_high` |
| `configs/noise_2.5/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25/fashion_amplitude/noise_low` |
| `configs/noise_2.5/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25/fashion_amplitude/noise_mid` |
| `configs/noise_2.5/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25/fashion_angle/noise_high` |
| `configs/noise_2.5/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25/fashion_angle/noise_low` |
| `configs/noise_2.5/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25/fashion_angle/noise_mid` |
| `configs/noise_2.5/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25/fashion_hybrid/noise_high` |
| `configs/noise_2.5/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25/fashion_hybrid/noise_low` |
| `configs/noise_2.5/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25/fashion_hybrid/noise_mid` |
| `configs/noise_2.5/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25/mnist_amplitude/noise_high` |
| `configs/noise_2.5/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25/mnist_amplitude/noise_low` |
| `configs/noise_2.5/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25/mnist_amplitude/noise_mid` |
| `configs/noise_2.5/mnist_angle_noise_high.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25/mnist_angle/noise_high` |
| `configs/noise_2.5/mnist_angle_noise_low.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25/mnist_angle/noise_low` |
| `configs/noise_2.5/mnist_angle_noise_mid.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25/mnist_angle/noise_mid` |
| `configs/noise_2.5/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25/mnist_hybrid/noise_high` |
| `configs/noise_2.5/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25/mnist_hybrid/noise_low` |
| `configs/noise_2.5/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25/mnist_hybrid/noise_mid` |

### configs/noise_2.5.1
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.5.1/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_1/fashion_amplitude/noise_high` |
| `configs/noise_2.5.1/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_1/fashion_amplitude/noise_low` |
| `configs/noise_2.5.1/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_1/fashion_amplitude/noise_mid` |
| `configs/noise_2.5.1/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_1/fashion_angle/noise_high` |
| `configs/noise_2.5.1/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_1/fashion_angle/noise_low` |
| `configs/noise_2.5.1/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_1/fashion_angle/noise_mid` |
| `configs/noise_2.5.1/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_1/fashion_hybrid/noise_high` |
| `configs/noise_2.5.1/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_1/fashion_hybrid/noise_low` |
| `configs/noise_2.5.1/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_1/fashion_hybrid/noise_mid` |
| `configs/noise_2.5.1/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_1/mnist_amplitude/noise_high` |
| `configs/noise_2.5.1/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_1/mnist_amplitude/noise_low` |
| `configs/noise_2.5.1/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_1/mnist_amplitude/noise_mid` |
| `configs/noise_2.5.1/mnist_angle_noise_high.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_1/mnist_angle/noise_high` |
| `configs/noise_2.5.1/mnist_angle_noise_low.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_1/mnist_angle/noise_low` |
| `configs/noise_2.5.1/mnist_angle_noise_mid.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_1/mnist_angle/noise_mid` |
| `configs/noise_2.5.1/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_1/mnist_hybrid/noise_high` |
| `configs/noise_2.5.1/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_1/mnist_hybrid/noise_low` |
| `configs/noise_2.5.1/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_1/mnist_hybrid/noise_mid` |

### configs/noise_2.5.2
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.5.2/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_2/fashion_amplitude/noise_high` |
| `configs/noise_2.5.2/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_2/fashion_amplitude/noise_low` |
| `configs/noise_2.5.2/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_2/fashion_amplitude/noise_mid` |
| `configs/noise_2.5.2/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_2/fashion_angle/noise_high` |
| `configs/noise_2.5.2/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_2/fashion_angle/noise_low` |
| `configs/noise_2.5.2/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_2/fashion_angle/noise_mid` |
| `configs/noise_2.5.2/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_2/fashion_hybrid/noise_high` |
| `configs/noise_2.5.2/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_2/fashion_hybrid/noise_low` |
| `configs/noise_2.5.2/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_2/fashion_hybrid/noise_mid` |
| `configs/noise_2.5.2/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_2/mnist_amplitude/noise_high` |
| `configs/noise_2.5.2/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_2/mnist_amplitude/noise_low` |
| `configs/noise_2.5.2/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_2/mnist_amplitude/noise_mid` |
| `configs/noise_2.5.2/mnist_angle_noise_high.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_2/mnist_angle/noise_high` |
| `configs/noise_2.5.2/mnist_angle_noise_low.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_2/mnist_angle/noise_low` |
| `configs/noise_2.5.2/mnist_angle_noise_mid.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_2/mnist_angle/noise_mid` |
| `configs/noise_2.5.2/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_2/mnist_hybrid/noise_high` |
| `configs/noise_2.5.2/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_2/mnist_hybrid/noise_low` |
| `configs/noise_2.5.2/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_2/mnist_hybrid/noise_mid` |

### configs/noise_2.5.3
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.5.3/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_3/fashion_amplitude/noise_high` |
| `configs/noise_2.5.3/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_3/fashion_amplitude/noise_low` |
| `configs/noise_2.5.3/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 6 | - | 0,7 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_3/fashion_amplitude/noise_mid` |
| `configs/noise_2.5.3/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_3/fashion_angle/noise_high` |
| `configs/noise_2.5.3/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_3/fashion_angle/noise_low` |
| `configs/noise_2.5.3/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 64 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_3/fashion_angle/noise_mid` |
| `configs/noise_2.5.3/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_3/fashion_hybrid/noise_high` |
| `configs/noise_2.5.3/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_3/fashion_hybrid/noise_low` |
| `configs/noise_2.5.3/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_3/fashion_hybrid/noise_mid` |
| `configs/noise_2.5.3/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_3/mnist_amplitude/noise_high` |
| `configs/noise_2.5.3/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_3/mnist_amplitude/noise_low` |
| `configs/noise_2.5.3/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 6 | - | 0,1 | 256 / 64 | 16 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_3/mnist_amplitude/noise_mid` |
| `configs/noise_2.5.3/mnist_angle_noise_high.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_3/mnist_angle/noise_high` |
| `configs/noise_2.5.3/mnist_angle_noise_low.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_3/mnist_angle/noise_low` |
| `configs/noise_2.5.3/mnist_angle_noise_mid.yaml` | MNIST | angle | 64 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_3/mnist_angle/noise_mid` |
| `configs/noise_2.5.3/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_3/mnist_hybrid/noise_high` |
| `configs/noise_2.5.3/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_3/mnist_hybrid/noise_low` |
| `configs/noise_2.5.3/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_3/mnist_hybrid/noise_mid` |

### configs/noise_2.5.4
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.5.4/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 512 / 128 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_4/fashion_hybrid/noise_high` |
| `configs/noise_2.5.4/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 512 / 128 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_4/fashion_hybrid/noise_low` |
| `configs/noise_2.5.4/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 8 | 64 | 0,7 | 512 / 128 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_4/fashion_hybrid/noise_mid` |
| `configs/noise_2.5.4/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 512 / 128 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise25_4/mnist_hybrid/noise_high` |
| `configs/noise_2.5.4/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 512 / 128 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise25_4/mnist_hybrid/noise_low` |
| `configs/noise_2.5.4/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 64 | 0,1 | 512 / 128 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise25_4/mnist_hybrid/noise_mid` |

### configs/noise_2.6
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.6/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise26/mnist_angle/noise_high` |
| `configs/noise_2.6/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise26/mnist_angle/noise_low` |
| `configs/noise_2.6/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise26/mnist_angle/noise_mid` |
| `configs/noise_2.6/mnist_angle_noise_none.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | none | `checkpoints_noise26/mnist_angle/noise_none` |
| `configs/noise_2.6/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise26/mnist_hybrid/noise_high` |
| `configs/noise_2.6/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise26/mnist_hybrid/noise_low` |
| `configs/noise_2.6/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise26/mnist_hybrid/noise_mid` |
| `configs/noise_2.6/mnist_hybrid_noise_none.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | none | `checkpoints_noise26/mnist_hybrid/noise_none` |

### configs/noise_2.6.3
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_2.6.3/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise263/mnist_angle/noise_high` |
| `configs/noise_2.6.3/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise263/mnist_angle/noise_low` |
| `configs/noise_2.6.3/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise263/mnist_angle/noise_mid` |
| `configs/noise_2.6.3/mnist_angle_noise_none.yaml` | MNIST | angle | 16 | 16 | 6,7 | 256 / 64 | 8 | 10 | none | `checkpoints_noise263/mnist_angle/noise_none` |
| `configs/noise_2.6.3/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.0004, p2=0.004 | `checkpoints_noise263/mnist_hybrid/noise_high` |
| `configs/noise_2.6.3/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=5e-05, p2=0.0005 | `checkpoints_noise263/mnist_hybrid/noise_low` |
| `configs/noise_2.6.3/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | p1=0.00015, p2=0.0015 | `checkpoints_noise263/mnist_hybrid/noise_mid` |
| `configs/noise_2.6.3/mnist_hybrid_noise_none.yaml` | MNIST | hybrid | 8 | 16 | 6,7 | 256 / 64 | 8 | 10 | none | `checkpoints_noise263/mnist_hybrid/noise_none` |

### configs/noise_light
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/noise_light/fashion_amplitude_noise_high.yaml` | FashionMNIST | amplitude | 10 | - | - | 8000 / 1600 | 16 | 6 | p1=0.0025, p2=0.025 | `checkpoints_light/fashion_amplitude/noise_high` |
| `configs/noise_light/fashion_amplitude_noise_low.yaml` | FashionMNIST | amplitude | 10 | - | - | 8000 / 1600 | 16 | 6 | p1=0.0003, p2=0.003 | `checkpoints_light/fashion_amplitude/noise_low` |
| `configs/noise_light/fashion_amplitude_noise_mid.yaml` | FashionMNIST | amplitude | 10 | - | - | 8000 / 1600 | 16 | 6 | p1=0.0008, p2=0.008 | `checkpoints_light/fashion_amplitude/noise_mid` |
| `configs/noise_light/fashion_angle_noise_high.yaml` | FashionMNIST | angle | 16 | 16 | - | 6000 / 1200 | 8 | 6 | p1=0.0025, p2=0.025 | `checkpoints_light/fashion_angle/noise_high` |
| `configs/noise_light/fashion_angle_noise_low.yaml` | FashionMNIST | angle | 16 | 16 | - | 6000 / 1200 | 8 | 6 | p1=0.0003, p2=0.003 | `checkpoints_light/fashion_angle/noise_low` |
| `configs/noise_light/fashion_angle_noise_mid.yaml` | FashionMNIST | angle | 16 | 16 | - | 6000 / 1200 | 8 | 6 | p1=0.0008, p2=0.008 | `checkpoints_light/fashion_angle/noise_mid` |
| `configs/noise_light/fashion_hybrid_noise_high.yaml` | FashionMNIST | hybrid | 12 | 144 | - | 6000 / 1200 | 12 | 6 | p1=0.0025, p2=0.025 | `checkpoints_light/fashion_hybrid/noise_high` |
| `configs/noise_light/fashion_hybrid_noise_low.yaml` | FashionMNIST | hybrid | 12 | 144 | - | 6000 / 1200 | 12 | 6 | p1=0.0003, p2=0.003 | `checkpoints_light/fashion_hybrid/noise_low` |
| `configs/noise_light/fashion_hybrid_noise_mid.yaml` | FashionMNIST | hybrid | 12 | 144 | - | 6000 / 1200 | 12 | 6 | p1=0.0008, p2=0.008 | `checkpoints_light/fashion_hybrid/noise_mid` |
| `configs/noise_light/mnist_amplitude_noise_high.yaml` | MNIST | amplitude | 10 | - | - | 8000 / 1600 | 16 | 6 | p1=0.0025, p2=0.025 | `checkpoints_light/mnist_amplitude/noise_high` |
| `configs/noise_light/mnist_amplitude_noise_low.yaml` | MNIST | amplitude | 10 | - | - | 8000 / 1600 | 16 | 6 | p1=0.0003, p2=0.003 | `checkpoints_light/mnist_amplitude/noise_low` |
| `configs/noise_light/mnist_amplitude_noise_mid.yaml` | MNIST | amplitude | 10 | - | - | 8000 / 1600 | 16 | 6 | p1=0.0008, p2=0.008 | `checkpoints_light/mnist_amplitude/noise_mid` |
| `configs/noise_light/mnist_angle_noise_high.yaml` | MNIST | angle | 16 | 16 | - | 6000 / 1200 | 8 | 6 | p1=0.0025, p2=0.025 | `checkpoints_light/mnist_angle/noise_high` |
| `configs/noise_light/mnist_angle_noise_low.yaml` | MNIST | angle | 16 | 16 | - | 6000 / 1200 | 8 | 6 | p1=0.0003, p2=0.003 | `checkpoints_light/mnist_angle/noise_low` |
| `configs/noise_light/mnist_angle_noise_mid.yaml` | MNIST | angle | 16 | 16 | - | 6000 / 1200 | 8 | 6 | p1=0.0008, p2=0.008 | `checkpoints_light/mnist_angle/noise_mid` |
| `configs/noise_light/mnist_hybrid_noise_high.yaml` | MNIST | hybrid | 12 | 144 | - | 6000 / 1200 | 12 | 6 | p1=0.0025, p2=0.025 | `checkpoints_light/mnist_hybrid/noise_high` |
| `configs/noise_light/mnist_hybrid_noise_low.yaml` | MNIST | hybrid | 12 | 144 | - | 6000 / 1200 | 12 | 6 | p1=0.0003, p2=0.003 | `checkpoints_light/mnist_hybrid/noise_low` |
| `configs/noise_light/mnist_hybrid_noise_mid.yaml` | MNIST | hybrid | 12 | 144 | - | 6000 / 1200 | 12 | 6 | p1=0.0008, p2=0.008 | `checkpoints_light/mnist_hybrid/noise_mid` |

### configs/profiling
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/profiling/mnist_hybrid_noise_mid_baseline.yaml` | MNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | p1=0.001, p2=0.01 | `checkpoints/profiling/mnist_hybrid_noise_mid_baseline` |
| `configs/profiling/mnist_hybrid_noise_mid_batch1.yaml` | MNIST | hybrid | 12 | 144 | - | 16 / 16 | 16 | 1 | p1=0.001, p2=0.01 | `checkpoints/profiling/mnist_hybrid_noise_mid_batch1` |
| `configs/profiling/mnist_hybrid_noise_mid_mini.yaml` | MNIST | hybrid | 12 | 144 | - | 1024 / 512 | 16 | 5 | p1=0.001, p2=0.01 | `checkpoints/profiling/mnist_hybrid_noise_mid_mini` |
| `configs/profiling/mnist_hybrid_noiseless_reference.yaml` | MNIST | hybrid | 12 | 144 | - | 15000 / 3000 | 16 | 10 | none | `checkpoints/profiling/mnist_hybrid_noiseless_reference` |

### configs (root)
| Config | Dataset | Encoding | Qubits | Features | Labels | Train/Test | Batch | Epochs | Noise | Checkpoint Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/fashion_amplitude.yaml` | FashionMNIST | amplitude | 10 | - | - | 256 / 128 | 8 | 5 | none | `-` |
| `configs/fashion_angle.yaml` | FashionMNIST | angle | 4 | - | - | 256 / 128 | 32 | 5 | none | `-` |
| `configs/fashion_hybrid.yaml` | FashionMNIST | hybrid | 4 | 16 | - | 256 / 128 | 32 | 5 | none | `-` |
| `configs/mnist_amplitude.yaml` | MNIST | amplitude | 10 | - | - | 128 / 64 | 8 | 3 | p1=0.001, p2=0.01 | `-` |
| `configs/mnist_amplitude_clean.yaml` | - | - | - | - | - | - / - | - | - | none | `-` |
| `configs/mnist_amplitude_fast.yaml` | MNIST | amplitude | 10 | - | - | 200 / 100 | 4 | 5 | none | `checkpoints/mnist_amplitude_fast` |
| `configs/mnist_amplitude_quick.yaml` | MNIST | amplitude | 10 | - | - | 8 / 4 | 2 | 1 | none | `-` |
| `configs/mnist_angle.yaml` | MNIST | angle | 4 | - | - | 256 / 128 | 32 | 5 | p1=0.001, p2=0.01 | `-` |
| `configs/mnist_angle_fast.yaml` | MNIST | angle | 8 | 64 | - | 200 / 100 | 8 | 5 | none | `-` |
| `configs/mnist_hybrid.yaml` | MNIST | hybrid | 4 | 16 | - | 256 / 128 | 32 | 5 | p1=0.001, p2=0.01 | `-` |

## Evaluation Results (noise_2.6 & noise_2.6.3)
所有 checkpoint 均在配置指定的测试 split 上重新推理得到下表的 `Eval Acc`。

### configs/noise_2.6.3/mnist_hybrid_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise263/mnist_hybrid/noise_high/best.pt` | 9 | 52.3438 | 48.44 | 0.700614 |
| `checkpoints_noise263/mnist_hybrid/noise_high/last.pt` | 10 | 50.7812 | 51.56 | 0.695418 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_1.pt` | 1 | 48.4375 | 54.69 | 0.737406 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_10.pt` | 10 | 50.7812 | 51.56 | 0.695418 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_2.pt` | 2 | 47.6562 | 51.56 | 0.695215 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_3.pt` | 3 | 46.0938 | 54.69 | 0.696114 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_4.pt` | 4 | 44.5312 | 39.06 | 0.695478 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_5.pt` | 5 | 50.0 | 48.44 | 0.694992 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_6.pt` | 6 | 47.6562 | 42.19 | 0.69547 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_7.pt` | 7 | 49.2188 | 48.44 | 0.698567 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_8.pt` | 8 | 46.0938 | 51.56 | 0.696738 |
| `checkpoints_noise263/mnist_hybrid/noise_high/mnist_hybrid_noise263_high_epoch_9.pt` | 9 | 52.3438 | 48.44 | 0.700614 |

### configs/noise_2.6.3/mnist_hybrid_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise263/mnist_hybrid/noise_low/best.pt` | 1 | 53.5156 | 59.38 | 0.716089 |
| `checkpoints_noise263/mnist_hybrid/noise_low/last.pt` | 10 | 53.5156 | 59.38 | 0.692074 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_1.pt` | 1 | 53.5156 | 59.38 | 0.716089 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_10.pt` | 10 | 53.5156 | 59.38 | 0.692074 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_2.pt` | 2 | 53.5156 | 59.38 | 0.693726 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_3.pt` | 3 | 53.5156 | 59.38 | 0.69341 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_4.pt` | 4 | 53.5156 | 59.38 | 0.693166 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_5.pt` | 5 | 53.5156 | 59.38 | 0.692295 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_6.pt` | 6 | 53.5156 | 59.38 | 0.691718 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_7.pt` | 7 | 53.5156 | 59.38 | 0.692178 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_8.pt` | 8 | 53.5156 | 59.38 | 0.693437 |
| `checkpoints_noise263/mnist_hybrid/noise_low/mnist_hybrid_noise263_low_epoch_9.pt` | 9 | 53.5156 | 59.38 | 0.692147 |

### configs/noise_2.6.3/mnist_hybrid_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise263/mnist_hybrid/noise_mid/best.pt` | 1 | 50.3906 | 46.88 | 0.699137 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/last.pt` | 10 | 47.2656 | 43.75 | 0.693994 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_1.pt` | 1 | 50.3906 | 46.88 | 0.699137 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_10.pt` | 10 | 47.2656 | 50.00 | 0.693994 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_2.pt` | 2 | 50.3906 | 46.88 | 0.696915 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_3.pt` | 3 | 50.3906 | 46.88 | 0.694076 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_4.pt` | 4 | 49.6094 | 46.88 | 0.694056 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_5.pt` | 5 | 50.3906 | 46.88 | 0.693876 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_6.pt` | 6 | 50.3906 | 46.88 | 0.693755 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_7.pt` | 7 | 50.3906 | 46.88 | 0.694746 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_8.pt` | 8 | 45.7031 | 50.00 | 0.693776 |
| `checkpoints_noise263/mnist_hybrid/noise_mid/mnist_hybrid_noise263_mid_epoch_9.pt` | 9 | 48.8281 | 46.88 | 0.694018 |

### configs/noise_2.6.3/mnist_hybrid_noise_none.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise263/mnist_hybrid/noise_none/best.pt` | 3 | 57.8125 | 62.50 | 0.686731 |
| `checkpoints_noise263/mnist_hybrid/noise_none/last.pt` | 10 | 57.8125 | 62.50 | 0.68389 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_1.pt` | 1 | 42.1875 | 37.50 | 0.764002 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_10.pt` | 10 | 57.8125 | 62.50 | 0.68389 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_2.pt` | 2 | 54.6875 | 62.50 | 0.686982 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_3.pt` | 3 | 57.8125 | 62.50 | 0.686731 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_4.pt` | 4 | 57.8125 | 62.50 | 0.68505 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_5.pt` | 5 | 57.8125 | 62.50 | 0.683093 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_6.pt` | 6 | 57.8125 | 62.50 | 0.688342 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_7.pt` | 7 | 57.8125 | 62.50 | 0.682262 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_8.pt` | 8 | 57.8125 | 62.50 | 0.681584 |
| `checkpoints_noise263/mnist_hybrid/noise_none/mnist_hybrid_noise263_none_epoch_9.pt` | 9 | 57.8125 | 62.50 | 0.683554 |

### configs/noise_2.6/mnist_hybrid_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise26/mnist_hybrid/noise_high/best.pt` | 1 | 56.25 | 57.81 | 0.694742 |
| `checkpoints_noise26/mnist_hybrid/noise_high/last.pt` | 10 | 50.7812 | 57.81 | 0.695031 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_1.pt` | 1 | 56.25 | 57.81 | 0.694742 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_10.pt` | 10 | 50.7812 | 57.81 | 0.695031 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_2.pt` | 2 | 53.9062 | 57.81 | 0.694636 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_3.pt` | 3 | 56.25 | 57.81 | 0.68746 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_4.pt` | 4 | 50.7812 | 57.81 | 0.699583 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_5.pt` | 5 | 50.7812 | 57.81 | 0.693538 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_6.pt` | 6 | 47.6562 | 57.81 | 0.695182 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_7.pt` | 7 | 50.7812 | 45.31 | 0.693589 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_8.pt` | 8 | 46.0938 | 54.69 | 0.695429 |
| `checkpoints_noise26/mnist_hybrid/noise_high/mnist_hybrid_noise26_high_epoch_9.pt` | 9 | 48.4375 | 57.81 | 0.692778 |

### configs/noise_2.6/mnist_hybrid_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise26/mnist_hybrid/noise_low/best.pt` | 10 | 51.9531 | 59.38 | 0.692856 |
| `checkpoints_noise26/mnist_hybrid/noise_low/last.pt` | 10 | 51.9531 | 59.38 | 0.692856 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_1.pt` | 1 | 49.6094 | 40.62 | 0.714508 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_10.pt` | 10 | 51.9531 | 59.38 | 0.692856 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_2.pt` | 2 | 41.7969 | 40.62 | 0.69773 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_3.pt` | 3 | 49.6094 | 59.38 | 0.697309 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_4.pt` | 4 | 47.2656 | 59.38 | 0.693542 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_5.pt` | 5 | 51.1719 | 59.38 | 0.693731 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_6.pt` | 6 | 51.1719 | 59.38 | 0.694746 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_7.pt` | 7 | 48.8281 | 59.38 | 0.693963 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_8.pt` | 8 | 51.1719 | 59.38 | 0.693457 |
| `checkpoints_noise26/mnist_hybrid/noise_low/mnist_hybrid_noise26_low_epoch_9.pt` | 9 | 45.7031 | 40.62 | 0.696573 |

### configs/noise_2.6/mnist_hybrid_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise26/mnist_hybrid/noise_mid/best.pt` | 1 | 52.3438 | 51.56 | 0.721068 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/last.pt` | 10 | 49.6094 | 48.44 | 0.694955 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_1.pt` | 1 | 52.3438 | 51.56 | 0.721068 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_10.pt` | 10 | 49.6094 | 48.44 | 0.694955 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_2.pt` | 2 | 52.3438 | 51.56 | 0.695692 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_3.pt` | 3 | 52.3438 | 51.56 | 0.693376 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_4.pt` | 4 | 50.3906 | 48.44 | 0.694935 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_5.pt` | 5 | 50.3906 | 48.44 | 0.693681 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_6.pt` | 6 | 51.1719 | 51.56 | 0.696085 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_7.pt` | 7 | 48.0469 | 48.44 | 0.701107 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_8.pt` | 8 | 51.1719 | 48.44 | 0.695527 |
| `checkpoints_noise26/mnist_hybrid/noise_mid/mnist_hybrid_noise26_mid_epoch_9.pt` | 9 | 46.4844 | 48.44 | 0.695873 |

### configs/noise_2.6/mnist_hybrid_noise_none.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise26/mnist_hybrid/noise_none/best.pt` | 3 | 54.2969 | 46.88 | 0.690149 |
| `checkpoints_noise26/mnist_hybrid/noise_none/last.pt` | 10 | 54.2969 | 46.88 | 0.690093 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_1.pt` | 1 | 51.1719 | 46.88 | 0.693245 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_10.pt` | 10 | 54.2969 | 46.88 | 0.690093 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_2.pt` | 2 | 53.5156 | 46.88 | 0.691857 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_3.pt` | 3 | 54.2969 | 46.88 | 0.690149 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_4.pt` | 4 | 54.2969 | 46.88 | 0.69017 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_5.pt` | 5 | 54.2969 | 46.88 | 0.690869 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_6.pt` | 6 | 54.2969 | 46.88 | 0.690166 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_7.pt` | 7 | 54.2969 | 46.88 | 0.690033 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_8.pt` | 8 | 54.2969 | 46.88 | 0.691285 |
| `checkpoints_noise26/mnist_hybrid/noise_none/mnist_hybrid_noise26_none_epoch_9.pt` | 9 | 54.2969 | 46.88 | 0.690111 |

## Evaluation Results (noise_2.3 / 2.3.1 / 2.4.1 / 2.5.1)
所有 checkpoint 均在配置指定的测试 split 上重新推理得到下表的 `Eval Acc`。

### configs/noise_2.3.1/fashion_amplitude_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/best.pt` | 6 | 95.3125 | 93.75 | 0.439852 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_1.pt` | 1 | 50.7812 | 62.50 | 0.893924 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_10.pt` | 10 | 93.3594 | 87.50 | 0.353903 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_2.pt` | 2 | 50.7812 | 56.25 | 0.754677 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_3.pt` | 3 | 50.7812 | 56.25 | 0.628461 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_4.pt` | 4 | 73.4375 | 93.75 | 0.537983 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_5.pt` | 5 | 94.5312 | 93.75 | 0.476554 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_6.pt` | 6 | 95.3125 | 81.25 | 0.439852 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_7.pt` | 7 | 95.3125 | 93.75 | 0.40694 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_8.pt` | 8 | 94.9219 | 100.00 | 0.386112 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/fashion_amplitude_noise23_1_high_epoch_9.pt` | 9 | 93.75 | 87.50 | 0.365367 |
| `checkpoints_noise23_1/fashion_amplitude/noise_high/last.pt` | 10 | 93.3594 | 93.75 | 0.353903 |

### configs/noise_2.3.1/fashion_amplitude_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/best.pt` | 10 | 86.3281 | 87.50 | 0.384361 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_1.pt` | 1 | 53.5156 | 43.75 | 0.792926 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_10.pt` | 10 | 86.3281 | 93.75 | 0.384361 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_2.pt` | 2 | 53.5156 | 68.75 | 0.695017 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_3.pt` | 3 | 55.4688 | 87.50 | 0.615578 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_4.pt` | 4 | 65.625 | 56.25 | 0.562254 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_5.pt` | 5 | 75.7812 | 81.25 | 0.516822 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_6.pt` | 6 | 80.0781 | 75.00 | 0.480274 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_7.pt` | 7 | 82.8125 | 87.50 | 0.451238 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_8.pt` | 8 | 85.1562 | 87.50 | 0.426477 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/fashion_amplitude_noise23_1_low_epoch_9.pt` | 9 | 85.5469 | 93.75 | 0.402216 |
| `checkpoints_noise23_1/fashion_amplitude/noise_low/last.pt` | 10 | 86.3281 | 93.75 | 0.384361 |

### configs/noise_2.3.1/fashion_amplitude_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/best.pt` | 10 | 88.2812 | 93.75 | 0.380342 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_1.pt` | 1 | 47.6562 | 43.75 | 0.840459 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_10.pt` | 10 | 88.2812 | 93.75 | 0.380342 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_2.pt` | 2 | 47.6562 | 75.00 | 0.739167 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_3.pt` | 3 | 49.6094 | 56.25 | 0.657228 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_4.pt` | 4 | 62.5 | 62.50 | 0.58716 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_5.pt` | 5 | 74.6094 | 87.50 | 0.535403 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_6.pt` | 6 | 82.0312 | 81.25 | 0.493236 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_7.pt` | 7 | 83.9844 | 81.25 | 0.457149 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_8.pt` | 8 | 87.5 | 87.50 | 0.425397 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/fashion_amplitude_noise23_1_mid_epoch_9.pt` | 9 | 87.8906 | 93.75 | 0.40045 |
| `checkpoints_noise23_1/fashion_amplitude/noise_mid/last.pt` | 10 | 88.2812 | 93.75 | 0.380342 |

### configs/noise_2.3.1/fashion_hybrid_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/best.pt` | 7 | 60.1562 | 43.75 | 0.685493 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_1.pt` | 1 | 48.8281 | 50.00 | 0.812143 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_10.pt` | 10 | 57.4219 | 62.50 | 0.681206 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_2.pt` | 2 | 48.8281 | 31.25 | 0.747966 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_3.pt` | 3 | 48.8281 | 43.75 | 0.713072 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_4.pt` | 4 | 48.8281 | 56.25 | 0.69905 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_5.pt` | 5 | 50.0 | 75.00 | 0.69448 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_6.pt` | 6 | 58.2031 | 56.25 | 0.688557 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_7.pt` | 7 | 60.1562 | 68.75 | 0.685493 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_8.pt` | 8 | 55.8594 | 68.75 | 0.683509 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/fashion_hybrid_noise23_1_high_epoch_9.pt` | 9 | 56.6406 | 68.75 | 0.682096 |
| `checkpoints_noise23_1/fashion_hybrid/noise_high/last.pt` | 10 | 57.4219 | 68.75 | 0.681206 |

### configs/noise_2.3.1/fashion_hybrid_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/best.pt` | 4 | 66.0156 | 68.75 | 0.671584 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_1.pt` | 1 | 47.6562 | 37.50 | 0.692332 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_10.pt` | 10 | 62.1094 | 62.50 | 0.66179 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_2.pt` | 2 | 57.4219 | 62.50 | 0.678216 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_3.pt` | 3 | 64.0625 | 56.25 | 0.672099 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_4.pt` | 4 | 66.0156 | 50.00 | 0.671584 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_5.pt` | 5 | 57.8125 | 62.50 | 0.670403 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_6.pt` | 6 | 63.6719 | 81.25 | 0.663932 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_7.pt` | 7 | 62.5 | 93.75 | 0.663963 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_8.pt` | 8 | 65.2344 | 68.75 | 0.662559 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/fashion_hybrid_noise23_1_low_epoch_9.pt` | 9 | 64.8438 | 50.00 | 0.661302 |
| `checkpoints_noise23_1/fashion_hybrid/noise_low/last.pt` | 10 | 62.1094 | 75.00 | 0.66179 |

### configs/noise_2.3.1/fashion_hybrid_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/best.pt` | 5 | 58.2031 | 37.50 | 0.68613 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_1.pt` | 1 | 47.2656 | 50.00 | 0.926278 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_10.pt` | 10 | 51.5625 | 62.50 | 0.675044 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_2.pt` | 2 | 47.2656 | 50.00 | 0.807595 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_3.pt` | 3 | 48.8281 | 75.00 | 0.735493 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_4.pt` | 4 | 53.125 | 75.00 | 0.701267 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_5.pt` | 5 | 58.2031 | 43.75 | 0.68613 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_6.pt` | 6 | 54.6875 | 31.25 | 0.679096 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_7.pt` | 7 | 48.8281 | 56.25 | 0.676228 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_8.pt` | 8 | 51.5625 | 43.75 | 0.674988 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/fashion_hybrid_noise23_1_mid_epoch_9.pt` | 9 | 51.5625 | 68.75 | 0.671579 |
| `checkpoints_noise23_1/fashion_hybrid/noise_mid/last.pt` | 10 | 51.5625 | 56.25 | 0.675044 |

### configs/noise_2.3.1/mnist_amplitude_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/best.pt` | 10 | 57.0312 | 56.25 | 0.670648 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/last.pt` | 10 | 57.0312 | 50.00 | 0.670648 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_1.pt` | 1 | 53.9062 | 37.50 | 1.132725 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_10.pt` | 10 | 57.0312 | 43.75 | 0.670648 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_2.pt` | 2 | 53.5156 | 50.00 | 1.002029 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_3.pt` | 3 | 53.125 | 50.00 | 0.896443 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_4.pt` | 4 | 51.9531 | 62.50 | 0.817794 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_5.pt` | 5 | 53.9062 | 43.75 | 0.764249 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_6.pt` | 6 | 53.5156 | 68.75 | 0.731601 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_7.pt` | 7 | 53.9062 | 56.25 | 0.708734 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_8.pt` | 8 | 53.9062 | 50.00 | 0.694568 |
| `checkpoints_noise23_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_1_high_epoch_9.pt` | 9 | 54.6875 | 68.75 | 0.683087 |

### configs/noise_2.3.1/mnist_amplitude_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/best.pt` | 10 | 93.75 | 100.00 | 0.409902 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/last.pt` | 10 | 93.75 | 100.00 | 0.409902 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_1.pt` | 1 | 50.0 | 43.75 | 0.858782 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_10.pt` | 10 | 93.75 | 93.75 | 0.409902 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_2.pt` | 2 | 50.0 | 43.75 | 0.795575 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_3.pt` | 3 | 50.0 | 37.50 | 0.732901 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_4.pt` | 4 | 50.0 | 56.25 | 0.662069 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_5.pt` | 5 | 79.6875 | 100.00 | 0.596713 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_6.pt` | 6 | 92.1875 | 87.50 | 0.533884 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_7.pt` | 7 | 92.9688 | 100.00 | 0.491295 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_8.pt` | 8 | 92.9688 | 93.75 | 0.455559 |
| `checkpoints_noise23_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_1_low_epoch_9.pt` | 9 | 93.3594 | 93.75 | 0.430073 |

### configs/noise_2.3.1/mnist_amplitude_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/best.pt` | 10 | 95.3125 | 75.00 | 0.333729 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/last.pt` | 10 | 95.3125 | 93.75 | 0.333729 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_1.pt` | 1 | 46.0938 | 43.75 | 0.851443 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_10.pt` | 10 | 95.3125 | 81.25 | 0.333729 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_2.pt` | 2 | 46.0938 | 56.25 | 0.736864 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_3.pt` | 3 | 56.25 | 56.25 | 0.634246 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_4.pt` | 4 | 81.25 | 87.50 | 0.548263 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_5.pt` | 5 | 91.0156 | 68.75 | 0.491316 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_6.pt` | 6 | 94.1406 | 87.50 | 0.437277 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_7.pt` | 7 | 94.5312 | 100.00 | 0.399246 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_8.pt` | 8 | 94.1406 | 87.50 | 0.373376 |
| `checkpoints_noise23_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_1_mid_epoch_9.pt` | 9 | 94.9219 | 87.50 | 0.351196 |

### configs/noise_2.3.1/mnist_angle_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/mnist_angle/noise_high/best.pt` | 2 | 52.3438 | 56.25 | 0.695322 |
| `checkpoints_noise23_1/mnist_angle/noise_high/last.pt` | 9 | 43.3594 | 56.25 | 0.697387 |
| `checkpoints_noise23_1/mnist_angle/noise_high/mnist_angle_noise23_1_high_epoch_1.pt` | 1 | 50.7812 | 31.25 | 0.693035 |
| `checkpoints_noise23_1/mnist_angle/noise_high/mnist_angle_noise23_1_high_epoch_2.pt` | 2 | 52.3438 | 62.50 | 0.695322 |
| `checkpoints_noise23_1/mnist_angle/noise_high/mnist_angle_noise23_1_high_epoch_3.pt` | 3 | 48.0469 | 18.75 | 0.696097 |
| `checkpoints_noise23_1/mnist_angle/noise_high/mnist_angle_noise23_1_high_epoch_4.pt` | 4 | 48.0469 | 68.75 | 0.698161 |
| `checkpoints_noise23_1/mnist_angle/noise_high/mnist_angle_noise23_1_high_epoch_5.pt` | 5 | 50.0 | 37.50 | 0.693979 |
| `checkpoints_noise23_1/mnist_angle/noise_high/mnist_angle_noise23_1_high_epoch_6.pt` | 6 | 48.0469 | 50.00 | 0.695901 |
| `checkpoints_noise23_1/mnist_angle/noise_high/mnist_angle_noise23_1_high_epoch_7.pt` | 7 | 46.4844 | 62.50 | 0.694709 |
| `checkpoints_noise23_1/mnist_angle/noise_high/mnist_angle_noise23_1_high_epoch_8.pt` | 8 | 50.7812 | 62.50 | 0.693083 |
| `checkpoints_noise23_1/mnist_angle/noise_high/mnist_angle_noise23_1_high_epoch_9.pt` | 9 | 43.3594 | 62.50 | 0.697387 |

### configs/noise_2.3.1/mnist_angle_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/mnist_angle/noise_low/best.pt` | 7 | 53.125 | 56.25 | 0.693257 |
| `checkpoints_noise23_1/mnist_angle/noise_low/last.pt` | 9 | 52.7344 | 75.00 | 0.691796 |
| `checkpoints_noise23_1/mnist_angle/noise_low/mnist_angle_noise23_1_low_epoch_1.pt` | 1 | 52.3438 | 62.50 | 0.743689 |
| `checkpoints_noise23_1/mnist_angle/noise_low/mnist_angle_noise23_1_low_epoch_2.pt` | 2 | 52.3438 | 81.25 | 0.69479 |
| `checkpoints_noise23_1/mnist_angle/noise_low/mnist_angle_noise23_1_low_epoch_3.pt` | 3 | 52.3438 | 81.25 | 0.69258 |
| `checkpoints_noise23_1/mnist_angle/noise_low/mnist_angle_noise23_1_low_epoch_4.pt` | 4 | 52.3438 | 75.00 | 0.693009 |
| `checkpoints_noise23_1/mnist_angle/noise_low/mnist_angle_noise23_1_low_epoch_5.pt` | 5 | 52.3438 | 37.50 | 0.691925 |
| `checkpoints_noise23_1/mnist_angle/noise_low/mnist_angle_noise23_1_low_epoch_6.pt` | 6 | 52.7344 | 50.00 | 0.692081 |

### configs/noise_2.3.1/mnist_angle_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/mnist_angle/noise_mid/best.pt` | 2 | 51.5625 | 43.75 | 0.69544 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/last.pt` | 9 | 51.5625 | 50.00 | 0.693227 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/mnist_angle_noise23_1_mid_epoch_1.pt` | 1 | 44.9219 | 68.75 | 0.69838 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/mnist_angle_noise23_1_mid_epoch_2.pt` | 2 | 51.5625 | 62.50 | 0.69544 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/mnist_angle_noise23_1_mid_epoch_3.pt` | 3 | 51.5625 | 56.25 | 0.694571 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/mnist_angle_noise23_1_mid_epoch_4.pt` | 4 | 46.875 | 62.50 | 0.694804 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/mnist_angle_noise23_1_mid_epoch_5.pt` | 5 | 46.0938 | 50.00 | 0.694211 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/mnist_angle_noise23_1_mid_epoch_6.pt` | 6 | 51.5625 | 50.00 | 0.692863 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/mnist_angle_noise23_1_mid_epoch_7.pt` | 7 | 51.5625 | 50.00 | 0.69308 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/mnist_angle_noise23_1_mid_epoch_8.pt` | 8 | 51.5625 | 50.00 | 0.693526 |
| `checkpoints_noise23_1/mnist_angle/noise_mid/mnist_angle_noise23_1_mid_epoch_9.pt` | 9 | 51.5625 | 62.50 | 0.693227 |

### configs/noise_2.3.1/mnist_hybrid_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/best.pt` | 4 | 61.3281 | 50.00 | 0.656937 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/last.pt` | 10 | 58.2031 | 56.25 | 0.648374 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_1.pt` | 1 | 45.7031 | 37.50 | 0.786193 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_10.pt` | 10 | 58.2031 | 56.25 | 0.648374 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_2.pt` | 2 | 50.7812 | 31.25 | 0.704128 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_3.pt` | 3 | 60.1562 | 50.00 | 0.667226 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_4.pt` | 4 | 61.3281 | 31.25 | 0.656937 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_5.pt` | 5 | 57.8125 | 62.50 | 0.653263 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_6.pt` | 6 | 58.9844 | 62.50 | 0.652307 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_7.pt` | 7 | 58.9844 | 50.00 | 0.648966 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_8.pt` | 8 | 57.8125 | 43.75 | 0.648909 |
| `checkpoints_noise23_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_1_high_epoch_9.pt` | 9 | 58.2031 | 31.25 | 0.648921 |

### configs/noise_2.3/mnist_amplitude_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23/mnist_amplitude/noise_high/best.pt` | 10 | 96.875 | 96.88 | 0.485732 |
| `checkpoints_noise23/mnist_amplitude/noise_high/last.pt` | 10 | 96.875 | 96.88 | 0.485732 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_1.pt` | 1 | 45.7031 | 46.88 | 0.839615 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_10.pt` | 10 | 96.875 | 96.88 | 0.485732 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_2.pt` | 2 | 45.7031 | 46.88 | 0.763569 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_3.pt` | 3 | 45.7031 | 46.88 | 0.716433 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_4.pt` | 4 | 45.7031 | 54.69 | 0.694306 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_5.pt` | 5 | 77.3438 | 82.81 | 0.674042 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_6.pt` | 6 | 83.2031 | 92.19 | 0.640921 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_7.pt` | 7 | 89.8438 | 95.31 | 0.59771 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_8.pt` | 8 | 95.7031 | 95.31 | 0.553767 |
| `checkpoints_noise23/mnist_amplitude/noise_high/mnist_amplitude_noise23_high_epoch_9.pt` | 9 | 94.9219 | 95.31 | 0.516467 |

### configs/noise_2.3/mnist_amplitude_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23/mnist_amplitude/noise_low/best.pt` | 8 | 96.4844 | 96.88 | 0.404318 |
| `checkpoints_noise23/mnist_amplitude/noise_low/last.pt` | 10 | 96.4844 | 98.44 | 0.36612 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_1.pt` | 1 | 58.9844 | 89.06 | 0.683572 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_10.pt` | 10 | 96.4844 | 98.44 | 0.36612 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_2.pt` | 2 | 78.125 | 90.62 | 0.644529 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_3.pt` | 3 | 82.8125 | 93.75 | 0.598828 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_4.pt` | 4 | 91.0156 | 96.88 | 0.551157 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_5.pt` | 5 | 95.3125 | 96.88 | 0.500473 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_6.pt` | 6 | 94.1406 | 98.44 | 0.465749 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_7.pt` | 7 | 94.9219 | 98.44 | 0.432303 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_8.pt` | 8 | 96.4844 | 98.44 | 0.404318 |
| `checkpoints_noise23/mnist_amplitude/noise_low/mnist_amplitude_noise23_low_epoch_9.pt` | 9 | 96.0938 | 96.88 | 0.382091 |

### configs/noise_2.3/mnist_amplitude_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23/mnist_amplitude/noise_mid/best.pt` | 9 | 95.7031 | 95.31 | 0.409923 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/last.pt` | 10 | 94.5312 | 96.88 | 0.396577 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_1.pt` | 1 | 53.9062 | 46.88 | 0.660034 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_10.pt` | 10 | 94.5312 | 96.88 | 0.396577 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_2.pt` | 2 | 54.6875 | 59.38 | 0.602586 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_3.pt` | 3 | 77.7344 | 82.81 | 0.56136 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_4.pt` | 4 | 91.7969 | 89.06 | 0.526798 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_5.pt` | 5 | 92.9688 | 95.31 | 0.498478 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_6.pt` | 6 | 94.1406 | 96.88 | 0.476155 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_7.pt` | 7 | 94.1406 | 95.31 | 0.452077 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_8.pt` | 8 | 94.1406 | 96.88 | 0.43179 |
| `checkpoints_noise23/mnist_amplitude/noise_mid/mnist_amplitude_noise23_mid_epoch_9.pt` | 9 | 95.7031 | 96.88 | 0.409923 |

### configs/noise_2.3/mnist_angle_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23/mnist_angle/noise_high/best.pt` | 2 | 51.1719 | 60.94 | 0.693479 |
| `checkpoints_noise23/mnist_angle/noise_high/last.pt` | 10 | 51.1719 | 60.94 | 0.69356 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_1.pt` | 1 | 46.875 | 60.94 | 0.698999 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_10.pt` | 10 | 51.1719 | 60.94 | 0.69356 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_2.pt` | 2 | 51.1719 | 60.94 | 0.693479 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_3.pt` | 3 | 46.4844 | 60.94 | 0.694378 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_4.pt` | 4 | 51.1719 | 60.94 | 0.694191 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_5.pt` | 5 | 51.1719 | 60.94 | 0.694198 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_6.pt` | 6 | 49.6094 | 60.94 | 0.693426 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_7.pt` | 7 | 51.1719 | 60.94 | 0.693454 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_8.pt` | 8 | 51.1719 | 60.94 | 0.693385 |
| `checkpoints_noise23/mnist_angle/noise_high/mnist_angle_noise23_high_epoch_9.pt` | 9 | 51.1719 | 60.94 | 0.694636 |

### configs/noise_2.3/mnist_angle_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23/mnist_angle/noise_low/best.pt` | 1 | 56.25 | 43.75 | 0.89634 |
| `checkpoints_noise23/mnist_angle/noise_low/last.pt` | 10 | 56.25 | 43.75 | 0.686608 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_1.pt` | 1 | 56.25 | 43.75 | 0.89634 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_10.pt` | 10 | 56.25 | 43.75 | 0.686608 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_2.pt` | 2 | 56.25 | 43.75 | 0.708274 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_3.pt` | 3 | 56.25 | 43.75 | 0.687868 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_4.pt` | 4 | 56.25 | 43.75 | 0.68638 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_5.pt` | 5 | 56.25 | 43.75 | 0.688038 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_6.pt` | 6 | 56.25 | 43.75 | 0.687171 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_7.pt` | 7 | 56.25 | 43.75 | 0.687414 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_8.pt` | 8 | 56.25 | 43.75 | 0.687598 |
| `checkpoints_noise23/mnist_angle/noise_low/mnist_angle_noise23_low_epoch_9.pt` | 9 | 56.25 | 43.75 | 0.686475 |

### configs/noise_2.3/mnist_angle_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23/mnist_angle/noise_mid/best.pt` | 2 | 52.7344 | 25.00 | 0.694129 |
| `checkpoints_noise23/mnist_angle/noise_mid/last.pt` | 10 | 52.7344 | 50.00 | 0.693259 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_1.pt` | 1 | 50.3906 | 50.00 | 0.694179 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_10.pt` | 10 | 52.7344 | 37.50 | 0.693259 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_2.pt` | 2 | 52.7344 | 50.00 | 0.694129 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_3.pt` | 3 | 52.7344 | 68.75 | 0.695624 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_4.pt` | 4 | 52.7344 | 37.50 | 0.694661 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_5.pt` | 5 | 52.7344 | 56.25 | 0.692067 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_6.pt` | 6 | 46.0938 | 50.00 | 0.69425 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_7.pt` | 7 | 52.7344 | 56.25 | 0.693624 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_8.pt` | 8 | 50.0 | 37.50 | 0.695707 |
| `checkpoints_noise23/mnist_angle/noise_mid/mnist_angle_noise23_mid_epoch_9.pt` | 9 | 52.7344 | 56.25 | 0.693601 |

### configs/noise_2.3/mnist_hybrid_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23/mnist_hybrid/noise_high/best.pt` | 9 | 60.9375 | 75.00 | 0.689519 |
| `checkpoints_noise23/mnist_hybrid/noise_high/last.pt` | 10 | 58.5938 | 75.00 | 0.691123 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_1.pt` | 1 | 51.9531 | 62.50 | 0.877595 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_10.pt` | 10 | 58.5938 | 100.00 | 0.691123 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_2.pt` | 2 | 51.9531 | 62.50 | 0.783601 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_3.pt` | 3 | 51.9531 | 56.25 | 0.730306 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_4.pt` | 4 | 52.7344 | 62.50 | 0.704266 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_5.pt` | 5 | 56.25 | 75.00 | 0.694333 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_6.pt` | 6 | 59.375 | 68.75 | 0.689737 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_7.pt` | 7 | 60.1562 | 62.50 | 0.689165 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_8.pt` | 8 | 59.7656 | 81.25 | 0.688845 |
| `checkpoints_noise23/mnist_hybrid/noise_high/mnist_hybrid_noise23_high_epoch_9.pt` | 9 | 60.9375 | 75.00 | 0.689519 |

### configs/noise_2.3/mnist_hybrid_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23/mnist_hybrid/noise_low/best.pt` | 8 | 52.7344 | 31.25 | 0.686712 |
| `checkpoints_noise23/mnist_hybrid/noise_low/last.pt` | 10 | 49.6094 | 62.50 | 0.685194 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_1.pt` | 1 | 49.6094 | 50.00 | 0.691355 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_10.pt` | 10 | 49.6094 | 37.50 | 0.685194 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_2.pt` | 2 | 47.6562 | 43.75 | 0.689406 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_3.pt` | 3 | 48.4375 | 50.00 | 0.690046 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_4.pt` | 4 | 52.3438 | 37.50 | 0.687396 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_5.pt` | 5 | 47.6562 | 43.75 | 0.687826 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_6.pt` | 6 | 49.2188 | 50.00 | 0.685848 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_7.pt` | 7 | 52.3438 | 68.75 | 0.687859 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_8.pt` | 8 | 52.7344 | 56.25 | 0.686712 |
| `checkpoints_noise23/mnist_hybrid/noise_low/mnist_hybrid_noise23_low_epoch_9.pt` | 9 | 50.7812 | 43.75 | 0.686739 |

### configs/noise_2.3/mnist_hybrid_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23/mnist_hybrid/noise_mid/best.pt` | 10 | 58.9844 | 81.25 | 0.689527 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/last.pt` | 10 | 58.9844 | 68.75 | 0.689527 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_1.pt` | 1 | 52.3438 | 37.50 | 0.724086 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_10.pt` | 10 | 58.9844 | 75.00 | 0.689527 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_2.pt` | 2 | 52.3438 | 43.75 | 0.699008 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_3.pt` | 3 | 52.3438 | 56.25 | 0.694304 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_4.pt` | 4 | 52.3438 | 37.50 | 0.692645 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_5.pt` | 5 | 53.125 | 56.25 | 0.692461 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_6.pt` | 6 | 52.7344 | 62.50 | 0.692437 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_7.pt` | 7 | 55.0781 | 68.75 | 0.691753 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_8.pt` | 8 | 57.4219 | 81.25 | 0.690686 |
| `checkpoints_noise23/mnist_hybrid/noise_mid/mnist_hybrid_noise23_mid_epoch_9.pt` | 9 | 58.5938 | 87.50 | 0.689935 |

### configs/noise_2.4.1/mnist_amplitude_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_high/best.pt` | 4 | 83.5938 | 81.25 | 0.52235 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_high/last.pt` | 4 | 83.5938 | 87.50 | 0.52235 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_4_1_high_epoch_1.pt` | 1 | 78.9062 | 81.25 | 0.579373 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_4_1_high_epoch_2.pt` | 2 | 82.4219 | 87.50 | 0.551703 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_4_1_high_epoch_3.pt` | 3 | 80.8594 | 68.75 | 0.537816 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_high/mnist_amplitude_noise23_4_1_high_epoch_4.pt` | 4 | 83.5938 | 93.75 | 0.52235 |

### configs/noise_2.4.1/mnist_amplitude_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_low/best.pt` | 4 | 69.1406 | 81.25 | 0.658015 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_low/last.pt` | 4 | 69.1406 | 81.25 | 0.658015 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_4_1_low_epoch_1.pt` | 1 | 49.6094 | 50.00 | 0.762665 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_4_1_low_epoch_2.pt` | 2 | 49.6094 | 31.25 | 0.716864 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_4_1_low_epoch_3.pt` | 3 | 50.3906 | 62.50 | 0.682072 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_low/mnist_amplitude_noise23_4_1_low_epoch_4.pt` | 4 | 69.1406 | 68.75 | 0.658015 |

### configs/noise_2.4.1/mnist_amplitude_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_mid/best.pt` | 4 | 85.1562 | 93.75 | 0.547625 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_mid/last.pt` | 4 | 85.1562 | 87.50 | 0.547625 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_4_1_mid_epoch_1.pt` | 1 | 81.6406 | 81.25 | 0.606512 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_4_1_mid_epoch_2.pt` | 2 | 83.9844 | 100.00 | 0.583547 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_4_1_mid_epoch_3.pt` | 3 | 83.5938 | 100.00 | 0.566372 |
| `checkpoints_noise23_4_1/mnist_amplitude/noise_mid/mnist_amplitude_noise23_4_1_mid_epoch_4.pt` | 4 | 85.1562 | 93.75 | 0.547625 |

### configs/noise_2.4.1/mnist_angle_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_4_1/mnist_angle/noise_low/best.pt` | 1 | 50.0 | 43.75 | 0.693275 |
| `checkpoints_noise23_4_1/mnist_angle/noise_low/last.pt` | 1 | 50.0 | 50.00 | 0.693275 |
| `checkpoints_noise23_4_1/mnist_angle/noise_low/mnist_angle_noise23_4_1_low_epoch_1.pt` | 1 | 50.0 | 56.25 | 0.693275 |

### configs/noise_2.4.1/mnist_hybrid_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/best.pt` | 3 | 58.2031 | 68.75 | 0.672913 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/last.pt` | 10 | 57.4219 | 75.00 | 0.667385 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_1.pt` | 1 | 50.0 | 50.00 | 0.745311 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_10.pt` | 10 | 57.4219 | 68.75 | 0.667385 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_2.pt` | 2 | 53.5156 | 62.50 | 0.693285 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_3.pt` | 3 | 58.2031 | 56.25 | 0.672913 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_4.pt` | 4 | 57.4219 | 75.00 | 0.667246 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_5.pt` | 5 | 57.0312 | 62.50 | 0.668709 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_6.pt` | 6 | 57.0312 | 81.25 | 0.665155 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_7.pt` | 7 | 54.2969 | 81.25 | 0.66926 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_8.pt` | 8 | 55.8594 | 81.25 | 0.66661 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_high/mnist_hybrid_noise23_4_1_high_epoch_9.pt` | 9 | 54.6875 | 75.00 | 0.668888 |

### configs/noise_2.4.1/mnist_hybrid_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/best.pt` | 3 | 58.5938 | 68.75 | 0.702448 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/last.pt` | 10 | 55.0781 | 43.75 | 0.68431 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_1.pt` | 1 | 54.2969 | 31.25 | 0.805074 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_10.pt` | 10 | 55.0781 | 50.00 | 0.68431 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_2.pt` | 2 | 55.0781 | 56.25 | 0.73728 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_3.pt` | 3 | 58.5938 | 56.25 | 0.702448 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_4.pt` | 4 | 58.2031 | 56.25 | 0.690969 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_5.pt` | 5 | 57.4219 | 75.00 | 0.686482 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_6.pt` | 6 | 57.8125 | 62.50 | 0.684671 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_7.pt` | 7 | 57.0312 | 68.75 | 0.683139 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_8.pt` | 8 | 56.6406 | 25.00 | 0.682867 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_low/mnist_hybrid_noise23_4_1_low_epoch_9.pt` | 9 | 55.8594 | 43.75 | 0.683574 |

### configs/noise_2.4.1/mnist_hybrid_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/best.pt` | 3 | 57.0312 | 56.25 | 0.677647 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/last.pt` | 10 | 51.9531 | 43.75 | 0.667343 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_1.pt` | 1 | 53.125 | 37.50 | 0.736617 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_10.pt` | 10 | 51.9531 | 43.75 | 0.667343 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_2.pt` | 2 | 53.5156 | 50.00 | 0.693174 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_3.pt` | 3 | 57.0312 | 43.75 | 0.677647 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_4.pt` | 4 | 52.3438 | 56.25 | 0.669846 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_5.pt` | 5 | 51.1719 | 50.00 | 0.667 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_6.pt` | 6 | 50.7812 | 50.00 | 0.669569 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_7.pt` | 7 | 51.1719 | 68.75 | 0.667486 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_8.pt` | 8 | 51.1719 | 56.25 | 0.66701 |
| `checkpoints_noise23_4_1/mnist_hybrid/noise_mid/mnist_hybrid_noise23_4_1_mid_epoch_9.pt` | 9 | 51.5625 | 62.50 | 0.666953 |

### configs/noise_2.5.1/fashion_amplitude_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/best.pt` | 8 | 70.7031 | 62.50 | 0.618216 |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/fashion_amplitude_noise25_1_high_epoch_1.pt` | 1 | 51.1719 | 56.25 | 0.727649 |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/fashion_amplitude_noise25_1_high_epoch_2.pt` | 2 | 51.1719 | 75.00 | 0.699987 |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/fashion_amplitude_noise25_1_high_epoch_3.pt` | 3 | 51.1719 | 62.50 | 0.684405 |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/fashion_amplitude_noise25_1_high_epoch_4.pt` | 4 | 53.125 | 43.75 | 0.669194 |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/fashion_amplitude_noise25_1_high_epoch_5.pt` | 5 | 57.8125 | 62.50 | 0.656969 |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/fashion_amplitude_noise25_1_high_epoch_6.pt` | 6 | 64.8438 | 50.00 | 0.641332 |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/fashion_amplitude_noise25_1_high_epoch_7.pt` | 7 | 67.1875 | 75.00 | 0.630076 |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/fashion_amplitude_noise25_1_high_epoch_8.pt` | 8 | 70.7031 | 93.75 | 0.618216 |
| `checkpoints_noise25_1/fashion_amplitude/noise_high/last.pt` | 8 | 70.7031 | 81.25 | 0.618216 |

### configs/noise_2.5.1/fashion_amplitude_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/best.pt` | 7 | 73.8281 | 87.50 | 0.632505 |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/fashion_amplitude_noise25_1_low_epoch_1.pt` | 1 | 45.7031 | 62.50 | 0.714521 |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/fashion_amplitude_noise25_1_low_epoch_2.pt` | 2 | 61.3281 | 62.50 | 0.689405 |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/fashion_amplitude_noise25_1_low_epoch_3.pt` | 3 | 64.8438 | 56.25 | 0.684054 |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/fashion_amplitude_noise25_1_low_epoch_4.pt` | 4 | 68.75 | 68.75 | 0.670074 |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/fashion_amplitude_noise25_1_low_epoch_5.pt` | 5 | 70.7031 | 68.75 | 0.655982 |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/fashion_amplitude_noise25_1_low_epoch_6.pt` | 6 | 71.4844 | 62.50 | 0.647749 |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/fashion_amplitude_noise25_1_low_epoch_7.pt` | 7 | 73.8281 | 68.75 | 0.632505 |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/fashion_amplitude_noise25_1_low_epoch_8.pt` | 8 | 73.0469 | 68.75 | 0.62559 |
| `checkpoints_noise25_1/fashion_amplitude/noise_low/last.pt` | 8 | 73.0469 | 87.50 | 0.62559 |

### configs/noise_2.5.1/fashion_amplitude_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/best.pt` | 8 | 73.8281 | 68.75 | 0.611324 |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/fashion_amplitude_noise25_1_mid_epoch_1.pt` | 1 | 58.5938 | 68.75 | 0.673048 |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/fashion_amplitude_noise25_1_mid_epoch_2.pt` | 2 | 66.0156 | 56.25 | 0.662117 |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/fashion_amplitude_noise25_1_mid_epoch_3.pt` | 3 | 67.1875 | 81.25 | 0.652085 |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/fashion_amplitude_noise25_1_mid_epoch_4.pt` | 4 | 69.9219 | 87.50 | 0.641874 |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/fashion_amplitude_noise25_1_mid_epoch_5.pt` | 5 | 73.4375 | 75.00 | 0.635655 |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/fashion_amplitude_noise25_1_mid_epoch_6.pt` | 6 | 72.2656 | 81.25 | 0.625133 |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/fashion_amplitude_noise25_1_mid_epoch_7.pt` | 7 | 72.2656 | 75.00 | 0.616819 |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/fashion_amplitude_noise25_1_mid_epoch_8.pt` | 8 | 73.8281 | 75.00 | 0.611324 |
| `checkpoints_noise25_1/fashion_amplitude/noise_mid/last.pt` | 8 | 73.8281 | 75.00 | 0.611324 |

### configs/noise_2.5.1/fashion_hybrid_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/best.pt` | 9 | 55.8594 | 75.00 | 0.671096 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_1.pt` | 1 | 51.1719 | 56.25 | 0.733239 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_10.pt` | 10 | 53.9062 | 37.50 | 0.671621 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_2.pt` | 2 | 53.125 | 31.25 | 0.69164 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_3.pt` | 3 | 51.1719 | 75.00 | 0.6783 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_4.pt` | 4 | 55.0781 | 62.50 | 0.673821 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_5.pt` | 5 | 53.125 | 50.00 | 0.675441 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_6.pt` | 6 | 52.3438 | 43.75 | 0.6738 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_7.pt` | 7 | 53.5156 | 62.50 | 0.674226 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_8.pt` | 8 | 54.6875 | 56.25 | 0.673434 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/fashion_hybrid_noise25_1_low_epoch_9.pt` | 9 | 55.8594 | 37.50 | 0.671096 |
| `checkpoints_noise25_1/fashion_hybrid/noise_low/last.pt` | 10 | 53.9062 | 50.00 | 0.671621 |

### configs/noise_2.5.1/fashion_hybrid_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/best.pt` | 8 | 60.9375 | 25.00 | 0.684269 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_1.pt` | 1 | 46.875 | 62.50 | 0.893645 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_10.pt` | 10 | 58.5938 | 37.50 | 0.681978 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_2.pt` | 2 | 49.2188 | 62.50 | 0.791728 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_3.pt` | 3 | 49.6094 | 50.00 | 0.728657 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_4.pt` | 4 | 52.3438 | 56.25 | 0.698571 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_5.pt` | 5 | 54.2969 | 43.75 | 0.690781 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_6.pt` | 6 | 58.5938 | 62.50 | 0.681983 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_7.pt` | 7 | 58.5938 | 50.00 | 0.682827 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_8.pt` | 8 | 60.9375 | 50.00 | 0.684269 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/fashion_hybrid_noise25_1_mid_epoch_9.pt` | 9 | 58.2031 | 37.50 | 0.685862 |
| `checkpoints_noise25_1/fashion_hybrid/noise_mid/last.pt` | 10 | 58.5938 | 62.50 | 0.681978 |

### configs/noise_2.5.1/mnist_amplitude_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/best.pt` | 9 | 62.1094 | 56.25 | 0.65005 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/last.pt` | 10 | 60.9375 | 50.00 | 0.6398 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_1.pt` | 1 | 43.3594 | 62.50 | 0.704379 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_10.pt` | 10 | 60.9375 | 68.75 | 0.6398 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_2.pt` | 2 | 50.0 | 62.50 | 0.693856 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_3.pt` | 3 | 54.2969 | 68.75 | 0.688859 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_4.pt` | 4 | 53.5156 | 68.75 | 0.684717 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_5.pt` | 5 | 55.0781 | 68.75 | 0.678075 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_6.pt` | 6 | 55.8594 | 50.00 | 0.673758 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_7.pt` | 7 | 58.5938 | 43.75 | 0.66506 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_8.pt` | 8 | 61.7188 | 56.25 | 0.658928 |
| `checkpoints_noise25_1/mnist_amplitude/noise_high/mnist_amplitude_noise25_1_high_epoch_9.pt` | 9 | 62.1094 | 68.75 | 0.65005 |

### configs/noise_2.5.1/mnist_amplitude_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/best.pt` | 10 | 61.3281 | 81.25 | 0.645142 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/last.pt` | 10 | 61.3281 | 56.25 | 0.645142 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_1.pt` | 1 | 46.875 | 68.75 | 0.802616 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_10.pt` | 10 | 61.3281 | 62.50 | 0.645142 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_2.pt` | 2 | 47.6562 | 75.00 | 0.738971 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_3.pt` | 3 | 54.2969 | 43.75 | 0.693784 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_4.pt` | 4 | 57.8125 | 56.25 | 0.676086 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_5.pt` | 5 | 56.6406 | 56.25 | 0.666323 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_6.pt` | 6 | 58.9844 | 50.00 | 0.663749 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_7.pt` | 7 | 59.7656 | 81.25 | 0.657629 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_8.pt` | 8 | 58.2031 | 50.00 | 0.652876 |
| `checkpoints_noise25_1/mnist_amplitude/noise_low/mnist_amplitude_noise25_1_low_epoch_9.pt` | 9 | 60.9375 | 68.75 | 0.648652 |

### configs/noise_2.5.1/mnist_amplitude_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/best.pt` | 10 | 68.75 | 81.25 | 0.639501 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/last.pt` | 10 | 68.75 | 75.00 | 0.639501 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_1.pt` | 1 | 51.1719 | 68.75 | 0.67114 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_10.pt` | 10 | 68.75 | 87.50 | 0.639501 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_2.pt` | 2 | 58.9844 | 75.00 | 0.65796 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_3.pt` | 3 | 67.1875 | 75.00 | 0.652748 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_4.pt` | 4 | 66.7969 | 68.75 | 0.648848 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_5.pt` | 5 | 64.0625 | 68.75 | 0.651138 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_6.pt` | 6 | 66.7969 | 81.25 | 0.643413 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_7.pt` | 7 | 68.3594 | 56.25 | 0.641177 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_8.pt` | 8 | 67.1875 | 75.00 | 0.644974 |
| `checkpoints_noise25_1/mnist_amplitude/noise_mid/mnist_amplitude_noise25_1_mid_epoch_9.pt` | 9 | 67.1875 | 75.00 | 0.639872 |

### configs/noise_2.5.1/mnist_angle_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/mnist_angle/noise_high/best.pt` | 2 | 58.9844 | 50.00 | 0.680963 |
| `checkpoints_noise25_1/mnist_angle/noise_high/last.pt` | 5 | 58.9844 | 56.25 | 0.678997 |
| `checkpoints_noise25_1/mnist_angle/noise_high/mnist_angle_noise25_1_high_epoch_1.pt` | 1 | 42.9688 | 37.50 | 0.739325 |
| `checkpoints_noise25_1/mnist_angle/noise_high/mnist_angle_noise25_1_high_epoch_2.pt` | 2 | 58.9844 | 68.75 | 0.680963 |
| `checkpoints_noise25_1/mnist_angle/noise_high/mnist_angle_noise25_1_high_epoch_3.pt` | 3 | 58.9844 | 68.75 | 0.679348 |
| `checkpoints_noise25_1/mnist_angle/noise_high/mnist_angle_noise25_1_high_epoch_4.pt` | 4 | 58.9844 | 62.50 | 0.678134 |
| `checkpoints_noise25_1/mnist_angle/noise_high/mnist_angle_noise25_1_high_epoch_5.pt` | 5 | 58.9844 | 56.25 | 0.678997 |

### configs/noise_2.5.1/mnist_angle_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/mnist_angle/noise_low/best.pt` | 1 | 53.9062 | 43.75 | 0.691471 |
| `checkpoints_noise25_1/mnist_angle/noise_low/last.pt` | 5 | 53.9062 | 68.75 | 0.69005 |
| `checkpoints_noise25_1/mnist_angle/noise_low/mnist_angle_noise25_1_low_epoch_1.pt` | 1 | 53.9062 | 68.75 | 0.691471 |
| `checkpoints_noise25_1/mnist_angle/noise_low/mnist_angle_noise25_1_low_epoch_2.pt` | 2 | 53.9062 | 62.50 | 0.691521 |
| `checkpoints_noise25_1/mnist_angle/noise_low/mnist_angle_noise25_1_low_epoch_3.pt` | 3 | 53.9062 | 75.00 | 0.691819 |
| `checkpoints_noise25_1/mnist_angle/noise_low/mnist_angle_noise25_1_low_epoch_4.pt` | 4 | 53.9062 | 75.00 | 0.692476 |
| `checkpoints_noise25_1/mnist_angle/noise_low/mnist_angle_noise25_1_low_epoch_5.pt` | 5 | 53.9062 | 62.50 | 0.69005 |

### configs/noise_2.5.1/mnist_angle_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/mnist_angle/noise_mid/best.pt` | 3 | 48.8281 | 37.50 | 0.693979 |
| `checkpoints_noise25_1/mnist_angle/noise_mid/last.pt` | 5 | 45.7031 | 37.50 | 0.696338 |
| `checkpoints_noise25_1/mnist_angle/noise_mid/mnist_angle_noise25_1_mid_epoch_1.pt` | 1 | 48.0469 | 37.50 | 0.696099 |
| `checkpoints_noise25_1/mnist_angle/noise_mid/mnist_angle_noise25_1_mid_epoch_2.pt` | 2 | 47.6562 | 43.75 | 0.694508 |
| `checkpoints_noise25_1/mnist_angle/noise_mid/mnist_angle_noise25_1_mid_epoch_3.pt` | 3 | 48.8281 | 56.25 | 0.693979 |
| `checkpoints_noise25_1/mnist_angle/noise_mid/mnist_angle_noise25_1_mid_epoch_4.pt` | 4 | 48.0469 | 56.25 | 0.694299 |
| `checkpoints_noise25_1/mnist_angle/noise_mid/mnist_angle_noise25_1_mid_epoch_5.pt` | 5 | 45.7031 | 43.75 | 0.696338 |

### configs/noise_2.5.1/mnist_hybrid_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/best.pt` | 10 | 48.8281 | 50.00 | 0.687942 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/last.pt` | 10 | 48.8281 | 37.50 | 0.687942 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_1.pt` | 1 | 48.4375 | 62.50 | 0.689051 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_10.pt` | 10 | 48.8281 | 43.75 | 0.687942 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_2.pt` | 2 | 46.0938 | 56.25 | 0.689541 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_3.pt` | 3 | 48.0469 | 25.00 | 0.688507 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_4.pt` | 4 | 41.4062 | 37.50 | 0.689227 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_5.pt` | 5 | 39.0625 | 37.50 | 0.690691 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_6.pt` | 6 | 47.6562 | 25.00 | 0.688961 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_7.pt` | 7 | 40.2344 | 31.25 | 0.690174 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_8.pt` | 8 | 41.4062 | 31.25 | 0.687561 |
| `checkpoints_noise25_1/mnist_hybrid/noise_high/mnist_hybrid_noise25_1_high_epoch_9.pt` | 9 | 42.9688 | 43.75 | 0.689464 |

### configs/noise_2.5.1/mnist_hybrid_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/best.pt` | 3 | 57.0312 | 56.25 | 0.691404 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/last.pt` | 10 | 42.5781 | 43.75 | 0.687817 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_1.pt` | 1 | 48.0469 | 50.00 | 0.714613 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_10.pt` | 10 | 42.5781 | 31.25 | 0.687817 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_2.pt` | 2 | 48.4375 | 43.75 | 0.697022 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_3.pt` | 3 | 57.0312 | 43.75 | 0.691404 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_4.pt` | 4 | 39.8438 | 25.00 | 0.688308 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_5.pt` | 5 | 44.9219 | 31.25 | 0.68674 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_6.pt` | 6 | 40.625 | 43.75 | 0.686217 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_7.pt` | 7 | 43.3594 | 43.75 | 0.686627 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_8.pt` | 8 | 40.2344 | 43.75 | 0.685111 |
| `checkpoints_noise25_1/mnist_hybrid/noise_low/mnist_hybrid_noise25_1_low_epoch_9.pt` | 9 | 44.1406 | 18.75 | 0.685974 |

### configs/noise_2.5.1/mnist_hybrid_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/best.pt` | 6 | 52.7344 | 68.75 | 0.694466 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/last.pt` | 10 | 52.7344 | 50.00 | 0.69227 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_1.pt` | 1 | 47.6562 | 25.00 | 0.761088 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_10.pt` | 10 | 52.7344 | 50.00 | 0.69227 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_2.pt` | 2 | 40.625 | 43.75 | 0.717815 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_3.pt` | 3 | 36.3281 | 37.50 | 0.701933 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_4.pt` | 4 | 37.1094 | 43.75 | 0.694485 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_5.pt` | 5 | 51.1719 | 43.75 | 0.695885 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_6.pt` | 6 | 52.7344 | 43.75 | 0.694466 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_7.pt` | 7 | 52.7344 | 68.75 | 0.692175 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_8.pt` | 8 | 50.3906 | 75.00 | 0.692574 |
| `checkpoints_noise25_1/mnist_hybrid/noise_mid/mnist_hybrid_noise25_1_mid_epoch_9.pt` | 9 | 51.5625 | 62.50 | 0.693955 |

## Evaluation Results (noise_2)
所有 checkpoint 均在配置指定的测试 split 上重新推理得到下表的 `Eval Acc`。

### configs/noise_2/fashion_hybrid_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/fashion_hybrid/noise_high/best.pt` | 1 | 51.7333 | 45.31 | 0.691731 |
| `checkpoints_noise2/fashion_hybrid/noise_high/last.pt` | 1 | 51.7333 | 57.81 | 0.691731 |

### configs/noise_2/mnist_amplitude_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/mnist_amplitude/noise_high/best.pt` | 1 | 73.55 | 92.19 | 0.611148 |
| `checkpoints_noise2/mnist_amplitude/noise_high/last.pt` | 1 | 73.55 | 87.50 | 0.611148 |

### configs/noise_2/mnist_amplitude_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/mnist_amplitude/noise_low/best.pt` | 5 | 86.275 | 93.75 | 0.391461 |
| `checkpoints_noise2/mnist_amplitude/noise_low/last.pt` | 5 | 86.275 | 81.25 | 0.391461 |
| `checkpoints_noise2/mnist_amplitude/noise_low/mnist_amplitude_noise2_low_epoch_3.pt` | 3 | 83.775 | 84.38 | 0.460595 |
| `checkpoints_noise2/mnist_amplitude/noise_low/mnist_amplitude_noise2_low_epoch_5.pt` | 5 | 86.275 | 92.19 | 0.391461 |

### configs/noise_2/mnist_amplitude_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/mnist_amplitude/noise_mid/best.pt` | 1 | 59.175 | 68.75 | 0.71789 |
| `checkpoints_noise2/mnist_amplitude/noise_mid/last.pt` | 1 | 59.175 | 71.88 | 0.71789 |

### configs/noise_2/mnist_angle_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/mnist_angle/noise_high/best.pt` | 1 | 52.9667 | 51.56 | 0.693164 |
| `checkpoints_noise2/mnist_angle/noise_high/last.pt` | 5 | 51.5 | 56.25 | 0.693941 |
| `checkpoints_noise2/mnist_angle/noise_high/mnist_angle_noise2_high_epoch_3.pt` | 3 | 52.0333 | 43.75 | 0.693947 |
| `checkpoints_noise2/mnist_angle/noise_high/mnist_angle_noise2_high_epoch_5.pt` | 5 | 51.5 | 54.69 | 0.693941 |

### configs/noise_2/mnist_angle_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/mnist_angle/noise_low/best.pt` | 3 | 51.5 | 59.38 | 0.694136 |
| `checkpoints_noise2/mnist_angle/noise_low/last.pt` | 5 | 51.1 | 60.94 | 0.694449 |
| `checkpoints_noise2/mnist_angle/noise_low/mnist_angle_noise2_low_epoch_3.pt` | 3 | 51.5 | 48.44 | 0.694136 |
| `checkpoints_noise2/mnist_angle/noise_low/mnist_angle_noise2_low_epoch_5.pt` | 5 | 51.1 | 53.12 | 0.694449 |

### configs/noise_2/mnist_angle_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/mnist_angle/noise_mid/best.pt` | 3 | 53.6 | 50.00 | 0.691503 |
| `checkpoints_noise2/mnist_angle/noise_mid/last.pt` | 5 | 53.1333 | 50.00 | 0.691972 |
| `checkpoints_noise2/mnist_angle/noise_mid/mnist_angle_noise2_mid_epoch_3.pt` | 3 | 53.6 | 50.00 | 0.691503 |
| `checkpoints_noise2/mnist_angle/noise_mid/mnist_angle_noise2_mid_epoch_5.pt` | 5 | 53.1333 | 45.31 | 0.691972 |

### configs/noise_2/mnist_hybrid_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/mnist_hybrid/noise_high/best.pt` | 5 | 88.3 | 100.00 | 0.347643 |
| `checkpoints_noise2/mnist_hybrid/noise_high/last.pt` | 7 | 88.2667 | 89.06 | 0.350459 |
| `checkpoints_noise2/mnist_hybrid/noise_high/mnist_hybrid_noise2_high_epoch_3.pt` | 3 | 88.0667 | 93.75 | 0.365839 |
| `checkpoints_noise2/mnist_hybrid/noise_high/mnist_hybrid_noise2_high_epoch_5.pt` | 5 | 88.3 | 95.31 | 0.347643 |
| `checkpoints_noise2/mnist_hybrid/noise_high/mnist_hybrid_noise2_high_epoch_7.pt` | 7 | 88.2667 | 90.62 | 0.350459 |

### configs/noise_2/mnist_hybrid_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/mnist_hybrid/noise_low/best.pt` | 1 | 81.0333 | 89.06 | 0.451218 |
| `checkpoints_noise2/mnist_hybrid/noise_low/last.pt` | 1 | 81.0333 | 89.06 | 0.451218 |

### configs/noise_2/mnist_hybrid_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise2/mnist_hybrid/noise_mid/best.pt` | 1 | 83.5333 | 89.06 | 0.449816 |
| `checkpoints_noise2/mnist_hybrid/noise_mid/last.pt` | 1 | 83.5333 | 95.31 | 0.449816 |

## Evaluation Results (noise_2.1)
所有 checkpoint 均在配置指定的测试 split 上重新推理得到下表的 `Eval Acc`。

### configs/noise_2.1/mnist_angle_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_noise21/mnist_angle/noise_mid/best.pt` | 1 | 46.875 | 52.08 | 1.128392 |
| `checkpoints_noise21/mnist_angle/noise_mid/last.pt` | 4 | 46.875 | 52.08 | 0.761925 |
| `checkpoints_noise21/mnist_angle/noise_mid/mnist_angle_noise21_mid_epoch_2.pt` | 2 | 46.875 | 52.08 | 0.912451 |
| `checkpoints_noise21/mnist_angle/noise_mid/mnist_angle_noise21_mid_epoch_4.pt` | 4 | 46.875 | 52.08 | 0.761925 |

## Evaluation Results (CNN Baselines)
经典 CNN checkpoint 在对应测试集上的准确率如下：

### configs/cnn_baselines/mnist_angle_4x4.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise23/mnist_angle_4x4/best.pt` | 9 | 98.4375 | 93.75 | 0.031958 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_1.pt` | 1 | 90.625 | 89.06 | 0.236 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_10.pt` | 10 | 98.4375 | 96.88 | 0.028068 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_2.pt` | 2 | 96.875 | 93.75 | 0.080429 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_3.pt` | 3 | 95.3125 | 95.31 | 0.086385 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_4.pt` | 4 | 93.75 | 92.19 | 0.053663 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_5.pt` | 5 | 95.3125 | 93.75 | 0.057769 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_6.pt` | 6 | 93.75 | 92.19 | 0.097662 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_7.pt` | 7 | 96.875 | 92.19 | 0.034695 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_8.pt` | 8 | 96.875 | 95.31 | 0.086732 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/cnn_mnist_angle_4x4_epoch_9.pt` | 9 | 98.4375 | 93.75 | 0.031958 |
| `checkpoints_cnn/noise23/mnist_angle_4x4/last.pt` | 10 | 98.4375 | 96.88 | 0.028068 |

### configs/cnn_baselines/mnist_angle_4x4_noise.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/best.pt` | 5 | 100.0 | 95.31 | 0.025996 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_1.pt` | 1 | 98.4375 | 98.44 | 0.26192 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_10.pt` | 10 | 98.4375 | 100.00 | 0.006217 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_2.pt` | 2 | 98.4375 | 100.00 | 0.060488 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_3.pt` | 3 | 98.4375 | 96.88 | 0.060492 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_4.pt` | 4 | 98.4375 | 100.00 | 0.020971 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_5.pt` | 5 | 100.0 | 95.31 | 0.025996 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_6.pt` | 6 | 98.4375 | 98.44 | 0.03259 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_7.pt` | 7 | 98.4375 | 93.75 | 0.028488 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_8.pt` | 8 | 98.4375 | 100.00 | 0.043328 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/cnn_mnist_angle_4x4_noise_epoch_9.pt` | 9 | 98.4375 | 98.44 | 0.012004 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_noise/last.pt` | 10 | 98.4375 | 100.00 | 0.006217 |

### configs/cnn_baselines/mnist_angle_4x4_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/best.pt` | 10 | 98.4375 | 100.00 | 0.026759 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_1.pt` | 1 | 93.75 | 95.31 | 0.273476 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_10.pt` | 10 | 98.4375 | 100.00 | 0.026759 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_2.pt` | 2 | 96.875 | 100.00 | 0.10385 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_3.pt` | 3 | 93.75 | 90.62 | 0.047976 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_4.pt` | 4 | 92.1875 | 100.00 | 0.055465 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_5.pt` | 5 | 93.75 | 100.00 | 0.045956 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_6.pt` | 6 | 96.875 | 100.00 | 0.046839 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_7.pt` | 7 | 93.75 | 96.88 | 0.044268 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_8.pt` | 8 | 96.875 | 96.88 | 0.069851 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/cnn_mnist_angle_4x4_noise_high_epoch_9.pt` | 9 | 96.875 | 100.00 | 0.038194 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_high/last.pt` | 10 | 98.4375 | 100.00 | 0.026759 |

### configs/cnn_baselines/mnist_angle_4x4_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/best.pt` | 2 | 98.4375 | 95.31 | 0.160787 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_1.pt` | 1 | 59.375 | 62.50 | 0.283554 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_10.pt` | 10 | 98.4375 | 96.88 | 0.086791 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_2.pt` | 2 | 98.4375 | 95.31 | 0.160787 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_3.pt` | 3 | 98.4375 | 100.00 | 0.07071 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_4.pt` | 4 | 95.3125 | 92.19 | 0.082488 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_5.pt` | 5 | 98.4375 | 98.44 | 0.038796 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_6.pt` | 6 | 98.4375 | 98.44 | 0.019419 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_7.pt` | 7 | 98.4375 | 98.44 | 0.036852 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_8.pt` | 8 | 96.875 | 98.44 | 0.038629 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/cnn_mnist_angle_4x4_noise_low_epoch_9.pt` | 9 | 98.4375 | 98.44 | 0.03334 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_low/last.pt` | 10 | 98.4375 | 96.88 | 0.086791 |

### configs/cnn_baselines/mnist_angle_4x4_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/best.pt` | 8 | 100.0 | 95.31 | 0.069357 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_1.pt` | 1 | 96.875 | 93.75 | 0.195386 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_10.pt` | 10 | 98.4375 | 93.75 | 0.03536 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_2.pt` | 2 | 98.4375 | 93.75 | 0.100482 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_3.pt` | 3 | 98.4375 | 95.31 | 0.070562 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_4.pt` | 4 | 98.4375 | 93.75 | 0.074515 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_5.pt` | 5 | 98.4375 | 93.75 | 0.055359 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_6.pt` | 6 | 98.4375 | 93.75 | 0.022148 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_7.pt` | 7 | 98.4375 | 93.75 | 0.079508 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_8.pt` | 8 | 100.0 | 95.31 | 0.069357 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/cnn_mnist_angle_4x4_noise_mid_epoch_9.pt` | 9 | 98.4375 | 95.31 | 0.027846 |
| `checkpoints_cnn/noise23/mnist_angle_4x4_mid/last.pt` | 10 | 98.4375 | 93.75 | 0.03536 |

### configs/cnn_baselines/mnist_angle_8x8.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise24/mnist_angle_8x8/best.pt` | 5 | 100.0 | 100.00 | 0.01303 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_1.pt` | 1 | 95.3125 | 98.44 | 0.179918 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_10.pt` | 10 | 100.0 | 98.44 | 0.001744 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_2.pt` | 2 | 96.875 | 95.31 | 0.037362 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_3.pt` | 3 | 96.875 | 95.31 | 0.021522 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_4.pt` | 4 | 93.75 | 87.50 | 0.087174 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_5.pt` | 5 | 100.0 | 100.00 | 0.01303 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_6.pt` | 6 | 100.0 | 98.44 | 0.01362 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_7.pt` | 7 | 100.0 | 98.44 | 0.007796 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_8.pt` | 8 | 100.0 | 100.00 | 0.003823 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/cnn_mnist_angle_8x8_epoch_9.pt` | 9 | 100.0 | 98.44 | 0.001152 |
| `checkpoints_cnn/noise24/mnist_angle_8x8/last.pt` | 10 | 100.0 | 98.44 | 0.001744 |

### configs/cnn_baselines/mnist_angle_8x8_noise.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/best.pt` | 1 | 100.0 | 95.31 | 0.332724 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_1.pt` | 1 | 100.0 | 95.31 | 0.332724 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_10.pt` | 10 | 100.0 | 100.00 | 0.015457 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_2.pt` | 2 | 100.0 | 96.88 | 0.052208 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_3.pt` | 3 | 100.0 | 100.00 | 0.051594 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_4.pt` | 4 | 100.0 | 100.00 | 0.018225 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_5.pt` | 5 | 100.0 | 98.44 | 0.014383 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_6.pt` | 6 | 98.4375 | 100.00 | 0.005799 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_7.pt` | 7 | 100.0 | 100.00 | 0.005558 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_8.pt` | 8 | 100.0 | 100.00 | 0.004467 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/cnn_mnist_angle_8x8_noise_epoch_9.pt` | 9 | 100.0 | 100.00 | 0.018079 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_noise/last.pt` | 10 | 100.0 | 100.00 | 0.015457 |

### configs/cnn_baselines/mnist_angle_8x8_noise_high.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/best.pt` | 2 | 100.0 | 100.00 | 0.046314 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_1.pt` | 1 | 95.3125 | 100.00 | 0.223373 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_10.pt` | 10 | 98.4375 | 100.00 | 0.001843 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_2.pt` | 2 | 100.0 | 100.00 | 0.046314 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_3.pt` | 3 | 96.875 | 100.00 | 0.026303 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_4.pt` | 4 | 96.875 | 100.00 | 0.01509 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_5.pt` | 5 | 92.1875 | 98.44 | 0.01007 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_6.pt` | 6 | 93.75 | 100.00 | 0.01789 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_7.pt` | 7 | 98.4375 | 100.00 | 0.004673 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_8.pt` | 8 | 98.4375 | 100.00 | 0.005233 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/cnn_mnist_angle_8x8_noise_high_epoch_9.pt` | 9 | 100.0 | 100.00 | 0.003212 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_high/last.pt` | 10 | 98.4375 | 100.00 | 0.001843 |

### configs/cnn_baselines/mnist_angle_8x8_noise_low.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/best.pt` | 8 | 100.0 | 98.44 | 0.00121 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_1.pt` | 1 | 96.875 | 96.88 | 0.153212 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_10.pt` | 10 | 98.4375 | 98.44 | 0.000644 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_2.pt` | 2 | 98.4375 | 96.88 | 0.073445 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_3.pt` | 3 | 98.4375 | 98.44 | 0.050515 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_4.pt` | 4 | 98.4375 | 98.44 | 0.028788 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_5.pt` | 5 | 96.875 | 96.88 | 0.013808 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_6.pt` | 6 | 98.4375 | 96.88 | 0.005871 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_7.pt` | 7 | 98.4375 | 98.44 | 0.00369 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_8.pt` | 8 | 100.0 | 98.44 | 0.00121 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/cnn_mnist_angle_8x8_noise_low_epoch_9.pt` | 9 | 98.4375 | 98.44 | 0.000766 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_low/last.pt` | 10 | 98.4375 | 98.44 | 0.000644 |

### configs/cnn_baselines/mnist_angle_8x8_noise_mid.yaml
| Checkpoint | Epoch | Train Acc (%) | Eval Acc (%) | Loss |
| --- | --- | --- | --- | --- |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/best.pt` | 3 | 100.0 | 96.88 | 0.034541 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_1.pt` | 1 | 98.4375 | 100.00 | 0.271324 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_10.pt` | 10 | 100.0 | 96.88 | 0.035998 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_2.pt` | 2 | 98.4375 | 100.00 | 0.091079 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_3.pt` | 3 | 100.0 | 96.88 | 0.034541 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_4.pt` | 4 | 96.875 | 100.00 | 0.022123 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_5.pt` | 5 | 100.0 | 96.88 | 0.014608 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_6.pt` | 6 | 98.4375 | 100.00 | 0.008671 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_7.pt` | 7 | 100.0 | 98.44 | 0.009061 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_8.pt` | 8 | 100.0 | 98.44 | 0.004723 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/cnn_mnist_angle_8x8_noise_mid_epoch_9.pt` | 9 | 96.875 | 100.00 | 0.006387 |
| `checkpoints_cnn/noise24/mnist_angle_8x8_mid/last.pt` | 10 | 100.0 | 96.88 | 0.035998 |
