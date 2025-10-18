# 🚀 CPU 实例快速启动指南

**适用场景**：从 GPU 实例迁移到高性能多核 CPU 实例

---

## 一键配置（推荐）

```bash
# 1. 克隆项目
git clone git@github.com:NikolaStarx/QCNN_project.git
cd QCNN_project

# 2. 切换到工作分支
git checkout feature/performance-optimization

# 3. 运行自动配置脚本
bash setup_cpu_env.sh

# 4. 激活虚拟环境
source qcnn_env/bin/activate

# 5. 开始训练！
python train_optimized.py --config configs/mnist_angle_fast.yaml
```

---

## 手动配置（如果自动脚本失败）

```bash
# 1-2. 克隆并切换分支（同上）
git clone git@github.com:NikolaStarx/QCNN_project.git
cd QCNN_project
git checkout feature/performance-optimization

# 3. 创建虚拟环境
python3 -m venv qcnn_env
source qcnn_env/bin/activate

# 4. 修改依赖文件
sed -i 's/qiskit-aer-gpu/qiskit-aer/g' requirements.txt

# 5. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 6. 修改配置文件
find configs/ -name "*.yaml" -exec sed -i 's/backend: GPU/backend: CPU/g' {} \;

# 7. 验证环境
python -c "import torch, qiskit; print('✅ Ready!')"
```

---

## 常用训练命令

### 快速测试（1-2分钟）
```bash
python train_optimized.py --config configs/mnist_angle_fast.yaml
```

### 完整训练
```bash
# Amplitude 编码
python train_optimized.py --config configs/full_scale/mnist_amplitude_full.yaml

# Angle 编码
python train_optimized.py --config configs/full_scale/mnist_angle_full.yaml

# Hybrid 编码
python train_optimized.py --config configs/full_scale/mnist_hybrid_full.yaml
```

### 噪声实验（18 个配置）
```bash
# 低噪声
python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_low.yaml

# 中等噪声
python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_mid.yaml

# 高噪声
python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_high.yaml
```

### 恢复训练（从检查点继续）
```bash
python train_optimized.py --config <config_file> --resume
```

---

## 多任务并行训练

### 使用 tmux（推荐）

```bash
# 创建第一个训练会话
tmux new -s train1
python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_low.yaml
# 按 Ctrl+B, 然后按 D 分离会话

# 创建第二个训练会话
tmux new -s train2
python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_mid.yaml
# 按 Ctrl+B, D 分离

# 查看所有会话
tmux ls

# 重新连接到会话
tmux attach -t train1

# 关闭会话
tmux kill-session -t train1
```

### 使用 nohup（后台运行）

```bash
# 创建日志目录
mkdir -p logs

# 启动多个后台训练
nohup python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_low.yaml > logs/low.log 2>&1 &
nohup python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_mid.yaml > logs/mid.log 2>&1 &
nohup python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_high.yaml > logs/high.log 2>&1 &

# 查看运行中的任务
ps aux | grep train_optimized

# 查看实时日志
tail -f logs/low.log

# 停止后台任务
pkill -f train_optimized
```

---

## 性能监控

### 查看 CPU 使用率
```bash
# 实时监控
htop

# 或使用 top
top -u $USER

# 查看 CPU 核心数
nproc
```

### 查看训练进程
```bash
# 查看所有 Python 训练进程
ps aux | grep train_optimized

# 查看特定进程的 CPU/内存使用
ps -p <PID> -o %cpu,%mem,etime,cmd
```

### 查看检查点
```bash
# 列出所有检查点
ls -lh checkpoints/

# 查看特定实验的检查点
ls -lh checkpoints/mnist_amplitude/
```

---

## 故障排查

### ❌ 问题：`No module named 'qiskit_aer'`
```bash
# 检查是否安装了正确的包
pip list | grep qiskit

# 重新安装
pip install qiskit-aer
```

### ❌ 问题：内存不足 (OOM)
在配置文件中减小批次大小：
```yaml
data:
  batch_size: 8  # 从 32 降到 8
  num_train_samples: 5000  # 减少样本数
```

### ❌ 问题：训练速度慢
```bash
# 检查 CPU 核心数
nproc

# 设置 PyTorch 使用所有核心（在训练脚本开头添加）
# torch.set_num_threads(32)  # 改为您的核心数
```

---

## 检查点同步（可选）

如果需要从旧实例同步检查点：

### 在旧实例上
```bash
# 打包检查点
tar -czf checkpoints_backup.tar.gz checkpoints/

# 上传到云存储或使用 scp 传输
# 示例：scp checkpoints_backup.tar.gz user@new-instance:/path/to/QCNN_project/
```

### 在新实例上
```bash
# 下载并解压
tar -xzf checkpoints_backup.tar.gz

# 使用 --resume 继续训练
python train_optimized.py --config <config_file> --resume
```

---

## 文件结构参考

```
QCNN_project/
├── train.py                    # 原始训练脚本
├── train_optimized.py          # 优化版训练脚本（推荐）
├── requirements.txt            # 依赖列表
├── setup_cpu_env.sh           # 自动配置脚本
├── configs/
│   ├── mnist_angle_fast.yaml  # 快速测试配置
│   ├── full_scale/            # 完整训练配置（6个）
│   └── config_noise/          # 噪声实验配置（18个）
├── models/
│   ├── qcnn.py                # 原始模型
│   └── qcnn_optimized.py      # 优化模型
├── checkpoints/               # 训练检查点
├── logs/                      # 训练日志
└── AI_logs/                   # 迁移指南和文档
    └── environment_migration_guide.md  # 详细迁移指南
```

---

## 获取帮助

- **详细迁移指南**: `AI_logs/environment_migration_guide.md`
- **项目架构**: `CLAUDE.md`
- **完整文档**: `README.md`

---

**最后更新**: 2025-10-18
