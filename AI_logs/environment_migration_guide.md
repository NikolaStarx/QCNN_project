# 环境迁移指南：从 GPU 实例迁移到高性能 CPU 实例

**目标**：在新的高性能多核 CPU 实例上快速复现当前训练环境。

**日期**：2025-10-18

---

## 一、当前环境配置信息

### 系统环境
- **操作系统**：Ubuntu 24.04.3 LTS
- **Python 版本**：3.12.11

### 核心依赖版本
```
torch==2.9.0
torchvision==0.24.0
qiskit==1.4.5
qiskit-aer-gpu==0.15.1  # ⚠️ 在 CPU 实例上需要改为 qiskit-aer
qiskit-machine-learning==0.8.4
numpy==2.3.4
pyyaml
```

### Git 仓库信息
- **远程仓库**：`git@github.com:NikolaStarx/QCNN_project.git`
- **工作分支**：`feature/performance-optimization`
- **主分支**：`main`

---

## 二、迁移步骤（新实例操作清单）

### Step 1: 配置 SSH 密钥（如果新实例还没配置）

```bash
# 1. 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 查看公钥并添加到 GitHub
cat ~/.ssh/id_ed25519.pub
# 复制输出内容，到 GitHub Settings > SSH Keys 添加

# 3. 测试连接
ssh -T git@github.com
```

### Step 2: 克隆项目并切换到工作分支

```bash
# 1. 克隆仓库
git clone git@github.com:NikolaStarx/QCNN_project.git
cd QCNN_project

# 2. 切换到性能优化分支
git checkout feature/performance-optimization

# 3. 验证分支状态
git status
git log -1  # 查看最新提交
```

### Step 3: 配置 Python 环境

#### 方案 A：使用 venv（推荐）

```bash
# 1. 创建虚拟环境
python3 -m venv qcnn_env

# 2. 激活虚拟环境
source qcnn_env/bin/activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装依赖（⚠️ 注意修改 requirements.txt）
# 先手动修改 requirements.txt，将 qiskit-aer-gpu 改为 qiskit-aer
sed -i 's/qiskit-aer-gpu/qiskit-aer/g' requirements.txt

# 5. 安装所有依赖
pip install -r requirements.txt
```

#### 方案 B：使用 conda（如果可用）

```bash
# 1. 创建 conda 环境
conda create -n qcnn_env python=3.12 -y
conda activate qcnn_env

# 2. 安装 PyTorch（CPU 版本）
conda install pytorch torchvision cpuonly -c pytorch -y

# 3. 修改并安装其他依赖
sed -i 's/qiskit-aer-gpu/qiskit-aer/g' requirements.txt
pip install -r requirements.txt
```

### Step 4: 验证环境

```bash
# 运行验证脚本
python -c "
import torch
import qiskit
import qiskit_aer
import yaml
import numpy as np
from torchvision import datasets

print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ Qiskit version: {qiskit.__version__}')
print(f'✅ Qiskit-Aer version: {qiskit_aer.__version__}')
print(f'✅ NumPy version: {np.__version__}')
print(f'✅ PyTorch device: {torch.device(\"cpu\")}')
print(f'✅ CPU count: {torch.get_num_threads()} threads')
print('✅ All dependencies loaded successfully!')
"
```

### Step 5: 修改配置文件以适配 CPU 训练

**重要**：所有配置文件中的 `environment.backend` 需要从 `GPU` 改为 `CPU`。

#### 方案 A：批量修改（推荐）

```bash
# 批量将所有配置文件中的 GPU 后端改为 CPU
find configs/ -name "*.yaml" -type f -exec sed -i 's/backend: GPU/backend: CPU/g' {} \;

# 验证修改
grep -r "backend:" configs/ | head -10
```

#### 方案 B：手动修改配置文件

编辑您要使用的配置文件（例如 `configs/full_scale/mnist_amplitude_full.yaml`）：

```yaml
environment:
  backend: CPU  # ← 从 GPU 改为 CPU
  add_noise: false
```

### Step 6: 启动训练

```bash
# 使用原始模型训练（如果需要）
python train.py --config configs/full_scale/mnist_amplitude_full.yaml

# 或使用优化版本训练（推荐）
python train_optimized.py --config configs/full_scale/mnist_amplitude_full.yaml

# 如果需要恢复之前的检查点，添加 --resume 参数
python train_optimized.py --config configs/full_scale/mnist_amplitude_full.yaml --resume
```

### Step 7: 利用多核 CPU 并行训练（可选）

在高性能多核 CPU 实例上，可以同时运行多个训练任务：

```bash
# 方案 A：使用 tmux 或 screen 分别启动多个训练
tmux new -s train1
python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_low.yaml
# Ctrl+B, D 分离会话

tmux new -s train2
python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_mid.yaml
# Ctrl+B, D 分离会话

# 查看所有会话
tmux ls

# 重新连接会话
tmux attach -t train1
```

```bash
# 方案 B：使用 nohup 后台运行
nohup python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_low.yaml > logs/low.log 2>&1 &
nohup python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_mid.yaml > logs/mid.log 2>&1 &
nohup python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_high.yaml > logs/high.log 2>&1 &

# 查看日志
tail -f logs/low.log
```

---

## 三、关键注意事项

### 1. 依赖包差异

| GPU 实例 | CPU 实例 | 说明 |
|---------|---------|------|
| `qiskit-aer-gpu` | `qiskit-aer` | ⚠️ 必须更换，否则会报错 |
| `torch` (CUDA) | `torch` (CPU) | PyTorch 会自动适配 |

### 2. 配置文件修改

**必须修改的字段**：
```yaml
environment:
  backend: CPU  # GPU → CPU
```

**可选修改的字段**（如果 CPU 内存不足）：
```yaml
data:
  batch_size: 16  # 从 32 降到 16
  num_train_samples: 10000  # 减少样本数
```

### 3. 性能优化建议（CPU 实例）

#### a. 设置 PyTorch 线程数
在训练脚本开头添加：
```python
import torch
torch.set_num_threads(32)  # 设置为 CPU 核心数
```

#### b. 启用 Qiskit-Aer 的多线程
检查 Qiskit-Aer 是否使用了所有 CPU 核心：
```python
from qiskit_aer import AerSimulator
backend = AerSimulator()
print(backend.configuration().max_parallel_threads)  # 应该接近 CPU 核心数
```

#### c. 避免内存溢出
- 减小 `batch_size`
- 减少 `num_train_samples`
- 使用 `--resume` 参数分阶段训练

### 4. 检查点同步

如果您想从当前 GPU 实例的训练进度继续训练：

```bash
# 在旧实例上，将检查点目录打包
tar -czf checkpoints_backup.tar.gz checkpoints/

# 传输到新实例（使用 scp、云存储等）
# 例如：上传到云存储，然后在新实例下载

# 在新实例上解压
tar -xzf checkpoints_backup.tar.gz

# 使用 --resume 参数继续训练
python train_optimized.py --config <config_file> --resume
```

---

## 四、验证清单

在新实例上完成迁移后，请逐项检查：

- [ ] Python 版本正确（3.12.x）
- [ ] 所有依赖包安装成功
- [ ] Git 分支切换到 `feature/performance-optimization`
- [ ] 配置文件中 `backend` 改为 `CPU`
- [ ] `requirements.txt` 中 `qiskit-aer-gpu` 改为 `qiskit-aer`
- [ ] 能够成功导入 `torch`, `qiskit`, `qiskit_aer`
- [ ] 至少运行一次快速测试（例如 `mnist_amplitude_fast.yaml`）
- [ ] 检查点目录已同步（如果需要继续之前的训练）

---

## 五、快速命令速查表

```bash
# 1. 克隆并切换分支
git clone git@github.com:NikolaStarx/QCNN_project.git && cd QCNN_project
git checkout feature/performance-optimization

# 2. 创建并激活虚拟环境
python3 -m venv qcnn_env && source qcnn_env/bin/activate

# 3. 修改依赖并安装
sed -i 's/qiskit-aer-gpu/qiskit-aer/g' requirements.txt
pip install --upgrade pip && pip install -r requirements.txt

# 4. 批量修改配置文件为 CPU 后端
find configs/ -name "*.yaml" -type f -exec sed -i 's/backend: GPU/backend: CPU/g' {} \;

# 5. 验证环境
python -c "import torch, qiskit, qiskit_aer; print('✅ Environment ready!')"

# 6. 启动训练
python train_optimized.py --config configs/full_scale/mnist_amplitude_full.yaml
```

---

## 六、故障排查

### 问题 1: `ModuleNotFoundError: No module named 'qiskit_aer'`
**解决**：检查是否已将 `qiskit-aer-gpu` 改为 `qiskit-aer` 并重新安装。

### 问题 2: 训练速度仍然很慢
**解决**：
- 确认 CPU 核心数：`lscpu | grep "^CPU(s):"`
- 设置 PyTorch 线程数：`torch.set_num_threads(<核心数>)`
- 检查 Qiskit-Aer 是否使用多线程

### 问题 3: 内存不足 (OOM)
**解决**：
- 减小 `batch_size`（例如从 32 → 16 → 8）
- 减少 `num_train_samples`
- 使用 swap 空间（不推荐，会很慢）

### 问题 4: Git 克隆失败
**解决**：
- 检查 SSH 密钥是否添加到 GitHub
- 使用 HTTPS 克隆：`git clone https://github.com/NikolaStarx/QCNN_project.git`

---

## 七、联系与支持

如遇到问题，请检查：
1. 项目 README: `/workspace/QCNN_project/README.md`
2. Claude Code 指南: `/workspace/QCNN_project/CLAUDE.md`
3. 本指南所在目录: `/workspace/QCNN_project/AI_logs/`

**最后更新**：2025-10-18
