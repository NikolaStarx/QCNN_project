#!/bin/bash
# 自动化环境配置脚本：CPU 实例快速配置
# 使用方法：bash setup_cpu_env.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "QCNN 项目 CPU 环境配置脚本"
echo "=========================================="
echo ""

# 1. 检查 Python 版本
echo "Step 1: 检查 Python 版本..."
PYTHON_VERSION=$(python3 --version)
echo "✅ $PYTHON_VERSION"
echo ""

# 2. 创建虚拟环境
echo "Step 2: 创建 Python 虚拟环境..."
if [ -d "qcnn_env" ]; then
    echo "⚠️  虚拟环境已存在，跳过创建"
else
    python3 -m venv qcnn_env
    echo "✅ 虚拟环境创建成功"
fi
echo ""

# 3. 激活虚拟环境
echo "Step 3: 激活虚拟环境..."
source qcnn_env/bin/activate
echo "✅ 虚拟环境已激活"
echo ""

# 4. 升级 pip
echo "Step 4: 升级 pip..."
pip install --upgrade pip -q
echo "✅ pip 升级完成"
echo ""

# 5. 修改 requirements.txt（将 GPU 版本改为 CPU 版本）
echo "Step 5: 修改 requirements.txt（GPU → CPU）..."
if grep -q "qiskit-aer-gpu" requirements.txt; then
    sed -i 's/qiskit-aer-gpu/qiskit-aer/g' requirements.txt
    echo "✅ 已将 qiskit-aer-gpu 替换为 qiskit-aer"
else
    echo "✅ requirements.txt 已经是 CPU 版本"
fi
echo ""

# 6. 安装依赖
echo "Step 6: 安装项目依赖（这可能需要几分钟）..."
pip install -r requirements.txt -q
echo "✅ 依赖安装完成"
echo ""

# 7. 批量修改配置文件为 CPU 后端
echo "Step 7: 批量修改配置文件（GPU → CPU backend）..."
MODIFIED_COUNT=$(find configs/ -name "*.yaml" -type f -exec grep -l "backend: GPU" {} \; | wc -l)
if [ "$MODIFIED_COUNT" -gt 0 ]; then
    find configs/ -name "*.yaml" -type f -exec sed -i 's/backend: GPU/backend: CPU/g' {} \;
    echo "✅ 已修改 $MODIFIED_COUNT 个配置文件"
else
    echo "✅ 配置文件已经是 CPU 后端"
fi
echo ""

# 8. 验证环境
echo "Step 8: 验证环境..."
python3 << 'VERIFY'
import sys
try:
    import torch
    import qiskit
    import qiskit_aer
    import yaml
    import numpy as np
    from torchvision import datasets

    print(f"  ✅ PyTorch: {torch.__version__}")
    print(f"  ✅ Qiskit: {qiskit.__version__}")
    print(f"  ✅ Qiskit-Aer: {qiskit_aer.__version__}")
    print(f"  ✅ NumPy: {np.__version__}")
    print(f"  ✅ CPU 核心数: {torch.get_num_threads()} threads")
    print("")
    print("✅ 所有依赖包验证通过！")
except Exception as e:
    print(f"❌ 验证失败: {e}")
    sys.exit(1)
VERIFY
echo ""

# 9. 创建必要的目录
echo "Step 9: 创建必要的目录..."
mkdir -p logs checkpoints data/raw data/processed
echo "✅ 目录创建完成"
echo ""

# 10. 显示 Git 状态
echo "Step 10: Git 仓库状态..."
echo "当前分支: $(git branch --show-current)"
echo "最新提交: $(git log -1 --oneline)"
echo ""

echo "=========================================="
echo "✅ 环境配置完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo "  1. 激活虚拟环境：source qcnn_env/bin/activate"
echo "  2. 运行快速测试："
echo "     python train_optimized.py --config configs/mnist_angle_fast.yaml"
echo "  3. 运行完整训练："
echo "     python train_optimized.py --config configs/full_scale/mnist_amplitude_full.yaml"
echo ""
echo "多任务并行训练示例："
echo "  tmux new -s train1"
echo "  python train_optimized.py --config configs/config_noise/mnist_amplitude_noise_low.yaml"
echo "  # 按 Ctrl+B, D 分离会话"
echo ""
