# Workflow Notes for Continuing the Project

这份笔记帮助你在新的环境上无缝接续现在的工作：准备依赖、重跑/追加评测、维护 `config_checkpoint_catalog.md`，以及把结果用于论文撰写，避免重复跑数浪费时间。

---
## 1. 环境准备

1. **Python & 依赖**  
   - 推荐 Python ≥ 3.10。  
   - 安装依赖：
     ```bash
     pip install -r requirements.txt
     ```
   - 主要依赖：PyTorch、Qiskit Aer、torchvision、tqdm 等。

2. **数据预处理**  
   QCNN 模型依赖已处理的数据。首次在新机器上运行前务必执行：
   ```bash
   python scripts/preprocess_downsampled.py
   python scripts/preprocess.py mnist amplitude --num_qubits 10
   python scripts/preprocess.py fashion_mnist amplitude --num_qubits 10
   ```
   - `preprocess_downsampled.py` 生成 4×4/8×8 特征，供 angle/hybrid/downsampled 任务使用。  
   - `preprocess.py` 生成 amplitude 编码所需的张量；如需不同 qubit 数，请调整 `--num_qubits`。

3. **工具脚本总览**
   - `scripts/eval_checkpoints.py`：量子 checkpoint 评测。  
   - `scripts/eval_cnn_checkpoints.py`：经典 CNN baseline 评测。  
   - `scripts/preprocess*.py`：数据预处理。  
   - 其他训练脚本如 `train_noise.py`, `train_cnn.py` 仅在需要重新训练时使用。

---
## 2. Checkpoint 评测

### 2.1 量子模型 `scripts/eval_checkpoints.py`

示例命令：
```bash
python scripts/eval_checkpoints.py \
  --prefix checkpoints_noise23/mnist_hybrid/noise_high \
  --output AI_logs/noise23_25_eval_results.json \
  --max-checkpoints 12 \
  --max-samples 64
```
- `--prefix`：一个或多个 checkpoint 目录前缀，可指向具体子目录或上级目录。脚本会自动发现对应的 YAML 配置。
- `--output`：评测结果写入的 JSON 文件。若文件已存在，脚本会自动追加新结果，不会覆盖。
- `--max-checkpoints`：限制本次新增评测的 checkpoint 数，数值 0 表示不限。适用于 64 qubit 等耗时场景。
- `--max-samples`：限制评测时使用的测试样本数量。默认全量；若想减小耗时可设为 16/32 等。

脚本逻辑：
- 优先使用 checkpoint 内保存的 `config`（若存在），确保模型结构与训练时一致，避免参数维度不匹配。
- 评测完成后条目会写入对应 JSON（字段：config、checkpoint、epoch、recorded_accuracy、evaluated_test_accuracy、loss）。
- JSON 中已有的 checkpoint 会被自动跳过，无需重复跑数。

常用 JSON：
- `AI_logs/noise26_eval_results.json` – noise_2.6 & noise_2.6.3 
- `AI_logs/noise23_25_eval_results.json` – noise_2.3 / 2.3.1 / 2.4.1 / 2.5.1 
- `AI_logs/noise2_eval_results.json` – noise_2 
- `AI_logs/noise21_eval_results.json` – noise_2.1 系列 
- `AI_logs/checkpoints_eval_results.json` – `checkpoints/` 根目录（全规模 & lightweight & profiling）。

### 2.2 经典 CNN `scripts/eval_cnn_checkpoints.py`

示例命令：
```bash
python scripts/eval_cnn_checkpoints.py \
  --prefix checkpoints_cnn \
  --output AI_logs/cnn_eval_results.json
```
- 会推断输入尺寸，构造 `models/classical_cnn.ClassicalCNN`，并在测试集上输出准确率。  
- 支持多个前缀；多次运行可追加更多结果。

### 2.3 避免重复评测
- 在跑脚本前先检查对应 JSON 是否已有该 checkpoint（`grep` 或简单加载 JSON）。  
- 如果结果已存在且不需要更新（如 `--max-samples`/`--max-checkpoints` 参数不变），可直接引用现有 JSON，而无需重新运行脚本。

---
## 3. `AI_logs/config_checkpoint_catalog.md` 的作用与维护

`config_checkpoint_catalog.md` 是整个项目的“索引”：
1. **配置总览**：自动遍历 `configs/`，用表格列出每个 YAML 的数据集、编码、样本规模、噪声参数、`checkpoint_dir` 等信息，快速定位实验设置。
2. **评测结果汇总**：将各 JSON 中的评测结果按套件分别整理成表格（包括 noise 系列、`checkpoints/` 根目录和 CNN baseline）。

**使用建议**：
- 写论文或查参数时，先查 catalog，能快速知道某个结果如何得到、是否已经评测，无需再回头打开 YAML / JSON。  
- 若 catalog 中已有表格，就说明对应 JSON 已经带有评测结果，无需再次运行脚本。
- 只有新增 checkpoint 或重新评估（例如改了 `--max-samples` 想要更精确结果）才需要跑脚本并更新 JSON，然后重建 catalog。

**如何重建 catalog**：
运行下述脚本（仓库中已包含）：
```bash
python - <<'PY'
# 生成 catalog 的脚本片段在 Workflow Notes 中，也可单独保存为脚本运行
...
PY
```
或将该片段另存为 `scripts/generate_catalog.py` 后执行。运行后 `AI_logs/config_checkpoint_catalog.md` 会被覆盖更新。

---
## 4. 论文写作衔接

1. **章节拆分**：`paper/` 目录下的 `chapter*_*.md` 已按章节拆好。继续写时直接编辑对应文件（Introduction、Literature Review、Methodology、Experiments、Evaluation 等）。
2. **数据引用**：
   - 加载 JSON（例如 `AI_logs/noise23_25_eval_results.json`）即可导出表格或绘图。
   - catalog 表格提供了 config ↔ checkpoint 的映射，写作时无需再查 YAML。
   - 图表建议放在 `paper_assets/`，命名清晰，方便引用。
3. **写作流程建议**：
   - 对每个章节标记需要引用的实验集，确认对应 JSON 已有数据。
   - 使用 pandas/matplotlib/Seaborn 等工具从 JSON 导出 csv/图片，减少人工复制。
   - 论文中引用表格时可直接指向 catalog 中的表格编号或 JSON 原始值。
4. **检查清单**：
   - 引用数据前确认 JSON 时间戳或最新评测条数。
   - 若新增评测，先更新 catalog，保证写作时引用的是最新结构化表格。
   - 记录图/表的来源脚本或 notebook，方便复现。

---
## 5. FAQ

- **评测太慢怎么办？** 调小 `--max-samples` 或分批 `--max-checkpoints`，先确认趋势；需要精确值再跑全量。
- **模型结构不匹配？** 可能是 YAML 改动导致参数维度不一致。脚本会优先用 checkpoint 内的 config；若仍报错，需要重训或调整配置。
- **JSON 损坏或误删？** 重新运行评测脚本即可重新生成；建议在提交前确认 JSON 正常。
- **只写论文不跑评测？** 参考 catalog 表格，直接引用已有结果即可，无需重新跑脚本。

以上步骤执行完后，即使在新环境也能快速接续实验、更新 catalog，并继续论文写作。祝进展顺利！
