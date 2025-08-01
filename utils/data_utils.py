"""
utils/data_utils.py
────────────────────────────────────────────────────────────
统一处理 KaggleHub 数据集下载 + Torch DataLoader 加载
支持数据集：
    • mnist           （hojjatk/mnist-dataset）
    • fashion_mnist   （zalando-research/fashionmnist）
依赖：
    pip install kagglehub torch torchvision tqdm
────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Tuple
import os
import shutil
import tarfile
import zipfile

# ---------------------------------------------------------------------
# 0️⃣ Kaggle 数据集 slug 对照表
# ---------------------------------------------------------------------
KAGGLE_DATASETS = {
    "mnist": "hojjatk/mnist-dataset",
    "fashion_mnist": "zalando-research/fashionmnist",
}

# ---------------------------------------------------------------------
# 🔧 工具函数
# ---------------------------------------------------------------------
def _download_via_kagglehub(slug: str) -> Path:
    """调用 kagglehub 下载并返回缓存目录"""
    import kagglehub
    return Path(kagglehub.dataset_download(slug))

# —— PATCH：忽略只读位复制文件 ——————————————————————
def _safe_copy(src: Path, dst: Path):
    """
    Windows 下 KaggleHub 解压文件常带只读属性，shutil.copy2 会报
    PermissionError；此函数先去掉只读位再复制。
    """
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        os.chmod(src, 0o644)         # 0o644 = 可读写、不执行
        shutil.copy2(src, dst)

def _extract_archive(archive: Path, dst: Path):
    """支持 .zip / .tar / .tgz 解压"""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dst)
    elif archive.suffix in {".tar", ".tgz", ".gz"}:
        with tarfile.open(archive) as tf:
            tf.extractall(dst)
    else:
        raise ValueError(f"Unknown archive type: {archive}")

# ---------------------------------------------------------------------
# 1️⃣ 主入口：下载并放入 data/raw/<dataset_name>
# ---------------------------------------------------------------------
def download_dataset(name: str, raw_root: str | Path = "data/raw") -> Path:
    """
    下载指定数据集到 data/raw/<name>/，若已存在则直接返回路径
    """
    if name not in KAGGLE_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. "
                         f"Choices: {list(KAGGLE_DATASETS)}")

    slug      = KAGGLE_DATASETS[name]
    cache_dir = _download_via_kagglehub(slug)

    dst_dir = Path(raw_root) / name
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 把缓存目录里的所有文件复制或解压到目标目录
    for p in cache_dir.iterdir():
        # MNIST / Fashion-MNIST 的四个 .gz 已被 kagglehub 解开成无扩展名文件
        if p.is_file() and p.suffix in {".gz", ".xz", ".bz2"}:
            _safe_copy(p, dst_dir / p.name)
        elif p.is_file() and p.suffix in {".zip", ".tar", ".tgz"}:
            _extract_archive(p, dst_dir)
        elif p.is_file():
            _safe_copy(p, dst_dir / p.name)
        else:
            # 若是目录，整份复制
            try:
                shutil.copytree(p, dst_dir / p.name, dirs_exist_ok=True)
            except PermissionError:
                # 处理目录中文件的权限问题
                for root, dirs, files in os.walk(p):
                    for file in files:
                        file_path = Path(root) / file
                        try:
                            os.chmod(file_path, 0o644)
                        except:
                            pass
                shutil.copytree(p, dst_dir / p.name, dirs_exist_ok=True)

    return dst_dir.resolve()

# ---------------------------------------------------------------------
# 2️⃣ Torch DataLoader 快捷封装
# ---------------------------------------------------------------------
def get_dataloaders(name: str,
                    batch_size: int = 64,
                    root: str | Path = "data/raw") -> Tuple["DataLoader","DataLoader"]:
    """下载（如需要）并返回 train_loader, test_loader"""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    root = Path(root).resolve()
    download_dataset(name, root)     # 确保存在

    tfm = transforms.ToTensor()
    data_dir = root / name

    if name == "mnist":
        train_ds = datasets.MNIST(data_dir, train=True,  download=False, transform=tfm)
        test_ds  = datasets.MNIST(data_dir, train=False, download=False, transform=tfm)
    elif name == "fashion_mnist":
        train_ds = datasets.FashionMNIST(data_dir, train=True,  download=False, transform=tfm)
        test_ds  = datasets.FashionMNIST(data_dir, train=False, download=False, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset {name}")

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2))

# ---------------------------------------------------------------------
# 3️⃣ CLI 使用示例：python utils/data_utils.py mnist
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        description="Download dataset to data/raw/<name>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python utils/data_utils.py mnist
              python utils/data_utils.py fashion_mnist
        """)
    )
    parser.add_argument("name", choices=KAGGLE_DATASETS)
    args = parser.parse_args()

    print("Downloading …")
    path = download_dataset(args.name)
    print("Saved to:", path)
