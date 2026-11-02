"""
公共定义：模型、预处理、指标、训练/评估函数。

为什么单独抽一个文件：train.py 与 eval.py 必须用**完全一致**的模型结构和预处理，
否则会出现"权重能加载、指标却全错"的静默 bug。放在一处就不会漂移。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------- 默认配置

DEFAULT_DATA_DIR = "data/process"
DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD = [0.5, 0.5, 0.5]
IMAGE_SIZE = 224
RESIZE_TO = 256


# ---------------------------------------------------------------- 复现性


def seed_everything(seed: int = 42) -> None:
    """固定所有随机源，保证同一 seed 可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str = "auto") -> torch.device:
    """auto 时有 CUDA 用 CUDA，否则 CPU。"""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


# ---------------------------------------------------------------- 预处理


def build_transforms(mean=None, std=None, train: bool = True):
    """
    训练用随机增强，验证/测试必须确定性。

    注意：这里用 RandomResizedCrop 而不是 Resize(224,224)——
    直接 Resize 到正方形会把 3:4 / 16:9 的画幅信息抹掉并让人物变形。
    """
    from torchvision import transforms as T

    mean = mean or DEFAULT_MEAN
    std = std or DEFAULT_STD

    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    return T.Compose(
        [
            T.Resize(RESIZE_TO),  # 单参数 = 短边缩到 256，保持长宽比
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


# ---------------------------------------------------------------- 模型


class TinyCNN(nn.Module):
    """
    最小 CNN：两层卷积 + 全局平均池化 + 单 logit 输出。

    输出不做 Sigmoid，交给 nn.BCEWithLogitsLoss 内部处理，数值更稳。
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------- 指标


def binary_metrics_from_logits(logits, targets, threshold: float = 0.5) -> dict:
    """
    从 logit 算 acc / precision / recall / f1，以及混淆矩阵四元素。

    logits: [B,1] 或 [B]；targets: [B] 的整型标签。
    注意这是"传入这批数据"的指标；跨 batch 汇总要先把 logits 和 targets
    拼成全量再算，否则各批比率不能直接平均。
    """
    probs = torch.sigmoid(logits.squeeze(1))
    preds = (probs >= threshold).long()
    targets = targets.long()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------- 训练 / 评估


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """训练一个 epoch，返回按样本数加权的平均 loss。"""
    model.train()
    total_loss, total_n = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        # BCEWithLogitsLoss 要 target 是 float 且形状与 logit 一致 [B,1]
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_n += images.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def run_inference(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """遍历 loader，返回全部 logits [N,1] 与 labels [N]。不打乱顺序。"""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def evaluate(model, loader, criterion, device, threshold: float = 0.5) -> dict:
    """在给定 loader 上算 loss 与各项指标（全集级，不是批次平均）。"""
    logits, labels = run_inference(model, loader, device)
    metrics = binary_metrics_from_logits(logits, labels, threshold)
    with torch.no_grad():
        loss = criterion(logits, labels.float().unsqueeze(1)).item()
    metrics["loss"] = loss
    return metrics


# ---------------------------------------------------------------- 目录与配置


def load_run_config(run_dir: Path) -> dict:
    """读 run 目录下的 config.json；不存在则返回空 dict（用默认值兜底）。"""
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def find_latest_run(runs_root: Path) -> Path | None:
    """
    取 runs/ 下最新的 run 目录。

    只认时间戳命名（YYYYMMDD_HHMMSS）的目录——否则 `_smoke`、`backup` 这类
    手建目录会按字符串排序插到前面（`_` 的 ASCII 比数字大），把 latest 抢走。
    """
    if not runs_root.is_dir():
        return None
    import re

    pattern = re.compile(r"^\d{8}_\d{6}$")
    runs = sorted(
        p for p in runs_root.iterdir() if p.is_dir() and pattern.match(p.name)
    )
    return runs[-1] if runs else None
