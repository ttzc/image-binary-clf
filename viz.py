"""
可视化脚本：把一个 run 的训练曲线、混淆矩阵、错误样本一次性画出来。

用法
----
# 最近一个 run，test 集，阈值 0.5
uv run python viz.py

# 指定 run，并额外画阈值扫描曲线
uv run python viz.py --run-dir runs/20261102_223838 --sweep

# 多看几张错图（默认 16 张）
uv run python viz.py --max-errors 24

产物（写到 <run>/figs/）
----------------------
    train_curves.png     train/val loss ＋ val 各项指标，标出 best epoch
    confusion_matrix.png 2×2 热图，带计数与百分比
    errors.png           分错的样本网格：真实→预测、正类概率、文件名（报告用这段）
    threshold_sweep.png  （--sweep 时）阈值 vs P/R/F1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口后端，避免在终端里弹窗

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from common import (
    DEFAULT_DATA_DIR,
    TinyCNN,
    build_transforms,
    load_run_config,
    resolve_device,
    run_inference,
)

# ---- 中文字体：不设置会全变成方块 ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


def parse_args():
    p = argparse.ArgumentParser(description="训练结果可视化")
    p.add_argument("--run-dir", default="latest",
                   help="latest ｜ run 目录路径（默认 runs/ 下最新的时间戳目录）")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-errors", type=int, default=16,
                   help="错误样本网格最多画几个（默认 16）")
    p.add_argument("--sweep", action="store_true", help="额外画阈值扫描曲线")
    p.add_argument("--out-dir", default=None, help="图片输出目录（默认 <run>/figs）")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def resolve_run_dir(value: str) -> Path:
    if value == "latest":
        root = Path("runs")
        import re
        pat = re.compile(r"^\d{8}_\d{6}$")
        cands = sorted(p for p in root.iterdir() if p.is_dir() and pat.match(p.name))
        if not cands:
            raise SystemExit(f"{root.resolve()} 下没有时间戳命名的 run 目录")
        return cands[-1]
    p = Path(value)
    if not p.is_dir():
        raise SystemExit(f"找不到 run 目录：{p}")
    return p


# ------------------------------------------------------------------ 1. 训练曲线


def plot_curves(run_dir: Path, out_dir: Path) -> Path | None:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        print(f"[skip] 没有 metrics.csv：{csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[skip] metrics.csv 是空的")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) loss
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], marker="o", ms=3, label="train loss")
    ax.plot(df["epoch"], df["val_loss"], marker="s", ms=3, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    # (b) 准确率
    ax = axes[1]
    ax.plot(df["epoch"], df["val_acc"], marker="o", ms=3, color="#2e7d32", label="val acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Val Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    # (c) P / R / F1 —— 报告里最该看的三条
    ax = axes[2]
    ax.plot(df["epoch"], df["val_precision"], marker="o", ms=3, label="precision")
    ax.plot(df["epoch"], df["val_recall"], marker="s", ms=3, label="recall")
    ax.plot(df["epoch"], df["val_f1"], marker="^", ms=4, lw=2, label="F1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Val Precision / Recall / F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    # 标出 best epoch（按 val_f1）
    if "is_best" in df.columns and df["is_best"].sum() > 0:
        best_row = df[df["is_best"] == 1].iloc[-1]
        for ax in axes:
            ax.axvline(best_row["epoch"], color="red", ls="--", lw=1, alpha=0.6)
        axes[2].annotate(
            f"best F1={best_row['val_f1']:.3f}\n@epoch {int(best_row['epoch'])}",
            xy=(best_row["epoch"], best_row["val_f1"]),
            xytext=(8, -28), textcoords="offset points",
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    fig.suptitle(f"training curves · {run_dir.name}", fontsize=12)
    fig.tight_layout()
    out = out_dir / "train_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")
    return out


# ------------------------------------------------------------------ 2. 混淆矩阵


def plot_confusion(cm: np.ndarray, names: list[str], out_dir: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    total = cm.sum()
    for i in range(len(names)):
        for j in range(len(names)):
            v = cm[i, j]
            pct = v / total * 100 if total else 0
            color = "white" if v > cm.max() / 2 else "black"
            ax.text(j, i, f"{v}\n{pct:.1f}%", ha="center", va="center",
                    color=color, fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = out_dir / "confusion_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")
    return out


# ------------------------------------------------------------------ 3. 错误样本


def pick_balanced_errors(wrong_idx: list[int], preds: np.ndarray,
                         labels: np.ndarray, max_items: int) -> list[int]:
    """
    两类错误交替取，避免网格里全是同一个方向。

    错误往往一边倒（比如 43 张 luo→miku、25 张 miku→luo），
    直接取前 N 张会全撞在同一类，看不出反向的问题。
    """
    fp = [i for i in wrong_idx if preds[i] == 1 and labels[i] == 0]  # 负判正
    fn = [i for i in wrong_idx if preds[i] == 0 and labels[i] == 1]  # 正判负
    sel: list[int] = []
    while len(sel) < max_items and (fp or fn):
        if fp:
            sel.append(fp.pop(0))
            if len(sel) >= max_items:
                break
        if fn:
            sel.append(fn.pop(0))
    return sel


def plot_errors(ds, wrong_idx: list[int], probs: np.ndarray, labels: np.ndarray,
                names: list[str], out_dir: Path, max_items: int,
                threshold: float = 0.5, preds: np.ndarray | None = None) -> Path | None:
    if not wrong_idx:
        print("[skip] 没有错误样本（acc=100%？）")
        return None

    if preds is not None:
        sel = pick_balanced_errors(wrong_idx, preds, labels, max_items)
    else:
        sel = wrong_idx[:max_items]

    cols = 4
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, sel):
        path, _ = ds.samples[idx]
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        true_i = int(labels[idx])
        p = float(probs[idx])
        pred_i = int(p >= threshold)
        ax.set_title(
            f"真: {names[true_i]} → 判: {names[pred_i]}\n"
            f"P({names[1]})={p:.3f}",
            fontsize=8,
        )
        name = Path(path).name
        if len(name) > 26:
            name = name[:23] + "..."
        ax.text(0.5, -0.06, name, transform=ax.transAxes,
                ha="center", va="top", fontsize=6, color="#555")

    fig.suptitle(f"错误样本（共 {len(wrong_idx)} 张，展示前 {len(sel)} 张）", fontsize=12)
    fig.tight_layout()
    out = out_dir / "errors.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")
    return out


# ------------------------------------------------------------------ 4. 阈值扫描


def plot_threshold_sweep(probs: np.ndarray, labels: np.ndarray, out_dir: Path) -> Path:
    ths = np.arange(0.05, 0.96, 0.01)
    rows = []
    y = labels.astype(int)
    for t in ths:
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        rows.append((t, p, r, f1, (tp + tn) / max(len(y), 1)))
    arr = np.array(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(arr[:, 0], arr[:, 1], label="precision")
    ax.plot(arr[:, 0], arr[:, 2], label="recall")
    ax.plot(arr[:, 0], arr[:, 3], label="F1", lw=2.2)
    ax.plot(arr[:, 0], arr[:, 4], label="accuracy", ls=":", alpha=0.8)

    best = arr[np.argmax(arr[:, 3])]
    ax.axvline(best[0], color="red", ls="--", lw=1, alpha=0.7)
    ax.annotate(f"best F1={best[3]:.3f} @ t={best[0]:.2f}",
                xy=(best[0], best[3]), xytext=(10, -30),
                textcoords="offset points", fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("阈值扫描")
    fig.tight_layout()
    out = out_dir / "threshold_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")
    return out


# ------------------------------------------------------------------ main


def main():
    args = parse_args()
    device = resolve_device(args.device)
    run_dir = resolve_run_dir(args.run_dir)
    cfg = load_run_config(run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {run_dir.resolve()}")
    print(f"[out] {out_dir.resolve()}")

    plot_curves(run_dir, out_dir)

    # ---- 推理，拿全量 logits ----
    data_dir = Path(args.data_dir) / args.split
    if not data_dir.is_dir():
        raise SystemExit(f"找不到数据目录：{data_dir}")

    mean = cfg.get("mean") or [0.5, 0.5, 0.5]
    std = cfg.get("std") or [0.5, 0.5, 0.5]

    ds = ImageFolder(root=str(data_dir), transform=build_transforms(mean, std, train=False))
    if cfg.get("class_to_idx") and ds.class_to_idx != cfg["class_to_idx"]:
        raise SystemExit(f"类别映射不一致：训练 {cfg['class_to_idx']} vs 现在 {ds.class_to_idx}")

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"找不到 best_model.pt：{ckpt_path}")

    model = TinyCNN(num_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    logits, labels = run_inference(model, loader, device)
    probs = torch.sigmoid(logits.squeeze(1)).numpy()
    labels_np = labels.numpy()

    names = [k for k, _ in sorted(ds.class_to_idx.items(), key=lambda kv: kv[1])]
    preds = (probs >= args.threshold).astype(int)

    # 混淆矩阵（行=真实，列=预测）
    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    plot_confusion(cm, names, out_dir,
                   f"{args.split} · t={args.threshold} · n={len(ds)}")

    # 错误样本
    wrong = np.where(preds != labels_np)[0].tolist()
    uniq, cnt = np.unique(labels_np[wrong], return_counts=True) if wrong else ([], [])
    if wrong:
        detail = "，".join(f"真={names[int(u)]} {int(c)} 张" for u, c in zip(uniq, cnt))
        print(f"[info] 错误 {len(wrong)} 张（{detail}）")
    plot_errors(ds, wrong, probs, labels_np, names, out_dir, args.max_errors,
                threshold=args.threshold, preds=preds)

    if args.sweep:
        plot_threshold_sweep(probs, labels_np, out_dir)

    print(f"\n图都在：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
