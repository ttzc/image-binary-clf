"""
测试脚本：加载某个 run 的 best_model.pt，在测试集上算完整指标。

用法示例
--------
# 1) 评估最近一个 run 的 best 模型（默认 test 集）
uv run python eval.py

# 2) 指定 run 目录
uv run python eval.py --ckpt runs/20261102_214648

# 3) 指定权重文件
uv run python eval.py --ckpt runs/20261102_214648/best_model.pt

# 4) 换阈值 / 额外扫一遍最优阈值
uv run python eval.py --threshold 0.6
uv run python eval.py --sweep

# 5) 在验证集上复查（注意：val 参与过选模型，别拿它当最终成绩）
uv run python eval.py --split val

为什么阈值可调：正负类张数虽然配平，但混淆方向常常不对称
（比如 fp 远多于 fn）时，把阈值从 0.5 抬高会同时提 precision、降 recall，
F1 往往还能再涨一点。--sweep 会给出 F1 最高的那个点。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from common import (
    DEFAULT_DATA_DIR,
    TinyCNN,
    build_transforms,
    find_latest_run,
    load_run_config,
    resolve_device,
    run_inference,
)


def parse_args():
    p = argparse.ArgumentParser(description="图像二分类测试/评估")
    p.add_argument("--ckpt", default="latest",
                   help="latest ｜ run 目录 ｜ best_model.pt 路径（默认 latest）")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"数据根目录，其下应有 train/val/test（默认 {DEFAULT_DATA_DIR}）")
    p.add_argument("--split", default="test", choices=["test", "val", "train"],
                   help="在哪个子集上评估（默认 test）")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sweep", action="store_true",
                   help="额外扫一遍阈值，输出 F1 最高的阈值（不改变 --threshold 的报告）")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--save-dir", default=None,
                   help="结果 csv 输出目录（默认写到 ckpt 所在 run 目录）")
    return p.parse_args()


def resolve_ckpt(value: str, runs_root: Path) -> Path:
    """把 --ckpt 的几种写法统一成权重文件路径。"""
    if value == "latest":
        latest = find_latest_run(runs_root)
        if latest is None:
            raise SystemExit(f"找不到任何 run：{runs_root.resolve()}")
        target = latest / "best_model.pt"
    else:
        p = Path(value)
        if p.is_dir():
            target = p / "best_model.pt"
        elif p.is_file():
            target = p
        else:
            guess = runs_root / value / "best_model.pt"
            if not guess.exists():
                raise SystemExit(f"找不到权重：{value}（也试过 {guess}）")
            target = guess

    if not target.exists():
        raise SystemExit(f"找不到权重文件：{target}")
    return target


def main():
    args = parse_args()
    device = resolve_device(args.device)

    ckpt_path = resolve_ckpt(args.ckpt, Path(args.runs_root))
    run_dir = ckpt_path.parent
    cfg = load_run_config(run_dir)

    # 预处理参数必须与训练时一致，否则指标会静默变差却看不出原因
    mean = cfg.get("mean") or [0.5, 0.5, 0.5]
    std = cfg.get("std") or [0.5, 0.5, 0.5]
    class_to_idx_train = cfg.get("class_to_idx")

    data_dir = Path(args.data_dir) / args.split
    if not data_dir.is_dir():
        raise SystemExit(f"找不到数据目录：{data_dir}")

    ds = ImageFolder(root=str(data_dir), transform=build_transforms(mean, std, train=False))
    if class_to_idx_train and ds.class_to_idx != class_to_idx_train:
        raise SystemExit(
            f"类别映射不一致：训练时 {class_to_idx_train}，现在 {ds.class_to_idx}\n"
            "文件名的类别顺序变了，正负类会整体翻转"
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,          # 不打乱，便于和文件列表逐张对齐
        num_workers=args.num_workers,
    )

    model = TinyCNN(num_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print(f"[ckpt] {ckpt_path.resolve()}")
    print(f"[data] split={args.split}  n={len(ds)}  class_to_idx={ds.class_to_idx}")
    print(f"[cfg ] mean={mean} std={std} device={device}")

    logits, labels = run_inference(model, loader, device)
    probs = torch.sigmoid(logits.squeeze(1))

    def metrics_at(th: float) -> dict:
        preds = (probs >= th).long()
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        acc = (tp + tn) / max(len(labels), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {"threshold": th, "acc": acc, "precision": precision,
                "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    m = metrics_at(args.threshold)

    # 用 logit 算 loss，与训练口径一致
    with torch.no_grad():
        loss = nn.BCEWithLogitsLoss()(logits, labels.float().unsqueeze(1)).item()

    print("\n" + "=" * 56)
    print(f"{args.split} 结果（threshold={args.threshold}）")
    print("=" * 56)
    print(f"loss      : {loss:.4f}")
    print(f"acc       : {m['acc']:.4f}")
    print(f"precision : {m['precision']:.4f}")
    print(f"recall    : {m['recall']:.4f}")
    print(f"F1        : {m['f1']:.4f}")
    print(f"混淆矩阵  : tp={m['tp']}  tn={m['tn']}  fp={m['fp']}  fn={m['fn']}")

    names = [k for k, _ in sorted(ds.class_to_idx.items(), key=lambda kv: kv[1])]
    print("\n混淆矩阵（行=真实，列=预测）")
    cm = confusion_matrix(labels.numpy(), (probs >= args.threshold).long().numpy())
    header = "          " + "".join(f"{n:>10}" for n in names)
    print(header)
    for i, n in enumerate(names):
        print(f"{n:>10}" + "".join(f"{v:>10d}" for v in cm[i]))

    print("\n分类报告")
    print(classification_report(
        labels.numpy(),
        (probs >= args.threshold).long().numpy(),
        target_names=names,
        digits=4,
        zero_division=0,
    ))

    if args.sweep:
        best = max((metrics_at(t / 100) for t in range(30, 81)), key=lambda d: d["f1"])
        print("阈值扫描（F1 最优）")
        print(f"  threshold={best['threshold']:.2f}  "
              f"acc={best['acc']:.4f}  P={best['precision']:.4f}  "
              f"R={best['recall']:.4f}  F1={best['f1']:.4f}")
        print("  ⚠️ 在 test 上调阈值＝用测试集调参，报告里若要写这个数，"
              "应说明是在 val 上选阈值、再到 test 上验证")

    # ---------------- 落盘 ----------------
    save_dir = Path(args.save_dir) if args.save_dir else run_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    out_csv = save_dir / f"{args.split}_metrics.csv"

    # 前 10 列与 notebook 里写出的 test_metrics.csv 保持一致，后面追加溯源信息
    fields = ["threshold", "loss", "acc", "precision", "recall", "f1",
              "tp", "tn", "fp", "fn",
              "split", "n", "ckpt", "evaluated_at"]
    from datetime import datetime
    row = {
        "split": args.split,
        "threshold": args.threshold,
        "loss": round(loss, 6),
        "acc": round(m["acc"], 6),
        "precision": round(m["precision"], 6),
        "recall": round(m["recall"], 6),
        "f1": round(m["f1"], 6),
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        "n": len(ds),
        "ckpt": str(ckpt_path.as_posix()),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    print(f"\n已写出：{out_csv.resolve()}")


if __name__ == "__main__":
    main()
