"""
训练脚本：从命令行控制「新开任务」或「接着上次的 last_ckpt 续训」。

用法示例
--------
# 1) 新开一个训练任务，跑 10 轮
uv run python train.py --epochs 10

# 2) 新开任务，指定超参
uv run python train.py --epochs 20 --lr 5e-4 --batch-size 32

# 3) 接着最近一个 run 的 last_ckpt 继续训练 5 轮（自动找 runs/ 下最新目录）
uv run python train.py --resume latest --epochs 5

# 4) 指定某个 run 续训
uv run python train.py --resume runs/20261102_214648 --epochs 5

# 5) 直接指向 checkpoint 文件
uv run python train.py --resume runs/20261102_214648/last_ckpt.pt --epochs 5

产物（每个 run 目录里）
----------------------
    config.json     超参与预处理参数（推理时靠它还原 transform）
    metrics.csv     每轮一行的权威实验记录（续训时**追加**，不覆盖）
    last_ckpt.pt    最新完整状态：model + optimizer + epoch + best_val_f1
    best_model.pt   验证集 F1 最高时的纯 state_dict（给 eval.py 用）
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from common import (
    DEFAULT_DATA_DIR,
    TinyCNN,
    build_transforms,
    evaluate,
    find_latest_run,
    load_run_config,
    resolve_device,
    seed_everything,
    train_one_epoch,
)

CSV_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "train_seconds",
    "val_seconds",
    "epoch_seconds",
    "elapsed_seconds",
    "lr",
    "is_best",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="图像二分类训练（洛天依 vs 初音）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"数据根目录，其下应有 train/val/test（默认 {DEFAULT_DATA_DIR}）")
    p.add_argument("--epochs", type=int, default=10,
                   help="本次要训练的轮数；续训时表示「在已训练轮数之后再跑多少轮」（默认 10）")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率（默认 1e-3）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto / cpu / cuda")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数（Windows 建议 0）")
    p.add_argument("--run-name", default=None,
                   help="新任务的目录名（默认用时间戳 YYYYMMDD_HHMMSS）")
    p.add_argument("--resume", default=None,
                   help="续训目标：latest / run 目录 / last_ckpt.pt 路径")
    p.add_argument("--runs-root", default="runs", help="产物根目录（默认 runs）")
    return p.parse_args()


def resolve_resume_target(value: str, runs_root: Path) -> Path:
    """
    把 --resume 的几种写法统一成 last_ckpt.pt 的路径。

    支持：latest ｜ runs/xxx（目录）｜ 任意 .pt 路径 ｜ run_id（如 20261102_214648）
    """
    if value == "latest":
        latest = find_latest_run(runs_root)
        if latest is None:
            raise SystemExit(f"--resume latest 失败：{runs_root.resolve()} 下没有任何 run 目录")
        target = latest / "last_ckpt.pt"
    else:
        p = Path(value)
        if p.is_dir():
            target = p / "last_ckpt.pt"
        elif p.is_file():
            target = p
        else:
            guess = runs_root / value / "last_ckpt.pt"
            if not guess.exists():
                raise SystemExit(f"找不到续训目标：{value}（也试过 {guess}）")
            target = guess

    if not target.exists():
        raise SystemExit(f"找不到 checkpoint：{target}")
    return target


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    data_root = Path(args.data_dir)
    train_dir, val_dir = data_root / "train", data_root / "val"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise SystemExit(f"找不到数据目录：{d}")

    # ---------------- 续训 or 新开 ----------------
    ckpt = None
    if args.resume:
        ckpt_path = resolve_resume_target(args.resume, Path(args.runs_root))
        ckpt = torch.load(ckpt_path, map_location=device)
        run_dir = ckpt_path.parent
        cfg = load_run_config(run_dir)
        mean = cfg.get("mean") or [0.5, 0.5, 0.5]
        std = cfg.get("std") or [0.5, 0.5, 0.5]
        start_epoch = int(ckpt.get("epoch", 0))
        # 续训沿用原 run 目录，CSV 追加写；--epochs 表示「再跑多少轮」
        epochs_this_run = args.epochs
        mode = "resume"
        print(f"[resume] {ckpt_path}")
        print(f"[resume] 已训练 {start_epoch} 轮，best_val_f1={ckpt.get('best_val_f1', -1):.4f}")
    else:
        run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.runs_root) / run_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"目录已存在且非空：{run_dir}（换个 --run-name，或改用 --resume）")
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {}
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        start_epoch = 0
        epochs_this_run = args.epochs
        mode = "new"
        print(f"[new] run dir: {run_dir.resolve()}")

    end_epoch = start_epoch + epochs_this_run
    print(f"[plan] 训练范围 epoch {start_epoch + 1} → {end_epoch}（本次 {epochs_this_run} 轮）  device={device}")

    # ---------------- 数据 ----------------
    train_tf = build_transforms(mean, std, train=True)
    eval_tf = build_transforms(mean, std, train=False)

    train_ds = ImageFolder(root=str(train_dir), transform=train_tf)
    val_ds = ImageFolder(root=str(val_dir), transform=eval_tf)
    print(f"[data] class_to_idx={train_ds.class_to_idx}  train={len(train_ds)}  val={len(val_ds)}")
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise SystemExit("train 与 val 的类别映射不一致，标签语义会错")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # ---------------- 模型 / 损失 / 优化器 ----------------
    model = TinyCNN(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    if ckpt is not None:
        # 结构必须与保存时一致，load_state_dict 才会过（不一致会直接报 key 不匹配）
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            # Adam 的一阶/二阶矩恢复了，但新传的 --lr 优先，便于「续训时降学习率」
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        print(f"[resume] optimizer 状态已恢复，lr 设为 {args.lr}")

    # ---------------- 落盘：config ----------------
    config = dict(cfg) if cfg else {}
    config.update({
        "run_id": run_dir.name,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "mean": mean,
        "std": std,
        "class_to_idx": train_ds.class_to_idx,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "device": str(device),
        "seed": args.seed,
    })
    if mode == "new":
        config["created_at"] = datetime.now().isoformat(timespec="seconds")
    else:
        config["resumed_at"] = datetime.now().isoformat(timespec="seconds")
    config["num_epochs"] = end_epoch  # 累计目标轮数
    config["rounds"] = config.get("rounds", []) + [
        {"mode": mode, "start": start_epoch + 1, "end": end_epoch,
         "lr": args.lr, "batch_size": args.batch_size,
         "at": datetime.now().isoformat(timespec="seconds")}
    ]

    metrics_csv = run_dir / "metrics.csv"
    config_json = run_dir / "config.json"
    last_ckpt = run_dir / "last_ckpt.pt"
    best_ckpt = run_dir / "best_model.pt"

    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 新任务写表头；续训追加，绝不覆盖已有实验记录
    if mode == "new":
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        if start_epoch == 0 and epochs_this_run <= 0:
            print("[warn] --epochs <= 0，不训练任何轮")
    else:
        if not metrics_csv.exists():
            with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    # ---------------- 训练 ----------------
    total_start = time.perf_counter()
    epoch_times: list[float] = []

    for epoch in range(start_epoch + 1, end_epoch + 1):
        epoch_start = time.perf_counter()

        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_time = time.perf_counter() - t1

        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - total_start
        epoch_times.append(epoch_time)

        is_best = val_metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1"]

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "train_seconds": round(train_time, 3),
            "val_seconds": round(val_time, 3),
            "epoch_seconds": round(epoch_time, 3),
            "elapsed_seconds": round(elapsed, 3),
            "lr": optimizer.param_groups[0]["lr"],
            "is_best": int(is_best),
        }
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

        # 完整状态：续训要靠 optimizer 的一阶/二阶矩，只存 model 等于热启动新优化器
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "class_to_idx": train_ds.class_to_idx,
                "config": config,
            },
            last_ckpt,
        )

        # 纯权重：给推理用，与 load_state_dict 直接兼容
        if is_best:
            torch.save(model.state_dict(), best_ckpt)

        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch * (end_epoch - epoch)
        print(
            f"[{epoch:02d}/{end_epoch}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} "
            f"P={val_metrics['precision']:.4f} "
            f"R={val_metrics['recall']:.4f} "
            f"F1={val_metrics['f1']:.4f} | "
            f"{epoch_time:.1f}s ETA {eta / 60:.1f}min"
            + ("  *best*" if is_best else "")
        )

    print(f"\ndone. best val_f1={best_val_f1:.4f}")
    print("run dir   :", run_dir.resolve())
    print("metrics   :", metrics_csv.resolve())
    print("last ckpt :", last_ckpt.resolve())
    print("best ckpt :", best_ckpt.resolve())


if __name__ == "__main__":
    main()
