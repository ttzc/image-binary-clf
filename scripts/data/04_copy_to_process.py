#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按 splits.csv 把图**复制**到 process/ 目录（原图保持不动）

为什么复制而不移动
------------------------------------------------------------------
`raw/` 是底本，任何一步预处理都能从它重跑；`process/` 是可丢弃的加工区。
所以这里只 **copy**，不 cut。

目录结构（默认 --layout split）
------------------------------------------------------------------
    data/process/
    ├── train/{luo,miku}/...
    ├── val/{luo,miku}/...
    └── test/{luo,miku}/...

⭐ 这个结构是 **torchvision `ImageFolder` 的标准形状**，训练代码两行就能读：

    ds = ImageFolder("data/process/train")   # 子目录名＝类别名 → luo=0, miku=1（按字母序）

⚠️ ImageFolder 的类别顺序是**按目录名字母序**（luo < miku），所以 `luo→0, miku→1`。
   和你 splits.csv 里的 label 一致（0=luo, 1=miku），不必自己重映射。

如果你更想按类平铺（`process/luo/`、`process/miku/`），加 `--layout class`，
但那样就分不出 train/val/test，训练时还得回去读 splits.csv。

用法
------------------------------------------------------------------
① 先看规模（不复制，只统计）：

    <python> -u -X utf8 04-复制到process.py --dry-run

② 真复制：

    <python> -u -X utf8 04-复制到process.py

    <python> -u -X utf8 04-复制到process.py --workers 4   # 磁盘快时可以开多线程

③ 只想快速浏览一下、不想占双份空间（同一块盘可用）：

    <python> -u -X utf8 04-复制到process.py --hardlink

    🔴 **硬链接警告**：hardlink 与原图是**同一份数据**。以后在 process/ 里
       **覆盖写**（如统一转 jpg 存回原路径）会**改动 raw/ 的底本**。
       只用于"看一眼就删"的场景；要做预处理就别用。

参数
    --splits    划分清单                     默认 data/splits.csv
    --raw-root  原图根目录                   默认 data/raw
    --out       输出根目录                   默认 data/process
    --layout    split（默认，ImageFolder 形状）｜class（按类平铺）
    --workers   并发复制线程数               默认 1（机械盘别开太高）
    --hardlink  用硬链接代替复制             —— 见上方警告
    --dry-run   只统计不写盘
    --overwrite 覆盖已存在的同名文件（默认跳过同样大小的）

退出码：0＝成功｜1＝有文件缺失｜2＝参数/清单问题
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

NAME_OF_CLASS = {0: "luo", 1: "miku"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(description="按 splits.csv 复制图片到 process/")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--out", default="data/process")
    ap.add_argument("--layout", choices=["split", "class"], default="split")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--hardlink", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sp, root, out = Path(args.splits), Path(args.raw_root), Path(args.out)
    if not sp.exists():
        print(f"[X] 找不到 {sp}（先跑 03-生成划分.py）", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(sp.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        print("[X] 清单为空", file=sys.stderr)
        return 2

    # 组装任务：src → dst
    tasks: list[tuple[Path, Path, int]] = []
    missing: list[str] = []
    per_split: dict[str, dict[str, int]] = {}
    total_bytes = 0

    for r in rows:
        src = root / r["path"]
        cls = NAME_OF_CLASS.get(int(r["label"]), str(r["label"]))
        if not src.exists():
            missing.append(r["path"])
            continue
        size = src.stat().st_size
        total_bytes += size
        if args.layout == "split":
            dst = out / r["split"] / cls / src.name
        else:
            dst = out / cls / src.name
        tasks.append((src, dst, size))
        per_split.setdefault(r["split"], {}).setdefault(cls, 0)
        per_split[r["split"]][cls] += 1

    print(f"[i] 清单 {len(rows)} 行｜可复制 {len(tasks)}｜缺失 {len(missing)}")
    print(f"[i] 结构：{'process/{split}/{class}/' if args.layout == 'split' else 'process/{class}/'}"
          f"｜方式：{'硬链接' if args.hardlink else '复制'}"
          f"{'（dry-run，不写盘）' if args.dry_run else ''}")
    print(f"{'集合':<8}{'luo':>7}{'miku':>7}{'合计':>8}")
    for s in ("train", "val", "test"):
        d = per_split.get(s, {})
        print(f"{s:<8}{d.get('luo', 0):>7}{d.get('miku', 0):>7}{sum(d.values()):>8}")
    print(f"[i] 预计占用 {human(total_bytes)}（D 盘剩余 {human(shutil.disk_usage(str(out.parent)).free)}）")

    if missing:
        print(f"[!] {len(missing)} 个源文件不存在，例：{missing[:3]}", file=sys.stderr)
    if args.dry_run:
        print("[i] dry-run 结束，未写任何文件")
        return 0 if not missing else 1

    t0 = time.perf_counter()
    done = skipped = failed = 0
    done_bytes = 0

    def work(src: Path, dst: Path, size: int) -> tuple[str, int]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.overwrite:
            if dst.stat().st_size == size:
                return "skip", size
        if args.hardlink:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
        return "ok", size

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(work, s, d, n): (s, d) for s, d, n in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                st, n = fut.result()
                if st == "ok":
                    done += 1
                    done_bytes += n
                else:
                    skipped += 1
            except Exception as e:                       # noqa: BLE001
                failed += 1
                s, d = futs[fut]
                print(f"  ! {s.name} → {d}：{e}")
            if k % 200 == 0 or k == len(tasks):
                el = time.perf_counter() - t0
                rate = k / el if el else 0
                eta = (len(tasks) - k) / rate if rate else 0
                print(f"      {k}/{len(tasks)}｜{human(done_bytes)}｜{rate:.0f} 个/s｜剩余约 {eta:.0f}s")

    el = time.perf_counter() - t0
    print("=" * 60)
    print(f"完成：复制 {done}｜跳过（已存在）{skipped}｜失败 {failed}｜"
          f"{human(done_bytes)}｜{el:.1f}s")
    print(f"输出：{out}")
    if failed:
        return 1
    if not missing:
        print("[✓] 全部就绪，训练代码可直接 ImageFolder(\"" + str(out / "train") + "\")")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
