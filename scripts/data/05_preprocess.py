#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一格式与尺寸：把 process/ 里的图预处理成「jpg · RGB · 长边 1024」

做的事（顺序不能乱）
------------------------------------------------------------------
1. **动图取首帧**（GIF/webp 动图）
2. **按 EXIF 摆正**（手机图会躺倒，不处理会横竖错乱）
3. **透明合成白底**  ← 最容易忘、后果最重的一步
   PNG/WebP 的透明区直接 `convert('RGB')` 会变成**黑色**。
   实测本数据集 luo 有 226 张 RGBA 模式，其中 3 张真有大片透明（68%/64%/49%）。
   不处理 → 那 3 张变成大片黑底 → 模型可能学成「黑底＝洛天依」，是典型的捷径特征。
   **合成到白底**（不是黑底）：动画图本来就多是白背景，白底最不容易引入新特征。
4. **缩到长边 1024**（Lanczos，只缩不放）
5. **存 jpg q95**（RGB、无 alpha、解码最快）

为什么输出到另一个目录（默认 process_1024/）
------------------------------------------------------------------
`process/` 是「划分后的原图副本」，留着它有三个用处：
  · 任何一步都能从这里重跑，不用回 raw 再挑
  · 可以对比「原图 vs 预处理后」两次训练的精度差 —— **这本身就是报告里的一小段实验**
  · 万一预处理参数选错（比如 1024 太小），还有原图可换
多占约 300 MB，换全部可回溯。

用法
------------------------------------------------------------------
① 先看规模：

    <python> -u -X utf8 05-预处理.py --dry-run

② 真处理：

    <python> -u -X utf8 05-预处理.py --workers 4

常用参数
    --src        源根目录（其下 {split}/{class}/）     默认 data/process
    --dst        输出根目录                            默认 data/process
    --max-side   长边像素（只缩不放）                  默认 1024
    --quality    JPEG 质量                             默认 95
    --bg         透明合成底色 hex（默认 ffffff 白）     可用 000000 / 808080
    --workers    并发线程数                            默认 4
    --overwrite  覆盖已存在的输出（默认跳过）
    --dry-run    只统计不写盘

输出
    目录结构与源一致：{dst}/{split}/{class}/xxx.jpg
    统计摘要：{dst}/_preprocess_stats.json  ← **写报告时"数据清洗规则"一节可直接引**
    失败清单（若有）：{dst}/_failed.csv

退出码：0＝成功｜1＝有失败项｜2＝参数问题
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("[X] 缺 Pillow：pip install pillow", file=sys.stderr)
    raise SystemExit(2)

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
SPLITS = ("train", "val", "test")
CLASSES = ("luo", "miku")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def collect(src: Path) -> list[tuple[Path, Path]]:
    """返回 [(源文件, 目标文件)]，目标一律 .jpg。"""
    jobs = []
    for split in SPLITS:
        for cls in CLASSES:
            d = src / split / cls
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXT:
                    jobs.append((p, Path(split) / cls / f"{p.stem}.jpg"))
    return jobs


def convert(src: Path, dst: Path, max_side: int, quality: int, bg: tuple[int, int, int]
            ) -> dict:
    im = Image.open(src)
    had_alpha = False
    if getattr(im, "is_animated", False):
        im.seek(0)
    im = ImageOps.exif_transpose(im)          # 按 EXIF 摆正（返回新图或原图）

    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        had_alpha = True
        im = im.convert("RGBA")
        canvas = Image.new("RGB", im.size, bg)
        canvas.paste(im, mask=im.split()[-1])  # 透明区透出底色
        im = canvas
    else:
        im = im.convert("RGB")

    w0, h0 = im.size
    long_side = max(w0, h0)
    if long_side > max_side:
        scale = max_side / long_side
        im = im.resize((max(1, round(w0 * scale)), max(1, round(h0 * scale))), Image.LANCZOS)

    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="JPEG", quality=quality, optimize=True, progressive=True)
    return {
        "w0": w0, "h0": h0, "w": im.size[0], "h": im.size[1],
        "had_alpha": had_alpha, "resized": long_side > max_side,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="统一格式与尺寸 → jpg/RGB/长边 N")
    ap.add_argument("--src", default="data/process")
    ap.add_argument("--dst", default="data/process")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--bg", default="ffffff")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        print(f"[X] 源目录不存在：{src}", file=sys.stderr)
        return 2
    bg = tuple(int(args.bg[i:i + 2], 16) for i in (0, 2, 4))

    jobs = collect(src)
    if not jobs:
        print(f"[X] {src} 下没找到 {{split}}/{{class}}/ 结构或图片", file=sys.stderr)
        return 2

    per = {s: {c: sum(1 for p, d in jobs if d.parts[0] == s and d.parts[1] == c)
               for c in CLASSES} for s in SPLITS}
    src_bytes = sum(p.stat().st_size for p, _ in jobs)

    print(f"[i] 源 {src}｜{len(jobs)} 张｜{human(src_bytes)}")
    print(f"[i] 目标 {dst}｜jpg q{args.quality}｜长边 {args.max_side}｜底色 #{args.bg}"
          f"｜workers {args.workers}{'（dry-run）' if args.dry_run else ''}")
    print(f"{'集合':<8}{'luo':>7}{'miku':>7}{'合计':>8}")
    for s in SPLITS:
        print(f"{s:<8}{per[s]['luo']:>7}{per[s]['miku']:>7}{sum(per[s].values()):>8}")

    if args.dry_run:
        print("[i] dry-run 结束，未写任何文件")
        return 0

    t0 = time.perf_counter()
    done = skipped = failed = 0
    out_bytes = 0
    alpha_n = resized_n = 0
    longs: list[int] = []
    failures: list[tuple[str, str]] = []

    def work(sf: Path, df: Path):
        target = dst / df
        if target.exists() and not args.overwrite:
            return "skip", None
        info = convert(sf, target, args.max_side, args.quality, bg)
        info["out"] = target.stat().st_size
        return "ok", info

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(work, sf, df): sf for sf, df in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            sf = futs[fut]
            try:
                st, info = fut.result()
                if st == "ok":
                    done += 1
                    out_bytes += info["out"]
                    alpha_n += int(info["had_alpha"])
                    resized_n += int(info["resized"])
                    longs.append(max(info["w"], info["h"]))
                else:
                    skipped += 1
            except Exception as e:                     # noqa: BLE001
                failed += 1
                failures.append((str(sf), str(e)))
                print(f"  ! {sf.name}：{e}")
            if k % 200 == 0 or k == len(jobs):
                el = time.perf_counter() - t0
                rate = k / el if el else 0
                print(f"      {k}/{len(jobs)}｜{human(out_bytes)}｜{rate:.1f} 张/s｜"
                      f"剩余约 {(len(jobs)-k)/rate if rate else 0:.0f}s")

    el = time.perf_counter() - t0
    stats = {
        "src": str(src), "dst": str(dst),
        "count": len(jobs), "done": done, "skipped": skipped, "failed": failed,
        "params": {"max_side": args.max_side, "quality": args.quality, "bg": args.bg},
        "bytes_in": src_bytes, "bytes_out": out_bytes,
        "ratio": round(out_bytes / src_bytes, 4) if src_bytes else None,
        "had_alpha": alpha_n, "resized": resized_n,
        "long_side": {
            "min": min(longs) if longs else None,
            "median": int(statistics.median(longs)) if longs else None,
            "max": max(longs) if longs else None,
        },
        "seconds": round(el, 1),
        "per_split": per,
    }
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "_preprocess_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        with (dst / "_failed.csv").open("w", encoding="utf-8-sig", newline="") as f:
            f.write("path,error\n")
            for p, e in failures:
                f.write(f'"{p}","{e}"\n')

    print("=" * 64)
    print(f"完成：处理 {done}｜跳过 {skipped}｜失败 {failed}｜{el:.1f}s")
    if done:
        print(f"体积：{human(src_bytes)} → {human(out_bytes)}"
              f"（压到 {out_bytes/src_bytes:.1%}）")
        print(f"其中：含透明通道 {alpha_n} 张（已合成到 #{args.bg}）｜发生缩放 {resized_n} 张")
        print(f"输出长边：min {stats['long_side']['min']}｜中位 {stats['long_side']['median']}"
              f"｜max {stats['long_side']['max']}")
    print(f"统计摘要：{dst/'_preprocess_stats.json'}")
    if failed:
        print(f"[X] {failed} 张失败，见 {dst/'_failed.csv'}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
