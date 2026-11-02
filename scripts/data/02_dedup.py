#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图集去重体检器：MD5 精确去重 ＋ pHash 近似去重（v2·整理版）
================================================================
用途：AI 概论 第一次作业（分类器 · 洛天依 vs 初音未来）的**数据清洗**。
     ⭐ **两个类通用**——脚本里没有任何"哪一类"的假设，只认目录和图片文件名。
        `luo`（洛天依，1678 张）与 `miku`（初音，1232 张）**用同一个脚本、换 `--dir` 即可**。
        初音那份的原始采集由 `01-数据采集脚本-初音（mikupara）.py` 负责。

历史产出（都是本脚本跑的，仅记录，别当作固定路径）：
       · `luo` → `data/_dedup_report.md` ＋ `_dedup_groups_20261029.csv`
       · `miku` → 见 `--out-md/--out-csv` 指定的路径

⭐ 相对当初那一版的整理（逻辑未改，只做工程化）：
  1. **去掉 opencv 依赖**：原来用 `cv2.dct` 算 8×8 DCT，现在用纯 numpy 的
     余弦基矩阵做同样的变换（只差一个全局常数，不影响 pHash 的"与中位数比"判定）
     → 只需 numpy + Pillow，少装一个大包。
  2. **路径全部参数化**：`--dir/--out-md/--out-csv`，不再写死路径（原先硬编码 `data/luo`）。
  3. 合并了原 `dedup_bench.py` 的实用开关：`--limit`（先跑 N 张看速度）、`--no-md5`。
  4. `np.bitwise_count`（numpy ≥ 2.0）带降级实现，老 numpy 也能跑。

为什么是"四个变体"：
    同一张图被转载时常被镜像、或被裁掉边（水印/黑边），单靠一次 pHash 认不出来。
    所以对每张图算 4 个 64 位 pHash：原图 / 原图镜像 / 中心 80% 裁剪 / 裁剪镜像，
    两两比较取**最小汉明距离**——任一组合撞上就算同一张。

判重口径：
    汉明距离 ≤ 阈值（默认 8）判为近似重复。实测参照：
    真正同图 0–8｜同角色不同画 10–16｜>24 基本无关。

⚠️ 本脚本**只读体检**，不改名、不删除任何图片。删留由人决定（报告里给了 A/B 档建议）。
   要真正落地删除，请自己看过 B 档再动手——实测 B 档存在误报。

用法（手动跑，换 --dir 就是换类）：
    <python> -u -X utf8 "02-数据清洗脚本-图集去重（MD5+pHash）.py" --dir "data/raw/luo"
    <python> -u -X utf8 "02-数据清洗脚本-图集去重（MD5+pHash）.py" --dir "data/raw/miku" ^
        --out-md "data/_dedup_report_miku_YYYYMMDD.md" --out-csv "data/_dedup_groups_miku_YYYYMMDD.csv"

常用参数：
    --dir         ⭐ **必填**：要体检的图片目录（如 data/raw/luo 或 data/raw/miku）
    --out-md      报告路径（默认 **图片目录之外**：`<父目录>/_dedup_report.md`）
    --out-csv     分组表路径（默认 **图片目录之外**：`<父目录>/_dedup_groups.csv`）
    --hamm 8      近似判重阈值（汉明距离）
    --workers 4   多进程数（建议 ≤ CPU 核数-1）
    --limit 100   只跑前 N 张（0＝全部，默认 0）
    --no-md5      跳过 MD5（省读盘；只算 pHash）
    --every 100   每多少张报一次进度

性能参考（1678 张、4 workers）：哈希约 26 ms/张（JPEG 用 draft 直接出小图），
    两两比对 1678×1678 分 16 次变体组合，全程约 3–5 分钟。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HAMM = 8          # 默认判重阈值（汉明距离）
DCT_N = 32        # pHash 的 DCT 尺寸
KEEP = 8          # 取左上 8×8 低频

# 32 点 DCT-II 余弦基（与 cv2.dct 等价，只差全局常数；pHash 只比大小，不受影响）
_n = np.arange(DCT_N)
_C = np.cos(np.pi * (2 * _n[None, :] + 1) * _n[:, None] / (2 * DCT_N)).astype(np.float32)


def _popcount_u64(a: np.ndarray) -> np.ndarray:
    """逐元素 popcount；优先用 numpy≥2.0 的 bitwise_count，否则解包成字节数。"""
    try:
        return np.bitwise_count(a)
    except AttributeError:
        flat = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape + (8,))
        return np.unpackbits(flat, axis=-1).sum(axis=-1)


def _dct2(x: np.ndarray) -> np.ndarray:
    """32×32 二维 DCT-II（等价 cv2.dct）。"""
    return _C @ x @ _C.T


def _phash_bits(gray32: np.ndarray) -> np.uint64:
    d = _dct2(gray32.astype(np.float32))[:KEEP, :KEEP]
    flat = d.flatten()[1:]                      # 丢掉 DC 分量
    bits = (flat > np.median(flat)).astype(np.uint64)
    v = np.uint64(0)
    for b in bits:
        v = (v << np.uint64(1)) | np.uint64(int(b))
    return v


def phash_variants(path: str) -> list[np.uint64] | None:
    """返回 [原图, 原图镜像, 中心80%裁剪, 裁剪镜像] 四个 64bit pHash；失败返回 None。"""
    try:
        im = Image.open(path)
        if getattr(im, "is_animated", False):
            im.seek(0)
        # 关键提速：JPEG 让解码器直接出小图，别整张解码（实测 300ms → 26ms/张）
        if im.format == "JPEG" and im.mode in ("RGB", "L", "CMYK"):
            im.draft("RGB", (64, 64))
        im = im.convert("RGB")
    except Exception:
        return None

    w, h = im.size
    crops = [im]
    if w > 40 and h > 40:
        dx, dy = int(w * 0.1), int(h * 0.1)
        crops.append(im.crop((dx, dy, w - dx, h - dy)))

    out: list[np.uint64] = []
    for c in crops:
        g = np.asarray(c.convert("L").resize((DCT_N, DCT_N), Image.LANCZOS), dtype=np.float32)
        out.append(_phash_bits(g))
        m = np.asarray(
            c.convert("L").transpose(Image.FLIP_LEFT_RIGHT).resize((DCT_N, DCT_N), Image.LANCZOS),
            dtype=np.float32,
        )
        out.append(_phash_bits(m))
    return out


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def work_one(job: tuple[str, str, bool]):
    """模块级 worker（Windows 多进程要求可序列化，不能是闭包）。"""
    d, name, want_md5 = job
    p = os.path.join(d, name)
    return name, os.path.getsize(p), (md5_of(p) if want_md5 else ""), phash_variants(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="图集 MD5+pHash 去重体检（只读·两类通用）")
    ap.add_argument("--dir", required=True, help="要体检的图片目录（两类通用，换目录即换类）")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--hamm", type=int, default=HAMM)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0＝全部")
    ap.add_argument("--no-md5", action="store_true", help="跳过 MD5，只算 pHash")
    ap.add_argument("--every", type=int, default=100)
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_dir():
        print(f"[X] 目录不存在：{d}", file=sys.stderr)
        return 2
    out_md = Path(args.out_md) if args.out_md else d.parent / "_dedup_report.md"
    out_csv = Path(args.out_csv) if args.out_csv else d.parent / "_dedup_groups.csv"
    hamm = args.hamm
    want_md5 = not args.no_md5

    names = sorted(e for e in os.listdir(d) if os.path.isfile(d / e))
    if args.limit and args.limit < len(names):
        names = names[: args.limit]
    total = len(names)
    print(f"[1/4] 目录：{d}")
    print(f"[1/4] 扫描完成：{total} 个文件｜workers={args.workers}｜判重阈值 ≤ {hamm}"
          f"｜MD5={'跳过' if not want_md5 else '计算'}", flush=True)

    t0 = time.perf_counter()
    results = []
    jobs = [(str(d), n, want_md5) for n in names]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            it = pool.imap_unordered(work_one, jobs, chunksize=8)
            for k, item in enumerate(it, 1):
                results.append(item)
                if k % args.every == 0 or k == total:
                    el = time.perf_counter() - t0
                    rate = k / el if el > 0 else 0
                    print(f"      {k}/{total} ({k*100//total}%)｜{el:5.1f}s｜"
                          f"{rate:5.0f} 张/s｜剩余约 {(total-k)/rate if rate else 0:4.1f}s", flush=True)
    else:
        for k, job in enumerate(jobs, 1):
            results.append(work_one(job))
            if k % args.every == 0 or k == total:
                el = time.perf_counter() - t0
                rate = k / el if el > 0 else 0
                print(f"      {k}/{total} ({k*100//total}%)｜{el:5.1f}s｜"
                      f"{rate:5.0f} 张/s｜剩余约 {(total-k)/rate if rate else 0:4.1f}s", flush=True)
    print(f"[2/4] 哈希完成：{len(results)} 张，用时 {time.perf_counter()-t0:.1f}s", flush=True)

    sizes, md5s, hashes, ok_names = {}, {}, [], []
    for n, sz, h, hv in results:
        sizes[n] = sz
        md5s[n] = h
        if hv is not None:
            hashes.append(hv)
            ok_names.append(n)
    print(f"      可 pHash：{len(ok_names)} 张｜解码失败：{len(names)-len(ok_names)} 张", flush=True)

    md5_dups: dict[str, list[str]] = {}
    if want_md5:
        by_md5: dict[str, list[str]] = {}
        for n in names:
            by_md5.setdefault(md5s[n], []).append(n)
        md5_dups = {k: v for k, v in by_md5.items() if len(v) > 1}

    arr = np.array(hashes, dtype=np.uint64)
    N, V = arr.shape
    print(f"[3/4] 两两比对：{N} 张 → {N*(N-1)//2} 对｜变体组合 {V}×{V}", flush=True)
    best = np.full((N, N), 255, dtype=np.uint8)
    t1 = time.perf_counter()
    for a in range(V):
        for b in range(V):
            dist = _popcount_u64(arr[:, a][:, None] ^ arr[:, b][None, :]).astype(np.uint8)
            np.minimum(best, dist, out=best)
    print(f"      比对完成，用时 {time.perf_counter()-t1:.1f}s", flush=True)

    iu = np.triu_indices(N, 1)
    dists = best[iu]
    pairs = np.argwhere(np.triu(best <= hamm, 1))
    print(f"      阈值 ≤ {hamm} 的近似对：{len(pairs)} 对 → 并查集分组", flush=True)

    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i, j in pairs:
        union(int(i), int(j))

    groups: dict[int, list[int]] = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    near = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)

    print(f"[4/4] 近似重复 {len(near)} 组 / MD5 重复 {len(md5_dups)} 组 → 写报告…", flush=True)

    L: list[str] = []
    L.append("# 图集去重体检报告\n")
    L.append(f"- 目录：`{d}`（只读体检，**未改动任何文件**）")
    L.append(f"- 文件总数：**{len(names)}**｜可用于 pHash：**{len(ok_names)}**")
    L.append(f"- 判重口径：64 位 pHash × 4 变体（原图／镜像／中心 80% 裁剪／裁剪镜像）取最小汉明距离，**≤ {hamm} 判近似重复**")
    L.append(f"- MD5 完全重复：**{len(md5_dups)} 组**，涉及 **{sum(len(v) for v in md5_dups.values())} 个文件**")
    L.append(f"- pHash 近似重复：**{len(near)} 组**，涉及 **{sum(len(g) for g in near)} 张**"
             f"（可归并 {sum(len(g) for g in near) - len(near)} 张）")
    L.append("")
    L.append("## 距离分布（判断阈值是否合适）\n")
    edges = [0, 2, 4, 6, 8, 10, 12, 16, 24, 32, 64]
    hist, _ = np.histogram(dists, bins=edges)
    for k in range(len(hist)):
        L.append(f"- {edges[k]}–{edges[k+1]}：{hist[k]}")
    L.append("")
    L.append("> 经验：真正同图通常 0–8；10–16 多为「同角色不同画」；>24 基本无关。")
    L.append("> 若 4–8 段数量陡增而 10+ 段也大，说明阈值接近边界，必须人工抽查。")
    L.append("")

    L.append(f"## 近似重复分组（共 {len(near)} 组，按组内张数降序）\n")
    L.append("> 档位：**A＝组内最小距离 ≤4（可直接归并）**｜**B＝6–8（必须人工过目，实测有误报）**")
    L.append("> `跨度` ＝ 组内编号 max−min：跨度大（≥50）多为同一张图重复下载；跨度小（≤3）多为同系列/调色版。")
    L.append("> ⚠️ 跨度只是提示，**不能替代看图**——实测跨度 249 的反而是误报（两张完全不同的图）。")
    L.append("")
    L.append("| 组 | 档 | 张数 | 跨度 | 建议保留 | 组内文件 |")
    L.append("|---|---|---|---|---|---|")
    csv_rows = [("group_id", "grade", "size", "span", "keep", "members", "min_hamming")]
    nA = nB = 0
    for gi, g in enumerate(near, 1):
        gn = sorted(ok_names[i] for i in g)
        try:
            # 文件名两种命名都要吃得下：`123.png`（luo）与 `12345_p0.webp`（miku）
            nums = [int(re.match(r"(\d+)", os.path.basename(x)).group(1)) for x in gn]
            span = max(nums) - min(nums)
        except (ValueError, AttributeError):
            span = -1
        keep = max(gn, key=lambda n: sizes.get(n, 0))
        sub = np.array([[best[i, j] for j in g] for i in g])
        mh = int(sub[sub > 0].min()) if (sub > 0).any() else 0
        grade = "A" if mh <= 4 else "B"
        nA += grade == "A"
        nB += grade == "B"
        L.append(f"| {gi} | {grade} | {len(g)} | {span} | `{keep}` | "
                 f"{', '.join(f'`{x}`' for x in gn[:10])}{' …' if len(gn) > 10 else ''} |")
        csv_rows.append((gi, grade, len(g), span, keep, "|".join(gn), mh))
    L.append("")
    L.append(f"**A 档 {nA} 组｜B 档 {nB} 组**（B 档请逐组看图再决定，**不要批量删**）")

    if md5_dups:
        L.append("")
        L.append("## MD5 完全重复（前 20 组）\n")
        L.append("| MD5 前 8 位 | 文件 |")
        L.append("|---|---|")
        for k, v in list(md5_dups.items())[:20]:
            L.append(f"| `{k[:8]}` | {', '.join(f'`{x}`' for x in v)} |")

    out_md.write_text("\r\n".join(L), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for r in csv_rows:
            w.writerow(r)

    print(f"近似重复 {len(near)} 组（A {nA} / B {nB}）｜MD5 重复 {len(md5_dups)} 组", flush=True)
    print(f"报告：{out_md}", flush=True)
    print(f"分组表：{out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
