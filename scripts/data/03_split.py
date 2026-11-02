#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成固定的 train / val / test 划分（一次定稿，之后只读不重划）

为什么要有这个脚本
------------------------------------------------------------------
**划分必须固化。** 如果每次训都 `random_split`，两次实验的训练集就不同 ——
「这次涨了 2%」到底是改模型改的、还是数据换了，你永远说不清。
本脚本把划分**一次性写进 `splits.csv`**，训练/评测代码只读这份清单。

三条硬约束（比比例更重要）
  1. **按组划分，不按图划分** —— 同组的东西不能跨集合：
       · miku：组＝**pixiv 作品号**（同一作品的 p0/p1/p2 是同一张画的变体，
         分居 train/test 会直接虚高精度）
       · luo：组＝**去重分组**（若给了 `--luo-groups`，用 10-29 体检出的 A/B 档组；
         不给则每张自成一组）
  2. **分层** —— 每类在每个集合里的比例一致（1:1 配平后，val/test 不会偏向某一类）
  3. **固定种子** —— 同一条命令永远产出同一份划分

用法
------------------------------------------------------------------
① 生成（先看统计，不满意再调参重跑）：

    <python> -u -X utf8 03-生成划分.py --raw-root "data/raw" --out "data/splits.csv"

② 校验（**手改过 csv 之后一定要跑这个**）：

    <python> -u -X utf8 03-生成划分.py --check "data/splits.csv" --raw-root "data/raw"

常用参数
    --raw-root     原图根目录（其下有 luo/ 与 miku/）        默认 data/raw
    --out          划分清单输出路径                          默认 data/splits.csv
    --ratio        三类比例，逗号分隔                        默认 0.8,0.1,0.1
    --seed         随机种子（固定才可复现）                  默认 42
    --per-class    每类取多少张（0＝按较小一类自动配平）      默认 0
    --miku-p0-only 初音只保留 p0（每作品一张，最干净）        默认开启
    --no-miku-p0-only  关掉上一条（保留多页）
    --luo-groups   洛天依去重分组表（`_dedup_groups_20261029.csv`）；
                   给了就按组划分，能避免近重复跨集合
    --check        只校验已有 csv，不重新生成

产出 `splits.csv` 的列（**可以直接手工改**）
    path   相对 raw-root 的路径，如 `miku/133412207_p0.webp`
    label  0 = luo（洛天依）｜1 = miku（初音）
    group  分组键（miku＝作品号；luo＝去重组号或文件名）
    split  train / val / test  ← **你要改就改这一列**

⚠️ 手改的注意
  · 改完跑 `--check`，它会检查：文件是否存在、有没有同组跨集合、比例是否还在合理区间
  · **别单独搬一张多页图到别的集合** —— 它的兄弟页还在原集合，--check 会报泄漏
  · 想整体换一种比例，改 `--ratio` 重新生成更省事（会覆盖旧 csv）
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
CLASS_OF_DIR = {"luo": 0, "miku": 1}
NAME_OF_CLASS = {0: "luo", 1: "miku"}


# ----------------------------------------------------------------- 扫描
def scan_miku(root: Path, p0_only: bool) -> list[tuple[str, int, str]]:
    """返回 [(relpath, label, group)]；group＝pixiv 作品号。"""
    d = root / "miku"
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        m = re.match(r"^(\d+)_p(\d+)$", p.stem)
        if not m:
            continue                       # 不认识的文件名跳过（不会静默算进去）
        pid, page = m.group(1), int(m.group(2))
        if p0_only and page != 0:
            continue
        out.append((f"miku/{p.name}", 1, pid))
    return out


def load_luo_groups(csv_path: Path) -> dict[str, str]:
    """从去重分组表读 {文件名: 组键}；没有表就返回空。"""
    mapping: dict[str, str] = {}
    if not csv_path or not csv_path.exists():
        return mapping
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            members = (row.get("members") or "").split("|")
            gid = row.get("group_id") or ""
            for m in members:
                m = m.strip()
                if m:
                    mapping[m] = f"luo_g{gid}"
    return mapping


def scan_luo(root: Path, groups: dict[str, str]) -> list[tuple[str, int, str]]:
    d = root / "luo"
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        out.append((f"luo/{p.name}", 0, groups.get(p.name, f"luo_f{p.stem}")))
    return out


# ----------------------------------------------------------------- 划分
def _split_one(items: list[tuple[str, int, str]], ratio: tuple[float, float, float],
               seed: int) -> dict[str, str]:
    """单个类内部的组级划分：打乱组 → 贪心塞满 val/test 目标张数，其余进 train。"""
    by_group: dict[str, list[str]] = defaultdict(list)
    for rel, _lab, g in items:
        by_group[g].append(rel)

    total = len(items)
    targets = {s: total * r for s, r in zip(SPLITS, ratio)}
    got = {s: 0 for s in SPLITS}
    assign: dict[str, str] = {}

    groups = list(by_group.keys())
    random.Random(seed).shuffle(groups)

    # 先塞 val 与 test（目标小、难凑准），剩下的自然全归 train
    for target_split in ("val", "test"):
        for g in groups:
            if g in assign:
                continue
            n = len(by_group[g])
            if got[target_split] + n <= targets[target_split] + 1e-9:
                assign[g] = target_split
                got[target_split] += n
        # 一轮塞不满（组太大跳过了）→ 再补一轮，允许轻微超目标
        for g in groups:
            if g in assign:
                continue
            if got[target_split] >= targets[target_split] * 0.95:
                break
            assign[g] = target_split
            got[target_split] += len(by_group[g])

    for g in groups:
        assign.setdefault(g, "train")

    return {rel: assign[g] for rel, _lab, g in items}


def split_by_group(items: list[tuple[str, int, str]], ratio: tuple[float, float, float],
                   seed: int) -> dict[str, str]:
    """⭐ 按类**分别**做组级划分（真正的分层）。

    早期版本对全体算一次目标数，结果总数虽是 80/10/10，但两类各自的比例会飘
    （实测 luo 970/121/109 vs miku 950/119/131）——那叫"总数对齐"、不叫"分层"。
    现在每类各按 ratio 划一次，保证**每类的 val/test 张数都接近同比例**。
    """
    assign: dict[str, str] = {}
    for label in sorted({it[1] for it in items}):
        sub = [it for it in items if it[1] == label]
        assign.update(_split_one(sub, ratio, seed + label * 1000))
    return assign


def balance(items: list[tuple[str, int, str]], per_class: int, seed: int
            ) -> list[tuple[str, int, str]]:
    """按类抽到相同张数；按组整体抽，避免把同一作品的页拆散。"""
    by_label: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
    for it in items:
        by_label[it[1]].append(it)

    n_target = per_class or min(len(v) for v in by_label.values())
    rng = random.Random(seed)
    kept: list[tuple[str, int, str]] = []
    for label, lst in sorted(by_label.items()):
        if len(lst) <= n_target:
            kept.extend(lst)
            continue
        by_group: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        for it in lst:
            by_group[it[2]].append(it)
        gs = list(by_group.keys())
        rng.shuffle(gs)
        take: list[tuple[str, int, str]] = []
        for g in gs:
            if len(take) >= n_target:
                break
            take.extend(by_group[g])
        kept.extend(take[:n_target])        # 组内可截断（同组仍同侧，不影响泄漏判断）
    return kept


# ----------------------------------------------------------------- 校验
def check(csv_path: Path, root: Path) -> int:
    if not csv_path.exists():
        print(f"[X] 找不到 {csv_path}", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        print("[X] 清单为空", file=sys.stderr)
        return 2

    bad_split = [r for r in rows if r.get("split") not in SPLITS]
    missing = [r for r in rows if not (root / r["path"]).exists()]
    grp_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        grp_splits[r["group"]].add(r["split"])
    leaks = {g: s for g, s in grp_splits.items() if len(s) > 1}

    print(f"[i] 清单 {len(rows)} 行｜目录 {root}")
    print(f"{'类':<6}{'train':>8}{'val':>8}{'test':>8}{'合计':>8}")
    for label in (0, 1):
        sub = [r for r in rows if r.get("label") == str(label)]
        cnt = {s: sum(1 for r in sub if r["split"] == s) for s in SPLITS}
        print(f"{NAME_OF_CLASS[label]:<6}{cnt['train']:>8}{cnt['val']:>8}{cnt['test']:>8}{len(sub):>8}")
    total_by = {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS}
    print(f"{'合计':<6}{total_by['train']:>8}{total_by['val']:>8}{total_by['test']:>8}{len(rows):>8}")
    print(f"[i] 比例：train {total_by['train']/len(rows):.1%}｜"
          f"val {total_by['val']/len(rows):.1%}｜test {total_by['test']/len(rows):.1%}")
    print(f"[i] 分组数 {len(grp_splits)}")

    ok = True
    if bad_split:
        print(f"[X] {len(bad_split)} 行 split 列不合法（只认 train/val/test）", file=sys.stderr)
        ok = False
    if missing:
        print(f"[X] {len(missing)} 个文件不存在，例：{missing[0]['path']}", file=sys.stderr)
        ok = False
    if leaks:
        ex = list(leaks.items())[:3]
        print(f"[X] 🔴 {len(leaks)} 个组跨了集合（数据泄漏）：{ex}", file=sys.stderr)
        ok = False
    # 比例偏离告警（±5 个百分点）——总数与**每一类**都要看
    for want, s in zip((0.8, 0.1, 0.1), SPLITS):
        if abs(total_by[s] / len(rows) - want) > 0.05:
            print(f"[!] 总体 {s} 占比 {total_by[s]/len(rows):.1%}，偏离目标 {want:.0%} 超过 5 个点")
    for label in (0, 1):
        sub = [r for r in rows if r.get("label") == str(label)]
        if not sub:
            continue
        for want, s in zip((0.8, 0.1, 0.1), SPLITS):
            frac = sum(1 for r in sub if r["split"] == s) / len(sub)
            if abs(frac - want) > 0.05:
                print(f"[!] {NAME_OF_CLASS[label]} 的 {s} 占比 {frac:.1%}，偏离目标 {want:.0%} 超过 5 个点")
    print("[✓] 校验通过" if ok else "[X] 校验未通过")
    return 0 if ok else 1


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="生成固定的 train/val/test 划分清单")
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--ratio", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", type=int, default=0, help="0＝按较小一类自动配平")
    ap.add_argument("--miku-p0-only", dest="p0only", action="store_true", default=True)
    ap.add_argument("--no-miku-p0-only", dest="p0only", action="store_false")
    ap.add_argument("--luo-groups", default="data/_dedup_groups.csv")
    ap.add_argument("--check", default="", help="只校验这个 csv，不重新生成")
    args = ap.parse_args()

    root = Path(args.raw_root)
    if args.check:
        return check(Path(args.check), root)

    ratio = tuple(float(x) for x in args.ratio.split(","))
    if len(ratio) != 3 or abs(sum(ratio) - 1) > 1e-6:
        print("[X] --ratio 要三个数且和为 1", file=sys.stderr)
        return 2

    luo_groups = load_luo_groups(Path(args.luo_groups)) if args.luo_groups else {}
    items = scan_miku(root, args.p0only) + scan_luo(root, luo_groups)
    if not items:
        print(f"[X] 在 {root} 下没扫到图（应有 luo/ 与 miku/）", file=sys.stderr)
        return 2

    n_miku = sum(1 for i in items if i[1] == 1)
    n_luo = len(items) - n_miku
    print(f"[1/4] 扫描：luo {n_luo} 张｜miku {n_miku} 张"
          f"（p0-only={'是' if args.p0only else '否'}）")
    if luo_groups:
        n_g = len({i[2] for i in items if i[1] == 0})
        print(f"      luo 分组来自去重表：{n_g} 组")

    balanced = balance(items, args.per_class, args.seed)
    n_miku = sum(1 for i in balanced if i[1] == 1)
    n_luo = len(balanced) - n_miku
    print(f"[2/4] 配平后：luo {n_luo} 张｜miku {n_miku} 张｜合计 {len(balanced)}")
    if n_miku != n_luo:
        print("[!] 两类张数不等 —— 训练时记得 class_weight 或 sampler")

    assign = split_by_group(balanced, ratio, args.seed)
    print("[3/4] 组级分层划分完成")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "split"])
        for rel, lab, g in sorted(balanced):
            w.writerow([rel, lab, g, assign[rel]])
    print(f"[4/4] 已写入 {out}")

    return check(out, root)


if __name__ == "__main__":
    raise SystemExit(main())
