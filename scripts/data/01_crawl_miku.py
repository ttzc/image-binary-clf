#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mikupara 初音未来图集采集器 v2
================================================================
用途：AI 概论 第一次作业（分类器 · 洛天依 vs 初音未来）的**初音类**数据源。
     洛天依类已在 `data/raw/luo`（1678 张，已改名＋已去重）。

⭐ v3 关键变化（2026-10-30 18:59 zaochen 指出两处）：
  1. **清单挪到图片目录之外**：默认 `<父目录>/_manifest_<目录名>.csv`
     （即 `data/raw/_manifest_miku.csv`）——**图片目录里只留图片**。
  2. **落一张写一条**：原来清单在整轮跑完才落盘，中途中断 → 已下的图没账。
     现在每成功保存一张立刻 append 一行，中断最多只丢"正在下的那一张"。
     ⚠️ 附带修掉一个隐患：`utf-8-sig` 在 append 模式下每次都会再写一个 BOM，
     逐张追加会把 csv 写坏 → 改成"新建才写 BOM，追加用纯 utf-8"。
  3. 新增 `--backfill` / `--backfill-only`：把目录里已有、清单却缺记录的图补登记
     （补的行 `source_url` 留空、`orientation` 由宽高比推定）。

⭐ v2 关键变化（2026-10-30 实测后改）：
  1. **默认走 `/api/`（不是 `/api/raw`）** —— 服务端返回 **WEBP**，与 /raw 的原图
     **分辨率完全相同**，但体积小 2–14 倍。3 组同作品号对照（webp / 原jpg）：
     0.54 vs 7.62 MB｜2.38 vs 4.38 MB｜5.72 vs 14.38 MB。
  2. 同分辨率下 WEBP 与原图 JPEG 的 **PSNR = 42.7 / 37.0 / 39.5 dB**
     （>35 dB 视觉难辨）→ **不影响特征提取**；1:1 裁剪看线条锐利、无色块。
  3. **支持多图作品**：Location 里除 `_p0` 还有 `_p1`、`_p2`…
     → 去重键改成 `{pixiv_id}_p{n}`（同作品不同页是**不同的画**，不能算重复）。
  4. 默认 **不缩图**（`--max-side 0`）：流量已由 WEBP 省掉，原图留着以后还能做
     pHash／放大／换模型；**从大往小永远可逆，反过来不可逆**。

接口行为（2026-10-30 实测）：
    GET https://api.mikupara.com/api/       → 302 → {horizontal|vertical}_pics/{id}_p{n}.webp
    GET https://api.mikupara.com/api/raw    → 302 → {horizontal|vertical}_pics/{id}_p{n}.jpg|png
    ⚠️ 两个端点都**不接受 `?w=` / `?width=` / `?size=`**（实测被忽略）；`/api/512` 直接失败；
       猜 `thumbnail_pics/` 会返回 **25,343 字节的 HTML 404 页**
       —— 所以脚本必须校验图片魔数，否则会把 404 网页存成 .jpg。

用法（他在自己机器上手动跑）：
    <python> -u -X utf8 miku_crawler.py --count 300 --out "data/raw/miku"

常用参数：
    --count 300           目标张数（作业要求 200–300 张/类）
    --out data/raw/miku  输出目录（已存在则续抓）
    --endpoint api        api＝WEBP（默认，省流量）｜raw＝原图 JPEG/PNG
    --max-side 0          0＝保留原尺寸（默认）；给 512/768 就缩到该长边并存 JPEG
    --delay 0.3           每次请求间隔秒
    --max-tries           最大请求次数（默认 count*6）
    --backfill            先把"目录里有、清单里没有"的图补登记，再继续抓
    --backfill-only       只补登记，不下载

清单默认落在图片目录**外面**（`data/raw/_manifest_miku.csv`），目录里只留图片。

特性：断点续传（读 manifest）｜三重校验（302 → 正则 → 图片魔数）｜
      内容级去重（sha256）｜多图作品按页区分｜失败不中断｜实测 ETA 与体积统计

⚠️ 脚本**不做**的事：
  · **多角色合照剔除**（要看图，脚本判不了）—— 必须人工过一遍
  · 部分图带**画师签名/水印小字**（实测有），是否保留由你定
  · 感知哈希（pHash）近重复 —— 沿用 `dedup_luo.py` 那套逻辑（在 .clash 里）
  · train/val 划分

退出码：0＝正常（即使部分失败）｜1＝一张没拿到｜2＝参数/环境问题
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("[X] 缺 requests：pip install requests", file=sys.stderr)
    raise SystemExit(2)

try:
    from PIL import Image
except ImportError:
    Image = None

BASE = "https://api.mikupara.com/api"
# 例：https://www.mikupara.com/horizontal_pics/128158158_p1.webp
LOC_PAT = re.compile(r"/(horizontal|vertical)_pics/(\d+)_p(\d+)\.(png|jpg|jpeg|webp)", re.I)

FIELDS = ["key", "pixiv_id", "page", "orientation", "format",
          "orig_w", "orig_h", "orig_bytes", "saved_bytes",
          "sha256", "filename", "source_url"]


def sniff_format(head: bytes) -> str | None:
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def load_manifest(path: Path) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                k = (row.get("key") or row.get("pixiv_id") or "").strip()
                if k:
                    seen[k] = row
    return seen


def append_manifest(path: Path, rows: list[dict]) -> None:
    """逐行追加；文件新建时才写 BOM 和表头。

    ⚠️ 必须区分编码：`utf-8-sig` 在 append 模式下**每次都会再写一个 BOM**，
    逐张追加会把文件写坏 → 已存在且非空时用纯 `utf-8`。
    """
    new_file = (not path.exists()) or path.stat().st_size == 0
    enc = "utf-8-sig" if new_file else "utf-8"
    with path.open("a", encoding=enc, newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def backfill(out: Path, mpath: Path, seen: dict[str, dict]) -> int:
    """把目录里已有、但清单里没有记录的图片补登记（上次中断的遗留）。

    ⚠️ 补的行：`source_url` 留空（原始下载地址已不可知）；
    `orientation` 由宽高比推定（w>=h → horizontal），不是从 URL 来的。
    """
    added = 0
    pat = re.compile(r"^(\d+)_p(\d+)\.(png|jpg|jpeg|webp)$", re.I)
    for p in sorted(out.iterdir()):
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        key = f"{m.group(1)}_p{m.group(2)}"
        if key in seen:
            continue
        raw = p.read_bytes()
        fmt = sniff_format(raw[:16]) or m.group(3).lower()
        w = h = 0
        if Image is not None:
            try:
                w, h = Image.open(p).size
            except Exception:                       # noqa: BLE001
                pass
        row = {
            "key": key, "pixiv_id": m.group(1), "page": m.group(2),
            "orientation": ("horizontal" if w >= h else "vertical") if w and h else "",
            "format": fmt, "orig_w": w, "orig_h": h,
            "orig_bytes": len(raw), "saved_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "filename": p.name, "source_url": "",
        }
        append_manifest(mpath, [row])
        seen[key] = row
        added += 1
        print(f"  ↺ 补登记 {p.name}")
    return added


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/126.0 Safari/537.36"),
        "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.8",
        "Referer": "https://www.mikupara.com/",
    })
    return s


def fetch_one(sess: requests.Session, endpoint: str, timeout: float):
    """返回 (location_url, orientation, pixiv_id, page, ext)；失败抛异常。"""
    url = f"{BASE}/{endpoint}" if endpoint == "raw" else f"{BASE}/"
    r = sess.get(url, allow_redirects=False, timeout=timeout)
    if r.status_code not in (301, 302, 303, 307, 308):
        raise RuntimeError(f"接口没给重定向：HTTP {r.status_code}")
    loc = r.headers.get("Location", "")
    m = LOC_PAT.search(loc)
    if not m:
        raise RuntimeError(f"Location 解析失败：{loc[:120]}")
    return loc, m.group(1).lower(), m.group(2), m.group(3), m.group(4).lower()


def shrink(raw: bytes, max_side: int, quality: int) -> tuple[bytes, str, int, int]:
    """按需缩放；返回 (落盘字节, 后缀, 原宽, 原高)。max_side<=0 时后缀为空＝原样保存。

    即使不缩放也读一下尺寸（PIL 只解析头部、不解码像素，几乎不耗时），
    这样 manifest 里能留下原始分辨率，写报告时不用再回头看图。
    """
    im = None
    w = h = 0
    if Image is not None:
        im = Image.open(io.BytesIO(raw))
        w, h = im.size
    if max_side <= 0:
        return raw, "", w, h
    if im is None:
        raise RuntimeError("用了 --max-side 但没装 Pillow：pip install pillow")
    im = im.convert("RGB")
    scale = max_side / max(w, h)
    if scale < 1:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), "jpg", w, h


def main() -> int:
    ap = argparse.ArgumentParser(description="mikupara 初音图集采集器 v2")
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--out", default="data/raw/miku")
    ap.add_argument("--endpoint", choices=["api", "raw"], default="api")
    ap.add_argument("--max-side", type=int, default=0, help="0＝保留原尺寸（默认）")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-tries", type=int, default=0)
    ap.add_argument("--manifest", default="",
                    help="清单路径；默认放图片目录之外：<父目录>/_manifest_<目录名>.csv")
    ap.add_argument("--backfill", action="store_true",
                    help="先补登记目录里已有、但清单缺记录的图片（中断遗留），再继续抓")
    ap.add_argument("--backfill-only", action="store_true", help="只补登记，不下载")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # ⭐ 清单默认放在图片目录**外面**——目录里只留图片
    mpath = Path(args.manifest) if args.manifest else out.parent / f"_manifest_{out.name}.csv"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    seen = load_manifest(mpath)
    start_have = len(seen)
    max_tries = args.max_tries or max(args.count * 6, args.count + 20)

    mode = "保留原尺寸" if args.max_side <= 0 else f"缩到长边 {args.max_side}px（JPEG q{args.quality}）"
    ep_label = "/api/" if args.endpoint == "api" else "/api/raw"
    print(f"[i] 目标 {args.count} 张 | 端点 {ep_label} | 图片目录 {out}")
    print(f"[i] 清单 {mpath}（在图片目录之外）｜已有记录 {start_have} 条")
    print(f"[i] 图像处理：{mode} | 间隔 {args.delay}s | 最多请求 {max_tries} 次")

    if args.backfill or args.backfill_only:
        n = backfill(out, mpath, seen)
        print(f"[i] 补登记完成：新增 {n} 条（清单总数 {len(seen)}）")
        if args.backfill_only:
            return 0

    sess = build_session()
    got = dup = fail = 0
    dl_bytes = sv_bytes = 0
    t0 = time.time()

    for attempt in range(1, max_tries + 1):
        if got >= args.count:
            break
        try:
            loc, orientation, pid, page, _ext = fetch_one(sess, args.endpoint, args.timeout)
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print(f"  ! #{attempt} 取重定向失败：{e}")
            time.sleep(args.delay)
            continue

        key = f"{pid}_p{page}"
        if key in seen:
            dup += 1
            time.sleep(args.delay)
            continue

        try:
            img = sess.get(loc, timeout=args.timeout)
            img.raise_for_status()
            raw = img.content
            fmt = sniff_format(raw[:16])
            if fmt is None:
                raise RuntimeError("返回内容不是图片（可能是 404 HTML 页）")
            if len(raw) < 1024:
                raise RuntimeError(f"图片过小（{len(raw)} B），疑似占位")

            sha = hashlib.sha256(raw).hexdigest()
            if any(r.get("sha256") == sha for r in seen.values()):
                dup += 1
                time.sleep(args.delay)
                continue

            data, new_ext, w, h = shrink(raw, args.max_side, args.quality)
            ext = new_ext or fmt
            fname = f"{pid}_p{page}.{ext}"
            fpath = out / fname
            if fpath.exists():
                dup += 1
                time.sleep(args.delay)
                continue

            fpath.write_bytes(data)
            row = {
                "key": key, "pixiv_id": pid, "page": page,
                "orientation": orientation, "format": fmt,
                "orig_w": w, "orig_h": h, "orig_bytes": len(raw),
                "saved_bytes": len(data), "sha256": sha,
                "filename": fname, "source_url": loc,
            }
            seen[key] = row
            append_manifest(mpath, [row])   # ⭐ 落一张写一条，中断也不丢账
            got += 1
            dl_bytes += len(raw)
            sv_bytes += len(data)

            if got % 20 == 0 or got <= 5:
                el = time.time() - t0
                eta = el / got * (args.count - got)
                print(f"  + [{got}/{args.count}] {fname}  {len(raw)/1024:.0f} KB → "
                      f"{len(data)/1024:.0f} KB  ({orientation})  "
                      f"已用 {el/60:.1f}min，预计还要 {eta/60:.1f}min")
        except Exception as e:                        # noqa: BLE001
            fail += 1
            print(f"  ! #{attempt} 下载失败 {loc[:90]}：{e}")

        time.sleep(args.delay)

    total = got + dup + fail
    el = time.time() - t0
    print("\n" + "=" * 62)
    print(f"完成：新下 {got} 张 | 重复跳过 {dup} | 失败 {fail} | 共请求 {total} 次 | 耗时 {el/60:.1f} min")
    if got:
        print(f"体积：下载 {dl_bytes/1048576:.1f} MB → 落盘 {sv_bytes/1048576:.1f} MB"
              f"（平均 {sv_bytes/got/1024:.0f} KB/张）")
        print(f"重复率 {dup/max(total,1):.1%} | 平均 {el/max(got,1):.2f}s/张")
    print(f"目录内记录总数：{start_have} → {len(seen)}")
    print(f"清单：{mpath}")

    if got == 0:
        print(f'[X] 一张都没拿到。先手动试：curl -I "{BASE}/"')
        return 1
    if got < args.count:
        print(f"[!] 还差 {args.count - got} 张（池子可能不够大）。"
              f"调大 --max-tries 再跑同一条命令即可续抓。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
