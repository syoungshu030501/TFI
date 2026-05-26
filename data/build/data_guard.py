"""数据契约体检：把 data/ 各层的实际现状和阈值对一遍，结果写到 data/meta/data_health.md。

跑法：
    python scripts/data/guard.py
    python scripts/data/guard.py --strict   # 任一硬契约失败 exit 1（CI 用）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"


def _count(p: Path, exts: Optional[Tuple[str, ...]] = None) -> int:
    """统计目录下符合 exts 的文件数；exts=None 表示所有文件。"""
    if not p.exists():
        return -1
    if exts is None:
        return sum(1 for f in p.iterdir() if f.is_file())
    return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts)


def _is_link_alive(p: Path) -> bool:
    return p.is_symlink() and p.resolve().exists()


def _section(title: str, rows: List[Tuple[str, str, str]]) -> List[str]:
    out = [f"## {title}", "", "| 项 | 值 | 判定 |", "|---|---|:-:|"]
    for k, v, ok in rows:
        out.append(f"| {k} | {v} | {ok} |")
    out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="任一硬契约失败时 exit 1（CI 用）")
    args = ap.parse_args()

    failures: List[str] = []
    md: List[str] = ["# data 健康报告", "",
                     f"`{DATA.relative_to(ROOT)}/` 当前快照（由 scripts/data/guard.py 生成）", ""]

    # --- ① raw 层 ---
    raw_rows = []
    for name, expected_min in [("train_resume/Black/Image", 600),
                               ("train_resume/White/Image", 100),
                               ("val/Black/Image", 100),
                               ("val/White/Image", 30),
                               ("test/Image", 100)]:
        p = DATA / "raw" / name
        c = _count(p)
        ok = "✓" if c >= expected_min else "✗"
        if c < expected_min:
            failures.append(f"raw/{name} count={c} < {expected_min}")
        raw_rows.append((f"`raw/{name}`", str(c) if c >= 0 else "MISSING", ok))
    md += _section("① raw 原料层", raw_rows)

    # --- ② processed 层 ---
    proc_rows = []
    for name, expected_min, exts in [
        ("synth/Image", 50, None),
        ("synth/Mask", 50, None),
        ("real_ext/Image", 800, None),
        ("real_ext/Caption", 800, (".md", ".txt")),
    ]:
        p = DATA / "processed" / name
        c = _count(p, exts=exts)
        ok = "✓" if c >= expected_min else "✗"
        if c < expected_min:
            failures.append(f"processed/{name} count={c} < {expected_min}")
        proc_rows.append((f"`processed/{name}`", str(c) if c >= 0 else "MISSING", ok))

    keep_file = DATA / "processed" / "synth" / "keep.txt"
    if keep_file.exists():
        n_keep = sum(1 for _ in keep_file.open(encoding="utf-8") if _.strip())
        proc_rows.append(("`processed/synth/keep.txt`", f"{n_keep} 行", "✓"))
    else:
        proc_rows.append(("`processed/synth/keep.txt`", "MISSING", "✗"))
        failures.append("processed/synth/keep.txt missing")

    v2_dir = DATA / "processed" / "caption_local_v2"
    if v2_dir.exists():
        v2_lines = sum(sum(1 for _ in p.open(encoding="utf-8"))
                       for p in v2_dir.glob("*.jsonl"))
        proc_rows.append(("`processed/caption_local_v2/*.jsonl`",
                          f"{v2_lines} 行（旧本地 9B）", "—"))
    md += _section("② processed 处理层", proc_rows)

    # --- ③ vlm 层（新 API 蒸馏目标）---
    vlm_rows = []
    api_dir = DATA / "vlm" / "caption_api_v3"
    if api_dir.exists():
        api_lines = sum(sum(1 for _ in p.open(encoding="utf-8"))
                        for p in api_dir.glob("*.jsonl"))
        target = 1280  # 640 stem × 2 versions
        ok = "✓" if api_lines >= target * 0.9 else "⏳"
        vlm_rows.append(("`vlm/caption_api_v3/*.jsonl`",
                         f"{api_lines}/{target} 行", ok))
        if api_lines < target * 0.9:
            failures.append(f"vlm/caption_api_v3 {api_lines} < {int(target * 0.9)} (90% 准入线)")
    else:
        vlm_rows.append(("`vlm/caption_api_v3/`", "未生成", "⏳"))
        failures.append("vlm/caption_api_v3 missing")
    md += _section("③ vlm 层（API 蒸馏）", vlm_rows)

    # --- ④ symlink 健康 ---
    link_rows = []
    for rel in ["raw/train_resume", "raw/val", "raw/test",
                "processed/synth", "processed/real_ext",
                "processed/caption_local_v2", "vlm/caption_api_v3"]:
        p = DATA / rel
        alive = _is_link_alive(p)
        link_rows.append((f"`data/{rel}`",
                          str(p.resolve()) if p.exists() else "BROKEN",
                          "✓" if alive else "✗"))
        if not alive:
            failures.append(f"symlink data/{rel} broken")
    md += _section("④ symlink 健康", link_rows)

    # --- ⑤ ckpt 进度 ---
    ckpt_rows = []
    for name in ["seg/segformer_fold0", "seg/segformer_fold1", "seg/segformer_fold2",
                 "seg/segformer_fold3", "seg/segformer_fold4",
                 "cls/efficientnet_fold0", "calibrator", "qwen35_9b"]:
        p = ROOT / "checkpoints" / name
        ckpt_rows.append((f"`checkpoints/{name}`",
                          "存在" if p.exists() else "缺失",
                          "✓" if p.exists() else "·"))
    md += _section("⑤ checkpoint 现状（参考，不卡阻）", ckpt_rows)

    # --- 汇总 ---
    if failures:
        md.append("## 失败的硬契约")
        md.append("")
        for f in failures:
            md.append(f"- {f}")
        md.append("")
    else:
        md.append("## 状态：✅ 全部硬契约通过")
        md.append("")

    out_path = DATA / "meta" / "data_health.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[guard] wrote {out_path.relative_to(ROOT)}")
    print(f"[guard] {len(failures)} failure(s)")

    if failures and args.strict:
        for f in failures:
            print(f"  ✗ {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
