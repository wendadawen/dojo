#!/usr/bin/env python3
"""把关系图孤岛的内链编辑应用到页面（或撤销）。

读取工作流产出的编辑清单，逐条插入。处理三类已知问题：
  · 重复：同一宿主文件、同一锚点、同一链接目标出现两次——只应用一次
  · 叠放：同一锚点被两条不同目标的编辑命中——保留优先给孤岛带来入链的那条
  · 硬凑：语义与宿主页无关的——由 SKIP 显式排除

同一文件的多条编辑必须**依次**作用于不断更新的文本；若每条都从原始文本写入，
后写的会覆盖先写的，同文件只有最后一条能落地。

    python3 .dojo/scripts/apply_link_edits.py <清单.json>            # 报告
    python3 .dojo/scripts/apply_link_edits.py <清单.json> --fix      # 应用
    python3 .dojo/scripts/apply_link_edits.py <清单.json> --undo     # 撤销
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 核验判定为硬凑，不应用：锚点所在段落与插入内容语义无关
SKIP = {("rmsnorm", "先算加权和，再过激活")}


def collect(proposals: list[dict]) -> list[dict]:
    """展平成编辑列表。入链编辑排在前——它才是消除入度 0 的那一侧，
    与出链编辑争同一个锚点时应当优先保留。"""
    incoming, outgoing = [], []
    for item in proposals:
        for edit in item.get("incoming_edits") or []:
            incoming.append(
                {
                    "host": edit["host_slug"],
                    "anchor": edit["anchor_text"],
                    "insert": edit["insert_html"],
                    "island": item["island"],
                    "kind": "in",
                }
            )
        for edit in item.get("outgoing_edits") or []:
            outgoing.append(
                {
                    "host": item["island"],
                    "anchor": edit["anchor_text"],
                    "insert": edit["insert_html"],
                    "island": item["island"],
                    "kind": "out",
                }
            )
    return incoming + outgoing


def select(edits: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    """按宿主文件分组后逐条判定，返回可应用清单与被跳过清单。"""
    by_host: dict[str, list[dict]] = defaultdict(list)
    for edit in edits:
        by_host[edit["host"]].append(edit)

    kept, dropped = [], []
    for host, group in by_host.items():
        path = Path("wiki") / host / "index.html"
        if not path.exists():
            for edit in group:
                dropped.append((edit, "宿主页不存在"))
            continue
        text = path.read_text(encoding="utf-8")
        used_anchor: set[str] = set()
        for edit in group:
            if (edit["island"], edit["anchor"]) in SKIP:
                dropped.append((edit, "核验判定为硬凑"))
                continue
            if edit["anchor"] in used_anchor:
                dropped.append((edit, "同锚点已被占用（重复或叠放）"))
                continue
            if text.count(edit["anchor"]) != 1:
                dropped.append(
                    (edit, f"锚点在宿主页出现 {text.count(edit['anchor'])} 次")
                )
                continue
            used_anchor.add(edit["anchor"])
            kept.append({**edit, "path": path})
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edits")
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--undo", action="store_true")
    args = parser.parse_args()

    proposals = json.loads(Path(args.edits).read_text(encoding="utf-8"))
    if isinstance(proposals, dict):
        proposals = proposals["proposals"]
    edits = collect(proposals)

    if args.undo:
        # 逐文件把 anchor+insert 还原成 anchor，从后往前避免偏移失效
        by_host: dict[str, list[dict]] = defaultdict(list)
        for edit in edits:
            by_host[edit["host"]].append(edit)
        removed = 0
        for host, group in by_host.items():
            path = Path("wiki") / host / "index.html"
            text = path.read_text(encoding="utf-8")
            for edit in group:
                pair = edit["anchor"] + edit["insert"]
                if pair in text:
                    text = text.replace(pair, edit["anchor"], 1)
                    removed += 1
            path.write_text(text, encoding="utf-8")
        print(f"撤销 {removed} 处插入，涉及 {len(by_host)} 个文件")
        return 0

    kept, dropped = select(edits)
    print(f"清单 {len(edits)} 条：应用 {len(kept)} 条，跳过 {len(dropped)} 条")
    for edit, why in dropped:
        print(f"  跳过 [{edit['island']}] host={edit['host']} {why}")
    if not args.fix:
        return 0

    # 同一文件的编辑依次作用于更新中的文本
    by_host: dict[str, list[dict]] = defaultdict(list)
    for edit in kept:
        by_host[edit["host"]].append(edit)
    for host, group in by_host.items():
        path = Path("wiki") / host / "index.html"
        text = path.read_text(encoding="utf-8")
        for edit in group:
            text = text.replace(edit["anchor"], edit["anchor"] + edit["insert"], 1)
        path.write_text(text, encoding="utf-8")
    print(f"已写入 {len(by_host)} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
