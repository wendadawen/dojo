#!/usr/bin/env python3
"""把 research/ 下的实测产物登记进 md，然后删除这些产物文件。

research/ 里除 .md 外的文件（实测脚本 .py、运行输出 .out、配置 .json、
模型源码快照 .cuh/.hpp 等）占了仓库的可观体积，且这些内容并不发布。
但直接删除会丢掉一件事：页面自称「实测」的依据——实测产物是否存在，
是「含实测」这一判断唯一可靠的信号，md 里并没有一致地记录过。

所以顺序是：先把清单写进 research/measured.md，再删文件。删掉之后，
「本页的实测产物是这些」这条记录仍在，只是产物本身不再随仓库分发。

    python3 .dojo/scripts/archive_research_artifacts.py            # 报告
    python3 .dojo/scripts/archive_research_artifacts.py --fix      # 登记并删除
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MARKER = "measured.md"
HEADER = """# 实测产物清单

本页的实测产物原先存放在本目录下，现已从仓库移除（内容不发布，且体积可观）。
下表是它们被移除时的登记，用于说明本页「含实测」的判定依据来自何处。

| 文件 | 体积 | 说明 |
|---|---|---|
"""


def describe(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "实测脚本",
        ".out": "运行输出存档",
        ".json": "配置或中间数据",
        ".sh": "执行脚本",
        ".cuh": "模型源码快照",
        ".hpp": "模型源码快照",
        ".txt": "提取的文本材料",
        ".pdf": "原始材料",
        ".html": "中间产物",
        ".js": "中间产物",
        ".jinja": "模型模板快照",
    }.get(suffix, "材料")


def human(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size / 1024:.1f} {unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    pages = 0
    files = 0
    total = 0
    for research in sorted(Path("wiki").glob("*/research")):
        artifacts = [
            item
            for item in sorted(research.rglob("*"))
            if item.is_file() and item.suffix != ".md"
        ]
        if not artifacts:
            continue
        pages += 1
        files += len(artifacts)
        rows = []
        for item in artifacts:
            rel = item.relative_to(research).as_posix()
            size = item.stat().st_size
            total += size
            rows.append(f"| `{rel}` | {human(size)} | {describe(item)} |")
        if args.fix:
            (research / MARKER).write_text(
                HEADER + "\n".join(rows) + "\n", encoding="utf-8"
            )
            for item in artifacts:
                item.unlink()
            for leftover in sorted(research.rglob("*"), reverse=True):
                if leftover.is_dir() and not any(leftover.iterdir()):
                    leftover.rmdir()

    verb = "已登记并删除" if args.fix else "将登记并删除"
    print(f"{verb}：{pages} 个页面的 {files} 个文件，共 {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
