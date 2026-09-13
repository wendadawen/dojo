#!/usr/bin/env python3
"""清掉页面里残留的工具注入属性。

部分页面的每一个元素上都挂着 data-page-node-id="…"，是某次编辑工具的调试
残留，一直随站点公开着。这些属性对读者没有任何作用，也会让 HTML 无谓膨胀。

    python3 .dojo/scripts/strip_injected_attrs.py            # 只报告
    python3 .dojo/scripts/strip_injected_attrs.py --fix      # 写回

同时把 tags 收集出来，便于判断注入来源是否只此一种。
"""

from __future__ import annotations

import argparse
import collections
import glob
import re
import sys
from pathlib import Path

# 形如  data-page-node-id="xxxx"  （前面可能有空格，也可能没有）
INJECTED_RE = re.compile(r'\s*data-page-node-id="[^"]*"')
DISCOVER_RE = re.compile(r'\s(data-[a-z-]+)="')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    targets = sorted(glob.glob("wiki/**/*.html", recursive=True)) + sorted(
        glob.glob(".dojo/templates/**/*.html", recursive=True)
    )
    total = 0
    found_attrs: collections.Counter[str] = collections.Counter()
    affected = []
    for path in targets:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        hits = INJECTED_RE.findall(text)
        found_attrs.update(DISCOVER_RE.findall(text))
        if not hits:
            continue
        affected.append((path, len(hits)))
        total += len(hits)
        if args.fix:
            Path(path).write_text(INJECTED_RE.sub("", text), encoding="utf-8")

    if not affected:
        print(f"no injected attributes in {len(targets)} files")
        return 0

    print(f"{'已清除' if args.fix else '发现'} {total} 处注入属性，涉及 {len(affected)} 个文件：")
    for path, count in affected:
        print(f"  {count:>5}  {path}")
    print("\n页面里出现过的全部 data-* 属性（确认注入源只有一种）：")
    for name, count in found_attrs.most_common():
        mark = " ← 注入残留" if name == "data-page-node-id" else ""
        print(f"  {count:>5}  {name}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
