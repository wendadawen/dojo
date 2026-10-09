"""概念页三个脚本共用的常量、「依据」检查和自检输出。

依据写法见 guides/concept/outline.md；位置是标题里的一段文字，或行号范围如 L120-L180。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

LONG_LIMIT = 5000   # 长页折叠块外的字数上限
SHORT_LIMIT = 1000  # 短页的字数上限
SHORT_SECTIONS = 3
LAST_CHAPTER = "适用条件与边界"
SELF_MADE = "自行构造"
LINES_RE = re.compile(r"[Ll](\d+)(?:\s*[-–~]\s*[Ll]?(\d+))?")
HEADING_RE = re.compile(r"#{1,6}\s+(.*)|\\(?:sub)*(?:section|paragraph)\*?\{(.*)\}")

# 两个检查脚本自检用的资料快照
SNAPSHOT = ("# 第 1 章 玩具算子\n\n玩具算子把输入乘 2。输入取 3.5 时输出 7.0。\n"
            "它只对数值输入成立。\n\n## 1.2 另一节\n\n更多内容。\n")


@lru_cache
def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _headings(path: Path) -> list[str]:
    found = (HEADING_RE.match(line.strip()) for line in _lines(path))
    return [(m.group(1) or m.group(2)).strip() for m in found if m]


def check_basis_item(item: str, sources: Path) -> str | None:
    """核一条「文件名 位置」，有问题返回原因。"""
    item = item.strip()
    m = re.match(r"(\S+)\s+(.+)", item)
    if not m:
        return f"「{item}」要写成「文件名 位置」"
    fname, where = m.group(1), m.group(2).strip()
    f = sources / fname
    if not f.is_file():
        return f"{fname} 不在 research/sources/ 下"
    lm = LINES_RE.fullmatch(where)
    if lm:
        total = len(_lines(f))
        a = int(lm.group(1))
        b = int(lm.group(2) or lm.group(1))
        if a < 1 or b < a or b > total:
            return f"{fname} 只有 {total} 行，位置 {where} 不对"
        return None
    if not any(where.lower() in h.lower() for h in _headings(f)):
        return f"{fname} 里没有标题含「{where}」；章节写标题里的一段文字，或写行号范围如 L120-L180"
    return None


def check_basis_list(basis: str, sources: Path) -> list[str]:
    """核一整条依据（多处用 ； 隔开），「自行构造」不核。返回每处的问题。"""
    if basis.strip() == SELF_MADE:
        return []
    items = (i for i in re.split(r"[;；]", basis) if i.strip())
    return [why for i in items if (why := check_basis_item(i, sources))]


def report(results: list[tuple[str, bool]], label: str) -> int:
    """打印自检结果，返回退出码。"""
    for name, ok in results:
        print(("  OK  " if ok else "  FAIL") + " " + name)
    failed = sum(not ok for _, ok in results)
    print(f"{label}自检失败：{failed} 项" if failed else f"{label}自检通过：{len(results)} 项")
    return 1 if failed else 0
