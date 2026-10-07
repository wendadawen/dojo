"""概念页检查的共用部分：「依据」怎么核，以及 outline、content 两个阶段都要用的常量和小函数。

「依据」的写法：
    依据：<文件名> <位置>[；<文件名> <位置>]
    文件名是 research/sources/ 下的文件。位置有两种写法：
      - 章节标题里的一段文字，如「2.3 Context-aware」，文件里要有一个标题行含这段文字；
      - 行号范围，如「L120-L180」，不能超出文件行数。
    「依据：自行构造」不核，只用于例子和推导，由复算审查者验证。
"""

from __future__ import annotations

import re
from pathlib import Path

LONG_LIMIT = 5000   # 长页折叠块外的字数上限
SHORT_LIMIT = 1000  # 短页的字数上限
SHORT_SECTIONS = 3
LAST_CHAPTER = "适用条件与边界"
SELF_MADE = "自行构造"
LINES_RE = re.compile(r"[Ll](\d+)(?:\s*[-–~]\s*[Ll]?(\d+))?")


def check_basis_item(item: str, sources: Path, cache: dict) -> str | None:
    """核一条「文件名 位置」，有问题返回原因，没问题返回 None。"""
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
        total = len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
        a = int(lm.group(1))
        b = int(lm.group(2) or lm.group(1))
        if a < 1 or b < a or b > total:
            return f"{fname} 只有 {total} 行，位置 {where} 不对"
        return None
    if f not in cache:
        cache[f] = _headings(f)
    if not any(where.lower() in h.lower() for h in cache[f]):
        return f"{fname} 里没有标题含「{where}」；章节写标题里的一段文字，或写行号范围如 L120-L180"
    return None


def _headings(path: Path) -> list[str]:
    """资料里的标题行：Markdown 的 # 行，或 LaTeX 的 \\section 一类。"""
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        m = re.match(r"#{1,6}\s+(.*)", s) or re.match(r"\\(?:sub)*(?:section|paragraph)\*?\{(.*)\}", s)
        if m:
            out.append(m.group(1).strip())
    return out


def check_basis_list(basis: str, sources: Path, cache: dict) -> list[str]:
    """核一整条依据（可含多处，用 ； 隔开）。「自行构造」不核。返回每处的问题。"""
    if basis.strip() == SELF_MADE:
        return []
    problems = []
    for item in re.split(r"[;；]", basis):
        if item.strip():
            why = check_basis_item(item, sources, cache)
            if why:
                problems.append(why)
    return problems
