#!/usr/bin/env python3
"""检查概念页模板：必须声明 dojo:type=concept，并引用共享样式表。

    python3 .dojo/scripts/site/check_template.py
    python3 .dojo/scripts/site/check_template.py --selftest

模板含占位符，不能走 check_page.py。
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = REPO_ROOT / ".dojo" / "templates" / "concept" / "index.html"
STYLESHEET = "libs/concept/dojo-concept.css"
TYPE_RE = re.compile(r'<meta\s+name="dojo:type"\s+content="([^"]*)"')
SHEET_RE = re.compile(r'<link\s+[^>]*href="([^"]+)"')


# ---------- 规则：每条一个函数，返回问题列表 ----------

def type_is_concept(text: str, root: Path) -> list[str]:
    m = TYPE_RE.search(text)
    kind = m.group(1) if m else ""
    if kind != "concept":
        return [f"模板的类型必须是 concept，现在是：{kind or '（缺失）'}"]
    return []


def stylesheet_is_linked(text: str, root: Path) -> list[str]:
    # 部署会给链接加 ?v=<sha>，比对时忽略查询串
    if not any(h.split("?", 1)[0].endswith(STYLESHEET) for h in SHEET_RE.findall(text)):
        return [f"模板要引用 ../../{STYLESHEET}"]
    return []


def stylesheet_exists(text: str, root: Path) -> list[str]:
    if not (root / STYLESHEET).exists():
        return [f"共享样式表不存在：{STYLESHEET}"]
    return []


RULES = [type_is_concept, stylesheet_is_linked, stylesheet_exists]


def check(template: Path = TEMPLATE, root: Path = REPO_ROOT) -> list[str]:
    text = template.read_text(encoding="utf-8", errors="ignore")
    errors: list[str] = []
    for rule in RULES:
        errors += rule(text, root)
    return errors


# ---------- 自检 ----------

GOOD = ('<html><head><meta name="dojo:type" content="concept">'
        f'<link rel="stylesheet" href="../../{STYLESHEET}"></head></html>')
CASES = [
    ("干净的模板没有报错", GOOD, True, None),
    ("类型不是 concept", GOOD.replace('"concept"', '"paper"'), True, "类型必须是 concept"),
    ("缺类型", GOOD.replace('<meta name="dojo:type" content="concept">', ""), True, "（缺失）"),
    ("没引用样式表", GOOD.replace(f'<link rel="stylesheet" href="../../{STYLESHEET}">', ""), True, "模板要引用"),
    ("带版本号的引用不算缺", GOOD.replace(STYLESHEET, STYLESHEET + "?v=abc"), True, None),
    ("样式表文件不存在", GOOD, False, "共享样式表不存在"),
]


def selftest() -> int:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "libs" / "concept").mkdir(parents=True)
        (root / STYLESHEET).write_text("", encoding="utf-8")
        t = root / "index.html"
        for name, text, has_sheet, want in CASES:
            t.write_text(text, encoding="utf-8")
            sheet = root / STYLESHEET
            if not has_sheet and sheet.exists():
                sheet.unlink()
            if has_sheet and not sheet.exists():
                sheet.write_text("", encoding="utf-8")
            errs = check(t, root)
            ok = (not errs) if want is None else any(want in e for e in errs)
            if want is None and errs:
                print("    意外报错：", errs)
            results.append((name, ok))
    for name, ok in results:
        print(("  OK  " if ok else "  FAIL") + " " + name)
    failed = sum(1 for _, ok in results if not ok)
    print(f"模板检查自检失败：{failed} 项" if failed else f"模板检查自检通过：{len(results)} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    problems = check()
    for p in problems:
        print(f"- {p}")
    print(f"模板检查失败：{len(problems)} 个问题" if problems else "模板检查通过")
    raise SystemExit(1 if problems else 0)
