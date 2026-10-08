#!/usr/bin/env python3
"""检查页面模板：必须声明 dojo:type，并引用该类型共享样式表。

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
# 每个模板：路径、dojo:type 的取值、引用的共享样式表
TEMPLATES = [
    (".dojo/templates/concept/index.html", "concept", "libs/concept/dojo-concept.css"),
    (".dojo/templates/dataflow/index.html", "dataflow", "libs/dataflow/dojo-flow.css"),
]
TYPE_RE = re.compile(r'<meta\s+name="dojo:type"\s+content="([^"]*)"')
SHEET_RE = re.compile(r'<link\s+[^>]*href="([^"]+)"')


def type_matches(text: str, kind: str, root: Path, rel: str) -> list[str]:
    m = TYPE_RE.search(text)
    now = m.group(1) if m else ""
    if now != kind:
        return [f"{rel}：类型必须是 {kind}，现在是：{now or '（缺失）'}"]
    return []


def stylesheet_is_linked(text: str, kind: str, root: Path, rel: str) -> list[str]:
    # 部署会给链接加 ?v=<sha>，比对时忽略查询串
    want = dict((k, s) for _, k, s in TEMPLATES)[kind]
    if not any(h.split("?", 1)[0].endswith(want) for h in SHEET_RE.findall(text)):
        return [f"{rel}：模板要引用 ../../{want}"]
    return []


def stylesheet_exists(text: str, kind: str, root: Path, rel: str) -> list[str]:
    want = dict((k, s) for _, k, s in TEMPLATES)[kind]
    if not (root / want).exists():
        return [f"{rel}：共享样式表不存在：{want}"]
    return []


RULES = [type_matches, stylesheet_is_linked, stylesheet_exists]


def check(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    for rel, kind, _sheet in TEMPLATES:
        template = root / rel
        if not template.exists():
            errors.append(f"{rel}：模板不存在")
            continue
        text = template.read_text(encoding="utf-8", errors="ignore")
        for rule in RULES:
            errors += rule(text, kind, root, rel)
    return errors


# ---------- 自检 ----------

GOOD = ('<html><head><meta name="dojo:type" content="concept">'
        '<link rel="stylesheet" href="../../libs/concept/dojo-concept.css"></head></html>')
CASES = [
    ("干净的模板没有报错", GOOD, True, None),
    ("类型不是 concept", GOOD.replace('"concept"', '"paper"'), True, "类型必须是 concept"),
    ("缺类型", GOOD.replace('<meta name="dojo:type" content="concept">', ""), True, "（缺失）"),
    ("没引用样式表", GOOD.replace('<link rel="stylesheet" href="../../libs/concept/dojo-concept.css">', ""),
     True, "模板要引用"),
    ("带版本号的引用不算缺", GOOD.replace("dojo-concept.css", "dojo-concept.css?v=abc"), True, None),
    ("样式表文件不存在", GOOD, False, "共享样式表不存在"),
]


def selftest() -> int:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "libs" / "concept").mkdir(parents=True)
        (root / "libs" / "dataflow").mkdir(parents=True)
        (root / ".dojo" / "templates" / "concept").mkdir(parents=True)
        (root / ".dojo" / "templates" / "dataflow").mkdir(parents=True)
        (root / "libs" / "concept" / "dojo-concept.css").write_text("", encoding="utf-8")
        (root / "libs" / "dataflow" / "dojo-flow.css").write_text("", encoding="utf-8")
        (root / ".dojo" / "templates" / "dataflow" / "index.html").write_text(
            '<html><head><meta name="dojo:type" content="dataflow">'
            '<link rel="stylesheet" href="../../libs/dataflow/dojo-flow.css"></head></html>',
            encoding="utf-8")
        t = root / ".dojo" / "templates" / "concept" / "index.html"
        for name, text, has_sheet, want in CASES:
            t.write_text(text, encoding="utf-8")
            sheet = root / "libs" / "concept" / "dojo-concept.css"
            if not has_sheet and sheet.exists():
                sheet.unlink()
            if has_sheet and not sheet.exists():
                sheet.write_text("", encoding="utf-8")
            errs = check(root)
            ok = (not errs) if want is None else any(want in e for e in errs)
            if want is None and errs:
                print("    意外报错：", errs)
            results.append((name, ok))
        # 数据流模板单独坏了要能报出
        t.write_text(GOOD, encoding="utf-8")
        flow = root / ".dojo" / "templates" / "dataflow" / "index.html"
        flow.write_text('<html><head><meta name="dojo:type" content="concept">'
                        '<link rel="stylesheet" href="../../libs/dataflow/dojo-flow.css"></head></html>',
                        encoding="utf-8")
        results.append(("数据流模板类型不对也会报出", any("dataflow" in e and "类型必须是" in e for e in check(root))))
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
