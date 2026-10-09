#!/usr/bin/env python3
"""检查共享样式表和页面内联 <style> 里有没有没人用的规则。

    python3 .dojo/scripts/site/check_css.py [--fix]
    python3 .dojo/scripts/site/check_css.py --selftest

「在用」的 class 来自这几处：模板、所有页面、生成器的源码、以及渲染时注入 class 的脚本。
生成器能产出的 class，样板页不一定用到（如表格的 table-scroll），所以要把生成器源码也算进来；
数据流页的 class 大半由 libs/dataflow/dojo-flow.js 在运行时加，所以那份脚本也要算。
--fix 直接删掉没人用的规则。
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "concept"))
from common import report  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)
SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.S | re.I)
CLASS_ATTR_RE = re.compile(r"""\bclass\s*=\s*["']([^"']*)["']""", re.I)
IDENT_RE = re.compile(r"[A-Za-z_][\w-]*")
CLASS_TOKEN_RE = re.compile(r"\.(-?[A-Za-z_][\w-]*)")
COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
# 库在运行时注入的 class，静态标记里查不到，必须视为已用
RUNTIME_PREFIXES = ("katex", "token", "language-", "prism", "line-numbers")


# ---------- 找出在用的 class ----------

def used_classes(path: Path) -> set[str]:
    """一份文档里可能被用上的 class：class 属性里的，以及脚本里出现的标识符。
    .js 文件整体当脚本看。"""
    text = path.read_text(encoding="utf-8")
    names: set[str] = set()
    for value in CLASS_ATTR_RE.findall(text):
        names.update(value.split())
    scripts = SCRIPT_RE.findall(text) if path.suffix in (".html", ".htm") else [text]
    for script in scripts:
        names.update(IDENT_RE.findall(script))
    return names


# ---------- 拆 CSS ----------

def iter_rules(css: str):
    """遍历 CSS 规则，产出 (选择器, 起始偏移, 结束偏移)，@media 等块里的规则各自产出。"""
    text = COMMENT_RE.sub(lambda m: " " * len(m.group(0)), css)
    stack: list[tuple[str, int]] = []   # (选择器, 规则起点)
    start = 0
    for cursor, char in enumerate(text):
        if char == "{":
            stack.append((text[start:cursor].strip(), start))
            start = cursor + 1
        elif char == "}":
            if stack:
                selector, begin = stack.pop()
                if selector and not selector.startswith("@"):
                    yield selector, begin, cursor + 1
            start = cursor + 1
        elif char == ";" and not stack:
            start = cursor + 1


def selector_is_dead(selector: str, used: set[str]) -> bool:
    """选择器里写了 class，而其中至少一个没人用，就是死的。
    带属性选择器、伪类函数、伪元素的判断不了，一律保留；不含 class 的也不判。"""
    if "[" in selector or "(" in selector or "::" in selector:
        return False
    tokens = CLASS_TOKEN_RE.findall(selector)
    if not tokens:
        return False
    return any(t not in used and not t.startswith(RUNTIME_PREFIXES) for t in tokens)


def split_selectors(selector: str) -> list[str]:
    return [part.strip() for part in selector.split(",") if part.strip()]


def dead_spans(css: str, used: set[str]) -> list[tuple[int, int, str]]:
    spans = []
    for selector, start, end in iter_rules(css):
        parts = split_selectors(selector)
        if parts and all(selector_is_dead(p, used) for p in parts):
            spans.append((start, end, selector.strip()))
    return spans


def drop(css: str, spans: list[tuple[int, int, str]]) -> str:
    for start, end, _ in reversed(spans):
        css = css[:start] + css[end:]
    return re.sub(r"\n{3,}", "\n\n", css).rstrip() + "\n"


def find_dead(path: Path, used: set[str], fix: bool) -> list[str]:
    """一个文件里没人用的规则。.css 整体查，.html 查里面的 <style>。fix 时直接写回。"""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".css":
        spans = dead_spans(text, used)
        if fix and spans:
            path.write_text(drop(text, spans), encoding="utf-8")
        return [s for _, _, s in spans]
    removed: list[str] = []
    for block in reversed(list(STYLE_RE.finditer(text))):
        spans = dead_spans(block.group(1), used)
        removed.extend(s for _, _, s in spans)
        if fix and spans:
            text = text[:block.start(1)] + drop(block.group(1), spans) + text[block.end(1):]
    if fix and removed:
        path.write_text(text, encoding="utf-8")
    return removed


def check(root: Path = REPO_ROOT, fix: bool = False) -> list[str]:
    """返回 ['文件：规则', ...]。"""
    sheet = root / "libs" / "concept" / "dojo-concept.css"
    template = root / ".dojo" / "templates" / "concept" / "index.html"
    generator = root / ".dojo" / "scripts" / "concept" / "build_page.py"
    flow_sheet = root / "libs" / "dataflow" / "dojo-flow.css"
    flow_template = root / ".dojo" / "templates" / "dataflow" / "index.html"
    flow_generator = root / ".dojo" / "scripts" / "dataflow" / "build_page.py"
    flow_engine = root / "libs" / "dataflow" / "dojo-flow.js"
    pages = sorted(root.glob("wiki/*/index.html"))
    for needed in (sheet, template, generator, flow_sheet, flow_template, flow_generator, flow_engine):
        if not needed.exists():
            return [f"错误：{needed} 不存在，无从检查"]
    if not pages:
        return ["错误：没有页面"]
    used: set[str] = set()
    for path in [template, generator, flow_template, flow_generator, flow_engine, *pages]:
        used |= used_classes(path)
    problems = []
    for path in [sheet, flow_sheet, *pages]:
        for selector in find_dead(path, used, fix):
            problems.append(f"{path.relative_to(root)}：{selector}")
    return problems


# ---------- 自检 ----------

def selftest() -> list[tuple[str, bool]]:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "libs" / "concept").mkdir(parents=True)
        (root / "libs" / "dataflow").mkdir(parents=True)
        (root / ".dojo" / "templates" / "concept").mkdir(parents=True)
        (root / ".dojo" / "templates" / "dataflow").mkdir(parents=True)
        (root / ".dojo" / "scripts" / "concept").mkdir(parents=True)
        (root / ".dojo" / "scripts" / "dataflow").mkdir(parents=True)
        (root / "wiki" / "toy").mkdir(parents=True)
        sheet = root / "libs" / "concept" / "dojo-concept.css"
        flow_sheet = root / "libs" / "dataflow" / "dojo-flow.css"
        (root / ".dojo" / "templates" / "concept" / "index.html").write_text('<p class="used">x</p>', encoding="utf-8")
        (root / ".dojo" / "templates" / "dataflow" / "index.html").write_text('<p class="flow-app">x</p>', encoding="utf-8")
        (root / ".dojo" / "scripts" / "concept" / "build_page.py").write_text(
            "out = '<div class=\"from-generator\">'", encoding="utf-8")
        (root / ".dojo" / "scripts" / "dataflow" / "build_page.py").write_text(
            "out = '<div class=\"from-flow-generator\">'", encoding="utf-8")
        (root / "libs" / "dataflow" / "dojo-flow.js").write_text(
            "el.className = 'flow-runtime'", encoding="utf-8")
        (root / "wiki" / "toy" / "index.html").write_text(
            '<style>.page-dead { color: red; }</style><p class="page-live">x</p>', encoding="utf-8")
        flow_sheet.write_text(".flow-app { a: b; }\n", encoding="utf-8")

        def run(css, fix=False):
            sheet.write_text(css, encoding="utf-8")
            return check(root, fix)

        base = ".used { a: b; }\n.from-generator { a: b; }\n.page-live { a: b; }\n"
        errs = run(base)
        results.append(("在用的 class 不报（模板、生成器、页面各一）", not any("dojo-concept.css" in e for e in errs)))
        results.append(("页面内联 <style> 里的死规则会报出", any("page-dead" in e for e in errs)))
        results.append(("没人用的规则会报出", any(".never-used" in e for e in run(base + ".never-used { a: b; }\n"))))
        results.append(("生成器不再产出时会报出", any(".gone" in e for e in run(base + ".gone { a: b; }\n"))))
        results.append(("katex 一类运行时 class 不报", not any("katex" in e for e in run(base + ".katex-html { a: b; }\n"))))
        results.append(("不含 class 的选择器不判", not any("body" in e for e in run(base + "body { a: b; }\n"))))
        media = "@media (max-width: 1px) { %s { a: b; } }\n"
        results.append(("@media 里没人用的规则会报出", any(".in-media" in e for e in run(base + media % ".in-media"))))
        results.append(("@media 里在用的规则不报", not any("dojo-concept.css" in e for e in run(base + media % ".used"))))
        results.append(("数据流模板、生成器、引擎的 class 不算死规则",
                        not any("dojo-flow.css" in e for e in run(base))))
        run(base + ".never-used { a: b; }\n", fix=True)
        results.append(("--fix 能删掉死规则", ".never-used" not in sheet.read_text(encoding="utf-8")
                        and ".used" in sheet.read_text(encoding="utf-8")))
    return report(results, "CSS 检查")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    fix = "--fix" in sys.argv
    problems = check(fix=fix)
    for p in problems:
        print(f"- {p}")
    if not problems:
        print("没有死 CSS")
    elif fix:
        print(f"已清掉 {len(problems)} 条死规则")
    else:
        print(f"发现 {len(problems)} 条死规则")
    raise SystemExit(1 if problems and not fix else 0)
