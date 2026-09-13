#!/usr/bin/env python3
"""找出页面内嵌 CSS 里没有任何元素会用到的规则。

判定方式：一条选择器如果含有 class 记号，而其中某个 class 名在"使用面"里
完全不出现，那它永远匹配不到元素，是死代码。

使用面由三部分组成，缺一不可：
  1. 目标文件自身的 class 属性（静态标记）
  2. 目标文件内联脚本里的标识符（class 可由脚本动态添加）
  3. --usage-from 指定的文件（模板的 body 是占位符，使用面必须取自它生成的页面）
库在运行时注入的 class（KaTeX 的 katex-*、Prism 的 token 等）由内置前缀表兜底。

脚本刻意保守：只清理能确证无用的规则。元素选择器、:not()、属性选择器、
含括号的伪类一律跳过，宁可漏删不误删。

    python3 .dojo/scripts/check_unused_css.py <目标文件...> \
        [--usage-from <提供使用面的页面...>] [--fix]

按模块处理（推荐，模块之间保持独立）：

    python3 .dojo/scripts/check_unused_css.py --module note \
        --template .dojo/templates/note/index.html [--fix]

--module 会扫描 wiki/ 下所有 dojo:type 等于该值的页面，用它们的并集作为使用面，
目标自动包含模板（若给了 --template）与这些页面。

退出码 1 表示存在死代码（可用于 CI），0 表示干净，2 表示用法错误。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)
SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.S | re.I)
CLASS_ATTR_RE = re.compile(r"""\bclass\s*=\s*["']([^"']*)["']""", re.I)
IDENT_RE = re.compile(r"[A-Za-z_][\w-]*")
CLASS_TOKEN_RE = re.compile(r"\.(-?[A-Za-z_][\w-]*)")
COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)

# 库在运行时注入的 class，静态标记里查不到，必须视为已用
RUNTIME_PREFIXES = ("katex", "token", "language-", "prism", "line-numbers")


def harvest(document: str) -> set[str]:
    """从一份文档里收集可能被应用到的 class 名。"""
    names: set[str] = set()
    for value in CLASS_ATTR_RE.findall(document):
        names.update(value.split())
    for script in SCRIPT_RE.findall(document):
        names.update(IDENT_RE.findall(script))
    return names


def build_usage(targets: list[Path], extra: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in list(targets) + list(extra):
        names |= harvest(path.read_text(encoding="utf-8"))
    return names


def iter_rules(css: str):
    """遍历 CSS 规则，产出 (选择器, 起始偏移, 结束偏移)。"""
    text = COMMENT_RE.sub(lambda m: " " * len(m.group(0)), css)
    stack: list[tuple[str, int]] = []
    start = 0
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            stack.append((text[start:cursor].strip(), cursor))
        elif char == "}":
            if stack:
                selector, _ = stack.pop()
                if selector and not selector.startswith("@"):
                    yield selector, start, cursor + 1
            start = cursor + 1
        elif char == ";" and not stack:
            start = cursor + 1
        cursor += 1


def selector_is_dead(selector: str, used: set[str]) -> bool:
    """判断不了的一律返回 False（保留）。"""
    if "[" in selector or ":not(" in selector or "(" in selector or "::" in selector:
        return False
    tokens = CLASS_TOKEN_RE.findall(selector)
    if not tokens:
        return False
    return any(
        token not in used
        and not token.startswith(RUNTIME_PREFIXES)
        for token in tokens
    )


def split_selectors(selector: str) -> list[str]:
    return [part.strip() for part in selector.split(",") if part.strip()]


def process(path: Path, usage: set[str], fix: bool) -> list[str]:
    document = path.read_text(encoding="utf-8")
    removed: list[str] = []
    for block in reversed(list(STYLE_RE.finditer(document))):
        css = block.group(1)
        spans: list[tuple[int, int, str]] = []
        for selector, local_start, local_end in iter_rules(css):
            parts = split_selectors(selector)
            if parts and all(selector_is_dead(part, usage) for part in parts):
                spans.append((local_start, local_end, selector.strip()))
        if not spans:
            continue
        removed.extend(s for _, _, s in spans)
        if fix:
            new_css = css
            for local_start, local_end, _ in reversed(spans):
                new_css = new_css[:local_start] + new_css[local_end:]
            new_css = re.sub(r"\n{3,}", "\n\n", new_css).rstrip() + "\n"
            document = document[: block.start(1)] + new_css + document[block.end(1) :]
    if fix and removed:
        path.write_text(document, encoding="utf-8")
    return removed


def pages_of_type(root: Path, kind: str) -> list[Path]:
    """wiki/ 下所有 dojo:type 等于 kind 的页面。"""
    pattern = re.compile(r'name="dojo:type"\s+content="' + re.escape(kind) + r'"')
    return [
        path
        for path in sorted(root.glob("wiki/*/index.html"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]


def main() -> int:
    args = sys.argv[1:]
    fix = "--fix" in args
    rest = [a for a in args if a != "--fix"]
    module = None
    template = None
    if "--module" in rest:
        index = rest.index("--module")
        module = rest[index + 1]
        del rest[index : index + 2]
    if "--template" in rest:
        index = rest.index("--template")
        template = Path(rest[index + 1])
        del rest[index : index + 2]

    if module:
        pages = pages_of_type(Path("."), module)
        if not pages:
            print(f"error: no pages with dojo:type={module}")
            return 2
        targets = ([template] if template else []) + pages
        extra = pages
    elif "--usage-from" in rest:
        index = rest.index("--usage-from")
        targets = [Path(p) for p in rest[:index]]
        extra = [Path(p) for p in rest[index + 1 :]]
    else:
        targets, extra = [Path(p) for p in rest], []

    if not targets:
        print(__doc__)
        return 2
    missing = [p for p in targets + extra if not p.exists()]
    if missing:
        print("error: not found: " + ", ".join(str(p) for p in missing))
        return 2

    usage = build_usage(targets, extra)
    total = 0
    for path in targets:
        removed = process(path, usage, fix)
        total += len(removed)
        if removed:
            print(f"{path}: {len(removed)} rules")
            for selector in removed:
                print(f"  - {selector}")
    if total:
        print(f"\n{'cleaned' if fix else 'found'} {total} dead rules in {len(targets)} files")
        return 0 if fix else 1
    print(f"no dead css in {len(targets)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
