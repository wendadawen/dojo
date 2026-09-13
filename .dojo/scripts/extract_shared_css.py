#!/usr/bin/env python3
"""把页面内嵌的整块 CSS 抽成共享样式文件，页面改为外链引用。

页面是自包含的——每个 index.html 里都内嵌了一整套模板 CSS 的副本，模板改动
无法回填，同一仓库里因此并存三十多种排版。本脚本按模块把这份副本抽出去：

  * 共享文件取该模块模板的 <style> 内容，写入 libs/dojo-<module>.css
  * 页面原来的 <style> 位置改成一个 <link>，指向共享文件
  * 页面自己独有的规则（共享文件里没有的选择器，或同名但声明不同的规则）
    原样保留在 <link> 之后的第二个 <style> 块里，保证覆盖关系不变

抽取后页面不再自包含，但页面本来就通过 ../../libs/ 外链 KaTeX 与 Prism，
所以并没有引入新的外部依赖。

    python3 .dojo/scripts/extract_shared_css.py --module concept \
        --source .dojo/templates/concept/index.html \
        --css libs/dojo-concept.css \
        --targets <页面...>

加 --check 只报告不写入。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)
COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonical_map(css: str) -> dict[str, set[str]]:
    """选择器 -> 该选择器在共享 CSS 里的全部声明体。"""
    cleaned = COMMENT_RE.sub("", css)
    out: dict[str, set[str]] = {}
    for match in RULE_RE.finditer(cleaned):
        selector = normalize(match.group(1))
        if selector.startswith("@"):
            continue
        body = normalize(match.group(2))
        for part in selector.split(","):
            out.setdefault(normalize(part), set()).add(body)
    return out


def residual_css(shell: str, canonical: dict[str, set[str]]) -> tuple[str, list[str]]:
    """返回页面需要保留的规则文本，以及被保留的选择器清单。"""
    spans = []
    cleaned = COMMENT_RE.sub(lambda m: " " * len(m.group(0)), shell)
    for match in RULE_RE.finditer(cleaned):
        selector_text = normalize(match.group(1))
        if selector_text.startswith("@"):
            continue
        body = normalize(match.group(2))
        needed = False
        for part in selector_text.split(","):
            key = normalize(part)
            if key not in canonical or body not in canonical[key]:
                needed = True
                break
        if needed:
            spans.append((match.start(), match.end()))
    if not spans:
        return "", []
    # 按原顺序取原文，保留原始排版
    pieces = [shell[start:end] for start, end in spans]
    kept = [normalize(piece) for piece in pieces]
    return "\n".join("    " + piece.strip() for piece in pieces), kept


def convert(path: Path, canonical: dict[str, set[str]], href: str, fix: bool) -> dict:
    document = path.read_text(encoding="utf-8")
    blocks = list(STYLE_RE.finditer(document))
    if not blocks:
        return {"path": str(path), "changed": False, "note": "no style block"}
    first = blocks[0]
    shell = first.group(1)
    residual, kept = residual_css(shell, canonical)
    link = f'<link rel="stylesheet" href="{href}">'
    replacement = link
    if residual:
        replacement += "\n    <style>\n" + residual + "\n    </style>"
    if fix:
        document = document[: first.start()] + replacement + document[first.end() :]
        path.write_text(document, encoding="utf-8")
    return {
        "path": str(path),
        "changed": True,
        "kept": len(kept),
        "kept_selectors": kept,
        "shell_chars": len(shell),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--source", required=True, help="取哪个文件的 <style> 作为共享 CSS")
    parser.add_argument("--css", required=True, help="共享 CSS 写到哪")
    parser.add_argument("--href", default=None, help="页面里的引用路径，默认 ../../libs/<css 文件名>")
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--check", action="store_true", help="只报告不写入")
    args = parser.parse_args()

    source = Path(args.source)
    css_out = Path(args.css)
    href = args.href or "../../" + str(css_out)

    source_css = STYLE_RE.findall(source.read_text(encoding="utf-8"))[0]
    canonical = canonical_map(source_css)

    fix = not args.check
    print(f"共享 CSS {len(source_css)} 字符 / {len(canonical)} 个选择器 -> {css_out}")
    if fix:
        css_out.write_text(
            f"/* Dojo {args.module} 模块共享样式。\n"
            f"   由页面内嵌副本抽出；改这里即改全部 {args.module} 页面。\n"
            f"   来源：{source} */\n\n" + source_css.strip() + "\n",
            encoding="utf-8",
        )

    total_kept = 0
    for target in args.targets:
        info = convert(Path(target), canonical, href, fix)
        total_kept += info.get("kept", 0)
        if info.get("kept"):
            print(f"  {info['path']}: 保留 {info['kept']} 条页面私有规则")
    print(f"\n{'转换' if fix else '将转换'} {len(args.targets)} 个文件，共保留 {total_kept} 条私有规则")
    return 0


if __name__ == "__main__":
    sys.exit(main())
