#!/usr/bin/env python3
"""把概览页与索引页的入口文案统一到同一套说法。

历史上有两个世代的文案并存：

  A 世代（早期）  eyebrow「概念快速阅读」、nav「深度教学 →」、footer「快速阅读」
  B 世代（模板）  eyebrow「概念概览」、   nav「完整说明 →」、footer「概览」

模板已固定为 B 世代，本脚本把早期的页面回填到同一套，并统一三类自由发挥：
箭头写法（LaTeX 与 Unicode 混用）、footer 的附加修饰、index 页概览入口文案。

    python3 .dojo/scripts/normalize_overview_copy.py --check
    python3 .dojo/scripts/normalize_overview_copy.py --fix
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 详细页第二页的称呼：概念叫「完整说明」，论文叫「完整解析」。
OVERVIEW_TERM = {"concept": "概览", "paper": "概览"}
DETAIL_TERM = {"concept": "完整说明", "paper": "完整解析"}

TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)
EYEBROW_RE = re.compile(r'(<span class="eyebrow">)(.*?)(</span>)', re.S)
NAV_RE = re.compile(r"(<nav>.*?</nav>)", re.S)
FOOTER_RE = re.compile(r"<footer>(.*?)</footer>", re.S)
LINK_RE = re.compile(r'<a class="overview-link"[^>]*title="[^"]*"[^>]*>.*?</a>', re.S)


def page_type(index_html: str) -> str:
    match = re.search(r'name="dojo:type"\s+content="([^"]+)"', index_html)
    return match.group(1) if match else ""


def clean_counters(text: str, keep: str) -> str:
    """把 LaTeX 箭头统一成 Unicode，并去掉计数器之外的修饰语。"""
    text = text.replace(r"$\leftarrow$", "←").replace(r"$\to$", "→")
    if keep == "detail":
        # nav 第二项只留「完整说明 →」这类固定形式
        text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_overview(text: str, ty: str) -> str:
    term = OVERVIEW_TERM.get(ty, "概览")
    detail = DETAIL_TERM.get(ty, "完整说明")
    if not term:
        return text

    # 正文里对详情页的称呼与旧标签一并统一
    text = text.replace("深度教学页", f"{detail}页").replace("深度教学", detail)
    text = text.replace("快速阅读", "概览")

    # title 统一成模板格式：`X · 概览 · Dojo`
    def title_sub(match: re.Match) -> str:
        title = match.group(1).strip()
        # 去掉末尾的「第二段 · Dojo」，只保留标题本体
        body = re.sub(r"\s*·\s*[^·]+?\s*·\s*Dojo\s*$", "", title)
        body = re.sub(r"\s*·\s*Dojo\s*$", "", body).strip()
        if body and body != title:
            return f"<title>{body} · 概览 · Dojo</title>"
        return match.group(0)

    text = TITLE_RE.sub(title_sub, text)

    # eyebrow：前缀统一，保留后面的主题标签
    def eyebrow_sub(match: re.Match) -> str:
        inner = match.group(2)
        tail = ""
        if "·" in inner:
            tail = inner.split("·", 1)[1].strip()
        label = "概念概览" if ty == "concept" else "论文概览"
        body = f"{label} · {tail}" if tail else label
        return f"{match.group(1)}{body}{match.group(3)}"

    text = EYEBROW_RE.sub(eyebrow_sub, text)

    # nav：返回项统一箭头写法，跳转项统一成「<detail> →」
    def nav_sub(match: re.Match) -> str:
        block = match.group(1)
        block = block.replace(r"$\leftarrow$", "←").replace(r"$\to$", "→")
        block = re.sub(
            r'(<a href="index\.html"[^>]*>).*?(</a>)',
            lambda m: m.group(1) + detail + " →" + m.group(2),
            block,
            flags=re.S,
        )
        return block

    text = NAV_RE.sub(nav_sub, text)

    # footer：统一成模板句式，去掉「与手算例子」这类自由修饰
    footer_text = f"Dojo · 概览 · 机制细节见{detail}页"
    text = FOOTER_RE.sub(lambda m: f"<footer>{footer_text}</footer>", text)
    return text


def fix_index_link(text: str) -> str:
    if '<a class="overview-link"' not in text:
        return text
    return LINK_RE.sub(
        '<a class="overview-link" href="overview.html" title="查看概览">概览</a>',
        text,
    )


def convert(path: Path, fix: bool) -> list[str]:
    changes: list[str] = []
    if path.name == "overview.html":
        index = path.parent / "index.html"
        ty = page_type(index.read_text(encoding="utf-8")) if index.exists() else ""
        original = path.read_text(encoding="utf-8")
        updated = fix_overview(original, ty)
        if updated != original:
            changes.append(f"{path}: 概览页文案")
            if fix:
                path.write_text(updated, encoding="utf-8")
    elif path.name == "index.html":
        original = path.read_text(encoding="utf-8")
        updated = fix_index_link(original)
        if updated != original:
            changes.append(f"{path}: 概览入口链接")
            if fix:
                path.write_text(updated, encoding="utf-8")
    return changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="写回；默认只报告")
    parser.add_argument("--check", action="store_true", help="只报告（默认行为）")
    args = parser.parse_args()

    targets = sorted(Path("wiki").glob("*/overview.html")) + sorted(
        Path("wiki").glob("*/index.html")
    )
    all_changes: list[str] = []
    for path in targets:
        all_changes.extend(convert(path, args.fix))

    if not all_changes:
        print(f"no copy drift in {len(targets)} files")
        return 0
    verb = "已统一" if args.fix else "待统一"
    print(f"{verb} {len(all_changes)} 处：")
    for line in all_changes:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
