#!/usr/bin/env python3
"""统一审查文件的命名，并给每份记录加上机器可读的头字段。

现状是两种命名并存：早期写成 review.md，后来改成 review-1/2/3.md。
「必须三轮独立审查」这条规则因此无法机械判定——审计原话是「不可判定的
规则等于不存在」。

本脚本做两件事：
  1. review.md 改名为 review-1.md（它本来就是第 1 轮）。dsa 与 rope 例外：
     这两页同时存在 review.md 和 review-1.md，且前者更早，说明它是采用
     review-N 命名之前的记录，改为 review-0.md。
  2. 每份记录开头插入头字段，其中 reviewed_content_sha256 取自**该审查
     文件被添加的那个提交**时的页面内容——即审查当时看到的版本。内容是
     指页面的可见正文，不含样式与脚本，因此排版改动不会让它失效。

    python3 .dojo/scripts/normalize_reviews.py            # 报告
    python3 .dojo/scripts/apply_review_headers.py         # 见 --fix
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HEADER_RE = re.compile(r"<!--\s*review-meta\b.*?-->\s*", re.S)
TAG_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
ANY_TAG_RE = re.compile(r"<[^>]+>")


def content_fingerprint(html: str) -> str:
    """页面可见正文的指纹。去掉样式、脚本与全部标签，只留文字。"""
    text = TAG_RE.sub(" ", html)
    text = ANY_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def adding_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", str(path)],
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.split("\n") if line.strip()]
    return lines[-1] if lines else None


def page_at_commit(commit: str, page: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{page}"],
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def build_header(round_number: int, page: str, fingerprint: str | None) -> str:
    lines = [
        "<!-- review-meta",
        f"round: {round_number}",
        f"page: {page}",
        f"reviewed_content_sha256: {fingerprint or '未记录'}",
        "-->",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    reviews = sorted(Path("wiki").glob("*/research/review*.md"))
    renamed = 0
    stamped = 0
    no_fingerprint = 0

    for path in reviews:
        slug = path.parts[1]
        page = f"wiki/{slug}/index.html"
        text = path.read_text(encoding="utf-8")
        name = path.name
        round_number = int(re.search(r"(\d+)", name).group(1)) if re.search(r"(\d+)", name) else 1

        # 1. 改名
        new_path = path
        if name == "review.md":
            sibling = path.parent / "review-1.md"
            target = "review-0.md" if sibling.exists() else "review-1.md"
            new_path = path.parent / target
            round_number = 0 if target == "review-0.md" else 1
            renamed += 1
            if args.fix:
                subprocess.run(["git", "mv", str(path), str(new_path)], check=True)

        # 2. 头字段
        body = HEADER_RE.sub("", text)
        commit = adding_commit(path)
        fingerprint = None
        if commit:
            html = page_at_commit(commit, page)
            if html:
                fingerprint = content_fingerprint(html)
        if fingerprint is None:
            no_fingerprint += 1
        header = build_header(round_number, page, fingerprint)
        if args.fix:
            new_path.write_text(header + body, encoding="utf-8")
        stamped += 1

    verb = "已处理" if args.fix else "待处理"
    print(f"{verb} {stamped} 份审查记录：改名 {renamed} 份，其中 {no_fingerprint} 份取不到审查时的页面内容（标为未记录）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
