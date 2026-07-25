#!/usr/bin/env python3
"""Audit concept pages against workflows/concept.md quality rules.

Checks the most commonly missed rules. Run before publishing:
    python scripts/audit_concept.py

Exit code 0 = all checks pass, 1 = issues found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
CONCEPTS_DIR = ROOT / "content" / "concepts"


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

class DetailsExtractor(HTMLParser):
    """Extract <details> blocks with their summary and inner text."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[dict] = []
        self._depth = 0
        self._current: dict | None = None
        self._capture_summary = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "details":
            self._depth += 1
            if self._depth == 1:
                attrs_dict = dict(attrs)
                self._current = {
                    "cls": attrs_dict.get("class", ""),
                    "summary": "",
                    "text": "",
                }
        elif tag == "summary" and self._current is not None and not self._current["summary"]:
            self._capture_summary = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "summary":
            self._capture_summary = False
        elif tag == "details":
            self._depth -= 1
            if self._depth == 0 and self._current is not None:
                self.blocks.append(self._current)
                self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            if self._capture_summary:
                self._current["summary"] += data
            else:
                self._current["text"] += data


def extract_details_blocks(html: str) -> list[dict]:
    parser = DetailsExtractor()
    parser.feed(html)
    return parser.blocks


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_no_prerequisite(text: str) -> list[str]:
    """No '前置' field in meta blockquote."""
    if re.search(r'<blockquote class="meta">[\s\S]*?<b>前置</b>', text):
        return ["meta 里还有前置列表，应删除（所有术语本页解释）"]
    return []


def check_checkpoint_structure(text: str) -> list[str]:
    """Checkpoint intro sentence should not be a sibling <li>."""
    for pattern in [r"<li>\s*读到这里", r"<li>\s*读到这", r"<li>\s*此时"]:
        if re.search(pattern, text):
            return ["检查点引导句不该是 <li>，应单独 <p>，检查项单独 <ul>"]
    return []


def check_derivation_blocks(html: str) -> list[str]:
    """Derivation <details> must have required structural titles."""
    required = ["问题陈述", "核心困难", "一句话总结", "符号总览"]
    errors: list[str] = []

    for block in extract_details_blocks(html):
        summary = block["summary"].strip()
        body = block["text"]

        # Skip code-details blocks — they are code, not derivations
        if "code-details" in block["cls"]:
            continue

        is_derivation = any(
            kw in body for kw in ["第 1 步", "第1步", "推导", "怎么来的"]
        ) or "推导" in summary

        if not is_derivation:
            continue

        for title in required:
            if title not in body:
                errors.append(f'推导折叠块"{summary[:30]}"缺少: {title}')
    return errors


def check_runnable_code_blocks(html: str) -> list[str]:
    """Runnable code <details class="code-details"> must have follow-up titles."""
    required = ["代码层次", "读者容易卡在哪", "和真实工程的差距"]
    errors: list[str] = []

    for block in extract_details_blocks(html):
        if "code-details" not in block["cls"]:
            continue
        if "language-python" not in block["text"]:
            continue

        summary = block["summary"].strip()
        if "伪代码" in summary:
            continue

        for title in required:
            if title not in block["text"]:
                errors.append(f'可运行代码折叠块"{summary[:30]}"缺少: {title}')
    return errors


def check_constructed_numbers(html: str) -> list[str]:
    """Numeric example <details> must label constructed numbers."""
    errors: list[str] = []

    for block in extract_details_blocks(html):
        summary = block["summary"].strip()
        body = block["text"]

        is_numeric = any(
            kw in summary for kw in ["数值", "数字", "例子", "演示", "计算"]
        )
        if not is_numeric:
            continue

        if not re.search(r"\d+\.\d+", body):
            continue

        if "本文构造" not in body and "教学用" not in body:
            errors.append(
                f'数值例子折叠块"{summary[:30]}"缺少"本文构造"标注'
            )
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKS = [
    check_no_prerequisite,
    check_checkpoint_structure,
    check_derivation_blocks,
    check_runnable_code_blocks,
    check_constructed_numbers,
]


def main() -> int:
    if not CONCEPTS_DIR.exists():
        print("no concepts directory")
        return 0

    pages = sorted(CONCEPTS_DIR.glob("*/index.html"))
    if not pages:
        print("no concept pages found")
        return 0

    all_errors: list[str] = []
    for page in pages:
        html = page.read_text(encoding="utf-8", errors="ignore")
        rel = page.relative_to(ROOT)
        for check_fn in CHECKS:
            for error in check_fn(html):
                all_errors.append(f"{rel}: {error}")

    if all_errors:
        print(f"concept audit: {len(all_errors)} issue(s) found\n")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print(f"concept audit ok ({len(pages)} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
