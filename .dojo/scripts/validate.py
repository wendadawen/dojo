#!/usr/bin/env python3
"""Validate a single concept or paper page (index.html).

Deterministic checks only: shell integrity, template leftovers, duplicate
ids, same-page anchors and broken local references. Semantic quality is out
of scope (handled by the independent review).

Usage:
    python3 .dojo/scripts/validate.py wiki/<name>/index.html
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path


LOCAL_ASSET_RE = re.compile(r'''(?:href|src)=["']([^"']+)["']''')
ID_RE = re.compile(r"""\bid=["']([^"']+)["']""")
SAME_PAGE_ANCHOR_RE = re.compile(r"""href=["']#([^"']*)["']""")
PLACEHOLDER_RE = re.compile(r"【[^】]*】")

IGNORE_PREFIXES = (
    "http://",
    "https://",
    "data:",
    "#",
    "mailto:",
)

MARKERS = ("@content", "@component", "TODO", "TBD")
REQUIRED_WIKI_META = (
    "description",
    "dojo:summary",
    "dojo:type",
    "dojo:topics",
    "dojo:tag",
)
INLINE_MATH_RE = re.compile(
    r"\$\$[\s\S]+?\$\$|"
    r"(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$(?!\$)|"
    r"\\\([\s\S]+?\\\)|"
    r"\\\[[\s\S]+?\\\]"
)


class PageInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.scripts: list[str] = []
        self.stylesheets: list[str] = []
        self.visible_parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple]) -> None:
        values = dict(attrs)
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        if tag == "meta":
            name = values.get("name", "").strip().lower()
            if name:
                self.meta[name] = values.get("content", "").strip()
        elif tag == "script" and values.get("src"):
            self.scripts.append(values["src"])
        elif tag == "link" and values.get("rel") == "stylesheet":
            self.stylesheets.append(values.get("href", ""))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.visible_parts.append(data)


def validate_page(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    inspector = PageInspector()
    inspector.feed(text)

    if not text.startswith("<!DOCTYPE html>"):
        errors.append("missing <!DOCTYPE html>")
    if not text.rstrip().endswith("</html>"):
        errors.append("missing </html>")

    searchable_text = re.sub(r"data:[^\"']+", "", text)
    for match in PLACEHOLDER_RE.findall(searchable_text):
        errors.append(f"template placeholder remains: {match}")
    for needle in MARKERS:
        if needle in searchable_text:
            errors.append(f"template marker remains: {needle}")

    ids = ID_RE.findall(text)
    seen: set[str] = set()
    for id_value in ids:
        if id_value in seen:
            errors.append(f"duplicate id: {id_value}")
        seen.add(id_value)

    id_set = set(ids)
    for anchor in SAME_PAGE_ANCHOR_RE.findall(text):
        if anchor and anchor not in id_set:
            errors.append(f"anchor points to missing id: #{anchor}")

    for ref in LOCAL_ASSET_RE.findall(text):
        if ref.startswith(IGNORE_PREFIXES):
            continue
        clean_ref = ref.split("#", 1)[0].split("?", 1)[0]
        if not clean_ref:
            continue
        target = (path.parent / clean_ref).resolve()
        if clean_ref.endswith("/"):
            target = target / "index.html"
        if not target.exists():
            errors.append(f"broken local reference {ref}")

    is_wiki_page = "wiki" in path.parts
    if is_wiki_page and path.name == "index.html":
        for name in REQUIRED_WIKI_META:
            if not inspector.meta.get(name):
                errors.append(f"missing metadata: {name}")

        description = inspector.meta.get("description", "")
        summary = inspector.meta.get("dojo:summary", "")
        if "$" in description:
            errors.append("description must be plain text; put formulas in dojo:summary")
        if "$$" in summary:
            errors.append("dojo:summary supports inline $...$ formulas only")
        if summary.count("$") % 2:
            errors.append("dojo:summary has unmatched $ delimiter")

    visible_text = " ".join(inspector.visible_parts)
    summary = inspector.meta.get("dojo:summary", "")
    if is_wiki_page and INLINE_MATH_RE.search(f"{summary} {visible_text}"):
        has_katex_js = any("katex" in ref for ref in inspector.scripts)
        has_katex_css = any("katex" in ref for ref in inspector.stylesheets)
        has_auto_render = (
            any("auto-render" in ref for ref in inspector.scripts)
            and "renderMathInElement" in text
        )
        if not has_katex_js or not has_katex_css:
            errors.append("math content requires local KaTeX JS and CSS")
        if not has_auto_render:
            errors.append("math content requires auto-render initialization")

    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    page = Path(sys.argv[1])
    if page.is_dir():
        page = page / "index.html"
    if not page.exists():
        print(f"error: page not found: {page}")
        return 2

    errors = validate_page(page)
    if errors:
        print(f"validation failed: {page}")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"validation ok: {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
