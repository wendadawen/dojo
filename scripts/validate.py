#!/usr/bin/env python3
"""Validate a single concept or paper page (index.html).

Deterministic checks only: shell integrity, template leftovers, duplicate
ids, same-page anchors and broken local references. Semantic quality is out
of scope (handled by the independent review).

Usage:
    python3 scripts/validate.py content/concepts/<name>/index.html
    python3 scripts/validate.py content/papers/<venue>/<name>/index.html
"""

from __future__ import annotations

import re
import sys
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


def validate_page(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")

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
