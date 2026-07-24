#!/usr/bin/env python3
"""Validate paper-reading manifest paths and generated pages."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "content.json"
INDEX = ROOT / "index.html"
LOCAL_ASSET_RE = re.compile(r'''(?:href|src)=["']([^"']+)["']''')


IGNORE_PREFIXES = (
    "http://",
    "https://",
    "data:",
    "#",
    "mailto:",
)


def load_manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


def target_for_path(path: str) -> Path:
    p = ROOT / path
    if path.endswith("/"):
        return p / "index.html"
    return p


def validate_manifest(manifest: dict) -> list[str]:
    errors: list[str] = []
    section_ids = {section["id"] for section in manifest.get("sections", [])}
    seen_ids: set[str] = set()

    for item in manifest.get("items", []):
        item_id = item.get("id", "<missing id>")
        if item_id in seen_ids:
            errors.append(f"duplicate item id: {item_id}")
        seen_ids.add(item_id)

        if item.get("section") not in section_ids:
            errors.append(f"{item_id}: unknown section {item.get('section')}")

        target = target_for_path(item.get("path", ""))
        if not target.exists():
            errors.append(f"{item_id}: missing path target {target.relative_to(ROOT)}")

        for key in ("tag", "title", "desc", "path"):
            if not item.get(key):
                errors.append(f"{item_id}: missing {key}")

    return errors


def validate_html_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    is_template = "templates" in path.parts

    if not text.startswith("<!DOCTYPE html>"):
        errors.append(f"{path.relative_to(ROOT)}: missing <!DOCTYPE html>")
    if not text.rstrip().endswith("</html>"):
        errors.append(f"{path.relative_to(ROOT)}: missing </html>")

    searchable_text = re.sub(r"data:[^\"']+", "", text)
    for needle in ("<概念名>", "<术语1>", "TODO", "TBD"):
        if needle in searchable_text and not is_template:
            errors.append(f"{path.relative_to(ROOT)}: template marker remains: {needle}")

    if is_template:
        return errors

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
            errors.append(
                f"{path.relative_to(ROOT)}: broken local reference {ref}"
            )
    return errors


def validate_index(manifest: dict) -> list[str]:
    errors: list[str] = []
    if not INDEX.exists():
        return ["missing index.html"]
    text = INDEX.read_text(encoding="utf-8", errors="ignore")
    for item in manifest.get("items", []):
        if item["path"] not in text:
            errors.append(f"index.html missing manifest path: {item['path']}")
    return errors


def main() -> int:
    manifest = load_manifest()
    errors = []
    errors.extend(validate_manifest(manifest))
    errors.extend(validate_index(manifest))

    for html in list((ROOT / "content").glob("**/*.html")) + list((ROOT / "templates").glob("**/*.html")):
        errors.extend(validate_html_file(html))

    if errors:
        print("validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("validation ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
