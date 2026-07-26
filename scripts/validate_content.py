#!/usr/bin/env python3
"""Validate paper-reading manifest paths and generated pages.

Only deterministic checks: manifest targets, required files, template
leftovers, duplicate ids, same-page anchors and broken local references.
Semantic quality is out of scope (handled by the stage-4 independent review).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "content.json"
INDEX = ROOT / "index.html"
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

REQUIRED_FILES = [
    "templates/concept/index.html",
    "templates/concept/components.html",
    "templates/paper/index.html",
    "workflows/concept.md",
    "workflows/concept/step-1-research.md",
    "workflows/concept/step-2-outline.md",
    "workflows/concept/step-3-draft.md",
    "workflows/concept/step-4-audit.md",
    "workflows/concept/step-5-fix.md",
    "workflows/concept/content-examples.md",
    "workflows/paper.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CODEBUDDY.md",
]

CONTENT_MARKERS = ("@content", "@component", "TODO", "TBD")


def load_manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


def target_for_path(path: str) -> Path:
    p = ROOT / path
    if path.endswith("/"):
        return p / "index.html"
    return p


def validate_required_files() -> list[str]:
    return [f"missing required file: {path}" for path in REQUIRED_FILES if not (ROOT / path).exists()]


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


def validate_shell(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    rel = path.relative_to(ROOT)
    if not text.startswith("<!DOCTYPE html>"):
        errors.append(f"{rel}: missing <!DOCTYPE html>")
    if not text.rstrip().endswith("</html>"):
        errors.append(f"{rel}: missing </html>")
    return errors


def validate_content_page(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    rel = path.relative_to(ROOT)
    searchable_text = re.sub(r"data:[^\"']+", "", text)

    for match in PLACEHOLDER_RE.findall(searchable_text):
        errors.append(f"{rel}: template placeholder remains: {match}")
    for needle in CONTENT_MARKERS:
        if needle in searchable_text:
            errors.append(f"{rel}: template marker remains: {needle}")

    ids = ID_RE.findall(text)
    seen: set[str] = set()
    for id_value in ids:
        if id_value in seen:
            errors.append(f"{rel}: duplicate id: {id_value}")
        seen.add(id_value)

    id_set = set(ids)
    for anchor in SAME_PAGE_ANCHOR_RE.findall(text):
        if anchor and anchor not in id_set:
            errors.append(f"{rel}: anchor points to missing id: #{anchor}")

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
            errors.append(f"{rel}: broken local reference {ref}")

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
    errors: list[str] = []
    errors.extend(validate_required_files())
    errors.extend(validate_manifest(manifest))
    errors.extend(validate_index(manifest))

    for html in sorted((ROOT / "content").glob("**/*.html")):
        text = html.read_text(encoding="utf-8", errors="ignore")
        errors.extend(validate_shell(html, text))
        errors.extend(validate_content_page(html, text))

    for html in sorted((ROOT / "templates").glob("**/*.html")):
        text = html.read_text(encoding="utf-8", errors="ignore")
        if "@component" in text:
            continue  # 组件库是片段集合，不是完整 HTML 文档
        errors.extend(validate_shell(html, text))

    if errors:
        print("validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("validation ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
