#!/usr/bin/env python3
"""Migrate legacy content.json metadata into wiki page <head> tags."""

from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path


TYPE_MAP = {"papers": "paper", "concepts": "concept", "notes": "note"}
META_NAMES = ("description", "dojo:type", "dojo:topics", "dojo:tag")
BODY_RE = re.compile(r"<body\b", re.IGNORECASE)


def metadata_for(item: dict) -> dict[str, str]:
    return {
        "description": item.get("desc", "").strip(),
        "dojo:type": TYPE_MAP.get(item.get("section", ""), "unknown"),
        "dojo:topics": item.get("group", "").strip(),
        "dojo:tag": item.get("tag", "").strip(),
    }


def migrate_text(text: str, values: dict[str, str]) -> str:
    body_match = BODY_RE.search(text)
    if body_match is None:
        raise ValueError("missing <body>")

    head = text[: body_match.start()]
    body = text[body_match.start() :]

    for name in META_NAMES:
        pattern = re.compile(
            rf"^[ \t]*<meta\s+name=[\"']{re.escape(name)}[\"'][^>]*>[ \t]*\n?",
            re.IGNORECASE | re.MULTILINE,
        )
        head = pattern.sub("", head)

    block = "\n".join(
        f'  <meta name="{name}" content="{escape(values[name], quote=True)}">'
        for name in META_NAMES
    )
    viewport = re.search(
        r"^[ \t]*<meta\s+name=[\"']viewport[\"'][^>]*>\s*$",
        head,
        re.IGNORECASE | re.MULTILINE,
    )
    if viewport:
        insert_at = viewport.end()
        head = head[:insert_at] + "\n" + block + "\n" + head[insert_at:].lstrip("\n")
    else:
        title = re.search(r"^[ \t]*<title\b", head, re.IGNORECASE | re.MULTILINE)
        if title is None:
            raise ValueError("missing viewport and <title>")
        head = head[: title.start()] + block + "\n" + head[title.start() :]

    return head + body


def migrate(root: Path, manifest: Path, write: bool) -> dict:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    changed: list[str] = []
    missing: list[str] = []
    listed: set[str] = set()

    for item in payload.get("items", []):
        relative = item.get("path", "")
        if not relative:
            continue
        listed.add(relative)
        page = root / relative
        if not page.exists():
            missing.append(relative)
            continue
        before = page.read_text(encoding="utf-8")
        after = migrate_text(before, metadata_for(item))
        if after != before:
            changed.append(relative)
            if write:
                page.write_text(after, encoding="utf-8")

    actual = {
        page.relative_to(root).as_posix()
        for page in (root / "wiki").glob("*/index.html")
    }
    return {
        "changed": sorted(changed),
        "missing": sorted(missing),
        "unlisted": sorted(actual - listed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = root / manifest

    report = migrate(root, manifest, args.write)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
