#!/usr/bin/env python3
"""Build homepage catalog data from wiki HTML pages."""

from __future__ import annotations

import json
import posixpath
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit


SPACE_RE = re.compile(r"\s+")
IGNORED_SCHEMES = {"http", "https", "mailto", "javascript", "data", "tel"}


def clean_text(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip()


def split_topics(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，]", value) if item.strip()]


class WikiHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.hrefs: list[str] = []
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.paragraphs: list[str] = []
        self.leads: list[str] = []
        self._title_depth = 0
        self._h1_depth = 0
        self._paragraph_depth = 0
        self._ignored_depth = 0
        self._paragraph_parts: list[str] = []
        self._paragraph_is_lead = False

    def handle_starttag(self, tag: str, attrs: list[tuple]) -> None:
        values = dict(attrs)
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "meta":
            name = values.get("name", "").strip().lower()
            content = values.get("content", "").strip()
            if name and content:
                self.meta[name] = content
        elif tag == "a":
            href = values.get("href")
            if href:
                self.hrefs.append(href.strip())
        elif tag == "title":
            self._title_depth += 1
        elif tag == "h1":
            self._h1_depth += 1
        elif tag == "p":
            self._paragraph_depth += 1
            self._paragraph_parts = []
            classes = values.get("class", "").split()
            self._paragraph_is_lead = bool({"lead", "summary"} & set(classes))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        elif tag == "h1" and self._h1_depth:
            self._h1_depth -= 1
        elif tag == "p" and self._paragraph_depth:
            paragraph = clean_text(" ".join(self._paragraph_parts))
            if paragraph:
                self.paragraphs.append(paragraph)
                if self._paragraph_is_lead:
                    self.leads.append(paragraph)
            self._paragraph_depth -= 1
            self._paragraph_parts = []
            self._paragraph_is_lead = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._title_depth:
            self.title_parts.append(data)
        if self._h1_depth:
            self.h1_parts.append(data)
        if self._paragraph_depth:
            self._paragraph_parts.append(data)


def discover_pages(root: Path) -> list[Path]:
    return sorted((root / "wiki").glob("*/index.html"))


def parse_page(root: Path, page_path: Path) -> dict:
    relative = page_path.relative_to(root).as_posix()
    parser = WikiHTMLParser()
    parser.feed(page_path.read_text(encoding="utf-8", errors="ignore"))

    h1 = clean_text(" ".join(parser.h1_parts))
    title_tag = clean_text(" ".join(parser.title_parts))
    title = h1 or title_tag or page_path.parent.name
    description = (
        parser.meta.get("description")
        or (parser.leads[0] if parser.leads else "")
        or (parser.paragraphs[0] if parser.paragraphs else "")
    )

    return {
        "id": relative,
        "path": relative,
        "title": title,
        "description": clean_text(description),
        "type": parser.meta.get("dojo:type", "").strip() or "unknown",
        "topics": split_topics(parser.meta.get("dojo:topics", "")),
        "tag": parser.meta.get("dojo:tag", "").strip(),
        "_hrefs": parser.hrefs,
    }


def normalize_target(source: str, href: str) -> Optional[str]:
    parts = urlsplit(href)
    if parts.scheme.lower() in IGNORED_SCHEMES or parts.netloc or not parts.path:
        return None
    base = posixpath.dirname(source)
    target = posixpath.normpath(posixpath.join(base, parts.path))
    if parts.path.endswith("/"):
        target = posixpath.join(target, "index.html")
    if target.endswith("/overview.html"):
        target = target[: -len("overview.html")] + "index.html"
    if not target.startswith("wiki/") or not target.endswith("/index.html"):
        return None
    if target == source:
        return None
    return target


def build_catalog(root: Path) -> dict:
    root = root.resolve()
    pages = [parse_page(root, path) for path in discover_pages(root)]
    page_ids = {page["id"] for page in pages}
    edge_counts: Counter = Counter()
    warnings: list[dict] = []

    for page in pages:
        if page["type"] == "unknown":
            warnings.append({"type": "unclassified", "source": page["id"]})
        if not page["topics"]:
            warnings.append({"type": "missing_topics", "source": page["id"]})
        if not page["tag"]:
            warnings.append({"type": "missing_tag", "source": page["id"]})
        for href in page.pop("_hrefs"):
            target = normalize_target(page["id"], href)
            if target is None:
                continue
            if target not in page_ids:
                warnings.append(
                    {"type": "missing_target", "source": page["id"], "target": target}
                )
                continue
            edge_counts[(page["id"], target)] += 1

    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    edges: list[dict] = []
    for (source, target), count in sorted(edge_counts.items()):
        incoming[target].append(source)
        outgoing[source].append(target)
        edges.append(
            {
                "id": f"{source}::{target}",
                "source": source,
                "target": target,
                "count": count,
            }
        )

    for page in pages:
        page["incoming"] = sorted(incoming[page["id"]])
        page["outgoing"] = sorted(outgoing[page["id"]])
        page["incoming_count"] = len(page["incoming"])
        page["outgoing_count"] = len(page["outgoing"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pages": sorted(pages, key=lambda page: (page["title"].casefold(), page["id"])),
        "edges": edges,
        "warnings": sorted(
            warnings,
            key=lambda item: (
                item.get("type", ""),
                item.get("source", ""),
                item.get("target", ""),
            ),
        ),
    }


def write_catalog(catalog: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
