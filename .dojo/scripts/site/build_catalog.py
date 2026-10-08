#!/usr/bin/env python3
"""生成首页用的 catalog.json：扫描 wiki/*/index.html，得到页面、链接关系和失效链接警告。

    python3 .dojo/scripts/site/build_catalog.py --output _site/catalog.json [--root <仓库根>]
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


def clean_text(value: str) -> str:
    return " ".join(value.split())


class PageParser(HTMLParser):
    """读出 meta、<a href>、<title> 和 <h1>。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.hrefs: list[str] = []
        self.text = {"title": "", "h1": ""}
        self._open = ""

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta" and values.get("name") and values.get("content"):
            self.meta[values["name"].strip().lower()] = values["content"].strip()
        elif tag == "a" and values.get("href"):
            self.hrefs.append(values["href"].strip())
        elif tag in self.text:
            self._open = tag

    def handle_endtag(self, tag):
        if tag == self._open:
            self._open = ""

    def handle_data(self, data):
        if self._open:
            self.text[self._open] += data


def first_seen_dates(root: Path) -> dict[str, str]:
    """wiki/ 下每个文件最早出现的提交日期；没有 git 历史时为空。"""
    cmd = ["git", "log", "--reverse", "--format=%aI", "--name-only", "--", "wiki/"]
    try:
        out = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    dates: dict[str, str] = {}
    date = ""
    for line in map(str.strip, out.splitlines()):
        if re.match(r"\d{4}-\d{2}-\d{2}T", line):
            date = line[:10]
        elif line.startswith("wiki/") and date:
            dates.setdefault(line, date)
    return dates


def parse_page(root: Path, path: Path, date: str) -> dict:
    relative = path.relative_to(root).as_posix()
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    meta = parser.meta
    return {
        "id": relative,
        "path": relative,
        "title": clean_text(parser.text["h1"]) or clean_text(parser.text["title"]) or path.parent.name,
        "description": clean_text(meta.get("description", "")),
        "summary": clean_text(meta.get("dojo:summary", "")),
        "topics": [t.strip() for t in re.split(r"[,，]", meta.get("dojo:topics", "")) if t.strip()],
        "tag": meta.get("dojo:tag", ""),
        "date": date,
        "_hrefs": parser.hrefs,
    }


def normalize_target(source: str, href: str) -> str | None:
    """href 指向的 wiki 页面（长页路径），不是站内页面链接则为 None。"""
    parts = urlsplit(href)
    if parts.scheme or parts.netloc or not parts.path:
        return None
    target = posixpath.normpath(posixpath.join(posixpath.dirname(source), parts.path))
    if parts.path.endswith("/"):
        target += "/index.html"
    if target.endswith("/overview.html"):
        target = target.removesuffix("overview.html") + "index.html"
    if not (target.startswith("wiki/") and target.endswith("/index.html")) or target == source:
        return None
    return target


def build_catalog(root: Path) -> dict:
    root = root.resolve()
    dates = first_seen_dates(root)
    pages = [parse_page(root, p, dates.get(p.relative_to(root).as_posix(), ""))
             for p in sorted((root / "wiki").glob("*/index.html"))]
    ids = {page["id"] for page in pages}
    edge_counts: dict[tuple[str, str], int] = {}
    warnings = []
    for page in pages:
        for href in page.pop("_hrefs"):
            target = normalize_target(page["id"], href)
            if target is None:
                continue
            if target in ids:
                edge_counts[(page["id"], target)] = edge_counts.get((page["id"], target), 0) + 1
            else:
                warnings.append({"type": "missing_target", "source": page["id"], "target": target})

    edges = [{"id": f"{s}::{t}", "source": s, "target": t, "count": n} for (s, t), n in sorted(edge_counts.items())]
    for page in pages:
        page["incoming"] = sorted(e["source"] for e in edges if e["target"] == page["id"])
        page["outgoing"] = sorted(e["target"] for e in edges if e["source"] == page["id"])
        page["incoming_count"] = len(page["incoming"])
        page["outgoing_count"] = len(page["outgoing"])
    pages.sort(key=lambda page: (page["title"].casefold(), page["id"]))
    return {"pages": pages, "edges": edges, "warnings": sorted(warnings, key=lambda w: (w["source"], w["target"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成首页用的 catalog.json")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="仓库根，默认当前目录")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = build_catalog(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {len(catalog['pages'])} 个页面、{len(catalog['edges'])} 条边、"
          f"{len(catalog['warnings'])} 条警告 -> {args.output}")


if __name__ == "__main__":
    main()
