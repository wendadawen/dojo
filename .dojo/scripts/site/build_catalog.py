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
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "concept"))
from common import report  # noqa: E402


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


# ---------- 自检 ----------

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head>
<meta name="description" content="  一句  描述 ">
<meta name="dojo:summary" content="摘要">
<meta name="dojo:topics" content="注意力机制, 内存与缓存">
<meta name="dojo:tag" content="注意力">
<title>标题</title></head><body><h1>页面名</h1>{links}</body></html>
"""


def selftest() -> int:
    results: list[tuple[str, bool]] = []

    # normalize_target：链接归一化。href 一律按「相对源页面所在目录」解析。
    cases = [
        ("同目录另一页", "wiki/a/index.html", "b/", "wiki/a/b/index.html"),
        ("上级目录另一页", "wiki/a/index.html", "../b/", "wiki/b/index.html"),
        ("相对路径", "wiki/a/index.html", "../b/index.html", "wiki/b/index.html"),
        ("概览页折到长页", "wiki/a/index.html", "../b/overview.html", "wiki/b/index.html"),
        ("指向自己不算", "wiki/a/index.html", "index.html", None),
        ("外部网址不算", "wiki/a/index.html", "https://x.com", None),
        ("非 wiki 路径不算", "wiki/a/index.html", "../../libs/x.js", None),
        ("锚点不算", "wiki/a/index.html", "#sec", None),
    ]
    for name, source, href, want in cases:
        got = normalize_target(source, href)
        results.append((f"链接：{name}", got == want))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "wiki" / "a").mkdir(parents=True)
        (root / "wiki" / "b").mkdir(parents=True)
        (root / "wiki" / "a" / "index.html").write_text(
            PAGE.format(links='<a href="../b/">去 b</a><a href="../nope/">断链</a>'), encoding="utf-8")
        (root / "wiki" / "b" / "index.html").write_text(PAGE.format(links=""), encoding="utf-8")
        cat = build_catalog(root)
        by_id = {p["id"]: p for p in cat["pages"]}

        results.append(("页面：扫到两个 wiki 页", len(cat["pages"]) == 2))
        results.append(("元数据：描述去多余空白",
                        by_id["wiki/a/index.html"]["description"] == "一句 描述"))
        results.append(("元数据：主题按中英文逗号拆开",
                        by_id["wiki/a/index.html"]["topics"] == ["注意力机制", "内存与缓存"]))
        results.append(("标题：优先取 h1",
                        by_id["wiki/a/index.html"]["title"] == "页面名"))
        results.append(("边：a 指向 b 记一条", len(cat["edges"]) == 1
                        and cat["edges"][0]["target"] == "wiki/b/index.html"))
        results.append(("边：数量累计", cat["edges"][0]["count"] == 1))
        results.append(("出入度：b 有一入边",
                        by_id["wiki/b/index.html"]["incoming"] == ["wiki/a/index.html"]))
        results.append(("出入度：a 有一出边",
                        by_id["wiki/a/index.html"]["outgoing"] == ["wiki/b/index.html"]))
        results.append(("警告：断链被记下",
                        [w["target"] for w in cat["warnings"]] == ["wiki/nope/index.html"]))
        results.append(("排序：页面按标题排",
                        [p["id"] for p in cat["pages"]] == ["wiki/a/index.html", "wiki/b/index.html"]))
        # 无 git 历史时不应崩（临时目录不是仓库）
        results.append(("无 git 历史时日期留空",
                        all(p["date"] == "" for p in cat["pages"])))

    return report(results, "catalog 生成")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    main()
