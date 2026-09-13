#!/usr/bin/env python3
"""从审查记录算出每个页面的审查状态。

页面本身不标注自己的可信度，读者无法分辨哪篇经过独立审查、哪篇没有——
审计把这一条列为最伤使用者的问题。本脚本把状态从 research/review-*.md
算出来，供首页在构建期渲染徽章；不往页面里写字段，免得每加一轮审查就要
回填 98 个文件。

每页算出四项：
  rounds      完成的独立审查轮次（review-0 是命名规范之前的记录，不计入）
  measured    research/ 下是否有实测产物（脚本或运行输出）
  stale       页面正文在最后一轮审查之后是否被改过
  reviewed_at 最后一轮审查的日期

    python3 .dojo/scripts/review_status.py [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

TAG_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
ANY_TAG_RE = re.compile(r"<[^>]+>")
HEADER_RE = re.compile(r"<!--\s*review-meta\b.*?-->\s*", re.S)
ROUND_RE = re.compile(r"review-(\d+)\.md$")


def content_fingerprint(html: str) -> str:
    text = TAG_RE.sub(" ", html)
    text = ANY_TAG_RE.sub(" ", text)
    return hashlib.sha256(
        re.sub(r"\s+", " ", text).strip().encode("utf-8")
    ).hexdigest()[:16]


def header_fingerprint(text: str) -> str | None:
    match = HEADER_RE.search(text)
    if not match:
        return None
    field = re.search(r"reviewed_content_sha256:\s*(\S+)", match.group(0))
    if not field or field.group(1) == "未记录":
        return None
    return field.group(1)


OPEN_RE = re.compile(r"统计：\s*阻断\s*(\d+)\s*/\s*重要\s*(\d+)")


def open_issues(path: Path) -> tuple[int, int]:
    """最新一轮审查记录末尾统计里仍未关闭的阻断数与重要数。

    徽章若只看「轮次够不够」，会把「审过但问题还没修」的页面显示成完整审查——
    而 check.md 的发布条件是阻断和重要全部关闭。
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    found = OPEN_RE.findall(text)
    if not found:
        return (0, 0)
    blocking, important = found[-1]
    return int(blocking), int(important)


def file_date(path: Path) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ad", "--date=short", "--", str(path)],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or ""


def status_of(slug: str, root: Path) -> dict:
    page = root / "wiki" / slug / "index.html"
    research = root / "wiki" / slug / "research"

    rounds = 0
    latest_round = None
    latest_path = None
    for path in sorted(research.glob("review-*.md")) if research.is_dir() else []:
        match = ROUND_RE.search(path.name)
        if not match:
            continue
        number = int(match.group(1))
        if number == 0:
            continue  # 命名规范之前的记录，不计入轮次
        rounds += 1
        if latest_round is None or number > latest_round:
            latest_round, latest_path = number, path

    fingerprint = None
    if latest_path is not None:
        fingerprint = header_fingerprint(latest_path.read_text(encoding="utf-8"))

    current = content_fingerprint(page.read_text(encoding="utf-8")) if page.exists() else None
    # 「含实测」看记录而不是看产物：实测产物已从仓库移除，清单留在 measured.md；
    # 早期页面在 evidence.md 里用 MEAS 记号标注实测论断。
    measured = False
    if research.is_dir():
        if (research / "measured.md").is_file():
            measured = True
        else:
            for note in research.glob("*.md"):
                if "MEAS" in note.read_text(encoding="utf-8", errors="replace"):
                    measured = True
                    break

    stale = None
    if fingerprint and current:
        stale = fingerprint != current

    open_blocking = open_important = 0
    if latest_path is not None:
        open_blocking, open_important = open_issues(latest_path)

    return {
        "slug": slug,
        "rounds": rounds,
        "measured": measured,
        "stale": stale,
        "reviewed_at": file_date(latest_path) if latest_path else "",
        "open_blocking": open_blocking,
        "open_important": open_important,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)

    slugs = sorted(p.parent.name for p in (root / "wiki").glob("*/index.html"))
    statuses = [status_of(slug, root) for slug in slugs]

    if args.json:
        print(json.dumps({s["slug"]: s for s in statuses}, ensure_ascii=False, indent=1))
        return 0

    full = [s for s in statuses if s["rounds"] >= 3]
    none = [s for s in statuses if s["rounds"] == 0]
    changed = [s for s in statuses if s["stale"]]
    with_open = [s for s in statuses if s["open_blocking"] or s["open_important"]]
    print(f"页面 {len(statuses)}")
    print(f"  走完 >=3 轮: {len(full)}")
    print(f"  0 轮:        {len(none)}")
    print(f"  审查后正文被改过: {len(changed)}")
    print(f"  有实测产物:  {sum(1 for s in statuses if s['measured'])}")
    print(f"  最后一轮仍有未关闭的阻断或重要: {len(with_open)}")
    print("\n轮次分布:")
    counts: dict[int, int] = {}
    for item in statuses:
        counts[item["rounds"]] = counts.get(item["rounds"], 0) + 1
    for rounds in sorted(counts):
        print(f"  {rounds} 轮: {counts[rounds]} 页")
    return 0


if __name__ == "__main__":
    sys.exit(main())
