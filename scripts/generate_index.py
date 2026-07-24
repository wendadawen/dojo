#!/usr/bin/env python3
"""Generate the Paper Reading homepage from content.json."""

from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "content.json"
OUTPUT = ROOT / "index.html"


def load_manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


def card(item: dict, tag_class: str) -> str:
    return f"""          <a class=\"card\" href=\"{escape(item['path'])}\">
            <span class=\"tag {tag_class}\">{escape(item['tag'])}</span>
            <h3 class=\"card-title\">{escape(item['title'])}</h3>
            <p class=\"card-desc\">{escape(item['desc'])}</p>
          </a>"""


def render_papers(items: list[dict]) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item.get("group", "未分组")].append(item)

    parts = []
    for group, group_items in groups.items():
        cards = "\n".join(card(item, "paper") for item in group_items)
        parts.append(f"""      <details class=\"group\">
        <summary class=\"group-summary\">
          <div class=\"left\">
            <span class=\"group-name\">{escape(group)}</span>
            <span class=\"group-count\">{len(group_items)} 篇</span>
          </div>
          <span class=\"group-arrow\">▶</span>
        </summary>
        <div class=\"group-body\">
{cards}
        </div>
      </details>""")
    return "\n".join(parts)


def render_concepts(items: list[dict]) -> str:
    cards = "\n".join(card(item, "concept") for item in items)
    return f"""      <div class=\"group-body\" style=\"padding:0; gap:0.75rem;\">
{cards}
      </div>"""


def render_section(section: dict, items: list[dict]) -> str:
    if section["id"] == "papers":
        body = render_papers(items)
    else:
        body = render_concepts(items)
    return f"""    <section class=\"section\">
      <h2 class=\"section-title\">{escape(section['title'])}</h2>
      <p class=\"section-desc\">{escape(section['description'])}</p>
{body}
    </section>"""


def render(manifest: dict) -> str:
    items_by_section: dict[str, list[dict]] = defaultdict(list)
    for item in manifest["items"]:
        items_by_section[item["section"]].append(item)

    sections = "\n\n".join(
        render_section(section, items_by_section[section["id"]])
        for section in manifest["sections"]
    )

    return f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>Paper Reading</title>
  <style>
    :root {{
      --font-body: Charter, 'Bitstream Charter', 'Sitka Text', Cambria, Georgia, serif;
      --font-ui: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      --font-mono: 'SF Mono', Monaco, Menlo, Consolas, 'Courier New', monospace;
      --text: rgba(0, 0, 0, 0.88);
      --text-light: rgba(0, 0, 0, 0.6);
      --text-muted: rgba(0, 0, 0, 0.4);
      --bg: #ffffff;
      --bg-soft: #fafafa;
      --border: rgba(0, 0, 0, 0.1);
      --border-light: rgba(0, 0, 0, 0.05);
      --blue: #0969da;
      --green: #1a7f37;
      --purple: #8250df;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --text: rgba(255, 255, 255, 0.88);
        --text-light: rgba(255, 255, 255, 0.6);
        --text-muted: rgba(255, 255, 255, 0.4);
        --bg: #0d1117;
        --bg-soft: #161b22;
        --border: rgba(255, 255, 255, 0.12);
        --border-light: rgba(255, 255, 255, 0.06);
        --blue: #58a6ff;
        --green: #3fb950;
        --purple: #bc8cff;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 0; font-family: var(--font-ui); background: var(--bg); color: var(--text); line-height: 1.6; -webkit-font-smoothing: antialiased; }}
    .container {{ max-width: 820px; margin: 0 auto; padding: 4rem 1.5rem 6rem; }}
    header {{ margin-bottom: 3rem; }}
    h1 {{ font-family: var(--font-body); font-size: 36px; font-weight: 600; margin: 0 0 0.5rem; letter-spacing: -0.01em; }}
    .subtitle {{ color: var(--text-light); font-size: 16px; margin: 0; }}
    .meta {{ color: var(--text-muted); font-size: 13px; margin-top: 1rem; font-family: var(--font-mono); }}
    .section {{ margin-top: 2.5rem; }}
    .section-title {{ font-family: var(--font-body); font-size: 24px; font-weight: 600; margin: 0 0 0.25rem; }}
    .section-desc {{ color: var(--text-light); font-size: 14px; margin: 0 0 1rem; }}
    .group {{ margin-bottom: 0.75rem; border: 1px solid var(--border-light); border-radius: 8px; overflow: hidden; background: var(--bg-soft); }}
    .group-summary {{ padding: 1rem 1.25rem; cursor: pointer; display: flex; align-items: center; justify-content: space-between; list-style: none; user-select: none; }}
    .group-summary::-webkit-details-marker {{ display: none; }}
    .group-summary .left {{ display: flex; align-items: baseline; gap: 0.75rem; }}
    .group-name {{ font-family: var(--font-body); font-size: 18px; font-weight: 600; }}
    .group-count {{ font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); }}
    .group-arrow {{ color: var(--text-muted); font-size: 14px; transition: transform 0.2s ease; }}
    .group[open] .group-arrow {{ transform: rotate(90deg); }}
    .group-body {{ padding: 0 1.25rem 1rem; display: flex; flex-direction: column; gap: 0.75rem; }}
    .card {{ display: block; padding: 1.25rem 1.5rem; background: var(--bg); border: 1px solid var(--border-light); border-radius: 6px; text-decoration: none; color: inherit; transition: border-color 0.15s ease, transform 0.15s ease; }}
    .card:hover {{ border-color: var(--border); transform: translateY(-1px); }}
    .card-title {{ font-family: var(--font-body); font-size: 18px; font-weight: 600; margin: 0 0 0.3rem; color: var(--text); }}
    .card-desc {{ font-size: 13px; color: var(--text-light); margin: 0; }}
    .tag {{ display: inline-block; font-family: var(--font-mono); font-size: 11px; color: var(--purple); margin-bottom: 0.4rem; }}
    .tag.paper {{ color: var(--green); }}
    .tag.concept {{ color: var(--blue); }}
    footer {{ margin-top: 4rem; padding-top: 2rem; border-top: 1px solid var(--border-light); color: var(--text-muted); font-size: 13px; }}
    footer a {{ color: var(--blue); text-decoration: none; }}
    footer a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class=\"container\">
    <header>
      <h1>Paper Reading</h1>
      <p class=\"subtitle\">论文解读与概念学习笔记合集</p>
      <p class=\"meta\">github.com/wendadawen/paper-reading</p>
    </header>

{sections}

    <footer>
      <a href=\"https://github.com/wendadawen/paper-reading\">查看仓库源码</a>
    </footer>
  </div>
</body>
</html>
"""


def main() -> None:
    OUTPUT.write_text(render(load_manifest()), encoding="utf-8")
    print(f"generated {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
