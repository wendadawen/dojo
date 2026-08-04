#!/usr/bin/env python3
"""Generate the Dojo homepage from content.json."""

from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "content.json"
OUTPUT = ROOT / "index.html"


def load_manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


def card(item: dict, tag_class: str) -> str:
    return f"""          <a class="card" href="{escape(item['path'])}">
            <span class="tag {tag_class}">{escape(item['tag'])}</span>
            <h3 class="card-title">{escape(item['title'])}</h3>
            <p class="card-desc">{escape(item['desc'])}</p>
          </a>"""


def render_papers(items: list[dict]) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item.get("group", "未分组")].append(item)

    parts = []
    for group, group_items in groups.items():
        cards = "\n".join(card(item, "paper") for item in group_items)
        parts.append(f"""      <details class="group">
        <summary class="group-summary">
          <div class="left">
            <span class="group-name">{escape(group)}</span>
            <span class="group-count">{len(group_items)} 篇</span>
          </div>
          <span class="group-arrow">▶</span>
        </summary>
        <div class="group-body">
{cards}
        </div>
      </details>""")
    return "\n".join(parts)


def render_cards(items: list[dict], section_id: str) -> str:
    if not items:
        return "      <p class=\"empty\">还没有内容。完成一次学习后再决定是否留档。</p>"
    cards = "\n".join(card(item, section_id.rstrip("s")) for item in items)
    return f"""      <div class="group-body flat">
{cards}
      </div>"""


def render_section(section: dict, items: list[dict]) -> str:
    if section["id"] == "papers" and items:
        body = render_papers(items)
    else:
        body = render_cards(items, section["id"])
    return f"""    <section class="section">
      <h2 class="section-title">{escape(section['title'])}</h2>
      <p class="section-desc">{escape(section['description'])}</p>
{body}
    </section>"""


def render(manifest: dict) -> str:
    site = manifest["site"]
    items_by_section: dict[str, list[dict]] = defaultdict(list)
    for item in manifest["items"]:
        items_by_section[item["section"]].append(item)

    sections = "\n\n".join(
        render_section(section, items_by_section[section["id"]])
        for section in manifest["sections"]
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escape(site['name'])}</title>
  <style>
    :root {{
      --font-body: Charter, 'Bitstream Charter', 'Sitka Text', Cambria, Georgia, serif;
      --font-ui: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      --font-mono: 'SF Mono', Monaco, Menlo, Consolas, 'Courier New', monospace;
      --text: rgba(0, 0, 0, 0.88); --text-light: rgba(0, 0, 0, 0.6); --text-muted: rgba(0, 0, 0, 0.4);
      --bg: #ffffff; --bg-soft: #fafafa; --border: rgba(0, 0, 0, 0.1); --border-light: rgba(0, 0, 0, 0.05);
      --blue: #0969da; --green: #1a7f37; --purple: #8250df; --orange: #bc4c00;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --text: rgba(255, 255, 255, 0.88); --text-light: rgba(255, 255, 255, 0.6); --text-muted: rgba(255, 255, 255, 0.4);
        --bg: #0d1117; --bg-soft: #161b22; --border: rgba(255, 255, 255, 0.12); --border-light: rgba(255, 255, 255, 0.06);
        --blue: #58a6ff; --green: #3fb950; --purple: #bc8cff; --orange: #ffa657;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: var(--font-ui); background: var(--bg); color: var(--text); line-height: 1.6; -webkit-font-smoothing: antialiased; }}
    .container {{ max-width: 920px; margin: 0 auto; padding: 4rem 1.5rem 6rem; }}
    header {{ margin-bottom: 3.5rem; }}
    h1 {{ font-family: var(--font-body); font-size: 44px; font-weight: 600; margin: 0 0 0.5rem; letter-spacing: -0.02em; }}
    .subtitle {{ max-width: 680px; color: var(--text-light); font-size: 17px; margin: 0; }}
    .meta {{ color: var(--text-muted); font-size: 13px; margin-top: 1rem; font-family: var(--font-mono); }}
    .section {{ margin-top: 3.25rem; }}
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
    .group-body.flat {{ padding: 0; }}
    .card {{ display: block; padding: 1.25rem 1.5rem; background: var(--bg); border: 1px solid var(--border-light); border-radius: 6px; text-decoration: none; color: inherit; }}
    .card:hover {{ border-color: var(--border); }}
    .card-title {{ font-family: var(--font-body); font-size: 18px; font-weight: 600; margin: 0 0 0.3rem; color: var(--text); }}
    .card-desc {{ font-size: 13px; color: var(--text-light); margin: 0; }}
    .tag {{ display: inline-block; font-family: var(--font-mono); font-size: 11px; color: var(--purple); margin-bottom: 0.4rem; }}
    .tag.paper {{ color: var(--green); }} .tag.concept {{ color: var(--blue); }} .tag.note {{ color: var(--orange); }}
    .empty {{ margin: 0; padding: 1.2rem 1.3rem; color: var(--text-muted); background: var(--bg-soft); border: 1px dashed var(--border); border-radius: 8px; font-size: 13px; }}
    footer {{ margin-top: 4rem; padding-top: 2rem; border-top: 1px solid var(--border-light); color: var(--text-muted); font-size: 13px; }}
    footer a {{ color: var(--blue); text-decoration: none; }} footer a:hover {{ text-decoration: underline; }}
    @media (max-width: 680px) {{ .container {{ padding-top: 2.5rem; }} h1 {{ font-size: 38px; }} }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>{escape(site['name'])}</h1>
      <p class="subtitle">{escape(site['subtitle'])}</p>
      <p class="meta">{escape(site['repo'])}</p>
    </header>

{sections}

    <footer>
      <a href="{escape(site['repo_url'])}">查看仓库源码</a>
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
