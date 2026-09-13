#!/usr/bin/env python3
"""用真实浏览器确认一批选择器在页面上匹配不到元素。

check_unused_css.py 是静态判断（class 名没出现在标记或脚本里）。本脚本是它的
运行时对照：把页面在无头 Chrome 里跑起来，收集**脚本执行完之后** DOM 上真实
存在的全部 class 名，再核对候选选择器里的 class 记号是否真的不在其中。

静态判断会漏掉脚本动态添加的 class，也会漏掉库运行时注入的 class；浏览器实测
能兜住这两类。两者一致时，删除该规则不会改变任何页面的渲染。

    python3 .dojo/scripts/verify_css_unused.py <页面...>

从标准输入读取待核对的选择器（每行一条，来自 check_unused_css.py 的输出）。
对每个选择器报告它在多少个页面上匹配到了元素；匹配数为 0 的记 ok。

退出码 0 表示全部选择器匹配数为 0（删除安全），1 表示有选择器能匹配到元素，
2 表示用法或环境错误（例如找不到 Chrome）。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

PROBE = """<script>
window.addEventListener('load', function () {
  var seen = {};
  var nodes = document.querySelectorAll('*');
  for (var i = 0; i < nodes.length; i++) {
    var list = nodes[i].classList;
    for (var j = 0; j < list.length; j++) seen[list[j]] = 1;
  }
  document.title = 'CLASSES:' + JSON.stringify(Object.keys(seen));
});
</script>
"""

TITLE_RE = re.compile(r"<title>CLASSES:(.*?)</title>", re.S)


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def live_classes(chrome: str, page: Path) -> set[str]:
    """把页面跑起来，返回脚本执行完之后 DOM 上真实存在的 class 名。"""
    document = page.read_text(encoding="utf-8", errors="replace")
    if "<head>" in document:
        document = document.replace("<head>", "<head>" + PROBE, 1)
    else:
        document = PROBE + document
    with tempfile.TemporaryDirectory() as tmp:
        probe_page = Path(tmp) / page.name
        probe_page.write_text(document, encoding="utf-8")
        # 让相对路径仍能解析到原页面的同级资源
        proc = subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--virtual-time-budget=4000",
                "--dump-dom",
                probe_page.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
    match = TITLE_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"probe did not report on {page}")
    return set(json.loads(match.group(1)))


def main() -> int:
    targets = [Path(a) for a in sys.argv[1:]]
    if not targets:
        print(__doc__)
        return 2
    missing = [p for p in targets if not p.exists()]
    if missing:
        print("error: not found: " + ", ".join(str(p) for p in missing))
        return 2

    chrome = find_chrome()
    if chrome is None:
        print("error: no Chrome/Chromium found; cannot verify in a browser")
        return 2

    selectors = [line.strip().lstrip("-").strip() for line in sys.stdin if line.strip()]
    # 只看含 class 记号的选择器，逗号分组
    wanted: dict[str, list[str]] = {}
    for selector in selectors:
        tokens = re.findall(r"\.(-?[A-Za-z_][\w-]*)", selector)
        if tokens:
            wanted[selector] = tokens
    if not wanted:
        print("no class selectors to verify")
        return 0

    print(f"verifying {len(wanted)} selectors against {len(targets)} pages")
    live = {}
    for page in targets:
        try:
            live[page] = live_classes(chrome, page)
        except Exception as error:  # noqa: BLE001 - 报告后继续
            print(f"error: {error}")
            return 2

    survivors = []
    for selector, tokens in wanted.items():
        hits = sum(1 for classes in live.values() if all(t in classes for t in tokens))
        if hits:
            survivors.append((selector, hits))

    if survivors:
        print(f"\n{len(survivors)} selectors still match elements — must not be removed:")
        for selector, hits in survivors:
            print(f"  ! {selector}  (matches on {hits} pages)")
        return 1

    print("\nall selectors match zero elements on every page — removal is safe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
