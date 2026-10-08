"""检查页面里内联 <script> 的语法，由 check.py 调用，需要 node。只查语法，不执行代码。"""

from __future__ import annotations

import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path


class Collector(HTMLParser):
    """收集页面里所有内联脚本的源码。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[str] = []
        self._attrs: dict[str, str] = {}
        self._buffer: list[str] = []
        self._inside = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "script":
            self._attrs = {k.lower(): (v or "") for k, v in attrs}
            self._buffer, self._inside = [], True

    def handle_data(self, data):
        if self._inside:
            self._buffer.append(data)

    def handle_endtag(self, tag):
        if tag.lower() != "script" or not self._inside:
            return
        self._inside = False
        kind = self._attrs.get("type", "").split(";")[0].strip().lower()
        body = "".join(self._buffer)
        if "src" not in self._attrs and kind in {"", "text/javascript", "module"} and body.strip():
            self.blocks.append(body)


def check(pages: list[Path]) -> tuple[list[str], int]:
    """返回 (问题, 脚本块数)。"""
    errors: list[str] = []
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for page in pages:
            collector = Collector()
            collector.feed(page.read_text(encoding="utf-8", errors="replace"))
            for index, body in enumerate(collector.blocks, start=1):
                total += 1
                target = Path(tmp) / f"{page.parent.name}-{page.stem}-{index}.js"
                target.write_text(body, encoding="utf-8")
                proc = subprocess.run(["node", "--check", str(target)], capture_output=True, text=True)
                if proc.returncode:
                    message = next((l for l in proc.stderr.splitlines() if "Error" in l), "语法错误")
                    errors.append(f"{page} 内联 <script> #{index}：{message}")
    return errors, total


CASES = [
    ("正确的脚本没有报错", "<script>var x = 1;</script>", None),
    ("括号不成对", "<script>var x = (1;</script>", "内联 <script> #1"),
    ("引号不成对", "<script>var s = 'a;</script>", "内联 <script> #1"),
    ("外链脚本不查", '<script src="a.js"></script>', None),
    ("type=text/plain 的块不查", '<script type="text/plain">这不是 js (</script>', None),
]


def selftest() -> list[tuple[str, bool]]:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "index.html"
        for name, html, want in CASES:
            page.write_text(f"<html><body>{html}</body></html>", encoding="utf-8")
            errs, _ = check([page])
            results.append((name, not errs if want is None else any(want in e for e in errs)))
    return results
