#!/usr/bin/env python3
"""Parse-check the inline <script> blocks of wiki pages.

This is a syntax gate and nothing more. It extracts every inline script and
runs `node --check` on it, so a typo or an unbalanced brace fails the build
instead of reaching a reader as a silently dead page.

What it does NOT catch, and nothing else in this repository does either:
load-order mistakes, references to globals that are never defined, and
failures inside third-party bundles. A script can be perfectly valid
JavaScript and still throw the moment the browser runs it.

Usage:
    python3 .dojo/scripts/check_inline_js.py [path ...]

Paths may be files or directories; directories are scanned for *.html.
Defaults to wiki/ when no argument is given. Requires node in PATH.
Exits non-zero if any requested path is missing, so that a glob which
stops matching cannot silently narrow the check.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

# HTML 规范里浏览器会当作 JavaScript 执行的 type 取值
JS_MIME_TYPES = frozenset(
    {
        "application/ecmascript",
        "application/javascript",
        "application/x-ecmascript",
        "application/x-javascript",
        "text/ecmascript",
        "text/javascript",
        "text/javascript1.0",
        "text/javascript1.1",
        "text/javascript1.2",
        "text/javascript1.3",
        "text/javascript1.4",
        "text/javascript1.5",
        "text/jscript",
        "text/livescript",
        "text/x-ecmascript",
        "text/x-javascript",
        "module",
    }
)

# 老式 "<!-- ... //-->" 包裹，去掉注释标记后再交给 node
LEGACY_OPEN_RE = re.compile(r"^\s*<!--")
LEGACY_CLOSE_RE = re.compile(r"//-->\s*$")


class ScriptCollector(HTMLParser):
    """收集页面里所有内联脚本的源码。

    用 HTMLParser 而不是正则：它按 HTML 规则解析属性（引号内的 > 不会截断
    标签）、正确跳过注释里的示例代码，也不要求 </script> 后面直接跟 >。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[str] = []
        self._attrs: dict[str, str] = {}
        self._buffer: list[str] = []
        self._inside = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        self._attrs = {name.lower(): (value or "") for name, value in attrs}
        self._buffer = []
        self._inside = True

    def handle_data(self, data: str) -> None:
        if self._inside:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "script" or not self._inside:
            return
        self._inside = False
        if "src" in self._attrs:
            return
        kind = self._attrs.get("type", "").split(";")[0].strip().lower()
        if kind and kind not in JS_MIME_TYPES:
            return
        body = LEGACY_OPEN_RE.sub("", "".join(self._buffer))
        body = LEGACY_CLOSE_RE.sub("", body)
        if body.strip():
            self.blocks.append(body)


def collect_pages(paths: list[str]) -> tuple[list[Path], list[str]]:
    pages: list[Path] = []
    missing: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            pages.extend(sorted(path.rglob("*.html")))
        elif path.exists():
            pages.append(path)
        else:
            missing.append(raw)
    return pages, missing


def check_blocks(pages: list[Path], tmp: Path) -> tuple[list[str], int]:
    """把所有内联脚本写盘后，用一次 node --check 批量校验。

    逐块起 node 进程时，200 多个脚本块要起 200 多次进程，耗时以秒计；
    每次 node 启动本身是固定开销。合并成一次调用后总耗时降到百毫秒级。
    """
    errors: list[str] = []
    total = 0
    targets: list[tuple[str, Path]] = []
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        parser = ScriptCollector()
        parser.feed(text)
        for index, body in enumerate(parser.blocks, start=1):
            total += 1
            target = tmp / f"{page.parent.name}-{page.stem}-{index}.js"
            target.write_text(body, encoding="utf-8")
            targets.append((f"{page} inline <script> #{index}", target))

    if not targets:
        return errors, total

    # node --check 只接受单个文件；若对每个块各起一次 node，200 多次进程
    # 启动就是全部耗时。这里在单个 node 进程内用 vm.Script 逐个编译：
    # 只做语法检查、不执行代码，语义与 --check 一致，且失败信息按
    # FILE: 前缀分段，便于映射回原始页面与脚本块号。
    script = (
        "const fs=require('fs'),vm=require('vm');"
        "let failed=false;"
        "for(const f of process.argv.slice(1)){"
        "try{new vm.Script(fs.readFileSync(f,'utf8'),{filename:f});}"
        "catch(e){failed=true;process.stderr.write('FILE:'+f+'\\n'+e.message.replace(/\\s+/g,' ')+'\\n');}"
        "}"
        "process.exit(failed?1:0);"
    )
    proc = subprocess.run(
        ["node", "-e", script, *[str(t) for _, t in targets]],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).replace(str(tmp.resolve()) + "/", "")
        # 按 FILE: 分段，把每个失败文件映射回原始页面与块号
        blocks: dict[str, str] = {}
        current = None
        for line in detail.splitlines():
            if line.startswith("FILE:"):
                current = line[len("FILE:"):].strip()
                blocks[current] = ""
            elif current:
                blocks[current] += (" " if blocks[current] else "") + line.strip()
        for label, target in targets:
            message = blocks.get(str(target))
            if message:
                errors.append(f"{label}: {' '.join(message.split())}")
    return errors, total


def main() -> int:
    if shutil.which("node") is None:
        print("error: node not found in PATH; install Node.js to run this check")
        return 2

    pages, missing = collect_pages(sys.argv[1:] or ["wiki"])
    if missing:
        print("error: path not found: " + ", ".join(missing))
        return 2
    if not pages:
        print("error: no HTML pages found")
        return 2

    with tempfile.TemporaryDirectory() as tmpdir:
        errors, total = check_blocks(pages, Path(tmpdir))

    if errors:
        print(f"inline script check failed: {len(errors)} of {total} blocks")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"inline script check ok: {len(pages)} pages, {total} blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
