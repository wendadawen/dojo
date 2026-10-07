#!/usr/bin/env python3
"""检查页面里内联 <script> 的语法。需要 node，缺失时跳过并提示。

    python3 .dojo/scripts/site/check_script.py <页面...>
    python3 .dojo/scripts/site/check_script.py --selftest

只查语法，不执行代码。所有脚本块在一个 node 进程里用 vm.Script 逐个编译。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

# 浏览器会当作 JavaScript 执行的 type 取值
JS_TYPES = frozenset({
    "application/ecmascript", "application/javascript", "application/x-ecmascript",
    "application/x-javascript", "text/ecmascript", "text/javascript", "text/jscript",
    "text/livescript", "text/x-ecmascript", "text/x-javascript", "module",
})
# 老式 "<!-- ... //-->" 包裹，去掉注释标记后再交给 node
LEGACY_OPEN_RE = re.compile(r"^\s*<!--")
LEGACY_CLOSE_RE = re.compile(r"//-->\s*$")
NODE_CHECK = (
    "const fs=require('fs'),vm=require('vm');let failed=false;"
    "for(const f of process.argv.slice(1)){"
    "try{new vm.Script(fs.readFileSync(f,'utf8'),{filename:f});}"
    "catch(e){failed=true;process.stderr.write('FILE:'+f+'\\n'+e.message.replace(/\\s+/g,' ')+'\\n');}}"
    "process.exit(failed?1:0);"
)


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
        if "src" in self._attrs:
            return
        kind = self._attrs.get("type", "").split(";")[0].strip().lower()
        if kind and kind not in JS_TYPES:
            return
        body = LEGACY_CLOSE_RE.sub("", LEGACY_OPEN_RE.sub("", "".join(self._buffer)))
        if body.strip():
            self.blocks.append(body)


def check(pages: list[Path]) -> tuple[list[str], int, bool]:
    """返回 (问题, 脚本块数, node 是否可用)。"""
    if shutil.which("node") is None:
        return [], 0, False
    errors: list[str] = []
    total = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        targets: list[tuple[str, Path]] = []
        for page in pages:
            collector = Collector()
            collector.feed(page.read_text(encoding="utf-8", errors="replace"))
            for index, body in enumerate(collector.blocks, start=1):
                total += 1
                target = tmp / f"{page.parent.name}-{page.stem}-{index}.js"
                target.write_text(body, encoding="utf-8")
                targets.append((f"{page} 内联 <script> #{index}", target))
        if not targets:
            return errors, total, True
        proc = subprocess.run(["node", "-e", NODE_CHECK, *[str(t) for _, t in targets]],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).replace(str(tmp.resolve()) + "/", "")
            messages: dict[str, str] = {}
            current = None
            for line in detail.splitlines():
                if line.startswith("FILE:"):
                    current = line[len("FILE:"):].strip()
                    messages[current] = ""
                elif current:
                    messages[current] += (" " if messages[current] else "") + line.strip()
            for label, target in targets:
                if messages.get(str(target)):
                    errors.append(f"{label}：{' '.join(messages[str(target)].split())}")
    return errors, total, True


# ---------- 自检 ----------

CASES = [
    ("正确的脚本没有报错", "<script>var x = 1;</script>", None),
    ("括号不成对", "<script>var x = (1;</script>", "内联 <script> #1"),
    ("引号不成对", "<script>var s = 'a;</script>", "内联 <script> #1"),
    ("外链脚本不查", '<script src="a.js"></script>', None),
    ("type=text/plain 的块不查", '<script type="text/plain">这不是 js (</script>', None),
    ("老式注释包裹", "<script><!--\nvar x = 1;\n//--></script>", None),
]


def selftest() -> int:
    if shutil.which("node") is None:
        print("没有 node，跳过脚本检查自检")
        return 0
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "index.html"
        for name, html, want in CASES:
            page.write_text(f"<html><body>{html}</body></html>", encoding="utf-8")
            errs, _, _ = check([page])
            ok = (not errs) if want is None else any(want in e for e in errs)
            if want is None and errs:
                print("    意外报错：", errs)
            results.append((name, ok))
    for name, ok in results:
        print(("  OK  " if ok else "  FAIL") + " " + name)
    failed = sum(1 for _, ok in results if not ok)
    print(f"脚本检查自检失败：{failed} 项" if failed else f"脚本检查自检通过：{len(results)} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selftest" in args:
        raise SystemExit(selftest())
    if not args:
        print(__doc__)
        raise SystemExit(2)
    paths = [Path(a) for a in args]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("错误：页面不存在：" + "、".join(missing))
    problems, total, ran = check(paths)
    if not ran:
        raise SystemExit("错误：PATH 里没有 node，这项检查需要安装 Node.js")
    for p in problems:
        print(f"- {p}")
    print(f"内联脚本检查失败：{total} 块里有 {len(problems)} 块有问题" if problems
          else f"内联脚本检查通过：{len(paths)} 个页面，{total} 块")
    raise SystemExit(1 if problems else 0)
