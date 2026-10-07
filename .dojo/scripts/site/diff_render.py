#!/usr/bin/env python3
"""改共享样式前后，比较页面渲染有没有变。需要 Chrome。

    python3 .dojo/scripts/site/diff_render.py <before-root> <after-root> <页面相对路径...> [--detail] [--jobs N]

before-root、after-root 是两份仓库根目录（例如改样式前后各一份）。每个页面在两边各渲染一次，
逐元素比较计算样式和几何，不同就报出。--detail 指出第一处差异是哪个 CSS 属性变了。
"""

from __future__ import annotations

import concurrent.futures
import functools
import http.server
import json
import re
import socketserver
import subprocess
import sys
import threading
from html import unescape
from pathlib import Path

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

# 逐元素取这些计算样式与几何，两侧不同即报出。
PROPS = (
    "display,position,top,right,bottom,left,float,clear,zIndex,"
    "width,height,minWidth,minHeight,maxWidth,maxHeight,"
    "marginTop,marginRight,marginBottom,marginLeft,"
    "paddingTop,paddingRight,paddingBottom,paddingLeft,"
    "borderTopWidth,borderRightWidth,borderBottomWidth,borderLeftWidth,"
    "borderTopStyle,borderRightStyle,borderBottomStyle,borderLeftStyle,"
    "borderTopColor,borderRightColor,borderBottomColor,borderLeftColor,"
    "borderTopLeftRadius,borderTopRightRadius,borderBottomLeftRadius,borderBottomRightRadius,"
    "background,backgroundColor,backgroundImage,backgroundSize,backgroundPosition,"
    "color,opacity,visibility,overflow,overflowX,overflowY,"
    "fontFamily,fontSize,fontWeight,fontStyle,lineHeight,letterSpacing,"
    "textAlign,textDecorationLine,textTransform,whiteSpace,wordBreak,textIndent,"
    "verticalAlign,listStyleType,listStylePosition,"
    "flexDirection,flexWrap,justifyContent,alignItems,alignSelf,flexGrow,flexShrink,flexBasis,gap,"
    "gridTemplateColumns,gridTemplateRows,gridColumn,gridRow,"
    "transform,transformOrigin,boxShadow,boxSizing,cursor"
).split(",")

PROBE = """
<script>
window.addEventListener('load', function () {
  // 等一拍再测：KaTeX 的 auto-render 在 defer 脚本的 onload 里跑，与 window
  // load 几乎同时，立刻取字形度量会抖动。用 setTimeout 而不是 document.fonts
  // .ready——后者在 --virtual-time-budget 下不兑现，探针会永远不执行。
  setTimeout(function () {
    var PROPS = %s;
    function pathOf(el) {
      var parts = [];
      var node = el;
      while (node && node.nodeType === 1) {
        var index = 0, sib = node;
        while ((sib = sib.previousElementSibling)) index++;
        parts.unshift(node.tagName + '[' + index + ']');
        node = node.parentElement;
      }
      return parts.join('/');
    }
    var SKIP = {HEAD:1, STYLE:1, LINK:1, SCRIPT:1, META:1, TITLE:1, BASE:1};
    var rows = [];
    var nodes = document.querySelectorAll('*');
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (SKIP[el.tagName]) continue;
      var cs = getComputedStyle(el);
      var style = '';
      for (var j = 0; j < PROPS.length; j++) style += PROPS[j] + ':' + cs[PROPS[j]] + ';';
      var r = el.getBoundingClientRect();
      var box = [r.x, r.y, r.width, r.height].map(function (v) {
        return Math.round(v * 100) / 100;
      }).join(',');
      rows.push(pathOf(el) + '|' + box + '|' + style);
    }
    var payload = JSON.stringify({count: nodes.length, rows: rows});
    var h = 5381;
    for (var k = 0; k < payload.length; k++) h = ((h * 33) ^ payload.charCodeAt(k)) >>> 0;
    document.title = 'RD:' + nodes.length + ':' + h;
    document.documentElement.setAttribute('data-rd-rows', payload);
  }, 400);
});
</script>
"""
TITLE_RE = re.compile(r"<title>RD:(\d+):(\d+)</title>")
ROWS_RE = re.compile(r'data-rd-rows="(.*?)"\s*>', re.S)


def find_chrome() -> str | None:
    return next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)


def probe_handler(probe: str):
    """在响应的 HTML 里注入探针。走 HTTP 而不是 file://：file:// 下外链 CSS 不会被加载，
    验证共享样式时会得到假差异。"""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - 基类命名
            target = Path(self.translate_path(self.path))
            if target.is_dir():
                target = target / "index.html"
            if target.suffix == ".html" and target.exists():
                body = target.read_text(encoding="utf-8", errors="replace")
                body = body.replace("<head>", "<head>" + probe, 1) if "<head>" in body else probe + body
                payload = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def log_message(self, *args):  # 静默
            pass

    return Handler


def serve(root: Path):
    handler = functools.partial(probe_handler(PROBE % json.dumps(PROPS)), directory=str(root))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def snapshot(chrome: str, url: str, want_rows: bool = False):
    """返回 (元素数, 计算样式指纹)；want_rows 时额外返回逐元素明细。"""
    try:
        proc = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                               "--virtual-time-budget=8000", "--dump-dom", url],
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    match = TITLE_RE.search(proc.stdout)
    if not match:
        return None
    digest = (int(match.group(1)), int(match.group(2)))
    if not want_rows:
        return digest
    rows_match = ROWS_RE.search(proc.stdout)
    rows = json.loads(unescape(rows_match.group(1)))["rows"] if rows_match else []
    return digest + (rows,)


def explain(before_rows, after_rows, limit=5) -> list[str]:
    """逐元素找出第一处差异，并指出是哪个 CSS 属性变了。"""
    lines = []
    for index, (left, right) in enumerate(zip(before_rows, after_rows)):
        if left == right:
            continue
        lpath, lbox, lstyle = left.split("|", 2)
        rpath, rbox, rstyle = right.split("|", 2)
        lines.append(f"  第 {index} 个元素")
        if lpath != rpath:
            lines.append(f"    路径  {lpath} -> {rpath}")
        if lbox != rbox:
            lines.append(f"    位置  {lbox} -> {rbox}")
        lprops = dict(kv.split(":", 1) for kv in lstyle.split(";") if ":" in kv)
        rprops = dict(kv.split(":", 1) for kv in rstyle.split(";") if ":" in kv)
        for key in lprops:
            if lprops.get(key) != rprops.get(key):
                lines.append(f"    {key}: {lprops.get(key)!r} -> {rprops.get(key)!r}")
        if len(lines) >= limit * 8:
            break
    if len(before_rows) != len(after_rows):
        lines.append(f"  元素数 {len(before_rows)} -> {len(after_rows)}")
    return lines


def main(args: list[str]) -> int:
    detail = "--detail" in args
    jobs = 1
    if "--jobs" in args:
        index = args.index("--jobs")
        jobs = max(1, int(args[index + 1]))
        del args[index:index + 2]
    args = [a for a in args if a != "--detail"]
    if len(args) < 3:
        print(__doc__)
        return 2
    before_root, after_root, pages = Path(args[0]), Path(args[1]), args[2:]
    chrome = find_chrome()
    if chrome is None:
        print("错误：未找到 Chrome/Chromium", file=sys.stderr)
        return 2

    before_server, before_port = serve(before_root)
    after_server, after_port = serve(after_root)

    def compare(rel: str):
        if not (before_root / rel).exists() or not (after_root / rel).exists():
            return f"ERR  {rel}  （一侧不存在）"
        before = snapshot(chrome, f"http://127.0.0.1:{before_port}/{rel}", detail)
        after = snapshot(chrome, f"http://127.0.0.1:{after_port}/{rel}", detail)
        if before is None or after is None:
            return f"ERR  {rel}  （页面没有回报，检查探针）"
        if before[:2] == after[:2]:
            return f"ok   {rel}  （{before[0]} 个元素）"
        lines = [f"DIFF {rel}  before={before[:2]} after={after[:2]}"]
        if detail and len(before) > 2 and len(after) > 2:
            lines.extend(explain(before[2], after[2]))
        return "\n".join(lines)

    failed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            for line in pool.map(compare, pages):
                print(line, flush=True)
                if line.startswith(("DIFF", "ERR")):
                    failed += 1
    finally:
        before_server.shutdown()
        after_server.shutdown()
    print(f"\n{len(pages) - failed}/{len(pages)} 个页面渲染一致", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
