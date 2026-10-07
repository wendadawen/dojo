"""样式检查：死 CSS 检测，以及改共享样式前后的渲染比对。--confirm 和 --diff 需要 Chrome。

由 validate.py 调度。
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
import tempfile
import threading
from html import unescape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from page import STYLE_RE, TYPE_STYLESHEET  # noqa: E402


# ============================================================
# 样式：死规则检测，以及改样式前后的渲染比对
# ============================================================
SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.S | re.I)
CLASS_ATTR_RE = re.compile(r"""\bclass\s*=\s*["']([^"']*)["']""", re.I)
IDENT_RE = re.compile(r"[A-Za-z_][\w-]*")
CLASS_TOKEN_RE = re.compile(r"\.(-?[A-Za-z_][\w-]*)")
COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)

# 库在运行时注入的 class，静态标记里查不到，必须视为已用
RUNTIME_PREFIXES = ("katex", "token", "language-", "prism", "line-numbers")

# --confirm 用的无头浏览器
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)
CLASS_PROBE = """<script>
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
CLASS_TITLE_RE = re.compile(r"<title>CLASSES:(.*?)</title>", re.S)


def harvest(document: str) -> set[str]:
    """从一份文档里收集可能被应用到的 class 名。"""
    names: set[str] = set()
    for value in CLASS_ATTR_RE.findall(document):
        names.update(value.split())
    for script in SCRIPT_RE.findall(document):
        names.update(IDENT_RE.findall(script))
    return names


def build_usage(targets: list[Path], extra: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in list(targets) + list(extra):
        names |= harvest(path.read_text(encoding="utf-8"))
    return names


def iter_rules(css: str):
    """遍历 CSS 规则，产出 (选择器, 起始偏移, 结束偏移)。"""
    text = COMMENT_RE.sub(lambda m: " " * len(m.group(0)), css)
    stack: list[tuple[str, int]] = []
    start = 0
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            stack.append((text[start:cursor].strip(), cursor))
        elif char == "}":
            if stack:
                selector, _ = stack.pop()
                if selector and not selector.startswith("@"):
                    yield selector, start, cursor + 1
            start = cursor + 1
        elif char == ";" and not stack:
            start = cursor + 1
        cursor += 1


def selector_is_dead(selector: str, used: set[str]) -> bool:
    """判断不了的一律返回 False（保留）。"""
    if "[" in selector or ":not(" in selector or "(" in selector or "::" in selector:
        return False
    tokens = CLASS_TOKEN_RE.findall(selector)
    if not tokens:
        return False
    return any(
        token not in used
        and not token.startswith(RUNTIME_PREFIXES)
        for token in tokens
    )


def split_selectors(selector: str) -> list[str]:
    return [part.strip() for part in selector.split(",") if part.strip()]


def process(path: Path, usage: set[str], fix: bool) -> list[str]:
    document = path.read_text(encoding="utf-8")
    removed: list[str] = []
    for block in reversed(list(STYLE_RE.finditer(document))):
        css = block.group(1)
        spans: list[tuple[int, int, str]] = []
        for selector, local_start, local_end in iter_rules(css):
            parts = split_selectors(selector)
            if parts and all(selector_is_dead(part, usage) for part in parts):
                spans.append((local_start, local_end, selector.strip()))
        if not spans:
            continue
        removed.extend(s for _, _, s in spans)
        if fix:
            new_css = css
            for local_start, local_end, _ in reversed(spans):
                new_css = new_css[:local_start] + new_css[local_end:]
            new_css = re.sub(r"\n{3,}", "\n\n", new_css).rstrip() + "\n"
            document = document[: block.start(1)] + new_css + document[block.end(1) :]
    if fix and removed:
        path.write_text(document, encoding="utf-8")
    return removed


def pages_of_type(root: Path, kind: str) -> list[Path]:
    """wiki/ 下所有 dojo:type 等于 kind 的页面。"""
    pattern = re.compile(r'name="dojo:type"\s+content="' + re.escape(kind) + r'"')
    return [
        path
        for path in sorted(root.glob("wiki/*/index.html"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def live_classes(chrome: str, page: Path) -> set[str]:
    """把页面跑起来，返回脚本执行完之后 DOM 上真实存在的 class 名。"""
    document = page.read_text(encoding="utf-8", errors="replace")
    if "<head>" in document:
        document = document.replace("<head>", "<head>" + CLASS_PROBE, 1)
    else:
        document = CLASS_PROBE + document
    with tempfile.TemporaryDirectory() as tmp:
        probe_page = Path(tmp) / page.name
        probe_page.write_text(document, encoding="utf-8")
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
    match = CLASS_TITLE_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"probe did not report on {page}")
    return set(json.loads(match.group(1)))


def confirm_removals(candidates: set[str], pages: list[Path]) -> int:
    """在真实浏览器里复核候选 class 确实不再出现。

    返回 0 表示全部确认可删；1 表示仍有候选在用（不能删）；2 表示环境错误。
    """
    chrome = find_chrome()
    if chrome is None:
        print("错误：--confirm 需要 Chrome/Chromium，未找到", file=sys.stderr)
        return 2

    live: set[str] = set()
    for page in pages:
        try:
            live |= live_classes(chrome, page)
        except Exception as error:  # noqa: BLE001 - 报告后返回
            print(f"错误：{error}", file=sys.stderr)
            return 2

    survivors = sorted(c for c in candidates if c in live)
    if survivors:
        print(f"{len(survivors)} class 在浏览器中仍存在，不能删：")
        for name in survivors:
            print(f"  ! .{name}")
        return 1
    print(f"复核通过：{len(candidates)} 个候选 class 在 {len(pages)} 个页面上均未出现")
    return 0


# ---------- 改样式前后的渲染比对 ----------

# 逐元素取这些计算样式与几何，两侧不同即报出。
DIFF_PROPS = (
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

DIFF_PROBE = """
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

DIFF_TITLE_RE = re.compile(r"<title>RD:(\d+):(\d+)</title>")
DIFF_ROWS_RE = re.compile(r'data-rd-rows="(.*?)"\s*>', re.S)


def make_probe_handler(probe: str):
    """在响应的 HTML 里注入探针。走 HTTP 而不是 file://——file:// 下外链 CSS
    不会被加载，验证共享样式时会得到假差异。"""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - 基类命名
            path = self.translate_path(self.path)
            target = Path(path)
            if target.is_dir():
                target = target / "index.html"
            if target.suffix == ".html" and target.exists():
                body = target.read_text(encoding="utf-8", errors="replace")
                body = (
                    body.replace("<head>", "<head>" + probe, 1)
                    if "<head>" in body else probe + body
                )
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


def serve_tree(root: Path):
    handler = functools.partial(
        make_probe_handler(DIFF_PROBE % json.dumps(DIFF_PROPS)), directory=str(root)
    )
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def snapshot(chrome: str, url: str, want_rows: bool = False):
    """返回 (元素数, 计算样式指纹)；want_rows 时额外返回逐元素明细。"""
    try:
        proc = subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=8000", "--dump-dom", url],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    match = DIFF_TITLE_RE.search(proc.stdout)
    if not match:
        return None
    digest = (int(match.group(1)), int(match.group(2)))
    if not want_rows:
        return digest
    rows_match = DIFF_ROWS_RE.search(proc.stdout)
    rows = json.loads(unescape(rows_match.group(1)))["rows"] if rows_match else []
    return digest + (rows,)


def explain_diff(before_rows, after_rows, limit=5) -> list[str]:
    """逐元素找出第一处差异，并指出是哪个 CSS 属性变了。"""
    lines = []
    for index, (left, right) in enumerate(zip(before_rows, after_rows)):
        if left == right:
            continue
        lpath, lbox, lstyle = left.split("|", 2)
        rpath, rbox, rstyle = right.split("|", 2)
        lines.append(f"  element #{index}")
        if lpath != rpath:
            lines.append(f"    path  {lpath} -> {rpath}")
        if lbox != rbox:
            lines.append(f"    box   {lbox} -> {rbox}")
        lprops = dict(kv.split(":", 1) for kv in lstyle.split(";") if ":" in kv)
        rprops = dict(kv.split(":", 1) for kv in rstyle.split(";") if ":" in kv)
        for key in lprops:
            if lprops.get(key) != rprops.get(key):
                lines.append(f"    {key}: {lprops.get(key)!r} -> {rprops.get(key)!r}")
        if len(lines) >= limit * 8:
            break
    if len(before_rows) != len(after_rows):
        lines.append(f"  element count {len(before_rows)} -> {len(after_rows)}")
    return lines


def run_render_diff(args: list[str]) -> int:
    """--diff 子命令：<before-root> <after-root> <页面...> [--detail] [--jobs N]"""
    detail = "--detail" in args
    jobs = 1
    if "--jobs" in args:
        index = args.index("--jobs")
        jobs = max(1, int(args[index + 1]))
        del args[index : index + 2]
    args = [a for a in args if a != "--detail"]
    if len(args) < 3:
        print("用法：--diff <before-root> <after-root> <相对路径...> [--detail] [--jobs N]")
        return 2
    before_root, after_root = Path(args[0]), Path(args[1])
    pages = args[2:]

    chrome = find_chrome()
    if chrome is None:
        print("错误：未找到 Chrome/Chromium", file=sys.stderr)
        return 2

    before_server, before_port = serve_tree(before_root)
    after_server, after_port = serve_tree(after_root)

    def check(rel: str):
        before_file = before_root / rel
        after_file = after_root / rel
        if not before_file.exists() or not after_file.exists():
            return f"ERR  {rel}  （一侧不存在）"
        before = snapshot(chrome, f"http://127.0.0.1:{before_port}/{rel}", detail)
        after = snapshot(chrome, f"http://127.0.0.1:{after_port}/{rel}", detail)
        if before is None or after is None:
            return f"ERR  {rel}  （页面没有回报，检查探针）"
        if before[:2] == after[:2]:
            return f"ok   {rel}  （{before[0]} 个元素）"
        lines = [f"DIFF {rel}  before={before[:2]} after={after[:2]}"]
        if detail and len(before) > 2 and len(after) > 2:
            lines.extend(explain_diff(before[2], after[2]))
        return "\n".join(lines)

    failed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            for line in pool.map(check, pages):
                print(line, flush=True)
                if line.startswith(("DIFF", "ERR")):
                    failed += 1
    finally:
        before_server.shutdown()
        after_server.shutdown()

    print(f"\n{len(pages) - failed}/{len(pages)} 个页面渲染一致", flush=True)
    return 1 if failed else 0




def run_css_check(argv: list[str]) -> int:
    """--css：对每个有样式表的页面类型（现行只有 concept）跑死规则检测。

    --confirm 在浏览器里复核候选；--fix 直接删。
    """
    fix = "--fix" in argv
    confirm = "--confirm" in argv

    total_removed = 0
    for name in TYPE_STYLESHEET:
        tpl = Path(f".dojo/templates/{name}/index.html")
        # 模板是这一类页面样式的权威副本，缺了就无从检查；静默跳过会让
        # 模板被误删时无人发觉，所以这里直接报错。
        if not tpl.exists():
            print(f"错误：模板不存在：{tpl}", file=sys.stderr)
            return 2
        pages = pages_of_type(Path("."), name)
        if not pages:
            print(f"错误：没有 dojo:type={name} 的页面", file=sys.stderr)
            return 2
        targets = [tpl] + pages
        usage = build_usage(targets, pages)
        candidates: set[str] = set()
        removed_here = 0
        for path in targets:
            removed = process(path, usage, fix)
            removed_here += len(removed)
            candidates.update(CLASS_TOKEN_RE.findall("\n".join(removed)))
            if removed:
                print(f"{path}：{len(removed)} 条规则")
                for selector in removed:
                    print(f"  - {selector}")
        if confirm and candidates:
            # 复核在真实页面里做：模板含占位符、相对路径也解析不到，
            # 浏览器跑模板没有意义。
            status = confirm_removals(candidates, pages)
            if status != 0:
                return status
        total_removed += removed_here
        if not removed_here:
            print(f"{len(targets)} 个文件里没有死 CSS")

    if total_removed:
        print(f"\n{'已清掉' if fix else '发现'} {total_removed} 条死规则")
        return 0 if fix else 1
    return 0


