#!/usr/bin/env python3
"""数据流页核查：事实对不对，以及连线画得干不干净。

数据流页最容易出的错不是排版，而是**写出源码里根本不存在的东西**。
这个脚本把断言变成可执行的检查：

  1. 源码位置 —— 传入计划文件时，「源码」列指的行不是空行或注释，且含节点名
  2. 节点名  —— 名字里出现的标识符必须在源码里出现；短词表里的词跳过
  3. 形状    —— 写出的权重形状，必须与 safetensors 头一致
  4. 材料    —— --script 指定的文件必须存在（旧页面用来确认构建脚本还在原处）
  5. 几何    —— --geometry 时用无头 Chrome 量连线穿框与重合（见下）
  6. 截图    —— --shots 时逐张图截图，供人看

前三项需要外部材料（源码 / 权重），所以不做进 ci-01-validate.py（那是纯静态
检查），单独跑这个脚本。

用法：
    # 计划文件的源码位置
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/research/dataflow.md \
        --source-root /path/to/transformers

    # 节点名与权重形状
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html \
        --source /path/to/modeling_x.py --shapes wiki/<name>/research/sources/shapes.json

    # 只看节点名（仍要 --source）
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html \
        --source /path/to/modeling_x.py --names-only

    # 几何：连线是否穿框、是否与别的线并排重合
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --geometry

    # 截图
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --shots <目录>

几何判据必须来自渲染后的像素，不能靠几何推算。

退出码：0 全部通过；1 有问题；2 用法错误。
"""

from __future__ import annotations

import argparse
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


# 节点名里允许出现、但不属于「源码标识符」的词：
# 数据类型、形状记号、数学符号、以及句式里的连接词。
ALLOWED_TOKENS = {
    # 数据类型与框架
    "bf16", "fp32", "fp16", "int32", "int64", "bool", "float", "float32", "dtype",
    # 常见的 nn 构造子（页面用 `nn.Embedding` 这种写法时会出现）
    "nn", "Embedding", "Linear", "RMSNorm", "LayerNorm", "Parameter", "Buffer",
    # 形状记号
    "B", "T", "S", "N", "D", "H", "L", "O", "tokens", "scalar", "dict",
    # 源码里作为局部变量出现、但短到容易误报的
    "x", "a", "b", "c", "k", "q", "v", "w", "p", "e", "i", "j", "n", "m",
    # 句式连接词
    "and", "or", "per", "the", "to", "of", "in", "on", "if", "else", "not",
    "None", "True", "False",
}


def load_views(page: Path) -> list[dict]:
    html = page.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"window\.DOJO_FLOW_DATA = (\{.*?\});</script>", html, re.S)
    if not m:
        raise SystemExit(f"error: {page} 里没有 DOJO_FLOW_DATA 数据块")
    return json.loads(m.group(1))["views"]


def idents(text: str) -> list[str]:
    """取出文本里的标识符（字母数字下划线，且不以数字开头）。"""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)


def check_names(views: list[dict], sources: dict[str, str]) -> list[str]:
    """节点名里出现的标识符必须在某份源码里找得到。

    这是防「凭空捏造」的核心检查。实测踩过的坑：节点写了 einsum，
    而源码里一次 einsum 都没有。
    """
    errors: list[str] = []
    for view in views:
        for node in view["nodes"]:
            for tok in idents(node["name"]):
                if tok in ALLOWED_TOKENS:
                    continue
                if any(tok in body for body in sources.values()):
                    continue
                # 允许「检查点键名」这类外部命名：detail 里注明来源即可
                detail = node.get("detail", "")
                if tok in detail and ("检查点" in detail or "键" in detail):
                    continue
                errors.append(
                    f"[{view['id']}/{node['id']}] 节点名 {node['name']!r} 里的 "
                    f"{tok!r} 在源码中找不到"
                )
    return errors


def check_shapes(views: list[dict], shapes: dict[str, list[int]]) -> tuple[list[str], int, list[str]]:
    """页面上写的权重形状必须与给定的形状表一致。

    shapes 由 --shapes 传入（name -> [dim...]），通常来自 safetensors 头。
    返回 (错误, 比对上的键数, 页面上写了形状但不在形状表里的键)。
    """
    errors: list[str] = []
    matched = 0
    missing: list[str] = []
    for view in views:
        for node in view["nodes"]:
            blob = " ".join(filter(None, [node.get("name"), node.get("detail"), node.get("shape")]))
            for m in re.finditer(r"([a-z_](?:[a-z0-9_.]|\{[a-z]+\})*)\s*\[([0-9][0-9,\s]*)\]", blob):
                # 层号、专家号写成 {i}、{e} 时，按第 0 个去形状表里比对
                key = re.sub(r"\{[a-z]+\}", "0", m.group(1))
                if key not in shapes:
                    missing.append(f"{view['id']}/{node['id']}: {key}")
                    continue
                matched += 1
                claimed = [int(x) for x in m.group(2).replace(" ", "").split(",")]
                expected = shapes[key]
                if claimed != expected:
                    errors.append(
                        f"[{view['id']}/{node['id']}] {key} 页面写 {claimed}，"
                        f"材料里是 {expected}"
                    )
    return errors, matched, missing


def check_structure(views: list[dict]) -> list[str]:
    """视图数据自洽：节点 id 不重复，边、分组、批注引用的节点和下钻目标都存在。"""
    errors: list[str] = []
    view_ids = {v["id"] for v in views}
    for v in views:
        ids = [n["id"] for n in v["nodes"]]
        known = set(ids)
        for nid in {i for i in ids if ids.count(i) > 1}:
            errors.append(f"[{v['id']}/{nid}] 节点 id 重复")
        for n in v["nodes"]:
            if n.get("drill") and n["drill"] not in view_ids:
                errors.append(f"[{v['id']}/{n['id']}] 下钻目标 {n['drill']} 不存在")
        for e in v["edges"]:
            for end in (e["from"], e["to"]):
                if end not in known:
                    errors.append(f"[{v['id']}] 边 {e['from']} → {e['to']} 的端点 {end} 不存在")
        for g in v.get("groups") or []:
            for m in g.get("members", []):
                if m not in known:
                    errors.append(f"[{v['id']}] 分组「{g.get('label', '')}」的成员 {m} 不存在")
        for note in v.get("notes") or []:
            if note.get("at") not in known:
                errors.append(f"[{v['id']}] 批注「{note.get('text', '')[:20]}」挂在不存在的节点 {note.get('at')}")
    return errors


def check_scripts_exist(scripts: list[Path]) -> list[str]:
    """--script 指定的构建脚本必须存在。

    早先的名字是 check_line_numbers，承诺校验源码行号；实际没有实现行号
    解析，只是确认文件存在。名字与行为不符会误导使用者，故改名并如实描述。
    """
    errors: list[str] = []
    for script in scripts:
        if not script.exists():
            errors.append(f"脚本不存在: {script}")
    return errors


# ---------- 几何：连线是否穿框、是否与别的线并排重合 ----------

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

GEOM_RESULT_RE = re.compile(r"FLOWGEOM(\{.*?\})FLOWGEOM", re.S)

# 页内探针：逐个视图测量「穿框」与「像素重合」。
#
# 关键点：applyView 会触发 ELK 的**异步**布局，切换后必须等布局完成再量，
# 否则量到的是上一个视图的残留数据（实测九个视图全报同一组数字 = 假通过）。
# 所以这里用递归的 step()：切一个视图 → 等一拍 → 量 → 再切下一个。
GEOM_PROBE = r"""
<script>
(function () {
  function edgesNow() { return [].slice.call(document.querySelectorAll('.flow-edge')); }

  function measureCurrent() {
    var fv = window.__flow;
    var paths = edgesNow();
    if (!paths.length) return null;
    var ps = Object.keys(fv.positions).map(function (k) {
      var p = fv.positions[k]; return { id: k, x: p.x, y: p.y, w: p.w, h: p.h };
    });
    var grids = paths.map(function (p) {
      var L = p.getTotalLength(), s = {};
      for (var d = 0; d <= L; d += 1) {
        var q = p.getPointAtLength(d);
        s[Math.round(q.x) + ',' + Math.round(q.y)] = 1;
      }
      return s;
    });
    var cross = 0;
    fv.view.edges.forEach(function (e, i) {
      var p = paths[i]; if (!p) return;
      var L = p.getTotalLength(), hit = false;
      for (var d = 0; d <= L && !hit; d += 3) {
        var q = p.getPointAtLength(d);
        for (var n = 0; n < ps.length; n++) {
          var b = ps[n];
          if (b.id === e.from || b.id === e.to) continue;
          if (q.x > b.x + 3 && q.x < b.x + b.w - 3 &&
              q.y > b.y + 3 && q.y < b.y + b.h - 3) { hit = true; break; }
        }
      }
      if (hit) cross++;
    });
    var overlap = 0;
    for (var i = 0; i < grids.length; i++) {
      var keys = Object.keys(grids[i]);
      for (var j = i + 1; j < grids.length; j++) {
        var shared = 0;
        for (var ki = 0; ki < keys.length; ki++) if (grids[j][keys[ki]]) shared++;
        if (shared > 60) overlap++;
      }
    }
    return { id: fv.view.id, edges: paths.length, cross: cross, overlap: overlap };
  }

  function finish(payload) {
    document.title = 'FLOWGEOM' + JSON.stringify(payload) + 'FLOWGEOM';
  }

  window.addEventListener('load', function () {
    var fv = window.__flow;
    if (!fv) { finish({ error: 'no __flow' }); return; }
    var ids = fv.data.views.map(function (v) { return v.id; });
    var out = [], idx = 0, waited = 0;
    function step() {
      if (idx >= ids.length) { finish({ views: out }); return; }
      var id = ids[idx];
      if (fv.view.id !== id) {
        fv.needsFit = true;
        fv.applyView(id);
        waited = 0;
        setTimeout(step, 250);
        return;
      }
      waited += 250;
      var paths = edgesNow();
      var expect = fv.view.edges.length;
      // 等布局稳定：边数对上了才认为渲染完成
      if (paths.length < expect && waited < 6000) { setTimeout(step, 250); return; }
      var m = measureCurrent();
      if (m) out.push(m);
      idx++;
      waited = 0;
      setTimeout(step, 250);
    }
    step();
  });
})();
</script>
"""


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    return None


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):        # 静默
        pass


def serve_root(root: Path):
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def run_geometry(page: Path, root: Path) -> int:
    """几何自检：逐视图量连线穿框与像素重合。"""
    chrome = find_chrome()
    if not chrome:
        print("error: 找不到 Chrome/Chromium，无法做几何自检", file=sys.stderr)
        return 2

    if not page.exists():
        print(f"error: 页面不存在: {page}", file=sys.stderr)
        return 2

    # --dump-dom 只吃页面自带的脚本，所以先把探针注入一份临时副本再加载。
    # 页面里的 <script> 都是相对 ../../libs/ 引资源，副本必须放在同一层级，
    # 否则 libs 加载不到、探针拿不到 __flow。
    injected = page.with_name("__geom_probe__.html")
    injected.write_text(
        page.read_text(encoding="utf-8").replace("</body>", GEOM_PROBE + "</body>"),
        encoding="utf-8",
    )

    httpd, port = serve_root(root)
    try:
        rel = injected.relative_to(root)
        dom = subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=20000", "--dump-dom",
             f"http://127.0.0.1:{port}/{rel}"],
            capture_output=True, text=True, timeout=180,
        ).stdout
    finally:
        httpd.shutdown()
        injected.unlink(missing_ok=True)

    m = GEOM_RESULT_RE.search(dom)
    if not m:
        print("error: 页面没有返回几何数据（探针未执行？）", file=sys.stderr)
        return 2
    payload = json.loads(unescape(m.group(1)))
    if payload.get("error"):
        print(f"error: 探针报错 {payload['error']}", file=sys.stderr)
        return 2

    print("几何自检（渲染后逐像素）：")
    for row in payload["views"]:
        flag = "OK " if row["cross"] == 0 and row["overlap"] == 0 else "!! "
        print(f"  {flag}{row['id']:<10} 边={row['edges']:>3}  穿框={row['cross']:>2}  "
              f"像素重合对={row['overlap']:>2}")
    bad = [r for r in payload["views"] if r["cross"] or r["overlap"]]
    if bad:
        print(f"几何自检失败：{len(bad)} 个视图有问题")
        return 1
    print("几何自检通过：全部视图穿框 0、像素重合 0")
    return 0


# 节点第一行照源码里的名字写。这里只给仍可能出现的短名留对照，
# 新的计划文件不再把 hidden_states 改写成 x。
NAME_ALIASES = {
    "x": ("hidden_states", "inputs_embeds", "last_hidden_state"),
    "q": ("query_states", "query"),
    "k": ("key_states", "key"),
    "v": ("value_states", "value"),
}
SKIP_HEADS = {"资料", "前提", "config.json"}


def _plan_rows(text: str):
    """逐张图产出 (标题, 行号, {列名: 值})，只读「节点」表。"""
    head = sub = None
    cols: list[str] = []
    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("## "):
            head, sub, cols = s[3:].strip(), None, []
            yield head, no, None
        elif s.startswith("### "):
            sub, cols = s[4:].strip(), []
        elif head not in SKIP_HEADS and sub == "节点" and s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not cols:
                cols = cells
            elif not set(s) <= set("|-: "):
                yield head, no, dict(zip(cols, cells))


def _refs(cell: str) -> list[tuple[str, int, int]]:
    """「a.py:1-3，5；b.py:7」→ [(a.py, 1, 3), (a.py, 5, 5), (b.py, 7, 7)]。"""
    out: list[tuple[str, int, int]] = []
    for seg in re.split(r"[；;]", cell):
        fname = ""
        for piece in re.split(r"[，,]", seg):
            piece = piece.strip()
            m = re.fullmatch(r"(?:(\S+?):)?(\d+)(?:-(\d+))?", piece)
            if not m:
                continue
            fname = m.group(1) or fname
            a = int(m.group(2))
            out.append((fname, a, int(m.group(3) or a)))
    return out


def check_plan_sources(plan: Path, root: Path) -> tuple[list[str], int]:
    """计划文件的源码位置回查：标题从 def forward 起，节点指的行不是空行或注释，且含节点名。"""
    files: dict[str, list[str]] = {}

    def lines_of(name: str) -> list[str] | None:
        if name not in files:
            hits = [root / name] if "/" in name else list(root.rglob(name))
            hits = [h for h in hits if h.is_file()]
            files[name] = hits[0].read_text(encoding="utf-8").splitlines() if len(hits) == 1 else None
        return files[name]

    errors: list[str] = []
    checked = 0
    for head, no, row in _plan_rows(plan.read_text(encoding="utf-8")):
        if head in SKIP_HEADS:
            continue
        where = f"第 {no} 行（{head.split('（')[0]}"
        if row is None:
            m = re.search(r"[（(](\S+?):(\d+)-\d+[）)]", head)
            src = m and lines_of(m.group(1))
            if m and src is not None and "def forward" not in src[int(m.group(2)) - 1]:
                errors.append(f"{where}）标题的起始行 {m.group(1)}:{m.group(2)} 不是 def forward")
            continue
        refs = _refs(row.get("源码", ""))
        if not refs:
            continue
        checked += 1
        where += f" / {row.get('id', '?')}）"
        text = []
        for fname, a, b in refs:
            src = lines_of(fname)
            if src is None:
                errors.append(f"{where}找不到源码文件 {fname}（或同名文件不止一个，写上相对 --source-root 的路径）")
                break
            if a > len(src) or not src[a - 1].strip() or src[a - 1].strip().startswith("#"):
                errors.append(f"{where}{fname}:{a} 是空行、注释或超出文件")
            text += [l.split("#")[0] for l in src[a - 1:b]]
        else:
            names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", row.get("第一行", "").replace("`", "")))
            words = set()
            for n in names:
                words |= set(NAME_ALIASES.get(n, ())) | {n}
                words |= {w for w in re.findall(r"[A-Z]?[a-z]+", n) if len(w) >= 3}
            blob = "\n".join(text).lower()
            if names and not any(w.lower() in blob for w in words):
                errors.append(f"{where}源码 {row.get('源码')} 里找不到第一行的名字")
    return errors, checked


def run_shots(page: Path, root: Path, out: Path) -> int:
    """逐张图截图：<视图>.png 是画布按宽度铺满，<视图>-info.png 是打开说明面板。

    file:// 打不开页面里的相对资源，要起本地服务；#视图 在无头模式下不生效，
    所以每张图注入一段脚本直接切过去。
    """
    chrome = find_chrome()
    if not chrome:
        print("error: 找不到 Chrome/Chromium，无法截图", file=sys.stderr)
        return 2
    ids = [v["id"] for v in load_views(page)]
    out.mkdir(parents=True, exist_ok=True)
    injected = page.with_name("__shot__.html")
    html = page.read_text(encoding="utf-8")
    httpd, port = serve_root(root)
    try:
        for vid in ids:
            for suffix, extra in (("", ""), ("-info", "var b=document.getElementById('flow-info-btn');if(b&&!b.hidden)b.click();")):
                probe = ("<script>window.addEventListener('load',function(){setTimeout(function(){"
                         f"window.__flow.fitMode='width';window.__flow.applyView('{vid}');{extra}"
                         "},400);});</script>")
                injected.write_text(html.replace("</body>", probe + "</body>"), encoding="utf-8")
                shot = out / f"{vid}{suffix}.png"
                subprocess.run(
                    [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                     "--window-size=1440,900", "--virtual-time-budget=9000",
                     f"--screenshot={shot.resolve()}",
                     f"http://127.0.0.1:{port}/{injected.relative_to(root)}"],
                    capture_output=True, timeout=120,
                )
                print(f"  {shot}")
    finally:
        httpd.shutdown()
        injected.unlink(missing_ok=True)
    print(f"截图完成：{len(ids)} 张图，每张两幅（画布、说明面板）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="数据流页事实核查")
    ap.add_argument("page", type=Path, help="wiki/<name>/index.html")
    ap.add_argument("--source", action="append", default=[], metavar="[LABEL=]PATH",
                    help="权威源码，可多次传入。带 LABEL 时（如 mtp=/path/mtp.py）"
                         "只有视图标题里含该 LABEL 才用它做校验，这样每个视图可以"
                         "有自己的权威来源（例如 MTP 视图的事实来源是 vLLM 的 mtp.py，"
                         "而不是 transformers）")
    ap.add_argument("--shapes", type=Path,
                    help="形状表 JSON：{tensor_key: [dim...]}，通常来自 safetensors 头")
    ap.add_argument("--script", action="append", default=[], type=Path,
                    help="构建脚本路径，确认其存在（可多次传入）")
    ap.add_argument("--names-only", action="store_true",
                    help="只做节点名核查")
    ap.add_argument("--geometry", action="store_true",
                    help="几何自检：无头 Chrome 量连线穿框与重合")
    ap.add_argument("--shots", type=Path, metavar="DIR",
                    help="逐张图截图，存到 DIR")
    ap.add_argument("--source-root", type=Path, metavar="DIR",
                    help="传入计划文件 dataflow.md 时，按「源码」列到这个目录下找文件回查")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="--geometry、--shots 时的站点根目录，默认当前目录")
    args = ap.parse_args()

    if not args.page.exists():
        print(f"error: 页面不存在: {args.page}", file=sys.stderr)
        return 2

    if args.page.suffix == ".md":
        if not args.source_root or not args.source_root.is_dir():
            print("error: 检查计划文件要用 --source-root 指定源码目录", file=sys.stderr)
            return 2
        errors, checked = check_plan_sources(args.page, args.source_root)
        if errors:
            print(f"源码位置核查失败：{len(errors)} 处")
            for e in errors:
                print(f"  - {e}")
            return 1
        print(f"源码位置核查通过：{checked} 个节点")
        return 0

    # 形状表缺失时直接报用法错误。让它抛 FileNotFoundError 会打出一串堆栈，
    # 使用者容易把「脚本崩了」误读成「核查过了」。
    if args.shapes and not args.shapes.exists():
        print(f"error: 形状表不存在: {args.shapes}", file=sys.stderr)
        return 2

    if args.geometry or args.shots:
        # 页面路径相对站点根解析，二者都取绝对路径，relative_to 才能用
        root = args.root.resolve()
        page = args.page if args.page.is_absolute() else (root / args.page)
        if args.shots:
            return run_shots(page, root, args.shots)
        return run_geometry(page, root)

    views = load_views(args.page)

    # 解析 --source：LABEL=PATH 或纯 PATH（纯 PATH 对所有视图生效）
    sources: dict[str, str] = {}
    labeled: list[tuple[str, str]] = []     # (label, body)
    labeled_files: set[str] = set()          # 已被某个 label 指派的文件
    for raw in args.source:
        label, _, path = str(raw).partition("=")
        if not _:
            label, path = "", label
        p = Path(path)
        if not p.exists():
            print(f"error: 源码不存在: {p}", file=sys.stderr)
            return 2
        body = p.read_text(encoding="utf-8", errors="ignore")
        sources[path] = body
        if label:
            labeled.append((label, body))
            labeled_files.add(path)

    if args.names_only and not sources:
        print("error: --names-only 仍需 --source 才能比对", file=sys.stderr)
        return 2

    # 选源规则：
    #   · 打了标签的源（LABEL=PATH）只作用于「标题或标签里含该 LABEL」的视图；
    #     这样 MTP 视图可以用 vLLM 的 mtp.py 校验，而其余视图用 transformers。
    #   · 没打标签的源对所有视图生效。
    labeled_bodies = {label: body for label, body in labeled}
    universal = [body for path, body in sources.items() if path not in labeled_files]

    errors: list[str] = check_structure(views)
    if sources:
        for view in views:
            heading = view.get("title", "") + " " + view.get("label", "")
            low = heading.lower()
            picked = [body for label, body in labeled_bodies.items() if label.lower() in low]
            if not picked:
                picked = universal
            if picked:
                errors += check_names([view], {"view": "\n".join(picked)})
    notes: list[str] = []
    shape_matched = 0
    if not args.names_only and args.shapes:
        # 形状表读不出来就报错退出，不要带着半截数据继续核查——
        # 那会给出「核查通过」，而形状其实一条都没比对。
        try:
            shapes = json.loads(args.shapes.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"error: 形状表无法解析: {args.shapes}（{error}）", file=sys.stderr)
            return 2
        shape_errors, shape_matched, shape_missing = check_shapes(views, shapes)
        errors += shape_errors
        if shape_matched == 0:
            errors.append("形状表一个键都没比对上：页面里要写完整键名，如 model.layers.0.mlp.gate_proj.weight [..]")
        for key in shape_missing:
            notes.append(f"形状没比对（不在形状表里）：{key}")
    if not args.names_only and args.script:
        errors += check_scripts_exist([Path(p) for p in args.script])

    for note in notes:
        print(f"  提示：{note}")

    nodes = sum(len(v["nodes"]) for v in views)
    if errors:
        print(f"核查失败：{len(errors)} 个问题（共 {len(views)} 视图 / {nodes} 节点）")
        for e in errors:
            print(f"  - {e}")
        return 1

    checked = []
    if sources:
        checked.append(f"节点名 vs {len(sources)} 份源码")
    if not args.names_only and args.shapes:
        checked.append(f"形状 vs 材料 {shape_matched} 个键")
    print(f"核查通过：{len(views)} 视图 / {nodes} 节点" + (
        "（" + "、".join(checked) + "）" if checked else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
