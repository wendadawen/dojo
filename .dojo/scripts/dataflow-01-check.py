#!/usr/bin/env python3
"""数据流页核查：事实对不对，以及连线画得干不干净。

数据流页最容易出的错不是排版，而是**写出源码里根本不存在的东西**。
这个脚本把断言变成可执行的检查：

  1. 节点名  —— 名字里出现的标识符，必须在源码里 grep 得到
  2. 形状    —— 写出的权重形状，必须与 safetensors 头一致
  3. 材料    —— --script 指定的文件必须存在（用于确认构建脚本还在原处）
  4. 几何    —— --geometry 时用无头 Chrome 量连线穿框与重合（见下）

前三项需要外部材料（源码 / 权重），所以不做进 ci-01-validate.py（那是纯静态
检查），单独跑这个脚本。

用法：
    # 源码在本地
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html \
        --source /path/to/modeling_x.py

    # 只看节点名（不需要外部材料）
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --names-only

    # 几何：连线是否穿框、是否与别的线并排重合
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --geometry

几何判据必须来自渲染后的像素，不能靠几何推算：曾用「两条线 x 差 < 3px」
判定并行，报 0 处；实际差 2px、重叠 318px，肉眼一看就是重复的线。

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


def check_shapes(views: list[dict], shapes: dict[str, list[int]]) -> list[str]:
    """页面上写的权重形状必须与给定的形状表一致。

    shapes 由 --shapes 传入（name -> [dim...]），通常来自 safetensors 头。
    """
    errors: list[str] = []
    if not shapes:
        return errors
    for view in views:
        for node in view["nodes"]:
            blob = " ".join(filter(None, [node.get("name"), node.get("detail"), node.get("shape")]))
            for m in re.finditer(r"([a-z_][a-z0-9_.]*)\s*\[([0-9][0-9,\s]*)\]", blob):
                key = m.group(1)
                if key not in shapes:
                    continue
                claimed = [int(x) for x in m.group(2).replace(" ", "").split(",")]
                expected = shapes[key]
                if claimed != expected:
                    errors.append(
                        f"[{view['id']}/{node['id']}] {key} 页面写 {claimed}，"
                        f"材料里是 {expected}"
                    )
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
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="--geometry 时的站点根目录，默认当前目录")
    args = ap.parse_args()

    if not args.page.exists():
        print(f"error: 页面不存在: {args.page}", file=sys.stderr)
        return 2

    # 形状表缺失时直接报用法错误。让它抛 FileNotFoundError 会打出一串堆栈，
    # 使用者容易把「脚本崩了」误读成「核查过了」。
    if args.shapes and not args.shapes.exists():
        print(f"error: 形状表不存在: {args.shapes}", file=sys.stderr)
        return 2

    if args.geometry:
        # 页面路径相对站点根解析，二者都取绝对路径，relative_to 才能用
        root = args.root.resolve()
        page = args.page if args.page.is_absolute() else (root / args.page)
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

    errors: list[str] = []
    if sources:
        for view in views:
            heading = view.get("title", "") + " " + view.get("label", "")
            low = heading.lower()
            picked = [body for label, body in labeled_bodies.items() if label.lower() in low]
            if not picked:
                picked = universal
            if picked:
                errors += check_names([view], {"view": "\n".join(picked)})
    if not args.names_only and args.shapes:
        # 形状表读不出来就报错退出，不要带着半截数据继续核查——
        # 那会给出「核查通过」，而形状其实一条都没比对。
        try:
            shapes = json.loads(args.shapes.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"error: 形状表无法解析: {args.shapes}（{error}）", file=sys.stderr)
            return 2
        errors += check_shapes(views, shapes)
    if not args.names_only and args.script:
        errors += check_scripts_exist([Path(p) for p in args.script])

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
        checked.append("形状 vs 材料")
    print(f"核查通过：{len(views)} 视图 / {nodes} 节点" + (
        "（" + "、".join(checked) + "）" if checked else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
