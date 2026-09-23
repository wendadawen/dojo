#!/usr/bin/env python3
"""数据流页的几何自检：连线是否穿框、是否与别的线并排重合。

为什么要有这个检查
------------------
这两类问题肉眼一看就烦，但只看数字推断容易被骗：
曾经用"两条线 x 差 < 3px"判定并行，报 0 处；实际差 2px、重叠 318px，
用户一眼就看出是重复的线。所以判据必须来自**渲染后的像素**，而不是几何推算。

检查方式：无头 Chrome 打开页面，把每条边按 1px 步长采样，落进栅格：
  · 穿框   —— 采样点落进非端点节点的框内（内缩 3px 避免贴边误报）
  · 并排   —— 两条边共用 >60 个相同像素（视觉上就是一条线）

用法：
    python3 .dojo/scripts/check_flow_geometry.py [--root .] [--port 0]

退出码 0 全部通过；1 有问题；2 环境错误（找不到 Chrome）。
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


CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

PAGE = "wiki/hy4-preview-dataflow/index.html"
RESULT_RE = re.compile(r"FLOWGEOM(\{.*?\})FLOWGEOM", re.S)

# 在页面里跑的探针：逐视图量穿框与像素重合
# 页内探针：逐个视图测量「穿框」与「像素重合」。
#
# 关键点：applyView 会触发 ELK 的**异步**布局，切换后必须等布局完成再量，
# 否则量到的是上一个视图的残留数据（实测九个视图全报同一组数字 = 假通过）。
# 所以这里用递归的 step()：切一个视图 → 等一拍 → 量 → 再切下一个。
#
# 指标口径也必须取自真实渲染：按坐标推算会漏报（差 2px 的两条线用
# 几何推算看不出来，用户一眼却能看出是重复的线）。
PROBE = r"""
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


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):        # 静默
        pass


def serve(root: Path):
    handler = functools.partial(Handler, directory=str(root))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="数据流页几何自检")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--page", default=PAGE, help="相对 root 的页面路径")
    args = ap.parse_args()

    chrome = find_chrome()
    if not chrome:
        print("error: 找不到 Chrome/Chromium，无法做几何自检", file=sys.stderr)
        return 2

    root = args.root.resolve()
    if not (root / args.page).exists():
        print(f"error: 页面不存在: {root / args.page}", file=sys.stderr)
        return 2

    # --dump-dom 只吃页面自带的脚本，所以先把探针注入一份临时副本再加载。
    # 页面里的 <script> 都是相对 ../../libs/ 引资源，副本必须放在同一层级，
    # 否则 libs 加载不到、探针拿不到 __flow。
    page_path = root / args.page
    injected = page_path.with_name("__geom_probe__.html")
    html = page_path.read_text(encoding="utf-8")
    injected.write_text(html.replace("</body>", PROBE + "</body>"), encoding="utf-8")

    httpd, port = serve(root)
    try:
        rel = injected.relative_to(root)
        url = f"http://127.0.0.1:{port}/{rel}"
        dom = subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=20000", "--dump-dom", url],
            capture_output=True, text=True, timeout=180,
        ).stdout
    finally:
        httpd.shutdown()
        injected.unlink(missing_ok=True)

    m = RESULT_RE.search(dom)
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


if __name__ == "__main__":
    sys.exit(main())
