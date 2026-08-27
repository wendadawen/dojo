"""
用 headless Chrome 实测页面渲染：① KaTeX 是否真的渲染（含 SVG foreignObject 内的公式）；
② 图内标签矩形是否两两重叠。
做法：把探针脚本注入页面副本，结果写进 document.title，再从 dump 出的 DOM 里取 <title>。
"""
from __future__ import annotations
import re, subprocess, sys, json, pathlib

PAGE = pathlib.Path("/Users/wendadawen/code/dojo/wiki/glm-5-3-flash-dataflow/index.html")
# 探针副本必须与原页面同目录，否则 ../../libs/ 解析不到，KaTeX 根本不会加载
TMP = PAGE.parent / "_render_probe.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PROBE = r"""
<script>
window.addEventListener('load', function () {
  setTimeout(function () {
    var out = {};
    out.katex = document.querySelectorAll('.katex').length;
    // 未渲染的 $...$ 残留（排除 script/style/pre/code）
    var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    var leftover = [], n;
    while ((n = walker.nextNode())) {
      var p = n.parentElement;
      if (!p) continue;
      if (p.closest('script,style,pre,code')) continue;
      if (p.closest('.katex')) continue;
      if (/\$[^$]{1,80}\$/.test(n.nodeValue)) leftover.push(n.nodeValue.trim().slice(0, 50));
    }
    out.leftover = leftover;
    out.fo = document.querySelectorAll('foreignObject').length;
    out.fo_katex = document.querySelectorAll('foreignObject .katex').length;

    // 每张图内标签矩形两两重叠检测
    var figs = [];
    document.querySelectorAll('figure svg').forEach(function (svg, si) {
      var boxes = [];
      svg.querySelectorAll('foreignObject > div, text').forEach(function (el) {
        var r = el.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) return;
        boxes.push({ t: (el.textContent || '').trim().slice(0, 26), x: r.left, y: r.top, w: r.width, h: r.height });
      });
      var hits = [];
      for (var i = 0; i < boxes.length; i++)
        for (var j = i + 1; j < boxes.length; j++) {
          var a = boxes[i], b = boxes[j];
          var ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
          var oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
          if (ox > 2 && oy > 2) hits.push([a.t, b.t, Math.round(ox), Math.round(oy)]);
        }
      figs.push({ svg: si, labels: boxes.length, overlaps: hits });
    });
    out.figs = figs;
    // 是否有元素横向溢出容器
    var ovf = [];
    document.querySelectorAll('figure svg, table, pre').forEach(function (el) {
      if (el.scrollWidth > el.clientWidth + 2 && el.tagName !== 'PRE')
        ovf.push(el.tagName + ':' + el.scrollWidth + '>' + el.clientWidth);
    });
    out.overflow = ovf;
    document.title = 'PROBE' + JSON.stringify(out) + 'ENDPROBE';
  }, 3500);
});
</script>
"""

html = PAGE.read_text(encoding="utf-8")
assert "</body>" in html
TMP.write_text(html.replace("</body>", PROBE + "</body>"), encoding="utf-8")

r = subprocess.run(
    [CHROME, "--headless", "--disable-gpu", "--no-sandbox", "--virtual-time-budget=9000",
     "--window-size=1200,900", "--dump-dom", f"file://{TMP}"],
    capture_output=True, text=True, timeout=180)
dom = r.stdout
m = re.search(r"PROBE(.*?)ENDPROBE", dom, re.S)
if not m:
    print("探针未回传结果。DOM 长度:", len(dom))
    print(dom[:600])
    sys.exit(1)

d = json.loads(m.group(1))
TMP.unlink(missing_ok=True)
print(f"KaTeX 渲染节点数        = {d['katex']}")
print(f"foreignObject 数        = {d['fo']}，其中含 KaTeX 的 = {d['fo_katex']}")
print(f"未渲染的 $...$ 残留     = {len(d['leftover'])} 处" +
      ("" if not d["leftover"] else f" → {d['leftover'][:5]}"))
print(f"横向溢出元素            = {d['overflow'] or '无'}")
print("\n各图标签重叠检测：")
bad = 0
for f in d["figs"]:
    n = len(f["overlaps"])
    bad += n
    print(f"  图 {f['svg']}: 标签 {f['labels']:3d} 个，重叠 {n} 对" +
          ("" if not n else f" → {f['overlaps'][:4]}"))
print(f"\n结论：重叠共 {bad} 对；未渲染公式 {len(d['leftover'])} 处")
sys.exit(1 if (bad or d["leftover"]) else 0)
