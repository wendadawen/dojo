"""headless Chrome 渲染实测：公式真的渲染、7 个视图数据完整、无悬垂边。

做法：把页面复制为同目录 _render_probe.html，注入探针脚本把结果写进 document.title，
headless Chrome --dump-dom 后从 <title> 提取 JSON。
"""
import json, re, subprocess, pathlib, html

SRC = pathlib.Path("/Users/wendadawen/code/dojo/wiki/qwen3-5-dataflow/index.html")
TMP = SRC.parent / "_render_probe.html"

PROBE = """
<script>
(function(){
  function probe(){
    var r = {};
    r.katex = document.querySelectorAll('.katex').length;
    r.tabs = document.querySelectorAll('.tab').length;
    var views = {};
    var badEdges = [];
    for (var name in VIEWS) {
      var V = VIEWS[name];
      var nn = Object.keys(V.nodes).length, ne = V.edges.length;
      var ids = new Set(Object.keys(V.nodes));
      var dangling = 0;
      for (var i = 0; i < V.edges.length; i++) {
        if (!ids.has(V.edges[i][0]) || !ids.has(V.edges[i][1])) dangling++;
      }
      views[name] = {nodes: nn, edges: ne, dangling: dangling};
    }
    r.views = views;
    r.curView = (typeof curView !== 'undefined') ? curView : null;
    // 未渲染的 $...$（排除 script/code）
    var unre = 0;
    document.querySelectorAll('main p, main li, main td, main h3').forEach(function(el){
      if (/\\$[^$]{2,80}\\$/.test(el.textContent)) unre++;
    });
    r.unrenderedDollar = unre;
    document.title = 'PROBE' + JSON.stringify(r);
  }
  if (document.readyState === 'complete') setTimeout(probe, 2500);
  else window.addEventListener('load', function(){ setTimeout(probe, 2500); });
})();
</script>
"""

t = SRC.read_text()
t = t.replace("</body>", PROBE + "\n</body>")
TMP.write_text(t)

out = subprocess.run(
    ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless",
     "--disable-gpu", "--no-sandbox", "--virtual-time-budget=12000", "--dump-dom",
     f"file://{TMP}"],
    capture_output=True, text=True, timeout=300).stdout

m = re.search(r"<title>PROBE(.*?)</title>", out, re.S)
if not m:
    print("FAIL: 探针未执行（无 PROBE title）")
    raise SystemExit(1)
r = json.loads(html.unescape(m.group(1)))
print(f"KaTeX 节点数      : {r['katex']}")
print(f"标签页数          : {r['tabs']}")
print(f"未渲染 $...$ 段落 : {r['unrenderedDollar']}")
ok = True
for name, v in r["views"].items():
    flag = "OK " if v["dangling"] == 0 else "BAD"
    ok &= v["dangling"] == 0
    print(f"  {flag} 视图 {name:<10s} nodes={v['nodes']:>2d} edges={v['edges']:>2d} 悬垂边={v['dangling']}")
expect = {"overview": (11, 10), "gdn": (16, 20), "fa": (13, 15), "moe": (8, 10),
          "mtp": (10, 9), "vision": (7, 6), "fusion": (8, 9)}
for name, (nn, ne) in expect.items():
    got = r["views"].get(name, {})
    match = got.get("nodes") == nn and got.get("edges") == ne
    ok &= match
    print(f"  {'OK ' if match else 'BAD'} 视图 {name:<10s} 期望 nodes={nn} edges={ne} 实际 {got.get('nodes')}/{got.get('edges')}")
import re as _re
_body = SRC.read_text()
_body = _body[_body.index("<main"):]
_body = _body[:_body.index("<script>")]
_expected = len(_re.findall(r"\$\$.*?\$\$", _body, _re.S)) + len(_re.findall(r"\$[^$\n]{2,200}\$", _body))
katex_ok = r['katex'] >= _expected
if not katex_ok:
    print(f"  正文公式数 {_expected} > 渲染节点数 {r['katex']}")
print("KaTeX 判定: " + ("通过" if katex_ok else "失败"))
print("总判定: " + ("全部通过" if ok and katex_ok and r['unrenderedDollar'] == 0 else "存在问题"))
TMP.unlink()
