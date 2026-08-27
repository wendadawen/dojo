"""用 headless Chrome 实测页面渲染：确认 KaTeX 真渲染、7 个视图的节点/边数据完整、悬停提示可生成。

做法：把探针脚本注入页面副本，结果 JSON.stringify 进 document.title，再从 dump 的 DOM 里取 <title>。
Chrome headless 不输出 console，这是唯一可行路径。
"""
import json, re, subprocess, pathlib, html

SRC = pathlib.Path("/Users/wendadawen/code/dojo/wiki/qwen3-8-flash-next-dataflow/index.html")
# 副本必须与原页面同目录，否则 ../../libs/ 相对路径解析不到本地库
TMP = SRC.parent / "_render_probe.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PROBE = r"""
<script>
setTimeout(function(){
  var r = {};
  try {
    r.katex = document.querySelectorAll('.katex').length;
    r.views = Object.keys(VIEWS);
    r.viewStats = {};
    var badEdges = [], badTeX = [];
    Object.keys(VIEWS).forEach(function(v){
      var V = VIEWS[v], ids = Object.keys(V.nodes);
      r.viewStats[v] = { nodes: ids.length, edges: V.edges.length };
      // 每条边的两端必须是本视图已定义的节点
      V.edges.forEach(function(e){
        if (ids.indexOf(e[0]) < 0) badEdges.push(v + ': source ' + e[0]);
        if (ids.indexOf(e[1]) < 0) badEdges.push(v + ': target ' + e[1]);
      });
      // 每个节点的公式必须能被 KaTeX 解析
      ids.forEach(function(id){
        var f = V.nodes[id].f;
        if (!f) return;
        try { katex.renderToString(f, {throwOnError:true, displayMode:true, strict:false}); }
        catch (err) { badTeX.push(v + '.' + id + ': ' + String(err).slice(0,80)); }
      });
      // drill 目标必须存在
      ids.forEach(function(id){
        var d = V.nodes[id].drill;
        if (d && !VIEWS[d]) badEdges.push(v + '.' + id + ': drill target ' + d + ' missing');
      });
    });
    r.badEdges = badEdges;
    r.badTeX = badTeX;
    // 逐个加载视图，确认 cytoscape 实际渲染出的节点数与定义一致
    var loaded = {};
    Object.keys(VIEWS).forEach(function(v){
      loadView(v);
      loaded[v] = { cyNodes: cy.nodes().length, cyEdges: cy.edges().length };
    });
    r.loaded = loaded;
    loadView('overview');
    // 正文中是否还有未渲染的 $...$
    var body = document.querySelector('main').innerText;
    var leftover = body.match(/\$[^$\n]{1,60}\$/g);
    r.leftoverMath = leftover ? leftover.slice(0,5) : [];
    r.tabs = document.querySelectorAll('.tab').length;
    r.tables = document.querySelectorAll('table').length;
    r.h2 = Array.prototype.map.call(document.querySelectorAll('h2'), function(h){return h.textContent.trim();});
  } catch (e) { r.error = String(e); }
  document.title = 'PROBE' + JSON.stringify(r);
}, 4000);
</script>
</body>
"""

src = SRC.read_text()
TMP.write_text(src.replace("</body>", PROBE))

out = subprocess.run(
    [CHROME, "--headless", "--disable-gpu", "--no-sandbox",
     "--virtual-time-budget=12000", "--dump-dom", f"file://{TMP}"],
    capture_output=True, text=True, timeout=180,
).stdout

m = re.search(r"<title>PROBE(.*?)</title>", out, re.S)
TMP.unlink(missing_ok=True)
if not m:
    print("未取到探针结果；title 片段：")
    print(re.findall(r"<title>.*?</title>", out, re.S)[:1])
    raise SystemExit(1)

r = json.loads(html.unescape(m.group(1)))
print("=" * 68)
print("headless Chrome 渲染实测")
print("=" * 68)
print(f"KaTeX 渲染节点数: {r['katex']}")
print(f"标签页数量: {r['tabs']}   表格数: {r['tables']}")
print(f"视图数: {len(r['views'])} -> {r['views']}")
print()
print(f"{'视图':<12s} {'定义节点':>8s} {'定义边':>7s} {'cy 节点':>8s} {'cy 边':>7s} {'一致':>5s}")
allok = True
for v in r["views"]:
    d, l = r["viewStats"][v], r["loaded"][v]
    ok = d["nodes"] == l["cyNodes"] and d["edges"] == l["cyEdges"]
    allok &= ok
    print(f"{v:<12s} {d['nodes']:>8d} {d['edges']:>7d} {l['cyNodes']:>8d} {l['cyEdges']:>7d} {'是' if ok else '否':>5s}")
print()
print(f"悬垂边/缺失下钻目标: {len(r['badEdges'])} {r['badEdges'] if r['badEdges'] else '(无)'}")
print(f"KaTeX 解析失败的公式: {len(r['badTeX'])} {r['badTeX'] if r['badTeX'] else '(无)'}")
print(f"正文残留未渲染 $...$: {len(r['leftoverMath'])} {r['leftoverMath'] if r['leftoverMath'] else '(无)'}")
print()
print("章节:")
for h in r["h2"]:
    print(f"  {h}")
print()
verdict = (allok and not r["badEdges"] and not r["badTeX"]
           and not r["leftoverMath"] and r["katex"] > 50 and "error" not in r)
print(f"总判定: {'通过' if verdict else '不通过'}")
if "error" in r:
    print("页面报错:", r["error"])
