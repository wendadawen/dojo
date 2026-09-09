#!/usr/bin/env python3
"""无头 Chrome 渲染实测：复制页面为测试副本，注入探针，dump-dom 读结果。"""
import json, re, subprocess, sys, os

PAGE = '/Users/wendadawen/code/dojo/wiki/hy4-preview-dataflow/index.html'
TEST = '/Users/wendadawen/code/dojo/wiki/hy4-preview-dataflow/_render_test.html'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

ERR_HOOK = r"""
<script>
window.__errs = [];
window.onerror = function(msg, src, line, col){ window.__errs.push(String(msg).slice(0,120) + ' @' + line + ':' + col); return true; };
window.addEventListener('error', function(ev){ if (ev.message) window.__errs.push('E:' + String(ev.message).slice(0,120)); }, true);
</script>
"""

PROBE = r"""
<script>
(function(){
  var R = {errors: (window.__errs || []).slice()};
  window.addEventListener('load', function(){
    try { R.katexCount = document.querySelectorAll('.katex').length; } catch(e){ R.katexCount = 'ERR:'+e.message; }
    try { R.katexError = document.querySelectorAll('.katex-error').length; } catch(e){ R.katexError = 'ERR'; }
    try { R.canvas = document.querySelectorAll('#cy canvas').length; } catch(e){ R.canvas = 'ERR'; }
    try { R.tabs = document.querySelectorAll('#tabs .tab').length; } catch(e){ R.tabs = 'ERR'; }
    try { R.legendItems = document.querySelectorAll('#legend > div').length; } catch(e){ R.legendItems = 'ERR'; }
    // 逐视图切换并记录
    R.views = {};
    var names = Object.keys(window.VIEWS || {});
    R.viewNames = names.join(',');
    for (var i = 0; i < names.length; i++) {
      try {
        loadView(names[i]);
        var V = VIEWS[names[i]];
        var nn = Object.keys(V.nodes).length, ne = V.edges.length;
        var vis = document.querySelectorAll('#cy canvas').length;
        // 检查每个视图的公式 f 都能被 katex 渲染
        var ferr = [];
        Object.keys(V.nodes).forEach(function(id){
          var f = V.nodes[id].f;
          if (!f) return;
          try {
            var html = katex.renderToString(f, {throwOnError: true, strict: false});
            if (html.indexOf('katex-error') >= 0) ferr.push(id);
          } catch(e) { ferr.push(id + ':' + e.message.slice(0, 60)); }
        });
        R.views[names[i]] = 'n=' + nn + ',e=' + ne + ',canvas=' + vis + (ferr.length ? ',FERR=[' + ferr.join(';') + ']' : ',f=ok');
      } catch(e) { R.views[names[i]] = 'EXC:' + e.message; }
    }
    // 切回 overview，测 tooltip 渲染
    loadView('overview');
    try {
      var o = VIEWS.overview.nodes.emb;
      var h = '<h4>' + o.label + '</h4><div class="fx">' + renderTeX(o.f) + '</div><div>' + renderD(o.d) + '</div>';
      var probeDiv = document.createElement('div');
      probeDiv.id = '__tooltip_probe';
      probeDiv.style.display = 'none';
      probeDiv.innerHTML = h;
      document.body.appendChild(probeDiv);
      R.tooltipProbe = {
        h4: probeDiv.querySelector('h4') !== null,
        katexInTooltip: probeDiv.querySelectorAll('.katex').length,
        katexErrInTooltip: probeDiv.querySelectorAll('.katex-error').length
      };
    } catch(e) { R.tooltipProbe = 'EXC:' + e.message; }
    // 可见元素两两重叠抽查
    try {
      var rects = [];
      document.querySelectorAll('h2, h3, td, th, .callout, .num').forEach(function(el){
        if (rects.length >= 400) return;
        var r = el.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) rects.push({tag: el.tagName, t: r.top, b: r.bottom, l: r.left, r2: r.right, txt: (el.textContent || '').slice(0, 12)});
      });
      var overlap = [];
      for (var a = 0; a < rects.length && overlap.length < 5; a++)
        for (var b = a + 1; b < rects.length && overlap.length < 5; b++) {
          var A = rects[a], B = rects[b];
          if (A.tag !== B.tag) continue;
          var ow = Math.min(A.r2, B.r2) - Math.max(A.l, B.l);
          var oh = Math.min(A.b, B.b) - Math.max(A.t, B.t);
          if (ow > 8 && oh > 8) overlap.push(A.tag + '["' + A.txt + '" vs "' + B.txt + '"]');
        }
      R.overlaps = overlap;
      R.rectSample = rects.length;
    } catch(e) { R.overlaps = 'EXC:' + e.message; }
    // 页面章节统计
    try {
      R.sections = Array.prototype.map.call(document.querySelectorAll('h2'), function(h){ return h.textContent.trim().slice(0, 30); });
    } catch(e) {}
    document.title = '__PROBE__' + JSON.stringify(R);
  });
})();
</script>
"""

html = open(PAGE, encoding='utf-8').read()
assert '</body>' in html
html = html.replace('</body>', PROBE + '</body>')
# 错误捕获器插到 <head...> 开标签之后，确保先于一切脚本执行（容错：标签可能带预览服务注入的属性）
m = re.search(r'<head[^>]*>', html)
assert m, 'no <head> tag'
html = html[:m.end()] + ERR_HOOK + html[m.end():]
open(TEST, 'w', encoding='utf-8').write(html)
print('test copy written')

url = 'file://' + TEST
cmd = [CHROME, '--headless=new', '--disable-gpu', '--no-sandbox',
       '--timeout=12000', '--dump-dom', url]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
dom = r.stdout
open('/tmp/hy4_dom_dump.html', 'w', encoding='utf-8').write(dom)
# 诊断：DOM 静态状态
print('--- diagnostics ---')
print('dom len:', len(dom))
print('canvas in dom:', dom.count('<canvas'))
print('katex spans in dom:', dom.count('class="katex'))
print('katex-error in dom:', dom.count('katex-error'))
print('tabs span in dom:', dom.count('class="tab'))
print('probe script present:', '__PROBE__' in dom)
print('window.__probe present:', '__probe' in dom)
print('dom saved to /tmp/hy4_dom_dump.html')
m = re.search(r'<title>(.*?)</title>', dom, re.S)
if not m:
    print('NO TITLE FOUND')
    print(r.stderr[:2000])
    sys.exit(1)
title = m.group(1)
if not title.startswith('__PROBE__'):
    print('PROBE NOT RUN, title =', title[:300])
    sys.exit(1)
data = json.loads(title[len('__PROBE__'):])
print(json.dumps(data, ensure_ascii=False, indent=1))
os.remove(TEST)
print('=== test copy removed ===')
