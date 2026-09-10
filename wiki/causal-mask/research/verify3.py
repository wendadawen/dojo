# -*- coding: utf-8 -*-
"""第 3 轮修复复验：引用闭合 / 相邻双上标 / Unicode 数学 / TAB / 占位符 / 代码实跑。"""
import io
import re
import subprocess
import sys

PATH = '/Users/wendadawen/code/dojo/wiki/causal-mask/index.html'
OVERVIEW = '/Users/wendadawen/code/dojo/wiki/causal-mask/overview.html'

UNICODE_MATH = '×−∞∑√≤≥≈αβγκλθμσ∈ℝ⋅→←'

ok = True


def report(name, passed, detail=''):
    global ok
    ok = ok and passed
    print(('PASS' if passed else 'FAIL') + ' | ' + name + (' | ' + detail if detail else ''))


t = io.open(PATH, encoding='utf-8').read()

# 1. 引用双向闭合
sup_refs = set()
for m in re.findall(r'<sup>\[([^\]]+)\]</sup>', t):
    for r in m.split(','):
        r = r.strip()
        if re.match(r'^[A-Z]\d+$', r):
            sup_refs.add(r)
defined = set(re.findall(r'<sup>\[([A-Z]\d+)\]</sup>', t))
# 定义出现在来源章节每个 <p> 开头
defs_in_sources = set()
sec = t.split('<h2 id="sources-and-scope-notes">')[-1]
for m in re.findall(r'<sup>\[([A-Z]\d+)\]</sup>\s*', sec):
    defs_in_sources.add(m)
report('引用双向闭合', sup_refs == defs_in_sources,
       '正文引用=%s 定义=%s' % (sorted(sup_refs), sorted(defs_in_sources)))
report('<sup> 开闭配对', t.count('<sup>') == t.count('</sup>'),
       '开=%d 闭=%d' % (t.count('<sup>'), t.count('</sup>')))

# 2. 相邻双上标
adj = re.findall(r'</sup>\s*<sup>', t)
report('无相邻双上标', len(adj) == 0, '命中 %d' % len(adj))

# 3. Unicode 数学字符
found = sorted(set(c for c in t if c in UNICODE_MATH))
report('无 Unicode 数学字符', len(found) == 0, '命中 %s' % found)

# 4. TAB
report('无 TAB', '\t' not in t, '命中 %d' % t.count('\t'))

# 5. 占位符
ph = [w for w in ['待生成', 'TODO', 'TBD', '占位', 'FIXME', 'XXX', 'Lorem'] if w in t]
report('无占位符', len(ph) == 0, '命中 %s' % ph)

# 6. 代码块实跑与预期输出一致
code = re.search(r'<pre><code class="language-python">(.*?)</code></pre>', t, re.S)
expected = re.search(r'<code class="language-text">(.*?)</code>', t, re.S)
if not code or not expected:
    report('代码实跑', False, '未找到代码块')
else:
    import html
    src = html.unescape(code.group(1))
    exp = html.unescape(expected.group(1)).strip()
    p = subprocess.run([sys.executable, '-c', src], capture_output=True, text=True)
    got = p.stdout.strip()
    report('代码实跑与预期输出一致', got == exp,
           'stderr=%s' % p.stderr.strip()[:200] if got != exp else '')

print()
print('OVERALL: ' + ('OK' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
