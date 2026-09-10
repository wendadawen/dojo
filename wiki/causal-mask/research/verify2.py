# -*- coding: utf-8 -*-
"""修复后复验脚本：引用闭环 / 双上标 / Unicode 数学 / TAB / 占位符 / 代码实跑比对。"""
import re, subprocess, sys, tempfile, os

PAGE = "/Users/wendadawen/code/dojo/wiki/causal-mask/index.html"
html = open(PAGE, encoding="utf-8").read()

# 1) 引用编号双向闭合
cited = set()
for m in re.findall(r"<sup>\[([^\]]+)\]</sup>", html):
    for part in m.split(","):
        part = part.strip()
        if part:
            cited.add(part)
defined = set(re.findall(r"<sup>\[(C\d+|F\d+|N\d+)\]</sup>\s*[^<]*[:：]", html))
defined |= set(re.findall(r"<sup>\[(C\d+|F\d+|N\d+)\]</sup>", html.split("<h3>论断与来源")[-1]))
# 更精确：来源章节里以 <sup>[X]</sup> 开头的定义段
src_block = html.split('id="sources-and-scope-notes"')[-1]
defined = set(re.findall(r"<p><sup>\[(C\d+|F\d+|N\d+)\]</sup>", src_block))
print("cited  =", sorted(cited))
print("defined=", sorted(defined))
print("cited - defined =", sorted(cited - defined))
print("defined - cited =", sorted(defined - cited))

# 2) 相邻双上标
adj = re.findall(r"</sup>\s*<sup>", html)
print("相邻双上标:", len(adj))

# 3) 无 Unicode 数学字符（去公式/代码/pre 后扫描）
body = html
body_nomath = re.sub(r"\$\$[\s\S]+?\$\$", " ", body)
body_nomath = re.sub(r"\\\([\s\S]+?\\\)", " ", body_nomath)
body_nomath = re.sub(r"\\\[[\s\S]+?\\\]", " ", body_nomath)
body_nomath = re.sub(r"(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$", " ", body_nomath)
body_nomath = re.sub(r"<pre[\s\S]*?</pre>", " ", body_nomath)
body_nomath = re.sub(r"<code[\s\S]*?</code>", " ", body_nomath)
math_chars = re.findall(
    "[\u0391-\u03a9\u03b1-\u03c9\u2202\u2207\u221a\u221d\u221e\u2211\u220f\u222b"
    "\u2248\u2260\u2261\u2264\u2265\u226a\u226b\u2208\u2209\u2282\u2283\u2229\u222a\u2205"
    "\u2295\u2297\u22c5\u2070-\u209f]",
    body_nomath,
)
print("Unicode 数学字符:", sorted(set(math_chars)))
# 额外扫 × ∈ → −
extra = [c for c in "×∈→−≤≥∞" if c in body_nomath]
print("额外符号（公式外）:", extra)
print("TAB:", body.count("\t"))
ph = re.findall(r"【[^】]*】|TODO|TBD|placeholder|FIXME|待生成", body)
print("残留占位符:", ph)

# 4) 抽取 language-python 代码块实跑并比对预期输出
py_code = re.search(r'<code class="language-python">([\s\S]*?)</code>', html).group(1)
import html as _h
py_code = _h.unescape(py_code)
expected = _h.unescape(re.search(r'<code class="language-text">([\s\S]*?)</code>', html).group(1))
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
    f.write(py_code)
    p = f.name
r = subprocess.run(["/usr/bin/python3", p], capture_output=True, text=True)
os.unlink(p)
print("returncode:", r.returncode, "stderr:", repr(r.stderr))
actual = r.stdout.rstrip("\n")
exp = expected.rstrip("\n")
print("逐字符一致:", actual == exp)
if actual != exp:
    import difflib
    print("\n".join(difflib.unified_diff(exp.splitlines(), actual.splitlines(), "expected", "actual")))
