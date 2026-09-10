# -*- coding: utf-8 -*-
"""以 qwen3-8-flash-next-dataflow 为外壳，生成 deepseek-v4-1-dataflow 页面。

替换四处：head 元信息、nav 品牌、正文区间、VIEWS+COLOR 区间。
"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "wiki/qwen3-8-flash-next-dataflow/index.html"
DST_DIR = ROOT / "wiki/deepseek-v4-1-dataflow"
DST = DST_DIR / "index.html"

tpl = io.open(SRC, encoding="utf-8").read()
body_new = io.open(ROOT / "wiki/deepseek-v4-1/research/_df_body.html", encoding="utf-8").read()
views_new = io.open(ROOT / "wiki/deepseek-v4-1/research/_df_views.js", encoding="utf-8").read()

# ---------- 1. head 元信息按行替换 ----------
HEAD_REPL = {
    '  <meta name="description"': '  <meta name="description" content="552B 主干加 196B Engram 参数、原生 1M 上下文：40 层按 Causal Encoder-Decoder 切成两半，解码器全局 KV 由编码器末层投影；全局 KV 用 CSA2 管理（每 r 个 token 池化一条、同压缩比跨层共享、Top-512 稀疏选择），主 KV 存 FP4、窗口 KV 存 FP8，每 token 缓存 890 字节；prefill 激活 8B、decode 16B。全部数字由官方配置、参考实现、真实 checkpoint 张量头与本机实测核对。">',
    '  <meta name="dojo:summary"': '  <meta name="dojo:summary" content="552B 主干 + 196B Engram；40 层 = 20 层编码器 + 20 层解码器，解码器全局 KV 由 $H_{L/2}$ 投影而来；主 KV 以 FP4 存储，每条目 288 字节，压缩比 2 的三个 source 层折合 144 B/token、压缩比 1 的层折合 288 B/token，加索引器 K 170 B 得每 token 890 字节；可见集 = 窗口 128 个位置 + 索引器选出的 Top-512 压缩条目，候选池 $2048 \\times 8 = 16384$；MoE 384 专家取 top-6 加 1 个共享专家；Engram 插在层 1、14，DSpark 为 3 个草稿块。">',
    '  <meta name="dojo:topics"': '  <meta name="dojo:topics" content="模型结构,内存与缓存">',
    '  <meta name="dojo:tag"': '  <meta name="dojo:tag" content="数据流速查">',
    '  <title>': '  <title>DeepSeek-V4.1-Flash 前向数据流 · Dojo</title>',
}

lines = tpl.split("\n")
for i, l in enumerate(lines):
    for prefix, newline in HEAD_REPL.items():
        if l.startswith(prefix):
            lines[i] = newline
tpl = "\n".join(lines)
assert 'DeepSeek-V4.1-Flash 前向数据流 · Dojo' in tpl

# nav 品牌
old_brand = '<span class="nav-brand">Qwen3.8-Flash-Next 前向数据流</span>'
assert tpl.count(old_brand) == 1
tpl = tpl.replace(old_brand, '<span class="nav-brand">DeepSeek-V4.1-Flash 前向数据流</span>')

# ---------- 2. 正文区间替换 ----------
a = tpl.find('<h1 class="title">')
b = tpl.find('<button class="back-to-top"')
assert a > 0 and b > a, (a, b)
tpl = tpl[:a] + body_new.strip("\n") + "\n\n" + tpl[b:]

# ---------- 3. VIEWS + COLOR 区间替换 ----------
c = tpl.find("var VIEWS = {")
d = tpl.find("var cy = cytoscape({")
assert c > 0 and d > c, (c, d)
tpl = tpl[:c] + views_new.strip("\n") + "\n\n" + tpl[d:]

# ---------- 3b. 标签页定义替换 ----------
p = tpl.find("var TABS=[")
q = tpl.find("];", p) + 2
assert p > 0 and q > p, (p, q)
tpl = tpl[:p] + "var TABS=[['overview','整体总览'],['attn','注意力内部'],['sparse','压缩与稀疏选择'],['moe','MoE 内部'],['engram','Engram 注入'],['dspark','DSpark 草稿链']];" + tpl[q:]

# ---------- 3c. 样式表换行修复 ----------
tpl = tpl.replace(
    "h2.collapsed .collapse-btn { transform: rotate(-90deg); }    /*",
    "h2.collapsed .collapse-btn { transform: rotate(-90deg); }\n\n    /*",
)

# ---------- 3d. 断言图库加载顺序正确（cytoscape → dagre → cytoscape-dagre）----------
# cytoscape-dagre 的 UMD 在加载期即读 window.dagre，顺序错了会在布局处抛 TypeError。
_p1 = tpl.find("libs/cytoscape.min.js")
_p2 = tpl.find("libs/dagre.min.js")
_p3 = tpl.find("libs/cytoscape-dagre.min.js")
assert -1 < _p1 < _p2 < _p3, f"图库加载顺序错误: {_p1}, {_p2}, {_p3}"

# ---------- 4. 落盘 ----------
DST_DIR.mkdir(parents=True, exist_ok=True)
io.open(DST, "w", encoding="utf-8").write(tpl)
print("已生成", DST)
print("行数:", len(tpl.split("\n")))
for bad in ("Qwen3.8", "qwen4_exp", "GDN", "QSA", "N-gram 注入层"):
    n = tpl.count(bad)
    if n:
        print(f"  **残留参照页字样 {bad}: {n} 处**")
print("残留占位符:", tpl.count("【"))
