"""
前置概念盘点：从页面正文实际出现的术语频次出发，找出读者理解本页所依赖的概念，
再逐个到全站 wiki 里验证是否有覆盖该概念的页面（看页面标题与章节，不看目录名）。
"""
from __future__ import annotations
import re, sys, pathlib
from collections import Counter
sys.path.insert(0, "/Users/wendadawen/code/dojo/.dojo/scripts")
from validate import PageInspector

ROOT = pathlib.Path("/Users/wendadawen/code/dojo")
PAGE = ROOT / "wiki/glm-5-3-flash-dataflow/index.html"

ins = PageInspector()
ins.feed(PAGE.read_text(encoding="utf-8"))
prose = " ".join(c for _, c in ins.prose_parts)

# 候选术语：本页涉及的机制类名词（含中英文写法）
TERMS = [
    "线性注意力", "delta rule", "KDA", "Kimi Delta", "稀疏注意力", "DSA",
    "MLA", "潜向量", "低秩", "深度卷积", "depthwise", "遗忘门", "门控",
    "残差", "超连接", "Sinkhorn", "双随机", "RMSNorm", "LayerNorm", "归一化",
    "MoE", "专家", "路由", "top-k", "sigmoid", "softmax", "SwiGLU", "SiLU",
    "位置编码", "RoPE", "NoPE", "KV cache", "prefill", "decode", "投机解码",
    "MTP", "量化", "FP8", "块量化", "bf16", "多模态", "patch", "ViT",
    "交叉熵", "因果掩码", "注意力", "嵌入", "Transformer", "chunk",
]
print("=== 页面术语频次（>0 才是真依赖）===")
hits = []
for t in TERMS:
    n = len(re.findall(re.escape(t), prose, re.I))
    if n:
        hits.append((n, t))
for n, t in sorted(hits, reverse=True):
    print(f"  {n:3d}  {t}")

# 全站页面：取每个 index.html 的 <title> 与 h1/h2，判断真实覆盖内容
print("\n=== 全站 wiki 页面清单（title + 前几个 h2）===")
pages = {}
for p in sorted((ROOT / "wiki").glob("*/index.html")):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    title = re.search(r"<title>(.*?)</title>", txt, re.S)
    h2s = re.findall(r"<h2[^>]*>(.*?)</h2>", txt, re.S)[:6]
    clean = lambda s: " ".join(re.sub(r"<[^>]+>", "", s).split())
    pages[p.parent.name] = (clean(title.group(1)) if title else "", [clean(x) for x in h2s])

for name, (title, h2s) in pages.items():
    print(f"  {name:34s} {title}")

# 已加的站内链接
links = re.findall(r'href="\.\./\.\./wiki/([^/]+)/', PAGE.read_text(encoding="utf-8"))
links += re.findall(r'href="\.\./([^/]+)/index\.html"', PAGE.read_text(encoding="utf-8"))
print(f"\n=== 本页已有的站内概念链接：{sorted(set(links)) or '（无）'} ===")
