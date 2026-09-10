#!/usr/bin/env python3
"""把内容与头部元数据拼进模板, 生成 wiki/cross-layer-kv-sharing/index.html。

避免用 Edit 工具处理含 CJK 的长串 (历史上会"假成功"), 统一用脚本做替换并回读校验。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # wiki/<name>/research/x.py -> 仓库根
TPL = ROOT / ".dojo" / "templates" / "concept" / "index.html"
CONTENT = Path(__file__).parent / "_content.html"
OUT = Path(__file__).resolve().parents[1] / "index.html"

tpl = TPL.read_text()
content = CONTENT.read_text().strip()

REPL = [
    ('<meta name="description" content="【首页摘要】">',
     '<meta name="description" content="让多层共用同一份全局 KV：上段各层不再自算全局 KV，'
     '缓存复杂度从 O(LND) 降到 O((N+L)D)，prefill 可提前退出；'
     '并区分共享 KV 与复用索引两级（Full/Reindex/Reuse），含运行期共享分组实测。">'),
    ('<meta name="dojo:summary" content="【首页可渲染摘要；公式使用 $...$】">',
     '<meta name="dojo:summary" content="上段各层复用下段末层隐状态投影出的全局 KV：'
     '$\\hat K = \\mathrm{LN}(X^{L/2})W_K$、$\\hat V = \\mathrm{LN}(X^{L/2})W_V$，'
     '缓存复杂度从 $\\mathcal{O}(LND)$ 降到 $\\mathcal{O}((N+L)D)$；'
     '共享 KV 与复用索引是两件事，实测共享分组为 2-7 / 8-13 / 14-19 / 20-39。">'),
    ('<meta name="dojo:topics" content="【主题，逗号分隔】">',
     '<meta name="dojo:topics" content="注意力机制,内存与缓存">'),
    ('<meta name="dojo:tag" content="【内容标签】">',
     '<meta name="dojo:tag" content="注意力机制">'),
    ('<title>【概念名】：【简要说明核心作用】</title>',
     '<title>跨层 KV 复用（Cross-Layer KV Sharing）：把全局 KV 的层维度消掉</title>'),
    ('<span class="nav-brand">【概念名】</span>',
     '<span class="nav-brand">跨层 KV 复用</span>'),
    ('<h1 class="title">【概念名】：【简要说明核心作用】</h1>',
     '<h1 class="title">跨层 KV 复用（Cross-Layer KV Sharing）：把全局 KV 的层维度消掉</h1>'),
]
for old, new in REPL:
    assert tpl.count(old) == 1, f"模板中未唯一匹配: {old[:60]}"
    tpl = tpl.replace(old, new)

marker_start = "<!-- @content"
i = tpl.index(marker_start)
j = tpl.index("-->", i) + 3
tpl = tpl[:i] + content + tpl[j:]

assert "【" not in tpl, "残留占位符"
assert "@content" not in tpl and "@copy-start" not in tpl
OUT.write_text(tpl)
print(f"写入 {OUT} ({len(tpl.splitlines())} 行)")
print("残留占位符检查: 通过")
