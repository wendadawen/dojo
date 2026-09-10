#!/usr/bin/env python3
"""把内容与头部元数据拼进模板, 生成 wiki/attention-sink/index.html。

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
     '<meta name="description" content="注意力汇聚点是被注意力倾倒的位置：softmax 要求权重非零且和为 1，'
     '模型必须把多余的注意力分配出去，序列开头的位置最容易被训练成这个去处；移除它会让分母缺一大块、模型崩溃。'
     '含两种实现形态（保留初始位置 / 每头一个可学习标量）与份额实测。">'),
    ('<meta name="dojo:summary" content="【首页可渲染摘要；公式使用 $...$】">',
     '<meta name="dojo:summary" content="注意力汇聚点是被注意力倾倒的位置：softmax 必须把权重分配完，'
     '分母里的汇聚点项 $e^{\\text{sink}-m}$ 只影响归一化、不携带内容；'
     'DeepSeek-V4.1-Flash 每头一个可学习标量（43 层 $\\times$ 64 头），'
     '$W=128$ 时 logit 从 0 增到 5 可让它的份额从 0.78% 升到 53.69%。">'),
    ('<meta name="dojo:topics" content="【主题，逗号分隔】">',
     '<meta name="dojo:topics" content="注意力机制">'),
    ('<meta name="dojo:tag" content="【内容标签】">',
     '<meta name="dojo:tag" content="注意力机制">'),
    ('<title>【概念名】：【简要说明核心作用】</title>',
     '<title>注意力汇聚点（Attention Sink）：被注意力倾倒的位置，以及为什么不能丢掉它</title>'),
    ('<span class="nav-brand">【概念名】</span>',
     '<span class="nav-brand">注意力汇聚点</span>'),
    ('<h1 class="title">【概念名】：【简要说明核心作用】</h1>',
     '<h1 class="title">注意力汇聚点（Attention Sink）：被注意力倾倒的位置，以及为什么不能丢掉它</h1>'),
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
