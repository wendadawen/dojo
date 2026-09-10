#!/usr/bin/env python3
"""把内容与头部元数据拼进模板, 生成 wiki/sliding-window-attention/index.html。

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
     '<meta name="description" content="滑动窗口注意力把每个 query 的可见范围限制在最近 W 个位置：'
     '可见集规则、每层打分次数从 N² 降到 N·W、环形缓冲的槽位映射与位置编码，'
     '以及 DeepSeek-V4.1-Flash（W=128）每层 64 KiB 的缓存账。">'),
    ('<meta name="dojo:summary" content="【首页可渲染摘要；公式使用 $...$】">',
     '<meta name="dojo:summary" content="每个 query 只看最近 $W$ 个位置：每层打分次数从 $N^2$ 降到 '
     '$N\\cdot W$，KV cache 固定为 $W$ 条（$W=128$、512 维、FP8 时每层 64 KiB）；层堆叠的感受野为 '
     '$k\\times W$，环形缓冲按槽位 $i \\bmod W$ 覆盖最旧。">'),
    ('<meta name="dojo:topics" content="【主题，逗号分隔】">',
     '<meta name="dojo:topics" content="注意力机制,内存与缓存">'),
    ('<meta name="dojo:tag" content="【内容标签】">',
     '<meta name="dojo:tag" content="注意力机制">'),
    ('<title>【概念名】：【简要说明核心作用】</title>',
     '<title>滑动窗口注意力（SWA）：把每个 query 的可见范围限制在最近 W 个位置</title>'),
    ('<span class="nav-brand">【概念名】</span>',
     '<span class="nav-brand">滑动窗口注意力</span>'),
    ('<h1 class="title">【概念名】：【简要说明核心作用】</h1>',
     '<h1 class="title">滑动窗口注意力（SWA）：把每个 query 的可见范围限制在最近 $W$ 个位置</h1>'),
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
