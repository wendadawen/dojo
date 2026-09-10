#!/usr/bin/env python3
"""核对页面标注的报告行号（支持跨行片段）。

把 tech_report.txt 拼成一个大串做跨行搜索，再把字符位置映射回行号区间。
"""
import io
import re
from pathlib import Path

HERE = Path(__file__).parent
RAW = io.open(HERE / "official" / "tech_report.txt", encoding="utf-8").read()
LINES = RAW.split("\n")

# 每行起始的字符偏移
OFF = []
pos = 0
for l in LINES:
    OFF.append(pos)
    pos += len(l) + 1


def line_of(char_idx):
    lo, hi = 0, len(OFF) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if OFF[mid] <= char_idx:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def locate(snippet):
    """snippet 中的空白按 \\s+ 匹配，返回 (起始行, 结束行)"""
    pat = re.compile(r"\s+".join(re.escape(w) for w in snippet.split()))
    m = pat.search(RAW)
    if not m:
        return None
    return (line_of(m.start()), line_of(m.end() - 1))


CHECKS = [
    ("[C1] 标 314–319", "Its language backbone comprises 40 causal Transformer layers, organized into a 20-layer causal encoder"),
    ("[C2] 标 490–520", "Full Mode. The layer computes its own main KV and indexer Q"),
    ("[C3] 标 691–694", "we select E2M1 with one E4M3 scale per 16 channels"),
    ("[C3] 标 698–700", "We retain FP8 for the SWA KV cache due to its sensitivity to quantization"),
    ("[C6] 标 388–393", "the KV entries are not derived from their respective hidden states"),
    ("[C6] 标 388–393", "For global attention, CED treats the bottom"),
    ("[C6] 标 537–540", "When CSA2 is combined with CED, the decoder layer assigned to Full Mode"),
    ("[C7] 标 542–570", "selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions"),
    ("[C7] 标 542–570", "assigned the maximum index score"),
    ("[C11] 标 404–410", "Bounded Replay"),
    ("[N1] 标 18–19", "890 bytes per token"),
    ("[N1] 标 133–140", "roughly 1/4"),
    ("[N2] 标 320–322", "activating 8B parameters per token during prefill and 16B during decode"),
    ("[N3] 标 320", "552B"),
    ("[N5] 标 691–700", "omitting its second-level global scale"),
    ("[N9] 标 35–38", "1/437"),
    ("[N9] 标 139–140", "1/8"),
    ("[F7] 标 388–393", "using layer-dependent projection weights"),
    ("[4.1] 标 586–588", "residual streams"),
    ("[3.3] 报告 L550–551", "still score the full causally visible context"),
    ("[3.3] 报告 L576–579", "bounded independently of context length"),
]

print(f"{'标注':<24} {'实际行号':<14} 判定")
print("-" * 62)
for label, snip in CHECKS:
    r = locate(snip)
    print(f"{label:<24} {str(r):<14} " + ("命中" if r else "**未找到**"))
