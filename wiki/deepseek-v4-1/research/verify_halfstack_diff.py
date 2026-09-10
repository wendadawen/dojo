#!/usr/bin/env python3
"""分解编码器半栈与解码器半栈的激活参数差额。

回答：为什么编码器半栈 7.8929B 比解码器半栈 7.5758B 高 317.1M？
数据源：ckpt/headers.json（真实分片头）+ inference/config.json。
"""
import json
import re
from collections import defaultdict
from pathlib import Path

HDR = json.loads((Path(__file__).parent / "ckpt" / "headers.json").read_text())
CFG = json.loads((Path(__file__).parent / "official" / "inference" / "config.json").read_text())

N_LAYERS = CFG["n_layers"]
TOP_K = CFG["n_activated_experts"]
N_ROUTED = CFG["n_routed_experts"]


def numel(meta):
    n = 1
    for s in meta["shape"]:
        n *= s
    if meta["dtype"] == "I8":
        n *= 2
    return n


def is_scale(name, meta):
    return meta["dtype"] == "F8_E8M0" or name.endswith(".scale")


per_layer = defaultdict(lambda: defaultdict(int))
for name, meta in HDR.items():
    m = re.match(r"layers\.(\d+)\.(.+)", name)
    if not m or is_scale(name, meta):
        continue
    l, rest = int(m.group(1)), m.group(2)
    n = numel(meta)
    if rest.startswith("attn."):
        per_layer[l]["attn"] += n
    elif rest.startswith("engram.embed."):
        per_layer[l]["engram_embed"] += n
    elif rest.startswith("engram."):
        per_layer[l]["engram_dense"] += n
    elif rest.startswith("ffn.experts."):
        per_layer[l]["routed_all"] += n
        per_layer[l]["routed_expert_count"] += 1 if rest.endswith("w1.weight") else 0
    else:
        per_layer[l]["other"] += n

print("=== 各层注意力参数量 ===")
for l in range(N_LAYERS):
    a = per_layer[l]["attn"] / 1e6
    tag = ""
    if l in (0, 1):
        tag = "  (纯 SWA，无压缩器)"
    elif per_layer[l]["attn"] > 130e6:
        tag = "  (含压缩器 + 索引器)"
    print(f"  层 {l:2d}: attn {a:8.2f}M{tag}")

print()
print("=== Engram 张量分解 ===")
for l in (1, 14):
    d = per_layer[l]
    print(f"  层 {l:2d}: engram_dense {d['engram_dense']/1e6:8.2f}M  "
          f"engram_embed {d['engram_embed']/1e6:8.2f}M")

engram_dense_total = sum(per_layer[l]["engram_dense"] for l in range(20))
print(f"  编码器 (层 0-19) engram_dense 合计: {engram_dense_total/1e6:.2f}M")

print()
print("=== 两半栈的注意力差异 ===")
enc_attn = sum(per_layer[l]["attn"] for l in range(20))
dec_attn = sum(per_layer[l]["attn"] for l in range(20, N_LAYERS))
print(f"  编码器 20 层 attn 合计: {enc_attn/1e6:.2f}M")
print(f"  解码器 20 层 attn 合计: {dec_attn/1e6:.2f}M")
print(f"  解码器 - 编码器 attn 差: {(dec_attn-enc_attn)/1e6:+.2f}M")

print()
print("=== 差额核对 ===")
enc_total, dec_total = 7.8929e9, 7.5758e9
diff = enc_total - dec_total
print(f"  编码器半栈 - 解码器半栈 = {diff/1e6:+.2f}M")
print(f"  其中 编码器 engram_dense        = {engram_dense_total/1e6:+.2f}M")
print(f"  其中 解码器多出的 attention     = {-(dec_attn-enc_attn)/1e6:+.2f}M")
resid = diff - engram_dense_total + (dec_attn - enc_attn)
print(f"  其余组件（MoE/gate/hc/norm 的逐层差异） = {resid/1e6:+.2f}M")
