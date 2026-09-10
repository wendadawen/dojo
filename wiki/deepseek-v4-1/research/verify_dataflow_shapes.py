# -*- coding: utf-8 -*-
"""从真实 checkpoint 张量头提取各模块形状，供数据流页使用。

数据源：ckpt/headers.json（HTTP Range 读到的 48 个分片头，96085 个张量）。
"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
HDR = json.loads((HERE / "ckpt" / "headers.json").read_text())
CFG = json.loads((HERE / "official" / "inference" / "config.json").read_text())


def shape(name):
    m = HDR.get(name)
    return m["shape"] if m else None


def dtype(name):
    m = HDR.get(name)
    return m["dtype"] if m else None


def merged(names):
    """把同一模块的多个张量形状合并展示"""
    out = []
    for n in names:
        if n in HDR:
            out.append((n, HDR[n]["shape"], HDR[n]["dtype"]))
    return out


print("=== 顶层 ===")
for n in ("embed.weight", "head.weight", "engram_hash.0", "engram_hash.1"):
    print(f"  {n:34s} {str(shape(n)):22s} {dtype(n)}")

print()
print("=== 层 0 的全部张量 ===")
rows = sorted(k for k in HDR if k.startswith("layers.0.") and ".experts." not in k)
for n in rows:
    print(f"  {n[10:]:40s} {str(HDR[n]['shape']):24s} {HDR[n]['dtype']}")

print()
print("=== 各层 attn 参数量分组（按模块归并） ===")
agg = defaultdict(lambda: defaultdict(int))
for name, meta in HDR.items():
    m = re.match(r"layers\.(\d+)\.(.+)", name)
    if not m:
        continue
    l, rest = int(m.group(1)), m.group(2)
    n = 1
    for s in meta["shape"]:
        n *= s
    if meta["dtype"] == "I8":
        n *= 2
    top = rest.split(".")[0]
    agg[l][top] += n
print("  层号   attn(M)   ffn(M)    hc(M)   其他(M)")
for l in (0, 1, 2, 14, 20, 24, 39):
    a = agg[l]
    print(f"  {l:3d}  {a['attn']/1e6:9.2f} {a['ffn']/1e6:9.2f} {a['hc']/1e6:8.2f}  "
          f"{(a.get('attn_norm',0)+a.get('ffn_norm',0))/1e6:6.3f}")

print()
print("=== MTP 块 0 的张量 ===")
rows = sorted(k for k in HDR if k.startswith("mtp.0.") and ".experts." not in k)
for n in rows:
    rest = n[6:]
    print(f"  {rest:40s} {str(HDR[n]['shape']):24s} {HDR[n]['dtype']}")

print()
print("=== 关键配置 ===")
for k in ("dim", "n_layers", "n_heads", "head_dim", "rope_head_dim", "q_lora_rank",
          "o_groups", "o_lora_rank", "window_size", "index_n_heads", "index_head_dim",
          "index_topk", "n_routed_experts", "n_activated_experts", "moe_inter_dim",
          "hc_mult", "vocab_size", "vision_n_layers", "vision_dim", "vision_patch_size"):
    print(f"  {k:22s} = {CFG[k]}")
