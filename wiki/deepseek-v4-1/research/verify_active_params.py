#!/usr/bin/env python3
"""核对报告宣称 "activating 8B parameters per token during prefill and 16B during decode".

依据 (报告 §2.1 与 §2.2 CED):
  - backbone 40 层 = 20 层因果编码器 + 20 层解码器
  - CED: 解码器的全局 KV 由第 L/2 层隐状态投影而来 -> prefill 阶段只跑编码器一半
  - 因此每 token 激活参数: prefill ≈ 编码器半栈, decode ≈ 全栈
口径 (与 count_params.py 一致):
  - I8 是 FP4 两值打包 -> 逻辑元素数 x2; F8_E8M0 是 scale, 不计
  - 每层激活 = 注意力全量 + MoE (1 共享 + top-6 路由) + gate/bias + hc + norm + engram(若有)
数据源: ckpt/headers.json (HTTP Range 读到的真实分片头)
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
DIM, MOE_INTER = CFG["dim"], CFG["moe_inter_dim"]
ENGRAM_LAYERS = set(CFG["engram_layer_ids"])


def numel(meta):
    n = 1
    for s in meta["shape"]:
        n *= s
    if meta["dtype"] == "I8":   # fp4 packed
        n *= 2
    return n


def is_scale(name, meta):
    return meta["dtype"] == "F8_E8M0" or name.endswith(".scale")


# 每层张量按用途归类
per_layer = defaultdict(lambda: defaultdict(int))
for name, meta in HDR.items():
    m = re.match(r"layers\.(\d+)\.(.+)", name)
    if not m or is_scale(name, meta):
        continue
    l, rest = int(m.group(1)), m.group(2)
    n = numel(meta)
    if rest.startswith("attn."):
        per_layer[l]["attn"] += n
    elif rest.startswith("ffn.gate."):
        per_layer[l]["gate"] += n
    elif rest.startswith("ffn.shared_experts."):
        per_layer[l]["shared_expert"] += n
    elif rest.startswith("ffn.experts."):
        per_layer[l]["routed_all"] += n
        per_layer[l]["routed_expert_count"] += 1 if rest.endswith("w1.weight") else 0
    elif rest.startswith("engram."):
        # engram 表是稀疏访问: 每 token 只查 n_hash_cols 行 (embed.weight 整体不计)
        if rest.startswith("engram.embed."):
            per_layer[l]["engram_lookup"] += 1  # 占位, 下面用 config 常量算
        else:
            per_layer[l]["engram"] += n
    elif rest.startswith("hc_"):
        per_layer[l]["hc"] += n
    elif "norm" in rest:
        per_layer[l]["norm"] += n
    else:
        per_layer[l]["other"] += n

# 路由专家: 每层存 384 个, 每 token 只激活 top-6 -> 按比例折算
N_HASH_COLS = (CFG["engram_max_ngram_size"] - 1) * CFG["engram_n_heads"]
ENGRAM_LOOKUP_PER_LAYER = N_HASH_COLS * CFG["engram_head_dim"]  # 每 token 实际取回的表元素
print(f"engram 每 token 查表: ({CFG['engram_max_ngram_size']}-1) x {CFG['engram_n_heads']} = "
      f"{N_HASH_COLS} 行 x {CFG['engram_head_dim']} 维 = {ENGRAM_LOOKUP_PER_LAYER:,} 个表元素")

active = {}
for l in range(N_LAYERS):
    d = per_layer[l]
    n_experts = d["routed_expert_count"]
    assert n_experts == N_ROUTED, f"层 {l} 路由专家数 {n_experts} != {N_ROUTED}"
    per_expert = d["routed_all"] / n_experts
    active[l] = (
        d["attn"] + d["gate"] + d["shared_expert"] + per_expert * TOP_K
        + d["engram"] + d["engram_lookup"] * ENGRAM_LOOKUP_PER_LAYER
        + d["hc"] + d["norm"] + d["other"]
    )

enc = sum(active[l] for l in range(20))
dec = sum(active[l] for l in range(20, N_LAYERS))
emb = numel(HDR["embed.weight"])
head = numel(HDR["head.weight"])
print("每层激活参数构成 (层 0 / 层 20):")
for l in (0, 20):
    d = per_layer[l]
    eng = (d["engram"] + d["engram_lookup"] * ENGRAM_LOOKUP_PER_LAYER) / 1e6
    print(f"  层 {l}: attn {d['attn']/1e6:8.2f}M  gate {d['gate']/1e6:6.2f}M  "
          f"共享专家 {d['shared_expert']/1e6:7.2f}M  top-{TOP_K} 路由 {d['routed_all']/N_ROUTED*TOP_K/1e6:7.2f}M  "
          f"engram {eng:8.2f}M  hc {d['hc']/1e6:6.2f}M  norm {d['norm']/1e6:5.3f}M")

print(f"\n编码器半栈 (层 0-19) 每 token 激活: {enc/1e9:.4f}B")
print(f"解码器半栈 (层 20-39) 每 token 激活: {dec/1e9:.4f}B")
print(f"embed: {emb/1e9:.4f}B, head: {head/1e9:.4f}B")
print(f"\nprefill (只跑编码器):        {enc/1e9:.4f}B  (+embed {(enc+emb)/1e9:.4f}B)")
print(f"decode  (编码器+解码器):     {(enc+dec)/1e9:.4f}B  (+embed+head {(enc+dec+emb+head)/1e9:.4f}B)")
print("报告宣称: prefill 8B / decode 16B")
print(f"比值 decode/prefill = {(enc+dec)/enc:.3f} (报告隐含 2.0)")
print(f"最接近的读数: prefill {enc/1e9:.1f}B / decode {(enc+dec)/1e9:.1f}B")
