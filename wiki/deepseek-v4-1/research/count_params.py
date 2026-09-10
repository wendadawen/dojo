#!/usr/bin/env python3
"""从 safetensors 头统计 DeepSeek-V4.1-Flash 参数量, 并逐类核对形状。

验证目标:
  1. 报告宣称 "552B backbone parameters + 196B Engram parameters" (tech report §2.1)
  2. 报告宣称 "activating 8B parameters per token during prefill and 16B during decode"
  3. config.json 的每个结构维度 vs 真实张量形状 (inference/model.py 的模块定义)
口径:
  - I8 张量是 FP4 两值打包 (convert.py: view(torch.float4_e2m1fn_x2)), 逻辑元素数 = 存储元素数 x2
  - F8_E8M0 是 scale, 不计参数量
  - F8_E4M3 / BF16 / F32 按存储元素数计
输出: ckpt/count_params.out
"""
import json
import re
from collections import defaultdict
from pathlib import Path

HDR = json.loads((Path(__file__).parent / "ckpt" / "headers.json").read_text())

# ---- config 维度的期望值 (来自 official/config.json, 手工转录) ----
VOCAB = 129280
DIM = 5120
MOE_INTER = 2304
N_LAYERS = 40
N_HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
Q_LORA = 1280
O_LORA = 1024
O_GROUPS = 8
N_ROUTED = 384
INDEX_HEADS = 32
INDEX_DIM = 128
HC_MULT = 4
ENGRAM_HEADS = 8
ENGRAM_HEAD_DIM = 256
ENGRAM_NGRAM = 4
DSPARK_ROUTED = 128
MARKOV_RANK = 256
VIS_DIM, VIS_HEADS, VIS_INTER, VIS_LAYERS, VIS_PATCH, VIS_DS = 1024, 16, 2816, 32, 14, 3


def logical_numel(name: str, meta: dict) -> int:
    """逻辑参数个数; scale 类张量返回 0."""
    n = 1
    for s in meta["shape"]:
        n *= s
    if meta["dtype"] == "I8":
        n *= 2  # fp4 packed, 两值一字节
    return n


def is_scale(name: str, meta: dict) -> bool:
    return meta["dtype"] == "F8_E8M0" or name.endswith(".scale")


# ---- 分组 ----
GROUPS = {
    "embed_head": lambda n: n in ("embed.weight", "head.weight"),
    "vision": lambda n: n.startswith("vision.") or n.startswith("aligner.") or n.startswith("image_"),
    "engram": lambda n: ".engram." in n and not n.startswith("mtp."),
    "mtp": lambda n: n.startswith("mtp."),
    "backbone_attn": lambda n: n.startswith("layers.") and ".attn." in n,
    "backbone_moe": lambda n: n.startswith("layers.") and ".ffn." in n,
    "backbone_hc": lambda n: n.startswith("layers.") and ".hc_" in n,
    "backbone_norm": lambda n: n.startswith("layers.") and ("attn_norm" in n or "ffn_norm" in n),
}

sums = defaultdict(int)
counts = defaultdict(int)
unassigned = []
for name, meta in HDR.items():
    if is_scale(name, meta):
        continue
    for g, pred in GROUPS.items():
        if pred(name):
            sums[g] += logical_numel(name, meta)
            counts[g] += 1
            break
    else:
        unassigned.append(name)

print("=== 参数量分组 (scale 张量不计) ===")
total = 0
for g in sorted(sums):
    print(f"{g:16s} {sums[g]:>15,}  ({sums[g]/1e9:.4f}B, {counts[g]} tensors)")
    total += sums[g]
print(f"{'TOTAL':16s} {total:>15,}  ({total/1e9:.4f}B)")
print(f"unassigned: {len(unassigned)}", unassigned[:5])

backbone = sums["embed_head"] + sums["backbone_attn"] + sums["backbone_moe"] + sums["backbone_hc"] + sums["backbone_norm"]
engram = sums["engram"]
print(f"\nbackbone (含 embed/head, 不含 vision/engram/mtp): {backbone:,} = {backbone/1e9:.4f}B")
print(f"engram: {engram:,} = {engram/1e9:.4f}B")
print(f"vision: {sums['vision']:,} = {sums['vision']/1e9:.4f}B")
print(f"mtp(dspark): {sums['mtp']:,} = {sums['mtp']/1e9:.4f}B")

# ---- 逐形状核对 ----
print("\n=== 形状核对 (期望 vs 实际) ===")
fails = []


def check(name, shape, dtype=None):
    meta = HDR.get(name)
    if meta is None:
        fails.append(f"MISSING {name}")
        print(f"FAIL  {name}: 不存在")
        return
    ok = meta["shape"] == shape and (dtype is None or meta["dtype"] == dtype)
    if not ok:
        fails.append(f"SHAPE {name}: {meta['shape']} {meta['dtype']} != {shape} {dtype}")
    print(f"{'ok  ' if ok else 'FAIL'}  {name}: {meta['shape']} {meta['dtype']}")


check("embed.weight", [VOCAB, DIM])
check("head.weight", [VOCAB, DIM], "BF16")  # ckpt bf16 -> 代码 ParallelHead 转 fp32
L = 0
check(f"layers.{L}.attn.wq_a.weight", [Q_LORA, DIM], "F8_E4M3")
check(f"layers.{L}.attn.wq_b.weight", [N_HEADS * HEAD_DIM, Q_LORA], "F8_E4M3")
check(f"layers.{L}.attn.wkv.weight", [HEAD_DIM, DIM], "F8_E4M3")
check(f"layers.{L}.attn.wo_a.weight", [O_GROUPS * O_LORA, N_HEADS * HEAD_DIM // O_GROUPS], "F8_E4M3")
check(f"layers.{L}.attn.wo_b.weight", [DIM, O_GROUPS * O_LORA], "F8_E4M3")
check(f"layers.{L}.attn.attn_sink", [N_HEADS], "F32")
check(f"layers.{L}.ffn.gate.weight", [N_ROUTED, DIM])
check(f"layers.{L}.ffn.gate.bias", [N_ROUTED], "F32")
check(f"layers.{L}.ffn.gate.bias_vl", [N_ROUTED], "F32")
check(f"layers.{L}.ffn.experts.0.w1.weight", [MOE_INTER, DIM // 2], "I8")
check(f"layers.{L}.ffn.experts.0.w2.weight", [DIM, MOE_INTER // 2], "I8")
check(f"layers.{L}.ffn.experts.0.w1.scale", [MOE_INTER, DIM // 32], "F8_E8M0")
check(f"layers.{L}.ffn.shared_experts.w1.weight", [MOE_INTER, DIM], "F8_E4M3")
check(f"layers.{L}.hc_attn_fn", [(2 + HC_MULT) * HC_MULT, HC_MULT * DIM], "F32")
check(f"layers.{L}.hc_attn_base", [(2 + HC_MULT) * HC_MULT], "F32")
check(f"layers.{L}.hc_attn_scale", [3], "F32")
# kv source 层 2 (ratio 2, encoder) 与层 20 (ratio 1, decoder)
check("layers.2.attn.compressor.wkv.weight", [HEAD_DIM, DIM], "BF16")  # ckpt bf16 -> 代码 ratio>1 提升 fp32
check("layers.2.attn.compressor.wgate.weight", [HEAD_DIM, DIM], "BF16")
check("layers.2.attn.indexer.wk.weight", [INDEX_DIM, HEAD_DIM], "BF16")
check("layers.2.attn.indexer.wq_b.weight", [INDEX_HEADS * INDEX_DIM, Q_LORA], "F8_E4M3")
check("layers.2.attn.indexer.weights_proj.weight", [INDEX_HEADS, DIM], "BF16")
check("layers.20.attn.compressor.wkv.weight", [HEAD_DIM, DIM], "BF16")  # ratio 1 -> bf16
print("layer20 compressor.wgate 存在?", "layers.20.attn.compressor.wgate.weight" in HDR, "(期望 False: ratio 1 无门控)")
# engram (dtype 为 checkpoint 存储 dtype; 推理代码加载后提升精度, 见 model.py 注释)
check("layers.1.engram.embed.weight", [384006168, ENGRAM_HEAD_DIM], "F8_E4M3")
check("layers.14.engram.embed.weight", [384016682, ENGRAM_HEAD_DIM], "F8_E4M3")
check("layers.1.engram.embed.scale", [384006168, ENGRAM_HEAD_DIM // 32], "F8_E8M0")
check("layers.1.engram.wkv.weight", [DIM * (HC_MULT + 1), (ENGRAM_NGRAM - 1) * ENGRAM_HEADS * ENGRAM_HEAD_DIM], "F8_E4M3")
check("layers.1.engram.q_weight", [HC_MULT, DIM], "BF16")  # ckpt bf16 -> 代码 fp32
# mtp / dspark (dtype 为 ckpt 存储; main_proj 代码默认 fp8, markov head/confidence 代码转 fp32)
check("mtp.0.main_proj.weight", [DIM, DIM * 3], "F8_E4M3")
check("mtp.2.markov_head.embed.weight", [VOCAB, MARKOV_RANK])
check("mtp.2.markov_head.head.weight", [VOCAB, MARKOV_RANK], "BF16")
check("mtp.2.confidence_head.proj.weight", [1, DIM + MARKOV_RANK], "BF16")
check("mtp.0.ffn.experts.0.w1.weight", [MOE_INTER, DIM // 2], "I8")
# vision
check("vision.patch_embed.proj.weight", [VIS_DIM, 3 * VIS_PATCH * VIS_PATCH])
check("vision.blocks.0.attn.wqkv.weight", [3 * VIS_DIM, VIS_DIM])
check("vision.blocks.0.mlp.w1.weight", [2 * VIS_INTER, VIS_DIM])
check("aligner.w1.weight", [DIM, VIS_DIM * VIS_DS * VIS_DS])
check("aligner.w2.weight", [DIM, DIM])

print(f"\n核对失败 {len(fails)} 项")
for f in fails:
    print(" ", f)

# ---- 层分布交叉验证 ----
print("\n=== 模块层分布 (张量存在性 vs config) ===")
compressor_layers = sorted({int(m.group(1)) for n in HDR for m in [re.match(r"layers\.(\d+)\.attn\.compressor\.wkv\.weight", n)] if m})
wgate_layers = sorted({int(m.group(1)) for n in HDR for m in [re.match(r"layers\.(\d+)\.attn\.compressor\.wgate\.weight", n)] if m})
indexer_wq_layers = sorted({int(m.group(1)) for n in HDR for m in [re.match(r"layers\.(\d+)\.attn\.indexer\.wq_b\.weight", n)] if m})
indexer_wk_layers = sorted({int(m.group(1)) for n in HDR for m in [re.match(r"layers\.(\d+)\.attn\.indexer\.wk\.weight", n)] if m})
engram_layers = sorted({int(m.group(1)) for n in HDR for m in [re.match(r"layers\.(\d+)\.engram\.embed\.weight", n)] if m})
print("compressor 层:", compressor_layers, "(期望 [2, 8, 14, 20] = kv_source_layer_ids)")
print("wgate 层:", wgate_layers, "(期望 [2, 8, 14]: ratio>1 才有门控)")
print("indexer(wq_b) 层:", indexer_wq_layers, "(期望 [2, 8, 14, 20, 24, 28, 32, 36] = index_source_layer_ids)")
print("indexer.wk 层:", indexer_wk_layers, "(期望 [2, 8, 14, 20]: owns_k)")
print("engram 层:", engram_layers, "(期望 [1, 14])")

# ---- 每 token 激活参数量 (报告: prefill 8B / decode 16B) ----
print("\n=== 每 token 激活参数 ===")
per_layer_moe_active = (6 + 1) * 3 * MOE_INTER * DIM + N_ROUTED * DIM + 2 * N_ROUTED  # 6 routed+1 shared, gate, bias x2
attn_common = None  # 逐层注意力参数差异在 compressor/indexer, 下面分开算


def layer_attn_params(l: int) -> int:
    p = 0
    for name, meta in HDR.items():
        if name.startswith(f"layers.{l}.attn.") and not is_scale(name, meta):
            p += logical_numel(name, meta)
    return p


enc_attn = sum(layer_attn_params(l) for l in range(20))
dec_attn = sum(layer_attn_params(l) for l in range(20, 40))
moe_all = sum(
    logical_numel(n, m) for n, m in HDR.items()
    if n.startswith("layers.") and ".ffn." in n and not is_scale(n, m)
)
hc_norm = sums["backbone_hc"] + sums["backbone_norm"]
embed_head = sums["embed_head"]
print(f"encoder 注意力参数 (0-19): {enc_attn/1e9:.4f}B")
print(f"decoder 注意力参数 (20-39): {dec_attn/1e9:.4f}B")
print(f"全部 MoE 参数 (40 层): {moe_all/1e9:.4f}B")
active_moe_layer = (6 + 1) * 3 * MOE_INTER * DIM
print(f"单层激活专家参数 (6 路由 + 1 共享): {active_moe_layer/1e9:.6f}B")
