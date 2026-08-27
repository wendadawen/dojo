"""参数量核算：按张量真实 shape 求和，分组统计。

数据来源：Qwen/Qwen3.8-Flash-Next@f5d08274 的 131 个 safetensors 分片 JSON 头。
不使用 metadata.total_size（那是磁盘字节数）。
"""
import json, re
from math import prod

H = json.load(open("/tmp/qwen38fn/headers.json"))


def bucket(name):
    if name.startswith("mtp."):
        return "MTP 草稿层"
    if name.startswith("model.visual."):
        return "视觉编码器"
    if "ple_embedding.ngram_embedding" in name:
        return "N-gram Embedding 表"
    if ".ple." in name:
        return "PLE 投影/卷积/norm"
    if name == "lm_head.weight":
        return "LM head"
    if "embed_tokens" in name:
        return "词表 Embedding"
    if "hyper_connection" in name:
        return "Gated Residual（超连接）"
    if ".mlp.experts." in name:
        return "MoE 路由专家"
    if ".mlp.shared_expert" in name:
        return "MoE 共享专家"
    if ".mlp.gate." in name:
        return "MoE 路由器"
    if ".linear_attn." in name:
        return "GDN 线性注意力"
    if ".self_attn." in name:
        return "QSA 全注意力（含 indexer）"
    return "其他"


tot = {}
for k, v in H.items():
    if v["dtype"] == "I64":  # buffer，不是可训练参数
        continue
    tot.setdefault(bucket(k), [0, 0])
    tot[bucket(k)][0] += prod(v["shape"])
    tot[bucket(k)][1] += 1

grand = sum(v[0] for v in tot.values())
print(f"{'分组':<28s} {'参数量':>18s} {'占比':>8s} {'张量数':>7s}")
print("-" * 66)
for k, (n, c) in sorted(tot.items(), key=lambda x: -x[1][0]):
    print(f"{k:<28s} {n:>18,d} {n / grand * 100:>7.2f}% {c:>7d}")
print("-" * 66)
print(f"{'合计（含 MTP+视觉）':<28s} {grand:>18,d} {'100.00%':>8s}")

ng = tot["N-gram Embedding 表"][0]
mtp = tot["MTP 草稿层"][0]
vis = tot["视觉编码器"][0]
print()
print(f"N-gram Embedding 表        : {ng:>18,d}  = {ng / 1e9:.2f} B")
print(f"主干（合计 - N-gram - MTP）: {grand - ng - mtp:>18,d}  = {(grand - ng - mtp) / 1e9:.2f} B")
print(f"  其中视觉编码器            : {vis:>18,d}  = {vis / 1e9:.3f} B")
print(f"  其中语言主干（减视觉）    : {grand - ng - mtp - vis:>18,d}  = {(grand - ng - mtp - vis) / 1e9:.2f} B")
print(f"MTP 草稿层                 : {mtp:>18,d}  = {mtp / 1e9:.2f} B")

print()
print("=== 单 token 激活参数量（推理时实际参与矩阵乘的权重）===")
CFG = json.load(open("/tmp/qwen38fn/config.json"))["text_config"]
d, E, K = CFG["hidden_size"], CFG["num_experts"], CFG["num_experts_per_tok"]
im, si = CFG["moe_intermediate_size"], CFG["shared_expert_intermediate_size"]
L = CFG["num_hidden_layers"]
n_lin = sum(1 for t in CFG["layer_types"] if t == "linear_attention")
n_qsa = L - n_lin

# 每层 MoE：top-K 路由专家 + 共享专家 + 路由器 + 共享门
per_expert = 2 * im * d + d * im
moe_act = K * per_expert + (2 * si * d + d * si) + E * d + d
# GDN 单层
gdn = (10240 * d) + (6144 * d) + 2 * (48 * d) + (2560 * 6144) + 10240 * 4 + 2 * 48 + 128
# QSA 单层
qsa = (12288 * d) + 2 * (512 * d) + (2560 * 6144) + (640 * d) + 2 * 128 + 2 * 256
# 超连接：每层两组
hc = 2 * (4 * d + 320 * 4 * d + 4 * d * 320 + 4 * 4 * d)

act = (n_lin * (gdn + moe_act + hc) + n_qsa * (qsa + moe_act + hc)
       + (4 * d + 320 * 4 * d + 4 * d * 320)          # 末端 hyper_connection_mixer
       + d + 248320 * d)                               # 输入 embedding 查表 + lm_head
# PLE 层的额外投影（只在第 2 层）+ 每 token 从 51.2B 表里查 16 个 head
ple_proj = 10240 * d + d * d + 3 * 10240 + 10240 * 4
ngram_lookup = 16 * 160
act_with_ple = act + ple_proj + ngram_lookup
print(f"  linear_attention 层数 = {n_lin}, qsa 层数 = {n_qsa}")
print(f"  每层 MoE 激活   = {moe_act:,d}")
print(f"  GDN 单层        = {gdn:,d}")
print(f"  QSA 单层        = {qsa:,d}")
print(f"  超连接 每层     = {hc:,d}")
print(f"  含 lm_head 的激活合计        = {act_with_ple:,d} = {act_with_ple / 1e9:.3f} B")
print(f"  不含 lm_head/embedding 的合计 = {act_with_ple - 248320 * d - d:,d} = {(act_with_ple - 248320 * d - d) / 1e9:.3f} B")
print(f"  每 token 实际查 N-gram 表元素 = {ngram_lookup} 个（表本身 {ng / 1e9:.1f} B 不参与矩阵乘）")
