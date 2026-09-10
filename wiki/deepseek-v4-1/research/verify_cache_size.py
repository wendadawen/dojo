#!/usr/bin/env python3
"""核对技术报告宣称的 "global KV cache footprint = 890 bytes per token".

来源:
  - 报告 2.4.4: 主 KV 用 MXFP4 (E2M1, 每 16 通道一个 E4M3 scale, 无二级 global scale), RoPE 后量化;
    SWA KV 保持 FP8; indexer Q/K 也是 FP4 (QAT).
  - 报告 2.3.1: 三种模式, 主 KV 与 indexer K 跨层共享 (同一 ratio 组共用一个, 由组内第一个层产生).
  - 代码常量: 主 KV 量化 fp4_act_quant(latent, 16, ..., scale_dtype=e4m3) (model.py L760);
    indexer K 量化 fp4_act_quant(k, fp4_block_size=32) (L546, kernel 默认 e8m0);
    head_dim=512 (config), index_head_dim=128 (config), 单 KV 头.
  - 共享结构: kv_source_layer_ids = [2,8,14,20], 组 = 相邻 source 之间.
本脚本只用 config + 代码常量推导每 token 字节数, 与报告宣称值对比.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
# 推理代码实际读取的配置 (generate.py: ModelArgs(**json.load(f)))
inf = json.loads((HERE / "official" / "inference" / "config.json").read_text())
# HF 仓库的 transformers 格式配置 (text_config 段)
hf = json.loads((HERE / "official" / "config.json").read_text())["text_config"]

# 两份配置对本脚本用到的字段必须一致 (独立交叉核对)
pairs = [
    ("head_dim", "head_dim"),
    ("index_head_dim", "index_head_dim"),
    ("kv_source_layers", "kv_source_layer_ids"),
    ("index_source_layers", "index_source_layer_ids"),
    ("compress_ratios", "compress_ratios"),
    ("n_layers", "num_hidden_layers"),
]
for a, b in pairs:
    assert inf[a] == hf[b], f"两份 config 不一致: inference.{a}={inf[a]} vs hf.{b}={hf[b]}"
print("两份 config 关键字段一致:", ", ".join(f"{a}" for a, _ in pairs))

cfg = inf
head_dim = cfg["head_dim"]                 # 512: 单 KV 头的 latent 维 (448 nope + 64 rope)
index_head_dim = cfg["index_head_dim"]     # 128
ratios = cfg["compress_ratios"]
kv_sources = cfg["kv_source_layers"]
n_layers = cfg["n_layers"]
n_mtp = cfg["n_mtp_layers"]

# ---- 代码常量 ----
MAIN_KV_BLOCK = 16      # model.py L760: fp4_act_quant(latent, 16, True, scale_dtype=e4m3)
INDEX_K_BLOCK = 32      # model.py L546: fp4_act_quant(k, fp4_block_size, True), fp4_block_size=32
FP4_BITS = 4            # E2M1
SCALE_BITS = 8          # e4m3 / e8m0 都是 1 字节


def entry_bytes(dim, block, scale_bits=SCALE_BITS, elem_bits=FP4_BITS):
    """一个压缩条目的字节数 = 元素位宽 + scale (每 block 个通道一个)."""
    elems = dim * elem_bits / 8
    scales = (dim + block - 1) // block * scale_bits / 8
    return elems + scales


main_entry = entry_bytes(head_dim, MAIN_KV_BLOCK)
index_entry = entry_bytes(index_head_dim, INDEX_K_BLOCK)
print("单条目字节数:")
print(f"  主 KV 条目 : {head_dim} 维 fp4 + 每 {MAIN_KV_BLOCK} 通道 1 字节 scale = {main_entry:.0f} B")
print(f"  indexer K  : {index_head_dim} 维 fp4 + 每 {INDEX_K_BLOCK} 通道 1 字节 scale = {index_entry:.0f} B")

# ---- 共享分组: 每个 kv_source 覆盖 [source, 下一个 source - 1] ----
groups = []
for i, s in enumerate(kv_sources):
    end = kv_sources[i + 1] - 1 if i + 1 < len(kv_sources) else n_layers - 1
    rs = set(ratios[s : end + 1])
    groups.append((s, end, end - s + 1, ratios[s], rs))
print("\n跨层共享分组 (来自 kv_source_layer_ids 与代码 '最近发布者' 语义):")
total_main = total_index = 0.0
for s, e, n, r, rs in groups:
    assert len(rs) == 1, f"组 {s}-{e} 内 ratio 不一致: {rs}"
    mb = main_entry / r
    ib = index_entry / r
    total_main += mb
    total_index += ib
    print(f"  层 {s:2d}-{e:2d} ({n:2d} 层, ratio={r}): 主 KV {mb:6.1f} B/token + indexer K {ib:5.1f} B/token")
print(f"\n合计: 主 KV {total_main:.0f} + indexer K {total_index:.0f} = {total_main + total_index:.0f} B/token")
print("报告宣称: 890 bytes/token")
print("一致:", abs(total_main + total_index - 890) < 0.5)

# ---- 各项设计贡献 ----
print("\n各项设计的贡献 (以每 token 字节计):")
base_layers = [l for l in range(n_layers)]
no_share_main = sum(main_entry / (ratios[l] if ratios[l] else 1) for l in range(2, n_layers))
no_share_index = sum(index_entry / (ratios[l] if ratios[l] else 1) for l in kv_sources)


def entry_bytes_fp8(dim, block):
    return dim * 8 / 8 + (dim + block - 1) // block * SCALE_BITS / 8


fp8_main_entry = entry_bytes_fp8(head_dim, MAIN_KV_BLOCK)
fp8_index_entry = entry_bytes_fp8(index_head_dim, INDEX_K_BLOCK)
fp8_main_total = sum(fp8_main_entry / r for _, _, _, r, _ in groups)
fp8_index_total = sum(fp8_index_entry / r for _, _, _, r, _ in groups)
print(f"  若无跨层共享 (层 2-39 各自存主 KV): {no_share_main:.0f} B/token")
print(f"  跨层共享后主 KV: {total_main:.0f} B/token (减少 {no_share_main / total_main:.2f}x)")
print(f"  主 KV 若用 fp8 (条目 {fp8_main_entry:.0f} B): {fp8_main_total:.0f} B/token "
      f"-> FP4 省 {fp8_main_total / total_main:.2f}x")
print(f"  indexer K 若用 fp8 (条目 {fp8_index_entry:.0f} B): {fp8_index_total:.0f} B/token "
      f"-> FP4 省 {fp8_index_total / total_index:.2f}x")

# ---- 与其它代的量级对照 (仅按结构可推的部分) ----
print("\n报告宣称的倍数关系 (需要对方 config 才能独立推导, 此处仅登记):")
print("  vs DeepSeek-V4-Flash: 约 1/4  (报告原文)")
print("  vs DeepSeek-V1:       约 1/437 (报告原文)")
print("  持久化 KV (SSD/host): 约 1/8  (报告原文)")
