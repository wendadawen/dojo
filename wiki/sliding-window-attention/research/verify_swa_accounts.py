#!/usr/bin/env python3
"""滑动窗口注意力的账: 缓存字节、感受野、复杂度比 (自包含, 只做算术与规则复算).

参数取值来自 DeepSeek-V4.1-Flash 的官方推理配置
(wiki/deepseek-v4-1/research/official/inference/config.json):
  window_size = 128, head_dim = 512, n_layers = 40, 窗口 KV 为 FP8 (1 B/元素)。
索引规则按官方 model.py 的 get_window_topk_idxs (环形缓冲, slot = 位置 mod W) 复算;
与官方函数本身的逐位置对照见
wiki/deepseek-v4-1/research/ckpt/verify_sparse_attn_window.out (那里直接跑官方函数)。
"""
import json
from pathlib import Path

CFG = json.loads(
    (Path(__file__).resolve().parents[2] / "deepseek-v4-1" / "research" / "official" / "inference" / "config.json").read_text()
)
W = CFG["window_size"]
HEAD_DIM = CFG["head_dim"]
N_LAYERS = CFG["n_layers"]
KV_BYTES = 1          # FP8
MAX_POS = 1048576     # 官方 HF config: max_position_embeddings

print(f"配置: window_size={W}, head_dim={HEAD_DIM}, n_layers={N_LAYERS}, 窗口 KV 精度 FP8({KV_BYTES} B/元素)")

# ---- 1. 环形缓冲的字节账 ----
ring = W * HEAD_DIM * KV_BYTES
print(f"\n[1] 环形缓冲: {W} 槽 x {HEAD_DIM} 维 x {KV_BYTES} B = {ring:,} B = {ring / 1024:.0f} KiB / 层")
print(f"    全部 {N_LAYERS} 层: {ring * N_LAYERS / 1024 / 1024:.2f} MiB (与序列长度无关)")
full = MAX_POS * HEAD_DIM * KV_BYTES
print(f"    若按全注意力缓存 {MAX_POS} 个位置: {full / 1024**3:.1f} GiB / 层")
print(f"    固定窗口相对全缓存的比值: {ring / full:.3e} = 1/{full / ring:.0f}")

# ---- 2. 感受野 ----
print("\n[2] 层堆叠的感受野 (k 层 x W):")
for k in (1, 2, 20, 40):
    print(f"    {k:2d} 层 -> {k * W:6d} 个 token")
print(f"    主干 {N_LAYERS} 层 = {N_LAYERS * W} 个 token, 占 1M 上下文的 {N_LAYERS * W / MAX_POS * 100:.2f}%")
print(f"    编码器 20 层 = {20 * W} 个 token")

# ---- 3. 复杂度 ----
print("\n[3] 每层注意力打分次数 (与序列长度 N 的关系):")
for N in (4096, 131072, MAX_POS):
    print(f"    N={N:>8}: 全注意力 {N * N:>16,} vs 窗口 {N * W:>16,}  ->  {N / W:>7.0f}x 更少")

# ---- 4. 索引规则: prefill 逐 query 因果窗口 vs decode 环形槽位列表 ----
print("\n[4] 索引规则复算 (W=128, 序列长 300):")
seq = 300
hold = {}
mismatch = []
for p in range(seq):
    hold[p % W] = p
    # decode: 官方 get_window_topk_idxs 在 start_pos>0 分支给出的槽位列表
    oldest = p % W + 1
    slots = list(range(oldest, W)) + list(range(oldest))
    slots = [s if s <= p else -1 for s in slots]
    dec_positions = sorted({hold[s] for s in slots if s != -1})
    expect = list(range(max(0, p - W + 1), p + 1))
    if dec_positions != expect:
        mismatch.append((p, dec_positions[:3], expect[:3]))
print(f"    decode 槽位映射回的位置集合 == 期望因果窗口: {not mismatch} (不符 {len(mismatch)} 处)")
print(f"    抽查 p=299: 槽位数 {len(slots)}, 位置范围 [{expect[0]}, {expect[-1]}], 共 {len(expect)} 个")
