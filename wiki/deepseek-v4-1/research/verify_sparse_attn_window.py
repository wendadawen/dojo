#!/usr/bin/env python3
"""核对 sparse_attn 槽位语义 + 窗口环形缓冲的 prefill/decode 位置等价.

核查项 (对照 kernel.py L310-403 sparse_attn_kernel 的语义):
  1. attn_sink 只进分母, 不参与 max; 构造解析解验证
  2. topk_idxs == -1 的槽位: 不贡献分子分母
  3. 整行 -1: 输出 0
  4. 窗口环形缓冲: decode 步给出的槽位序列, 映射回的"位置集合"与 prefill 同 query 的因果窗口一致
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import mini_kernel as MK  # noqa: E402
from mini_model import load_model_module  # noqa: E402

M = load_model_module()

# ---- 1-3: 构造解析解 ----
d, h = 4, 1
q = torch.zeros(1, 1, h, d)          # 分数全 0 -> softmax 权重均等
v1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
v2 = torch.tensor([3.0, 4.0, 5.0, 6.0])
kv = torch.stack([v1, v2, torch.tensor([9.0, 9.0, 9.0, 9.0])]).view(1, 3, d)

idx_two = torch.tensor([[[0, 1]]])
idx_two_plus_null = torch.tensor([[[0, 1, -1]]])
idx_all_null = torch.tensor([[[-1, -1, -1]]])

print("[1] attn_sink 只进分母 (解析: 输出 = (v1+v2)/(2+exp(sink)))")
for sink in (-float("inf"), 0.0, 2.0):
    o = MK.sparse_attn(q, kv, torch.tensor([sink]), idx_two, 1.0)[0, 0, 0]
    ref = (v1 + v2) / (2 + (0.0 if sink == -float("inf") else torch.exp(torch.tensor(sink))))
    print(f"    sink={sink:>6}: 输出 {o.tolist()}  解析解 {ref.tolist()}  最大差 {(o - ref).abs().max().item():.2e}")

print("\n[2] -1 槽位不贡献:")
o2 = MK.sparse_attn(q, kv, torch.tensor([0.0]), idx_two, 1.0)
o2n = MK.sparse_attn(q, kv, torch.tensor([0.0]), idx_two_plus_null, 1.0)
print(f"    两槽 vs 两槽+一个 -1 槽: 最大差 {(o2 - o2n).abs().max().item():.2e} (应为 0)")
o3 = MK.sparse_attn(q, kv, torch.tensor([0.0]), idx_all_null, 1.0)
print(f"\n[3] 整行 -1 的输出: {o3.flatten().tolist()} (应为 0)")

# ---- 4: 窗口环形缓冲等价 ----
win, P = 8, 20
print(f"\n[4] 窗口环形缓冲 (window_size={win}), 序列长度 {P}")
slots_hold = {}  # slot -> 该槽位当前保存的位置
ok = True
for p in range(P):
    slots_hold[p % win] = p          # 写入位置 p
    if p == 0:
        pre_idxs = M.get_window_topk_idxs(win, 1, p + 1, 0)[0]      # prefill: 每 query 一行
        dec_idxs = M.get_window_topk_idxs(win, 1, 1, p)[0, 0]
    else:
        dec_idxs = M.get_window_topk_idxs(win, 1, 1, p)[0, 0]
    dec_positions = sorted({slots_hold[s] for s in dec_idxs.tolist() if s != -1})
    pre_positions = sorted(pre_idxs[p].tolist()) if p < pre_idxs.size(0) else None
    expect = list(range(max(0, p - win + 1), p + 1))
    if dec_positions != expect:
        ok = False
        print(f"    step {p}: decode 位置集合 {dec_positions} != 期望因果窗口 {expect}")
    if p < pre_idxs.size(0):
        if pre_positions != expect:
            ok = False
            print(f"    step {p}: prefill 位置集合 {pre_positions} != 期望 {expect}")
print(f"    decode 槽位映射回的位置集合与 prefill 因果窗口一致 (所有 {P} 步): {ok}")

# 槽位顺序 (不影响结果, 但记录)
p = 12
print(f"    step {p}: decode 槽位序列 {M.get_window_topk_idxs(win, 1, 1, p)[0, 0].tolist()}"
      f" -> 位置 {[slots_hold[s] for s in M.get_window_topk_idxs(win, 1, 1, p)[0, 0].tolist()]}")
print(f"    prefill query {p} 的索引 {M.get_window_topk_idxs(win, 1, p + 1, 0)[0, p].tolist()}")
