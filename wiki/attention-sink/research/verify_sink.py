#!/usr/bin/env python3
"""注意力汇聚点的语义与真实参数形状核对.

三部分:
  1. 解析解: 汇聚点项只进 softmax 分母 (构造 q=0, 两个等权槽位, 改变 sink)
  2. 边界: -1 槽位不贡献; 整行 -1 时输出 0 (汇聚点项被行最大值推成 inf)
  3. 真实 checkpoint: attn_sink 张量的数量与形状 (HTTP Range 读到的分片头)

官方 kernel 是 tilelang CUDA 内核, 本机无法执行; 第 1、2 部分跑的是
wiki/deepseek-v4-1/research/mini_kernel.py —— 它逐行对照 kernel.py L310-403
(sparse_attn_kernel) 的语义复现。
"""
import json
import sys
from pathlib import Path

import torch

DSV41 = Path(__file__).resolve().parents[2] / "deepseek-v4-1" / "research"
sys.path.insert(0, str(DSV41))
import mini_kernel as MK  # noqa: E402

d, h = 4, 1
q = torch.zeros(1, 1, h, d)                      # 分数全 0 -> 每个可见槽位等权
v1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
v2 = torch.tensor([3.0, 4.0, 5.0, 6.0])
kv = torch.stack([v1, v2, torch.tensor([9.0] * 4)]).view(1, 3, d)
idx_two = torch.tensor([[[0, 1]]])
idx_null = torch.tensor([[[-1, -1, -1]]])

print("[1] 汇聚点项只进分母: 输出 = (v1+v2) / (2 + exp(sink))")
for sink in (-float("inf"), 0.0, 2.0, 5.0):
    o = MK.sparse_attn(q, kv, torch.tensor([sink]), idx_two, 1.0)[0, 0, 0]
    denom = 2 + (0.0 if sink == -float("inf") else torch.exp(torch.tensor(sink)))
    ref = (v1 + v2) / denom
    print(f"    sink={sink:>6}: 输出 {o.tolist()}  解析解 {ref.tolist()}  最大差 {(o - ref).abs().max().item():.2e}")

print("\n[2] 边界: 整行 -1 的输出 =", MK.sparse_attn(q, kv, torch.tensor([0.0]), idx_null, 1.0).flatten().tolist())

print("\n[3] 汇聚点在分母中的占比 (W=128 个窗口槽位, 分数同为 0):")
W = 128
for sink in (0.0, 2.0, 5.0, 8.0):
    share = torch.exp(torch.tensor(sink)) / (W + torch.exp(torch.tensor(sink)))
    print(f"    sink={sink:>4}: exp(sink)/({W}+exp(sink)) = {share.item():.4f} = {share.item() * 100:.2f}%")
print("    说明: 汇聚点 logit 越大, 它吃掉的注意力份额越多; 128 个等权槽位下 sink=0 只占 0.78%")

print("\n[4] 真实 checkpoint 的 attn_sink (HTTP Range 读到的分片头):")
hdr = json.loads((DSV41 / "ckpt" / "headers.json").read_text())
ks = [k for k in hdr if k.endswith("attn_sink")]
backbone = sorted((int(k.split(".")[1]) for k in ks if k.startswith("layers.")))
mtp = sorted((int(k.split(".")[1]) for k in ks if k.startswith("mtp.")))
shapes = {tuple(hdr[k]["shape"]) for k in ks}
dtypes = {hdr[k]["dtype"] for k in ks}
print(f"    张量总数 {len(ks)} (主干 {len(backbone)} 层 + MTP {len(mtp)} 层), "
      f"形状集合 {shapes}, dtype 集合 {dtypes}")
print(f"    主干层号 {backbone[0]}..{backbone[-1]} 连续: {backbone == list(range(40))}")
print("    含义: 每层每个注意力头一个可学习标量 (64 头 -> 形状 [64], fp32)")
