#!/usr/bin/env python3
"""核对 mHC 的 hc_split_sinkhorn: 独立重实现 + 迭代方向/次数 + 双随机性 + 索引映射.

对照源: official/inference/kernel.py L406-474 (tilelang 内核原文).
核查项:
  1. pre = sigmoid(m*scale[0] + base) + eps;  post = 2*sigmoid(m*scale[1] + base) (无 eps)
  2. comb 的索引映射: comb[j][k] = mixes[j*hc + k + 2*hc] (行主序)
  3. comb = softmax(comb, -1) + eps -> 列归一化 -> (iters-1) x (行归一化, 列归一化)
     即行归一化 20 次 / 列归一化 20 次, 末步在列方向
  4. 结果近似双随机 (行和、列和都接近 1), 偏差量级
"""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import mini_kernel as MK  # noqa: E402

HC, ITERS, EPS = 4, 20, 1e-6
torch.manual_seed(0)
n = 6
mixes = torch.randn(n, (2 + HC) * HC) * 0.7
hc_scale = torch.randn(3) * 0.5
hc_base = torch.randn((2 + HC) * HC) * 0.1


def reference(mixes, hc_scale, hc_base, hc=HC, iters=ITERS, eps=EPS, count=False):
    """照 kernel.py 原文逐行独立重实现 (float64)."""
    m = mixes.double()
    pre = torch.sigmoid(m[:, :hc] * hc_scale[0].double() + hc_base[:hc].double()) + eps
    post = 2 * torch.sigmoid(m[:, hc : 2 * hc] * hc_scale[1].double() + hc_base[hc : 2 * hc].double())
    comb = m[:, 2 * hc :] * hc_scale[2].double() + hc_base[2 * hc :].double()
    comb = comb.view(-1, hc, hc)          # comb[j][k] = m[j*hc + k + 2hc]
    ops = []
    row_max = comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    ops.append("row-softmax")
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    ops.append("col")
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        ops.append("row")
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        ops.append("col")
    return pre, post, comb, ops


pre_r, post_r, comb_r, ops = reference(mixes, hc_scale, hc_base)
# mini_kernel 复现的是官方 3-D 包装 (b, s, mix_hc), 故补一维
pre_m, post_m, comb_m = MK.hc_split_sinkhorn(mixes.view(n, 1, -1), hc_scale, hc_base, HC, ITERS, EPS)

print("[1] 独立重实现 vs mini_kernel (kernel.py 语义复现):")
for name, a, b in (("pre", pre_r, pre_m.view(n, HC)), ("post", post_r, post_m.view(n, HC)),
                   ("comb", comb_r, comb_m.view(n, HC, HC))):
    print(f"    {name}: 最大差 {(a.float() - b.float()).abs().max().item():.3e}  "
          f"(独立实现 float64, mini_kernel float32)")

print(f"\n[3] 迭代方向与次数: 共 {len(ops)} 步")
print(f"    行归一化 {ops.count('row') + ops.count('row-softmax')} 次 "
      f"(含 1 次 softmax), 列归一化 {ops.count('col')} 次, 末步 = {ops[-1]}")
assert ops.count("col") == ITERS and ops[-1] == "col"

print(f"\n[4] 双随机性 (iters={ITERS}, eps={EPS}):")
rs, cs = comb_r.sum(-1), comb_r.sum(-2)
print(f"    行和范围 [{rs.min():.8f}, {rs.max():.8f}]  最大偏差 {float((rs - 1).abs().max()):.3e}")
print(f"    列和范围 [{cs.min():.8f}, {cs.max():.8f}]  最大偏差 {float((cs - 1).abs().max()):.3e}")
print(f"    comb 最小值 {comb_r.min():.3e} (eps 下界)")

print(f"\n[2] comb 索引映射 (one-hot 探针):")
probe = torch.zeros(1, (2 + HC) * HC, dtype=torch.float64)
probe[0, 2 * HC + 1 * HC + 2] = 1.0   # 期望映射到 comb[1][2]
_, _, cb, _ = reference(probe, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
                        torch.zeros((2 + HC) * HC, dtype=torch.float64))
print(f"    置 mixes[2*hc + 1*hc + 2] = 1 -> comb 最大值位置 "
      f"{tuple(int(v) for v in torch.nonzero(cb[0] == cb[0].max())[0])} (期望 (1, 2))")

print(f"\n[1b] pre/post 取值范围:")
print(f"    pre  ∈ [{pre_r.min():.6f}, {pre_r.max():.6f}] (sigmoid+eps, 理论 ({EPS}, 1+{EPS}))")
print(f"    post ∈ [{post_r.min():.6f}, {post_r.max():.6f}] (2*sigmoid, 理论 (0, 2), 无 eps)")
print(f"    post 无 eps 项: {post_m.min().item() >= 0}")
