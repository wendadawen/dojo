#!/usr/bin/env python3
"""核对 MoE 路由 (Gate) 公式: sqrtsoftplus 打分 + noaux_tc 偏置只影响选择.

核查项:
  1. 打分函数: score = sqrt(softplus(x·W^T / gate_temp)), gate_temp=1.0 -> 等于 sqrt(softplus(x·W^T))
  2. 偏置只进选择: indices = topk(score + bias), 权重取未加偏置的 score
  3. norm_topk_prob: 权重按 top-k 内和归一化 (常数 1e-20)
  4. route_scale=1.5 乘在权重上
  5. 图像 span token 用 bias_vl 替代 bias
  6. 与 softmax 打分的差别 (sqrtsoftplus 分数不构成概率分布)
用真实维度 (384 专家, dim 5120, top-6) 跑官方 Gate 模块, 与独立重实现逐位对照.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from mini_model import load_model_module  # noqa: E402

M = load_model_module()
torch.set_default_dtype(torch.bfloat16)

args = M.ModelArgs(
    n_layers=40, n_routed_experts=384, n_shared_experts=1, n_activated_experts=6,
    dim=5120, score_func="sqrtsoftplus", route_scale=1.5, norm_topk_prob=True,
    vision_n_layers=32, dspark_n_routed_experts=128, dspark_n_activated_experts=3,
)
g = M.Gate(0, args)
torch.manual_seed(0)
with torch.no_grad():
    g.weight.normal_(0, 0.02)
    g.bias.normal_(0, 0.001)
    g.bias_vl.normal_(0, 0.001)

n = 8
x = torch.randn(n, args.dim, dtype=torch.bfloat16)

# ---- 独立重实现 ----
W = g.weight.float()
sc = (x.float() @ W.T) / g.gate_temp
sc = torch.sqrt(torch.nn.functional.softplus(sc))
bias = g.bias
idx_ref = (sc + bias).topk(g.topk, dim=-1)[1]
w_ref = sc.gather(1, idx_ref)
w_ref = w_ref / (w_ref.sum(-1, keepdim=True) + 1e-20) * g.route_scale

w_off, idx_off = g(x)

print("[1] 打分函数与权重:")
print(f"    官方 vs 独立重实现 indices 完全一致: {bool((idx_off == idx_ref).all())}")
print(f"    权重最大差: {(w_off.float() - w_ref).abs().max().item():.3e}")
print(f"    score = sqrt(softplus(xW^T)) 逐元素与手算一致: "
      f"{bool(torch.allclose(torch.sqrt(torch.nn.functional.softplus((x.float() @ W.T))), sc, atol=1e-6))}")
print(f"    权重行和 = route_scale: {w_off.float().sum(-1)[:4].tolist()}")

print("\n[2] 偏置只影响选择, 不影响权重:")
with torch.no_grad():
    g.bias.add_(torch.randn_like(g.bias) * 0.05)
w2, idx2 = g(x)
print(f"    扰动 bias 后 indices 变化比例: {(idx2 != idx_off).float().mean().item():.3f}")
print(f"    扰动 bias 后权重最大差: {(w2.float() - w_off.float()).abs().max().item():.3e} "
      f"(仅因选中不同专家的 score 不同, 未被 bias 缩放)")

print("\n[3] 权重归一化口径:")
raw = sc.gather(1, idx_ref)
print(f"    归一化后行和 (不含 route_scale): {(raw / (raw.sum(-1, keepdim=True) + 1e-20)).sum(-1)[0].detach().item():.6f}")

print("\n[4] route_scale:")
print(f"    route_scale = {g.route_scale}; 权重行和 = {w_off.float().sum(-1).mean().item():.6f}")

print("\n[5] 图像 span 用 bias_vl:")
img_mask = torch.zeros(n, dtype=torch.bool)
img_mask[2:4] = True
w_img, idx_img = g(x, img_mask)
sc_plus_vl = sc + torch.where(img_mask.unsqueeze(-1), g.bias_vl, g.bias)
idx_img_ref = (sc_plus_vl).topk(g.topk, dim=-1)[1]
print(f"    图像 token 的选择与 (score + bias_vl) 的 top-k 一致: {bool((idx_img == idx_img_ref).all())}")
print(f"    文本 token 的选择未受 image_mask 影响: {bool((idx_img[0] == idx2[0]).all())}")

print("\n[6] 与 softmax 打分的差别:")
sm = torch.softmax((x.float() @ W.T), dim=-1)
print(f"    sqrtsoftplus 分数范围: [{sc.min().item():.4f}, {sc.max().item():.4f}] (非概率, 不归一)")
print(f"    softmax 分数范围:      [{sm.min().item():.2e}, {sm.max().item():.4f}] (行和 1)")
print(f"    sqrtsoftplus 分数行和 (第 0 行): {sc[0].sum().item():.2f} != 1 -> 权重必须显式归一化")
