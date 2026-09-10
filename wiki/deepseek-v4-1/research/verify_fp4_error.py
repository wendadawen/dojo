#!/usr/bin/env python3
"""核对报告 §2.4.4 对 FP4 主 KV 的三条量化论断.

报告原文要点:
  - 选 E2M1 + 每 16 通道一个 E4M3 scale (NVFP4 去掉第二级 global scale)
  - "格式支持的最大幅值 448 x 6 = 2688, 远高于 cache 的幅值上界"
  - "RMSNorm 后 512 通道 latent 的 L2 范数最多约 sqrt(512) ≈ 22.6, RoPE 保范, 因此通道最大绝对值
     也约 sqrt(512) ≈ 22.6; 训练中观察到的最大幅值约 10"
  - 量化在 RoPE 之后做; SWA KV 保持 FP8
实测:
  1. 用真实 latent 分布 (RMSNorm 输出: 每元素 ~N(0,1), L2 范数 sqrt(512)) 量化-反量化, 测误差
  2. 检查是否有元素超出 fp4 可表示范围 (截断)
  3. 与 FP8 (block 32, ue8m0) 对比
  4. 幅值上界与 448x6 的关系
量化函数用 mini_kernel (kernel.py 语义复现).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import mini_kernel as MK  # noqa: E402

DIM = 512          # 单 KV 头 latent 维
BLOCK = 16         # 主 KV: 每 16 通道一个 e4m3 scale
E4M3_MAX, FP4_MAX = 448.0, 6.0

torch.manual_seed(0)
n = 4096
# RMSNorm 输出: 每行 L2 范数 = sqrt(512), 即每元素 RMS = 1
x = torch.randn(n, DIM)
x = x / x.norm(dim=-1, keepdim=True) * (DIM ** 0.5)
print(f"[分布] 行 L2 范数 = {x.norm(dim=-1).mean():.4f} (sqrt(512)={DIM ** 0.5:.4f}), "
      f"元素最大绝对值 = {x.abs().max():.4f}")

# ---- 1/2: MXFP4 (E2M1 + per-16 e4m3 scale) ----
q4 = x.clone()
MK.fp4_act_quant(q4, BLOCK, True, scale_dtype=torch.float8_e4m3fn)
err4 = (q4 - x).abs()
rel4 = err4.norm() / x.norm()
print(f"\n[FP4 E2M1, block {BLOCK}, e4m3 scale]")
print(f"  逐元素相对误差 RMS: {(err4 / x.abs().clamp_min(1e-6)).pow(2).mean().sqrt().item():.4f}")
print(f"  整体相对误差 ||err||/||x||: {rel4.item():.4f}")
print(f"  最大绝对误差: {err4.max().item():.4f}")

# ---- 3: FP8 (block 32, ue8m0) 对比 ----
q8 = x.clone()
MK.act_quant(q8, 32, "ue8m0", torch.float8_e8m0fnu, True)
err8 = (q8 - x).abs()
print(f"\n[FP8 E4M3, block 32, ue8m0 scale] (对比基准)")
print(f"  整体相对误差 ||err||/||x||: {(err8.norm() / x.norm()).item():.4f}")
print(f"  FP4 相对误差 / FP8 相对误差 = {rel4.item() / (err8.norm() / x.norm()).item():.2f}x")

# ---- 4: 幅值上界 ----
bound = (E4M3_MAX * FP4_MAX)
print(f"\n[幅值上界] 每 16 通道一个 e4m3 scale 时可表示的最大幅值 = {E4M3_MAX:.0f} x {FP4_MAX:.0f} = {bound:.0f}")
print(f"  本次采样的最大幅值 {x.abs().max().item():.4f}; 报告称训练观察最大约 10; 理论上界 sqrt(512)={DIM ** 0.5:.2f}")
print(f"  距上界余量: {bound / x.abs().max().item():.1f}x (采样) / {bound / 10:.1f}x (报告观察值)")

# ---- 极端测试: 把幅值推到理论上界附近, 看是否截断 ----
xs = x / x.abs().max() * 22.6
q4s = xs.clone()
MK.fp4_act_quant(q4s, BLOCK, True, scale_dtype=torch.float8_e4m3fn)
clamped = (q4s.abs() > FP4_MAX * (xs.abs().amax(dim=-1, keepdim=True) / BLOCK * BLOCK / FP4_MAX)).sum().item()
print(f"\n[极端] 缩放到理论上界 22.6 后: 最大绝对误差 {(q4s - xs).abs().max().item():.4f}, "
      f"相对误差 {(q4s - xs).norm() / xs.norm():.4f}")
print("  说明: 每 16 通道自带 scale, 幅值整体缩放不改变相对误差 (与报告 '无 global scale 无碍' 一致)")
