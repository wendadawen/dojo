#!/usr/bin/env python3
"""核对 Compressor 池化公式与 prefill/decode 状态机等价性.

核查项:
  1. ratio=1: 只做 norm(wkv(x)), 无 gate、无 fp32 提升
  2. ratio>1: pooled = sum_j softmax_j(score) * kv_j, 组内 softmax; 先池化再 cast 回 x.dtype 再 RMSNorm
  3. prefill 整段批量池化 == decode 逐 token 状态机池化 (同一权重下逐位对照)
  4. 边界: prefill 15 token (尾部余 1) 留下的残态 + 1 步 decode 完成最后一组, 与整段 prefill 的第 8 组一致
  5. 组未填满时 decode 返回 None (不产出条目)
用真实维度 (dim 5120, head_dim 512) 跑官方 Compressor.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from mini_model import load_model_module  # noqa: E402

M = load_model_module()
torch.set_default_dtype(torch.bfloat16)

DIM, HD = 5120, 512
args = M.ModelArgs(dim=DIM, head_dim=HD, n_layers=40, compress_ratios=tuple([0, 0] + [2] * 18 + [1] * 20 + [0] * 3),
                   max_batch_size=1, max_seq_len=1024)

torch.manual_seed(0)


def build(layer_id):
    torch.manual_seed(0)  # 每次构建同一份权重, 否则各实例不可比
    c = M.Compressor(args, layer_id)
    with torch.no_grad():
        c.wkv.weight.normal_(0, 0.02)
        if c.compress_ratio > 1:
            c.wgate.weight.normal_(0, 0.02)
        c.norm.weight.fill_(1.0)
    return c


x = torch.randn(1, 16, DIM, dtype=torch.bfloat16) * 0.5

# ---- ratio = 2 ----
c2 = build(2)
assert c2.compress_ratio == 2
pre = c2(x, 0)
print(f"[2] ratio=2 prefill 16 token -> 输出 {tuple(pre.shape)} (8 组)")
print(f"    wkv dtype={c2.wkv.weight.dtype} (ratio>1 应为 fp32), "
      f"wgate 存在={hasattr(c2, 'wgate')}, 残态 buffer 存在={hasattr(c2, 'kv_state')}")

# 独立重实现
xf = x.float()
kv_all = c2.wkv(xf)      # [1,16,512] fp32
sc_all = c2.wgate(xf)
kv_g = kv_all.unflatten(1, (-1, 2))
sc_g = sc_all.unflatten(1, (-1, 2))
pooled = (kv_g * sc_g.softmax(dim=2)).sum(dim=2)
ref = c2.norm(pooled.to(torch.bfloat16))
print(f"    独立重实现 (softmax 池化 -> cast bf16 -> RMSNorm) 最大差: "
      f"{(pre.float() - ref.float()).abs().max().item():.3e}")

# decode 状态机
c2d = build(2)
outs, none_steps = [], []
for p in range(16):
    o = c2d(x[:, p : p + 1], p)
    if o is None:
        none_steps.append(p)
    else:
        outs.append(o)
dec = torch.cat(outs, dim=1)
print(f"    decode 逐 token: 产出 {tuple(dec.shape)}, 返回 None 的步 = {none_steps}")
print(f"    prefill 批量 vs decode 状态机 最大差: {(pre.float() - dec.float()).abs().max().item():.3e}")
print(f"    逐位一致: {torch.equal(pre, dec)}")
print(f"    输出中不一致元素: {(pre != dec).sum().item()}/{pre.numel()} (量级 {pre.abs().mean().item():.3f})")

# 把对照下沉到 fp32 层面: 池化数学本身是否一致
kv_b = c2.wkv(x.float())      # m=16 的 GEMM
sc_b = c2.wgate(x.float())
pool_b = (kv_b.unflatten(1, (-1, 2)) * sc_b.unflatten(1, (-1, 2)).softmax(2)).sum(2)
kv_d = torch.cat([c2.wkv(x[:, p : p + 1].float()) for p in range(16)], dim=1)   # m=1 的 GEMV
sc_d = torch.cat([c2.wgate(x[:, p : p + 1].float()) for p in range(16)], dim=1)
pool_d = (kv_d.unflatten(1, (-1, 2)) * sc_d.unflatten(1, (-1, 2)).softmax(2)).sum(2)
print(f"    fp32 层面池化结果最大差 (批量 GEMM vs 逐 token GEMV): "
      f"{(pool_b - pool_d).abs().max().item():.3e} (相对 {((pool_b - pool_d).abs().max() / pool_b.abs().max()).item():.2e})")
print(f"    池化结果 cast 到 bf16 后不一致元素: {(pool_b.to(torch.bfloat16) != pool_d.to(torch.bfloat16)).sum().item()}"
      f"/{pool_b.numel()} (1 个 bf16 ulp ≈ {2 ** -8 * pool_b.abs().max().item():.3e})")
print("    结论: 池化数学一致; 差异来自 fp32 GEMM 形状相关的累加序 (~1e-7) 经 bf16 舍入放大到 1 ulp")

# ---- 边界: prefill 15 + decode 1 ----
c2b = build(2)
p15 = c2b(x[:, :15], 0)          # 7 组, 尾部 1 个 token 进残态
print(f"\n[4] prefill 15 token -> {tuple(p15.shape)} (7 组), 残态已填 1 个槽")
o15 = c2b(x[:, 15:16], 15)       # 组 7 完成
print(f"    第 16 步产出: {None if o15 is None else tuple(o15.shape)}")
print(f"    与整段 prefill 的第 8 组逐位一致: {torch.equal(o15, pre[:, 7:8])}, "
      f"最大差 {(o15.float() - pre[:, 7:8].float()).abs().max().item():.3e}")

# ---- ratio = 1 ----
c1 = build(20)
assert c1.compress_ratio == 1
o1 = c1(x, 0)
print(f"\n[1] ratio=1 prefill 16 token -> {tuple(o1.shape)} (逐 token, 无压缩)")
print(f"    wkv dtype={c1.wkv.weight.dtype} (应为 bf16), wgate 存在={hasattr(c1, 'wgate')}, "
      f"残态 buffer 存在={hasattr(c1, 'kv_state')}")
ref1 = c1.norm(c1.wkv(x))
print(f"    等于 norm(wkv(x)): {torch.equal(o1, ref1)}")

# ---- 组未填满语义 ----
c2e = build(2)
steps = [c2e(x[:, p : p + 1], p) for p in range(4)]
print(f"\n[5] 前 4 步产出 None 的模式: {[s is None for s in steps]} (奇数步完成一组: 步 1、3 产出)")
