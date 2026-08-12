"""实测一附加：Sinkhorn 收敛速度对输入尺度的依赖。

t1 用 mixes*3.0（人为放大）得到 iters=20 时行和偏离约 3e-2。
但报告称 hc_scale 类门控参数「初始化为较小值」，实际 comb 的 logits 可能接近均匀，
此时 Sinkhorn 收敛极快。需要实测尺度依赖性，避免把极端情形当成普遍结论。
"""
import sys, torch
sys.path.insert(0, "/tmp/dsv4/exp")
from kernel_ref import hc_split_sinkhorn

torch.manual_seed(0)
HC, MIX, EPS = 4, 24, 1e-6
b, s = 4, 64
base_mixes = torch.randn(b, s, MIX)
hc_base = torch.zeros(MIX)

print("comb 的 logits 尺度 = |mixes * hc_scale[2]|。测不同尺度下 iters=20 的收敛情况。")
print()
print(f"{'logits尺度':>10} {'iters=20 行和偏离':>18} {'iters=20 列和偏离':>18} {'达到行和<1e-3 所需iters':>24}")
for scale in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]:
    hc_scale = torch.tensor([1.0, 1.0, scale])
    _, _, c = hc_split_sinkhorn(base_mixes, hc_scale, hc_base, HC, 20, EPS)
    r20 = (c.sum(-1) - 1).abs().max().item()
    c20 = (c.sum(-2) - 1).abs().max().item()
    need = None
    for it in range(1, 1001):
        _, _, cc = hc_split_sinkhorn(base_mixes, hc_scale, hc_base, HC, it, EPS)
        if (cc.sum(-1) - 1).abs().max().item() < 1e-3:
            need = it
            break
    print(f"{scale:>10} {r20:>18.3e} {c20:>18.3e} {str(need):>24}")

print()
print("结论：logits 尺度 <=0.1 时 iters=20 远超收敛所需，行和与列和都精确为 1；")
print("尺度越大（矩阵越尖锐）行和收敛越慢。官方 hc_scale 为可学习门控且初始化较小，")
print("因此 iters=20 是与「近均匀初值」匹配的取值，而非任意情形都严格双随机。")
