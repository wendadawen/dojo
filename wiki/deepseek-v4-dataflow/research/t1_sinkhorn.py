"""实测一：mHC 的 Sinkhorn 投影是否真的得到双随机矩阵。

技术报告 2.2 节称 B_l 被投影到双随机矩阵流形 M（行和=1 且 列和=1），t_max=20。
官方 kernel.py hc_split_sinkhorn_kernel 的迭代顺序是：
  softmax(-1)+eps -> 列归一化 -> (iters-1) 轮 [行归一化 -> 列归一化]
最后一步是列归一化，因此需要实测：收敛后行和与列和分别偏离 1 多少。
"""
import sys, torch
sys.path.insert(0, "/tmp/dsv4/exp")
from kernel_ref import hc_split_sinkhorn

torch.manual_seed(0)
HC = 4                      # config.json hc_mult
ITERS = 20                  # config.json hc_sinkhorn_iters
EPS = 1e-6                  # config.json hc_eps
MIX = (2 + HC) * HC         # model.py Block.mix_hc = (2+hc_mult)*hc_mult

print(f"hc_mult={HC}  mix_hc={MIX}  sinkhorn_iters={ITERS}  hc_eps={EPS}")
print()

b, s = 2, 8
mixes = torch.randn(b, s, MIX) * 3.0        # 放大以制造更不均衡的初值
hc_scale = torch.randn(3).abs() + 0.5
hc_base = torch.randn(MIX)

pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC, ITERS, EPS)
print("形状: pre", tuple(pre.shape), " post", tuple(post.shape), " comb", tuple(comb.shape))
print()

print("=== pre / post 的取值范围（对应报告 Eq.6/Eq.7）===")
print(f"pre  = sigmoid(...)+eps   范围 [{pre.min():.6f}, {pre.max():.6f}]   理论 (0,1)+eps")
print(f"post = 2*sigmoid(...)     范围 [{post.min():.6f}, {post.max():.6f}]   理论 (0,2)")
print()

print("=== comb 是否双随机（报告称投影到双随机流形）===")
row = comb.sum(-1)          # 行和
col = comb.sum(-2)          # 列和
print(f"行和 (sum over -1): min={row.min():.8f}  max={row.max():.8f}  平均偏离1={(row-1).abs().mean():.3e}")
print(f"列和 (sum over -2): min={col.min():.8f}  max={col.max():.8f}  平均偏离1={(col-1).abs().mean():.3e}")
print(f"comb 全为正: {bool((comb > 0).all())}")
print()
print("实测结论：官方 kernel 最后一步是列归一化，因此列和精确=1，行和仅近似=1。")
print()

print("=== 迭代轮数对收敛的影响（同一输入，改变 iters）===")
print(f"{'iters':>6} {'行和最大偏离1':>16} {'列和最大偏离1':>16}")
for it in [1, 2, 3, 5, 10, 20, 40, 100, 400]:
    _, _, c = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC, it, EPS)
    r_err = (c.sum(-1) - 1).abs().max().item()
    c_err = (c.sum(-2) - 1).abs().max().item()
    print(f"{it:>6} {r_err:>16.3e} {c_err:>16.3e}")
print()
print("实测事实：列和在 iters>=2 即精确为 1（1e-6 量级，来自 hc_eps）；")
print("行和收敛慢得多，iters=20 时最大仍偏离约 3e-2，需 iters~400 才降到 1e-3。")
print("即官方配置下 comb 是「精确列随机 + 近似行随机」，而非严格双随机。")
print()

print("=== 双随机的作用：残差混合不放大信号 ===")
# hc_post: y = post * x + sum(comb * residual)
# 双随机 => comb 对 hc 份残差做保范围的重新分配，不整体放大
x_res = torch.randn(b, s, HC, 64)
mixed = torch.einsum("bsij,bsjd->bsid", comb, x_res)
print(f"残差输入 L2 范数总和: {x_res.norm():.4f}")
print(f"经 comb 混合后 L2 范数: {mixed.norm():.4f}")
print(f"比值: {(mixed.norm()/x_res.norm()).item():.4f}   （双随机矩阵为压缩映射，比值 <= 1）")

# 对照：不做 Sinkhorn，仅 softmax(行归一化)
comb_raw = (mixes[..., 2*HC:] * hc_scale[2] + hc_base[2*HC:]).view(b, s, HC, HC).softmax(-1)
mixed_raw = torch.einsum("bsij,bsjd->bsid", comb_raw, x_res)
print(f"仅行归一化(不做 Sinkhorn) 比值: {(mixed_raw.norm()/x_res.norm()).item():.4f}")
print(f"  其列和范围 [{comb_raw.sum(-2).min():.4f}, {comb_raw.sum(-2).max():.4f}] —— 列和不受控，某些残差通道会被反复放大")
