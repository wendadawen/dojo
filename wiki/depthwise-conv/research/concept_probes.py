"""
四份概念页共用的数值验证探针。每节聚焦一个论断，输出直接供页面引用。
运行环境：系统 python3 + torch 2.8.0（CPU）。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)


def banner(t):
    print("=" * 78)
    print(t)
    print("=" * 78)


# ============================================================
# 探针 A：Sinkhorn-Knopp 与双随机矩阵（hyper-connections 页）
# ============================================================
banner("探针 A：Sinkhorn-Knott 与双随机矩阵")

# A1. 一个 3x3 正矩阵手动跑 Sinkhorn：先列后行，观察行列和收敛
M0 = torch.tensor([[3.0, 1.0, 0.5],
                   [0.2, 4.0, 1.0],
                   [1.0, 0.3, 5.0]])
print("[A1] 初始正矩阵 M0 =")
print(M0)
M = M0.clone()
for t in range(1, 6):
    M = M / M.sum(dim=0, keepdim=True)   # 列归一化（每列和为 1）
    M = M / M.sum(dim=1, keepdim=True)   # 行归一化（每行和为 1）
    print(f"  迭代 {t}: 行和 = {M.sum(1).tolist()}, 列和 = {[round(v,6) for v in M.sum(0).tolist()]}")
for t in range(6, 21):
    M = M / M.sum(dim=0, keepdim=True)
    M = M / M.sum(dim=1, keepdim=True)
print(f"  迭代 20: 行和最大偏差 {float((M.sum(1)-1).abs().max()):.3e}, "
      f"列和最大偏差 {float((M.sum(0)-1).abs().max()):.3e}")
print(f"  全部元素非负: {bool((M >= 0).all())}")

# A2. 复合封闭性：两个双随机矩阵的乘积仍是双随机（数值验证）
D1 = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
D2 = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.4, 0.3, 0.3]])
P = D1 @ D2
print(f"\n[A2] 双随机×双随机: 乘积行和 {P.sum(1).tolist()}, 列和 {[round(v,10) for v in P.sum(0).tolist()]}")
print(f"     偏差量级 {(P.sum(1)-1).abs().max():.1e}（浮点舍入级别）")

# A3. 谱范数：双随机矩阵的谱范数（最大奇异值）≤ 1
sv = torch.linalg.svdvals(M)
print(f"\n[A3] Sinkhorn 结果的最大奇异值 = {float(sv[0]):.6f}  ≤ 1: {bool(sv[0] <= 1.0 + 1e-9)}")
sv2 = torch.linalg.svdvals(D1 @ D2 @ M)
print(f"     三个双随机矩阵连乘的最大奇异值 = {float(sv2[0]):.6f}")

# A4. 对照：无约束矩阵链的行和放大（HC 不稳定的机制演示，构造示例）
print("\n[A4] 构造示例：无约束随机矩阵链 vs 双随机矩阵链的行和增长")
print("     每层 H_res 从 N(0,1) 采样（4x4），链 24 层，看复合矩阵行和：")
G = torch.eye(4)
for _ in range(24):
    G = G @ torch.randn(4, 4)
print(f"     无约束链 24 层: 复合矩阵行和绝对值最大 = {float(G.abs().sum(1).max()):.3e}")
G2 = torch.eye(4)
for _ in range(24):
    # 每层做一个随机双随机矩阵（先随机正矩阵再 Sinkhorn 20 次）
    X = torch.rand(4, 4) + 0.1
    for _ in range(20):
        X = X / X.sum(0, keepdim=True)
        X = X / X.sum(1, keepdim=True)
    G2 = G2 @ X
print(f"     双随机链 24 层: 复合矩阵行和 = {G2.sum(1).tolist()}")
print("     注：随机矩阵乘积的幅度随层数指数变化，量级不稳定；双随机链的行和恒为 1")

# A5. n=1 退化：1x1 双随机矩阵只能是 [1]
print(f"\n[A5] 1x1 双随机矩阵（行和=列和=1 的非负数）只能是 [1]，即恒等映射")

# ============================================================
# 探针 B：RMSNorm 与 LayerNorm（rmsnorm 页）
# ============================================================
banner("探针 B：RMSNorm vs LayerNorm")


def layernorm(a, g=None, b=None, eps=0.0):
    mu = a.mean()
    sigma = ((a - mu) ** 2).mean().sqrt()
    out = (a - mu) / sigma
    return out * g + b if g is not None else out


def rmsnorm(a, g=None, b=None, eps=0.0):
    rms = (a ** 2).mean().sqrt()
    out = a / rms
    return out * g + b if g is not None else out


a = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"[B1] a = {a.tolist()}")
print(f"     LayerNorm: mu={float(a.mean()):.4f}, sigma={float(((a-a.mean())**2).mean().sqrt()):.4f}, "
      f"输出 = {[round(v,6) for v in layernorm(a).tolist()]}")
print(f"     RMSNorm:   RMS={float((a**2).mean().sqrt()):.4f}, "
      f"输出 = {[round(v,6) for v in rmsnorm(a).tolist()]}")

# B2. 零均值时两者相等（论文论断的数值验证）
a0 = torch.tensor([1.0, -1.0, 3.0, -3.0])
d = (layernorm(a0) - rmsnorm(a0)).abs().max()
print(f"\n[B2] 零均值向量 {a0.tolist()}: LayerNorm 与 RMSNorm 最大差 = {float(d):.3e}")

# B3. 与 PyTorch 官方实现对齐（nn.RMSNorm 在 torch 2.4+ 存在）
try:
    ref = torch.nn.RMSNorm(4, eps=0.0)
    with torch.no_grad():
        ref.weight.fill_(1.0)
    print(f"[B3] torch.nn.RMSNorm 输出 = {[round(v,6) for v in ref(a).tolist()]}")
    print(f"     与手写 RMSNorm 最大差 = {float((ref(a)-rmsnorm(a)).abs().max()):.3e}")
except AttributeError:
    print("[B3] 本机 torch 无 nn.RMSNorm")

# B4. GLM 实现：fp32 内部计算 + eps（与 Glm5NextTextRMSNorm 语义一致）
def glm_rmsnorm(x, weight, eps):
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return weight * x32.to(x.dtype)


w = torch.ones(4)
print(f"\n[B4] GLM 式（fp32+eps=1e-5）输出 = {[round(v,6) for v in glm_rmsnorm(a, w, 1e-5).tolist()]}")
print(f"     与论文公式（无 eps）差 = {float((glm_rmsnorm(a,w,1e-5)-rmsnorm(a)).abs().max()):.3e}")
print("     论文 v1 公式不含 eps；实现普遍加 eps 保证除零安全")

# B5. 运算量对比（计数推导的数值验证：n=4096）
n = 4096
print(f"\n[B5] n={n} 时：LayerNorm 比 RMSNorm 多的标量运算 ≈ 3n = {3*n} 次"
      f"（求均值 n 加法 + 方差 n 减法 + 归一化 n 减法）")

# ============================================================
# 探针 C：深度可分离卷积（depthwise-conv 页）
# ============================================================
banner("探针 C：深度可分离卷积")

# C1. 1D depthwise 卷积（KDA 用法）：groups=通道数，每通道独立核
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).view(1, 1, -1)   # 1 通道
k = torch.tensor([1.0, 0.5, -0.5, -1.0]).view(1, 1, 4)        # kernel=4
conv = torch.nn.Conv1d(1, 1, kernel_size=4, groups=1, bias=False,
                       padding=3)   # causal：右 padding
with torch.no_grad():
    conv.weight.copy_(k)
out = conv(x)[:, :, 3:8]    # 去掉左 padding 产生的输出，只留因果部分
print(f"[C1] 因果 depthwise conv1d（kernel=4）: 输入 {x.flatten().tolist()}")
print(f"     手算 t=3（0 起）: 1*1 + 2*0.5 + 3*(-0.5) + 4*(-1) = {1*1+2*0.5+3*(-0.5)+4*(-1)}")
print(f"     torch 输出 = {[round(v,4) for v in out.flatten().tolist()]}")

# C2. groups=C 验证：2 通道各用各的核
x2 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]).view(1, 2, 4)
dw = torch.nn.Conv1d(2, 2, kernel_size=2, groups=2, bias=False, padding=1)
with torch.no_grad():
    dw.weight.copy_(torch.tensor([[[1.0, 1.0]], [[0.0, 1.0]]]))   # ch0: 求和, ch1: 右移
o2 = dw(x2)[:, :, 1:5]
print(f"\n[C2] groups=2: 通道 0（核 [1,1]）输出 {[round(v,1) for v in o2[0,0].tolist()]}, "
      f"通道 1（核 [0,1]）输出 {[round(v,1) for v in o2[0,1].tolist()]}")
print("     通道 1 的核不碰通道 0 的数据——逐通道独立")

# C3. FLOP 对比（Eq 2 vs Eq 5 的具体数字）
Dk, M, N, Df = 3, 64, 64, 16
std_cost = Dk * Dk * M * N * Df * Df
sep_cost = Dk * Dk * M * Df * Df + M * N * Df * Df
print(f"\n[C3] D_K={Dk}, M={M}, N={N}, D_F={Df}:")
print(f"     标准卷积: {std_cost:,} 次乘加")
print(f"     深度可分离: {sep_cost:,} 次（depthwise {Dk*Dk*M*Df*Df:,} + pointwise {M*N*Df*Df:,}）")
print(f"     比值 = {sep_cost/std_cost:.6f} = 1/N + 1/D_K^2 = {1/N + 1/Dk**2:.6f}")
print(f"     缩减倍数 = {std_cost/sep_cost:.1f}x")

# ============================================================
# 探针 D：E4M3 与块量化（fp8-block-quant 页）
# ============================================================
banner("探针 D：E4M3 与块量化")

# D1. torch 的 E4M3 finfo（对应论文 Table 1）
fi = torch.finfo(torch.float8_e4m3fn)
print(f"[D1] torch float8_e4m3fn: max={fi.max}, min(负)={fi.min}, "
      f"tiny(最小正正规数)={fi.tiny:.3e}")
print(f"     2^-6 = {2.0**-6:.3e}, 2^-9 = {2.0**-9:.3e}")
print(f"     448 = 1.75 * 2^8 = {1.75 * 2**8}")

# D2. 位模式解码验证：S.1111.110 → 1.75 × 2^8
bits = 0b01111110   # 符号0 指数1111(=15) 尾数110
s = bits >> 7
e = (bits >> 3) & 0b1111
m = bits & 0b111
val = (-1) ** s * 2 ** (e - 7) * (1 + m / 8)
print(f"\n[D2] 位模式 0.1111.110: 指数域={e}, 偏置 7, 尾数={m}/8")
print(f"     值 = (-1)^{s} × 2^{e}-7 × (1+{m}/8) = {val}")

# D3. 块量化 roundtrip（transformers quantizer 语义复现）
W = torch.tensor([[1.0, -6.0, 0.01, 2.0],
                  [0.5, 0.02, -0.008, 4.0],
                  [3.0, -0.001, 0.03, -1.0],
                  [0.007, 5.0, 0.02, 0.5]])
max_abs = W.abs().amax()
scale = fi.max / max_abs
Wq = torch.clamp(W * scale, min=fi.min, max=fi.max).to(torch.float8_e4m3fn)
W_back = Wq.float() / scale
err = (W_back - W).abs()
print(f"\n[D3] 4x4 块量化（scale = 448/max_abs = 448/{max_abs:.3f} = {scale:.4f}）:")
print(f"     反量化绝对误差: max={float(err.max()):.6f}, mean={float(err.mean()):.6f}")
print(f"     相对误差(max): {float((err / W.abs()).max()) * 100:.3f}%")
print(f"     注：0.001 与 6.0 同块，小值几乎丢失——块内动态范围是块量化的精度边界")

# D4. scale 张量形状公式（GLM checkpoint 语义）
for shape in [(2048, 4096), (16384, 1536), (12288, 4096)]:
    rows, cols = shape
    exp_shape = (math.ceil(rows / 128), math.ceil(cols / 128))
    print(f"[D4] 权重 {shape} → scale {exp_shape} = ceil({rows}/128)×ceil({cols}/128)")

# D5. 平均每参数字节数（混合精度下的换算）
print(f"\n[D5] FP8 权重 1 字节/参数；块 scale 每块 4 字节(fp32)")
print(f"     2048x4096 块：权重 {(2048*4096):,} B + scale {16*32*4} B → "
      f"平均 {1 + 16*32*4/(2048*4096):.6f} B/参数")
print("     scale 的存储开销随块大小增大而摊薄；GLM 实测全模型平均 1.022 B/参数")
