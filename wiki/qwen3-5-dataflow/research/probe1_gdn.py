"""实测 1：GatedDeltaNet 分块（prefill）与递归（decode）等价性 + 手算复核 + 状态占用。

验证目标（对应源码 modeling_qwen3_5.py L249-380, L387-546）：
  1A. torch_chunk_gated_delta_rule（prefill 路径）与逐 token 递归（decode 路径）数值等价
  1B. 2 token 正交键手算复核递归状态更新公式（delta rule）
  1C. 397B 真实配置下的每层递归状态与卷积状态占用
  1D. beta=sigmoid(b)、g=-exp(A_log)*softplus(a+dt_bias) 的数值范围
依赖：torch（系统 python 自带 2.8.0）。Qwen3.5 真实维度：16 k头×128，64 v头×128，conv kernel 4。
"""
import math, torch

torch.manual_seed(0)

# ---- 真实 config 维度（缩头数不缩头维，保持结构） ----
NK, DK = 16, 128   # linear_num_key_heads=16, linear_key_head_dim=128
NV, DV = 64, 128   # linear_num_value_heads=64, linear_value_head_dim=128
HID = 4096
CONV_K = 4
T = 20

# ============ 1A. chunk vs recurrent 等价性 ============
# 逐字复刻源码两套实现的语义（float32 下计算）
def chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64):
    """源码 L249-327：分块前向。q [B,T,NV,DK] k [B,T,NV,DK] v [B,T,NV,DV] g/beta [B,T,NV]"""
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
    B, NH, S, KD = k.shape
    pad = (chunk_size - S % chunk_size) % chunk_size
    q, k, v = [torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
    beta = torch.nn.functional.pad(beta, (0, pad)); g = torch.nn.functional.pad(g, (0, pad))
    tot = S + pad
    scale = 1 / (KD ** 0.5); q = q * scale
    v_beta = v * beta.unsqueeze(-1); k_beta = k * beta.unsqueeze(-1)
    q, k, v, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone(); sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(B, NH, KD, v.shape[-1])
    out = torch.zeros_like(value)
    mask2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    for i in range(tot // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], value[:, :, i]   # 注意：源码 L299 value=attn@v_beta 重赋值，此处取变换后的值
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        out[:, :, i] = attn_inter + attn_i @ v_new
        state = state * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])   # 合并 chunk 维回序列维（源码 L324）
    return out[:, :, :S].transpose(1, 2), state

def recurrent_gated_delta_rule(q, k, v, g, beta, state=None):
    """源码 L331-380：逐 token 递归（decode 路径）。"""
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
    B, NH, S, KD = k.shape
    scale = 1 / (KD ** 0.5); q = q * scale
    out = torch.zeros(B, NH, S, v.shape[-1])
    state = torch.zeros(B, NH, KD, v.shape[-1]) if state is None else state.float()
    for i in range(S):
        q_t, k_t, v_t = q[:, :, i], k[:, :, i], v[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2), state

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

B = 2
q = torch.randn(B, T, NV, DK); k = torch.randn(B, T, NV, DK); v = torch.randn(B, T, NV, DV)
beta = torch.rand(B, T, NV)                       # 已过 sigmoid 的量级
g = -torch.rand(B, T, NV) * 0.1                   # 负对数量级
q, k = l2norm(q), l2norm(k)                       # 源码 use_qk_l2norm_in_kernel=True

out_c, st_c = chunk_gated_delta_rule(q, k, v, g, beta)
out_r, st_r = recurrent_gated_delta_rule(q, k, v, g, beta)
print("=== 1A. 分块 vs 递归（真实头数/头维，T=20） ===")
print(f"  输出最大绝对差     = {(out_c - out_r).abs().max().item():.3e}")
print(f"  末态最大绝对差     = {(st_c - st_r).abs().max().item():.3e}")
print(f"  判定：{'等价' if (out_c - out_r).abs().max().item() < 1e-5 else '不等价'}")

# ============ 1B. 手算复核（2 token 正交键） ============
print()
print("=== 1B. 递归状态手算复核（1 头，2 token 正交键，g=0, beta=1） ===")
qh = torch.tensor([[[[1.0, 0.0]]]])               # [1,1,1,2] 单头单 token
k1 = torch.tensor([[[[1.0, 0.0]]]]); k2 = torch.tensor([[[[0.0, 1.0]]]])
v1 = torch.tensor([[[[3.0, 5.0]]]]); v2 = torch.tensor([[[[7.0, 11.0]]]])
zero = torch.zeros(1, 1, 1)
q2 = torch.cat([qh, qh], dim=1); k12 = torch.cat([k1, k2], dim=1)   # 沿 T 维拼接 → [1,2,1,2]
v12 = torch.cat([v1, v2], dim=1); b12 = torch.ones(1, 2, 1); g12 = torch.zeros(1, 2, 1)
q2, k12 = l2norm(q2), l2norm(k12)
_, st = recurrent_gated_delta_rule(q2, k12, v12, g12, b12)
# 手算：S0=0；t1: S = k1 v1^T = [[3,5],[0,0]]；t2: kv_mem = S k2 = [0,0]（k1⊥k2），delta = v2，S = [[3,5],[0,0]] + k2 v2^T = [[3,5],[7,11]]
manual = torch.tensor([[[[3.0, 5.0], [7.0, 11.0]]]])
print(f"  手算末态 S = [[3,5],[7,11]]")
print(f"  实跑末态   = {st.squeeze().tolist()}")
print(f"  最大差     = {(st - manual).abs().max().item():.3e}")
print(f"  判定：{'一致' if (st - manual).abs().max().item() < 1e-5 else '不一致'}")

# ============ 1C. 真实配置状态占用 ============
print()
print("=== 1C. 397B 配置每层 GDN 状态占用（常数，不随序列长度增长） ===")
kd, vd = NK * DK, NV * DV
rec_elems = NV * DK * DV
print(f"  递归状态 = {NV} v头 × {DK} × {DV} = {rec_elems:,} 元素")
print(f"    fp32（config mamba_ssm_dtype=float32）= {rec_elems*4/2**20:.2f} MiB/层")
print(f"    × 45 个 GDN 层 = {45*rec_elems*4/2**20:.1f} MiB")
conv_elems = (kd*2 + vd) * (CONV_K - 1)
print(f"  卷积状态 = ({kd}×2+{vd}) × (kernel-1={CONV_K-1}) = {conv_elems:,} 元素 = {conv_elems*2/2**10:.2f} KiB/层（bf16）")
print(f"  对照：全注意力层 KV cache 每 token 增长 = 2×{2}×{256} = {2*2*256:,} 元素/层")

# ============ 1D. beta / g 数值范围 ============
print()
print("=== 1D. beta 与 g 的数值行为 ===")
b_raw = torch.randn(1000) * 2
a_raw = torch.randn(1000)
A_log = torch.log(torch.empty(64).uniform_(0.01, 16))
dt_bias = torch.ones(64)
beta_v = torch.sigmoid(b_raw)
g_v = -A_log.exp() * torch.nn.functional.softplus(a_raw[:, None] + dt_bias)
print(f"  beta = sigmoid(b)          ∈ [{beta_v.min():.4f}, {beta_v.max():.4f}]")
print(f"  g = -exp(A_log)·softplus(a+dt_bias) ∈ [{g_v.min():.3f}, {g_v.max():.3f}]")
print(f"  每 token 记忆保留率 exp(g) ∈ [{g_v.exp().min():.6f}, {g_v.exp().max():.6f}]")
print(f"  → A_log 初始化于 U(0.01,16)：exp(A_log)∈[0.01,16]，g 恒负，状态指数衰减")
