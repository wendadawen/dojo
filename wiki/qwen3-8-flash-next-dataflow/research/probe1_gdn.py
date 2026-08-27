"""实测 1：Gated DeltaNet 递归形式与分块形式的等价性，以及递归状态的实际尺寸。

验证目标：
  1. 官方 torch_recurrent_gated_delta_rule 与 torch_chunk_gated_delta_rule 在同一输入下输出一致
     -> 证明「prefill 走分块、decode 走单步递归」两条路径计算同一个函数
  2. delta rule 的状态更新公式 S_t = S_{t-1}*exp(g_t) + k_t (v_t - S_{t-1}^T k_t)^T beta_t
  3. GDN 每层常数大小的递归状态，与 KV cache 随长度线性增长形成对比

对应源码：transformers@36deb0b5
  src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
    L266 torch_chunk_gated_delta_rule
    L348 torch_recurrent_gated_delta_rule
    L519 g = -A_log.exp() * softplus(a + dt_bias)
    L517 beta = b.sigmoid()
config：Qwen/Qwen3.8-Flash-Next@f5d08274
"""
import json, math, sys
import torch
import torch.nn.functional as F

torch.manual_seed(0)
sys.path.insert(0, "/tmp/qwen38fn/tf/src")

C = json.load(open("/tmp/qwen38fn/config.json"))["text_config"]
NK, NV = C["linear_num_key_heads"], C["linear_num_value_heads"]
DK, DV = C["linear_key_head_dim"], C["linear_value_head_dim"]


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent(query, key, value, g, beta):
    """逐 token 递归，对应源码 L348 torch_recurrent_gated_delta_rule。"""
    query, key = l2norm(query), l2norm(key)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)]
    B, H, T, DKh = key.shape
    DVh = value.shape[-1]
    query = query * (1 / DKh**0.5)
    out = torch.zeros(B, H, T, DVh)
    S = torch.zeros(B, H, DKh, DVh)
    for i in range(T):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        S = S * g_t
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (S * q_t.unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2), S


def chunked(query, key, value, g, beta, chunk_size=64):
    """分块并行，对应源码 L266 torch_chunk_gated_delta_rule（逐行照搬）。"""
    query, key = l2norm(query), l2norm(key)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)]
    B, H, T, DKh = key.shape
    DVh = value.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad)) for x in (query, key, value)]
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    Tt = T + pad
    query = query * (1 / query.shape[-1] ** 0.5)
    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    S = torch.zeros(B, H, DKh, DVh)
    out = torch.zeros_like(value)
    for i in range(Tt // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ S
        out[:, :, i] = attn_inter + a @ v_new
        S = S * g[:, :, i, -1, None, None].exp() + (
            k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :T]
    return out.transpose(1, 2), S


print("=" * 72)
print("实测 1A：分块形式 vs 递归形式的数值等价性")
print("=" * 72)
B, T = 1, 160
# 官方 config 的真实头数/维度；num_v_heads=48, num_k_heads=16 -> q/k 需 repeat_interleave 3 次
q = torch.randn(B, T, NK, DK)
k = torch.randn(B, T, NK, DK)
v = torch.randn(B, T, NV, DV)
rep = NV // NK
q_e = q.repeat_interleave(rep, dim=2)
k_e = k.repeat_interleave(rep, dim=2)
print(f"  config: linear_num_key_heads={NK}, linear_num_value_heads={NV}, 比值={rep}")
print(f"  q 扩展前 {tuple(q.shape)} -> 扩展后 {tuple(q_e.shape)}   (源码 L520-522 repeat_interleave)")

# 按源码 L517-519 生成 beta 与 g
a_raw = torch.randn(B, T, NV)
b_raw = torch.randn(B, T, NV)
A_log = torch.log(torch.empty(NV).uniform_(0.01, 16))
dt_bias = torch.ones(NV)
beta = b_raw.sigmoid()
g = -A_log.float().exp() * F.softplus(a_raw.float() + dt_bias)
print(f"  beta = sigmoid(b)  范围 [{beta.min():.4f}, {beta.max():.4f}]  (源码 L517)")
print(f"  g = -exp(A_log) * softplus(a + dt_bias)  范围 [{g.min():.4f}, {g.max():.4f}]  (源码 L519)")
print(f"  g 恒为负 -> exp(g) in (0,1) 是衰减因子: {bool((g < 0).all())}")

out_r, S_r = recurrent(q_e, k_e, v, g, beta)
out_c, S_c = chunked(q_e, k_e, v, g, beta)
d_out = (out_r - out_c).abs().max().item()
d_S = (S_r - S_c).abs().max().item()
print()
print(f"  输出最大绝对差   = {d_out:.3e}")
print(f"  末态最大绝对差   = {d_S:.3e}")
print(f"  输出相对量级     = {out_r.abs().mean().item():.4f}")
print(f"  判定（差 < 1e-4）: {'一致' if d_out < 1e-4 else '不一致'}")

print()
print("=" * 72)
print("实测 1B：delta rule 状态更新公式的逐步复算")
print("=" * 72)
# 用极小规模手算核对：1 头，DK=DV=2，2 个 token
qs = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]).permute(1, 0, 2, 3)
ks = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]).permute(1, 0, 2, 3)
vs = torch.tensor([[[[2.0, 3.0]]], [[[5.0, 7.0]]]]).permute(1, 0, 2, 3)
gs = torch.zeros(1, 2, 1)          # g=0 -> exp(g)=1，无衰减，便于手算
bs = torch.ones(1, 2, 1)           # beta=1 -> 完全写入
o_ref, S_ref = recurrent(qs, ks, vs, gs, bs)
# 手算：k 已 l2norm（本例已是单位向量），q 也 l2norm 后再乘 1/sqrt(2)
# t=0: S = k0 (v0 - 0) = e1 ⊗ [2,3]；out0 = S^T q0 * scale
# t=1: kv_mem = S^T k1 = 0（k1=e2 与 e1 正交）；S += e2 ⊗ [5,7]
scale = 1 / math.sqrt(2)
S_manual = torch.zeros(2, 2)
S_manual[0] = torch.tensor([2.0, 3.0])
S_manual[1] = torch.tensor([5.0, 7.0])
o0_manual = S_manual.T @ (torch.tensor([1.0, 0.0]) * scale)
o1_manual = S_manual.T @ (torch.tensor([0.0, 1.0]) * scale)
print(f"  构造输入：k0=[1,0] k1=[0,1] 正交，v0=[2,3] v1=[5,7]，g=0，beta=1")
print(f"  手算末态 S =\n{S_manual.numpy()}")
print(f"  实跑末态 S =\n{S_ref[0, 0].numpy()}")
print(f"  末态差 = {(S_ref[0, 0] - S_manual).abs().max().item():.3e}")
print(f"  手算 out0 = {o0_manual.numpy()}   实跑 out0 = {o_ref[0, 0, 0].numpy()}")
print(f"  手算 out1 = {o1_manual.numpy()}   实跑 out1 = {o_ref[0, 1, 0].numpy()}")
print(f"  输出差 = {max((o_ref[0,0,0]-o0_manual).abs().max().item(), (o_ref[0,1,0]-o1_manual).abs().max().item()):.3e}")

print()
print("=" * 72)
print("实测 1C：GDN 递归状态 vs QSA KV cache 的显存量级（按真实 config 计算）")
print("=" * 72)
n_lin = sum(1 for t in C["layer_types"] if t == "linear_attention")
n_qsa = len(C["layer_types"]) - n_lin
# GDN 状态：每层 num_v_heads * head_k_dim * head_v_dim，config 指定 fp32
gdn_state = NV * DK * DV
gdn_bytes = gdn_state * 4                                    # mamba_ssm_dtype=float32
# conv 状态只需保留 kernel-1 个位置（实测见 probe6_convstate.py）
conv_state = (2 * NK * DK + NV * DV) * (C["linear_conv_kernel_dim"] - 1) * 2   # bf16
print(f"  GDN 每层递归状态 = {NV} 头 x {DK} x {DV} = {gdn_state:,} 元素 = {gdn_bytes/2**20:.2f} MiB (fp32, config mamba_ssm_dtype)")
print(f"  GDN 每层卷积状态 = conv_dim({2*NK*DK+NV*DV}) x (kernel-1={C['linear_conv_kernel_dim']-1}) x 2B = {conv_state/2**10:.2f} KiB (bf16)")
print(f"  36 个 GDN 层合计 = {(gdn_bytes+conv_state)*n_lin/2**30:.4f} GiB —— 与序列长度无关")
print()
nkv, hd = C["num_key_value_heads"], C["head_dim"]
per_tok_per_layer = 2 * nkv * hd * 2                          # K+V, bf16
idx_per_tok_per_layer = C["indexer_head_dim"] * 2             # indexer 每 token 存 1 个 key head
for L_seq in (4096, 32768, 262144, 1048576):
    kv = per_tok_per_layer * n_qsa * L_seq
    idx = idx_per_tok_per_layer * n_qsa * L_seq
    tot = kv + idx + (gdn_bytes + conv_state) * n_lin
    print(f"  seq={L_seq:>9,}: KV={kv/2**30:>7.3f} GiB + indexer={idx/2**30:>6.3f} GiB + GDN={((gdn_bytes+conv_state)*n_lin)/2**30:.3f} GiB = {tot/2**30:>7.3f} GiB")
print()
print(f"  对比：若 48 层全是 QSA 型全注意力，seq=1M 时 KV 为 {per_tok_per_layer*48*1048576/2**30:.3f} GiB")
print(f"  实际 12 层：{per_tok_per_layer*n_qsa*1048576/2**30:.3f} GiB，降为 {n_qsa/48:.4f} = 1/{48/n_qsa:.0f}")
