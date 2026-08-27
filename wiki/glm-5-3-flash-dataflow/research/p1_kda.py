"""
探针 1：KDA（Kimi Delta Attention）线性注意力层。

验证目标
  1. 真实 config 维度下 KDA 层的张量形状与参数量，与 checkpoint 张量头一致
  2. 遗忘门公式：safe 分支 g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
     —— 确认取的是 safe 分支而非 softplus 分支，且 g 的数值范围
  3. delta rule 递推：S_t = diag(g_t) ... 逐 token 状态更新（recurrent 实现）
  4. chunk 实现（prefill）与 recurrent 实现（decode）在同一输入上等价
  5. 循环状态尺寸与序列长度无关（线性注意力的核心性质）

对应源码
  Glm5NextTextForgetGate.forward           modeling_glm5_next.py L319-335
  recurrent_kimi_delta_attention           L428-478
  chunk_kimi_delta_attention               L482-578
  Glm5NextTextLinearAttention.forward      L628-733
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/tmp/glm53f/probe")
import torch
from harness import (real_config, banner, Glm5NextTextLinearAttention, Glm5NextTextForgetGate,
                     recurrent_kimi_delta_attention, chunk_kimi_delta_attention, l2norm, MiniCache)

torch.manual_seed(0)
C = real_config()
DEV, DT = "cpu", torch.float32

banner("探针 1：KDA 线性注意力（真实 config 维度）")

# ---------- 1.1 层结构与参数量 ----------
layer = Glm5NextTextLinearAttention(C, layer_idx=0).to(DT).eval()
print("[1.1] KDA 层参数张量（真实维度）")
tot = 0
for n, p in layer.named_parameters():
    tot += p.numel()
    print(f"      {n:26s} {tuple(p.shape)}  {p.numel():>10,}")
print(f"      合计 {tot:,} 参数 = {tot/1e6:.2f} M")

H, D = C.linear_num_heads, C.linear_head_dim
print(f"[1.2] 派生维度：qkv_dim = num_heads*head_dim = {H}*{D} = {layer.qkv_dim}")
print(f"      conv_dim = qkv_dim*3 = {layer.conv_dim}（q/k/v 拼成一路做 depthwise conv）")
print(f"      conv1d.weight {tuple(layer.conv1d.weight.shape)}  groups={layer.conv1d.groups}")
print(f"      checkpoint 中拆为 q/k/v_conv1d 各 [8192,1,4] → 合计 [{3*8192},1,4] = "
      f"{tuple(layer.conv1d.weight.shape)}  一致={tuple(layer.conv1d.weight.shape)==(24576,1,4)}")

# ---------- 1.3 遗忘门：确认走 safe 分支 ----------
banner("1.3 遗忘门（forget gate）取值范围")
fg = layer.forget_gate
print(f"      safe_gate_lower_bound = {fg.safe_gate_lower_bound}  (config.linear_attn_config.gate_lower_bound)")
print(f"      → 走 safe 分支：g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))")
# 官方 _init_weights：safe 分支下 A_log 初始化为 zeros → decay_rate = exp(0) = 1
with torch.no_grad():
    fg.A_log.zero_()
    fg.dt_bias.zero_()
x = torch.randn(1, 16, C.hidden_size, dtype=DT)
g = fg(x)
print(f"      g.shape = {tuple(g.shape)}  (B, S, num_heads, head_dim) → 逐头逐通道独立衰减")
print(f"      g 范围 [{g.min():.6f}, {g.max():.6f}]，全部 < 0 = {bool((g<0).all())}")
print(f"      递推里用的是 g.exp()，即衰减因子范围 [{g.exp().min():.6f}, {g.exp().max():.6f}] ⊂ (0,1)")
print(f"      A_log=0,dt_bias=0 时理论值 = lower_bound*sigmoid(0) = {fg.safe_gate_lower_bound}*0.5 "
      f"= {fg.safe_gate_lower_bound*0.5}；实测均值 = {g.mean():.6f}")
print(f"      衰减因子理论 exp(-2.5) = {torch.tensor(-2.5).exp():.6f}")

# ---------- 1.4 delta rule 递推：手工复算一步，与官方 kernel 对比 ----------
banner("1.4 delta rule 递推公式核对（手算 vs 官方 recurrent kernel）")
B, S, Hh, Dh = 1, 5, 2, 4          # 缩小到可手算规模；公式与真实维度无关
q = torch.randn(B, S, Hh, Dh, dtype=DT)
k = torch.randn(B, S, Hh, Dh, dtype=DT)
v = torch.randn(B, S, Hh, Dh, dtype=DT)
gg = -torch.rand(B, S, Hh, Dh, dtype=DT)          # 与真实 g 同号（负）
beta = torch.rand(B, S, Hh, dtype=DT)

out_ref, state_ref = recurrent_kimi_delta_attention(
    q, k, v, g=gg, beta=beta, initial_state=None, output_final_state=True,
    use_qk_l2norm_in_kernel=True)

# 手算：严格按 L464-476 的语义重写一遍（独立实现，用于核对公式理解）
qn = l2norm(q.float(), dim=-1, eps=1e-6) * (Dh ** -0.5)
kn = l2norm(k.float(), dim=-1, eps=1e-6)
S_t = torch.zeros(B, Hh, Dh, Dh, dtype=DT)         # [B,H,Dk,Dv]
manual = torch.zeros(B, S, Hh, Dh, dtype=DT)
for t in range(S):
    decay = gg[:, t][..., None].exp()              # [B,H,Dk,1]
    S_t = S_t * decay                              # 逐 key 通道衰减
    kv_mem = (S_t * kn[:, t][..., None]).sum(dim=-2)          # k_t^T S  → [B,H,Dv]
    delta = (v[:, t] - kv_mem) * beta[:, t][..., None]        # 预测残差 × 写入强度
    S_t = S_t + kn[:, t].unsqueeze(-1) * delta.unsqueeze(-2)  # 外积写回
    manual[:, t] = (S_t * qn[:, t][..., None]).sum(dim=-2)    # 读出
print(f"      官方 kernel 输出 vs 手算最大绝对误差 = {(out_ref-manual).abs().max():.3e}")
print(f"      终态最大绝对误差 = {(state_ref-S_t).abs().max():.3e}")
print(f"      注：q 在 kernel 内乘 scale = 1/sqrt(head_dim) = {Dh**-0.5:.6f}；k 只做 l2norm 不乘 scale")

# ---------- 1.5 chunk（prefill）与 recurrent（decode）等价性 ----------
banner("1.5 chunk 并行实现 与 recurrent 串行实现的等价性")
for Slen in [1, 7, 64, 65, 130]:
    q2 = torch.randn(1, Slen, 2, 4, dtype=DT)
    k2 = torch.randn(1, Slen, 2, 4, dtype=DT)
    v2 = torch.randn(1, Slen, 2, 4, dtype=DT)
    g2 = -torch.rand(1, Slen, 2, 4, dtype=DT)
    b2 = torch.rand(1, Slen, 2, dtype=DT)
    kw = dict(g=g2, beta=b2, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_rec, s_rec = recurrent_kimi_delta_attention(q2, k2, v2, **kw)
    o_chk, s_chk = chunk_kimi_delta_attention(q2, k2, v2, **kw)
    print(f"      S={Slen:4d}  输出误差={float((o_rec-o_chk).abs().max()):.3e}  "
          f"终态误差={float((s_rec-s_chk).abs().max()):.3e}  "
          f"(chunk_size=64, pad={(64-Slen%64)%64})")

# ---------- 1.6 状态尺寸与序列长度无关 ----------
banner("1.6 KDA 循环状态尺寸（线性注意力的核心性质）")
print(f"      单层单序列状态 = [num_heads, head_dim, head_dim] = "
      f"[{H}, {D}, {D}] = {H*D*D:,} 个元素")
print(f"      官方 update_recurrent_state 存 float32 → {H*D*D*4/1024/1024:.2f} MiB / 层 / 序列")
print(f"      另有 conv 状态 = [conv_dim, kernel-1] = [{layer.conv_dim}, {C.linear_conv_kernel_dim-1}] = "
      f"{layer.conv_dim*(C.linear_conv_kernel_dim-1):,} 个元素")
n_kda = sum(1 for t in C.layer_types if t == "linear_attention")
print(f"      全模型 KDA 层数 = {n_kda} → 状态合计 "
      f"{n_kda*H*D*D*4/1024/1024:.2f} MiB / 序列，与序列长度无关")

# ---------- 1.7 完整 KDA 层前向：prefill + 逐 token decode ----------
banner("1.7 KDA 层完整前向（prefill 后接 decode，检查形状与数值健康）")
cache = MiniCache(num_layers=1)
xp = torch.randn(1, 12, C.hidden_size, dtype=DT)
mask = torch.ones(1, 12, dtype=torch.bool)
with torch.no_grad():
    yp = layer(hidden_states=xp, cache_params=cache, attention_mask=mask)
print(f"      prefill  输入 {tuple(xp.shape)} → 输出 {tuple(yp.shape)}  "
      f"finite={bool(torch.isfinite(yp).all())}")
rs = cache.layers[0].recurrent_states[0]
cs = cache.layers[0].conv_states[0]
print(f"      循环状态 {tuple(rs.shape)} dtype={rs.dtype}；conv 状态 {tuple(cs.shape)}")
with torch.no_grad():
    for step in range(3):
        xd = torch.randn(1, 1, C.hidden_size, dtype=DT)
        yd = layer(hidden_states=xd, cache_params=cache, attention_mask=torch.ones(1, 1, dtype=torch.bool))
        print(f"      decode {step+1}  输出 {tuple(yd.shape)}  finite={bool(torch.isfinite(yd).all())}  "
              f"状态 shape 不变={tuple(cache.layers[0].recurrent_states[0].shape)==tuple(rs.shape)}")
