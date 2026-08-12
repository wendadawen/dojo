"""实测十一：输出侧反向 RoPE，以及 q 的额外归一化。

报告 2.3.3「Partial Rotary Positional Embedding」称：
  KV 条目同时充当 key 和 value，因此注意力输出会携带绝对位置信息；
  对策是对输出的后 64 维施加位置为 -i 的 RoPE，使贡献变回相对位置。
model.py Attention.forward: apply_rotary_emb(o[..., -rd:], freqs_cis, True)  # inverse=True

同时验证 model.py 中 q 在 wq_b 之后额外做了一次 RMS 归一（无权重）：
  q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
"""
import sys, types, torch
sys.path.insert(0, "/tmp/dsv4/exp")
import kernel_ref
fake = types.ModuleType("kernel")
for k in ["act_quant","fp4_act_quant","sparse_attn","hc_split_sinkhorn","fp8_gemm","fp4_gemm"]:
    setattr(fake, k, getattr(kernel_ref, k))
sys.modules["kernel"] = fake
sys.path.insert(0, "/tmp/dsv4/inference")
import model as M

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

print("=== 1. apply_rotary_emb 的 inverse=True 确实是共轭（反向旋转）===")
d, s = 64, 6
freqs = M.precompute_freqs_cis(d, s, 0, 10000.0, 1.0, 32, 1)
x = torch.randn(1, s, 1, d)
fwd = M.apply_rotary_emb(x.clone(), freqs)
back = M.apply_rotary_emb(fwd.clone(), freqs, True)
print(f"正向旋转后再反向旋转，与原始的最大差异 = {(back - x).abs().max().item():.3e}")
print("  -> inverse 用 freqs_cis.conj()，是精确的逆变换")
print()

print("=== 2. 为什么输出需要反向旋转：KV 同时当 key 和 value ===")
print("注意力输出 o_t = sum_j softmax(q_t·k_j) * v_j，而 k_j = v_j = 同一个 KV 条目。")
print("KV 条目的后 64 维已被施加位置 j 的 RoPE，故 o_t 里混入了各 j 的绝对位置相位。")
print("对 o_t 施加位置 -t 的反向旋转后，每个 j 的贡献相位变为 (j - t)，即相对距离。")
print()
# 数值演示：单个 KV 主导时，输出相位差应等于 j-t
kv = torch.zeros(1, s, d)
kv[0, :, :] = torch.randn(d)                       # 各位置内容相同
kv_r = M.apply_rotary_emb(kv.clone().unsqueeze(2), freqs).squeeze(2)   # 施加位置 j 的 RoPE
q = torch.randn(1, 1, 1, d) * 0.0
q[0, 0, 0, 0] = 10.0
t = 4
# 让 query 只看位置 j=1
idx = torch.tensor([[[1]]], dtype=torch.int32)
o = kernel_ref.sparse_attn(q, kv_r, torch.tensor([-1e30]), idx, 1.0)
o_inv = M.apply_rotary_emb(o.clone()[..., -d:], freqs[t:t+1], True)
raw_phase = kv_r[0, 1, :2]
print(f"仅看位置 j=1 时：")
print(f"  未反向旋转的输出前2维 = {o[0,0,0,:2].tolist()}   （含位置 1 的绝对相位）")
print(f"  反向旋转(位置 t=4)后 = {o_inv[0,0,0,:2].tolist()}   （相位变为 1-4=-3，即相对距离）")
print()

print("=== 3. q 在 wq_b 之后的额外 RMS 归一（无可学习权重）===")
print("model.py: q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)")
qv = torch.randn(2, 3, 4, 128) * 5.0
rms_before = qv.square().mean(-1).sqrt()
qn = qv * torch.rsqrt(qv.square().mean(-1, keepdim=True) + 1e-6)
rms_after = qn.square().mean(-1).sqrt()
print(f"归一前每头 RMS 范围 [{rms_before.min():.4f}, {rms_before.max():.4f}]")
print(f"归一后每头 RMS 范围 [{rms_after.min():.6f}, {rms_after.max():.6f}]  -> 每个头被拉到单位 RMS")
print("对应报告 2.3.3「Query and Key-Value Entry Normalization」：对每个 query 头和唯一的")
print("KV 头在核心注意力前额外做 RMSNorm，避免 attention logits 爆炸。")
print("注意：q 这一步没有可学习权重（纯归一），kv 侧则是带权重的 kv_norm。")
