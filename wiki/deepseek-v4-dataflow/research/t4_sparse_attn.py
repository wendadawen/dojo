"""实测四：sparse_attn 的语义——attn_sink、-1 掩码、以及它与标准注意力的关系。

对应 kernel.py sparse_attn_kernel 与报告 Eq.27。
验证：
  1. 当 topk_idxs 覆盖全部位置且 attn_sink=-inf 时，退化为标准 MQA 注意力
  2. attn_sink 使注意力权重之和 < 1（报告：allows total attention scores to be near 0）
  3. -1 索引确实被完全排除（不贡献、不占权重）
"""
import sys, torch
sys.path.insert(0, "/tmp/dsv4/exp")
from kernel_ref import sparse_attn

torch.manual_seed(0)
b, m, h, d, n = 1, 4, 8, 64, 32
q = torch.randn(b, m, h, d)
kv = torch.randn(b, n, d)
scale = d ** -0.5

print("=== 1. attn_sink 极小时退化为标准 MQA 注意力 ===")
full_idx = torch.arange(n).view(1, 1, n).expand(b, m, n).int()
sink_off = torch.full((h,), -1e30)
o_sparse = sparse_attn(q, kv, sink_off, full_idx, scale)
# 标准 MQA：单 KV 头被所有 query 头共享
ref = torch.einsum("bmhd,bnd->bmhn", q, kv) * scale
ref = ref.softmax(-1)
o_ref = torch.einsum("bmhn,bnd->bmhd", ref, kv)
print(f"最大绝对差异 = {(o_sparse - o_ref).abs().max().item():.3e}   -> sparse_attn 是带索引门控的标准注意力")
print()

print("=== 2. attn_sink 的作用：注意力权重和可以小于 1 ===")
for sv in [-1e30, -2.0, 0.0, 2.0, 5.0]:
    sink = torch.full((h,), float(sv))
    logits = torch.einsum("bmhd,bnd->bmhn", q, kv) * scale
    mx = logits.amax(-1, keepdim=True)
    e = torch.exp(logits - mx)
    denom = e.sum(-1) + torch.exp(sink.view(1, 1, h) - mx.squeeze(-1))
    w = (e.sum(-1) / denom).mean().item()
    print(f"  attn_sink={sv:>8}: 平均注意力权重之和 = {w:.6f}")
print("  sink 越大，模型越可以「不看任何 KV」，即整层注意力输出趋近 0")
print()

print("=== 3. -1 索引被完全排除 ===")
idx = torch.arange(n).view(1, 1, n).expand(b, m, n).clone().int()
idx[:, :, 16:] = -1                              # 后一半置为无效
o_masked = sparse_attn(q, kv, sink_off, idx, scale)
o_half = sparse_attn(q, kv[:, :16], sink_off, torch.arange(16).view(1,1,16).expand(b,m,16).int(), scale)
print(f"「32 位置但后16个为-1」 vs 「只有前16个位置」 最大差异 = {(o_masked - o_half).abs().max().item():.3e}")
print("  -> -1 既不贡献 value 也不占 softmax 分母，等价于该位置不存在")
print()

print("=== 4. 滑窗索引与压缩索引拼接后一起进 softmax（model.py Attention.forward）===")
print("model.py: topk_idxs = torch.cat([window_idxs, compress_idxs], dim=-1)")
print("          o = sparse_attn(q, kv, attn_sink, topk_idxs, scale)")
print("  含义：滑窗的未压缩 KV 与选中的压缩 KV 处于同一个 softmax 归一化空间，")
print("        而非两条分支各算一次注意力再加权求和。")
win_idx = torch.arange(8).view(1,1,8).expand(b,m,8).int()
cmp_idx = torch.arange(20, 28).view(1,1,8).expand(b,m,8).int()
cat_idx = torch.cat([win_idx, cmp_idx], dim=-1)
o_cat = sparse_attn(q, kv, sink_off, cat_idx, scale)
o_w = sparse_attn(q, kv, sink_off, win_idx, scale)
print(f"  拼接后输出 与 仅滑窗输出 的差异 = {(o_cat-o_w).abs().max().item():.4f}  (非0，证明压缩条目参与同一归一化)")
