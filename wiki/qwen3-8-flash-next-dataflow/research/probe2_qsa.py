"""实测 2：QSA indexer 的块选择机制。

验证目标：
  1. indexer 用「块内 compress_ratio 个 token key 求均值」压缩，再算 query 与块的相关性
  2. 得分为 relu(q·k) 沿 4 个 index head 求和后除 sqrt(indexer_head_dim)
  3. 选中 block_topk = budget/compress_ratio 个完整块，尾部不完整块无条件保留
  4. 实际每个 query 可见的 token 数随位置变化，并在超过 budget 后封顶
  5. RoPE 只作用于 head_dim 的前 rotary_dim 维（partial_rotary_factor）

对应源码：transformers@36deb0b5
  modeling_qwen4_exp.py L611 Qwen4ExpTextQSAIndexer
    L622 block_topk = token_budget // compress_ratio
    L681 pooled_keys = key_groups.float().mean(dim=1)
    L693 scores = relu(scores).sum(dim=-1) / sqrt(index_head_dim)
    L695 topk(min(block_topk, num_complete_blocks))
    L700 tail = 不完整块的尾部 token，无条件加入
config：Qwen/Qwen3.8-Flash-Next@f5d08274
"""
import json, math
import torch

torch.manual_seed(0)
C = json.load(open("/tmp/qwen38fn/config.json"))["text_config"]

INH = C["indexer_n_heads"]
IKV = C["indexer_kv_heads"]
IH = C["indexer_head_dim"]
BUDGET = C["indexer_budget"]
CR = C["indexer_compress_ratio"]
BLOCK_TOPK = BUDGET // CR
HD = C["head_dim"]
PRF = C["rope_parameters"]["partial_rotary_factor"]
ROT = int(HD * PRF)

print("=" * 72)
print("实测 2A：indexer 配置量的实际取值")
print("=" * 72)
print(f"  indexer_n_heads       = {INH}   (query 头数)")
print(f"  indexer_kv_heads      = {IKV}   (key 头数，源码强制必须为 1)")
print(f"  indexer_head_dim      = {IH}")
print(f"  indexer_compress_ratio= {CR}   (每 {CR} 个连续 token 压成 1 块)")
print(f"  indexer_budget        = {BUDGET}")
print(f"  block_topk            = {BUDGET}//{CR} = {BLOCK_TOPK}  (源码 L622)")
print(f"  主注意力 head_dim     = {HD}, partial_rotary_factor = {PRF} -> rotary_dim = {ROT}")
print(f"  约束 rotary_dim <= indexer_head_dim: {ROT} <= {IH} -> {ROT <= IH}")
print(f"  索引投影输出宽度 = ({INH}+{IKV})*{IH} = {(INH+IKV)*IH}  (与权重 index_qk_proj [640,2560] 对齐)")


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_partial(x, cos, sin, unsqueeze_dim=1):
    """源码 L573 apply_rotary_pos_emb：只旋转前 cos.shape[-1] 维，其余原样拼回。"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    out = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([out, x_pass], dim=-1)


print()
print("=" * 72)
print("实测 2B：partial RoPE 只旋转前 64 维，后 64 维原样通过")
print("=" * 72)
theta = C["rope_parameters"]["rope_theta"]
inv_freq = 1.0 / (theta ** (torch.arange(0, ROT, 2).float() / ROT))
pos = torch.tensor([7.0])
freqs = pos[:, None] * inv_freq[None, :]
emb = torch.cat([freqs, freqs], dim=-1)
cos, sin = emb.cos(), emb.sin()
x = torch.randn(1, 1, IH)
y = apply_rope_partial(x, cos, sin)
print(f"  rope_theta = {theta:,}")
print(f"  inv_freq 长度 = {len(inv_freq)} = rotary_dim/2 = {ROT}/2")
print(f"  输入 indexer key 维度 = {IH}")
print(f"  前 {ROT} 维被旋转，最大变化 = {(y[...,:ROT]-x[...,:ROT]).abs().max().item():.4f}")
print(f"  后 {IH-ROT} 维原样通过，最大变化 = {(y[...,ROT:]-x[...,ROT:]).abs().max().item():.3e}")
print(f"  旋转保范数（前 {ROT} 维）: 输入 {x[...,:ROT].norm().item():.6f} -> 输出 {y[...,:ROT].norm().item():.6f}")


def indexer_select(raw_keys, q_one, query_idx, full_cos, full_sin):
    """照搬源码 L667-702 的单 query 选择逻辑。"""
    visible = torch.arange(query_idx + 1)          # 因果可见范围
    n_blocks = visible.shape[-1] // CR
    if n_blocks > 0:
        blk = visible[: n_blocks * CR].view(n_blocks, CR)
        kg = raw_keys.index_select(0, blk.flatten()).view(n_blocks, CR, IH)
        pooled = kg.float().mean(dim=1)                       # L681 块内均值
        starts = blk[:, 0]
        bk = apply_rope_partial(
            pooled.unsqueeze(1), full_cos.index_select(0, starts), full_sin.index_select(0, starts)
        ).squeeze(1)
        scores = torch.matmul(q_one.float(), bk.float().transpose(-1, -2)).transpose(-1, -2)
        scores = torch.relu(scores).sum(dim=-1) / math.sqrt(IH)   # L693
        sel_blk = scores.topk(min(BLOCK_TOPK, n_blocks), dim=0).indices
        sel = blk.index_select(0, sel_blk).flatten()
    else:
        sel = torch.tensor([], dtype=torch.long)
    tail = visible[n_blocks * CR:]                            # L700 尾部无条件保留
    return torch.cat([sel, tail]).long(), n_blocks, tail.numel()


print()
print("=" * 72)
print("实测 2C：不同 query 位置的实际可见 token 数")
print("=" * 72)
T = 12000
raw_keys = torch.randn(T, IH)
freqs_all = torch.arange(T).float()[:, None] * inv_freq[None, :]
emb_all = torch.cat([freqs_all, freqs_all], dim=-1)
cos_all, sin_all = emb_all.cos(), emb_all.sin()

print(f"  {'query 位置':>10s} {'因果可见':>9s} {'完整块数':>9s} {'选中块':>7s} {'块内token':>9s} {'尾部':>5s} {'实际可见':>9s} {'占比':>8s}")
rows = []
for qi in (3, 100, 1000, 2047, 2048, 2051, 4000, 8000, 11999):
    q_one = torch.randn(INH, IH)
    sel, nb, ntail = indexer_select(raw_keys, q_one, qi, cos_all, sin_all)
    n_uniq = torch.unique(sel).numel()
    causal = qi + 1
    picked_blocks = min(BLOCK_TOPK, nb)
    print(f"  {qi:>10,d} {causal:>9,d} {nb:>9,d} {picked_blocks:>7,d} {picked_blocks*CR:>9,d} {ntail:>5d} {n_uniq:>9,d} {n_uniq/causal*100:>7.2f}%")
    rows.append((qi, causal, n_uniq))

print()
print(f"  可见数上界 = block_topk*compress_ratio + (compress_ratio-1) = {BLOCK_TOPK}*{CR}+{CR-1} = {BLOCK_TOPK*CR+CR-1}")
print(f"  即 budget({BUDGET}) + compress_ratio-1({CR-1}) = {BUDGET+CR-1}，与源码 L662 分配的张量宽度一致")
maxv = max(r[2] for r in rows)
print(f"  实测最大可见 token 数 = {maxv}  (是否 <= 上界 {BUDGET+CR-1}: {maxv <= BUDGET+CR-1})")

print()
print("=" * 72)
print("实测 2D：块得分公式 relu 求和的作用 —— 与直接求和的差异")
print("=" * 72)
q_one = torch.randn(INH, IH)
qi = 4000
visible = torch.arange(qi + 1)
nb = visible.shape[-1] // CR
blk = visible[: nb * CR].view(nb, CR)
kg = raw_keys.index_select(0, blk.flatten()).view(nb, CR, IH)
pooled = kg.float().mean(dim=1)
starts = blk[:, 0]
bk = apply_rope_partial(pooled.unsqueeze(1), cos_all.index_select(0, starts), sin_all.index_select(0, starts)).squeeze(1)
raw = torch.matmul(q_one.float(), bk.float().transpose(-1, -2)).transpose(-1, -2)
s_relu = torch.relu(raw).sum(dim=-1) / math.sqrt(IH)
s_plain = raw.sum(dim=-1) / math.sqrt(IH)
print(f"  完整块数 = {nb}, 每块得分由 {INH} 个 index head 聚合")
print(f"  relu 求和：范围 [{s_relu.min():.4f}, {s_relu.max():.4f}]，恒非负 = {bool((s_relu>=0).all())}")
print(f"  直接求和：范围 [{s_plain.min():.4f}, {s_plain.max():.4f}]，恒非负 = {bool((s_plain>=0).all())}")
top_relu = set(s_relu.topk(BLOCK_TOPK).indices.tolist())
top_plain = set(s_plain.topk(BLOCK_TOPK).indices.tolist())
print(f"  两种打分选出的 top-{BLOCK_TOPK} 块重叠 = {len(top_relu & top_plain)} / {BLOCK_TOPK} = {len(top_relu&top_plain)/BLOCK_TOPK*100:.1f}%")
print(f"  -> relu 使单个 head 的负相关不能抵消其他 head 的正相关，选择结果确实不同")

print()
print("=" * 72)
print("实测 2E：块内均值池化对得分的影响（压缩带来的信息损失）")
print("=" * 72)
# 对比：用块内 4 个 token 的真实最大得分 vs 用池化 key 的得分
q_v = torch.randn(INH, IH)
tok_k = raw_keys[: nb * CR].view(nb, CR, IH)
# 每个 token 的得分：q [INH,IH] 与 token key 内积，沿 INH 求 relu 后求和
per_tok = torch.relu(torch.einsum("hd,bcd->bch", q_v.float(), tok_k.float())).sum(dim=-1) / math.sqrt(IH)
per_tok_max = per_tok.max(dim=-1).values          # 每块内 token 的最大得分 [nb]
pooled_nr = tok_k.float().mean(dim=1)             # [nb, IH]
per_blk = torch.relu(torch.einsum("hd,bd->bh", q_v.float(), pooled_nr)).sum(dim=-1) / math.sqrt(IH)
print(f"  （未加 RoPE，隔离位置因素）块数 = {nb}, per_tok_max shape={tuple(per_tok_max.shape)}, per_blk shape={tuple(per_blk.shape)}")
corr = torch.corrcoef(torch.stack([per_tok_max, per_blk]))[0, 1].item()
a = set(per_tok_max.topk(BLOCK_TOPK).indices.tolist())
b = set(per_blk.topk(BLOCK_TOPK).indices.tolist())
print(f"  「块内 token 最大得分」与「池化 key 得分」的相关系数 = {corr:.4f}")
print(f"  两者 top-{BLOCK_TOPK} 块重叠 = {len(a&b)}/{BLOCK_TOPK} = {len(a&b)/BLOCK_TOPK*100:.1f}%")
print(f"  -> 池化是有损的：块粒度选择无法完全还原 token 粒度的重要性排序")
