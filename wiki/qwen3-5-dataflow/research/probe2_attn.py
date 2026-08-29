"""实测 2：门控注意力与交错 MRoPE（Qwen3.5 真实配置）。

验证目标（对应源码 modeling_qwen3_5.py L557-704, L84-164）：
  2A. q_proj 双宽 → chunk 出 query 与 gate；sigmoid(gate) 缩放注意力输出
  2B. QK-norm 逐头 RMSNorm（head_dim=256），归一后每头 RMS≈1
  2C. GQA：2 KV 头 → 32 Q 头（repeat 16×）
  2D. 部分 RoPE：只旋转前 64/256 维，旋转保范数、其余原样通过
  2E. 交错 MRoPE 槽位排布（mrope_section=[11,11,10]，rotary_dim=64）
  2F. 位置推进：视觉段 max(h,w)/merge（get_rope_index 语义）
真实维度：32 Q 头 × 256，2 KV 头，partial_rotary_factor 0.25，rope_theta 1e7。
"""
import math, torch

torch.manual_seed(1)

# ---- 真实 config ----
NH, NKV, HD = 32, 2, 256
HID = 4096
PARTIAL = 0.25
ROT = int(HD * PARTIAL)          # 64
THETA = 1000000
SEC = [11, 11, 10]               # mrope_section，和 = 32 = ROT/2

# ============ 2A. 双宽 q_proj 与输出门 ============
print("=== 2A. q_proj 双宽与 sigmoid 输出门 ===")
Wq = torch.randn(NH * HD * 2, HID)      # [16384, 4096] 与真实权重一致
Wo = torch.randn(HID, NH * HD)
x = torch.randn(1, 5, HID)
proj = (x @ Wq.T).view(1, 5, NH, HD * 2)  # 源码 L670-671：view 到 (..., heads, head_dim*2)
q, gate = torch.chunk(proj, 2, dim=-1)
gate = gate.reshape(1, 5, -1)
print(f"  q_proj 输出宽 = {NH*HD*2} = 2 × (头数 × 头维)，chunk 后 query {tuple(q.shape)} + gate {tuple(gate.shape)}")
attn_out = torch.randn(1, 5, NH * HD)
gated = attn_out * torch.sigmoid(gate)
print(f"  sigmoid(gate) 范围 = [{torch.sigmoid(gate).min():.4f}, {torch.sigmoid(gate).max():.4f}]")
print(f"  门控逐 token 逐头缩放：输出范数比 = {(gated.norm(dim=-1) / attn_out.norm(dim=-1)).mean():.4f}")
print(f"  → 每个头每个 token 有独立的输出缩放系数（共 {NH}×T 个），门在 o_proj 之前")

# ============ 2B. QK-norm 逐头归一 ============
print()
print("=== 2B. QK-norm（RMSNorm on head_dim=256） ===")
w_qn = torch.randn(HD)
def rmsnorm(t, w):
    var = t.float().pow(2).mean(-1, keepdim=True)
    return (w * (t * torch.rsqrt(var + 1e-6)).to(t.dtype))
q_n = rmsnorm(q, w_qn)
rms = q_n.float().pow(2).mean(-1).sqrt()
print(f"  归一后每头 RMS ≈ {rms.mean():.4f}（weight=随机时≠1，结构上先除 RMS 再乘 weight）")
print(f"  归一前每头 RMS ≈ {q.float().pow(2).mean(-1).sqrt().mean():.3f} → 归一化消除了头间幅度差异")
print(f"  → Q/K 各有独立 RMSNorm（q_norm/k_norm weight 形状 [256]，逐头维共享参数）")

# ============ 2C. GQA ============
print()
print("=== 2C. GQA 分组 ===")
groups = NH // NKV
print(f"  {NH} 个 Q 头共享 {NKV} 个 KV 头，每组 {groups} 个 Q 头复用同一 KV")
print(f"  KV cache 每 token 每层 = 2(K和V) × {NKV} × {HD} = {2*NKV*HD:,} 元素")
print(f"  若无 GQA（32 KV 头）需 {2*NH*HD:,} 元素 → 压缩 {NH//NKV}×")

# ============ 2D. 部分 RoPE 保范数 ============
print()
print("=== 2D. 部分 RoPE（只旋转前 64 维） ===")
def rotate_half(t):
    t1, t2 = t[..., : t.shape[-1] // 2], t[..., t.shape[-1] // 2:]
    return torch.cat((-t2, t1), dim=-1)
inv_freq = 1.0 / (THETA ** (torch.arange(0, ROT, 2).float() / ROT))
pos = torch.tensor([0.0, 5.0, 1000.0, 100000.0])
freqs = torch.outer(pos, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
cos, sin = emb.cos(), emb.sin()
qh = torch.randn(len(pos), 1, HD)     # [T, 1, 256]
q_rot, q_pass = qh[..., :ROT], qh[..., ROT:]
q_embed = q_rot * cos.unsqueeze(1) + rotate_half(q_rot) * sin.unsqueeze(1)
q_full = torch.cat([q_embed, q_pass], dim=-1)
print(f"  旋转维 {ROT}/256 = {ROT/HD:.0%}，其余 {HD-ROT} 维不旋转直接拼接")
n_before = qh.norm(dim=-1).squeeze(-1)
n_after = q_full.norm(dim=-1).squeeze(-1)
print(f"  旋转前后范数: {[f'{v:.4f}' for v in n_before.tolist()]} → {[f'{v:.4f}' for v in n_after.tolist()]}")
print(f"  最大范数变化 = {(n_after - n_before).abs().max():.2e}（旋转子空间正交，保范数）")
print(f"  inv_freq 首/末 = {inv_freq[0]:.6f} / {inv_freq[-1]:.3e}（theta=1e7，比标准 1e4 低频更多）")

# ============ 2E. 交错 MRoPE 槽位 ============
print()
print("=== 2E. 交错 MRoPE 槽位排布（mrope_section=[11,11,10]） ===")
print(f"  频率槽位共 {ROT//2} = {sum(SEC)}（T:{SEC[0]} H:{SEC[1]} W:{SEC[2]}）")
freqs3 = torch.zeros(3, 1, 1, ROT // 2)
for d in range(3):
    freqs3[d] = d + 1
ft = freqs3[0].clone()
for dim, offset in enumerate((1, 2), start=1):
    length = SEC[dim] * 3
    idx = slice(offset, length, 3)
    ft[..., idx] = freqs3[dim, ..., idx]
layout = ft.flatten().int().tolist()
for i in range(0, len(layout), 16):
    print(f"    槽位[{i:>2d}:{min(i+16,len(layout)):>2d}] {''.join('THW'[v-1] for v in layout[i:i+16])}")
from collections import Counter
cnt = Counter(layout)
print(f"  槽位归属: T={cnt[1]} H={cnt[2]} W={cnt[3]}（与 mrope_section {SEC} 一致: {[cnt[1],cnt[2],cnt[3]]==SEC}）")
h_slots = list(range(1, min(SEC[1]*3, ROT//2), 3))
w_slots = list(range(2, min(SEC[2]*3, ROT//2), 3))
print(f"  H 覆盖槽位 {h_slots[0]}..{h_slots[-1]}，W 覆盖 {w_slots[0]}..{w_slots[-1]}：三维交错铺满全部 32 槽")

# ============ 2F. 位置推进 ============
print()
print("=== 2F. 位置推进（视觉段后推进 max(h,w)/merge） ===")
MERGE = 2
def vision_pos(start, t, h, w):
    gt, gh, gw = t, h // MERGE, w // MERGE
    T_ = torch.arange(gt) * 1 + start
    H_ = torch.arange(gh) + start
    W_ = torch.arange(gw) + start
    Tm, Hm, Wm = torch.meshgrid(T_, H_, W_, indexing="ij")
    return torch.stack([Tm, Hm, Wm], dim=0).reshape(3, -1)
start = 8
vp = vision_pos(start, 1, 28, 28)
ntok = vp.shape[1]
print(f"  8 文本 + (1,28,28) 图：视觉 token {ntok} 个，位置推进 {max(28,28)//MERGE}")
print(f"  图后第一个文本 token 位置 = {start + max(28,28)//MERGE}（而非 {start+ntok}）")
print(f"  {'图像 grid':>14s} {'视觉 token':>10s} {'位置推进':>9s} {'压缩比':>7s}")
for h, w in [(28, 28), (56, 56), (84, 84), (66, 120)]:
    n = (h // MERGE) * (w // MERGE)
    adv = max(h, w) // MERGE
    print(f"  {f'(1,{h},{w})':>14s} {n:>10,d} {adv:>9,d} {n/adv:>6.1f}x")
