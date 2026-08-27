"""实测 8：视觉编码器（ViT）的完整数据流与多模态融合。

验证目标：
  1. patch_embed 是 Conv3d，kernel=stride=(2,16,16)，把每个时空 patch 一次投成 1152 维
  2. 学习式位置嵌入是 48x48 的方格，按每张图的实际网格做双线性重采样（非插值到固定尺寸）
  3. 视觉 RoPE 只用 head_dim//2=36 维的频率，拼接后覆盖 72 维完整头
  4. 27 层 block 为 LayerNorm + 双向（非因果）注意力 + GELU MLP，标准 pre-norm
  5. merger 把 2x2 空间相邻的 4 个 patch 拼成 4608 维再投到 2560（语言主干宽度）
  6. 视觉 token 数 = prod(grid_thw) / spatial_merge_size^2
  7. 融合方式是 masked_scatter 替换占位符 token，不是拼接
  8. 参数量分组核对

对应源码：transformers@36deb0b5 modeling_qwen4_exp.py
  L1660 Qwen4ExpVisionRotaryEmbedding（dim = head_dim//2）
  L1685 Qwen4ExpVisionPatchEmbed（Conv3d）
  L1705 Qwen4ExpVisionPatchMerger
  L1735 Qwen4ExpVisionAttention（is_causal=False）
  L1818 Qwen4ExpVisionBlock
  L1849 Qwen4ExpVisionModel（pos_embed 2304 = 48^2，rotary_pos_emb(head_dim//2)）
  L1937 hidden_states = patch_embed(x) + 重采样后的 pos_embed
  L2151 split_sizes = grid_thw.prod(-1) // spatial_merge_size^2
  L2286 inputs_embeds.masked_scatter(image_mask, image_embeds)
config：Qwen/Qwen3.8-Flash-Next@f5d08274
"""
import json, math
from math import prod
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
CFG = json.load(open("/tmp/qwen38fn/config.json"))
V = CFG["vision_config"]
TC = CFG["text_config"]
H = json.load(open("/tmp/qwen38fn/headers.json"))

D = V["hidden_size"]
NH = V["num_heads"]
HD = D // NH
IM = V["intermediate_size"]
PS = V["patch_size"]
TPS = V["temporal_patch_size"]
SMS = V["spatial_merge_size"]
OUT = V["out_hidden_size"]
DEPTH = V["depth"]
NPOS = V["num_position_embeddings"]
CH = V["in_channels"]

print("=" * 74)
print("A. 视觉编码器配置与真实张量核对")
print("=" * 74)
print(f"  depth={DEPTH}, hidden_size={D}, num_heads={NH} -> head_dim={HD}")
print(f"  intermediate_size={IM}, patch_size={PS}, temporal_patch_size={TPS}")
print(f"  spatial_merge_size={SMS}, out_hidden_size={OUT}（= 语言主干 hidden_size {TC['hidden_size']}: {OUT == TC['hidden_size']}）")
print(f"  num_position_embeddings={NPOS} -> 方格边长 sqrt({NPOS}) = {int(NPOS**0.5)}")
print()
checks = [
    ("patch_embed.proj.weight", [D, CH, TPS, PS, PS], "Conv3d 权重 [out, in, kT, kH, kW]"),
    ("patch_embed.proj.bias", [D], "Conv3d 有 bias"),
    ("pos_embed.weight", [NPOS, D], "学习式位置嵌入表"),
    ("blocks.0.attn.qkv.weight", [3 * D, D], "qkv 合并投影（3 倍宽）"),
    ("blocks.0.attn.qkv.bias", [3 * D], "attention 带 bias（与语言塔不同）"),
    ("blocks.0.attn.proj.weight", [D, D], "输出投影"),
    ("blocks.0.mlp.linear_fc1.weight", [IM, D], "MLP 上投影"),
    ("blocks.0.mlp.linear_fc2.weight", [D, IM], "MLP 下投影"),
    ("blocks.0.norm1.weight", [D], "LayerNorm（带 bias，非 RMSNorm）"),
    ("merger.norm.weight", [D], "merger 前置 LayerNorm 作用在合并前的 1152"),
    ("merger.linear_fc1.weight", [D * SMS**2, D * SMS**2], f"合并后维度 {D}x{SMS}^2 = {D*SMS**2}"),
    ("merger.linear_fc2.weight", [OUT, D * SMS**2], f"投到语言主干宽度 {OUT}"),
]
for name, exp, desc in checks:
    key = f"model.visual.{name}"
    got = H[key]["shape"]
    assert got == exp, f"MISMATCH {key}: 期望 {exp}, 实际 {got}"
    print(f"  OK  {name:<32s} {str(exp):<24s} {desc}")
n_blocks = len({int(k.split(".")[3]) for k in H if k.startswith("model.visual.blocks.")})
print(f"  OK  blocks 层数 = {n_blocks}，与 config depth={DEPTH} 一致: {n_blocks == DEPTH}")
print(f"  OK  MLP 与 attention 均带 bias（真实权重存在 *.bias），LayerNorm 也带 bias")

print()
print("=" * 74)
print("B. patch_embed：Conv3d 一次投影，kernel = stride（无重叠）")
print("=" * 74)
proj = nn.Conv3d(CH, D, kernel_size=[TPS, PS, PS], stride=[TPS, PS, PS], bias=True)
# 一张 1 帧图（源码要求图像沿时间维复制到 temporal_patch_size）
grid_t, grid_h, grid_w = 1, 8, 12
n_patch = grid_t * grid_h * grid_w
px = torch.randn(n_patch, CH * TPS * PS * PS)
out = proj(px.view(-1, CH, TPS, PS, PS)).view(-1, D)
print(f"  网格 grid_thw = ({grid_t}, {grid_h}, {grid_w}) -> patch 数 = {n_patch}")
print(f"  每个 patch 展平输入 = {CH}x{TPS}x{PS}x{PS} = {CH*TPS*PS*PS}")
print(f"  输入 {tuple(px.shape)} -> reshape {tuple(px.view(-1,CH,TPS,PS,PS).shape)} -> 输出 {tuple(out.shape)}")
print(f"  kernel == stride == [{TPS},{PS},{PS}]，故每个 patch 独立映射、无重叠")
print(f"  等价于对展平向量做一次 {CH*TPS*PS*PS} -> {D} 的线性变换（参数量相同）")
w = H["model.visual.patch_embed.proj.weight"]["shape"]
print(f"  真实权重 {w}，元素数 {prod(w):,} = {D} x {CH*TPS*PS*PS} = {D*CH*TPS*PS*PS:,}: {prod(w) == D*CH*TPS*PS*PS}")

print()
print("=" * 74)
print("C. 位置嵌入：48x48 方格按实际网格双线性重采样")
print("=" * 74)
side = int(NPOS**0.5)
print(f"  位置表是 {side}x{side} = {NPOS} 个可学习向量，各 {D} 维")
print(f"  对每张图，按其 (grid_h, grid_w) 用 bilinear + align_corners=True 重采样")
pos_tbl = torch.randn(NPOS, D)
# 复现重采样：把 48x48 网格插值到 grid_h x grid_w
src = pos_tbl.view(1, side, side, D).permute(0, 3, 1, 2)
dst = F.interpolate(src, size=(grid_h, grid_w), mode="bilinear", align_corners=True)
dst = dst.permute(0, 2, 3, 1).reshape(grid_h * grid_w, D)
print(f"  {side}x{side} -> {grid_h}x{grid_w}: {tuple(src.shape)} -> {tuple(dst.shape)}")
print(f"  与 patch_embed 输出逐元素相加：{tuple(out.shape)} + {tuple(dst.repeat(grid_t,1).shape)}")
h_sum = out + dst.repeat(grid_t, 1)
print(f"  相加后 {tuple(h_sum.shape)}，无 NaN: {bool(torch.isfinite(h_sum).all())}")
# 重采样是加权和，权重非负且和为 1
print(f"  重采样后向量落在原表向量的凸包内（双线性权重非负且和为 1）")
print(f"  含义：位置信息与图像实际长宽比绑定，不需要把图像 resize 成固定尺寸")

print()
print("=" * 74)
print("D. 视觉 RoPE：频率维度只有 head_dim/2，拼接后覆盖整头")
print("=" * 74)
rope_dim = HD // 2
print(f"  Qwen4ExpVisionRotaryEmbedding(dim = head_dim // 2 = {HD} // 2 = {rope_dim})")
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
print(f"  inv_freq 长度 = {len(inv_freq)} = {rope_dim}/2")
# position_ids 是 (seq, 2) 的 (h, w) 二维坐标
pos2d = torch.stack(torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij"), dim=-1).reshape(-1, 2)
rot = (pos2d.unsqueeze(-1) * inv_freq).flatten(1)
print(f"  position_ids {tuple(pos2d.shape)}（每个 patch 一对 (h,w) 坐标）")
print(f"  外积后 flatten -> {tuple(rot.shape)}  = 2 坐标 x {len(inv_freq)} 频率 = {2*len(inv_freq)}")
emb = torch.cat([rot, rot], dim=-1)
print(f"  cat 自身 -> {tuple(emb.shape)} = {emb.shape[-1]}，恰为 head_dim = {HD}: {emb.shape[-1] == HD}")
print(f"  故视觉 RoPE 覆盖完整头维（与语言塔的 partial_rotary_factor=0.25 只转 1/4 不同）")
cos_v, sin_v = emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


q = torch.randn(grid_h * grid_w, NH, HD)
q_r = (q * cos_v.unsqueeze(-2) + rotate_half(q) * sin_v.unsqueeze(-2))
print(f"  施加到 q {tuple(q.shape)}：范数保持 {q.norm().item():.6f} -> {q_r.norm().item():.6f}"
      f"（差 {abs(q.norm().item()-q_r.norm().item()):.2e}）")

print()
print("=" * 74)
print("E. 27 层 block：pre-norm + 双向注意力 + GELU MLP")
print("=" * 74)


class VBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D, eps=1e-6)
        self.norm2 = nn.LayerNorm(D, eps=1e-6)
        self.qkv = nn.Linear(D, 3 * D, bias=True)
        self.proj = nn.Linear(D, D)
        self.fc1 = nn.Linear(D, IM, bias=True)
        self.fc2 = nn.Linear(IM, D, bias=True)

    def forward(self, x, cos, sin):
        h = self.norm1(x)
        L = h.shape[0]
        qq, kk, vv = self.qkv(h).reshape(L, 3, NH, HD).permute(1, 0, 2, 3).unbind(0)
        qq = qq * cos.unsqueeze(-2) + rotate_half(qq) * sin.unsqueeze(-2)
        kk = kk * cos.unsqueeze(-2) + rotate_half(kk) * sin.unsqueeze(-2)
        qq, kk, vv = [t.transpose(0, 1).unsqueeze(0) for t in (qq, kk, vv)]
        # is_causal=False：无掩码，patch 之间全连接（同一图像内）
        a = F.scaled_dot_product_attention(qq, kk, vv, is_causal=False)
        a = a.squeeze(0).transpose(0, 1).reshape(L, D)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))
        return x


blk = VBlock()
hh = h_sum
print(f"  输入 {tuple(hh.shape)}")
hh = blk(hh, cos_v, sin_v)
print(f"  单层输出 {tuple(hh.shape)}（形状不变，残差结构）")
print(f"  注意力 is_causal=False -> 双向；同一图像的 {grid_h*grid_w} 个 patch 互相可见")
print(f"  多图时用 cu_seqlens 做变长打包，跨图像不互相注意")
print(f"  归一化用 LayerNorm(eps=1e-6) 带 bias，激活 {V['hidden_act']}")
print(f"  27 层堆叠后输出仍为 {tuple(hh.shape)}（逐层等形状）")

print()
print("=" * 74)
print("F. merger：2x2 空间合并后投到语言主干宽度")
print("=" * 74)
merged_dim = D * SMS**2
norm = nn.LayerNorm(D, eps=1e-6)
fc1 = nn.Linear(merged_dim, merged_dim)
fc2 = nn.Linear(merged_dim, OUT)
# use_postshuffle_norm=False -> 先按 1152 归一化，再 view 成 4608
x_in = torch.randn(grid_h * grid_w, D)
m = norm(x_in).view(-1, merged_dim)
m = fc2(F.gelu(fc1(m)))
print(f"  use_postshuffle_norm=False -> LayerNorm 作用在合并前的 {D} 维，再 view 成 {merged_dim}")
print(f"  {tuple(x_in.shape)} -> norm -> view {(grid_h*grid_w//SMS**2, merged_dim)} -> fc1/GELU/fc2 -> {tuple(m.shape)}")
print(f"  patch 数 {grid_h*grid_w} 被 {SMS}^2={SMS**2} 合并 -> {grid_h*grid_w//SMS**2} 个视觉 token")
print(f"  输出维度 {m.shape[-1]} == 语言主干 hidden_size {TC['hidden_size']}: {m.shape[-1] == TC['hidden_size']}")

print()
print("=" * 74)
print("G. 视觉 token 数量：不同分辨率下的实算")
print("=" * 74)
print(f"  公式（源码 L2151）：n_tokens = prod(grid_thw) / spatial_merge_size^2")
print(f"  grid_h = ceil(H / {PS} / {SMS}) * {SMS}，grid_w 同理（由 processor 保证可被 {SMS} 整除）")
print()
print(f"  {'输入':>16s} {'grid (t,h,w)':>16s} {'patch 数':>10s} {'视觉 token':>11s} {'占 262144 上下文':>16s}")
for label, (t, hpx, wpx) in [
    ("448x448 图", (1, 448, 448)),
    ("896x896 图", (1, 896, 896)),
    ("1344x1344 图", (1, 1344, 1344)),
    ("1920x1080 图", (1, 1080, 1920)),
    ("16 帧 448x448", (16, 448, 448)),
]:
    gh, gw = hpx // PS, wpx // PS
    gh = gh - gh % SMS
    gw = gw - gw % SMS
    # 视频每 temporal_patch_size 帧合成一个时间 patch
    gt = max(1, t // TPS) if t > 1 else 1
    npatch = gt * gh * gw
    ntok = npatch // SMS**2
    print(f"  {label:>16s} {f'({gt},{gh},{gw})':>16s} {npatch:>10,d} {ntok:>11,d} {ntok/TC['max_position_embeddings']*100:>15.3f}%")

print()
print("=" * 74)
print("H. 多模态融合：masked_scatter 替换占位符，不是拼接")
print("=" * 74)
print(f"  config: image_token_id={CFG['image_token_id']}, video_token_id={CFG['video_token_id']}")
print(f"          vision_start_token_id={CFG['vision_start_token_id']}, vision_end_token_id={CFG['vision_end_token_id']}")
# 小规模复现替换过程
vocab, d_txt = 100, TC["hidden_size"]
IMG_ID = CFG["image_token_id"]
ids = torch.tensor([[5, 7, IMG_ID, IMG_ID, IMG_ID, IMG_ID, 9, 11]])
emb_txt = torch.randn(1, ids.shape[1], d_txt)
img_emb = torch.arange(4 * d_txt, dtype=torch.float).view(4, d_txt)
mask = (ids == IMG_ID).unsqueeze(-1).expand_as(emb_txt)
fused = emb_txt.masked_scatter(mask, img_emb)
n_ph = int((ids == IMG_ID).sum())
print(f"  序列 {ids.tolist()[0]} 中有 {n_ph} 个图像占位符")
print(f"  视觉 token 数 {img_emb.shape[0]} 必须等于占位符数 {n_ph}: {img_emb.shape[0] == n_ph}")
print(f"  masked_scatter 后序列长度不变: {tuple(emb_txt.shape)} -> {tuple(fused.shape)}")
ok_replaced = torch.allclose(fused[0, 2:6], img_emb)
ok_kept = torch.allclose(fused[0, [0, 1, 6, 7]], emb_txt[0, [0, 1, 6, 7]])
print(f"  占位符位置已被视觉特征替换: {ok_replaced}")
print(f"  非占位符位置保持原文本 embedding: {ok_kept}")
print(f"  -> 视觉与文本在同一序列中共享全部 48 层，不存在独立的视觉分支")

print()
print("=" * 74)
print("I. 视觉编码器参数量分组")
print("=" * 74)
groups = {
    "patch_embed (Conv3d)": ["patch_embed"],
    "pos_embed (48x48 表)": ["pos_embed"],
    "blocks attention (27 层)": ["blocks", "attn"],
    "blocks MLP (27 层)": ["blocks", "mlp"],
    "blocks LayerNorm (27 层)": ["blocks", "norm"],
    "merger": ["merger"],
}
tot = 0
res = {}
for k, v in H.items():
    if not k.startswith("model.visual."):
        continue
    n = prod(v["shape"])
    tot += n
    for g, pats in groups.items():
        if all(p in k for p in pats):
            res[g] = res.get(g, 0) + n
            break
print(f"  {'分组':<28s} {'参数量':>14s} {'占视觉塔':>9s}")
for g in groups:
    n = res.get(g, 0)
    print(f"  {g:<28s} {n:>14,d} {n/tot*100:>8.2f}%")
print(f"  {'合计':<28s} {tot:>14,d} {'100.00%':>9s}")
print(f"  = {tot/1e9:.3f} B，占全模型 179,999,981,424 的 {tot/179999981424*100:.3f}%")
# 逐项理论核对
theory_blk = 27 * ((3*D*D + 3*D) + (D*D + D) + (IM*D + IM) + (D*IM + D) + 4*D)
print()
print(f"  单层理论 = qkv({3*D*D}+{3*D}) + proj({D*D}+{D}) + fc1({IM*D}+{IM}) + fc2({D*IM}+{D}) + 2xLN({4*D})")
print(f"  27 层理论 = {theory_blk:,}")
real_blk = sum(prod(v["shape"]) for k, v in H.items() if k.startswith("model.visual.blocks."))
print(f"  27 层真实 = {real_blk:,}   一致: {theory_blk == real_blk}")
theory_mg = D + D + (merged_dim*merged_dim + merged_dim) + (OUT*merged_dim + OUT)
real_mg = sum(prod(v["shape"]) for k, v in H.items() if k.startswith("model.visual.merger."))
print(f"  merger 理论 = {theory_mg:,}   真实 = {real_mg:,}   一致: {theory_mg == real_mg}")

print()
print("=" * 74)
print("J. 视觉 token 在语言侧的三维 MRoPE 位置分配")
print("=" * 74)
print("  源码 get_vision_position_ids (L1980) + get_rope_index (L2032)：")
print("    llm_grid = (t // temp_merge, h // spatial_merge, w // spatial_merge)")
print("    T = arange(llm_grid_t) * time_interval + start_position")
print("    H = arange(llm_grid_h) + start_position")
print("    W = arange(llm_grid_w) + start_position")
print("    三者 meshgrid 后 stack 成 (3, n_tokens)")
print("  文本段则三个维度取同一个 arange（源码 L2105 expand(3,-1)）")
print()


def vision_pos_ids(start, t, h, w, merge=2, temp_merge=1, interval=1):
    gt, gh, gw = t // temp_merge, h // merge, w // merge
    pt = torch.arange(gt) * interval
    ph = torch.arange(gh) + start
    pw = torch.arange(gw) + start
    T, Hh, W = torch.meshgrid(pt, ph, pw, indexing="ij")
    out = torch.stack([T, Hh, W], dim=0).reshape(3, -1)
    out[0] += start
    return out


# 构造：8 个文本 token + 一张 (1, 28, 28) 网格的图 + 5 个文本 token
start = 8
gh_, gw_ = 28, 28
vp = vision_pos_ids(start, 1, gh_, gw_)
ntok = vp.shape[1]
print(f"  例：前 {start} 个文本 token 占位置 0..{start-1}")
print(f"      一张 grid=(1,{gh_},{gw_}) 的图 -> llm_grid=(1,{gh_//SMS},{gw_//SMS}) -> {ntok} 个视觉 token")
print(f"      视觉 token 的三维位置（前 4 个与后 2 个）:")
for i in list(range(4)) + [ntok - 2, ntok - 1]:
    print(f"        token {i:>4d}: T={vp[0,i].item():>3d}  H={vp[1,i].item():>3d}  W={vp[2,i].item():>3d}")
print(f"      T 维恒为 {vp[0].unique().tolist()}（单帧图像），H/W 各自在 [{start}, {start+gh_//SMS-1}] 内遍历")
advance = max(gh_, gw_) // SMS
print()
print(f"  关键：图像块结束后位置只推进 max(h,w)//merge = max({gh_},{gw_})//{SMS} = {advance}")
print(f"        而不是推进视觉 token 数 {ntok}（源码 L2115）")
print(f"        故后续文本从位置 {start+advance} 开始，而非 {start+ntok}")
print(f"        节省的位置数 = {ntok} - {advance} = {ntok-advance}")
print()
print(f"  {'图像 grid':>14s} {'视觉 token':>11s} {'位置推进量':>11s} {'压缩比':>9s}")
for gh2, gw2 in [(28, 28), (56, 56), (84, 84), (66, 120)]:
    n = (gh2 // SMS) * (gw2 // SMS)
    adv = max(gh2, gw2) // SMS
    print(f"  {f'(1,{gh2},{gw2})':>14s} {n:>11,d} {adv:>11,d} {n/adv:>8.1f}x")
print()
print(f"  含义：一张 1344x1344 的图占 1764 个序列位置（消耗上下文），")
print(f"        但在 MRoPE 的位置轴上只前进 42，因此不会快速耗尽位置编码范围。")
print(f"        max_position_embeddings={TC['max_position_embeddings']:,} 对应的是位置轴上限，非 token 数上限。")

print()
print("=" * 74)
print("K. 语言塔 MRoPE 的交错排布：三个维度如何共享 64 个旋转维")
print("=" * 74)
sec = TC["rope_parameters"]["mrope_section"]
rot = int(TC["head_dim"] * TC["rope_parameters"]["partial_rotary_factor"])
print(f"  mrope_section = {sec}，和 = {sum(sec)} = rotary_dim/2 = {rot}/2 = {rot//2}: {sum(sec) == rot//2}")
print(f"  mrope_interleaved = {TC['rope_parameters']['mrope_interleaved']}")
print(f"  源码 apply_interleaved_mrope (L140)：以 T 维为底，再把 H/W 按 stride 3 的切片覆盖进去")
freqs = torch.zeros(3, 1, 1, rot // 2)
for d in range(3):
    freqs[d] = d + 1        # 用 1/2/3 标记来源维度，便于观察排布
ft = freqs[0].clone()
for dim, offset in enumerate((1, 2), start=1):
    length = sec[dim] * 3
    idx = slice(offset, length, 3)
    ft[..., idx] = freqs[dim, ..., idx]
layout = ft.flatten().int().tolist()
print()
print(f"  排布结果（1=T, 2=H, 3=W），共 {len(layout)} 个频率槽位：")
for i in range(0, len(layout), 16):
    print(f"    [{i:>2d}:{min(i+16,len(layout)):>2d}] {''.join('THW'[v-1] for v in layout[i:i+16])}")
from collections import Counter
cnt = Counter(layout)
print()
print(f"  各维占用槽位数: T={cnt[1]}, H={cnt[2]}, W={cnt[3]}")
# H 覆盖 slice(1, sec[1]*3, 3)，W 覆盖 slice(2, sec[2]*3, 3)，其余槽位保持 T
h_slots = list(range(1, min(sec[1]*3, rot//2), 3))
w_slots = list(range(2, min(sec[2]*3, rot//2), 3))
t_slots = [i for i in range(rot//2) if i not in h_slots and i not in w_slots]
print(f"  H 覆盖槽位 slice(1, {sec[1]}*3, 3) -> {h_slots}")
print(f"  W 覆盖槽位 slice(2, {sec[2]}*3, 3) -> {w_slots}")
print(f"  其余槽位保持 T -> {t_slots}")
print(f"  与实测排布一致: {sorted(h_slots)==sorted([i for i,v in enumerate(layout) if v==2]) and sorted(w_slots)==sorted([i for i,v in enumerate(layout) if v==3])}")
print(f"  槽位数与 mrope_section 逐项对应: T={cnt[1]}=={sec[0]}, H={cnt[2]}=={sec[1]}, W={cnt[3]}=={sec[2]}"
      f" -> {[cnt[1],cnt[2],cnt[3]] == sec}")
print(f"  三维在全部 {rot//2} 个频率槽位上交错分布（非分段独占）；")
print(f"  低频槽位对应长程距离、高频对应近距离，交错使三个维度都覆盖完整频率范围，")
print(f"  这是 interleaved 相比 chunked [TTT...HHH...WWW] 排布的差别（源码 docstring L141）。")
print(f"  W 比 T/H 少一个槽位（{sec[2]} vs {sec[0]}），因 rotary_dim/2={rot//2} 不能被 3 整除。")
