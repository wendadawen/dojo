"""实测 6：视觉编码器数据流（397B 真实配置：27 层 / hidden 1152 / patch 16 / 输出 4096）。

验证目标（对应源码 modeling_qwen3_5.py L833-1121 + transformers.vision_utils）：
  6A. Conv3d patch embed：kernel==stride → 每 patch 独立映射，等价于 1536→1152 线性变换
  6B. 可学习位置网格 48×48 双线性重采样到任意图像网格（权重和=1）
  6C. 视觉二维 RoPE：rot 维 36（head_dim//2），(h,w) 坐标，保范数
  6D. vision block：LayerNorm+biases、双向注意力（按图分段）、GELU MLP
  6E. merger：合并前 LayerNorm 在 1152 维（use_postshuffle_norm=False），4608→4608→4096
  6F. 常见分辨率 token 数
"""
import math, torch, torch.nn.functional as F

torch.manual_seed(5)
VH, VD, VHEADS = 1152, 27, 16
PATCH, TMERGE, MERGE = 16, 2, 2
OUT = 4096
POS_N = 2304                     # num_position_embeddings = 48×48
GRID = int(POS_N ** 0.5)         # 48
HD = VH // VHEADS                # 72

# ============ 6A. Conv3d patch embed ============
print("=== 6A. Conv3d patch embed（kernel == stride） ===")
W = torch.randn(VH, 3, TMERGE, PATCH, PATCH) * 0.02
n_params = W.numel() + VH        # + bias
x = torch.randn(1, 3, TMERGE, PATCH, PATCH)
y = F.conv3d(x, W, stride=(TMERGE, PATCH, PATCH))
# 等价线性：reshape patch → 3*2*16*16 = 1536
Wl = W.view(VH, -1)
yl = (x.view(1, -1) @ Wl.T).view(1, VH, 1, 1, 1)   # 对齐 conv3d 输出形状 [1,VH,1,1,1]
print(f"  Conv3d kernel=stride=(2,16,16)：输出 {tuple(y.shape)}，无重叠无遗漏")
print(f"  与线性变换等价: 最大差 {(y - yl).abs().max().item():.3e}")
print(f"  patch_embed 参数量 = {n_params:,} = {VH}×(3×2×16×16+1)（真实权重元素数应为此值）")
print(f"  输入像素按 (2帧×16×16) 分块，每块独立映射为 1 个 patch 向量")

# ============ 6B. 位置网格重采样 ============
print()
print("=== 6B. 48×48 可学习网格 → 双线性重采样 ===")
pos_embed = torch.randn(POS_N, VH) * 0.02
# transformers get_vision_interpolation_indices_and_weights 语义：
# 对目标网格 (h, w) 的每个坐标，在 48×48 网格上双线性插值（align_corners=True）
def interp_indices_weights(grid_h, grid_w, side=GRID):
    # 归一化坐标（与源一致：以 corner 对齐把目标格点映射到源格点）
    ys = torch.linspace(0, side - 1, grid_h)
    xs = torch.linspace(0, side - 1, grid_w)
    y0 = ys.floor().long().clamp(0, side - 1); y1 = (y0 + 1).clamp(0, side - 1)
    x0 = xs.floor().long().clamp(0, side - 1); x1 = (x0 + 1).clamp(0, side - 1)
    wy = (ys - y0.float()).clamp(0, 1); wx = (xs - x0.float()).clamp(0, 1)
    idxs, wts = [], []
    for i in range(grid_h):
        for j in range(grid_w):
            corners = [y0[i]*side + x0[j], y0[i]*side + x1[j], y1[i]*side + x0[j], y1[i]*side + x1[j]]
            weights = [(1-wy[i])*(1-wx[j]), (1-wy[i])*wx[j], wy[i]*(1-wx[j]), wy[i]*wx[j]]
            idxs.append(corners); wts.append(weights)
    return torch.tensor(idxs).long(), torch.tensor(wts).float()
for gh, gw in [(14, 14), (28, 28), (42, 42), (33, 60)]:
    idx, wt = interp_indices_weights(gh, gw)
    emb = (pos_embed[idx] * wt[:, :, None]).sum(1)
    print(f"  图像网格 ({gh},{gw}) → {gh*gw} 个位置向量；每位置 = 4 个网格点双线性混合（权重和 = {wt.sum(-1)[0]:.1f}）")
print(f"  位置信息跟随图像实际长宽比（无需 resize 到正方形）")

# ============ 6C. 视觉二维 RoPE ============
print()
print("=== 6C. 视觉二维 RoPE（rot=36） ===")
ROT = HD // 2                    # 36
inv = 1.0 / (10000.0 ** (torch.arange(0, ROT, 2).float() / ROT))
def rotate_half(t):
    a, b = t[..., :t.shape[-1]//2], t[..., t.shape[-1]//2:]
    return torch.cat((-b, a), dim=-1)
# 源码 rot_pos_emb：position_ids [n,2]=(h,w)，(pos × inv_freq).flatten(1) → [n,36]（前18=h,后18=w）
# emb = cat(freqs, freqs) → [n,72]，rotate_half 配对 (i, i+36) 共享同一频率
h, w = 3, 5
freqs = torch.cat([torch.outer(torch.tensor([float(h)]), inv),
                   torch.outer(torch.tensor([float(w)]), inv)], -1)   # [1, 36]
cos = torch.cat([freqs.cos(), freqs.cos()], -1)     # [1, 72]
sin = torch.cat([freqs.sin(), freqs.sin()], -1)
qh = torch.randn(1, HD)                      # 72 维全旋转
out = qh * cos + rotate_half(qh) * sin
print(f"  rot 维 = {ROT}（head_dim//2 = {HD}//2），全头维 {HD} 参与，(i, i+{HD//2}) 配对共享频率")
print(f"  前 18 对用 h 坐标、后 18 对用 w 坐标（二维位置进同一头维）")
print(f"  旋转范数保持: {qh.norm():.4f} → {out.norm():.4f}（差 {(out.norm()-qh.norm()).abs():.2e}）")
print(f"  与语言塔差别：视觉旋转全头维（72/72 中的 72 维按对旋转），语言塔只转 1/4 头维（64/256）")

# ============ 6D/6E. block 与 merger ============
print()
print("=== 6D. vision block 结构（真实张量头对照见 verify_structure） ===")
H = __import__("json").load(open("/tmp/qwen35/headers.json"))
vb = {k.replace("model.visual.blocks.0.", ""): v["shape"] for k, v in H.items()
      if k.startswith("model.visual.blocks.0.")}
for k in sorted(vb): print(f"    {k:<44s} {vb[k]}")
print()
print("=== 6E. merger ===")
mg = {k.replace("model.visual.merger.", ""): v["shape"] for k, v in H.items()
      if k.startswith("model.visual.merger.")}
for k in sorted(mg): print(f"    {k:<24s} {mg[k]}")
print(f"  → norm 在 1152 维（合并前，use_postshuffle_norm=False），linear_fc1 4608→4608，linear_fc2 4608→{OUT}")

# ============ 6F. token 数 ============
print()
print("=== 6F. 常见分辨率 token 数 ===")
def tokens(res):
    h, w = res // PATCH, res // PATCH
    return (h // MERGE) * (w // MERGE)
for res in [448, 896, 1344]:
    h = w = res // PATCH
    print(f"  {res}×{res}: grid ({h},{w}) → {(h//MERGE)*(w//MERGE):,} 个视觉 token")
# 1080p (1920×1080)
h, w = 1080 // PATCH, 1920 // PATCH
print(f"  1080p 1080×1920: grid ({h},{w}) → {(h//MERGE)*(w//MERGE):,} 个视觉 token")
