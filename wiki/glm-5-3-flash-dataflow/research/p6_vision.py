"""
探针 6：视觉塔与多模态 token 排布（GLM-5 系列首个原生多模态模型）。

验证目标
  1. 视觉塔各阶段真实形状：patch_embed(Conv3d) → 24 个 block → post_layernorm →
     downsample(Conv2d, spatial_merge) → merger
  2. patch 数与视觉 token 数的换算关系：token = prod(grid_thw) / spatial_merge_size^2
  3. 视觉输出维度 out_hidden_size 必须等于文本 hidden_size 才能替换占位符
  4. 图像与视频共用同一个 token id，靠 start/end span 区分（这是与其他 VLM 的关键差异）
  5. masked_scatter 注入路径的形状约束
  6. 视觉塔参数量与 checkpoint 交叉验证

对应源码
  Glm5NextVisionModel.forward         modeling_glm5_next.py L1781-1824
  Glm5NextVisionPatchEmbed.forward    L1723-1729
  Glm5NextVisionPatchMerger.forward   L1529-1537
  Glm5NextModel.get_image_features    L1869-1881
  Glm5NextModel.get_placeholder_mask  L1883-1933
  Glm5NextModel.forward               L1937-1990
"""
from __future__ import annotations
import sys, json, math
sys.path.insert(0, "/tmp/glm53f/probe")
import torch
import torch.nn as nn

CFG = json.load(open("/tmp/glm53f/config.json"))
V = CFG["vision_config"]
T = CFG["text_config"]


def banner(t):
    print("=" * 78); print(t); print("=" * 78)


banner("探针 6：视觉塔与多模态 token 排布")

# ---------- 6.1 视觉配置 ----------
print("[6.1] vision_config 全字段（来自官方 config.json）")
for k in sorted(V):
    print(f"      {k:32s} {V[k]!r}")

print(f"\n      out_hidden_size={V['out_hidden_size']}  text hidden_size={T['hidden_size']}  "
      f"相等={V['out_hidden_size']==T['hidden_size']}")
print(f"      → 视觉输出可直接 masked_scatter 进文本 embedding，无需额外投影层")

# ---------- 6.2 patch → token 换算 ----------
banner("6.2 patch 数与视觉 token 数换算")
ps, sm, tp = V["patch_size"], V["spatial_merge_size"], V["temporal_patch_size"]
print(f"      patch_size={ps}  spatial_merge_size={sm}  temporal_patch_size={tp}")
print(f"      源码 L1877：split_sizes = grid_thw.prod(-1) // spatial_merge_size**2")
print(f"      → 视觉 token 数 = (T*H_patch*W_patch) / {sm}^2 = (T*H_p*W_p)/{sm*sm}")
print()
print(f"      {'输入图像':>16s} {'grid(t,h,w)':>16s} {'patch 数':>10s} {'视觉 token':>11s}  压缩比")
for px in [448, 896, 1344]:
    hp = wp = px // ps
    grid = (1, hp, wp)
    npatch = hp * wp
    ntok = npatch // (sm * sm)
    print(f"      {f'{px}x{px}':>16s} {str(grid):>16s} {npatch:>10d} {ntok:>11d}  "
          f"{px*px//ntok:>5d} 像素/token")
# 视频
print(f"\n      视频（8 帧 448x448，temporal_patch_size={tp}）：")
hp = 448 // ps
print(f"        get_video_features（L1845-1864）先把 [b,3] 的 video_grid_thw 展平成每帧 [1,h,w]")
print(f"        8 帧 → grid_thw=(8,{hp},{hp}) → 视觉 token = 8*{hp}*{hp}//{sm*sm} = "
      f"{8*hp*hp//(sm*sm)}")

# ---------- 6.3 视觉塔实跑（缩小 depth） ----------
banner("6.3 视觉塔实跑（depth 24→2 以便本机跑通，其余维度全为真实值）")


class VC:
    def __init__(self, depth):
        for k, v in V.items():
            setattr(self, k, v)
        self.depth = depth
        self._attn_implementation = "eager"
        self.training = False


sys.path.insert(0, "/tmp/glm53f/probe")
import harness as HN
import re

# 抽取官方视觉塔代码（L1497-1824），同样一行不改
lines = open("/tmp/glm53f/modeling_glm5_next.py").read().split("\n")
vis_src = "\n".join(lines[1496:1824])
assert "class Glm5NextVisionMLP" in vis_src and "class Glm5NextVisionModel" in vis_src


# 官方视觉塔依赖的两个 vision_utils 函数，按官方语义补齐（Qwen2-VL 系列通用实现）
def get_vision_position_ids(grid_thw, spatial_merge_size, kwargs=None):
    """按 spatial_merge_size 分块重排后的 (h_idx, w_idx) 位置对。"""
    pos = []
    for t, h, w in grid_thw.tolist():
        hi = torch.arange(h).unsqueeze(1).expand(h, w)
        wi = torch.arange(w).unsqueeze(0).expand(h, w)
        hi = hi.reshape(h // spatial_merge_size, spatial_merge_size,
                        w // spatial_merge_size, spatial_merge_size).permute(0, 2, 1, 3).flatten()
        wi = wi.reshape(h // spatial_merge_size, spatial_merge_size,
                        w // spatial_merge_size, spatial_merge_size).permute(0, 2, 1, 3).flatten()
        pos.append(torch.stack([hi, wi], dim=-1).repeat(t, 1))
    return torch.cat(pos, dim=0)


def get_vision_attention_seqlens(grid_thw, config, kwargs=None):
    lens = [int(t * h * w) for t, h, w in grid_thw.tolist()]
    cu = torch.tensor([0] + lens).cumsum(0).to(torch.int32)
    return cu, (max(lens) if lens else 0)


def is_flash_attention_requested(config):
    return False


def get_max_seqlen(*a, **k):
    return 0


class BaseModelOutputWithPooling:
    def __init__(self, last_hidden_state=None, pooler_output=None):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output


class _VisionBase(nn.Module):
    """替身基类：官方 Glm5NextVisionModel 继承 Glm5NextPreTrainedModel（框架类，
    __init__ 接收 config 并挂载 self.config）。这里只保留该两点行为。"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def post_init(self):
        pass

    @property
    def dtype(self):
        return next(self.parameters()).dtype


ns = dict(HN.__dict__)
ns.update(dict(get_vision_position_ids=get_vision_position_ids,
               get_vision_attention_seqlens=get_vision_attention_seqlens,
               is_flash_attention_requested=is_flash_attention_requested,
               get_max_seqlen=get_max_seqlen,
               BaseModelOutputWithPooling=BaseModelOutputWithPooling,
               Glm5NextPreTrainedModel=_VisionBase,
               merge_with_config_defaults=HN._identity_decorator,
               capture_outputs=HN._identity_decorator,
               nn=nn, torch=torch))
exec(compile("from __future__ import annotations\n" + vis_src,
             "<official modeling_glm5_next.py L1497-L1824>", "exec"), ns)

VisionModel = ns["Glm5NextVisionModel"]
torch.manual_seed(0)
cfg = VC(depth=2)
vm = VisionModel(cfg).to(torch.float32).eval()
with torch.no_grad():
    for p in vm.parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.02)
        else:
            p.zero_()

hp = wp = 448 // ps
grid = torch.tensor([[1, hp, wp]])
npatch = hp * wp
pix = torch.randn(npatch, V["in_channels"] * tp * ps * ps, dtype=torch.float32)
print(f"      输入 pixel_values {tuple(pix.shape)}  = (patch 数, C*t_patch*p*p = "
      f"{V['in_channels']}*{tp}*{ps}*{ps})")
print(f"      grid_thw = {grid.tolist()}")
with torch.no_grad():
    out = vm(pix, grid_thw=grid)
print(f"      patch_embed 后           ({npatch}, {V['hidden_size']})")
print(f"      经 {cfg.depth} 个 block + post_layernorm 后  ({npatch}, {V['hidden_size']})")
print(f"      last_hidden_state（downsample 后） {tuple(out.last_hidden_state.shape)}")
print(f"      pooler_output（merger 后）        {tuple(out.pooler_output.shape)}")
print(f"      期望视觉 token 数 = {npatch}//{sm*sm} = {npatch//(sm*sm)}  实测="
      f"{out.pooler_output.shape[0]}  一致={out.pooler_output.shape[0]==npatch//(sm*sm)}")
print(f"      输出维度 {out.pooler_output.shape[-1]} == 文本 hidden_size {T['hidden_size']} = "
      f"{out.pooler_output.shape[-1]==T['hidden_size']}")
print(f"      finite={bool(torch.isfinite(out.pooler_output).all())}")

print(f"\n      downsample: Conv2d({V['hidden_size']}→{V['out_hidden_size']}, "
      f"kernel={sm}, stride={sm}) —— 空间 2x2 合并同时升维到文本宽度")
print(f"      merger: proj → LayerNorm → GELU → SwiGLU(context={V['projection_intermediate_size']}) → down")

# ---------- 6.4 图像/视频共用 token id ----------
banner("6.4 图像与视频共用同一 token id（与其他 VLM 的关键差异）")
print(f"      image_token_id = {CFG['image_token_id']}   video_token_id = {CFG['video_token_id']}")
print(f"      image_start/end = {CFG['image_start_token_id']}/{CFG['image_end_token_id']}")
print(f"      video_start/end = {CFG['video_start_token_id']}/{CFG['video_end_token_id']}")
print(f"      源码 L1909-1916 注释明写：img token == vid token，靠 start/end span 区分")
print(f"      判定式：in_video_span = cumsum(==video_start) > cumsum(==video_end)")
print(f"               special_image_mask = (ids==image_token_id) & ~in_video_span")
print(f"               special_video_mask = (ids==image_token_id) &  in_video_span")

# 实测该判定逻辑
IMG, VS, VE = CFG["image_token_id"], CFG["video_start_token_id"], CFG["video_end_token_id"]
ids = torch.tensor([[7, IMG, IMG, 7, VS, IMG, IMG, IMG, VE, 7, IMG, 7]])
in_span = (ids == VS).cumsum(-1) > (ids == VE).cumsum(-1)
mm = ids == IMG
print(f"\n      构造 input_ids（7=普通文本 token）：")
print(f"        {[('IMG' if t==IMG else 'V_S' if t==VS else 'V_E' if t==VE else 'txt') for t in ids[0].tolist()]}")
print(f"        in_video_span  {in_span[0].int().tolist()}")
print(f"        image_mask     {(mm & ~in_span)[0].int().tolist()}  → {int((mm&~in_span).sum())} 个图像 token")
print(f"        video_mask     {(mm &  in_span)[0].int().tolist()}  → {int((mm& in_span).sum())} 个视频 token")
print(f"      注意 video_token_id={CFG['video_token_id']} 在此路径下未被用于掩码判定")

# ---------- 6.5 注入路径 ----------
banner("6.5 视觉特征注入文本序列（masked_scatter）")
D = T["hidden_size"]
emb = torch.zeros(1, 12, D)
nimg = int((mm & ~in_span).sum())
feat = torch.arange(nimg * D, dtype=torch.float32).view(nimg, D)
merged = emb.masked_scatter((mm & ~in_span).unsqueeze(-1), feat)
print(f"      inputs_embeds {tuple(emb.shape)}  image_features {tuple(feat.shape)}")
print(f"      形状校验（L1921-1924）：n_image_tokens * hidden == features.numel() → "
      f"{nimg}*{D} == {feat.numel()}  {nimg*D==feat.numel()}")
print(f"      masked_scatter 后被替换的位置 = "
      f"{[i for i in range(12) if merged[0,i].abs().sum()>0]}")
print(f"      期望位置 = {[i for i,v in enumerate((mm&~in_span)[0].tolist()) if v]}  一致="
      f"{[i for i in range(12) if merged[0,i].abs().sum()>0]==[i for i,v in enumerate((mm&~in_span)[0].tolist()) if v]}")
print(f"      注入后走 language_model（input_ids=None, inputs_embeds=merged），L1973-1981")

# ---------- 6.6 视觉塔参数量 ----------
banner("6.6 视觉塔参数量（真实 depth=24）与 checkpoint 对照")
full = VC(depth=V["depth"])
vm_full = VisionModel(full)
np_full = sum(p.numel() for p in vm_full.parameters())
print(f"      按官方 depth={V['depth']} 构建，参数量 = {np_full:,} = {np_full/1e6:.2f} M")
print(f"      checkpoint 实测 visual 命名空间参数量 = 563,627,008 = 563.63 M")
print(f"      一致 = {np_full == 563_627_008}   差值 = {np_full - 563_627_008}")
print(f"      视觉塔占全模型（321.32 B）比例 = {np_full/321_323_031_390*100:.3f}%")
print(f"      视觉塔全部为 BF16（未量化），checkpoint dtype 交叉验证见 verify_structure [8]")
