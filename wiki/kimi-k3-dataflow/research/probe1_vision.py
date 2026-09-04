# 实测一：视觉编码器（MoonViT-V2 / MoonViT3d）数据流。
# 组件全部来自官方 modeling_kimi_k3.py 原文（经 _loader.py 加载），
# 维度取官方 config.json 的真实值：vt_hidden_size=1024、27 层、12 头、
# qkv_hidden_size=1536（头维 128）、patch 14、pos_emb 64x64x4、merger 2x2。
import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from _loader import load_official

off = load_official("modeling_kimi_k3.py")
torch.manual_seed(0)

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name} {detail}")
    else:
        fail += 1
        print(f"  FAIL  {name} {detail}")

D = 1024   # vt_hidden_size
L = 27     # vt_num_hidden_layers
H = 12     # vt_num_attention_heads
QKV = 1536 # qkv_hidden_size
HEAD = QKV // H  # 128
PATCH = 14
MID = 4096 # vt_intermediate_size

print("== 1. patch_embed (Conv2d, kernel=stride=14, 无 bias) ==")
pe = off.MoonVision3dPatchEmbed(
    out_dim=D, patch_size=PATCH, pos_emb_height=64, pos_emb_width=64,
    pos_emb_time=4, pos_emb_type="divided_fixed",
    patch_embed_proj_bias=False, pos_emb_interpolation_mode="bilinear")
check("proj 是 Conv2d", isinstance(pe.proj, nn.Conv2d),
      f"kernel={pe.proj.kernel_size} stride={pe.proj.stride} bias={pe.proj.bias is not None}")
w = pe.proj.weight
check("proj 权重形状", tuple(w.shape) == (D, 3, PATCH, PATCH), f"{tuple(w.shape)}")
n_param = w.numel()
print(f"  INFO  patch_embed 权重参数 {n_param:,} = 3*14*14*{D}")

x = torch.randn(4, 3, PATCH, PATCH)  # 4 个 patch 的像素
grid = torch.tensor([[1, 2, 2]])     # 1 帧 2x2 patch
out = pe(x, grid)
check("patch+位置嵌入输出形状", tuple(out.shape) == (4, D), f"{tuple(out.shape)}")

print("== 2. 位置嵌入 divided_fixed（64x64 可学习 + 时间 sincos 冻结） ==")
pos = pe.pos_emb
check("可学习 2D 表", tuple(pos.weight.shape) == (64, 64, D), f"{tuple(pos.weight.shape)}")
check("time_weight 是 buffer", "time_weight" not in dict(pos.named_parameters()))
check("时间表形状", tuple(pos.time_weight.shape) == (4, 1, D), f"{tuple(pos.time_weight.shape)}")
# 双线性插值到 (28,40)
emb = pos.weight.detach()
interp = torch.nn.functional.interpolate(
    emb.permute(2, 0, 1).unsqueeze(0), size=(28, 40), mode="bilinear"
).squeeze(0).permute(1, 2, 0).flatten(end_dim=1)
check("64x64 -> (28,40) 插值", tuple(interp.shape) == (28 * 40, D), f"{tuple(interp.shape)}")
# t=3：3 份 2D + 前 3 帧时间嵌入相加（官方排列：[t, hw, D] 广播相加后 reshape(-1, D)）
# 官方 forward 返回 x + pos_embs，manual 比较 pos_embs 部分需从 out3 减去 x3
x3 = torch.randn(3 * 28 * 40, D)
g3 = torch.tensor([[3, 28, 40]])
out3 = pos(x3, g3)
manual = interp.repeat(3, 1) + pos.time_weight[0:3].squeeze(1).repeat_interleave(28 * 40, dim=0)
check("t=3 = 2D 重复 3 份 + time_weight[0:3]",
      torch.allclose(out3 - x3, manual, atol=1e-4),
      f"max diff {(out3 - x3 - manual).abs().max().item():.2e}")
# t 上限 assert（init_pos_emb_time=4）
try:
    pos(torch.randn(5 * 4 * 4, D), torch.tensor([[5, 4, 4]]))
    check("t>4 报错", False, "未触发 assert")
except AssertionError:
    check("t>4 报错", True, "assert t <= num_frames 生效（时间表仅 4 帧）")

print("== 3. 2D RoPE（Rope2DPosEmbRepeated，x/y 交错频率） ==")
rope = off.Rope2DPosEmbRepeated(dim=HEAD, max_height=512, max_width=512)
check("dim 必须被 4 整除", HEAD % 4 == 0, f"head_dim={HEAD}")
fc = rope.get_freqs_cis(torch.tensor([[1, 3, 5]]), torch.device("cpu"))
check("freqs_cis 返回展平形状", tuple(fc.shape) == (15, HEAD // 2), f"{tuple(fc.shape)} (complex64={fc.dtype})")
fcm = rope.freqs_cis  # 内部 2D 网格 buffer [512, 512, 64]
check("内部网格形状", tuple(fcm.shape) == (512, 512, HEAD // 2), f"{tuple(fcm.shape)}")
check("展平返回与网格一致", torch.allclose(fc, fcm[:3, :5].reshape(-1, HEAD // 2)))
check("偶数槽只依赖 w", torch.allclose(fcm[1, 0, 0::2], fcm[3, 0, 0::2]),
      "代码 x_pos=flat%W(宽) 进偶数槽：freqs_cis[h,w,2i] 与 h 无关")
check("奇数槽只依赖 h", torch.allclose(fcm[1, 0, 1::2], fcm[1, 4, 1::2]),
      "代码 y_pos=flat//W(高) 进奇数槽：freqs_cis[h,w,2i+1] 与 w 无关")
# 语义严格验证：偶数槽随 w 变、奇数槽随 h 变（以代码为准，docstring 标注与此相反）
check("偶数槽随 w 变", not torch.allclose(fcm[1, 0, 0::2], fcm[1, 4, 0::2]))
check("奇数槽随 h 变", not torch.allclose(fcm[1, 0, 1::2], fcm[3, 0, 1::2]))
print("  INFO  源码 docstring 写 ret[h,w,2i]=cis(h*theta)，但实现里偶数槽是 x_pos=flat%W（宽），"
      "与 docstring 相反；本表按代码实际行为记录")
# 保范性
q = torch.randn(15, H, HEAD) * 3.0
k = torch.randn(15, H, HEAD) * 3.0
qr, kr = off.apply_rope(q, k, fc)
check("旋转保范数(q)", torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4),
      f"{q.norm(dim=-1).mean():.6f} -> {qr.norm(dim=-1).mean():.6f}")
check("旋转保范数(k)", torch.allclose(k.norm(dim=-1), kr.norm(dim=-1), atol=1e-4))
# 超出 512 报错
try:
    rope.get_freqs_cis(torch.tensor([[1, 600, 4]]), torch.device("cpu"))
    check("h>512 报错", False)
except AssertionError:
    check("h>512 报错", True, "max_height=max_width=512 硬上限")

print("== 4. 27 层 block（RMSNorm + GELU-MLP2 + 双向注意力，无 bias） ==")
layer = off.MoonViTEncoderLayer(
    num_heads=H, hidden_dim=D, mlp_dim=MID, qkv_hidden_size=QKV,
    norm_type="rmsnorm", mlp_type="mlp2",
    attn_implementation="eager",
    attn_bias=False, linear_bias=False)
check("wqkv 形状无 bias", tuple(layer.wqkv.weight.shape) == (QKV * 3, D) and layer.wqkv.bias is None,
      f"{tuple(layer.wqkv.weight.shape)}")
check("wo 形状无 bias", tuple(layer.wo.weight.shape) == (D, QKV) and layer.wo.bias is None)
check("norm0/norm1 是 RMSNorm", isinstance(layer.norm0, nn.RMSNorm) and isinstance(layer.norm1, nn.RMSNorm))
check("mlp fc0/fc1 无 bias",
      tuple(layer.mlp.fc0.weight.shape) == (MID, D) and layer.mlp.fc0.bias is None
      and tuple(layer.mlp.fc1.weight.shape) == (D, MID) and layer.mlp.fc1.bias is None)
# 头维 = qkv_hidden_size/num_heads = 128，不等于 hidden/heads
check("头维 128 != hidden/heads", HEAD == 128 and D // H != HEAD,
      f"head_dim={HEAD}, hidden/heads={D / H:.2f}")

print("== 5. 变长打包：两图互不可见 ==")
enc = off.MoonViT3dEncoder(
    hidden_dim=D, num_layers=L,
    block_cfg=dict(num_heads=H, hidden_dim=D, qkv_hidden_size=QKV,
                   mlp_dim=MID, norm_type="rmsnorm", mlp_type="mlp2",
                   activation=off.PytorchGELUTanh(), attn_bias=False,
                   linear_bias=False, attn_implementation="eager"),
    use_deterministic_attn=False)
check("encoder 层数", len(enc.blocks) == L, f"{len(enc.blocks)}")
check("final_layernorm 是 RMSNorm", isinstance(enc.final_layernorm, nn.RMSNorm))
enc.eval()
with torch.no_grad():
    # 图 A: (1,2,2)=4 patch；图 B: (1,3,3)=9 patch
    gA, gB = torch.tensor([[1, 2, 2]]), torch.tensor([[1, 3, 3]])
    xA, xB = torch.randn(4, D), torch.randn(9, D)
    gAB = torch.tensor([[1, 2, 2], [1, 3, 3]])
    xAB = torch.cat([xA, xB], 0)
    outA = enc(xA.clone(), gA)
    outB = enc(xB.clone(), gB)
    outAB = enc(xAB, gAB)
dA = (outA - outAB[:4]).abs().max().item()
dB = (outB - outAB[4:]).abs().max().item()
check("拼接跑 == 分开跑（图A）", dA < 1e-4, f"max diff {dA:.2e}")
check("拼接跑 == 分开跑（图B）", dB < 1e-4, f"max diff {dB:.2e}")
print("  INFO  两图各自输出与拼在一起逐元素一致 => 跨图注意力为 0（cu_seqlens 块对角掩码）")

print("== 6. tpool_patch_merger（时间全池化 + 2x2 空间合并） ==")
with torch.no_grad():
    # (t=3, h=8, w=10, d)：手工构造使每帧可辨识
    t, h, w_ = 3, 8, 10
    seq = torch.zeros(t * h * w_, D)
    for ti in range(t):
        seq[ti * h * w_:(ti + 1) * h * w_] = float(ti + 1)  # 第 1/2/3 帧填 1/2/3
    outs = off.tpool_patch_merger(seq.unsqueeze(0).squeeze(0) if False else seq,
                                  torch.tensor([[t, h, w_]]))
    check("输出段数", len(outs) == 1)
    o = outs[0]
    check("输出形状", tuple(o.shape) == (h // 2 * w_ // 2, 2 * 2, D),
          f"{tuple(o.shape)} = (40 token, 4 patch, {D})")
    # 时间池化 = 3 帧平均：全帧均值应处处为 (1+2+3)/3=2
    check("时间池化是均值", torch.allclose(o, torch.full_like(o, 2.0)),
          "第 1/2/3 帧填 1/2/3，合并后处处 = 2.0")
    # 空间 2x2：帧内只有 (2,4) 位置的 patch 填 1 其余 0，对应输出 token 应恰为 1/4? 直接验证 t=1 时输出 token 数
    o1 = off.tpool_patch_merger(torch.randn(1 * h * w_, D), torch.tensor([[1, h, w_]]))[0]
    check("t=1 与 t=3 输出 token 数相同", o1.shape[0] == o.shape[0],
          f"t=1 -> {tuple(o1.shape)}；输出 token 数与帧数 t 无关")
    # t=4（上限）
    o4 = off.tpool_patch_merger(torch.randn(4 * h * w_, D), torch.tensor([[4, h, w_]]))[0]
    check("t=4 同样 20 token", o4.shape[0] == 20)

print("== 7. PatchMergerMLPV2（4096->4096->7168 + RMSNorm，无 bias） ==")
class PCfg:
    mm_hidden_size = D
    hidden_size = 7168
    merge_kernel_size = [2, 2]
    projector_ln_eps = 1e-5
pm = off.PatchMergerMLPV2(PCfg())
check("proj[0] Linear(4096,4096) 无 bias",
      tuple(pm.proj[0].weight.shape) == (4096, 4096) and pm.proj[0].bias is None)
check("proj[2] Linear(4096,7168) 无 bias",
      tuple(pm.proj[2].weight.shape) == (7168, 4096) and pm.proj[2].bias is None)
check("post_norm RMSNorm(7168)", isinstance(pm.post_norm, nn.RMSNorm) and tuple(pm.post_norm.weight.shape) == (7168,))
check("V2 无 pre_norm", not hasattr(pm, "pre_norm"),
      "V1 才有 pre_norm(LayerNorm@1024)；V2 直接 proj 后接 RMSNorm")
import inspect
src_pm = inspect.getsource(off.PatchMergerMLPV2.forward)
check("forward 未使用 LayerNorm 前置", "LayerNorm" not in src_pm and "pre_norm" not in src_pm)
with torch.no_grad():
    feats = [torch.randn(40, 4, D)]
    y = pm(feats)
    check("merger 输出形状", tuple(y[0].shape) == (40, 7168), f"{tuple(y[0].shape)}")

print("== 8. 视觉 token 数与参数量 ==")
def n_vis(pw, ph):  # 像素宽高 -> patch grid -> 合并后 token
    gw, gh = pw // PATCH, ph // PATCH
    return (gh // 2) * (gw // 2)
for (pw, ph), name in [((448, 448), "448x448"), ((896, 896), "896x896"),
                       ((1344, 1344), "1344x1344"), ((1820, 1008), "1820x1008")]:
    if pw % PATCH or ph % PATCH:
        print(f"  INFO  {name}: 非 14 整除，processor 需先 resize")
        continue
    print(f"  INFO  {name}: patch grid {ph//PATCH}x{pw//PATCH} -> 视觉 token {n_vis(pw, ph)}（与帧数无关）")
print(f"  INFO  4 帧 448x448 视频: t=4 -> 时间池化后仍 {n_vis(448,448)} token（init_pos_emb_time=4 限制 t<=4）")

# 参数量
p_layer = (QKV * 3 * D) + (D * QKV) + (D * MID) + (MID * D) + 2 * D  # wqkv+wo+mlp+2*RMSNorm
p_enc = L * p_layer + D
p_patch = 3 * PATCH * PATCH * D
p_posemb = 64 * 64 * D
p_tower = p_enc + p_patch + p_posemb
p_merger = (4096 * 4096) + (4096 * 7168) + 7168  # proj 两层 + post_norm(RMSNorm 仅 weight)
print(f"  INFO  单层参数 {p_layer:,}")
print(f"  INFO  视觉塔参数 = 27层 {L*p_layer:,} + final_norm {D} + patch_embed {p_patch:,} + pos_emb {p_posemb:,} = {p_tower:,} ({p_tower/1e6:.1f}M)")
print(f"  INFO  merger 参数 = {p_merger:,} ({p_merger/1e6:.1f}M)")
print(f"  INFO  视觉侧合计 {p_tower + p_merger:,} ({(p_tower+p_merger)/1e6:.1f}M)；技术报告称 MoonViT-V2 401M -> 塔本体 {p_tower/1e6:.1f}M 吻合")

print(f"\n== 汇总: PASS {ok} / FAIL {fail} ==")
sys.exit(1 if fail else 0)
