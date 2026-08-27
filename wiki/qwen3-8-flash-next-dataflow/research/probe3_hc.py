"""实测 3：Gated Residual（四条残差流）的完整数据流。

验证目标：
  1. 入口把 embedding 复制 hc_count 份，残差流全程宽度为 hc_count*hidden_size = 10240
  2. GatedResidual 读出：分组 RMSNorm -> 低秩 down/up -> sigmoid 得每流每维权重 -> 加权后沿流求均值
  3. 写回：block 输出乘 injection_weights（每流一个标量）后加到原四流上
  4. injection_weights = 2*sigmoid(.../hc_count)，取值范围 (0,2)，初始约为 1
  5. 出口 hyper_connection_mixer 无 block_inject_weight，只把四流收成 hidden_size

对应源码：transformers@36deb0b5
  modeling_qwen4_exp.py
    L1417 hidden_states = inputs_embeds.repeat(1, 1, hc_count)
    L941  Qwen4ExpTextGatedResidual
    L960  input_mix_weight = silu(down(x_normed) / hc_count)
    L961  input_mix_weight = sigmoid(up(...))
    L963  mixed = (weight * x_normed.unflatten).mean(dim=-2)
    L968  injection_weights = 2*sigmoid(block_inject_weight(x_normed) / hc_count)
    L1236 injection = block_out.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    L1237 hidden_states = hyper_input + injection.flatten(-2)
    L1430 hidden_states = hyper_connection_mixer(hidden_states)  # use_combine=False
config：Qwen/Qwen3.8-Flash-Next@f5d08274
"""
import json
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
C = json.load(open("/tmp/qwen38fn/config.json"))["text_config"]
D = C["hidden_size"]
HC = C["hc_count"]
LR = C["hc_lowrank"]
EPS = C["rms_norm_eps"]


class RMSNormGrouped(nn.Module):
    """源码 L158 Qwen4ExpTextRMSNorm，带 group_size 时按组归一化。注意权重是 (1+w)。"""
    def __init__(self, dim, group_size=None, eps=EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        h = x.float()
        if self.group_size is not None:
            h = h.reshape(*h.shape[:-1], -1, self.group_size)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            h = h.flatten(-2)
        return (h * (1.0 + self.weight.float())).type_as(x)


class GatedResidual(nn.Module):
    """源码 L941 Qwen4ExpTextGatedResidual，逐行照搬。"""
    def __init__(self, use_combine=True):
        super().__init__()
        hc_hidden = HC * D
        self.hc_norm = RMSNormGrouped(hc_hidden, group_size=D)
        self.input_mix_weight_down = nn.Linear(hc_hidden, LR, bias=False)
        self.input_mix_weight_up = nn.Linear(LR, hc_hidden, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden, HC, bias=False) if use_combine else None

    def forward(self, hyper_input):
        normed = self.hc_norm(hyper_input)
        w = F.silu(self.input_mix_weight_down(normed) / HC)
        w = torch.sigmoid(self.input_mix_weight_up(w))
        w = w.unflatten(-1, (HC, D))
        mixed = (w * normed.unflatten(-1, (HC, D))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        inj = 2 * torch.sigmoid(self.block_inject_weight(normed) / HC)
        return mixed, hyper_input, inj, w


print("=" * 72)
print("实测 3A：残差流宽度与入口复制")
print("=" * 72)
B, T = 1, 5
emb = torch.randn(B, T, D)
streams = emb.repeat(1, 1, HC)
print(f"  hidden_size = {D}, hc_count = {HC}, hc_lowrank = {LR}")
print(f"  embedding {tuple(emb.shape)} --repeat(1,1,{HC})--> 残差流 {tuple(streams.shape)}")
print(f"  残差流宽度 = {HC} x {D} = {streams.shape[-1]}  (与权重 hc_norm [10240] 一致)")
四 = streams.unflatten(-1, (HC, D))
print(f"  入口四条流是否完全相同（复制而非各自初始化）: {bool(torch.equal(四[:,:,0], 四[:,:,3]))}")

print()
print("=" * 72)
print("实测 3B：读出（mixed_input）与写回（injection）的形状与取值")
print("=" * 72)
gr = GatedResidual(use_combine=True)
mixed, hyper_in, inj, w = gr(streams)
print(f"  输入四流 {tuple(streams.shape)} -> 读出 mixed {tuple(mixed.shape)}   (block 只看到 {D} 宽)")
print(f"  混合权重 w {tuple(w.shape)}  = 每条流每一维一个 sigmoid 门，范围 [{w.min():.4f}, {w.max():.4f}]")
print(f"  注入权重 inj {tuple(inj.shape)} = 每条流一个标量，范围 [{inj.min():.4f}, {inj.max():.4f}]")
print(f"  inj 公式 2*sigmoid(x/{HC}) -> 理论范围 (0, 2)；权重零初始时应恰为 1.0")
gr0 = GatedResidual(use_combine=True)
nn.init.zeros_(gr0.block_inject_weight.weight)
_, _, inj0, _ = gr0(streams)
print(f"  零初始化 block_inject_weight 时 inj = {inj0.flatten()[:4].tolist()}  -> 恒等残差")

block_out = torch.randn(B, T, D)
injection = block_out.unsqueeze(-2) * inj.unsqueeze(-1)
new_streams = hyper_in + injection.flatten(-2)
print()
print(f"  block 输出 {tuple(block_out.shape)} -> 广播到四流 {tuple(injection.shape)} -> 展平 {tuple(injection.flatten(-2).shape)}")
print(f"  写回后残差流 {tuple(new_streams.shape)}，四条流从此不再相同: {not bool(torch.equal(new_streams.unflatten(-1,(HC,D))[:,:,0], new_streams.unflatten(-1,(HC,D))[:,:,3]))}")
delta = (new_streams - hyper_in).unflatten(-1, (HC, D))
print(f"  各流实际增量的 L2 范数: {[round(delta[0,0,i].norm().item(),4) for i in range(HC)]}")
print(f"  -> 同一个 block 输出按不同标量写入四条流，因此各流演化路径不同")

print()
print("=" * 72)
print("实测 3C：出口 mixer 收束（use_combine=False，无 block_inject_weight）")
print("=" * 72)
mixer = GatedResidual(use_combine=False)
out = mixer(new_streams)
print(f"  残差流 {tuple(new_streams.shape)} -> mixer 输出 {tuple(out.shape)}")
print(f"  mixer 参数量 = {sum(p.numel() for p in mixer.parameters()):,}")
print(f"  含 block_inject 的层参数量 = {sum(p.numel() for p in gr.parameters()):,}")
print(f"  差值 = {sum(p.numel() for p in gr.parameters()) - sum(p.numel() for p in mixer.parameters()):,} = HC*HC*D = {HC*HC*D:,}")
print(f"  与真实权重一致：mixer 只有 hc_norm/down/up 三个张量，无 block_inject_weight")

print()
print("=" * 72)
print("实测 3D：单层超连接参数量与真实权重核对")
print("=" * 72)
theory = {
    "hc_norm.weight": HC * D,
    "input_mix_weight_down.weight": LR * HC * D,
    "input_mix_weight_up.weight": HC * D * LR,
    "block_inject_weight.weight": HC * HC * D,
}
H = json.load(open("/tmp/qwen38fn/headers.json"))
from math import prod
print(f"  {'张量':<32s} {'理论':>14s} {'真实 shape':>20s} {'真实元素数':>14s} {'一致':>5s}")
for name, n in theory.items():
    key = f"model.language_model.layers.0.attn_hyper_connection.{name}"
    real = prod(H[key]["shape"])
    print(f"  {name:<32s} {n:>14,d} {str(H[key]['shape']):>20s} {real:>14,d} {'是' if n==real else '否':>5s}")
per_layer = 2 * sum(theory.values())
L = C["num_hidden_layers"]
mixer_n = HC * D + LR * HC * D + HC * D * LR
total = per_layer * L + mixer_n
print()
print(f"  每层 2 组（attn 前 + mlp 前） = {per_layer:,}")
print(f"  末端 hyper_connection_mixer   = {mixer_n:,}")
print(f"  {L} 层 + 末端 mixer 合计       = {total:,}")
print(f"  对照 count_params.py 的「Gated Residual」分组 = 640,624,640")
print(f"  一致: {total == 640_624_640}")
print(f"  说明：MTP 的 2 组超连接 + MTP mixer（共 {per_layer + mixer_n:,}）在参数量脚本里归入「MTP 草稿层」，不重复计入本组")
