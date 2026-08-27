"""实测 5：MoE 路由 + 端到端逐层数据流形状。

验证目标：
  1. 路由是 softmax 后 top-k，再对被选中的 k 个概率重新归一化（norm_topk_prob=True）
  2. 共享专家始终参与，且被一个 sigmoid 标量门缩放
  3. 单层 MoE 的激活参数量核算
  4. 用一个按真实结构等比缩小的模型跑通完整前向，抓每层输入输出形状，
     确认残差流全程 hc_count*hidden_size、block 内部 hidden_size 的收放关系
  5. QSA 层与 GDN 层在缓存上的差异：前者 KV 随长度增长，后者常数状态

对应源码：transformers@36deb0b5
  modeling_qwen4_exp.py
    L898  Qwen4ExpTextTopKRouter：softmax -> topk -> 归一化
    L919  Qwen4ExpTextSparseMoeBlock：共享专家 + sigmoid 门
    L1207 Qwen4ExpTextDecoderLayer.forward 的完整顺序
config：Qwen/Qwen3.8-Flash-Next@f5d08274
"""
import json, math
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
C = json.load(open("/tmp/qwen38fn/config.json"))["text_config"]
D = C["hidden_size"]
E = C["num_experts"]
K = C["num_experts_per_tok"]
IM = C["moe_intermediate_size"]
SI = C["shared_expert_intermediate_size"]
HC = C["hc_count"]

print("=" * 72)
print("实测 5A：路由的 softmax -> top-k -> 重归一化")
print("=" * 72)
T = 8
h = torch.randn(T, D)
W = torch.randn(E, D) * 0.02
logits = F.linear(h, W)
probs = F.softmax(logits, dtype=torch.float, dim=-1)
top_v, top_i = torch.topk(probs, K, dim=-1)
print(f"  num_experts = {E}, num_experts_per_tok = {K}, norm_topk_prob = {C.get('norm_topk_prob', True)}")
print(f"  router_logits {tuple(logits.shape)} -> softmax 全 {E} 个专家，和 = {probs[0].sum().item():.6f}")
print(f"  top-{K} 原始概率和（token 0）= {top_v[0].sum().item():.6f}  （远小于 1，因为只取了 {K}/{E}）")
normed = top_v / top_v.sum(dim=-1, keepdim=True)
print(f"  重归一化后和 = {normed[0].sum().item():.6f}")
print(f"  token 0 选中专家 = {top_i[0].tolist()}")
print(f"  token 0 归一化权重 = {[round(x,4) for x in normed[0].tolist()]}")
print(f"  -> 先在全 {E} 个专家上 softmax，再对选中的 {K} 个重新归一化，与 num_experts_per_tok 之和恒为 1")

print()
print("=" * 72)
print("实测 5B：共享专家的 sigmoid 标量门")
print("=" * 72)
shared_gate = nn.Linear(D, 1, bias=False)
g = torch.sigmoid(shared_gate(h))
print(f"  shared_expert_gate 权重 shape = {tuple(shared_gate.weight.shape)}  (真实权重 [1, 2560])")
print(f"  每 token 一个标量门，范围 [{g.min():.4f}, {g.max():.4f}]，恒在 (0,1)")
print(f"  源码 L934: shared_out = sigmoid(gate(x)) * shared_expert(x)")
print(f"  源码 L936: 最终输出 = 路由专家加权和 + 门控后的共享专家（共享专家无条件参与）")

print()
print("=" * 72)
print("实测 5C：单层 MoE 参数量与激活量")
print("=" * 72)
per_expert = 2 * IM * D + D * IM
routed_total = E * per_expert
shared = 2 * SI * D + D * SI
router = E * D
sgate = D
print(f"  单个路由专家 = gate_up(2*{IM}*{D}) + down({D}*{IM}) = {per_expert:,}")
print(f"  {E} 个路由专家全量 = {routed_total:,} = {routed_total/1e9:.3f} B")
print(f"  共享专家 = {shared:,}")
print(f"  路由器 = {E}*{D} = {router:,}，共享门 = {sgate:,}")
print(f"  单层 MoE 全量 = {routed_total+shared+router+sgate:,}")
act = K * per_expert + shared + router + sgate
print(f"  单层 MoE 激活 = {K}*{per_expert:,} + 共享 + 路由器 = {act:,}")
print(f"  激活占全量比例 = {act/(routed_total+shared+router+sgate)*100:.3f}%  （约 {K}/{E} = {K/E*100:.2f}% 加上共享部分）")
L = C["num_hidden_layers"]
print(f"  {L} 层路由专家合计 = {routed_total*L:,} = {routed_total*L/1e9:.2f} B")
from math import prod
Hd = json.load(open("/tmp/qwen38fn/headers.json"))
real_routed = sum(prod(Hd[k]["shape"]) for k in Hd if ".mlp.experts." in k and not k.startswith("mtp."))
print(f"  真实权重中语言主干路由专家 = {real_routed:,} = {real_routed/1e9:.2f} B   一致: {real_routed == routed_total*L}")

print()
print("=" * 72)
print("实测 5D：按真实结构等比缩小的模型跑通完整前向，抓逐层形状")
print("=" * 72)
# 缩小配置：保留全部结构特征（GDN/QSA 每 4 层交替、四流超连接、PLE 在第 2 层、MoE top-k+共享）
sD, sHC, sE, sK = 64, 4, 8, 2
sNK, sNV, sDK, sDV = 2, 6, 8, 8
sNH, sNKV, sHD = 4, 2, 16
sLayers = ["linear_attention", "linear_attention", "linear_attention", "qwen_sparse_attention"] * 2
print(f"  缩小配置：hidden={sD}, hc_count={sHC}, 层类型={['L','L','L','Q','L','L','L','Q']}")
print(f"  GDN: {sNK} k 头 x {sDK} / {sNV} v 头 x {sDV}    QSA: {sNH} q 头 / {sNKV} kv 头 x {sHD}")
print(f"  MoE: {sE} 专家 top-{sK} + 共享专家")


class RMSN(nn.Module):
    def __init__(self, dim, group_size=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size

    def forward(self, x):
        h = x.float()
        if self.group_size is not None:
            h = h.reshape(*h.shape[:-1], -1, self.group_size)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        if self.group_size is not None:
            h = h.flatten(-2)
        return (h * (1.0 + self.weight.float())).type_as(x)


class GR(nn.Module):
    def __init__(self, use_combine=True):
        super().__init__()
        w = sHC * sD
        self.hc_norm = RMSN(w, group_size=sD)
        self.down = nn.Linear(w, 16, bias=False)
        self.up = nn.Linear(16, w, bias=False)
        self.inject = nn.Linear(w, sHC, bias=False) if use_combine else None

    def forward(self, x):
        n = self.hc_norm(x)
        w = torch.sigmoid(self.up(F.silu(self.down(n) / sHC))).unflatten(-1, (sHC, sD))
        mixed = (w * n.unflatten(-1, (sHC, sD))).mean(dim=-2)
        if self.inject is None:
            return mixed
        return mixed, x, 2 * torch.sigmoid(self.inject(n) / sHC)


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(sD, sE, bias=False)
        self.gu = nn.Parameter(torch.randn(sE, 2 * sD, sD) * 0.02)
        self.dn = nn.Parameter(torch.randn(sE, sD, sD) * 0.02)
        self.sh_gu = nn.Linear(sD, 2 * sD, bias=False)
        self.sh_dn = nn.Linear(sD, sD, bias=False)
        self.sh_gate = nn.Linear(sD, 1, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        f = x.reshape(-1, sD)
        sg, su = self.sh_gu(f).chunk(2, dim=-1)
        shared = self.sh_dn(F.silu(sg) * su)
        p = F.softmax(self.gate(f), dtype=torch.float, dim=-1)
        v, i = torch.topk(p, sK, dim=-1)
        v = (v / v.sum(-1, keepdim=True)).to(f.dtype)
        out = torch.zeros_like(f)
        for e in range(sE):
            pos, tok = torch.where(F.one_hot(i, sE).permute(2, 1, 0)[e])
            if tok.numel() == 0:
                continue
            cs = f[tok]
            gg, uu = F.linear(cs, self.gu[e]).chunk(2, dim=-1)
            o = F.linear(F.silu(gg) * uu, self.dn[e]) * v[tok, pos, None]
            out.index_add_(0, tok, o.to(out.dtype))
        out = out + torch.sigmoid(self.sh_gate(f)) * shared
        return out.reshape(B, T, sD)


class GDN(nn.Module):
    def __init__(self):
        super().__init__()
        kd, vd = sNK * sDK, sNV * sDV
        self.cd = 2 * kd + vd
        self.qkv = nn.Linear(sD, self.cd, bias=False)
        self.z = nn.Linear(sD, vd, bias=False)
        self.b = nn.Linear(sD, sNV, bias=False)
        self.a = nn.Linear(sD, sNV, bias=False)
        self.conv = nn.Conv1d(self.cd, self.cd, 4, groups=self.cd, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(sNV).uniform_(0.01, 16)))
        self.dt = nn.Parameter(torch.ones(sNV))
        self.norm = RMSN(sDV)
        self.out = nn.Linear(vd, sD, bias=False)
        self.state_shape = (sNV, sDK, sDV)

    def forward(self, x, cache=None):
        """cache = (recurrent_state, conv_state)；conv_state 长度 kernel-1=3，
        对应官方 cache_params.update_conv_state（源码 L486 / L477）。"""
        B, T, _ = x.shape
        state, conv_state = (None, None) if cache is None else cache
        raw = self.qkv(x).transpose(1, 2)                       # [B, cd, T]
        if conv_state is None:
            conv_state = raw.new_zeros(B, self.cd, 3)
        win = torch.cat([conv_state, raw], dim=-1)              # 前置 3 个历史位置
        qkv = F.silu(self.conv(win)).transpose(1, 2)            # 输出长度恰为 T
        new_conv_state = win[..., -3:]
        kd = sNK * sDK
        q, k, v = torch.split(qkv, [kd, kd, sNV * sDV], dim=-1)
        q = q.reshape(B, T, sNK, sDK).repeat_interleave(sNV // sNK, dim=2)
        k = k.reshape(B, T, sNK, sDK).repeat_interleave(sNV // sNK, dim=2)
        v = v.reshape(B, T, sNV, sDV)
        beta = self.b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.a(x).float() + self.dt)
        qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        qn, kn, vv, bb, gg = [t.transpose(1, 2).float() for t in (qn, kn, v, beta, g)]
        qn = qn * (1 / sDK**0.5)
        S = torch.zeros(B, sNV, sDK, sDV) if state is None else state.clone()
        o = torch.zeros(B, sNV, T, sDV)
        for t in range(T):
            S = S * gg[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
            mem = (S * kn[:, :, t].unsqueeze(-1)).sum(-2)
            dl = (vv[:, :, t] - mem) * bb[:, :, t].unsqueeze(-1)
            S = S + kn[:, :, t].unsqueeze(-1) * dl.unsqueeze(-2)
            o[:, :, t] = (S * qn[:, :, t].unsqueeze(-1)).sum(-2)
        o = o.transpose(1, 2).reshape(-1, sDV)
        o = self.norm(o) * torch.sigmoid(self.z(x).reshape(-1, sDV))
        return self.out(o.reshape(B, T, -1)), (S, new_conv_state)


class QSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(sD, sNH * sHD * 2, bias=False)
        self.k = nn.Linear(sD, sNKV * sHD, bias=False)
        self.v = nn.Linear(sD, sNKV * sHD, bias=False)
        self.o = nn.Linear(sNH * sHD, sD, bias=False)
        self.qn, self.kn = RMSN(sHD), RMSN(sHD)
        self.idx = nn.Linear(sD, (2 + 1) * 8, bias=False)

    def forward(self, x, kv=None):
        B, T, _ = x.shape
        q, gate = torch.chunk(self.q(x).view(B, T, -1, sHD * 2), 2, dim=-1)
        q = self.qn(q.view(B, T, sNH, sHD)).transpose(1, 2)
        k = self.kn(self.k(x).view(B, T, sNKV, sHD)).transpose(1, 2)
        v = self.v(x).view(B, T, sNKV, sHD).transpose(1, 2)
        if kv is not None:
            k = torch.cat([kv[0], k], dim=2)
            v = torch.cat([kv[1], v], dim=2)
        kk = k.repeat_interleave(sNH // sNKV, dim=1)
        vv = v.repeat_interleave(sNH // sNKV, dim=1)
        S = (q @ kk.transpose(-1, -2)) / math.sqrt(sHD)
        Tk = kk.shape[2]
        m = torch.ones(T, Tk, dtype=torch.bool).tril(diagonal=Tk - T)
        S = S.masked_fill(~m, float("-inf"))
        a = (S.softmax(-1) @ vv).transpose(1, 2).reshape(B, T, -1)
        a = a * torch.sigmoid(gate.reshape(B, T, -1))
        return self.o(a), (k, v)


class Layer(nn.Module):
    def __init__(self, lt, has_ple):
        super().__init__()
        self.lt = lt
        self.block = GDN() if lt == "linear_attention" else QSA()
        self.moe = MoE()
        self.hc1, self.hc2 = GR(), GR()
        self.ple = nn.Linear(sD, sHC * sD, bias=False) if has_ple else None

    def forward(self, s, cache=None, ple_feat=None):
        if self.ple is not None:
            s = s + self.ple(ple_feat)
        h, hyper, inj = self.hc1(s)
        if self.lt == "linear_attention":
            h, new_cache = self.block(h, cache)
        else:
            h, new_cache = self.block(h, cache)
        s = hyper + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        h, hyper, inj = self.hc2(s)
        h = self.moe(h)
        s = hyper + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        return s, new_cache


emb = nn.Embedding(100, sD)
layers = nn.ModuleList([Layer(lt, i == 1) for i, lt in enumerate(sLayers)])
mixer = GR(use_combine=False)
head = nn.Linear(sD, 100, bias=False)

ids = torch.randint(0, 100, (1, 12))
x = emb(ids)
s = x.repeat(1, 1, sHC)
ple_feat = torch.randn(1, 12, sD)
print()
print(f"  input_ids {tuple(ids.shape)} -> embedding {tuple(x.shape)} -> repeat x{sHC} -> 残差流 {tuple(s.shape)}")
caches = [None] * len(sLayers)
for i, lyr in enumerate(layers):
    before = tuple(s.shape)
    s, caches[i] = lyr(s, caches[i], ple_feat)
    tag = "GDN " if lyr.lt == "linear_attention" else "QSA "
    ple_tag = " +PLE" if lyr.ple is not None else "     "
    if lyr.lt == "linear_attention":
        cinfo = f"递归状态 {tuple(caches[i][0].shape)} + conv 状态 {tuple(caches[i][1].shape)}（均为常数）"
    else:
        cinfo = f"KV {tuple(caches[i][0].shape)}（随长度增长）"
    print(f"  层 {i} {tag}{ple_tag}: 残差流 {before} -> {tuple(s.shape)}  | {cinfo}")
out = mixer(s)
logits = head(out)
print(f"  末端 mixer: {tuple(s.shape)} -> {tuple(out.shape)}   lm_head -> {tuple(logits.shape)}")
print(f"  前向输出无 NaN/Inf: {bool(torch.isfinite(logits).all())}")

print()
print("=" * 72)
print("实测 5E：逐 token decode 与整段 prefill 的一致性（缩小模型）")
print("=" * 72)
caches2 = [None] * len(sLayers)
s2 = emb(ids[:, :1]).repeat(1, 1, sHC)
outs = []
for t in range(ids.shape[1]):
    if t > 0:
        s2 = emb(ids[:, t:t+1]).repeat(1, 1, sHC)
    pf = ple_feat[:, t:t+1]
    for i, lyr in enumerate(layers):
        s2, caches2[i] = lyr(s2, caches2[i], pf)
    outs.append(head(mixer(s2)))
step = torch.cat(outs, dim=1)
print(f"  整段 prefill 输出 {tuple(logits.shape)}，逐 token 输出 {tuple(step.shape)}")
d_last = (logits[:, -1] - step[:, -1]).abs().max().item()
d_all = (logits - step).abs().max().item()
print(f"  最后一个位置的最大绝对差 = {d_last:.3e}")
print(f"  全部位置的最大绝对差     = {d_all:.3e}")
print(f"  判定（< 1e-4）: {'一致' if d_all < 1e-4 else '不一致'}")
print(f"  关键点：GDN 的 depthwise causal conv 必须跨 decode 步保留 kernel-1=3 个位置的通道状态")
print(f"          （官方 cache_params.update_conv_state，源码 L477/L486）；")
print(f"          若每步独立卷积则前 3 个历史位置被当作 0，误差达 1e-1 量级（见 probe6）")
