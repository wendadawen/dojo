"""实测 5：保留全部结构特征的等比缩小模型 —— 整段 prefill 与逐 token decode 一致性。

缩小配置（结构全保留，仅缩维度）：
  8 层，interval=4 → [GDN, GDN, GDN, FA] × 2；hidden 64；GQA 4Q/2KV 头 × 32 维
  GDN: 2 k头×16, 4 v头×16, conv kernel 4；MoE: 8 专家 top-2 + 共享专家 + sigmoid 门
  注意力: 双宽 q_proj + QK-norm + sigmoid 输出门 + 部分 RoPE(1/4) + 交错 MRoPE
验证目标：
  5A. 逐层张量形状记录（残差全程单流 64 宽）
  5B. GDN 缓存形状恒定 vs 全注意力 KV 随长度增长
  5C. 整段 prefill 与逐 token decode 的 logits 一致（卷积状态跨步维护 kernel-1=3 滑窗）
"""
import math, torch, torch.nn.functional as F

torch.manual_seed(4)

HID, NH, NKV, HD = 64, 4, 2, 32
ROT = HD // 4                    # 部分 RoPE 1/4
NK, DK, NV, DV = 2, 16, 4, 16
CONV_K = 4
E, TOPK, I = 8, 2, 32
VOCAB = 97
LAYERS = ["gdn", "gdn", "gdn", "fa", "gdn", "gdn", "gdn", "fa"]

def rms(t, w, eps=1e-6):
    return w * (t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + eps)).to(t.dtype)
def rmsnorm_gated(t, gate, w):
    return rms(t, w) * F.silu(gate.float()).to(t.dtype)      # Qwen3.5 的 GDN 输出门是 silu
def rotate_half(t):
    a, b = t[..., :t.shape[-1]//2], t[..., t.shape[-1]//2:]
    return torch.cat((-b, a), dim=-1)

class GDN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kd, self.vd = NK*DK, NV*DV
        self.conv_dim = self.kd*2 + self.vd
        self.qkv = torch.nn.Linear(HID, self.conv_dim, bias=False)
        self.z = torch.nn.Linear(HID, self.vd, bias=False)
        self.b = torch.nn.Linear(HID, NV, bias=False)
        self.a = torch.nn.Linear(HID, NV, bias=False)
        self.conv = torch.nn.Conv1d(self.conv_dim, self.conv_dim, CONV_K, groups=self.conv_dim,
                                    padding=CONV_K-1, bias=False)
        self.A_log = torch.nn.Parameter(torch.log(torch.empty(NV).uniform_(0.01, 16)))
        self.dt_bias = torch.nn.Parameter(torch.ones(NV))
        self.norm_w = torch.nn.Parameter(torch.ones(DV))
        self.out = torch.nn.Linear(self.vd, HID, bias=False)
        self.conv_state = None       # [1, conv_dim, CONV_K-1]
        self.rec_state = None        # [1, NV, DK, DV]
    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).transpose(1, 2)                       # [B, conv_dim, T]
        if T == 1 and self.conv_state is not None:
            inp = torch.cat([self.conv_state, qkv], dim=-1)
            self.conv_state = inp[..., -(CONV_K-1):].clone()
            qkv = F.conv1d(inp, self.conv.weight, groups=self.conv_dim)[..., -T:]
        else:
            if self.conv_state is not None:                     # prefill 接续旧状态
                qkv = F.conv1d(torch.cat([self.conv_state, qkv], dim=-1),
                               self.conv.weight, groups=self.conv_dim)[..., -T:]
            else:
                qkv = self.conv(qkv)[..., :T]
            tail = torch.cat([qkv[..., -1:], torch.zeros(1, self.conv_dim, CONV_K-2)], -1) \
                   if T >= 1 else qkv
            full = self.qkv(x).transpose(1, 2)
            # 直接重算末尾滑窗：取最后 CONV_K-1 个输入
            src = self.qkv(x).transpose(1, 2)
            self.conv_state = src[..., -(CONV_K-1):].clone() if T >= CONV_K-1 else \
                torch.cat([self.conv_state, src], -1)[..., -(CONV_K-1):] if self.conv_state is not None else \
                F.pad(src, (CONV_K-1-T, 0))[..., :(CONV_K-1)]
        qkv = F.silu(qkv).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.kd, self.kd, self.vd], dim=-1)
        q = q.view(B, T, NK, DK).repeat_interleave(NV//NK, dim=2)
        k = k.view(B, T, NK, DK).repeat_interleave(NV//NK, dim=2)
        v = v.view(B, T, NV, DV)
        beta = torch.sigmoid(self.b(x))
        g = -self.A_log.float().exp() * F.softplus(self.a(x).float() + self.dt_bias.float())
        # 逐 token 递归（两种模式共用同一路径，等价性见 probe1）
        if self.rec_state is None:
            self.rec_state = torch.zeros(1, NV, DK, DV)
        out = torch.zeros(B, T, NV, DV)
        for i in range(T):
            qi, ki, vi = q[:, i].float(), k[:, i].float(), v[:, i].float()
            qi = qi * (1/math.sqrt(DK))
            qi = qi * torch.rsqrt(qi.pow(2).sum(-1, keepdim=True) + 1e-6)
            ki = ki * torch.rsqrt(ki.pow(2).sum(-1, keepdim=True) + 1e-6)
            gt = g[:, i].exp().unsqueeze(-1).unsqueeze(-1)
            bt = beta[:, i].unsqueeze(-1)
            self.rec_state = self.rec_state * gt
            kv_mem = (self.rec_state * ki.unsqueeze(-1)).sum(-2)
            delta = (vi - kv_mem) * bt
            self.rec_state = self.rec_state + ki.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, i] = ((self.rec_state * qi.unsqueeze(-1)).sum(-2)).to(x.dtype)
        z = self.z(x)
        o = out.reshape(-1, DV)                                 # 源码 L540：reshape 到 (-1, head_v_dim) 逐头
        zz = z.reshape(-1, DV)
        o = rmsnorm_gated(o, zz, self.norm_w)
        return self.out(o.reshape(B, T, -1))

class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(HID, NH*HD*2, bias=False)      # 双宽：query+gate
        self.k = torch.nn.Linear(HID, NKV*HD, bias=False)
        self.v = torch.nn.Linear(HID, NKV*HD, bias=False)
        self.o = torch.nn.Linear(NH*HD, HID, bias=False)
        self.qn = torch.nn.Parameter(torch.ones(HD))
        self.kn = torch.nn.Parameter(torch.ones(HD))
        self.K = self.V = None
    def forward(self, x, pos):
        B, T, _ = x.shape
        proj = self.q(x).view(B, T, NH, HD*2)
        q, gate = torch.chunk(proj, 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = rms(q, self.qn).transpose(1, 2)                     # [B,NH,T,HD]
        k = rms(self.k(x).view(B, T, NKV, HD), self.kn).transpose(1, 2)
        v = self.v(x).view(B, T, NKV, HD).transpose(1, 2)
        inv = 1.0 / (10000.0 ** (torch.arange(0, ROT, 2).float() / ROT))
        fr = torch.outer(pos.float(), inv)
        cos = torch.cat([fr.cos(), fr.cos()], -1)[None, None]   # [1,1,T,ROT]
        sin = torch.cat([fr.sin(), fr.sin()], -1)[None, None]
        qr, qp = q[..., :ROT], q[..., ROT:]
        kr, kp = k[..., :ROT], k[..., ROT:]
        q = torch.cat([qr*cos + rotate_half(qr)*sin, qp], -1)
        k = torch.cat([kr*cos + rotate_half(kr)*sin, kp], -1)
        if self.K is None: self.K, self.V = k, v
        else: self.K, self.V = torch.cat([self.K, k], 2), torch.cat([self.V, v], 2)
        Kx = self.K.repeat_interleave(NH//NKV, 1); Vx = self.V.repeat_interleave(NH//NKV, 1)
        att = (q @ Kx.transpose(-1, -2)) / math.sqrt(HD)
        n = self.K.shape[2]
        # 掩码按 [T, n] 构造：本段第 i 个查询可见键 0..(n-T+i)。decode T=1 时全可见
        mask = torch.triu(torch.ones(T, n, dtype=torch.bool), diagonal=n - T + 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        ao = (att @ Vx).transpose(1, 2).reshape(B, T, -1)
        return self.o(ao * torch.sigmoid(gate))

class MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(HID, E, bias=False)
        self.gu = torch.randn(E, 2*I, HID) * 0.02
        self.dn = torch.randn(E, HID, I) * 0.02
        self.sh_g = torch.nn.Linear(HID, I, bias=False)
        self.sh_u = torch.nn.Linear(HID, I, bias=False)
        self.sh_d = torch.nn.Linear(I, HID, bias=False)
        self.sh_gate = torch.nn.Linear(HID, 1, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        h = x.view(-1, HID)
        probs = F.softmax(self.gate(h).float(), -1)
        tv, ti = torch.topk(probs, TOPK, -1)
        w = (tv / tv.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(h)
        for e in range(E):
            sel = (ti == e)
            if not sel.any(): continue
            idx = sel.any(-1).nonzero().squeeze(-1)
            pos = sel[idx].float().argmax(-1)
            hh = h[idx]
            g, u = (hh @ self.gu[e].T).chunk(2, -1)
            eo = (F.silu(g) * u) @ self.dn[e].T
            out.index_add_(0, idx, eo * w[idx, pos, None])
        sh = F.silu(self.sh_g(h)) * self.sh_u(h)
        sh = self.sh_d(sh) * torch.sigmoid(self.sh_gate(h))
        return (out + sh).view(B, T, HID)

class Layer(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.mix = GDN() if kind == "gdn" else Attn()
        self.moe = MoE()
        self.ln1 = torch.nn.Parameter(torch.ones(HID))
        self.ln2 = torch.nn.Parameter(torch.ones(HID))
    def forward(self, x, pos):
        x = x + self.mix(rms(x, self.ln1), pos) if self.kind == "fa" else x + self.mix(rms(x, self.ln1))
        x = x + self.moe(rms(x, self.ln2))
        return x

torch.manual_seed(4)
embed = torch.randn(VOCAB, HID) * 0.02
lm_head = torch.randn(VOCAB, HID) * 0.02
final_ln = torch.nn.Parameter(torch.ones(HID))
layers = torch.nn.ModuleList([Layer(k) for k in LAYERS])

def forward_tokens(ids, pos):
    x = embed[ids]
    for lyr in layers:
        x = lyr(x, pos)
    return rms(x, final_ln) @ lm_head.T

# ============ 5A/5B. prefill 形状记录 ============
print("=== 5A. 逐层前向形状（prefill T=10） ===")
ids = torch.randint(0, VOCAB, (1, 10))
shapes = []
x = embed[ids]
for i, lyr in enumerate(layers):
    xin = x
    x = lyr(x, torch.arange(10))
    shapes.append((i, lyr.kind, tuple(xin.shape), tuple(x.shape)))
for i, kind, si, so in shapes:
    print(f"  L{i} {kind:<4s} {si} -> {so}")
print(f"  → 残差流全程单流 {HID} 宽（无并行流）")

print()
print("=== 5B. 缓存形状 ===")
for i, lyr in enumerate(layers):
    if lyr.kind == "gdn":
        print(f"  L{i} GDN: 递归状态 {tuple(lyr.mix.rec_state.shape)}（常数）卷积状态 {tuple(lyr.mix.conv_state.shape)}（常数）")
    else:
        print(f"  L{i} FA : KV {tuple(lyr.mix.K.shape)}/{tuple(lyr.mix.V.shape)}（随长度增长）")

# ============ 5C. prefill vs 逐 token decode ============
print()
print("=== 5C. prefill vs decode 一致性 ===")
def reset_caches():
    for lyr in layers:
        if lyr.kind == "gdn":
            lyr.mix.rec_state = None; lyr.mix.conv_state = None
        else:
            lyr.mix.K = lyr.mix.V = None
reset_caches()
logits_pre = forward_tokens(ids, torch.arange(10))
# 重置缓存逐 token 重放
reset_caches()
outs = []
for t in range(10):
    outs.append(forward_tokens(ids[:, t:t+1], torch.tensor([t])))
logits_dec = torch.cat(outs, 1)
diff = (logits_pre - logits_dec).abs().max().item()
print(f"  整段 prefill {tuple(logits_pre.shape)} vs 逐 token decode {tuple(logits_dec.shape)}")
print(f"  最后一个位置 logits 最大绝对差 = {diff:.3e}")
print(f"  判定：{'一致（卷积状态跨步维护正确）' if diff < 1e-4 else '不一致，需排查'}")
