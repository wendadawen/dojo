"""实测 4：PLE / N-gram Embedding 的哈希寻址与门控注入。

验证目标：
  1. 16 个 n-gram head 各自使用一个不同的质数词表大小（20,000,000 之后的第 1..16 个质数）
  2. 质数之和 + 128 对齐 = 真实权重的行数总和 320,001,536
  3. n-gram id 由 splitmix64 派生的乘子做异或混合后取模得到，是哈希而非精确查表
  4. bigram 用 2 个 shift、trigram 用 3 个 shift，各占 8 个 head
  5. 哈希冲突率的实测量级
  6. PLE 门控注入：key/query 内积 -> 符号保持开方 -> sigmoid 门 -> 空洞深度卷积残差
  7. 每 token 实际只查 16 个 160 维向量，即 2560 个元素，而表本身有 51.2B 参数

对应源码：transformers@36deb0b5
  modeling_qwen4_exp.py
    L979  _splitmix64
    L986  _build_layer_multipliers
    L1009 _find_nth_prime_after
    L1039 size = _find_nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1)
    L1050 padded_vocab_size = ceil(total/divisor)*divisor
    L1098-1110 异或混合 + 取模 + 加偏移
    L1180 gate = (key_normed * query_normed).sum(-1) / sqrt(hidden_size)
    L1181 gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    L1188 output = gated_value + short_conv(gated_value_normed)
config：Qwen/Qwen3.8-Flash-Next@f5d08274
"""
import json, math
from math import prod
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
C = json.load(open("/tmp/qwen38fn/config.json"))["text_config"]
H = json.load(open("/tmp/qwen38fn/headers.json"))

D = C["hidden_size"]
HC = C["hc_count"]
NGRAM = C["ngram_size"]
HPN = C["heads_per_ngram"]
NG_HEADS = (NGRAM - 1) * HPN
PLE_D = C["ple_embed_dim"]
BASE = C["ngram_vocab_size_base"]
DIVISOR = C["make_ngram_vocab_size_divisible_by"]
VOCAB = C["vocab_size"]
SEED = 1234  # config 未显式给出，用 configuration 默认值
EPS = C["rms_norm_eps"]

_MASK64 = (1 << 64) - 1
_G = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def splitmix64(v):
    v = (v + _G) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    i = 3
    while i * i <= v:
        if v % i == 0:
            return False
        i += 2
    return True


def find_nth_prime_after(start, count):
    p, found = start, 0
    while found < count:
        p += 1
        if is_prime(p):
            found += 1
    return p


print("=" * 72)
print("实测 4A：16 个 n-gram head 的质数词表大小与真实权重行数核对")
print("=" * 72)
sizes, offsets, tot = [], [], 0
for h in range(NG_HEADS):
    s = find_nth_prime_after(BASE - 1, h + 1)
    sizes.append(s)
    offsets.append(tot)
    tot += s
print(f"  ngram_size={NGRAM} -> n-gram 阶数 {list(range(2, NGRAM+1))}，heads_per_ngram={HPN}")
print(f"  head 总数 = ({NGRAM}-1)*{HPN} = {NG_HEADS}")
print(f"  ngram_vocab_size_base = {BASE:,}")
print(f"  各 head 词表大小（{BASE:,} 之后的连续质数）:")
for i in range(0, NG_HEADS, 4):
    print(f"    head {i:2d}-{i+3:2d}: {', '.join(f'{s:,}' for s in sizes[i:i+4])}")
print(f"  质数之和 total_vocab_size = {tot:,}")
padded = math.ceil(tot / DIVISOR) * DIVISOR
print(f"  按 {DIVISOR} 对齐 -> padded_vocab_size = {padded:,}  (补 {padded-tot} 行)")

real_rows = sum(H[k]["shape"][0] for k in H if "ngram_embedding" in k)
n_shards = sum(1 for k in H if "ngram_embedding" in k)
print()
print(f"  真实权重 {n_shards} 个分片行数之和 = {real_rows:,}")
print(f"  理论 padded_vocab_size        = {padded:,}")
print(f"  一致: {real_rows == padded}")
head_dim_ng = PLE_D // NG_HEADS
print(f"  每 head 向量维度 = ple_embed_dim/head 数 = {PLE_D}/{NG_HEADS} = {head_dim_ng}  (真实 shape 列维 = {H[[k for k in H if 'shard_0' in k][0]]['shape'][1]})")
print(f"  表参数量 = {padded:,} x {head_dim_ng} = {padded*head_dim_ng:,} = {padded*head_dim_ng/1e9:.2f} B")
real_tot = sum(prod(H[k]["shape"]) for k in H if "ngram_embedding" in k)
print(f"  真实表参数量 = {real_tot:,} = {real_tot/1e9:.2f} B   一致: {real_tot == padded*head_dim_ng}")

print()
print("=" * 72)
print("实测 4B：splitmix64 派生的层乘子")
print("=" * 72)
ple_layer_index = 0
mults = []
for pos in range(NGRAM):
    raw = splitmix64(SEED + ple_layer_index * _PRIME_1 + pos)
    m = raw % VOCAB
    if m == 0:
        m = 1
    mults.append(m)
print(f"  seed={SEED}, ple_layer_index={ple_layer_index}, vocab_size={VOCAB:,}")
print(f"  层乘子（{NGRAM} 个，对应 shift 0..{NGRAM-1}）= {mults}")
real_mult = H["model.language_model.layers.1.ple.ple_embedding.layer_multipliers"]
print(f"  真实权重 layer_multipliers shape = {real_mult['shape']}, dtype = {real_mult['dtype']}")
print(f"  形状一致（{NGRAM} 个乘子）: {real_mult['shape'] == [NGRAM]}")
print(f"  注：乘子具体数值是 buffer，需下载权重才能核对，此处只核对推导公式与形状")

print()
print("=" * 72)
print("实测 4C：n-gram id 的哈希混合与冲突率")
print("=" * 72)
T = 4000
ids = torch.randint(0, VOCAB, (1, T))
eos = C["eos_token_id"]


def shift_right(tok, shift):
    """简化版：不处理 eos 分段（源码 L1053 会在 eos 处重置），只验证哈希混合本身。"""
    if shift == 0:
        return tok
    return F.pad(tok[:, :-shift], (shift, 0), value=eos)


shifted = [shift_right(ids, s) for s in range(NGRAM)]
blocks = []
for ngram in range(2, NGRAM + 1):
    st = (ngram - 2) * HPN
    en = st + HPN
    mixed = shifted[0] * mults[0]
    for pos in range(1, ngram):
        mixed = torch.bitwise_xor(mixed, shifted[pos] * mults[pos])
    hs = torch.tensor(sizes[st:en])
    ho = torch.tensor(offsets[st:en])
    ng_ids = torch.remainder(mixed.unsqueeze(-1), hs.view(1, 1, -1))
    blocks.append(ng_ids + ho.view(1, 1, -1))
    print(f"  {ngram}-gram: 用 shift 0..{ngram-1} 共 {ngram} 个位置异或混合 -> head {st}..{en-1}")
    print(f"      混合后 id 范围 [{mixed.min().item():,}, {mixed.max().item():,}]")
    print(f"      取模后落入各 head 词表，加偏移后全局 id 范围 [{blocks[-1].min().item():,}, {blocks[-1].max().item():,}]")

all_ids = torch.cat(blocks, dim=-1)
print()
print(f"  拼接后每 token 的 n-gram id 张量 shape = {tuple(all_ids.shape)}  (每 token {all_ids.shape[-1]} 个 head)")
print(f"  全局 id 最大值 = {all_ids.max().item():,} < padded_vocab_size {padded:,}: {all_ids.max().item() < padded}")

# 冲突率：同一 head 内，不同 n-gram 上下文映射到同一行的比例
for hidx in (0, 8):
    col = all_ids[0, :, hidx]
    uniq = torch.unique(col).numel()
    ngram_order = 2 if hidx < HPN else 3
    print(f"  head {hidx}（{ngram_order}-gram）: {T} 个位置 -> {uniq} 个不同行，冲突率 = {(T-uniq)/T*100:.3f}%")
print(f"  -> 表容量（每 head 约 2000 万行）远大于实际 n-gram 数时，冲突率很低")

print()
print("=" * 72)
print("实测 4D：PLE 门控注入与空洞卷积")
print("=" * 72)


class RMSNormGrouped(nn.Module):
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


KS = C["ple_conv_kernel_size"]
DIL = NGRAM
state_len = (KS - 1) * DIL
hc_hidden = HC * D
key_proj = nn.Linear(PLE_D, hc_hidden, bias=False)
value_proj = nn.Linear(PLE_D, D, bias=False)
norm_key = RMSNormGrouped(hc_hidden, group_size=D)
norm_query = RMSNormGrouped(hc_hidden, group_size=D)
norm_conv = RMSNormGrouped(hc_hidden, group_size=D)
conv1d = nn.Conv1d(hc_hidden, hc_hidden, kernel_size=KS, groups=hc_hidden, dilation=DIL, bias=False)

Tp = 20
embeddings = torch.randn(1, Tp, PLE_D)          # 16 个 head 各 160 维拼接
streams = torch.randn(1, Tp, hc_hidden)
key_normed = norm_key(key_proj(embeddings)).unflatten(-1, (HC, D))
value = value_proj(embeddings)
query_normed = norm_query(streams).unflatten(-1, (HC, D))
gate_raw = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(D)
gate = gate_raw.abs().clamp_min(1e-6).sqrt() * gate_raw.sign()
gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
print(f"  n-gram 拼接 embedding {tuple(embeddings.shape)}  (= {NG_HEADS} head x {head_dim_ng} 维)")
print(f"  key_proj -> {tuple(key_proj(embeddings).shape)}（每条流一个 key）, value_proj -> {tuple(value.shape)}（四流共享）")
print(f"  gate 原始（内积/sqrt({D})）范围 [{gate_raw.min():.4f}, {gate_raw.max():.4f}]")
print(f"  符号保持开方后范围 [{gate.min():.4f}, {gate.max():.4f}]")
print(f"  -> 开方压缩大值、放大小值，且 sign 保证不改变正负: {bool(torch.equal(gate.sign(), gate_raw.sign()))}")
print(f"  sigmoid(gate) 范围 [{torch.sigmoid(gate).min():.4f}, {torch.sigmoid(gate).max():.4f}]")
print(f"  门控后 gated_value {tuple(gated_value.shape)} -> 展平 {tuple(gated_value.flatten(-2).shape)}")

gv = gated_value.flatten(-2)
gvn = norm_conv(gv)
x = gvn.transpose(1, 2)
x = F.pad(x, (state_len, 0))
conv_out = F.silu(conv1d(x)).transpose(1, 2)
output = gv + conv_out
print()
print(f"  空洞深度卷积：kernel={KS}, dilation=ngram_size={DIL}, groups={hc_hidden}（逐通道）")
print(f"  卷积状态长度 = (kernel-1)*dilation = ({KS}-1)*{DIL} = {state_len}")
print(f"  左 pad {state_len} -> 卷积输出 {tuple(conv_out.shape)}，与输入等长: {conv_out.shape[1] == Tp}")
print(f"  感受野覆盖相对位置 = {[-i*DIL for i in range(KS-1, -1, -1)]}  （步长 {DIL}，跨 {state_len} 个 token）")
print(f"  PLE 最终输出 {tuple(output.shape)}，直接以残差形式加到四流上（源码 L1218）")

print()
print("=" * 72)
print("实测 4E：每 token 的实际访存量 vs 表总参数量")
print("=" * 72)
per_tok_elems = NG_HEADS * head_dim_ng
print(f"  每 token 查 {NG_HEADS} 个 head，各取 1 行 {head_dim_ng} 维 = {per_tok_elems:,} 个元素 = {per_tok_elems} = ple_embed_dim({PLE_D}): {per_tok_elems == PLE_D}")
print(f"  bf16 下每 token 从表中读取 {per_tok_elems*2/1024:.2f} KiB")
print(f"  表总参数 {real_tot:,} = {real_tot/1e9:.2f} B，bf16 占 {real_tot*2/2**30:.2f} GiB")
print(f"  单 token 实际触达比例 = {per_tok_elems/real_tot*100:.3e} %")
print(f"  -> 表是纯查表结构，不参与任何矩阵乘；config 的 base_model_tp_plan 把它标为 colwise_gather_output 并注释其体积约 45B/90GiB")
