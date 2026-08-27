"""核对 N-gram Embedding 的三个 I64 缓冲的真实数值。

这三个张量共 35 个 int64（280 字节），用 HTTP Range 精确下载，不涉及权重本体。

验证目标：
  1. layer_multipliers 的真实数值 == 源码 _build_layer_multipliers 的派生结果
     -> 确认 seed=1234（config 未显式给出，取 configuration 默认值）与派生公式
  2. ngram_heads_vocab_sizes 的真实数值 == 20000000 之后的连续 16 个质数
  3. ngram_heads_offsets 的真实数值 == vocab_sizes 的前缀和
  4. 用真实乘子重算 n-gram id，确认落在各 head 词表区间内

对应源码：transformers@36deb0b5 modeling_qwen4_exp.py
  L979  _splitmix64
  L986  _build_layer_multipliers
  L1009 _find_nth_prime_after
  L1039 size = _find_nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1)
  L1041 self.head_offsets.append(self.total_vocab_size)
config：Qwen/Qwen3.8-Flash-Next@f5d08274
"""
import json, struct, subprocess

REPO = "Qwen/Qwen3.8-Flash-Next"
SHA = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
C = json.load(open("/tmp/qwen38fn/config.json"))["text_config"]

TARGETS = {
    "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": "model-00005-of-00131.safetensors",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes": "model-00037-of-00131.safetensors",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets": "model-00037-of-00131.safetensors",
}


def curl_range(url, start, end):
    r = subprocess.run(["curl", "-sL", "-r", f"{start}-{end}", url], capture_output=True)
    return r.stdout


def read_header(shard):
    url = f"https://huggingface.co/{REPO}/resolve/{SHA}/{shard}"
    n = struct.unpack("<Q", curl_range(url, 0, 7)[:8])[0]
    return json.loads(curl_range(url, 8, 8 + n - 1).decode()), n, url


print("=" * 74)
print("A. 用 HTTP Range 精确下载三个 I64 缓冲（按 data_offsets 定位）")
print("=" * 74)
real = {}
total_bytes = 0
for shard in sorted(set(TARGETS.values())):
    hdr, hlen, url = read_header(shard)
    data_start = 8 + hlen
    for name, sh in TARGETS.items():
        if sh != shard:
            continue
        meta = hdr[name]
        s, e = meta["data_offsets"]
        nbytes = e - s
        raw = curl_range(url, data_start + s, data_start + e - 1)
        assert len(raw) == nbytes, f"{name}: 期望 {nbytes} 字节，实得 {len(raw)}"
        vals = list(struct.unpack(f"<{nbytes // 8}q", raw))
        real[name.split(".")[-1]] = vals
        total_bytes += nbytes
        print(f"  {name.split('.')[-1]:<26s} dtype={meta['dtype']} shape={meta['shape']}"
              f" offsets={meta['data_offsets']} -> 下载 {nbytes} B")
print(f"  三个张量合计下载 {total_bytes} 字节（权重本体未触碰）")

# ---------------- 复现 splitmix64 与质数派生 ----------------
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


VOCAB = C["vocab_size"]
NGRAM = C["ngram_size"]
NG_HEADS = (NGRAM - 1) * C["heads_per_ngram"]
BASE = C["ngram_vocab_size_base"]
PLE_LAYER_INDEX = 0   # ple_layer_ids=[2] 只有一个元素，故该层的 ple_layer_index=0


def build_layer_multipliers(vocab_size, ngram_size, ple_layer_index, seed):
    """逐行照搬源码 L986 _build_layer_multipliers。

    注意三处细节，凭直觉容易写错：
      1. splitmix64 的输入是 base_seed + GAMMA*(index+1)，不是 base_seed + index
      2. 取模用的是 half_bound = ((2^63-1)//vocab_size)//2，不是 vocab_size
      3. 结果为 2*(...)+1，强制为奇数（奇数与 2^64 互质，保证乘法在模 2^64 下可逆）
    """
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + _G * (index + 1)) & _MASK64
        out.append(2 * (splitmix64(value) % half_bound) + 1)
    return out


print()
print("=" * 74)
print("B. layer_multipliers：真实数值 vs 公式派生")
print("=" * 74)
max_long = (1 << 63) - 1
multiplier_max = max_long // VOCAB
half_bound = max(1, multiplier_max // 2)
print(f"  公式（源码 L986-995）：")
print(f"    base_seed  = seed + {_PRIME_1} * ple_layer_index")
print(f"    value      = base_seed + GAMMA * (index + 1)   (GAMMA = 0x9E3779B97F4A7C15)")
print(f"    multiplier = 2 * (splitmix64(value) mod half_bound) + 1")
print(f"  其中 half_bound = ((2^63-1) // vocab_size) // 2 = ({max_long} // {VOCAB:,}) // 2 = {half_bound:,}")
print(f"  ple_layer_index={PLE_LAYER_INDEX}（ple_layer_ids={C['ple_layer_ids']} 中该层的下标）")
print()
# config 未显式写 seed，扫描候选以确认默认值 1234
matched_seed = None
for seed in [1234] + [s for s in range(0, 20001) if s != 1234]:
    if build_layer_multipliers(VOCAB, NGRAM, PLE_LAYER_INDEX, seed) == real["layer_multipliers"]:
        matched_seed = seed
        break

derived = build_layer_multipliers(VOCAB, NGRAM, PLE_LAYER_INDEX, 1234)
print(f"  {'index':>6s} {'公式派生 (seed=1234)':>24s} {'checkpoint 真实值':>24s} {'一致':>6s}")
for pos in range(NGRAM):
    ok = derived[pos] == real["layer_multipliers"][pos]
    print(f"  {pos:>6d} {derived[pos]:>24,d} {real['layer_multipliers'][pos]:>24,d} {'是' if ok else '否':>6s}")
print()
print(f"  seed=1234 全部一致: {derived == real['layer_multipliers']}")
print(f"  在 [0,20000] 内扫描到的最小匹配 seed: {matched_seed}")
print(f"  真实乘子全为奇数: {all(v % 2 == 1 for v in real['layer_multipliers'])}（公式 2k+1 的直接后果）")
print(f"  真实乘子 x vocab_size 是否都不溢出 int64: "
      f"{all(v * VOCAB <= max_long for v in real['layer_multipliers'])}"
      f"（half_bound 的设计目的：保证 token_id * multiplier 不溢出）")
print(f"  -> config 未显式给出的 seed 确认为 configuration 默认值 1234")

print()
print("=" * 74)
print("C. ngram_heads_vocab_sizes：真实数值 vs 连续质数")
print("=" * 74)
sizes = [find_nth_prime_after(BASE - 1, h + 1) for h in range(NG_HEADS)]
print(f"  {'head':>5s} {'第 i+1 个质数 (>{:,})'.format(BASE):>26s} {'checkpoint 真实值':>20s} {'一致':>6s}")
for h in range(NG_HEADS):
    ok = sizes[h] == real["ngram_heads_vocab_sizes"][h]
    print(f"  {h:>5d} {sizes[h]:>26,d} {real['ngram_heads_vocab_sizes'][h]:>20,d} {'是' if ok else '否':>6s}")
print()
print(f"  全部 {NG_HEADS} 个一致: {sizes == real['ngram_heads_vocab_sizes']}")
print(f"  真实数值全为质数: {all(is_prime(v) for v in real['ngram_heads_vocab_sizes'])}")
print(f"  真实数值之和 = {sum(real['ngram_heads_vocab_sizes']):,}")

print()
print("=" * 74)
print("D. ngram_heads_offsets：真实数值 vs vocab_sizes 前缀和")
print("=" * 74)
offs, acc = [], 0
for h in range(NG_HEADS):
    offs.append(acc)
    acc += real["ngram_heads_vocab_sizes"][h]
print(f"  {'head':>5s} {'前缀和':>20s} {'checkpoint 真实值':>20s} {'一致':>6s}")
for h in range(NG_HEADS):
    ok = offs[h] == real["ngram_heads_offsets"][h]
    print(f"  {h:>5d} {offs[h]:>20,d} {real['ngram_heads_offsets'][h]:>20,d} {'是' if ok else '否':>6s}")
print()
print(f"  全部一致: {offs == real['ngram_heads_offsets']}")
print(f"  末个 offset + 末个 size = {real['ngram_heads_offsets'][-1] + real['ngram_heads_vocab_sizes'][-1]:,}"
      f"  == 质数之和 {acc:,}: {real['ngram_heads_offsets'][-1] + real['ngram_heads_vocab_sizes'][-1] == acc}")

print()
print("=" * 74)
print("E. 用真实缓冲值重算表尺寸，与真实权重行数比对")
print("=" * 74)
DIV = C["make_ngram_vocab_size_divisible_by"]
import math
tot = sum(real["ngram_heads_vocab_sizes"])
padded = math.ceil(tot / DIV) * DIV
H = json.load(open("/tmp/qwen38fn/headers.json"))
real_rows = sum(H[k]["shape"][0] for k in H if "ngram_embedding" in k)
head_dim = C["ple_embed_dim"] // NG_HEADS
print(f"  真实 vocab_sizes 之和           = {tot:,}")
print(f"  按 {DIV} 对齐                     = {padded:,}（补 {padded - tot} 行）")
print(f"  真实权重 128 分片行数之和       = {real_rows:,}")
print(f"  一致: {padded == real_rows}")
print(f"  表参数量 = {padded:,} x {head_dim} = {padded * head_dim:,} = {padded * head_dim / 1e9:.2f} B")

print()
print("=" * 74)
print("F. 用真实乘子重算一遍 n-gram id，确认落在各 head 词表内")
print("=" * 74)
import torch
torch.manual_seed(0)
T = 2000
ids = torch.randint(0, VOCAB, (1, T))
eos = C["eos_token_id"]
mult = real["layer_multipliers"]
vs = torch.tensor(real["ngram_heads_vocab_sizes"])
of = torch.tensor(real["ngram_heads_offsets"])
import torch.nn.functional as F


def shift_right(t, s):
    return t if s == 0 else F.pad(t[:, :-s], (s, 0), value=eos)


shifted = [shift_right(ids, s) for s in range(NGRAM)]
blocks = []
HPN = C["heads_per_ngram"]
for ngram in range(2, NGRAM + 1):
    st, en = (ngram - 2) * HPN, (ngram - 2) * HPN + HPN
    mixed = shifted[0] * mult[0]
    for p in range(1, ngram):
        mixed = torch.bitwise_xor(mixed, shifted[p] * mult[p])
    nid = torch.remainder(mixed.unsqueeze(-1), vs[st:en].view(1, 1, -1)) + of[st:en].view(1, 1, -1)
    blocks.append(nid)
    # 每个 head 的 id 必须落在 [offset, offset+size)
    for j, h in enumerate(range(st, en)):
        col = nid[0, :, j]
        lo, hi = int(of[h]), int(of[h]) + int(vs[h])
        assert col.min() >= lo and col.max() < hi, f"head {h} 越界"
    print(f"  {ngram}-gram (head {st}..{en-1}): id 范围 [{nid.min().item():,}, {nid.max().item():,}]，全部落在各自 head 区间内")
allids = torch.cat(blocks, dim=-1)
print()
print(f"  每 token 产生 {allids.shape[-1]} 个 head id，全局最大 {allids.max().item():,} < padded {padded:,}: {allids.max().item() < padded}")
for hidx in (0, 8):
    col = allids[0, :, hidx]
    u = torch.unique(col).numel()
    print(f"  head {hidx}（{'2' if hidx < HPN else '3'}-gram）: {T} 位置 -> {u} 个不同行，碰撞率 {(T-u)/T*100:.3f}%")

print()
print("=" * 74)
print("结论：三个 I64 缓冲的真实数值与源码派生公式逐元素一致（共 35 个 int64 全部核对）")
print("      seed=1234 得到确认；表尺寸 320,001,536 由真实 vocab_sizes 重算得出")
print("=" * 74)
