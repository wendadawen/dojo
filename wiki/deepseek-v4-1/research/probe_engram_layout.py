#!/usr/bin/env python3
"""Engram 布局实测: 用真实 tokenizer + 官方 engram.py 原样代码核对三件事。

验证目标 (对应 official/inference/engram.py 与 config.json):
  1. build_compressed_token_map 在真实 tokenizer 上得到的压缩词表大小
     == config.engram_compressed_vocab_size (99092)  [NgramHashState.__init__ 的 assert]
  2. EngramLayout 的 24 个素数桶之和 == config.engram_num_embeddings
     (384006168 / 384016682), 且所有素数 >= engram_vocab_size 起点、两两不同
  3. compute_hash_multipliers 的派生规则: 每层 RNG 种子 10007*layer_id,
     全部乘子为奇数且 < (int64max // compressed_vocab) // 2
  4. NgramHashState.forward 的哈希行为: n-gram 被 DEAD(图像) 阻断时用 pad 填充
运行: /usr/bin/python3 probe_engram_layout.py   (需 transformers/tokenizers/sympy)
输出: ckpt/probe_engram_layout.out
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "official" / "inference"))
sys.path.insert(0, str(Path(__file__).parent / "official"))

from transformers import AutoTokenizer  # noqa: E402

# 官方 engram.py 原样 exec; 只前置 future import 让 py3.9 不在运行时求值 `X | None` 注解
_src = (Path(__file__).parent / "official" / "inference" / "engram.py").read_text()
E = type(sys)("official_engram")
sys.modules["official_engram"] = E
exec(compile("from __future__ import annotations\n" + _src, "official/inference/engram.py", "exec"), E.__dict__)

CKPT = Path(__file__).parent / "official"

tokenizer = AutoTokenizer.from_pretrained(str(CKPT))
print("tokenizer vocab:", len(tokenizer))

# ---- 1. 压缩词表 ----
lookup, vocab_size = E.build_compressed_token_map(tokenizer)
print("compressed vocab size:", vocab_size, "(config 期望 99092:", vocab_size == 99092, ")")
print("原始词表大小:", len(lookup), "-> 压缩比:", f"{len(lookup)/vocab_size:.4f}")
# 抽样: ' The'/'the'/'THE' 应映射到同一压缩 id (源码 docstring 的断言)
ids = {t: tokenizer.encode(t, add_special_tokens=False) for t in [" The", "the", "THE"]}
print("抽样 token ids:", ids)
same = len({lookup[i[0]] for i in ids.values() if len(i) == 1}) == 1
print("' The'/'the'/'THE' 同压缩 id:", same)


# ---- 2. 素数表布局 ----
class A:  # 最小 ModelArgs, 只给 from_args 用的字段
    engram_layer_ids = (1, 14)
    engram_max_ngram_size = 4
    engram_n_heads = 8
    engram_vocab_size = 16000000
    engram_head_dim = 256
    engram_num_embeddings = (384006168, 384016682)


layout = E.EngramLayout.from_args(A)
print("\nlayers:", layout.layer_ids, "n-gram 阶数:", layout.max_ngram_size - 1, "每阶头数:", layout.n_heads)
all_primes = [p for layer in layout.primes for per_ngram in layer for p in per_ngram]
print("总桶数:", len(all_primes), "(期望 2 层 x 3 阶 x 8 头 = 48)")
print("全部素数互不相同:", len(set(all_primes)) == len(all_primes))
print("最小素数:", min(all_primes), ">= engram_vocab_size-1 =", A.engram_vocab_size - 1, ":", min(all_primes) > A.engram_vocab_size - 1)
for li, layer in enumerate(layout.primes):
    s = sum(p for per_ngram in layer for p in per_ngram)
    print(f"layer {layout.layer_ids[li]}: 24 素数之和 = {s} == engram_num_embeddings[{li}] = {A.engram_num_embeddings[li]}: {s == A.engram_num_embeddings[li]}")
    for o, per_ngram in enumerate(layer):
        print(f"  {o+2}-gram 桶: {list(per_ngram)}")

# ---- 3. 哈希乘子 ----
mults = E.compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
print("\nmultipliers shape:", tuple(mults.shape), "(期望 (2, 4): 每层每 lookback 一个)")
print("multipliers:", mults.tolist())
print("全部为奇数:", bool((mults % 2 == 1).all()))
# bound 是 RNG 采样上界; 乘子 = 2v+1 可达 2*bound-1. 设计约束是 token_id*multiplier 不溢出 int64
print("乘子上界核对: max =", int(mults.max()), "<= 2*bound-1 =", 2 * bound - 1, ":", bool((mults <= 2 * bound - 1).all()))
print("不溢出 int64: max*compressed_vocab =", int(mults.max()) * vocab_size, "< 2^63-1 :", int(mults.max()) * vocab_size < 2**63 - 1)
print("种子规则核对: layer 1 种子 10007, layer 14 种子 140098 -> 两层乘子不同:", not torch.equal(mults[0], mults[1]))

# ---- 4. 哈希行为 (NgramHashState) ----
args = type(
    "Args",
    (),
    dict(
        engram_layer_ids=(1, 14),
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_vocab_size=16000000,
        engram_head_dim=256,
        engram_num_embeddings=(384006168, 384016682),
        engram_compressed_vocab_size=vocab_size,
        engram_pad_id=2,
        max_batch_size=1,
        max_seq_len=64,
    ),
)()
state = E.NgramHashState(args, layout, tokenizer)
print("\npad_id (压缩后):", state.pad_id, "= token_map[2]:", lookup[2])
ids = torch.tensor([tokenizer.encode("deepseek v4 flash", add_special_tokens=False)])
h = state(ids, 0)
print("hash ids shape:", tuple(h.shape), "(期望 [1, L, 2 层, 24 桶])")
print("最大 hash id:", int(h.max()), "< 表大小:", int(h.max()) < max(A.engram_num_embeddings))
# 起始位置: 位置 0 的 2/3/4-gram 都因越界被 pad
print("位置 0 的三个桶 (层 1):", h[0, 0, 0, :8].tolist(), "...")
# DEAD 阻断: 中间插入图像 span
mask = torch.ones_like(ids, dtype=torch.bool)
mask[0, 1] = False  # 位置 1 是图像
h2 = state(ids, 0, mask)
print("位置 1 为 DEAD 后, 位置 2 的 hash 是否变化:", not torch.equal(h[0, 2], h2[0, 2]), "(n-gram 跨越 DEAD -> pad)")
