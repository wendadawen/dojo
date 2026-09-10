#!/usr/bin/env python3
"""核对 CSA2 三模式在真实层分配下的运行行为.

层分配来自 official/inference/config.json (非缩放): compress_ratios / kv_source_layers /
index_source_layers / candidate_source_layer. 维度等比缩小以便本机运行, 结构字段保持真实.

核查项:
  1. 每个 ratio>0 的层都归入 Full / Reindex / Reuse 之一, 且判定条件与代码属性一致
  2. Full 层 = kv_source ∩ index_source (自己算主 KV + 自己跑 indexer)
  3. Reindex 层 = index_source ∖ kv_source (复用主 KV, 自己跑 indexer)
  4. Reuse 层 = 其余 ratio>0 层 (复用主 KV + 复用 topk)
  5. 运行期: 同组层读到的 compress_kv 是同一个张量对象 (跨层共享真的发生)
  6. 候选池: 只有 candidate_source_layer=20 建池, 其后的 index_source 层在池内搜索
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from mini_model import init_params, load_model_module  # noqa: E402

M = load_model_module()
torch.set_default_dtype(torch.bfloat16)
cfg = json.loads((HERE / "official" / "inference" / "config.json").read_text())

N = cfg["n_layers"]
ratios = tuple(cfg["compress_ratios"][:N])
kv_src = set(cfg["kv_source_layers"])
ix_src = set(cfg["index_source_layers"])
cand = cfg["candidate_source_layer"]

args = M.ModelArgs(
    max_batch_size=1, max_seq_len=256, dtype="bf16", expert_dtype=None, vocab_size=129280,
    dim=64, moe_inter_dim=48, n_layers=N, n_mtp_layers=0,
    n_heads=4, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
    score_func="sqrtsoftplus", norm_topk_prob=True, route_scale=1.5, swiglu_limit=10.0,
    q_lora_rank=32, head_dim=32, rope_head_dim=8, o_groups=2, o_lora_rank=16,
    window_size=8, compress_ratios=ratios, kv_source_layers=tuple(cfg["kv_source_layers"]),
    index_source_layers=tuple(cfg["index_source_layers"]),
    compress_rope_theta=160000.0, original_seq_len=32, rope_theta=10000.0, rope_factor=4.0,
    beta_fast=8, beta_slow=2, index_n_heads=4, index_head_dim=64, index_topk=512,
    candidate_source_layer=cand, candidate_topk_blocks=cfg["candidate_topk_blocks"],
    candidate_block_size=cfg["candidate_block_size"],
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, engram_layer_ids=(), vision_n_layers=0,
    dspark_block_size=0,
)
torch.manual_seed(0)
model = M.Transformer(args, None)
init_params(model, seed=0)
model.eval()

# 运行期记录: 每个压缩层实际读到的主 KV cache 属于哪个 kv_source 层
read_log = []
_orig_ckv = M.Attention._compress_kv
cache_owner = {
    id(l.attn.compress_kv_cache): i for i, l in enumerate(model.layers) if l.attn.is_kv_source
}


def _ckv(self, x, qr, start_pos, offset):
    out = _orig_ckv(self, x, qr, start_pos, offset)
    # 源层在返回前已发布自己的 cache, 故此刻的全局值就是它读到的那个
    used = M.shared_attn.compress_kv
    read_log.append(("compress_kv", self.layer_id, cache_owner.get(id(used)), self.is_kv_source))
    return out


M.Attention._compress_kv = _ckv

T = 64
ids = torch.randint(0, 129280, (1, T), generator=torch.Generator().manual_seed(7))
model(ids, 0)

print("层表 (ratio / 组件 / 模式 / 共享):")
print(" layer ratio compr index owns_k cand_src use_cand  模式   compress_kv 来源")
mode_count = {}
for i, layer in enumerate(model.layers):
    a = layer.attn
    r = a.compress_ratio
    has_c = a.compressor is not None
    has_i = a.indexer is not None
    owns = a.indexer.owns_k if has_i else None
    csrc = a.indexer.is_candidate_source if has_i else None
    ucan = a.indexer.uses_candidates if has_i else None
    if r == 0:
        mode = "SWA"
    elif a.is_kv_source and a.is_index_source:
        mode = "Full"
    elif a.is_index_source:
        mode = "Reindex"
    else:
        mode = "Reuse"
    mode_count[mode] = mode_count.get(mode, 0) + 1
    src = next((o for k, l, o, _ in read_log if l == i and k == "compress_kv"), None)
    print(f" {i:5d} {r:5d} {str(has_c):>5} {str(has_i):>5} {str(owns):>6} {str(csrc):>8} {str(ucan):>8}  "
          f"{mode:>7}   读到的 cache 属于层 {src}")

print("\n模式计数:", mode_count)
full = [i for i, l in enumerate(model.layers) if l.attn.is_kv_source and l.attn.is_index_source]
reindex = [i for i, l in enumerate(model.layers) if l.attn.is_index_source and not l.attn.is_kv_source]
reuse = [i for i, l in enumerate(model.layers)
         if l.attn.compress_ratio > 0 and not l.attn.is_index_source]
print(f"Full    = {full}   (config: kv_source ∩ index_source = {sorted(kv_src & ix_src)})")
print(f"Reindex = {reindex} (config: index_source ∖ kv_source = {sorted(ix_src - kv_src)})")
print(f"Reuse   = {reuse}")
print(f"SWA     = {[i for i, l in enumerate(model.layers) if l.attn.compress_ratio == 0]}")
assert full == sorted(kv_src & ix_src) and reindex == sorted(ix_src - kv_src)

print("\n[5] 跨层共享 (运行期 cache 归属):")
sources = sorted(kv_src)
expect_owner = {}
for s in sources:
    nxt = sources[sources.index(s) + 1] if s != sources[-1] else N
    for l in range(s, nxt):
        expect_owner[l] = s
bad = [(l, o, expect_owner.get(l)) for k, l, o, _ in read_log if k == "compress_kv" and o != expect_owner.get(l)]
print(f"    每个压缩层读到的 cache 都来自其组内 source 层: {not bad}  (不符: {bad})")
print(f"    源层读到自己刚发布的 cache: "
      f"{all(o == l for k, l, o, is_src in read_log if k == 'compress_kv' and is_src)}")
for s in sources:
    nxt = sources[sources.index(s) + 1] if s != sources[-1] else N
    print(f"    层 {s} 的 cache 被层 {s}-{nxt - 1} 共享 ({nxt - s} 层)")

print("\n[6] 候选池: 建池层 =", cand, "; 池内搜索的层 =",
      [i for i, l in enumerate(model.layers) if l.attn.indexer is not None and l.attn.indexer.uses_candidates])
print("    候选池形状 (运行后):", tuple(M.shared_attn.candidates.shape) if M.shared_attn.candidates is not None else None)
print(f"    candidate_topk_blocks={cfg['candidate_topk_blocks']} × block_size={cfg['candidate_block_size']} "
      f"= {cfg['candidate_topk_blocks'] * cfg['candidate_block_size']} 个候选位置 (报告: 16384)")
