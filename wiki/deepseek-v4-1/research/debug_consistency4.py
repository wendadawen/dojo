#!/usr/bin/env python3
"""决定性测量: layer 2 indexer 在 prefill 与 decode 两条路径下的 index_score 原值.

E0/E1/E2 三组对照显示 topk 翻转位置与幅度不随 bf16/fp32、量化开关变化, 且 index_k 缓存
逐位一致 -> 怀疑不是数值噪声, 而是 topk 候选张量宽度不同 (prefill: 全长 8 槽含 -inf 填充;
decode: 截到 end_pos//ratio = 5 槽) 导致的平局打破差异. 这里把两条路径每个位置的分数
逐元素打印出来核对.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mini_model import init_params, load_model_module, mini_args

M = load_model_module()
torch.set_default_dtype(torch.bfloat16)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(str(Path(__file__).parent / "official"))
T, SPLIT = 16, 8
input_ids = torch.randint(0, 129280, (1, T), generator=torch.Generator().manual_seed(42))

SCORE_LOG = []  # (id(indexer), start_pos, seqlen, compress_lens, scores, idxs)
_orig = M.Indexer.forward


def _probe(self, x, qr, latent, start_pos, offset):
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
    idxs = _orig(self, x, qr, latent, start_pos, offset)
    # 重算分数 (与官方 forward 同式, 只加记录)
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
    M.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
    M.fp4_act_quant(q, 32, True)
    index_k = M.shared_attn.index_k[:bsz, : end_pos // ratio]
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
    sc = torch.einsum("bshd,btd->bsht", q, index_k)
    sc = (sc.relu() * weights.unsqueeze(-1)).sum(dim=2)[0]  # [s, t]
    SCORE_LOG.append((id(self), start_pos, seqlen, end_pos // ratio, sc.detach().clone(), idxs.clone()))
    return idxs


M.Indexer.forward = _probe


def run(full):
    args = mini_args(M)
    m = M.Transformer(args, tokenizer)
    init_params(m, seed=0)
    m.eval()
    reg = {id(l.attn.indexer): i for i, l in enumerate(m.layers) if l.attn.indexer is not None}
    mark = len(SCORE_LOG)
    if full:
        m(input_ids, 0)
    else:
        m(input_ids[:, :SPLIT], 0)
        for p in range(SPLIT, T):
            m(input_ids[:, p : p + 1], p)
    log = {}
    for idx_id, start_pos, seqlen, clen, sc, idxs in SCORE_LOG[mark:]:
        layer = reg[idx_id]
        for j in range(seqlen):
            log.setdefault(layer, {})[start_pos + j] = (clen, sc[j], idxs[0, j])
    return log


A = run(True)
B = run(False)
lay = 2
print(f"=== layer {lay} indexer 分数对比 (prefill16 vs prefill8+decode8) ===")
for p in range(8, 16):
    clenA, scA, idxA = A[lay][p]
    clenB, scB, idxB = B[lay][p]
    print(f"\n-- query pos {p} --")
    print(f"   A prefill : compress_len={clenA} 槽数={scA.numel()}")
    print(f"   B decode  : compress_len={clenB} 槽数={scB.numel()}")
    print(f"   A scores: {[f'{v:+.6e}' for v in scA.tolist()]}")
    print(f"   B scores: {[f'{v:+.6e}' for v in scB.tolist()]}")
    print(f"   A idxs(归一化前): {idxA.tolist()}")
    print(f"   B idxs(归一化前): {idxB.tolist()}")
    if scA.numel() == scB.numel():
        print(f"   分数逐位最大差: {(scA - scB).abs().max().item():.3e}")
    else:
        print(f"   分数共同前缀最大差: {(scA[: scB.numel()] - scB).abs().max().item():.3e}")
    # 平局检查: 第 4 大与第 5 大的间隔
    for tag, sc, clen in (("A", scA, clenA), ("B", scB, clenB)):
        vals = sc[:clen].sort(descending=True).values
        if vals.numel() > 4:
            print(f"   {tag} 第4/第5名间隔: {vals[3].item():+.6e} vs {vals[4].item():+.6e} "
                  f"(gap {abs(vals[3] - vals[4]).item():.3e})")
        zeros = (vals == 0).sum().item()
        print(f"   {tag} 可达候选中恰为 0 的个数: {zeros}/{vals.numel()}")
