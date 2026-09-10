#!/usr/bin/env python3
"""定位 layer 2 indexer 分数差异的因子: q / weights / index_k 哪个不同?

debug4 结果: 偶数 query 位置 (8,10,12,14) 分数整体不同 (2.4e-2), 奇数位置 (9,11,13,15)
逐位相同; index_k 缓存终态逐位一致. 这里把三个因子分别 dump 出来对比.
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

DUMP = []
_orig = M.Indexer.forward


def _probe(self, x, qr, latent, start_pos, offset):
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
    idxs = _orig(self, x, qr, latent, start_pos, offset)
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
    M.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
    index_k = M.shared_attn.index_k[:bsz, : end_pos // ratio]
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
    DUMP.append(
        dict(
            iid=id(self),
            start_pos=start_pos,
            seqlen=seqlen,
            x=x.detach().clone(),
            qr=qr.detach().clone(),
            q=q.detach().clone(),
            latent=None if latent is None else latent.detach().clone(),
            index_k=index_k.detach().clone(),
            weights=weights.detach().clone(),
            idxs=idxs.clone(),
        )
    )
    return idxs


M.Indexer.forward = _probe


def run(full):
    args = mini_args(M)
    m = M.Transformer(args, tokenizer)
    init_params(m, seed=0)
    m.eval()
    reg = {id(l.attn.indexer): i for i, l in enumerate(m.layers) if l.attn.indexer is not None}
    mark = len(DUMP)
    if full:
        m(input_ids, 0)
    else:
        m(input_ids[:, :SPLIT], 0)
        for p in range(SPLIT, T):
            m(input_ids[:, p : p + 1], p)
    recs = []
    for d in DUMP[mark:]:
        d["layer"] = reg[d["iid"]]
        recs.append(d)
    return recs


def by_query(recs, layer, prefill_seqlen):
    """{query pos: record} 只保留指定层."""
    out = {}
    for d in recs:
        if d["layer"] != layer:
            continue
        if d["seqlen"] == prefill_seqlen:
            for p in range(prefill_seqlen):
                out[p] = d
        else:
            out[d["start_pos"]] = d
    return out


A = by_query(run(True), 2, 16)
B = by_query(run(False), 2, 8)


def show(t, n):
    return "[" + ", ".join(f"{x:+.6e}" for x in t.flatten()[:n].float().tolist()) + "]"


def dv(a, b):
    return (a.float() - b.float()).abs().max().item()


for p in (8, 9, 10, 11, 12):
    a, b = A[p], B[p]
    ia = p if a["seqlen"] == 16 else 0
    ib = p if b["seqlen"] == 16 else 0
    print(f"\n===== layer2 indexer, query pos {p} =====")
    xa, xb = a["x"][0, ia], b["x"][0, ib]
    print(f"  x       A {show(xa, 4)}  B {show(xb, 4)}  差 {dv(xa, xb):.3e}")
    qa, qb = a["qr"][0, ia], b["qr"][0, ib]
    print(f"  qr      A {show(qa, 4)}  B {show(qb, 4)}  差 {dv(qa, qb):.3e}")
    qqa, qqb = a["q"][0, ia, 0], b["q"][0, ib, 0]
    print(f"  q[h0]   A {show(qqa, 4)}  B {show(qqb, 4)}  差 {dv(qqa, qqb):.3e}")
    wa, wb = a["weights"][0, ia], b["weights"][0, ib]
    print(f"  w       A {show(wa, 4)}  B {show(wb, 4)}  差 {dv(wa, wb):.3e}")
    ka, kb = a["index_k"][0, 0], b["index_k"][0, 0]
    print(f"  k[0]    A {show(ka, 4)}  B {show(kb, 4)}  差 {dv(ka, kb):.3e}")
    print(f"  index_k 行数 A {a['index_k'].size(1)}  B {b['index_k'].size(1)}")
    print(f"  latent  A {'None' if a['latent'] is None else 'tensor'}  "
          f"B {'None' if b['latent'] is None else 'tensor'}")
    print(f"  idxs    A {a['idxs'].tolist()}  B {b['idxs'].tolist()}")
