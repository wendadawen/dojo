#!/usr/bin/env python3
"""决定性验证: 修掉 "owner 层读到自己 cache" 后, prefill/decode 差异是否归零.

诊断 (debug5): 偶数 decode 步 (latent is None) 时, ratio=2 的 index source 层不执行
`shared_attn.index_k = self.k_cache`, 却无条件读 `shared_attn.index_k`, 于是读到上一个
forward 中最后发布者的 cache (ratio=1 的层) -> 索引键错位 -> topk 选择不同.

最小修正: owner 层无条件发布自己的 cache (等价于 "owner 读自己的 cache").
对照: 不修 (原样) vs 修 (补丁), 各跑 prefill16 / prefill8+decode8 一致性.
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

_orig_indexer_forward = M.Indexer.forward
FIX = {"on": False}


def _patched(self, x, qr, latent, start_pos, offset):
    if FIX["on"] and self.owns_k:
        # owner 层无条件发布自己的 cache: 修正 "组未满时读到别的层 cache"
        M.shared_attn.index_k = self.k_cache
    return _orig_indexer_forward(self, x, qr, latent, start_pos, offset)


M.Indexer.forward = _patched


def run(full):
    args = mini_args(M)
    m = M.Transformer(args, tokenizer)
    init_params(m, seed=0)
    m.eval()
    outs, topk = {i: [] for i in range(args.n_layers)}, {}
    reg = {id(l.attn.indexer): i for i, l in enumerate(m.layers) if l.attn.indexer is not None}

    hs = []
    for i, layer in enumerate(m.layers):
        def mk(i):
            def hook(mod, inp, out):
                outs[i].append(out[0].detach().clone())
            return hook
        hs.append(layer.register_forward_hook(mk(i)))

    if full:
        m(input_ids, 0)
    else:
        m(input_ids[:, :SPLIT], 0)
        for p in range(SPLIT, T):
            m(input_ids[:, p : p + 1], p)
    for h in hs:
        h.remove()

    outs = [torch.cat(outs[i], dim=1).float() for i in range(args.n_layers)]
    for i, layer in enumerate(m.layers):
        if layer.attn.indexer is not None and layer.attn.indexer.owns_k:
            topk[i] = layer.attn.indexer.k_cache.detach().clone()
    return outs, topk


def compare(tag):
    A = run(True)
    B = run(False)
    print(f"\n=== {tag} ===")
    allzero = True
    for i in range(len(A[0])):
        d = (A[0][i] - B[0][i]).abs()
        m_ = d.max().item()
        nz = int((d.amax(dim=(0, 2, 3)) > 0).sum())
        if m_ > 0:
            allzero = False
        print(f"  layer {i}: max {m_:.3e}, 有差位置数 {nz}")
    print(f"  -> {'全部逐位一致' if allzero else '存在差异'}")
    # 缓存终态
    for i in sorted(A[1]):
        a, b = A[1][i].float(), B[1][i].float()
        print(f"  indexer.k_cache layer {i}: max {((a - b).abs().max().item()):.3e}, "
              f"不一致 {(a != b).sum().item()}/{a.numel()}")
    return allzero


print("---- 原样 (不修) ----")
FIX["on"] = False
compare("原样")
print("\n---- 修正后 (owner 无条件发布) ----")
FIX["on"] = True
compare("修正")
