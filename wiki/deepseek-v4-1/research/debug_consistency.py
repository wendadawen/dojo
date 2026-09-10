#!/usr/bin/env python3
"""逐层定位 prefill/decode 分叉点: 两个同种子模型, 每层 Block 输出在共同位置对比."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mini_model import init_params, load_model_module, mini_args

M = load_model_module()
torch.set_default_dtype(torch.bfloat16)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(str(Path(__file__).parent / "official"))
args = mini_args(M)

T, SPLIT = 16, 8
input_ids = torch.randint(0, 129280, (1, T), generator=torch.Generator().manual_seed(42))

outs = {"A": {}, "B": {}}


def layer_hooks(model, tag):
    hs = []
    for i, layer in enumerate(model.layers):
        def mk(i):
            def hook(mod, inp, out):
                outs[tag][i] = out[0].detach().clone()
            return hook
        hs.append(layer.register_forward_hook(mk(i)))
    return hs


mA = M.Transformer(args, tokenizer)
init_params(mA, seed=0)
mA.eval()
hs = layer_hooks(mA, "A")
mA(input_ids, 0)
for h in hs:
    h.remove()

mB = M.Transformer(args, tokenizer)
init_params(mB, seed=0)
mB.eval()
hs = layer_hooks(mB, "B")
mB(input_ids[:, :SPLIT], 0)
for pos in range(SPLIT, T):
    mB(input_ids[:, pos : pos + 1], pos)
for h in hs:
    h.remove()

print("层 | 路径A形状 | 路径B形状 | 共同位置最大差")
for i in range(args.n_layers):
    a, b = outs["A"][i], outs["B"][i]
    sa, sb = a.size(1), b.size(1)
    # 共同位置: B 的最后 sb 个位置对应 A 的 (T-sb)..T-1
    ca, cb = a[:, T - sb :], b
    d = (ca.float() - cb.float()).abs().max().item()
    print(f"layer {i}: A[1,{sa},...] B[1,{sb},...] 共同 {sb} 位置 max_diff = {d:.6e}")
