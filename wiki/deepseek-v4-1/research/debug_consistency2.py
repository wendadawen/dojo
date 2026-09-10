#!/usr/bin/env python3
"""逐位置逐层对比 prefill(16) vs prefill(8)+decode(8), 并抓 topk 选择的离散跳变.

假设: bf16 GEMM 路径噪声 (~1e-3) 经 topk 离散选择放大 -> 表观大差异.
验证: 1) 同长度两次 prefill 逐位一致 (确定性); 2) 位置 7 (两条 prefill 共有) 的差是
纯数值噪声量级; 3) 差异放大处伴随 topk idxs 不同.
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
args = mini_args(M)
T, SPLIT = 16, 8
input_ids = torch.randint(0, 129280, (1, T), generator=torch.Generator().manual_seed(42))

# ---- 0. 确定性: 同长度两次 prefill ----
mA = M.Transformer(args, tokenizer)
init_params(mA, seed=0)
mA.eval()
hk = []
caps = {}


def cap_hook(tag):
    def hook(mod, inp, out):
        caps.setdefault(tag, []).append(inp[0].detach().clone())
    return hook


hk.append(mA.head.register_forward_hook(cap_hook("A1")))
mA(input_ids, 0)
hk.pop().remove()
hk.append(mA.head.register_forward_hook(cap_hook("A2")))
mA(input_ids, 0)
hk.pop().remove()
d = (caps["A1"][0].float() - caps["A2"][0].float()).abs().max().item()
print(f"[确定性] 两次 prefill(16) head 输入最大差: {d:.6e}")

# ---- 1. 逐位置逐层对比 ----
layers_A, layers_B = {}, {}


def layer_hooks(model, store, tag):
    hs = []
    for i, layer in enumerate(model.layers):
        def mk(i, tag):
            def hook(mod, inp, out):
                store.setdefault(i, {})[tag] = out[0].detach().clone()
            return hook
        hs.append(layer.register_forward_hook(mk(i, tag)))
    return hs


# A: prefill 16 (记录每层全位置输出, 同时抓 topk)
topk_log = {"A": {}, "B": {}}
orig_forward = M.Indexer.forward


def logged_indexer_forward(self, x, qr, latent, start_pos, offset):
    idxs = orig_forward(self, x, qr, latent, start_pos, offset)
    tag = "A" if start_pos == 0 and x.size(1) == T else "B"
    topk_log[tag].setdefault(self.layer_id if hasattr(self, "layer_id") else id(self), []).append(
        (start_pos, x.size(1), idxs.detach().clone())
    )
    return idxs


mA2 = M.Transformer(args, tokenizer)
init_params(mA2, seed=0)
mA2.eval()
hs = layer_hooks(mA2, layers_A, "full")
mA2(input_ids, 0)
for h in hs:
    h.remove()

mB2 = M.Transformer(args, tokenizer)
init_params(mB2, seed=0)
mB2.eval()
# B 每步都记录, 用 step 标记
hs = []
for i, layer in enumerate(mB2.layers):
    def mk(i):
        def hook(mod, inp, out):
            layers_B.setdefault(i, []).append(out[0].detach().clone())
        return hook
    hs.append(mB2.layers[i].register_forward_hook(mk(i)))
mB2(input_ids[:, :SPLIT], 0)
for pos in range(SPLIT, T):
    mB2(input_ids[:, pos : pos + 1], pos)
for h in hs:
    h.remove()

print("\n[逐位置] 每层在位置 7/8/12/15 的 max_diff (A prefill16 vs B prefill8+decode)")
print("layer | pos7 | pos8 | pos12 | pos15")
for i in range(args.n_layers):
    a = layers_A[i]["full"].float()  # [1, 16, hc, d]
    steps = layers_B[i]  # [prefill8, dec8, ..., dec15]
    b_full = torch.cat([steps[0]] + steps[1:], dim=1).float()  # [1, 16, ...] 但 prefill 只有 8 位置 + 8 decode = 16
    diffs = []
    for pos in (7, 8, 12, 15):
        d = (a[:, pos] - b_full[:, pos]).abs().max().item()
        diffs.append(f"{d:.3e}")
    print(f"layer {i}: {' | '.join(diffs)}")
