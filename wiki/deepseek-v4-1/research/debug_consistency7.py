#!/usr/bin/env python3
"""三重取证 + 分离两个原因.

(1) 取证 stale-global: layer 2 indexer 在偶数 decode 步读到的 index_k 是不是自己的 cache
(2) 修正后 bf16 残差: 剩余差异来自 bf16 GEMM 形状噪声 + fp4 量化翻转 (数值, 不可消除)
(3) 修正 + fp32: 若差异归零, 说明 (2) 的判断成立, 且 (1) 是唯一的语义问题
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

_orig = M.Indexer.forward
FIX = {"on": False}
READS = []  # (layer_id 占位, start_pos, owns_k, read_is_own, k_first4)


def _probe(self, x, qr, latent, start_pos, offset):
    if FIX["on"] and self.owns_k:
        M.shared_attn.index_k = self.k_cache
    rec = dict(owns_k=self.owns_k, start_pos=start_pos, latent_none=latent is None)
    if self.owns_k:
        pre = M.shared_attn.index_k  # 官方代码在 latent is None 时读的就是它
        rec["pre_is_own"] = pre is self.k_cache
        rec["pre_vals"] = None if pre is None else pre[0, 0, :4].detach().clone()
    out = _orig(self, x, qr, latent, start_pos, offset)
    # latent 非 None 时官方会先把自己的 cache 发布再读 -> 读到自己
    rec["read_own"] = True if not rec.get("latent_none", True) else rec.get("pre_is_own")
    READS.append(rec)
    return out


M.Indexer.forward = _probe

# fp32 对照需要: 官方 engram 查找表硬编码 .to(bfloat16), 提升精度时保持 fp32
_orig_emb = M.ParallelEngramEmbedding.forward


def _emb_keepdtype(self, indices):
    out = _orig_emb(self, indices)
    return out.float() if self.weight.dtype == torch.float32 else out


def run(full, float_model=False, layer_watch=None):
    M.ParallelEngramEmbedding.forward = _emb_keepdtype if float_model else _orig_emb
    args = mini_args(M)
    m = M.Transformer(args, tokenizer)
    init_params(m, seed=0)
    if float_model:
        m.float()
    m.eval()
    outs = {i: [] for i in range(args.n_layers)}
    hs = []
    for i, layer in enumerate(m.layers):
        def mk(i):
            def hook(mod, inp, out):
                outs[i].append(out[0].detach().clone())
            return hook
        hs.append(layer.register_forward_hook(mk(i)))
    mark = len(READS)
    if full:
        m(input_ids, 0)
    else:
        m(input_ids[:, :SPLIT], 0)
        for p in range(SPLIT, T):
            m(input_ids[:, p : p + 1], p)
    for h in hs:
        h.remove()
    outs = [torch.cat(outs[i], dim=1).float() for i in range(args.n_layers)]
    return outs, READS[mark:], m


def diffs(A, B):
    rows = []
    for i in range(len(A)):
        d = (A[i] - B[i]).abs()
        rows.append((i, d.max().item(), int((d.amax(dim=(0, 2, 3)) > 0).sum())))
    return rows


print("========== (1) 取证: layer 2 indexer 读到的 index_k 属于谁 ==========")
FIX["on"] = False
outsA, readsA, mA = run(True)
outsB, readsB, mB = run(False)
print("  decode 段 (start_pos>=8) 的 layer2 indexer 读 cache 情况:")
for r in readsB:
    if r["owns_k"] and r["start_pos"] >= 8:
        vals = "None" if r["pre_vals"] is None else [f"{v:+.4f}" for v in r["pre_vals"].float().tolist()]
        print(f"    step {r['start_pos']}: latent None? {r['latent_none']}  "
              f"读到自己的 cache? {r['read_own']}  读到的行0首4值 {vals}")
print("  layer2 自己的 k_cache 行0 首4值:",
      [f"{v:+.4f}" for v in mB.layers[2].attn.indexer.k_cache[0, 0, :4].float().tolist()])
print("  layer4 的 k_cache 行0 首4值:",
      [f"{v:+.4f}" for v in mB.layers[4].attn.indexer.k_cache[0, 0, :4].float().tolist()])

print("\n========== (2) 修正后 bf16 ==========")
FIX["on"] = True
a2, _, _ = run(True)
b2, _, _ = run(False)
for i, mx, nz in diffs(a2, b2):
    print(f"  layer {i}: max {mx:.3e}, 有差位置数 {nz}")

print("\n========== (3) 修正 + fp32 ==========")
FIX["on"] = True
a3, _, _ = run(True, float_model=True)
b3, _, _ = run(False, float_model=True)
for i, mx, nz in diffs(a3, b3):
    print(f"  layer {i}: max {mx:.3e}, 有差位置数 {nz}")
