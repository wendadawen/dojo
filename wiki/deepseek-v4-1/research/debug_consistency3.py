#!/usr/bin/env python3
"""根因定位: prefill(16) vs prefill(8)+decode(8) 的逐位置差异从哪来.

E0 bf16 基线:
  - 16 位置 x 7 层全量 diff map (找出第一个非零差异的真实位置, 不靠采样)
  - 两条路径跑完后逐位对比 compress_kv_cache / indexer.k_cache / window_kv_cache /
    compressor 残态 (kv_state/score_state)
  - 对比三个 index source 层 (2/4/5) 每个 query 位置的 topk 选择 (归一化 offset 后)
E1 去量化对照: act_quant/fp4_act_quant 换恒等, 其余同 E0
E2 fp32 对照: 模型整体 .float() 提升, 量化保留

判定:
  - E1/E2 差异消失 -> bf16 GEMM 形状噪声经量化/离散 topk 放大 (纯数值, 非语义 bug)
  - 仍有大差异 -> 语义不一致, 继续深挖
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

# ---- Indexer topk 日志 (全局补丁, 按 id(self) 归属层) ----
TOPK_LOG = []  # (id(indexer), start_pos, seqlen, offset, idxs)
_orig_indexer_forward = M.Indexer.forward


def _logged_indexer_forward(self, x, qr, latent, start_pos, offset):
    idxs = _orig_indexer_forward(self, x, qr, latent, start_pos, offset)
    TOPK_LOG.append((id(self), start_pos, x.size(1), offset, idxs.detach().clone()))
    return idxs


M.Indexer.forward = _logged_indexer_forward


def _registry(model):
    return {
        id(layer.attn.indexer): i
        for i, layer in enumerate(model.layers)
        if layer.attn.indexer is not None
    }


def run_path(full_prefill: bool, float_model: bool):
    """full_prefill=True -> prefill(T); False -> prefill(SPLIT)+decode(T-SPLIT).
    返回 (每层输出按位置拼接 [n_layers][1,T,hc,d], topk {layer: [T 行]}, 缓存快照)."""
    args = mini_args(M)
    m = M.Transformer(args, tokenizer)
    init_params(m, seed=0)
    if float_model:
        m.float()
    m.eval()
    reg = _registry(m)

    layer_outs = {i: [] for i in range(args.n_layers)}
    hs = []
    for i, layer in enumerate(m.layers):
        def mk(i):
            def hook(mod, inp, out):
                layer_outs[i].append(out[0].detach().clone())
            return hook
        hs.append(layer.register_forward_hook(mk(i)))

    mark = len(TOPK_LOG)
    if full_prefill:
        m(input_ids, 0)
    else:
        m(input_ids[:, :SPLIT], 0)
        for pos in range(SPLIT, T):
            m(input_ids[:, pos : pos + 1], pos)
    for h in hs:
        h.remove()

    outs = [torch.cat(layer_outs[i], dim=1).float() for i in range(args.n_layers)]

    # topk 日志 -> {layer: [T 条归一化 idx 行]}  (val - offset, -1 保持)
    topk = {}
    for idx_id, start_pos, seqlen, offset, idxs in TOPK_LOG[mark:]:
        layer = reg[idx_id]
        rows = torch.where(idxs[0] >= 0, idxs[0] - offset, idxs[0])
        topk.setdefault(layer, []).append((start_pos, seqlen, rows))

    caches = {}
    for i, layer in enumerate(m.layers):
        a = layer.attn
        caches[f"win{i}"] = a.window_kv_cache.detach().clone()
        if a.compressor is not None:
            caches[f"ckv{i}"] = a.compress_kv_cache.detach().clone()
            if hasattr(a.compressor, "kv_state"):  # ratio>1 才有跨步残态
                caches[f"kvstate{i}"] = a.compressor.kv_state.detach().clone()
                caches[f"scstate{i}"] = a.compressor.score_state.detach().clone()
        if a.indexer is not None and a.indexer.owns_k:
            caches[f"ik{i}"] = a.indexer.k_cache.detach().clone()
    return outs, topk, caches


def cmp_runs(A, B, tag):
    outsA, topkA, cachesA = A
    outsB, topkB, cachesB = B
    print(f"  逐层 diff (位置数>0 的层):")
    first = None
    for i in range(len(outsA)):
        d = (outsA[i] - outsB[i]).abs().amax(dim=(0, 2, 3))  # [T]
        nz = [p for p in range(T) if d[p] > 0]
        if nz:
            print(f"    layer {i}: {len(nz)} 个位置有差, 首个 pos {nz[0]}, "
                  f"max {d.max().item():.3e} (pos {d.argmax().item()})")
            if first is None:
                first = (i, nz[0])
    if first is None:
        print("    全部位置全部层逐位一致")
    print(f"  首个非零差异: {first}")
    print(f"  缓存逐位对比:")
    for k in cachesA:
        a, b = cachesA[k].float(), cachesB[k].float()
        n_mis = (a != b).sum().item()
        print(f"    {k}: max {((a - b).abs().max().item()):.3e}, 不一致元素 {n_mis}/{a.numel()}")
    print(f"  topk 选择对比 (归一化后):")
    for layer in sorted(topkA):
        rowsA = {}
        for start_pos, seqlen, rows in topkA[layer]:
            for j in range(seqlen):
                rowsA[start_pos + j] = rows[j]
        rowsB = {}
        for start_pos, seqlen, rows in topkB[layer]:
            for j in range(seqlen):
                rowsB[start_pos + j] = rows[j]
        bad = []
        for p in range(T):
            sa = sorted(rowsA[p].tolist())
            sb = sorted(rowsB[p].tolist())
            # 长度可能不同 (topk=min(topk, len)); 按集合比较非 -1 部分
            sa = [v for v in sa if v >= 0]
            sb = [v for v in sb if v >= 0]
            if sa != sb:
                bad.append((p, sa, sb))
        if bad:
            print(f"    layer {layer}: {len(bad)} 个位置选择不同 -> {bad[:4]}")
        else:
            print(f"    layer {layer}: 全部 {T} 个位置选择一致")
    return first


def run_pair(tag, float_model=False):
    print(f"[{tag}]")
    A = run_path(True, float_model)
    B = run_path(False, float_model)
    first = cmp_runs(A, B, tag)
    # decode 路径自一致性 (确定性)
    B2 = run_path(False, float_model)
    dmax = max((B[0][i] - B2[0][i]).abs().max().item() for i in range(len(B[0])))
    print(f"  decode 路径两次重复 max diff: {dmax:.3e}\n")
    return first


run_pair("E0 bf16 基线")

_orig_act_quant, _orig_fp4 = M.act_quant, M.fp4_act_quant
M.act_quant = lambda x, *a, **k: x
M.fp4_act_quant = lambda x, *a, **k: x
run_pair("E1 去量化对照")
M.act_quant, M.fp4_act_quant = _orig_act_quant, _orig_fp4

# E2 专用: 官方 engram 查找表硬编码 .to(bfloat16), fp32 对照时保持 fp32
_orig_emb_fwd = M.ParallelEngramEmbedding.forward


def _emb_fwd_keepdtype(self, indices):
    out = _orig_emb_fwd(self, indices)
    return out.float() if self.weight.dtype == torch.float32 else out


M.ParallelEngramEmbedding.forward = _emb_fwd_keepdtype
run_pair("E2 fp32 对照 (量化保留)", float_model=True)
M.ParallelEngramEmbedding.forward = _orig_emb_fwd
