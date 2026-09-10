#!/usr/bin/env python3
"""tilelang kernel 的纯 PyTorch 语义复现, 用于在 CPU 上注入官方 model.py。

每个函数对照 official/inference/kernel.py 的同名 kernel 逐行语义:
  - act_quant:      kernel.py L40-124  (act_quant_kernel)
  - fp4_act_quant:  kernel.py L127-204 (fp4_quant_kernel)
  - hc_split_sinkhorn: kernel.py L406-474
  - sparse_attn:    kernel.py L310-403 (sparse_attn_kernel)
  - fp8_gemm/fp4_gemm: kernel.py L207-308 / L477-591, 数学等价的反量化实现
FP4 值表取自 official/inference/convert.py L13-15 (FP4_TABLE)。
"""
import torch

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
FP8_MAX = 448.0
FP4_MAX = 6.0


def _round_scale_pow2(x: torch.Tensor) -> torch.Tensor:
    """fast_round_scale: 2^ceil(log2(x)), kernel.py L22-37."""
    return torch.pow(2.0, torch.ceil(torch.log2(x)))


def _to_fp4(v: torch.Tensor) -> torch.Tensor:
    """cast(FP4, v): 到 E2M1 值表的最近舍入 (round-to-nearest, 平局向偶数值侧).

    E2M1 相邻点中点为 0.25/0.75/1.25/1.75/2.5/3.5/5; RNTE 下平局落向尾数偶数
    (0,1,2,4 侧). 用查表实现, 平局取下界值即对应偶数尾数.
    """
    abs_v = v.abs().clamp(max=FP4_MAX)
    pos = FP4_TABLE[:8]
    idx = torch.searchsorted(pos, abs_v.contiguous(), right=True)
    idx = idx.clamp(1, 7)
    lo, hi = pos[idx - 1], pos[idx]
    # 平局 (abs_v - lo == hi - abs_v) 时取 lo: 0.25->0, 0.75->0.5? 0.5 的二进制尾数是 0... 实际上
    # E2M1 的 RNTE 平局落向编码偶数. 0.25: 0(编码偶) vs 0.5(编码奇) -> 0 ✓ 下界
    # 0.75: 0.5(奇) vs 1.0(偶) -> 1.0 上界; 1.25: 1.0(偶) vs 1.5(奇) -> 1.0 下界
    # 1.75: 1.5(奇) vs 2.0(偶) -> 2.0 上界; 2.5: 2(偶) vs 3(奇) -> 2 下界
    # 3.5: 3(奇) vs 4(偶) -> 4 上界; 5: 4(偶) vs 6(奇) -> 4 下界
    take_hi = (abs_v - lo) > (hi - abs_v)
    # 平局单独处理: 0.75/1.75/3.5 取上界(偶数编码), 其余平局取下界
    tie = (abs_v - lo) == (hi - abs_v)
    tie_hi = tie & ((abs_v == 0.75) | (abs_v == 1.75) | (abs_v == 3.5))
    rounded = torch.where(take_hi | tie_hi, hi, lo)
    return rounded.copysign(v)


def act_quant(x, block_size, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """块级 FP8 量化. inplace=True 时量化+反量化回写原 dtype (QAT 模拟)."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float()
    blocks = z.unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(-1).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _round_scale_pow2(amax / FP8_MAX)
    else:
        s = amax / FP8_MAX
    q = (blocks / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    if inplace:
        y = q.to(torch.float8_e4m3fn).float() * s.unsqueeze(-1)
        y = y.flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    y = q.to(torch.float8_e4m3fn)
    return y.flatten(-2), s.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """FP4(E2M1) 量化. e4m3 scale (compressed KV, block 16) 或 e8m0 (indexer, block 32)."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float()
    blocks = z.unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * (2**-9))
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()  # cast(FP8, amax/6) 有舍入
    else:
        amax = amax.clamp_min(6 * (2**-126))
        s = _round_scale_pow2(amax / FP4_MAX)
    q = (blocks / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX)
    if inplace:
        y = _to_fp4(q) * s.unsqueeze(-1)
        y = y.flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    # 打包路径本机用不上 (推理代码总是 inplace=True 调用), 返回浮点逻辑值
    return _to_fp4(q).flatten(-2), s.to(scale_dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """kernel.py L426-460:
    pre  = sigmoid(m * scale[0] + base) + eps
    post = 2 * sigmoid(m * scale[1] + base)
    comb = softmax(-1)(m * scale[2] + base) + eps  -> 列归一化 -> (iters-1) x (行, 列) 归一化
    即行归一化共 iters 次 (softmax 那次 + 循环 iters-1 次), 列归一化 iters 次, 末步在列方向.
    """
    b, s_, _ = mixes.size()
    m = mixes.view(-1, (2 + hc_mult) * hc_mult).float()
    pre = torch.sigmoid(m[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(m[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = m[:, 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    comb = comb.view(-1, hc_mult, hc_mult)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return (
        pre.view(b, s_, hc_mult),
        post.view(b, s_, hc_mult),
        comb.view(b, s_, hc_mult, hc_mult),
    )


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """按索引聚集的稀疏注意力, kernel.py L310-403 的数学等价朴素实现.

    kernel 语义要点:
      - topk_idxs == -1 的槽位: kv 取 0, score 取 -inf
      - 在线 softmax 的 max 只来自 score (不含 sink), 最后 sum_exp += exp(sink - max)
      - 全 -1 的行: max = -1e30 (有限下界), 分子 0, 输出 0
    """
    b, m, h, d = q.size()
    n = kv.size(1)
    idxs = topk_idxs.long()
    valid = idxs != -1
    safe = idxs.clamp_min(0)
    gathered = kv[torch.arange(b).unsqueeze(-1).unsqueeze(-1), safe]  # [b, m, topk, d]
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    row_max = scores.amax(-1, keepdim=True)
    row_max = torch.where(torch.isinf(row_max), torch.full_like(row_max, -1e30), row_max)
    p = (scores - row_max).exp()
    p = p.masked_fill(~valid.unsqueeze(2), 0.0)
    denom = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float() - row_max.squeeze(-1)).unsqueeze(-1)
    o = torch.einsum("bmht,bmtd->bmhd", p, gathered.float()) / denom
    return o.to(q.dtype)


def _dequant_fp8_blocks(w, s, block_size):
    blocks = w.float().unflatten(0, (-1, block_size)).unflatten(-1, (-1, block_size))
    s = s.float().unflatten(0, (-1, s.size(0) // blocks.size(0) if False else 1))
    return w  # placeholder, 缩小测试不走量化 GEMM


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """数学等价: 分块反量化后 matmul (kernel 是分块累加, 数值略有差异)."""
    K = a.size(-1)
    af = a.float().unflatten(-1, (-1, block_size)) * a_s.float().unsqueeze(-1)
    af = af.flatten(-2)
    bs = block_size
    bf = b.float().unflatten(0, (-1, bs)).unflatten(-1, (-1, bs))
    bf = bf * b_s.float().unsqueeze(1).unsqueeze(-1)
    bf = bf.flatten(0, 1).flatten(-2)
    return (af @ bf.T).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """FP8 激活 x FP4 权重. b: [N, K//2] 打包或逻辑 [N, K] 浮点."""
    K = a.size(-1)
    af = a.float().unflatten(-1, (-1, act_block_size)) * a_s.float().unsqueeze(-1)
    af = af.flatten(-2)
    if b.dtype == torch.int8 or (hasattr(b, "dtype") and "float4" in str(b.dtype)):
        raw = b.view(torch.uint8) if b.dtype != torch.uint8 else b
        lo, hi = raw & 0x0F, (raw >> 4) & 0x0F
        bf = torch.stack([FP4_TABLE[lo.long()], FP4_TABLE[hi.long()]], dim=-1).flatten(1)
    else:
        bf = b.float()
    bf = bf.unflatten(-1, (-1, 32)) * b_s.float().unsqueeze(-1)
    bf = bf.flatten(-2)
    return (af @ bf.T).to(torch.get_default_dtype())
