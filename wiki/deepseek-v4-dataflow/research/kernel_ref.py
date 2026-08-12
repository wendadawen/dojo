"""纯 PyTorch 等价实现 DeepSeek-V4-Pro inference/kernel.py 中的 CUDA/tilelang 算子。

官方 kernel.py 依赖 tilelang，只能在 NVIDIA GPU 上编译运行。本文件按官方 kernel 的
逐行语义在 CPU 上等价复现，用于在无 GPU 环境下实测数据流。

对应关系：
  act_quant          <- kernel.py act_quant_kernel      (block FP8 量化，inplace=量化后反量化)
  fp4_act_quant      <- kernel.py fp4_quant_kernel
  sparse_attn        <- kernel.py sparse_attn_kernel    (index gather + online softmax + attn_sink)
  hc_split_sinkhorn  <- kernel.py hc_split_sinkhorn_kernel
  fp8_gemm/fp4_gemm  <- 以 bf16 matmul 近似（本文件仅用于形状/数据流验证）
"""
import torch
import torch.nn.functional as F


def _fast_round_scale(amax, max_inv):
    """对应 kernel.py fast_round_scale：scale 取 2^ceil(log2(amax/max))。"""
    return torch.pow(2.0, torch.ceil(torch.log2(amax * max_inv)))


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """对应 kernel.py act_quant / act_quant_kernel。

    每 block_size 个元素（沿最后一维）共享一个 scale，amax 下限 1e-4，FP8 范围 ±448。
    inplace=True 时执行「量化再反量化」写回原张量（QAT 模拟），返回 x。
    """
    N = x.size(-1)
    assert N % block_size == 0
    fp8_max = 448.0
    z = x.contiguous()
    flat = z.view(-1, N)
    grouped = flat.view(flat.size(0), N // block_size, block_size).float()
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _fast_round_scale(amax, 1.0 / fp8_max)
    else:
        s = amax / fp8_max
    q = torch.clamp(grouped / s.unsqueeze(-1), -fp8_max, fp8_max)
    q8 = q.to(torch.float8_e4m3fn)
    if inplace:
        deq = (q8.float() * s.unsqueeze(-1)).reshape(flat.shape).view_as(x)
        x.copy_(deq.to(x.dtype))
        return x
    return q8.reshape(flat.shape).view_as(x), s.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False):
    """对应 kernel.py fp4_act_quant / fp4_quant_kernel。FP4(e2m1) 最大值 6，scale 取 2 的幂。"""
    N = x.size(-1)
    assert N % block_size == 0
    fp4_max = 6.0
    flat = x.contiguous().view(-1, N)
    grouped = flat.view(flat.size(0), N // block_size, block_size).float()
    amax = grouped.abs().amax(dim=-1).clamp_min(6 * (2.0 ** -126))
    s = _fast_round_scale(amax, 1.0 / fp4_max)
    q = torch.clamp(grouped / s.unsqueeze(-1), -fp4_max, fp4_max)
    # e2m1 可表示值：0,0.5,1,1.5,2,3,4,6（含符号）
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    sign = torch.sign(q)
    qa = q.abs()
    idx = (qa.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    qr = sign * levels[idx]
    if inplace:
        deq = (qr * s.unsqueeze(-1)).reshape(flat.shape).view_as(x)
        x.copy_(deq.to(x.dtype))
        return x
    return qr.reshape(flat.shape).view_as(x), s


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """对应 kernel.py sparse_attn / sparse_attn_kernel。

    q: [b,m,h,d]  kv: [b,n,d]（MQA，单一 KV 头，key 与 value 同一张量）
    topk_idxs: [b,m,topk] int，-1 表示无效位置（被 mask 掉）
    attn_sink: [h] 学习到的 sink logit，Exp(sink) 加进 softmax 分母。
    """
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    idx = topk_idxs.long()
    valid = idx != -1
    safe = idx.clamp_min(0)
    # gather KV：[b,m,topk,d]
    gathered = torch.gather(
        kv.unsqueeze(1).expand(b, m, kv.size(1), d),
        2,
        safe.unsqueeze(-1).expand(b, m, topk, d),
    )
    gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    # logits: [b,m,h,topk]
    logits = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered.float()) * softmax_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = logits.amax(dim=-1, keepdim=True)
    mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
    e = torch.exp(logits - mx)
    denom = e.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - mx.squeeze(-1))
    o = torch.einsum("bmht,bmtd->bmhd", e, gathered.float()) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """对应 kernel.py hc_split_sinkhorn / hc_split_sinkhorn_kernel。

    mixes: [b,s,(2+hc)*hc] -> 切成三段：
      pre  = sigmoid(mixes[:hc]   * scale0 + base) + eps      -> [b,s,hc]
      post = 2*sigmoid(mixes[hc:2hc]* scale1 + base)          -> [b,s,hc]
      comb = Sinkhorn(mixes[2hc:] * scale2 + base)            -> [b,s,hc,hc]
    Sinkhorn：先 softmax(-1)+eps，再列归一化，随后 (iters-1) 轮「行归一化 -> 列归一化」。
    """
    b, s, _ = mixes.shape
    m = mixes.float()
    hc = hc_mult
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.view(b, s, hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """形状等价占位：本机验证数据流用，不追求 FP8 数值一致。"""
    raise NotImplementedError("本机实测用 bf16 linear，不走 fp8_gemm")


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    raise NotImplementedError("本机实测用 bf16 linear，不走 fp4_gemm")
