"""实测二：Compressor 的压缩行为——CSA(ratio=4, overlap) 与 HCA(ratio=128, 非overlap)。

验证目标（对应技术报告 Eq.9-12 与 model.py Compressor）：
  1. ratio=4 时 self.overlap=True，权重矩阵宽度翻倍（coff=2），每个压缩条目由 2m 个原始条目产生
  2. softmax 在 2m 个元素上归一化（报告：normalization across the total of 2m elements）
  3. 序列长度被压缩到 1/ratio（不是 1/2m）
  4. ratio=128 时 overlap=False，仅 m 个条目参与
  5. prefill 与 decode 增量压缩结果一致
"""
import sys, math, torch
sys.path.insert(0, "/tmp/dsv4")
sys.path.insert(0, "/tmp/dsv4/exp")

import torch.nn as nn
import torch.nn.functional as F
from kernel_ref import act_quant, fp4_act_quant

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)


class RMSNorm(nn.Module):
    """同 model.py RMSNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        d = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(d)


class Compressor(nn.Module):
    """严格照抄 model.py Compressor 的 forward 结构，去掉 RoPE/量化以聚焦压缩语义。"""
    def __init__(self, dim, compress_ratio, head_dim, max_bsz=2, eps=1e-6):
        super().__init__()
        self.dim, self.head_dim = dim, head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4          # model.py: self.overlap = compress_ratio == 4
        coff = 1 + self.overlap
        self.coff = coff
        self.ape = nn.Parameter(torch.randn(compress_ratio, coff * head_dim) * 0.1)
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        self.norm = RMSNorm(head_dim, eps)
        self.register_buffer("kv_state", torch.zeros(max_bsz, coff * compress_ratio, coff * head_dim))
        self.register_buffer("score_state", torch.full((max_bsz, coff * compress_ratio, coff * head_dim), float("-inf")))

    def overlap_transform(self, tensor, value=0):
        """同 model.py：把 [b,s,r,2d] 重排为 [b,s,2r,d]，前 r 个取上一块的前半维，后 r 个取本块后半维。"""
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x, start_pos):
        bsz, seqlen, _ = x.size()
        ratio, overlap, d = self.compress_ratio, self.overlap, self.head_dim
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio:cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff-ratio:cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset:offset+remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset:offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            self._last_softmax_width = score.size(2)
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    self._last_softmax_width = score_state.size(1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    self._last_softmax_width = self.score_state[:bsz].size(1)
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv)


DIM, HEAD_DIM = 256, 64
print("=== CSA 层（compress_ratio=4，config.json 中偶数层取值）===")
c4 = Compressor(DIM, 4, HEAD_DIM)
print(f"overlap={c4.overlap}  coff={c4.coff}")
print(f"wkv 输出维度 = {c4.wkv.out_features} = coff({c4.coff}) x head_dim({HEAD_DIM})")
print(f"ape 形状 = {tuple(c4.ape.shape)} = (ratio, coff*head_dim)   对应报告 B_a,B_b in R^(m x c)")
x = torch.randn(2, 64, DIM)
out4 = c4(x, 0)
print(f"输入 seqlen=64 -> 压缩条目数 {out4.size(1)} = 64/4，序列压缩到 1/ratio")
print(f"softmax 归一化宽度 = {c4._last_softmax_width} = 2m (m=4)  <- 报告称在 2m 个元素上归一化")
print()

print("=== HCA 层（compress_ratio=128，config.json 中奇数层取值）===")
c128 = Compressor(DIM, 128, HEAD_DIM)
print(f"overlap={c128.overlap}  coff={c128.coff}")
print(f"wkv 输出维度 = {c128.wkv.out_features} = coff({c128.coff}) x head_dim({HEAD_DIM})")
x2 = torch.randn(2, 512, DIM)
out128 = c128(x2, 0)
print(f"输入 seqlen=512 -> 压缩条目数 {out128.size(1)} = 512/128")
print(f"softmax 归一化宽度 = {c128._last_softmax_width} = m (m=128)  <- 非 overlap，仅 m 个条目")
print()

print("=== 压缩权重确实是数据相关的 softmax（不是均值池化）===")
c4b = Compressor(DIM, 4, HEAD_DIM)
xa = torch.randn(1, 8, DIM)
o1 = c4b(xa, 0)
# 均值池化对照
mean_pool = c4b.wkv(xa.float())[..., HEAD_DIM:].unflatten(1, (-1, 4)).mean(2)
print(f"compressor 输出 与 均值池化 的相对差异: {((o1 - c4b.norm(mean_pool)).norm()/o1.norm()).item():.4f}")
print("  非 0 => 压缩是学习到的加权求和，权重由 wgate 从隐藏状态算出")
print()

print("=== decode 增量压缩 与 prefill 一次性压缩 是否一致（HCA, ratio=128 便于对齐）===")
R = 128
cA = Compressor(DIM, R, HEAD_DIM, max_bsz=1)
cB = Compressor(DIM, R, HEAD_DIM, max_bsz=1)
cB.load_state_dict(cA.state_dict())
xs = torch.randn(1, R, DIM)
pre = cA(xs, 0)                                  # prefill 一次压 128 个 token
inc = None
for t in range(R):                               # decode 每步喂 1 个 token
    r = cB(xs[:, t:t+1], t if t > 0 else 0) if t > 0 else cB(xs[:, 0:1], 0)
    if r is not None:
        inc = r
# t=0 走 prefill 分支(seqlen=1 < ratio 不压缩)，其余走 decode 分支
print(f"prefill 输出形状 {tuple(pre.shape)}   decode 最终输出形状 {tuple(inc.shape) if inc is not None else None}")
if inc is not None:
    print(f"最大绝对差异: {(pre[:, -1] - inc[:, 0]).abs().max().item():.3e}  （同一 state 缓冲区语义一致）")
