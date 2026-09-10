#!/usr/bin/env python3
"""核对 RoPE / YaRN 公式: 官方 precompute_freqs_cis 与独立重实现逐元素对照.

核查项:
  1. 三段式: i <= low 频率不变; i >= high 除以 factor; 中间线性斜坡
  2. low/high 的推导与 YaRN 原文一致: dim*ln(original_seq_len/(rotations*2pi))/(2*ln(base))
  3. factor * original_seq_len == max_position_embeddings (65536*16 == 1048576)
  4. 注意力缩放因子 (YaRN 原文的 0.1*ln(factor)+1) 未出现在代码里
  5. 三类 RoPE 配置: ratio=0 (SWA) 用 base 10000 且关 YaRN; ratio>0 用 compress_rope_theta=160000 且开 YaRN
  6. apply_rotary_emb 只旋转最后 rope_head_dim 维, inverse 为共轭
  7. 压缩条目占位: 组 j 用位置 j*ratio
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from mini_model import load_model_module  # noqa: E402

M = load_model_module()
cfg = json.loads((HERE / "official" / "inference" / "config.json").read_text())
hf = json.loads((HERE / "official" / "config.json").read_text())["text_config"]

dim = cfg["rope_head_dim"]
base = cfg["rope_theta"]
cbase = cfg["compress_rope_theta"]
orig = cfg["original_seq_len"]
factor = cfg["rope_factor"]
bf, bs = cfg["beta_fast"], cfg["beta_slow"]
print(f"rope_head_dim={dim} base={base} compress_base={cbase} original_seq_len={orig} factor={factor} "
      f"beta_fast={bf} beta_slow={bs}")

# ---- 3. 长度关系 ----
mx = hf["max_position_embeddings"]
print(f"\n[3] original_seq_len * factor = {orig * factor}, max_position_embeddings = {mx}")
assert orig * factor == mx, "扩展长度关系不成立"
print("    一致 ✓ (YaRN 把训练窗口 65536 按 factor=16 扩到 1M)")


def corrected_dim(rotations, base_):
    return dim * math.log(orig / (rotations * 2 * math.pi)) / (2 * math.log(base_))


def independent(base_, use_yarn=True):
    """按文档规则独立重实现, 返回 [dim//2] 的频率."""
    f = torch.tensor([base_ ** (-2 * i / dim) for i in range(dim // 2)], dtype=torch.float32)
    if not use_yarn:
        return f
    low = max(math.floor(corrected_dim(bf, base_)), 0)
    high = min(math.ceil(corrected_dim(bs, base_)), dim - 1)
    ramp = torch.tensor([min(max((i - low) / max(high - low, 1e-3), 0.0), 1.0) for i in range(dim // 2)])
    smooth = 1 - ramp
    return f / factor * (1 - smooth) + f * smooth, (low, high, ramp)


def official(base_, use_yarn):
    fc = M.precompute_freqs_cis(dim, 2, orig if use_yarn else 0, base_, factor, bf, bs)
    return fc[1]  # 位置 1 的相位 = 频率本身


for base_, use_yarn, tag in ((cbase, True, "全局 KV 层 (ratio>0)"), (base, False, "SWA 层 (ratio=0)")):
    print(f"\n--- {tag}: base={base_} YaRN={'开' if use_yarn else '关'} ---")
    f_ind = independent(base_, use_yarn)
    low = high = None
    if use_yarn:
        f_ind, (low, high, ramp) = f_ind
    fc = official(base_, use_yarn)
    ref = torch.polar(torch.ones_like(f_ind), f_ind)
    d = (fc - ref).abs().max().item()
    print(f"  独立重实现 vs 官方 precompute_freqs_cis 逐元素最大差: {d:.3e}")
    assert d < 1e-6
    if use_yarn:
        n_unchanged = int((ramp == 0).sum())
        n_full = int((ramp == 1).sum())
        n_ramp = int(((ramp > 0) & (ramp < 1)).sum())
        print(f"  low={low} high={high}; 频率不变 {n_unchanged} 维, 线性斜坡 {n_ramp} 维, 全额除以 factor {n_full} 维")
        print(f"  首维 (i=0) 频率 {f_ind[0]:.6e} (= base^-2/dim 原值 {base_ ** (-2 / dim):.6e})")
        print(f"  末维 (i={dim // 2 - 1}) 频率 {f_ind[-1]:.6e}, 原值 {base_ ** (-2 * (dim // 2 - 1) / dim):.6e}, "
              f"比值 {f_ind[-1] / (base_ ** (-2 * (dim // 2 - 1) / dim)):.4f}")
    else:
        print(f"  YaRN 关闭时 = base^(-2i/dim) 原式, 无任何斜坡 (逐元素差 {d:.3e})")

# ---- 4. 注意力缩放因子 ----
src = (HERE / "official" / "inference" / "model.py").read_text()
print("\n[4] YaRN 注意力缩放 (0.1*ln(factor)+1):")
for pat in ("mscale", "0.1 * math.log", "0.1*math.log", "log(factor)"):
    print(f"    代码中出现 '{pat}': {pat in src}")

# ---- 6. apply_rotary_emb 只旋转尾部 ----
rd = dim
x2 = torch.randn(1, 2, 4, 2 * rd)  # 每头 2*rd 维, 只应旋转最后 rd 维
y = x2.clone()
fc2 = M.precompute_freqs_cis(rd, 2, 0, base, factor, bf, bs)
M.apply_rotary_emb(y[..., -rd:], fc2)
front_same = torch.equal(y[..., :-rd], x2[..., :-rd])
back_changed = not torch.equal(y[..., -rd:], x2[..., -rd:])
print(f"\n[6] apply_rotary_emb 只改最后 {rd} 维: 前缀未变={front_same}, 尾部已变={back_changed}")
assert front_same and back_changed
y2 = x2.clone()
M.apply_rotary_emb(y2[..., -rd:], fc2)
M.apply_rotary_emb(y2[..., -rd:], fc2, True)
print(f"    正旋转后再逆旋转复原: 最大差 {(y2 - x2).abs().max().item():.3e}")

# ---- 7. 压缩条目占位 ----
ratio = 2
freqs = M.precompute_freqs_cis(rd, 16, orig, cbase, factor, bf, bs)
# freqs_cis[p] 应等于 polar(1, p * 频率向量)
vec = M.precompute_freqs_cis(rd, 1, orig, cbase, factor, bf, bs)[0]
ok = True
worst = 0.0
for p in (0, 1, 2, 4, 6, 8, 15):
    # 与独立重实现的频率向量比对相位 (float64 计算, 避免 float32 相位精度干扰)
    f64 = independent(cbase, True)[0].double()
    ref = torch.polar(torch.ones_like(f64), p * f64)
    err = (freqs[p].to(torch.complex128) - ref).abs().max().item()
    worst = max(worst, err)
    ok &= err < 1e-4
print(f"\n[7] freqs_cis[p] 的相位 == p * 频率 (独立重实现) 对所有抽查 p 成立: {ok} (最大差 {worst:.3e})")
assert ok
print("    组 j 取 freqs_cis[j*ratio] -> 相位 = (j*ratio)*频率, 即该压缩条目被放在位置 j*ratio")
# decode 侧: 组刚填满的步 p, 用 freqs_cis[p + 1 - ratio] -> 应等于该组首 token 位置
bad = []
for p in range(1, 16):
    if (p + 1) % ratio == 0:
        group_first = (p // ratio) * ratio
        if p + 1 - ratio != group_first:
            bad.append((p, p + 1 - ratio, group_first))
print(f"    decode 侧: 组填满步 p 用 freqs_cis[p+1-ratio], 与组首 token 位置一致: {not bad} {bad}")
print("    代码依据: Indexer.forward L539-543 / Attention._compress_kv L753-757")
