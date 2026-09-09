#!/usr/bin/env python3
"""按独立审查报告修复 3 处 P1 + 4 处 P2，并清理预览服务注入的 data-page-node-id 属性。"""
import re, sys

P = '/Users/wendadawen/code/dojo/wiki/hy4-preview-dataflow/index.html'
s = open(P, encoding='utf-8').read()
orig_len = len(s)

def rep(old, new, n=1, tag=''):
    global s
    c = s.count(old)
    assert c == n, f'{tag}: expected {n}, got {c}: {old[:60]!r}'
    s = s.replace(old, new)
    print(f'OK  {tag} ({n}x)')

# --- 0. 清理预览注入属性 ---
n_attr = len(re.findall(r' data-page-node-id="[^"]*"', s))
s = re.sub(r' data-page-node-id="[^"]*"', '', s)
print(f'OK  清理 data-page-node-id x{n_attr}')
assert 'data-page-node-id' not in s

# --- P1-1: vLLM 路径（先长后短） ---
rep('vllm/model_executor/models/hy_v4/attention.py', 'vllm/models/hy_v4/nvidia/attention.py', 1, 'P1-1a')
rep('vllm/model_executor/models/hy_v4/', 'vllm/models/hy_v4/', 1, 'P1-1b')

# --- P1-2: finetune 补丁路径 ---
rep('finetune_hy_v4_patches.py', 'finetune/llama_factory_support/hy_v4_patches.py', 1, 'P1-2')

# --- P1-3: config 键名 ---
rep('rope_theta</code>', 'rope_parameters.rope_theta</code>', 1, 'P1-3')

# --- P2-4: 单位口径统一（2^20 token + 二进制单位） ---
rep('4.88 MB', '4.88 MiB', 2, 'P2-4 MB')
rep('4.88 TB', '4.88 TiB', 3, 'P2-4 TB')
rep('87.8 KB', '87.8 KiB', 2, 'P2-4 KB')
rep('87.8 GB', '87.8 GiB', 4, 'P2-4 GB-main')
rep('2.8–5.4 GB', '2.7–5.3 GiB', 2, 'P2-4 idx-range')
rep('2.8 GB', '2.7 GiB', 2, 'P2-4 idx-single')
rep('90.6 GB', '90.5 GiB', 1, 'P2-4 sum-d')
rep(r'90.6\\ \\mathrm{GB}', r'90.5\\ \\mathrm{GiB}', 1, 'P2-4 sum-f')
rep('（主 cache，按真实 config 实算）：',
    '（主 cache，按真实 config 实算；容量均按 1M = $2^{20}$ token 与二进制单位 KiB/MiB/GiB/TiB 换算）：',
    1, 'P2-4 note')

# --- P2-5: permute 函数文件位置 ---
rep('另有 <code>permute_hyv4_indexer_weight</code> 把索引器',
    '另有 <code>python/sglang/srt/models/hunyuan_v4.py</code> 的 <code>permute_hyv4_indexer_weight</code> 把索引器',
    1, 'P2-5')

# --- P2-6: tooltip 数学补 $...$（JS 字符串内反斜杠双写） ---
rep(r'此后整个主干残差宽度为 4×6144=24576。',
    r'此后整个主干残差宽度为 $4\\times 6144=24576$。', 1, 'P2-6a')
rep(r'初始 -log3）、hc_head_scale [1]',
    r'初始 $-\\log 3$）、hc_head_scale [1]', 1, 'P2-6b')
rep(r'初始门 ≈ 0.25（sigmoid(-log3)）。等比模型实测 mean 0.2500',
    r'初始门 $\\approx 0.25$（$\\sigma(-\\log 3)$）。等比模型实测 mean $0.2500$', 1, 'P2-6c')
rep(r'初始 post 门 = 2σ(0) = 1.0（恒等残差），实测 mean 0.9999',
    r'初始 post 门 $= 2\\sigma(0) = 1.0$（恒等残差），实测 mean $0.9999$', 1, 'P2-6d')
rep(r'初始 b = -log3、s = 0.01，初始门 ≈ 0.25。实测 mean 0.2500',
    r'初始 $b = -\\log 3$、$s = 0.01$，初始门 $\\approx 0.25$。实测 mean $0.2500$', 1, 'P2-6e')
rep(r'各乘 ≈0.25 的门', r'各乘 $\\approx 0.25$ 的门', 1, 'P2-6f')
rep(r'初始门同样 ≈0.25', r'初始门同样 $\\approx 0.25$', 1, 'P2-6g')
rep(r'单元实测 silu(10)×(-7) = -69.9968',
    r'单元实测 $\\mathrm{silu}(10)\\times(-7) = -69.9968$', 1, 'P2-6h')

# --- P2-7: 删除 research/ 文件清单段 ---
m = re.search(r'\n  <p>实测脚本与原始输出存于 <code>research/</code>.*?</p>\n', s)
assert m, 'P2-7 paragraph not found'
s = s[:m.start()] + '\n' + s[m.end():]
print('OK  P2-7 删除清单段')

open(P, 'w', encoding='utf-8').write(s)
s2 = open(P, encoding='utf-8').read()

# --- 反向验证 ---
checks = [
    ('vllm/models/hy_v4/nvidia/attention.py', 1), ('model_executor/models/hy_v4', 0),
    ('finetune/llama_factory_support/hy_v4_patches.py', 1), ('finetune_hy_v4_patches', 0),
    ('rope_parameters.rope_theta', 1), ('data-page-node-id', 0),
    ('4.88 MiB', 2), ('4.88 TiB', 3), ('87.8 KiB', 2), ('87.8 GiB', 4),
    ('2.7–5.3 GiB', 2), ('90.5 GiB', 2), ('2^{20}', 1),
    ('python/sglang/srt/models/hunyuan_v4.py', 1),
    ('$-\\log 3$', 2), ('$2\\sigma(0) = 1.0$', 1), ('$\\mathrm{silu}(10)\\times(-7)', 1),
    ('实测脚本与原始输出', 0),
    # 残留裸数学应为 0
]
bad = []
for kw, want in checks:
    got = s2.count(kw)
    if got != want:
        bad.append(f'{kw!r}: want {want}, got {got}')
    else:
        print(f'VERIFIED {kw!r} = {got}')
# 残留裸 GB/KB 检查（B/token 的 B 字节单位除外）
for m2 in re.finditer(r'\d (GB|MB|KB|TB)\b', s2):
    ctx = s2[max(0, m2.start()-30):m2.end()]
    bad.append(f'residual unit: {ctx!r}')
if bad:
    print('FAILURES:'); [print(' ', b) for b in bad]; sys.exit(1)
print(f'*** ALL FIXES APPLIED (len {orig_len} -> {len(s2)}) ***')
