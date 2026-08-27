"""MRoPE 概念页实测：三维位置 id 生成 + 交错槽位归属 + 推进量。

验证目标：
  1. 文本段三维 id 退化（三值相同）
  2. 视觉段 (t,h,w) 分配规则（单帧图 t 恒定、h/w 网格遍历）
  3. 跨模态衔接与推进量：图像块后位置轴只推进 max(h,w)/merge
  4. 交错排布的槽位归属与 mrope_section 逐项一致
对应来源：transformers@36deb0b5 qwen4_exp L140-155（apply_interleaved_mrope）、
L1980-2030（get_vision_position_ids）、L2032-2123（get_rope_index）。
"""

# ---- 第 1 部分：三维位置 id（构造序列：8 文本 + (1,28,28) 图 + 5 文本） ----
S = 8           # 前置文本长度
GH, GW, MERGE = 28, 28, 2
llm_h, llm_w = GH // MERGE, GW // MERGE

def text_pos(start, n):
    return [(start + i,) * 3 for i in range(n)]

def vision_pos(start, gh, gw, merge=MERGE):
    lh, lw = gh // merge, gw // merge
    out = []
    for r in range(lh):
        for c in range(lw):
            out.append((start, start + r, start + c))
    return out

vis = vision_pos(S, GH, GW)
advance = max(GH, GW) // MERGE
tail = text_pos(S + advance, 5)

print("=== A. 构造序列的三维位置 id ===")
print(f"序列 = 8 个文本 token + 一张 (1,{GH},{GW}) 图（{len(vis)} 个视觉 token）+ 5 个文本 token")
print(f"文本段 1（位置 {S-8}..{S-1}）示例: {text_pos(0, 8)[:2]} ...（三个分量相同）")
print(f"视觉段前 3 个: {vis[:3]}")
print(f"视觉段最后 1 个: {vis[-1]}")
print(f"视觉段分量取值: t 恒为 {vis[0][0]}，h ∈ [{vis[0][1]}, {vis[-1][1]}]，w ∈ [{vis[0][2]}, {vis[-1][2]}]")
print(f"视觉 token 数 = {len(vis)}，但位置推进量 = max({GH},{GW})/{MERGE} = {advance}")
print(f"文本段 2 从位置 {S + advance} 开始（而非 {S + len(vis)}）: 首个 {tail[0]}")

print()
print("=== B. 推进量对照 ===")
print(f"  {'图像 grid':>14s} {'视觉 token':>10s} {'推进量':>8s} {'比值':>7s}")
for gh, gw in [(28, 28), (56, 56), (84, 84), (66, 120)]:
    n = (gh // MERGE) * (gw // MERGE)
    adv = max(gh, gw) // MERGE
    print(f"  {f'(1,{gh},{gw})':>14s} {n:>10,d} {adv:>8d} {n/adv:>6.1f}x")

# ---- 第 2 部分：交错槽位归属 ----
print()
print("=== C. 交错排布：32 个频率槽位的归属 ===")
SECTION = [11, 11, 10]
ROT_HALF = 32          # rotary_dim 64 / 2
layout = ["T"] * ROT_HALF
h_slots = list(range(1, min(SECTION[1] * 3, ROT_HALF), 3))
w_slots = list(range(2, min(SECTION[2] * 3, ROT_HALF), 3))
for i in h_slots:
    layout[i] = "H"
for i in w_slots:
    layout[i] = "W"
print("  槽位归属（T/H/W）:")
for r in range(2):
    print(f"    [{r*16:2d}:{(r+1)*16:2d}] " + " ".join(layout[r*16:(r+1)*16]))
from collections import Counter
cnt = Counter(layout)
print(f"  各分量占用: T={cnt['T']}, H={cnt['H']}, W={cnt['W']}")
print(f"  与 mrope_section {SECTION} 逐项一致: {[cnt['T'],cnt['H'],cnt['W']] == SECTION}")
print(f"  H 槽位 = slice(1, {SECTION[1]}*3, 3) -> {h_slots}")
print(f"  W 槽位 = slice(2, {SECTION[2]}*3, 3) -> {w_slots}")
print(f"  其余归 T -> {[i for i in range(ROT_HALF) if layout[i]=='T']}")

# ---- 第 3 部分：分段排布对照（qwen2_vl，用其真实参数） ----
print()
print("=== D. 分段排布对照（qwen2_vl：head_dim=128，mrope_section=[16,24,24]） ===")
Q2_SECTION = [16, 24, 24]              # 频率配额，和 = 64 = 128/2
dims = [s * 2 for s in Q2_SECTION]     # 维度配额 [32,48,48]，在 128 维 cos/sin 上切段
print(f"  频率槽位共 {sum(Q2_SECTION)} 个（head_dim 128 / 2）")
start = 0
for name, s in zip("THW", Q2_SECTION):
    print(f"  {name} 独占频率槽位 [{start}, {start + s - 1}]（cos/sin 维度 [{start*2}, {start*2 + s*2 - 1}]）")
    start += s
print(f"  源码：mrope_section*2 = {dims}，cos.split({dims}) 后按 i%3 取段")
print(f"  交错排布（qwen4_exp）则以 3 为步长穿插：[TTT...HHH...WWW] -> [THWTHW...]（docstring 原文）")

# ---- 第 4 部分：视频的 t 递增 ----
print()
print("=== E. 视频：t 逐帧递增 ===")
frames = [vision_pos(S, GH, GW) for _ in range(3)]
# 实际源码中同一视频各帧共享 t 间隔 1：第 k 帧 t = start + k
for k in range(3):
    print(f"  第 {k} 帧首个 token: t={S + k}, h={S}, w={S}")
print(f"  3 帧视频的 t 分量取值: {[S + k for k in range(3)]}（逐帧 +1），h/w 每帧重新从 {S} 遍历")
