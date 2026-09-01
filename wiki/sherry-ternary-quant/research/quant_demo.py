import random

random.seed(20260901)

# 构造示例：固定种子的伪随机权重与伪随机重要性，
# 模拟一个 256 权重块（64 组，每组 4 个）及其 imatrix
weights = [random.gauss(0.0, 0.5) for _ in range(256)]
imatrix = [random.uniform(0.2, 2.0) for _ in range(256)]
GROUPS = [weights[i:i + 4] for i in range(0, 256, 4)]
IM_GROUPS = [imatrix[i:i + 4] for i in range(0, 256, 4)]


def quantize(groups, im_groups, scale_mode, zero_mode, rounds=1):
    """对 64 个四权重组做 3:4 稀疏三值量化，返回 (d, qs, 加权SSD)。"""
    d = max(abs(x) for g in groups for x in g)  # 初值 amax
    qs = None
    for _ in range(rounds):
        qs = []
        for g, w in zip(groups, im_groups):
            if zero_mode == "argmin":
                z = min(range(4), key=lambda j: abs(g[j]))
            else:  # imatrix 增量代价：置零 w*(x^2-(|x|-d)^2) 最小者
                z = min(range(4),
                        key=lambda j: w[j] * (g[j] ** 2 - (abs(g[j]) - d) ** 2))
            qs.append([0 if j == z else (1 if g[j] > 0 else -1) for j in range(4)])
        if scale_mode == "amax":
            d = max(abs(x) for g in groups for x in g)
        else:  # 加权最小二乘：d = Σ w·x·q / Σ w·q^2，仅对非零 lane 求和
            num = sum(wi * x * q
                      for g, w, qs_g in zip(groups, im_groups, qs)
                      for wi, x, q in zip(w, g, qs_g))
            den = sum(wi * q * q
                      for w, qs_g in zip(im_groups, qs)
                      for wi, q in zip(w, qs_g))
            d = num / den
    wssd = sum(wi * (x - d * q) ** 2
               for g, w, qs_g in zip(groups, im_groups, qs)
               for wi, x, q in zip(w, g, qs_g))
    return d, qs, wssd


for name, scale_mode, zero_mode, rounds in [
    ("amax + argmin 零位（上游量化器）", "amax", "argmin", 1),
    ("WLS + argmin 零位", "wls", "argmin", 1),
    ("WLS + imatrix 零位，交替 3 轮（AngelSlim）", "wls", "incremental", 3),
]:
    d, qs, wssd = quantize(GROUPS, IM_GROUPS, scale_mode, zero_mode, rounds)
    print(f"{name}: d={d:.4f} 加权SSD={wssd:.4f}")

# 均匀重要性下两种零位规则是否一致
d0 = 0.6
same = all(
    min(range(4), key=lambda j: abs(g[j]))
    == min(range(4), key=lambda j: g[j] ** 2 - (abs(g[j]) - d0) ** 2)
    for g in GROUPS
)
print(f"\n重要性全 1 时 argmin 与增量代价零位一致: {same}"
      "（代价 2|x|d-d^2 随 |x| 递增）")

# 贯穿示例：一组 4 个权重 (0.6, -0.1, 0.5, -0.7)，d=0.6
x = [0.6, -0.1, 0.5, -0.7]
d = 0.6
z = min(range(4), key=lambda j: abs(x[j]))
q = [0 if j == z else (1 if x[j] > 0 else -1) for j in range(4)]
deq = [d * v for v in q]
err = [abs(a - b) for a, b in zip(x, deq)]
print(f"\n贯穿示例: x={x} -> q={q} (零在第 {z + 1} 位)")
print(f"反量化={[round(v, 4) for v in deq]} "
      f"逐元素误差={[round(e, 4) for e in err]} "
      f"SSD={sum(e * e for e in err):.4f}")
d_amax = max(abs(v) for v in x)
ssd_a = sum((a - d_amax * b) ** 2 for a, b in zip(x, q))
print(f"同一组用 amax d={d_amax}: SSD={ssd_a:.4f}")

# 编码空间与字节账
print(f"\n4 权重组合数: C(4,3)*2^3 = {4 * 8} = 2^5（5 bit 恰好装下）")
index_bytes = 64 * 4 // 8
sign_bytes = 64 // 8
total = index_bytes + sign_bytes + 2
print(f"256 权重: 索引 {index_bytes}B + 符号 {sign_bytes}B + fp16 scale 2B = {total}B")
print(f"bpw = {total}*8/256 = {total * 8 / 256}")
