"""复算两个机制：压缩条目可达数、Top-K 选择语义。

对应概念页第 3.2 节（可达性公式 F3）与第 3.3 节（Top-K 选择语义）。
纯 Python，无第三方依赖。
"""


def reachable_formula(i, r):
    """F3 公式：floor((i+1)/r)"""
    return (i + 1) // r


def reachable_bruteforce(i, r):
    """按「整组出现」逐组枚举：组 j 覆盖 [jr, (j+1)r)，组尾 <= i 才算可达"""
    count = 0
    j = 0
    while (j + 1) * r - 1 <= i:
        count += 1
        j += 1
    return count


print("== 1. 可达条目数：公式 vs 枚举 ==")
bad = 0
for r in (1, 2, 4):
    for i in range(0, 200):
        if reachable_formula(i, r) != reachable_bruteforce(i, r):
            bad += 1
            print(f"  不符: i={i} r={r}")
print(f"  r in (1,2,4), i in 0..199 全部比对：不符 {bad} 处")

for (i, r) in ((4999, 2), (4999, 1), (4, 2)):
    n = reachable_formula(i, r)
    print(f"  i={i} r={r}: 可达 {n} 条（覆盖 token 0..{n * r - 1}）")

print()
print("== 2. Top-K 选择：不可达条目置 -inf ==")
NEG = float("-inf")
n_entries, r, i, K = 12, 2, 9, 4
n_reach = reachable_formula(i, r)
raw = [3.0, 1.0, 4.0, 1.5, 2.0, 0.5, 2.5, 3.5, 0.8, 1.2, 2.2, 0.9]

naive_order = sorted(range(n_entries), key=lambda j: (-raw[j], j))
masked = [raw[j] if j < n_reach else NEG for j in range(n_entries)]
masked_order = sorted(range(n_entries), key=lambda j: (-masked[j], j))

print(f"  槽位总数 {n_entries}，可达 {n_reach} 条，K={K}")
print(f"  全部槽位分数  : {raw}")
print(f"  未屏蔽时前 {K} 名: {sorted(naive_order[:K])}  <- 含未完成组的槽位")
print(f"  屏蔽后前 {K} 名  : {sorted(masked_order[:K])}")
print(f"  屏蔽后选中项全在可达范围内: "
      f"{all(j < n_reach for j in masked_order[:K])}")
