"""n-gram 概念页实测：I am Sam 语料的 bigram 统计 + 哈希行号。

验证目标：
  1. SLP3 §3.1.2 的 I am Sam 语料上，bigram MLE 概率与教材列出的数值逐项一致
  2. 整句概率 = 条件概率连乘
  3. 同一语料换成 token id 后，Qwen 式哈希把 bigram/trigram 映射到行号
  4. 表容量与碰撞率的关系（用真实量级的奇数乘子，Python 大整数精确计算）
对应来源：SLP3 (2026-08-19 draft) ch.3；transformers@36deb0b5 L1018-1110。
"""
from collections import Counter

# ---- 第 1 部分：bigram 计数与概率（教材 §3.1.2 语料） ----
corpus = [
    "<s> I am Sam </s>",
    "<s> Sam I am </s>",
    "<s> I do not like green eggs and ham </s>",
]
bigram_counts = Counter()
unigram_counts = Counter()
for sent in corpus:
    toks = sent.split()
    for i in range(len(toks) - 1):
        bigram_counts[(toks[i], toks[i + 1])] += 1
    for t in toks:
        unigram_counts[t] += 1

def P(w_prev, w_next):
    """bigram MLE：P(w_next | w_prev) = C(w_prev w_next) / C(w_prev)"""
    return bigram_counts[(w_prev, w_next)] / unigram_counts[w_prev]

print("=== A. bigram MLE 概率（对照教材数值） ===")
for w_prev, w_next, expect in [
    ("<s>", "I", "2/3"), ("<s>", "Sam", "1/3"),
    ("I", "am", "2/3"), ("I", "do", "1/3"),
    ("am", "Sam", "1/2"), ("Sam", "</s>", "1/2"),
]:
    got = P(w_prev, w_next)
    print(f"  P({w_next} | {w_prev}) = {bigram_counts[(w_prev,w_next)]}/{unigram_counts[w_prev]}"
          f" = {got:.4f}   教材值 {expect}")

print()
print("=== B. 整句概率：P(<s> I am Sam </s>) ===")
sent = "<s> I am Sam </s>".split()
prob = 1.0
steps = []
for i in range(1, len(sent)):
    p = P(sent[i - 1], sent[i])
    steps.append(f"P({sent[i]}|{sent[i-1]})={p:.4f}")
    prob *= p
print("  " + " × ".join(steps))
print(f"  = {prob:.6f}")

print()
print("=== C. 稀疏检查：语料 11 个词型，理论 bigram 组合 121，实际出现 ===")
print(f"  实际出现的不同 bigram 数 = {len(bigram_counts)}")
print(f"  未出现组合占比 = {(11*11 - len(bigram_counts))/121*100:.1f}%")

# ---- 第 2 部分：哈希行号（Qwen 式，真实量级乘子） ----
# 乘子取 Qwen3.8-Flash-Next checkpoint 的真实值（数据流页 probe7 已核对）
MULTS = [23703573157769, 20109073645365, 8052911324071]
P_H = [20000003, 20000023]     # 真实表的前两个质数大小（构造示例取 2 个头）
OFFSETS = [0, 20000003]
tok = {"<s>": 1, "I": 2, "am": 3, "Sam": 4, "do": 5, "</s>": 6}   # 构造 id 映射
ids = [tok[t] for t in "<s> I am Sam </s>".split()]

def shift_right(seq, k):
    """右移 k 位，左侧以 <s> 的 id 填充；k=0 原样返回。"""
    return [1] * k + seq[:-k] if k else list(seq)

sh = [shift_right(ids, k) for k in range(3)]
print()
print("=== D. 同一语料按 Qwen 式哈希取行号（构造示例：id 为人为指定） ===")
for ngram in (2, 3):
    h = ngram - 2
    print(f"  {ngram}-gram（头 {h}，质数 {P_H[h]:,}）:")
    for t in range(2, len(ids)):
        mixed = sh[0][t] * MULTS[0]
        for p in range(1, ngram):
            mixed ^= sh[p][t] * MULTS[p]
        row = mixed % P_H[h] + OFFSETS[h]
        print(f"    位置 {t}（{'<s> I am Sam </s>'.split()[t]}）: 混合值 {mixed:,} -> 行号 {row:,}")

print()
print("=== E. 表容量与碰撞率（真实乘子 + 随机 bigram，Python 大整数） ===")
import random
random.seed(0)
V = 248320     # 与真实 vocab 同量级
for size, label in [(20000003, "单头质数表"), (211, "小表对照")]:
    seen, coll = set(), 0
    for _ in range(1000):
        a, b = random.randrange(V), random.randrange(V)
        row = ((a * MULTS[0]) ^ (b * MULTS[1])) % size
        if row in seen:
            coll += 1
        seen.add(row)
    print(f"  {label}（{size:,} 行）: 1000 个随机 bigram 碰撞 {coll} 次（{coll/10:.1f}%）")
