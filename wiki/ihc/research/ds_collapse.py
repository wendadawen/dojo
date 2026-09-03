"""双随机矩阵连乘坍缩：构造实验。

模拟 mHC 的做法——每个"层"产生一个正矩阵，经 Sinkhorn 投影到
（近似）双随机矩阵，然后把各层的矩阵连乘，观察深层累积混合矩阵
的行是否趋于均匀向量。构造示例：矩阵为随机生成，非训练产物。
"""
import numpy as np

rng = np.random.default_rng(0)
n, L, steps = 4, 12, 20

def sinkhorn(A, steps):
    """交替行、列归一化，把正矩阵投影到近似双随机矩阵。"""
    for _ in range(steps):
        A = A / A.sum(axis=1, keepdims=True)   # 行和归一
        A = A / A.sum(axis=0, keepdims=True)   # 列和归一
    return A

# 每层一个独立的随机正矩阵，投影到双随机
mats = [sinkhorn(rng.uniform(0.2, 1.0, size=(n, n)), steps) for _ in range(L)]

print("第 1 层混合矩阵（Sinkhorn 20 步后）:")
print(np.round(mats[0], 4))
print("行和:", np.round(mats[0].sum(axis=1), 4))
print("列和:", np.round(mats[0].sum(axis=0), 4))
print("最小元素: %.4f" % mats[0].min())

# 深层累积混合矩阵 = 逐层矩阵连乘
P = np.eye(n)
print()
print("层  0   第 1 行 =", np.round(P[0], 4), " max|P-1/n| = %.4f" % np.max(np.abs(P - 1.0 / n)))
for l, M in enumerate(mats, 1):
    P = M @ P
    print("层 %2d   第 1 行 = %s   max|P-1/n| = %.4f"
          % (l, np.round(P[0], 4), np.max(np.abs(P - 1.0 / n))))
