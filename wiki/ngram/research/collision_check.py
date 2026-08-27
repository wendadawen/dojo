"""碰撞率口径实测：16 个头（真实质数/偏移/乘子）× 2000/4000 随机位置。

口径：每头独立统计，重复行号计一次碰撞；数字以 (T-unique)/T 口径给出最差单头。
乘子/质数/偏移均为 checkpoint 真实值（见数据流页 probe7）。
"""
import random
MULTS = [23703573157769, 20109073645365, 8052911324071]
P = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
     20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]
OFF = []
acc = 0
for x in P:
    OFF.append(acc); acc += x
random.seed(0)
V = 248320
for T in (2000, 4000):
    tot_coll, worst = 0, 0
    for h in range(16):
        seen, coll = set(), 0
        ngram = 2 if h < 8 else 3
        for _ in range(T):
            ids = [random.randrange(V) for _ in range(ngram)]
            m = ids[0] * MULTS[0]
            for p in range(1, ngram):
                m ^= ids[p] * MULTS[p]
            row = m % P[h] + OFF[h]
            if row in seen:
                coll += 1
            seen.add(row)
        tot_coll += coll
        worst = max(worst, coll)
    print(f"T={T}: 16 头合计碰撞 {tot_coll} 次 / {16*T} 个查询，最差单头 {worst} 次（{worst/T*100:.3f}%）")
