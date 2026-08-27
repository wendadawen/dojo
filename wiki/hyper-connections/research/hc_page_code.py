# hyper-connections 页的三个可运行代码块（独立运行，输出进页面）
import torch

torch.manual_seed(0)

# ============ 代码 1：n=2 的 HC 层与 n=1 退化（第 2 章） ============
def hc_layer(H, A_m, A_r, B, T):
    """H: [n,d], A_m: [n], A_r: [n,n], B: [n], T: 子层(函数)"""
    h0 = A_m @ H                       # pre：加权求和成单路
    out = T(h0)                        # 子层
    return A_r @ H + torch.outer(B, out)   # res 混合 + post 写回

d = 4
x = torch.arange(1, d + 1, dtype=torch.float32)
T = lambda h: h * 0.1                  # 构造子层：缩放 0.1

# n=1：所有映射退化成标量 1 → 恒等残差
H1 = x.unsqueeze(0)
h_n1 = hc_layer(H1, torch.ones(1), torch.ones(1, 1), torch.ones(1), T)
plain = x + T(x)
print("n=1 输出:", [round(v, 6) for v in h_n1[0].tolist()])
print("普通残差:", [round(v, 6) for v in plain.tolist()])
print("最大差:", float((h_n1[0] - plain).abs().max()))

# n=2：入口复制两份，随机映射
H2 = x.unsqueeze(0).repeat(2, 1)
A_m = torch.tensor([0.7, 0.3])
A_r = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
B = torch.tensor([1.0, 0.5])
h_n2 = hc_layer(H2, A_m, A_r, B, T)
print("n=2 输出行 0:", [round(v, 6) for v in h_n2[0].tolist()])
print("n=2 输出行 1:", [round(v, 6) for v in h_n2[1].tolist()])
print("两行是否已不同:", not torch.allclose(h_n2[0], h_n2[1]))

# ============ 代码 2：无约束链 vs 双随机链（第 3 章） ============
def sinkhorn(M, iters=20):
    for _ in range(iters):
        M = M / M.sum(dim=0, keepdim=True)   # 列归一化
        M = M / M.sum(dim=1, keepdim=True)   # 行归一化
    return M

G = torch.eye(4)
for _ in range(24):
    G = G @ torch.randn(4, 4)
print("无约束链 24 层, 复合矩阵行和绝对值最大:", f"{float(G.abs().sum(1).max()):.3e}")

G2 = torch.eye(4)
for _ in range(24):
    X = torch.rand(4, 4) + 0.1
    G2 = G2 @ sinkhorn(X)
print("双随机链 24 层, 复合矩阵行和:", [round(v, 6) for v in G2.sum(1).tolist()])

# ============ 代码 3：Sinkhorn 收敛（第 4 章） ============
M0 = torch.tensor([[3.0, 1.0, 0.5],
                   [0.2, 4.0, 1.0],
                   [1.0, 0.3, 5.0]])
M = M0.clone()
for t in range(1, 6):
    M = M / M.sum(dim=0, keepdim=True)
    M = M / M.sum(dim=1, keepdim=True)
    print(f"迭代 {t}: 行和 {[round(v, 6) for v in M.sum(1).tolist()]}, "
          f"列和 {[round(v, 6) for v in M.sum(0).tolist()]}")
for t in range(6, 21):
    M = M / M.sum(dim=0, keepdim=True)
    M = M / M.sum(dim=1, keepdim=True)
print(f"迭代 20: 行和最大偏差 {float((M.sum(1) - 1).abs().max()):.1e}, "
      f"列和最大偏差 {float((M.sum(0) - 1).abs().max()):.1e}")
print("最大奇异值:", round(float(torch.linalg.svdvals(M)[0]), 6))
