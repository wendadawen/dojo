"""迷你 iHC 前向：4 条流、d=2、两个子块、一个 head。

复算页面贯穿示例的第一幕（初始化状态）与第二幕（训练后的门）。
子层用简单线性映射代替——注意力 / MLP 内部不是本页主题。
构造示例：权重与门值为教学构造，非 Hy4 checkpoint 数值。
"""
import numpy as np

n, d = 4, 2                                # 4 条流，每条 2 维
m, eps_hc, eps_rms = 2.0, 1e-6, 1e-5       # 幅度、门上的 epsilon、RMS 的 epsilon

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def rms_norm(x, eps=eps_rms):              # 逐 token、最后一维
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)

def pre_block(x, W, scale, base):
    """读门合并 + 顺手算出写门。x: [n, d] -> y: [d]"""
    x_flat = x.reshape(-1)                              # [n*d]
    r = 1.0 / np.sqrt(np.mean(x_flat**2) + eps_rms)
    mixes = (W @ x_flat) * r                            # [2n]
    H_pre = sigmoid(mixes[:n] * scale[0] + base[:n]) + eps_hc
    H_post = m * sigmoid(mixes[n:] * scale[1] + base[n:]) + eps_hc
    y = np.sum(H_pre[:, None] * x, axis=0)              # [d]
    return y, H_pre, H_post

def post_block(x, z, H_post):
    """写门回加：每条流加同一份子层输出的不同倍数。"""
    return H_post[:, None] * z + x                      # [n, d]

def head_block(x, W_h, scale_h, base_h):
    """末端门控合并。"""
    x_flat = x.reshape(-1)
    r = 1.0 / np.sqrt(np.mean(x_flat**2) + eps_rms)
    mixes = (W_h @ x_flat) * r                          # [n]
    H_head = sigmoid(mixes * scale_h[0] + base_h) + eps_hc
    return np.sum(H_head[:, None] * x, axis=0), H_head

# 子层（线性映射代替注意力 / MLP）
W_attn = np.array([[0.6, -0.4], [0.3, 0.8]])
W_mlp = np.array([[0.5, 0.2], [-0.7, 0.4]])

# 初始化参数：scale=0.01，读偏置 -ln(n-1)，写偏置 0
scale = np.array([0.01, 0.01])
base = np.concatenate([-np.log(n - 1) * np.ones(n), np.zeros(n)])
scale_h = np.array([0.01])
base_h = -np.log(n - 1) * np.ones(n)

# ===== 第一幕：初始化状态（hc_fn 置零，门精确停在偏置值） =====
W0 = np.zeros((2 * n, n * d))
W_h0 = np.zeros((n, n * d))
x0 = np.tile(np.array([1.0, -0.5]), (n, 1))    # 入口复制 4 份
x = x0.copy()
print("入口 4 条流:")
print(x)
for name, W_sub in [("子块 1（注意力位置）", W_attn), ("子块 2（MLP 位置）", W_mlp)]:
    y, H_pre, H_post = pre_block(x, W0, scale, base)
    ny = rms_norm(y)
    z = W_sub @ ny
    x = post_block(x, z, H_post)
    print()
    print(name)
    print("  读门", np.round(H_pre, 6), " 写门", np.round(H_post, 6), " 读门之和 %.6f" % H_pre.sum())
    print("  合并输入 y =", np.round(y, 6), " 归一化输入 =", np.round(ny, 6))
    print("  子层输出 z =", np.round(z, 6))
    print("  出口 4 条流（流间最大差 %.1e）:" % np.max(np.ptp(x, axis=0)))
    print(np.round(x, 6))
y_h, H_head = head_block(x, W_h0, scale_h, base_h)
print()
print("head 读门", np.round(H_head, 6), " 合并输出", np.round(y_h, 6))

# 对照：标准 Pre-Norm 残差走同样的两个子层
h = x0[0].copy()
for W_sub in (W_attn, W_mlp):
    h = W_sub @ rms_norm(h) + h
print()
print("标准 Pre-Norm 残差结果:", np.round(h, 6))
print("两者之差的最大分量: %.2e" % np.max(np.abs(y_h - h)))

# ===== 第二幕：训练后的门（构造值） =====
H_pre2 = np.array([0.55, 0.20, 0.10, 0.15])    # 读门偏科
H_post2 = np.array([1.8, 0.6, 1.2, 0.3])       # 写门强弱不一
x2 = np.array([[1.00, -0.50],
               [0.90, -0.55],
               [1.10, -0.45],
               [1.05, -0.48]])                 # 已轻微分化的 4 条流
y2 = np.sum(H_pre2[:, None] * x2, axis=0)
z2 = W_attn @ rms_norm(y2)
x2_new = post_block(x2, z2, H_post2)
print()
print("第二幕：读门", H_pre2, " 写门", H_post2)
print("合并输入 y2 =", np.round(y2, 6), "（读门偏向流 1，y2 最接近流 1 的值）")
print("子层输出 z2 =", np.round(z2, 6))
print("子块前 4 条流:")
print(x2)
print("子块后 4 条流:")
print(np.round(x2_new, 6))
print("流间每维极差（前）:", np.round(np.ptp(x2, axis=0), 4))
print("流间每维极差（后）:", np.round(np.ptp(x2_new, axis=0), 4))
