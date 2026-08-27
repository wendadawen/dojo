# depthwise-conv 页的两个可运行代码块
import torch

# ============ 代码 1：成本计数与比值（第 3 章） ============
Dk, M, N, Df = 3, 64, 64, 16
std_cost = Dk * Dk * M * N * Df * Df
dw_cost = Dk * Dk * M * Df * Df
pw_cost = M * N * Df * Df
sep_cost = dw_cost + pw_cost
print("标准卷积乘加:", f"{std_cost:,}")
print("depthwise 乘加:", f"{dw_cost:,}")
print("pointwise 乘加:", f"{pw_cost:,}")
print("深度可分离合计:", f"{sep_cost:,}")
print("比值 sep/std =", f"{sep_cost / std_cost:.6f}",
      "  公式 1/N + 1/D_K^2 =", f"{1 / N + 1 / Dk ** 2:.6f}")
print("缩减倍数:", f"{std_cost / sep_cost:.1f}x")

# ============ 代码 2：因果 1D depthwise 卷积（第 4 章） ============
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).view(1, 1, -1)
k = torch.tensor([1.0, 0.5, -0.5, -1.0]).view(1, 1, 4)
conv = torch.nn.Conv1d(1, 1, kernel_size=4, groups=1, bias=False, padding=3)
with torch.no_grad():
    conv.weight.copy_(k)
out = conv(x)[:, :, 3:8]
print("输入:", x.flatten().tolist())
print("核:", k.flatten().tolist())
print("手算 t=3: 1*1 + 2*0.5 + 3*(-0.5) + 4*(-1) =", 1 * 1 + 2 * 0.5 + 3 * (-0.5) + 4 * (-1))
print("torch 输出:", [round(v, 4) for v in out.flatten().tolist()])

x2 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]).view(1, 2, 4)
dw = torch.nn.Conv1d(2, 2, kernel_size=2, groups=2, bias=False, padding=1)
with torch.no_grad():
    dw.weight.copy_(torch.tensor([[[1.0, 1.0]], [[0.0, 1.0]]]))
o2 = dw(x2)[:, :, 1:5]
print("groups=2 通道 0（核 [1,1]）:", [round(v, 1) for v in o2[0, 0].tolist()])
print("groups=2 通道 1（核 [0,1]）:", [round(v, 1) for v in o2[0, 1].tolist()])
