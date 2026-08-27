# rmsnorm 页的两个可运行代码块
import torch

# ============ 代码 1：LayerNorm vs RMSNorm（第 2 章） ============
def layernorm(a):
    mu = a.mean()
    sigma = ((a - mu) ** 2).mean().sqrt()
    return (a - mu) / sigma

def rmsnorm(a):
    rms = (a ** 2).mean().sqrt()
    return a / rms

a = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("输入:", a.tolist())
print("LayerNorm: mu =", round(float(a.mean()), 4),
      ", sigma =", round(float(((a - a.mean()) ** 2).mean().sqrt()), 4))
print("  输出:", [round(v, 6) for v in layernorm(a).tolist()])
print("RMSNorm:   RMS =", round(float((a ** 2).mean().sqrt()), 4))
print("  输出:", [round(v, 6) for v in rmsnorm(a).tolist()])

a0 = torch.tensor([1.0, -1.0, 3.0, -3.0])
print("零均值输入:", a0.tolist())
print("  两者最大差:", float((layernorm(a0) - rmsnorm(a0)).abs().max()))

# ============ 代码 2：与官方实现对齐 + GLM 式实现（第 4 章） ============
ref = torch.nn.RMSNorm(4, eps=0.0)
with torch.no_grad():
    ref.weight.fill_(1.0)
print("torch.nn.RMSNorm 输出:", [round(v, 6) for v in ref(a).tolist()])
print("与手写 RMSNorm 最大差:", float((ref(a) - rmsnorm(a)).abs().max()))

def glm_rmsnorm(x, weight, eps):
    x32 = x.to(torch.float32)                      # fp32 里算
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)             # 分母加 eps
    return weight * x32.to(x.dtype)                # 权重乘在转回后

w = torch.ones(4)
print("GLM 式（fp32 + eps=1e-5）输出:", [round(v, 6) for v in glm_rmsnorm(a, w, 1e-5).tolist()])
print("与论文公式（无 eps）差:", float((glm_rmsnorm(a, w, 1e-5) - rmsnorm(a)).abs().max()))
