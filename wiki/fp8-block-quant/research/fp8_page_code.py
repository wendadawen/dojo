# fp8-block-quant 页的两个可运行代码块
import torch
import math

# ============ 代码 1：E4M3 的范围与位解码（第 1 章） ============
fi = torch.finfo(torch.float8_e4m3fn)
print("torch float8_e4m3fn: max =", fi.max, ", min =", fi.min,
      ", tiny(最小正正规数) =", f"{fi.tiny:.3e}")
print("2^-6 =", f"{2.0 ** -6:.3e}", "  2^-9 =", f"{2.0 ** -9:.3e}")
print("1.75 * 2^8 =", 1.75 * 2 ** 8)

bits = 0b01111110            # 0.1111.110：符号 0、指数 1111、尾数 110
s = bits >> 7
e = (bits >> 3) & 0b1111
m = bits & 0b111
val = (-1) ** s * 2 ** (e - 7) * (1 + m / 8)
print("位模式 0.1111.110: 指数域 =", e, "尾数 =", m, "/8")
print("值 = (-1)^" + str(s), "* 2^(%d-7)" % e, "* (1 + %d/8)" % m, "=", val)

# ============ 代码 2：块量化 roundtrip 与 scale 形状（第 4 章） ============
W = torch.tensor([[1.0, -6.0, 0.01, 2.0],
                  [0.5, 0.02, -0.008, 4.0],
                  [3.0, -0.001, 0.03, -1.0],
                  [0.007, 5.0, 0.02, 0.5]])
max_abs = W.abs().amax()
scale = fi.max / max_abs
Wq = torch.clamp(W * scale, min=fi.min, max=fi.max).to(torch.float8_e4m3fn)
W_back = Wq.float() / scale
err = (W_back - W).abs()
print("块内 max_abs =", float(max_abs), " scale = 448/max =", round(float(scale), 4))
print("反量化绝对误差: max =", round(float(err.max()), 6),
      " mean =", round(float(err.mean()), 6))
print("最大相对误差:", f"{float((err / W.abs()).max()) * 100:.3f}%")

for shape in [(2048, 4096), (16384, 1536)]:
    rows, cols = shape
    print(f"权重 {shape} 的 scale 形状: "
          f"({math.ceil(rows / 128)}, {math.ceil(cols / 128)})")
