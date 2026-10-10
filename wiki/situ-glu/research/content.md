# SiTU-GLU

- 描述：SiTU-GLU 是前馈层的一种写法，把两个乘性因子各自换成一个平滑饱和的上界函数，让它们的乘积有固定上界。
- 摘要：$\mathrm{SiTU\text{-}GLU}(x)=\big[\beta_1\tanh(W_g x/\beta_1)\odot\sigma(W_g x)\big]\odot\big[\beta_2\tanh(W_u x/\beta_2)\big]$，把 SwiGLU 里两个无界的线性因子各自换成 $\beta\tanh(x/\beta)$，输出逐坐标落在 $[-\beta_1\beta_2,\beta_1\beta_2]$ 内。
- 主题：模型结构
- 标签：模型架构

## 资料

- Kimi Team. Kimi K3: Open Frontier Intelligence. arXiv:2607.24653v2, 2026. [https://arxiv.org/abs/2607.24653](https://arxiv.org/abs/2607.24653)

## 短页

### 两个因子都加上界的前馈激活

SiTU-GLU 把 SwiGLU 的两个乘性因子（门的线性因子和值支的投影）各换成一个平滑饱和的上界函数，让二者的乘积落进一段固定区间。SwiGLU 是前馈层（网络里逐个位置各算各的一层）的一种写法：中间一步是两条线性投影（输入向量乘权重矩阵得到的向量）的逐元素乘积，其中一条先过 Swish 当门。Swish 指 $\mathrm{Swish}(z)=z\sigma(z)$，$\sigma$ 是 sigmoid，$\sigma(z)=1/(1+e^{-z})$，把实数压到 $0$ 与 $1$ 之间。
依据：paper/k3-src/2-model-architecture.tex L253-L254；paper/k3-src/2-model-architecture.tex L262-L271

### SwiGLU 的两个无界因子

SwiGLU 的两个乘性因子都没有上界：门的线性因子 $W_g x$ 和值投影 $W_u x$ 都随输入增大而增大，其中 $W_g$、$W_u$ 是两个权重矩阵，$x$ 是输入向量。某个坐标上两支同时很大时，乘积可能出现一个激活外点（前馈层那个非线性函数的输出里，个别坐标的值异常大），在低精度算术下有溢出风险。
依据：paper/k3-src/2-model-architecture.tex L257

GLU 原本的 sigmoid 门不会有这种增大的问题，但它丢掉了 Swish 在正侧近似线性的响应。
依据：paper/k3-src/2-model-architecture.tex L258

### 用 softcap 分别饱和两个因子

softcap 是 $\mathrm{softcap}(z,\beta)=\beta\tanh(z/\beta)$，$\beta>0$ 是一个正的常数，$\tanh$ 把实数平滑压进区间 $(-1,1)$，乘 $\beta$ 后值域是 $(-\beta,\beta)$。SiTU-GLU 把它分别施加到两支上：

$$\mathrm{SiTU\text{-}GLU}(x)=\Big[\beta_1\tanh\!\Big(\frac{W_g x}{\beta_1}\Big)\odot\sigma(W_g x)\Big]\odot\Big[\beta_2\tanh\!\Big(\frac{W_u x}{\beta_2}\Big)\Big].$$
依据：paper/k3-src/2-model-architecture.tex L262-L271

$\odot$ 是逐元素乘积。门支只把 Swish 的线性因子换成 softcap、保留 $\sigma$ 因子，值支把线性的 $W_u x$ 换成 softcap。两支各被自己的 $\beta$ 限制在 $(-\beta,\beta)$ 内，输出逐坐标落在 $\pm\beta_1\beta_2$ 内；原点附近 softcap 近似恒等，输出仍近似于 SwiGLU。
依据：paper/k3-src/2-model-architecture.tex L262-L271；paper/k3-src/appendix.tex L15-L31

## 长页

前馈层是网络里逐个位置各算各的一层，中间夹一个逐元素作用的非线性函数。[SwiGLU](wiki/swiglu) 是前馈层的一种写法：用一道门逐元素缩放一条线性投影（输入向量乘权重矩阵得到的向量）。SiTU-GLU 在 [GLU](wiki/glu) 家族的这个位置上做一处改动，给两个乘性因子（门支的线性因子 $W_g x$、值支的投影 $W_u x$，$W_g$ 与 $W_u$ 是权重矩阵）各施加一个 softcap（一种把实数平滑压进有限区间的函数），让两支都饱和，乘积因此有固定上界，同时近原点保持 SwiGLU 的形状。
依据：paper/k3-src/2-model-architecture.tex L253-L271；paper/k3-src/appendix.tex L7-L13

### 为什么 SwiGLU 需要上界

[GLU](wiki/glu) 用一道门逐元素缩放一条线性投影，门由 sigmoid 算出：$\sigma(z)=1/(1+e^{-z})$，把实数压到 $0$ 与 $1$ 之间。SwiGLU 把这道门换成 Swish，$\mathrm{Swish}(z)=z\sigma(z)$，门的输出是 $W_g x\odot\sigma(W_g x)$（$\odot$ 是逐元素乘积）；$W_g$ 是门分支的权重矩阵，$x$ 是输入向量。
依据：paper/k3-src/2-model-architecture.tex L253-L254

这么写之后，$x$ 的坐标很大时，门的线性因子 $W_g x$ 会跟着变大，值投影 $W_u x$ 也如此（$W_u$ 是值分支的权重矩阵），两个乘性因子都无界。某个坐标上两支同时很大时，乘积可能出现一个激活外点（前馈层那个非线性函数的输出里，个别坐标的值异常大），在低精度算术下有溢出风险。
依据：paper/k3-src/2-model-architecture.tex L257

GLU 原本的 sigmoid 门不会无限增大，但它丢掉了 Swish 在正侧近似线性的响应：$W_g x$ 很大时 Swish 的门趋近 $W_g x$ 本身，而 sigmoid 门趋近 $1$。
依据：paper/k3-src/2-model-architecture.tex L258

Kimi K3 是提出 SiTU-GLU 的模型；它的路由分支（被选中少数专家的那条支路）把若干次矩阵相乘连成一串，叠上 2.8 万亿参数规模，内部激活会出现异常大的值。
依据：paper/k3-src/2-model-architecture.tex L218-L221

### SiTU-GLU 的公式

SiTU-GLU 用一个 softcap 函数：

$$\mathrm{softcap}(z,\beta)=\beta\tanh(z/\beta).$$
依据：paper/k3-src/2-model-architecture.tex L263

$\tanh$ 是把实数平滑压进区间 $(-1,1)$ 的函数，$\beta>0$ 是一个正的常数。乘 $\beta$ 之后值域是 $(-\beta,\beta)$：$z$ 远大于 $\beta$ 时输出趋近 $\beta$，$z$ 远小于 $-\beta$ 时趋近 $-\beta$。
依据：paper/k3-src/2-model-architecture.tex L263

SiTU-GLU 对 SwiGLU 的门分支和值分支各施加一次 softcap：

$$\mathrm{SiTU\text{-}GLU}(x)=\Big[\beta_1\tanh\!\Big(\frac{W_g x}{\beta_1}\Big)\odot\sigma(W_g x)\Big]\odot\Big[\beta_2\tanh\!\Big(\frac{W_u x}{\beta_2}\Big)\Big].$$
依据：paper/k3-src/2-model-architecture.tex L262-L267

门支 $\beta_1\tanh(W_g x/\beta_1)\odot\sigma(W_g x)$ 是把 Swish 门 $W_g x\odot\sigma(W_g x)$ 里的线性因子 $W_g x$ 换成 softcap，sigmoid 因子原样保留。值支 $\beta_2\tanh(W_u x/\beta_2)$ 是把线性的 $W_u x$ 换成 softcap。
依据：paper/k3-src/2-model-architecture.tex L262-L267；paper/k3-src/appendix.tex L10-L13

这些量都是同形状的向量，$\odot$ 在对应分量上逐元素做，不像矩阵乘那样改变形状。设定的取值是 $\beta_1=4$ 用于门分支、$\beta_2=25$ 用于值分支。
依据：paper/k3-src/2-model-architecture.tex L270

### 手算一遍

取标量输入，并把两支的线性输出都取同一个值 $a$，这样能把函数本身的行为单独考察，不受两个权重矩阵影响。代入 $\beta_1=4$、$\beta_2=25$，输出随着 $a$ 增大一直落在上界 $100$ 之内，而 SwiGLU 在同一取值下继续无界增长。
依据：自行构造

::: 三点代入的完整过程
标量例子里两支的线性输出都取 $a$，于是门支是 $4\tanh(a/4)\cdot\sigma(a)$，值支是 $25\tanh(a/25)$，输出是两者之积。$\sigma(a)=1/(1+e^{-a})$。

1. $a=0$：门支 $4\tanh(0)\cdot\sigma(0)=4\times 0\times 0.5=0$，值支 $25\tanh(0)=0$，输出 $0\times 0=0$。
2. $a=10$：$\tanh(2.5)\approx 0.9866$，门支 $\approx 4\times 0.9866\times 0.99995\approx 3.9463$；$\tanh(0.4)\approx 0.3799$，值支 $\approx 25\times 0.3799\approx 9.4987$；输出 $\approx 3.9463\times 9.4987\approx 37.4846$。
3. $a=100$：$\tanh(25)\approx 1$，门支 $\approx 4\times 1\times 1=4$；$\tanh(4)\approx 0.99933$，值支 $\approx 25\times 0.99933\approx 24.9832$；输出 $\approx 4\times 24.9832\approx 99.9329$，已贴近上界 $100$。
4. 对照 SwiGLU 在 $a=100$：门支 $100\times\sigma(100)\approx 100$，值支 $100$，输出 $\approx 100\times 100=10000$。
依据：自行构造
:::

::: 用代码核对
```python
import math

beta1, beta2 = 4, 25
sig = lambda z: 1 / (1 + math.exp(-z))

for a in (0, 10, 100):
    gate = beta1 * math.tanh(a / beta1) * sig(a)
    up = beta2 * math.tanh(a / beta2)
    print(f"a={a:3d} gate={gate:.4f} up={up:.4f} y={gate * up:.4f}")
```

```text
a=  0 gate=0.0000 up=0.0000 y=0.0000
a= 10 gate=3.9463 up=9.4987 y=37.4846
a=100 gate=4.0000 up=24.9832 y=99.9329
```
依据：自行构造
:::

### 近原点像 SwiGLU、正侧饱和到上界

softcap 在原点附近近似恒等：对 $|z|\ll\beta$，$\beta\tanh(z/\beta)=z+O(z^3/\beta^2)$，即比 $\beta$ 小得多的输入被原样传过。代进公式：$|W_g x|\ll\beta_1$ 时门支的 softcap 退化为 $W_g x$，门支回到 Swish 门 $W_g x\odot\sigma(W_g x)$；$|W_u x|\ll\beta_2$ 时值支退化为 $W_u x$。两支都近原点时，SiTU-GLU 的一阶项与 SwiGLU 相同。
依据：paper/k3-src/appendix.tex L15-L22

远点则被限制在有限区间里。$\|\cdot\|_\infty$ 是各坐标绝对值里最大的那个。因为 $|\tanh(z)|<1$ 且 $0<\sigma(z)<1$，每一维的输出都满足

$$\|\mathrm{SiTU\text{-}GLU}(x)\|_\infty\le\beta_1\beta_2=100.$$

正侧输入很大时，输出趋近这个上界；当 $\beta_1,\beta_2\to\infty$ 时，$\beta\tanh(z/\beta)\to z$，SiTU-GLU 逐点收敛到 SwiGLU。
依据：paper/k3-src/appendix.tex L21-L31；paper/k3-src/2-model-architecture.tex L248-L251

::: 局部展开的来历
$\tanh$ 在原点附近可以用泰勒级数展开。

1. 先写 $\tanh$ 在原点附近的展开。由 $\tanh u=\frac{e^{u}-e^{-u}}{e^{u}+e^{-u}}$，把 $e^{\pm u}$ 都展到 $u^3$ 项，得 $\tanh u=u-\frac{u^3}{3}+O(u^5)$。
2. 令 $u=z/\beta$，代入：$\beta\tanh(z/\beta)=\beta\Big[\frac{z}{\beta}-\frac{1}{3}\Big(\frac{z}{\beta}\Big)^3+O\Big(\Big(\frac{z}{\beta}\Big)^5\Big)\Big]=z-\frac{z^3}{3\beta^2}+O\Big(\frac{z^5}{\beta^4}\Big)$。
3. 当 $|z|\ll\beta$ 时 $\frac{z^3}{3\beta^2}$ 相对 $z$ 是高阶小量，所以 $\beta\tanh(z/\beta)=z+O\big(z^3/\beta^2\big)$。
依据：自行构造
:::

### 适用条件与边界

近原点一阶等价于 SwiGLU，前提是 $|W_g x|\ll\beta_1$ 且 $|W_u x|\ll\beta_2$：输入一旦和 $\beta$ 同量级，高阶项就不能再丢。上界的数值 $100$ 只在 $\beta_1=4$、$\beta_2=25$ 这组设定下成立。
依据：paper/k3-src/appendix.tex L15-L22；paper/k3-src/appendix.tex L24-L32

和直接对门预激活（门分支的线性输出 $W_g x$）做硬截断（把超过阈值的值截平）相比，softcap 的区别在梯度：硬截断在饱和后梯度为零，平滑饱和在离开饱和边界处仍保留非零梯度。
依据：paper/k3-src/appendix.tex L33
