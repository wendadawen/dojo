# SiTU-GLU 核心论断与证据

来源优先级：原始论文 > 权威教材/综述 > 官方文档 > 固定版本源码。本文核心论断全部来自 Kimi K3 技术报告（§2.3.2 与 §B），通过本地 `/tmp/kimi-k3-research/k3-report.txt` 与 `/tmp/kimi-k3-research/k3-report.pdf` 逐条核对。SwiGLU 与 GLU 家族相关论断引用 Shazeer 2020 / Dauphin 2017，已在 [GLU 概念页 evidence.md](../../wiki/glu/research/evidence.md) 登记，本文不重复。

## C 论断（机制/事实）

- **C1**：SiTU-GLU 定义为
  $$\mathrm{SiTU\text{-}GLU}(x)=\big[\beta_1\tanh(W_g x/\beta_1)\odot\sigma(W_g x)\big]\odot\big[\beta_2\tanh(W_u x/\beta_2)\big],$$
  $\sigma$ 为 sigmoid，$\odot$ 为逐元素乘积。
  - 来源：Kimi K3 Technical Report §2.3.2 Eq.(12)。
  - 适用条件：$\beta_1,\beta_2>0$；输入 $x$ 与 $W_g,W_u$ 维度匹配。
  - 置信状态：已确认。

- **C2**：SiTU-GLU 通过把 SwiGLU 的两个无界乘性因子各自换成 softcap $\beta\tanh(x/\beta)$，从根源控制激活爆炸；softcap 同时被套到门支的线性因子与值支。
  - 来源：K3 §2.3.2 末段 "SiTU-GLU applies the smooth cap softcap(x, β) = β tanh(x/β) to the linear factor of the Swish gate and independently to the up branch"。
  - 适用条件：以 SwiGLU 的"门 = Swish($W_g x$)= $W_g x\cdot\sigma(W_g x)$、值 = $W_u x$"为修改起点。
  - 置信状态：已确认。

- **C3**：SwiGLU 的两个乘性因子（门 $W_g x\cdot\sigma(W_g x)$ 的线性因子 $W_g x$、值 $W_u x$）都无界；GLU 的 sigmoid 门有界但值 $W_u x$ 无界，且 GLU 不保留 Swish 近原点的线性响应。
  - 来源：K3 §2.3.2 第二段 "both multiplicative factors in SwiGLU are unbounded, so coincident large coordinates can produce activation outliers and increase overflow risk in low-precision arithmetic. The sigmoid gate of the original GLU avoids unbounded gate growth, but it does not retain the approximately linear positive regime of Swish."
  - 适用条件：在 SwiGLU/GLU 标准定义下成立。
  - 置信状态：已确认。

- **C4**：K3 把 soft-cap 超参设为 $\beta_1=4$（门支）、$\beta_2=25$（值支）。
  - 来源：K3 §2.3.2 末段 "For Kimi K3, we set the soft-cap hyperparameters to β1 = 4 for the gate branch and β2 = 25 for the up branch"。
  - 适用条件：K3 工程选择，不构成普适最优。
  - 置信状态：已确认。

- **C5**：softcap 的局部展开 $\beta\tanh(z/\beta)=z+O((z/\beta)^3)$，使 SiTU-GLU 在原点附近与 SwiGLU 一阶等价。
  - 来源：K3 §B Eq.(18) "For a scalar z near the origin, the scaled tanh satisfies β tanh(z/β) = z + O((z/β)^3). SiTU-GLU therefore matches SwiGLU to first order around the origin."
  - 适用条件：$|z|\ll\beta$；一阶等价指局部展开到一阶相同。
  - 置信状态：已确认。

- **C6**：当 $\beta_1,\beta_2\to\infty$ 时，SiTU-GLU 逐点收敛到 SwiGLU。
  - 来源：K3 §B Eq.(18) 后一句 "It also recovers SwiGLU pointwise as β1, β2 → ∞."
  - 适用条件：极限过程，$\beta\to\infty$ 时 $\beta\tanh(z/\beta)\to z$（因 $\tanh(u)\to u$ 当 $u\to 0$）。
  - 置信状态：已确认。

- **C7**：每个输出坐标满足 $|\mathrm{SiTU\text{-}GLU}(x)|\le\beta_1\beta_2=100$（$\beta_1=4,\beta_2=25$）。
  - 来源：K3 §B Eq.(19) "Since |tanh(z)| < 1 and 0 < Sigmoid(z) < 1, every output coordinate satisfies ∥SiTU-GLU(x)∥∞ ≤ β1 β2 = 100."
  - 适用条件：在 K3 设定的 $\beta_1,\beta_2$ 下；逐坐标成立，故无穷范数 $\|\cdot\|_\infty$ 也成立。
  - 置信状态：已确认。

- **C8**：与 hard clamping 不同，softcap 在饱和边界外仍保留非零梯度，K3 称其训练行为更好。
  - 来源：K3 §B 末段 "Unlike hard clamping of gate pre-activations, the smooth cap preserves nonzero gradients away from saturation boundaries, which we find to give better training behavior."
  - 适用条件：$|x|$ 远离 $\pm\beta$ 时；饱和后 $\tanh'$ 指数衰减而非严格非零。
  - 置信状态：已确认（"better training behavior" 是报告经验陈述，未给对照实验）。

- **C9**：SiTU-GLU 在 K3 中用于 Stable LatentMoE 路由分支与 Dense FFN，目的是抑制 Stable LatentMoE 在四连矩阵相乘结构下的激活爆炸。
  - 来源：K3 §2.3 开头 "This ill-conditioned structure, combined with the 2.8-trillion-parameter scale, produces exploding internal activations in the routed branch" + §2.3.2 标题上下文 + §2.3.2 第一段 "controls large-value growth while preserving the characteristic local and positive-side response of SwiGLU"。
  - 适用条件：K3 架构下；SiTU-GLU 是 Stable LatentMoE 三件套（RMSNorm + SiTU-GLU + QB）之一。
  - 置信状态：已确认。

- **C10**：K3 整体把激活函数从 K2 的 SwiGLU 换成 SiTU-GLU（架构对比）。
  - 来源：K3 §4 对比表 "Activation Function: SwiGLU (K2) → SiTU-GLU (K3)"。
  - 适用条件：K2→K3 架构对比语境。
  - 置信状态：已确认。

## F 公式（核心公式与来源）

- **F1**：SiTU-GLU 定义（同 C1）— K3 §2.3.2 Eq.(12)。
- **F2**：softcap $\mathrm{softcap}(x,\beta)=\beta\tanh(x/\beta)$ — K3 §2.3.2 Eq.(12) 前定义式。
- **F3**：局部展开 $\beta\tanh(z/\beta)=z+O((z/\beta)^3)$ — K3 §B Eq.(18)。
- **F4**：上界 $|\mathrm{SiTU\text{-}GLU}(x)|\le\beta_1\beta_2$ — K3 §B Eq.(19)。
- **F5**：极限行为 $\lim_{\beta_1,\beta_2\to\infty}\mathrm{SiTU\text{-}GLU}(x)=\mathrm{SwiGLU}(x)$ — K3 §B Eq.(18) 后一句。
- **F6**：SwiGLU $\mathrm{SwiGLU}(x)=(W_g x\cdot\sigma(W_g x))\odot(W_u x)$ — Shazeer 2020 §2 Eq.(5)，经 [GLU 概念页 evidence F6](../../wiki/glu/research/evidence.md) 已登记。本文以引用方式使用。
- **F7**：$\tanh$ 导数 $\tanh'(z)=1-\tanh^2(z)$ — 标准定义（基础记号）。
- **F8**：hard clamping $\mathrm{clip}(x,c)=\min(\max(x,-c),c)$ — 标准定义（基础记号）。
- **F9**：hard clamping 导数 $\frac{d}{dx}\mathrm{clip}(x,c)=\begin{cases}1 & |x|<c\\ 0 & |x|>c\end{cases}$（边界处不可导） — 标准定义。

## N 数字（外部数字与实验条件）

- **N1**：K3 设定 $\beta_1=4$、$\beta_2=25$，输出上界 $\beta_1\beta_2=100$。
  - 来源：K3 §2.3.2 末段、§B Eq.(19)。
  - 适用条件：K3 工程设定。
  - 置信状态：已确认。

- **N2**：K2→K3 激活函数对比表行：K2 SwiGLU → K3 SiTU-GLU。
  - 来源：K3 §4 对比表。
  - 适用条件：架构对比，无具体性能数字。
  - 置信状态：已确认。

- **N3**（手算验证，非外部数字；归类于"教学示例"，登记于此供 draft-check 复核）：
  - $x=0$：$g=4\tanh(0)\cdot\sigma(0)=0\cdot 0.5=0$，$u=25\tanh(0)=0$，$y=0$。
  - $x=10$：$g=4\tanh(2.5)\cdot\sigma(10)\approx 4\times 0.98661\times 0.99995\approx 3.9463$，$u=25\tanh(0.4)\approx 25\times 0.37995\approx 9.4987$，$y\approx 37.485$。
  - $x=100$：$g=4\tanh(25)\cdot\sigma(100)\approx 4\times 1\times 1=4$，$u=25\tanh(4)\approx 25\times 0.99933\approx 24.983$，$y\approx 99.933$（接近上界 100）。
  - 对照 SwiGLU 同输入（设两支 pre-act 均为 $x$）：$x=0\to 0$，$x=10\to 10\cdot\sigma(10)\cdot 10\approx 99.995$，$x=100\to 100\cdot\sigma(100)\cdot 100=10000$（无界增长）。
  - 数值经 Python `math.tanh / math.exp` 复算，保留四位有效数字。

无其它外部数字。所有手算数字为教学构造，登记于"教学示例"节，不计入 N。
