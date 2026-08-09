# GLU 核心论断与证据

来源优先级：原始论文 > 综述/权威二手。本文核心论断全部来自两篇原始论文 arXiv:1612.08083（Dauphin et al. 2017）与 arXiv:2002.05202（Shazeer 2020），通过 ar5iv HTML 版逐条核对。

## C 论断（机制/事实）

- **C1**：GLU 定义为 $h(\mathbf{X})=(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$，$\sigma$ 为 sigmoid，$\otimes$ 为逐元素乘积，$*$ 为卷积/线性变换。
  - 来源：Dauphin et al. 2017, §2, Eq.(1)。ar5iv 核对一致。
  - 适用条件：输入与两套权重/偏置同形可逐元素相乘。
  - 置信状态：已确认。

- **C2**：GLU 通过提供"梯度的线性通路"缓解深层网络的梯度消失，同时保留非线性能力。
  - 来源：Dauphin §2 原文 "Our gated linear units reduce the vanishing gradient problem for deep architectures by providing a linear path for the gradients while retaining non-linear capabilities (§5.2)."
  - 适用条件：与 GTU（Gated Tanh Unit）等带 $\tanh'$ 缩放的门控对比成立。
  - 置信状态：已确认。

- **C3**：GLU 梯度 $\nabla[\mathbf{X}\otimes\sigma(\mathbf{X})]=\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\mathbf{X}\otimes\sigma'(\mathbf{X})\nabla\mathbf{X}$ 有一项 $\nabla\mathbf{X}\otimes\sigma(\mathbf{X})$ 不含 $\sigma'$ 或 $\tanh'$ 缩放因子，对"开门"单元近似为 $\nabla\mathbf{X}$，可视为"乘性跳连"。
  - 来源：Dauphin §3, Eq.(3)，原文 "has a path ∇X ⊗ σ(X) without downscaling for the activated gating units ... a multiplicative skip connection which helps gradients flow."
  - 适用条件：门 $\sigma(\mathbf{X})$ 未饱和到 0 时该通路有意义；门饱和到 0 时通路亦趋零。
  - 置信状态：已确认。

- **C4**：GTU（Gated Tanh Unit，$\tanh(\mathbf{X})\otimes\sigma(\mathbf{X})$）的梯度含 $\tanh'(\mathbf{X})$ 与 $\sigma'(\mathbf{X})$ 缩放因子，随 $|\mathbf{X}|$ 增大趋零，因而深层会梯度消失；GLU 没有这个 $\tanh'$ 项。
  - 来源：Dauphin §3, Eq.(2)。
  - 适用条件：对比前向链式分解成立。
  - 置信状态：已确认。

- **C5**：Bilinear 层是 GLU 去掉 sigmoid 的形式 $h(\mathbf{X})=(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes(\mathbf{X}*\mathbf{V}+\mathbf{c})$，归因于 Mnih & Hinton 2007。
  - 来源：Dauphin §5.3 "bilinear layers (Mnih & Hinton, 2007) which take the form ..."。
  - 适用条件：去掉门控激活即退化为此形式。
  - 置信状态：已确认。

- **C6**：Shazeer 2020 把 GLU 的 sigmoid 替换为其它激活得到变体：ReGLU（ReLU）、GEGLU（GELU）、SwiGLU（Swish），并可去掉激活得 Bilinear。
  - 来源：Shazeer 2020, §2, Eq.(4)(5)。
  - 适用条件：替换的是"门分支的激活"，值分支保持线性。
  - 置信状态：已确认。

- **C7**：Shazeer 的 FFN_GLU 形式 $\mathrm{FFN}_{GLU}(x,W,V,W_2)=(\sigma(xW)\otimes xV)W_2$ 用三个权重矩阵、省略偏置；为保持参数量与计算量与双矩阵 FFN 相等，把隐藏维 $d_{ff}$ 缩为 $2/3$。
  - 来源：Shazeer 2020, §2, Eq.(6) 与 §2 末段 "reduce the number of hidden units d_ff ... by a factor of ⅔"；§3.1 实例 $d_{ff}:3072\to2048$。
  - 适用条件：参数量等式 $3\cdot d\cdot d_{ff}' = 2\cdot d\cdot d_{ff}$，得 $d_{ff}'=\tfrac23 d_{ff}$。
  - 置信状态：已确认。

- **C8**：Shazeer 在 T5 语言建模上经验观察到所有 GLU 变体都优于 ReLU/GELU 基线，GEGLU 与 SwiGLU 最优；但原文未给理论解释。
  - 来源：Shazeer 2020, §2/§3 结果、Table 1；§4 结语 "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."
  - 适用条件：T5 base、segment-filling 任务、参数与计算量匹配。
  - 置信状态：已确认（经验结论，限定实验条件）。

- **C9**：Dauphin 原文记 $\sigma$ 在 $V$ 分支、$W$ 为值分支；Shazeer 记 $\sigma$ 在 $W$ 分支、$V$ 为值分支。两者是 $W\leftrightarrow V$ 标签互换，因 $\otimes$ 可交换而数学等价，不是矛盾。
  - 来源：Dauphin §2 Eq.(1) vs Shazeer §2 Eq.(4)。
  - 适用条件：跨论文阅读时需对齐。
  - 置信状态：已确认。

## F 公式（核心公式与来源）

- **F1**：$h(\mathbf{X})=(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$ — Dauphin §2 Eq.(1)。
- **F2**：Sigmoid $\sigma(z)=\dfrac{1}{1+e^{-z}}$，$\sigma(z)\in(0,1)$，导数 $\sigma'(z)=\sigma(z)(1-\sigma(z))$ — 标准定义（基础记号）。
- **F3**：GTU 梯度 $\nabla[\tanh(\mathbf{X})\otimes\sigma(\mathbf{X})]=\tanh'(\mathbf{X})\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\sigma'(\mathbf{X})\nabla\mathbf{X}\otimes\tanh(\mathbf{X})$ — Dauphin §3 Eq.(2)。
- **F4**：GLU 梯度 $\nabla[\mathbf{X}\otimes\sigma(\mathbf{X})]=\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\mathbf{X}\otimes\sigma'(\mathbf{X})\nabla\mathbf{X}$ — Dauphin §3 Eq.(3)。
- **F5**：Bilinear $h(\mathbf{X})=(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes(\mathbf{X}*\mathbf{V}+\mathbf{c})$ — Dauphin §5.3。
- **F6**：Shazeer GLU 变体（§2 Eq.(5)）：
  - $\mathrm{ReGLU}(x,W,V,b,c)=\max(0,xW+b)\otimes(xV+c)$
  - $\mathrm{GEGLU}(x,W,V,b,c)=\mathrm{GELU}(xW+b)\otimes(xV+c)$
  - $\mathrm{SwiGLU}(x,W,V,b,c,\beta)=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$
  - 注：此处沿用 Shazeer 记法（$\sigma$/激活在 $W$ 分支）；与 Dauphin F1 相差 $W\leftrightarrow V$ 标签。
- **F7**：$\mathrm{FFN}_{GLU}(x,W,V,W_2)=(\sigma(xW)\otimes xV)W_2$，三矩阵、无偏置 — Shazeer §2 Eq.(6)。
- **F8**：参数量等式 $3\cdot d\cdot d_{ff}'=2\cdot d\cdot d_{ff}\Rightarrow d_{ff}'=\tfrac23 d_{ff}$ — 由 F7 推出（Shazeer §2 明确）。

## N 数字（外部数字与实验条件）

- **N1**：Shazeer 2020 Table 1（T5 segment-filling，heldout log-perplexity，524,288 steps，参数与计算量匹配）：
  - FFN_ReLU（基线）1.677；FFN_GELU 1.679；FFN_Swish 1.683；
  - FFN_GLU 1.663；FFN_Bilinear 1.648；FFN_ReGLU 1.645；
  - FFN_GEGLU 1.633（最优）；FFN_SwiGLU 1.636。
  - 来源：Shazeer 2020 Table 1。
  - 适用条件：T5 base 架构、$d_{model}=768$、12 头、$d_{ff}=3072$（GLU 变体缩为 2048）。
  - 置信状态：已确认（经验数字，非普适）。

- **N2**：Shazeer §3.1 实例：标准 FFN $d_{ff}=3072$；GLU 变体 FFN 缩为 $d_{ff}=2048$（$3072\times\tfrac23=2048$）。
  - 来源：Shazeer §3.1。
  - 置信状态：已确认。

无其它外部数字。所有手算例子为教学构造，登记于教学示例节，不计入 N。
