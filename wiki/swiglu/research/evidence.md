# SwiGLU 核心论断与证据

来源优先级：原始论文 > 权威二手。本文核心论断全部来自 Shazeer 2020（arXiv:2002.05202）原始论文，通过 ar5iv HTML 版逐条核对；LLaMA/PaLM 部署事实通过搜索引擎+二手综述交叉核对。GLU 家族基础论断引用 [GLU 概念页 evidence.md](../../wiki/glu/research/evidence.md)，不重复。

## C 论断（机制/事实）

- **C1**：SwiGLU 定义为 $\mathrm{SwiGLU}(x,W,V,b,c,\beta)=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$，其中 $\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$，$\sigma$ 为 sigmoid，$\otimes$ 为逐元素乘积。
  - 来源：Shazeer 2020, §2, Eq.(5)。ar5iv 核对一致。
  - 适用条件：$xW+b$ 与 $xV+c$ 同形可逐元素相乘；$\beta>0$。
  - 置信状态：已确认。

- **C2**：Swish 的边界与渐近行为：$\mathrm{Swish}_1(0)=0$、$\mathrm{Swish}_1(1)=1\cdot\sigma(1)\approx0.731$、$\mathrm{Swish}_1(-1)=-1\cdot\sigma(-1)\approx-0.269$；$z\to+\infty$ 时 $\mathrm{Swish}_1(z)\sim z$（因 $\sigma(z)\to1$）；$z\to-\infty$ 时 $\mathrm{Swish}_1(z)\to0$（因 $\sigma(z)\to0$ 且 $z\cdot\sigma(z)$ 中 $\sigma$ 指数衰减压过线性增长）。
  - 来源：Swish 定义 $\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$ 来自 Shazeer §1（引用 Ramachandran et al. 2017）；边界值由定义直接代入计算。
  - 适用条件：$\beta=1$。
  - 置信状态：已确认（定义直接推出）。

- **C3**：Shazeer 2020 实验固定 $\beta=1$，即 $\mathrm{Swish}_1(z)=z\cdot\sigma(z)$；现代 LLM 部署（LLaMA、PaLM 等）也固定 $\beta=1$。
  - 来源：Shazeer §1 末段 "we use $\beta=1$ in our experiments"；Shazeer §3.1 实验设置未提 $\beta$ 调参。
  - 适用条件：Shazeer 实验 + 主流 LLM 部署。
  - 置信状态：已确认。

- **C4**：SwiGLU 的门分支 $\mathrm{Swish}_\beta(xW+b)$ 与 GLU 的门分支 $\sigma(xV+c)$ 形状不同：sigmoid 恒正有界 $(0,1)$；Swish 可取负值（$z<0$ 时 $\mathrm{Swish}(z)<0$）且正侧无界（$z\to+\infty$ 时 $\mathrm{Swish}(z)\sim z$）。
  - 来源：由 C2 与 sigmoid 性质直接推出。
  - 适用条件：$\beta>0$。
  - 置信状态：已确认。

- **C5**：Shazeer 2020 §2 Eq.(6) 把 SwiGLU 塞进 FFN：$\mathrm{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2)=(\mathrm{Swish}_1(xW)\otimes xV)W_2$，三矩阵、无偏置。
  - 来源：Shazeer §2, Eq.(6)。
  - 适用条件：FFN 部署语境。
  - 置信状态：已确认。

- **C6**：三矩阵 FFN 为保持参数量与计算量与双矩阵 FFN 相等，把隐藏维 $d_{ff}$ 缩为 $2/3$。
  - 来源：Shazeer §2 末段原文 "All of these layers have three weight matrices, as opposed to two for the original FFN. To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units $d_{ff}$ ... by a factor of $\frac{2}{3}$ when comparing these layers to the original two-matrix version."
  - 适用条件：参数量等式 $3\cdot d\cdot d_{ff}'=2\cdot d\cdot d_{ff}\Rightarrow d_{ff}'=\tfrac23 d_{ff}$。
  - 置信状态：已确认。

- **C7**：Shazeer §3.1 实验设置：T5 base 架构，$d_{model}=768$，12 层编码器 + 12 层解码器，$h=12$ 头，$d_k=d_v=64$；基线 FFN $d_{ff}=3072$，SwiGLU 变体 $d_{ff}=2048$（即 $3072\times\tfrac23$）。
  - 来源：Shazeer §3.1 原文。
  - 适用条件：Shazeer 2020 实验。
  - 置信状态：已确认。

- **C8**：Shazeer Table 1 经验数字（heldout log-perplexity，524,288 步，越低越好）：FFN_ReLU 基线 $1.677$；FFN_GELU $1.679$；FFN_Swish $1.683$；FFN_GLU $1.663$；FFN_Bilinear $1.648$；FFN_ReGLU $1.645$；FFN_GEGLU $1.633$（最优）；FFN_SwiGLU $1.636$。
  - 来源：Shazeer 2020, Table 1。
  - 适用条件：T5 base、segment-filling 任务、参数与计算量匹配、524,288 步。
  - 置信状态：已确认（经验数字，非普适）。

- **C9**：Shazeer §4 结语原文 "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."——明确标注未给理论解释。
  - 来源：Shazeer §4 末段。
  - 适用条件：所有 GLU 变体（含 SwiGLU）的经验增益。
  - 置信状态：已确认。

- **C10**：LLaMA（Touvron et al. 2023）采用 SwiGLU 作为 FFN 激活，hidden 维取 $\tfrac83 d$；PaLM（Chowdhery et al. 2022）也采用 SwiGLU。后续 Mistral、Qwen、DeepSeek 等主流 LLM 跟随。
  - 来源：LLaMA 论文 §2 架构描述（"SwiGLU activation functions [Shazeer et al., 2020]"，引用 PaLM），通过搜索引擎交叉核对；aiwiki.ai/swiglu 综述列出 PaLM、LLaMA 1/2/3、Mistral 7B、Mixtral、Falcon 3、DeepSeek V2/V3、Qwen2/Qwen3、OLMo。
  - 适用条件：2022 年至今的主流开权重 LLM。
  - 置信状态：已确认（多源交叉）。

- **C11**：$\tfrac83 d$ 推导：标准 Transformer FFN 取 $d_{ff}=4d$；SwiGLU FFN 取 $d_{ff}'=\tfrac23 d_{ff}=\tfrac23\times 4d=\tfrac83 d$ 以保参数量相等。
  - 来源：由 C6 直接推出；LLaMA 论文与多篇工程综述确认 LLaMA 取此值。
  - 适用条件：基线 $d_{ff}=4d$。
  - 置信状态：已确认。

- **C12**：SiTU-GLU 是对 SwiGLU 加 softcap 有界化的下游改进——把 SwiGLU 门支与值支的线性因子同时换成 $\beta\tanh(\cdot/\beta)$，使乘积有界。
  - 来源：Kimi K3 Technical Report §2.3.2；详见 [SiTU-GLU 概念页](../../wiki/situ-glu/index.html)。
  - 适用条件：K3 Stable LatentMoE 路由分支。
  - 置信状态：已确认（下游概念页已发布）。

## F 公式（核心公式与来源）

- **F1**：$\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$ — Shazeer §1（引用 Ramachandran et al. 2017）。
- **F2**：$\mathrm{SwiGLU}(x,W,V,b,c,\beta)=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$ — Shazeer §2 Eq.(5)。注：Shazeer 记法（激活在 $W$ 分支）；与 Dauphin GLU 公式相差 $W\leftrightarrow V$ 标签。
- **F3**：$\mathrm{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2)=(\mathrm{Swish}_1(xW)\otimes xV)W_2$ — Shazeer §2 Eq.(6)。
- **F4**：参数量等式 $3\cdot d\cdot d_{ff}'=2\cdot d\cdot d_{ff}\Rightarrow d_{ff}'=\tfrac23 d_{ff}$ — 由 F3 推出，Shazeer §2 明确。
- **F5**：LLaMA 风格 $d_{ff}'=\tfrac83 d$ — 由 F4 + 基线 $d_{ff}=4d$ 推出。

## N 数字（外部数字与实验条件）

- **N1**：Shazeer 2020 Table 1（T5 segment-filling，heldout log-perplexity，524,288 steps，参数与计算量匹配）：
  - FFN_ReLU（基线）$1.677$；FFN_GELU $1.679$；FFN_Swish $1.683$；
  - FFN_GLU $1.663$；FFN_Bilinear $1.648$；FFN_ReGLU $1.645$；
  - FFN_GEGLU $1.633$（最优）；FFN_SwiGLU $1.636$。
  - 来源：Shazeer 2020 Table 1。
  - 适用条件：T5 base、$d_{model}=768$、12 层编码/解码、$d_{ff}=3072$（SwiGLU 变体缩为 $2048$）。
  - 置信状态：已确认（经验数字，非普适）。

- **N2**：Shazeer §3.1 实例：基线 $d_{ff}=3072$；SwiGLU 变体 $d_{ff}=2048$（$3072\times\tfrac23=2048$）。
  - 来源：Shazeer §3.1。
  - 置信状态：已确认。

- **N3**：手算例子数值（教学构造，登记于教学示例节）：
  - $\sigma(1.0)=1/(1+e^{-1})\approx0.7311$；$\sigma(-0.5)=1/(1+e^{0.5})\approx0.3775$。
  - $\mathrm{Swish}_1(0)=0$；$\mathrm{Swish}_1(1.0)=1.0\times0.7311\approx0.7311$；$\mathrm{Swish}_1(-0.5)=-0.5\times0.3775\approx-0.1888$。
  - SwiGLU 输出 $[1.0,0.5]\to[0.7311,\,-0.0944]$（详细推导见 outline §4）。

无其它外部数字。所有手算例子为教学构造，不计入 N 的"外部实验数字"。
