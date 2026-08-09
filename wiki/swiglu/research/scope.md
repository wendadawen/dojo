# SwiGLU 内容范围

## 1. 概念歧义处理

- 状态：**已裁定**。
- "SwiGLU" 在深度学习语境下主流且唯一相关含义即 Swish-Gated Linear Unit（Swish 门控线性单元），由 Shazeer 2020 §2 Eq.(5) 定义为 $\mathrm{SwiGLU}(x,W,V,b,c,\beta)=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$。无其它领域同名缩写。
- 记号歧义（需在正文显式处理）：Shazeer 2020 把 $\sigma$/激活放在 $W$ 分支、$V$ 作值分支；Dauphin 2017 原文把 $\sigma$ 放在 $V$ 分支、$W$ 作值分支。两者因 $\otimes$ 可交换而数学等价，只是 $W\leftrightarrow V$ 标签互换。本文沿用 Shazeer 记法（与原始提出一致，也是 LLaMA/PaLM 等部署代码的记法），在 S2 公式处明确标注，避免与 [GLU 概念页](../../wiki/glu/index.html)（Dauphin 主记法）混淆。
- $\beta$ 取值：Shazeer 2020 §2 Eq.(5) 写出含 $\beta$ 的一般形式，但 §1 与 §3.1 实验固定 $\beta=1$（即 $\mathrm{Swish}_1(z)=z\cdot\sigma(z)$），现代 LLM 部署（LLaMA、PaLM 等）也固定 $\beta=1$。本文以 $\beta=1$ 为主，§2 末尾说明 $\beta$ 是可调超参但实践中固定为 1。

## 2.1 概念含义

- 概念名称：SwiGLU（Swish-Gated Linear Unit，Swish 门控线性单元）。
- 一句话定义：SwiGLU 是一种 GLU 变体，把 GLU 门分支的 sigmoid 换成 Swish 激活 $\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$，让门分支可以输出负值和无界正值，再与另一份线性投影逐元素相乘。
- 正式定义（与 Shazeer 2020 §2 Eq.(5) 一致）：$\mathrm{SwiGLU}(x,W,V,b,c,\beta)=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$，其中 $\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$，$\sigma$ 为 sigmoid，$\otimes$ 为逐元素乘积。
- 本文语境：作为 GLU 家族中事实上最常被部署的变体，从"为什么换门"动机出发讲清机制，再讲它在 Transformer FFN 中的三矩阵部署与 $2/3$ 缩放，最后讲经验结论与边界。SiTU-GLU 作为下游有界化改进被点名，不展开内部机制。

### 包括什么

- SwiGLU 的定义公式与每个符号的含义（含 Shazeer 记法标注）。
- Swish 激活的定义、边界值（$\mathrm{Swish}(0)=0$、$\mathrm{Swish}(1)\approx0.731$、$\mathrm{Swish}(-1)\approx-0.269$）与正负侧渐近行为。
- 手算数字例子：给定小输入与权重，逐步算出 SwiGLU 输出，并与同输入下 GLU 的输出对照（突出 Swish 门可负、sigmoid 门恒正的差别）。
- SwiGLU 在 Transformer FFN 中的三矩阵部署：$\mathrm{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2)=(\mathrm{Swish}_1(xW)\otimes xV)W_2$。
- 参数量等式 $3\cdot d\cdot d_{ff}'=2\cdot d\cdot d_{ff}\Rightarrow d_{ff}'=\tfrac23 d_{ff}$，Shazeer §3.1 实例 $3072\to2048$，以及 LLaMA 风格 $d_{ff}'=\tfrac83 d$ 的推导。
- Shazeer Table 1 经验数字与 "divine benevolence" 结语。
- LLaMA/PaLM/Mistral/Qwen/DeepSeek 等模型采用 SwiGLU 的事实。

### 不包括什么

- GLU 家族本身的完整讲解（已在 [GLU 概念页](../../wiki/glu/index.html) 讲过，本文只引用结果）。
- SiTU-GLU 的内部机制（K3 自有 softcap 改进，是独立概念页 [SiTU-GLU](../../wiki/situ-glu/index.html)，本文只点名它是 SwiGLU 的有界化下游）。
- Swish 激活函数的完整讲解（Ramachandran et al. 2017 通过神经搜索找到 Swish；本文只给定义与边界值，不展开搜索过程与其它激活对比）。
- GELU/ReLU 的内部机制（仅在派生对比表中作为名字出现）。
- Transformer 完整架构（注意力、位置编码等；本文只需"FFN 是每层后的两层 MLP"这一最小语境）。
- 各 LLM 的具体部署代码与张量并行细节。

### 相邻概念

- **GLU**：SwiGLU 的父概念。已有概念页 [GLU](../../wiki/glu/index.html)，本文引用其结果（梯度通路、家族派生、Shazeer 记法差异），不重复推导。
- **SiLU**：$\mathrm{Swish}_1$ 的别称（Sigmoid Linear Unit）。本文作为同义名出现一次，不展开。
- **SiTU-GLU**：SwiGLU 加 softcap 有界化的下游改进。已有概念页 [SiTU-GLU](../../wiki/situ-glu/index.html)，本文只在 S4 边界处点名"SwiGLU 门无界是 SiTU-GLU 想解决的问题"。
- **Swish**：本文给出 $\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$ 的最小定义与边界值，不建独立概念页（与 GLU 页对 sigmoid 的处理一致）。
- **GEGLU / ReGLU**：GLU 家族中与 SwiGLU 并列的变体。本文只在派生对照表中出现，不展开。

## 2.2 学习目标

### Q1：用一句话说清 SwiGLU 在做什么，并说明它与 GLU 的关系

- 完成答案：读者应能说出"SwiGLU 把 GLU 门分支的 sigmoid 换成 Swish，让门分支可以输出负值和无界正值，再与另一份线性投影逐元素相乘"，并指出它是 GLU 家族中由 Shazeer 2020 §2 Eq.(5) 定义的变体。
- 为什么是核心目标：不理解"换门"这一动作，后续公式、手算、部署都失去落点。
- 依赖内容：GLU 概念页（门控机制、Shazeer 记法）、Swish 定义、Shazeer 2020 来源。

### Q2：写出 SwiGLU 公式与 Swish 定义，逐符号解释；手算一个小例子

- 完成答案：读者应能写出 $\mathrm{SwiGLU}(x,W,V,b,c,\beta)=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$ 与 $\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$；指出 $x,W,V,b,c,\beta,\otimes$ 各自含义；验证 $\mathrm{Swish}(0)=0$、$\mathrm{Swish}(1)\approx0.731$；用小输入 $x=[1.0,0.5]$、$W=\mathrm{diag}(1,-1)$、$V=I$、$b=c=0$ 算出 $\mathrm{SwiGLU}\approx[0.731,-0.0945]$。
- 为什么是核心目标：确认读者理解了公式与运算顺序，并把 Swish 的边界值落到具体数字。
- 依赖内容：Shazeer §2 Eq.(5)、sigmoid 数值 $\sigma(1)\approx0.731$、$\sigma(-0.5)\approx0.378$、Shazeer 记法。

### Q3：解释为什么 SwiGLU 在 FFN 中要缩 $2/3$ 维度，并算出参数量等式与 LLaMA 的 $8/3\,d$ 推导

- 完成答案：读者应能算出三矩阵 FFN 参数量 $3\cdot d\cdot d_{ff}'$ 等于双矩阵 $2\cdot d\cdot d_{ff}$ 需 $d_{ff}'=\tfrac23 d_{ff}$（Shazeer §3.1 实例 $3072\to2048$）；并指出若基线 FFN 取 $d_{ff}=4d$，则 SwiGLU FFN 取 $d_{ff}'=\tfrac23\times 4d=\tfrac83 d$（LLaMA 风格）。
- 为什么是核心目标：现代 LLM FFN 内部维度取 $\tfrac83 d$ 的来源是 SwiGLU 的直接工程后果，读者必须能推出。
- 依赖内容：Shazeer §2 Eq.(6)、§2 末段 $2/3$ 缩放、§3.1 实例、LLaMA 部署事实。

### Q4：说明 SwiGLU 与 GLU/ReGLU/GEGLU 的派生关系，并指出 Swish 门比 sigmoid 门多了什么能力

- 完成答案：读者应能说出派生规则——GLU 用 sigmoid、ReGLU 用 ReLU、GEGLU 用 GELU、SwiGLU 用 Swish，统一形式为 $\mathrm{激活}(xW+b)\otimes(xV+c)$；并指出 Swish 与 sigmoid 的关键差别：sigmoid 恒正有界 $(0,1)$，Swish 可负（$z<0$ 时 $\mathrm{Swish}(z)<0$）且正侧无界（$z\to+\infty$ 时 $\mathrm{Swish}(z)\sim z$）。结合手算例子第二维 $xW=-0.5$ 时 Swish 门 $=-0.189$ 而 sigmoid 门 $=0.378$，说明 SwiGLU 输出可翻号而 GLU 不能。
- 为什么是核心目标：把"换门"从公式操作提升到机制差别，避免读者把 SwiGLU 等同为 GLU。
- 依赖内容：Shazeer §2 Eq.(4)(5)、Swish 边界值、手算例子。

### Q5：说明 SwiGLU 的经验结论与边界——Shazeer 实验说了什么、没说什么，以及为什么社区选了 SwiGLU

- 完成答案：读者应能说出 Shazeer 2020 Table 1 中 SwiGLU 的 heldout log-perplexity 为 $1.636$（vs ReLU 基线 $1.677$、GEGLU $1.633$），即 SwiGLU 不是 Shazeer 实验中最优（GEGLU 略低），但与 GEGLU 同列"最优两变体之一"；指出 Shazeer §4 明确未给理论解释（"divine benevolence"）；说出社区选 SwiGLU 的工程原因（PaLM 2022 采用 → LLaMA 2023 跟随 → 工具链锁定），不构成"SwiGLU 在所有任务上都比 GEGLU 好"的保证。
- 为什么是核心目标：边界意识——避免把单实验结论与路径依赖推广成普适保证。
- 依赖内容：Shazeer Table 1、§4 结语、LLaMA/PaLM 部署事实、SiTU-GLU 作为下游改进。

## 2.3 内容分级

### 核心内容

- SwiGLU 定义公式与符号（→Q1、Q2）。必须讲清：Shazeer 记法（激活在 $W$ 分支、$V$ 为值分支）、$\otimes$、Swish 的角色。
- Swish 定义与边界值（→Q2）。必须讲清：$\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$、$\mathrm{Swish}(0)=0$、$\mathrm{Swish}(1)\approx0.731$、$\mathrm{Swish}(-1)\approx-0.269$、正侧渐近 $z$、负侧趋 $0$。
- 手算数字例子（→Q2、Q4）。必须讲清：代入、Swish 数值、逐元素乘、与 GLU 同输入对照。
- FFN 三矩阵部署与 $2/3$ 缩放（→Q3）。必须讲清：$\mathrm{FFN}_{\mathrm{SwiGLU}}$ 公式、参数量等式、$3072\to2048$ 实例、$\tfrac83 d$ 推导。
- 派生对照与 Swish vs sigmoid 机制差别（→Q4）。必须讲清：GLU/ReGLU/GEGLU/SwiGLU 的激活替换、Swish 可负无界、sigmoid 恒正有界。
- 经验结论与边界（→Q5）。必须讲清：Table 1 数字、GEGLU 略优于 SwiGLU、divine benevolence、社区采用路径、SiTU-GLU 下游关系。

### 辅助内容

- Shazeer §1 ReLU/GELU/Swish FFN 基线（服务 Q3 的"双矩阵 FFN"对照）。
- LLaMA 部署事实（服务 Q3 的 $\tfrac83 d$ 与 Q5 的社区采用）。
- SiTU-GLU 作为下游有界化改进的点名（服务 Q5 的边界）。

### 扩展内容

- Swish 激活函数的完整讲解（Ramachandran 2017 的神经搜索过程、与 GELU 的对比）——排除，属另一独立概念。
- 各 LLM 的具体部署代码（HuggingFace、vLLM、llama.cpp）——排除，属工程实现。
- SwiGLU 在视觉/强化学习/小数据场景的迁移——排除，Shazeer 实验限定 T5 语言建模，无可靠来源支持迁移结论。

## 2.4 前置知识映射

- **GLU（Gated Linear Unit）**：SwiGLU 的父概念。已有概念页 [GLU](../../wiki/glu/index.html)。本文首次依赖时给出链接，不内联重复讲解门控机制、梯度通路、家族派生规则——这些都在 GLU 页讲过。Q1/Q4 依赖。
- **Sigmoid** $\sigma(z)=1/(1+e^{-z})$：基础记号。GLU 页已作为基础记号内联一行定义；本文沿用，不展开概念页。Q2 依赖。
- **逐元素（Hadamard）乘积** $\otimes$：基础记号。GLU 页已内联定义；本文沿用。Q2 依赖。
- **线性/仿射变换** $xW+b$：基础记号。GLU 页已内联定义；本文沿用。Q2/Q3 依赖。
- **Transformer FFN**：本文首次使用时给一行定义"FFN 是每层后的两层 MLP"，不展开注意力与完整架构。Q3 依赖。

递归生成判断：GLU 已有概念页，本文引用；sigmoid、$\otimes$、$xW+b$、FFN 是基础记号或最小语境，不构成需要独立概念页的"前置概念"。SiTU-GLU 是**下游**概念（依赖 SwiGLU），本文只点名不展开、不放占位链接（已有概念页）。

## 2.5 明确不展开的内容

- **GLU 完整机制**：父概念，已有概念页，本文只引用结果（Shazeer 记法、家族派生规则、梯度通路结论）。不展开原因：避免与 GLU 页重复，保持 SwiGLU 主线。
- **SiTU-GLU 内部机制**：下游有界化改进，已有概念页。不展开原因：属另一独立概念，本文只在 S4 点名"SwiGLU 门无界是 SiTU-GLU 想解决的问题"。
- **Swish 激活函数完整讲解**：Ramachandran 2017 的神经搜索、与 GELU 的对比、$\beta$ 调参规律。不展开原因：属另一独立概念，本文只需 $\mathrm{Swish}_\beta$ 的定义与边界值。
- **Transformer 完整架构**：注意力、位置编码、归一化。不展开原因：本文只需"FFN 是每层后的两层 MLP"这一最小语境。
- **各 LLM 部署代码与张量并行**：不展开原因：属工程实现，不影响理解 SwiGLU 机制。

## 2.6 常见误解和适用边界

### 常见误解

1. **误解**：SwiGLU 就是 GLU 换个名字。
   **正确**：SwiGLU 是 GLU 家族中的一个变体，把门分支的 sigmoid 换成 Swish。差别在 Swish 可负、正侧无界，sigmoid 恒正有界——这改变门分支对值分支的调制方式（输出可翻号、可放大）。
   **成因**：把"家族"与"成员"混同。
   **影响**：Q1、Q4。

2. **误解**：Shazeer 2020 证明了 SwiGLU 是最优激活。
   **正确**：Shazeer Table 1 中 GEGLU（$1.633$）略优于 SwiGLU（$1.636$），SwiGLU 不是实验最优；且 Shazeer §4 明确"未给理论解释"（"divine benevolence"）。社区选 SwiGLU 是 PaLM/LLaMA 采用 + 工具链锁定的工程路径，不构成"在所有任务上都最优"的保证。
   **成因**：把单实验结论与路径依赖推广成普适保证。
   **影响**：Q5。

3. **误解**：SwiGLU 的门和 GLU 一样有界。
   **正确**：GLU 的 sigmoid 门有界 $(0,1)$；SwiGLU 的 Swish 门正侧无界（$z\to+\infty$ 时 $\mathrm{Swish}(z)\sim z$），负侧趋 $0$ 但可取负值。这是 SiTU-GLU 加 softcap 想解决的问题。
   **成因**：把"门"统一想象成"$(0,1)$ 的阀门"。
   **影响**：Q4、Q5。

4. **误解**：$\beta$ 是可学习参数。
   **正确**：Shazeer §1 与 §3.1 实验固定 $\beta=1$；现代 LLM 部署（LLaMA/PaLM 等）也固定 $\beta=1$。$\beta$ 是可调超参，不是可学习参数（Ramachandran 2017 原始 Swish 论文中 $\beta$ 可以是常数或可学习，但 SwiGLU 部署中固定为 $1$）。
   **成因**：把 Swish 论文的"可学习 $\beta$"默认带入 SwiGLU 部署。
   **影响**：Q2。

5. **误解**：LLaMA 的 $d_{ff}=\tfrac83 d$ 是新发明。
   **正确**：$d_{ff}'=\tfrac23 d_{ff}$ 来自 Shazeer 2020 §2 末段；若基线 $d_{ff}=4d$（标准 Transformer 取法），则 SwiGLU FFN 取 $d_{ff}'=\tfrac23\times 4d=\tfrac83 d$。LLaMA 只是把 Shazeer 的 $2/3$ 规则套在 $4d$ 基线上，不是新发明。
   **成因**：把 Shazeer 的工程规则误归给 LLaMA。
   **影响**：Q3。

### 适用边界

- **SwiGLU 解决**：在 GLU 框架下把门分支从"恒正有界的 sigmoid"换成"可负、正侧无界的 Swish"，让门分支能翻号、能放大；在 T5 语言建模上经验性优于 ReLU/GELU 基线。
- **SwiGLU 不解决**：不保证在所有任务/模态上最优（Shazeer 实验限定 T5）；不替代归一化；不做跨位置信息聚合（那是注意力）；门无界可能引发激活爆炸（SiTU-GLU 想解决）；不保证全局最优。
- **条件**：经验增益结论成立的条件是 T5 base、segment-filling 任务、参数与计算量匹配（基线 $d_{ff}=3072$、SwiGLU 变体 $d_{ff}=2048$）；$\beta=1$。
- **条件不满足时**：换任务/模态/规模时 SwiGLU 不一定仍优于 ReLU/GELU；Swish 门正侧无界在深层堆叠或路由分支可能放大激活（SiTU-GLU 的动机）。
