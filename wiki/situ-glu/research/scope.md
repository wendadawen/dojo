# SiTU-GLU 内容范围

## 1. 概念歧义处理

- 状态：**已裁定**。
- "SiTU-GLU" 是 Kimi K3 技术报告 §2.3.2 / §B 提出的、专属于 K3 的激活函数，无同名缩写歧义。全称 "Sigmoid Tanh Unit GLU" 沿用报告原文（§2.3.2 标题），缩写 SiTU-GLU 来自报告同节首段。
- 与同领域其它激活名（SwiGLU、GeGLU、ReGLU、GLU、Swish、SiLU）的关系在 §2.1 "家族角色" 与正文 S4 中显式区分；不存在名称混淆。
- 记号约定：本文沿用 K3 报告 Eq.(12) 的记法，门分支与值分支共用 $W_g$ 的部分按报告原文 "the linear factor of the Swish gate" 理解——即门分支的 $\sigma(W_g x)$ 与 $\beta_1\tanh(W_g x/\beta_1)$ 共用同一个 $W_g$，softcap 只替换 Swish 的线性因子 $x$。读者不必与 Dauphin/Shazeer 的 $W\leftrightarrow V$ 标签切换纠缠，详见 [GLU 概念页](../../wiki/glu/index.html) §"家族" 一章。

## 2.1 概念含义

- 概念名称：Sigmoid Tanh Unit GLU（Sigmoid Tanh 单元 GLU），缩写 SiTU-GLU。
- 一句话定义：SiTU-GLU 是一种把 SwiGLU 的门分支线性因子和值分支同时套上 softcap $\beta\cdot\tanh(x/\beta)$ 的 GLU 变体，让两个乘性因子都平滑饱和，从而给乘积一个固定上界，从根源抑制激活爆炸。
- 正式定义（与 Kimi K3 技术报告 §2.3.2 Eq.(12) 一致）：

$$\mathrm{SiTU\text{-}GLU}(x)=\Big[\beta_1\tanh\!\Big(\frac{W_g x}{\beta_1}\Big)\odot\sigma(W_g x)\Big]\odot\Big[\beta_2\tanh\!\Big(\frac{W_u x}{\beta_2}\Big)\Big],$$

  其中 $\sigma$ 为 sigmoid，$\odot$ 为逐元素乘积，$\beta_1=4$、$\beta_2=25$ 为 K3 的设定值（§2.3.2 末段）。
- 本文语境：作为 Kimi K3 在 Stable LatentMoE 路由分支与 Dense FFN 中使用的激活函数。读者读 K3 精读页 Stable LatentMoE / Dense FFN 之前需先理解本页。

### 包括什么

- SiTU-GLU 的定义公式与每个符号的含义（含维度）。
- 为什么需要它：SwiGLU 两个乘性因子都无界，会导致激活爆炸；GLU 的 sigmoid 门有界但值无界，且没有 Swish 的近原点线性响应。
- softcap $\beta\tanh(x/\beta)$ 的两个性质：近原点近似线性（局部展开 $z+O((z/\beta)^3)$）、远点饱和到 $\pm\beta$。
- 上界证明：$|\mathrm{SiTU\text{-}GLU}|\le \beta_1\beta_2=100$，依据 $|\tanh|<1$ 与 $0<\sigma<1$（§B Eq.(19)）。
- 局部与极限行为：近原点与 SwiGLU 一阶等价（§B Eq.(18)），$\beta_1,\beta_2\to\infty$ 逐点收敛到 SwiGLU（§B 末段）。
- 与 hard clamping 的对比：softcap 远离饱和边界处保留非零梯度（§B 末段）。
- 手算数字例子：$x=0$、$x=10$、$x=100$ 三个标量输入的 SiTU-GLU 输出，并对比同输入下 SwiGLU 的无界增长。

### 不包括什么

- K3 整体架构（Stable LatentMoE、Quantile Balancing、KDA–MLA 等是独立概念，本文只用其作为使用场景的最小语境）。
- SwiGLU 完整教学（依赖 GLU，是 [GLU 概念页](../../wiki/glu/index.html) 已展开的家族成员，本文只引用结论）。
- Low-precision arithmetic（FP8/INT8）的完整训练稳定性问题——只作为激活爆炸后果的最小语境提及。
- 其它 GLU 变体（ReGLU/GEGLU/Bilinear）的内部机制（[GLU 概念页](../../wiki/glu/index.html) 已展开家族派生）。
- softcap 在其它场景（attention logits、router logits）的使用——本文只讲 SiTU-GLU 这一具体应用。

### 相邻概念

- **SwiGLU**：SiTU-GLU 的修改起点。Shazeer 2020 提出，公式 $\mathrm{SwiGLU}(x) = (xW_g\cdot\sigma(xW_g))\odot(xW_u)$。SiTU-GLU 把它的两个 $x$（线性因子）各自换成 $\beta\tanh(x/\beta)$。完整家族派生见 [GLU 概念页](../../wiki/glu/index.html) §"家族"。本文只引用结论，不展开。
- **GLU**：家族基础，见 [GLU 概念页](../../wiki/glu/index.html)。本文不重复门控基础动机。
- **softcap**：函数 $\mathrm{softcap}(x,\beta)=\beta\tanh(x/\beta)$。在 SiTU-GLU 之前已被用于 LLaMA 等模型的 attention logits 与 router logits 限幅，本文只讲它在 SiTU-GLU 这一具体应用中的角色，不展开其它应用。
- **hard clamping**：$\min(\max(x,-c),c)$ 一类硬裁剪。作为 softcap 的对比项纳入本文（§B 末段直接对比），不独立展开。

## 2.2 学习目标

### Q1：用一句话说清 SiTU-GLU 在做什么，并说明它解决什么问题

- 完成答案：读者应能说出"SiTU-GLU 把 SwiGLU 的门分支线性因子和值分支同时换成 softcap $\beta\tanh(x/\beta)$，让两个乘性因子都平滑饱和、给乘积一个固定上界 $\beta_1\beta_2$"，并指出它解决 K3 在 2.8T 参数 + Stable LatentMoE 四连矩阵相乘结构下出现的"激活爆炸"问题（§2.3 开头与 §2.3.2 开头）。
- 为什么是核心目标：不理解"softcap 同时套到两支"和"为什么要套"，后续公式与上界证明都失去落点。
- 依赖内容：SwiGLU 公式（来自 [GLU 概念页](../../wiki/glu/index.html)）、softcap 函数、$\tanh$ 与 $\sigma$ 的值域。

### Q2：从公式说明为什么 SiTU-GLU 同时实现"近原点近似 SwiGLU"和"输出有界"

- 完成答案：读者应能指出两件事——
  (a) 近原点：$\beta\tanh(z/\beta)=z+O((z/\beta)^3)$（§B Eq.(18)），代入 Eq.(12) 后两支在 $|W_g x|,|W_u x|\ll\beta_1,\beta_2$ 时分别退化为 $W_g x\cdot\sigma(W_g x)$ 与 $W_u x$，与 SwiGLU 一阶等价；
  (b) 有界：$|\tanh|\le 1$ 与 $0<\sigma<1$ ⇒ 每个坐标 $|\mathrm{SiTU\text{-}GLU}|\le\beta_1\cdot 1\cdot\beta_2\cdot 1=\beta_1\beta_2=100$（§B Eq.(19)）。
- 为什么是核心目标：这是 SiTU-GLU 设计目标的两个支点，缺一就退化为"只是加个上界"或"只是变个形状"，不再构成 K3 报告意义上的 SiTU-GLU。
- 依赖内容：softcap 局部展开、$\tanh$ 与 $\sigma$ 的界、上界证明。

### Q3：手算 $x=0$、$x=10$、$x=100$ 三个标量输入的 SiTU-GLU 输出

- 完成答案：给定标量输入 $x$（设两支 pre-act 均为 $x$ 以隔离函数行为），读者能算出门支 $\beta_1\tanh(x/\beta_1)\sigma(x)$、值支 $\beta_2\tanh(x/\beta_2)$、乘积，并解释：$x=0$ 输出 0；$x=10$ 输出约 37.5；$x=100$ 输出约 99.93 接近上界 100；对比 SwiGLU 在 $x=100$ 输出 10000（已无界增长）。
- 为什么是核心目标：确认读者理解了公式与三个关键点——原点附近与 SwiGLU 几乎重合、正侧大输入平滑饱和到上界、负侧被 sigmoid 杀到接近 0。
- 依赖内容：SiTU-GLU 公式、$\tanh$ 与 $\sigma$ 在 $0/2.5/25$ 附近的数值、上界 100。

### Q4：说明 softcap 与 hard clamping 在梯度上的关键差别

- 完成答案：读者应能指出——hard clamping $\min(\max(x,-c),c)$ 在 $|x|>c$ 时梯度为 0，进入饱和就"死"；softcap $\beta\tanh(x/\beta)$ 在 $|x|\to\infty$ 时虽然饱和但 $\tanh'$ 不严格等于 0（指数衰减，远点处仍非零），K3 §B 末段原文据此称 "preserves nonzero gradients away from saturation boundaries"。这影响训练行为：饱和区仍能传梯度，模型不至于在饱和后完全失能。
- 为什么是核心目标：K3 §B 末段明确把 SiTU-GLU 与 hard clamping 做了对比，这是 SiTU-GLU 选择 softcap 而非简单 clip 的根本理由，不点出这一对比等于没讲完设计动机。
- 依赖内容：hard clamping 定义、$\tanh$ 的导数 $\tanh'=1-\tanh^2$、$\tanh$ 的指数渐近形式。

### Q5：说明 SiTU-GLU 在 K3 中的使用位置与不解决的问题

- 完成答案：读者应能指出——使用位置：Stable LatentMoE 的 routed expert FFN（抑制路由分支四连矩阵相乘的激活爆炸，§2.3 开头）和 Dense FFN（K3 整体 FFN 激活函数，对比表 K2→K3 "SwiGLU→SiTU-GLU"，§4 表）；不解决：(a) 不解决 Quantile Balancing 的负载均衡问题（§2.3.3 独立处理）；(b) 不解决 MLA 注意力内部的稳定性；(c) 不保证训练损失一定下降——报告 §2.3.2 末段说"preserves the local response of SwiGLU while controlling both factors"，是结构上的稳定性保证，不是性能保证。
- 为什么是核心目标：边界意识——避免把"抑制激活爆炸"扩大为"训练更稳/性能更好"的普适结论。
- 依赖内容：K3 §2.3、§2.3.2 末段、K2→K3 对比表 §4。

## 2.3 内容分级

### 核心内容

- SiTU-GLU 定义公式与符号（→Q1、Q3）。必须讲清：门支 $\beta_1\tanh(W_g x/\beta_1)\odot\sigma(W_g x)$、值支 $\beta_2\tanh(W_u x/\beta_2)$、$\odot$、$\sigma$、$\beta_1=4/\beta_2=25$。
- 激活爆炸动机：SwiGLU 两因子无界、K3 四连矩阵放大（→Q1）。必须讲清：$x\cdot\sigma(x)$ 在 $x\to\infty$ 时近似 $x$、值 $x$ 也无界，乘积近似 $x^2$；K3 路由分支 $W_\downarrow\to$FFN$\to W_\uparrow$ 四连乘放大。
- softcap 两个性质（局部展开、远点饱和）（→Q2）。必须讲清：$\beta\tanh(z/\beta)=z+O((z/\beta)^3)$、$|\beta\tanh(z/\beta)|\le\beta$。
- 上界证明（→Q2）。必须讲清：$|\tanh|\le1$、$0<\sigma<1$、$\beta_1\beta_2=100$。
- 手算 $x=0/10/100$（→Q3）。必须讲清：三组数值代入、中间结果、最终结果与上界 100 的接近度。
- softcap vs hard clamping 的梯度对比（→Q4）。必须讲清：hard clamping 饱和后梯度为 0、softcap 饱和后 $\tanh'$ 指数衰减非零。
- K3 使用位置与不解决项（→Q5）。必须讲清：Stable LatentMoE 路由分支 + Dense FFN、不解决 QB 与 MLA。

### 辅助内容

- $\beta_1,\beta_2\to\infty$ 时 SiTU-GLU 逐点收敛到 SwiGLU（服务 Q2，澄清"SiTU-GLU 是 SwiGLU 的有界化版本"）。
- SwiGLU 在 $x=100$ 输出 10000 的对照数字（服务 Q3 的对照）。
- K2→K3 激活函数对比表中 SwiGLU→SiTU-GLU 一行（服务 Q5）。

### 扩展内容

- softcap 在 attention logits / router logits 上的其它应用（排除，属另一主题）。
- 其它有界激活（GeGLU、GELU 自身的有界性讨论）（排除，属家族成员各自概念页）。
- K3 的低精度训练（FP8）完整方案（排除，只作为动机语境）。
- $\tanh$ 的泰勒展开完整推导（排除，本文只用一阶结论）。

## 2.4 前置知识映射

本文读者需理解以下概念，按如下方式处理：

- **GLU（Gated Linear Unit）**：SiTU-GLU 是 GLU 家族下游成员。已有概念页：[wiki/glu/index.html](../../wiki/glu/index.html)。正文首次提及时给链接，不内联重复讲解家族派生。Q1 依赖。
- **SwiGLU**：SiTU-GLU 的修改起点。在 [GLU 概念页](../../wiki/glu/index.html) §"家族" 一章已展开公式 $\mathrm{SwiGLU}(x)=(xW_g\cdot\sigma(xW_g))\odot(xW_u)$。本文在正文 S1 引用时给出公式并指向 GLU 页面，不重复讲解。Q1/Q3 依赖。
- **Sigmoid** $\sigma(z)=1/(1+e^{-z})$：基础记号。无概念页。本文内联一行定义。Q1/Q2 依赖。
- **$\tanh$**：基础记号。无概念页。本文内联一行定义并给出值域 $(-1,1)$ 与导数 $1-\tanh^2$。Q2/Q4 依赖。
- **逐元素乘积 $\odot$**：基础记号。无概念页。本文内联一行定义（与 GLU 页一致）。Q1 依赖。
- **线性变换 $W_g x, W_u x$**：基础记号。无概念页。内联一行。Q1 依赖。
- **hard clamping** $\min(\max(x,-c),c)$：作为对比项，本文内联一行定义。无概念页。Q4 依赖。
- **渐近记号 $O(\cdot)$**：基础记号。内联一行。Q2 依赖。

递归生成判断：GLU 是已有概念页（深度 0）；SwiGLU 在 GLU 页内已展开，不单独建页；其余均为基础数学原语级（类比 $+$、$\exp$），不构成需要独立概念页的"前置概念"。SiTU-GLU 的下游概念（K3 Stable LatentMoE 等）不属本任务范围，本文不引用其占位链接，只在 Q5 中以一句话指出使用场景。

## 2.5 明确不展开的内容

- **K3 Stable LatentMoE 完整架构**：$W_\downarrow$、专家分支聚合、RMSNorm、Quantile Balancing 等是独立概念。不展开原因：属另一独立概念，本文只需"路由分支四连矩阵相乘会放大激活"这一最小动机语境。
- **SwiGLU 的完整教学**：依赖 GLU 的家族成员。不展开原因：[GLU 概念页](../../wiki/glu/index.html) 已展开，本文只引用公式。
- **softcap 在 attention logits / router logits 的其它应用**：不展开原因：只影响 K3 的其它设计选择，不影响理解 SiTU-GLU 本身。
- **FP8/INT8 训练稳定性**：不展开原因：只作为"激活爆炸 → 低精度溢出"这一最小后果语境，不展开低精度训练的完整方案。
- **$\tanh$ 的泰勒展开完整推导**：不展开原因：本文只用一阶局部展开结论（§B Eq.(18)），完整推导属数学分析主题。

## 2.6 常见误解和适用边界

### 常见误解

1. **误解**：SiTU-GLU = "把 SwiGLU 输出 clip 到 100"。
   **正确**：SiTU-GLU 在乘积的**两个因子内部**套 softcap，不是对最终输出做 clip。区别在于：clip 只对超过阈值的部分裁剪、且裁剪后梯度为 0；softcap 平滑饱和、保留非零梯度；且因为套在两个因子上，乘积的渐近上界是 $\beta_1\beta_2$ 而非硬阈值。
   **成因**：从字面"输出有界 100"误读成"输出 clip 到 100"。
   **影响**：Q1、Q4。

2. **误解**：SiTU-GLU 一定比 SwiGLU 训练更稳/性能更好。
   **正确**：K3 §2.3.2 末段只承诺"preserves the local response of SwiGLU while controlling both factors"——结构上的有界性，不是训练稳定性或性能的保证。报告未给 SiTU-GLU vs SwiGLU 的对照实验，只有 K3 整体架构与 K2 的对比。
   **成因**：把"激活爆炸被抑制"扩大为"训练一定更稳"。
   **影响**：Q5。

3. **误解**：$\beta_1=4, \beta_2=25$ 是普适最优。
   **正确**：报告只在 §2.3.2 末段给 "we set the soft-cap hyperparameters to $\beta_1=4$ for the gate branch and $\beta_2=25$ for the up branch"，没有讨论选择依据、没有对照实验。这是 K3 的工程选择，不是普适最优。
   **成因**：把工程设定值当成设计原则。
   **影响**：Q1。

4. **误解**：softcap 套到门支就够了，值支不必。
   **正确**：K3 §B 第二段明确 "Kimi K3 applies the same construction to the up branch ... preventing either branch from dominating the product"。如果只套门支，值支仍无界，乘积仍无界——上界证明依赖两支都套。
   **成因**：只看到门控的"调节"角色，没看到乘积有界需要两支同时饱和。
   **影响**：Q2。

5. **误解**：饱和区里 softcap 等价于 hard clamping。
   **正确**：在 $|x|\gg\beta$ 时 softcap 与 hard clamping 的输出都接近 $\pm\beta$，但 softcap 的导数 $\beta\tanh'(x/\beta)\cdot(1/\beta)=1-\tanh^2(x/\beta)$ 指数衰减非零；hard clamping 的导数在 $|x|>c$ 时严格为 0。这一差别是 K3 §B 末段选择 softcap 的核心依据。
   **成因**：从"输出接近"误读成"梯度等价"。
   **影响**：Q4。

### 适用边界

- **SiTU-GLU 解决**：乘积 $W_g x\cdot\sigma(W_g x)\cdot W_u x$ 在 $|W_g x|,|W_u x|$ 同时大时的激活爆炸——把两支的线性因子各自饱和到 $\beta_1, \beta_2$。
- **SiTU-GLU 不解决**：不解决路由负载均衡（QB 处理）、不解决 MLA 注意力稳定性、不解决 low-precision 算术的全部问题（只降低溢出风险）。
- **条件**：上界 $\beta_1\beta_2$ 在 $\beta_1=4,\beta_2=25$ 时成立；局部展开 $z+O((z/\beta)^3)$ 在 $|z|\ll\beta$ 时有效（§B Eq.(18) 适用条件）；非零梯度结论在"远离饱和边界"成立，饱和后仍指数衰减趋于 0 而非严格非零。
- **条件不满足时**：若 $|W_g x|/4$ 与 $|W_u x|/25$ 都很大但符号相反，输出仍可能接近 $-\beta_1\beta_2=-100$（被 sigmoid 杀到接近 0 的负门会进一步压低绝对值，实际负侧上界更紧）；若 $\beta_1\to\infty$ 或 $\beta_2\to\infty$，SiTU-GLU 退化为 SwiGLU 的对应分支无界形式（§B 末段），上界失效。
