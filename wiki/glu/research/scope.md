# GLU 内容范围

## 1. 概念歧义处理

- 状态：**已裁定**。
- "GLU" 在深度学习语境下主流且唯一相关含义即 Gated Linear Unit（门控线性单元），由 Dauphin et al. 2017 提出。存在一个无关的领域缩写（图形/OpenGL 语境下的 GLU 工具库），与本文无关，不纳入。
- 记号歧义（需在正文显式处理）：Dauphin 原文记 GLU 为 $h(\mathbf{X})=(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$，其中 $\mathbf{W}$ 是"值"分支、$\mathbf{V}$ 是"门"分支；Shazeer 2020 记 $\mathrm{GLU}(x,W,V,b,c)=\sigma(xW+b)\otimes(xV+c)$，把 $\sigma$ 放在 $W$ 分支、$V$ 作值分支。两者仅是 $W\leftrightarrow V$ 标签互换（因 $\otimes$ 可交换），数学等价。本文以 Dauphin 记法为主（与任务背景一致），在家族章节明确指出 Shazeer 的标签差异，避免读者读 SwiGLU 时混淆。

## 2.1 概念含义

- 概念名称：Gated Linear Unit（门控线性单元），缩写 GLU。
- 一句话定义：GLU 是一种神经网络单元，它把输入做两个线性投影，其中一个经过 sigmoid 压到 $(0,1)$ 当作"门"，再与另一个线性投影逐元素相乘，让门控制每个维度放多少信息过去。
- 正式定义（与 Dauphin et al. 2017, §2, Eq.(1) 一致）：给定输入 $\mathbf{X}$，$\;h(\mathbf{X})=(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$，其中 $*$ 为卷积/线性变换，$\sigma$ 为 sigmoid，$\otimes$ 为逐元素乘积。
- 本文语境：作为门控激活函数家族的基础概念，既讲它在卷积语言模型中被提出的原始动机（梯度通路），也讲它在 Transformer FFN 中作为"换激活"的派生起点（Shazeer 2020）。SiTU-GLU、SwiGLU 作为下游家族成员被点名，不展开内部机制。

### 包括什么

- GLU 的定义公式与每个符号的含义（含维度）。
- 门控直觉：门 $\sigma(\cdot)\in(0,1)$ 逐维度缩放值分支。
- GLU 的梯度路径分析：为什么它给梯度留了线性通路（与 GTU 对比）。
- 手算数字例子：给定小输入与权重，逐步算出 GLU 输出。
- GLU 家族的派生规则：换激活（ReGLU/GEGLU/SwiGLU）、去激活（Bilinear）。
- GLU 在 Transformer FFN 中的用法：三矩阵、$d_{ff}$ 缩 $2/3$ 保持参数量。

### 不包括什么

- SiTU-GLU 的内部机制（K3 自有激活，是独立概念，本文只点名它是 GLU 家族成员）。
- SwiGLU 的完整教学（依赖 GLU，是独立概念页，本文只说明派生关系）。
- 卷积神经网络语言模型的完整架构（Gated CNN 是 GLU 的提出场景，本文只用其动机，不展开 CNN 架构）。
- 反向传播的完整推导（本文只讲 GLU 局部梯度路径，不讲全网络反传）。
- 各 GLU 变体在具体大模型（LLaMA/PaLM 等）中的部署细节。

### 相邻概念

- **Sigmoid** $\sigma$：GLU 的门控函数，把任意实数压到 $(0,1)$。本文作为基础记号给一行定义，不展开其概念页（见 §2.4）。
- **Bilinear 层**：GLU 去掉 sigmoid 的退化形式，纳入本页（家族派生）。
- **SwiGLU / GeGLU / ReGLU**：把 GLU 的 sigmoid 换成 Swish/GELU/ReLU，纳入本页（家族派生，只给公式与派生规则，不展开各自机制）。
- **Transformer FFN**：GLU 变体的应用场景，本文只讲"GLU 怎么塞进 FFN"这一最小必要，不展开注意力与 FFN 的完整架构。

## 2.2 学习目标

### Q1：用一句话说清 GLU 在做什么，并说明它解决什么问题

- 完成答案：读者应能说出"GLU 用一个 sigmoid 门逐元素缩放一个线性投影"，并能指出它解决的两类问题——深层网络中 sigmoid/tanh 门控杀梯度（Dauphin 语境），以及单一线性+激活表达力不足、缺少数据相关的乘性调制（Shazeer 语境）。
- 为什么是核心目标：不理解"门控逐元素缩放"和"为什么需要门"，后续公式与梯度分析都失去落点。
- 依赖内容：sigmoid 把实数压到 $(0,1)$、逐元素乘积、线性投影 $xW+b$、动机场景。

### Q2：从公式解释为什么 GLU 既有非线性又给梯度留了线性通路

- 完成答案：读者应能指出 GLU 的梯度 $\nabla[\mathbf{X}\otimes\sigma(\mathbf{X})]=\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\mathbf{X}\otimes\sigma'(\mathbf{X})\nabla\mathbf{X}$ 中，第一项 $\nabla\mathbf{X}\otimes\sigma(\mathbf{X})$ 是不被 $\sigma'$ 或 $\tanh'$ 缩放的通路（对"开门"单元 $\sigma(\mathbf{X})\approx1$ 时近似等于 $\nabla\mathbf{X}$），并对比 GTU 中 $\tanh'(\mathbf{X})$ 随 $|\mathbf{X}|$ 增大趋零。
- 为什么是核心目标：这是 GLU 被提出的核心理由，区分了"门控"与"杀梯度门控"。
- 依赖内容：链式法则、sigmoid 导数 $\sigma'=\sigma(1-\sigma)$、Dauphin §3 Eq.(2)(3)。

### Q3：手算一个 GLU 的小数字例子

- 完成答案：给定小输入 $x$ 与小权重 $W,V,b,c$，读者能算出值分支 $xW+b$、门分支 $\sigma(xV+c)$、逐元素乘出 GLU 输出，并能把数字翻译成"哪个维度被门放行/压低"。
- 为什么是核心目标：确认读者理解了公式与运算顺序，而不只是记住形状。
- 依赖内容：GLU 公式、sigmoid 数值、逐元素乘积。

### Q4：说明 GLU 家族怎么派生（Bilinear、SwiGLU/GeGLU/ReGLU），并指出记号差异

- 完成答案：读者应能说出派生规则——Bilinear 去掉 sigmoid；ReGLU/GEGLU/SwiGLU 把 sigmoid 换成 ReLU/GELU/Swish。并指出 Dauphin 记法（$\sigma$ 在 $V$ 分支）与 Shazeer 记法（$\sigma$ 在 $W$ 分支）是 $W\leftrightarrow V$ 标签互换，数学等价。
- 为什么是核心目标：GLU 是家族基础，读者读 SwiGLU/SiTU-GLU 前必须知道派生规则与记号差异。
- 依赖内容：Shazeer §2 Eq.(4)(5)、Swish $x\cdot\sigma(x)$（作为激活名，不展开）。

### Q5：说明在 Transformer FFN 用 GLU 时为什么要缩 $2/3$ 维度，以及 GLU 不保证什么

- 完成答案：读者应能算出三矩阵 FFN 参数量 $3\cdot d\cdot d_{ff}'$ 等于双矩阵 $2\cdot d\cdot d_{ff}$ 需 $d_{ff}'=\tfrac23 d_{ff}$（例 $3072\to2048$），并说明 GLU 的经验增益没有理论解释（Shazeer 原文"divine benevolence"），不保证在所有任务/模态上都更好。
- 为什么是核心目标：边界意识——避免把经验结论推广成普遍保证。
- 依赖内容：Shazeer §2、§3.1、Table 1、§4 结语。

## 2.3 内容分级

### 核心内容

- GLU 定义公式与符号（→Q1、Q3）。必须讲清：值分支、门分支、$\otimes$、$\sigma$ 的角色与维度。
- 门控直觉与动机（→Q1）。必须讲清：门 $\in(0,1)$ 逐维度缩放，"开门≈放行、关门≈屏蔽"。
- GLU 梯度通路分析（→Q2）。必须讲清：$\nabla\mathbf{X}\otimes\sigma(\mathbf{X})$ 通路与 GTU 的 $\tanh'$ 对比。
- 手算数字例子（→Q3）。必须讲清：代入、sigmoid 数值、逐元素乘、结果含义。
- 家族派生规则与记号差异（→Q4）。必须讲清：Bilinear 去门、换激活变体、Dauphin/Shazeer 标签差异。
- FFN 三矩阵与 $2/3$ 缩放（→Q5）。必须讲清：参数量等式与 $d_{ff}'=\tfrac23 d_{ff}$。

### 辅助内容

- GTU（Gated Tanh Unit）公式与梯度，作为 GLU 的对比项（服务 Q2）。
- Shazeer Table 1 困惑度数字（服务 Q5 的"经验增益"）。
- "divine benevolence" 结语（服务 Q5 的"不保证"）。
- 卷积 vs 线性投影的形式等价说明（澄清 Dauphin 用 $*$ 不只是矩阵乘）。

### 扩展内容

- SiTU-GLU / SwiGLU 在具体大模型中的部署（排除，属下游独立概念）。
- Gated CNN 完整架构（排除，属另一主题）。
- LSTM 门控对比（排除，仅一句话提及"和 LSTM 门同源"不展开）。

## 2.4 前置知识映射

本文读者需理解以下基础记号，均按"基础记号"处理——在首次使用处给一行定义，不展开概念页（与 content-examples.md 对 sigmoid 的处理一致）：

- **Sigmoid** $\sigma(z)=1/(1+e^{-z})$，把实数压到 $(0,1)$。wiki/ 无概念页。本文作为基础记号内联一行定义。学习目标 Q1/Q2/Q3 依赖。
- **逐元素（Hadamard）乘积** $\otimes$：两个同形向量逐位相乘。无概念页。内联一行定义。Q1/Q3 依赖。
- **线性/仿射变换** $xW+b$：输入向量乘权重矩阵加偏置。无概念页。内联一行定义。Q1/Q3 依赖。
- **链式法则 / 梯度**：用于 Q2 梯度路径分析。无概念页。本文只用到"复合函数导数=各段导数相乘"这一最小形式，内联说明。

递归生成判断：以上均为基础数学记号级（类比 $+$、$\exp$），不构成需要独立概念页的"前置概念"。若未来建立 sigmoid 概念页，本文可改为链接引用。SiTU-GLU、SwiGLU 是**下游**概念（依赖 GLU），不是前置，本文只点名不展开、不放占位链接（它们各自的概念页不在本任务范围）。

## 2.5 明确不展开的内容

- **SwiGLU 内部机制**：依赖 GLU，是独立概念页，本文只给派生公式与记号差异。不展开的原因：属另一独立概念，展开会偏离 GLU 主线。
- **SiTU-GLU 内部机制**：K3 自有激活，独立概念。不展开原因同上。
- **Gated CNN 架构**：Dauphin 的卷积语言模型整体结构。不展开原因：只影响工程规模与提出场景，不影响理解 GLU 单元本身。
- **Transformer 完整架构**：注意力、位置编码等。不展开原因：本文只需"FFN 是每层后的两层 MLP"这一最小语境。
- **反向传播全网络推导**：不展开原因：本文只讲 GLU 局部梯度路径，全网络反传是另一主题。

## 2.6 常见误解和适用边界

### 常见误解

1. **误解**：GLU 就是"sigmoid 激活函数"。
   **正确**：GLU 的 sigmoid 只作用于"门"分支，值分支是纯线性的；GLU 的非线性来自乘性门控，不是把 sigmoid 当激活套在整条输出上。
   **成因**：把"门控"与"激活"混同。
   **影响**：Q1。

2. **误解**：GLU 一定比 ReLU 好。
   **正确**：Shazeer 2020 在 T5 语言建模上经验性观察到 GLU 变体优于 ReLU/GELU 基线（Table 1），但原文明确未给理论解释，不构成普适保证。
   **成因**：把单实验结论推广。
   **影响**：Q5。

3. **误解**：门 $\sigma(\cdot)$ 必须用 sigmoid。
   **正确**：原始 GLU 用 sigmoid，但 Shazeer 2020 的"GLU 变体"框架把 sigmoid 替换为任意激活（ReLU/GELU/Swish）甚至去掉（Bilinear），统称 GLU 家族。
   **成因**：把"原始定义"当成"家族定义"。
   **影响**：Q4。

4. **误解**：Dauphin 与 Shazeer 的 $W,V$ 是同一个东西。
   **正确**：两篇论文把 $\sigma$ 放在不同分支上（Dauphin 在 $V$，Shazeer 在 $W$），是标签互换，不是矛盾；读 SwiGLU 时需用 Shazeer 记法。
   **成因**：跨论文阅读未对齐记号。
   **影响**：Q4。

### 适用边界

- **GLU 解决**：深层前向网络中引入"数据相关的乘性门控"，同时给梯度留一条不被导数压缩的通路。
- **GLU 不解决**：不保证全局最优、不替代归一化、不解决注意力机制要解决的全局信息聚合；门控"控制带宽"不是"聚合多源信息"。
- **条件**：梯度通路结论 $\nabla\mathbf{X}\otimes\sigma(\mathbf{X})$ 在门未饱和（$\sigma(\mathbf{X})$ 不极接近 0）时成立；门完全关闭时该通路也趋于 0。
- **条件不满足时**：若所有门都饱和到 0，GLU 退化为近似零输出、梯度也近零；若门饱和到 1，GLU 退化为近似线性（值分支直通）。
