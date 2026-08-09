# SigLIP 教学大纲

## 1. 页面开头

**钩子问题**：用 32k batch 训 CLIP 要把整个 batch 的图文相似度都聚合到每个设备上算 softmax，能不能让每个 (图, 文) 对只看自己就被评分？

**一句话解释**：SigLIP 是把 CLIP 的 softmax 对比损失换成 sigmoid 二分类损失——每个 (image, text) 对独立判断"配不配"，不需要 batch 内全局归一化。

**要解决的具体问题**：CLIP softmax 损失的分母 $\sum_j e^{t x_i\cdot y_j}$ 让每个对的梯度依赖整个 batch，导致小 batch 下 hardest negative 太弱、梯度噪声大；同时分布式训练需要 expensive all-gather。

**学习承诺**：读完这一页，你应该能够——
1. 写出 SigLIP 损失公式与 CLIP softmax 损失公式，指出形式差异；
2. 说明 batch size 与损失函数如何解耦、为什么小 batch 下 SigLIP 仍优于 softmax；
3. 说明可学习温度 $t$ 与可学习 bias $b$ 的各自职责、为什么 $b$ 初始化为 $-10$；
4. 说出 SigLIP 在小算力场景相对 CLIP 的实际优势与边界。

**首个具体场景**：4 个 TPUv4 chip、2 天训练、ImageNet zero-shot 84.5%——这是 CLIP 几乎无法做到的算力级别。

**过渡到第一章**：先看清 CLIP softmax 损失"分母依赖整个 batch"这件事，才能理解 SigLIP 把它去掉的代价与收益。

## 2. 章节设计

### S1 CLIP softmax 损失的"全局耦合"问题——为什么需要换掉它

- **主要教学问题**：CLIP 损失中每个对的梯度通过 softmax 分母与整个 batch 耦合，这带来什么问题？
- **对应范围**：C2（CLIP softmax 公式与分母耦合）、C5（small batch 下 hardest negative 弱、梯度噪声大）、Q1、Q2。
- **正文要点**：
  - CLIP 双塔架构与训练目标的最小回顾（图像塔 + 文本塔 + L2 归一化嵌入 + 共享嵌入空间 + 温度参数 $t=\exp(t')$）；图像塔常用 ViT（已有概念页 [ViT](../../wiki/vit/index.html)），文本塔常用 Transformer。
  - CLIP softmax 损失公式 F2：写出来、解释每个符号（$x_i, y_j, t, |B|, \sum_j$）；对称的 image→text 与 text→image 两次 softmax。
  - **关键机制**：分母 $\sum_j e^{t x_i\cdot y_j}$ 让每个对的损失依赖整个 batch 的所有 $y_j$；某个对的梯度信号被"对 hardest negative 的相对优势"决定。
  - 小 batch 下的问题：hardest negative 往往太弱（运气不好没采到难样本），梯度信号噪声大；论文 §4.2 实证。
  - 分布式训练的工程问题：每个设备需要 all-gather 全部嵌入才能算分母（论文 §3.3 提及）。
- **讲解材料及职责**：
  - 公式 F2：精确表达 CLIP 损失形式。
  - 对照表格：CLIP 与 SigLIP 在"分母依赖 / batch 依赖 / 分布式通信"三栏的对比（在 S4 给出，本节只提）。
  - 图示（ASCII）：softmax 分母耦合全 batch 的依赖关系。
- **前置知识安排**：
  - 正文首次引用 [ViT](../../wiki/vit/index.html)（图像塔架构，已有概念页）。
  - softmax 在 [标准注意力](../../wiki/standard-attention/index.html) 概念页给出（已有概念页）；本页只使用其结果，不展开 softmax 推导。
  - L2 归一化、sigmoid 二分类——基础机器学习概念，正文内联最小定义。
- **完成检查**：
  - 用一句话说出 CLIP softmax 分母形式 $\sum_j e^{t x_i\cdot y_j}$；
  - 说出"每个对的梯度依赖整个 batch"这一关键性质；
  - 说出小 batch 下 hardest negative 弱导致梯度噪声大的机制。
- **过渡**：知道了"分母依赖 batch"是问题所在，下一章看 SigLIP 怎么把它去掉。

### S2 SigLIP 损失公式——逐对独立的二分类

- **主要教学问题**：SigLIP 损失的数学形式是什么？为什么说"每个对独立"？
- **对应范围**：C1（SigLIP 公式与符号）、C3（每对独立性）、Q1。
- **正文要点**：
  - SigLIP 损失公式 F1：写出 $-\frac{1}{|B|}\sum_{i,j}\log\sigma(z_{ij}(t\,x_i\cdot y_j-b))$ 与等价形式 $-\frac{1}{|B|}\sum_{i,j}\log\frac{1}{1+e^{z_{ij}(-t\,x_i\cdot y_j+b)}}$。
  - 逐项符号解释：$x_i, y_j$ 与 CLIP 共享（L2 归一化）；$z_{ij}\in\{+1,-1\}$（配对/不配对）；$t=\exp(t')$ 可学习温度；$b$ 可学习 bias（与 CLIP 相比新增的项）。
  - 与 CLIP 的形式对照：(a) CLIP 分母 $\sum_j$ → SigLIP 无分母；(b) CLIP 两次 softmax（image→text + text→image）→ SigLIP 一次性对 $|B|^2$ 对求和；(c) CLIP 用 cross-entropy on softmax → SigLIP 用 binary cross-entropy on sigmoid；(d) CLIP 无 bias → SigLIP 有 bias $b$。
  - **关键机制**：每个对 $\log\sigma(z_{ij}(t\,x_i\cdot y_j-b))$ 是一项独立的二分类交叉熵；对 $(i,j)$ 的梯度只依赖该对的 $x_i, y_j, z_{ij}, t, b$。
  - 用 $|B|=4$ 的小例子手算（在数字例子部分给出）：4×4 = 16 个对，4 个正、12 个负；展示无分母的逐项求和。
- **讲解材料及职责**：
  - 公式 F1：精确表达 SigLIP 损失。
  - 公式对照表（S1 已提，本节细化）：4 项形式差异。
  - 数字例子（折叠块）：$|B|=4$ 的逐项手算。
- **前置知识安排**：sigmoid 函数（基础概念，内联最小定义）。
- **完成检查**：
  - 写出 SigLIP 损失公式与等价形式；
  - 说出 $z_{ij}$ 的含义与取值；
  - 说出 SigLIP 与 CLIP 的 4 项形式差异。
- **过渡**：形式清楚了。但 $b$ 这个新项为什么需要？为什么 CLIP 不需要它？下一章讲 $b$ 与 $t$ 的职责。

### S3 可学习温度 $t$ 与可学习 bias $b$——为什么 $b$ 是关键

- **主要教学问题**：$t$ 与 $b$ 各承担什么职责？为什么 $b$ 初始化为 $-10$？
- **对应范围**：C4（bias 抵消正负不平衡）、Q3。
- **正文要点**：
  - 温度 $t=\exp(t')$：控制相似度被放大的程度（同 CLIP §3.1）；$t'=\log 10$ 即 $t=10$。
  - 正负比例：batch 内正对仅 $|B|$ 个、负对 $|B|^2-|B|$ 个；正负比例 $1:(|B|-1)$。在 $|B|=32\text{k}$ 时比例为 $1:32767$。
  - **若没有 $b$ 会怎样**：初始模型未训练，$x_i\cdot y_j\approx 0$，正对与负对的损失都约为 $-\log\sigma(0)=-\log 0.5=\log 2$；但因为负对数量压倒性多（32767 倍），总损失被负对主导，梯度方向试图让所有对都"更负"，包括本应是正的对。
  - **$b$ 的作用**：初始 $b=-10$，使 $\sigma(-b)=\sigma(10)\approx 0.99995$——等价于"初始时假设任何对都偏向正类"。这抵消了"1 正 vs 32767 负"的极不平衡，让初始损失接近均匀、初始梯度不偏向错误方向。
  - $b$ 是可学习的：训练过程中模型逐渐学到正确判别能力后，$b$ 会自动调整到合理值（论文 §4.5 图 6 显示 $b$ 最终值与负采样策略有关）。
  - CLIP 为什么不需要 $b$：softmax 的分母已经隐式做了"相对优势"的归一化，每个对的损失是相对量，不存在"绝对数量失衡"问题；SigLIP 把每个对独立化后，必须显式引入 $b$ 处理失衡。
- **讲解材料及职责**：
  - 数字代入：$b=-10$ 时 $\sigma(10)\approx 0.99995$ 的精确计算。
  - 对照表：$|B|=4/32\text{k}/1\text{M}$ 下的正负比例（1:3 / 1:32767 / 1:999999）。
  - 数字例子（折叠块）：$|B|=4$、$t=10$、$b=-10$ 的完整损失手算，展示初始时各对损失接近均匀。
- **前置知识安排**：sigmoid 单调性（基础概念）。
- **完成检查**：
  - 说出 $t'=\log 10$ 即 $t=10$、$b=-10$；
  - 说出"若没有 $b$，初始损失被负对主导、梯度方向错误"；
  - 说出 $b=-10$ 让 $\sigma(-b)\approx 0.99995$ 抵消不平衡；
  - 说出 CLIP 为什么不需要 $b$（softmax 隐式相对归一化）。
- **过渡**：公式与符号全清楚了。下一章看 SigLIP 在不同 batch size 下的实际表现，以及为什么 K3 选择"不从 SigLIP 初始化"。

### S4 batch size 解耦与实验边界——SigLIP 在小算力下的优势与边界

- **主要教学问题**：SigLIP 在不同 batch size 与算力条件下相对 CLIP 的实际优势是什么？该结论的边界在哪？
- **对应范围**：C5（小 batch 下显著优势）、C6（SigLiT 84.5% 与 SigLIP 72.1%）、C7（batch 增大时优势缩小、32k 最优、1M 边际收益消失）、Q2、Q4。
- **正文要点**：
  - **batch size 与损失解耦的机制重述**（与 S1 衔接）：softmax 依赖 batch 内 hard negative 池；sigmoid 不依赖；这是 §4.2 实证结论"sigmoid 在 batch < 16k 显著优于 softmax"的机制原因。
  - SigLiT 84.5% 数字（C6）：4 TPUv4 / 2 天 / batch 20k / ViT-g/14 frozen → ImageNet zero-shot 84.5%。强调这是 CLIP 几乎做不到的算力级别。
  - SigLIP from scratch 数字（C7）：32 TPUv4 / 2 天 / batch 32k → 72.1%（5 天 → 73.4%）；对照 CLIP 72.6% 用约 2500 TPUv3-days。
  - **batch size 扫描**（C8）：多语言 mSigLIP Table 2——batch 从 16k → 32k → 64k → 128k → 240k，ImageNet zero-shot 几乎不变（71.6 → 73.2 → 73.2 → 73.2 → 73.1），但多语言 retrieval 反而下降（34.8 → 34.9 → 34.4 → 33.6 → 32.7）；摘要 batch = 1M 边际收益快速消失。
  - **适用边界**：
    - batch 增大时优势缩小（sigmoid 显著优于 softmax 仅在 batch < 16k）；
    - batch > 32k 后 ImageNet zero-shot 几乎不再提升，多语言 retrieval 反而下降；
    - softmax 在"batch 内有真实相似样本"（false-negative 风险）时可能强制判别更细（mixpeek 2025 综述观点，标注为综述非论文结论）；
    - 极小 batch（如 1:1）时 $b$ 作用减弱。
  - **K3 选择"不从 SigLIP 初始化"的衔接**：SigLIP 的训练目标是"静态图像 + 对比损失"；MoonViT-V2 的目标是"视频 + 自定义训练目标"；K3 选择从零训练更稳定（与 SigLIP 质量无关，是设计取向不同）。MoonViT-V2 已有概念页 [MoonViT-V2](../../wiki/moonvit-v2/index.html)。
- **讲解材料及职责**：
  - 实验数据表格：SigLiT / SigLIP / CLIP 三组关键数字对照（N1 + N2）。
  - 多语言 batch size 扫描表（N3）。
  - 关键结论 callout：32k 接近最优、1M 边际收益消失。
  - K3 衔接段（不展开 MoonViT-V2 机制，只说明选择动机）。
- **前置知识安排**：无新增；引用 [ViT](../../wiki/vit/index.html) 与 [MoonViT-V2](../../wiki/moonvit-v2/index.html)。
- **完成检查**：
  - 说出 SigLiT 84.5% 的训练条件（4 TPUv4 / 2 天 / batch 20k / ViT-g/14）；
  - 说出 SigLIP from scratch 72.1% 与 CLIP 72.6% 用约 2500 TPUv3-days 的对比；
  - 说出 32k 最优、> 32k 多语言 retrieval 反而下降、1M 边际收益消失；
  - 说出至少一个边界（softmax 在 false-negative 场景可能更优 / batch 极小时 $b$ 作用减弱）；
  - 说出 K3 不从 SigLIP 初始化的原因（设计取向不同，非 SigLIP 不好）。
- **过渡**：机制与边界都清楚了。文末给出来源与教学说明。

## 3. 讲解顺序

1. 先讲为什么需要换掉 CLIP softmax（S1：分母耦合、小 batch 弱）；
2. 再讲 SigLIP 损失的形式（S2：公式、符号、4 项差异）；
3. 再讲 $b$ 与 $t$ 的职责（S3：$b=-10$ 抵消不平衡）；
4. 最后讲实验边界与 K3 衔接（S4：小算力优势 + 32k 最优 + K3 不从 SigLIP 初始化）。

一次只引入一个新变量：S1 只用 CLIP 已有概念；S2 引入 $z_{ij}$ 与 $b$；S3 才解释 $b$ 的初始化动机；S4 才引入实验数字。前置概念（ViT、标准注意力、softmax）已在 wiki 下有概念页，正文引用即可。

## 4. 贯穿例子

**贯穿例子**：batch size $|B|=4$ 的最小对比学习场景。

- **第一次出现**（S2）：4 张图 + 4 段文本，构成 4×4 = 16 个对，其中 4 个正对（对角线）+ 12 个负对（非对角线）；用 $t=10, b=-10$ 与某个简化的相似度（如 $x_i\cdot y_j=0$ 初始状态）手算每个对的损失，展示无分母的逐项求和。
- **复用**（S3）：同一 $|B|=4$ 例子，加入 $b=-10$，展示初始时各对损失接近均匀（$\sigma(10)\approx 0.99995$），抵消"1 正 vs 3 负"的不平衡；再展示若没有 $b$，4 个正对损失 + 12 个负对损失的总和被负对主导。
- **数字便于手算**：$t=10, b=-10$，初始 $x_i\cdot y_j=0$，每对 $\log\sigma(z_{ij}(0-(-10)))=\log\sigma(10 z_{ij})$；正对 $z=+1$ → $\log\sigma(10)\approx\log 0.99995\approx -0.00005$；负对 $z=-1$ → $\log\sigma(-10)\approx\log 0.0000454\approx -10.0000$。展示初始时虽然每对损失接近均匀（$\sigma(-b)\approx 0.99995$），但 z 的方向不同导致梯度方向正确。
- **延展到 $|B|=32\text{k}$**（S4）：同样比例下正负比 1:32767，说明 $b$ 在大 batch 下更关键。

## 5. 讲解材料职责

| 材料 | 服务章节 | 教学问题 | 类型 |
|---|---|---|---|
| F2 CLIP softmax 公式 | S1 | CLIP 损失形式与分母依赖 | 公式 |
| F1 SigLIP 公式 | S2 | SigLIP 损失形式与逐对独立 | 公式 |
| F3 sigmoid 定义 | S2 | 等价形式推导 | 公式 |
| F4 嵌入归一化 | S2 | $x_i, y_j$ 的来源 | 公式 |
| F5 温度参数化 | S3 | $t=\exp(t')$ | 公式 |
| F6 初始化值 | S3 | $t'=\log 10, b=-10$ | 公式 |
| ASCII 图示：softmax 分母耦合全 batch | S1 | 视觉化"每个对依赖整个 batch" | 图示 |
| 对照表：CLIP vs SigLIP 4 项形式差异 | S2 | 形式差异结构化 | 对照表 |
| $|B|=4$ 手算例子 | S2、S3 | 公式可复算、$b$ 作用可视化 | 数字例子 |
| 正负比例对照表（$|B|=4/32\text{k}/1\text{M}$） | S3 | $b$ 必要性量化 | 数字例子 |
| SigLiT / SigLIP / CLIP 数字对照表 | S4 | 算力优势实证 | 数字例子 |
| 多语言 batch size 扫描表 | S4 | 32k 最优、> 32k 下降 | 数字例子 |
| 关键结论 callout | S4 | 32k 最优结论 | callout |
| K3 衔接段 | S4 | MoonViT-V2 不从 SigLIP 初始化的动机 | 衔接段 |

不安排可运行代码——本概念是损失函数的数学形式与机制理解，可运行代码（如 numpy 实现 SigLIP 损失）虽可帮助验证公式但不必要；用 $|B|=4$ 的手算数字例子即可承担"可复算"的职责。

## 6. 正文与折叠块分工

### 必须放正文

- CLIP softmax 损失公式 F2 与分母依赖（S1）。
- SigLIP 损失公式 F1 与逐项符号、4 项形式差异（S2）。
- 可学习 $t$ 与 $b$ 的职责、$b=-10$ 抵消不平衡的机制（S3）。
- batch size 解耦机制（S4）。
- SigLiT 84.5% / SigLIP 72.1% / CLIP ~2500 TPUv3-days 关键数字（S4）。
- 32k 最优、1M 边际收益消失结论（S4）。
- K3 不从 SigLIP 初始化的衔接段（S4）。
- 公式符号说明。

### 可放折叠块

- $|B|=4$ 的完整逐项手算（S2、S3 数字例子折叠块）。
- 初始 $b=-10$ 时 $\sigma(-b)=\sigma(10)\approx 0.99995$ 的精确推导（S3 折叠块）。
- 多语言 mSigLIP 完整 Table 2（S4 折叠块）。
- softmax 数值稳定性的 log-sum-exp trick 说明（S1 折叠块，回应"避免数值不稳定是次要收益"）。
- CLIP 与 SigLIP 分布式 chunked 实现的对比图（S4 折叠块，说明不需 all-gather）。

折叠块全部收起时正文仍须回答全部学习目标：Q1 由 S1+S2 正文公式回答；Q2 由 S1 机制段 + S4 解耦段回答；Q3 由 S3 正文 $b$ 机制段回答；Q4 由 S4 正文数字与边界段回答。

## 7. 范围与证据约束

- 大纲只使用 scope.md 中已纳入范围的内容。
- 未发现缺少必需定义或机制。
- 不需要新增学习目标、不需要纳入已排除内容、无新增事实改变概念边界。
- 所有 C1–C8 论断有 evidence.md 中已确认来源。

## 8. 术语表（glossary.md 单独文件）

详见 glossary.md。

## 完成条件检查

- 章节单一任务：S1 讲 CLIP 问题、S2 讲 SigLIP 公式、S3 讲 $b$ 作用、S4 讲实验与边界。
- 讲解顺序：先为什么、再是什么；一次一个新变量。
- 贯穿例子：$|B|=4$ 跨 S2、S3 复用，$|B|=32\text{k}$ 在 S4 延展。
- 材料职责：每项材料关联具体教学问题。
- 正文与折叠块分工：核心机制在正文，手算与扩展表在折叠块。
- 范围与证据约束：全部来自 scope.md 与 evidence.md。
