# 线性注意力独立审查

- 审查者：独立上下文（AI 模拟 / 真实目标读者）
- 页面版本：index.html `bbd15d5c659941bccaa16ca0d6ddcebd32f1b51d` / overview.html `7584e313ddd1fdd26dd3c5be0a6e73a9991cb1bb`
- 时间：2026-08-09

## 问题

- [轻微·盲读] index.html 开头段："注意力计算就要 $9 \times 10^{10}$ 次运算（即 $300000^2 \approx 9 \times 10^{10}$，对应 $N \times N$ 注意力矩阵的元素数）"：称矩阵元素数为"次运算"，用词不准确；后文 S1 正确给出 $O(N^2 d)$ 次乘加，但开头 hook 把元素数和运算数混用。修法：将"次运算"改为"个元素"或补注"每个元素需 $d$ 次乘加，故总运算约 $9 \times 10^{10} \times d$"。｜ 修复： ｜ 复验：
- [轻微·盲读] index.html S2「用核函数 φ 把相似度分解」节：$d'$（特征映射输出维度）首次出现在"$\phi(K)^T V$（$d' \times d$ 矩阵，与 $N$ 无关）"及图中矩阵维度标注，但正文从未显式定义"$d'$ 是 $\phi$ 的输出维度"。读者只能从图示矩阵形状推断。修法：在 F2 或结合律公式前补一句"$d'$ 是 $\phi$ 将 $d$ 维输入映射后的输出维度"。｜ 修复： ｜ 复验：
- [轻微·盲读] index.html S3/S4 + overview.html：多次引用"K3 知识树""KDA（Kimi Delta Attention）"作为选型动机，但从未解释这两个术语是什么。不影响线性注意力本身的理解，但对完全小白是未解释的项目内部术语。修法：首次出现时加括注说明 K3 和 KDA 的身份（如"K3 是本项目研究的模型体系，KDA 是其采用的注意力模块"），或改为泛化表述不绑定特定模型。｜ 修复： ｜ 复验：
- [轻微·技术] index.html「来源与教学说明」N2："MNIST bits/dim linear 0.644 vs softmax 0.621；WSJ PER linear 8.08 vs softmax 5.12（Katharopoulos Table 1，论文实验模型规模）"：经核实论文 PDF，MNIST 数据确实在 Table 1，但 WSJ PER 数据在 Table 3（非 Table 1）。表号引用错误；数字本身正确。修法：将 N2 拆为两条或修正表号——"MNIST bits/dim …（Katharopoulos Table 1）；WSJ PER …（Katharopoulos Table 3）"。｜ 修复： ｜ 复验：
- [轻微·技术] index.html「来源与教学说明」C1：引文"the computational cost of softmax attention scales with O(N²)"标注为"§3.2.1 原文"，经核实论文 PDF 该引文出现在 §3.2 正文（§3.2.1 标题"Feature Maps and Computational Cost"之前）。章节号标注有误；引文内容正确。修法：将"§3.2.1 原文"改为"§3.2 原文"。｜ 修复： ｜ 复验：
- [轻微·技术] index.html S2 结合律公式 $(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V)$ 对应论文 Eq.(6)，但「来源与教学说明」F2 只标注"Katharopoulos Eq.(3)(4)(5)"，未将 Eq.(6) 纳入来源编号。虽属同一推导链，但严格说引用不完整。修法：在 F2 来源后补注"结合律公式对应 Eq.(6)"。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：进入修复

### 段 A 盲读小结

按页面顺序以小白视角阅读，主线理解无阻断卡点。四个学习目标（Q1 softmax 瓶颈定位、Q2 核分解+结合律降复杂度、Q3 因果递归固定状态、Q4 表达力代价来源）均由 S1–S4 正文章节完整回答，不依赖折叠块。$N=3, d=2$ 手算例子贯穿全文且全部可复算（已逐一验算正确）。教学简化 $\phi(x)=x+1$ 的适用边界已在来源节明确标注。

发现 3 个轻微盲读问题：(1) 开头"次运算"用词把矩阵元素数与运算次数混用；(2) $d'$ 首现未显式定义；(3) K3/KDA 项目术语未解释。三者均不影响核心机制理解。

### 段 B 对照来源小结

逐条核对页面表述与 Katharopoulos 2020 论文（arXiv:2006.16236, ICML 2020）及 Choromanski 2020（Performer）一致性：

1. **定义与机制**：C1–C5 核心论断全部与来源一致。非负约束原文"the only constraint we need to impose to sim(·)… is to be non-negative"逐字匹配（§3.2）。elu+1 选择理由原文"We prefer elu(·) over relu(·) to avoid setting the gradients to 0 when x is negative"逐字匹配（§3.2.1）。
2. **公式与推导**：F1–F5 全部与论文 Eq.(2)–(7)、Eq.(9)–(12)、Eq.(16)–(20) 一致。F2 重排、F3 递归、完整 RNN 形式均逐式核对无误。$N=3$ 手算例子的所有矩阵运算（$\phi(K)^T V$、$\phi(K)^T \mathbf{1}$、$s_1/s_2/s_3$ 递推、$V'_1$ 因果/非因果）已全部复算正确。
3. **可运行代码**：页面无可运行代码块，不适用。
4. **事实与推断**：实验数字 N1（CIFAR-10 4000× 加速）与论文 Table 2 及摘要"up to 4000x"一致；N2（MNIST 0.644 vs 0.621）与 Table 1 一致，但 WSJ PER（8.08 vs 5.12）表号误标为 Table 1（实为 Table 3），数字本身正确。softmax 核无穷维的泰勒展开论证数学正确，页面标注为教学推导而非来源结论。
5. **前置知识引用**："standard-attention 概念页（待生成）"已标注占位提示，符合规范。
6. **教学简化**：$\phi(x)=x+1$ 仅适用非负输入的限制已说明；略去 $\sqrt{d}$ 缩放不影响定性对比的结论已标注；略去多头/位置编码/梯度/FAVOR+ 细节均有说明且不影响 Q1–Q4 回答。
7. **页面功能**：KaTeX 公式渲染正常；折叠块（details）收起后主线仍可理解；目录锚点跳转正常。

发现 3 个轻微技术问题：WSJ 表号引用错误（Table 1 → 实为 Table 3）、O(N²) 引文章节号标注错误（§3.2.1 → 实为 §3.2）、结合律公式（Eq.6）未在来源节标注。
