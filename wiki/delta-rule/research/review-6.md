<!-- review-meta
round: 6
page: wiki/delta-rule/index.html
reviewed_content_sha256: 677151d97a89675d
-->
# Delta 规则与 DeltaNet 审查记录（第 6 轮）

- 页面版本：3ff8a775457707e2cc9d93f70598e825f10b5f64
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题（5 题含解答折叠块）→ 最容易误解 → 1. 为什么线性注意力会"记不清"（1.1 retrieval 含展开折叠块、1.2 手算例子、本章问题）→ 2. Delta 规则的紧凑公式与手算（2.1 来源与命名含命名演化图、2.2 手算一步、本章问题）→ 3. 等价改写（含代入验证折叠块、3.1 回到手算例子、3.2 几何直觉含 SVG 图、本章问题）→ 4. 边界情况与 β_t 的退化（4.1/4.2/4.3/4.4、本章问题）→ 5. DeltaNet 与相邻模型对比（5.1–5.5、代码折叠块、本章问题）→ 全文总结 → 来源与范围说明（C1–C10、F1–F3、N1–N2、构造示例、类比边界、简化条件）→ overview.html。已核对来源：arXiv:2102.11174、arXiv:2406.06484v3、arXiv:2412.06464v3、arXiv:2607.24653v1 原文 HTML，以及 github.com/fla-org/flash-linear-attention。

## 核对通过（不需修复）

- §1 手算例子：$S=\begin{pmatrix}1&1\\1&2\end{pmatrix}$、$Sk_2=(2,3)^\top$，分项 $v_1,v_2,v_3$ 求和 $(2,3)^\top$，与矩阵乘法一致；$(2,3)-(0,2)=(2,1)=v_1+v_3$ 复算通过。
- §2.2 / §3.1 / §4.3 手算：$S_1=\begin{pmatrix}1&0\\0&0\end{pmatrix}$、$S_2=\begin{pmatrix}0&0\\1&0\end{pmatrix}$、$\beta_2=0.5$ 时 $S_2=\begin{pmatrix}0.5&0\\0.5&0\end{pmatrix}$ 均复算通过。
- 折叠代码块实际执行（numpy）：输出与页面「预期输出」逐字一致（Linear S_2=[[1,0],[1,0]]、S_2@k_2=[1,1]；Delta S_2=[[0,0],[1,0]]、S_2@k_2=[0,1]；Matches compact form: True）。
- §5.4 表格数字与 C8/N1/N2 逐一核对：Table 1（MAD）Transformer 94.1/29.8/86.8/99.6/85.2/74.5、Mamba 90.4/6.7/90.1/86.3/89.5/69.3、GLA 80.8/6.9/81.6/88.6/63.3/60.0、DeltaNet 100/35.7/100/100/52.8/71.8 与原文完全一致；Compress DeltaNet 42.2 / Mamba 52.7 与原文一致；Average 确为 6 任务均值（(100+35.7+100+100+52.8+42.2)/6=71.78）。Table 2（1.3B/100B）DeltaNet 16.87/12.21、Mamba 17.06/13.89、GLA 17.25/14.92、Transformer++ 16.85/13.44 与原文完全一致。
- 机制引文核对：「a purely additive update rule makes it difficult to deallocate past key-value associations, eventually leading to key "collisions" when L>d, as pointed out by Schlag et al.」（2406.06484 §2.2）、"writing strength"（§2.2）、"I−k_tk_t^T becomes a projection matrix, erasing information in one subspace while preserving the other d−1 subspaces"（§3.3）、"generalized Householder transformation"（§3.1）、"storing more than d_dot associations will result in a retrieval error"（2102.11174 §4.1）、"akin to the famous error-correcting delta-rule (Widrow & Hoff, 1960)"（2102.11174 §1）、Eq. 23 remove+write 形式（§4.2）、"We refer to the Linear Transformer with our delta update rule as a Delta Network"（2102.11174 §6.3）、Gated DeltaNet gated delta rule 公式 $S_t=S_{t-1}(\alpha_t(I-\beta_t k_tk_t^\top))+\beta_t v_t k_t^\top$（§3.1 Eq. 10）、Mamba2 公式 $S_t=\alpha_t S_{t-1}+v_t k_t^\top$（§2.1）、"$\beta_t$ represents the (adaptive) learning rate"（§3.1）均与来源一致。
- 结构：$d_v=d_k=d$ 维度自洽检查正确；4 处边界退化推导正确；幂等/对称验证正确；每章均有「本章问题」折叠解答，核心问题答案指明章节。`validate.py` 返回 validation ok；页面引用的全部本地资源与 `wiki/linear-attention/index.html` 均存在；无「（待生成）」占位；SVG 无 `$...$` alt，公式在 `<foreignObject>` 中。

## 问题

- [重要·技术] 来源与范围说明 C5（第 733 行）：C5 把引文出处标为「Yang, Kautz & Hatamizadeh ICLR 2025 arXiv:2412.06464 §2.3」，但该论文没有 §2.3——第 2 章只有 2.1 Mamba2 与 2.2 Delta Networks；所引句子实际位于 §2.2。C3（第 729 行）同样把该论文的对比基准标为 §2.3。｜引文依据：2412.06464v3 章节列表为「2 Preliminary / 2.1 Mamba2: Linear Attention with decay / 2.2 Delta Networks: Linear Attention with Delta Rule / 3 Gated Delta Networks」，无 2.3；被引句 "The delta update rule (Widrow et al., 1960; Schlag et al., 2021b) dynamically erases the value..." 位于 2.2 标题正下方。｜修复要求：把 C5、C3 中的「§2.3」改为「§2.2」。｜修复：｜复验：
- [重要·技术] 来源与范围说明 N2（第 757 行）：N2 把「340M 模型 / 15B tokens 设置」标为 §4.1 Table 1（MAD benchmark）的实验设置，但该来源的 §4.1 与 Table 1 未出现 340M/15B；该数字只出现在 §4.2 Table 2 的语言建模表。正文 §5.4（第 590 行）据此把 MAD 结果呈现为在该规模下取得。｜引文依据：§4.1 原文 "We use Arora et al. [4]'s training setting and for DeltaNet we use 2 heads. We do not use convolutions for these experiments."；Table 1 标题 "Results on the synthetic MAD benchmark. Results other than DeltaNet are directly borrowed from Poli et al. [82]."；「340M params / 15B tokens」仅见于 Table 2 标题（"The 340M/1.3B models are trained for 15B/100B tokens respectively."）。｜修复要求：核对 MAD benchmark 实际训练规模的真实出处（Poli et al. 2024 或本文附录），把 N2 的条件标注改到实际出处章节；定位不到则删除 N2 中「340M 模型 / 15B tokens 设置」一句。｜修复：｜复验：
- [轻微·技术] 来源与范围说明 C9（第 741 行）：C9 把「KDA 在 delta rule 基础上加入 channel-wise 遗忘门」标为 arXiv:2607.24653 §1，但 §1 只提到 K3 使用 Kimi Delta Attention，未给机制描述。｜引文依据：§1 原文 "Kimi K3 is built on Kimi Delta Attention [56] and Attention Residuals [59]..."；机制句 "KDA extends the delta-rule recurrence [104, 137] with a channel-wise forget gate [56]" 位于 §2.1.1「Kimi Delta Attention」。｜修复要求：把 C9 的「§1」改为「§2.1.1」。｜修复：｜复验：
- [轻微·技术] 来源与范围说明 C4（第 731 行）：C4 以英文引号给出 "Letting $v_t^{\text{old}} = S_{t-1} k_t$, ..."，读作逐字引用，但 §2.2 原文无此措辞。｜引文依据：§2.2 原文 "it first retrieves the old value using the current key, $\bm{v}_t^{\text{old}}=\mathbf{S}_{t-1}\bm{k}_t$"；「Letting」在全文仅出现于 §3.2。｜修复要求：把该引文片段改为 §2.2 原文措辞，或去掉引号改为释义。｜修复：｜复验：
- [轻微·表述] 全文多处：以「本页 / 本文 / 本概念 / 本代码」为主语的自我指代与元话语（第 68、71、225、269、550、612、614、691、741、761 行），例如「本页以 Yang 2024 NeurIPS 的公式为正式定义」（269）、「本页讲的是…本页不展开」（550）、「本页不展开」（612）、「本页构造了两组教学数字」（761）、「本代码不实现 batch」（691）。｜引文依据：不适用｜修复要求：改为以内容为主语的客观陈述（例「本页以 Yang 2024 的公式为正式定义」→「正式定义采用 Yang 2024 的公式」；「本页构造了两组教学数字」→「两组教学数字均为构造示例」），把自我指代收敛到最少。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：修复（对 C3/C5 的 §2.3 与 N2 的 340M/15B 条件标注先回源核对再改；其余按上述要求逐条修正后交下一轮复验）