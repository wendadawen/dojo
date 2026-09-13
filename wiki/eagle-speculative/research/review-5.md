<!-- review-meta
round: 5
page: wiki/eagle-speculative/index.html
reviewed_content_sha256: 2f94616e1147b5eb
-->
# EAGLE-3 投机解码 draft 模型审查记录（第 5 轮）

- 页面版本：4425e90ec2f058a753cccd2cea4ac6df08c11459
- 审查时间：2026-09-13 20:11
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：引言与「核心问题」；1. 为什么独立小模型 draft 有两难；2. EAGLE-1 的核心机制；3. EAGLE-3 的两项架构改变（3.1 直接 token 预测 / 3.2 多层特征融合 / 3.3 架构总览）；4. 推理时的自回归 draft（4.1–4.4，含「展开」与「代码」两个折叠块）；5. 训练 draft 模型（5.1 TTT / 5.2 两种损失 / 5.3 LK vs KL，含「补充」折叠块）；6. 工程实例（6.1–6.4）；来源与范围说明；overview.html

## 问题

- [重要·技术] 引言（第 70 行）：把 7B draft 的失败原因写成「draft 自身成本 $c$ 接近 target」，与同页 §1（第 117、143 行）自述的「$c$ 不可忽略（按参数量比 $7/70\approx0.1$ 得到的量级估算，约 0.05–0.1）」互相矛盾——0.05–0.1 与 c→1 相差一个数量级，读者对「为什么 7B draft 不好」会得到两个互斥的模型。｜引文依据：EAGLE-1 §1 原文「Despite the 7B model's potential as a draft model, its high overhead diminishes acceleration gains.」（只说开销高削弱收益，未说成本接近 target）；本页第 117 行自述「约 0.05-0.1，属推断而非论文给出的数值」。｜修复要求：把第 70 行「draft 自身成本 $c$ 接近 target」改为与第 117/143 行一致的表述（如「draft 单步成本 $c$ 不可忽略」），使全页对 7B draft 成本量级只有一种说法。｜修复：｜复验：
- [轻微·表述] 第 656 行：中文句子里夹入未翻译的英文动词——「训练中 $W_{E3}$ 逐渐学到 incorporate low/mid feature」。｜引文依据：K3 报告 §4.1.4 原文「…initialized as [0 0 I] so that the fused representation coincides at initialization with the high-level feature h_h — the input on which the MTP layer was pre-trained.」（来源无 incorporate 一词，且该句本身不含「训练中逐渐学到」的陈述）。｜修复要求：把「学到 incorporate low/mid feature」改为纯中文（如「逐渐学到引入 low/mid feature」）。｜修复：｜复验：
- [轻微·格式] 同一构造场景的 draft token 记号前后不一致：§4.2/§4.3 与图注用全局位置记号，§4.4 手算改用相对序号。｜引文依据：第 325 行「采样得到第一个 draft token $x_{t+2}$」、第 354 行图注「本轮产出 draft token $x_{t+2}, x_{t+3}, x_{t+4}$」，而第 386/396/402 行同一场景写作「$x_1=\text{do}$」「$x_2=\text{it}$」「$x_3=\text{now}$」。｜修复要求：§4.4 统一使用 $x_{t+2}, x_{t+3}, x_{t+4}$（或在 §4.4 首处显式声明 x_1 即 x_{t+2} 的简写并全章沿用），使同一对象的符号全文单义。｜修复：｜复验：
- [轻微·技术] 来源说明 [C10] 引「Leviathan et al. 2023 §3 Theorem 3.5」作为「不改变投机解码框架 / lossless」的依据，但该编号的定理不是正确性结论。｜引文依据：arXiv:2211.17192 §3 Theorem 3.5 = 「β = 1 − DLK(p,q)」（接受率与散度的关系；接受率定义在 Definition 3.1 / Corollary 3.6），输出分布不变的证明在附录 A.1（页面括号内已写「正确性证明另见附录 A.1」）。｜修复要求：把 [C10] 中的「§3 Theorem 3.5」改为指向附录 A.1 的正确性证明（或删除该编号，只保留 A.1），使标注位置的实际内容支持 lossless 论断。｜修复：｜复验：
- [轻微·表述] 章节衔接使用同一固定句式：全文 7 处以「下一章讲/展开/给出」收尾（第 135、197、199、288、359、480、619 行），第 135 行另有「本章说明…」的自我指代开头。｜引文依据：不适用（guides/concept/style-guide.md §8「用一至两句说明前一节结论与下一节问题的关系。不使用固定句式」）。｜修复要求：保留必要的章节衔接信息，但打散句式，使不与「下一章…」逐章雷同；第 135 行去掉以「本章」为主语的元话语，直接陈述结论与下一章问题的关系。｜修复：｜复验：
- [轻微·技术] 第 242 行把「无偏」当作 EAGLE-3 融合矩阵的一般属性：「$W_{\text{fuse}} \in \mathbb{R}^{k \times 3k}$ 是无偏融合矩阵（K3 命名为 $W_{E3}$）」，但「bias-free」只在 K3 实现里出现。｜引文依据：EAGLE-3 §3.1 原文「…then pass it through a fully connected (FC) layer to reduce it to k-dimensions…」（未提 bias-free）；「a bias-free matrix WE3」见 K3 报告 §4.1.4。｜修复要求：把「无偏」限定为 K3 实现（如「K3 实现为无偏矩阵 $W_{E3}$」），或删去该修饰，避免把 K3 细节写成 EAGLE-3 的一般性质。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 核对覆盖（回源定位并比对原文，均取到片段/数值）：C1、C2、N1、N5 见 arXiv:2401.15077 摘要与 §1；C3、C4、C5、N2、N3 见 arXiv:2503.01840 摘要、§3.1、§3.2、§4 与 Table 1（Vicuna 13B 温度=0 时 EAGLE=3.07x、EAGLE-3=5.58x）；C6、C7、C8、C9、N4、F4 见 arXiv:2607.24653 §4.1.4（Eq.(16) LK loss、temperature 1 且无辅助 cross-entropy 项、MTP 初始化、1st/4th/final AttnRes、W_E3=[0 0 I]、unroll seven steps、MXFP4/MXFP8）；C10、F5、F6 见 arXiv:2211.17192 §3（Definition 3.1、Corollary 3.6、Theorem 3.8 的 $(1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c+1))$、附录 A.1）；F1、F2 见 arXiv:2503.01840 §3.1（3k→k 的 FC、第二处 FC 降到 k 后入单层 decoder）；[N3] 亦见 Hugging Face yuhuili/EAGLE-Qwen2-72B-Instruct 模型卡。EAGLE-3 §3.2 经核实确未给出显式损失公式，页面把 $L_{E3}$ 标为「本页形式化」正确。
- 算式复算：§4.4 构造示例三步全部复算通过——Step1 $W_a\cdot[0.4,0.2,0.1,0.6,1,0,0,0]=[0.08,0.45,0.14,0.13]$、$\tanh$ 后 $a_{\text I}=[0.080,0.422,0.139,0.129]$、logits $[0.160,0.844,0.278,0.259,0.385]$、softmax 分母 7.585、$q_1\approx[0.155,0.307,0.174,0.171,0.194]$（argmax "do"）；Step2/Step3 各行乘加、logits、softmax 分母（6.66、7.39）与 argmax（"it"、"now"）均相符；偏差段 $g_{\text I}-a_{\text I}=[0.370,-0.022,-0.039,0.071]$、L2≈0.380 正确。加速比公式 [F6] 与 Theorem 3.8 一致；$\alpha=\sum_x\min(p,q)=1-\mathrm{TV}(p,q)$ 由 $\min(a,b)=(a+b-|a-b|)/2$ 成立。
- 机械项：`python3 .dojo/scripts/validate.py wiki/eagle-speculative/index.html` 返回 `validation ok`；`check_inline_js.py` 通过；公式定界符外无 Unicode 数学字符；结构图为 HTML（无等宽框线图）；details 的 summary 前缀（解答/展开/代码/补充）合规；两级问题块均有解答且页面级答案指明所辖章节；index.html 与 overview.html 互链；链接的 wiki/speculative-decoding/index.html、wiki/mxfp4-qat/index.html 均存在；正文与来源说明未出现 research/ 下文件路径。
- 处置：修复（1 条重要 + 5 条轻微需关闭；无阻断、无返回规划项）