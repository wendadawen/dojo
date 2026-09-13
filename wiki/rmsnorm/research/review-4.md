<!-- review-meta
round: 4
page: wiki/rmsnorm/index.html
reviewed_content_sha256: 43d7f46d35677acf
-->
# RMSNorm 审查记录（第 4 轮）

- 页面版本：b6005c5376b06fb69d32ded236d4ca3dbe87deb4（工作树；sha256 844cf60b6a6ad574…）
- 审查时间：2026-09-13 19:48
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何规划、修复或前序审查记录）
- 已完整阅读章节（含折叠块与图注）：头部 meta/description/summary、引言、核心问题（4 条）、常见误解、1. 归一化的对象：神经元的加权和、2. RMSNorm：只保留缩放、3. 为什么敢去掉中心化、4. 计算与实现、来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）、两段代码折叠块、流水线对照图图注

## 问题

- [轻微·技术] 核心问题 Q4 解答（index.html L97）：「实现普遍加 $\varepsilon$ 并在 fp32 里算」把 fp32 计算写成普遍做法｜引文依据：GLM 源码 `Glm4MoeRMSNorm.forward` 有 `hidden_states.to(torch.float32)`（[C8]）；PyTorch docstring 仅记 `eps` 默认 `torch.finfo(x.dtype).eps` 且公式为 $\sqrt{\varepsilon+\frac1n\sum x_i^2}$（[C9]），未证明上浮 fp32｜修复要求：改为「实现普遍加 $\varepsilon$；GLM 另把统计量放在 fp32 里计算」，与正文 L290「GLM 的实现先转到 fp32 计算」口径一致｜修复：｜复验：
- [轻微·技术] 「3. 为什么敢去掉中心化」证据二（index.html L245）：「论文系统对比了两种方法的不变性集合（其 Table 1）」把来源表格的覆盖范围缩窄为两种方法｜引文依据：论文 Table 1 行为 BatchNorm / WeightNorm / LayerNorm / RMSNorm / ppRMSNorm，表题 "Invariance properties of different normalization methods."｜修复要求：改为「论文在 Table 1 中并列各归一化方法的不变性；其中 RMSNorm 相对 LayerNorm 只少了权重矩阵 re-centering（其余层级保留）」｜修复：｜复验：
- [轻微·技术] 「来源与范围说明 · 简化条件及其限制」（index.html L387）：「大语言模型上的「质量相当」由后续广泛采用间接支持」是对学界采用情况的事实判断，未给出可定位来源｜引文依据：不适用（无来源支持；论文与 [C8] GLM 源码均不覆盖"广泛采用"）｜修复要求：删除该分句，或改写为明确标注无引用的背景说明（如「除下列来源外，本页不评价其他模型上的表现」）｜修复：｜复验：

## 本轮来源核对摘录（已定位原文/数值）

- 公式：Eq(2) `ā_i=((a_i−μ)/σ)g_i`；Eq(3) `μ=(1/n)Σa_i, σ=√((1/n)Σ(a_i−μ)²)`；Eq(4) `ā_i=(a_i/RMS(a))g_i, RMS(a)=√((1/n)Σa_i²)`；Eq(5) `y=f((Wx/RMS(a))⊙g+b)`——均无 ε。论文仅 v1（arXiv Submission history 只有 v1），L108/L290/L337「v1 式(3)(4) 无 ε」成立。
- 假设原句（§1）：「We argue that this mean normalization does not reduce the variance of hidden states or model gradients, and hypothesize that it has little impact on the success of LayerNorm.」与 L241 引文一致。
- 零均值等价（§4）：「When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm.」与 L172/L183/L232 一致。
- 不变性（Table 1）：RMSNorm 行 = 权重矩阵 re-scaling ✓、权重矩阵 re-centering ✗、向量 re-scaling ✗、数据集 re-scaling ✓、数据集 re-centering ✗、单样本 re-scaling ✓；相对 LayerNorm 仅多丢权重矩阵 re-centering——L245/L348 [C3] 的保留项（权重矩阵缩放、数据集缩放、单样本缩放）正确。
- 均值波动（Table 5）：Baseline M = −2.60/−1.19/−1.43/−1.53（ALL −1.60）；RMSNorm M = −0.40/−0.60/−0.69/−0.74（ALL −0.73）→ L247/L369 [N2] 的「−0.40~−0.74 对 −2.60~−1.19」数值正确。Table 5 正文「Although the mean in RMSNorm is not normalized, in practice it is more stable than the mean of the baseline.」→ L247「论文只把 RMSNorm 均值与基线对比，未对 LayerNorm 均值作相当性判断」成立。
- 加速数字复算：Table 2 501s vs 665s = 24.7%（TF，默认硬体 TITAN X）；Table 3 TL 652 vs 988 = 34.0%、Py 763 vs 857 = 11.0%；Table 4 231 vs 248 = 6.9%（Tesla V100）；Table 6 333 vs 392 = 15.1%（Theano，默认 TITAN X）；Table 8 7.12 vs 12.02 = 40.8%（Theano）；Table 10 31 vs 39 = 20.5%（GeForce RTX 2080 Ti）——与 L277-283 表格逐行一致；摘要「reduces the running time by 7%~64%」= 6.9%…63.9%，L288/L368 [N1] 的解释成立。
- 质量（Table 4/§6.1）：LayerNorm 26.6/27.7、RMSNorm 26.8/27.7、Baseline「-」（训练失败 BLEU 0）→ L249/L352 [C7] 一致。
- 鲁棒性（§6.1）：「change the center of weight initialization to 0.2 … LayerNorm becomes very unstable with abnormal initialization, but RMSNorm is more robust」→ L249/L351 [C6] 一致。
- 梯度（§4.2）：「the gradient of g is proportional to the normalized summed inputs, rather than raw inputs」「The negative correlation acts as an implicit learning rate adaptor」+ Santurkar et al. 2018 → L253/L349 [C4] 一致。
- GLM 源码（本机 transformers 4.57.6）：`glm4_moe/modeling_glm4_moe.py` L275 `class Glm4MoeRMSNorm`，forward 内 `.to(torch.float32)` → `pow(2).mean(-1)` → `rsqrt(variance+eps)` → `self.weight * ....to(input_dtype)`（被 L154/155/362/363/472 实际使用）；`glm4_moe/configuration_glm4_moe.py` L182 `rms_norm_eps=1e-5` → L290/L353 [C8] 一致。
- PyTorch 源码（本机 torch 2.8.0）：`torch/nn/modules/normalization.py` `class RMSNorm`，docstring `RMS(x)=√(ε+(1/n)Σx_i²)`，`eps` 默认 `torch.finfo(x.dtype).eps` → L290/L354 [C9] 一致。
- 代码执行：两段代码块在本机 torch 2.8.0 实跑，输出与页面「预期输出」逐字符一致；`torch.nn.RMSNorm`（eps=0,g=1）对手写公式最大差 1.19e-7、GLM 式差 9.54e-7，与 L319/L363 [F5] 标注的 1.2e-7 / 9.5e-7 相符。
- 机械项：`python3 .dojo/scripts/validate.py wiki/rmsnorm/index.html` → `validation ok`；`dojo:topics=模型结构`、`dojo:tag=网络结构` 均在 ALLOWED_TOPICS/ALLOWED_TAGS 内；`wiki/residual-connection/index.html` 存在（L65 前置链接有效）；index.html ↔ overview.html 互链；无 research/ 路径、无「（待生成）」占位、无 Unicode 数学字符；结构图为 HTML（`.dg-stack`），无等宽字符框线；公式与数字全文无内部矛盾（7%~64% 与 6.9%~40.8% 的口径在 L288 已说明）。
- 表述维度：全文未见会话指代「你/我/我们」（L241 的「我们」位于论文引文内，属引述）；末「本页的推导/本页按…清点」为 style-guide §12 明文允许的自称；引言 L67 的结构说明为 style-guide §7 要求；未发现调试叙事、临场评价、公文连接词堆叠或把「场景」当术语。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（仅 3 处轻微项）；无阻断/重要问题，核心结论、公式、代码输出与全部来源数值均已回源核对一致，轻微项修复后即可发布
