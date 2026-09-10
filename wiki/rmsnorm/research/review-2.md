# RMSNorm 审查记录（第 2 轮）

- 页面版本：`85fb8e468b227dc59345bb9beefec0d8a9ab5379`（index.html 工作树哈希）
- 审查时间：2026-09-10 22:45
- 审查者：独立审查者（未参与写作，未参与第 1 轮审查与修复；本轮仅读取 index.html、overview.html、check.md、style-guide.md、content-examples.md、arxiv 论文页面与本机源码，未读取 research/ 下任何文件）
- 已完整阅读章节：h1/reading-time/blockquote.meta/引言/核心问题（4 组折叠答案）/常见误解 → 1. 归一化的对象：神经元的加权和（含本章问题与解答、两种归一化流水线对照图）→ 2. RMSNorm：只保留缩放（含计算示例、代码折叠块、本章问题与解答）→ 3. 为什么敢去掉中心化（含「补充：隐式学习率自适应」折叠块、本章问题与解答）→ 4. 计算与实现（含加速明细表、代码折叠块、本章问题与解答）→ 来源与范围说明（论断与来源（C）/公式与来源（F）/外部数字与实验条件（N）/构造示例/辅助解释与类比边界/简化条件及其限制）→ overview.html 全文

## 机械验证结果

`/usr/bin/python3 .dojo/scripts/validate.py wiki/rmsnorm/index.html` → `validation ok`，退出码 0。

| 检查项 | 结果 |
|---|---|
| 引用双向闭合 | 通过。正文上标共 15 处（C2×2、C3×2、C5、C6+N2、C7、C8、C9+C10×4、C9×3、F1、F2、F3×2、F4、F5×3、N1×2）；来源章节条目与正文上标一一对应，无「有条目未引用」或「有引用无条目」。注：C1、C4 编号空缺（见问题 5） |
| 相邻双上标 | 通过。无 `</sup><sup>` 形式；三处 `</sup>…<sup>` 均为不同句之间的正常引用 |
| Unicode 数学字符 | 通过。全文（含 summary、表格、图内 HTML 结构、`dojo:summary`）无 σ/μ/ε/α/Σ/≈/√/∈/⊙/∑/ᵢ/⁻ 等字符；`⌂ ◐ ☀ ↑ ✓ ▼ →` 为非数学装饰字符。overview.html 同样通过 |
| TAB | 通过。index.html 与 overview.html 均无制表符 |
| 占位符 | 通过。无「待生成/TODO/TBD/FIXME/占位/待补」 |
| `<details>` / `</details>` | 通过。均为 12，数量相等。构成：解答×9（核心问题 4 + 各章本章问题 1/1/1/2）+ 补充×1 + 代码×2；前缀全部为规范的「解答：」「补充：」「代码：」 |
| `<head>` 五项元数据 | 通过。`description`（纯文本）、`dojo:summary`（含可渲染 `$...$`）、`dojo:type=concept`、`dojo:topics=模型结构`（在 AGENTS.md 固定大类表内）、`dojo:tag=归一化` 均存在且有效 |
| overview 互链 | 通过。index.html:727 有 `href="overview.html"`；overview.html:38 有 `href="index.html"` |
| 前置概念链接 | 页面无任何概念页链接（只有 `../../index.html` 与 `overview.html`）；wiki/ 下 97 个目录中无 LayerNorm 页。LayerNorm 在第 1 章以公式与两项操作给出完整最小定义，阅读不受阻（见问题 6） |
| 本地资源 | 通过。`../../index.html`、`libs/katex.min.css`、`katex.min.js`、`auto-render.min.js`、`prism.min.js`、`prism-python.min.js`、两份 prism 主题文件均存在 |
| 代码块实跑 | 通过。用 HTML 抽取原文代码块后以 torch 2.8.0 解释器逐字运行，两块输出与页面「预期输出」逐字符相同（`identical = True`）：块 1 输出 `LayerNorm: mu = 2.5 , sigma = 1.118`、`[-1.341641, -0.447214, 0.447214, 1.341641]`、`RMSNorm:   RMS = 2.7386`、`[0.365148, 0.730297, 1.095445, 1.460593]`、`两者最大差: 0.0`；块 2 输出 `torch.nn.RMSNorm 输出: [0.365148, 0.730297, 1.095445, 1.460593]`、`GLM 式（fp32 + eps=1e-5）输出: [0.365148, 0.730296, 1.095444, 1.460593]` |
| 页面内声称的差值复算 | 通过。`torch.nn.RMSNorm(eps=0)` 对上手写 RMS 公式最大差 `1.1920928955078125e-07`（页面写 $1.2\times10^{-7}$）；GLM 式实现对手写公式最大差 `9.5367431640625e-07`（页面写 $9.5\times10^{-7}$） |
| 手算数值复算 | 通过。$\sigma=\sqrt{1.25}=1.11803$、$\mathrm{RMS}=\sqrt{7.5}=2.73861$、RMSNorm 输出均值 $0.91287\approx0.913$，与页面一致 |
| 锚点 | 通过。全部 h2/h3 有 id（normalization-target、target-questions、rmsnorm-definition、rmsnorm-questions、why-effective、why-questions、computation、computation-questions、sources-and-scope-notes），TOC 由 JS 按同一元素集生成，无死锚点 |

## 来源核对记录（逐条对照，check.md §2.2 四步）

主要依据：arXiv:1910.07467（v1，2019-10-16，Biao Zhang & Rico Sennrich）。核对用页面为 https://arxiv.org/abs/1910.07467 与 https://ar5iv.labs.arxiv.org/html/1910.07467 。

| 页面论断 | 定位 | 引文依据（原文片段／代码片段） | 结论 |
|---|---|---|---|
| [F1] 式 (1) 加权和 | 论文 §3 Background | `a_i = Σ_{j=1}^{m} w_{ij} x_j , y_i = f( a_i + b_i )  (1)` | 一致 |
| [F2] 式 (2)(3) LayerNorm | 论文 §3 Background | `ā_i = ( a_i − μ ) / σ · g_i  (2)`；`μ = (1/n) Σ a_i , σ = sqrt( (1/n) Σ (a_i − μ)^2 )  (3)` | 一致 |
| [F3] 式 (4)(5) RMSNorm | 论文 §4 RMSNorm（§4.1 标题为 Invariance Analysis） | `ā_i = ( a_i / RMS(a) ) g_i , where RMS(a) = sqrt( (1/n) Σ a_i^2 )  (4)`；`y = f( ( Wx / RMS(a) ) ⊙ g + b )  (5)` | 一致 |
| [C2] 核心假设 | 论文 §1 与 §4 | §1：`We argue that this mean normalization does not reduce the variance of hidden states or model gradients, and hypothesize that it has little impact on the success of LayerNorm.`；§4：`we hypothesize that the re-scaling invariance is the reason for success of LayerNorm, rather than re-centering invariance.` | 一致（页面中文转述与原文逐句对应） |
| [C3] 零均值时精确相等 | 论文 §4 原句 | `When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm.` | 一致 |
| [C6, N2] 激活均值与标准差 | 论文 §6.1，Table 5（RNNSearch 解码器 GRU hidden-to-hidden，newstest2013） | `Although the mean in RMSNorm is not normalized, in practice it is more stable than the mean of the baseline.`；`Due to their normalization properties, both RMSNorm and LayerNorm stabilize standard deviation.`；Table 5 数值：Baseline M `-2.60 / -1.19 / -1.43 / -1.53`，RMSNorm M `-0.40 / -0.60 / -0.69 / -0.74` | 一致。页面「$-0.40$ 漂移到 $-0.74$」「基线 $-2.60$ 到 $-1.19$」与表值一致；页面「论文的文字只把 RMSNorm 的均值与基线对比，未对 LayerNorm 的均值作相当性判断」与原文一致（原文确无 LayerNorm 均值稳定的正面判断） |
| [C7] 异常初始化鲁棒性 | 论文 §6.1「On the Robustness of RMSNorm」与 Figure 4 | `change the center of weight initialization to 0.2`；`Figure 4: SacreBLEU score curve of LayerNorm and RMSNorm on newstest2013 (devset) when the initialization center is 0.2.`；`LayerNorm becomes very unstable with abnormal initialization, but RMSNorm is more robust`；`RMSNorm is similarly robust as LayerNorm, or more.` | 一致 |
| [C8] 质量相当与无归一化失败 | 论文 §6.1 与 Table 4（WMT14 En-De） | Table 4：LayerNorm `26.6 / 27.7`，RMSNorm `26.8 / 27.7`，Baseline `-`（`“-” indicates that we fail to train this model and BLEU score is 0`）；§6.1：`without which training fails` | 一致 |
| [C5] 梯度分析 | 论文 §4.2 Gradient Analysis，式 (8)(9)(10) | 式 (8)：`∂L/∂g = ∂L/∂v ⊙ Wx / RMS(a)`，`the gradient of g is proportional to the normalized summed inputs, rather than raw inputs`；式 (9)(10) 后：`keeps the negative correlation with weight matrix scaling`，`the negative correlation acts as an implicit learning rate adaptor`；`see also Santurkar et al. 2018 who argue that the success of normalization methods does not come from the added stability to layer inputs, but due to increased smoothness of the optimization landscape` | 一致 |
| [N1] 加速数字 | 论文摘要、§6 开头、Table 2/3/4/6/8/10 | 摘要：`reduces the running time by 7%~64% on different models`；Conclusion：`we empirically observed speedups of 7%∼64% across different models and implementations`；§6 开头：`Unless otherwise noted, all speed-related statistics are measured on one TITAN X (Pascal).`；Table 2 `24.7%`、Table 3 `34.0%`/`11.0%`、Table 4 `231 ± 0.04s (6.9%)`（`measured using Tesla V100`）、Table 6 `15.1%`、Table 8 `40.8%`（pRMSNorm `63.9%`）、Table 10 `20.5%`（`Time is measured with GeForce RTX 2080 Ti`） | 一致。表中 7 行速度、6.9%~40.8% 区间、摘要 7%~64%（含 pRMSNorm，Table 8 的 63.9%）均与论文相符；CIFAR 行只填硬件不填框架，与论文 Table 10 未标框架的事实相符 |
| [C9] GLM 实现三点 | 本机 transformers `models/glm4_moe/modeling_glm4_moe.py` L275-289、`configuration_glm4_moe.py` L182 | L275-289 实测：`class Glm4MoeRMSNorm(nn.Module)` / `hidden_states = hidden_states.to(torch.float32)` / `variance = hidden_states.pow(2).mean(-1, keepdim=True)` / `hidden_states * torch.rsqrt(variance + self.variance_epsilon)` / `return self.weight * hidden_states.to(input_dtype)`；L182 实测：`rms_norm_eps=1e-5,` | 一致。上一轮「文件不存在」问题已消除：路径存在、行号精确（类体恰好 L275-289），三点细节（fp32、eps 加在根号内、权重乘在转回之后）逐行可验 |
| [C10] PyTorch nn.RMSNorm | 本机 torch 2.8.0 `torch/nn/modules/normalization.py`（`class RMSNorm` 在 L321） | L329：`\text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}`；L346：`eps: a value added to the denominator for numerical stability. Default: ``torch.finfo(x.dtype).eps``` | 一致 |
| [F4] 3n 次运算差 | 页面自述为本页推导（论文无逐项计数） | 复算：LayerNorm 多出「均值求和 $n$ 次加法 + 方差 $(a_i-\mu)$ $n$ 次减法 + 归一化分子 $a_i-\mu$ $n$ 次减法」= $3n$；两者共有的平方、求和、开方、除法、增益乘确实相同 | 一致，且已按 §2.2 标注为推导而非来源结论 |
| [F5] 对照实测 | 页面自述为构造示例实测 | 本轮独立复跑：两块代码输出与页面「预期输出」逐字符相同；差值 $1.192\times10^{-7}$、$9.537\times10^{-7}$ 与页面 $1.2\times10^{-7}$、$9.5\times10^{-7}$ 相符 | 一致（未读取 research/，靠独立复跑得到同值） |
| pRMSNorm | 论文摘要 | `We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties.` | 来源支持该词存在（页面未解释该词，见问题 3） |

未发现需要删除或降级的来源论断；页面所有机制描述、归因与引文编号均有可定位依据，且构造数据（$(1,2,3,4)$、$(1,-1,3,-3)$、$\varepsilon$ 影响测量）已明确标注为构造示例。

## 问题

- [轻微·技术] 来源与范围说明 [C9] 与 blockquote.meta：标注的 transformers 源码路径与行号未注明版本，本机同时存在两个版本且行号含义不同｜引文依据：transformers 4.57.1 的 `models/glm4_moe/modeling_glm4_moe.py` L275 = `class Glm4MoeRMSNorm(nn.Module)`（与页面一致）；transformers 5.9.0 的同一文件 L275 = `class Glm4MoeMLP(nn.Module)`（`Glm4MoeRMSNorm` 已移出该文件）｜修复要求：在 [C9] 条目与 blockquote.meta 的路径后补注版本写作 `transformers 4.57.1 models/glm4_moe/modeling_glm4_moe.py L275-289`，并在 [C10] 旁同等标注 torch 归属环境；补注后重新运行 validate.py｜修复：｜复验：
- [轻微·技术] 引言段：「已被 GLM 系列模型的官方实现采用」超出了所引来源的支持范围｜引文依据：blockquote.meta 与 [C9] 指向的来源是 Hugging Face 的 `transformers models/glm4_moe/modeling_glm4_moe.py`，页面未给出 GLM 官方仓库（如 zai-org 系列）路径｜修复要求：把该句改为来源实际支持的表述——「已被 transformers 的 GLM-4-MoE 实现采用<sup>[C9]</sup>」，或另补一条 GLM 官方仓库的路径与行号作为新来源条目再保留「官方实现」的措辞｜修复：｜复验：
- [轻微·可读性] 4. 计算与实现正文与「常见误解」列表：`pRMSNorm` 与 `BLEU` 首次出现时未解释｜引文依据：论文摘要 `We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties.`；页面 4 章正文写「连同变体 pRMSNorm 计为 7%~64%」，[N1] 同｜修复要求：在正文首次出现 pRMSNorm 处补一句最小定义（pRMSNorm 为 partial RMSNorm，用前 p% 的分量估计均方根数值），在首次出现 BLEU 处补一句（机器翻译评测指标，数值越高越好）｜修复：｜复验：
- [轻微·可读性] 1. 归一化的对象：式 (1) 后的 `<ul>` 未定义输入维度 $m$ 与求和索引 $j$｜引文依据：不适用（依 style-guide §11「公式后紧跟 `<ul>` 逐项定义每个符号」与 content-examples A2 正例，A2 明确列出「$j$：只在分母求和中使用的索引」）｜修复要求：在该 `<ul>` 中补两条——$m$ 为输入维度 $\mathbf{x}$ 的维数，$j$ 为只在求和中取遍 $1$ 到 $m$ 的索引｜修复：｜复验：
- [轻微·格式] 来源与范围说明：变量编号存在空号，来源章节首条为 [C2]，全文无 [C1]、[C4]｜引文依据：不适用｜修复要求：把「论断与来源（C）」按出现顺序连续重排为 [C1] 起，并同步替换正文全部 `<sup>[Cx]</sup>` 上标；重排后逐条复验每个上标仍指向正确条目｜修复：｜复验：
- [轻微·格式] 页面链接：LayerNorm 是本页第 1 章整章定义的前置概念，但页面除首页与 overview 外无任何概念页链接，wiki/ 下也无 LayerNorm 页｜引文依据：不适用（依 style-guide §13；同类页面均有前置概念链接，如 wiki/rope/index.html 链到 `../../wiki/positional-encoding/index.html`，wiki/glu/index.html 链到 `../swiglu/index.html`）｜修复要求：按 style-guide §13 递归生成 LayerNorm 概念页，并在引言或第 1 章首次出现 LayerNorm 处加入该链接；若决定不生成，则需在规划文件中记录该前置概念以文字定义代替的判断（check.md §4：改变范围的问题返回规划处理）｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：可发布（6 条轻微问题不影响核心结论、来源一致性与阅读连续性；建议在本轮一并修复问题 1、2、3、4、5 后进入第 3 轮，问题 6 交编排者决定是否递归生成 LayerNorm 页）。已通过项：validate.py 成功；来源论断全部有引文依据且定位准确（含上一轮疑点的 GLM 源码路径与行号，实测 L275-289 恰为 `class Glm4MoeRMSNorm`）；两块代码实跑输出与「预期输出」逐字符一致；overview 与 index 互链；`<details>` 配对 12/12；无 Unicode 数学字符、无 TAB、无占位符；数学符号全部由 KaTeX 渲染。
