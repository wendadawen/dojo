<!-- review-meta
round: 1
page: wiki/rmsnorm/index.html
reviewed_content_sha256: f96615c5336e58b4
-->
# RMSNorm 审查记录（第 1 轮）

- 页面版本：`8ad70d00f9c6ce61f9b41bb6d51dcd91609666f7`（index.html，`git hash-object` 取）；overview.html：`a318a99f4008ad894c57288eae8e5a994ae58a7e`
- 审查时间：2026-09-10 20:57
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节：引言与核心问题、常见误解；1. 归一化的对象：神经元的加权和（含本章问题解答折叠块）；2. RMSNorm：只保留缩放（含「代码：LayerNorm 与 RMSNorm 的对照实现」折叠块、本章问题解答折叠块）；3. 为什么敢去掉中心化（含「补充：隐式学习率自适应——梯度视角的论据」折叠块、本章问题解答折叠块）；4. 计算与实现（含「代码：与 torch 官方实现对齐，并复现 GLM 式实现」折叠块、两条本章问题解答折叠块）；来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制）。全文所有 `<details>` 折叠块均已展开阅读。

## 机械验证结果

命令与结果：

```
$ /usr/bin/python3 .dojo/scripts/validate.py wiki/rmsnorm/index.html
validation ok: wiki/rmsnorm/index.html   （exit=0）
```

- **引用编号双向闭合**：正文 `<sup>` 引用集合 = {C2,C3,C5,C6,C7,C8,C9,F1,F2,F3,F4,F5,N1,N2}；来源章节定义集合 = {C2,C3,C5,C6,C7,C8,C9,F1,F2,F3,F4,F5,N1,N2}；两个差集均为空，闭合成立。注：编号不连续（无 C1、C4），不影响闭合，仅属维护性观察。
- **相邻双上标**：检出 1 处 `<sup>[C6][N2]</sup>`（index.html:934），见问题 2。
- **Unicode 数学字符**：全文（含非代码区）扫描 U+2212 计 0、U+00D7 计 0；χ 类希腊字母/上下标/运算符计 0；非代码区仅出现 `—`、`→`（图内流程箭头）、`⌂ ◐ ☀ ↑`（导航控件图标），均非数学运算符。
- **TAB 字符**：0 个。
- **残留占位符**：`【…】` 计 0；`@content`/`@component`/`TODO`/`TBD` 计 0。
- **`<head>` 五项元数据**：`description`（纯文本、无 `$`）、`dojo:summary`（仅行内 `$…$`，`$` 配对）、`dojo:type=concept`、`dojo:topics=模型结构`（在 AGENTS.md 词表内）、`dojo:tag=归一化` 均存在且有效，validate.py 通过。
- **overview ↔ index 互链**：index.html:727 `<a href="overview.html">概览</a>`；overview.html:38 `<a href="index.html">完整说明 →</a>`，另有回家链接 `../../index.html`，双向有效。
- **前置概念链接**：正文无任何概念页 `<a>` 链接（LayerNorm 在第 1 章就地给出公式与解释），不存在失效链接；本地资源（katex/prism）引用经 validate.py 校验均存在，同页锚点全部命中 ID。
- **可运行代码实跑**：两个代码块在 torch 2.8.0（`/usr/bin/python3`）下逐一实跑，输出与页面「预期输出」**逐字符一致**（`exact match = True` ×2）。
  - 块 1 实测：`LayerNorm: mu = 2.5 , sigma = 1.118`；`输出: [-1.341641, -0.447214, 0.447214, 1.341641]`；`RMSNorm:   RMS = 2.7386`；`输出: [0.365148, 0.730297, 1.095445, 1.460593]`；`两者最大差: 0.0`。
  - 块 2 实测：`torch.nn.RMSNorm 输出: [0.365148, 0.730297, 1.095445, 1.460593]`；`GLM 式（fp32 + eps=1e-5）输出: [0.365148, 0.730296, 1.095444, 1.460593]`。
  - 复算正文声称的差值：`torch.nn.RMSNorm`(eps=0) 对手写公式 maxdiff = `1.1920928955078125e-07` ≈ 页面所写 $1.2\times10^{-7}$；GLM 式对主写公式 maxdiff = `9.5367431640625e-07` ≈ 页面所写 $9.5\times10^{-7}$；均一致。
  - 手算例复算：(1,2,3,4) 的 $\sigma=\sqrt{1.25}\approx1.1180$、$\mathrm{RMS}=\sqrt{7.5}\approx2.7386$、RMSNorm 输出均值 0.912871≈0.913、(1,−1,3,−3) 两式差 0，全部与页面一致。

### 来源逐条核对（引文依据）

来源：arXiv:1910.07467（ar5iv HTML 版逐句核对）。

| 编号 | 页面标注 | 来源原文片段（核对依据） | 结论 |
|---|---|---|---|
| [F1] | §3 Background 式 1 | `a_i = Σ_{j=1}^{m} w_{ij}x_j, y_i = f(a_i + b_i)` (1) | 一致 |
| [F2] | §3 式 2、3 | `ā_i=(a_i−μ)/σ·g_i` (2)；`μ=(1/n)Σa_i, σ=√((1/n)Σ(a_i−μ)²)` (3) | 一致 |
| [F3] | §4 式 4、5；§4.1 | `ā_i=a_i/RMS(a)·g_i, RMS(a)=√((1/n)Σa_i²)` (4)；`y=f(Wx/RMS(a)⊙g+b)` (5，general form)；§4.1 Invariance Analysis、§4.2 Gradient Analysis 均存在 | 一致 |
| [C2] | §1 与 §4 | §1：`We argue that this mean normalization does not reduce the variance of hidden states or model gradients, and hypothesize that it has little impact on the success of LayerNorm.`；§4：`we hypothesize that the re-scaling invariance is the reason for success of LayerNorm, rather than re-centering invariance.` | 一致 |
| [C3] | §4 原句 | `When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm.` | 一致 |
| [C5] | §4.2 式 8-10 | 式(8) `∂ℒ/∂g = ∂ℒ/∂v ⊙ Wx/RMS(a)`；式(9) `R = 1/RMS(a)(I − (Wx)(Wx)ᵀ/(n RMS(a)²))`；式(10) `R′ = (1/δ)R`；`the negative correlation acts as an implicit learning rate adaptor`；`Santurkar et al. 2018 ... increased smoothness of the optimization landscape` | 一致 |
| [C6] | §6.1 Table 5 | Table 5 的 M 行 RMSNorm：−0.40/−0.60/−0.69/−0.74；Baseline：−2.60/−1.19/−1.43/−1.53/−1.60；原句 `in practice it is more stable than the mean of the baseline` | 数值一致；「与 LayerNorm 相当」超出原句，见问题 3 |
| [C7] | §6.1 Figure 4 | `when the initialization center is 0.2`；`LayerNorm becomes very unstable with abnormal initialization, but RMSNorm is more robust`；`RMSNorm is similarly robust as LayerNorm, or more` | 一致 |
| [C8] | §6.1 Table 4 | RMSNorm 26.8/27.7，LayerNorm 26.6/27.7；表注 `"-" indicates that we fail to train this model`；`without which training fails` | 一致 |
| [C9] | `models/glm5_next/modeling_glm5_next.py` L66-83 与 config | 本机 transformers（`~/Library/Python/3.9/lib/python/site-packages/transformers`）`models/` 下无 `glm5_next` 目录或 `modeling_glm5_next.py`；全盘未找到该文件 | 无法定位，见问题 1 |
| [F4] | 本页推导，论文无计数 | 逐项复算 n 次加法 + n 次减法 + n 次减法 = 3n，其余项两式相同 | 推导成立 |
| [F5] | research/ 实跑（构造示例） | 见上「可运行代码实跑」 | 一致 |
| [N1] | Table 2/3/4/6/8/10；Abstract | Table 2 24.7%、Table 3(Th) 34.0%/(Py) 11.0%、Table 4 6.9%、Table 6 15.1%、Table 8 40.8%、Table 10 20.5%；Abstract `reduces the running time by 7%∼64%`；框架/硬件：Table 2 TensorFlow（标题）、Table 3 Theano/PyTorch（标题）、Table 4 Tesla V100、Table 10 RTX 2080 Ti；Table 6/8 表标题未点名框架与 GPU，但正文通用说明 `all speed-related statistics are measured on one TITAN X (Pascal)`，且 §6.2/§6.3 正文提到 Theano | 数值一致；页面表内框架/硬件标注可支持 |
| [N2] | §6.1 Table 5 | 同 [C6] | 一致 |

## 问题

- [重要·技术] index.html:750 与 index.html:1039（[C9]，及 `blockquote.meta` 的「主要依据」）：来源标注 `transformers models/glm5_next/modeling_glm5_next.py` L66-83 无法按标注定位——本机唯一安装的 transformers（`~/Library/Python/3.9/lib/python/site-packages/transformers`）`models/` 下只有 `glm`、`glm4`、`glm4_moe`、`glm4v`、`glm4v_moe`，无 `glm5_next`；全盘查找 `modeling_glm5_next*` 无结果。按 check.md §2.2 第 1/4 步，标注位置定位不到即视为未核对。｜引文依据：可支持的同类实现为 `transformers/models/glm4_moe/modeling_glm4_moe.py` L275-289 `class Glm4MoeRMSNorm`：`hidden_states = hidden_states.to(torch.float32)` → `variance = hidden_states.pow(2).mean(-1, keepdim=True)` → `hidden_states * torch.rsqrt(variance + self.variance_epsilon)` → `return self.weight * hidden_states.to(input_dtype)`，配置 `rms_norm_eps=1e-5`（`configuration_glm4_moe.py` L182）；该实现的「先转 fp32 / eps 加在根号内 / 权重乘在转回之后」与页面机制描述一致，但文件路径与 L66-83 行号无法核对。｜修复要求：给出 GLM-5.3-Flash 源码中 RMSNorm 的确切可访问路径与行号并附原文片段；若不可提供，则把 [C9] 的路径/行号改写为可定位的已验证实现（如上方 `glm4_moe` 路径与 L275-289），或将该论断降级为明确标注的推断并同步修改 `blockquote.meta` 中「主要依据」的表述。｜修复：先独立复确认路径不存在——本机 transformers 4.57.6（`~/Library/Python/3.9/lib/python/site-packages/transformers`）`models/` 下仅 `glm`、`glm4`、`glm4_moe`、`glm4v`、`glm4v_moe`、`xglm`，`find` 全盘无 `modeling_glm5_next*`、无任何 `*glm5*`。按质检规范（不保留无法定位的路径、可改引到实际存在且机制等价的实现）改引到本机可定位的等价实现：`transformers/models/glm4_moe/modeling_glm4_moe.py` L275-289 `class Glm4MoeRMSNorm`（原文：`hidden_states = hidden_states.to(torch.float32)` → `variance = hidden_states.pow(2).mean(-1, keepdim=True)` → `hidden_states * torch.rsqrt(variance + self.variance_epsilon)` → `return self.weight * hidden_states.to(input_dtype)`），配置 `transformers/models/glm4_moe/configuration_glm4_moe.py` L182 `rms_norm_eps=1e-5`。同步改写：`blockquote.meta` 主要依据、[C9] 条目、引言、§4 实现描述、简化条件；并把实现描述由「GLM-5.3-Flash」更名为「GLM 系列实现」。另新增 [C10] 承载 PyTorch `nn.RMSNorm` 的可定位来源。全文已无 `glm5_next` 与「GLM-5.3-Flash」。｜复验：`grep -c "glm5_next\|GLM-5.3-Flash\|modeling_glm5"` = 0；`sed -n '275,289p' modeling_glm4_moe.py` 与 [C9] 标注的行号、代码逐行一致；`sed -n '182p' configuration_glm4_moe.py` = `rms_norm_eps=1e-5`；[C9]、[C10] 在正文与来源章节双向闭合（正文引用集合 = 来源定义集合 = {C2,C3,C5,C6,C7,C8,C9,C10,F1,F2,F3,F4,F5,N1,N2}，两差集为空）。

- [轻微·格式] index.html:934：相邻双上标 `<sup>[C6][N2]</sup>`，不符合 style-guide §6 的组合写法（应为 `<sup>[C6, N2]</sup>`，与页面其余组合引用一致）。｜引文依据：不适用｜修复要求：改为单个上标内逗号分隔的 `<sup>[C6, N2]</sup>`。｜修复：已与问题 3 的同句改写一并完成，写为 `<sup>[C6, N2]</sup>`。｜复验：全文 `grep -c "<sup>[C6][N2]</sup>"` = 0；正则 `<sup>\[[^\]]*\]\[`（单上标内相邻方括号）命中 0；`</sup><sup>`（相邻双上标）命中 0。

- [轻微·技术] index.html:934（[C6]）：页面称 RMSNorm 激活均值「与 LayerNorm 相当」，但论文 §6.1「Effect of Normalization on Mean and Standard Deviation」只把 RMSNorm 的均值与 baseline 对比，未对 LayerNorm 的均值作相当性判断。｜引文依据：`Although the mean in RMSNorm is not normalized, in practice it is more stable than the mean of the baseline.`；`both RMSNorm and LayerNorm stabilize standard deviation`（对比对象仅为标准差，且明确是 baseline 而非 LayerNorm 的均值）；Table 5 数值 LayerNorm M≈−0.43~−0.51 与 RMSNorm M≈−0.40~−0.74 量级相近但 RMSNorm 波动更大。｜修复要求：将该句改为论文支持的表述（RMSNorm 均值比 baseline 更稳定；两者都稳定了标准差），或保留该比较但注明「由 Table 5 数值读出，论文文字未作此结论」。｜修复：按论文 §6.1 改写为来源支持的表述——「RMSNorm 的激活均值随 token 位置从 $-0.40$ 漂移到 $-0.74$，比不归一化的基线（$-2.60$ 到 $-1.19$）稳定得多」，并引论文原句「RMSNorm 的均值虽未归一化，但实践中比基线的均值更稳定」；补论文原句「RMSNorm 与 LayerNorm 都稳定了标准差」；再显式声明「论文的文字只把 RMSNorm 的均值与基线对比，未对 LayerNorm 的均值作相当性判断」。删去了原句「——不显式中心化，均值实际上也被缩放操作间接稳住了」这一无来源的因果归因。同步改写核心问题答案（§证据三）、本章问题答案与来源条目 [C6] 的同义表述。｜复验：ar5iv HTML §6.1 逐句核对，原句 `Although the mean in RMSNorm is not normalized, in practice it is more stable than the mean of the baseline.`、`Due to their normalization properties, both RMSNorm and LayerNorm stabilize standard deviation.` 均命中，且该节无任何比较 RMSNorm 均值与 LayerNorm 均值的句子；Table 5 M 值为 LayerNorm −0.43/−0.48/−0.50/−0.50/−0.51、RMSNorm −0.40/−0.60/−0.69/−0.74/−0.73，与「随 token 位置从 $-0.40$ 漂移到 $-0.74$」一致；全文已无用于均值的「与 LayerNorm 相当」。

- [轻微·技术] index.html:977：「PyTorch 的 `nn.RMSNorm` 与 GLM 都这么做，位置与默认值各不相同」——该实现断言未标注来源，且「位置……各不相同」与事实不符：二者都把 $\varepsilon$ 加在根号内。｜引文依据：`torch.nn.RMSNorm` 文档给出 $\mathrm{RMS}(x)=\sqrt{\varepsilon+\frac{1}{n}\sum x_i^2}$（`torch/nn/modules/normalization.py`，`eps` 默认 `torch.finfo(x.dtype).eps`），GLM 用 `torch.rsqrt(variance + eps)`（`glm4_moe` L288），两者位置相同、仅默认值不同（PyTorch 随 dtype 取 finfo，GLM 为 $10^{-5}$）。｜修复要求：删去「位置……各不相同」或改为「两者的 $\varepsilon$ 都在根号内，仅默认值不同」，并为 PyTorch 行为补可定位来源（本机 torch 2.8.0 源码可定位）。｜修复：删去「位置与默认值各不相同」，改为「PyTorch 的 `nn.RMSNorm` 与 GLM 都把它加在根号内，仅默认值不同」（PyTorch $\mathrm{RMS}(x)=\sqrt{\varepsilon+\frac{1}{n}\sum_{i=1}^{n}x_i^2}$、默认 `torch.finfo(x.dtype).eps`；GLM `torch.rsqrt(var + eps)`、rms_norm_eps $=10^{-5}$）。新增来源条目 [C10] 承载 PyTorch 行为，并在 §4 正文、常见误解、本章问题答案与 [C9] 处补上 `<sup>[C9, C10]</sup>` 引用。｜复验：本机 torch 2.8.0 `torch/nn/modules/normalization.py` 的 `class RMSNorm` docstring 确认 $\mathrm{RMS}(x)=\sqrt{\epsilon+\frac{1}{n}\sum x_i^2}$（ε 在根号内）与 `eps: ... Default: torch.finfo(x.dtype).eps`；`glm4_moe` L288 `torch.rsqrt(variance + self.variance_epsilon)` 同样 ε 在根号内；[C10] 正文与来源双向闭合。

- [轻微·技术] index.html:753（引言）：「今天主流大模型（包括 GLM-5.3-Flash）的归一化层用的都是它」为无来源的适用范围扩大陈述，且与页面「简化条件及其限制」中「只引用论文实验与 GLM 实现事实，不自行外推」的声明不一致。｜引文依据：不适用（无对应来源；论文不支持「主流大模型普遍采用」这一范围性结论）｜修复要求：补充可定位的采用情况来源，或收窄为可核实表述（如「RMSNorm 已被 GLM-5.3-Flash 等模型的实现采用」）。｜修复：收窄为可核实表述——「这个删减版就是 RMSNorm，已被 GLM 系列模型的官方实现采用<sup>[C9]</sup>」，去掉「今天主流大模型……的归一化层用的都是它」这一无来源的范围性结论；[C9] 现指向本机可定位的 `models/glm4_moe/modeling_glm4_moe.py` L275-289。｜复验：该句不再含范围性结论；`grep -c "主流大模型"` = 0；[C9] 双向闭合。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（核心结论、公式、推导与全部数值来源均与论文一致，两个可运行代码块逐字符复现；无阻断问题。问题 1 为来源标注无法定位，须先修复再进入下一轮；问题 2–5 为格式与表述层面的轻微问题，可在同一轮一并修复。）
