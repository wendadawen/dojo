<!-- review-meta
round: 3
page: wiki/rmsnorm/index.html
reviewed_content_sha256: a0b5f46a75d0e016
-->
# RMSNorm 审查记录（第 3 轮）

- 页面版本：8ff504b6d1e530d060c17859316483fae9a9e61e
- 审查时间：2026-09-13 19:2x
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：引言 / 核心问题 / 常见误解 / 1. 归一化的对象：神经元的加权和 / 2. RMSNorm：只保留缩放 / 3. 为什么敢去掉中心化 / 4. 计算与实现 / 来源与范围说明（含全部折叠块、图注与表格）
- 核对来源：arXiv:1910.07467 v1 PDF（14 页）与 ar5iv 全文（Table 1–10、Eq 1–10、§1/§3/§4/§4.1/§4.2/§6.1 原句）；transformers 4.57.6 `models/glm4_moe/modeling_glm4_moe.py` L275-289 与 `configuration_glm4_moe.py` L182；本机 torch 2.8.0 `torch/nn/modules/normalization.py` `class RMSNorm`；页面两段代码实跑复算。

## 核对结论摘要（无阻断）

- 代码块一（LayerNorm/RMSNorm 对照）：实跑输出与「预期输出」逐字符一致；零均值差 0.0。
- 代码块二（torch 对齐 + GLM 式）：实跑 `[0.365148, 0.730297, 1.095445, 1.460593]` / `[0.365148, 0.730296, 1.095444, 1.460593]`，与页面一致；实测差 torch 对齐 1.192e-7、GLM 式 9.537e-7，页面写 1.2e-7 / 9.5e-7，相符。
- 速度表 7 行逐项核对 Table 2/3/4/6/8/10：24.7% / 34.0% / 11.0% / 6.9% / 15.1% / 40.8% / 20.5% 全部一致；硬件「TITAN X（默认）/ V100（Table 4）/ 2080 Ti（Table 10）」与原文相符；摘要 7%~64% 原句存在。
- Table 5 原值：RMSNorm M = -0.40→-0.74，Baseline M = -2.60→-1.19；BLEU 26.8/27.7 对 26.6/27.7；Figure 4「center 0.2」；「without normalization training fails」——全部与页面相符。
- Eq 编号：v1 中 Eq(1) 加权和、Eq(2) LayerNorm、Eq(3) μ/σ、Eq(4) RMSNorm、Eq(5) 一般形式、Eq(8-10) 梯度分析，均与页面 [F1]-[F3]、[C5] 相符；v1 全文无 ε。
- `3n` 推导自洽（μ 的 n 次加法 + 方差 n 次减法 + 归一化分子 n 次减法 = 3n）。
- GLM 源码 L275-289 逐行核对：fp32 转换、`rsqrt(var+eps)`、权重乘在转回后；config L182 `rms_norm_eps=1e-5`。PyTorch `nn.RMSNorm` 文档公式 `sqrt(ε+1/nΣx²)`、默认 `torch.finfo(x.dtype).eps`。均相符。
- `python3 .dojo/scripts/validate.py wiki/rmsnorm/index.html` → `validation ok`。

## 问题

- [重要·技术] 「2. RMSNorm：只保留缩放」line 172（并见 line 76 核心问题答案、line 106 常见误解、line 231/83 本章问题）：把「LayerNorm 的输出均值恒为 0」写成无条件论断。但页面 line 130 自己把 $g_i$ 定义为逐维可学习增益，而 $\bar a_i=(a_i-\mu)g_i/\sigma$ 的均值 $=\frac{\sum(a_i-\mu)g_i}{n\sigma}$，只有在 $g$ 各维相同时才为 0。构造 $a=(1,2,3,4)$、$g=(1,2,3,4)$ 实算，LayerNorm 输出为 $(-1.3416,-0.8944,1.3416,5.3666)$，均值 $1.118\ne0$。｜引文依据：页面 line 76「使输出零均值」、line 172「LayerNorm 的输出均值恒为 0」、line 106「LayerNorm 输出零均值」；对照实测均值 1.118034。｜修复要求：把「恒为 0」限定为「增益 $g$ 各维相同时（本页对照取 $g=1$）」；或改写为「归一化后的值 $(a_i-\mu)/\sigma$ 均值恒为 0，乘逐维增益 $g_i$ 后一般不再为 0」。四处表述一并改，保持与该页 $g_i$ 定义一致。｜修复：｜复验：

- [重要·来源] 「来源与范围说明」line 361 [F5]：来源依据写作「research/ 实跑输出（构造示例）」，但本页 research/ 下的实测产物已按 research/measured.md 登记后从仓库移除，该指针已无处可打开，违反 check.md §2.2「打开来源，定位到页面标注的位置」与 §5「每条来源论断都有引文依据记录」。｜引文依据：research/measured.md「本页的实测产物原先存放在本目录下，现已从仓库移除」；页面 [F5] 登记 `rms_page_code.py/out`、`concept_probes.py/out` 均已删除。｜修复要求：把 [F5] 的来源改写为可定位的当前依据——即本页两段代码块（实跑可复现，本轮已复算：torch 对齐差 1.2e-7、GLM 式差 9.5e-7、零均值差 0.0），或指向 research/measured.md 的登记。删除对已移除 research/ 产物的引用。｜修复：｜复验：

- [轻微·技术] 「3. 为什么敢去掉中心化」line 90 核心问题第 3 题答案：「RMSNorm 完整保留各类 re-scaling 不变性（只放弃 re-centering 类性质）」表述过宽。论文 Table 1 中 RMSNorm 的「Weight vector re-scaling」为 ✗（与 LayerNorm 相同），并非保留「各类」re-scaling。｜引文依据：Table 1 行 LayerNorm = ✓ ✓ ✗ ✓ ✗ ✓、RMSNorm = ✓ ✗ ✗ ✓ ✗ ✓（列序：Weight matrix re-scaling / re-centering / Weight vector re-scaling / Dataset re-scaling / re-centering / Single case re-scaling）。｜修复要求：改为「完整保留 LayerNorm 所具备的权重矩阵缩放、数据集缩放、单样本缩放不变性（只放弃权重矩阵 re-centering）」，与 line 244 证据二的精确表述一致。｜修复：｜复验：

- [轻微·格式] 「来源与范围说明」论断与来源（C）小节断号：条目从 [C2] 开始，[C1]、[C4] 全页既未在正文引用也无对应条目（正文实引 [C2][C3][C5][C6][C7][C8][C9][C10]）。｜引文依据：不适用。｜修复要求：补回缺失编号或把 C 编号连续重排，使正文上标与来源条目一一对应（style-guide §6）。｜修复：｜复验：

- [轻微·来源] 「3. 为什么敢去掉中心化」line 244「证据二：不变性的完整保留」整段（Table 1 不变性对照）没有任何 `<sup>[Cx]</sup>` 上标，也未在来源章节落条目；其余证据均有标注（证据三 [C6, N2]、证据四 [C8]、补充 [C5]）。｜引文依据：Table 1 原文「Invariance properties of different normalization methods」；页面该段无上标。｜修复要求：为该段补一个来源条目（如 [C4]）并在段末标注上标，或在 [F3] 中明确「§4.1 不变性分析（Table 1）」并在正文引用。｜修复：｜复验：

- [轻微·表述] line 65 引言末句「本文讲清它删了什么、为什么敢删、省了多少，以及实现里两处与论文公式不同的细节。」属 check.md §2 第 12 项所禁元话语（「本页将…」句式），且与 line 67「结构：先…」的路线图功能重复。｜引文依据：不适用。｜修复要求：删除该预告句，或将范围陈述并入 line 67 的路线图一句，避免「本文将讲清…」式元话语。｜修复：｜复验：

- [轻微·表述] line 385「大语言模型场景的『质量相当』由后续广泛采用间接支持」把「场景」当术语使用，属 check.md §2 第 12 项「AI 拼接腔」。｜引文依据：不适用。｜修复要求：改为「大语言模型上的『质量相当』由后续广泛采用间接支持」之类不含「场景」作术语的表述。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复（无阻断；两条重要问题为「无条件论断漏条件」与「来源指针指向已移除产物」，均限于局部改写即可关闭；轻微问题逐条修复）。核心结论、公式、数字、代码输出与全部速度/质量表格均已回到来源核对通过。
