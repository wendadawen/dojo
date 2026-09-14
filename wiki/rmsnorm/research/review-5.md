<!-- review-meta
round: 5
page: wiki/rmsnorm/index.html
reviewed_content_sha256: 02bc6f3fc971e8f6
-->
# RMSNorm 审查记录（第 5 轮）

- 页面版本：index.html 工作树哈希 2c46f4834bc53f61189cf4c753d00a2af0592702
- 审查时间：2026-09-14 17:11
- 审查者：编排者派发的独立审查者（未参与写作与前序审查）
- 已完整阅读章节（按顺序）：引言 → 核心问题（页面级）→ 常见误解 → 1. 归一化的对象：神经元的加权和（含本章问题）→ 2. RMSNorm：只保留缩放（含代码折叠块、本章问题）→ 3. 为什么敢去掉中心化（含「补充」折叠块、本章问题）→ 4. 计算与实现（含速度表、代码折叠块、本章问题）→ 来源与范围说明 → head 元信息与页内脚本
- 适用规范：guides/concept/check.md（head 中 `dojo:type=concept`）
- 来源获取：arXiv:1910.07467 v1 全文（PDF 抽取文本）；本机 torch 2.8.0 `torch/nn/modules/normalization.py`；本机 transformers 4.57.6 `models/glm4_moe/modeling_glm4_moe.py` 与 `configuration_glm4_moe.py`

## 逐条核对（来源论断与引文依据）

- [C1] §1 “We argue that this mean normalization does not reduce the variance of hidden states or model gradients, and hypothesize that it has little impact on the success of LayerNorm.”；§4 “we hypothesize that the re-scaling invariance is the reason for success of LayerNorm, rather than re-centering invariance.” —— 与正文「假设」段一致；页面明确「假设+证据、不是定理」。
- [C2] §4 “When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm.” —— 正文第 2 章与第 3 章证据一一致。
- [C3] Table 1 两行：LayerNorm `3,3,7,3,7,3`、RMSNorm `3,7,7,3,7,3`，仅在 “Weight matrix re-centering” 一列不同 —— 与正文「只少权重矩阵 re-centering，其余缩放不变性保留」一致。
- [C4] §4.2 “the gradient of g is proportional to the normalized summed inputs, rather than raw inputs.”；“the negative correlation acts as an implicit learning rate adaptor”；援引 Santurkar et al.[23] “does not come from the added stability to layer inputs, but due to increased smoothness of the optimization landscape” —— 与折叠块内容一致；式 8-10 编号正确。
- [C5] §6.1 “Although the mean in RMSNorm is not normalized, in practice it is more stable than the mean of the baseline.”；“both RMSNorm and LayerNorm stabilize standard deviation.” —— 正文「证据三」只比基线、未对 LayerNorm 均值作相当性判断，处理准确。
- [C6] §6.1 “change the center of weight initialization to 0.2 … RMSNorm is more robust”；结论 “RMSNorm is similarly robust as LayerNorm, or more.” —— 与「异常初始化下表现不差于 LayerNorm 甚至更稳」一致。
- [C7] Table 4：LayerNorm 26.6/27.7、RMSNorm 26.8/27.7、Baseline “-”（训练失败）—— 与正文 BLEU 26.8/27.7 对 26.6/27.7 及「无归一化训练失败」一致。
- [C8] `modeling_glm4_moe.py` L275 `class Glm4MoeRMSNorm`，forward 内 `hidden_states.to(torch.float32)` → `variance = hidden_states.pow(2).mean(-1, keepdim=True)` → `hidden_states * torch.rsqrt(variance + self.variance_epsilon)` → `return self.weight * hidden_states.to(input_dtype)`；`configuration_glm4_moe.py` L182 `rms_norm_eps=1e-5` —— 页码与正文「fp32 计算、权重乘在转回后、eps 加在根号内、默认 1e-5」全部吻合。
- [C9] torch 2.8.0 `normalization.py::RMSNorm` docstring 公式 `RMS(x)=\sqrt{\epsilon+\frac{1}{n}\sum_i x_i^2}`，且 “eps: … Default: `torch.finfo(x.dtype).eps`” —— 与正文一致。
- [N1] Table 2/3/4/6/8/10 RMSNorm 加速：24.7% / 34.0% / 11.0% / 6.9% / 15.1% / 40.8% / 20.5%（7 行表逐行吻合）；范围 6.9%~40.8%；摘要 “reduces the running time by 7%∼64%”；Table 8 pRMSNorm 4.34s 对 LayerNorm 12.02s ≈ 63.9% —— 页面「连同 pRMSNorm 计为 7%~64%」的归因正确。实验条件（TITAN X / V100 / RTX 2080 Ti；TensorFlow/Theano/PyTorch）与 §6 一致。
- [N2] Table 5：RMSNorm 均值 -0.40/-0.60/-0.69/-0.74/-0.73，基线 -2.60/-1.19/-1.43/-1.53/-1.60 —— 页面取值区间与来源一致。
- [F1] 式 (1)；[F2] 式 (2)(3)；[F3] 式 (4)(5) —— 页面公式与编号逐一对上；「v1 式 (3)(4) 无 ε」经原文核对属实。
- [F4] 3n 计数按页面口径复算：求 μ 的 n 次加法 + 方差偏差的 n 次减法 + 归一化分子的 n 次减法 = 3n，其余（n 次平方、n 次求和、开方、除法、增益乘）两式相同 —— 推导成立。
- [F5] 两段代码块在本地实跑：第一段输出与页面「预期输出」逐位一致（mu 2.5 / sigma 1.118 / RMS 2.7386 / 零均值最大差 0.0）；第二段 `torch.nn.RMSNorm` 输出 [0.365148, 0.730297, 1.095445, 1.460593]、GLM 式输出 [0.365148, 0.730296, 1.095444, 1.460593]，误差 `1.1920928955078125e-07` 与 `9.5367431640625e-07`，与页面所写 1.2e-7 / 9.5e-7 一致。
- 计算示例 (1,2,3,4)：μ=2.5、σ=√1.25≈1.1180、输出 ±1.3416/±0.4472、RMS=√7.5≈2.7386、输出均值≈0.913 —— 全部复算正确。

## 可读性、表述与格式

- 术语（加权和、re-centering/re-scaling、不变性、pRMSNorm）首次出现即解释；前置概念「残差连接」有链接且目标页 `wiki/residual-connection/index.html` 真实存在，无「（待生成）」占位。
- 页面级「核心问题」4 条、各章「本章问题」（1/1/1/2 条）均有 `解答：` 折叠块，答案独立可读、与正文结论一致；核心问题答案均指明完整论证所在章节。折叠块收起后正文仍能建立结论。
- 通读全文（含折叠块与图注）未发现会话指代（我/我们/你——正文唯一的「我们」出现在被明确标注为「原文的表述是「…」」的论文译文引文中，非页面自称）、调试叙事、临场评价或 AI 拼接腔；`本页` 自称共 3 处（L269/L330/L362），符合 style-guide §12「自称使用「本页」或「本文」」。
- 图注与图示：结构图为 HTML（`dg-stack`）而非等宽框线，公式经 KaTeX 渲染；无 `alt` 含 `$...$`；无 Unicode 数学字符；无交互视图依赖脚本。
- 格式一致性：h2 编号连续（1–4 + 不编号的「来源与范围说明」）、h3 均为固定「本章问题」与来源小节的固定命名；h2 副标题用「：」（全仓通用写法，如 positional-encoding「1. 现象：…」）；details summary 前缀为「解答：/代码：/补充：」，符合 style-guide §5/§9。
- `dojo:topics=模型结构`、`dojo:tag=网络结构` 均在 `.dojo/scripts/catalog_builder.py` 的 `ALLOWED_TOPICS` / `ALLOWED_TAGS` 内；`description` 为纯文本、`dojo:summary` 公式可渲染；`index.html` 与 `overview.html` 相互链接。
- `.dojo/scripts/validate.py wiki/rmsnorm/index.html` 返回 `validation ok`（退出码 0）。

## 问题

（无）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布。本轮所有来源论断均有可定位依据并已写下原文片段或关键数值；两段可运行代码实跑输出与页面一致；公式可复算、符号单义、summary 公式可渲染；未发现元话语、会话指代、同页数字/引文编号不一致或图注读数与来源不符的情况。