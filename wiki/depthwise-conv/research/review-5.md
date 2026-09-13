<!-- review-meta
round: 5
page: wiki/depthwise-conv/index.html
reviewed_content_sha256: f5b5a68b29e61abe
-->
# 深度可分离卷积审查记录（第 5 轮）

- 页面版本：index.html 工作树哈希 `e1d42f0849d4cba90093054d6c617d3e4f59ec8a`（内容指纹 `f5b5a68b29e61abe`）
- 审查时间：2026-09-13 21:08 CST
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节（按顺序）：核心问题（4 题及解答折叠块）、常见误解、1. 标准卷积——滤波与组合一步完成（含公式、符号表、成本示例、本章问题）、2. 拆成两步——depthwise 与 pointwise（含两步公式、结构图与图注、本章问题）、3. 省了多少——比值推导（含约分推导、代码折叠块「成本计数与比值验证」、本章问题）、4. 大模型里的用法——KDA 的短深度卷积（含源码配置、代码折叠块「因果 1D depthwise 卷积与 groups 语义」、本章问题）、来源与范围说明（论断与来源 C / 公式与来源 F / 外部数字与实验条件 N / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）；overview.html 全部小节

## 来源核对记录（逐条摘录引文依据）

- **[F2] 标准卷积成本**：MobileNets（arXiv:1704.04861）§3.1 原文成本式 "DK·DK·M·N·DF·DF"。页面 `$D_K\cdot D_K\cdot M\cdot N\cdot D_F\cdot D_F$` 一致。
- **[F3]/[F4] depthwise 与 pointwise 成本**：同节，depthwise "DK·DK·M·DF·DF"、pointwise "M·N·DF·DF"，两步相加即页面合计式。
- **[F5] 缩减比**：同节给出 "1/N+1/DK²"。页面推导逐项约分（`$D_K^2MD_F^2$`→`$1/N$`；`$MND_F^2$`→`$1/D_K^2$`）与之一致。
- **[C1] 一步完成滤波与组合**：§3.1 原句 "A standard convolution both filters and combines inputs into a new set of outputs in one step."。页面「滤波与组合一步完成」与之逐字对应。
- **[C2]/[F3] depthwise 单滤波器单通道**：§3.1 原句 "For MobileNets the depthwise convolution applies a single filter to each input channel."。
- **[C2] pointwise 1×1 跨通道组合**：§3.1 原句 "The pointwise convolution then applies a 1×1 convolution to combine the outputs the depthwise convolution."。
- **[C3]/[N1] 8~9 倍**：§3.1 原句 "MobileNet uses 3×3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy."。页面「计算省 8 到 9 倍、精度只有小幅下降」逐字对应。
- **[简化条件] Table 4 数字**：论文 Table 4（§4 Experiments，4.1 Model Choices 小节）"Conv MobileNet 71.7% / 4866M / 29.3M" 对 "MobileNet 70.6% / 569M / 4.2M"。页面的 71.7%/70.6%、4866M/569M、29.3M/4.2M 与顺序均一致；4866/569=8.55≈8.5 倍、71.7−70.6=1.1 个百分点，页面自算正确。
- **[C4] KDA conv 配置**：transformers main `src/transformers/models/glm5_next/modeling_glm5_next.py` L606-618 —— `self.q_proj/k_proj/v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)`；`self.conv_dim = self.qkv_dim * 3`；`self.conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1)`；L645-652 `mixed_qkv = torch.cat([q_proj, k_proj, v_proj], dim=-1)`。页面「通道数 24576、kernel 4、groups 等于通道数、无偏置、q/k/v 三路拼接后过一个一维卷积」逐项对上。
- **[C4] 维度 4096→8192**：同文件 L596-608 `hidden_size=config.hidden_size`、`num_heads=config.linear_num_heads`、`head_dim=config.linear_head_dim`、`qkv_dim=head_dim*num_heads`；`configuration_glm5_next.py` 默认 `hidden_size=4096`、`linear_head_dim=128`、`linear_num_heads=64`、`linear_conv_kernel_dim=4`。故 64×128=8192、8192×3=24576，页面「24576（=3×8192）」「4096→8192」成立。
- **[C4]/本章问题] 因果实现**：同文件 L395-415 `def causal_conv1d_fn(...)`：`padding = weight.shape[-1] - 1`，`out = F.conv1d(hidden_states..., weight=weight.unsqueeze(1), bias=bias, padding=padding, groups=hidden_size)[:, :, :seq_len]`。与页面所写 `F.conv1d(padding=k-1)` 后取 `[:, :, :seq_len]`、行号 L396-412 一致（文件在 main 上确为 396–412 行）。
- **[C5] 跨通道由 q/k/v 投影承担**：源码只有「先投影后卷积」的顺序事实，无「投影承担跨通道混合」的语句；页面已在该句后标「结构解读，非源码直陈」，处理正确。
- **[C3]/[F5] 构造数字例与比值**：见下「复算」小节，独立复算全部吻合。

### 复算与实跑（机械核对）

- 成本式复算：9×64×64×256=9,437,184；9×64×256=147,456；64×64×256=1,048,576；合计 1,196,032；比值 1,196,032/9,437,184=0.126736=1/64+1/9；缩减 9,437,184/1,196,032=7.9。与页面正文、[N2] 与代码「预期输出」完全一致。
- 导语按位置计数复算：9×64×64=36,864；9×64+64×64=4,672≈4,700；比值 0.126736→约 12.7%。与正文一致。
- §3 本章问题上界复算：收益 = 1/(1/N+1/D_K^2) < min(N, D_K^2) = min(64,9)=9，页面「不超过 $N$ 与 $D_K^2$ 中较小者的值」成立。
- 代码块 1「成本计数与比值验证」实跑（python3）：`标准卷积乘加: 9,437,184 / depthwise 乘加: 147,456 / pointwise 乘加: 1,048,576 / 深度可分离合计: 1,196,032 / 比值 sep/std = 0.126736   公式 1/N + 1/D_K^2 = 0.126736 / 缩减倍数: 7.9x`，与页面「预期输出」逐行相同。
- 代码块 2「因果 1D depthwise 卷积与 groups 语义」实跑（torch 2.8.0）：`torch 输出: [-1.0, -2.5, -3.5, -3.5, -3.5]`、`groups=2 通道 0（核 [1,1]）: [1.0, 3.0, 5.0, 7.0]`、`groups=2 通道 1（核 [0,1]）: [10.0, 20.0, 30.0, 40.0]`，与页面「预期输出」及「观察重点」中的手算、单调不增（−1→−3.5）、t≥2 起恒定 −3.5 的说明逐项一致。
- 页面功能：`.dojo/scripts/validate.py wiki/depthwise-conv/index.html` → `validation ok`（锚点、重复 id、本地引用、模板残留均通过）；正文引用类（dg-label/dg-label-center/dg-box/dg-accent/diagram/diagram-caption/code-details/code-block/chapter-questions/learning-goals/misconceptions）在 `libs/dojo-concept.css` 中均存在。
- 前置链接：`../kda/index.html`、`../glm-5-3-flash-dataflow/index.html` 均真实存在；正文与 `overview.html` 互相链接。无「（待生成）」占位。
- 公式书写：标题/summary/正文/列表/表格中无 Unicode 数学字符（×、≈、² 等均未直接出现），数学全由 KaTeX 渲染；结构图为内联 SVG，公式写在 `<foreignObject>` 内；`<img>`/`aria-label` 中无 `$...$`。
- head 字段：`description`（纯文本）、`dojo:summary`（可渲染）、`dojo:type=concept`、`dojo:topics=模型结构`（在 AGENTS.md 固定大类内）、`dojo:tag=网络结构`（在 ALLOWED_TAGS 内）均有效。
- 正文与 summary/overview 数字交叉核对：乘加量、比值、8~9 倍、KDA 配置（24576 / kernel 4 / 无 pointwise / q/k/v 投影承担跨通道）、简化条件（步长 1、正方形核与特征图）三处表述一致，无互相矛盾。

## 问题

- [轻微·表述] §1「标准卷积」成本示例段（`计算示例（贯穿本页的构造数字）`）：括号内「贯穿本页」是对页面自身结构的元话语与「本页」自我指代，属规范列举的不合格表述。｜引文依据：不适用｜修复要求：删去或改写该自指短语，去掉「本页」二字（例如改为「两章共用的构造数字」或直接删除整个括号），不改变其后数字。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（唯一轻微问题为表述用词，不影响正确性与主线理解；本轮无阻断、无重要，第 5 轮无需追加修复）
