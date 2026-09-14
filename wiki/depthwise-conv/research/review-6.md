<!-- review-meta
round: 6
page: wiki/depthwise-conv/index.html
reviewed_content_sha256: dbdf5c111522324f
-->
# 深度可分离卷积审查记录（第 6 轮）

- 页面版本：bd0cb42ca277c3716ebe118c2556e34d8abc8495
- 审查时间：2026-09-14 16:50
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 / 常见误解 / 1. 标准卷积——滤波与组合一步完成 / 2. 拆成两步——depthwise 与 pointwise / 3. 省了多少——比值推导 / 4. 大模型里的用法——KDA 的短深度卷积 / 来源与范围说明（含全部折叠解答块、两段代码块与图注）

## 问题

- [重要·技术] 图（第 211–213 行，SVG 右下角 foreignObject 标签「①只在单通道内滤波，②只在通道间组合，两步各做一件事」）：标签文本宽度超出 SVG 视口，渲染时末字「事」被裁掉，读者看到的是残句「…两步各做一件事」缺尾，实际显示为「…两步各做一件」。｜引文依据：无头 Chrome 实测（Dump-DOM 取 getBoundingClientRect + Range）——窗口 1100px 时 `.diagram svg` 包围盒 right=1040.0，该 foreignObject 文本 right=1063.0（超出 23px）；窗口 1400px 时 SVG right=1247.5、文本 right=1273.0（同样超出 25.5px）。因 viewBox 宽 680、标签自 x=360 起、文本宽约 335.6 单位、末端 695.6>680，超出量在视口坐标系恒为 15.6 单位，与窗口宽度无关，故任何屏宽下都截断；截图放大（1100px 与 1400px 两次）末端均止于「件」。｜修复要求：使该标签文本末端落入 viewBox 宽度内——将 viewBox 宽由 680 增至 ≥700（并同步调整右侧竖条/箭头/底部标签的坐标与宽度），或把该标签起点左移、拆成两行；修复后用无头浏览器在 1100px 与 1400px 两种宽度各截一次图，确认完整显示「…两步各做一件事」且无其他标签被裁。｜修复：采用「增大 viewBox」方案：`<svg>` 的 `viewBox="0 0 680 270"` 改为 `viewBox="0 0 710 270"`，并把该标签的 `foreignObject` 由 `x="360" width="310"` 改为 `x="360" width="345"`（右端 705），使标签文本末端（用户单位 695.9）落入 viewBox 内并留约 14 单位右边距。右侧竖条/箭头/底部标签的坐标**无需调整**——复核全部元素：右列最右矩形右端 580、其余标签文本最右 586.2（FO#8），均原本就落在 680 以内，唯一越界项就是本标签，故未改动任何其他坐标。改动仅 2 行属性。｜复验：无头 Chrome 实测（1100px 与 1400px 两档，window 900 高）：两档下页面内全部 10 个 foreignObject 的文本右端均 ≤ 710（本标签两档均为 695.9，`CLIPPED=false`），其余标签最右 586.2，无被裁项；两档各截图并放大确认完整显示「①只在单通道内滤波，②只在通道间组合，两步各做一件事」含末字「事」。`python3 .dojo/scripts/validate.py wiki/depthwise-conv/index.html` 返回 `validation ok`；改动未触及正文、数字、引文编号与 summary/overview，故无需同步。｜

## 来源核对（本轮实测片段）

- MobileNets（arXiv:1704.04861）§3.1 原文：式(2) `DK · DK · M · N · DF · DF`；式(3) `Ĝk,l,m = Σi,j K̂i,j,m · Fk+i−1,l+j−1,m`；式(4) `DK · DK · M · DF · DF`；式(5) `DK · DK · M · DF · DF + M · N · DF · DF`；约分结果 `1/N + 1/DK²`；「MobileNet uses 3 × 3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy」——与页面 [C1][C2][C3]、[F1]–[F5]、[N1] 一致。§3.1 亦含「A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers」——与 [C1] 一致。
- 同论文 Table 1（原文）「Table 4. Depthwise Separable vs Full Convolution MobileNet」：`Conv MobileNet 71.7% 4866 29.3`、`MobileNet 70.6% 569 4.2`——与页面第 398 行「71.7% 对 70.6%、4866M 对 569M、29.3M 对 4.2M」逐项一致（4866/569=8.55，对应「约 8.5 倍」）。
- transformers `main` 的 `src/transformers/models/glm5_next/modeling_glm5_next.py`：L599 `self.qkv_dim = self.head_dim * self.num_heads`；L610 `self.conv_dim = self.qkv_dim * 3`；L611–618 `nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1)`；L640–648 先 `q_proj/k_proj/v_proj` 再 `torch.cat([...], dim=-1)`；L396–412 `causal_conv1d_fn` 内 `padding = weight.shape[-1] - 1` 与 `)[:, :, :seq_len]`——与 [C4] 及第 4 章正文（24576、kernel 4、groups=通道数、无偏置、因果、q/k/v 投影在前）逐条一致。
- zai-org/GLM-5.3-Flash `config.json` 的 `linear_attn_config`：`num_heads: 64`、`head_dim: 128`、`short_conv_kernel_size: 4`——推出 8192 与 24576；`text_config.qk_rope_head_dim: 0`——支撑「主干无位置编码」这一标注为结构解读的说法。
- checkpoint `model.safetensors.index.json`：存在 `model.language_model.layers.N.self_attn.{q,k,v}_conv1d.weight`——与 [N3]「checkpoint 拆为 q/k/v_conv1d 三个 [8192,1,4] 张量」一致。
- 代码实跑（python3 3.9 + torch）：成本计数块输出与「预期输出」逐字一致（9,437,184 / 147,456 / 1,048,576 / 1,196,032 / 0.126736 / 0.126736 / 7.9x）；因果卷积块输出 `[-1.0, -2.5, -3.5, -3.5, -3.5]`、通道 0 `[1.0, 3.0, 5.0, 7.0]`、通道 1 `[10.0, 20.0, 30.0, 40.0]`，与手算注释逐一吻合。
- 机械项：`../kda/index.html`、`../glm-5-3-flash-dataflow/index.html` 均真实存在；`.dojo/scripts/validate.py wiki/depthwise-conv/index.html` 返回 `validation ok`；`dojo:topics=模型结构` 属词表大类，`dojo:tag=网络结构` 在 ALLOWED_TAGS 内；`dojo:summary` 的 `$...$` 均为合法 KaTeX；标题、正文、列表、表格内无 Unicode 数学字符；无 `alt` 属性含 `$`；无「（待生成）」占位；overview.html 与 index.html 互链。
- 表述维度：通读全文（含四段折叠解答、两段代码块、图注）未见元话语（「本页」「下面来看」「需要注意的是」）、会话指代（我/我们/你）、调试叙事、临场评价或 AI 拼接腔；构造示例均标注为构造（[N2]、「构造示例」小节），非源码直陈的解读均标注「（结构解读，非源码直陈）」。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 0
- 处置：修复（修复上图注标签截断后即可发布）