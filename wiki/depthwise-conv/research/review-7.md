<!-- review-meta
round: 7
page: wiki/depthwise-conv/index.html
reviewed_content_sha256: dbdf5c111522324f
-->
# 深度可分离卷积审查记录（第 7 轮）

- 页面版本：05d7f1855cc97d3a0e977e569316bc56fffa4f4f（`git hash-object wiki/depthwise-conv/index.html`）
- 审查时间：2026-09-14 17:39
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查与修复；仅读 `wiki/depthwise-conv/index.html`、`overview.html`、页面引用的外部来源、`guides/concept/check.md` 与 `guides/concept/style-guide.md`）
- 已完整阅读章节（含折叠块、图注、来源章节，按顺序）：核心问题 → 常见误解 → 1. 标准卷积——滤波与组合一步完成 → 本章问题 → 2. 拆成两步——depthwise 与 pointwise → 本章问题 → 3. 省了多少——比值推导 → 本章问题 → 4. 大模型里的用法——KDA 的短深度卷积 → 本章问题 → 来源与范围说明

## 来源核对（引文依据）

- **MobileNets（arXiv:1704.04861）**：abs 页显示最新版即 v1，故按 **v1** 逐条核对。§3.1 原句“uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy”；成本式 Eq.2 “D_K·D_K·M·N·D_F·D_F”、Eq.4 “D_K·D_K·M·D_F·D_F”、Eq.5 “D_K·D_K·M·D_F·D_F + M·N·D_F·D_F”，比值式“(D_K·D_K·M·D_F·D_F + M·N·D_F·D_F)/(D_K·D_K·M·N·D_F·D_F)” 化简为 “1/N + 1/D_K^2”。§4.1 Table 4（Depthwise Separable vs Full Convolution MobileNet）：Conv MobileNet 71.7% / 4866 Mult-Adds / 29.3M 参数；MobileNet 70.6% / 569 / 4.2。页面 [C1]–[C3]、[F1]–[F5]、[N1] 与“简化条件”第 2 条所述数字逐项吻合；4866/569=8.55≈“约 8.5 倍”、71.7−70.6=1.1 个百分点，均与页面一致。
- **GLM-5.3-Flash（transformers main，`src/transformers/models/glm5_next/`）**：`modeling_glm5_next.py` L396 `def causal_conv1d_fn(...)`，函数体内 `padding = weight.shape[-1] - 1` 与 `F.conv1d(..., padding=padding, groups=hidden_size)[:, :, :seq_len]`（对应页面第 350 行与 [C4] 的 L396-412）；L610 `self.conv_dim = self.qkv_dim * 3`；L611-618 `nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1)`。`configuration_glm5_next.py` L143-145 `linear_head_dim=128, linear_num_heads=64, linear_conv_kernel_dim=4` ⇒ 64×128=8192、conv_dim=24576、kernel=4、groups=通道数、无偏置、因果，与正文与 [C4][N3] 逐项吻合。checkpoint 拆 `q/k/v_conv1d` 三个 `[8192,1,4]` 张量：外部端口（sglang PR #38616、vllm.cpp）与 GLM 数据流页交叉一致。
- **代码实跑**：「省了多少」成本计数代码块输出与页面【预期输出】逐行相同（9,437,184 / 147,456 / 1,048,576 / 1,196,032 / 0.126736 / 7.9x）；「大模型里的用法」torch 代码块实跑输出 `[-1.0, -2.5, -3.5, -3.5, -3.5]`、`[1.0, 3.0, 5.0, 7.0]`、`[10.0, 20.0, 30.0, 40.0]`，与页面一致，手算注释 t=0..4 逐个复算无误。
- **页面功能**：`python3 .dojo/scripts/validate.py wiki/depthwise-conv/index.html` → `validation ok`；KaTeX 渲染后正文残留 `$` 计数为 0（h1/h2/h3/summary/正文/表格无 Unicode 数学字符，唯一 `·` 在 JS 字符串内）；`../../libs/*` 全部存在，`../kda/`、`../glm-5-3-flash-dataflow/`、`overview.html` 均存在且与 index 双向互链。

## 问题

- [轻微·图示] 「2. 拆成两步」图内 “核 $D_K\times D_K\times M\times N$” 标签：其 foreignObject 宽 60 单位，内部 `.katex` 内容实测宽 113.1 单位（近 2 倍），超宽行居中失效、文字向右溢出外框，落到箭头正下方并以约 1 单位压到右侧输出框左边缘（该 rect 绘制在 foreignObject 之后，覆盖约 1 px）；当前字号与 900 px 视口（SVG 缩放 1.1197）下与箭头尚有约 7 单位净空、肉眼未见重叠，故不构成可读性缺陷，但标签未落在预定槽位，字号/字体微调即会压线。｜引文依据：像素测量——foreignObject x=92 width=60；内部 `.katex` 实宽 113.1；箭头 path x=158..200、y=106；输出框 rect x=204｜修复要求：把该 foreignObject 宽度由 60 调到 ≥120（或调小该标签字号），使文字不与箭头、输出框相接｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布。每条来源论断均已回到所核对的版本逐条确认（MobileNets v1 §3.1 与 §4.1 Table 4；transformers main `glm5_next` 源码及其 config 默认值），两段代码块实跑输出与页面【预期输出】一致，validate.py 通过。唯一 1 条轻微为图内标签外框宽度，当前渲染下无可读性影响（接受理由：净空约 7 单位、无可见重叠），可在后续编辑窗口顺手修正。