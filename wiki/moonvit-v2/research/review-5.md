<!-- review-meta
round: 5
page: wiki/moonvit-v2/index.html
reviewed_content_sha256: ceb05790a1a081fa
-->
# MoonViT-V2（K3 视觉编码器）审查记录（第 5 轮）

- 页面版本：b343316cbc41cb9e64e513d734c8860fa78b4257（git hash-object wiki/moonvit-v2/index.html）
- 审查时间：2026-09-14 17:04
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次；未读取本页 research/ 下任何文件）
- 已完整阅读章节：核心问题（5 条，含全部解答折叠块）；1. 主流做法遇到什么问题——SigLIP 初始化为何不稳（含本章问题）；2. 从零训练的方案——next-token prediction 如何替代对比预训练（含本章问题）；3. 训练是怎么进行的——接入 LLM 的联合预训练（3.1 训练用什么数据、3.2 训练配置：整个模型统一一套、3.3 系统层面怎么跑得动，含本章问题）；4. 服务稳定性的架构——27 层 ViT、RMSNorm、去 bias（含「补充」折叠块与本章问题）；5. 图像/视频共享与高分辨率——分解注意力、时间池化、pixel-shuffle（含本章问题）；6. 结果与结论——对比预训练初始化是否必要（含本章问题）；来源与范围说明（C/F/N 三节 + 构造示例 + 辅助解释与类比边界 + 简化条件及其限制）；overview.html。含全部折叠块、图注与三张内联 SVG 图逐像素阅读。

来源核对（本轮实际执行）：K3 报告 arXiv 2607.24653v2（PDF 文本抽取）逐字核对 §2.4 Native Vision、§3.1 Pre-Training Data、§3.3 Training Recipe、§5.2.3 Multimodal Encoder Optimization，以及 Fig.6 标题与 Table 1；HuggingFace moonshotai/Kimi-K3 config.json 的 vision_config 全字段核对；F1/F2 手算与页面「可运行代码」实跑。

核对通过项（用于界定问题范围）：C1–C11 全部引文逐字吻合（如 C4「a 27-layer vision transformer with roughly 0.4B parameters that adopts RMSNorm and removes all bias terms...a design that further stabilizes the from-scratch optimization above」、C6「pixel-shuffle operation with 2 × 2 downsampling...up to 3584 × 3584 pixels affordable within the 1M-token context」、C9 两处、C10 三处、C11 三处均与原文一致）；Fig.6 横轴「Training step (×10³)」刻度 7/10/15/20/25/30、纵轴 0/0.2/0.4/0.6，与本页简化对照图刻度逐一相符；Table 1「Total Parameters of ViT = 401M」「27 layers / patch 14 / 12 heads」「Total Parameters 2.78T」与正文一致；config.json 的 vt_num_hidden_layers=27、vt_hidden_size=1024、qkv_hidden_size=1536、vt_intermediate_size=4096、vt_num_attention_heads=12、patch_size=14、norm_type=rmsnorm、attn_bias/linear_bias/patch_embed_proj_bias=false、merge_kernel_size=[2,2]、merge_type=sd2_tpool、text_hidden_size=7168 全部吻合；F1（6,291,456 + 8,388,608 + 2,048 = 14,682,112；×27 = 396,417,024 ≈ 0.40B）与 F2（3584/14=256 → 65,536 → ÷4=16,384）复算无误；页面代码实跑输出与文中「预期输出」逐字一致；三张图均为内联 SVG，图内公式经 `<foreignObject>` 由 KaTeX 渲染、`<text>` 内无 ASCII 近似数学；无 `alt` 含 `$...$`；前置概念链接 ../siglip、../vit、../standard-attention 与 overview.html/根 index.html 均真实存在；`dojo:type=concept`、topics「多模态」、tag「视觉与多模态」均在词表内；`.dojo/scripts/validate.py` 返回 `validation ok`。

## 问题

- [重要·技术] 第 4 章正文（第 407 行，"MLP 投影器不小"一句）：`$1024 times 4=4096$` 中的 `\t` 被落成了**字面制表符**（0x09），只剩 `imes`，KaTeX 实际把该式渲染为 "1024imes4=4096"（`imes` 被当作变量 i·m·e·s），公式不可读且语义错误，应为 "1024×4=4096"。｜引文依据：文件字节为 `24 31 30 32 34 09 69 6d 65 73`（即 `$1024` + TAB + `imes 4=4096$`）；以 KaTeX 实机渲染 `katex.renderToString("1024\t imes 4=4096",{throwOnError:false})` 得到的纯文本为 `1024imes4=4096`。config.json `text_hidden_size=7168`、pixel-shuffle 后通道维确为 1024×4=4096，支持该式本应含 ×4。｜修复要求：将此处的制表符+`imes` 改回 `\times`（改为 `$1024\times 4=4096$`），并在浏览器/KaTeX 复验渲染结果为 "1024×4=4096"（不含 "imes"）。｜修复：｜复验：

- [轻微·技术] 来源章节「外部数字与实验条件（N）」N4：引号引文 `"2.8-trillion-parameter Mixture-of-Experts"` 标注出处为「K3 报告摘要」，但摘要原文为 `a 2.8T parameter Mixture-of-Experts model`；该逐字串实际出现在报告 Conclusion（"an open 2.8-trillion-parameter Mixture-of-Experts model"）与 §1（"the 2.8-trillion-parameter scale"）。数字本身（2.8T/2.78T）无误，仅引文出处的章节标注不准。｜引文依据：摘要 "We introduce Kimi K3, a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters"；Conclusion "We present Kimi K3, an open 2.8-trillion-parameter Mixture-of-Experts model"。｜修复要求：将 N4 的出处由「摘要」改为「§结论/Conclusion」，或把引号内文字替换为摘要原文（`2.8T parameter Mixture-of-Experts model`），二者择一，使用词与出处一致。｜修复：｜复验：

- [轻微·表述] 第 4 章末（第 479 行）："这一点常被误读成"现代 ViT 惯例""——以结论口吻断言社区常见认知，无来源支持。｜引文依据：不适用｜修复要求：删除该判断句，或改写为明确标注为本页判断的推断（如"一种常见看法是把它当作现代 ViT 惯例"并说明无来源），不得保留为无来源的断言。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（修复后需复验第 4 章该式渲染与 N4 出处；本轮无阻断项）