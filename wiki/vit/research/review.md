# Vision Transformer 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源核查）
- 页面版本：index.html `ac3d5c45e9a082c0035c96c916687d6cd951b066` / overview.html `b750f7cc80776dae392fb1e9453c6a29d35ea5c5`
- 时间：2026-08-09
- 来源：Dosovitskiy et al. 2021, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021, arXiv:2010.11929v2（https://arxiv.org/abs/2010.11929 ；HTML 全文 https://arxiv.org/html/2010.11929v2 ）
- 审查范围：wiki/vit/index.html + wiki/vit/overview.html，段 A 盲读 + 段 B 对照来源

## 问题

- [重要·来源] index.html §4 核心结论 callout + [C1] + overview.html §"关键结论与边界"末条：将原文 "large scale training trumps inductive bias" 标注为"Dosovitskiy 2021 §4.4 原文"。论文 §4.4 为 Scaling Study，该句原文实际出现在 §1 Introduction（"However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias."）。位置标注错误会使读者按 §4.4 检索时找不到该原文。修法：将该句出处由"§4.4 原文"改为"§1 原文"；如保留 §4.4 引用，则删除"原文"二字并改为"§4.4 Scaling Study 实验支持此论断"。overview.html 同步修正。 ｜ 修复：已将 index.html §4 核心结论 callout（"§4.4 原文"→"§1 原文"）、[C1]（§4.4→§1，两处引文统一归 §1）、overview.html §"关键结论与边界"末条（"§4.4 原文"→"§1 原文"）三处同步修正 ｜ 复验：
- [重要·来源] index.html §4 "小数据从零训练的边界"段 + [C7]：标注"§3.2 报告：在 ImageNet-1k（1.3M 图）上从零训练 ViT 不如 ResNet"。论文 §3.2 标题为 "Fine-tuning and Higher Resolution"，讨论微调与高分辨率调整，不含小数据从零训练对比；小数据下 ViT 不如 ResNet 的报告实际在 §1（"a few percentage points below ResNets of comparable size"）与 §4.3 Pre-training Data Requirements（"Vision Transformers overfit more than ResNets ... the convolutional inductive bias is useful for smaller datasets"）。修法：将"§3.2 报告"改为"§4.3 报告"（或"§1 与 §4.3 报告"）；[C7] 中"§3.2（小数据结果）"同步改为"§4.3（小数据结果）"。 ｜ 修复：已将 index.html §4"小数据从零训练的边界"段（§3.2→§4.3）、§4"常见误解"段（§3.2→§4.3）、[C7]（§3.2→§4.3）三处同步修正 ｜ 复验：
- [重要·来源] index.html §2 "补充：class token vs GAP（全局平均池化）的消融"折叠块：标注"附录 D.1 给出消融"。论文该消融实际位于附录 D.3 "Head Type and class token"（结论：class token 与 GAP 性能相当，差异由学习率要求不同所致）。小节编号错误会让读者按 D.1 检索时定位到无关内容。修法：将"附录 D.1"改为"附录 D.3"。 ｜ 修复：已将 index.html §2"class token vs GAP"折叠块"附录 D.1 给出消融"改为"附录 D.3 给出消融" ｜ 复验：
- [轻微·来源] index.html §4 数据集表 JFT-300M 行 + chapter-summary：标"300M 图像"。论文 §4.1 原文为 "JFT with 18k classes and 303M high-resolution images"。JFT-300M 为数据集惯用名，但严格图像数为 303M 高分辨率图像。修法：表格"图像数"列由"300M"改为"303M"，chapter-summary 补一句"JFT-300M 为惯用名，实际含 303M 高分辨率图像"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html §1 CNN vs ViT 对比表 + chapter-summary：CNN 行"远距离交互路径 $O(n/k)$ 层"以"n 为图像线性化后的 token 数"衡量 CNN 层数；CNN 处理的是像素而非 token，用 token 数衡量 CNN 概念口径不一致，小白会困惑"为什么用 token 数算 CNN"。修法：将 n 定义改为"图像线性尺寸（如边长像素数）"或"序列化后的位置数"，并使 CNN 与 ViT 两行采用同一口径。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html §3 "手算 ViT-B/12/16 配置" + 学习目标第3项"ViT-B/12/16"：命名约定不标准。论文惯例为"ViT-B/16"（斜杠后为 patch 大小），层数 12 属 Table 1 配置项；"ViT-B/12/16" 非论文写法，易让小白误以为这是模型正式名称。修法：改为"ViT-B/16（12 层）"或"ViT-Base/16"，学习目标同步。 ｜ 修复： ｜ 复验：
- [轻微·来源] index.html §3 "与标准 Transformer 的三处差异"表 "$Q,K,V$ 来源"行：表述"NLP 是词嵌入的线性投影，ViT 是 patch 嵌入 $z_{\ell-1}$ 的线性投影"。严格说每层 Q/K/V 来自上一层层输出经线性投影，并非直接来自词嵌入/patch 嵌入；该表述易让小白误以为 Q/K/V 内部计算不同（虽然 misconception 第5条与 §3 关键结论 callout 已澄清 Q/K/V 内部计算与标准注意力完全一致）。修法：将行标题改为"输入 token 类型"或加注"指编码块输入端 token 类型；Q/K/V 内部计算见 [标准注意力] 概念页"。 ｜ 修复： ｜ 复验：
- [轻微·来源] index.html §2 "class token vs GAP"折叠块 + §"最容易误解"第3条：表述"在 JFT-300M 预训练下，用 GAP 与 class token 性能相近"。论文附录 D.3 结论为整体相当且需不同学习率，未强调 JFT-300M 这一条件作为限定。加条件标注依据不足。修法：删除"在 JFT-300M 预训练下"限定，改为"附录 D.3 消融显示 GAP 与 class token 整体性能相当（需不同学习率）"；misconception 第3条同步。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html §1 引子"让 12 层自注意力自己决定"：12 层对应 ViT-B 配置，但 §1 未交代配置来源，小白在 §3 Table 1 出现前会困惑"12 层从哪来"。修法：改为"多层自注意力"或加注"（ViT-B 配置，见 §3）"。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 6
- 处置：进入修复（重要问题均为来源定位/小节编号标注错误，不涉及核心论断、公式、数字或机制；修复范围小且可逐条回填复验）
- 段 A 盲读：学习目标 4 项（z_0 公式与手算、编码块公式与三处差异、数据规模边界、ViT 要解决的问题）均由正文章节完整回答，不依赖折叠块；前置概念占位提示齐全（CNN、矩阵乘法、LayerNorm、GELU 待生成），已给最小概念。主线无阻断级卡点。
- 段 B 对照来源：核心公式 F1–F4 与论文 Eq.(1)–(4) 完全一致并确认 pre-LN；Table 1 模型配置（B/L/H 的 L、D、MLP size、h、参数量）与论文一致；Table 2 ImageNet top-1 与 TPUv3-core-days（85.30/0.23k、87.76/0.68k、88.55/2.5k、87.54/9.9k、88.4–88.5/12.3k）与论文一致；224×224 patch 16 → 196 patch / 197 序列长度手算正确；d_k=64 手算正确。问题集中在三类来源定位标注（§4.4 vs §1、§3.2 vs §4.3、附录 D.1 vs D.3）。
- 未核查项：前置概念页（standard-attention、residual-connection、positional-encoding、moonvit-v2）链接目标存在性与层级，按审查任务包约束"禁止读取仓库中其他页面"未打开核对，仅确认链接路径格式正确；页面功能（公式渲染、折叠交互、目录锚点浏览器实际打开、validate.py 退出码）属发布门控，本审查未执行。
