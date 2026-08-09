# Vision Transformer（ViT）初稿检查

## 输入版本

- `research/scope.md`：完成（概念边界、4 个学习目标 Q1–Q4、内容分级、5 项常见误解、4 项适用边界、前置知识映射 3 项已有 + 4 项缺失登记、明确不展开项 10 项）。
- `research/evidence.md`：完成（8 条核心论断 C1–C8、4 条公式 F1–F4、3 条数字 N1–N3，全部来自 Dosovitskiy et al. 2021 原论文 §3.1 / Table 1 / §4.2 Table 2 / §4.4 与附录 D；无存在冲突或证据不足的论断）。
- `research/outline.md`：完成（4 章 S1–S4 + 文末来源、贯穿 224×224 patch 16 例子在 S2/S3/S4 复用、讲解材料职责表、正文与折叠块分工、范围与证据约束）。
- `research/glossary.md`：完成（术语 39 项、符号 24 项、缩写 12 项；$x, P, N, E, x_{\text{class}}, E_{\text{pos}}, D, z_0, L, \ell, \text{LN}, \text{MSA}, \text{MLP}, y, h, d_k$ 全文含义一致）。

## 大纲落实

- **章节**：S1 ViT 要解决什么问题 / S2 图像如何变成 token 序列 / S3 Transformer 编码块 / S4 数据规模边界 / 文末来源与教学说明。全部 5 个 h2 章节标题与 outline.md 一致。
- **学习目标**：4 个学习目标（Q1–Q4）逐项对应 S1–S4 正文章节，没有由折叠块独占的目标。
- **前置知识**：3 项已有概念页（`standard-attention`、`residual-connection`、`positional-encoding`）在正文首次依赖时给链接。4 项登记的缺失概念（`matrix-multiplication`、`cnn`、`layer-norm`、`gelu`）在正文首次依赖时给占位提示 + 一句话衔接，不内联大段背景。
- **贯穿例子**：224×224 + patch 16 + ViT-Base 配置（$D=768$、12 层、12 头）在 S2 算 $N=196$、$N+1=197$、$z_0\in\mathbb{R}^{197\times 768}$；在 S3 复用算每头 $d_k=D/h=64$、注意力矩阵 $197\times 197$；在 S4 复用对照 ViT-H/14 patch 14 → $N=256$。
- **误解和边界**：5 项误解（ViT 用 attention 替换卷积 / patch 就是 token / class token 可去 / 88.55% 普遍胜 CNN / ViT 是新架构）在页面开头 misconceptions 块；4 项适用边界（小数据从零训练 / 大数据预训练 / 检测分割下游 / 视频时序 / 多模态对齐）在 S4 边界对照表。
- **过渡**：S1→S2（"为什么试纯 Transformer" → "图像怎么变成 token"）、S2→S3（"$z_0$ 全有了" → "Transformer 编码"）、S3→S4（"机制全清楚了" → "数据规模边界"）、S4 文末收束（"理解了它就理解了所有变体要保住什么改什么"）。每章末过渡都有。

## 学习目标闭环

- **Q1（ViT 要解决什么问题）**：S1 正文完整回答。CNN 两个归纳偏置（locality + translation invariance）写在正文；小数据下是优势、大数据下成为限制的解释写在正文；ViT 设计取向"无空间先验、让模型从数据学"写在正文；CNN vs ViT 对照表展示空间先验/远距离交互路径/可学关系对比；常见误解 callout 区分"用 attention 替换卷积"vs"用纯 Transformer 替代 CNN"；与 NLP Transformer 对应一句。完成检查三问覆盖。
- **Q2（patch embedding + class token + 位置编码）**：S2 正文完整回答。三步流程（patch 切分 → 展平+线性投影 → class token + 位置编码）逐步说明；$z_0$ 合成公式 F1 + 完整符号解释 + ASCII 图示 + 224×224 patch 16 手算（$N=196$、$N+1=197$、$z_0\in\mathbb{R}^{197\times 768}$）+ 形状对照表；折叠块放 BERT [CLS] 来源、1D vs 2D 消融、GAP vs class token 消融；patch 大小 $P$ 影响一句；常见误解 callout 区分 patch vs token。完成检查四问覆盖。
- **Q3（编码块公式与差异）**：S3 正文完整回答。$z_\ell', z_\ell, y$ 三条公式 F2/F3/F4 + 全部符号（$L, \ell, \text{LN}, \text{MSA}, \text{MLP}, z_L^0$）+ pre-LN 解释 + 与标准 Transformer 三处差异对照表（pre-LN / encoder-only / Q,K,V 来源）+ 关键结论 callout + ViT-B/12/16 每头维度 $d_k=64$ 与注意力矩阵 $197\times 197$ 手算 + 模型配置表 N1。完成检查四问覆盖。
- **Q4（数据规模边界）**：S4 正文完整回答。三个数据集表格（ImageNet-1k/21k/JFT-300M）+ ImageNet top-1 + TPUv3-core-days 对照表（ViT-L/16 87.76%、ViT-H/14 88.55%、BiT-L 87.54% 等）+ "大数据胜过归纳偏置"核心结论 callout + 适用边界对照表（5 项场景）+ 常见误解 callout + ViT 家族变体一句话提及（含 MoonViT-V2 链接）。完成检查四问覆盖。

所有目标在折叠块全部收起时仍能由正文回答。

## 代码运行

无可运行代码。本页核心机制（patch 切分 + 线性投影 + 编码块复用标准注意力）用 224×224 patch 16 手算（$N=HW/P^2=224\times 224/16^2=196$、$N+1=197$）与 ViT-B/12/16 配置手算（$d_k=D/h=768/12=64$）在 S2/S3 正文逐步代入验证。MSA 内部公式直接引用 [标准注意力](../../wiki/standard-attention/index.html) 概念页（已有页面给出 2×2 手算验证），不重复。所有数字均经独立 Python 复算确认（$224/16=14$、$14\times 14=196$、$768/12=64$）。

## 机械检查

```bash
$ python3 .dojo/scripts/validate.py wiki/vit/index.html
validation ok: wiki/vit/index.html
$ python3 .dojo/scripts/validate.py wiki/vit/overview.html
validation ok: wiki/vit/overview.html
```

两次运行退出码均为 0。无残留占位符 `【…】`、无 `@content`/`@component`/`TODO`/`TBD` 标记、无重复 id、无指向缺失 id 的同页锚点、无指向不存在文件的本地引用。本地引用链接（`../../wiki/standard-attention/index.html`、`../../wiki/residual-connection/index.html`、`../../wiki/positional-encoding/index.html`、`../../wiki/moonvit-v2/index.html`）均指向已存在页面。

## 公式渲染与交互

- **KaTeX 渲染**：index.html 引用 `../../libs/katex.min.css` 与 `../../libs/katex.min.js` + `../../libs/auto-render.min.js`（自动渲染 `$...$` 与 `$$...$$`）。本地路径相对页面解析。文中 6 处 display 公式（F1 $z_0$ 合成、F2 MSA 子层、F3 MLP 子层、F4 分类头）与若干 inline 公式（$x, P, N, E, x_{\text{class}}, E_{\text{pos}}, D, z_0, L, \ell, \text{LN}, \text{MSA}, \text{MLP}, y, h, d_k$ 等）均使用标准 KaTeX 分隔符。
- **目录生成**：外壳脚本 `document.querySelectorAll('body > h2, body > h3')` 扫描所有 h2/h3；本页 5 个 h2（4 章 + 来源与教学说明）+ 6 个 h3（来源小节）均已带显式 id（`why-vit`、`patch-embedding`、`transformer-encoder`、`data-scale-boundary`、`sources-and-teaching-notes`），目录与 j/k 快捷键工作正常。
- **章节折叠按钮**：每个 h2 末尾自动添加 ▼ 按钮，点击切换后续元素 display。本页折叠块（3 个 details：BERT [CLS] 来源、1D vs 2D 消融、pre-LN 稳定性 + 1 个 GAP 消融）独立工作，不受 h2 折叠影响。
- **亮/暗主题**：外壳脚本读 localStorage 与 prefers-color-scheme，自动切换 KaTeX 与 Prism 主题 CSS。本页未引入自定义颜色，全部使用 CSS 变量。
- **代码块复制**：无代码块（不安排可运行代码），无复制按钮生成需求。
- **图片点击放大**：无 `<img>`，本页用 ASCII 图示（pre.diagram）替代，不需要 lightbox。
- **224×224 手算验证**：$N=HW/P^2=224\times 224/16^2=50176/256=196$；$N+1=197$；$d_k=D/h=768/12=64$；注意力矩阵形状 $197\times 197$。所有数字经 Python 复算确认。
- **MoonViT-V2 链接验证**：`../../wiki/moonvit-v2/index.html` 经 `ls` 确认存在。
- **前置概念链接验证**：`standard-attention`、`residual-connection`、`positional-encoding` 三个已有概念页均经 `ls` 与 `grep <h2` 确认存在。

## 写作偏差

无偏差。

- 未自行增删核心章节（4 章 + 来源与教学说明，与 outline.md 一致）
- 未增加新学习目标（4 个 Q 与 scope.md 一致）
- 未更换贯穿例子（224×224 patch 16 + ViT-B 配置在 S2/S3/S4 复用）
- 未改变前置知识映射（3 项已有 + 4 项缺失登记，正文用占位提示）
- 未把正文必要内容移入折叠块（所有学习目标结论在正文，折叠块只放 BERT 来源消融、1D vs 2D 消融、GAP vs class token 消融、pre-LN 稳定性补充）
- 未使用证据不足或无法消歧的论断（所有核心论断来自 Dosovitskiy 2021 原论文 §3.1 / Table 1 / §4.2 Table 2 / §4.4 与附录 D；pre-LN 稳定性的 Xiong 2020 引用是辅助对照，不作为核心论断依据）

写作中补充的局部内容（未改变大纲）：
- S1 callout 中"归纳偏置 = 架构给数据预先设定的规则"的类比与失效边界，已在文末"教学解释与类比边界"中登记。
- S2 折叠块中 BERT [CLS] token 来源、1D vs 2D 消融、GAP vs class token 消融的简要说明，对应 glossary.md 中已登记的"BERT [CLS] token""可学习 1D 绝对位置编码""GAP"边界，未引入新概念。
- S3 折叠块中 pre-LN 与 post-LN 稳定性的对照（Xiong 2020 一句话提及），对应 glossary.md 中已登记的"pre-LN"与"post-LN"，未引入新概念。
- S4 边界对照表中"视频时序建模"一行提及 MoonViT-V2 是变体扩展，已在变体列表处链接到 MoonViT-V2 概念页。
