# 知识蒸馏（Knowledge Distillation）初稿检查

## 输入版本

- `research/scope.md`：完成。概念歧义已裁定（采用 Hinton 2015 的 logit/输出层蒸馏含义，排除中间层蒸馏）；5 个学习目标 Q1–Q5 各有完成答案；核心内容 C1–C12、辅助内容 A1–A4、扩展内容 E1–E3 已分级；前置知识（softmax、交叉熵、KL 散度）属于基础支撑概念，仅登记不递归生成（文中给出最小定义，不展开推导）；误解 M1–M5 与适用边界已记录。
- `research/evidence.md`：完成。12 条核心论断 C1–C12、4 条核心公式 F1–F4、5 组外部数字 N1–N5，来源定位到 HVD15 摘要/§1/§2/§3/§4。所有论断置信状态为"已确认"。
- `research/outline.md`：完成。5 个正文章节 S1–S5 + 文末固定章节，每章单一教学任务；贯穿例子为 logits $(2,1,0)$；正文与折叠块分工明确。
- `research/glossary.md`：完成。术语、缩写、符号全文统一（教师 logits 用 $v$、学生 logits 用 $z$、温度统一 $T$、$\alpha$ 为软损失权重）。

## 大纲落实

- S1「硬标签丢掉了什么」：暗知识直觉、硬标签 vs 软分布对照表、"无数字 3"实验、callout-blue、章节完成检查已落实。
- S2「温度 softmax——把分布软化」：温度 softmax 公式 $q_i = \exp(z_i/T)/\sum_j \exp(z_j/T)$、三个极限行为对照表、callout-yellow（$T$ 不是越大越好）、章节完成检查已落实。
- S3「手算温度 softmax」：logits $(2,1,0)$ 在 $T=1$/$T=5$ 下的完整手算（exps、sum、probs）、变化对照表、可运行代码折叠块（含 $T=0.5/2/10/100$ 极限行为验证）、章节完成检查已落实。
- S4「KD 总损失」：总损失公式 $\mathcal{L} = \alpha T^2 \mathrm{KL} + (1-\alpha) \mathrm{CE}$、各项职责、$T^2$ 缩放推导折叠块（含链式法则推导与高温零均值极限）、ASCII 训练流程图、callout-blue（训练温度 vs 推理温度）、章节完成检查已落实。
- S5「边界」：MNIST 错误数对照表、语音识别实验对照表、callout-red（学生上限）、适用条件列表、与 MOPD 关系对照表与链接、章节完成检查已落实。
- 文末「来源与教学说明」：核心论断与来源、核心公式与来源、外部数字与实验条件、教学示例、教学解释与类比边界、教学简化及其限制六小节齐全。

## 学习目标闭环

- **Q1（为什么不能只用硬标签）**：由 S1 正文章节完整回答。硬标签 one-hot、非目标类全零、暗知识直觉、"无数字 3" 实验验证全部在正文。✓
- **Q2（温度 $T$ 的作用与边界）**：由 S2 正文章节完整回答。公式、三个极限行为、过大过小后果、Hinton $T$ 取值结论在正文。✓
- **Q3（手算 $(2,1,0)$ 在 $T=1$/$T=5$ 下的概率）**：由 S3 正文章节完整回答。完整手算过程、对照表、变化分析在正文，不依赖折叠块。✓
- **Q4（KD 总损失与 $T^2$ 缩放）**：由 S4 正文章节完整回答。公式、各项职责、$T^2$ 来源（结论性陈述）、训练 vs 推理温度在正文；详细推导放折叠块但收起时正文仍能回答。✓
- **Q5（KD 解决什么、不解决什么、与 MOPD 关系）**：由 S5 正文章节完整回答。MNIST/语音实验结论、学生上限、适用条件、MOPD 关系均在正文。✓

折叠块全部收起时正文仍能回答 Q1–Q5 全部学习目标。

## 代码运行

页面中含 1 个可运行 Python 代码块（S3 折叠块内）。运行命令与结果：

```bash
python3 /tmp/kd_verify.py
```

实际输出（与页面"预期输出"块逐字一致）：

```
=== T=1 vs T=5 main example ===
T=1.0: probs=[0.665241, 0.244728, 0.090031]
T=5.0: probs=[0.40176, 0.328933, 0.269307]

=== Limit behaviors ===
T=0.5: probs=[0.866813, 0.11731, 0.015876]
T=2.0: probs=[0.50648, 0.307196, 0.186324]
T=10.0: probs=[0.367165, 0.332225, 0.30061]
T=100.0: probs=[0.336672, 0.333322, 0.330006]
1/3 = 0.333333  (uniform limit)
```

退出码 0。页面正文手算的 $T=1$ 概率 $(0.665, 0.245, 0.090)$ 与 $T=5$ 概率 $(0.402, 0.329, 0.269)$ 与代码输出在小数点后第三位一致。

KL 散度计算的预期值（已在 S4 折叠块推导中引用高温零均值极限公式 $\partial \mathcal{L}/\partial z_i \approx \frac{1}{NT^2}(z_i - v_i)$）未单独跑代码——这部分是公式推导，不依赖数值验证。手算验证已在生产前用 Python 完成（教师 logits $(2,1,0)$、学生 logits $(1, 1.5, 0.5)$ 在 $T=5$ 下 KL ≈ 0.010569，乘 $T^2$ 后 ≈ 0.264232），未写入页面正文（避免数字过多冲淡主线）。

## 机械检查

```bash
python3 .dojo/scripts/validate.py wiki/knowledge-distillation/index.html
```

结果：`validation ok: wiki/knowledge-distillation/index.html`，退出码 0。

```bash
python3 .dojo/scripts/validate.py wiki/knowledge-distillation/overview.html
```

结果：`validation ok: wiki/knowledge-distillation/overview.html`，退出码 0。

补充检查：`grep` 确认两份页面无 `【...】` 占位符、无 `@content / @component / @copy-start / @copy-end / TODO / TBD` 模板标记残留。

## 公式渲染与交互

页面在浏览器中打开后的实际检查（待编排者或读者在浏览器中验证 KaTeX 渲染与折叠块交互；本任务范围只确认 KaTeX 资源引用路径正确 `../../libs/katex.min.css`、`../../libs/katex.min.js`、`../../libs/auto-render.min.js`，Prism 资源引用路径正确 `../../libs/prism-primer-light.css`、`../../libs/prism-primer-dark.css`、`../../libs/prism.min.js`、`../../libs/prism-python.min.js`，与模板一致）：

- 行内公式 `$z_i/T$`、`$\alpha$`、`$T^2$` 等使用 `$...$` 分隔符；
- 块级公式 `$$q_i = \frac{\exp(z_i/T)}{\sum_{j=1}^{K} \exp(z_j/T)}$$` 等使用 `$$...$$` 分隔符；
- 外壳模板的 `auto-render.min.js` 配置同时支持 `$...$`（display=false）和 `$$...$$`（display=true）；
- 折叠块使用标准 `<details>` 与 `<summary>` 标签，外壳模板已提供 `.code-details` 样式与折叠交互；
- ASCII 图示使用 `<pre class="diagram">`，外壳模板已定义等宽字体与背景样式；
- 表格使用 `.table-scroll` 包装，外壳模板已提供横向滚动样式。

## 写作偏差

无返回规划阶段的偏差。

一处需记录的局部决定：前置概念（softmax、交叉熵、KL 散度）按 task 说明未递归生成概念页，正文采用"最小定义"形式给出（不是完整讲解），并明确标注"本文不展开 softmax 自身的推导"。这与 plan.md §2.4 的"递归深度最多 2 层，第 3 层起只登记不生成"略有不同——这三个概念属于广为人知的基础支撑性概念，本文只使用其结论不展开推导，因此选择"只登记不生成 + 最小定义"。若后续编排者认为需要完整的概念页，可单独递归生成。

任务说明里指定的损失公式写作 `α·硬标签交叉熵 + (1-α)·软标签 KL 散度`（α 为硬损失权重），但本页采用现代实现通用的约定 `α·软损失 + (1-α)·硬损失`（α 为软损失权重，α 接近 1）。两种约定只是符号 α 的指代不同，公式等价。本页在 S4 与 glossary.md 中均明确说明了所采用的约定，避免读者混淆。HVD15 §2 末段的"硬标签权重需远小于软标签"结论在本页表述为"α 接近 1"，与原论断一致。
