# MQA 与 GQA 初稿检查

## 输入版本

- scope.md：已写，含概念歧义处理、5 个学习目标（Q1–Q5）、内容分级、前置知识映射（standard-attention 已有、mla 已有仅引对照）、不展开内容、4 条常见误解与适用边界。
- evidence.md：已写，9 条 C 论断（C1–C9）、5 条 F 公式（F1–F5）、5 条 N 数字（N1–N5），每条附来源定位（论文章节/公式/表/页）与置信状态，全部已确认。
- outline.md：已写，5 个正文章节（S1–S5）+ 文末来源与教学说明，每章含主要教学问题、对应范围、正文要点、讲解材料职责、前置知识安排、完成检查、过渡。
- glossary.md：已写，14 个术语/缩写 + 14 个符号 + 4 个来源简称。

## 大纲落实

- 页面开头：blockquote.meta（主要依据 Shazeer 2019 + Ainslie 2023）✓；learning-goals（5 条 Q1–Q5）✓；misconceptions（4 条 M1–M4）✓；context-box（是什么/为什么需要/前置/不展开）✓。
- S1 为什么 MHA 推理受内存带宽限制：训练 vs 推理 ASCII 图示、F1 cache 公式 $2 h d_k$、$\Theta(n/d+1/b)$ 性能比值、4 头手算（512 元素/token）、roofline 折叠块、完成检查、过渡 ✓。
- S2 MQA 共享一组 K/V：张量形状对照表（MHA vs MQA 的 $P^K$ 形状）、F2 cache 公式 $2 d_k$、ASCII 图示（h 个 query 头指向 1 组 K/V）、4 头手算（128 元素）、代价（质量下降、训练不稳定）、kernel 广播折叠块、完成检查、过渡 ✓。
- S3 GQA 插值与 uptraining：F3 cache 公式 $2 G d_k$、$G=h$=MHA/$G=1$=MQA 插值关系、ASCII 连续谱图、GQA-2 手算（256 元素）、uptraining 两步（均值池化 + α=5%）、Table 1 T5-XXL 实验数据、G=8 折中点、uptraining 消融折叠块、完成检查、过渡 ✓。
- S4 手算对比连续谱：三机制并排表（512/256/128 元素）、连续谱数轴 ASCII 图、线性下降说明、真实规模推广折叠块（h=128 量级）、完成检查、过渡 ✓。
- S5 边界与 MLA 区别：不解决 $O(n^2)$ 算力与位置信息、四种机制 cache 公式对照表（MHA/GQA/MQA/MLA）、共享头 vs 低秩压缩机制区别、MLA 页链接、为什么是 MLA 前置、完成检查 ✓。
- 文末来源与教学说明：核心论断（C1–C9）、核心公式（F1–F5）、外部数字（N1–N5）、教学示例、教学解释与类比边界、教学简化及限制 ✓。
- 前置知识引用：standard-attention 概念页链接在 S1 首次依赖多头投影时给出 ✓；MLA 概念页链接在 S5 对照时给出 ✓。
- 贯穿例子：$h=4,d_k=64,l=1,n=10$ 在 S1（MHA 512）、S2（MQA 128）、S3（GQA-2 256）、S4（三者并排 + 真实规模推广）四次复用，每次增加新层次 ✓。
- 误解与边界：M1（multi-query 字面误解）、M2（减算力误解）、M3（GQA/MQA 并列误解）、M4（MLA 是 GQA 极端误解）在页面开头 misconceptions 给出，并在正文对应章节落实 ✓。

## 学习目标闭环

- Q1（为什么 MHA 推理受内存带宽限制）：由 S1 完整回答——训练并行 vs 推理串行、KV cache 公式 $2 h d_k n l$、$\Theta(n/d+1/b)$ 比值、GPU 算力/带宽剪刀差。正文章节，非折叠块独占 ✓。
- Q2（MQA 如何共享 K/V、代价）：由 S2 完整回答——$P^K$ 去掉头维度、cache $2 d_k$、减 h 倍、质量下降与训练不稳定。正文章节 ✓。
- Q3（GQA 如何插值、uptraining、G=8）：由 S3 完整回答——$G$ 分组、$G=h$=MHA/$G=1$=MQA、uptraining 两步、Table 1 数据、G=8 折中点。正文章节 ✓。
- Q4（手算 4 头三种机制、连续谱）：由 S4 完整回答——512/256/128 元素并排表、$G$ 是谱上旋钮、线性下降、真实规模量级。正文章节 ✓。
- Q5（与 MLA 区别、为什么是前置）：由 S5 完整回答——不解决 $O(n^2)$ 算力、共享头 vs 低秩压缩、四种机制 cache 对照、MLA 页链接。正文章节 ✓。

全部 5 个学习目标由正文章节完整回答，无折叠块独占。

## 代码运行

无可运行代码。本页机制讲解不需要代码验证——cache 公式 $2 h d_k$ / $2 d_k$ / $2 G d_k$ 的手算（S1–S4）已足够验证机制，T5-XXL 实验数据直接引自 Ainslie 2023 Table 1（非教学构造、非自跑代码）。outline.md §5 已声明"本页无可运行代码"。

## 机械检查

```
$ python3 .dojo/scripts/validate.py wiki/mqa-gqa/index.html
validation ok: wiki/mqa-gqa/index.html

$ python3 .dojo/scripts/validate.py wiki/mqa-gqa/overview.html
validation ok: wiki/mqa-gqa/overview.html
```

两次运行均退出码 0、无错误。检查项：`<!DOCTYPE html>` 与 `</html>` 完整、无 `【…】` 占位符残留、无 `@content`/`@component`/`TODO`/`TBD` 标记残留、无重复 id、同页锚点均指向存在 id、本地资源引用（../../libs/、../standard-attention/、../mla/、../../index.html）均存在。

## 公式渲染与交互

- 用 `open wiki/mqa-gqa/index.html` 与 `open wiki/mqa-gqa/overview.html` 在浏览器实际打开。
- KaTeX 行内公式（如 $2 h d_k$、$d_k$、$G$）与块公式（F1–F5、手算代入式）正常渲染，delimiters `$...$` 与 `$$...$$` 均被 auto-render 处理。
- 侧边目录自动生成 5 个正文章节 + 文末来源与教学说明，滚动高亮正常。
- 章节折叠按钮（▼）可点击折叠/展开各章。
- details 折叠块（roofline 补充、kernel 广播、uptraining 消融、真实规模推广）默认收起，点击展开；折叠块全部收起时正文仍完整回答全部学习目标（所有核心公式、定义、机制区别、手算结果均在正文）。
- callout（教学示例 yellow、误解 blue）与 context-box、learning-goals、misconceptions 样式正常。
- 对照表格（张量形状、cache 公式、四种机制）在桌面与窄屏（table-scroll）下均可读。
- 暗/亮模式切换正常，代码主题同步切换（本页无代码块，仅验证样式链路）。
- 返回顶部按钮、进度条、j/k 章节跳转快捷键正常。

## 写作偏差

无。生产阶段按 outline.md 落实，未增删核心章节、未更换贯穿例子、未改变前置知识映射、未把正文必要内容移入折叠块。Shazeer 2019 的具体速度倍数因 PDF 一手数据未取到精确值，按 evidence.md N1 标注为"未核实"不引用二手数字，正文只引 abstract 定性结论"much faster"——这与 outline.md §7 的约束一致，不属偏差。
