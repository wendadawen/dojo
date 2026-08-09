# FlashKDA 与 KDA Context Parallelism 初稿检查

## 输入版本

- scope.md：已写，含概念含义、5 个学习目标（Q1–Q5）、内容分级（C1–C5 核心 + 辅助 + 扩展）、前置知识映射（3 份已有概念页）、不展开内容、误解（M1–M5）与适用边界。
- evidence.md：已写，C1–C5 论断 + F1–F3 公式，全部来自 K3 报告 §5.1.1/§5.1.2/§5.4.2，逐条定位，置信状态均为已确认，无冲突或证据不足项。无外部性能数字。
- outline.md：已写，6 章（S1–S5 + 文末来源），贯穿例子为 2 rank × 2 token KDA 递归，讲解顺序、材料职责、正文折叠块分工齐备。
- glossary.md：已写，术语 + 符号 + 约定齐全，与 KDA 页状态约定一致（$S\in\mathbb{R}^{d_k\times d_v}$，状态在右）。

## 大纲落实

逐项核对 outline.md 的章节、学习目标、前置知识、贯穿例子、误解和边界、过渡：

- **页面开头**：context-box（本页所处位置）、learning-goals（Q1–Q5）、misconceptions（M1–M4，页首版）、blockquote.meta（主要依据 K3 §5.1/§5.4.2）——落实。
- **S1 串行状态 vs GPU 并行**：h2 id=serial-vs-parallel，含 Eq.1 引用、GPU 并行偏好链接、四 regime 瓶颈对照表、chunkwise 空隙图示、完成检查 3 项、过渡到 S2——落实。
- **S2 FlashKDA**：h2 id=flashkda，含朴素两阶段交替、token-parallel + head-parallel 分解、重叠时间线图示、对照表、CUTLASS/flash-linear-attention 工程定位、完成检查 3 项、过渡到 S3——落实。
- **S3 设备内 CP**：h2 id=intra-device-cp，含 TP 局限、关键观察（段转移可独立于入状态计算）、SM 级 CP planner、与 KCP 通信区别、TP vs 设备内 CP 图示、完成检查 3 项、过渡到 S4——落实。
- **S4 KCP**：h2 id=kcp，含三个 h3 小节（vanilla LA 可求和 / KDA 不可求和 / $M+\tilde S$ 分解 / all-gather+prefix scan / 手算验证）、Eq.17 boxed、展开求和形式、三方案通信量对照表、手算例子正文结论 + 折叠块完整计算（含误用直接求和的错误对照）、完成检查 4 项、过渡到 S5——落实。
- **S5 KDA 解码**：h2 id=kda-decoding，含三个 h3 小节（朴素快照代价 / 投影输入缓存 / 载荷不变）、状态快照 vs 投影输入图示、ReplaySSM 并发、融合 kernel、亚线性延迟、完成检查 4 项、过渡到文末——落实。
- **文末来源与教学说明**：h2 id=sources-and-teaching-notes，含核心论断与来源（C1–C5 逐条 K3 报告原文引用）、核心公式与来源（F1–F3）、教学示例（2×2 递归构造）、教学解释与类比边界（4 条 + 失效边界）、教学简化及其限制（5 项）——落实。无外部数字小节（本页无外部性能数字，已删除）。
- **误解和边界**：页首 M1–M4 + 文末教学解释与类比边界 4 条 + 适用边界在 scope.md；正文 S2/S3/S4/S5 各章末完成检查含边界性问题——落实。
- **过渡**：每章末尾均有"本章已得结论 + 指出下一章要解决的问题"的过渡句——落实。

## 学习目标闭环

逐题核对：

- **Q1（冲突根源 + 四 regime 瓶颈）**：S1 正文完整回答。Eq.1 的 $S_t$ 依赖 $S_{t-1}$ 是串行根源；四 regime 瓶颈对照表逐一定位；chunkwise 空隙图示。折叠块全收起时正文仍完整回答（四 regime 表在正文）。✓
- **Q2（FlashKDA 重叠）**：S2 正文完整回答。朴素两阶段交替的空隙、token-parallel + head-parallel 分解、重叠时间线、CUTLASS-based + flash-linear-attention 后端 + 服务训练/prefill。✓
- **Q3（设备内 CP 无跨设备通信）**：S3 正文完整回答。TP 不缩短递归、关键观察（段转移可独立于入状态计算）、SM 级切序列、无跨设备通信、与 KCP 通信区别。✓
- **Q4（KCP 不能直接求和 + $M+\tilde S$ 分解 + 固定通信）**：S4 正文完整回答。vanilla LA 加性递归可求和（F3）→ KDA 因 $M_t$ 不可求和（F1）→ $M+\tilde S$ 分解（F2/Eq.17）→ 一次 all-gather + prefix scan → 通信量固定（对照表）。手算例子正文给结论、折叠块给完整计算 + 错误对照。折叠块收起时正文有"$S_T^{[2]}$ 与 $S_4$ 一致"的结论 + 关键数字。✓
- **Q5（解码回滚 + 投影输入缓存）**：S5 正文完整回答。原地更新 + MTP 拒绝回滚问题、状态快照代价、投影输入比状态小、片上重建、不离开 decode 阶段对 PD 分离的意义。✓

全部 5 个目标由正文章节完整回答，无折叠块独占。

## 代码运行

无可运行代码。本页四套方案都是 kernel/系统级机制，伪代码会与图示重复（KCP prefix scan 流程、解码重建流程已用 ASCII 图示表达）；可运行代码需要 CUTLASS/分布式框架（flash-linear-attention、NCCL），超出教学职责。手算例子（S4 折叠块）已足够验证 KCP 分解的正确性。

手算验证（非代码，纯矩阵计算）：S4 的 2 rank × 2 token KDA 递归，ground truth $S_4=\begin{pmatrix}5.5&7\\8.5&10\end{pmatrix}$，KCP 分解重组 $S_T^{[2]}=\tilde S_2+M累积_2\cdot S_T^{[1]}=\begin{pmatrix}5&6\\7&8\end{pmatrix}+0.5I\cdot\begin{pmatrix}1&2\\3&4\end{pmatrix}=\begin{pmatrix}5.5&7\\8.5&10\end{pmatrix}$，一致。误用直接求和得 $\begin{pmatrix}6&8\\10&12\end{pmatrix}\neq S_4$，验证"direct summation is insufficient"。所有矩阵乘法已逐元素手算复算（见折叠块）。

## 机械检查

```
$ python3 .dojo/scripts/validate.py wiki/flash-kda/index.html
validation ok: wiki/flash-kda/index.html

$ python3 .dojo/scripts/validate.py wiki/flash-kda/overview.html
validation ok: wiki/flash-kda/overview.html
```

两项均通过。检查项：<!DOCTYPE html> 与 </html> 完整、无 【】 占位符残留、无 @content/@component/TODO/TBD 标记、无重复 id、无指向缺失 id 的锚点、无断裂本地引用。

补充程序化检查：
- 占位符 【：index.html 0 处、overview.html 0 处。
- 模板标记：index.html 0 处、overview.html 0 处。
- 本地链接解析：../../wiki/kda/index.html、../../wiki/gpu-execution-model/index.html、../../wiki/linear-attention/index.html、../../index.html、index.html、overview.html 均存在。
- h2 id：6 个（serial-vs-parallel, flashkda, intra-device-cp, kcp, kda-decoding, sources-and-teaching-notes），全部唯一。
- 显示公式 delimiter：10 个（5 对），平衡。
- 7 个 diagram 块内无 $ 符号（不会干扰 KaTeX 渲染）。
- 1 个 details 折叠块（S4 手算完整计算）。
- 库文件 ../../libs/katex.min.js、katex.min.css、auto-render.min.js、prism.min.js、prism-python.min.js、prism-primer-light.css、prism-primer-dark.css 均存在。

## 公式渲染与交互

程序化验证（未在浏览器中人工打开，但已逐项核查渲染前提）：
- KaTeX delimiters 配置正确（模板 onload 脚本：`$$` display + `$` inline，throwOnError:false）。
- 5 个显示公式语法均为标准 KaTeX（\boxed、\mathrm、\Diag、\prod、\sum、\tilde、\begin{pmatrix} 等均支持）。
- 显示 delimiter 平衡（5 对）。
- diagram 块内无 $ 干扰。
- 外壳脚本（TOC 生成、滚动高亮、阅读时间、章节折叠、j/k 快捷键、主题切换、代码复制、lightbox）均由模板提供，未改动。
- Prism 代码高亮：本页无 language-python 代码块（无运行代码），仅 1 个 language-text 折叠块内含手算矩阵（非代码，但标 language-text 不会被 Prism 误处理）。

注：本环境无法启动浏览器人工目视渲染结果。上述检查覆盖了渲染正确性的全部可程序化前提；若浏览器中仍有渲染异常，应属 KaTeX 对个别 LaTeX 命令的边界处理，需在 check 阶段人工复核。

## 写作偏差

无。生产阶段严格按 outline.md 落实，未增删核心章节、未增加学习目标、未更换贯穿例子、未改变前置知识映射、未把正文必要内容移入折叠块。S4 手算例子的"误用直接求和错误对照"是 evidence.md C4 论断的自然展开（"direct summation is insufficient"的数值体现），未引入范围外内容。
