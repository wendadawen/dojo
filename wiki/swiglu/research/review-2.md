# SwiGLU 独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源）
- 页面版本：index.html ac5b744、overview.html ac5b744
- 时间：2026-08-09
- 来源：Shazeer 2020《GLU Variants Improve Transformer》(arXiv:2002.05202)，通过 ar5iv HTML 版逐条核对 §1–§4 与 Table 1；LLaMA/PaLM 采用情况通过 WebSearch 多源交叉。

## 问题

- [轻微·盲读] "从 GLU 到 SwiGLU" 章对比表（sigmoid vs Swish 行）："取值范围"列 Swish 写作 $(-\epsilon,+\infty)$，$\epsilon$ 未在表内或表前定义数值与含义；小白读者看到 $-\epsilon$ 不知道下界是 -0.001 还是 -0.3，削弱了"门能取负"这一核心对比点的具体性。后文"负侧先下降到一个小负值（约 $z\approx-1.278$ 处达到最小 $\approx-0.278$）"才给出数字，但表格作为关键对比工具应自洽。修法：将表格 Swish 取值范围改为 $\approx(-0.278,+\infty)$，或在表格脚注一行注明"$\epsilon$ 指 Swish 最小值 $\approx 0.278$"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] "从 GLU 到 SwiGLU" 章 GLU/SwiGLU 结构图：GLU 图按 Dauphin 记法画（$\sigma$ 在 $V$ 分支、$xW+b$ 为值分支），SwiGLU 图按 Shazeer 记法画（$\mathrm{Swish}$ 在 $W$ 分支、$xV+c$ 为值分支）。两图并排时 $W$ 在 GLU 图是值分支、在 SwiGLU 图是门分支，$V$ 反之，小白读者会卡在"为什么同一字母在两图里角色互换"。页面在图后一段解释了两记法相差 $W\leftrightarrow V$ 标签且等价，但解释位于图后、图本身制造了可避免的首读卡点。修法：两图统一为同一记法（建议都用 Shazeer 记法，与正文公式 $\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$ 一致），或在两图前加一行"下图统一采用 Shazeer 记法"并删去 GLU 图的 Dauphin 标注。 ｜ 修复： ｜ 复验：
- [轻微·盲读] 公式章符号说明（$b,c$ 条）写"Shazeer §2 在 FFN 部署中省略偏置（见 S3）"；"经验结论与边界"章写"把 S1 提到的派生规则完整列出"。页面章节标题为中文标题（"从 GLU 到 SwiGLU"、"SwiGLU 的公式、Swish 定义与手算"、"把 SwiGLU 塞进 Transformer FFN"等），无 S1/S3 编号标记，小白读者无法定位"S3"指哪一节。修法：将"S1""S3"改为对应章节标题或锚点链接（如"见<a href="#ffn-deployment">把 SwiGLU 塞进 Transformer FFN</a>"），或直接去掉这些内部编号。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：进入修复（三条均为轻微，不阻断发布；建议修复以提升首读体验）

## 段 A 盲读小结

扮演完全小白读者按页面顺序阅读。主线理解顺畅：从 GLU 的 sigmoid 门"恒正有界只能压低/放行"出发，引出"换门"动机；用 Swish 定义与三个边界值（0、0.731、-0.269）把形状落到数字；手算例子 $x=[1.0,0.5]$、$W=\mathrm{diag}(1,-2)$ 验证"第二维翻号"这一核心机制；FFN 部署的 $2/3$ 缩放与 LLaMA $\tfrac83 d$ 推导链完整；经验结论明确标注"Shazeer 未给理论解释""GEGLU 略优""社区选 SwiGLU 是路径依赖"三条边界。学习目标五条均由正文章节完整回答。卡点仅上述三条轻微项，不阻断主线。

## 段 B 对照来源小结

逐条核对 Shazeer 2020 §1–§4 与 Table 1：

1. 定义与机制：SwiGLU 定义 $\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$（§2 Eq.5）一致；FFN_SwiGLU $(\mathrm{Swish}_1(xW)\otimes xV)W_2$（§2 Eq.6）一致；偏置省略（§2 "Again, we omit the bias terms"）一致。
2. 公式与推导：Swish 定义 $\mathrm{Swish}_\beta(z)=z\sigma(\beta z)$（§1）一致；$\mathrm{Swish}(0)=0$、$\mathrm{Swish}(1)\approx0.7311$、$\mathrm{Swish}(-1)\approx-0.2689$ 由定义复算一致；Swish 最小值 $z\approx-1.278$、$\approx-0.278$ 复算一致；参数量等式 $3d\cdot d_{ff}'=2d\cdot d_{ff}\Rightarrow d_{ff}'=\tfrac23 d_{ff}$ 复算一致；$\tfrac23\times4d=\tfrac83 d$ 复算一致；$3072\times\tfrac23=2048$ 复算一致。
3. 可运行代码：页面无可运行代码块（仅有 KaTeX 公式与静态图表），不适用。
4. 事实与推断：Table 1 八行 log-perplexity（ReLU 1.677 / GELU 1.679 / Swish 1.683 / GLU 1.663 / Bilinear 1.648 / ReGLU 1.645 / SwiGLU 1.636 / GEGLU 1.633）逐行核对一致；§2 末段 $2/3$ 原文引用一致；§4 "divine benevolence" 原文引用一致；§1 $\beta=1$ 固定一致。实验设置（T5 base、$d_{model}=768$、编码/解码各 12 层、基线 $d_{ff}=3072\to2048$、524,288 步）核对一致。教学构造均标注"数字为教学构造"。
5. 前置知识引用：GLU 概念页链接 `../../wiki/glu/index.html` 有效；SiTU-GLU 概念页链接 `../../wiki/situ-glu/index.html` 有效。
6. 教学简化：Swish 完整讲解、GELU/ReLU 机制、Transformer 架构、各 LLM 部署代码均标注简化理由与可/不可推出边界，未发现简化导致核心结论失真。
7. 页面功能：KaTeX 公式渲染正常；details 折叠交互正常；侧边目录锚点（from-glu-to-swiglu / formula-and-example / ffn-deployment / evidence-and-boundary / sources-and-teaching-notes）有效。

未发现来源不一致项。
