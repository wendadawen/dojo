# 标准缩放点积注意力初稿检查

## 输入版本

- `research/scope.md`：完成（概念边界、4 个学习目标 Q1–Q4、内容分级、5 项常见误解、3 个适用边界、前置知识映射 4 项缺失登记、明确不展开项 8 项）。
- `research/evidence.md`：完成（9 条核心论断 C1–C9、6 条公式 F1–F6、3 条数字 N1–N3，全部来自 Vaswani et al. 2017 原论文及 §3.2.1 脚注 4；无存在冲突或证据不足的论断）。
- `research/outline.md`：完成（5 章 S1–S5 + 文末来源、贯穿 2×2 例子在 S2/S3/S4 复用、讲解材料职责表、正文与折叠块分工、范围与证据约束）。
- `research/glossary.md`：完成（术语 30 项、符号 26 项、缩写 7 项；$Q,K,V,d_k,h$ 全文含义一致）。

## 大纲落实

- **章节**：S1 注意力要解决什么问题 / S2 缩放点积公式 / S3 为什么除以 √d_k / S4 多头注意力 / S5 复杂度与边界 / 文末来源与教学说明。全部 6 个 h2 章节标题与 outline.md 一致。
- **学习目标**：4 个学习目标（Q1–Q4）逐项对应 S1–S5 正文章节，没有由折叠块独占的目标。
- **前置知识**：4 项登记的缺失概念（`matrix-multiplication`、`dot-product`、`softmax`、`variance`）在正文首次依赖时给占位提示 + 一句话衔接，不内联大段背景。已有概念页 `residual-connection` 不在本文正文依赖路径上（只在 S5 边界对照表间接提及"残差"作为子层外层），未引入新前置。
- **贯穿例子**：2×2 注意力（$Q=K=\begin{pmatrix}1&0\\0&1\end{pmatrix}$、$V=\begin{pmatrix}1&2\\3&4\end{pmatrix}$、$d_k=2$）在 S2 走完全流程、在 S3 复用作"未缩放 vs 缩放"对照、在 S4 用"两个头各 1 维"的思路示意多头拆分；S5 用一般表达式与对照表。
- **误解和边界**：5 项误解（√dk 经验常数 / 多头=独立模型 / QKV 三种数据 / 标准注意力解决长程依赖 / 注意力权重即模型关注点）分布在页面开头 misconceptions 块与 S4 callout；3 项适用边界（不引入位置、不压缩 KV、不改变复杂度）在 S5 边界对照表。
- **过渡**：S1→S2（"机制在做什么清楚了，但具体怎么匹配加权？"）、S2→S3（"√dk 这步还没解释清楚"）、S3→S4（"单头公式全清楚了，实际不用单头"）、S4→S5（"完整机制有了，看复杂度与边界"）、S5 文末收束（"理解了它就理解了所有变体要保住什么"）。每章末过渡都有。

## 学习目标闭环

- **Q1（注意力解决什么问题）**：S1 正文完整回答。RNN 路径长 $O(n)$ + 顺序操作 $O(n)$ 两个局限写在正文；复杂度对照表展示 RNN/CNN/自注意力对比；数据库类比 callout 列失效边界；自注意力 $Q=XW^Q,K=XW^K,V=XW^V$ 同源投影写在正文。完成检查三问覆盖。
- **Q2（公式与符号与为什么除 √dk）**：S2 + S3 正文完整回答。S2 给公式 F1 + 四步流程 ASCII 图 + 每步形状说明 + 2×2 手算例子（$QK^\top\to\div\sqrt{2}\to\text{softmax}\to\cdot V$ 全部代入与中间结果）+ 形状对照表。S3 给方差推导结论 $\text{Var}(q\cdot k)=d_k$（正文）+ 完整推导（折叠块）+ 不除时 softmax 雅可比与梯度消失（折叠块）+ $d_k=64$ 饱和数字对照（折叠块，标为教学构造）+ 为什么除 $\sqrt{d_k}$ 而不是 $d_k$。
- **Q3（多头公式、拼接、参数量等价）**：S4 正文完整回答。多头公式 F2 + 拼接 ASCII 图 + 参数量等价手算（$h\cdot d_{model}\cdot d_k=d_{model}^2$）+ 论文数字（$h=8,d_k=64,d_{model}=512$）+ 多头 vs 单头对照 + 常见误解 callout + 因果遮罩公式 F5 + 3×3 遮罩例子（折叠块，完整 softmax 计算）。
- **Q4（复杂度瓶颈与边界）**：S5 正文完整回答。复杂度拆解表（$QK^\top$/softmax/$AV$ 三步）+ 瓶颈定位（$QK^\top$ 产生 $n\times n$）+ Flash vs Linear callout（公式不变 vs 改公式）+ 位置无关 + 不压缩 KV + 多头冗余 + 边界对照表（能解决 vs 不能解决 → 后续变体）。

所有目标在折叠块全部收起时仍能由正文回答。

## 代码运行

无可运行代码。本页核心机制（点积 + softmax + 加权）用 2×2 手算例子在 S2 正文逐步代入验证（$QK^\top=\begin{pmatrix}1&0\\0&1\end{pmatrix}$、softmax 得 $\begin{pmatrix}0.670&0.330\\0.330&0.670\end{pmatrix}$、$AV=\begin{pmatrix}1.66&2.66\\2.34&3.34\end{pmatrix}$）；不缩放 vs 缩放对照在 S3 正文；3×3 遮罩 softmax 在 S4 折叠块。所有数字均经独立 Python 复算确认（脚本见 `/tmp` 验证日志，使用 numpy 与 math.sqrt(2)）。

## 机械检查

```bash
$ python3 .dojo/scripts/validate.py wiki/standard-attention/index.html
validation ok: wiki/standard-attention/index.html
$ python3 .dojo/scripts/validate.py wiki/standard-attention/overview.html
validation ok: wiki/standard-attention/overview.html
```

两次运行退出码均为 0。无残留占位符 `【…】`、无 `@content`/`@component`/`TODO`/`TBD` 标记、无重复 id、无指向缺失 id 的同页锚点、无指向不存在文件的本地引用。

## 公式渲染与交互

- **KaTeX 渲染**：index.html 引用 `../../libs/katex.min.css` 与 `../../libs/katex.min.js` + `../../libs/auto-render.min.js`（自动渲染 `$...$` 与 `$$...$$`）。本地路径相对页面解析，文件存在性已用 `curl` 验证（HTTP 200）。文中 13 处 display 公式（F1–F6 + 2×2 例子分步 + 多头公式 + 因果遮罩公式）与若干 inline 公式（$Q,K,V,d_k,\sqrt{d_k},W^O$ 等）均使用标准 KaTeX 分隔符。
- **目录生成**：外壳脚本 `document.querySelectorAll('body > h2, body > h3')` 扫描所有 h2/h3；本页 6 个 h2（5 章 + 来源）+ 6 个 h3（来源小节）均已带显式 id，目录与 j/k 快捷键工作正常。
- **章节折叠按钮**：每个 h2 末尾自动添加 ▼ 按钮，点击切换后续元素 display。本页折叠块（5 个 details）独立工作，不受 h2 折叠影响。
- **亮/暗主题**：外壳脚本读 localStorage 与 prefers-color-scheme，自动切换 KaTeX 与 Prism 主题 CSS。本页未引入自定义颜色，全部使用 CSS 变量。
- **代码块复制**：无代码块（不安排可运行代码），无复制按钮生成需求。
- **图片点击放大**：无 `<img>`，本页用 ASCII 图示（pre.diagram）替代，不需要 lightbox。
- **2×2 数字验证**：手算结果与 Python numpy 复算完全一致（$A=\begin{pmatrix}0.6698&0.3302\\0.3302&0.6698\end{pmatrix}$ 取三位为 $\begin{pmatrix}0.670&0.330\\0.330&0.670\end{pmatrix}$；$AV=\begin{pmatrix}1.660&2.660\\2.340&3.340\end{pmatrix}$）。
- **3×3 遮罩数字验证**：与 Python 复算完全一致（第 1 行 $[1,0,0]$、第 2 行 $[0.401,0.599,0]$、第 3 行 $[0.258,0.316,0.426]$）。

## 写作偏差

无偏差。

- 未自行增删核心章节（5 章 + 来源与教学说明，与 outline.md 一致）
- 未增加新学习目标（4 个 Q 与 scope.md 一致）
- 未更换贯穿例子（2×2 在 S2/S3/S4 复用）
- 未改变前置知识映射（4 项缺失登记，正文用占位提示）
- 未把正文必要内容移入折叠块（所有学习目标结论在正文，折叠块只放完整推导/补充数字/补充例子）
- 未使用证据不足或无法消歧的论断（所有核心论断来自 Vaswani 2017 原论文）

写作中补充的局部内容（未改变大纲）：
- S3 折叠块中"$\sqrt{d_k}$ 是温度 $T=\sqrt{d_k}$"的视角类比，已在文末"教学解释与类比边界"中登记失效边界。
- S4 折叠块中 3×3 遮罩例子的具体分数（$\begin{pmatrix}0.5&1.0&0.8\\0.3&0.7&1.2\\0.4&0.6&0.9\end{pmatrix}$）是教学构造，已在该折叠块开头标注"教学构造"并在文末"教学示例"登记。
- S2 折叠块"softmax 数值稳定实现（max 减法）"是补充材料，对应 glossary.md 中已登记的"softmax 数值稳定实现"边界，未引入新概念。
