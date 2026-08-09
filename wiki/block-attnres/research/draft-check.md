# Block AttnRes 初稿检查

## 输入版本

- `scope.md`：已完成，5 个学习目标（Q1-Q5），核心内容 11 项，辅助内容 5 项，扩展内容 4 项（全排除），前置知识 4 项（残差连接已有概念页、softmax/RMSNorm/自注意力 不递归生成），误解 5 条 + 边界 5 条。
- `evidence.md`：已完成，9 条 C 论断、5 条 F 公式、3 条 N 数字，全部标注来源定位与置信状态。C8（K3 加权三次）与 N2（$N\approx 8$）标注为间接证据。
- `outline.md`：已完成，5 章正文（S1-S5）+ 来源与教学说明，讲解顺序 S1→S2→S3→S4→S5，贯穿例子 $N=3$/$S=2$/$d=2$。
- `glossary.md`：已完成，30 个术语 + 22 个符号，全部首次出现位置已登记。

## 大纲落实

- **章节**：S1 标准残差瓶颈 / S2 Full AttnRes 公式 / S3 Block AttnRes 分块 / S4 K3 具体配置 / S5 RMSNorm 设计选择 + 来源与教学说明，共 6 个 h2（含来源说明）。✓
- **学习目标**：5 个核心问题在页面开头 `learning-goals` 组件中列出，与 scope.md §1.2 一致。✓
- **前置知识**：残差连接在页面开头与 S1 引用 `../residual-connection/index.html`；softmax 与 RMSNorm 在 S2 给最小说明，不递归生成（理由在 scope.md §1.4 与文末教学简化中说明）；自注意力在 S1 用一句话类比，不展开。✓
- **贯穿例子**：$N=3$、$S=2$、$d=2$ 小网络在 S1 末尾引入，S2 手算 Full AttnRes（6 候选、不加 RMSNorm）、S3 手算 Block AttnRes（4 候选、不加 RMSNorm）、S5 手算加 RMSNorm 对比。✓
- **误解和边界**：5 条误解在 scope.md §1.6 登记，正文中处理——"AttnRes 不取消标准残差"在 S1 与 callout 中明确；"9 个候选不是固定值"在 S3 与 S4 中明确；"pseudo-query 不依赖输入"在 S2 中明确；"RMSNorm 不是为数值稳定"在 S5 中明确；"块内不是单纯标准残差"在 S3 中明确。5 条边界在 S5（RMSNorm 两个失效边界）与文末教学简化中处理。✓
- **过渡**：每章末尾有过渡段（S1→S2 公式、S2→S3 内存瓶颈、S3→S4 K3 实例化、S4→S5 RMSNorm 细节），S5 末尾指向文末来源说明。✓

## 学习目标闭环

- **Q1（标准残差瓶颈与 AttnRes 思路）**：S1 正文回答——等权累加把所有历史压成单一流（RNN-over-depth 类比）、AttnRes 用 attention 替代等权累加。正文给出对照表与 ASCII 图示。折叠块全收起时仍完整。✓
- **Q2（Full AttnRes 公式与符号）**：S2 正文回答——Eq.(8)(9) 公式、pseudo-query/keys/values/softmax kernel 符号定义、RMSNorm 最小说明、两个边界检查（key 方向相同/不同）。正文给出小例子的计算结果（$h_6\approx[0.703,0.703]$、权重最大 0.228）。折叠块全收起时仍完整。✓
- **Q3（Block AttnRes 分块与内存）**：S3 正文回答——分块、块内求和 $b_n$、候选集合 Eq.(10)、内存从 $O(Ld)$ 降到 $O(Nd)$ 的来源。正文给出小例子的计算结果（$h_6\approx[1.059,1.241]$、权重最大 0.382）。折叠块全收起时仍完整。✓
- **Q4（K3 具体配置）**：S4 正文回答——93 层、8 块、12 层 size、最后块 9 层、9 个候选来源构成、加权三次位置（含证据层级标注）。正文给出对照表与配置数值表。折叠块全收起时仍完整。✓
- **Q5（RMSNorm 设计选择）**：S5 正文回答——不加 RMSNorm 的大值主导问题、RMSNorm 按方向选择的作用、两个失效边界（所有 key 方向接近时 softmax 仍均匀；RMSNorm 不防 exp 溢出）。正文给出加/不加 RMSNorm 的权重对比结论（$b_1$ 权重从 0.38 降到 0.26）。折叠块全收起时仍完整。✓

## 代码运行

无可运行代码。本页只有伪代码（S3 折叠块内的 Block AttnRes 前向伪代码），用 `language-text` 标记，不是 Python。Block AttnRes 的实际实现涉及大量张量操作（online softmax、TP/SP 通信、checkpointing），教学代码会隐藏核心机制或过于冗长，伪代码已足够展示机制。

## 机械检查

命令：`python3 .dojo/scripts/validate.py wiki/block-attnres/index.html`

结果（2026-08-09）：
```
validation ok: wiki/block-attnres/index.html
```

命令：`python3 .dojo/scripts/validate.py wiki/block-attnres/overview.html`

结果（2026-08-09）：
```
validation ok: wiki/block-attnres/overview.html
```

两个页面均通过机械检查：无模板占位符、无组件标记、无重复 id、无指向缺失 id 的同页锚点、无 broken local reference。

## 公式渲染与交互

- **KaTeX 公式**：所有公式用 `$...$`（行内）或 `$$...$$`（展示）包裹，符合外壳脚本的 auto-render 配置。已检查的公式包括：Eq.(8) keys/values 定义、Eq.(9) softmax kernel 与权重、Eq.(10) 候选集合、RMSNorm 定义、$h_L=h_1+\sum_l F_l(h_l)$、$\alpha_{i\to l}$、$b_n=\sum_{j\in B_n}f_j(h_j)$ 等。未发现未包裹的 LaTeX 命令。
- **ASCII 图示**：3 个 `<pre class="diagram">` 块（S1 深度方向 RNN 瓶颈、S3 Full vs Block 候选集合对比、S4 K3 的 9 个候选来源），用纯 ASCII 字符（`_` 表示下标、`─►` 表示箭头），不参与 KaTeX 渲染，避免冲突。
- **表格**：5 个表格（S1 标准残差 vs AttnRes 对照、S4 Full vs Block vs K3 实例化、S4 config.json 字段、S5 加/不加 RMSNorm 权重对比、以及 S2/S3 折叠块内的小例子逐项计算），全部用 `<div class="table-scroll"><table>` 结构。
- **折叠块**：8 个 `<details>` 块（S2 Full AttnRes 手算、S2 softmax 敏感性推导、S3 Block AttnRes 手算、S3 伪代码、S5 加 RMSNorm 手算、S5 softmax 敏感性推导），全部有具体 summary 标题，不写"更多""详情"。
- **浏览器交互**：外壳脚本提供目录、章节折叠、暗/亮模式、代码复制、图片放大、j/k 快捷键。本页无图片，代码复制按钮对伪代码块生效。暗/亮模式切换时 Prism 主题同步切换（本页无 Python 代码，但伪代码用 `language-text` 仍会被 Prism 处理）。
- **侧边目录**：外壳脚本自动从 h2/h3 生成。本页 6 个 h2（S1-S5 + 来源与教学说明），目录应正常显示。
- **未在浏览器中实际打开**：本检查基于代码审查，未在真实浏览器中打开页面验证 KaTeX 渲染与交互。如需最终确认，应在浏览器中打开 `wiki/block-attnres/index.html` 检查公式渲染、目录跳转、折叠块展开、暗/亮模式切换。

## 写作偏差

无重大偏差。两处局部修正：

1. **S2 的 Full AttnRes 小例子**：outline.md 计划"先不加 RMSNorm 简化，让读者先理解 attention 检索的本质"。实际写作时发现若不先解释 RMSNorm，读者会疑惑为什么 S5 才加。处理方式：S2 正文明确标注"本节暂时不加 RMSNorm，S5 再加 RMSNorm 重算并对比"，并在 S2 末尾加一个"softmax 大值敏感性"补充推导折叠块作为 S5 伏笔。这是大纲内的局部衔接，不改变大纲结构。

2. **S4 的证据层级标注**：outline.md 计划"明确标注哪些来自 K3 报告原文、哪些来自源码核对"。实际写作时把 C8 的证据层级拆成两个 bullet（K3 报告原文确认 + 源码核对间接证据），并在正文与文末来源说明中重复标注。这是为了读者能在正文就看到证据层级，不需要翻到文末。不改变大纲结构。

## 完成条件自检

- 输入产物齐全：scope/evidence/outline/glossary 均已完成。✓
- 大纲全部章节、学习目标、前置知识、完成检查、过渡均已落实。✓
- 学习目标闭环：5 个目标全部由正文章节完整回答，折叠块全收起时正文仍完整。✓
- 来源事实附来源定位：C1-C9、F1-F5、N1-N3 在文末来源说明中逐条标注。✓
- 教学构造和教学解释均已标记：所有教学示例标注"教学示例"，类比标注"教学解释"并给出失效边界，简化在文末"教学简化及其限制"中逐项说明。✓
- C/F/N 引用与 evidence.md 一致：正文中 C 论断的描述与 evidence.md 一致，公式编号 F1-F5 与 evidence.md 一致。✓
- 占位符、组件标记、写作注释已清除：validate.py 通过。✓
- `.dojo/scripts/validate.py` 通过：两个页面均通过。✓
- 折叠块全部收起时正文仍能回答全部学习目标：已在"学习目标闭环"中逐题核对。✓
- draft-check.md 已填写：本文件。✓

## 遗留问题

1. **公式渲染未在浏览器中实际检查**：本检查基于代码审查，未在真实浏览器中打开页面。建议在浏览器中打开 `wiki/block-attnres/index.html` 确认 KaTeX 渲染（特别是 Eq.(8)(9)(10) 的展示公式）与目录跳转、折叠块展开等交互。
2. **K3 加权三次的间接证据**：C8 的具体三次位置（attention 前 / MLP 前 / final norm 前）来自 `wiki/kimi-k3-dataflow/` 对源码的核对，本页未直接复核 `modeling_kimi_linear.py`。若 dataflow note 修订，本页相应描述需同步修订。
3. **AttnRes 原 preprint [57] 未获取**：$N\approx 8$ 的最优性、原 preprint 的消融实验等结论只用 K3 报告 §2.2 的转述。若需核对，应获取原 preprint。
4. **partial block 的具体层数为推算**：K3 报告未明说最后一个 block 有几层，本页由 $93=7\times 12+9$ 推算为 9 层，已标注为推算结果。
