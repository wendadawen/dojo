# Kimi Delta Attention（KDA）初稿检查

## 输入版本

- `scope.md`：完成。概念歧义已裁定（采用 K3 §2.1.1 定义的 KDA）；5 个学习目标（Q1-Q5）均有完成答案；核心/辅助/扩展内容分级完成；前置知识映射到 `wiki/delta-rule/index.html`、`wiki/linear-attention/index.html`（均已有，递归深度 0）；4 条常见误解 + 适用边界记录完成。
- `evidence.md`：完成。10 条 C 类核心论断（C1-C10）、6 条 F 类公式（F1-F6）、6 条 N 类数字（N1-N6）全部完成来源定位与置信状态；无冲突；无未确认项。
- `outline.md`：完成。6 个正文章节（S1-S6）+ 文末来源与教学说明；每章单一教学任务、讲解顺序、贯穿例子（4 维 3 步迷你序列，4 次出场）、材料职责、正文与折叠块分工齐全。
- `glossary.md`：完成。术语表 + 缩写表 + 符号表 + 约定说明（状态布局转置）齐全。

## 大纲落实

- **章节**：S1（why-k3-needs-kda）、S2（kda-recurrence）、S3（lower-bounded-decay）、S4（parameterization-and-gate）、S5（chunkwise-parallel）、S6（k3-config-and-boundaries）+ 文末来源与教学说明（sources-and-teaching-notes）。6 个 h2 + 文末 h2，与 outline 一致。
- **学习目标**：页面开头 learning-goals 组件列出 Q1-Q5 五条，与 scope.md 一致。
- **前置知识**：S1 首次依赖线性注意力结论时给链接（`../../wiki/linear-attention/index.html`）；S1、S2 首次依赖 delta rule 结论时给链接（`../../wiki/delta-rule/index.html`）。共 3 处链接，不内联重复讲解。
- **贯穿例子**：4 维 3 步迷你序列，4 次出场——S2 第 1、2 步（衰减+擦写）、S3 反推 $g$ 与 $z$ + $z=1,A=0$ 对比、S5 $C=3$ chunk 手算 $\Gamma$。全部标注"教学示例"。
- **误解和边界**：页面开头 misconceptions 组件列 4 条（KDA≠DeltaNet、lower-bound≠negative-softplus、forget gate 顺序、KDA≠K3 全部注意力）；S6 末尾"适用边界"小节。
- **过渡**：每章末尾有过渡句指向下一章（S1→S2 "递归具体长什么样"、S2→S3 "$\alpha$ 从哪来"、S3→S4 "还要能训练"、S4→S5 "串行跑不动"、S5→S6 "落到具体数值"）。

## 学习目标闭环

- **Q1（KDA 在 K3 里做什么）**：S1 正文回答。包含 KV cache 爆炸估算（教学估算）、线性注意力固定状态引用、delta rule key 碰撞引用、69/24/93 比例与 3:1 混合。完成答案与 scope.md 一致。
- **Q2（channel-wise forget gate 怎么改写递归）**：S2 正文回答。Eq.1 三步分解（衰减→擦除→写入）、$\alpha$ vs $\beta$ 职责差异、第 2 步手算。完成答案一致。
- **Q3（lower-bounded decay vs negative-softplus）**：S3 正文回答。两种映射公式与范围对比、$g_{\min}=-5$、$\alpha > e^{-5}$、$1/\Gamma < e^{80}$、对角 tile 用 Tensor Core。完成答案一致，含手算 $z=1,A=0$。
- **Q4（full-rank gate 与 chunkwise 并行）**：S4 + S5 正文回答。参数化链、full-rank gate 公式、$\Gamma$、Tril、inter/intra 分工。完成答案一致。
- **Q5（K3 配置与边界）**：S6 正文回答。69/24/93、数值表（$g_{\min}$、head_dim、num_heads、short_conv、hidden、max_pos）、KDA 不解决的三件事、适用边界。完成答案一致。

全部目标由正文章节完整回答，无折叠块独占。

## 代码运行

无可运行代码。outline.md 已说明：KDA 是模型层机制，不是独立算法；伪代码会与 Eq.1/2/4 重复；可运行代码需要完整训练框架（flash-linear-attention、CUTLASS），超出教学职责。chunkwise 形式本身已是并行算法描述，不再额外伪代码化。

## 机械检查

- `python3 .dojo/scripts/validate.py wiki/kda/index.html` → 退出码 0，输出 `validation ok: wiki/kda/index.html`。
- `python3 .dojo/scripts/validate.py wiki/kda/overview.html` → 退出码 0，输出 `validation ok: wiki/kda/overview.html`。
- 占位符检查：`【|@content|@copy|@component` 在 index.html 与 overview.html 均无匹配。

## 公式渲染与交互

- KaTeX 公式使用标准 `$...$`（行内）与 `$$...$$`（展示）标记，外壳脚本已配置 auto-render。
- 未在浏览器实际打开页面验证渲染与交互（本阶段为 plan+write，check 阶段由编排者另行安排独立审查）。validate.py 通过保证结构合规；公式语法经自查与前置页（delta-rule、linear-attention 用相同标记）一致。

## 写作偏差

无返回规划的偏差。两处局部补充（不改变大纲）：

1. outline.md 在贯穿例子第 1 次出场处记录了构造修正（$k_1$ 从 $(1,0,0,0)^\top$ 改为 $(1,1,1,1)^\top/2$，让 $\alpha$ 的通道差异在 $S_1$ 多行非零时可见）。实际写作按修正后的构造落笔，与 outline 最终版一致。
2. S1 的 1M KV cache 估算在正文用教学估算形式呈现（48 GB 量级），完整代入与简化放折叠块。outline.md 已规划此分工。

## 遗留问题

1. **公式渲染待浏览器验证**：本阶段未在浏览器实际打开页面确认 KaTeX 渲染与折叠块交互，留待 check 阶段在浏览器验证。
2. **UT 变换未展开**：Eq.4 的 $V_e = U - WS$ 按 K3 报告指示指向 Kimi Linear [63]，本文只引用结论。读者若需理解 $U, W$ 的具体形式需查 [63]。这是大纲明确决定的排除项，不是缺口。
3. **content.json 未更新**：按任务要求，不更新 content.json 与首页，由编排者安排。
4. **check 阶段未执行**：按任务要求，check 阶段由编排者另行安排独立审查。
