# GLU 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成（research/ 目录），规划完成条件全部满足：概念歧义已裁定（已采纳 Dauphin 主流含义，记号差异在 scope §1 与正文 S4 处理）；Q1–Q5 五个互不重复学习目标各有书面完成答案；核心内容逐项对应学习目标；前置知识（sigmoid、⊗、xW+b、链式法则）按基础记号内联处理，已记录生成状态；核心论断 C1–C9 均完成来源定位与置信状态（全部已确认）；误解与边界具体可查；大纲章节单一任务、讲解顺序、贯穿例子、材料职责、正文折叠块分工齐备；术语表齐全。

- 大纲落实（逐项一行）：
  - 页面开头：钩子（callout）、context-box（位置/解决/提出场景/家族角色）、learning-goals（Q1–Q5）、blockquote.meta（两篇论文）——落实。
  - S1 为什么需要"门"：动机两层缺口（按维度调节 / 不杀梯度）、门控直觉、一句话定位、ASCII 对照图、完成检查——落实。
  - S2 公式与手算：F1 公式 + 逐符号、维度约定、三个边界检查、贯穿例子 $x=[1.0,0.5]$ 手算、极端对照折叠块、完成检查——落实。
  - S3 梯度通路：简化形式说明、F4 梯度两项职责、对照 F3 GTU、对照表、推导折叠块、边界（门全关仍消失）、完成检查——落实。
  - S4 家族：Bilinear F5、ReGLU/GEGLU/SwiGLU F6、记号差异对照表、FFN_GLU F7、参数量 F8 推导、3072→2048 实例、参数量折叠块、完成检查——落实。
  - S5 经验与边界：Table 1（N1）、divine benevolence 原句、不解决项、4 条误解收尾、完成检查——落实。
  - 文末来源与教学说明：核心论断/公式/数字/教学示例/教学解释边界/教学简化限制六小节齐全——落实。

- 学习目标闭环（逐题）：
  - Q1（做什么/解决什么）：S1 给动机与一句话定位 + S2 给定义公式与符号 ⇒ 正文完整回答。不依赖折叠块。
  - Q2（梯度线性通路）：S3 给 F4 两项职责、F3 GTU 对照、对照表、边界 ⇒ 正文完整回答。推导在折叠块但结论在正文。
  - Q3（手算）：S2 正文给出 $x=[1.0,0.5]$ 的值分支、门分支、相乘三步与结果 $[0.7311,0.1888]$ ⇒ 正文完整回答。极端对照在折叠块但主线例子在正文。
  - Q4（家族派生 + 记号差异）：S4 给派生规则、F5/F6、Dauphin vs Shazeer 对照表 ⇒ 正文完整回答。
  - Q5（2/3 缩放 + 不保证）：S4 给 F8 等式与 3072→2048 实例；S5 给 Table 1、divine benevolence、不解决项 ⇒ 正文完整回答。
  - 折叠块全收起测试：S2 极端对照、S3 推导、S4 参数量验算三处折叠块收起后，正文仍含 Q1–Q5 完整答案 ⇒ 通过。

- 代码运行：本页无可运行代码（大纲未分配可运行代码组件——手算例子与公式推导已承担机制验证职责）。无代码块需运行。

- 机械检查：
  - 命令：`python3 .dojo/scripts/validate.py wiki/glu/index.html` → 退出码 0，输出 "validation ok: wiki/glu/index.html"。
  - 命令：`python3 .dojo/scripts/validate.py wiki/glu/overview.html` → 退出码 0，输出 "validation ok: wiki/glu/overview.html"。
  - 无占位符【…】残留、无 @content/@component/TODO/TBD 标记残留、无重复 id、无指向缺失 id 的锚点、无断裂本地引用（../../libs/ 与 ../../index.html、index.html、overview.html 均存在）。

- 公式渲染与交互：KaTeX 本地资源（../../libs/katex.min.css / katex.min.js / auto-render.min.js）在两页 head 中正确引用；行内 $...$ 与行间 $$...$$ 定界符与 auto-render 配置一致；Prism 资源在 index.html 中引用。结构上目录、章节折叠按钮、j/k 快捷键、复制按钮、返回顶部均由外壳脚本提供。待浏览器实际打开复核渲染（见 review 阶段）。

- 手算数字复核：
  - σ(1.0)=1/(1+e^-1)=0.73106≈0.7311 ✓；σ(-0.5)=1/(1+e^0.5)=0.37754≈0.3775 ✓；GLU=[1.0×0.7311, 0.5×0.3775]=[0.7311, 0.1888] ✓。
  - 极端：σ(10)≈0.99995、σ(-5)≈0.00669 ⇒ [1.0, 0.0] ✓。
  - 参数量：2×768×3072=4,718,592；3×768×2048=4,718,592 ✓。

- 写作偏差：无。所有内容来自 scope.md 已纳入范围与 evidence.md 已确认论断；教学数字均标记"教学示例"；教学解释与类比均标明职责与失效边界；教学简化（$*$ 退化为矩阵乘、梯度分析用共享输入简化形式、Swish/GELU/ReLU 作为名不展开、基础记号内联）均在文末"教学简化及其限制"逐项写明。C/F/N 引用与 evidence.md 一致。

- 来源事实与外部来源核对：
  - Dauphin §2 Eq.(1) GLU 定义、§3 Eq.(2)(3) GTU/GLU 梯度、§5.3 Bilinear、§2"linear path for the gradients"、§3"multiplicative skip connection"——经 ar5iv 核对一致。
  - Shazeer §2 Eq.(4)(5)(6) 变体与 FFN 定义、§2 末段 2/3 缩放、§3.1 3072→2048、Table 1 八行 log-perplexity、§4"divine benevolence"——经 ar5iv 核对一致。
  - 记号差异 C9：Dauphin σ 在 V 分支、Shazeer σ 在 W 分支——两篇原文公式直接比对确认。
