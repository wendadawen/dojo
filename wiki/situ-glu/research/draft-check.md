# SiTU-GLU 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成（research/ 目录），规划完成条件全部满足：概念歧义已裁定（K3 §2.3.2 专有术语，无同名歧义）；Q1–Q5 五个互不重复学习目标各有书面完成答案；核心内容逐项对应学习目标；前置知识（GLU/SwiGLU 已有概念页 [wiki/glu/index.html](../../wiki/glu/index.html) 链接，sigmoid/$\tanh$/$\odot$/$W x$/hard clamping/$O(\cdot)$ 按基础记号内联）；核心论断 C1–C10 均完成来源定位与置信状态（全部已确认）；误解与边界具体可查（5 条误解 + 适用边界 4 项）；大纲章节单一任务、讲解顺序、贯穿例子、材料职责、正文折叠块分工齐备；术语表齐全。

- 大纲落实（逐项一行）：
  - 页面开头：钩子 callout（"两个数乘积怎么涨"）、context-box（是什么/解决什么/提出场景/家族角色，含 GLU 链接）、learning-goals（Q1–Q5）、misconceptions（4 条最容易误解）、blockquote.meta（K3 报告 + GLU 概念页 evidence 链接）——落实。
  - S1 为什么需要给 SwiGLU 加上界：SwiGLU 公式回顾（含 GLU 页链接）、两个无界因子、K3 四连矩阵放大场景、GLU 不够的两个理由、softcap 函数定义、一句话定位、三变体对照表（GLU/SwiGLU/SiTU-GLU 的门支线性因子/激活/值支/上界）、完成检查——落实。
  - S2 公式与手算：F1 公式 + 逐符号（含 β1=4/β2=25、保留 σ 因子的"线性因子"说明）、三个边界检查（x=0/+∞/-∞）、贯穿例子 x=0/10/100 与 SwiGLU 对照表、$x=2,50$ 折叠块、完成检查——落实。
  - S3 近原点像 SwiGLU、远点饱和：F3 局部展开 + 代入 F1 推一阶等价、F4 上界证明 + "两支都套"必要条件、F5 极限行为、$\beta$ 选择标注工程设定；$\tanh$ 泰勒展开推导折叠块、上界证明逐项拆解折叠块；完成检查——落实。
  - S4 softcap vs hard clamping：clip 定义 F8、softcap F2、输出/导数对照表、梯度对比 F7/F9、K3 §B 末段原文引用、"away from saturation boundaries"边界澄清、"better training behavior"经验陈述标注；$\tanh$ 指数渐近推导折叠块；完成检查——落实。
  - S5 在 K3 中的使用位置与不解决：Stable LatentMoE 路由 FFN + Dense FFN 两位置（C9/C10）、三件套对照表（RMSNorm/SiTU-GLU/QB）、不解决项（QB/MLA/性能/低精度）、5 条误解收尾、完成检查——落实。
  - 文末来源与教学说明：核心论断/公式/数字/教学示例/教学解释边界/教学简化限制六小节齐全——落实。

- 学习目标闭环（逐题）：
  - Q1（做什么/解决什么）：S1 给动机（SwiGLU 两因子无界 + K3 四连矩阵放大 + GLU 不够）+ 一句话定位 + 三变体对照表；S2 给定义公式与符号 ⇒ 正文完整回答。不依赖折叠块。
  - Q2（近原点 + 有界同时成立）：S3 给 F3 局部展开代入 F1 推一阶等价、F4 上界证明、F5 极限行为、上界必要条件（误解 4）⇒ 正文完整回答。推导在折叠块但结论在正文。
  - Q3（手算 x=0/10/100）：S2 正文给出三点的门支/值支/输出代入与中间值（$0, 37.485, 99.933$）+ SwiGLU 对照（$0, 99.995, 10000$）⇒ 正文完整回答。中间点 $x=2,50$ 在折叠块但主线三点在正文。
  - Q4（softcap vs clip 梯度差别）：S4 给 clip 与 softcap 的输出/导数对照表、clip 在 $|x|>c$ 严格 0、softcap $1-\tanh^2(x/\beta)$ 指数衰减非零、K3 §B 末段原文 ⇒ 正文完整回答。$\tanh$ 渐近推导在折叠块但结论在正文。
  - Q5（使用位置 + 不解决）：S5 给两位置（Stable LatentMoE 路由 FFN + Dense FFN）+ 三件套对照表 + 4 项不解决 ⇒ 正文完整回答。
  - 折叠块全收起测试：S2 中间点手算、S3 泰勒展开与上界拆解、S4 $\tanh$ 渐近推导三处折叠块收起后，正文仍含 Q1–Q5 完整答案 ⇒ 通过。

- 代码运行：本页无可运行代码（大纲未分配可运行代码组件——手算例子与公式推导已承担机制验证职责，与 [GLU 概念页](../../wiki/glu/index.html) 一致）。无代码块需运行。

- 机械检查：
  - 命令：`python3 .dojo/scripts/validate.py wiki/situ-glu/index.html` → 退出码 0，输出 "validation ok: wiki/situ-glu/index.html"。
  - 命令：`python3 .dojo/scripts/validate.py wiki/situ-glu/overview.html` → 退出码 0，输出 "validation ok: wiki/situ-glu/overview.html"。
  - 无占位符【…】残留、无 @content/@component/TODO/TBD 标记残留、无重复 id、无指向缺失 id 的锚点、无断裂本地引用（../../libs/ 与 ../../index.html、index.html、overview.html、../../wiki/glu/index.html、../../wiki/glu/research/evidence.md 均存在）。

- 公式渲染与交互：KaTeX 本地资源（../../libs/katex.min.css / katex.min.js / auto-render.min.js）在两页 head 中正确引用；行内 $...$ 与行间 $$...$$ 定界符与 auto-render 配置一致；Prism 资源在 index.html 中引用。结构上目录、章节折叠按钮、j/k 快捷键、复制按钮、返回顶部均由外壳脚本提供。待浏览器实际打开复核渲染（见 check 阶段）。

- 手算数字复核（用 Python `math.tanh / math.exp` 复算，保留四位有效数字）：
  - SiTU-GLU $x=0$：$g=0, u=0, y=0$ ✓
  - SiTU-GLU $x=10$：$g=4\tanh(2.5)\sigma(10)=4\times 0.98661\times 0.99995=3.9463$，$u=25\tanh(0.4)=25\times 0.37995=9.4987$，$y=37.4846\approx 37.485$ ✓
  - SiTU-GLU $x=100$：$g=4\tanh(25)\sigma(100)\approx 4\times 1\times 1=4.0000$，$u=25\tanh(4)=25\times 0.99933=24.9832$，$y=99.9329\approx 99.933$ ✓
  - SwiGLU $x=0\to 0$ ✓；$x=10\to 10\sigma(10)\cdot 10=99.995$ ✓；$x=100\to 100\sigma(100)\cdot 100=10000$ ✓
  - 中间点 $x=2$：SiTU $g=1.6281, u=1.9957, y=3.2493$；SwiGLU $2\sigma(2)\cdot 2=3.5232$ ✓（修正后）
  - 中间点 $x=50$：SiTU $g\approx 4, u=24.101, y=96.403$；SwiGLU $50\sigma(50)\cdot 50=2500$ ✓
  - 局部展开验证 $z=0.5,\beta=4$：$\beta\tanh(z/\beta)=4\tanh(0.125)=0.49741$，与 $z=0.5$ 差 $-0.00259$；理论 $-z^3/(3\beta^2)=-0.125/(3\times 16)=-0.00260$，吻合四位有效数字 ✓
  - $\tanh(4)=0.99933$，$1-2e^{-8}=0.99933$，吻合 ✓
  - 饱和区梯度 $x=100,\beta=4$：$4e^{-50}=7.7e-22$（修正后，原误写为 1.4e-21）；$x=100,\beta=25$：$4e^{-8}=1.3e-3$ ✓

- 写作偏差：
  - 初稿写出后复核发现两处手算数字错误并已修正：(a) S2 折叠块 SwiGLU(2) 由 3.5286 改为 3.5232；(b) S4 折叠块 $4e^{-50}$ 由 1.4e-21 改为 7.7e-22。修正后与 Python 复算结果一致。
  - 无其它偏差。所有内容来自 scope.md 已纳入范围与 evidence.md 已确认论断；教学数字均标记"教学示例"；教学解释与类比均标明职责与失效边界；教学简化（两支共享同一 pre-act 标量、用标量代替向量、clip 单一定义、Stable LatentMoE 四连矩阵只用一句话、基础记号内联）均在文末"教学简化及其限制"逐项写明。C/F/N 引用与 evidence.md 一致。

- 来源事实与外部来源核对：
  - K3 §2.3.2 Eq.(12) SiTU-GLU 定义、§2.3.2 末段 β1=4/β2=25 设定、§2.3.2 第二段 "linear factor of the Swish gate" 与 GLU 不够的两理由、§2.3 开头四连矩阵 + 2.8T 激活爆炸、§B Eq.(18) 局部展开 + 极限行为、§B Eq.(19) 上界证明、§B 末段 softcap vs hard clamping "preserves nonzero gradients ... better training behavior"、§4 对比表 SwiGLU→SiTU-GLU——经 `/tmp/kimi-k3-research/k3-report.txt` 行号定位核对一致。
  - F6 SwiGLU 公式引用 Shazeer 2020 §2 Eq.(5)，已由 [GLU 概念页 evidence F6](../../wiki/glu/research/evidence.md) 登记，本文以链接引用方式使用。
  - 未引用范围外来源；未使用网络博客或聚合文章作为核心论断依据。
