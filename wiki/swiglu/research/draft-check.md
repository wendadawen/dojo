# SwiGLU 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成（research/ 目录），规划完成条件全部满足：概念歧义已裁定（SwiGLU 主流唯一含义，Shazeer 记法 vs Dauphin 记法在 scope §1 与正文 S2 处理）；Q1–Q5 五个互不重复学习目标各有书面完成答案；核心内容逐项对应学习目标；前置知识（GLU 已有概念页、sigmoid/⊗/xW+b/FFN 基础记号）已记录生成状态；核心论断 C1–C12 均完成来源定位与置信状态（全部已确认）；误解与边界具体可查（scope §2.6 五条误解）；大纲章节单一任务、讲解顺序、贯穿例子、材料职责、正文折叠块分工齐备；术语表齐全。

- 大纲落实（逐项一行）：
  - 页面开头：callout 钩子（LLM 标配提问）、context-box（是什么/解决什么/提出场景/家族角色，含 GLU 与 SiTU-GLU 链接）、learning-goals（Q1–Q5）、blockquote.meta（Shazeer 2020 主要依据）——落实。
  - S1 从 GLU 到 SwiGLU：GLU 回顾+链接、sigmoid 门形状限制（恒正有界）、Shazeer 换门动作、Swish 定义、sigmoid vs Swish 对照表、ASCII 结构对照图、一句话定位、完成检查——落实。
  - S2 公式与手算：F2 SwiGLU 公式+逐符号、Shazeer 记法标注、β 说明、Swish 边界值（Swish(0)=0、Swish(1)≈0.731、Swish(-1)≈-0.269）、维度约定、贯穿例子 x=[1.0,0.5] 手算、与 GLU 同输入对照（折叠块）、Swish 边界值逐步代入（折叠块）、完成检查——落实。
  - S3 FFN 部署：FFN_ReLU 双矩阵、FFN_SwiGLU 三矩阵 F3、参数量等式 F4、Shazeer §2 末段原文引用、§3.1 实例 3072→2048、LLaMA 风格 8/3 d 推导 F5、LLaMA/PaLM 部署事实 C10、对照表、参数量验算折叠块、完成检查——落实。
  - S4 经验与边界：Table 1（N1）、GEGLU 略优于 SwiGLU、divine benevolence 原句、社区采用路径（PaLM/LLaMA/工具链锁定）、GLU 家族五变体派生对照表、不解决项（含 SiTU-GLU 下游关系 C12）、5 条误解收尾、完成检查——落实。
  - 文末来源与教学说明：核心论断/公式/数字/教学示例/教学解释边界/教学简化限制六小节齐全——落实。

- 学习目标闭环（逐题）：
  - Q1（做什么/与 GLU 关系）：S1 给动机（sigmoid 门恒正有界 vs Swish 可负无界）与一句话定位 + S2 给定义公式与符号 ⇒ 正文完整回答。不依赖折叠块。
  - Q2（公式+Swish 定义+手算+边界值）：S2 正文给出 F2 公式、F1 Swish 定义、Swish(0)=0/Swish(1)≈0.731/Swish(-1)≈-0.269 边界值、x=[1.0,0.5] 手算三步与结果 [0.7311,-0.1345] ⇒ 正文完整回答。同输入 GLU 对照与 Swish 边界值逐步代入在折叠块但主线手算在正文。
  - Q3（2/3 缩放+参数量+8/3 d）：S3 给 F3 FFN 公式、F4 参数量等式推导、§3.1 实例 3072→2048、F5 LLaMA 风格 8/3 d 推导 ⇒ 正文完整回答。参数量验算在折叠块但等式与结论在正文。
  - Q4（派生关系+Swish vs sigmoid 机制差别）：S1 给 Swish vs sigmoid 形状对照表 + S2 手算对照（SwiGLU 第二维 -0.1345 vs GLU +0.1345）+ S4 派生对照表 ⇒ 正文完整回答。
  - Q5（经验结论+边界+社区选择）：S4 给 Table 1 数字、GEGLU 略优、divine benevolence、社区采用路径、不解决项、SiTU-GLU 下游关系 ⇒ 正文完整回答。
  - 折叠块全收起测试：S2 同输入对照、S2 Swish 边界值代入、S3 参数量验算三处折叠块收起后，正文仍含 Q1–Q5 完整答案 ⇒ 通过。

- 代码运行：本页无可运行代码（大纲未分配可运行代码组件——手算例子与公式推导已承担机制验证职责）。无代码块需运行。

- 机械检查：
  - 命令：`python3 .dojo/scripts/validate.py wiki/swiglu/index.html` → 退出码 0，输出 "validation ok: wiki/swiglu/index.html"。
  - 命令：`python3 .dojo/scripts/validate.py wiki/swiglu/overview.html` → 退出码 0，输出 "validation ok: wiki/swiglu/overview.html"。
  - 无占位符【…】残留、无 @content/@component/TODO/TBD 标记残留、无重复 id、无指向缺失 id 的锚点、无断裂本地引用（../../libs/ 与 ../../index.html、index.html、overview.html、../../wiki/glu/index.html、../../wiki/situ-glu/index.html、../../wiki/glu/research/evidence.md 均存在）。

- 公式渲染与交互：KaTeX 本地资源（../../libs/katex.min.css / katex.min.js / auto-render.min.js）在两页 head 中正确引用；行内 $...$ 与行间 $$...$$ 定界符与 auto-render 配置一致；Prism 资源在 index.html 中引用。结构上目录、章节折叠按钮、j/k 快捷键、复制按钮、返回顶部均由外壳脚本提供。待浏览器实际打开复核渲染（见 review 阶段）。

- 手算数字复核（Python 独立运行，2026-08-09）：
  - σ(0)=0.5、σ(1)=0.731059≈0.7311、σ(-1)=0.268941≈0.2689、σ(-0.5)=0.377541≈0.3775 ✓
  - Swish(0)=0×0.5=0 ✓；Swish(1)=1×0.731059≈0.7311 ✓；Swish(-1)=-1×0.268941≈-0.2689 ✓；Swish(-0.5)=-0.5×0.377541≈-0.1888 ✓
  - Swish 最小值在 z≈-1.278，值≈-0.278 ✓（与正文"约 z≈-1.278 处达到最小 ≈-0.278"一致）
  - 贯穿例子：x=[1.0,0.5]、W=diag(1,-1)、V=I、b=c=0、β=1
    - xW=[1.0, -1.0] ✓
    - Swish(xW)=[0.7311, -0.2689] ✓
    - xV=[1.0, 0.5] ✓
    - SwiGLU=[0.7311, -0.1345] ✓（Python 验证 -0.268941×0.5=-0.1344705，4dp=-0.1345）
  - GLU 对照：σ(xW)=[0.7311, 0.2689] ✓；GLU=[0.7311, 0.1345] ✓
  - 参数量 Shazeer 原始 d=768：2×768×3072=4,718,592；3×768×2048=4,718,592 ✓（相等）
  - 参数量 LLaMA 风格 d=4096：基线 2d×4d=8d²=134,217,728；SwiGLU 3d×(8/3)d=8d²=134,217,728 ✓（相等）；LLaMA-7B 实际取 d_ff=11008（向上取整到 256 倍数，与 8/3×4096≈10922.67 略有差异，属工程取整）

- 写作偏差：无。所有内容来自 scope.md 已纳入范围与 evidence.md 已确认论断；教学数字均标记"教学示例"；教学解释与类比均标明职责与失效边界；教学简化（Swish 完整讲解不展开、GELU/ReLU 作为名不展开、Transformer 完整架构不展开、各 LLM 部署代码不展开、基础记号内联、LLaMA-7B 取整规则）均在文末"教学简化及其限制"逐项写明。C/F/N 引用与 evidence.md 一致。写作中修正了 3 处 Markdown 风格链接 `[...](...)` 为 HTML `<a>` 标签（index.html 2 处、overview.html 1 处），修正了 4 处手算数字四舍五入（$-0.1344\to-0.1345$、$+0.1344\to+0.1345$，Python 验证 $-0.268941\times0.5=-0.1344705$ 4dp=$-0.1345$），修正了参数量验算表述（用 $8d^2$ 精确等式替代 $3\times4096\times10923\approx134{,}217{,}728$ 的近似）。

- 来源事实与外部来源核对：
  - Shazeer §1 Swish 定义与 β=1、§2 Eq.(5) SwiGLU 定义、§2 Eq.(6) FFN_SwiGLU 定义、§2 末段 2/3 缩放原文、§3.1 实验设置（T5 base、d_model=768、12 层编码/解码、d_ff=3072→2048）、Table 1 八行 log-perplexity（GEGLU 1.633 最优、SwiGLU 1.636）、§4"divine benevolence"——经 ar5iv (https://ar5iv.labs.arxiv.org/html/2002.05202) 逐条核对一致。
  - LLaMA 采用 SwiGLU：通过 WebSearch 交叉核对（LLaMA 论文 §2 引用 PaLM，aiwiki.ai/swiglu 综述列出 PaLM/LLaMA 1-3/Mistral/Qwen/DeepSeek 等）。
  - Shazeer 记法 vs Dauphin 记法：与 [GLU 概念页] evidence.md C9 一致。
  - SiTU-GLU 作为 SwiGLU 下游改进：与 [SiTU-GLU 概念页] 一致。
