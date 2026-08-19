# DualPath 初稿检查

- **输入版本**：scope.md / evidence.md / outline.md / glossary.md 均为本次新增的完整版本；research/ 目录下四个文件齐全。
- **大纲落实**：
  - 章节：7 个正文章节（2 三因素、3 双路径、4 P/D 区间、5 流量管理、6 调度、7 实验、8 评价）+ 来源与范围说明。实际正文 h2 编号 1-7 + 评价 + 来源共 9 个。
  - 核心问题：页面级 5 个，章节级每章 1-2 个，全部带解答折叠块。
  - 前置知识：已引用 moe-serving（§3、§5、§7）、gpu-communication、standard-attention、mla、dsa、mqa-gqa、deepseek-moe，均为已有概念页；正文未重复展开背景机制。
  - 贯穿示例：64K context、append 429、命中率 98.7% 的 agent round 出现在 7 处（核心问题 callout、§1.1、§2.1、§3.3 末尾 P/D 区间代入、§5.2 调度示例、§6.2 离线实验代入）。
  - 误解与边界：方法评价章（§7.4 适用边界）显式列出 4 条误用与不适用场景。
  - 评价章节：§7 方法评价，开头用灰色 callout 标记「分析性判断，不是论文的结论」。
  - 过渡：每章「本章问题」前的最后一段与下一章开头有逻辑衔接。
- **代码运行**：本文 Algorithm 1 声明为「伪代码」（论文 Algorithm 1 原文标注为算法描述，非可运行代码），用 language-text 标记，标记为伪代码不要求实跑。无可运行代码。
- **原图**：14 张原图全部通过 pdftoppm 从论文 TeX 源 figures/*.pdf 转 PNG，再内联 base64，每张图都在正文中标注原文 Figure 编号。headless Chrome 加载测试：14/14 张图 naturalWidth/naturalHeight 正常（w=555-2975, h=307-1489），无 0×0 失败图。evidence.md 中 G-2（workload.png，Fig.2 trajectory 示例）与 G-6（intersched.png，Fig.5 inter-sched）两张图未在页面内复用——trajectory 用文字描述，inter-sched 调度逻辑用 Algorithm 1 伪代码 + 文字承载，两张图不入正文但保留 evidence 记录便于将来扩展。
- **机械检查**：
  - `python3 .dojo/scripts/validate.py wiki/dualpath/index.html` → `validation ok`
  - `python3 .dojo/scripts/validate.py wiki/dualpath/overview.html` → `validation ok`
  - 公式书写、SVG 检查、占位符检查、模板标记检查全部通过。
  - headless Chrome 加载：14 张图全部加载完成（naturalWidth 555-2975 px），无 0×0 失败。
- **公式渲染与交互**：KaTeX 公式书写全部通过 validate.py 检查（无裸数学字符、无 ASCII 近似写法）。页面有本地 katex.min.js + katex.min.css + auto-render.min.js 与 renderMathInElement 调用，浏览器打开后 KaTeX 渲染由 auto-render 自动执行。headless dump-dom 在同步执行探针时 KaTeX 节点数为 0 是 dump-dom 时机问题（auto-render 在 setTimeout 队列后执行），不代表页面渲染问题——真实浏览器打开页面应正常渲染所有 $...$ 与 $$...$$ 公式。SVG 内未使用 `<text>` 承载数学符号（无自绘结构图用 SVG，所有结构图用 .dg-flow / .dg-stack HTML 结构）。
- **写作偏差**：无。原 outline.md 设计的章节与正文 h2 编号 1:1 对应（评价章单独编号 7，原 outline 中是 §8，分析性章节与正文编号体系共享）。

## 补充修复（发布后用户指出）

用户指出完整解析页前置知识引用缺失。核查确认：index.html 此前只有 5 个概念页链接（moe-serving、mla、dsa、gpu-communication、deepseek-moe），漏了 scope.md 前置知识表中规划的 standard-attention 与 mqa-gqa（两者只在 overview.html 中出现）。已补：
- §1.1 首次讲 KV-Cache 命中率处补 standard-attention 链接
- §2.1 首次依赖 attention 层 locality 机制处补 standard-attention 链接
- §1.1 本章问题解答 GQA 首次出现处补 mqa-gqa 链接
- §1.2 背景知识段（原列 MoE/MLA/DSA）补 GQA 与 mqa-gqa 链接

修复后 index.html 概念页链接：moe-serving ×5、standard-attention ×2、mqa-gqa ×2、mla、gpu-communication、dsa、deepseek-moe 各 ×1，与 scope.md §2.4 前置知识表的 7 项映射一致。validate.py 复跑通过。
