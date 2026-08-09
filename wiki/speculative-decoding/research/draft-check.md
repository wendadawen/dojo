# 投机解码初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成（research/ 目录下），规划完成条件全部满足。
- 大纲落实：
  - 页面开头：钩子问题 + 一句话定义 + blockquote.meta 来源摘要 + learning-goals（5 个学习目标） ✓
  - S1 为什么串行解码慢：带宽受限结论、验证墙钟时间相近的硬件原因、负载条件边界 ✓
  - S2 Draft-then-Verify：五步骤、接受概率 $\min(1,p/q)$ 与边界、残差分布与触发条件、bonus 来源、伪代码折叠 ✓
  - S3 为什么能保分布：F1-F4 公式、路径 A/B 互斥分解、残差归一化常数等于拒绝概率、贪心退化 ✓
  - S4 能快多少：$\mathbb{E}[L]$ 与 $S$ 公式、Leviathan 数字复算折叠、$\alpha=0.2$ 反例折叠、三种退化条件、$\gamma$ 最优值 ✓
  - S5 把机制跑一遍：3-token 贯穿例子、逐位 $a_i$ 计算、残差归一化、单位置证明验证、完整手算折叠 ✓
  - S6 工程实例与边界：K3+EAGLE-3 占位、两篇论文实测表、边界总结 ✓
  - 文末来源与教学说明：核心论断/公式/数字/教学示例/教学解释/教学简化 六小节齐全 ✓
- 学习目标闭环：
  - Q1（带宽受限 + 验证几乎免费）：S1 正文章节完整回答 ✓
  - Q2（一轮五步 + 接受/拒绝条件）：S2 正文章节完整回答，伪代码折叠只形式化已述内容 ✓
  - Q3（保分布证明）：S3 正文章节给出 F3 结论与恒等式提示，推导折叠补全代数 ✓
  - Q4（加速比公式 + 退化条件）：S4 正文章节给出公式、实测数字、退化条件，手算折叠补全数字 ✓
  - Q5（贪心退化）：S3 正文末尾完整回答 ✓
  - 折叠块全部收起时正文仍能回答全部学习目标 ✓（已逐题核对）
- 代码运行：无可运行代码（本页无 Python 代码组件）。伪代码折叠块标记为 language-text，非 Python，符合组件 11 规则。
- 机械检查：
  - 命令：`python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py wiki/speculative-decoding/index.html`
  - 结果：`validation ok: /Users/wendadawen/code/dojo/wiki/speculative-decoding/index.html`，退出码 0
  - overview.html 同样通过：`validation ok`，退出码 0
  - 占位符检查：无【…】残留、无 @content/@component/@copy-start/@copy-end/TODO/TBD 残留
  - 数学分隔符检查：display $$ 20 对、inline $ 320 对（去除 script/pre/code 后，KaTeX auto-render 忽略这些标签）；5 个 $ 在 JavaScript 中（regex 与模板字符串），KaTeX 不解析
- 公式渲染与交互：
  - KaTeX 本地资源路径正确（../../libs/katex.min.css / katex.min.js / auto-render.min.js）
  - Prism 本地资源路径正确（../../libs/prism-primer-light.css 等）
  - 目录锚点：所有 h2 均有显式 id（why-sequential-slow / draft-then-verify / why-distribution-preserved / how-much-faster / worked-example / engineering-and-boundaries / sources-and-teaching-notes），validate.py 未报告缺失锚点
  - 折叠块：6 个 details 元素（伪代码、F3 推导、F4 推导隐含在 F3 中、$\alpha=0.8$ 手算、$\alpha=0.2$ 反例、贯穿例子完整手算），summary 均写明内容
  - 完成检查项：已移除所有括注答案，符合组件 08 规则
- 写作偏差：无。全部章节、学习目标、前置知识、贯穿例子、误解和边界均按 outline.md 落实。两处完成检查项原含括注答案，已在生产阶段移除（不改变大纲，属局部修正）。

## 数字一致性验证（Python 复算）

用 Python 独立复算页面所有手算数字，全部一致：

- 贯穿例子：$a_1=0.8, a_2=1.0, a_3=0.4$ ✓
- 残差 $p'_3 = (2/3, 1/3, 0)$，$Z=0.3$ ✓
- 单位置证明 pos 1：$A: 0.4+0=0.4$, $B: 0.3+0.1=0.4$, $C: 0.2+0=0.2$，均等于 $p_1(x)$ ✓
- 单位置证明 pos 3：$A: 0.2+0.2=0.4$, $B: 0.3+0.1=0.4$, $C: 0.2+0=0.2$，均等于 $p_3(x)$ ✓
- 接受率：$\alpha_1=0.9, \alpha_2=0.9, \alpha_3=0.7$，平均 $\bar\alpha \approx 0.833$ ✓
- 贯穿例子 $\mathbb{E}[L] \approx 3.11$（精确 3.1065）✓
- Leviathan 复算：$\alpha=0.8, \gamma=5, c=0.05 \Rightarrow \mathbb{E}[L]=3.689, S=2.95\times$，与论文 2-3× 一致 ✓
- 反例：$\alpha=0.2, \gamma=5, c=0.05 \Rightarrow \mathbb{E}[L]=1.250, S=1.00\times$（盈亏平衡）✓
- S4 完成检查题 $\alpha=0.7, \gamma=4, c=0.1$：独立复算 $S \approx 1.98\times$（页面未印答案，此处仅记录验证）
