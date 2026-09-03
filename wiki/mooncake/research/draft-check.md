# Mooncake 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 全部就绪（与本页同一时间完成）
- 大纲落实：
  - 章节：1. 为什么把 prefill 与 decode 拆开；2. Mooncake 架构总览：组件、KVCache 池与四步工作流；3. KVCache-centric 调度算法；4. 长上下文 prefill 的多节点与传输重叠；5. 过载场景下的早拒绝与预测；末尾「来源与范围说明」（5 章 + 来源说明 + 开头 meta/learning-goals/misconceptions）。
  - 学习目标：5 个核心问题（Q1–Q5），覆盖动机、架构、调度算法、prefill 实现、过载处理。
  - 前置知识：链接到 [KV cache](../../wiki/kv-cache/index.html)、[PagedAttention](../../wiki/paged-attention/index.html)、[Prefix caching](../../wiki/prefix-caching/index.html)、[Chunked Prefill](../../wiki/chunked-prefill/index.html)、[Model parallelism](../../wiki/model-parallelism/index.html)、[PCP 与 DCP](../../wiki/pcp-dcp/index.html)，全部 concept 页面已存在，无递归生成。
  - 贯穿示例：3 个 prefill 实例（A/B/C），12288 tokens 请求，构造时间模型。用于第 3 章 Algorithm 1 演示与策略对比。
  - 误解和边界：3 条常见误解（Mooncake = Transfer Engine、PD 分离 = 全部创新、朴素早拒绝 = 限流）；适用边界包含长上下文场景、非外推声明。
  - 过渡：每章开头有 chapter-summary 概述；学习目标答复中末尾指明完整论证所在章节。
- 目标覆盖检查：
  - Q1（动机）：第 1 章两阶段特性与 SLO、chunked prefill 不足以替代 PD 分离 → 完成
  - Q2（架构）：第 2 章五大组件、KVCache 池存储形态、四步工作流 → 完成
  - Q3（调度算法）：第 3 章 Algorithm 1 两条分支、热点迁移、跨案例策略对比 → 完成
  - Q4（prefill 实现）：第 4 章 TP/SP/CPP 对比、layer-wise prefill → 完成
  - Q5（过载）：第 5 章 goodput/早拒绝/震荡/系统级预测/端到端结果 → 完成
- 代码运行：本页无可独立运行的代码（Algorithm 1 为论文伪代码，不作为可执行代码块）。构造示例中的时间模型仅用于手算演示，标注「构造示例」。
- 机械检查：`python3 .dojo/scripts/validate.py wiki/mooncake/index.html` → validation ok。`python3 .dojo/scripts/validate.py wiki/mooncake/overview.html` → validation ok。
- 公式渲染与交互：Headless Chrome（`--dump-dom --virtual-time-budget=8000`）输出包含 92 个 `.katex` 与 92 个 `.katex-html` 节点，KaTeX 渲染成功；92 个公式分布在正文、折叠块、表格与来源说明。18 个 `<details>` 块（核心问题 5 + 章节问题 5×2 = 10 + Algorithm 1 + 构造时间模型 + 第 2 章辅助解释 = 14；具体数量由 JS 自动生成）。
- 写作偏差：无。SVG 标签重叠问题在自检中发现并修复（Stage 标签从「Stage 1（节点 1）」缩短为「Stage 1」以避开第一列 rect 的视觉重叠）。