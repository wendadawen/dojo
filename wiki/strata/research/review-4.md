<!-- review-meta
round: 4
page: wiki/strata/index.html
reviewed_content_sha256: d46175d85361eadc
-->
# Strata 审查记录（第 4 轮）

- 页面版本：index.html 1f85f327845eae417c564b4bac0a3051c3b4cb1d；overview.html 6964b781b6fb113f7f21aca096d819b2c49ad555
- 论文版本：arXiv:2508.18572v1（Strata: Hierarchical Context Caching for Long Context Language Model Serving, Xie et al.；OSDI 2026 会议版）
- 审查时间：2026-09-13 19:54
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节：核心问题（5 题）、术语表、1. 瓶颈定位（1.1 容量账 / 1.2 分页碎片化传输 / 1.3 大页不是出路 / 1.4 调度器假设失效与 delay hit / 本章问题）、2. GPU-assisted I/O（2.1 机制 / 2.2 干扰控制 / 本章问题）、3. 布局解耦与存储层（3.1 / 3.2 / 3.3 存储层预取 / 本章问题）、4. Cache-aware 调度（4.1 系统全景 / 4.2 延迟命中 / 4.3 均衡 batch / 4.4 Bubble filling / 本章问题）、5. 实验验证（5.1–5.6 / 本章问题）、6. 方法评价（6.1–6.3 / 本章问题）、来源与范围说明（含全部 details 折叠块、Algorithm 1 代码块、SVG 图注）、overview.html
- 核对来源：WebFetch 抓取 https://arxiv.org/abs/2508.18572（abstract）与 https://arxiv.org/html/2508.18572v1（正文全文），按 C/F/N 编号逐条定位核对；页面内链指向的 8 个前置概念页（kv-cache / paged-attention / prefix-caching / standard-attention / gpu-execution-model / gpu-communication / kv-cache-layout / dualpath）逐一确认存在；dualpath 页确认承载 L189 的 agentic 命中率论断
- 机械校验：`python3 .dojo/scripts/validate.py wiki/strata/index.html` 返回 `validation ok`；页面无 research/ 路径，无「（待生成）」占位

## 问题

- [重要·技术] index.html L591（§6.2 局限 其一）与 L614（第 6 章本章问题解答）：断言「I/O kernel …部署在 decode 主导的短上下文负载上仍净亏」，与论文「短上下文无退化」的实测结论冲突，也与本页 L551（§5.6）「长上下文优化不以短上下文为代价[C24]」及 L102（核心问题 4 解答）「短上下文无退化（ShareGPT）」自相矛盾。｜引文依据：arXiv:2508.18572v1 abstract“……achieves up to 5x lower Time-To-First-Token (TTFT) compared to vLLM + LMCache and 3.75x speedup over NVIDIA TensorRT-LLM on long-context benchmarks, **without degrading short-context performance**”；§4.2 微基准“less than 5% performance degradation on prefill and 10% on decoding”（该 10% 是 I/O kernel 与计算同跑时的微基准观测值，页面已标 N5；不是短上下文稳态的整机数字）｜修复要求：删除 L591、L614 两处「decode 主导的短上下文负载上仍净亏」的无条件结论；若要保留，改写为明确标注的推断并限定条件——该 decode 损失只在 I/O kernel 与计算同跑时出现，短上下文加载量小、论文实测无退化，不得保留与 C24 相反的无条件表述。｜修复：｜复验：
- [轻微·格式] index.html L253+L255（图 5）、L290+L292（图 6）、L302+L304（图 12）、L341+L343（图 4）：四张图前各有两条重复引入句。例：L290「图 6 对比 layer-first 与 page-first 两种布局。」紧跟 L292「图 6 显示 layer-first（GPU 计算友好）与 page-first（传输友好）布局的差别。」；图 4/5/12 同型（L341「图 4 为架构图。」+L343「图 4 展示 Strata 的数据流——…」；L253「图 5 的微基准（…）：」+L255「图 5 显示资源配额…」；L302「图 12 的微基准：…」+L304「图 12 给出磁盘加载下 page-first 与 layer-first 布局的延迟对比。」）。｜引文依据：不适用｜修复要求：图 4/5/6/12 各保留一条引入句，删去重复的另一条。｜修复：｜复验：
- [轻微·表述] index.html L632「本页沿用」：以「本页」为主语的自我指代。同时 L149「与本文贯穿示例对齐」、L408「本文讨论的 P-D co-location 下」、L72「本文假设读者已熟悉」——「本文」在本页指页面自身，与全文指代所解析论文的「论文」并存，易混淆。｜引文依据：不适用｜修复要求：L632 改为「F3 定义沿用如下……」；指本页处改用「本解析」或省略主语，去除「本页/本文」作主语的自我指代。｜修复：｜复验：
- [轻微·表述] index.html L179（§1.3）：「但放大页同时改了三个目标的方向，而且两个反向。」下文只展开两个目标（传输效率、缓存命中率），第三个目标未出现，「两个反向」的具体所指也不明确。｜引文依据：不适用｜修复要求：改为「两个目标」，或明确补出第三个目标；「两个反向」改为可辨识的具体陈述（如「传输效率上升、命中率下降」）。｜修复：｜复验：
- [轻微·表述] index.html L81 / L150 / L155：称「打满 PCIe 5.0 需 1–2 MB 传输」，与 L165「打满 PCIe 5.0 的 75%–80% 需要 1–2 MB 传输」口径不一；来源中 1–2 MB 对应的是 75–80% 带宽，而非 100%。｜引文依据：arXiv:2508.18572v1 §3.1“……requires transfer sizes (S) in the megabyte range (i.e., 1-2MB)……achieve 75-80% of theoretical PCIe 5.0 bandwidth”｜修复要求：L81/L150/L155 统一改为「达到 PCIe 5.0 约 75–80% 带宽需 1–2 MB」。｜修复：｜复验：
- [轻微·表述] index.html L155（§1.2）「vLLM 16 token」与 L479（§5.1）「vLLM 页 32」并存，未说明前者是引擎背景默认、后者是本文实验设定，读者会读作同页矛盾。｜引文依据：arXiv:2508.18572v1 §2.2“Typical page sizes are small—e.g., 32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang”；§5.1“the vLLM page size was set to 32 as default”｜修复要求：在 §1.2 或 §5.1 补一句说明二者分别是背景默认值与实验设定（对齐先前工作）。｜修复：｜复验：
- [轻微·表述] index.html L68「把缓存从 CPU 内存搬回 GPU 的带宽利用率惨不忍睹」：口语化临场评价，非中性陈述。｜引文依据：不适用｜修复要求：改为中性陈述（如「带宽利用率远低于硬件上限」）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 6
- 处置：修复（1 项重要 + 6 项轻微待修；本轮全部来源论断已回源核对，数字与机制描述与 arXiv:2508.18572v1 一致，validate.py 通过，无阻断项）
