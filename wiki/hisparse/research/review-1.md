# HiSparse 审查记录（第 1 轮）

- 页面版本：`index.html` 88ae001c228f8007b29319ed17ab0b12e42d468f，`overview.html` 7dee9775f87e59f726dba7624d0561bb52cf53df
- 论文版本：arXiv:2608.07009v1（NeurIPS 2026 投稿预印本）
- 审查时间：2026-09-03
- 审查者：编排者派发的独立审查者（独立上下文，未参与写作）
- 已完整阅读章节（按页面顺序）：术语表与核心问题块；第 1 章 容量墙（1.1–1.4 + 本章问题）；第 2 章 解耦可用与驻留（2.1–2.4 + 本章问题）；第 3 章 LRU（3.1–3.4 + 本章问题）；第 4 章 Resolve kernel（4.1–4.3 + 本章问题）；第 5 章 精确预取（5.1–5.5 + 本章问题）；第 6 章 收益与代价（6.1–6.5 + 本章问题）；第 7 章 方法评价（7.1–7.3）；来源与范围说明；以及 overview.html 全文
- 已对照的论文原文：`0_abstract.tex`、`1_intro.tex`、`2_background.tex`、`3_design.tex`、`4_eval.tex`、`5_discussion.tex`、`6_related.tex`、`7_conclusion.tex`、`8_appendix.tex`、`main.tex`、`main.bbl`
- 机械验证：
  - 页面自带的可运行代码（3.4 节 Python LRU trace）实际执行，输出 `Swap-vanilla (B=k=8) miss 1048/1600 = 65.5%；LRU (B=2k=16) miss 546/1600 = 34.1%；LRU (B=4k=32) miss 63/1600 = 3.9%`，与页面预期输出一致
  - 本地资源（`libs/katex.min.css` 等、`assets/*.png` 6 张原图）齐全
  - 概念链接目标 `wiki/kv-cache/index.html`、`wiki/dsa/index.html`、`wiki/vllm-cudagraph/index.html`、`wiki/strata/index.html` 均存在
  - `.dojo/scripts/validate.py` 对两页均返回 `validation ok`（含数学字符与结构图检查）
  - 6 张原图（motivation / hisparse_overview2 / topk_miss_rate_trace / swap_kernel / prefetch_sweep / peak_throughput_comparison_paper）与 Figure 1/2/6/3/8/5 编号、图中数据、图注文字一一对应：Figure 5 柱状图数字（2430/2668、511/1824、111/520、2288/2280、624/1919、232/680）、Figure 6 七条缓存配置曲线与均值（30%/13.4%/17.2%/16.1%/8.2%/6.7%）、Figure 8 四配置（Baseline / HiSparse w/o prefetch / w/ prefetch / no-IO oracle）并发 8–256、Figure 3 Resolve 五阶段、Figure 2 五步数据流编号 1–5 均与正文和论文 caption 吻合
  - `dojo:topics="推理系统,内存与缓存,注意力机制"` 全部命中 AGENTS.md 固定大类词表

## 问题

- [重要·技术] `index.html` §2.1（行 866）"HiSparse 的结构由四条不变量刻画（§3.1）"：页面仅列出四条（完整 KV 可用性、有界设备占用、精确输出、indexer 无关），但论文 `3_design.tex` §3.1 实际有五条 paragraphhead——完整 KV 可用性、有界设备占用、精确稀疏注意力输出、indexer 无关接口、**miss 延迟离开关键路径**；第五条被遗漏。引文依据：`3_design.tex` §3.1 五个 `\paragraphhead{...}`，最后一条为 "Miss latency off the critical path"（"Bounding residency must not trade throughput for per-token latency..."）。修复要求：补齐第五条 "Miss latency off the critical path"（用 §3.1 原文表述）或将"四条"改为包含全部五条并重新表述"刻画"措辞，避免读者对照 §3.1 时发现遗漏。修复：标题 `2.1 四条设计目标与两条公式` → `2.1 五条设计目标与两条公式`；正文 `HiSparse 的结构由四条不变量刻画` → `五条`；在 indexer 无关后追加第 5 条 `miss 延迟不进入关键路径`，说明论文把 miss 解析的代价视为一等公民，三件套（cache 局部性、融合 kernel、prefetch）即为此目标设计，并标注 §3.2 / §3.4 / §3.5 / §4 的对应位置。复验：`grep -n "五条设计目标\|五条不变量\|miss 延迟不进入关键路径" wiki/hisparse/index.html` 三个关键词均命中；validate.py ok；Chrome headless `.katex` 数从 271 → 278，无错误。
- [轻微·技术] `index.html` §4.1（行 1122）"（这个类比出自论文 §3.4：地址翻译换成了 KV 记录定位，页表错失换成了 host 拉取）"：冒号前的论文定位正确，冒号后的"地址翻译↔KV 记录定位、页表错失↔host 拉取"对应关系为页面自身的展开解释，不在论文原句中。引文依据：`3_design.tex` §3.4 原句仅 "logical indices in, physical slots out, much like a software-managed TLB"，未给出地址翻译/页表错失到 KV/host fetch 的逐项对应。修复要求：把冒号后内容标注为"页面的对应解读"或移除逐项对应、只保留论文原句（"逻辑索引进、物理槽位出，类比软件管理的 TLB"），或拆为独立句不嵌在"出自 §3.4"的引文边界内；保持与来源说明区（行 1459-1460）"不延伸到 TLB 的其他语义"的口径一致。修复：｜复验：
- [轻微·技术] `index.html` §4.1（行 1120）"GLM-5.2 有 78 个稀疏层，每生成一个 token 就要执行 78 次"：GLM-5.2 是 IndexShare 目标模型，78 层中 21 个 anchor 跑 Resolve、57 个 shared 层"跳过解析的全部阶段"（§5.2），实际 Resolve 调用次数为 21 次/层·token，不是 78 次；与第 5 章的描述形成内部不一致。引文依据：`4_eval.tex` §4.6 "of its 78 layers, 21 anchor layers run the indexer and the remaining 57 layers reuse the selection of their preceding anchor"；`3_design.tex` §3.5 "a shared layer reuses the anchor's slot table outright ... skips resolution entirely"。修复要求：把"GLM-5.2 有 78 个稀疏层，每生成一个 token 就要执行 78 次"改为"在无共享选择模型上每生成一个 token 每个 sparse layer 各跑一次 Resolve"（或换用 DeepSeek-V4-Flash/NSA 等无共享选择模型做数字例子）；或加上"同步解析路径下"的限定，与第 5 章预取路径对齐。修复：改为"在没有跨层共享选择的模型上，以 78 个稀疏层为例，每生成一个 token 就要跑 78 次。GLM-5.2 用 IndexShare 把 21 个 anchor 与 57 个 shared 层配成组，shared 层跳过解析的全部分阶段、直接复用 anchor 的 slot 表，所以实际每 token 只跑 21 次 Resolve（§3.5、§4.6）。"——既保留"78 次"用于阐释无共享选择模型的代价，又给出 GLM-5.2 的真实数字 21，消除内部不一致。复验：grep 命中"21 个 anchor 与 57 个 shared 层配成组"与"§3.5、§4.6"，与第 5 章的 §3.5 引用、§4.6 数字一致。
- [轻微·格式] `index.html` §6.2 表格（行 1349-1356）倍数写法不统一：Qwen 4K 行用 LaTeX "$\approx 1\times$"，其余 6 行（1.10×、3.6×、4.7×、3.1×、2.9×、2.1×）用 Unicode "×"；同表格内数学写法不一致。引文依据：`write.md` §4.2 "运算符和关系符一律写为 `$...$` ... 不使用 Unicode 数学字符替代 ... 同一变量全页保持同一种写法"。修复要求：把表格内全部倍数写法统一为 LaTeX（`\times`），或全部保留 Unicode "×"，不可在同一表格中混用。修复：｜复验：
- [轻微·格式] `index.html` §7（方法评价，行 1419-1437）末尾无 h3「本章问题」块：`write.md` §4.10 要求"每个正文章节末尾 h3 固定为「本章问题」"。第 7 章是分析性评价章节，没有事实性问题需要回答，但其形式上仍属正文章节。引文依据：`write.md` §4.10 "页面开头问题块的 h2 固定为「核心问题」... 每个正文章节末尾 h3 固定为「本章问题」... 两级都不使用其他措辞"。修复要求：第 7 章末尾补 h3「本章问题」并配套解答折叠块；如认为该章为评价章节不适用，需通过规划文件确认豁免（按 `check.md` §4 "需要改变范围或大纲的问题返回规划文件处理"）。修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复——重要问题"四条 vs 五条"必须在本轮修复；轻微问题建议一并处理；轻微问题 §4.1 GLM-5.2 "78 次"虽归轻微，但与 §5 共享选择机制直接相关，建议与重要问题同步修复以消除内部不一致
- 待第二轮独立审查者复核修复结果