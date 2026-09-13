<!-- review-meta
round: 6
page: wiki/mooncake/index.html
reviewed_content_sha256: 956a4578deb26a17
-->
# Mooncake 审查记录（第 6 轮）

- 页面版本：9f244d9505dc4ea14dad6db3fc6ffcd1c77e85e3
- 审查时间：2026-09-13 22:50
- 审查者：独立子代理（未参与写作与前序轮次）
- 适用规范：`guides/concept/check.md`（页面 `<meta name="dojo:type" content="concept">`）
- 外部来源获取：arXiv:2407.00079v4（arXiv HTML 全文 + PDF 第 11 页 Figure 8 高分辨率渲染像素测量）
- 已完整阅读章节：核心问题、常见误解、1. 为什么把 prefill 与 decode 拆开、2. Mooncake 架构总览：组件、KVCache 池与四步工作流、3. KVCache-centric 调度算法、4. 长上下文 prefill 的多节点与传输重叠、5. 过载场景下的早拒绝与预测、来源与范围说明（含全部折叠块、图注与内联 SVG）

## 已核对且无误的要点（记录核对依据，不列为问题）

- 图 8 四个平均 TTFT：从论文 PDF 第 11 页以 400 dpi 渲染后逐像素读数，绿色均值三角标注依次为 KVCache-centric 6.26、cache-aware 14.36、load-balancing 60.41、random 92.07；y 轴刻度 0/50/…/250 定位后算出虚线 SLO 线 ≈30.3 s，与页面「≈30s」一致。论文正文 §6.2 只作定性描述，四个数字仅在 Figure 8 内，页面标注「§6.2；Figure 8」定位正确。
- §3.4 构造示例全部可复算：12288/512=24 块；A 分支 1 得 0.5+4096/4000=1.524、TTFT 3.0+1.524=4.524；B 得 1.012、9.012；C 分支 2 得 10240/10000=1.024、T_prefill 1.012、TTFT 1.0+1.024+1.012=3.036；纯 LB 1.0+0.5+12288/4000=4.572。阈值判断 best/prefix 1.25、1.0、∞ 与分支选择一致，热点迁移触发条件与 Algorithm 1 第 30 行一致。
- 伪代码逐条对照论文 Algorithm 1（PrefixHash、FindBestPrefixMatch、两条分支的条件与 T_prefill 自变量、SelectDecodingInstance、TTFT/TBT_SLO 拒绝、`best_prefix_len / p.prefix_len > kvcache_balancing_threshold` 触发 TransferKVCache）一致。
- N1 23,608 条/1 小时（§4）、N2 输入 7,590/输出 182（§4.2）、N3 块 512（§4.1）、N4 Table 1（1000→0.30、50000→0.50）、N5 LRU 最优（§4.2）、N6 >50% 零访问（§4.2/Figure 6）、N7 prefill_chunk>1000（§3）、N9 Table 3 4183/3771/3589（§8.2）、N10 TTFT_P90=10×、TBT_P90=5×（§2）、N11 +20%/+40%（§8.1.1）、N12 50%–525%（§8.1.2）、N13 +75%、100% vs 57%（§8.1.3）、N14 30s / 0.1s per token（§8.1.3）、N15 8×A800-SXM4-80GB + 800 Gbps（§8.1 Testbed）、N16 50%/90%（§9）——逐条在论文中定位到原文片段，一致。
- F2 `S·T`（§5.2 原文 "its occupation cost is S*T"）、§5.2 执行时间 ≈ max(载入, prefill) 与「调度只需考虑 DRAM」、「Similar to pipeline parallelism in training」「over 50% of cache blocks remaining unused while certain blocks are accessed tens of thousands of times」、「HTTP 429 Too Many Requests」、Messenger「operates as an independent process within its respective inference instance, receiving signals」等表述均与来源吻合。
- §2.3 折叠块 KVCache 估算可复算：2·80·8·128·2 B = 327,680 B = 320 KiB/token；12,288 tokens ≈ 3.75 GiB；131,072 tokens ≈ 40 GiB。LLaMA2-70B 的 L/H_kv/d_head 与 F1 引用一致。
- 链接有效（`wiki/paged-attention`、`wiki/kv-cache`、`wiki/pcp-dcp`、`wiki/chunked-prefill`、`overview.html` 均存在），无「（待生成）」，无 `alt` 含 `$...$`，`dojo:type/topics/tag` 取值在词表内，`validate.py` 返回 `validation ok`。

## 问题

- [重要·技术] 第 4 章 §4.2 第 1 段：句子自相矛盾且与来源不符。前半句「但把切分维度反过来——不是按层切节点」否定按层划分节点，同句括号内却写「（每节点持有一部分层）」，本章内联 SVG 也把两个节点标为「Stage 1 / chunk 1，前半层」与「Stage 2 / chunk 1，后半层」。论文对 CPP 的定位是沿用训练流水线并行、按层划分 pipeline stage，读者据此无法判断 CPP 是否按层划分节点。｜引文依据：论文 §5.1 "To address this, Mooncake ... implements chunked pipeline parallelism (CPP) for long context prefill. We group every X nodes in the prefill cluster into a pipelined prefill node group. ... Similar to pipeline parallelism in training, it requires cross-node communication only at the boundaries of each pipeline stage, which can be easily overlapped with computation."（CPP 与训练 PP 同构，按层划分 stage，故节点按层划分）｜修复要求：改写该句，删去与同句括号及 §4.2 SVG 冲突的「不是按层切节点」，明确写成「CPP 沿用训练 PP 的按层 stage 划分，把每请求输入切成 chunk 作为流水线 microbatch 在相邻 stage 间推进」；「把切分维度反过来」若不能给出论文出处一并删除。｜修复：｜复验：

- [轻微·格式] 第 4 章 §4.2 的内联 SVG：图内 `<rect>` / `<line>` 使用页面自定义的 `.box` / `.accent` 类并在 `<svg>` 内用 `<style>` 定义，未使用 `guides/concept/style-guide.md` A5 要求、且 `libs/dojo-concept.css` 第 582–584 行已定义的 `dg-box` / `dg-line` / `dg-accent` 类；全仓库仅本页采用这种自定义写法（其余含 SVG 的页面均用 `dg-box`）。｜引文依据：不适用｜修复要求：把 SVG 内方框 `rect` 的 class 改为 `dg-box`、箭头 `line` 改为 `dg-accent`，删除 `<svg>` 内自定义 `.box` / `.accent` 规则；`text` 的主题变量填充引用可保留。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 1
- 处置：修复
