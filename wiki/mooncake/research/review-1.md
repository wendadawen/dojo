# Mooncake 审查记录（第 1 轮）

- 页面版本：e45742d3f5621d96f0b5c0b4748897a4b6a70b60（index.html 工作树哈希）
- 审查时间：2026-09-03
- 审查者：编排者派发的独立审查者（未参与写作与前序审查）
- 已完整阅读章节（按顺序）：index.html：导语与范围声明、核心问题（5 题及解答）、常见误解、第 1 章（1.1、1.2、本章问题）、第 2 章（2.1、2.2、2.3 含折叠块、本章问题）、第 3 章（3.1、3.2 含伪代码折叠块、3.3、3.4 含构造模型折叠块、本章问题）、第 4 章（4.1、4.2、4.3、本章问题）、第 5 章（5.1–5.4、本章问题）、来源与范围说明（C/F/N 表、构造示例、类比边界、简化条件）；overview.html：问题背景、核心机制、关键结论与边界、来源行。全部折叠块已展开阅读。

审查输入仅限：两个待审 HTML、`guides/concept/check.md`、论文文本 `/tmp/mooncake-research/mooncake.txt`（arXiv:2407.00079v4，全文 1513 行已通读）。未读取 `research/` 下任何写作过程产物。style-guide.md 与 AGENTS.md 不在本轮允许输入内，涉及处已在问题中注明局限，不下结论。

机械验证：`../../libs/` 下 KaTeX/Prism 资源全部存在；`wiki/paged-attention`、`wiki/prefix-caching`、`wiki/kv-cache`、`wiki/pcp-dcp`、`wiki/chunked-prefill` 五个链接目标页面均存在；index.html 与 overview.html 相互链接正常。

## 问题

### 重要

- [重要·技术] index.html 2.2 节第 1 条、2.3 节折叠块（"以 [KV cache](../../wiki/kv-cache/index.html) 页面的公式"）、4.1 节末段（"[Ring Attention](../../wiki/pcp-dcp/index.html)"）、4.2 节（"[Chunked Prefill](../../wiki/chunked-prefill/index.html)"）及 overview.html 核心机制第 1、2 条（"[vLLM 的 PagedAttention](...)"、"[Prefix caching](...)"）：共 7 处使用 Markdown 链接语法 `[文字](路径)` 直接写在 HTML 里，浏览器不渲染为链接，正文会显示原始方括号文本，前置概念链接全部失效｜引文依据：不适用（HTML 语法事实；对照同页其余位置均用 `<a href>`）｜修复要求：全部改为 `<a href="../../wiki/.../index.html">文字</a>`，并逐处确认渲染后可点击｜修复：｜复验：

- [重要·技术] overview.html「关键结论与边界」第 2 条「只有 cache-aware + balance-aware 组合同时满足 SLO」：与来源不符，且与 index.html 3.4 末段「只有前两者满足 SLO 线（≈50s）」互相矛盾。按论文 Figure 8，cache-aware 为 14.36s，同样低于 SLO 线（≈50s），满足 SLO 的策略有两个（KVCache-centric 与 cache-aware），load-balancing（60.41s）与 random（92.07s）不满足｜引文依据：论文 §6.2/Figure 8 数值 "92.07 / 60.41 / 14.36 / 6.26"，图中另有 SLO 参考线；index.html 3.4 原文「并且只有前两者满足 SLO 线（≈50s）」｜修复要求：改写 overview 该句为与 Figure 8 一致的表述（如「cache-aware 与 KVCache-centric 均满足 SLO 线，其中只有 KVCache-centric 同时兼顾负载均衡」），并消除与 index 的矛盾｜修复：｜复验：

- [重要·技术] index.html 3.4 末段与 overview.html「关键结论与边界」第 2 条「把平均 TTFT 中位数从 random 92.07s 降到 6.26s」：统计量名称与来源不符。论文原文为 "We assessed the performance of each scheduling algorithm using the average TTFT and the TTFT SLO attainment rate"（平均 TTFT），全文未出现 median；「平均 TTFT 中位数」本身自相矛盾，来源表 N8 亦写「median TTFT」，三处不一致｜引文依据：论文 §6.2 原文 "using the average TTFT and the TTFT SLO attainment rate"｜修复要求：index 3.4 正文、N8 表条目、overview 相关句统一改为「平均 TTFT」；同时改写 index「只有前两者满足 SLO 线」中指代不明的「前两者」为明确的主语（如「KVCache-centric 与 cache-aware」）｜修复：｜复验：

- [重要·技术] index.html 2.3 节折叠块「12,288 tokens 约 3.84 GiB」：数值示例不可复算。12,288 × 320 KiB = 3,932,160 KiB = 3,840 MiB = 3.75 GiB；3.84 是把 MiB 数值误当 GiB。同段「128k 上下文的请求就接近 40 GiB」复算正确（131,072 × 327,680 B = 40 GiB），进一步印证 3.84 为换算错误。来源说明节「构造示例」中同样写「≈ 3.84 GiB」需一并修正｜引文依据：不适用（复算：12,288 × 2 B × 80 × 8 × 128 × 2 = 4,026,531,840 B = 3.75 GiB）｜修复要求：两处「3.84 GiB」改为「3.75 GiB」（或「3,840 MiB」），并复核改后数字｜修复：｜复验：

- [重要·技术] index.html 3.4 节构造场景表「实例 A：prefix = 8,192 tokens（16 块，40% 复用）」：复用比例标注错误。8,192 / 12,288 = 66.7%，不是 40%（16 块数正确；40% 无法由表中任何数字得出）｜引文依据：不适用（复算：8192/12288 = 2/3 ≈ 66.7%）｜修复要求：改为「66.7% 复用」或删去百分比只保留块数，保证表中每个数字可由场景参数复算｜修复：｜复验：

- [重要·技术] index.html 第 3 章本章问题 1 解答「（多走 20% 的 prefix 增量）成本可控」：20% 为无来源数字，且与 threshold 语义不符。Algorithm 1 分支 1 的条件是 best_prefix_len/prefix_len < threshold（threshold 取 2 时意味着本地 prefix 至少为全局最佳的一半，即最多多算的增量为 prefix_len 本身），任何阈值取值都推不出「20%」｜引文依据：论文 Algorithm 1 第 8 行 "if best_prefix_len/prefix_len < kvcache_balancing_threshold then"；论文全文无 20% 相关表述｜修复要求：删除「20%」或改为与阈值定义一致的定性描述（如「与全局最佳的差距在阈值倍数以内」）｜修复：｜复验：

- [重要·技术] index.html 1.2 节「分离后能节省 VRAM——prefill 实例只需装下单请求的 KVCache，剩下的 VRAM 可以给 decode 实例装尽可能大的 batch」：机制归因无来源支持且与分离架构矛盾。prefill 与 decode 是不同集群的不同实例，prefill 实例的 VRAM 无法「给」decode 实例。论文 §5.1 理由 2 仅说 "It presents a unique opportunity to save VRAM (§5.2)"，§5.2 的机制是 layer-wise prefill 降低占用代价 S·T、使 prefill 调度可忽略 VRAM；decode 侧的对应事实是 decode 实例的 VRAM 不再被 prefill 挤占（论文 §1 "restricted ... by the total size of the aggregated KVCache that can be contained in the VRAM"）。第 1 章本章问题 2 解答复述了同一说法，需一并修正｜引文依据：论文 §5.1 "2) It presents a unique opportunity to save VRAM (§5.2)"；§5.2 "its occupation cost is S*T ... enables us to disregard the available VRAM size in prefill scheduling, as long as it can contain a single request"｜修复要求：改写为与来源一致的表述（如「分离后 prefill 实例的 VRAM 只需容纳单个请求（配合 layer-wise prefill），decode 实例的 VRAM 不再被 prefill 竞争，可全部用于聚合 batch」），第 1 章问题 2 解答同步修改｜修复：｜复验：

- [重要·技术] index.html 导语「把同 batch 上所有 decode 序列的 TBT 抬高数十毫秒甚至数秒」、1.1 节「一个长 prefill 占满算力的几十到几百毫秒里」、第 1 章本章问题 1 解答「长 prefill 占满算力的几十到几百毫秒期间」：三处时间量级数字均无来源支持，也未标注为推断。论文 §2 只给出超线性/次线性的定性结论与 Figure 2 的归一化延迟，未给出 prefill 时长量级；且对长上下文请求（12k+ tokens）prefill 常为秒级，「几十到几百毫秒」会形成错误量级认知｜引文依据：论文 §2 原文仅 "computation time in the prefill stage generally increases superlinearly with input length, as shown in the left part of Figure 2"，无任何毫秒量级数字｜修复要求：删除具体量级数字，改为与来源一致的定性表述（如「长 prefill 占满算力的整个期间」）；或明确标注为帮助理解的估计值并说明依据｜修复：｜复验：

- [重要·技术] index.html「来源与范围说明」表不完整：正文「简化条件及其限制」节引用了 <sup>[C34, N16]</sup> 与 <sup>[C35]</sup>，但「论断与来源（C）」表止于 C33、「外部数字（N）」表止于 N15，C34、C35、N16 均无定位条目；5.4 节的 <sup>[C29–C32]</sup> 范围引用中 C30、C31 同样在表中无行（表从 C29 直接跳到 C32）。这违反「每条来源论断都有引文依据记录」的发布条件。经核对论文，C34/C35/N16 对应内容本身存在（§9 的 50%/90% 复用上界、§1.2/§8.1 的 dummy 模型声明），属表格漏收而非论断错误｜引文依据：论文 §9 "up to only 50% of the KVCache can be reused ... can be as large as 90% ... such as our chat-to-paper service"；§1.2 "all the experimental results reported in this paper are based on replayed traces of real workloads, but using a dummy model that follows the same architecture as LLaMA2-70B"｜修复要求：在 C 表补 C34、C35（含 C30、C31）条目并在 N 表补 N16，每条给出论文定位；或改用正文实际标注的连续编号重排｜修复：｜复验：

- [重要·技术] index.html 3.3 节末句「这与系统级预测一起，让缓存复用与负载均衡在不显式预测未来访问的情况下同时被优化」：把 §7.4 早拒绝用的「系统级预测」与 §6.2 的热点迁移错误关联。论文 §6.2 明确将热点迁移定位为启发式方案、以区别于预测未来用法（"a heuristic-based automated hot-spot migration scheme ... without requiring precise predictions of future KVCache usage"）；系统级预测属于过载早拒绝（§7.4），不参与缓存复用与负载均衡的优化｜引文依据：论文 §6.2 原文 "we propose a heuristic-based automated hot-spot migration scheme"；§7.4 才引入 "Early Rejection Based on Prediction"｜修复要求：删除「与系统级预测一起」，或改为「热点迁移与 Algorithm 1 的两条分支共同实现缓存复用与负载均衡的折中」这类与 §6.2 一致的表述｜修复：｜复验：

### 轻微

- [轻微·可读性] index.html 导语（MaaS、SLO 首次出现）及 4.1 节（MFU 首次出现）、1.1 节（HBM）、2.2 节（RDMA 已在导语出现）、1.1 节（TTFT<sub>P90</sub> 的 P90）：MaaS（Model as a Service）、SLO（Service Level Objective）、MFU（Model FLOPs Utilization）、RDMA、HBM、P90（第 90 百分位）在首次使用处均未给全称或解释；论文 §1/§8.1 均有对应定义可引｜引文依据：论文 §1 "As a Model as a Service (MaaS) provider"；§1 "the Model FLOPs Utilization (MFU)"；§8.1 "we use the 90th percentile (P90) values of TTFT and TBT"｜修复要求：在各术语首次出现处补中文全称（至少 MaaS、SLO、MFU、P90 四个）｜修复：｜复验：

- [轻微·技术] index.html 2.2 节「部分块被访问上万次」与来源表 N6：论文为 "certain blocks are accessed tens of thousands of times"，即「数万次」；「上万次」（≥1 万）弱于原文｜引文依据：论文 §4.2 原文 "certain blocks are accessed tens of thousands of times"｜修复要求：改为「数万次」｜修复：｜复验：

- [轻微·格式] index.html 常见误解第 1 条「同名开源项目 kvcake-ai/Mooncake」：组织名拼写错误，应为 `kvcache-ai`（同页「主要依据」与论文 §1.2 均为 kvcache-ai）｜引文依据：论文 §1.2 "The trace is open-sourced at https://github.com/kvcache-ai/Mooncake"｜修复要求：改为 kvcache-ai/Mooncake｜修复：｜复验：

- [轻微·格式] index.html 来源表 C10 论断「传可复用 KVCache → 分分块/层预fill 并流式传 KVCache → decode 加入连续批处理」：「分分块」为错别字（应为「分块」），且 C10 未在正文任何位置标注、表中论断与正文 2.3 的「四步工作流」口径不统一（论文 §1 为三步、§3 为四步，两者并存但页内未说明关系）｜引文依据：论文 §1 三步列表 "1) transfer as much reusable KVCache as possible ... 2) complete the prefill stage in chunks/layers ... 3) load the KVCache and add the request to the continuous batching process"｜修复要求：改正错别字；为 C10 补正文标注或从表中移除；在 2.3 节一句话说明 §1 三步与 §3 四步的对应关系｜修复：｜复验：

- [轻微·技术] index.html 3.2 节「Algorithm 1 完整伪代码」折叠块：与论文 Algorithm 1 相比缺第 3 行 "p ← ∅"（循环前初始化选中的实例为空），伪代码标称「完整」但转录不完整｜引文依据：论文 Algorithm 1 第 3 行 "3: p ← ∅"｜修复要求：补上该行｜修复：｜复验：

- [轻微·格式] index.html 2.3 节四步工作流图注「① 把可复用前缀搬到 GPU；② 算剩余部分；④ 边算边往 decode 节点传；④ 收齐后解码」：第三个步骤编号误写为「④」，应为「③」，图注出现两个④｜引文依据：不适用（对照同图节点标题「③ KVCache 传输」）｜修复要求：改为「③ 边算边往 decode 节点传」｜修复：｜复验：

- [轻微·格式] index.html 4.2 节 CPP 时序 SVG 的 `<text>` 元素内出现 "chunk 1, layers 1–L/2"、"layers L/2+1–L"：图内数学表达（L/2、L/2+1–L）以纯文本书写，未按规范放在 `<foreignObject>` 中由 KaTeX 渲染｜引文依据：不适用（check.md 2.2 第 10 条对图内公式的要求）｜修复要求：将层范围标签改用 foreignObject + KaTeX，或改写为不含数学式的文字（如「前半层」「后半层」）｜修复：｜复验：

- [轻微·技术] index.html 2.2 节「Messenger……绕开了传统 TCP 栈的多次拷贝」「在节点间搬运而不占用对端 CPU」：论文只称 Messenger 为 "a separate (GPUDirect) RDMA-based component"，未描述 TCP 拷贝规避或对端 CPU 占用；属无来源的机制扩展且未标注推断｜引文依据：论文 §3 "The transfer of these KVCache blocks across CPUs and GPUs is handled by a separate (GPUDirect) RDMA-based component called Messenger"｜修复要求：删除这两处扩展描述，或明确标注为基于 RDMA 一般特性的推断｜修复：｜复验：

- [轻微·技术] index.html 若干无来源机制描述未标注推断：2.2 节「分页让分配/淘汰与传输都以等大块为单位，避免外部碎片与不规则拷贝」；第 2 章本章问题 1 解答「CPU DRAM 通常几 TB 每节点，单卡 VRAM 仅数十 GB」与「T_queue 小通常意味着缓存也少（重负载实例才会被复用热点缓存填满）」；4.3 节「VRAM 中可以同时驻留多个请求的部分层 KVCache 与正在载入的层」及第 4 章本章问题 2 解答「VRAM 中同时驻留的是『正在载入的一层 + 正在算的一层』」；4.2 节「顺带消掉部分流水线气泡」。以上各条论文均无对应表述（论文未述分页碎片收益、未给 DRAM 容量、未述 T_queue 与缓存量的因果、未述 VRAM 同时驻留内容的细节、未讨论流水线气泡）｜引文依据：论文对应章节（§3、§6.2、§5.2）均无上述表述，如 §5.2 仅 "the model waits for the asynchronous loading of that layer's KVCache to complete and triggers the next layer's asynchronous KVCache loading"｜修复要求：逐条删除，或标注为「推断/帮助理解的解释」，不得以事实语气保留｜修复：｜复验：

- [轻微·格式] index.html 变量与术语写法不一致：核心问题 3 解答用「best/prefix < 阈值」而 3.2 节用「best/prefix$_i$ &lt; threshold」；「阈值」与「threshold」中英混用；2.3 表格用「prefill_chunk」code 体而 1.2 节用普通文字。同一变量/参数全页写法应统一｜引文依据：不适用｜修复要求：统一为一种写法（建议 code 体的 `kvcache_balancing_threshold`/`prefill_chunk` + KaTeX 下标实例记号），全页替换｜修复：｜复验：

- [轻微·格式] index.html「来源与范围说明」表收录了正文未标注的条目（C1、C2、C10、N1、N2、N14、N15），且 C 编号存在跳号（C3→C6、C11→C15、C20→C22、C29→C32），与表头「下表列出正文中所有标注 [Cx] 的论断」的声明不符，影响维护性｜引文依据：不适用（正文 sup 标注与表的对照）｜修复要求：为未标注条目补正文标注，或从表中移除；跳号处改为连续编号或注明删除原因｜修复：｜复验：

- [轻微·格式] index.html 2.2 节折叠块 F1 公式「$B_{\text{kv}}=2\cdot L\cdot H_{\text{kv}}\cdot d_{\text{head}}\cdot b$ …… $b=2$（fp16）」：公式首因子 2（K 与 V 两份）未解释；用 b 表示「每元素字节数」与常见记法（batch size）冲突，且未说明运算范围（单 token、fp16）｜引文依据：不适用（符号说明完整性检查；数值本身 320 KiB/token 复算正确）｜修复要求：补全符号说明（2 = K/V 两份，b = 每元素字节数，fp16 时为 2），或改用不易混淆的符号｜修复：｜复验：

- [轻微·格式] index.html 5.4 节「与『分离 + cache-aware 调度 + CPP + layer-wise prefill』组合是为此类场景设计一致」：语句不通（应为「与……的设计意图一致」）｜引文依据：不适用｜修复要求：改写为「与『分离 + cache-aware 调度 + CPP + layer-wise prefill』面向此类场景的设计意图一致」｜修复：｜复验：

- [轻微·格式] overview.html：其一，`<head>` 无 `description` 与 `dojo:*` meta（index.html 均有），按 check.md 第 5 节发布条件的字面要求「页面 head 包含……」，若该要求覆盖 overview 页则缺失；本轮无 AGENTS.md/style-guide 访问权限，无法确认概览页是否豁免，记录待规划确认。其二，overview 核心机制第 1 条把 PagedAttention 表述为「传统本地 KVCache 缓存（如 vLLM 的 PagedAttention）」——论文中 PagedAttention 是分页内存管理，本地缓存复用逻辑对应的是 §6.1 "Similar reuse logic is already implemented in vLLM"（指 vLLM 的 prefix caching），两者混称会形成轻微误解｜引文依据：论文 §6.1 "Similar reuse logic is already implemented in vLLM, but the open-source version of vLLM only supports local KVCache caching"；§9 "vLLM [13] leverages dynamic KVCache management"（[13] 即 PagedAttention 论文）｜修复要求：与规划确认 overview 是否需要 dojo meta；把「本地 KVCache 缓存（如 vLLM 的 PagedAttention）」改为「如 vLLM 的本地 prefix caching」或拆开表述｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 10 / 轻微 15
- 处置：修复（无需返回规划的问题；overview.html 的 dojo meta 豁免问题需规划侧确认后归档）
- 补充说明：本轮未发现核心结论与来源冲突的阻断级问题；四步工作流、Algorithm 1 主体、CPP/layer-wise prefill、早拒绝与预测、端到端数字（+20%/+40%/50%–525%/+75%、57% vs ~100%、4183→3771→3589、块大小 512、LRU 30%→50% 等）已逐条对照论文原文核对无误。style-guide.md 与 AGENTS.md 主题词表不在本轮允许输入内，2.2.12（格式一致性）与 dojo:topics 词表合规性未审查，留待后续轮次或由编排者补充核对。
