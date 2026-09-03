# Mooncake 审查记录（第 3 轮）

- 页面版本：index.html a2c3ddb5b58e56f04a0ecd64078b2eb7bce0a8e0；overview.html c739e76979383c5e431378b6a063ba94b1c5c5be（git hash-object，工作树）
- 审查时间：2026-09-03
- 审查者：独立审查者（第 3 轮，未参与写作与前两轮审查，未读取 research/ 下任何文件）
- 输入：wiki/mooncake/index.html、wiki/mooncake/overview.html、/tmp/mooncake-research/mooncake.txt（arXiv:2407.00079v4 全文）、guides/concept/check.md
- 已完整阅读章节（index.html，按顺序）：核心问题、常见误解、第 1 章（1.1/1.2/本章问题）、第 2 章（2.1/2.2/2.3/本章问题）、第 3 章（3.1/3.2/3.3/3.4/本章问题）、第 4 章（4.1/4.2/4.3/本章问题）、第 5 章（5.1/5.2/5.3/5.4/本章问题）、来源与范围说明（C/F/N 表、构造示例、简化条件）；overview.html 全文（问题背景、核心机制、关键结论与边界）
- 机械验证：`.dojo/scripts/validate.py wiki/mooncake/index.html` 返回 `validation ok`；概念链接（paged-attention / kv-cache / pcp-dcp / chunked-prefill / prefix-caching）与 libs 本地资源全部存在；两页相互链接存在；dojo:topics 取值（推理系统、内存与缓存、并行与通信）均在 AGENTS.md 固定大类内
- 复算通过项：12,288 = 24×512；8,192/12,288 = 66.7%；2×80×8×128×2 = 320 KiB/token；12,288×320 KiB = 3,840 MiB = 3.75 GiB；131,072×320 KiB = 40 GiB；贯穿示例 A/B/C 三实例 TTFT（4.524/9.012/3.036）、纯负载均衡 4.072、纯缓存优先 9.012 全部可复算；(3771−3589)/3771 ≈ 4.8% ≈ "~5%"；Algorithm 1 伪代码与论文 Algorithm 1 逐步一致；C6–C28、C33–C35、N1–N16、F1/F2 的数值与定位逐条核对通过（下述问题除外）

## 问题

### 重要

- [重要·技术] index.html:1431（「来源与范围说明 → 构造示例」）vs index.html:972（2.3 折叠块）与 index.html:1397（F1）：同一估算两处数值矛盾——构造示例节写「12,288 tokens × 320 KiB/token ≈ 3.84 GiB」，2.3 折叠块与 F1 写「约 3.75 GiB」。复算：12,288×320 KiB = 3,932,160 KiB = 3,840 MiB = 3.75 GiB；3.84 是把 3,840 MiB 的数值误标为 GiB，无法从给定公式复算得出。｜引文依据：不适用（构造示例，输入为 LLaMA2-70B 配置 L=80、H_kv=8、d_head=128、b=2，见 F1）；论文 §8.1 Testbed "8 NVIDIA-A800-SXM4-80GB GPUs" 支持「占单卡一半」的语境。｜修复要求：将 1431 行的 3.84 GiB 改为 3.75 GiB，与 972 行、1397 行一致。｜修复：｜复验：

- [重要·技术] index.html:818（1.1 节第 2 段）：「decode 每批每请求只能前进 1 个 token，是访存密集型：计算量随 batch 大小只次线性增长」——论文原文是「computation **time** … increase sublinearly with batch size」（计算时间次线性），不是「计算量」次线性；计算量随 batch 大小是线性增长的，次线性的成因是访存受限（同一份 KVCache 读取服务多个请求）。C7 表项本身写对（「时间随 batch 大小次线性增长」），正文与 C7 不一致。｜引文依据：论文 §2 "This makes it memory-constrained and causes computation time to increase sublinearly with batch size, as shown in the right part of Figure 2."。｜修复要求：把「计算量随 batch 大小只次线性增长」改为「计算时间随 batch 大小次线性增长」，与本页 dg-layer 图示（「时间随 batch 大小次线性」）和 C7 一致。｜修复：｜复验：

- [重要·技术] index.html:1256（5.1 节第 2 段）：对论文 §7.1 的转述语义反转。页面写「在传统耦合系统里，prefill 与 decode 互相干扰，难以用单一负载度量决定拒收」；论文原文是耦合系统里 TTFT/TBT 预测被干扰复杂化，因此负载「常用处理中请求数与系统最大容量之比这个简单（单一）度量」衡量——即耦合系统恰恰用一个粗粒度单一度量，Mooncake 分离后才改用 SLO 满足度。页面丢掉了「请求数比例」这一关键对照，且「难以用单一负载度量」与原文方向相反。｜引文依据：论文 §7.1 "In conventional coupled systems, the prediction of TTFT and TBT can be complicated by interference between the prefill and decoding stages. Therefore, the load is often measured simply by the ratio of the number of requests being processed to the system's maximum capacity. In contrast, Mooncake … Thus we use SLO satisfaction as a direct load measurement."。｜修复要求：改写该句为：耦合系统因两阶段干扰难以准确预测 TTFT/TBT，负载只能用「处理中请求数 / 最大容量」的简单比值衡量；Mooncake 分离后可直接用 SLO 满足度（预测实例上的 TTFT、TBT 是否超 l_ttft / l_tbt）作为负载度量。｜修复：｜复验：

- [重要·可读性] index.html:866（1.2 本章问题 2 解答）：「(2) 彻底分离后 prefill 实例只需装下单请求，VRAM 可以留给 decode 装尽可能大的 batch」——主语从 prefill 实例跳到 decode，字面可读成「prefill 实例的 VRAM 留给 decode 使用」；在分离集群中两者是不同机器，prefill 实例省下的 VRAM 并不能给 decode 用。论文的理由 2 也只是 prefill 侧节省 VRAM（§5.2）。｜引文依据：论文 §5 "There are two main reasons for this decision: 1) Prefill nodes require different cross-node parallelism settings to handle long contexts (§5.1). 2) It presents a unique opportunity to save VRAM (§5.2)."；§1 "restricted … by the total size of the aggregated KVCache that can be contained in the VRAM"。｜修复要求：改写为两个独立分句：prefill 实例的 VRAM 只需容纳单请求（layer-wise prefill，见第 4 章）；decode 实例的 VRAM 不再被 prefill 占用，可全部用于聚合 decode batch。｜修复：｜复验：

### 轻微

- [轻微·格式] index.html:820（1.1 节）：「TTFT<sub>P90</sub> = 10×、TBT<sub>P90</sub> = $5\times$」——同句中 10× 用 Unicode 乘号、5× 用 KaTeX，违反「数学符号全部由 KaTeX 渲染、正文无 Unicode 数学字符直接出现」。｜引文依据：不适用。｜修复要求：10× 改为 $10\times$。｜修复：｜复验：

- [轻微·格式] index.html:1099–1101、1117（3.4 节两个表格）：单元格内算式「10240/8192 = 1.25 &lt; 2」「0.5 + 4096/4000 = 1.524」「1.0 + (12288 - 0)/4000 = 4.072」为纯文本，与同节表头/正文的 KaTeX 写法（$T_{\text{prefill}}$、best/prefix$_i$、$≥$、$∞$）不一致；比较符混用 HTML 实体（&lt;/&gt;）与 KaTeX（$≥$/$≤$），同一变量全页写法不统一。｜引文依据：不适用。｜修复要求：表格内算式与比较符统一由 KaTeX 渲染，或全部改用与全页一致的纯文本约定（二选一，保持一致）。｜修复：｜复验：

- [轻微·来源] index.html:1408（N1）：来源定位标「§4.1」，实际「23,608 entries / 1 小时采样」出自 §4 开头引言段（§4.1 Data Details 之前）。｜引文依据：论文 §4 引言 "we sampled a subset of online request data from a 1-hour period … The trace dataset comprises 23,608 entries"（§4.1 标题 "Data Details" 之前）。｜修复要求：N1 来源改为「§4」。｜修复：｜复验：

- [轻微·来源] index.html:820 与 N10（§2 来源定位）：页面写「常用 TTFT_P90 = 10×、TBT_P90 = 5× 来约束」——「常用」暗示通用惯例；论文是本文端到端实验的设定（§8.1 Metric 处重复为 "multiplying the lowest observed RPS values by factors of 10 and 5"）。｜引文依据：论文 §2 "Specifically, in the end-to-end experiment of this paper (§8.1), we set TTFTP 90 = 10× and TBTP 90 = 5×."。｜修复要求：「常用」改为「本论文端到端实验中设定」。｜修复：｜复验：

- [轻微·来源] index.html:1361/1390（C1、C36 表项）：C36（HTTP 429）在来源表定义，但正文相应论断（781 行核心问题 3 解答「若超出 SLO 直接 HTTP 429 拒绝」）无 [C36] 标注；C1、C2 同样只在来源表出现，intro/正文对应论断无标注。表项与正文标注不对应。｜引文依据：论文 §6.1 "If the SLO is not achievable, Conductor directly returns the HTTP 429 Too Many Requests response status code to the upper layers."（内容支持，仅标注缺失）。｜修复要求：在核心问题 3 解答的 HTTP 429 论断处补 <sup>[C36]</sup>（intro 的 C1/C2 论断处补注，或删除表中无正文对应的孤儿条目）。｜修复：｜复验：

- [轻微·来源] index.html:882、1074：「按分页块（默认 $B=512$ tokens）管理」「$B$ = 512 tokens，块大小（论文 §4.1）」——论文中 512 出现在 §4.1 的 trace hash 生成上下文（"hashing token blocks (with a block size of 512)"），未把 512 声明为系统级「默认」配置；「默认」一词扩大了来源适用范围。｜引文依据：论文 §4.1 "It is generated by hashing token blocks (with a block size of 512) into prefix hash values …"；论文未在其他位置给出系统块大小默认值。｜修复要求：改为「trace 与示例采用 $B=512$（论文 §4.1）」或等价表述，去掉「默认」。｜修复：｜复验：

- [轻微·来源] index.html:1177（4.2 节收益 2）：「短请求可以走单节点 PP 不切块，长请求切块流水线」——论文仅说 CPP "naturally fits both short and long contexts, bringing no significant overhead for short context prefill"，未描述短请求走单节点 PP 的具体路径；此为推断未标注。｜引文依据：论文 §5.1 "2) It naturally fits both short and long contexts, bringing no significant overhead for short context prefill and avoiding frequent dynamic adjustment of node partitioning."。｜修复要求：改为论文原意的「对短上下文 prefill 无显著额外开销」，或标注「（推断：短请求可由单 stage 组处理）」。｜修复：｜复验：

- [轻微·可读性] index.html:1176（4.2 节收益 1）：文字重复——「与 stage 内计算重叠，因此 通信少且可重叠，MFU 优于……」，「通信少且可重叠」在同句出现两次。｜引文依据：不适用。｜修复要求：删除「因此 通信少且可重叠，」中的重复片段，改为「因此 MFU 优于跨节点 TP 与 SP（论文未给与单节点 TP 的直接对比）」。｜修复：｜复验：

- [轻微·可读性] index.html:754（intro 段）与 1157（4.1 表格）：术语首次使用未解释——「SLO」在 intro 首次出现（1.1 才定义）；「MFU」首次出现于核心问题 4 解答（788 行）与 4.1 表格列头（1160 行附近），定义在 4.1 正文（1167 行）之后；「RDMA」「GPUDirect RDMA」「HBM」全文未展开。｜引文依据：不适用。｜修复要求：SLO 在 intro 首次出现处给出全称或移至 1.1 首次出现；MFU 的全称移到首次出现处；RDMA/HBM 首次出现处补一句简短说明。｜修复：｜复验：

- [轻微·格式] overview.html:67（关键结论第 2 条）：「KVCache-centric 调度把平均 TTFT从 random 92.07s 降到 6.26s，把 cache-aware（14.36s）和 load-balancing（60.41s）也都优于随机」——「TTFT从」缺空格；「把 X 和 Y 也都优于随机」句式不通（缺谓语改写）。｜引文依据：论文 §6.2/Figure 8 数值 92.07、60.41、14.36、6.26（数值本身核对无误）。｜修复要求：改为「……平均 TTFT 从 random 的 92.07s 降到 6.26s；cache-aware（14.36s）与 load-balancing（60.41s）也均优于 random」。｜修复：｜复验：

- [轻微·可读性] index.html:1126（3.4 节末段）：「只有前两者满足 SLO 线（$≈$50s）」——「前两者」指代歧义：按本句列举顺序（random 92.07 → KVCache-centric 6.26 → cache-aware 14.36 → load-balancing 60.41）会被读成 random 与 KVCache-centric；实际指 KVCache-centric 与 cache-aware（overview.html 表述明确）。｜引文依据：论文 Figure 8：柱值 6.26、14.36、60.41、92.07，SLO 线位于 50s。｜修复要求：改为「其中只有 KVCache-centric（6.26s）与 cache-aware（14.36s）满足 SLO 线（$≈$50s）」。｜修复：｜复验：

- [轻微·格式] index.html:882（2.1 组件列表第 4 项）：`<li>` 未以 `</li>` 闭合，却在句末多余一个 `</p>`（列表项内无起始 `<p>`），HTML 标签错配；浏览器容错后渲染正常，但结构不规范。｜引文依据：不适用。｜修复要求：删除多余的 `</p>`，补 `</li>`。｜修复：｜复验：

- [轻微·来源] index.html:972（2.3 折叠块）：「把 GPU 集群里长期闲置的『邻居』容量（不仅是便宜）纳为己用；同时 RDMA 互联为 KVCache 搬运提供了独立于 GPU 算力的高带宽通路」——C3 仅支持「利用闲置 CPU/DRAM/SSD 资源提供缓存容量与传输带宽」；「邻居容量」「不仅是便宜」「独立于 GPU 算力」均为页面推断，未标注。｜引文依据：论文 Abstract "leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster"；§3 "harnesses underutilized resources to provide ample cache capacity and transfer bandwidth"。｜修复要求：删去「（不仅是便宜）」，其余推断性短语标注「（推断）」。｜修复：｜复验：

- [轻微·来源] index.html:848（1.2 节分离理由 2）：「decode 实例的 VRAM 不再被 prefill 抢占，可全部用于聚合 decode batch」——论文的两条理由中，理由 2 只涉及 prefill 侧节省 VRAM；decode 侧收益是页面对分离架构的直接推论，未标注推断。｜引文依据：论文 §5 "2) It presents a unique opportunity to save VRAM (§5.2)."（§5.2 全节讨论 prefill 实例的 VRAM 占用）。｜修复要求：decode 侧分句标注「（推断，分离的直接结果）」或移入下一句作为独立说明。｜修复：｜复验：

- [轻微·可读性] index.html:754（intro 段）：「常常把同 batch 上所有 decode 序列的 TBT 拉高一个或更多 step」——TBT 是 token 间隔时间，与「拉高……step」搭配不当（应为「推迟若干步」或「拉长若干个 step 的时长」）。｜引文依据：不适用。｜修复要求：改为「把同 batch 上所有 decode 序列推迟一个或更多 step，拉高其 TBT」。｜修复：｜复验：

- [轻微·来源] index.html:965（2.3 工程要点 1）：「剩余输入 $≤$ 阈值时不分块，避免流水线填充开销」——论文只说阈值「选择以充分利用对应 GPU 的算力」（"This threshold is selected to fully utilize the corresponding GPU's computational power"），未给出「避免流水线填充开销」这一理由；推断未标注。｜引文依据：论文 §3 step 2 "This threshold is selected to fully utilize the corresponding GPU's computational power and is typically larger than 1000 tokens."。｜修复要求：「避免流水线填充开销」标注「（推断）」或删除。｜修复：｜复验：

- [轻微·来源] index.html:1088（3.4 折叠块）：「论文实验中 prefill 时间与传输速率受实际负载与网络拥塞影响，由 §6.1 描述的预测模型估计」——§6.1 的离线拟合预测模型只针对 prefill 执行时间；论文明确说传输时间预测「更难」且未描述其预测模型。｜引文依据：论文 §6.1 "we employ a predictive model derived from offline test data. This model estimates the prefill duration…"；同节 "More difficulty lies in predicting the transfer time because it is determined not only by the size of the transferred data but also by the current network status"。｜修复要求：拆开表述：prefill 时间由 §6.1 预测模型估计；传输时间更难预测（论文未给出其模型细节）。｜修复：｜复验：

- [轻微·来源] index.html:912/1006 与 C33（1387 行）：「这一实现与 vLLM 的本地缓存复用一致（块哈希链）」——论文只说 "Similar reuse logic is already implemented in vLLM"，未指明 vLLM 的实现是块哈希链；「（块哈希链）」是页面对「similar reuse logic」的具体化，未标注。｜引文依据：论文 §6.1 "Similar reuse logic is already implemented in vLLM, but the open-source version of vLLM only supports local KVCache caching."。｜修复要求：删去「（块哈希链）」或改为「与 vLLM 已实现的复用逻辑类似（论文未展开其实现细节）」。｜修复：｜复验：

- [轻微·来源] index.html:843（1.2 节）：「vLLM 等系统引入的连续批处理」——按论文引文 [12]（Orca），iteration-level/continuous batching 的引入者是 Orca，vLLM [13] 是采用者；「vLLM 引入」归因不准（C8 表项写「Orca 等」是对的，正文与之不一致）。｜引文依据：论文 §2 "A widely used optimization in the decoding stage is continuous batching [12, 13]"；参考文献 [12] Orca "a distributed serving system…"、[13] vLLM "…with pagedattention"。｜修复要求：改为「Orca、vLLM 等系统采用的连续批处理」。｜修复：｜复验：

- [轻微·来源] index.html:1224（4.3 节）与 F2：「$T$ 为该请求在 GPU 上的驻留时间」——论文原文为 processing time（处理时间）；「驻留时间」与「处理时间」在该论证（inline 使 T 增大）中结论同向但定义不同，未说明差异。｜引文依据：论文 §5.2 "if the KVCache size of a request is S and the processing time is T, its occupation cost is S ∗ T."。｜修复要求：改为「$T$ 为该请求的处理时间」或注明「论文称 processing time」。｜修复：｜复验：

- [轻微·可读性] index.html:1134（3 章本章问题 1 解答）：「当 best_prefix_len/prefix_len$_i$ &lt; threshold，$p_i$ 的本地前缀与全局最佳差距不大（与全局最佳差距在阈值倍数以内）成本可控」——括号与前文重复且「成本可控」前缺连接词，句子不通。｜引文依据：不适用。｜修复要求：改为「……差距不大（在阈值倍数以内）、直接本地复用的成本可控，……」。｜修复：｜复验：

- [轻微·可读性] overview.html:66（关键结论第 1 条）：「该结论在 cache ratio &gt; 0% 的多轮/系统提示词场景下成立」表述含糊——同页第 3 条又说 ArXiv（cache ratio ≈ 0%）「收益仍存在但不显著」，两者需要读者自行调和；且「> 0%」与「多轮/系统提示词」的对应关系未说明。｜引文依据：论文 Table 2（ArXiv ~0%、L-Eval >80%、模拟 50%、真实 ~50%）。｜修复要求：改为「显著增益（50%–525%）出现在 cache ratio 较高或上下文较长的场景；低复用场景（ArXiv ~0%）增益缩小到 +20% 但仍存在」。｜修复：｜复验：

- [轻微·图示] index.html:1183–1219（4.2 CPP SVG）：第 3 个任务框（x=560–680）右边界超出时间轴末端（x=660），且时间轴标签只到 t5（x=560，与框左边界重合），最后一个时段缺少结束刻度；不影响语义（stage 流水线时序、箭头指向、act 传递标注均正确），但时间轴与框边界不对齐。｜引文依据：不适用（图为页面构造示意）。｜修复要求：时间轴延长至 x=680 或补 t6 标签，使框边界与刻度对齐。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 23
- 处置：修复（4 条重要问题修复后需复验；全部修复后重跑 `.dojo/scripts/validate.py`）
- 总体评价：来源覆盖完整（C/N/F 条目均可在论文中定位到支持片段，范围引用 C29–C32 覆盖 C30/C31），数值除 3.84 GiB 一处内部矛盾外全部与论文一致，贯穿示例与 Algorithm 1 分支全部可复算，跨页在 Figure 8、五大组件、端到端数字上表述一致，第 5 章对 §7.1–§7.4 的机制覆盖完整；主要残留问题集中在个别转述与原文语义偏差（计算量/计算时间、§7.1 耦合系统度量）、一处数值矛盾（3.84 GiB）、推断未标注与格式一致性。
