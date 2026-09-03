# EPLB 第 2 轮审查（独立）

- 审查对象：`wiki/eplb/index.html`、`wiki/eplb/overview.html`
- 审查时间：2026-09-03
- 审查方式：独立重读页面全文（含折叠块），用 WebFetch 重新抓取 EPLB README/eplb.py、DeepSeek-V3 报告（arXiv:2412.19437 HTML 版）、vLLM 文档与源码（default.py、parallel.py），实际运行页面内复刻代码，运行 `validate.py`。未读取 research/ 下任何既有文件。
- 机械校验：`python3 .dojo/scripts/validate.py wiki/eplb/index.html` 与 `overview.html` 均通过。
- 代码复验：将页面 3.2 节复刻代码原样保存运行，实际输出（组负载 [262, 330, 116, 325]、组→节点 [1, 0, 0, 1]、两节点副本数、各 GPU 负载 121.5/86.5/125/113 与 147.5/131.5/156/152、phy2log = [5,6,5,7,8,4,3,4,10,9,10,2,0,1,11,1]、与官方一致 True）与页面「预期输出」逐行一致；另跑全局策略复算得 8 卡负载 130.5/95.5/130/138/138.5/134.5/134/132、max/mean = 1.0726、e0/e2 在 GPU2、e1 两副本同在 GPU7，与 3.3 节正文一致。

## 上轮修复复核

- 上轮问题 1（C9 整除方向）：**部分修** — C9 表格现为「num_logical_experts % num_groups == 0、num_groups % num_nodes == 0、num_gpus % num_nodes == 0（即组数整除逻辑专家数、节点数整除组数、节点数整除 GPU 数）」，与 eplb.py 四条 assert 逐条一致；3.2「当节点数整除组数时」、3.3「当组数不被节点数整除时」、层次/全局对照表触发条件、3.2 问题解答的约束链均与源码 `if num_groups % num_nodes == 0` 一致。但 2.2 正文末尾「逻辑专家数要整除组数、组数要整除节点数、GPU 数要整除节点数」与伪代码约束「num_log 整除 num_groups、num_groups 整除 num_nodes、num_gpus 整除 num_nodes」按中文标准数学语义（A 整除 B = B % A == 0）方向全部相反，伪代码分支条件「若 num_groups 整除 num_nodes」同样相反；与页内其他位置的正确用法并存，形成页内自相矛盾（详见新发现 1）。
- 上轮问题 2（C15 内存公式）：**已修** — 正文 4.3 现引文档字面公式「每 rank 总专家存储 = MoE 层数 × 每专家字节数 × (专家总数 + 冗余专家数) / EP rank 数」，并把 2.4 GB 明确为「由冗余带来的增量开销」、给出「减去无冗余项，等价于 MoE 层数 × 每专家字节数 × 冗余专家数 / EP rank 数」的换算。文档原文「This overhead equals NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS. For DeepSeekV3, this is approximately 2.4 GB for one redundant expert per EP rank」支持该处理。
- 上轮问题 3（F2 下界→上界）：**已修** — 1.1 现写「最大负载与均值之比至少 $187/129.125 \approx 1.45$，层利用率至多约 69%」，方向正确（129.125/187 ≈ 0.691）；F 节交代构造条件与 $(0,1]$ 值域、并声明未考虑流水线/all-toall 重叠。
- 上轮问题 4（C17 定位）：**已修** — C17 标注 [v3] §4.2 Model Hyper-Parameters；报告原文确认「61 层、前 3 层 dense、每 MoE 层 1 共享 + 256 路由专家、每 token 激活 8 路由专家、至多 4 节点」全部出自该节。
- 上轮问题 5（h1 括号）：**已修** — h1 为「专家并行负载均衡（EPLB）：复制热专家、均衡摆放，拉平忙闲不均」，符合 style-guide「概念名（英文缩写）：核心作用或结论」格式。
- 上轮问题 6（blockquote 顺序）：**部分修** — blockquote.meta 已移至 learning-goals 之前，但仍在引言（负载表两段）之后；style-guide §2 固定顺序为 reading-time → blockquote.meta → 引言 → learning-goals，blockquote 应在引言之前（详见新发现 3）。
- 上轮问题 7（F1 符号 <ul> 形式）：**部分修** — $w_e$、$c_e$ 已在公式前正文定义、含义明确，但公式 $$\ell_e = w_e/c_e$$ 之后仍无 `<ul>` 逐项定义符号的列表，与 style-guide §11「公式后紧跟 <ul> 逐项定义每个符号」的形式要求不符（4.3 节两公式符号为中文词，可豁免）。
- 上轮问题 8（C19 通信重叠偏强）：**已修** — 3.1 现仅表述「V3 正是靠限制每 token 至多 4 个节点来压住 IB 流量」，与报告原文「we limit each token to be dispatched to at most 4 nodes, thereby reducing IB traffic」对应，不再宣称计算-通信重叠；「木桶效应」的类比边界也声明不涉及重叠执行模型。
- 上轮问题 9（SVG 节点 1 蓝框位置）：**已修** — 逐槽核对 SVG 与官方 phy2log = [5,6,5,7,8,4,3,4,10,9,10,2,0,1,11,1]：GPU1 = [e5·副本(左), e7] 对应 slot 2-3 = (5,7)；GPU2 = [e8, e4·副本(右)] 对应 slot 4-5 = (8,4)；GPU5 = [e10·副本(左), e2] 对应 slot 10-11 = (10,2)；GPU6 = [e0, e1·副本(右)] 对应 slot 12-13 = (0,1)；GPU7 = [e11, e1] 对应 slot 14-15 = (11,1)。节点 0、节点 1 的全部 16 个槽位顺序及蓝框（副本）位置均与官方输出一致。
- 上轮问题 10（overview 计算示例）：**已修** — 概览不再包含手算示例；保留数字（4 节点 32 卡 EP32 每卡 8+1、40 节点 320 卡每卡 1 专家每 token 9 专家、vLLM 3000/1000、2.4 GB）均为来源事实且与 index.html 一致。
- 上轮问题 11（N3 正文引用）：**部分修** — 正文 4.2 现引用 [C10, N4]（prefill）与 [C11, N5]（decode），但 N3 在全文无任何正文引用；且来源节将 N3–N5 合并为一条定义，读者无法分辨 N4、N5 各自覆盖哪些数字，与「双向对应」要求不完全满足（详见新发现 5）。
- 上轮问题 12（毫秒级/轮流分给/expert_placement_strategy）：**部分修** — expert_placement_strategy 已补 C18a 引用（源码 parallel.py 确有 `ExpertPlacementStrategy = Literal["linear", "round_robin"]`，默认 "linear"），但 C18a 来源定位写作「PlacementPolicy 枚举」，与源码实际名称/形式不符，且「初始摆放选项」的定性在源码 docstring 中无逐字依据、未标注为推断；「按副本轮流分发」加了「具体分发机制不在 EPLB 范围内」的限定，但「常见的做法是按副本轮流分发」仍是未标注来源的一般性机制断言（README 仅对负载预测陈述过 moving average）；「两个贪心……毫秒级就能跑完」仍是完全无来源、未标注推断的性能断言（详见新发现 2）。

## 新发现

1. **[重要·技术] 2.2 正文与伪代码的整除链方向与源码相反、页内自相矛盾**（index.html 2.2 节末段「层次策略路径上还有一条整除链：逻辑专家数要整除组数、组数要整除节点数、GPU 数要整除节点数」；伪代码「约束：……num_log 整除 num_groups、num_groups 整除 num_nodes、num_gpus 整除 num_nodes」及「若 num_groups 整除 num_nodes（层次策略）」）。引文依据：eplb.py `assert num_logical_experts % num_groups == 0`、`assert num_groups % num_nodes == 0`、`assert num_gpus % num_nodes == 0`，分支 `if num_groups % num_nodes == 0`；README「When the number of server nodes divides the number of expert groups」。按「A 整除 B」的标准语义，源码要求是「组数整除逻辑专家数、节点数整除组数、节点数整除 GPU 数」——即 C9 表格、3.2/3.3 正文、对照表与 3.2 问题解答采用的方向；2.2 末段与伪代码三处约束加一处分支条件的表述方向恰好全部相反。同一页两种相反用法并存，读者无法判断哪一个是接口真实约束。
2. **[重要·技术] 4.1「重算本身便宜——两个贪心在十几个到几百个专家的规模上毫秒级就能跑完」为无来源性能断言**。引文依据：不适用（README/eplb.py/V3 报告/vLLM 文档均无重算耗时的任何数字或量级陈述）。按 check.md 2.2「无来源支持的机制描述……删除该论断及其推论，或降级为明确标注的推断」，该句既无 C/F/N 引用也无推断标注，且给出了具体量级（毫秒级）与规模范围（十几个到几百个专家），读者会当作已核实事实。同段「贵的是搬运权重」的定性对比亦建立在该断言之上。
3. **[轻微·格式] blockquote.meta 位于引言之后，违反 style-guide §2 前置 section 固定顺序**（reading-time → blockquote.meta → 引言 → learning-goals → misconceptions）。页面现为 reading-time → 引言（负载表 + 定义段）→ blockquote.meta → learning-goals → misconceptions。引文依据：不适用。
4. **[轻微·技术] 1.1 与 N2 的「至少 187 / 至多约 69%」隐含假设未交代**。「若不复制 e10，它所在的卡至少还要再放一个专家（每卡 2 槽）」——若允许空槽（12 专家 16 槽），e10 独占一卡时最大卡负载为 183，比值为 183/129.125 ≈ 1.42、利用率至多约 70.8%；「至少 187」依赖「e10 所在卡必须放第二个专家」这一未说明的前提（在 num_phy = 16 且槽位填满的 EPLB 约束下成立，在「不做处理」的对比情景中未声明）。结论方向不受影响，但下界数值的论证有缝隙。引文依据：不适用（构造对比）。
5. **[轻微·格式] N3 无正文引用，N3–N5 合并定义边界不明**。来源节「N3–N5：V3 部署数字（prefill……；decode……）」一条打包三个编号，正文仅出现 [C10, N4] 与 [C11, N5]，N3 全文无引用；无法从来源节文字判断 N3、N4、N5 各自的覆盖范围，双向对应不完整。引文依据：不适用。
6. **[轻微·技术] C18a 来源定位名称与源码不符、定性未标注推断**。parallel.py 中实际是类型别名 `ExpertPlacementStrategy = Literal["linear", "round_robin"]`（非「PlacementPolicy 枚举」，也非 Enum）；其 docstring 只描述两种摆放的布局规则，未出现「initial placement」或「不依赖统计」的表述。「那是初始摆放选项，与按统计动态重排的 EPLB 是两回事」是页面对两套配置分离这一事实的合理综合，但属推断，未按规范标注。引文依据：parallel.py `expert_placement_strategy: ExpertPlacementStrategy = "linear"` 及 docstring；`enable_eplb: bool = False` 为独立配置项。
7. **[轻微·技术] C15 来源定位未覆盖「大规模建议 32」的出处**。「We recommend setting --eplb-config '{"num_redundant_experts":32}' to 32 in large scale use cases」位于 vLLM 文档 Example Command 小节，而 C15 的来源定位仅写「[vllm-doc] Expert Distribution Formula 与 Memory Footprint Overhead 节」。论断本身与文档一致。引文依据：见前句原文。
8. **[轻微·技术] 「常见的做法是按副本轮流分发」无来源标注**（2.1 节括号内）。README 仅对负载预测陈述「A common method is to use moving average of historical statistics」，未提及副本分发方式；分发机制的「常见做法」在任何引用来源中均无对应陈述。已加「具体分发机制不在 EPLB 范围内」限定、危害有限，但按 check.md 应删除或降级为标注推断。引文依据：README 原文只覆盖负载预测。
9. **[轻微·格式] 正文章节引用使用编号而非章节标题**（「也会在第 2 章由冗余专家一并解决」「量化公式放在第 4 章 vLLM 一节」「两种策略的差异见 3.3 节」「见 3.3 节」等）。style-guide §1「正文引用其他章节时使用章节标题」；页面核心问题解答中已用标题（「完整论证见『1. 专家并行的木桶效应』」），正文与章节解答混用编号引用。引文依据：不适用。
10. **[轻微·技术] 边界表将 MoonEP 列为推理部署重排间隙漂移的「谁管」，未标注其为训练场景方案**。MoonEP 页自述「MoonEP 是训练方案，不适用于推理场景」；EPLB 页第 5 章语境为推理部署（V3 两层均衡），「重排间隙的负载漂移｜谁管：UltraEP（配额实时规划）、MoonEP（完美均衡）」未说明 MoonEP 针对的是训练期 EP。对 MoonEP 机制本身的描述（「追求每个 EP rank 恰好收到固定数量 token 的完美均衡」）与其页面一致、无事实误述。引文依据：moonep/overview.html「MoonEP 是训练方案，不适用于推理场景」。
11. **[轻微·格式] 3.2 笔误：「每组数恰为节点数的两倍」多一「每」字**，应为「组数恰为节点数的两倍」。引文依据：不适用。
12. **[轻微·可读性] 3 章问题 3 解答中的「G0」未在该解答内定义**（组划分定义在同章问题 1 的解答与 3.1 正文，不在本解答内），独立可读性有小的折扣。引文依据：不适用。

正面核对记录（本轮独立复核、均通过）：

- 负载表 12 个数字与合计 1033 与 README 逐字一致；贪心步骤表（183→91.5、165→82.5、132→66、104→52 及各步当前最大值）复算无误；2.3 折叠块的逐步 $w_e/c_e$ 终值表复算无误。
- 组负载 262/330/116/325、节点负载 446/587、层次 8 卡负载与 max/mean ≈ 1.21、全局 8 卡负载与 max/mean ≈ 1.07、全局 GPU7 = [e1, e1] 合计 132、G0 专家落在 GPU2/GPU7——全部由独立运行复算证实。
- 复刻代码实际运行输出与页面「预期输出」逐行一致，「与官方输出一致: True」属实。
- C1–C19 逐条对照来源：C1（README 首段）、C2（redundant experts 策略原文）、C3（§2.1.2 Node-Limited Routing「at most $M$ nodes...the sum of the highest $K_r/M$ affinity scores」；README「group-limited expert routing」）、C4（out of scope + moving average）、C5（README The Algorithm 节 + eplb.py 分支）、C6（replicate_experts docstring 与实现）、C7（balanced_packing docstring 与实现）、C8（tokens_per_phy = tokens_per_mlog / mlogcnt 的 gather）、C9（四条 assert）、C10（§3.4.1 全部数字含「e.g., every 10 minutes」「without increasing the cross-node all-to-all communication overhead」）、C11（§3.4.2「treat the shared expert as a routed one...9 experts...40 nodes with 320 GPUs...TP4 with SP, combined with DP80...EP320...each GPU hosts only one expert, and 64 GPUs are responsible for hosting redundant experts and shared experts」）、C12（每步末 ±γ 更新与训练/推理均不丢 token 原文；推理期 bias 不更新的推断已在正文标注）、C13（default.py 模块 docstring「The rearrangement algorithm is adapted from [DeepSeek EPLB]」逐字存在，论断成立）、C14（文档参数表与 parallel.py EPLBConfig 默认值逐一吻合；step_interval「大于窗口时只用最近窗口」有 parallel.py docstring「if this is greater than the EPLB window size, only the metrics of the last lb_window_size steps will be used」依据）、C16（preserve_intragpu_slots docstring 与「GPU 数与槽位不变时才应用」一致）、C17、C18（§3.4.1 动态冗余「e.g., 16 experts...only 9 will be activated...compute the globally optimal routing scheme on the fly」）、C19（§3.2.2「NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)」）均有可定位原文支持。
- F1/F2 的适用条件在正文与来源节均已交代（F1 均摊假设在 2.1 正文与 F 节双处；F2 构造性质在 1.1 正文标注「本页构造，非来源公式」、条件与值域在 F 节）。
- 相邻概念链接全部有效（moe-serving、aux-loss-free-routing、model-parallelism、ultraep、moonep 目录均存在），对 UltraEP（配额耦合副本与 token 接量、实时规划）与 MoonEP（每 rank 恰好固定数量 token 的完美均衡）的描述与各自页面一致，无事实性误述；deepseek-moe 页未链接，但正文仅顺带提及「V3 细粒度专家」而非依赖该概念，不构成违规。
- 各级问题块（核心问题 5 条、每章本章问题）均有「解答：」折叠块；抽查各解答均独立成段、自带结论与关键数字，不依赖回看正文（除新发现 12 一处小折扣）。
- 7 张表数据（负载表、训练 vs 推理、贪心步骤、层次 vs 全局、prefill vs decode、边界对照、vLLM 参数）逐项与来源及复算一致；「毫秒级」与「轮流分发」两处无来源表述见新发现 2、8。
- overview.html 与 index.html 相互链接；overview 数字与 index 及来源一致；overview 无计算示例。
- validate.py 两个页面均通过。

## 总结

- 阻断性问题：无。核心机制、算法手算、复刻代码输出、SVG 槽位顺序、7 张表格数据经独立复算与来源核对均正确。
- 重要问题：2 项 — ① 2.2 正文整除链与伪代码（含分支条件）的「整除」方向与 eplb.py 源码相反且与页内其他位置自相矛盾；② 4.1「毫秒级就能跑完」为无来源、未标注推断的性能断言。
- 次要问题：10 项 — blockquote 顺序、F1 无 <ul> 符号列表、N3 无正文引用且 N3–N5 边界不明、「至少 187」隐含假设未交代、C18a 来源定位名称不符且定性未标注推断、C15 来源定位未含「建议 32」出处、「轮流分发」无来源标注、章节编号引用、MoonEP 场景差异未标注、「每组数」笔误（另附 3 章问题 3 解答「G0」指代一处极轻微折扣）。
