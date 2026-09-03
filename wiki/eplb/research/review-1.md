# EPLB 第 1 轮审查（独立）

- 页面：wiki/eplb/index.html、wiki/eplb/overview.html
- 审查时间：2026-09-03
- 审查者：独立审查者（未参与写作，未读取 research/ 下任何规划与笔记文件）
- 机械校验：`python3 .dojo/scripts/validate.py wiki/eplb/index.html` 与 `overview.html` 均返回 validation ok
- 独立获取并核对的来源原文：deepseek-ai/EPLB main 分支 README.md 与 eplb.py；DeepSeek-V3 报告（arXiv:2412.19437，HTML 全文）；vLLM main 分支 docs/serving/expert_parallel_deployment.md、vllm/distributed/eplb/policy/default.py、vllm/config/parallel.py（补充核对 C14 与 expert_placement_strategy）
- 页面可运行代码已由审查者独立提取执行：退出码 0，输出与页面「预期输出」逐行一致，phy2log 与官方 tensor 逐槽一致
- 构造例中间量（组/节点/GPU 负载、全局策略 8 卡负载、max/mean）已由审查者独立手算复核算，全部一致

## 来源论断逐条核对（C1–C19）

- C1：[专家负载随 workload 变化，需保持各 GPU 负载均衡] — 来源定位：[repo] README 首段（L3–4）— 一致 — "the load of different experts may vary depending on the current workload, it is important to keep the load of different GPUs balanced"
- C2：[冗余专家策略：复制高负载专家、启发式打包、同组尽量同节点] — 来源定位：[repo] README 首段（L5–8）— 一致 — "duplicates heavy-loaded experts… heuristically pack… place the experts of the same group to the same node to reduce inter-node data traffic"
- C3：[V3 节点受限路由：至多 $M$ 个节点，按最高 $K_r/M$ 亲和分数和选节点] — 来源定位：[v3] §2.1.2 Node-Limited Routing 段 — 一致 — "at most M nodes, which are selected according to the sum of the highest K_r/M affinity scores of the experts distributed on each node"；README L7 称 group-limited expert routing 亦核实
- C4：[算法基于估计负载；负载预测 out of scope，常见做法移动平均] — 来源定位：[repo] README（L11–13）— 一致 — "based on the estimated expert loads… out of this repo's scope… moving average of historical statistics"
- C5：[层次策略条件（节点数整除组数）、三步；全局策略其余情况、prefill 小 EP / decode 大 EP 建议] — 来源定位：[repo] README The Algorithm 节（L19–31）+ eplb.py L150–156 分支 — 一致 — "When the number of server nodes divides the number of expert groups… can be used in prefilling stage with a smaller expert-parallel size"；`if num_groups % num_nodes == 0` 分支核实
- C6：[贪心复制：每次给 weight/logcnt 最大者加副本，最小化最大副本负载] — 来源定位：[repo] eplb.py L44–71（docstring + L67 `(weight / logcnt).max(dim=-1).indices`）— 一致 — docstring "the maximum load of all replicas is minimized"
- C7：[均衡打包：降序、放入仍有容量的最轻包、每包恰 n/m 个] — 来源定位：[repo] eplb.py L5–41 — 一致 — docstring "each bin contains exactly n/m objects"，实现为降序 sort + min(pack_weights) 带容量过滤
- C8：[物理专家打包负载 = 逻辑负载 ÷ 副本数] — 来源定位：[repo] eplb.py L117 `tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)` — 一致 — 每份副本（含 rank 0）均记均摊值
- C9：[接口约束：num_replicas 是 num_gpus 的倍数；层次路径「逻辑专家数整除组数、组数整除节点数、GPU 数整除节点数」] — 来源定位：[repo] eplb.py L90/L92/L94/L95 四个 assert 与 docstring — 部分支持 — assert 实为 `num_logical_experts % num_groups == 0`、`num_groups % num_nodes == 0`、`num_gpus % num_nodes == 0`，即「组数整除逻辑专家数、节点数整除组数、节点数整除 GPU 数」；页面 2.2、C9 表、第 3 章本章问题答案三处把整除方向写反（页面 3.2 正文与 overview 的「节点数整除组数」是正确的，页面自相矛盾）
- C10：[V3 prefill：4 节点 32 卡、TP4+SP+DP8、EP32、32 冗余、每卡 8+1、约每 10 分钟、节点内重排不加跨节点 all-toall] — 来源定位：[v3] §3.4.1 — 一致 — "4 nodes with 32 GPUs… TP4 with SP, combined with DP8… EP32… 32 redundant experts… one additional redundant expert… every 10 minutes… without increasing the cross-node all-to-all communication overhead"
- C11：[V3 decode：共享专家视为路由专家（每 token 9 专家）、40 节点 320 卡、TP4+SP+DP80、EP320、每卡 1 专家、64 卡承载冗余与共享、无需重排] — 来源定位：[v3] §3.4.2 — 一致 — "treat the shared expert as a routed one… select 9 experts… 40 nodes with 320 GPUs… TP4 with SP, combined with DP80… EP320… each GPU hosts only one expert, and 64 GPUs are responsible for hosting redundant experts and shared experts… do not need to rearrange experts"
- C12：[bias 每训练步末按过载/欠载 ±γ 更新；训练与推理都不丢 token] — 来源定位：[v3] §2.1.2 Auxiliary-Loss-Free 段与 No Token-Dropping 段 — 一致 — "At the end of each step, we will decrease the bias term by γ… does not drop any tokens during training… also does not drop tokens during inference"；「推理期 bias 不再更新」页面已明确标注为推断，处理合规
- C13：[vLLM 内置 EPLB：--enable-eplb、每前向收集统计周期性重排；default 策略改编自 DeepSeek EPLB] — 来源定位：[vllm-doc] L142–144 + [vllm-src] 模块 docstring L8–9 — 一致 — "Enable EPLB with the --enable-eplb flag… collects load statistics with every forward pass"；"The rearrangement algorithm is adapted from DeepSeek EPLB"
- C14：[vLLM 参数默认值 window_size=1000、step_interval=3000、num_redundant_experts=0、log_balancedness=关、policy=default] — 来源定位：[vllm-doc] L150–157 参数表 + vllm/config/parallel.py L59–90 EPLBConfig — 一致 — 表值与 Field 默认值逐一相符；step_interval「大于窗口时只用最近窗口统计」与 parallel.py L68–70 docstring 相符
- C15：[每 rank 专家数 = (总数+冗余)/EP rank 数；内存开销 = MoE 层数 × 每专家字节 × 冗余数 / EP rank 数；V3 每 rank 1 冗余约 2.4 GB；大规模建议 32] — 来源定位：[vllm-doc] L179–205（Expert Distribution Formula 与 Memory Footprint Overhead 节）— 部分支持 — 每 rank 专家数公式、2.4 GB、建议 32 均逐字相符；但内存开销公式页面写作「× 冗余专家数」，文档字面为「× (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS」，两者不一致（详见公式核对与问题清单）
- C16：[preserve_intragpu_slots：留在本卡的专家保持原槽位、新专家填空槽] — 来源定位：[vllm-src] preserve_intragpu_slots docstring（L192–204）— 一致 — "experts that remain on the same GPU keep their previous slot positions… Incoming experts… fill any remaining available slots"，适用条件（GPU 数与每卡槽位不变）亦相符
- C17：[V3 配置：61 层（前 3 层 dense）、每 MoE 层 1 共享 + 256 路由、每 token 激活 8、至多 4 节点] — 来源定位：页面标注 [v3] §4.2 与 §3.3 — 部分支持 — 全部数字在 §4.2 Hyper-Parameters（Model Hyper-Parameters 段）逐字可定位："61… first three layers… 1 shared expert and 256 routed experts… 8 experts will be activated… at most 4 nodes"；但 §3.3 为 FP8 Training，不含任何该组数字，属定位错误（数字本身无误）
- C18：[V3 探索中的动态冗余：每卡 16 专家、每步激活 9、每层 all-toall 前在线算最优路由、prefill 语境开销可忽略] — 来源定位：[v3] §3.4.1 末段 — 一致 — "each GPU hosts more experts (e.g., 16 experts), but only 9 will be activated… compute the globally optimal routing scheme on the fly… prefilling stage… almost negligible"
- C19：[NVLink 160 GB/s 约 IB 50 GB/s 的 3.2 倍；限制每 token 至多 4 节点压 IB 流量] — 来源定位：[v3] §3.2.2 — 一致 — "NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)… limit each token to be dispatched to at most 4 nodes, thereby reducing IB traffic"；同节另有 "both dispatching and combining kernels overlap with the computation stream" 支持页面「让通信与计算重叠」的说法（详见问题清单第 8 条，因果链表述偏强）

## 公式核对（F1, F2）

- F1（$\ell_e = w_e/c_e$）：一致。eplb.py L117 `tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)` 即该式的直接形式化；页面已声明「均摊为显式假设、分发层职责」，与来源无冲突。审查者按算法独立复算了层次与全局两条路径的全部中间负载（组 262/330/116/325、节点 446/587、8 卡 121.5/86.5/125/113 与 147.5/131.5/156/152、全局 130.5/95.5/130/138/138.5/134.5/134/132），与页面数字全部一致。
- F2（层利用率 = 平均负载 ÷ 最大负载）：页面已声明为本页构造、非来源公式，处理合规；但正文 1.1 称之为「层利用率**下界**」在数学方向上错误——实际层时间 ≥ 最大负载对应时间，故利用率 = 理想时间/实际时间 ≤ 平均/最大，「平均 ÷ 最大」是利用率的**上界**（两卡 800/200 的 62.5% 是该简化模型下的最优可能值）。数值本身无误，方向词需修正（见问题清单第 3 条）。
- 附：页面 4.3 的内存开销公式「MoE 层数 × 每专家字节数 × 冗余专家数 ÷ EP rank 数」与其标注来源 [vllm-doc] Memory Footprint Overhead 节的字面公式「NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS」不一致：文档字面公式的分子是「专家总数+冗余专家数」，页面改为「冗余专家数」。页面的版本与文档同节的算例（V3 每 rank 1 个冗余约 2.4 GB，即 58 个 MoE 层 × 每专家约 41 MB）自洽，文档字面公式按 (256+32)/32=9 个专家/rank 计算约 21–23 GB，与 2.4 GB 算例矛盾——来源文档自身公式与算例存在张力，页面取了与算例一致的改写形式，但未声明与来源字面公式的差异（见问题清单第 2 条）。

## 数字与外部实验条件核对（N1–N6）

- N1：一致。README L43–56：层 0 负载 [90,132,40,61,104,165,39,4,73,56,183,86]、num_replicas=16、num_groups=4、num_nodes=2、num_gpus=8、输出层 0 = [5,6,5,7,8,4,3,4,10,9,10,2,0,1,11,1]、层 1 = [7,10,6,8,6,11,8,9,2,4,5,1,5,0,3,1]，逐项核对相符；页面复刻代码由审查者独立执行，输出与页面「预期输出」及官方 tensor 完全一致（含「组→节点 [1,0,0,1]」「副本数向量」）。
- N2：一致（审查者独立手算复核）。总负载 1033、均值 129.125、贪心四步 e10/e5/e1/e4（183→91.5、165→82.5、132→66、104→52）、层次 max/mean = 156/129.125 ≈ 1.21、全局 max/mean = 138.5/129.125 ≈ 1.07、GPU 7 = [e1,e1] 合计 132、187 = 183+4（约 1.45、69%）全部复算相符。注：「不复制 e10 时其所在卡至少 187」是反事实构造推理而非算法输出，页面在 N 节与「构造示例」节均已声明为构造量，处理合规。
- N3：数字与 [v3] §3.4.1 相符（4 节点 32 卡、EP32、每卡 8+1、约每 10 分钟）。但 N3 编号在正文任何位置均未被 `<sup>` 引用（正文 prefill 数字只引了 C10 与 N4），C/F/N 双向对应存在缺口。
- N4：一致。[v3] §3.4.1："32 redundant experts… one additional redundant expert… every 10 minutes… rearrange experts among GPUs within a node… without increasing the cross-node all-to-all communication overhead"。
- N5：一致。[v3] §3.4.2："40 nodes with 320 GPUs… EP320… each GPU hosts only one expert, and 64 GPUs… redundant experts and shared experts… 9 experts"。
- N6：一致。[vllm-doc] L163–166（Qwen/Qwen3-30B-A3B + --enable-eplb 示例）、L197–202（DeepSeek-V3-0324 示例）、L205（"recommend… num_redundant_experts… 32 in large scale use cases"）；默认值与 C14 同源核对相符。

## 页面规范问题（标题/格式/数学/问题块/结构图）

1. [重要·技术] 2.2 末段、来源表 C9、第 3 章本章问题答案：「逻辑专家数整除组数、组数整除节点数、GPU 数整除节点数」整除方向与 eplb.py assert（L90 `num_logical_experts % num_groups == 0`、L92 `num_groups % num_nodes == 0`、L94 `num_gpus % num_nodes == 0`）正好相反，也与页面 3.2 正文、3.3、对照表及 overview 的「节点数整除组数」自相矛盾｜引文依据：eplb.py L90/L92/L94 四个 assert｜修复要求：统一改为与 assert 一致的方向表述。
2. [重要·技术] 4.3 内存开销公式与 C15：公式分子「冗余专家数」与 vllm-doc 字面「(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS)」不一致，页面未声明改写｜引文依据：vllm-doc L188 "This overhead equals NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS"｜修复要求：改为与来源字面一致，或明确标注为对来源算例（2.4 GB）的改写及两者差异。
3. [重要·数学] 1.1「$R$ 张卡的层利用率下界 = 平均负载 ÷ 最大负载」：方向词错误，平均÷最大是利用率的**上界**（实际层时间 ≥ 最慢卡时间）；两卡 800/200 的 62.5% 为该模型下最优值而非下界｜引文依据：不适用（本页构造公式 F2 自身推导）｜修复要求：「下界」改为「上界」（或改述为「利用率的乐观估计」），F2 节说明同步。
4. [轻微·来源] C17 定位「[v3] §4.2 与 §3.3」：§3.3 为 FP8 Training，不含该组任何数字；全部数字在 §4.2 Hyper-Parameters（Model Hyper-Parameters 段）｜引文依据：报告目录及 §4.2 原文｜修复要求：删除 §3.3 或改为正确章节。
5. [轻微·格式] h1「EPLB 专家并行负载均衡：……」未采用 style-guide 第 1 节「概念名（英文缩写）：核心作用」的括号形式（如「专家并行负载均衡（EPLB）：……」）｜引文依据：不适用｜修复要求：按规范调整或维持时给出接受理由。
6. [轻微·格式] 前置顺序：页面为 reading-time → 引言（负载表+定义段）→ blockquote.meta → learning-goals，style-guide 第 2 节固定顺序为 reading-time → blockquote.meta → 引言 → learning-goals｜引文依据：不适用｜修复要求：将「主要依据」blockquote 移至引言之前。
7. [轻微·格式] 公式符号说明：F1 公式（2.1）的符号在公式前的正文定义而非「公式后紧跟 <ul> 逐项定义」（style-guide 第 11 节）；4.3 两个文字公式同样无符号列表。信息完整、仅形式不符｜引文依据：不适用｜修复要求：按需补 <ul> 或注明接受理由。
8. [轻微·技术] 3.1「V3 正是靠限制每 token 至多 4 个节点来压住 IB 流量、让通信与计算重叠」：「压住 IB 流量」在 §3.2.2 有直接因果支持；「通信与计算重叠」的因果支持在 §2.1.2（"Under this constraint… nearly achieve full computation-communication overlap"，主语为训练框架）及 §3.2.2 的 kernel 实现描述，与 C19 的定位（§3.2.2）不完全对应，因果链表述偏强｜引文依据：§2.1.2/§3.2.2 原文｜修复要求：弱化为「减少 IB 流量」（或补充定位）。
9. [轻微·技术] 结构图（3.2）：节点 1 的 e1 副本蓝框标在 GPU 7，按 eplb.py 的 rank 语义（replicate_experts 返回的 phyrank），冗余槽副本实际落在 GPU 6（phy2mlog 冗余槽 7 经第三步打包进卡 6 rank 1），GPU 7 上的 e1 为原件；phy2log 逐槽一致、两份 e1 参数与负载完全相同（66），仅「蓝框=副本」的标注与官方 rank 语义相反。节点 0 的 e5/e4 副本蓝框位置经复算正确｜引文依据：eplb.py L66–71 与第三步打包复算｜修复要求：对调 GPU 6/GPU 7 的 e1 蓝框，或图注说明蓝框仅为示意。
10. [轻微·格式] overview.html「关键事实」含代入计算「(256+32)/32 = 9」，write.md 第 5 节规定概览「不包含公式推导、不包含计算示例」｜引文依据：不适用｜修复要求：改为直接陈述「每卡 9 个（8+1）」。
11. [轻微·格式] N3 在正文无 `<sup>[N3]</sup>` 引用，C/F/N「双向对应」存在缺口（N1/N2/N4/N5/N6 均有正文引用）｜引文依据：不适用｜修复要求：在 4.2 prefill 段补引 N3，或将其并入 N4。
12. [轻微·来源] 三处无编号的机制/性能陈述：4.1「两个贪心在十几个到几百个专家的规模上毫秒级就能跑完」（无来源的性能断言）；2.1「dispatch 时把发往 $e$ 的 token 轮流分给它的各个副本」（「轮流」为具体分发机制，来源不含分发实现，紧邻的失效条件说明尚可但未标记为辅助解释）；4.3「expert_placement_strategy，如 linear、round_robin」（可在 vllm/config/parallel.py L185–194 定位，但正文无编号、未入 C 表）｜引文依据：parallel.py L185–194｜修复要求：补编号/标注辅助解释，或删去「毫秒级」这类不可验证的量级断言。

其余规范项核查结果：h2/h3 编号连续、格式正确，「核心问题」「本章问题」命名与「解答：」前缀合规，两级问题块每题均有独立可读的答案且核心问题答案指明论证章节；details summary 前缀（展开/代码/补充）合规；misconceptions 每条含错误理解与正确方向；结构图为内联 SVG、使用 dg-box/dg-accent 类、`<text>` 内无数学符号与 ASCII 近似写法、蓝框与图注已定义；正文章节引用使用章节标题而非代号；前置概念链接经 validate.py 检查均存在；数学符号全部 LaTeX 书写（validate.py 通过）；页面复刻代码实际运行且输出与描述一致。

## 总结

- 阻断性问题（必须修）：无。
- 重要问题（建议修）：
  1. 整除链方向颠倒（C9 及 2.2/来源表/第 3 章本章问题答案，与 eplb.py assert 相反且页面自相矛盾）；
  2. 内存开销公式与 vllm-doc 字面公式不一致且未声明改写（C15、4.3）；
  3. 「层利用率下界」数学方向错误，应为上界（1.1、F2）。
- 次要问题（可选）：C17 的 §3.3 定位错误；h1 括号格式；blockquote.meta 顺序；公式符号 <ul> 形式；3.1 通信重叠因果链偏强；SVG 节点 1 e1 副本蓝框与 rank 语义相反；overview 含计算示例；N3 正文无引用；三处无编号机制/性能陈述（「毫秒级」「轮流分给」「expert_placement_strategy」）。

总体：19 条来源论断中 16 条完全一致、3 条部分支持（C9 方向颠倒、C15 公式改写未声明、C17 定位错误）；数字与外部实验条件 N1–N6 全部核对相符；可运行代码与构造例中间量经独立复算全部一致。核心机制（冗余专家、贪心复制、三步排布、周期性重平衡）的表述与来源忠实，无事实性遗漏或相邻概念误述；三处重要问题修复后页面质量可显著提升。
