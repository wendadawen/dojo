# EPLB 第 3 轮审查（独立，最终）

- 页面版本：index.html 工作树哈希 `f51d075b3bef2e6c0308260cac4448a7da0a8206`；overview.html `5b9d5d98fe4bb135861225ae2e6ca9c8a7b3b020`
- 审查时间：2026-09-03 14:39
- 审查者：独立审查者（第 3 轮，未参与写作与前两轮审查及修复）
- 已完整阅读章节（按顺序）：引言与 12 专家负载表、核心问题（含全部解答）、常见误解、1. 专家并行的木桶效应（1.1/1.2/本章问题）、2. 冗余专家（2.1/2.2/2.3/本章问题）、3. 三步排布（3.1/3.2/3.3/本章问题，含 SVG、伪代码折叠块、可运行代码折叠块）、4. 周期性重平衡（4.1/4.2/4.3/本章问题）、5. 边界（本章问题）、来源与范围说明（C/F/N/构造示例/辅助解释/简化条件）、overview.html 全文
- 来源核对方式：独立重新抓取 EPLB README 与 eplb.py（raw.githubusercontent.com main 分支）、DeepSeek-V3 报告（arXiv:2412.19437v2 HTML 全文）、vLLM docs/serving/expert_parallel_deployment.md、vllm/distributed/eplb/policy/default.py、vllm/config/parallel.py（均 main 分支）；页面可运行代码与全部中间量另行用独立脚本复算
- 机械验证：`python3 .dojo/scripts/validate.py wiki/eplb/index.html` → `validation ok`；`python3 .dojo/scripts/validate.py wiki/eplb/overview.html` → `validation ok`

## 前两轮修复复核

**独立性说明**：按本质检规范 check.md §1「审查者不读取 research/ 中的规划、修复记录和前序审查结果」及本次任务指令，本轮未读取 review-1.md、review-2.md、draft-check.md，无法将本轮发现逐条映射到第 1/2 轮的问题编号。因此「第 1 轮问题 1-12」「第 2 轮问题 1-12」的逐条已修/未修/部分修状态**无法核对**，改以「修复敏感区域的最终状态全量复核」替代——若某区域最终状态正确，则相关修复视为到位；若仍有问题，列出如下。未读取前轮记录也意味着无法判断下列问题属残留还是新引入。

最终状态全量复核结果（每行：区域｜结论｜证据）：

- C1（负载随 workload 变化）｜与来源一致｜README 首段原文 "the load of different experts may vary depending on the current workload"
- C2（冗余专家 + 启发式打包 + 同组同节点）｜与来源一致｜README "duplicates heavy-loaded experts…heuristically pack…place the experts of the same group to the same node"
- C3（节点受限路由）｜与来源一致｜V3 报告 §2.1.2 "each token will be sent to at most $M$ nodes, which are selected according to the sum of the highest $K_r/M$ affinity scores"
- C4（负载预测 out of scope、移动平均）｜与来源一致｜README "out of this repo's scope. A common method is to use moving average of historical statistics"
- C5（层次/全局触发条件与 prefill/decode 建议）｜与来源一致｜README "When the number of server nodes divides the number of expert groups"（节点数整除组数）；eplb.py `if num_groups % num_nodes == 0` 分支
- C6（贪心复制）｜与来源一致｜eplb.py replicate_experts docstring "such that the maximum load of all replicas is minimized"，实现为 `(weight / logcnt).max`
- C7（均衡打包）｜与来源一致｜eplb.py balanced_packing docstring "each bin contains exactly n/m objects"，实现为降序 + 最轻有容量包
- C8（物理负载 = 逻辑负载 ÷ 副本数）｜与来源一致｜eplb.py `tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)`，原副本同样记均摊值
- C9（接口约束）｜论断表条目正确，但正文两处方向表述反向（见重要问题 1）｜eplb.py 四个 assert：`num_logical_experts % num_groups == 0`、`num_groups % num_nodes == 0`、`num_gpus % num_nodes == 0`、`num_physical_experts % num_gpus == 0`
- C10（V3 prefill）｜与来源一致｜§3.4.1 "4 nodes with 32 GPUs…TP4…SP…DP8…EP32…32 redundant experts…besides the original 8 experts…one additional redundant expert…every 10 minutes…without increasing the cross-node all-to-all communication overhead"
- C11（V3 decode）｜与来源一致｜§3.4.2 "treat the shared expert as a routed one…select 9 experts…40 nodes with 320 GPUs…DP80…EP320…each GPU hosts only one expert, and 64 GPUs are responsible for hosting redundant experts and shared experts…we do not need to rearrange"
- C12（bias 步末 ±γ 更新、不丢 token；推理期不更新为标注推断）｜与来源一致｜§2.1.2 "At the end of each step…decrease the bias term by γ if…overloaded, and increase it by γ if…underloaded"；"does not drop any tokens during training…also does not drop tokens during inference"；推断已在正文与 C12 条目双重标注
- C13（vLLM EPLB、default 改编自 DeepSeek EPLB）｜与来源一致｜vLLM 文档 "Enable EPLB with the `--enable-eplb` flag…collects load statistics with every forward pass and periodically rebalances"；default.py 模块 docstring "The rearrangement algorithm is adapted from [DeepSeek EPLB]"
- C14（vLLM 参数默认值）｜与来源一致｜文档参数表与 parallel.py EPLBConfig：window_size=1000、step_interval=3000、num_redundant_experts=0、log_balancedness=False、policy="default"；step_interval 的「大于窗口只用最近窗口」注记与 parallel.py 原文一致
- C15（两个公式、2.4 GB、建议 32）｜与来源一致｜文档 "Each EP rank has (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS"；"This overhead equals NUM_MOE_LAYERS * BYTES_PER_EXPERT * (…) ÷ NUM_EP_RANKS…approximately 2.4 GB for one redundant expert per EP rank"；"recommend setting…to 32 in large scale use cases"；页面将 2.4 GB 解读为「每 rank 1 个冗余的增量」并给出等价式，与文档语义一致且可复算
- C16（preserve_intragpu_slots）｜与来源一致｜default.py docstring "experts that remain on the same GPU keep their previous slot positions…Incoming experts…fill any remaining available slots…only when the number of GPUs is unchanged and the slots per GPU remain the same"
- C17（V3 超参）｜与来源一致｜§4.2 "61…first three layers…1 shared expert and 256 routed experts…8 experts will be activated…at most 4 nodes"
- C18（探索性动态冗余）｜与来源一致｜§3.4.1 "each GPU hosts more experts (e.g., 16 experts), but only 9 will be activated…compute the globally optimal routing scheme on the fly…overhead…almost negligible"
- C18a（静态摆放）｜实质内容与来源一致，来源定位名不准确（见次要问题 2）｜parallel.py `ExpertPlacementStrategy = Literal["linear", "round_robin"]`，字段默认 "linear"；类型别名不叫 PlacementPolicy，也不是枚举
- C19（带宽与 4 节点限制）｜与来源一致｜§3.2.2 "NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)"；"limit each token to be dispatched to at most 4 nodes, thereby reducing IB traffic"
- F1/F2｜与标注一致｜F1 为 eplb.py tokens_per_phy 的形式化；F2 已明确标注「本页构造，非来源公式」并给出适用条件
- N1/N2/N4/N5/N6｜与来源/复算一致｜N1 数值与 README 示例逐项相同；N2 全部中间量经独立复算一致（见最终核查 ⑧）；N4/N5 与 §3.4.1/§3.4.2 一致；N6 与 vLLM 文档一致
- 结构规范（h1/h2/h3 编号、问题块、折叠块前缀、meta、链接）｜到位｜h1 符合「概念名（英文缩写）：核心作用」；h2 编号 1–5 连续；「核心问题」「本章问题」「来源与范围说明」命名正确；23 个 summary 全部为 解答：/补充：/展开：/代码： 前缀；5 个被引用概念页（moe-serving、aux-loss-free-routing、model-parallelism、ultraep、moonep）均存在；overview 与 index 互链；dojo:topics「并行与通信,推理系统」均在 ALLOWED_TOPICS 词表内
- 可运行代码｜实际执行且输出与页面「预期输出」逐行一致｜独立复跑结果：组负载 [262, 330, 116, 325]、组->节点 [1, 0, 0, 1]、节点0/1 副本数、各 GPU 负载、phy2log 与官方 tensor 一致 = True

## 最终核查

### ① 整除链方向最终一致性 —— 存在 3 处反向表述（重要问题 1）

页面 9 处整除表述方向正确（严格用法「A 整除 B」= A | B）：§2.2「物理专家总数必须被 GPU 数整除」、C9 条目中文注释「组数整除逻辑专家数、节点数整除组数、节点数整除 GPU 数」、§3.2「当节点数整除组数时」、层次/全局对照表触发条件行、§3.1 答案约束链、伪代码分支行「若 num_nodes 整除 num_groups」、§3.3「当组数不被节点数整除时」、§2 答案 2、§5 答案 2（「GPU 数须整除物理专家数」，8|16 成立）。

但以下 3 处方向相反（严格读法与示例数值矛盾：12|4、4|2、8|2 均不成立）：

- index.html:892（§2.2 正文）：「逻辑专家数要整除组数、组数要整除节点数、GPU 数要整除节点数」——按标准数学用法（A 整除 B = B 是 A 的倍数），这三项全部反向；与同页 C9 条目的正确注释直接冲突。
- index.html:1065-1067（伪代码折叠块约束行）：「num_log 整除 num_groups、num_groups 整除 num_nodes、num_gpus 整除 num_nodes」——同一折叠块内第 1076 行分支条件「若 num_nodes 整除 num_groups」用的是正确方向，块内自相矛盾。
- index.html:1232（§3 本章问题 3 的 summary）：「组数不整除节点数时启用」——严格读「4 不整除 2」为真，但示例（4 组 2 节点）走的恰是层次策略，与本章内容矛盾；应为「组数不被节点数整除时启用」。同一答案正文（1233 行）表述正确。

### ② C/F/N 双向对应 —— 完整，两处编号瑕疵（次要）

正文使用的编号：C1–C19（含 C18a）全部出现且均对应文末条目；文末 20 条 C 条目全部在正文被引用；F1/F2 双向对应；N1、N2、N4、N5、N6 双向对应。无「只引不列」或「只列不引」。

- N 编号跳号：存在 N1、N2、N4、N5、N6，无 N3（index.html:1436）。不影响对应关系，疑为删项后未重排。
- C18a 的来源定位写「PlacementPolicy 枚举」（index.html:1422）；parallel.py 中实际标识符为类型别名 `ExpertPlacementStrategy = Literal["linear", "round_robin"]`，既非枚举也不叫 PlacementPolicy。论断内容本身（linear、round_robin、不依赖统计的初始摆放）与源码一致。

### ③ 5 张表格数据与来源对照 —— 全部一致

1. 12 专家负载表（755 行）：90/132/40/61/104/165/39/4/73/56/183/86，合计 1033 —— 与 README 示例 weight tensor 层 0 逐项一致；合计复算 1033 ✓。
2. 贪心复制 4 步表（904-907 行）：e10 183→91.5、e5 165→82.5、e1 132→66、e4 104→52；「当前最大」列 165/132/104/91.5 —— 独立手算逐格一致。
3. 训练期 bias vs 推理期 EPLB 对照表（836-846 行）：各行与 C12/C10 及 README 表述一致。
4. V3 prefill vs decode 部署表（1279-1292 行）：逐行与 V3 报告 §3.4.1/§3.4.2 原文一致（4 节点 32 卡/40 节点 320 卡、TP4+SP DP8/DP80、EP32/EP320、8+1/1、32/64 卡、重排方式、每 token 8/9 专家）。
5. vLLM 参数表（1305-1316 行）：window_size 1000、step_interval 3000（含「大于窗口只用最近窗口」）、num_redundant_experts 0、log_balancedness 关（含均衡度公式说明）、policy default —— 与 vLLM 文档参数表及 parallel.py EPLBConfig 一致（文档另有 use_async、communicator 两参数，页面自称「主要参数」，未列不构成错误）。

另核两张未编号对照表：层次/全局策略表（1200-1211 行）触发条件、复制范围、8 卡负载区间（86.5–156 vs 95.5–138.5）、max/mean（1.21 vs 1.07）、README 建议用途——全部与 README、eplb.py 及复算一致；边界表（1365-1376 行）为定性归纳，与来源表述一致。

### ④ 章节引用标题化 —— 主体到位，4 处裸编号残留（次要）

无 S1/S2/S3 式章节代号。核心问题答案、§1.1、§3.3、§5 等处以「N. 标题」引用（如「完整论证见『1. 专家并行的木桶效应』」「两类部署的实际差别见『4. 周期性重平衡』」）✓。以下 4 处使用裸编号、未带章节标题，与 style-guide §1「正文引用其他章节时使用章节标题」不一致：

- index.html:884「量化公式放在第 4 章 vLLM 一节」
- index.html:892「『组』是什么、为什么要整除，第 3 章展开」
- index.html:920「（第 3 章第二步）……两种策略的差异见 3.3 节」
- index.html:1275「可以用第 4.3 节的公式验证」（对比 1363 行「见『4. 周期性重平衡』的 4.3 节」写法正确）

### ⑤ 折叠块预告 —— 4/5 有锚点，1 处无预告（次要）

- 「展开：贪心复制 12→16 的逐步验证」：块前正文已给出贪心结论（91.5 < 129.125 预算）✓
- 「展开：第三步节点 0 的完整手算」：块前正文已给出四卡负载结果 ✓
- 「代码：三步算法的伪代码」：块前已完整描述三步机制 ✓
- 「代码：单层层次策略的纯 Python 复刻」：块前正文（1053 行）已陈述「与官方 tensor 逐槽一致」的结论 ✓
- 「补充：V3 探索中的动态冗余策略」（1295 行）：块前正文只讲了 prefill/decode 的实际部署，未提及任何「探索方向」，折叠块引入的是全新内容而非对已陈述结论的展开。按组件库 09「块前正文须先说明它补充哪个已陈述的结论」，此块缺预告锚点。

### ⑥ 章节问题答案独立可读性 —— 合格

5 个「本章问题」共 12 问、页面级「核心问题」5 问，全部有「解答：」折叠块；答案自含结论、关键数字与成立条件，不依赖回看正文（如 §3 答案 2 自带组负载、复制对象、四卡负载与 phy2log 全量数字；§4 答案 3 自带公式与 2.4 GB 数字）；核心问题答案均以「完整论证见『N. 标题』」指明章节且不逐句重复正文 ✓。

### ⑦ 构造示例与辅助解释的边界标注 —— 合格

- 层利用率公式标注「本页构造，非来源公式」（824 行）且 F2 条目复述构造条件与失效边界 ✓
- 分发层均摊假设标注「显式假设」并声明分发机制不在仓库范围、相关表述「仅为辅助解释」（882 行）✓
- 「推理期 bias 不再更新」在正文（832 行）与 C12 条目双重标注为推断 ✓
- 文末「构造示例」「辅助解释与类比边界」「简化条件及其限制」三节齐备，覆盖两卡 800/200、冷专家对照、GPU 7 = [e1, e1] 等全部构造中间量，并声明已用独立复刻核对 ✓

### ⑧ SVG 槽位顺序与复制算法中间量 —— 与官方逐槽一致

SVG 排布图（967-1051 行）逐槽对照官方 phy2log 层 0 = [5,6,5,7,8,4,3,4,10,9,10,2,0,1,11,1]：

- GPU 0 = [e5, e6] → 槽 0-1 = 5,6 ✓；GPU 1 = [e5·副本, e7] → 槽 2-3 = 5,7 ✓；GPU 2 = [e8, e4·副本] → 槽 4-5 = 8,4 ✓；GPU 3 = [e3, e4] → 槽 6-7 = 3,4 ✓
- GPU 4 = [e10, e9] → 槽 8-9 = 10,9 ✓；GPU 5 = [e10·副本, e2] → 槽 10-11 = 10,2 ✓；GPU 6 = [e0, e1·副本] → 槽 12-13 = 0,1 ✓；GPU 7 = [e11, e1] → 槽 14-15 = 11,1 ✓
- 图中各槽均摊负载（82.5/39/82.5/4/73/52/61/52 与 91.5/56/91.5/40/90/66/86/66）、8 张卡总负载（121.5/86.5/125/113/147.5/131.5/156/152）、节点总负载（446/587）、组归属（节点 0 = G1+G2、节点 1 = G0+G3）全部与独立复算一致；原图注「每张卡的两个槽即映射中相邻的两项」成立。

复制算法中间量独立手算：组负载 G0-G3 = 262/330/116/325 ✓；第一步打包 G1→节点 0、G3→节点 1、G0→节点 1、G2→节点 0（负载 446/587）✓；第二步节点 0 复制 e5、e4，节点 1 复制 e10、e1（副本数向量 [1,2,2,1,1,1] 与 [1,2,1,1,2,1]）✓；第三步节点 0 手算（含平手取先遍历包的细节：e6→卡 0）与节点 1 手算（56→卡 4 因 91.5 先被遍历）逐句复算成立 ✓；全局策略 8 卡负载 130.5/95.5/130/138/138.5/134.5/134/132、max/mean = 1.0726 ≈ 1.07 ✓；G0 的 e0、e2 落 GPU 2、e1 两副本落 GPU 7（合计 132）✓。页面可运行代码经独立执行，输出与页面「预期输出」逐行一致，「与官方输出一致: True」。

### ⑨ 其他核查

- 数学符号：validate.py 通过（含裸数学字符、SVG text、ASCII 近似、框线图检查）；summary/表格内的 $w_e/c_e$、$\approx$、$\gamma$ 等均为 LaTeX。轻微残留：§4 答案 3（1348 行）以纯文本写「(专家总数 + 冗余专家数) ÷ EP rank 数」「MoE 层数 × 每专家字节数 × 冗余专家数 ÷ EP rank 数」，§2 答案 2（935 行）写「16 = 8×2」，§1.1（824 行）正文写「层利用率 = 平均负载 ÷ 最大负载」——×、÷ 不在 validate.py 禁用清单内，但与正文 $$...$$ 公式及 $\div$ 写法并存，全页公式书写风格不统一。
- 数值精度：§1.1（826 行）与 §1 答案 1（854 行）两处「层利用率至多约 70.8%」——复算 129.125/183 = 0.7056，应为约 70.6%；同句的 69%（129.125/187 = 0.6905）与 1.42、1.45 均正确。
- 用词：无「我们/你们/您」、无「待生成」占位、自称用「本页」✓。
- overview.html：全部数字（10 分钟、3000 步、1000 步窗口、8+1、9 专家、320 卡、2.4 GB、每卡 9 = (256+32)/32）与来源及 index 一致；KaTeX 已加载可渲染 $w_e/c_e$；无公式推导与计算示例，符合概览规范。

### ⑩ 新发现问题（无法判定是否前两轮已存在）

- 重要问题 2：「最糟的搭配」用词与算术相反。index.html:826：「若 e10 与另一个专家共卡，最糟的搭配是 e10 与最冷的 e7 共卡，最大负载 187」。共卡时卡负载 = 183 + 搭档负载，最糟（最大）搭配应是搭档最重的 e5（348）；e10+e7（187）恰是共卡搭配中最好的一种，其作用是给出「至少 187」的下界（与同句「max/mean 至少 ≈ 1.45」的方向一致）。数字与下界结论均正确，但「最糟」一词让该句在字面上为假。§1 答案 1 未复述该词，仅此一处。

## 总结

- 阻断性问题：无。核心结论（三步排布、贪心复制、均摊公式、phy2log 逐槽一致、V3/vLLM 事实）全部经来源原文与独立复算验证成立。
- 重要问题：
  1. 整除链方向 3 处反向（index.html:892、1065-1067、1232），与同页 9 处正确表述及 C9 条目冲突，按严格读法与示例数值（12、4、2、8）矛盾。
  2. index.html:826「最糟的搭配是 e10 与最冷的 e7 共卡」方向说反（最糟应为搭档最重；e7 搭配是下界情形），字面为假但下界数字与结论正确。
- 次要问题：
  1. N 编号跳号（无 N3，index.html:1436）。
  2. C18a 来源定位写「PlacementPolicy 枚举」，实际为 parallel.py 的类型别名 `ExpertPlacementStrategy`（index.html:1422）。
  3. 「层利用率至多约 70.8%」应为约 70.6%（129.125/183 = 0.7056），两处（index.html:826、854）。
  4. 4 处裸编号章节引用（884、892、920、1275 行）未用章节标题，与其余「N. 标题」引用方式不一致。
  5. 「补充：V3 探索中的动态冗余策略」折叠块（1295 行）块前正文无预告锚点。
  6. 答案折叠块与个别正文位置以纯文本 ×、÷ 书写公式（824、935、1348 行），与全页 LaTeX 公式风格不统一（validate.py 不拦截）。
- 发布是否就绪：**否**。按 check.md §5「阻断和重要问题均已关闭」方可发布，当前存在 2 项重要问题未关闭。两项均为局部文字修正（改 3 处整除措辞方向、改 1 处「最糟」表述），不涉及结构、数据与来源核对，修复后逐条复验即可，无需第四轮全量审查；6 项次要问题建议随同修正，其中次要问题 3（70.8%→70.6%）为数字错误，应一并修复。
