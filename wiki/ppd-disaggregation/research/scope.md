# 内容范围

## 1. 论文定位

- 标题：Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving
- 作者：Zongze Li、Jingyu Liu、Zhen Xu、Yineng Zhang（Independent Researcher）、Tahseen Rabbani、Ce Zhang
- 单位：University of Chicago（5 人）+ Independent Researcher（1 人）
- 发表：ICML 2026；arXiv:2603.13358v2（2026-05-05 提交，本次固定版本）
- 链接：https://arxiv.org/abs/2603.13358
- 代码：https://github.com/freelulul/vllm-ppd（作者 Twitter 与 CV 确认；论文正文称原型基于 vLLM disaggregated serving 构建，附录 B 给出实现细节；仓库内容仅作辅助核对，不作核心证据）
- 版本消歧：v1（2026-03-09）与 v2（2026-05-05）无影响定位的差异报道；本页全部论断以 v2 TeX 源码为准。无同名方法冲突；"PPD" 在本文专指 Prefill Prefill-capable Decode。

一句话定位：论文用「区分 full prefill 与 append-prefill 的干扰差异 + 按请求动态路由」解决「PD 分离架构在多轮对话下反复重算历史 KV、KV 传输饱和带宽」的问题。

论文宣称的贡献（§1 列表，逐项对应原文）：
1. 识别 PD 在多轮对话下的低效：AP 走 prefill 节点带来额外重算，并通过频繁 KV 传输饱和网络带宽（§1 贡献 i，指向 §4.1/干扰分析）
2. 发现 full prefill 与 append-prefill 对 decode 的干扰差一个数量级（batch 200 下 48% vs 2% TPOT 减速），支持选择性 AP-to-D 路由（§1 贡献 ii）
3. 把多轮推理服务形式化为优化问题，传统 PD 是 $x\equiv 0$ 特例，且无单一固定策略占优（§1 贡献 iii，指向 §4.4）
4. 提出 PPD 动态路由系统：按当前负载、算子权重、初始节点分配逐请求选 $x$（§1 贡献 iv，指向 §5）
5. 实验显示 PPD 优于标准 PD 与固定策略：合成扫描上 Turn 2+ TTFT 降 48–73%，TPOT 保持竞争力（§1 贡献 v，指向 §6）

论文没做什么（容易被误认属于本文贡献，逐项排除依据）：
- 不修改 PD 的单向 KV 传输协议：KV 仍严格 P→D，P 节点从不接收反向 KV（§2.2 One-way KV Transfer Protocol；附录 B "The transfer protocol follows vLLM's standard disaggregated serving format, requiring no custom modifications"）
- 不构建分布式 KV 存储层：Mooncake/MemServe/LMCache 是外挂存储层路线，论文只讨论互补关系（§2.3、§7）
- 不强制端到端 SLO 硬约束：PPD 是"routing actuator"，端到端 SLO 界需要闭环准入控制与批调度，超出本文（§5.5 原文）
- 不做多节点真实部署实验：单节点 4×H100 NVLink，跨节点慢链路用带宽注入模拟（§4.2、§6.2）
- 不深入分析混合 R+P/D 配置：主分析排除，附录 A 给两条排除理由与胜率数字，未给理论解释（附录 A）
- 不做模型训练或模型结构改动：纯系统层路由调度
- 不解决 Turn 1 的效率：Turn 1 恒走 P（§3 "Turn 1 always uses the PD path since no cached KV exists"）

相邻工作（关键区别，是否纳入范围）：
- AMPD（he2026，并发工作）：共享「把增量 prefill 路由到 decode 节点」的高层直觉，用实时队列状态估计决策；PPD 的区别是微架构干扰分析 + 优化形式化 + 单旋钮离线控制面（§1、§7）。纳入方式：方法评价章一句话定位，不展开 AMPD 机制细节（论文也只有定性描述）
- DistServe/Splitwise：PD 分离的基础工作，本文的前提而非对比对象。前置知识由 moe-serving 页承担
- Mooncake/MemServe/LMCache/CachedAttention：分布式/外部 KV 层。它们决定「前缀状态存在哪」，PPD 决定「命中/未命中请求怎么调度」，两层互补可组合（§7）。只记录关系，不展开机制
- DuetServe/Nexus/TaiChi：GPU 内 SM 切分/资源再分配路线（§2.2）。一句话定位，不展开
- chunked prefill（Sarathi 类）/DeepSpeed-FastGen Dynamic SplitFuse：干扰缓解但不消除（§2.1）。背景一句话

## 2. 核心问题

### Q1：PD 分离在多轮对话下到底低效在哪，根因是什么？

- 预期答案：两个低效——(a) 每轮 Turn 2+ 都要把整段历史（此前响应 + 新输入）送回 P 重算 KV 再传回 D，根因是 KV 传输单向（P 生产、D 消费、无反向通道），D 上已有的响应 KV 对 P 不可达；(b) 每轮一次的 KV 传输在高负载下饱和带宽，导致排队、超时与服务降级。重算占比高达 99% 是 CachedAttention（gao2024）在真实聊天负载上的测量结论（文献已有结论），不是本文实测。
- 重要性：这是论文存在的理由；不理解它就无法理解 PPD 路由的对象是什么。
- 依赖内容：单向协议（§2.2）、多轮重算机制（§1、§2.3）、KV 传输量级（§2.2 的 256MB@2K）、降级现象（§6.2）、moe-serving（PD 分离与 KV cache）、prefix-caching（KV 复用的可能性）。

### Q2：为什么 append-prefill 放到 decode 节点本地跑不会严重拖慢 decode？

- 预期答案：full prefill 对 $n$ 个输入 token 计算 attention，复杂度 $O(n^2)$；append-prefill 只对 $m$ 个新 token 计算（每个关注 $n+m$ 个 key），复杂度 $O(m(n+m))$，$m\ll n$ 时比同长度 full prefill 便宜约 $n/m$ 倍。H100 单卡 Llama-3.1-8B 微基准：batch 200 时 full prefill 使 decode TPOT 减速约 48%，append-prefill 仅约 2%，差一个数量级；4 并发 prefill 时 +57% vs +21%；full prefill 干扰在 32K token 时增长到 3–4×，append-prefill 在 64K 时仍低于 25%。结论：decode 节点可以安全地本地处理 Turn 2+。
- 重要性：这是全文的地基论断；推翻了「所有 prefill 都重度干扰 decode」这个 PD 分离的隐含前提。
- 依赖内容：§4.1 复杂度分析、Fig.2/Fig.A1/Fig.A2、standard-attention（$O(n^2)$ 来源）、prefix-caching（复用缓存 KV 的含义）、moe-serving（prefill/decode 阶段划分、TPOT）。

### Q3：既然本地处理干扰这么小，为什么不固定把所有 append-prefill 都留在 decode 节点（$x{=}1$）？

- 预期答案：3060 个配置点（17 配置 × 18 负载 × 10 QPS）的系统扫描显示没有单一固定策略同时在 TTFT、TPOT、吞吐上占优：92.2% 的（负载, QPS）组合中 Turn 2 TTFT 的最优配置与 Avg TPOT 的最优配置不同；winner 分布上 Replica 拿走 63.3% 的 TTFT 胜场却几乎不赢 TPOT/吞吐，$x{=}0$ 拿走 38.3% 的 TPOT 胜场却 0% TTFT 胜场，$x{=}1$ 最均衡（平均 27.0%）但单项不垄断。TPOT 对在线体验重要（流式输出的平滑度），生产部署可能宁可保 TPOT。
- 重要性：这是「为什么要动态路由而不是选一个静态最优」的直接证据；跳过它 PPD 就显得多余。
- 依赖内容：§4.2 配置空间与负载设计、Table 1、Table 2、§4.4、moe-serving（TPOT/SLO）。

### Q4：PPD 如何对每个请求做路由决策，为什么这样设计？

- 预期答案：打分函数 $S(\psi;\pi,\mathbf{w}) = w_{\text{ttft}}\Delta_{\text{ttft}} - w_{\text{tpot}}\Delta_{\text{tpot}}$（Eq.1），$\Delta_{\text{ttft}}$/$\Delta_{\text{tpot}}$ 是本地处理相对走 P 的 TTFT 相对改善/TPOT 相对退化，$\mathbf{w}$ 是算子给的 SLO 权重；$S>0$ 本地处理，否则走 P；Turn 1 恒 $x{=}0$。两阶段：离线在粗粒度负载网格上实测每个 cell 的 $x{=}0$/$x{=}1$ 端点指标、按 $S$ 符号存布尔决策表；在线把请求按（累积上下文长度、输入/输出比、当前 QPS）量化到最近 cell，查表返回决策，耗时 <1ms。传统 PD 是 $x\equiv 0$ 特例，极端权重可回收 $x{=}0$ 或 $x{=}1$。设计收益：P:D 比例管 Turn 1 吞吐、$\mathbf{w}$ 管 Turn 2+ 工作点，两个旋钮解耦。
- 重要性：这是论文的方法核心。
- 依赖内容：§3、§5、Alg.1、附录 B（vLLM 实现：kv_role producer/consumer、ZeroMQ、Quart 代理、MD5 会话哈希、60min TTL、10s/30s 心跳）、附录 B.1 离散化三维。

### Q5：PPD 在真实负载、慢网络和不同权重下实际拿到多少收益？

- 预期答案：真实数据集（ShareGPT、WildChat，balanced 权重）：PPD 比 $x{=}0$ 平均查询延迟低 15–25%（1P_3D ShareGPT）；$x{=}0$ 在 2P_2D/3P_1D 大量 QPS 点服务降级（成功率 <95%），PPD 全部 100%；根因是 KV 传输饱和，平均 3.1 轮/会话下 PPD 削减约 75% KV 传输（~3× 网络负载差）。对比最强静态基线 $x{=}1$（WildChat 27 测试点）：PPD TPOT 胜 12/27（$x{=}0$ 10/27、$x{=}1$ 5/27），TTFT 胜 14/27（$x{=}1$ 13/27、$x{=}0$ 0/27），成功率同 27/27——PPD 是唯一三项全竞争力的模式。带宽模拟（NVLink 150→100GbE 10 GB/s）：PD 的 Turn 2+ TTFT 143.7→170.6ms（+18.7%），PPD 恒 ~51ms，相对改善从 64% 扩大到 70%，NVLink 实验是保守下界。权重旋钮：$w_{\text{tpot}}$ 从 1→3→6，D-local 比例 95%→50%→20%，单调插值在 $x{=}1$（94–96% TTFT 降、7–12% TPOT 退化）与 $x{=}0$ 之间。整体：真实负载 Turn 2+ TTFT 平均降约 68%（abstract/§8）。
- 重要性：论文的实证分量；回答「这套设计值不值」。
- 依赖内容：§6.1–§6.4、Fig.4/Fig.5/Fig.6、Table 3、附录 C/D/E、gpu-communication（NVLink/IB/RoCE 带宽量级）。

## 3. 内容分级

核心内容（缺少则核心问题无法完整回答）：
- 单向 KV 协议及其多轮后果（Q1）
- 多轮重算与 KV 传输饱和两个低效（Q1）
- full/append-prefill 定义与复杂度对比（Q2）
- 干扰微基准三组数字（单并发、4 并发、上下文长度敏感性）（Q2）
- 配置空间、负载设计与 3060 扫描方法（Q3）
- Table 1（$x{=}0\to x{=}1$ TTFT 改善）与「P 越稀缺收益越大、负载越高收益越大」两个模式（Q3）
- 92.2% 无静态最优与 winner 分布（Q3）
- Eq.1 打分函数、双重含义的 $x$、Turn 1 恒 0（Q4）
- 两阶段算法与在线 <1ms（Q4）
- 解耦论断（P:D 比 vs 权重）（Q4）
- 真实数据集延迟与稳定性结果、75% KV 削减根因（Q5）
- vs $x{=}1$ 的 per-metric 分解（Q5）
- 带宽模拟四档与「NVLink 是保守下界」（Q5）
- 权重扫描单调控制面（Q5）

辅助内容（消除理解障碍或澄清误解）：
- KV 传输量手算：$s_{\text{kv}}=128$ KiB/token 的构成与 2K 上下文 ~256 MB（服务 Q1 量级感；128 KiB 来自附录 D，构成推导是页面构造的验证计算，需标注）
- Replica（4R）的含义与角色（Q3 的 winner 表需要）
- vLLM 实现细节：kv_role、ZeroMQ、Quart 代理、会话表、心跳（Q4 的落地形态；折叠块）
- 失败率表（Table A2）（Q5 稳定性的补充证据；折叠块）
- 混合配置排除理由（附录 A）（防止「为什么不试混合」的疑问）
- 与 AMPD/Mooncake 的关系（方法评价章）
- turn/model scaling ~70% 稳定性（Q5 泛化性，正文简述）

扩展内容（逐项决定）：
- Pareto 9-panel 附录分析（Fig.A3 之外的 Fig.A4 结构模式）——排除：Table 2 + Fig.1 已足够支撑「无静态最优」；9-panel 的面板级结构是细化不改变结论
- 失败率 30s/10s 阈值敏感性讨论——纳入一句（附录 E 末段：10s 阈值进一步压缩 $x{=}0$ 可用域、不影响 $x{=}1$/PPD），不展开
- WildChat P90 5115 token → 670MB/次的传输量数字（附录 D）——纳入（服务 Q5 带宽模拟的量级感）
- vLLM APC 的 cache_salt/多模态哈希等安全细节——排除：属于 prefix-caching 概念页范围
- CSDN 等中文二手解读——排除：不作为证据

## 4. 前置知识映射

| 前置知识 | 状态 | 链接 | 被哪些核心内容依赖 |
|---|---|---|---|
| prefill/decode 两阶段与 KV cache | 已有 | wiki/moe-serving/ | Q1、Q2 全部 |
| TTFT/TPOT/SLO/goodput | 已有 | wiki/moe-serving/ | Q1–Q5 全部指标 |
| PD 合设/分离与干扰 | 已有 | wiki/moe-serving/ | Q1、Q2、Q3 |
| 注意力 $O(n^2)$ 复杂度 | 已有 | wiki/standard-attention/ | Q2 复杂度对比 |
| GQA 与 KV heads | 已有 | wiki/mqa-gqa/ | $s_{\text{kv}}$ 构成（辅助） |
| NVLink/IB/RoCE 带宽量级 | 已有 | wiki/gpu-communication/ | Q5 带宽模拟 |
| prefix caching（block 级 KV 复用机制） | 缺失 | 按 concept 流程递归生成 wiki/prefix-caching/ | Q1（D 上 KV 可复用的前提）、Q2（append-prefill 的定义基础）、Q4（D 本地 KV 的来源） |

prefix-caching 概念页范围（在 concept 规划中细化）：block 级组织、哈希链式前缀依赖、full-block 命中粒度、LRU 逐出、多轮对话命中场景、与 PD 分离交互时「缓存跟着执行位置走」的性质。证据：vLLM 官方设计文档（docs.vllm.ai design/prefix_caching）+ vLLM APC 功能文档 + 论文 §2.3 一句定位。其自身前置（KV cache、prefill/decode）由 moe-serving 覆盖，无更深递归。

## 5. 明确不展开的内容

- DistServe/Splitwise 的 PD 分离设计细节：属于 PD 分离本身的机制，moe-serving 已覆盖；本文把 PD 当前提
- Mooncake/MemServe/LMCache/CachedAttention 的存储层机制：属于独立工作；本文只引用其定位与互补关系（§7）
- AMPD 的实时队列状态方法：并发工作，论文只有定性对比（§7），无细节可写
- chunked prefill / Dynamic SplitFuse 机制：背景一句话（§2.1），机制属于独立工作
- GQA 的原理细节：mqa-gqa 已有，本页只用其结论（8 KV heads）算 $s_{\text{kv}}$
- RDMA/IB/RoCE 协议栈细节：gpu-communication 已有，本页只用带宽数字
- 混合 R+P/D 配置为何普遍劣于纯 PD：论文自己未给理论解释（附录 A 只给现象与两条排除理由），页面照实记录，不推测
- 网格离散化阈值的具体取值：论文指向附录 B.1 的三个轴（small/large/huge 等），阈值本身论文未列数值，页面只写维度不臆造数值

## 6. 常见误解和适用边界

误解 1：「PPD 打破了 PD 的单向 KV 协议（让 KV 从 D 流回 P）」。
- 正确结论：协议不动，KV 仍只 P→D。PPD 改的是请求路由——Turn 2+ 请求根本不送 P，直接在 D 上用本地缓存做 append-prefill。附录 B 明言传输协议无需定制修改。
- 形成原因：把「消除反向重算」误读成「建立反向通道」。
- 影响：Q1、Q4。

误解 2：「append-prefill 干扰小，说明 prefill-decode 干扰问题被夸大了，PD 分离没必要」。
- 正确结论：full prefill（Turn 1、全新会话）的干扰仍然严重（batch 200 时 -48%，32K 时 3–4×）。PD 分离对 full prefill 仍然必要；PPD 是把「哪些 prefill 必须隔离」精细化，不是否定分离。
- 形成原因：把「一类 prefill 干扰小」推广到「所有 prefill 干扰小」。
- 影响：Q2、Q3。

误解 3：「$x{=}1$ 全面占优，直接全本地就行，不需要 PPD」。
- 正确结论：TPOT 维度 $x{=}0$ 胜率 38.3% vs $x{=}1$ 15.6%；92.2% 组合无静态最优；真实负载 PPD TPOT 胜 12/27 vs $x{=}1$ 5/27。$x{=}1$ 在 TPOT 上付 7–12% 代价（权重案例的 $x{=}1$ 极限端）。
- 形成原因：只看 TTFT 单指标。
- 影响：Q3、Q4、Q5。

误解 4：「~68% TTFT 降低是无条件结论」。
- 正确结论：成立条件——多轮负载（Turn 2+）、4×H100 NVLink 或更慢互连、Llama-3.1-8B 为主（8B/14B/30B 验证过 ~70% 稳定）、balanced 权重。合成扫描范围是 48–73%，随 P:D 配比与 QPS 变化（1P_3D 高 QPS 最大 -73.3%，3P_1D 高 QPS 最小 -24.9%）。单轮为主的负载无此收益（Turn 1 恒走 P）。
- 形成原因：把摘要数字当普适常数。
- 影响：Q5。

误解 5：「PPD 是一个在线优化器/SLO 控制器」。
- 正确结论：决策离线预计算，在线只查表（<1ms）；PPD 不强制端到端 SLO 界——那是闭环准入控制+批调度的事，PPD 只暴露权重旋钮给上层控制器（§5.5 原文称 routing actuator）。
- 形成原因：把「动态路由」误解为「在线求解优化问题」。
- 影响：Q4。

误解 6：「重算占多轮 prefill 成本 99% 是本文的测量」。
- 正确结论：这是 CachedAttention（gao2024）在真实聊天负载上的测量结论，本文引用（§1）。标注为文献已有结论。
- 形成：没区分本文实测与引用。
- 影响：Q1。

适用边界：
- 方法解决：多轮对话/agentic 负载下 Turn 2+ 的 TTFT 与 KV 传输拥塞；不解决：Turn 1 效率、单轮负载收益（Turn 1 恒走 P）、端到端 SLO 硬约束
- 成立条件：D 节点有 prefix cache 且会话粘在同一个 D 上（实现细节：会话哈希路由）；模型单卡放得下（replica 对照需要；主实验 8B 的原因之一，附录 C.3）
- 条件不满足时：负载/硬件漂移远离婚线校准集时离线表优雅降级但次优（§7）；TPOT 极敏感部署应回退 $x{=}0$（极端权重）
- 实验未覆盖：真实多节点 RDMA 部署（仅带宽注入模拟）；更大模型的全量扫描（初始化开销 prohibitive，附录 C.3）；多个 session 竞争 D 节点 KV 槽位时的缓存抖动（论文未讨论）

## 7. 论断分级

- 论文明确声称：C1–C16 全部（见 evidence.md，逐条附原文定位）
- 文献已有结论：多轮重算占 prefill 成本 99%（gao2024/CachedAttention，§1 引用）；PD 干扰现象（zhong2024 DistServe 等，§2.1 引用）；互连慢 2–8×（patel2024/qin2025，§6.2 引用）
- 基于证据的推断：「92.2% 无静态最优 ⇒ 逐请求动态决策是必要的」是论文自己的论证链（§4.4→§5），页面按论文声称呈现；「单轮为主的负载收益小」由「Turn 1 恒 x=0 + 收益全部来自 Turn 2+」直接推出，页面标注为推断
- 缺失假设的猜测：无（页面不写）。特别注意：$s_{\text{kv}}=128$ KiB 的「构成推导」（8 heads × 128 head_dim × 2(KV) × 2 bytes × 32 layers）论文未展示，页面作为构造的验证计算呈现并标注；Llama-3.1-8B 的 head_dim=128 与 32 layers 是公开模型配置，可作为外部事实引入并注明来源（Hugging Face 模型卡）
