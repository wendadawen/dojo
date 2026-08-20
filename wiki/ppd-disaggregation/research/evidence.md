# 核心论断与证据

固定版本：arXiv:2603.13358v2 TeX 源码（2026-05-05）。定位缩写：§=正文章节，App.X=附录，Eq.X=公式，Tab.X=表，Fig.X=图。

## C 论断（机制与结论）

- C1：PD 的 KV 传输严格单向：P 是生产者、D 是消费者，无 D→P 反向通道；所有主流生产引擎保持该契约。｜定位：§2.2 One-way KV Transfer Protocol（"P nodes act as KV producers and D nodes as consumers, with no reverse channel from D back to P. This producer/consumer contract is preserved across all major production engines"）｜条件：canonical PD 设计｜已确认
- C2：单向协议的直接后果：D 上算出的 KV（含全部已解码响应）对 P 不可达，因此每个新 turn 都要重走完整 P→D 流水线。｜定位：§2.2 末句（"any KV state computed on D, including all decoded responses, is unreachable from P, so each new turn re-traverses the full P→D pipeline"）｜条件：无外挂 KV 层时｜已确认
- C3：多轮场景下 P 必须重算整段对话历史（此前响应 + 新 prompt）的 KV 再传回 D。｜定位：§1（"the prefill node must therefore recompute the KV cache for the entire conversation history (prior responses plus the new prompt) before transferring it back to D"）、§2.3 首段｜条件：传统 PD（$x{=}0$）｜已确认
- C4：重算占多轮 prefill 成本高达 99%（文献已有结论，gao2024 CachedAttention 测量）。｜定位：§1（"Recent measurements on real chat workloads find that this recomputation accounts for up to 99% of multi-turn prefill cost"）+ 引文 [gao2024]｜条件：真实聊天负载、up to（上界表述）｜已确认（作为引用事实）
- C5：full prefill 处理无缓存上下文的新 prompt，$n$ 个 token 的 attention 复杂度 $O(n^2)$；append-prefill 只对 $m$ 个新 token 计算（每个关注 $n+m$ 个 key），复杂度 $O(m(n+m))$；$m\ll n$ 时比同长度 full prefill 便宜约 $n/m$ 倍。｜定位：§4.1 Full Prefill vs. Append Prefill 段｜条件：$m$ 个新 token 接在 $n$ 个已缓存 token 后｜已确认
- C6：Turn 1 恒走 PD 路径（$x{=}0$），因为无缓存 KV。｜定位：§3（"Turn 1 always uses the PD path since no cached KV exists"）、Alg.1 第 10 行｜已确认
- C7：路由参数 $x$ 双重含义：硬件级分数 $x\in[0,1]$（AP 路由到 D 的比例，跨请求统一）与每请求二元决策 $x\in\{0,1\}$。｜定位：§3 开头｜已确认
- C8：传统 PD 是 $x\equiv 0$ 特例；PPD 极端权重下可回收 $x{=}0$ 或 $x{=}1$。｜定位：§3（"traditional PD is the special case $x\equiv 0$"）、§5.3（"PPD recovers $x{=}0$ or $x{=}1$ as special cases at extreme settings"）｜已确认
- C9：吞吐不进每请求目标函数，是 KV 传输量下降的系统级副产物。｜定位：§3 末句（"Throughput is a system-level emergent metric and is not optimized per-request, but it improves as KV transfer volume drops"）｜已确认
- C10：PPD 两阶段：离线在负载网格上实测每 cell 的 $x{=}0$/$x{=}1$ 端点 TTFT/TPOT、按 $S$ 符号存布尔决策；在线按三维（累积上下文长度、输入/输出比、系统 QPS）量化到最近 cell 查表，<1ms 返回。｜定位：§5（"Offline, it builds a lookup table...directly measuring TTFT and TPOT at the two extremes $x{=}0$ and $x{=}1$ for each cell, then storing the sign-of-score decision. Online, each Turn 2+ request is mapped to the nearest cell along three axes (accumulated context length, input/output ratio, system QPS) and the precomputed decision is returned in <1 ms"）、Alg.1｜条件：网格覆盖负载分布｜已确认
- C11：在线请求特征元组 $\psi=(t,n_{\text{in}},n_{\text{out}},n_{\text{ctx}},q)$（轮次、新输入 token 数、预期输出 token 数、累积上下文、当前 QPS）；三维离散化：context class（small ≤512 / large 512–4096 / huge >4096）、workload type（输入输出比九类）、QPS bin。｜定位：App.B.1｜已确认
- C12：P:D 比例管 Turn 1 吞吐，权重 $\mathbf{w}$ 选 Turn 2+ 在 Pareto 前沿上的工作点，两旋钮解耦；传统 PD 里同一个比例被迫同时承担两个目标。｜定位：§5 Decoupling 段｜已确认
- C13：原型基于 vLLM disaggregated serving：P 用 kv_role=kv_producer、D 用 kv_role=kv_consumer（收到的 KV 存入本地 prefix cache 供 Turn 2+ 使用）、ZeroMQ 传输、标准协议无需定制修改；路由代理用 Quart 实现，会话表以首条用户消息的 MD5 哈希为键，60 分钟失活逐出（对齐 vLLM prefix cache 默认 TTL），后端每 10 秒 ZeroMQ 心跳、30 秒无心跳移除。｜定位：§5 末段、App.B.2–B.5｜已确认
- C14：$x{=}1$ 优势随负载增大（传输排队增长、本地缓存访问价值上升）；P 越稀缺本地处理收益越大（P 成瓶颈时 $x{=}1$ 完全绕过它）。｜定位：§4.3（"the $x{=}1$ advantage increases with load"、"the fewer prefill nodes available, the greater the benefit of local processing on D"）、Tab.1｜条件：高 QPS 下传输排队显著｜已确认
- C15：真实负载下 $x{=}0$ 降级的根因是 KV 传输饱和（非显存耗尽或硬件故障）；平均 3.1 轮/会话下 PPD 把 KV 传输量削减约 75%，形成 ~3× 网络负载差。｜定位：§6.2 Root Cause 段（"cutting KV transfer load by ~75% at the observed average of 3.1 turns per conversation. At 3.1 average turns per conversation, this creates a ~3× difference in network load"）｜条件：balanced 权重、3.1 轮均值｜已确认
- C16：PPD 与分布式 KV 层是不同层且可组合：后者决定前缀状态存在集群何处，前者决定缓存命中/未命中请求如何调度避免互相干扰。｜定位：§7 Relationship with Distributed KV Cache 段｜已确认
- C17：PPD 的主要局限：离线表在硬件或负载分布远离婚线校准集时优雅降级但次优；AMPD 的在线队列估计是补齐该缺口的自然机制，两者可组合。｜定位：§7 Relationship with Concurrent Work 段｜已确认
- C18：PPD 不强制端到端 SLO 界（那需要闭环准入控制与批调度），它暴露一个可预测的权重旋钮，上层 SLO 控制器可由观测到的 P99 指标驱动。｜定位：§5.5 末句（"PPD is a routing actuator: it does not enforce end-to-end SLO bounds...but exposes a predictable knob that higher-level SLO controllers can drive from observed P99 metrics"）｜已确认
- C19：混合 R+P/D 配置被排除出主分析的两个理由：(a) R 节点无分离隔离收益（R 上 prefill 干扰自身 decode，路由参数 $x$ 对 R 不适用，与 AP 路由正交）；(b) 混合配置挤占专用 P/D 资源。混合配置胜率仅 6.1%（TTFT）/12.2%（TPOT）/16.1%（吞吐）。｜定位：App.A（"hybrid configurations rarely achieved the best performance on any metric, winning only 6.1% of TTFT, 12.2% of TPOT, and 16.1% of throughput test points"）｜条件：4 GPU 预算内对比｜已确认
- C20：$x{=}1$ 的 Turn 2+ TTFT 优势随轮数与模型尺寸泛化：2–16 轮与 8B/14B/30B 上维持 ~70% 改善，说明收益来自架构性质而非模型特性。｜定位：§4.3 Validation 段、App.C.3（"the relative improvement remains stable at ~70%"）｜条件：合成负载、NVLink｜已确认
- C21：带宽模拟方法：对每个 PD 路由请求在 vLLM P2P NCCL connector 的 decode 侧接收路径注入校准延迟，PPD 本地路径不受影响；注入延迟按 Eq.2 计算，扣除 NVLink 实际传输时间避免重复计数。｜定位：§6.2 开头、App.D｜条件：模拟（非真实多节点）｜已确认
- C22：E2E 相似但 per-metric 不同的解释：balanced 权重下 PPD 把部分请求路由回 P，用少量 TTFT 换 TPOT 改善，两者在复合 E2E 指标中近似抵消，所以 $x{=}1$ 与 PPD 的 E2E 曲线视觉重合。｜定位：§6.3 开头、App.E（"these two effects nearly cancel in the composite E2E metric"）｜已确认

## F 公式

- F1（Eq.1）：$S(\psi;\pi,\mathbf{w}) = w_{\text{ttft}}\Delta_{\text{ttft}} - w_{\text{tpot}}\Delta_{\text{tpot}}$。$\Delta_{\text{ttft}}$：本地处理相对走 P 的 TTFT 相对改善；$\Delta_{\text{tpot}}$：相对 TPOT 退化；$\mathbf{w}=(w_{\text{ttft}},w_{\text{tpot}})$ 为算子权重。决策：Turn 2+ 且 $S>0$ → 本地（$x{=}1$），否则走 P（$x{=}0$）。｜定位：§3 Eq.(1) 及其后段落｜已确认
- F2（§4.1 文中）：full prefill 复杂度 $O(n^2)$（$n$ 个输入 token）；append-prefill 复杂度 $O(m(n+m))$（$m$ 个新 token 接在 $n$ 个缓存 token 后，每个关注 $n+m$ 个 key）；$m\ll n$ 时约便宜 $n/m$ 倍。｜定位：§4.1、§2.1（$O(n^2)$ 同时在 §2.1 出现并引 FlashAttention 只降内存复杂度）｜已确认
- F3（Eq.2）：$\Delta t = \max(0, \tfrac{B(\psi)}{\beta_{\text{target}}} - t_{\text{NVLink}})$，$B(\psi)=n_{\text{tokens}}\cdot s_{\text{kv}}$。带宽模拟注入延迟。｜定位：App.D Eq.(2)｜条件：模拟慢互连｜已确认
- F4（页面构造的验证计算，非论文公式）：$s_{\text{kv}}=128$ KiB/token 的构成 = 2 bytes（BF16）× 8 KV heads × 128 head_dim × 2（K 与 V）× 32 layers = 131072 bytes = 128 KiB。论文只给结果数字与条件（"For Llama-3.1-8B (BF16, GQA with 8 KV heads, 32 layers) the per-token footprint is $s_{\text{kv}}=128$ KiB"，App.D）；8 KV heads 与 32 layers 来自论文同一句，head_dim=128 来自 Hugging Face Llama-3.1-8B 模型配置（head_dim 字段，外部事实）。页面必须标注此为构造验证计算。｜已确认（结果与论文一致：$2048\times 128\,\text{KiB}=256\,\text{MiB}$ 对上 §2.2 的 ~256 MB）
- F5（页面构造的换算，非论文公式）：2K token 上下文单次 KV 传输 $\approx 2048 \times 128\,\text{KiB} = 256\,\text{MiB}$，对应 §2.2 "$\sim$256 MB"。｜已确认（与论文数字吻合）

## N 数字

- N1：干扰微基准（单 H100、Llama-3.1-8B、co-locate 一个处理 1024 token 的 prefill）：batch size 200 时 full prefill 使 decode TPOT 减速 ~48%，append-prefill 仅 ~2%，差一个数量级。｜定位：§1 贡献 ii、§4.1 Interference Measurement 段、Fig.2｜条件：1 并发 prefill、1024 token、batch 200｜已确认
- N2：4 并发 prefill 时 full +57% vs append +21%（batch 200）。｜定位：§4.1（"With 4 concurrent prefills, full reaches +57% while append stays at +21%"）、Fig.A1｜已确认
- N3：上下文长度敏感性：full prefill 干扰在 32K token 时增长到 3–4×；append-prefill 在 64K 时仍 <25%。｜定位：§4.1 末段、Fig.A2（App.C.1）｜已确认
- N4：配置扫描规模：17 配置 × 18 合成负载 × 10 QPS = 3060 数据点；每点固定 10 秒、泊松到达。｜定位：§4.2 Experimental Setup｜条件：4×H100 80GB HBM3 NVLink 单节点｜已确认
- N5（Tab.1 全表，$x{=}0\to x{=}1$ 的 Turn 2 TTFT 改善，负值=改善）：1P_3D：低 QPS(0.5–2) -57.8%、中(4–8) -65.2%、高(12–20) -73.3%；2P_2D：-47.7%/-51.6%/-56.2%；3P_1D：-44.3%/-38.1%/-24.9%。合成扫描总范围 48–73%。｜定位：Tab.1、§1 贡献 v｜条件：合成负载、NVLink｜已确认
- N6：92.2% 的（负载, QPS）组合中 Turn 2 TTFT 最优配置与 Avg TPOT 最优配置不同。｜定位：§4.4 Core Finding｜条件：3060 点扫描内（注：百分比基数是 18 负载 × 10 QPS = 180 组合）｜已确认
- N7（Tab.2 winner 分布，各配置类别在三目标上的胜场百分比）：Replica(4R)：TTFT 63.3%/TPOT 0.6%/吞吐 0%/平均 21.3%；x=0：0%/38.3%/4.4%/14.2%；0<x<1：3.3%/33.3%/27.8%/21.5%；x=1：27.2%/15.6%/38.3%/平均 27.0%（最高）。列和不为 100% 因混合配置（极少胜）被排除。｜定位：Tab.2、§4.4｜条件：合成扫描｜已确认
- N8：真实负载延迟：1P_3D ShareGPT 上 PPD 平均查询延迟比 $x{=}0$ 低 15–25%（全 QPS 范围）。｜定位：§6.2 Finding 1｜条件：balanced 权重、两数据集全部配置 PPD 曲线低于基线｜已确认
- N9：稳定性：$x{=}0$ 在 2P_2D 与 3P_1D 于两数据集多数 QPS 点服务降级（成功率 <95%），仅 1P_3D 稳定；PPD 全部 QPS 与数据点 100% 成功率。｜定位：§6.2 Finding 2｜条件：成功 = ≥95% 请求不超时；降级主因请求排队与超时（30s 阈值，App.E）｜已确认
- N10（Tab.3 per-metric 胜场，WildChat 3 配置 × 9 QPS = 27 点）：TPOT 最优：$x{=}0$ 10/27、$x{=}1$ 5/27、PPD 12/27；TTFT 最优：$x{=}0$ 0/27、$x{=}1$ 13/27、PPD 14/27；成功率 100%：$x{=}0$ 4/27、$x{=}1$ 27/27、PPD 27/27。｜定位：Tab.3、§6.3｜条件：WildChat、balanced 权重｜已确认
- N11（带宽模拟，1P_3D、QPS=1、WildChat 500 会话，四档互连）：NVLink ~150 GB/s / IB NDR 50 / IB HDR 25 / 100GbE 10 GB/s。PD 的 Turn 2+ TTFT：143.7→170.6 ms（+18.7%）；PPD 恒 ~51 ms。相对 TTFT 改善从 64% 扩大到 70%。PD TPOT 8.07→8.38 ms（+3.8%），PPD ~8.1 ms。E2E：PD +4.7%（3028→3169 ms），PPD +0.9%（仅 Turn 1 贡献）。｜定位：Fig.5 caption、§6.2 末段｜条件：带宽注入模拟｜已确认
- N12（权重扫描，1P_3D、prefill-heavy 500 会话、QPS 8 与 16）：$w_{\text{tpot}}$=1（balanced）→95% D-local；=3→50%；=6→20%；$x{=}0$ 端 0%。$x{=}1$ 极限：94–96% TTFT 降低、7–12% TPOT 退化。｜定位：Fig.6 caption、§6.4｜条件：prefill-heavy 切片｜已确认
- N13：PPD 使 Turn 2+ TTFT 平均降低 ~68%（真实负载）。｜定位：abstract（"reduces Turn 2+ time-to-first-token (TTFT) by ∼68%"）、§8（"cutting Turn 2+ TTFT by an average of ∼68% on real-world workloads while keeping TPOT competitive and adding <1 ms of per-request overhead"）｜条件：多轮真实负载、NVLink、balanced 权重｜已确认
- N14：Llama-3.1-8B 2K token 上下文单次 KV 传输 ~256 MB。｜定位：§2.2｜已确认
- N15：$s_{\text{kv}}=128$ KiB/token（Llama-3.1-8B BF16、GQA 8 KV heads、32 layers）；WildChat P90 上下文 5115 token 时每个 Turn 2+ 传输 ~670 MB：NVLink ~4.5 ms、IB HDR ~27 ms、100GbE ~67 ms。｜定位：App.D｜已确认
- N16（Tab.A2 高 QPS 失败率）：3P_1D x=0：QPS8 11%/12 44%/16 61%/20 89%；3P_1D x=1：6%/22%/44%/67%；2P_2D x=0：0%/6%/11%/22%；2P_2D x=1：0%/0%/6%/11%；4R 全 0%。超时阈值 30s。｜定位：App.C.4｜条件：合成负载｜已确认
- N17：$x{=}1$ 优势泛化：2–16 轮与 8B/14B/30B 模型上 Turn 2+ TTFT 改善稳定 ~70%。｜定位：App.C.3、§4.3｜已确认
- N18：QPS 档位 10 档（0.5/1/2/4/6/8/10/12/16/20）；18 合成负载 = 2 个 Turn 1 设置 × 9 个 Turn 2 设置（decode-heavy 4 个如 $n_{\text{in}}{=}32,n_{\text{out}}{=}512$、balanced 2 个、prefill-heavy 3 个如 $n_{\text{in}}{=}1024,n_{\text{out}}{=}32$）；模型 Llama-3.1-8B 为主，Qwen2.5-14B 与 Qwen3-30B-A3B 验证。｜定位：§4.2｜已确认
- N19：真实负载平均 3.1 轮/会话。｜定位：§6.2｜已确认
- N20：在线查表开销 <1 ms/请求。｜定位：§5、§8｜已确认
- N21：timeout 阈值敏感性：30s→10s 进一步压缩 $x{=}0$ 的可用域，$x{=}1$ 与 PPD 全 QPS 范围不变（本地缓存路径把 Turn 2+ E2E 延迟压在两个阈值之下，即使 QPS=14）。｜定位：App.E 末段｜条件：WildChat 三方对比｜已确认

## 原图候选（获取途径：TeX 源码包 /tmp/ppd-research/src/figures/，最高优先级）

- G1：Fig.2 interference_tpot.png——full vs append prefill 的 decode TPOT 干扰对比（N1 的图证据）。说明 Q2 的核心数量级差。
- G2：Fig.3 ppd.pdf（需转 PNG）——Replica / PD / PPD 三种架构的多轮数据流概念图。说明 Q1 与 Q4 的机制对比。
- G3：Fig.4 real_validation_e2e.png——ShareGPT/WildChat 上平均查询延迟 vs QPS，虚线 PD、实线 PPD，×为降级点。说明 N8/N9。
- G4：Fig.5 scaling_simulation.png——四档互连下 TTFT/TPOT/E2E 三面板。说明 N11。
- G5：Fig.6 weight_tradeoff_curve.png——$w_{\text{tpot}}$ 扫描的 TTFT-TPOT 前沿。说明 N12。
- G6：Fig.1 fig1_pareto.png——P99 TTFT vs TPS Pareto 前沿总览。说明 Q3 的「无静态最优+PPD 选点」。
- 备选（暂不纳入正文，需要时取）：Fig.A1（4 并发）、Fig.A2（上下文敏感性）、Fig.A4（9-panel Pareto）、Fig.A5（三方 E2E）、Fig.A6（turn/model scaling）。

## 冲突与存疑处理

- 无版本间冲突（以 v2 为准，v1 差异不影响本页论断）。
- "256 MB" vs 计算 256 MiB：论文用 MB 表述，页面按原文记录 ~256 MB 并在验证计算中注明 MiB 换算，不视为冲突。
- $x{=}1$ 在 Fig.A5 caption 中记作 "pD"：同一模式的两种记号（pD = prefill-capable Decode），术语表登记，正文统一用「$x{=}1$（Full AP-to-D）」，首次出现时注明别名。
- 92.2% 的基数：正文未明说分母；由扫描设计（18 负载 × 10 QPS = 180 组合）可推出分母应为 180（17 配置在每组合内竞争），页面按"（负载, QPS）组合"转述论文原话，不擅自补分母数字，只写明扫描总量 3060 点。
