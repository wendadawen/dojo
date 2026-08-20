# 术语表

全文术语与符号统一以此为准。首次出现位置 = 章节号。

## 基础推理术语

| 术语 | 首次出现 | 定义 |
|---|---|---|
| prefill（预填充） | 第 1 章 | 并行处理输入 prompt、产生第一个输出 token 并填充 KV cache 的阶段，计算密集 |
| decode（解码） | 第 1 章 | 自回归逐 token 生成阶段，每 token 都要加载全部模型权重，访存带宽受限 |
| KV cache | 第 1 章 | 此前 token 的 key/value 状态缓存，随上下文线性增长（详见 moe-serving） |
| full prefill（全量预填充） | 第 2 章 | 处理无缓存上下文的新 prompt，$n$ 个 token，attention 复杂度 $O(n^2)$；Turn 1 场景 |
| append-prefill（AP，追加预填充） | 第 2 章 | 只对 $m$ 个新 token 计算 attention（每个关注 $n+m$ 个 key），复用已缓存 KV；复杂度 $O(m(n+m))$ |
| prefix caching（前缀缓存） | 第 1 章 | 以 block 为单位缓存已处理请求的 KV，新请求共享相同前缀时直接复用（详见 prefix-caching 概念页） |
| TTFT | 第 1 章 | time-to-first-token，首 token 延迟 |
| TPOT | 第 1 章 | time-per-output-token，平均每输出 token 延迟，流式体验的关键指标 |
| TPS | 第 3 章 | tokens-per-second，系统吞吐 |
| SLO | 第 3 章 | Service Level Objective，服务等级目标 |
| goodput | 不使用 | （moe-serving 页概念，本页不引入） |

## 架构与节点

| 术语 | 首次出现 | 定义 |
|---|---|---|
| PD 分离 | 第 1 章 | prefill 与 decode 物理分置不同 GPU 池（详见 moe-serving） |
| P 节点 | 第 1 章 | prefill-only 节点，KV 生产者 |
| D 节点 | 第 1 章 | decode 节点，KV 消费者 |
| R 节点（Replica） | 第 3 章 | 复制节点，本地同时做 prefill 与 decode，等价于不分离的完整实例 |
| 单向 KV 协议 | 第 1 章 | KV 传输严格 P→D：P 生产、D 消费、无反向通道；主流引擎共有 |
| 1P_3D / 2P_2D / 3P_1D / 4R | 第 3 章 | 4 GPU 预算下的 P:D 配比记号（xP = x 个 prefill 节点，类推） |
| 混合配置 | 第 3 章 | R 与 P/D 混布的配置（如 1R_1P_2D），被主分析排除 |
| 服务降级 | 第 5 章 | 成功率 <95%（请求超时为主因）；成功 = ≥95% 请求不超时完成 |

## 路由与决策

| 符号/术语 | 首次出现 | 定义 |
|---|---|---|
| $x$（硬件级分数） | 第 3 章 | $x\in[0,1]$，AP 操作路由到 D 的比例（跨请求统一；静态策略用） |
| $x$（请求级决策） | 第 4 章 | $x\in\{0,1\}$，单个请求走 P（0）或 D 本地（1） |
| $x{=}0$（传统 PD） | 第 1 章 | 所有 Turn 2+ 送 P，每轮 KV 传输；传统 PD 本身 |
| $x{=}1$（Full AP-to-D，又称 pD / D-local） | 第 2 章 | 全部 Turn 2+ 在 D 本地 append-prefill，仅 Turn 1 传 KV |
| $0{<}x{<}1$（部分路由） | 第 3 章 | 固定比例（1/3、1/2、2/3）路由到 D 的静态策略 |
| Turn 1 / Turn 2+ | 第 1 章 | 会话首轮 / 第二轮及以后 |
| $\psi$ | 第 4 章 | 请求工作负载特征元组 $(t, n_{\text{in}}, n_{\text{out}}, n_{\text{ctx}}, q)$：轮次、新输入 token 数、预期输出 token 数、累积上下文长度、当前 QPS |
| $\pi$ | 第 4 章 | 初始节点分配（P:D 配比） |
| $\mathbf{w}=(w_{\text{ttft}},w_{\text{tpot}})$ | 第 4 章 | 算子指定的 SLO 权重 |
| $S(\psi;\pi,\mathbf{w})$ | 第 4 章 | 打分函数（Eq.1）：本地处理相对走 P 的加权收益 |
| $\Delta_{\text{ttft}}$ / $\Delta_{\text{tpot}}$ | 第 4 章 | 本地处理相对走 P 的 TTFT 相对改善 / TPOT 相对退化 |
| 离线建表 | 第 4 章 | Phase 1：负载网格逐 cell 实测 $x{=}0$/$x{=}1$ 端点指标，按 $S$ 符号存布尔决策 |
| 在线查表 | 第 4 章 | Phase 2：请求量化到最近 cell 取预存决策，<1ms |
| context class / workload type / QPS bin | 第 4 章（折叠） | 三维离散化轴：累积上下文长度（small ≤512 / large 512–4096 / huge >4096）、输入输出比九类、QPS 档位 |
| routing actuator（路由执行器） | 第 4 章 | PPD 的定位：不强制端到端 SLO 界，只暴露权重旋钮 |

## 实验与实现

| 术语 | 首次出现 | 定义 |
|---|---|---|
| QPS | 第 3 章 | queries per second，请求到达率；10 档 0.5–20 |
| 泊松到达 | 第 3 章 | 会话按 Poisson 过程到达目标 QPS |
| 18 合成负载 | 第 3 章 | 2 个 Turn 1 设置 × 9 个 Turn 2 设置（decode-heavy 4 / balanced 2 / prefill-heavy 3） |
| 3060 数据点 | 第 3 章 | 17 配置 × 18 负载 × 10 QPS 的完整扫描 |
| winner 分布 | 第 3 章 | 各配置类别在（负载, QPS）组合上按目标取最优的胜场百分比 |
| Pareto 前沿 | 第 3 章 | P99 TTFT vs TPS 的权衡边界，左上为理想 |
| ShareGPT / WildChat | 第 5 章 | 两个公开多轮对话数据集 |
| 带宽模拟（延迟注入） | 第 5 章 | 对 PD 路由请求注入 $\Delta t=\max(0,B(\psi)/\beta_{\text{target}}-t_{\text{NVLink}})$ 的校准延迟，模拟慢互连；PPD 本地路径不注入 |
| $s_{\text{kv}}$ | 第 1 章 | 每 token KV 足迹；Llama-3.1-8B BF16（GQA 8 KV heads、32 layers）为 128 KiB |
| $B(\psi)$ | 第 5 章 | 请求 KV 传输量 $n_{\text{tokens}}\cdot s_{\text{kv}}$ |
| $\beta_{\text{target}}$ / $t_{\text{NVLink}}$ | 第 5 章 | 模拟目标带宽 / NVLink 实际传输时间（扣除以避免重复计数） |
| NVLink / IB NDR / IB HDR / 100GbE | 第 5 章 | 四档互连：~150 / 50 / 25 / 10 GB/s 有效带宽 |
| kv_role=kv_producer / kv_consumer | 第 4 章（折叠） | vLLM 分离部署角色：P 生成并发送 KV；D 接收并存入本地 prefix cache |
| 会话哈希 | 第 4 章（折叠） | 以首条用户消息 MD5 为键的会话路由表；60 分钟失活逐出 |
| 心跳 | 第 4 章（折叠） | 后端 ZeroMQ 每 10 秒心跳，30 秒无心跳移除 |

## 相关工作（一句话定位，不展开）

| 术语 | 首次出现 | 定位 |
|---|---|---|
| AMPD | 第 6 章 | 并发工作：同样把增量 prefill 路由到 D，但用实时队列状态估计而非离线建表 |
| Mooncake / MemServe / LMCache / CachedAttention | 第 6 章 | 外挂分布式/分层 KV 存储层路线，决定「前缀状态存哪」；与 PPD 的调度层互补 |
| DistServe / Splitwise | 第 1 章 | PD 分离基础工作（背景，不展开） |
| chunked prefill / Dynamic SplitFuse | 第 1 章 | 同卡干扰缓解路线：切小块交错执行；缓解但不消除干扰 |
| DuetServe / Nexus / TaiChi | 第 6 章 | GPU 内 SM 切分/动态资源再分配路线 |

## 写法约定

- 「$x{=}1$（Full AP-to-D）」为全页统一称呼；Fig.A5 的「pD」与「D-local capable」在首次出现处注明是别名。
- TTFT/TPOT/QPS/KV cache 等通用缩写首次出现给全称后直接用缩写。
- 数字一律带条件（QPS 档位、配置、数据集、权重）。
- 百分比改善方向：负值/「降低」均按论文原义转述为自然语言，避免符号歧义。
