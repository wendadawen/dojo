# 核心论断与证据：PP 负载均衡

来源缩写：
- TP = TeraPipe 论文（arXiv:2102.07988v2，ICML 2021，Zhuohan Li et al.）
- MC = Mooncake 论文（arXiv:2407.00079，Qin et al.）
- SG = SGLang 官方博客《Pipeline Parallelism in SGLang》（lmsys.org，2026-01-15）
- GL = gLLM 论文（arXiv:2504.14775v2，Guo et al.，中山大学）
- YN = 腾讯一念演讲实录（InfoQ《腾讯一念 LLM 分布式推理优化实践》）

核心论断编号只覆盖核心内容；辅助材料在文末单列。所有条目置信状态默认「已确认」，例外单独标注。

## C 论断

| 编号 | 论断 | 来源定位 | 适用条件 |
|---|---|---|---|
| C1 | PP 推理中 GPU 空闲来自两类依赖：inter-stage（后级必须等前级完成）与 inter-batch（并发 micro-batch 数受流水线深度限制） | GL §2.4（Figure 3 及正文二分定义） | 同步流水线执行 |
| C2 | 流水线负载失衡分两类：inter-stage（stage 间计算分布不均）与 inter-batch（不同 micro-batch 计算量不同）；gLLM 只解决后者，inter-stage 留作未来工作 | GL §2.4 | gLLM 的研究范围声明 |
| C3 | 超长 prompt 以整体批次进入流水线时，下游 GPU 长时间空闲，形成大气泡；同时单次前向要为整个序列保存/传输中间 hidden state，峰值内存高（memory wall） | SG「The Pipeline Bubble」「The Memory Wall」两节 | 128K–1M token 级输入、未切分时 |
| C4 | 固定 chunk 大小下，self-attention 的增量计算特性使同尺寸 chunk 的处理时间随前缀长度增长非线性增加；时序错配沿流水线传播、在更高 PP rank 上复合放大，实际气泡比率超过理论值 | SG「Dynamic Chunking」一节 | chunked prefill + PP，长前缀 |
| C5 | decoder-only 模型中位置 $t$ 的计算只依赖 $\le t$ 的 token（自回归性质），因此可以在单条序列内部做流水线：当前 token 在当前层与之前 token 在下一层并行计算 | TP §1 关键观察、§3.2 | 单向自回归（causal）注意力 |
| C6 | 序列中越靠后的位置注意力计算量越大；流水线总时延由最慢 stage 决定，因此最优切分应前长后短；均分序列导致后段片段更慢、气泡更大 | TP §3.1 公式(2)、§3.2 末段与 Figure 4 | 同 C5 |
| C7 | Mooncake 在 prefill 池实现 CPP：每 $X$ 个节点组成流水线组，请求输入切成不超过 `prefill_chunk` 的 chunk，同一请求的不同 chunk 同时由不同节点处理以降低 TTFT；跨节点通信只发生在 stage 边界、可与计算重叠；对短上下文无显著开销，无需动态调整节点分组 | MC §5.1「Multi-node Prefill」、§3 工作流第 2 步 | 长上下文 prefill；chunk 阈值通常大于 1000 token |
| C8 | Mooncake 选 CPP 而非跨节点 TP/SP 的动机：跨节点 TP 每层需两次基于 RDMA 的 all-reduce，显著降低 prefill 节点 MFU；SP 仍有频繁跨节点通信且静态分组导致集群利用率低、需频繁弹性伸缩 | MC §5.1 动机段 | 多节点 prefill 集群 |
| C9 | Sarathi-Serve 用固定 token budget 混排 chunked prefill 与 decode：波动来自两方面——错过 decode 与 prefill 混批的机会、decode token 在 batch 间分布不均；缩小预算理论上可平滑波动，但会不成比例地惩罚 prefill 速率、限制总吞吐 | GL Abstract、Introduction（Figure 1 讨论）、§2.5 | PP serving、chunked prefill + decode 混排 |
| C10 | gLLM decode 侧调度：把运行中 decode token 总数 $\#RD$ 均摊到全部 micro-batch，$\#D=\#RD/\#PP_{\text{depth}}$；剩余不足 $\#D$ 时全排，否则恰好排 $\#D$ | GL §3.2.1 公式(4) | decode 变化相对平缓的假设 |
| C11 | gLLM prefill 侧调度：合并公式 $\#P=\max(\min(\#WP/\#T,\ \#MaxP\times\frac{KV_{\text{free}}-KV_{\text{thresh}}}{1-KV_{\text{thresh}}}),\ \#MinP)$；$\#WP$ 为等待 prefill 的 token 数（分摊到 $\#T$ 次迭代）、$KV_{\text{free}}$ 为 KV cache 空闲率；$KV_{\text{free}}<KV_{\text{thresh}}$ 时暂停 prefill，防止 KV cache 溢出导致 decode 请求被过早抢占而产生重算 | GL §3.1.1–§3.1.3 公式(1)(2)(3)、§3.1.3 溢出保护 | gLLM 超参 #T=8、#MaxP=2048、#MinP=32、KV_thresh=0.05（实验取值） |
| C12 | TeraPipe 支持 batch 维与 token 维联合切分：对每个候选 batch size $b$ 跑 token 维 DP 得最优 $T_b$ 与切分方案，再决定 batch 维切分 $b_1+\dots+b_D=B$ 使 $\sum T_{b_i}$ 最小，该子问题归约为一维背包问题 | TP §3.4 | 训练场景推导；推理侧作为两维度可组合的依据 |
| C13 | batch 足够大时，联合 DP 的最优解退化为只切 batch 维、不切 token 维（GPT3-1B 两个大 batch setting 下 TeraPipe 无加速） | TP §4 实验分析 | 大 batch、小模型（GPT3-1B） |
| C14 | TeraPipe 切分过细的问题：GPU 需要大块计算才能充分利用，GPT3-1B 单层上单个 token 与 256 个 token 的前向时间相同 | TP §3.2 第一个挑战 | 小模型单层；说明 chunk 下限的来源 |
| C15 | SGLang 的 PP 实现与其他并行策略、PD 分离、HiCache 兼容 | SG 引言与 PP Roadmap 提法 | 组合可能性的事实依据 |

## F 公式

| 编号 | 公式 | 来源定位 | 说明 |
|---|---|---|---|
| F1 | 气泡比率 $=\frac{K-1}{K-1+M}$（$K$=stage 数，$M$=micro-batch/chunk 数） | GPipe 结论；`model-parallelism` 页已覆盖（该页记号为 $p,m$） | 本页引用不推导，记号映射需在正文说明 |
| F2 | $T^*=\min_{l_1,\dots,l_M}\{\sum_{i=1}^{M}t_i+(K-1)\cdot\max_{1\le j\le M} t_j\}$ | TP §3.3 公式(5) | 第一项=单 stage 总前向时间；第二项=流水线开销，由最慢片段决定 |
| F3 | $t_i=t_{\text{fwd}}(l_i,\sum_{j=1}^{i-1}l_j)$ | TP §3.3 公式(4) | 片段时间依赖自身长度与全部前缀长度 |
| F4 | $\mathrm{Runtime}(L+\Delta L)-\mathrm{Runtime}(L)=\mathrm{Runtime}(\text{初始 chunk})$ | SG「Dynamic Chunking」 | 累计运行时间建模为 $L$ 的二次函数后解出 $\Delta L$ |
| F5 | $\#D=\#RD/\#PP_{\text{depth}}$；$\#P$ 见 C11 合并式 | GL §3.2.1 公式(4)、§3.1.3 公式(3) | gLLM 两侧调节公式 |

## N 数字

| 编号 | 数字 | 来源定位 | 条件 |
|---|---|---|---|
| N1 | TeraPipe 训练加速 5.0×（GPT-3 175B，48 台 AWS p3.16xlarge，对比 SOTA 模型并行） | TP Abstract | 同步训练场景 |
| N2 | DP 非均匀切分 vs 最优均匀切分：1.12×（GPT3-44B）、1.04×（GPT3-175B）；GPT3-175B 的 DP 方案为 $[120]\times4+[112]\times6+[104]\times8+[64]$（前长后短） | TP §4.2 | 训练场景 |
| N3 | Mooncake `prefill_chunk` 阈值通常大于 1000 token（为吃满单 GPU 算力） | MC §3 | Mooncake 部署经验 |
| N4 | SGLang：DeepSeek-V3.1（H20）PP4×TP8 prefill 吞吐 = TP8 的 3.31×（chunked prefill 12K），比 TP32 方案高 30.5%；Qwen3-235B-A22B-FP8 PP8 加速 6.14×；TTFT 降 67.9%（DeepSeek，48.5s→15.5s）与 81.1%（Qwen3-235B，55.5s→10.5s）；PP4 保持 82.8% 强扩展效率；动态 chunk 的 smooth factor 默认 0.75、推荐 0.6–0.85；chunk 预测值向下对齐到 $\max(\text{page size},64)$ 倍数 | SG 基准与调参章节 | 各数字的模型/硬件/配置如文中所述 |
| N5 | gLLM：最大吞吐比 SOTA PP/TP 系统高 11%–398%；TTFT 排队拐点出现在 2–6× 更高请求率下；实验模型 Qwen2.5-14B/32B、Llama-3.1-100B，硬件 L20/A100/A800；基线 vLLM v0.8.1（PP）与 SGLang v0.4.3（TP），基线沿用 Sarathi 调度、token budget=2048 | GL Abstract、§4.1 | 条件如列 |
| N6 | 一念：多阶段流水线 + batch 间负载均衡（「首次在大规模语言模型推理这种有状态服务中实现」），该优化使吞吐从 5K 提升到 9K | YN 演讲实录 | 辅助佐证，工业口径数字、无公开论文细节 |

## 辅助材料（不进 C/F/N 编号，正文引用时文字标注来源）

- TeraPipe DP 递推 $S^*(i;t_{\max})=\min_{1\le k\le i}\{S^*(i-k;t_{\max})+t_{\text{fwd}}(k,i-k)\mid t_{\text{fwd}}(k,i-k)\le t_{\max}\}$（TP §3.3 公式(8)，Algorithm 1）；固定 $t_{\max}$ 复杂度 $O(L^2)$，总体 $O(L^4)$，配合提前终止与 $\varepsilon$ 间隔可在分钟内完成（TP §3.3）。
- $t_{\text{fwd}}(i,j)=t_{\text{fwd}}(i,0)+t_{\text{ctx}}(i,j)$，上下文开销用线性模型 $t_{\text{ctx}}(i,j)=a_0+a_1 i+a_2 j+a_3 ij$ 拟合，相对预测误差 <2%（TP §3.3 公式(9)）。
- SGLang 异步 P2P（`async_send` 返回 `P2PWork` 句柄、同步推迟到 commit）与多流执行（forward_stream/copy_stream）（SG 工程细节节）。
- SGLang 动态 chunk 三步调参：先找最优固定 chunk，初始值取其 2–3×（极大 ITL 建议 4×），动态预测保证下限为初始值 1/4（SG 调参指南）。
- 一念背景：61 层 DeepSeek 每输出一个 token 需 122 次跨机通信，推动其用多阶段流水线替代部分跨机通信（YN）。

## 证据缺口与降级处理

- 「gLLM token throttling 可与 CPP 叠加」：无直接文献证据，不得写成论断。Ch5 组合一节只用 C12（TeraPipe 联合切分）与 C15（SGLang PP 兼容 PD/HiCache）支撑，两者都不声称 gLLM×CPP 组合。
- Mooncake「首个推理阶段应用流水线加速」为作者自述（MC §5.1「to our knowledge」），引用时保留限定语。
- 一念数字（N6）无公开论文，只作折叠块佐证，不进正文主论证链。
- 章明星「forward-only 不需要 DP、设最小 chunk 即可」的观点来自方佳瑞文章转述，未核实一手出处，不写入正文；页面只呈现 Mooncake 实际采用的简化方案与 SGLang 的在线模型两条工程路线（M5 误解说明背景，不点名争论）。
