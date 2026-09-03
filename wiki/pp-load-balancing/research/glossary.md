# 术语表：PP 负载均衡

登记全文首次出现的术语、缩写和符号。写作与审查以此为准，防止同一对象出现多种记法。

## 术语

| 术语 | 首次出现 | 定义 |
|---|---|---|
| PP（pipeline parallelism，流水线并行） | 页面开头 | 把模型按层堆叠切成连续片段、分到不同设备依次处理的并行方式；本文只在开头引用模型并行页的结论 |
| stage（流水级） | 页面开头 | 流水线中的一台/一组设备，负责一段连续的层 |
| micro-batch（微批次） | Ch1 | 一次迭代中并行推进的最小批次单位；并发 micro-batch 数受流水线深度限制 |
| 气泡（pipeline bubble） | Ch1 | stage 等待输入/等待下游时的 GPU 空闲时段 |
| inter-stage 依赖 / inter-batch 依赖 | Ch1 | 后级等前级完成 / 并发 micro-batch 数受流水线深度限制，两类等待来源（gLLM 命名） |
| inter-stage 失衡 / inter-batch 失衡 | Ch1 | stage 间计算分布不均 / 不同 micro-batch 计算量不同，两类失衡（gLLM 命名；本页不处理前者） |
| 前缀长度 | Ch1/Ch3 | 已处理的历史 token 数；决定注意力交互次数 |
| 增量计算特性 | Ch1 | 位置 $t$ 的注意力要与全部 $\le t$ 的位置交互，同尺寸片段的成本随前缀增长 |
| token 维度切分 | 页面开头 | 把单条请求的输入序列切成片段送入不同 stage 的做法（TeraPipe/CPP 一系） |
| batch 维度均衡 | 页面开头 | 在并发请求间配平各 micro-batch 计算量的做法（token throttling 一系） |
| 切片（slice）/ 片段 | Ch2 | token 维度切分得到的一段连续 token；本文「切片」与「片段」同义，全文统一用「片段」，引用 TeraPipe 处保留「切片」并声明同义 |
| 前长后短 | Ch2 | 最优切分的形态：靠前的片段 token 更多、靠后的更少 |
| 目标函数 $T^*$ | Ch2 | TeraPipe 的流水线时延：单 stage 总前向时间 + $(K-1)\times$ 最慢片段时间 |
| DP（dynamic programming，动态规划） | Ch2 | TeraPipe 求最优切分的算法；递推放在折叠块 |
| CPP（chunked pipeline parallelism，分块流水线并行） | Ch2.4 | Mooncake 对推理 prefill 的 token 维切分实现 |
| `prefill_chunk` | Ch2.4 | Mooncake 的 chunk 长度上限，通常大于 1000 token |
| chunk | Ch2.4 起 | 进入流水线的 token 块；与 chunked prefill 的 chunk 同一概念（引用前置页） |
| 漂移（drift） | Ch3 | 固定 chunk 下片段处理时间随前缀增长、逐步偏离对齐状态的现象（本文用词，指 SGLang 描述的 timing mismatch） |
| 级联放大 | Ch3 | 错配沿流水线向后传播并在更高 PP rank 累积 |
| 动态 chunk（dynamic chunking） | Ch3 | SGLang 按二次运行时间模型逐步缩小 chunk 的机制 |
| smooth factor（平滑因子） | Ch3 | 控制 chunk 缩小幅度的系数，默认 0.75 |
| token budget（token 预算） | Ch4 | Sarathi-Serve 给每次迭代 prefill+decode 总 token 数设的固定上限（引用 chunked-prefill 页） |
| token throttling（token 节流） | Ch4 | gLLM 独立调节 prefill/decode token 数量的调度策略 |
| 一念 | Ch4 折叠块 | 腾讯 PCG 的 LLM 推理框架，多阶段流水线 + batch 间均衡的工业案例 |

## 符号

| 符号 | 首次出现 | 定义 | 来源与一致性 |
|---|---|---|---|
| $K$ | Ch1 | 流水线 stage 数 | TeraPipe 记号；模型并行页气泡公式用 $p$，本文统一为 $K$ 并在引用处声明映射 |
| $M$ | Ch1 | micro-batch（或 chunk）数 | TeraPipe 记号；模型并行页用 $m$，同上映射 |
| $L$ | Ch2 | 序列总长度（token 数） | TeraPipe 记号 |
| $l_i$、$t_i$ | Ch2 | 第 $i$ 个片段的长度与处理时间 | TeraPipe Eq.(4)(5) |
| $t_{\text{fwd}}(l,\cdot)$ | Ch2 折叠块 | 长度 $l$ 片段的前向时间函数，第二参数为前缀总长 | TeraPipe Eq.(4) |
| $\Delta L$ | Ch3 | 下一个 chunk 的大小 | SGLang 博客记号 |
| $\mathrm{Runtime}(L)$ | Ch3 | 前缀长度 $L$ 时的累计运行时间（二次函数模型） | SGLang 博客 |
| $\#RD$ | Ch4 | 运行中的 decode token 总数 | gLLM 记号 |
| $\#PP_{\text{depth}}$ | Ch4 | 流水线深度（= micro-batch 并发上限） | gLLM 记号 |
| $\#D$ | Ch4 | 每个 micro-batch 排入的 decode token 数 | gLLM Eq.(4) |
| $\#WP$ | Ch4 | 等待 prefill 的 token 数 | gLLM 记号 |
| $\#T$ | Ch4 | 把积压 prefill 分摊到的迭代数（超参，实验取 8） | gLLM 记号 |
| $\#P$ | Ch4 | 本次迭代排入的 prefill token 数 | gLLM Eq.(3) |
| $\#MaxP$、$\#MinP$ | Ch4 | prefill batch 大小上下限（超参） | gLLM 记号 |
| $KV_{\text{free}}$、$KV_{\text{thresh}}$ | Ch4 | KV cache 空闲率与溢出保护阈值 | gLLM 记号 |

## 用词约定

- 「stage」不写作「阶段」（避免与 prefill/decode 的「阶段」混淆）；prefill/decode 统一称「阶段」或直接用英文名。
- 「片段」用于 token 维度切分的结果；「chunk」在 CPP/chunked prefill 语境使用；两者同指时（Ch2.4 起）统一「chunk」。
- PPLB 一词只出现在开头语境说明，不作为正文术语。
- 数学符号一律 `$...$` 书写；gLLM 的 `#` 前缀是论文记号，正文写作 $\#D$ 形式。
