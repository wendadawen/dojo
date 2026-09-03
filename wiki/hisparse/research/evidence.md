# HiSparse 核心论断与证据

来源优先级：固定版本论文原文（arXiv:2608.07009v1 TeX 源码）> 官方源码（SGLang 上游，未直接浏览，仅按论文附录 A 定位）> 引文原始论文。

## C 论断（机制与定性结论）

| 编号 | 论断 | 来源定位 | 适用条件 | 置信状态 |
|---|---|---|---|---|
| C1 | top-$k$ 稀疏注意力每步读 $k$ 个选中条目，但选择集随步与层漂移，被跳过的条目之后可能被选中，因此 serving 系统把完整 KV cache 常驻 HBM | §1 第 3 段、§2.2 第 1 段 | top-$k$ 类稀疏注意力模型 | 已确认 |
| C2 | 容量墙的量化形式：decode batch 必须 $N_{\text{batch}} \times L_{\text{ctx}}$ 个 token 的 KV 状态放进权重之后的 HBM，而每步注意力只读 $N_{\text{batch}} \times k$ | §2.2（"The resulting admission constraint is the quantitative form of the capacity wall"） | 全 KV 驻留的 serving 系统 | 已确认 |
| C3 | HiSparse 在 host pinned DRAM 维护每请求完整 KV 权威副本；GPU 侧每请求每层预留固定 $B$ 格 GPU cache（$B \ge k$），带页表（逻辑位置→槽位或 host-only 哨兵）与 LRU 元数据 | §3.2 第 1–2 段、Table 2 | — | 已确认 |
| C4 | 解码 footprint 公式：$N_{\text{batch}} N_{\ell} B W_{\text{KV}} s$，替代全量驻留的 $N_{\text{batch}} N_{\ell} L_{\text{ctx}} W_{\text{KV}} s$（不计元数据） | §3.1 "Bounded device footprint" 段 | $B$ 与 $L_{\text{ctx}}$ 无关、部署时固定 | 已确认 |
| C5 | 精确性不变量：HiSparse 只改变未选中条目的驻留位置，不改变选中位置、注意力分数或输出 | abstract（"Because only KV placement changes, model outputs are unchanged"）、§3.1 "Exact sparse-attention outputs" | 每层选择集在注意力前完成解析 | 已确认 |
| C6 | indexer 无关：HiSparse 不假设选择集如何产生，只消费每请求每层发出的逻辑位置；DSA/NSA/Quest 均可接入 | §3.1 "Indexer-agnostic interface"、§2.1 三共性（选择状态紧凑、先选择后读、输出同形） | 选择器满足 §2.1 三共性 | 已确认 |
| C7 | 选择状态（indexer key、块 key、页摘要）与 HiSparse 页表、LRU 元数据常驻 GPU 不分页；合计至多几百字节/token 量级，对比 KV 记录约 100 KB/token | §3.2 第 3 段 | — | 已确认 |
| C8 | 替换策略：每请求每层独立 LRU；同一步内 hit 条目提升到新 fetch 的 miss 之上（多次复用者留下、一次性选择先走）；层间不协调驻留 | §3.2 "Replacement policy" 段 | — | 已确认 |
| C9 | 请求生命周期四阶段：prefill 各层 KV 写入 host pool（indexer 状态留在设备）→ host 状态就绪且每层 cache 预留后可调度 → 每步每层先解析再跑注意力 → 新 token KV 直接产出到 reserved slot、由 backup stream 经事件排序 write-through 到 host pool | §3.3 四个阶段段 | colocated 本地写 / disaggregated 经传输路径写 | 已确认 |
| C10 | 融合理由：解析在每个稀疏层每步重复，拆多个 CUDA launch 会反复物化中间态到 HBM 并加 launch 延迟；中间状态紧耦合于同一选择集与驻留元数据 | §3.4 第 1 段 | — | 已确认 |
| C11 | Resolve kernel：每稀疏层启动一次，一个 CUDA block 处理一个请求的工作项，捕获在 SGLang 稳态 decode CUDA graph 内；五阶段：①线程协作把选择集装入共享内存哈希表 ②并行探测 $B$ 个 slot 标记 hit/可逐出 ③并行 scan 压缩可逐出槽位、选 victim、更新 LRU（hit 提升至 MRU 端、fetched miss 排 hit 之后）④负责 miss 的线程从 pinned host 池拷贝记录到认领槽位 ⑤更新页表并发布 top_k_device_locs（与选中逻辑位置对齐的物理槽位稠密向量） | §3.4 五个 Phase 段、Figure 3 | — | 已确认 |
| C12 | fetch 用 Strata 的 GPU-assisted IO：GPU 线程对 pinned host 内存直接发向量化非一致加载（ld.global.nc.v2.b64），容忍分散地址、降低 PCIe/NVLink-C2C 事务开销；per-thread 传输块大小调优使碎片化 miss 读接近链路带宽 | §3.4 Phase 4 | PCIe 与 NVLink-C2C 系统 | 已确认 |
| C13 | IndexCache 把层划分为 anchor 层（跑 top-$k$ indexer）与 shared 层（复用前面 anchor 的选择）；GLM-5.2 以 IndexShare 原生每 4 层一组共享一个 indexer | §3.5 第 1 段；引文 bai2026indexcache、glm52_blog_2026 | 模型显式共享选择 | 已确认 |
| C14 | plan-then-IO 精确预取：anchor 的 Resolve 顺带记录 miss plan（哪些 host 记录进哪些槽位）；side stream 上 copy-only kernel 把该 plan 重放进每个 shared 层的 cache，传输与中间层计算重叠；shared 层 cache 与 anchor 槽位布局同步，等待 prefetch 完成事件后直接复用 anchor 的 slot 表、完全跳过解析（无探测、无 LRU 更新、无同步 host 加载、无浪费流量） | §3.5 "Exact prefetch" 段 | 共享选择模型 | 已确认 |
| C15 | 推测式预取（用层 $\ell$ 的选择提示层 $\ell{+}1$，错误 hint 只浪费传输不影响正确性）实测无端到端收益：hint 的位置大多已驻留，剩余 miss 恰是预测不了的新进入位置 | §3.5 末段、§4.6 末段 | 无共享选择的模型 | 已确认 |
| C16 | no-IO oracle：跳过 resolve kernel 的全部 host IO（miss 时用过期记录、输出无效）；固定输出长度下其时序是任何 IO 隐藏方案的有效上界，强于完美重叠的预取 | §4.6 第 1 段 | 固定输出长度的 benchmark | 已确认 |
| C17 | 与 SGLang HiCache 的关系：HiCache 是分层前缀缓存（跨请求复用）；HiSparse 通过 mixin 复用其 host-tier 基础设施（pinned host pool 与 IO backend），管理对象是请求内由模型稀疏选择主导的解码工作集；部署中两者可组成 prefill 节点用 HiCache、decode 节点用 HiSparse 的对偶 | 附录 A "Relationship to HiCache" 段 | SGLang 集成 | 已确认 |
| C18 | 实现合并进上游 SGLang：约 2200 行新 Python（六模块）+ 一个 CUDA kernel header，另有 scheduler、model runner、attention backend、disaggregation 路径的集成改动；`--enable-hisparse` 启用，`--hisparse-config` JSON 设 top_k、device_buffer_size（$B$）、host_to_device_ratio、swap-in 传输块大小 | §1 第 6 段、附录 A "New components" 与 "Configuration" 段 | — | 已确认 |
| C19 | 三种 selector 的选择状态与粒度（Table 1）：DSA 每 token 紧凑 indexer key（token 级、co-trained）；NSA 每 token 块一个压缩 key（block 级、trained）；Quest 每 page 的 min/max key 向量（page 级、training-free） | §2.1 Table 1 | — | 已确认 |
| C20 | DeepSeek-V4-Flash 的选择在 4-token 压缩 KV 条目粒度上运作，top-512 覆盖 2048 token；HiSparse 管理这些压缩条目（占该模型 KV 占用主体），其余分支状态保持 GPU 驻留 | §4.1 "Models" 段 | — | 已确认 |
| C21 | 混布模式 TTFT 机制：decode 饱和使 prefill 块与新请求排队，TTFT 被队列时间主导随负载陡增；HiSparse 留出更多 HBM 余量并更快排空 decode 工作 | §2.2 第 2 段、§4.2 第 3 段 | PD-colocated | 已确认（机制 §2.2 原文；与 §4.2 数字的因果拼接含推断成分，正文标注） |
| C22 | host 容量局限：GB200/GB300 每个 Grace CPU 约 480 GB LPDDR，与配对 GPU 的聚合 HBM 相当（GB300 上更小）；第二层不再大于第一层时 HiSparse 的容量乘数缩小；NVMe/网络层可恢复容量但延迟更高 | §5 "Host-memory capacity" 段 | Grace 类平台 | 已确认 |
| C23 | $B$ 是部署时静态固定的 serving 配置参数，按平台 profiling 选择；实验中偏好设置主要取决于平台 host 链路带宽而非工作负载；动态调整留 future work | §4.5 第 2 段 | — | 已确认 |

## F 公式

| 编号 | 公式 | 来源定位 | 适用条件 | 置信状态 |
|---|---|---|---|---|
| F1 | 解码 KV footprint：$N_{\text{batch}} N_{\ell} B W_{\text{KV}} s$（有界）对照 $N_{\text{batch}} N_{\ell} L_{\text{ctx}} W_{\text{KV}} s$（全量），不计元数据 | §3.1 "Bounded device footprint"（Table 2 定义 $W_{\text{KV}}$ 为每 token 每层 KV 元素数、$s$ 为每元素字节数） | $B \ge k$、$B$ 与 $L_{\text{ctx}}$ 无关 | 已确认 |
| F2 | admission 时每请求预留：$N_{\ell} B W_{\text{KV}} s$，对照全量的 $N_{\ell} L_{\text{ctx}} W_{\text{KV}} s$ | §3.3 "(2) Admission" 段 | 同 F1 | 已确认 |
| F3 | naive offload 带宽估算：GLM-5.1 每 token 约 200 MB KV 记录移动（$k{=}2048$、跨层求和约 100 KB/token），TPOT 30 ms 下约 7 GB/s 持续 host→device 流量每请求；十几个并发即饱和 PCIe Gen5 ×16（约 64 GB/s 每方向）——这是"解耦本身不够、必须靠局部性"的论证 | §2.3 第 2 段 | 每层选择都从 host 取的假设（反设） | 已确认 |

## N 数字（实验与定量结论）

| 编号 | 数字 | 来源定位 | 实验条件 | 置信状态 |
|---|---|---|---|---|
| N1 | 峰值生成吞吐最高 4.7×：Qwen3+Quest GH200 200K 输入 111→520 tokens/s；32K 时 3.6×（511→1824）；4K 时 2430→2668（1.10×） | abstract、§4.2、Figure 5 | Qwen3-30B-A3B-Thinking-2507 + Quest，GH200，k=2048，峰值吞吐对比，baseline SGLang v0.5.11 全 KV | 已确认 |
| N2 | GLM-5.1：32K 3.1×（624→1919）、160K 2.9×（232→680）、4K 几乎不变（2288→2280） | §4.2、Figure 5 | GLM-5.1-FP8（DSA），8×H200，k=2048 | 已确认 |
| N3 | DeepSeek-V4-Flash 32K/8K 并发 64：600→1257 tokens/s（2.1×）；decode-only 1511→4308（2.9×）；并发 8 时 TTFT 26 s、并发 64 时 baseline 829 s vs HiSparse 171 s；TPOT 并发 16 时 15.9 vs 16.0 ms | §4.2、Figure 4 | DeepSeek-V4-Flash（NSA 式，top-512 over 4-token 条目），2×B200，PD-colocated，32K 输入 8K 输出 | 已确认 |
| N4 | GLM-5.1 128K 请求 13.09 GB BF16 KV；$B{=}4096$ 时每请求约 0.4 GB（约 30×）；1M token 请求超 100 GB，H200 为 141 GB HBM；1M 请求在权重驻留后无法 admission | §1 第 3 段、§3.3 "(2) Admission"、§4.2 第 4 段 | GLM-5.1 BF16 KV | 已确认 |
| N5 | GLM-5.1 32K/8K 每请求至多 4 GB KV；全 KV baseline 约 60 并发饱和（约 240 GB KV，权重/激活/CUDA graph 状态之外的部分）；128K 只能约 15 请求；全 KV decode-only 777 tokens/s | §2.2 第 3 段、§4.2 第 4 段 | 8×H200（聚合 1.1 TB HBM） | 已确认 |
| N6 | miss-rate trace：GLM-5.1 服务 100,384 token LongBenchV2 prompt、解码 1799 步、78 稀疏层；统计前 1000 步。$B{=}k{=}2048$ 只存 top-k（Swap-vanilla）平均 miss 30%；$B{=}4096$：LRU 13.4% < random 16.1% < FIFO 17.2%，Bélady 8.2%；$B{=}8192$ LRU 6.7%；LRU@B=2k 命中率约 87% | §1 第 5 段（87%）、§2.3、§4.3、Figure 6 | 单 trace 重放，各策略重放同一每层 top-k 选择流 | 已确认 |
| N7 | GLM-5.1 32K/8K（Figure 1）：baseline 吞吐在 HBM 饱和后走平，HiSparse 继续扩展；混布模式 baseline TTFT 从并发 8 的低值升至并发 64 的 829 s，HiSparse 171 s；重叠区间 TPOT 可比（15.9 vs 16.0 ms @并发 16） | §4.2 第 3 段（TTFT 数字）、Figure 1 | GLM-5.1-FP8（DSA，k=2048）、8×H200、32K/8K | 已确认 |
| N8 | prefetch 实验（GLM-5.2-FP8，IndexShare，8×H200，32K/8K，k=2048，B=4096，并发 8–256）：baseline 峰值 618 tokens/s；HiSparse 同步解析 1515；精确预取 1727（对 baseline 2.8×）；no-IO oracle 2034，预取达其 85%（无预取 74%）；decode-only：oracle 4671、预取 3410（73%）；TPOT 降低 13–15%、吞吐提升 14–17%（全并发范围）；同步解析 IO 暴露 7.7 ms/token @并发 8、22.0 @并发 256，预取后 3.0、11.2；oracle TPOT 24.1 vs baseline 24.8 ms @并发 8；baseline TTFT 16 s @并发 16 → 91 s @32 → 275 s @64，HiSparse 全变体扩到 256 并发；并发 8 时 TTFT 10.7 vs 10.7–10.8 s（staging 免费近似） | §4.6、Figure 8 | GLM-5.2-FP8（DSA + IndexShare，78 层 = 21 anchor + 57 shared） | 已确认 |
| N9 | kernel 分解：GLM-5.1 $B{=}2k$、batch 16 时 IO 分量 112 μs（H200，PCIe Gen5）→ 29 μs（GH200，NVLink-C2C）；probe&scan 平台无关（GH200 曲线贴 H200 带 ±5% 内）；残差相位（哈希表构建+输出发布）各处 1–4 μs；对照：稀疏注意力 kernel 本身约 60 μs/层（GLM-5.2 解码 profile，per-GPU batch 8，H200）；未隐藏的 100–200 μs resolve 会让某层注意力关键路径翻倍以上 | §4.4、Figure 7 | 三端到端模型 + DeepSeek-V4-Pro（token 级 k=1024），各自原生选择粒度 | 已确认 |
| N10 | $B$ 的实用范围 $[2k, 4k]$；更快的 host-device 链路把最优推向 $B{=}2k$（保留更多设备内存给并发）；$B{=}2k$ 是稳健默认 | §3.2 "Sizing the cache"、§4.4、§4.5 | 跨被测模型 | 已确认 |
| N11 | H200 节点 8 GPU 配 2 TB host DRAM；最大运行点（256 并发、32K/8K）host pool 约 1 TB pinned；PCIe Gen5 ×16 约 64 GB/s 每方向 | §2.3、§4.1 "Platforms" | — | 已确认 |
| N12 | baseline 为未修改 SGLang v0.5.11，全 KV 驻留 HBM，其余部署配置（模型、并行、精度、调度器设置）一致；对比无 offload 系统（ESS 仿真原型、ECHO NSA 专用，为并发工作） | §4.1 "Baseline" | — | 已确认 |
| N13 | anchor 层自身选择无法提前知道，21/78 层（均匀分摊下约 27% IO）保持同步，构成预取的结构性下界；扣除该 floor 后预取隐藏其可作用 IO 的 2/3 到 4/5，剩余为重叠缺口（预取搬移而非消除流量，高 batch 下与 demand miss 争同一 host 链路） | §4.6 第 4 段 | GLM-5.2 层结构 | 已确认 |
| N14 | $B{=}4096$ 下 GLM-5.1 约 60 请求 batch 的解码 KV 预算从约 240 GB 降到约 25 GB；容量红利可兑换为更少 GPU 或更便宜的低 HBM 部件，代价是每 token IO 开销 | §4.2 第 4 段 | 32K、8×H200 | 已确认 |
| N15 | 模型选择状态部署中合计数百 MB，与每请求数 GB 的 KV 相比小 | §3.2 第 3 段 | 论文部署 | 已确认 |
| N16 | 混布模式下 HiSparse TPOT 随进入更高吞吐工作点而上升，但重叠区间与 baseline 可比 | §4.2 第 3 段 | DeepSeek-V4-Flash 实验 | 已确认 |
| N17 | GB200/GB300 平台 Grace CPU LPDRAM 约 480 GB | §5 | — | 已确认 |
| N18 | 87% 命中率表述：LRU（2×top-k 大小）把 87% 的选择变成设备命中 | §1 第 5 段、§2.3 末段 | LongBenchV2 trace | 已确认（与 N6 的 13.4% miss 一致，86.6%≈87%） |

## 原文图候选

| 页面使用编号 | 原文 Figure | 内容摘要 | 可说明的结论 | 获取途径 |
|---|---|---|---|---|
| IMG-1 | Figure 1 | (a) 吞吐-并发曲线 baseline 走平 vs HiSparse 扩展；(b) 混布模式 TTFT-吞吐 | Q1 容量墙与 TTFT 后果 | TeX 源码 figs/motivation.pdf |
| IMG-2 | Figure 2 | 系统总览：host pool、GPU cache、五步数据流 | Q2 层级结构与生命周期 | figs/hisparse_overview2.pdf |
| IMG-3 | Figure 3 | Resolve kernel 五阶段 | Q4a 融合解析 | figs/swap_kernel.pdf |
| IMG-4 | Figure 5 | 两平台两模型的峰值吞吐-输入长度柱状对比 | Q5 4.7×/3.1× 的来源 | figs/peak_throughput_comparison_paper.pdf |
| IMG-5 | Figure 6 | 七种 cache 配置的每步 miss 率曲线 | Q3 LRU 优于 FIFO/random、B 加倍再减半 | figs/topk_miss_rate_trace.pdf |
| IMG-6 | Figure 8 | prefetch 四配置的吞吐/TTFT/TPOT | Q4b 预取效果与 oracle 上界 | figs/prefetch_sweep.pdf |

不纳入页面的图：Figure 4（DeepSeek-V4-Flash 扫描，关键数字 N3 以表格完整呈现，基线不选择性删减）、Figure 7（kernel 分解，关键数字 N9 以文字呈现）。
