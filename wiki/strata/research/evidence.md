# Strata 核心论断与证据

编号规则：C 论断 / F 公式 / N 数字。来源定位以 arXiv:2508.18572v2 TeX 源码（文件名:行号）+ 论文章节为准。所有条目置信状态：已确认（除非另注）。

## C 论断（机制与定性结论）

- C1：长上下文 KV cache 存储占用超出 GPU 显存，生产系统采用分层缓存（CPU 内存、本地 SSD、远端内存池）。示例数字：40GB HBM 对 Llama-8B 只能存约 0.3M token。来源：1_intro.tex L10（§1）；分层的引用 eurosys25:yu_pensieve / atc24:gao_cachedattention / arxiv24:hu_memserve / fast25:qin_mooncake。条件：Llama-8B、40GB HBM（0.3M token 由 40GB ÷ 128KB/token ≈ 305K 推得，与原文一致）。
- C2：现有系统页大小很小——TensorRT-LLM 32、vLLM 16、SGLang 1 token；每 token KV cache 从几十 KB 到几 MB。来源：1.5_background.tex L14（§2.2）。
- C3：分页导致数据碎片化：一个序列的 KV cache 散布在多个不连续页，传输量小到只有几 KB，无法打满 PCIe 带宽。来源：1_intro.tex L30–32（§1）。
- C4：现有调度器假设新 token 的 prefill 计算足以掩盖历史 KV cache 的加载延迟（SGLang/vLLM/TensorRT-LLM 均如此假设）；上下文变长后该假设失效，系统 loading-bound。来源：1_intro.tex L34–38（§1）、2_motivation.tex L58（§3.2 引用 pensieve/cachedattention 的该假设）。
- C5：Little's Law：$C=\lambda\cdot L$；吞吐 $X=\lambda\cdot S$；合并得 $X=C\cdot S/L$。提高吞吐需要高并发 C、大传输 S 或低延迟 L；CPU-GPU 场景下增大 S 是最实际的杠杆，打满 PCIe 5.0 的 75–80% 需 1–2MB 传输。来源：2_motivation.tex L25–34（§3.1）。F1/F2 见下。
- C6：加大页虽提高 S，但 hit rate 与 TTFT 显著恶化——ShareGPT、Mistral-24B、H200、SGLang 上页 1→1024，hit rate 显著下降，平均 TTFT 最多升 2 倍、P90 升 2.9 倍；原因是缓存按页匹配，页越大匹配粒度越粗。来源：2_motivation.tex L37–43（§3.1，图 Figure 2）；"cache matching is performed on a per-page basis" L39。
- C7：延迟命中（delay hit）：多个请求在 cache miss 尚未解决期间到达并排队，被编入同一批则产生冗余 prefill；异步调度器（提前准备下一批）把 miss 解析窗口拉长到整批执行时间，加剧该现象。来源：2_motivation.tex L65–68（§3.2），现象引自网络社区 delayhit 文献。
- C8：GPU-assisted I/O 机制：不反复调用 cudaMemcpyAsync，而是启动 CUDA kernel，数千线程各自把一小块数据从源（GPU 全局内存或 CPU 注册 pinned 内存）读入本地寄存器文件再流出到目的。来源：3_design.tex L30–31（§4.2）。
- C9：GPU-assisted I/O 三个优点：①并发 C——GPU 提供数千并发 I/O（CPU 通常只有数十）；②小传输友好 S——高效粒度仅 128 字节（多数架构），单页 KB 级也高效，无需为效率放大页；③布局灵活——I/O kernel 内轻量计算几乎免费，布局转换可忽略开销。来源：3_design.tex L33–36（§4.2）。
- C10：I/O kernel 与计算 kernel 存在运行时干扰（寄存器、执行周期、cache 污染；GPU 硬件调度器难以管理该竞争，引 rammer）。Strata 的控制：启动少量大 CUDA block，诱使硬件调度器把 I/O kernel 限制在极少数 SM（可低至 1 个），并用底层指令绕过 cache 减轻污染。来源：3_design.tex L48–54（§4.2）。
- C11：布局解耦：GPU 内存保持计算友好的 layer-first 布局（与 LLM 逐层计算对齐），host 内存与外部存储采用 page-first 布局（页内各层连续，利于大块传输）；转换由 I/O 线程对分配偏移多做一次算术运算完成，开销可忽略。来源：3_design.tex L79–86（§4.2.1，图 Figure 6）。
- C12：外部存储参与时，cache controller 在存储层命中后机会性预取到 host 内存，预取延迟与排队延迟重叠；请求被调度时终止在途预取、利用已在 host/GPU 的数据（best-effort）。来源：3_design.tex L73–76（§4.2.1）。
- C13：调度器三阶段：①识别可能 delay hit 的请求延迟执行；②构建均衡 batch（加载与计算配平）；③仍 loading-bound 时插入有用计算填空泡。来源：3_design.tex L101–106（§4.3）。
- C14：delay hit 延迟执行机制：HiRadixTree 引入 transient node，用 token ID 作遍历键但不指向内存索引，携带 in-queue（有请求引用新上下文）或 in-flight（对应 token 的 cache 正在计算）标记；命中 transient node 的请求推迟到下一轮并置于等待队列队首；执行时其 transient node 转 in-flight，完成后转为带内存索引的标准节点；阈值：transient node 上 token 匹配数超过阈值才延迟，默认 100。来源：3_design.tex L111–115（§4.3.1）。
- C15：均衡 batch：默认引擎按 FIFO 组批直到批满（token 上限或显存耗尽）；Strata 组批时检查加入请求是否使批 loading-bound（聚合 load/compute 比例超阈值，阈值硬件/模型相关、可分别 profiling，默认 100，对应 Figure 1 中 stall 开始出现的位置）；不 bound 则加入并优先吸收与其 bundle hit 的请求（共享上下文，既配平又省显存与片上带宽）；bound 则进降级列表，批不满时补充；防饿死：降级请求保持原序、每轮组批总从队首请求开始。来源：3_design.tex L150–167（§4.3.2，Algorithm 1）。
- C16：bubble filling：批仍 loading-bound 时推迟 prefill 计算、向模型执行器发 decode batch 与上下文加载并行；decode batch 虽也是 I/O-bound 但饱和 HBM 带宽、加载饱和 PCIe 带宽，资源基本不冲突；也可插 prefill 批（更适用于 P-D 分离系统，引 distserve）。来源：3_design.tex L169–174（§4.3.3）。
- C17：Strata 用 P-D co-location：同一 GPU 时间上交替执行 prefill 与 decode batch，沿用 SGLang 优先 prefill（缩 TTFT）、decode 合大批（提吞吐）的实践；prefill 完成的请求经 continuous batching 合并进统一 decode batch。来源：3_design.tex L23–25（§4.1）。
- C18：系统架构：请求进等待队列；Scheduler 在当前批执行期间持续估计系统资源与队列需求、选子集组成下一批，发批给 GPU 执行器并向 Cache Controller 发起 KV cache 加载请求；prefill 执行期间 GPU 执行器与 Cache Controller 同步确保特定层 KV cache 就绪；Cache Controller 异步管理 KV cache 向低层的备份与逐出。来源：3_design.tex L19–26（§4.1，图 Figure 4）。
- C19：分层缓存对长上下文 serving 是必要的：LooGLE 上非分层系统迅速耗尽 GPU 显存导致 prefill 频繁 miss、反复重算、低吞吐高延迟；分层系统借 CPU 内存稳定达到约 95% hit rate。来源：5_eval.tex L86–87（§5.2.1）。
- C20：消融归因：Strata-IO 与 Strata-Schedule-Only 相对 SGLang-HiCache 分别最高 2.3x 与 1.8x 峰值吞吐；低请求率下调度收益更大（批小、I/O 压力轻），高请求率下 I/O 子系统成为主导、GPU-assisted I/O 是维持高吞吐的关键。vLLM-LMCache 与 TensorRT-HiCache 也用 CUDA kernel 加速 KV cache I/O，与 Strata-IO 低请求率下相当，高请求率下 Strata-IO 维持更高吞吐（干扰抑制更有效）。Strata-IO-LPM（最长前缀匹配调度）低请求率下因提高在设备页复用而获益，高请求率下因逐出频繁无法保持；Strata 显式考虑带宽资源故持续领先。来源：5_eval.tex L126–138（§5.3.1，图 Figure 9）。
- C21：页大小负担消除：GPU-assisted I/O 使 I/O 效率对页大小不敏感；SGLang-HiCache 最优页（512）也只有 Strata-IO 的 93% 峰值吞吐，主因 hit rate 低 2.4%（页 512 比页 32 低）。来源：5_eval.tex L150–154（§5.3.2，图 Figure 10）。
- C22：cache distance 实验：最小距离（同上下文请求连续）时无需分层缓存（局部性完美），delay hit 缓解贡献最大（峰值吞吐 +42%）；shuffle 与最大距离时 I/O 机制分别 +76% / +95%（距离越大命中 CPU DRAM 越多）；均衡 batch 再 +11% / +12%；stall hiding 再 +8% / +3%（shuffle 方差大收益更高）；最大距离时 delay hit 缓解无收益。来源：5_eval.tex L157–168（§5.3.3，图 Figure 11）。
- C23：GH200 实验：标准 DMA 拷贝不加大页无法有效利用硬件改进；仅硬件改进（GH200 上的 SGLang 基线）不敌 H200 上的 Strata-IO；Strata-IO 把持续带宽从 40 提到 150 GB/s；但 Strata-IO-GH 仍不敌 Strata-PCIe（完整版，H200）——调度改进是吃满 Grace Hopper 类平台的必要条件；完整 Strata-GH 接近 Oracle（无限带宽模拟）性能。来源：5_eval.tex L212–225（§5.4，图 Figure 13/14）。
- C24：短上下文（ShareGPT）上 Strata 与其他 SOTA 系统性能相当；注意底层 SGLang 引擎在 Llama-8B/70B 上因 kernel 差异本身略弱于 vLLM/TensorRT-LLM 基础引擎。来源：5_eval.tex L106–111（§5.2.3）。
- C25：Strata 已在一家领先 AI 公司的生产环境部署（论文原文未具名）。来源：1_intro.tex L46（§1）。
- C26：实现基于 SGLang（广泛使用的开源框架）；ROCm 后端下 kernel 实现兼容 AMD GPU（仅声明，未见实验）。来源：1_intro.tex L46、3_design.tex L55。
- C27：相关工作定位：SGLang 用 RadixTree、vLLM/Mooncake 用哈希（token ID + 前缀页哈希生成页唯一标识）、LMDeploy 混合粗粒度 trie；Strata 把 RadixTree 扩展为 HiRadixTree。CacheGen/CacheBlend 是超越精确前缀的近似缓存，Strata 不影响请求精度。CachedAttention/Pensieve 层间重叠策略、FlashGen 加重排序执行调度（已实现进 SGLang 并用于本文基线）；Mooncake/MemServe 是大规模分离式方向，Strata 可集成其传输引擎但不依赖高速网络等专用硬件。来源：6_related.tex L2–21（§6）。on-chip I/O 加速器的展望见 7_discussion.tex L2（§7 Conclusion "motivating the design of more versatile on-chip memory I/O accelerators"），正文 §6 章引述 C27（§6 定义 Strata 相对位置），§7 给出后续方向。

## F 公式

- F1（C5 内）：$C=\lambda\cdot L$（Little's Law，$\lambda$ 为 I/O 操作到达率、$C$ 平均并发 I/O 数、$L$ 平均每次操作延迟）。来源：2_motivation.tex L26–27。
- F2（C5 内）：$X=\lambda\cdot S=C\cdot S/L$（$X$ 持续数据吞吐 GB/s、$S$ 平均每次传输的数据量）。来源：2_motivation.tex L28。
- F3（前置页 kv-cache 引入，页面直接引用）：KV cache 每 token 字节数 $=2\cdot L_{\text{layers}}\cdot H_{\text{kv}}\cdot d_{\text{head}}\cdot b$（K 与 V 两份、层数、KV 头数、头维度、每元素字节数）。用于验证 N1 的 0.3M token：Llama-3.1-8B（32 层、8 KV 头、$d_{\text{head}}=128$、bf16）得 128KB/token，$40\text{GB}/128\text{KB}=312{,}500\approx 0.3\text{M}$。模型配置来自 Llama 3.1 模型卡（外部事实）；"roughly 0.3M" 为论文原文。置信：已确认（计算复算一致）。

## N 数字（实验结果与配置）

- N1：40GB HBM ≈ Llama-8B 的 0.3M token。来源：1_intro.tex L10。条件：Llama-8B、bf16（由 F3 推得，原文给 "roughly"）。
- N2：LooGLE + SGLang offload（按先前工作标准配置）：74% prefill 时间阻塞在 KV 传输（红曲线），吞吐最多降 4 倍；配置了优化 I/O 机制后（绿线）仍有最高 24% prefill 执行时间阻塞在 cache 加载。来源：1_intro.tex L14（Figure 1）、L37（24%）、2_motivation.tex L63（"even with an I/O mechanism achieving 75% of theoretical PCIe bandwidth, stalls still account for up to 24%"）。条件：Qwen2.5-14B、LooGLE（Figure 1 caption）。
- N3：8192 token KV cache（页 32、Llama-3.1-8B）CPU→GPU：仅约 22% 理论 PCIe 5.0 带宽；GH200 类平台（NVLink 替代 PCIe、峰值带宽 6 倍）低至约 5%。来源：2_motivation.tex L47–49（§3.1，图 Figure 3）。条件：页 32 是分层 KV cache 先前工作推荐值与 vLLM CUDA GPU 最大支持值。
- N4：页大小扫描：ShareGPT、Mistral-24B、H200、SGLang，页 1→1024，平均 TTFT 最多 2x、P90 最多 2.9x 恶化。来源：2_motivation.tex L41–42（§3.1，图 Figure 2）。
- N5：干扰微基准：H200 上 I/O kernel（2 个 CUDA block、每 block 1024 线程）与 prefill（2 请求 × 4k 输入）同跑：约 50 GB/s 传输吞吐、prefill 性能损失 <5%、decode（16 请求 × 4k）损失 10%。默认配额：CPU→GPU 加载 2 block（关键路径）、GPU→CPU 备份 1 block（非关键路径）。端到端确认整体性能影响 <5%。来源：3_design.tex L56–59（§4.2，图 Figure 5）。
- N6：GPU-assisted I/O 有效粒度 128 字节（多数架构，引 PTX 文档）。来源：3_design.tex L35。
- N7：delay hit 延迟阈值默认 100 个活跃 token 匹配。来源：3_design.tex L115（§4.3.1）。
- N8：loading_bound 阈值默认 100（聚合 load/compute 比），对应 Figure 1 中 stall 开始出现处；阈值硬件/模型相关、可分别 profiling。来源：3_design.tex L162（§4.3.2）。
- N9：磁盘微基准：同样页 32、8192 token、从本地磁盘到 CPU 内存，page-first 布局延迟最多降 4 倍。来源：5_eval.tex L186–190（§5.3.4，图 Figure 12）。条件：微基准，disk 未纳入端到端对比（基线支持有限，L66）。
- N10：端到端（H200，图 Figure 8）：LooGLE 同 TTFT 吞吐倍数——Llama-8B：vs SGLang-HiCache 3.2x、vs vLLM-LMCache 2.6x、vs TensorRT-HiCache 1.9x；Qwen-14B：3.9x / 2.1x / 1.9x；Llama-70B：5x / 5x / 3.75x。ReviewMT（Llama-8B）：vs vLLM-LMCache 2.3x、vs TensorRT-HiCache 2.3x、vs SGLang-HiCache 1.7x。来源：5_eval.tex L90–91（§5.2.1）。abstract 的 "up to 5x lower TTFT vs vLLM+LMCache、3.75x vs TensorRT-LLM" 与 Llama-70B 行对应。
- N11：NarrativeQA 预热稳态（CPU 内存预计算全部上下文 KV、GPU 刷新后重启负载；TensorRT-HiCache 不支持预热未测）：vs vLLM-LMCache 吞吐 Llama-8B 2.3x、Qwen-14B 2.6x、Llama-70B 2.5x。来源：5_eval.tex L98–103（§5.2.2）。
- N12：消融（Qwen-14B、H200、LooGLE，图 Figure 9）：Strata-Schedule-Only 最高 1.8x、Strata-IO 最高 2.3x 峰值吞吐（vs SGLang-HiCache）。来源：5_eval.tex L133（§5.3.1）。
- N13：页大小扫描（图 Figure 10）：SGLang-HiCache 最优页 512 达 Strata-IO 的 93%，hit rate 低 2.4%。来源：5_eval.tex L154（§5.3.2）。
- N14：cache distance（图 Figure 11，LooGLE 派生）：min 距离 delay hit 缓解 +42%；I/O 机制 shuffle +76%、max +95%；均衡 batch shuffle +11%、max +12%；stall hiding shuffle +8%、max +3%；max 距离 delay hit 缓解无收益。来源：5_eval.tex L163–168（§5.3.3）。
- N15：GH200（Llama-3.1-8B、LooGLE，图 Figure 13/14）：Strata-IO 持续带宽 40→150 GB/s；完整 Strata-GH 接近 Oracle。来源：5_eval.tex L221–224（§5.4）。
- N16：实验平台：H200 节点（8×H200、NVLink 互联、Intel Sapphire Rapids CPU、1.6TB DRAM、每 GPU PCIe 5.0 x16 上行 64GB/s 峰值）；GH200 节点（H100 + Grace 64 核 ARM、464GB LPDDR5X、384GB/s 上行）。来源：5_eval.tex L4–8（§5.1）。
- N17：基线版本与配置：vLLM v0.8.5；LMCache v0.2.1（chunk 256）；vLLM 页 32；TensorRT-LLM v0.17.0（页 32）；SGLang v0.4.5（SGLang 与 Strata 页 1、SGLang-HiCache 页 32）；SGLang-HiCache 为作者实现（layer-wise 重叠 + cudaMemcpyAsync，对齐 CachedAttention/Pensieve/FlashGen）。来源：5_eval.tex L13–25（§5.1）。
- N18：模型：Llama-3.1-8B-Instruct（128k 窗口）、Qwen2.5-14B-Instruct-1M（1M 窗口）、Llama-3.1-70B-Instruct（128k 窗口，4 GPU 张量并行）；8B/14B 单 GPU。来源：5_eval.tex L29–30（§5.1）。
- N19：数据集统计（Table 1）：LooGLE 平均输入 21613 / 输出 15.60、105 上下文 2410 查询；NarrativeQA 54797 / 13.00、50 / 1461；ReviewMT 17708 / 208.3、100 / 1092；ShareGPT 680.9 / 260.9、200869 查询。来源：5_eval.tex L31–46（Table 1）。Poisson 到达模拟；对话数据集保留轮次依赖、ShareGPT 插 60 秒思考时间（沿 Pensieve 方法）；在途查询上限 128；ShareGPT 限 GPU 内存约 500K token；CPU pinned 内存 1TB（GH200 400GB）；所有基准不用磁盘。来源：5_eval.tex L59–66。
- N20：分层系统达到约 95% cache hit rate（借 CPU 内存）。来源：5_eval.tex L87（§5.2.1）。

## 原图候选（编号｜原文 Figure｜内容｜可说明的结论｜获取途径）

- 图 A｜Figure 1｜Load/Compute Ratio 的 CDF 与 I/O stall 百分比（右轴）｜Q1：74% stall 与优化 I/O 后仍剩 24%｜TeX 源码 figs/cdf_stall.pdf
- 图 B｜Figure 3｜8192 token 加载延迟与带宽利用率（多平台，页 32）｜Q1：22%/5% 带宽利用率｜figs/motivation_poor_utilization.pdf
- 图 C｜Figure 2｜页大小 vs hit rate 与 TTFT（ShareGPT、Mistral-24B）｜Q1：大页代价｜figs/page_size_ttft.pdf
- 图 D｜Figure 4｜Strata 系统架构｜Q2/Q3：Cache Controller 与 Scheduler 两组件及数据流｜figs/strata_system_diagram.pdf
- 图 E｜Figure 5｜I/O kernel 资源配额 vs 干扰｜Q2：2 block 的取舍依据｜figs/io_overhead.pdf
- 图 F｜Figure 6｜layer-first vs page-first 布局对比｜Q2：布局解耦的对象｜figs/layout_difference.pdf
- 图 G｜Figure 7｜调度策略示意（橙/绿/紫/蓝/灰块）｜Q3：delay hit、均衡批、bundle hit、bubble filling 全景｜figs/schedule_diagram.pdf
- 图 H｜Figure 8｜端到端吞吐-TTFT 全景（3 模型 × 4 数据集）｜Q4：N10/N11 的来源｜figs/e2e_bench_all.png（PNG 已可直接用）
- 图 I｜Figure 9｜消融吞吐-延迟曲线｜Q4：C20/N12｜figs/loogle_break.pdf
- 图 J｜Figure 10｜页大小扫描归一化对比｜Q4：C21/N13｜figs/loogle_page_size_scan.pdf
- 图 K｜Figure 11｜各机制在不同 cache distance 下的贡献分解｜Q4/Q5：C22/N14｜figs/incremental_contributions.pdf
- 图 L｜Figure 12｜磁盘加载不同布局延迟对比｜Q2：C11/N9｜figs/loading_latency_comparison.pdf
- 图 M｜Figure 13｜PCIe-5.0 vs Grace-Hopper TTFT 对比｜Q4：C23｜figs/grace_hopper_compare.pdf
- 图 N｜Figure 14｜持续带宽对比｜Q4：N15｜figs/grace_hopper_bandwidth.pdf

选用原则由 outline.md 决定；全部候选获取途径为 TeX 源码包（最高优先级），PDF 图需转 PNG 后 base64 内联。

## 冲突与不确定项

- abstract 说 "up to $5\times$ lower TTFT" 而 §5.2.1 正文说的是 "higher throughput at the same TTFT"（Llama-70B 对 vLLM-LMCache 与 SGLang-HiCache 均 5x）。两者出自同一组实验的不同读法（吞吐-延迟曲线），页面按正文口径表述并注明 abstract 口径，不视为冲突。
- "deployed in production at a leading AI company"：未具名，页面如实转述，不猜测。
- SGLang-HiCache 基线是作者自建（非社区版 HiCache）：页面在基线介绍处明确，避免与 HiCache 博客混淆。
- Strata/SGLang 页大小为 1、分层基线页 32：这是论文的实验配置（§5.1 L25），如实记录，不做额外解释性推断。
