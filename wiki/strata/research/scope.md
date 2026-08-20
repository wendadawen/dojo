# Strata 内容范围

## 1. 论文定位

- 标题：Strata: Hierarchical Context Caching for Long Context Language Model Serving
- 作者：Zhiqiang Xie（Stanford & NVIDIA）、Ziyi Xu（上海交通大学）、Mark Zhao（CU Boulder）、Yuwei An（CMU）、Vikram Sharma Mailthody（NVIDIA）、Scott Mahlke（NVIDIA & 密歇根大学）、Michael Garland（NVIDIA）、Christos Kozyrakis（NVIDIA & Stanford）
- 发表：OSDI 2026（USENIX Symposium on Operating Systems Design and Implementation，Track 1 "KV Cache and Long Context" session）；arXiv:2508.18572v2（2025-08-27 提交的 TeX 源码，与 OSDI'26 会议版一致）
- 链接：https://arxiv.org/abs/2508.18572
- 代码：论文未给出官方代码仓库。一作 Zhiqiang Xie 是 SGLang HiCache（LMSYS 博客 2025-09-10）作者，HiCache 与 Strata 机制高度相关，但论文本身未声明开源对应版本，页面不以 HiCache 源码作为论文论断依据
- 版本固定：本页全部论断以 arXiv TeX 源码 v2（下载于 /tmp/strata-research/src，文件时间戳 2025-08-26/27）为准；arXiv HTML 版不存在，图编号按 LaTeX 源码中 figure 环境出现顺序并经 ar5iv 确认为 Figure 1–14

简要说明：论文用「GPU-assisted I/O + 布局解耦 + cache-aware 调度」解决长上下文 LLM serving 中把 KV cache 从 CPU/存储层搬回 GPU 时的 I/O 瓶颈（loading-bound）问题。

论文宣称的贡献（与原文 abstract 与 §1 一致）：

1. GPU-assisted I/O：用 CUDA kernel 代替 cudaMemcpyAsync 应对小页碎片化传输，提高并发与带宽利用率（§4.2）
2. GPU 与 CPU/存储内存布局解耦：GPU 保持 layer-first 计算友好布局，CPU/存储用 page-first 传输友好布局，转换在 I/O kernel 中近乎零开销完成（§4.2.1）
3. Cache-aware 请求调度：delay hit 延迟执行、均衡 batch 构建、bubble filling 填充 I/O 空泡（§4.3）
4. 基于 SGLang 实现、已在生产环境部署；长上下文基准上 TTFT 相比 vLLM+LMCache 最多降 5 倍、相比 TensorRT-LLM 提速最多 3.75 倍，短上下文无退化（abstract、§5）

论文没做什么（每项有排除依据）：

- 不做 KV cache 压缩/量化/近似缓存：相关工作明确说明 CacheGen/CacheBlend 属于"approximate caching schemes"，Strata "does not impact the accuracy of requests"（§6）
- 不做大规模分离式 KV cache 池与全局协调器：那是 Mooncake/MemServe 的方向；Strata 关注"memory management and scheduling within single compute instances"（§6）
- 不做 P-D 分离部署：Strata 用 P-D co-location（同一 GPU 时间上交替执行 prefill/decode batch，§4.1）；bubble filling 一节顺带指出插入 prefill 更适用于 P-D 分离系统，但本文实验不是分离式（§4.3.3）
- 不做稀疏注意力、模型结构改动：论文不涉及
- 不做 KV cache 加载的重计算替代方案分析之外的调度理论研究：重计算 vs 加载的取舍只在动机里引用了 jin2024compute

相邻工作（容易混淆的方法，记录关键区别）：

- LMCache（vLLM 的分层缓存扩展）：基线之一，用 cudaMemcpyAsync 传输；vLLM-LMCache 基线版本 v0.8.5 + LMCache v0.2.1，chunk 256、page 32
- TensorRT-HiCache：TensorRT-LLM v0.17.0 的 CPU offload，page 32
- SGLang-HiCache：作者自建的基线（SGLang v0.4.5 + layer-wise 传输重叠 + cudaMemcpyAsync），与 CachedAttention/Pensieve/FlashGen 的做法一致；注意这不是 SGLang 社区版 HiCache
- CachedAttention / Pensieve / FlashGen：layer-wise 重叠加载与计算的一系工作，Strata 动机部分论证该假设在长上下文下失效
- Mooncake / MemServe：分离式全局内存池，与 Strata 正交可集成

## 2. 核心问题

### Q1：把缓存的 KV cache 从 CPU 搬回 GPU 为什么会成为主要瓶颈？

- 预期答案：两个来源。其一，PagedAttention 分页布局为 GPU 显存管理而设计（页小至 1–32 token），导致一个序列的 KV cache 散布在大量不连续页中，传输粒度只有几 KB，无法打满 PCIe（8192 token 的加载只达理论带宽 22%，GH200 上低至 5%；打满 PCIe 5.0 需 1–2 MB 传输）。其二，现有调度器假设 prefill 计算足以掩盖加载延迟，长上下文（加载量大、新 token 少）下该假设失效，系统变成 loading-bound（LooGLE 上 74% prefill 时间阻塞在 I/O，吞吐降 4 倍）。加大页可以改善 I/O 但 hit rate 与 TTFT 显著恶化（页 1→1024，平均 TTFT 最多 2 倍、P90 2.9 倍），存在根本性权衡。
- 重要性：论文全部设计针对这两个来源；不理解它们就无法评价 Strata 的机制选择。
- 依赖内容：KV cache 存储量随 token 线性增长（kv-cache 页）、PagedAttention 分页（paged-attention 页）、前缀缓存 hit rate（prefix-caching 页）、Little's Law（论文 §3.1 给出，页面内推导）。

### Q2：GPU-assisted I/O 如何让小页传输打满带宽，代价是什么？

- 预期答案：不再反复调用 cudaMemcpyAsync（受 CPU 侧并发与驱动队列限制），而是启动一个 CUDA kernel，数千线程各自把一小块数据从源（GPU 显存或 CPU pinned memory）读入寄存器再写出。优点：并发 C 从 CPU 的数十提高到 GPU 的数千；有效粒度低至 128 字节，单页 KB 级传输也高效；kernel 内可做任意轻量计算，布局转换几乎免费。代价与控制：I/O kernel 与计算 kernel 抢资源（寄存器、执行周期、cache 污染），Strata 用少量大 block（2 个 1024 线程 block）诱使硬件调度器把 I/O 限制在极少数 SM，配合绕过 cache 的指令；实测 50 GB/s 吞吐下 prefill 性能损失 <5%、decode 10%，默认 2 block 加载 + 1 block 备份。布局解耦：GPU 保持 layer-first、host/存储用 page-first，线程只多做一次地址算术即可在途转换；磁盘加载 8192 token 延迟最多降 4 倍。
- 重要性：这是论文数据面核心机制，回答 Strata"怎么把带宽用起来"。
- 依赖内容：CUDA kernel/block/SM 结构（gpu-execution-model 页）、GPU 互联与带宽（gpu-communication 页）、KV cache 布局（本页讲，paged-attention 页作前置）。

### Q3：cache-aware 调度如何避免 loading-bound 与 delay hit？

- 预期答案：三阶段。① delay hit 延迟执行：HiRadixTree 引入 transient node（in-queue / in-flight 两种标记），匹配到 transient node 的请求推迟一轮、置于队首，避免对同一未就绪上下文重复 prefill；阈值默认 100 个匹配 token。② 均衡 batch：按 load/compute 比例（默认阈值 100）判断加入请求是否使 batch loading-bound；不 bound 则加入，并优先吸收与其 bundle hit（共享上下文）的请求；bound 则进降级列表，批不满时再补。③ bubble filling：仍 loading-bound 时推迟 prefill、插入 decode batch 与加载并行——decode 饱和 HBM 带宽、加载饱和 PCIe 带宽，两者资源基本不冲突。
- 重要性：这是论文控制面核心机制；I/O 机制解决"搬得快"，调度解决"搬的同时 GPU 不闲着、不重复搬"。
- 依赖内容：Q1 的 loading-bound 概念、Q2 的带宽资源区分、batch/prefill/decode 流程（页面内以贯穿示例建立）。

### Q4：Strata 的收益有多大，在什么条件下成立？

- 预期答案：H200、三模型四数据集。LooGLE 上同 TTFT 吞吐提升：Llama-8B 对 SGLang-HiCache / vLLM-LMCache / TensorRT-HiCache 分别 3.2x / 2.6x / 1.9x，Qwen-14B 为 3.9x / 2.1x / 1.9x，Llama-70B 为 5x / 5x / 3.75x；ReviewMT（Llama-8B）2.3x / 2.3x / 1.7x；NarrativeQA 预热稳态对 vLLM-LMCache 2.3x / 2.6x / 2.5x。abstract 的 "up to 5x lower TTFT vs vLLM+LMCache、3.75x vs TensorRT-LLM" 即来自 Llama-70B 组。条件：长上下文 prefill 主导、分层缓存已必要（非分层系统在 LooGLE 上迅速耗尽显存）；分层数 95% hit rate 依赖 1TB pinned DRAM 配置；disk 未纳入端到端实验（基线不支持）。消融：调度单独 1.8x、I/O 单独 2.3x 峰值吞吐（相对 SGLang-HiCache）；低请求率下调度收益更大，高请求率下 I/O 机制成为关键。页大小：SGLang-HiCache 最优页 512 也只有 Strata-IO 的 93% 且 hit rate 低 2.4%——Strata 免去页大小调参。GH200：I/O 机制把持续带宽 40→150 GB/s；仅有 I/O 机制（Strata-IO-GH）仍不敌 H200 上完整 Strata，调度是吃满新硬件的必要条件。
- 重要性：核心结论章；读者需要知道数字与条件，避免"5x 无条件成立"的误读。
- 依赖内容：Q1–Q3 的机制、实验设置（模型/数据集/基线版本）。

### Q5：Strata 不解决什么问题，适用边界在哪里？

- 预期答案：短上下文无退化（ShareGPT，Strata 与其他系统相当；注意底层 SGLang 引擎本身在 Llama-8B/70B 上因 kernel 差异略有劣势，Strata 未消除引擎差距）。单机内问题：不做全局分布式缓存池（Mooncake/MemServe 方向）、不依赖高速网络；可以与它们集成传输引擎但论文未实验。不做近似/压缩缓存。P-D 分离系统不是本文实验对象。cache distance 实验给出各机制的收益随负载模式变化：最小距离（高局部性）时 delay hit 缓解 +42%、无需分层；最大距离时 I/O 机制 +95%、调度机制收益小——机制收益与负载模式相关。
- 重要性：评价章与误解处理的依据；直接服务"方法评价"章。
- 依赖内容：Q1–Q4。

## 3. 内容分级

核心内容（缺一则核心问题无法完整回答）：

- KV cache 分层缓存的必要性与容量数字（40GB HBM ≈ 0.3M token，Llama-8B）→ Q1
- 两个瓶颈来源：碎片化小传输（22%/5% 带宽、1–2MB 才能打满）与调度器忽略加载延迟（74% stall、4x 吞吐损失）→ Q1
- 大页的代价实验（hit rate 下降、TTFT 2x/2.9x）→ Q1
- Little's Law 及其推论 X = C·S/L → Q1、Q2
- GPU-assisted I/O 机制（kernel、128 字节粒度、三优点）→ Q2
- 干扰实测与控制（2 block、50GB/s、<5%/10% 降速、默认配额）→ Q2
- 布局解耦（layer-first vs page-first、一次地址算术、4x 磁盘延迟降低）→ Q2
- HiRadixTree/transient node/delay hit 延迟执行（阈值 100）→ Q3
- 均衡 batch 算法（load/compute 比例 100、bundle hit、防饿死）→ Q3
- bubble filling（decode 填空泡、HBM/PCIe 区分）→ Q3
- P-D co-location 概述 → Q3
- 端到端数字全套（含基线版本与实验条件）→ Q4
- 消融与 cache distance 归因 → Q4、Q5
- 页大小无关性与 GH200 结果 → Q4、Q5
- 短上下文无退化及其条件 → Q5

辅助内容（消除理解障碍或澄清误解）：

- 术语速查表（TTFT、prefill/decode、pinned memory、DMA 等）
- delay hit 现象的排队论来源（网络社区的既有概念）
- SGLang-HiCache 基线的含义（作者自建、对齐 CachedAttention/Pensieve/FlashGen 做法）——避免读者以为是社区版
- FlashGen 的 re-order 执行调度已被 SGLang 实现并用于基线设置
- Strata 生产部署声明（"a leading AI company"，未具名）

扩展内容：

- 纳入：CUDA Graph 相关性不做展开；ROCm/AMD 兼容一句话（§4.2 提到 ROCm 后端兼容）
- 排除：CacheGen/CacheBlend 的 KV 压缩细节（另一独立工作，§6 一句话区别足够）
- 排除：Mooncake/MemServe 的全局架构细节（另一方向，§6 关系一句话）
- 排除：SSD 存储层的预取细节（§4.2.1 有描述，端到端实验未用 disk；正文一段讲机制、标注实验未覆盖）
- 排除：GH200 硬件架构细节（LPDDR5X/NVLink-C2C 规格，论文给数字即可）

## 4. 前置知识

| 前置概念 | 被哪些核心内容依赖 | 概念页 |
|---|---|---|
| KV cache 与 prefill/decode 两阶段、存储量公式 | Q1 容量数字、Q2 传输对象、贯穿示例 | `wiki/kv-cache/`（递归生成，concept 流程） |
| PagedAttention 分页管理、页大小、逻辑/物理分离 | Q1 碎片化来源、Q2 布局、Q4 页大小实验 | `wiki/paged-attention/`（递归生成） |
| 前缀缓存、radix tree、hit rate | Q1 大页代价、Q3 delay hit/bundle hit、Q4 hit rate 数字 | `wiki/prefix-caching/`（递归生成） |
| CUDA kernel/block/SM、线程层次 | Q2 I/O kernel 机制、SM 隔离 | `wiki/gpu-execution-model/`（已有） |
| GPU 互联、PCIe/NVLink、带宽 | Q1/Q2/Q4 带宽数字 | `wiki/gpu-communication/`（已有） |
| 标准注意力（K/V 矩阵是什么） | KV cache 定义的底层 | `wiki/standard-attention/`（已有，kv-cache 页引用） |

递归生成的三个概念页在 Strata 页面写作前完成各自质检。

## 5. 明确不展开的内容

- KV cache 量化/压缩（CacheGen、KIVI 等）：属于近似缓存方向，论文明确不做且声明不影响精度；展开会偏离页面主题。
- Mooncake/MemServe 分布式架构：属于跨实例全局缓存方向，与本文单机内存管理正交；论文只声明可集成。
- PCIe/NVLink 硬件协议细节：gpu-communication 页已有互联内容，本页只使用论文给的带宽数字。
- TensorRT-LLM/vLLM/SGLang 引擎自身的 kernel 差异：论文在短上下文实验提到 SGLang 底层引擎略弱，本页如实转述，不展开引擎实现。
- 延迟命中（delay hit）在网络社区的原始研究：论文引用 delayhit 文献，本页只解释现象本身与 LLM 场景的对应，不展开网络排队论文。
- KV cache 传输的 RDMA 路径：论文不涉及（单机内 PCIe/NVLink 场景）。

## 6. 常见误解和适用边界

误解：

1. "Strata 比 vLLM+LMCache 快 5 倍"无条件成立——实际是 Llama-70B LooGLE 上同 TTFT 吞吐 5x（abstract 的 5x lower TTFT 同源）；Llama-8B 是 2.6x，ReviewMT 2.3x；且条件为长上下文 prefill 主导负载 + 1TB pinned DRAM。形成原因：二手报道普遍只引 abstract 的 up to 数字。影响 Q4。
2. GPU-assisted I/O 没有代价——实际 I/O kernel 与计算 kernel 竞争寄存器/执行周期/cache，需要 SM 隔离与绕过 cache 的设计；配 2 block 时 prefill 仍有 <5% 性能损失、decode 10%。影响 Q2。
3. "加大页就能解决带宽问题"——实验显示页 1→1024 时 hit rate 显著下降，平均 TTFT 最多 2x、P90 2.9x 恶化；页大小是传输效率与缓存收益的权衡，不是自由旋钮。影响 Q1、Q4。
4. Strata 是 PD 分离系统——不是。P-D co-location 指同一 GPU 时间上交替执行 prefill 与 decode batch；bubble filling 反而提到插 prefill 更适合 P-D 分离系统，但本文实验非分离式。影响 Q3。
5. 分层缓存各系统都能拿到 95% hit rate，所以差异全在传输——hit rate 95% 是分层系统在该配置下的结果，但页大小等配置会改变 hit rate（页 512 比页 32 低 2.4%），SGLang-HiCache 最优配置也输在 hit rate 与带宽两端。影响 Q4。
6. Strata 解决了 KV cache 显存不够的根本问题——它解决的是"搬运效率与调度"，显存容量扩展靠分层（CPU DRAM/SSD）本身是已有做法；容量是否够取决于 CPU DRAM 配额（实验 1TB/400GB）。影响 Q1、Q5。

适用边界：

- 方法解决：单机内 GPU↔CPU（及存储）KV cache 搬运带宽利用、加载与计算的资源平衡、重复 prefill。
- 不解决：跨实例全局缓存协调（Mooncake 方向）、KV 压缩（近似缓存方向）、显存容量本身的物理限制、模型结构（稀疏注意力）。
- 结论成立条件：长上下文（prefill 主导）负载；分层缓存已必要（短上下文上各系统差异本就小）；实验用 pinned DRAM 1TB（GH200 400GB）；基线版本固定（vLLM v0.8.5、LMCache v0.2.1、TRT-LLM v0.17.0、SGLang v0.4.5）；disk 未纳入端到端对比。
- 条件不满足时：短上下文上收益趋近于零（但无退化）；cache distance 最小（高局部性）时分层缓存本身不必要、delay hit 缓解贡献最大（+42%）；最大距离时 I/O 机制贡献最大（+95%）而调度贡献小。
- 未覆盖场景：P-D 分离部署、多机分布式、AMD（ROCm 兼容仅声明未实验）、disk 端到端。

## 7. 论断分级

- 论文明确声称：§1–§7 全部机制描述、实验数字、abstract 结论（见 evidence.md 编号）。
- 文献已有结论：PagedAttention 分页机制与碎片（vLLM SOSP'23）；RadixAttention/radix tree 前缀缓存（SGLang NeurIPS'24）；Little's Law（排队论经典，论文内引用）；delay hit 现象（网络社区 delayhit 文献，论文引用）；层间重叠假设（CachedAttention/Pensieve/FlashGen）；Llama-3.1-8B 模型配置（Meta 模型卡，用于 KV cache 手算，属外部事实）。
- 基于证据的推断：无官方开源版本对应（HiCache 博客为一作相关工作但论文未声明对应）；"调度与 I/O 机制收益随请求率此消彼长"由 Fig 9 曲线趋势支撑（论文原话是定性描述，页面引用原话）。
- 缺失假设的猜测：无。生产部署公司未具名（论文原文如此），页面如实标注不猜测。
