# paged-attention 核心论断与证据

## C 论断

- C1：vLLM 之前的系统为每请求按最大序列长度预分配连续显存存放 KV cache，产生三类浪费：为未来 token 的预留、按最大长度过配的内部碎片、分配器空洞的外部碎片。来源：vLLM 论文（arXiv:2309.06180）§3 Figure 3 及正文（"reserved slots for future tokens, internal fragmentation due to over-provisioning for potential maximum sequence lengths, and external fragmentation from the memory allocator like the buddy allocator"）。
- C2：既有系统实测只有 20.4%–38.2% 的 KV cache 显存真正存了 token 状态。来源：vLLM 论文 §3（"our profiling results in Fig. 2 show that only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems"）。
- C3：PagedAttention 把 KV cache 划分为固定 token 数的块，块不必连续存储；管理方式类比 OS 虚拟内存分页（块-页、token-字节、请求-进程）。来源：vLLM 论文 §2（"PagedAttention divides the request's KV cache into blocks... not necessarily stored in contiguous space. Therefore, we can manage the KV cache in a more flexible way as in OS's virtual memory: one can think of blocks as pages, tokens as bytes, and requests as processes"）。
- C4：分页用相对小的块按需分配缓解内部碎片、消除外部碎片（所有块同大小），并以块粒度支持同请求与跨请求共享。来源：vLLM 论文 §2（"This design alleviates internal fragmentation by using relatively small blocks and allocating them on demand. Moreover, it eliminates external fragmentation as all blocks have the same size. Finally, it enables memory sharing at the granularity of a block"）。
- C5：vLLM 实现按需分配：块在需要时才分配（token 写满当前块才申请新块），逻辑块号到物理块号的映射由块表维护。来源：vLLM 论文 §4.1（block table 与调度器按需分配的描述）。置信：已确认（论文 §4 "PagedAttention... kernel... block table" 相关段落）。
- C6：内部碎片被限制在每请求最后一块内（最多浪费一块，均匀假设下平均半块——推断标注：论文说 "at most one block per request" 类似表述，平均半块为本页推断）。来源：vLLM 论文 §4.1。
- C7：各引擎页大小取值：TensorRT-LLM 32、vLLM 16、SGLang 1 token；每 token KV cache 从几十 KB 到几 MB。来源：Strata 论文（arXiv:2508.18572v2）§2.2（"Typical page sizes are small—e.g., 32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang—where each token may span from tens of kilobytes to several megabytes"）。另 vLLM CUDA GPU 页大小上限 32（Strata §3.1 引 vLLM 文档 "a maximum supported size in vLLM for CUDA GPUs"）。
- C8：分页导致数据碎片化：一个序列的 KV cache 散布在多个不连续页中，跨内存层传输粒度只有几 KB，无法打满 PCIe 带宽。来源：Strata 论文 §1（"paging causes data fragmentation, as the KV cache for a given sequence is spread across multiple non-contiguous pages. This leads to small data transfers, sometimes only a few kilobytes, which fail to saturate PCIe bandwidth"）。
- C9：分页与连续批处理、张量并行等机制兼容。来源：vLLM 论文 §5/§6 实现；SGLang 论文 §3.2（RadixAttention "is compatible with techniques like continuous batching, paged attention, and tensor parallelism"）。

## F 公式

- F1（构造算例用）：页数 $=\lceil \text{token 数}/\text{页大小}\rceil$；内部碎片 $=$ 页数 × 页大小 − token 数（最后一块的空余）。构造示例专用，非论文公式。
- F2（引用 kv-cache 页 F1）：每 token 字节数 $2\cdot L_{\text{layers}}\cdot H_{\text{kv}}\cdot d_{\text{head}}\cdot b$——用于把页大小换算成字节（如 Llama-3.1-8B、页 1 → 每页 128 KB；页 32 → 每页 4 MB）。来源：kv-cache 页。

## N 数字

- N1：既有系统 KV cache 显存有效利用率 20.4%–38.2%。来源：vLLM 论文 §3 Figure 2。
- N2：页大小：TensorRT-LLM 32、vLLM 16、SGLang 1 token（vLLM CUDA 上限 32）。来源：Strata 论文 §2.2、§3.1。
- N3：Llama-3.1-8B 每 token 128 KB（kv-cache 页 N2）；页 32 → 单页 4 MB；20,000 token 序列 → 625 页。来源：kv-cache 页 + 构造算例（标注构造）。
- N4：8192 token（页 32、Llama-3.1-8B）从 CPU 加载仅约 22% PCIe 5.0 理论带宽。来源：Strata 论文 §3.1 Figure 3（本页过渡段引用，完整分析归 Strata 页）。

## 冲突与不确定项

- 无。C6 的"平均半块"为推断标注；页大小默认值随引擎版本可能变化，页面注明"论文引用时的取值"。
