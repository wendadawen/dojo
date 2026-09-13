<!-- review-meta
round: 6
page: wiki/strata/index.html
reviewed_content_sha256: 323d8ef57589f7c1
-->
# Strata 审查记录（第 6 轮）

- 页面版本：6077d1e36e6d88f632f0e9bfe6622c570ff798fe（index.html 工作树哈希）
- 论文版本：arXiv:2508.18572v1（2025-08-26 提交的 TeX 源码；会议版 OSDI 2026，USENIX）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复；本轮未读取 research/ 下任何文件）
- 已完整阅读章节（按顺序）：核心问题（learning-goals）与术语表 → 1 瓶颈定位（1.1 容量账／1.2 瓶颈一／1.3 大页不是出路／1.4 瓶颈二／本章问题）→ 2 GPU-assisted I/O（2.1 机制／2.2 干扰控制／本章问题）→ 3 布局解耦与存储层（3.1 两种布局／3.2 在途转换／3.3 存储层预取／本章问题）→ 4 Cache-aware 调度（4.1 系统全景／4.2 延迟命中／4.3 均衡 batch／4.4 Bubble filling／本章问题）→ 5 实验验证（5.1 设置／5.2 端到端／5.3 消融／5.4 页大小与 cache distance／5.5 GH200／5.6 短上下文／本章问题）→ 6 方法评价（6.1 优点／6.2 局限／6.3 适用场景／本章问题）→ 来源与范围说明。全文含全部折叠块、图注与自绘 SVG。

## 问题

- [重要·技术] 第 1 章「1.1 分层缓存成为标配——容量账」及贯穿示例、第 1 章本章问题、来源说明（index.html 第 72／143／149／203／624 行）：Llama-3.1-8B 20,000 token 手册缓存被写成约 2.56 GB，据此得出「占 40 GB HBM 的 6.4%」，并把 40 GB HBM 容量写成约 0.31M token。这组数字与本页明确声明「定义沿用」的前置页 KV cache 的换算口径直接矛盾。｜引文依据：本页来源说明写「F3（KV cache 每 token 字节数）见 KV cache F1，定义沿用如下：$2\cdot L_{\text{layers}}\cdot H_{\text{kv}}\cdot d_{\text{head}}\cdot b$，Llama-3.1-8B 32 层/8 KV 头/head_dim 128/bf16 得 128 KB/token，20,000 token 手册约 2.56 GB」。而被引用页 KV cache 明示换算基准并给出同一对象的结果：「本文按 $1\,\text{KB}=1{,}024$ 字节、$1\,\text{GB}=2^{30}$ 字节换算。按此口径 128 KB/token 下，20,000 token 手册约 2.44 GB」，并给出「代入公式 $2\times 32\times 8\times 128\times 2=131{,}072$ 字节 = 128 KB/token。20,000 token 的手册共 $20{,}000\times 131{,}072\,\text{B}\approx 2.44\,\text{GB}$」与「$40\times 2^{30}\,\text{B}/131{,}072\,\text{B}\approx 0.33\text{M}$」。本页 F3 复算结果为 131,072 B/token，在任一统一口径下都得不出 2.56 GB（二进制口径 2.44 GiB、十进制口径 2.62 GB）；2.56 只可能来自把「128 KB」当作 128,000 字节后乘以 20,000，即先把 131,072 B 向下取整为 128,000 B。同一对象在前置页为 2.44 GB／0.33M、在本页为 2.56 GB／0.31M。｜修复要求：与前置页 KV cache 统一换算口径。把第 72／143／149／203／624 行的「2.56 GB」改为「约 2.44 GB」，「40 GB HBM…约 0.31 M」改为「约 0.33M」，并把由 2.56/40 得出的「6.4%」改为由 2.44/40 得出的「约 6.1%」（第 149、203 行）；若坚持采用十进制口径，则须在页内显式声明「1 KB=1000 字节、1 GB=10^9 字节」并同步把复算值改为 2.62 GB，二者取一，不得保留现写法。｜修复：｜复验：
- [轻微·可读性] 第 1 章「1.4 瓶颈二：调度器假设失效与 delay hit」末段末句（index.html 第 193 行）：把因果链压进长名词短语，指代与语义都需回读。原句「这是分层缓存后命中率虽高（实测约 95%）但"缓存还没准备好"的请求也能命中瞬时不到位的缓存的来源。」中，「命中…缓存」与「瞬时不到位的缓存」语义相抵，读者难以还原「高命中率只对已就绪缓存成立、缓存尚未算完时仍会触发多余 prefill」这一本意。｜引文依据：不适用（表述问题）。｜修复要求：拆成直陈句，例如「命中率的统计只覆盖已就绪的缓存；缓存尚在计算中的请求仍会落空，这正是 delay hit 的来源」，删除「…的来源」式长名词化结构。｜修复：｜复验：

## 本轮核对摘要（原文片段，供复验）

- 图注与正文页大小取值：原文 Figure 2 caption「benchmarked on H200 for Mistral-24B using the ShareGPT dataset」、Figure 3 caption「8192 tokens (using page size 32) of Llama-3.1-8B from CPU to GPU」、Figure 12 caption「different models from a local disk to CPU memory」——与本页 1.3／1.2／3.2 图注一致。
- 带宽数字：原文「achieves only approximately 22% of the theoretical PCIe 5.0 bandwidth」「falling to as low as 5% on … Grace-Hopper」「74% of prefill time is blocked on KV transfers … up to a 4× throughput reduction」「the granularity required for efficient GPU-assisted I/O is only 128 bytes on most architectures」——与 1.2、2.1 一致。
- 公式与阈值：原文「C=λ⋅L」「X=λ⋅S」「X=C⋅S/L describing attainable throughput in a stable state」「achieving 75-80% … necessitates transfer sizes (S) … (i.e., 1-2MB)」；「two CUDA blocks of 1024 threads each … nearly 50 GB/s … less than 5% performance degradation on prefill and 10% on decoding」「keeping overall performance impact under 5%」；阈值「default threshold of 100 active token matches」「a default ratio of 100, corresponding to the point where stalls begin to appear」——与 1.2、2.1、2.2、4.2、4.3 一致。
- 端到端与消融：原文 LooGLE「up to 3.2×, 2.6×, and 1.9×」「3.9×, 2.1×, and 1.9×」「5×, 5×, and 3.75×」；ReviewMT「outperforms vLLM-LMCache by 2.3×, TensorRT-HiCache by 2.3× and SGLang-HiCache by 1.7×」；NarrativeQA「up to 2.3×」「2.6×」「2.5×」；消融「Strata-IO … up to 2.3× higher peak throughput」「Strata-Schedule-Only … up to 1.8×」；页大小「93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate」——与第 5 章表格与正文一致。
- cache distance：原文「minimal cache distance, there is no need for hierarchical caching due to the perfect locality」「42%」「76%」「95%」「11%/12%」「8% and 3% for shuffle and max cache distance, with higher benefit to the shuffle pattern due to higher variance」——与 5.4 一致。
- GH200：原文「up to 384 GB/s of memory bandwidth (unidirectional)」「6x higher peak bandwidth」「from 40 to 150 GB/s」——与 5.1、5.5 一致。
- HiRadixTree 元数据字段：原文正文只说「serving as a page table and stores metadata about each KV cache page」，未列字段；核对原文 Figure 4 矢量图（strata_system_diagram.svg）内部文字为「Token_ids: { GPU_indices, CPU_indices, hit_count, … }」——本页 4.1「存各页元数据（GPU 索引、CPU 索引、hit 计数等）」由图 4 直接支持，无误。
- 其他均已回源核对成立：SGLang-HiCache 系作者自建（「in line with prior work including CachedAttention, Pensieve and FlashGen」）、ROCm 兼容 AMD（「With the ROCm backend … compatible with AMD GPUs」）、Intro 部署表述（「deployed in production environments at a leading AI company」）、NarrativeQA 预热流程与 TensorRT 不支持预热、ShareGPT 60 秒思考时间／在途 128／500K token 限显存——第 5、6 章表述与原文一致。
- 前置概念链接 kv-cache／paged-attention／prefix-caching／standard-attention／gpu-execution-model／gpu-communication／kv-cache-layout／dualpath 均真实存在；1.4 节对 DualPath 的转引（agentic 命中率 ≥95%、prefill 由计算密集退化为 I/O 密集、新增第二条 KV 加载路径）与 DualPath 页表述一致。
- 无 alt/aria-label 内含 `$...$`；`.dojo/scripts/validate.py wiki/strata/index.html` 返回 `validation ok`；自绘结构图为内联 SVG，公式均 KaTeX 渲染。表述维度逐段通读未发现元话语、会话指代（我/我们/你）、调试叙事或临场评价。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 1
- 处置：修复（改第 72／143／149／203／624 行容量数字与口径，使之与前置页 KV cache 一致；并改写 1.4 节末句）