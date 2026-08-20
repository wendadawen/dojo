# PagedAttention 审查记录（第 2 轮）

- 页面版本：index.html 工作树哈希 `a93c0579c9614937b5fd2ef48c18a6ae72898b01`；overview.html 工作树哈希 `b373b4493caca5df24dcfa58064a2d4adb2b2880`
- 审查时间：2026-08-20 11:53
- 审查者：独立子代理（reviewer-paged-attention-2）
- 已完整阅读章节（按顺序）：index.html 头部元信息与导语、核心问题（含 4 个解答折叠块）、1. 连续预留的三类浪费（含 A/B 例与本章问题）、2. 分页机制——块、页表与按需分配（含流程图示、A/B 页 16 完整账目折叠块与本章问题）、3. 页间不连续——为显存设计，被传输惩罚（含页大小上限折叠块与本章问题）、4. 页大小怎么选——一个没有免费午餐的旋钮（含权衡表与本章问题）、来源与范围说明全部五小节；overview.html 全文。外部来源：vLLM 论文全文（§1–§10）、Strata 论文 TeX 源码全部章节（§1–§7，含 3_design.tex 与 5_eval.tex）。未读取 research/ 下任何其他文件。

## 问题

- [重要·技术] index.html §3 折叠块「补充：页大小上限与可配置性」：「Strata 论文做分层缓存实验时统一取页 32」与来源不符：Strata 自身（\oursys）与 SGLang 在评估中取默认页 1，仅三个分层缓存基线取 32；且该表述与第 4 章「摘掉传输约束后小页方案整体占优」（Strata-IO 即用小页）存在阅读张力｜引文依据：5_eval.tex §5.1 Baselines "We set the page size for \textit{SGLang} and \oursys to 1 (SGLang's default), and the page size for \textit{SGLang-HiCache} to 32 to be consistent with other hierarchical cache baselines"（另 vLLM-LMCache "page size was set to 32"、TensorRT-HiCache "page size also set to 32"）｜修复要求：将「统一取页 32」限定为「分层缓存基线（vLLM-LMCache、TensorRT-HiCache、SGLang-HiCache）统一取页 32」，并注明 Strata 自身与 SGLang 取默认页 1；或删除「统一」改为「分层缓存基线取页 32」｜修复：｜复验：
- [轻微·格式] index.html 来源与范围说明·核心公式 F1 行：「页数$\times$页大小−token 数」中的数学负号「−」（U+2212）位于 $...$ 之外，违反「数学符号全部由 KaTeX 渲染、正文无 Unicode 数学字符直接出现」｜引文依据：不适用（机械检查：U+2212 定位于该行，处于两个数学环境之间）｜修复要求：整式写入数学环境（如 $\text{页数}\times\text{页大小}-\text{token 数}$）或改为文字表述（如「页数×页大小减 token 数」改为「页数乘页大小再减 token 数」的文字描述）｜修复：｜复验：
- [轻微·技术] index.html §4 本章问题 2 解答：「小页传输也能打满带宽」超出来源口径，来源仅称无论页大小都保持高 I/O 效率，未声称「打满」带宽｜引文依据：5_eval.tex §5.3.2 "with the GPU-assisted I/O mechanisms described in \S\ref{sec:design_io}, \oursys achieves consistently high I/O efficiency regardless of page size"｜修复要求：改为「小页传输也能保持高 I/O 效率」或「传输效率不再依赖页大小」（与同页核心问题 4 解答、§4 正文的既有口径一致）｜修复：｜复验：
- [轻微·技术] index.html §3 正文与来源说明 N3：20,000 token 手册计为「2.5 GB」，而 N3 声称「每 token 字节数经 KV cache 一文核实」，该文在声明二进制换算下同一手册为「约 2.44 GB」，两页口径不一致且本页未注明换算口径（625×4 MB=2500 MB≈2.5 GB 的读法依赖十进制 MB）｜引文依据：kv-cache 页「本页 KB/GB 按存储行业惯例换算：$1\,\text{KB}=1{,}024$ 字节、$1\,\text{GB}=2^{30}$ 字节。按此口径 128 KB/token 下，20,000 token 手册约 2.44 GB」；Strata 页现为「手册缓存约 2.5 GB」｜修复要求：统一口径——本页与 Strata 页均改为「约 2.44 GB」，或在两页注明十进制换算口径；正文「该是 2.5 GB 还是 2.5 GB」随口径同步修改｜修复：｜复验：

## 来源核对记录（逐条引文依据）

- C1（连续预分配与三类浪费）：vLLM §3.1 "they pre-allocate a contiguous chunk of memory with the request's maximum length (e.g., 2048 tokens)"；"three primary sources of memory wastes: reserved slots for future tokens, internal fragmentation due to over-provisioning for potential maximum sequence lengths, and external fragmentation from the memory allocator like the buddy allocator"。页面三类定义及「内部碎片生成结束后才确认」的时序区分与 "Internal fragmentation also remains unused, but this is only realized after a request has finished sampling" 一致。通过。
- C2（20.4%–38.2%）：vLLM §1 "our profiling results in Fig. 2 show that only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems"；§3.1 "as low as 20.4%"。页面补充说明图例 "External frag. & Others" 与 Figure 2 图例一致。通过。
- C3（OS 类比）：vLLM §1 "one can think of blocks as pages, tokens as bytes, and requests as processes"；§4.2 "analogous to the virtual memory in operating systems"。页面「块-页、token-字节、请求-进程」对应准确。通过。
- C4（按需分配缓解内部碎片、同大小页消除外部碎片、页级共享）：vLLM §1 "This design alleviates internal fragmentation by using relatively small blocks and allocating them on demand. Moreover, it eliminates external fragmentation as all blocks have the same size. Finally, it enables memory sharing at the granularity of a block"；§4.2 "a new physical block is only allocated when all previous blocks are full"。通过。
- C5（页表与按需分配）：vLLM §4.2 "The KV block manager also maintains block tables—the mapping between logical and physical KV blocks of each request"；§4.3 逐步示例（页满才分配新物理块 3）。通过。
- C6（内部碎片至多一页/请求）：vLLM §4.2 "vLLM limits all the memory wastes for a request within one block"。页面「至多一页」为论文口径，「平均约半页」已在正文与来源说明标注为均匀假设下的推断。通过。
- C7（页大小 32/16/1、每 token 几十 KB 到几 MB、vLLM CUDA 上限 32）：Strata §2.2 "Typical page sizes are small—e.g., 32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang—where each token may span from tens of kilobytes to several megabytes"；§3.1 "a value recommended in prior works on hierarchical KV cache and a maximum supported size in vLLM for CUDA GPUs"。vLLM 默认 16 另有 vLLM §7.2 "vLLM sets its default block size as 16" 佐证。通过。
- C8（分页导致数据碎片化、小传输打不满 PCIe）：Strata §1 "paging causes \textit{data} fragmentation, as the KV cache for a given sequence is spread across multiple non-contiguous pages. This leads to small data transfers, sometimes only a few kilobytes, which fail to saturate PCIe bandwidth"。通过。
- N1：同 C2，Figure 2（§1 与 §3.1 均引用）。通过。
- N2：同 C7 前半（Strata §2.2）。通过。
- N3（构造算例 20,000 token→625 页×4 MB）：20,000/32=625、32×128 KB=4 MB 复算正确；Llama-3.1-8B 128 KB/token 与 kv-cache 页 $2\times 32\times 8\times 128\times 2=131{,}072$ 字节一致。标注为构造算例，未写成来源结论。GB 口径问题见问题第 4 条。
- N4（8192 token 约 22% PCIe 5.0、GH200 约 5%）：Strata §3.1 "transferring KV cache data for 8192 tokens, achieves only approximately 22% of the theoretical PCIe 5.0 bandwidth... falling to as low as ~5% on systems like NVIDIA's Grace-Hopper platform that replaces PCIe with NVLink"。定位标注「§3.1 Figure 3」与 fig:loading（第 3 幅图、§3.1 内）一致，图注含 "using page size 32... Llama-3.1-8B"。通过。
- N5（SGLang-HiCache 最优页 512 吞吐为 Strata-IO 93%、命中率低 2.4%）：Strata §5.3.2 "Even at its best-performing setting (page size 512), \textit{SGLang-HiCache} achieves only 93\% of \textit{\oursys-IO}'s performance, primarily due to a 2.4\% lower cache hit rate"。定位准确：§5.3.2 即 "Can \oursys alleviate the burden of choosing a page size?"；条件标注 Qwen-14B、H200、LooGLE 与 §5.3 开头 "All analyses presented here were conducted using the Qwen-14B model on an H200 platform" 及 loogle_page_size_scan 图一致。页面解读「摘掉传输约束后小页方案整体占优」与该节结论相符。通过。
- N6（GPU-assisted I/O 高效粒度 128 字节）：Strata §4.2 "the granularity required for efficient GPU-assisted I/O is only 128 bytes on most architectures"。定位准确（§4.2 即 Efficient KV Cache I/O）。通过。
- 第 1 轮修复验证：① 权衡表「平均约半页，至多一页」与 C6 一致，「平均」已在 §2 本章问题 2 解答与来源说明标注为推断——通过；② 引擎动机归因已标注「此为推断，非来源结论」（§4 正文）——通过；③ 「每 token 从几十 KB 到几 MB」与 Strata "tens of kilobytes to several megabytes" 口径一致——通过；④ 「利用率从 30% 提到 96%，可用缓存约 3.2 倍」复算 96/30=3.2 正确，30% 落在实测区间 20.4%–38.2% 内、96% 取自 A/B 构造例——通过。
- A/B 例子复算：连续预留 A 浪费 2048−100=1948、B 浪费 512−500=12、合计 2560 槽装 600 真实数据、600/2560≈23% 均正确；分页后 ⌈100/16⌉=7 页=112 槽、碎片 12，⌈500/16⌉=32 页=512 槽、碎片 12，合计 624 槽、600/624≈96% 均正确；与 vLLM Figure 3 的 A（上限 2048）/B（上限 512）结构一致且已标注为构造。通过。
- 跨页引用核对：kv-cache 页第 4 章确含「逐出或分层（把不常用的缓存搬到 CPU 内存或 SSD，用时再搬回）」，本页 §3「分层存储的常规操作，见 KV cache 第 4 章」指向恰当；prefix-caching 页第 4 章为「命中率——页大小是隐藏的自变量」，本页两处「前缀缓存第 4 章」指向恰当。通过。
- 问题块核对：页面级「核心问题」4 题与四章「本章问题」各 2 题均有解答折叠块，答案独立可读、与正文结论一致，核心问题答案均指明所在章节。通过。
- 机械项：正文（不含脚本）$ 共 32 个、逐行均为偶数，配对完整；raw「×」「→」等数学符号均只出现在 $...$ 内（唯一例外见问题第 2 条）；概念链接（kv-cache、prefix-caching、strata、../../index.html、overview.html、index.html）全部存在且双向互链；KaTeX/Prism 本地资源存在；`.dojo/scripts/validate.py` 对 index.html 与 overview.html 均返回 validation ok。通过。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复


## Round 2 修复记录

| 编号 | 问题 | 修复 | 复验 |
|---|---|---|---|
| 重要 #1 | "Strata 分层实验统一取页 32"不准确 | 改"分层缓存基线统一取页 32；Strata 自身与 SGLang 基线用默认页 1（§5.1）" | 与 §5.1 原文一致 |
| 轻微 #2 | F1 裸 − 残留（上轮 replace 未生效，根因：Python 单引号串内 \t 被解析为制表符） | 用 raw 字符串 r'...' 重新替换 | grep 确认新旧文本状态 |
| 轻微 #3 | "小页传输也能打满带宽"超原文口径 | 改"保持高 I/O 效率（原文口径 consistently high I/O efficiency）" | 验证 |
| 轻微 #4 | "2.5 GB"与 kv-cache 页 2.44 口径不一致 | 改"手册的缓存仍是约 2.44 GB"（顺带修复"该是 2.5 GB 还是 2.5 GB"病句） | 验证 |
