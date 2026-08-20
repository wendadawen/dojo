# PagedAttention 审查记录（第 3 轮）

- 页面版本：index.html fb64fe1dfe91f60d1579496e3d6e52b949a7f059 / overview.html b373b4493caca5df24dcfa58064a2d4adb2b2880
- 审查时间：2026-08-20 12:01
- 审查者：独立子代理（reviewer-paged-attention-3）
- 已完整阅读章节：index.html 全文含全部折叠块——导语与元信息、核心问题（4 题及解答折叠块）、「1. 连续预留的三类浪费」（含构造示例与本章问题）、「2. 分页机制——块、页表与按需分配」（含写入-读取路径图示、A/B 页 16 账目折叠块、本章问题）、「3. 页间不连续——为显存设计，被传输惩罚」（含页大小上限与可配置性折叠块、本章问题）、「4. 页大小怎么选——一个没有免费午餐的旋钮」（含四维权衡表、本章问题）、「来源与范围说明」全部小节；overview.html 全文（这是什么 / 解决什么问题 / 核心机制 / 关键结论与边界）。对照来源：vllm-paper.txt 全文、Strata tex 源码（1_intro / 1.5_background / 2_motivation / 3_design / 5_eval）。

## 第 2 轮修复验证

1. 「分层缓存基线统一取页 32（Strata 自身与 SGLang 用默认页 1，§5.1）」——与原文一致。Strata §5.1 Baselines："We set the page size for \emph{SGLang} and \oursys{} to 1 (SGLang's default), and the page size for \emph{SGLang-HiCache} to 32 to be consistent with other hierarchical cache baselines."；vLLM-LMCache "vLLM page size was set to 32 in line with prior work"；TensorRT-HiCache "the page size also set to 32 as default"。页面折叠块中「先前分层 KV cache 工作的推荐值（MemServe、CachedAttention、FlashGen、Pensieve）」与 §3.1 引文一致："a value recommended in prior works on hierarchical KV cache~\cite{arxiv24:hu_memserve, atc24:gao_cachedattention, asplos25:flashgen, eurosys25:yu_pensieve}"。
2. 「小页传输也能保持高 I/O 效率（原文口径 consistently high I/O efficiency）」——与原文一致。Strata §5.3.2："with the GPU-assisted I/O mechanisms described in §4.2, \oursys{} achieves consistently high I/O efficiency regardless of page size"。
3. 「手册的缓存仍是约 2.44 GB」——复算通过。20,000 token × 128 KB/token = 2,621,440,000 B = 2.44 GiB，与 625 页 × 4 MB 一致。
4. F1 的 $-\text{token 数}$ 在 math 环境内——通过。来源说明原句为「$\text{页数}\times\text{页大小}-\text{token 数}$」，减号与被减项整体在 `$...$` 内。

## 来源核对抽查

- C2 / N1（20.4%–38.2%）：vLLM §3.1 "our profiling results in Fig. 2 show that only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems"；Figure 2 图例含 "External frag. & Others"，与页面「论文图例为 External frag. & Others，含少量未归类项」一致。核对通过。
- C3（OS 类比）：vLLM §1 "one can think of blocks as pages, tokens as bytes, and requests as processes"；§4.2 "The key idea behind vLLM's memory manager is analogous to the virtual memory in operating systems"。核对通过。
- N4（22% / GH200 约 5%）：Strata §3.1 "transferring KV cache data for 8192 tokens, achieves only approximately 22% of the theoretical PCIe 5.0 bandwidth. Page size is set to 32 … falling to as low as ~5% on systems like NVIDIA's Grace-Hopper platform that replaces PCIe with NVLink"。核对通过。
- N5（最优页 512、93%、2.4%）：Strata §5.3.2 "Even at its best-performing setting (page size 512), \emph{SGLang-HiCache} achieves only 93\% of \emph{\oursys-IO}'s performance, primarily due to a 2.4\% lower cache hit rate"；实验条件 §5.3 "All analyses presented here were conducted using the Qwen-14B model on an H200 platform"（LooGLE 扫描），与页面标注条件一致。核对通过。
- 其余条目本轮顺带全核：C1（§3.1 静态按最大长度预分配 + Figure 3 三类浪费）、C4（"alleviates internal fragmentation … allocating them on demand … eliminates external fragmentation as all blocks have the same size … enables memory sharing at the granularity of a block"）、C5（§4.2 block tables 与 "a new physical block is only allocated when all previous blocks are full"）、C6（"vLLM limits all the memory wastes for a request within one block"）、C7/N2（background §2.2 "Typical page sizes are small—e.g., 32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang—where each token may span from tens of kilobytes to several megabytes"；§3.1 "a maximum supported size in vLLM for CUDA GPUs"）、C8（§1 "paging causes \textit{data} fragmentation … spread across multiple non-contiguous pages. This leads to small data transfers … fail to saturate PCIe bandwidth"）、N3（构造算例，页 32 × 128 KB = 4 MB、20000/32 = 625 页复算通过，每 token 128 KB 与 Strata §1 "40 GB … roughly 0.3M tokens for Llama-8B" 量级吻合）、N6（§4.2 "the granularity required for efficient GPU-assisted I/O is only 128 bytes on most architectures"）。全部通过。

## A/B 例子数字复算

- A 上限 2048 / 实际 100：浪费 2048 − 100 = 1948 槽 ✓；B 上限 512 / 实际 500：浪费 12 槽 ✓；A+B 占 2560 槽、真实数据 600 槽、连续预留利用率 600/2560 ≈ 23% ✓。
- 分页后（页 16）：A 占 ⌈100/16⌉ = 7 页 = 112 槽，内部碎片 112 − 100 = 12 ✓；B 占 ⌈500/16⌉ = 32 页 = 512 槽，碎片 12 ✓；合计 624 槽，利用率 600/624 ≈ 96% ✓；外部碎片为零、生成中无预留 ✓。
- 40 GB 利用率 30% → 96% 得 3.2 倍可用缓存 ✓（示意演算，未写成来源结论）。

## 机械项检查

- `$` 配对：两页面正文每行 `$` 均成对；唯一奇数行为 index.html 脚本内正则（KaTeX auto-render 默认忽略 script 标签），不构成问题。
- math 字符均在 `$...$` 内：正文、表格、标题、summary 无 Unicode 数学运算符残留（检出 `·` 为间隔号、overview 导航 `←` 与图示流程箭头为 UI/流程符号，非数学字符）。
- 链接有效：../kv-cache/index.html、../prefix-caching/index.html、../strata/index.html、overview.html、../../index.html、libs 下 7 个资源文件均存在；两页面互链成立；跨页章节引用有效（kv-cache 第 4 章「为什么显存成为瓶颈」、prefix-caching 第 4 章「命中率——页大小是隐藏的自变量」均存在）。../gpu-communication 在两页面中均未被引用，无需占位。
- 问题块结构：页面级核心问题 4 题 + 四章本章问题各 2 题，共 12 题，均有「解答」折叠块（12 个解答 summary），答案独立可读、与正文结论一致，核心问题答案均指明论证所在章节。
- `python3 .dojo/scripts/validate.py wiki/paged-attention/index.html` 返回 "validation ok"。

## 问题

- [轻微·可读性] index.html 第 2 章折叠块「展开：A/B 例子在页 16 下的完整账目」：向上取整符号 $\lceil 100/16\rceil$ 为全页首现，符号含义（向上取整）未在首现处或此前任何位置说明，F1 的符号定义亦在文末来源说明且未解释记号。｜引文依据：不适用｜修复要求：在首现处（如「需要 $\lceil 100/16\rceil=7$ 页」后）加「⌈·⌉ 为向上取整」或同等文字说明。｜修复：｜复验：
- [轻微·来源] index.html 第 4 章表格「内部碎片与预留」行与「来源与范围说明」：「来源与范围说明」称「'平均半块'为推断（均匀假设下的期望），标注于正文」，但第 2 章正文仅有论文口径「每请求至多浪费一页」，推断标注实际位于第 2 章「本章问题」解答折叠块，指引位置不准确；且第 4 章表格单元格「浪费小（平均约半页，至多一页）」中「平均约半页」为推断值却未就地标注，与论文口径「至多一页」并列易被当作来源结论。｜引文依据：vLLM 论文原文仅支持"至多一页"口径："vLLM limits all the memory wastes for a request within one block"（§4.2），无"平均半块"表述。｜修复要求：将来源说明中「标注于正文」改为实际位置（第 2 章本章问题解答），并在第 4 章表格「平均约半页」处补推断标注（如「平均约半页（推断）」），或将该格改为仅保留论文口径。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：修复（仅 2 条轻微，逐条修复并复验后即可发布；第 2 轮 4 项修复全部验证通过，来源核对与数字复算无阻断、无重要问题，validate.py 通过）


## Round 3 修复记录

| 编号 | 问题 | 修复 | 复验 |
|---|---|---|---|
| 轻微 #1 | ⌈⌉ 向上取整首现未说明 | 折叠块加 "$\lceil\cdot\rceil$ 为向上取整（不足一页按一页计）" | 验证 |
| 轻微 #2 | "平均半页"推断标注位置修正 | 第 4 章表格就地标"（至多一页；平均约半页为推断）"；来源说明改为指向两处标注 + 论文口径 within one block | 验证 |

## 发布结论

三轮审查完成（轮 1：8 问题、轮 2：4 问题、轮 3：2 问题，全部修复）。阻断 0、重要 0、轻微 0 未决。可发布。
