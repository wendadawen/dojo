# 前缀缓存 审查记录（第 3 轮）

- 页面版本：index.html `1f0af8ba92ee0c8312b7607a5870eaa58e6c9108`；overview.html `fee7a021258ed43db6a9e73ed5f90544ae200b08`
- 审查时间：2026-08-20 11:45
- 审查者：独立子代理（reviewer-prefix-caching-3）
- 已完整阅读章节：index.html 按顺序——导语与主要依据、核心问题（5 题及全部解答折叠块）、1. 什么能复用（含表格、本章问题 2 题）、2. radix tree（含 SVG 图示、折叠块「三个请求走树的完整过程」、本章问题 2 题）、3. 缓存满了怎么办（本章问题 2 题）、4. 命中率（含公式 F1、页大小命中表、本章问题 2 题）、5. 两种实现（含对比表、本章问题 2 题）、来源与范围说明（核心论断与来源 / 核心公式与来源 / 外部数字与实验条件 / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）；overview.html 全文（这是什么 / 哪些负载能复用 / 核心机制 / 关键结论与边界）。

## 审查输入

- `wiki/prefix-caching/index.html`、`wiki/prefix-caching/overview.html`
- `/tmp/strata-research/sglang-paper.txt`（SGLang，arXiv:2312.07104v2）
- `/tmp/strata-research/src/`：1.5_background.tex、2_motivation.tex、3_design.tex、5_eval.tex、6_related.tex（Strata，arXiv:2508.18572v2）
- `guides/concept/check.md`

## 数值复核

1. **2.4% 对照对象**。正文（§4）、核心问题 4 解答、overview「关键结论与边界」、来源说明 N2 四处均写"页 512 配置的 SGLang-HiCache 命中率比 **Strata-IO** 低 2.4%"，与 Strata §5.3.2 原文一致（"Even at its best-performing setting (page size 512), SGLang-HiCache achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate"，5_eval.tex:154）。**但第 4 章本章问题第 1 题解答末句写成"实测页 512 比页 32 命中率低 2.4%"，对照对象错误，见问题区。**
2. **三请求走树算例**。$S=200$、$m_1=30$、$m_2=40$、$S'=200$、$m_3=30$：R1 全算 $200+30=230$；R2 命中 $S$ 段 200、只算 $m_2$ 的 40；R3 根部分叉全算 $200+30=230$。总计 $230+40+230=500$，无缓存 $230\times 3=690$。页面折叠块全部数字复算一致。
3. **100 token × 页 1/32/256 算例**。页 1：$\lfloor 100/1\rfloor=100$ 页、命中 100 token；页 32：$\lfloor 100/32\rfloor=3$ 页、命中 96、落空 4；页 256：$\lfloor 100/256\rfloor=0$、命中 0。正文表格、核心问题 4 解答、本章问题 1 解答、overview 均一致且可复算。

## 来源核对抽查记录

按 check.md §2.2 逐条打开来源定位，C1–C9、N1–N4 各抽 2 条（C 系列抽 C1、C7；N 系列抽 N1、N2；其余条目在通读与数值复核中顺带核对，均有原文支持）：

- **C1**（既有系统完成后丢弃缓存、阻止复用）：SGLang §1："In existing inference engines, the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls and significantly slowing down the execution."（sglang-paper.txt:110-112）✓
- **C7**（SGLang RadixTree / vLLM+Mooncake 哈希 / LMDeploy 粗粒度 trie / Strata 扩展 HiRadixTree）：Strata §6："SGLang employs a RadixTree for tracking shared context. Other serving engines, such as vLLM and Mooncake, utilize hashing mechanisms that generate unique page identifiers based on token IDs and prefix page hashes. LMDeploy adopts a hybrid approach by constructing coarser-grained tries. \oursys builds upon SGLang by extending its RadixTree to a HiRadixTree."（6_related.tex:3-6）✓
- **N1**（SGLang 页大小 = 1 token）：SGLang §3："These KV cache tensors are stored in a non-contiguous, paged layout, where the size of each page is equivalent to one token."（sglang-paper.txt:288-289）；Strata §5.1 "We set the page size for \textit{SGLang} and \oursys to 1 (SGLang's default)"（5_eval.tex:25）✓
- **N2**（2.4% / 93%）：Strata §5.3.2 引文见数值复核第 1 条（5_eval.tex:154）；实验条件标注（Qwen2.5-14B、H200、LooGLE、对照 Strata-IO）与 §5.3 开头 "All analyses presented here were conducted using the Qwen-14B model on an H200 platform"（5_eval.tex:116）及图 loogle_page_size_scan 一致 ✓
- 顺带核对：C3（"edges of a radix tree can be labeled … with sequences of elements of varying lengths … mapping between sequences of tokens, and their corresponding KV cache tensors"，sglang-paper.txt:284-288）✓；C4（"evicts the least recently used leaf first … enable the re-use of their common ancestors until those ancestors become leaves"，sglang-paper.txt:289-292）✓；C5（"each node maintains a reference counter … share the same memory pool"，sglang-paper.txt:293-299）✓；C6（"widely adopted by providers such as OpenAI and Google"，1.5_background.tex:18，即 §2.3）✓；C8（"cache matching is performed on a per-page basis"，2_motivation.tex:39，即 §3.1）✓；C9（"retains the cache for prompts and generation results in a radix tree"，sglang-paper.txt:279）✓；N3（构造算例，已标注）✓；N4（"consistently reaching approximately 95\% cache hit rate by leveraging CPU memory"，5_eval.tex:87；1TB pinned DRAM 见 5_eval.tex:65）✓；页大小脚注（TensorRT-LLM 32 / vLLM 16 见 1.5_background.tex:14 "32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang"；"32 是 vLLM CUDA GPU 最大支持值" 见 2_motivation.tex:48 "a maximum supported size in vLLM for CUDA GPUs"）✓

## 问题

- [重要·技术] index.html §4「本章问题」第 1 题解答（id=hit-rate-questions 内第 1 个 details）：解答末句"实测页 512 比页 32 命中率低 2.4%"对照对象错误——2.4% 是 SGLang-HiCache（页 512）相对 Strata-IO 的命中率差距，不是页 512 相对页 32 配置间的差距，与正文 §4、核心问题 4 解答及 N2 来源说明的表述不一致，读者会把 2.4% 误解为页大小之间的对比｜引文依据：Strata §5.3.2 "Even at its best-performing setting (page size 512), SGLang-HiCache achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate."（5_eval.tex:154）｜修复要求：将该句改为与正文一致的表述"实测页 512 配置的 SGLang-HiCache 命中率比 Strata-IO 低 2.4%"（或等义改写，对照对象必须是 Strata-IO）｜修复：｜复验：

## 机械项检查结果

- **$ 配对**：index.html 正文（剔除 script 块与注释）单 `$` 共 180 个、偶数配对，`$$` 1 对完整；overview.html 单 `$` 14 个、偶数配对。通过。
- **math 字符**：两页正文中 ×、→、⌊、⌋ 及数字间 +/- 均只出现在 `$...$` 内（检出项仅为日期 2026-08-19 与 Qwen2.5-14B 版本号，非算式）。通过。
- **链接有效**：`../kv-cache/index.html`、`../paged-attention/index.html`、`../strata/index.html`、`../../index.html`、`overview.html` 均存在；libs 下 katex.min.css/js、auto-render.min.js、prism 系列资源齐备；overview 与 index 相互链接。通过。
- **radix tree SVG**：4 个 foreignObject 标签（$S$：系统提示词 / $m_1$：消息 1 / $m_2$：消息 2 / $S'$：另一提示词）逐一与 5 个节点框、4 条连线做包围盒计算，均无重叠、无压线。通过。
- **问题块**：页面级核心问题 5 题、各章本章问题 2×5=10 题均带解答折叠块；解答独立可读（重述结论而非指代正文）；核心问题 5 题答案均指明"见「某章」"。通过。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 0
- 处置：修复。仅 1 条重要问题（2.4% 对照对象写错一处），按修复要求改正并复验、运行 `.dojo/scripts/validate.py` 成功后即可发布；其余全部审查项（数值复算、来源抽查、问题块、机械项）均通过。


## Round 3 修复记录

| 编号 | 问题 | 修复 | 复验 |
|---|---|---|---|
| 重要 #1 | §4 本章问题 1 解答末句"比页 32 命中率低 2.4%"残留 | 改"页 512 配置的 SGLang-HiCache 命中率比 Strata-IO 低 2.4%（Strata §5.3.2）"；顺带把该处 floor 混排改为单一 math 段 | grep 清零；validate.py ok |

## 发布结论

三轮审查完成（轮 1：8 问题、轮 2：6 问题、轮 3：1 问题，全部修复）。阻断 0、重要 0、轻微 0 未决。可发布。
