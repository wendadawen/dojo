<!-- review-meta
round: 4
page: wiki/vllm-cudagraph/index.html
reviewed_content_sha256: 028f391b3721d68a
-->
# vllm-cudagraph 审查记录（第 4 轮）

- 页面版本：fbdbb2519baacb952999933709572c25c0301aa5（index.html 工作树哈希）
- 页面路径：wiki/vllm-cudagraph/index.html
- 页面类型：note（dojo:type=note，适用规范 guides/note.md；记录格式按 check.md 第 3 节）
- 审查时间：2026-09-13 20:32
- 审查者：独立子代理（第 4 轮，未参与写作与前序轮次）
- 已完整阅读章节：1. torch.compile 的两阶段流程 / 2. Dynamo 图捕获的处理边界 / 3. CUDA Graph 的录制与重放 / 4. 捕获尺寸的确定 / 来源与范围说明
- 机械验证：`.dojo/scripts/validate.py wiki/vllm-cudagraph/index.html` 返回 validation ok；页面无 `<pre>`/`<details>`/`<svg>`，无折叠块、无图注、无可运行代码块（第 3 项"代码实测"落空，改用实测核对警告文本，见下）

## 核对过的关键论断（均与来源一致，未列问题）

- torch.compile 两阶段：TorchDynamo 捕获图（Frame Evaluation API）、Inductor 为默认后端并借助 Triton 生成 kernel —— PyTorch 官方页 `torch.compiler`："TorchInductor is the default torch.compile deep learning compiler... For NVIDIA, AMD and Intel GPUs, it leverages OpenAI Triton as the key building block"；`torch/__init__.py:2512` "inductor" is the default backend。
- fullgraph 语义：torch 2.8.0 docstring "If True, then we require that the entire function be capturable into a single graph... this will raise an error"（`torch/__init__.py:2500-2502`）。
- C 扩展的 graph break 与警告文本：本机 torch 2.8.0 实跑 `hashlib.sha1` 得到 UserWarning 原文 `Dynamo does not know how to trace the builtin `_hashlib.openssl_sha1.``，执行继续；`fullgraph=True` 时抛 `Unsupported`；graph_break 计数键为 `Attempted to call function marked as skipped`。页面表述与实测一致。
- 捕获尺寸默认规则：`vllm/config/compilation.py:700-701` docstring 原文 `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))`；`vllm/config/vllm.py:1859-1864` 原文 `decode_query_len = 1 + self.num_speculative_tokens`、`min(self.scheduler_config.max_num_seqs * decode_query_len * 2, 512)`、`min(max_num_tokens, max_cudagraph_capture_size)`；提交 `5ac268497` 日期 `2026-08-07 02:42:34 +0000`，与来源标注一致。
- padding 与 dispatch：`vllm/v1/cudagraph_dispatcher.py:141` 与 `:82-91` 以 num_tokens 向上取整；`dispatch` 在 `num_tokens > max_size` 或键未命中时返回 `CUDAGraphMode.NONE`；`FULL_AND_PIECEWISE` 为默认（compilation.py:615、vllm.py:291）。

## 问题

- [重要·技术] §3 CUDA Graph 的录制与重放 · 「显存代价」条：该条断言"内存池…按最大容量预留、即使未用满也不释放，因此显存占用随捕获图数量线性增长，vLLM 通过限制捕获图数量控制显存"。vLLM 对全部捕获图使用同一个全局 cudagraph 内存池，显存由最大捕获尺寸的工作集决定，不随捕获图数量线性增长；且本页"来源与范围说明"未给该机制任何来源。｜引文依据：`vllm/platforms/interface.py:1152-1159` `def get_global_graph_pool` 返回类级单例 `cls._global_graph_pool`（`_global_graph_pool: Any | None = None`，:174）；`vllm/compilation/cuda_graph.py:200` `self.graph_pool = current_platform.get_global_graph_pool()`，:313-317 `with torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=current_stream()):`；`vllm/config/compilation.py:703-706` 注释 "This voids OOM in tight memory scenarios with small max_num_seqs, and prevents capture of many large graphs (>512) that would greatly increase startup time with limited performance benefit."；PyTorch CUDA semantics 页 "By default, the allocator creates a separate private pool for each capture... but sometimes needlessly wastes memory. Sharing memory across captures # To economize the memory stashed in private pools, torch.cuda.graph ... optionally allow different captures to share the same private pool."｜修复要求：删除"显存占用随捕获图数量线性增长"这一无条件论断与其对 vLLM 的归因；改为说明 vLLM 所有捕获图共用一个全局 cudagraph 内存池（get_global_graph_pool），显存由最大捕获尺寸的工作集决定，并给出捕获尺寸上限的真实理由（compilation.py 注释：紧凑显存下的 OOM 与启动时间）。若保留原表述，须改标为推断并附上述依据。｜修复：已删除「显存占用随捕获图数量线性增长」及其对 vLLM 的归因，§3「显存代价」改为：vLLM 为全部捕获图共用同一全局 cudagraph 内存池（`get_global_graph_pool` 返回类级单例，录制时以该池提交），故显存由最大捕获尺寸对应的工作集决定，而非随捕获图数量线性增长；图本身每张数十至数百 MB 的量级保留并内联标注为推断。同时补入捕获尺寸上限（默认不超过 512）的真实理由——规避紧凑显存下的 OOM、避免为收益有限的大批量捕获大量图而大幅拉长启动时间。「来源与范围说明」补入 `vllm/platforms/interface.py::get_global_graph_pool` 与 `vllm/compilation/cuda_graph.py`。｜复验：

- [轻微·技术] §4 捕获尺寸的确定："实际请求数不落在合法捕获尺寸上时，按邻近的合法尺寸向上补齐（padding）"。padding 的索引是总 token 数（num_tokens），不是请求数；两者仅在本节前文假设的"无投机解码"下相等，当 `decode_query_len = 1 + num_speculative_tokens > 1` 时并不一致。｜引文依据：`vllm/v1/cudagraph_dispatcher.py:141` `num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]`；:82-91 `_bs_to_padded_graph_size` 以 num_tokens 为下标向上取整。｜修复要求：把"实际请求数"改为"实际总 token 数"，或在该句标明仅适用于无投机解码的 decode 批次。｜修复：已将 §4「实际请求数不落在合法捕获尺寸上时…」改为「实际总 token 数不落在合法捕获尺寸上时…」，与 padding 以 num_tokens 为下标一致。｜复验：

- [轻微·维护] 页头 page-meta："更新于 2026-08-06 · 类型：学习笔记" 早于本页所引来源的提交日期（vLLM commit 5ac2684，2026-08-07），更新时间与实际来源及后续改动不一致。｜引文依据：来源与范围说明"vLLM 源码（本地检出 commit 5ac2684，2026-08-07）"；`git show -s --format=%ci 5ac268497` = `2026-08-07 02:42:34 +0000`；`git log -1 --format=%ci -- wiki/vllm-cudagraph/index.html` = `2026-09-13 20:06:41 +0800`。｜修复要求：把"更新于"改为不早于所引来源日期、且与最近一次内容改动一致的值。｜修复：页头「更新于」由 2026-08-06 改为 2026-09-13，不早于所引来源 commit 5ac2684 的 2026-08-07，且与本次内容改动一致。｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（重要问题需关闭后方可发布）

## 说明

表述维度（第 4 项）已逐段通读正文与导语，未发现元话语（"本页将…""下面来看…""需要注意的是"）、会话指代（我/我们/你）、调试与复现踩坑叙事、临场评价或把"场景"当术语的 AI 拼接腔，故无该类问题上报。页面级无「核心问题/本章问题」结构（note 类不要求），无图示与折叠块，均不属于本页缺陷。