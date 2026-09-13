<!-- review-meta
round: 5
page: wiki/vllm-cudagraph/index.html
reviewed_content_sha256: 5b6a1afe0c48756b
-->
# torch.compile 图捕获与 CUDA Graph 审查记录（第 5 轮）

- 页面版本：bc5b5ec9e5129ef5bdd900f0b1252ed431afa31a
- 审查时间：2026-09-13 21:22
- 审查者：编排者派发的独立审查者（未参与写作，也未参与前序轮次）
- 页面类型：note（dojo:type=note），适用规范 guides/note.md
- 已完整阅读章节：1. torch.compile 的两阶段流程；2. Dynamo 图捕获的处理边界；3. CUDA Graph 的录制与重放；4. 捕获尺寸的确定；来源与范围说明
- 来源获取方式：PyTorch 官方文档（docs.pytorch.org/torch.compiler 及 programming_model 系列）与 torch 2.8.0 本机实测；vLLM 源码本地检出 commit 5ac2684（2026-08-07，与页面标注一致，git 核对 commit 日期 2026-08-07）。

## 问题

- 无。本轮逐项核对后未发现阻断、重要或轻微问题。

## 核对依据（本轮实际核到的原文片段与关键数值）

- 二阶段与默认后端：torch.compile docstring（torch 2.8.0）「Optimizes given model/function using TorchDynamo and specified backend.」「"inductor" is the default backend」；programming_model 系列文档陈述 Dynamo 将函数追踪为 FX 图、图外代码回落 Python。与第 1 节一致。
- Inductor 以 Triton 产出 kernel：torch.compile docstring「max-autotune ... leverages Triton or template based matrix multiplications ... on GPU」。本机为 CPU，未复现 GPU Triton 产物，按官方文档核对，与第 1 节一致。
- C 扩展 graph break 与警告文本：本机 torch 2.8.0 实跑 `torch.compile(f)`（f 内含 `hashlib.sha1`），警告原文「Dynamo does not know how to trace the builtin `_hashlib.openssl_sha1.`」，执行继续并返回结果；`fullgraph=True` 下抛 `torch._dynamo.exc.Unsupported`，消息「Attempted to call function marked as skipped」。与第 2 节表格与正文逐字一致（含「抛出 Unsupported」「标记为 skipped」「一次警告」warn_once）。
- 捕获尺寸默认生成：vllm/config/vllm.py:1858-1864 `decode_query_len = 1 + self.num_speculative_tokens`；`max_cudagraph_capture_size = min(self.scheduler_config.max_num_seqs * decode_query_len * 2, 512)`；`max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)`；docstring 生成式 `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_graph_size + 1, 16))`；compilation.py:1131 断言 `cudagraph_capture_sizes[-1] == max_cudagraph_capture_size`。与第 4 节「末元素即最大捕获尺寸」「min(max_num_seqs × decode_query_len × 2, 512) 再与 max_num_batched_tokens 取较小值」一致。
- 上限 512 的动机：vllm/config/compilation.py:703-706 docstring「max_cudagraph_capture_size is set to min(max_num_seqs*2, 512) by default. This voids OOM in tight memory scenarios with small max_num_seqs, and prevents capture of many large graphs (>512) that would greatly increase startup time with limited performance benefit.」与第 3 节「紧凑显存避免 OOM、避免拉长启动时间」一致。
- padding 与退回 eager：vllm/v1/cudagraph_dispatcher.py:72-91 `_compute_bs_to_padded_graph_size` 将 batch 映射到不小于它的最小合法捕获尺寸（向上补齐）；:274-281 `num_tokens > max_size` 时返回 `CUDAGraphMode.NONE`；:320-324 无匹配 key 亦返回 NONE。与第 4 节「向上 padding」「超出上限或未命中退回 eager」一致。
- 全局内存池：vllm/platforms/interface.py:174 `_global_graph_pool: Any | None = None`（类属性），:1152-1159 `get_global_graph_pool` 惰性写入该类的单例并复用；vllm/compilation/cuda_graph.py:200 `self.graph_pool = current_platform.get_global_graph_pool()`、:315 `pool=self.graph_pool`。与第 3 节「全图共用同一全局 cudagraph 内存池、录制时以该池提交」一致。
- 查表位置：vllm/v1/worker/gpu_model_runner.py:904 构造 `CudagraphDispatcher`，:2981 与 :4026 调用 `.dispatch(...)`。支持第 3 节「由 GPUModelRunner 查表决定」。
- 模式语义：vllm/config/compilation.py:607-639 cudagraph_mode docstring「FULL_AND_PIECEWISE mode: Capture full cudagraph for decode batches and piecewise cudagraph for prefill and mixed prefill-decode batches. This is the most performant mode for most models and is the default.」支持第 3 节 prefill 采用 PIECEWISE 的陈述。
- 引用路径存在性：vllm/config/vllm.py、vllm/config/compilation.py、vllm/v1/cudagraph_dispatcher.py、vllm/platforms/interface.py、vllm/compilation/cuda_graph.py 均存在；站内引用页 vllm-v1-two-process-arch、vllm-mm-unified-embeds-cudagraph 均存在，且 mm 页第 101 行回链本页；本页第 124 行「decode 命中 FULL、大 prefill 落到 eager」与 mm 页第 133-148 行 dispatch 表一致。
- 表述维度：逐段通读（含来源说明段），未发现元话语（「本页将…」「下面来看…」「需要注意的是」）、以「本页」为主语的自我指代、会话指代（我/我们/你）、调试与复现踩坑叙事或临场评价；公式式表达统一置于 `<code>` 内，`×` 为站内通用写法（50+ 页使用）；alt 属性未出现 `$...$`。
- 交叉核对数字：512、[1,2,4]、range(8,256,8)、range(256,…,16)、decode_query_len=1+num_speculative_tokens 在正文、summary、来源说明三处一致，无两处互相矛盾。
- 机械项：`.dojo/scripts/validate.py wiki/vllm-cudagraph/index.html` → validation ok。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布
