<!-- review-meta
round: 4
page: wiki/vllm-framework-map/index.html
reviewed_content_sha256: 179403915be2119c
-->
# vLLM 框架全景审查记录（第 4 轮）

- 页面版本：22c23ba11ce6d45cff2a67d1e002eeea1f034290
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，亦未参与前序轮次审查）
- 适用规范：`guides/note.md`（`dojo:type=note`）
- 来源获取：本地检出 `github/github-wendadawen/vllm`，`git describe` = `v0.23.1rc0-1383-g95e073e17`，与页面 page-meta 声明一致；设计文档取自该检出 `docs/design/`；页面无 arXiv/GitHub 等外部 URL，无需 WebFetch。
- 已完整阅读章节：1. 进程架构；2. 五层代码地图（入口层／引擎层／执行层／模型层／编译层）；3. 跨层子系统地图；4. 关键设计决策；5. 官方设计文档索引；6. 请求生命周期；来源与范围说明（含两幅内联 SVG 的 aria-label 与全部 `<text>`）

## 已回源核对通过的关键论断（本轮未发现问题的部分）

- 进程数量：页面「单机 4 卡 TP=4 共 6 进程（1+1+4）」「8 卡 TP=2 DP=4 共 17 进程」，与 `docs/design/arch_overview.md` 第 131、139 行「1 API server + 1 engine core + 4 GPU workers = **6 processes**」「4 API servers + 4 engine cores + 8 GPU workers + 1 DP coordinator = **17 processes**」逐字一致；表内公式 `Worker = DP×PP×TP`、`API Server 默认 = DP`、`EngineCore = DP`、`DP Coordinator = DP>1 时 1 个` 与同文件第 119–125 行汇总表一致；代入 4×1×2=8，4+4+8+1=17，分项之和等于合计。
- 三条设计决策：`VllmConfig`/统一构造签名 `__init__(self, *, vllm_config: VllmConfig, prefix: str = "")`/分片与量化在初始化完成，与 `arch_overview.md` 第 279–305 行「405B model (with roughly 810GB weights) with 16 H100 80GB GPUs... load the full 810GB weights to every GPU and then shard」一致（810GB = 405B×2B，与「每卡都要装下 810GB」相符）。
- 模块计数（逐一在该检出上实测）：`vllm/model_executor/models/*.py` = 290（页面「约 290」✓）；`layers/quantization/` 的 `QuantizationMethods` 字面量 25 项（页面「约 25 种量化方法」✓）+ 6 个 online 简写（`fp8_per_tensor/fp8_per_block/fp8_per_channel/int8_per_channel_weight_only/nvfp4_per_token/mxfp8`，页面「另有 6 个 online 简写」✓）；`model_loader/` 注册加载器 default/sharded_state/runai_streamer/tensorizer/bitsandbytes/dummy/modelexpress 共 7 个（页面「7 种加载器」✓，该版本已无 gguf_loader，故不含 gguf 正确）；`v1/spec_decode/*.py` 17 个文件去掉 `__init__.py` = 16（页面「约 16 个模块」✓）；`docs/design/` 实测 29 篇，页面点名的 29 个文件名与该目录实际列表逐项吻合（含 `endpoint_plugins.md`、`nixl_kv_push_connector.md`，二者在该检出中真实存在）✓。
- 目录与文件路径：`entrypoints/`（openai/anthropic/grpc_server.py/mcp/cli/llm.py/pooling/speech_to_text）、`renderers/`、`multimodal/`（含 image.py、audio.py、video.py、hasher.py、cache.py）、`v1/engine/`（async_llm.py、core_client.py、core.py 内 `EngineCoreProc`、coordinator.py）、`v1/core/sched/`、`v1/core/kv_cache_*`（block_pool.py、kv_cache_manager.py、kv_cache_coordinator.py、kv_cache_utils.py 内含 prefix caching 哈希 `hash_block_tokens`）、`v1/executor/`（uniproc/multiproc/ray）、`v1/worker/`、`v1/attention/backends/`（flash_attn/flashinfer/triton_attn/gdn_attn/mamba*/mla/rocm_attn）、`compilation/`（`CUDAGraphWrapper` 定义于 cuda_graph.py）、`platforms/`（cuda/rocm/xpu/cpu）、`lora/`、`v1/pool/`、`v1/structured_output/`（4 个 backend ✓）、`distributed/`（eplb、stateless_coordinator、device_communicators/custom_all_reduce）、`distributed/kv_transfer/kv_connector/v1/` 之 mooncake/nixl/lmcache/hf3fs/moriio/flexkv 均存在。
- 通信机制：`distributed/device_communicators/shm_broadcast.py` 使用 ZMQ `XPUB`，`v1/executor/multiproc_executor.py` 使用 `ShmRingBuffer`，支持页面「EngineCore 到 worker 的调用广播走共享内存环形缓冲 + ZMQ 发布订阅」。
- 表述维度：全文（含折叠块、图注、SVG 内文字）通读后未发现元话语、以「本页」为主语的自我指代、会话指代（我/我们/你，实测 0 处）、调试复现叙事、临场评价或 AI 拼接腔；两幅 SVG 的 `aria-label` 与 `<text>` 中无 `$...$`，页面无公式，无 Unicode 数学字符，无声称可运行的代码。
- 链接与校验：正文引用的三篇关联笔记 `../vllm-v1-two-process-arch/`、`../vllm-mm-image-two-stage/`、`../vllm-mm-unified-embeds-cudagraph/` 均真实存在；`.dojo/scripts/validate.py wiki/vllm-framework-map/index.html` 返回 `validation ok`。

## 问题

- [轻微·技术] §6 请求生命周期 第二段：括注把 `embed_input_ids` 与 `_gather_mm_embeddings`、`_prepare_mm_inputs` 并列，写成 `v1/worker/gpu_model_runner.py` 的 `_preprocess`「所含」的函数；`_gather_mm_embeddings`（第 3171 行）与 `_prepare_mm_inputs`（第 3478 行）确为 `GPUModelRunner` 的方法，但 `embed_input_ids` 不是该文件的函数，而是模型对象的方法（该处代码为 `inputs_embeds_scheduled = self.model.embed_input_ids(...)`），定义在各模型类中｜引文依据：`gpu_model_runner.py:3539 inputs_embeds_scheduled = self.model.embed_input_ids(`；`grep -rn "def embed_input_ids" vllm/` 在 `v1/worker/` 下 0 命中，仅命中 `model_executor/models/*.py`（协议见 `model_executor/models/interfaces.py:350`）｜修复要求：把 `embed_input_ids` 从该括注移出，或改写为「`_preprocess` 调用的 `self.model.embed_input_ids`」，使括注中列出的函数与实际定义位置一致；不改变该段其余表述｜修复：｜复验：

## 说明（不构成问题）

- 来源说明中「14 个 v1 侧 KV connector」与 `kv_connector/factory.py` 的 16 处 `register_connector` 不矛盾：16 个注册项中 `ExampleConnector`、`ExampleHiddenStatesConnector` 为示例连接器，其余 14 个为生产连接器，页面亦以「及 mooncake/nixl 等子目录实现」另行交代子目录，故按可核对口径成立，不报。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布；上述轻微项不影响正确性与主线，按要求在上述位置顺手修正即可。

统计：阻断 0 / 重要 0 / 轻微 1

> 本轮所列问题的处理结果见 `minor-fixes.md`。
