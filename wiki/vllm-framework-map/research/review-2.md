<!-- review-meta
round: 2
page: wiki/vllm-framework-map/index.html
reviewed_content_sha256: 1d43a2fefc62b150
-->
# vLLM 框架全景审查记录（第 2 轮）

- 页面版本：9e9c452ce290e0dc476c7b3d7d2eab4fcf12f383（本轮实际读到的 index.html 工作树版本）
- 审查时间：2026-09-13
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查与修复）
- 适用规范：guides/note.md（head 中 dojo:type=note）；记录格式按 guides/concept/check.md 第 3 节
- 已完整阅读章节：head 元数据 → 1. 进程架构（含进程拓扑 SVG）→ 2. 五层代码地图 → 3. 跨层子系统地图（含五层全景 SVG）→ 4. 关键设计决策 → 5. 官方设计文档索引 → 6. 请求生命周期 → 来源与范围说明
- 核对来源：本地检出 v0.23.1rc0-1383-g95e073e17（/Users/wendadawen/code/github/github-wendadawen/vllm，2026-07-22）及其 docs/design/ 全部 29 篇、arch_overview.md、multiprocessing.md、paged_attention.md、prefix_caching.md、vllm/ 源码目录；两处内链的目标页面存在性

## 问题

- [重要·技术] 6. 请求生命周期第 2 段：把生命周期 ① 的「前端预处理」落点在 v1/worker/gpu_model_runner.py 的 _preprocess 与 v1/sample/ 的采样器上，位置与阶段均错——该句与本页 1. 进程架构表对「前端预处理」位置的表述冲突，且 _preprocess 是 worker 侧模型输入准备（属 ⑤ forward），采样器属 ⑥ 采样，都不属 ①。｜引文依据：本页表「API Server…HTTP 请求、tokenize、多模态预处理、流式返回」；本页 §6「其中前端的预处理集中在 v1/worker/gpu_model_runner.py 的 _preprocess（含 _gather_mm_embeddings、embed_input_ids、_prepare_mm_inputs 等函数）与 v1/sample/ 的采样器」；源码 gpu_model_runner.py:3489 `def _preprocess(...)`（worker 进程内执行，由 execute_model 调用，属 ⑤）｜修复要求：改写该句，分别指明 _preprocess 是 worker 侧的模型输入准备（⑤）、采样器属 ⑥ 采样，不再用「前端的预处理」统称二者；或删去与 ① 的关联｜修复：｜复验：

- [轻微·技术] 来源与范围说明第 1 条及其对应的 3. 跨层子系统地图：「14 个 v1 侧 KV connector」与源码注册数不符，计数口径未说明。｜引文依据：kv_connector/factory.py 中 `KVConnectorFactory.register_connector(...)` 共 16 处，注册名依次为 ExampleConnector、ExampleHiddenStatesConnector、LMCacheConnectorV1、LMCacheMPConnector、NixlConnector、NixlPullConnector、NixlPushConnector、MultiConnector、MoRIIOConnector、OffloadingConnector、DecodeBenchConnector、MooncakeConnector、MooncakeStoreConnector、FlexKVConnectorV1、SimpleCPUOffloadConnector、HF3FSKVConnector｜修复要求：改为 16，或注明计数口径（如排除 Example* 示例连接器后为 14）｜修复：｜复验：

- [轻微·技术] 2. 五层代码地图·模型层：「约 290 个模型文件」计数偏低且口径未说明（dojo:summary 同值）。｜引文依据：vllm/model_executor/models/ 递归 .py 文件 306 个（顶层 .py 290 个，另含子目录 transformers/ 16 个）｜修复要求：明确计数口径，或按「约 300」并说明含 transformers/ 子目录｜修复：｜复验：

- [轻微·表述] 4. 关键设计决策第 1 条：出现口语化用词与无来源支撑的评价性补语。｜引文依据：本页「所有类接收同一个完整配置对象而非散装参数…——这是应对快速演进的扩展性设计」；arch_overview.md 对应段落无「散装参数」「快速演进」字样｜修复要求：「散装参数」改为书面表述（如「而非逐项参数」）；评价性补语删除或改为来源支持的表述｜修复：｜复验：

## 已核对无问题（本轮抽查结论）

- 进程数量公式与示例：与 arch_overview.md「Process Count Summary」一致——4 卡 TP=4 = 1+1+4 = 6；8 卡 TP=2 DP=4 = 4+4+8+1 = 17；GPU Worker = DP×PP×TP，API Server 默认 = DP。分项之和与合计一致。
- 关键设计决策 2、3：`__init__(*, vllm_config, prefix="")`、405B/810GB/16 卡、按 prefix 做非均匀量化，均与 arch_overview.md 对应段落一致。
- 官方设计文档索引：表内 29 篇与 docs/design/ 目录 29 篇逐一对应（分项 1+5+5+10+3+5=29，合计写 29，一致）；paged_attention 标注「历史内核文档」与文档开头 warning 一致；prefix_caching 标注「块哈希与多模态哈希」与文档正文一致；multiprocessing 标注「fork/spawn 取舍」与文档正文一致。
- 目录结构类论断：entrypoints（openai/anthropic/grpc_server/mcp/cli/llm.py/pooling/speech_to_text）、renderers、multimodal、v1/engine（async_llm.py/core.py:EngineCore,EngineCoreProc/core_client.py:EngineCoreClient）、v1/core（block_pool.py、kv_cache_manager.py、kv_cache_coordinator.py、kv_cache_utils.py）、v1/executor（uniproc/multiproc/ray）、v1/worker（gpu_worker.py:Worker、worker_base.py:WorkerWrapperBase、gpu_model_runner.py:GPUModelRunner）、v1/attention（selector.py + backends）、v1/sample（thinking_budget_state.py）、v1/structured_output（4 个 backend：xgrammar/guidance/outlines/lm_format_enforcer）、distributed（eplb/stateless_coordinator/device_communicators）、lora、v1/pool、platforms 等均存在。
- 通信机制：请求/结果走 ZMQ（local 走 IPC path、跨机走 tcp，vllm/v1/utils.py:get_engine_client_zmq_addr）；EngineCore→worker 走共享内存环形缓冲 + ZMQ 发布订阅（shm_broadcast.py:ShmRingBuffer + XPUB/SUB）——与页面表述一致。
- 版本标注「vLLM 0.23.1rc0」与来源条「v0.23.1rc0-1383-g95e073e17」及 `git describe` 一致；量化「约 25 种 + 6 个 online 简写」与 QuantizationMethods 字面量（25 核心 + 6 简写）一致；加载器「7 种」（default/sharded_state/runai_streamer/tensorizer/bitsandbytes/dummy/modelexpress）与 7 个 loader 类一致；推测解码「约 16 个模块」与 v1/spec_decode/ 顶层 16 个 .py（除 __init__.py）一致；gguf 已移除，页面不再列 gguf。
- 内链：../vllm-v1-two-process-arch/index.html、../vllm-mm-image-two-stage/index.html 均存在；head 元数据 dojo:type=note、dojo:topics=推理系统、dojo:tag=推理系统 均在词表内；`.dojo/scripts/validate.py` 返回 validation ok。
- 表述维度：正文（含两张 SVG 图注）无「本页将…/下面来看…/需要注意的是」类元话语，无「我/我们/你」会话指代，无调试与复现踩坑叙事，无「本页/本表」自我指代（前序版本中的个人进度仪表盘已移除）。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（1 条重要问题需关闭；3 条轻微问题可一并处理）
