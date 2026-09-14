<!-- review-meta
round: 6
page: wiki/vllm-v1-two-process-arch/index.html
reviewed_content_sha256: ca0381a6d131e9ad
-->
# vllm-v1-two-process-arch 第 6 轮审查记录

审查对象：`/Users/wendadawen/code/dojo/wiki/vllm-v1-two-process-arch/index.html`
页面类型：`note`（依据 head 中 `dojo:type` = note，适用规范 `guides/note.md`）
本轮审查者未参与写作，也未读取本页 `research/` 下任何文件。

## 核对基准版本（记录用）

- 源码核对基线：本地检出 `/Users/wendadawen/code/HCF-Distributed/vllm`，commit `ee4ed6d71d300d6d19333a80d1c7cdef17058a31`，`git describe` = `v0.20.0-718-gee4ed6d71d`（2026-09-08）。与页脚声明的源码基线串完全一致，本页全部源码类论断按此版本逐条回源。
- 「上游 vLLM 同名流程中只有 post_step」一条的对照版本：`/Users/wendadawen/code/github/github-community/vllm`，commit `5ac2684976ee22c04fe0d2f968c6cf6096b383f2`，`git describe` = `v0.26.1rc0-454-g5ac268497`。
- 页面无编号式引文（无 `[n]` 标记），故无「参考文献编号与所引版本文献表不符」类核对项。
- 渲染实测：headless Chrome 截图（亮/暗两套 CSS 变量各测一遍）三张内联 SVG；`python3 .dojo/scripts/validate.py <页面>` 输出 `validation ok`。页面正文无 `$...$` 公式，无 KaTeX 渲染项。

## 已逐条回源且核对通过的事实性论断（摘要，均写有定位片段）

- §1 两进程划分：`AsyncLLM.__init__` 中 `self.engine_core = EngineCoreClient.make_async_mp_client(...)`（`vllm/v1/engine/async_llm.py:167`）；`engine/protocol.py:40 class EngineClient(ABC)`；`core_client.py` 中 `EngineCoreClient`(71)、`InprocClient`(293)、`MPClient`(480)、`SyncMPClient`(776)、`AsyncMPClient`(971)；`grep class .*EngineClient)` 命中仅 `async_llm.py:71 class AsyncLLM(EngineClient)`，故「EngineClient 的唯一实现」成立。
- §2 `run_busy_loop`（`vllm/v1/engine/core.py:1517`）原文逐字符为 `while self._handle_shutdown(): self._process_input_queue(); self._process_engine_step()`，与页内代码块一致；`_process_engine_step`（1558）依次 `pre_engine_step()` → `step_fn()` → `post_engine_step()` → `post_step()`；条件休眠原文 `if not model_executed and self.scheduler.has_requests(): time.sleep(0.001)` 及注释「yield the GIL briefly to allow background transfer threads to make progress」。
- §2 钩子来源：`vllm/hcf_mixin/v1/engine/core_mixin.py:28/32` 定义 `pre_engine_step` / `post_engine_step`（`post_engine_step` 累加 `engine_busy_seconds_delta`），而对照版本上游 `core.py` 只有 `def post_step`（612 行）；「上游只有 post_step」由 v0.26.1rc0 检出确认。`EngineCore(HyKVEngineCoreMixin, EngineCoreMixin)`（`core.py:106`）。
- §2 `post_step` 条件：原文 `if (not self.async_scheduling and self.use_spec_decode and model_executed and self.batch_queue_size == 1)`，及注释「update draft token ids in the worker process」「refreshed inside step_with_batch_queue via draft_future」——与页述逐条吻合。
- §2 `step_fn` 选择：`self.step_fn = (self.step if self.batch_queue is None else self.step_with_batch_queue)`；`batch_queue_size = self.model_executor.max_concurrent_batches`；`uniproc_executor.py:77 return 2 if self.scheduler_config.async_scheduling else 1`。`async_scheduling` 默认 `None`（`vllm/config/scheduler.py:182`），`vllm/config/vllm.py:995 elif self.scheduler_config.async_scheduling is None:` 分支在无不相容项时置 True。
- §3 `driver_worker = WorkerWrapperBase(rpc_rank=0)`（`uniproc_executor.py:48`）；`worker_base.py:191` docstring「remembers the worker module and class name ... real initialization happens in init_worker」，`__getattr__`（327）转发，`execute_model`（336）先 `self._apply_mm_cache(...)`。`Worker(HyKVGPUWorkerMixin, WorkerBase)`（`gpu_worker.py:113`）含 init_device/load_model/determine_available_memory/initialize_from_config/compile_or_warm_up_model/execute_model/sample_tokens/check_health/sleep/wake_up/add_lora/remove_lora/list_loras/pin_lora/profile，全部实存。
- §3 `worker_cls` 解析：`vllm/config/parallel.py:255 worker_cls: str = "auto"`；`vllm/platforms/__init__.py:60 def cuda_platform_plugin`，其中 `pynvml.nvmlDeviceGetCount() > 0`（75）；`__init__.py:262 def __getattr__` 懒加载 `current_platform`；`vllm/platforms/cuda.py:248-249 if parallel_config.worker_cls == "auto": parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"`；`init_worker` 内 `resolve_obj_by_qualname(parallel_config.worker_cls)`。
- §3 v1/v2 runner：`vllm/config/vllm.py:71 DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset({"Qwen3ForCausalLM"})`，`use_v2_model_runner` 检查 `HAS_TRITON` 与 `_is_default_v2_model_runner_model()`，后者要求 `runner_type == "generate"`、架构命中名单、`not is_moe and not is_quantized`；`gpu_worker.py:348-362` 以 `GPUModelRunner as GPUModelRunnerV2` / `GPUModelRunner as GPUModelRunnerV1` 两次导入——两实现同名 `GPUModelRunner`（`v1/worker/gpu_model_runner.py:437`、`v1/worker/gpu/model_runner.py:111`），页述成立。`Qwen3_5ForConditionalGeneration` 确为仓库内存在的架构名（`model_executor/models/config.py:632`）且不在名单内。
- §4 `run_method`（`serial_utils.py:486`）：str 走 `getattr(obj, method)`；`uniproc_executor.py:93/97 result = run_method(self.driver_worker, method, args, kwargs)`（该行阻塞与非阻塞分支各出现一次，页注「非阻塞分支」不误）。`multiproc_executor.py:405-409` 中 `send_method = cloudpickle.dumps(method, protocol=...)` 用于非字符串方法；`FutureWrapper`(82)、`WorkerProc`(575)；返回 future 由 `get_response()` 遍历 `response_mqs` 聚合。`shm_broadcast.py` `enqueue`（720）在 `total_bytes + len(all_buffers[0]) >= self.buffer.max_chunk_bytes` 时置 overflow 并 `local_socket.send_multipart(...)`，`local_socket = context.socket(XPUB)`（389）——「小消息走共享内存环形缓冲、超阈值退化为 ZMQ 发布订阅」成立。
- §5 四动作调用链（`core.py:538 step`）：`scheduler.schedule()` → `model_executor.execute_model(...)` → 必要时 `model_executor.sample_tokens(grammar_output)` → `scheduler.update_from_output(...)`；调度器注释原文「There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec. ... the scheduler tries to assign tokens to the requests so that each request's num_computed_tokens can catch up its num_tokens_with_spec.」（`vllm/v1/core/sched/scheduler.py:406-414`）。往返五要素：`utils.py:960 get_engine_zmq_addresses` 中本地走 `get_open_zmq_ipc_path()`、跨机走 `get_tcp_uri(host, ...)`；`core_client.py:232 add_request_async`；`async_llm.py:657 _run_output_handler` / `677 async def output_handler()`；`output_processor.py:464 class OutputProcessor` docstring「Process EngineCoreOutputs into RequestOutputs.」
- §6 全部类定位与注释：`core.py:106 class EngineCore ... """Inner loop of vLLM's Engine."""`；`llm_engine.py:47 class LLMEngine: """Legacy LLMEngine for backwards compatibility."""`（无基类，确不继承 EngineClient）；`executor/abstract.py:37 class Executor(ABC)` 与 `get_class`(49)；进程创建 `utils.py:149 context.Process(target=EngineCoreProc.run_engine_core, ...)`、`multiproc_executor.py:724 context.Process(target=WorkerProc.worker_main, ...)`。
- 「源码定位」19 条路径全部 `[ -f ]` 存在；列出的符号（`_add_request`、`_run_output_handler`、`AsyncOutputFuture`、`FutureWrapper`、`WorkerProc`、`get_engine_zmq_addresses`、`_is_default_v2_model_runner_model` 等）全部实存。
- 站内链接 `../vllm-framework-map/index.html`、`../vllm-mm-unified-embeds-cudagraph/index.html` 均存在；后者的「大 prefill 超出捕获尺寸走 eager、decode 命中 FULL 重放」与本页 983 token 的表述一致。
- 表述维度：全文（含表格、图注、范围说明）检索「我们/本页/本文/下面/接下来/值得注意的是/需要说明的是/综上/由此可见」等元话语与会话指代，均无命中；未发现调试叙事、临场评价或 AI 拼接腔。

---

## 问题记录

轻微｜不适用（图示自身）｜§2 图「忙等循环结构示意」（`<svg viewBox="0 0 720 150">`，`path d="M 270 108 Q 360 138 458 108"`）｜标为「循环」的虚线箭头不回指左侧节点：其终点 x=458 落在第二个节点（矩形 x 270–450）右下方之外，箭头方向由控制点 (360,138) 指向 (458,108) 为「向右上」，两端都不与任何节点相接，读者无法从箭头本身看出「循环回到 _process_input_queue」这一含义，只能靠文字标签猜。｜引文依据：headless Chrome 渲染截图 `/tmp/svgshot1.png`，虚线弧起于第二节点左下方、止于其右下外侧，箭头悬空；图内无任何连接回第一节点的路径。｜修复要求：让循环箭头真正闭合（例如自第二节点下方折回第一节点下方并加箭头），或删去箭头仅保留虚线弧＋「循环」标签。｜修复结果：未修复。｜复验结果：待复验。

轻微｜不适用（图示自身）｜§6 图「类关系全景图」（`<svg viewBox="0 0 760 400">`）｜两条关系箭头指向不明：其一 `path d="M 570 154 L 570 186"`（EngineCore → 下行，旁标「持有」）终点 x=570 恰落在 Scheduler（x 420–565）与 UniProcExecutor（x 585–720）之间的空隙，未触及任一节点，读者无法判断「持有」指向哪一个或哪几个类；其二 `path d="M 652 224 L 652 251"`（UniProcExecutor → WorkerWrapperBase）没有任何含义标注，而同图其余四条关系箭头均有「继承 EngineCore / 持有 / 包装 / ZMQ」标注。｜引文依据：渲染截图 `/tmp/crop_hold3.png` 显示「持有」箭头头部悬在两框之间空白处；SVG 源码中该 `path` 附近除 `text x="580" y="309"`（属 Wrapper→Worker 那条）外无标签。｜修复要求：把「持有」箭头改为分别指向 Scheduler 与 UniProcExecutor 的两条（或让终点落在节点上并标注「持有 Scheduler / UniProcExecutor」），并为 UniProcExecutor → WorkerWrapperBase 补标注。｜修复结果：未修复。｜复验结果：待复验。

轻微｜源码核对基线 `ee4ed6d71d…`（v0.20.0-718-gee4ed6d71d）｜§3 正文「由此可推，三层职责彼此隔离：更换并行度只需改 Executor，更换模型只需改 Runner。」｜后半句缺乏来源支撑，且与该页自身给出的分层不符：V1 的 `GPUModelRunner` 是架构无关的执行器（换模型依赖 `vllm/model_executor/models/` 下的模型类与注册表，而非改动 Runner），把「更换模型」归到 Runner 层会把读者引向错误的心智模型；该判断仅以「由此可推」四字标注，未说明其推断性质与适用边界。｜引文依据：同页 §3 表格将 GPUModelRunner 定义为「准备输入、调用模型、处理采样输出」的通用层；`vllm/model_executor/models/` 中按架构提供模型类，仓库内无「换模型需改 Runner」的对应实现或注释。｜修复要求：删除该句，或改写为「并行度变化集中在 Executor 层；模型相关逻辑集中在 Runner 层」等不含「只需」断言的表述，并显式标为推断。｜修复结果：未修复。｜复验结果：待复验。

轻微｜本地检出 `ee4ed6d71d…`（v0.20.0-718-gee4ed6d71d）｜文末「来源与范围说明 › 范围与限定」第三条｜该条讨论 UniProcExecutor / MultiprocExecutor 的 CPU 线程管理（torch intra-op 线程数、`OMP_NUM_THREADS=1`），但正文自始至终没有任何关于 CPU 线程管理的论断，读者在正文里找不到它要限定的对象，属与页面内容不对应的孤立限定项。｜引文依据：正文检索「线程」仅命中「ZMQ socket 线程 process_input_sockets」「后台的 KV 传输线程」「output_handler 后台循环」，无 CPU/intra-op 线程内容；其所述代码事实本身正确（`multiproc_executor.py:1065-1085` 有 `default_omp_num_threads = 1` 且设置 `os.environ["OMP_NUM_THREADS"]`，`uniproc_executor.py` 无对应代码）。｜修复要求：删除该条，或先补充它对应的正文论断再保留。｜修复结果：未修复。｜复验结果：待复验。

轻微｜本地检出 `ee4ed6d71d…`（v0.20.0-718-gee4ed6d71d）｜文末「来源与范围说明 › 源码定位」｜§3「model_runner 的 v1/v2」一节明确区分 v1 与 v2 两个 runner 实现，但源码定位只给出 v1 路径 `vllm/v1/worker/gpu_model_runner.py`，v2 实现所在模块 `vllm/v1/worker/gpu/model_runner.py` 未列出，该节结论缺少对应来源条目。｜引文依据：`vllm/v1/worker/gpu_worker.py:348-362` 两处导入分别为 `from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2` 与 `from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1`；两个文件均实存。｜修复要求：在源码定位中补 `vllm/v1/worker/gpu/model_runner.py：GPUModelRunner（v2 实现）`。｜修复结果：未修复。｜复验结果：待复验。

---

## 说明

- 本页在事实层面高度可靠：§1–§6 的全部机制描述、条件分支、代码片段、类归属与进程归属，均按上列版本逐条回源并给出原文片段，未发现定位不到、来源不支持、实验观察被写成无条件结论或推断被包装成来源结论的情形。
- 页内两处自认不可复核的内容（trace 的 23 个回合数、函数热点分布）已在「trace 证据」中显式声明为「原始 trace 记录已不随页面留存……不可再复核」，符合 note 规范「无法核实的内容标记为未核实或推断」，本轮不计为问题。
- 未发现以下不合格项：同页两处数字互相矛盾、算式与结论不符、图注读数与图上刻度不符、同一数字在正文/summary/图注间不一致、无来源支持的判断写成结论、构造示例写成来源事实、指向仓库不存在文件的路径、`alt`/`aria-label` 中出现 `$...$`、交互视图无脚本不可读（正文与表格不依赖脚本，目录为 JS 增强）。
- 口语化措辞一项经逐句核对后未作记录：「工人 / 包装壳 / 干活」等词集中出现在 §6「命名规律」，是对 `Worker / Wrapper / Base` 类名后缀字面含义的解释性用语，属该节体例所需，未降级页面表达质量。

统计：阻断 0 / 重要 0 / 轻微 5