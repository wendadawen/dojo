<!-- review-meta
round: 3
page: wiki/vllm-v1-two-process-arch/index.html
reviewed_content_sha256: 38f2142b00e66026
-->
# vllm-v1-two-process-arch 审查记录（第 3 轮）

- 页面版本：git blob db7140f38860fdd15abc1fc880e9d707c94d6da5（工作树，43066 字节；sha256 bbdc3e916e83c916…）
- 源码基线：页面自述基线为本地检出 HCF-Distributed/vllm `v0.20.0-718-gee4ed6d71d`，本轮据此核对
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序审查；未读取本页 research/ 下任何文件）
- 已完整阅读章节：1. 两进程的职责划分与证据；2. 忙等循环与回合结构（含「回合内的钩子、空转与容错」「step 与 step_with_batch_queue 的选择」）；3. 三层执行架构（含「Worker 的具体职责」「WorkerWrapperBase 包装层」「worker_cls="auto" 的解析」「model_runner 的 v1/v2」）；4. collective_rpc（含 4.1 / 4.2）；5. 每回合四动作与一次请求往返；6. 类与进程大全辨析（命名规律 / APIServer 的类 / EngineCore 的类 / 继承与持有关系全景 / 最易混淆的对照 / 类与进程的边界）；来源与范围说明。
- 机械验证：`.dojo/scripts/validate.py wiki/vllm-v1-two-process-arch/index.html` → `validation ok`；两条内链 `../vllm-framework-map/index.html`、`../vllm-mm-unified-embeds-cudagraph/index.html` 均存在；页面无指向 research/ 的链接，无「（待生成）」占位。

## 问题

- [重要·技术] 6. EngineCore 进程的类表（index.html:238）：`RayExecutor` 这个类名在基线源码中不存在，读者按该名检索不到对应类。｜引文依据：`vllm/v1/executor/ray_executor.py:64: class RayDistributedExecutor(Executor):`；`vllm/v1/executor/ray_executor_v2.py:206: class RayExecutorV2(MultiprocExecutor):`；`vllm/v1/executor/abstract.py:61-69` 中 `distributed_executor_backend == "ray"` 时依 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` 分别选 `RayExecutorV2` / `RayDistributedExecutor`；对基线仓库检索 `class RayExecutor` 无任何命中。该行「用 Ray actor 管理 worker 的实现」的描述本身成立（`ray_executor.py:151 _init_workers_ray` 使用 `RayWorkerWrapper` actor）。｜修复要求：把该行类名改为 `RayDistributedExecutor`（或并列 `RayExecutorV2`），描述文字保留。｜修复：｜复验：

- [轻微·技术] 2. 回合内的钩子（index.html:138）：`pre_engine_step`、`post_engine_step` 由本地 fork 的 `vllm.hcf_mixin` 注入，上游 vLLM 的同名流程中没有这两个名字，正文把它们与 `post_step` 并列陈述为引擎行为，未在正文标注其来源层级。｜引文依据：`vllm/hcf_mixin/v1/engine/core_mixin.py:28 def pre_engine_step(self)`、`:32 def post_engine_step(...)`；上游 `vllm/v1/engine/core.py`（github-community/vllm、github-wendadawen/vllm）中 `pre_engine_step`/`post_engine_step` 均无命中，仅有 `post_step`。｜修复要求：在该处补一句说明这两个钩子来自本地 fork 的 `hcf_mixin`（页面来源表已列出该文件），使读者在比对上游代码时不致检索落空；或明确接受「本页仅面向该本地检出」。｜修复：｜复验：

- [轻微·格式] 三张内联 SVG（index.html:121-135、155-167、246-287）：图内的代码标识符（`_process_input_queue`、`_process_engine_step`、`Executor`、`EngineCoreProc`、`Worker` 等）写在 `<text>` 元素中，未按规范用 `foreignObject` 承载。｜引文依据：`guides/note.md` 第 40 行「图内公式与代码标识符用 foreignObject 承载」；页面形如 `<text x="110" y="65" text-anchor="middle" font-size="14" font-family="monospace" fill="var(--text)">_process_input_queue</text>`。｜修复要求：把图内代码标识符移入 `<foreignObject>`（纯中文标签可保留在 `<text>`）；若确认仓库对无公式的等宽标识符沿用 `<text>` 是既有惯例，则在结论中写明接受理由。｜修复：｜复验：

- [轻微·表述] APIServer 的类表与来源与范围说明（index.html:224、327、331、334）：含元话语、以「本页」为主语的自我指代与复现叙事。｜引文依据：`224: …注意与 <code>EngineClient</code> 是两个东西`；`327: <p>以下为实测观察；原始 trace 记录已不随页面留存…`；`331: …本页源码核对基线为本地检出 <code>v0.20.0-718-gee4ed6d71d</code>`；`334: …（该 torch 版本 profiler 的 with_stack 无法捕获调用栈，已用最小用例验证）`。｜修复要求：改为无人称陈述，例如「`EngineCoreClient` 与 `EngineClient` 是两个不同的接口：……」；「实测环境的版本串已无对应检出，源码核对基线为 `v0.20.0-718-gee4ed6d71d`」；「该 torch 版本 profiler 的 `with_stack` 不能捕获调用栈」——删除「以下为…」「本页」「已用最小用例验证」。｜修复：｜复验：

- [轻微·格式] head 的 `dojo:summary`（index.html:7）把类名写作 `WorkerWrapper`，实际类名为 `WorkerWrapperBase`，与正文第 172、239 行的写法不一致。｜引文依据：`<meta name="dojo:summary" content="…WorkerWrapper、Worker、GPUModelRunner 等全部类…">`；`vllm/v1/worker/worker_base.py:191 class WorkerWrapperBase:`。｜修复要求：summary 中改为 `WorkerWrapperBase`。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（关闭上述 1 项重要与 4 项轻微后可发布）

### 核对情况说明（支撑上述结论）

已逐条回源核对且与基线一致的关键论断（引文依据）：

- 忙等循环与代码片段：`vllm/v1/engine/core.py:1517-1523` `while self._handle_shutdown(): … self._process_input_queue() … self._process_engine_step()`，与页面第 113-117 行片段一致。
- 回合内钩子顺序：`core.py:1558-1570` 依次 `self.pre_engine_step()` → `outputs, model_executed = self.step_fn()` → `self.post_engine_step(outputs, model_executed)` → `self.post_step(model_executed)`；第 1581-1584 行条件休眠 `if not model_executed and self.scheduler.has_requests(): time.sleep(0.001)`。
- `post_step` 条件：`core.py:587-609` `if (not self.async_scheduling and self.use_spec_decode and model_executed and self.batch_queue_size == 1): … update_draft_token_ids`，与页面第 138 行改写后的表述一致。
- `log_error_detail`：`core.py:420-432` `@contextmanager` + `except Exception as err: dump_engine_exception(self.vllm_config, scheduler_output, self.scheduler.make_stats()); raise err`。
- `step_fn` 选择与 batch 队列：`core.py:199-205` `self.batch_queue_size = self.model_executor.max_concurrent_batches`，`> 1` 时建 `deque(maxlen=…)`；`core.py:233-235` `self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue`；`vllm/v1/executor/uniproc_executor.py:76-78` `max_concurrent_batches = 2 if self.scheduler_config.async_scheduling else 1`。
- `async_scheduling` 默认自动开启：`vllm/config/scheduler.py:182 async_scheduling: bool | None = None`；`vllm/config/vllm.py:995-1035` `elif … is None:` 分支在 pooling 模型、不兼容推测解码、不支持的 executor 下置 False，否则置 True。
- 请求类型分发：`core.py:1621-1649 _handle_client_request` 处理 `WAKEUP/ADD/ABORT/UTILITY/EXECUTOR_FAILED`；输入队列由 `process_input_sockets`（`core.py:1730+`，`MsgpackDecoder(EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver)`）投递——与页面第 118 行改写后的表述一致。
- EngineCore/EngineCoreProc：`core.py:106 class EngineCore(HyKVEngineCoreMixin, EngineCoreMixin)` docstring `"""Inner loop of vLLM's Engine."""`；`core.py:1157 class EngineCoreProc(EngineCore)`；`core.py:1417 run_engine_core`；进程创建点 `vllm/v1/engine/utils.py:149-150 context.Process(target=EngineCoreProc.run_engine_core, …)`；TP>1 侧 `vllm/v1/executor/multiproc_executor.py:724 context.Process(target=WorkerProc.worker_main, …)`、`:575 class WorkerProc`。
- Worker 方法集：`vllm/v1/worker/gpu_worker.py` 存在 `init_device(269)/load_model(379)/determine_available_memory(402)/initialize_from_config(587)/compile_or_warm_up_model(616)/execute_model(831)/sample_tokens(825)/check_health(1020)/sleep(181)/wake_up(205)/add_lora(1008)/remove_lora(1011)/list_loras(1014)/pin_lora(1017)/profile(937)`。
- WorkerWrapperBase：`vllm/v1/worker/worker_base.py:191`，docstring 自述「remembers the worker module and class name… real initialization happens in init_worker」；`:331 def __getattr__`；`:341-347 execute_model` 先 `self._apply_mm_cache(scheduler_output)` 再 `self.worker.execute_model`；`init_worker` 用 `resolve_obj_by_qualname(parallel_config.worker_cls)`（`:254-256`）。
- worker_cls 解析：`vllm/config/parallel.py:255 worker_cls: str = "auto"`；`vllm/platforms/cuda.py:248-249 if parallel_config.worker_cls == "auto": parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"`；`vllm/platforms/__init__.py:60-78 cuda_platform_plugin` 用 `pynvml.nvmlDeviceGetCount() > 0`，`:262-278 __getattr__` 懒加载 `current_platform`，`:212 resolve_current_platform_cls_qualname` 逐个调用插件函数。
- runner v1/v2：`vllm/config/vllm.py:71 DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset({"Qwen3ForCausalLM"})`；`:549-560 _is_default_v2_model_runner_model` 要求 `runner_type == "generate"`、命中名单、`not is_moe and not is_quantized`；`:524-546` 另要求 `HAS_TRITON`；`gpu_worker.py:347-361` v2 从 `vllm.v1.worker.gpu.model_runner` 导入 `GPUModelRunner as GPUModelRunnerV2`，v1 从 `vllm.v1.worker.gpu_model_runner` 导入——两者同名 `GPUModelRunner`，与页面第 179 行一致。
- collective_rpc：`uniproc_executor.py:79-107` 阻塞/非阻塞两分支均调用 `run_method(self.driver_worker, method, args, kwargs)`；`vllm/v1/serial_utils.py:486-510 run_method` 对 `str` 走 `getattr`、对 `bytes` 走 `cloudpickle.loads`、对可调用对象走 `partial`；`multiproc_executor.py:405-409 if isinstance(method, str): send_method = method else: send_method = cloudpickle.dumps(...)`，`:409 self.rpc_broadcast_mq.enqueue(...)`，`:415-431 get_response` 聚合 `response_mqs`，`:433 FutureWrapper`。
- 广播队列：`vllm/distributed/device_communicators/shm_broadcast.py:358 class MessageQueue`，`:209 class ShmRingBuffer`；`enqueue`（`:720-763`）在 `total_bytes + len(buf) >= self.buffer.max_chunk_bytes` 时置 overflow 并 `self.local_socket.send_multipart(...)`（XPUB），否则写入共享内存环形缓冲——与页面第 194 行一致。
- 调度四动作：`vllm/v1/core/sched/scheduler.py:406 def schedule` → `SchedulerOutput`（`vllm/v1/core/sched/output.py:183`）；`vllm/v1/engine/core.py:636 model_executor.execute_model(..., non_block=True)`、`:667 model_executor.sample_tokens(grammar_output, non_block=True)`；`scheduler.py:1577 def update_from_output`。
- 请求 token 计数：`vllm/v1/request.py:146 self.num_computed_tokens = 0`、`:281 def num_tokens_with_spec`。
- 类清单：`vllm/engine/protocol.py:40 class EngineClient(ABC)`（含 `generate:65`、`encode:87`、`abort:102`）；`vllm/v1/engine/async_llm.py:71 class AsyncLLM(EngineClient)`（`:415 _add_request`、`:657 _run_output_handler`）；`vllm/v1/engine/core_client.py:71 EngineCoreClient(ABC)`、`:293 InprocClient`、`:480 MPClient`；`vllm/v1/engine/llm_engine.py:47 class LLMEngine`，docstring `"""Legacy LLMEngine for backwards compatibility."""`；`vllm/v1/engine/output_processor.py:42/251` 持有 `IncrementalDetokenizer`；`vllm/v1/executor/abstract.py:49 get_class`；`vllm/v1/executor/uniproc_executor.py:26 AsyncOutputFuture`、`multiproc_executor.py:82 FutureWrapper`。
- ZMQ 地址：`vllm/v1/engine/utils.py:960-1005 get_engine_zmq_addresses`，本机 `get_open_zmq_ipc_path()`（ipc://），跨机 `get_tcp_uri(host, …)`（tcp://host:port）。
- 线程限制限定：`multiproc_executor.py:1057-1086 set_multiprocessing_worker_envs` 置 `default_omp_num_threads = 1` 并写 `os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)`；`uniproc_executor.py` 无对应代码——与页面「范围与限定」一致。
- 未核实项（页面已自分标注）：trace 观察（23 个回合、983 token prefill 单回合、函数热点分布、`step_with_batch_queue` 出现而 `step` 未出现）来自实测环境 vLLM 0.22.2.dev / torch 2.11.0+cu130 / Qwen3.5-0.8B / TP=1，原始记录已不留存，页面第 327 行已声明「不可再复核」，按规范属已标注的实测观察，不计为不合格论断。
