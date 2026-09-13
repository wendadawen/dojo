<!-- review-meta
round: 1
page: wiki/vllm-cudagraph/index.html
reviewed_content_sha256: ea0ee03c7969ae96
-->
# torch.compile 图捕获与 CUDA Graph 审查记录（第 1 轮）

- 页面版本：wiki/vllm-cudagraph/index.html 工作树 sha256 b0a30a7281bd8f3e
- 页面类型：`dojo:type=note`，按 `guides/note.md` 审查；记录格式按 `guides/concept/check.md` 第 3 节
- 审查时间：2026-09-13 18:47
- 审查者：独立子代理（未参与写作，未参与前序审查与修复）
- 已完整阅读章节：head 元数据与页内脚本；§1 torch.compile 的两阶段流程；§2 Dynamo 图捕获的处理边界（含表格与两段正文）；§3 CUDA Graph 的录制与重放（含四个列表项）；§4 捕获尺寸的确定；§5 来源与范围说明
- 来源核对方式：
  - vLLM 源码（本地检出 `/Users/wendadawen/code/github/github-community/vllm`，HEAD `5ac2684`，2026-08-07）：`vllm/config/compilation.py`、`vllm/config/vllm.py`、`vllm/v1/cudagraph_dispatcher.py`、`vllm/v1/worker/gpu_model_runner.py`
  - PyTorch 官方文档与源码：`docs/source/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.md`、`programming_model.skipped_functions.md`、`docs/source/torch.compiler.rst`；`torch/_dynamo/exc.py`、`torch/_dynamo/variables/functions.py`、`torch/_dynamo/convert_frame.py`
  - 页内互链页 wiki/vllm-mm-unified-embeds-cudagraph/index.html（核对 §4 的 FULL/eager 对照）
- 未读：wiki/vllm-cudagraph/research/（规范禁止读取）
- 机械验证：`.dojo/scripts/validate.py wiki/vllm-cudagraph/index.html` → `validation ok`。页面无公式、无插图、无折叠块（章节折叠由脚本生成）、无可运行代码块，故 KaTeX 渲染、图示、代码执行三项改为静态审查。

## 问题

- [阻断·技术] index.html:107（§2 表格第 3 行）、109–110（正文）：把「C 扩展被 Dynamo 标记为 skipped」写成**硬性失败、抛出异常终止捕获**（表格「图捕获中断」，正文「Dynamo 直接抛出 Unsupported 异常终止捕获」）。来源反证该情形同样是 graph break（软性失效）：给出 "Dynamo does not know how to trace the builtin …" 说明的代码路径以 `unimplemented(...)` 结束，而该函数的 docstring 明确「Called within dynamo to cause a graph break」，且 `Unsupported` 在 `convert_frame` 中被判定为 `soft_fail`——编译继续进行，仅在 `fullgraph=True` / `error_on_graph_break` 时才成为硬错误。因此 §2「graph break 软性 / C 扩展硬性」这一全节核心区分不成立。引文依据：`torch/_dynamo/exc.py`：`def unimplemented(...)` docstring「Called within dynamo to cause a graph break.」；`torch/_dynamo/variables/functions.py`（`SkipFunctionVariable.call_function`）在 `explanation = f"Dynamo does not know how to trace the builtin \`{module_name}.{qualname}.\` …"` 之后调用 `unimplemented(gb_type="Attempted to call function marked as skipped", …)`；`torch/_dynamo/convert_frame.py:2434`：`soft_fail = isinstance(e, (Unsupported, UserError))`；官方文档 `programming_model.skipped_functions.md`：「Then `fn` is skipped and run eagerly - `inner1` and `inner2` are compiled when they are called.」。修复要求：将表格第 3 行「结果」列与正文 109 行改写为「抛出 `Unsupported`，被 `convert_frame` 判为 soft_fail → 与 graph break 同属软性失效，编译继续（仅 `fullgraph=True` 时为硬错误）」；若坚持保留「硬性失败」表述，须在正文该句处直接给出 `fullgraph=True` 前提，不得只把限制放在末尾来源说明。｜修复：｜复验：

- [重要·技术] index.html:106（§2 表格第 2 行）：把「Python 原生控制流（`if`、`for`、`len`）」一律列为 graph break，属扩大适用范围。官方文档只把**数据相关（data-dependent）**控制流列为 graph break，且把「把控制流改为常量」作为规避手段，即常量条件下的 `if`、可展开的静态 `for` 不会 graph break。另外 `len` 不属于控制流，`len` 本身可被追踪。该行未标注为推断。引文依据：`programming_model.common_graph_breaks.md`：「`torch.compile` graph breaks on data-dependent operations such as data-dependent control flow (if-statements, loops with tensors) and direct tensor data accesses (`.item`, `.data_ptr`).」；同页规避项：「If your control flow doesn't actually depend on data values, consider modifying your code to perform control flow on constants.」。修复要求：该行改为「数据相关的控制流（分支/循环条件依赖张量取值）→ graph break」，并说明常量条件下的 `if`、静态 `for` 由 Dynamo 特化/展开；从「控制流」一栏删去 `len`。｜修复：｜复验：

- [轻微·技术] index.html:110（§2 正文）：引号内报错原文与源码模板不一致。页面写作 `Dynamo does not know how to trace the builtin _hashlib.openssl_sha1.`，源码模板为 ``Dynamo does not know how to trace the builtin `{module_name}.{qualname}.` ``（函数名外有反引号，且多出的句点位于反引号内）。引文依据：`torch/_dynamo/variables/functions.py`：`explanation = (f"Dynamo does not know how to trace the builtin \`{module_name}.{qualname}.\` " f"This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).")`。修复要求：按源码模板补回反引号与句点位置，或明确标注为「节选/大意」。｜修复：｜复验：

- [轻微·技术] index.html:123（§4）：捕获尺寸表达式写为 `list(range(256, max, 16))`，其中 `max` 未被定义，且上界缺 `+ 1`。引文依据：`vllm/config/vllm.py:1814-1815`：`cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_graph_size + 1, 16))`；`vllm/config/compilation.py:700-701` docstring 同式（`range(256, max_cudagraph_capture_size + 1, 16)`）。修复要求：写为完整表达式 `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))`，并在同句说明 `max_cudagraph_capture_size` 即下一句给出上限。｜修复：｜复验：

- [轻微·技术] index.html:128（§5 来源说明）：捕获尺寸默认生成规则归到 `vllm/config/compilation.py`。该规则的可执行代码在 `vllm/config/vllm.py` 的 `VllmConfig._set_cudagraph_sizes`，`compilation.py` 仅在其字段 docstring 复述；且实际默认上限为 `min(max_num_seqs * (1 + num_speculative_tokens) * 2, 512)` 再与 `max_num_batched_tokens` 取 min，页面 123 行的 `min(max_num_seqs * 2, 512)` 只是无投机解码时的形式。引文依据：`vllm/config/vllm.py:1805` `def _set_cudagraph_sizes(self):`；同文件 1858-1864：`decode_query_len = 1 + self.num_speculative_tokens` … `max_cudagraph_capture_size = min(self.scheduler_config.max_num_seqs * decode_query_len * 2, 512)` … `max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)`。修复要求：来源补 `vllm/config/vllm.py::VllmConfig._set_cudagraph_sizes`；并在此处注明投机解码与 `max_num_batched_tokens` 会收紧默认上限（或在 §4 写明该式为 `num_speculative_tokens=0` 的情形）。｜修复：｜复验：

- [轻微·表述] index.html:95（§1 首段）：「每一条 Python 语句对应一次算子（kernel）提交」与「torch.compile …来消除这部分开销」为过强表述：一条 Python 语句可对应零个或多个算子；默认 Inductor 产物并不启用 CUDA Graph，kernel 启动开销只是被更少的算子与更短的 Python 调用路径**减少**，未被「消除」。引文依据：不适用（表述类）。修复要求：改为「通常对应一次或多次算子提交」「大幅降低这部分开销」；如要保留「消除」须限定在启用 CUDA Graph 重放的场景。｜修复：｜复验：

## 已核对且通过

- §1 两阶段：Dynamo 图捕获 + Inductor 为默认后端、GPU 上以 Triton 生成 kernel，与 PyTorch 官方文档一致（`docs/source/torch.compiler.rst`：「TorchDynamo … uses a CPython feature called the Frame Evaluation API to safely capture PyTorch graphs.」「TorchInductor is the default torch.compile deep learning compiler …」「For NVIDIA and AMD GPUs, it leverages OpenAI Triton as the key building block.」）。
- §2 表格第 1 行（张量运算→图节点）与「graph break 是软性失效」成立。
- §3 机制描述（录制 kernel 顺序/参数/输入输出地址与数据搬运、CPU 一次下发整图重放、地址固化要求固定内存池、KV cache 缓冲区按固定地址设计）与 CUDA Graph/vLLM V1 实现一致；「每步是否重放捕获图由 GPUModelRunner 查表决定」与源码一致（`vllm/v1/worker/gpu_model_runner.py:904` 构造 `CudagraphDispatcher`，`gpu_model_runner.py:2981`/`4026` 调 `self.cudagraph_dispatcher.dispatch(...)`）。
- §4 捕获尺寸以 decode 总 token 数为单位、decode 每请求每步一 token、按邻近合法尺寸**向上**补齐（`vllm/v1/cudagraph_dispatcher.py:_compute_bs_to_padded_graph_size`：`if bs == start: …[bs] = start else: …[bs] = end`，即落到下一个捕获尺寸）、超上限或未命中则退回 eager（`dispatch`：`num_tokens > max_size` 或键不存在 → 返回 `CUDAGraphMode.NONE`）均核对通过。
- §4 与互链页一致：wiki/vllm-mm-unified-embeds-cudagraph/index.html 的走读表确认「decode 批次以 FULL 模式重放」「983 token 单批超出捕获尺寸 → NONE，eager 执行」（该页 index.html:139-147、150）。
- 表述维度：通读全文（含表格与列表项）未发现元话语（「本页将…」「下面来看…」「需要注意的是」）、会话指代（我/我们/你）、调试叙事与临场评价、AI 拼接腔或把「场景」当术语的用法。
- 元数据与机械项：`dojo:type=note` 与正文形态相符；`dojo:topics=推理系统`、`dojo:tag=推理系统` 均在 `.dojo/scripts/catalog_builder.py` 词表内；`description` 为纯文本、`dojo:summary` 无未渲染公式；两个互链页 wiki/vllm-v1-two-process-arch/index.html、wiki/vllm-mm-unified-embeds-cudagraph/index.html 均真实存在；无「（待生成）」占位；validate.py 通过。

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：修复（§2 表格第 3 行与正文的「C 扩展硬性失败/终止捕获」须与 PyTorch 源码对齐后方可复验；§2 表格第 2 行须限定为「数据相关控制流」）