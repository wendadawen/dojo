<!-- review-meta
round: 2
page: wiki/vllm-mm-unified-embeds-cudagraph/index.html
reviewed_content_sha256: 01f590bf1abe155a
-->
# 多模态统一输入与 cudagraph 审查记录（第 2 轮）

- 页面版本：wiki/vllm-mm-unified-embeds-cudagraph/index.html 工作树 git hash-object fddb2bc16a0ec86249f8d84451a4c9322acc7a5a
- 页面类型：`dojo:type=note`，按 `guides/note.md` 审查；记录格式按 `guides/concept/check.md` 第 3 节
- 审查时间：2026-09-13 19:07
- 审查者：独立子代理（未参与写作，未参与前序审查与修复）
- 已完整阅读章节：head 元数据与页内脚本；§1 问题：两种输入格式与 cudagraph 的固定性矛盾；§2 解法：统一成 inputs_embeds（含内联 SVG 与 §2.1 混合 batch、§2.2 全纯文本 batch、§2.3 对比：纯文本模型走 input_ids、§2.4 v0 双编译到 v1 单图）；§3 prefill 与 decode 的模式差异（含表格与正文）；§4 来源与范围说明（源码定位 / trace 证据 / 边界）
- 来源核对方式（逐条回到源码，非只看页面自身表述）：
  - vLLM 官方源码检出 1：`/Users/wendadawen/code/github/github-community/vllm`，HEAD `5ac2684`（v0.26.1rc0，2026-08-07）：`vllm/v1/worker/gpu_model_runner.py`、`vllm/model_executor/models/qwen3_vl.py`、`vllm/model_executor/models/utils.py`、`vllm/v1/cudagraph_dispatcher.py`、`vllm/config/compilation.py`
  - vLLM 官方源码检出 2（用于定位页面所标行号）：`/Users/wendadawen/code/github/github-wendadawen/vllm` 历史提交 `03878d1c2`（2026-06-11）等
  - 页内互链页：`wiki/vllm-cudagraph/index.html`、`wiki/vllm-mm-image-two-stage/index.html`（均存在）
- 未读：`wiki/vllm-mm-unified-embeds-cudagraph/research/`（规范禁止读取规划、修复与前序审查记录；仅列目录确认文件构成）
- 机械验证：`.dojo/scripts/validate.py wiki/vllm-mm-unified-embeds-cudagraph/index.html` → `validation ok`。页面无 <details> 折叠块（章节折叠由脚本生成）、无可运行代码块、无 KaTeX 公式，故「代码执行」「公式复算」改为静态审查。

## 问题

- [重要·来源] index.html:161–165（§4「trace 证据」）：§3 结论所依赖的三个数字（prefill 单批 983 token、decode 段 19 次 replay、prefill 段约 85 ms）标注来自「EngineCore 进程 trace」，但本页 `research/` 目录内无 `measured.md` 登记，页面也未给出可访问的实测产物或外部链接，独立审查者无法写出任何核对片段，按 check.md 2.2 属「未核对」条目；页面亦未把这几个数字标注为「未核实/推断」。引文依据：index.html:161–165 原文「……prefill 段（execute_context_1(983)_generation_0(0)，约 85ms）内 0 次 replay；其后 decode 段 19 次 replay（torch/cuda/graphs.py 的 replay）」；同目录 `ls wiki/vllm-mm-unified-embeds-cudagraph/research/` 仅返回 `review-1.md`，无登记表（同仓他页如 `wiki/qwen3-5-dataflow/research/measured.md` 对已移除产物建了登记表）。修复要求：把该 trace 产物登记进本页 `research/measured.md`（文件名/体积/说明，与他页格式一致）；或删去这三个数字，改用不依赖 trace 的表述——§3 结论（大 prefill 走 eager）本可直接由源码推出（见「已核对且通过」第 9 条），无需 trace 数字支撑。

- [轻微·表述] index.html:95（§1 首句）：「被 support_torch_compile 装饰的模型主体（如 Qwen3LLMModel）的 forward 里有一个输入分支」。该 forward 实际是 `if inputs_embeds is not None … else …` 两支，「一个输入分支」与代码不符，易被读成只有一支。引文依据：`qwen3_vl.py` 的 `Qwen3LLMModel.forward`：`if inputs_embeds is not None: hidden_states = inputs_embeds` / `else: hidden_states = self.embed_input_ids(input_ids)`（作者 Jue 2026 检出一致）。修复要求：改为「forward 对输入来源做了判断：`inputs_embeds` 非空则直接使用，否则用 `input_ids` 过 embedding 层」。

- [轻微·表述] index.html:136（§3 表格表头）：「场景」用作列名，属把「场景」当术语（check.md 2.2 表述维度明确列为不合格表述）；该列内容实为执行阶段/批次类型。引文依据：不适用。修复要求：改为「执行阶段」或「批次类型」。

- [轻微·格式] index.html:106–119（§2 内联 SVG）：图内代码标识符用普通 `<text>` + `font-family="monospace"` 承载，且写作「is multimodal」（115 行）、「inputs embeds」（112 行，aria-label 亦同），与正文的 `is_multimodal`、`inputs_embeds` 写法不一致；未按 `guides/note.md`「图内公式与代码标识符用 foreignObject 承载」处理。引文依据：不适用。修复要求：图内标识符与正文完全一致；按 note.md 用 `<foreignObject>` 承载代码标识符，或改写为不含代码标识符的中文描述。

- [轻微·表述] index.html:101（§1 末句）与 index.html:128（§2.3 末句）：「管理与调度都会变复杂」「自然选择让 embedding 层入图」为无来源支持的判断，被写成结论。引文依据：`gpu_model_runner.py` 对应注释只说明「The v0 engine avoids this by "double compiling" the CUDA graph…」，未评价调度复杂度，也未给「自然选择」的依据。修复要求：删去「管理与调度都会变复杂」与「自然选择」，或改为可核对的陈述（如「需要为两种输入各维护/编译一张图」）。

## 已核对且通过

- §1 引用注释与 v0 说明属实：`gpu_model_runner.py` 多模态分支内含 `# NOTE(woosuk): To unify token ids and soft tokens (vision embeddings), we always use embeddings (rather than token ids) as input to the multimodal model, even when the input is text.`；文本分支注释含 `it is not desirable for performance since then the embedding layer is not included in the CUDA graph`；`enable_prompt_embeds` 分支注释含 `The v0 engine avoids this by "double compiling" the CUDA graph, once with input_ids and again with inputs_embeds, for all num_tokens.`（与页面 §2.4 引文逐字一致）。
- §2 三个条件属实：分支条件为 `if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:`，与页面「模型支持多模态、流水线第一段、非编解码架构」一致。
- §2 覆写语句属实：`vllm/model_executor/models/utils.py` 的 `_merge_multimodal_embeddings` 内 `inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)`，与页面「inputs_embeds[is_multimodal] = 图像特征」一致。
- §2.2「空特征列表跳过覆写」属实：`qwen3_vl.py` 的 `embed_input_ids` 内 `if multimodal_embeddings is None or len(multimodal_embeddings) == 0: return inputs_embeds`；`_execute_mm_encoder` 在 `if not mm_kwargs: return []` 处直接返回。
- §2 图注掩码语义属实：True 位置换入 mm 特征、False 位置保留文本 embedding（同 `_merge_multimodal_embeddings` 的布尔索引语义）。
- §2「按图片哈希缓存」属实：`gpu_model_runner.py` 中 `# mm_hash -> encoder_output`、`self.encoder_cache[mm_hashes[i]] = …`。
- §3「dispatch 查表、超限或未命中返回 NONE」属实：`vllm/v1/cudagraph_dispatcher.py::CudagraphDispatcher.dispatch` 中 `if (not self.keys_initialized or self.cudagraph_mode == CUDAGraphMode.NONE or max_size is None or num_tokens > max_size or allowed_modes <= {CUDAGraphMode.NONE}): return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)`。
- §3 模式定义与默认值属实：`vllm/config/compilation.py` 的 `CUDAGraphMode` 含 `FULL = 2`、`FULL_AND_PIECEWISE = (FULL, PIECEWISE)`，docstring 记 `FULL_AND_PIECEWISE. (v1 default)`，并写明 `Capture full cudagraph for decode batches and piecewise cudagraph for prefill and mixed prefill-decode batches.`，与页面「decode 用 FULL、prefill 与混合批次用 PIECEWISE」一致；`decode_mode() = value[0] = FULL`。
- §3「983 token 超出捕获尺寸上限」与默认上限相符：`max_cudagraph_capture_size` 默认 `min(max_num_seqs*2, 512)`（docstring），983 > 512 → dispatch 返回 NONE，页面的「NONE，eager」与该推导一致。
- §4 源码定位可核对：页面所标行号 `gpu_model_runner.py` 多模态分支「3432 起」与 `compilation.py` 模式说明 docstring「590 起」，在作者所用构建（`03878d1c2` 2026-06-11）附近可对应（该提交 `The mode of the cudagraph:` 在 590 行、`supports_mm_inputs and is_first_rank` 在 3430 行，偏差 ≤2 行），不构成行号错误。
- 表述维度其余项：通读全文（含图注与表格）未发现元话语（「本页将…」「下面来看…」「需要注意的是」）、会话指代（我/我们/你）、调试叙事与临场评价、抽象名词堆叠式 AI 拼接腔；页面级结构（标题/导语/来源与范围说明）齐备，标题与导语职责不同。
- 元数据与机械项：`dojo:type=note` 与正文形态相符；`description` 为纯文本、`dojo:summary` 无未渲染公式；两个互链页 `wiki/vllm-cudagraph/index.html`、`wiki/vllm-mm-image-two-stage/index.html` 真实存在；无「（待生成）」占位与悬空的 `research/` 路径引用；validate.py 通过。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（重要项「trace 数字无登记产物、无法核对」须登记或删去后复验；四项轻微表述/格式问题一并修正；核心技术论断与来源核对全部通过，无阻断项）