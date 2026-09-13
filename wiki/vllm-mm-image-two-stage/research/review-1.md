<!-- review-meta
round: 1
page: wiki/vllm-mm-image-two-stage/index.html
reviewed_content_sha256: 26f1beb5c08af4cc
-->
# 多模态图片的两阶段处理 审查记录（第 1 轮）

- 页面版本：`3f46328ef661237fc2ea50161eed60b111e7a1df`（wiki/vllm-mm-image-two-stage/index.html 工作树 blob 哈希，2026-09-13 17:57 最后提交）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 适用规范：`dojo:type = note` → `guides/note.md`
- 已完整阅读章节：导语与 page-meta、1. 分界在哪里、2. 前端段、3. 后端段（收集／合批机制／编码／缓存）、4. 为什么这样分、来源与范围说明（源码定位／验证数据／边界），以及两张内联 SVG 的全部图内文字
- 核对来源：被页面引用为 smart_resize 出处的 transformers `models/qwen2_vl/image_processing_qwen2_vl.py`（本地 transforms 4.57.6 实跑）；vLLM 官方仓库 `vllm-project/vllm`（v0.16.0 tag 与 main 分支）的 `vllm/multimodal/utils.py`、`vllm/multimodal/inputs.py`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/renderers/base.py`、`vllm/renderers/hf.py`、`vllm/transformers_utils/processor.py`、`tests/multimodal/test_utils.py`；vLLM 官方 API 文档（docs.vllm.ai v0.16.0 / latest）；vLLM 相关 PR #38018、#49686
- 机械验证：`python3 .dojo/scripts/validate.py wiki/vllm-mm-image-two-stage/index.html` → `validation ok`；页内两处内链 `../vllm-v1-two-process-arch/`、`../vllm-mm-unified-embeds-cudagraph/` 均真实存在；正文无「我／我们／你」、「本页将／下面来看／需要注意的是」等会话指代与元话语标记
- 说明：本页 `wiki/vllm-mm-image-two-stage/` 下只有 index.html，无 research/ 目录、无 research/measured.md，故页面「验证数据」列举的实测产物（smart_resize 数值、合批实验、profiler trace）在仓库内均无登记可查（见 I1）。

## 问题

- [阻断·技术] 来源：页面自引的 smart_resize（transformers `models/qwen2_vl/image_processing_qwen2_vl.py`，页面「源码定位」标为行 62）+ 本地实跑｜位置：行 141（正文），行 193（验证数据）｜问题：页面以「两个真实例子」给出 `980×986 → grid [1, 62, 62]`（3844 patch）与 `2310×2310 → grid [1, 144, 144]`（20736 patch），并称由模型自带 HF processor「实际运行得到」。用页面引用的同一 smart_resize（默认 factor=28、min_pixels=56×56、max_pixels=28×28×1280）与 `Qwen2VLImageProcessor` 默认配置实跑，两图都得到 `image_grid_thw = [[1, 70, 70]]`（4900 个 patch 位置、1225 个合并 token），与页面数字均不符。量纲上也不成立：`[1,62,62]` 要求缩放到 1736×1736、`[1,144,144]` 要求缩放到 4032×4032，均大于原图，只有 min_pixels 约 3.0M / 16.3M（两图互不一致）时 smart_resize 才会上采样，任何合理配置都得不到这两个网格｜引文依据：页面行 141「两个真实例子：980×986 的图片被缩放后切为 grid [1, 62, 62]，对应 3844 个 patch；2310×2310 的图片切为 grid [1, 144, 144]，对应 20736 个 patch」、行 193「smart_resize 数值：对 testdata 真实图片（980×986 与 2310×2310）用模型自带的 HF processor 实际运行得到」；实测输出 `smart_resize(980,986)=(980,980)`、`smart_resize(2310,2310)=(980,980)`，`Qwen2VLImageProcessor()(images=...)["image_grid_thw"]` 两图均为 `[[1, 70, 70]]`｜修复要求：用实际 processor 重新跑并按真实值改写（默认配置下两图同为 [1,70,70]，则应改用真正产生不同网格的两张图），或删除这两个例子；同时补出该模型 preprocessor 配置里的 min_pixels / max_pixels 取值｜修复：｜复验：

- [阻断·技术] 来源：被页面引用的 smart_resize 与 image_grid_thw 语义（transformers `image_processing_qwen2_vl.py`）｜位置：行 141｜问题：页面写「把图片高宽各自取整到 factor=28 的倍数……再除以 28 得到网格数」。实际 `image_grid_thw` 是「缩放后高/宽 ÷ patch_size」：Qwen2-VL/Qwen3-VL 的 patch_size=14、merge_size=2，factor=28 只是 smart_resize 的取整粒度（= patch_size×merge_size），网格按 14 切。980×980 缩放后得到 `[1,70,70]`，70 = 980 ÷ 14，而非 ÷ 28。页面把换算系数写错为 28，且与该页自己给出的 [1,62,62] 也不自洽（980 ÷ 28 = 35）。这是算式与结论不符｜引文依据：页面行 141「……尽量保持宽高比，再除以 28 得到网格数」；实测 `image_grid_thw=[1,70,70]` 对应 980÷14｜修复要求：改为「除以 patch_size（Qwen 系列为 14）得到网格数」，或写成「按 patch_size 切 patch，再按 merge_size（2）合并成视觉 token，token 网格 = 网格数 ÷ merge_size」｜修复：｜复验：

- [重要·技术] 来源：不适用（来源缺失）｜位置：行 111，行 193、194、195（验证数据）｜问题：页面用确定语气陈述三组实测结论——(a) 两份 profiler trace 的函数分布（行 111：APIServer trace 出现 `fetch_image`、`processor.py`、`renderers/`，无视觉编码器 forward；EngineCore trace 出现 `_execute_mm_encoder`、`_process_image_input`、`embed_multimodal`；APIServer 内部 `load_file`/`rgba_to_rgb` 与 HF processor 调用分处不同线程），(b) smart_resize 数值（行 193），(c) 合批实验（行 194）。本页无 research/ 目录、无 research/measured.md，这些实测产物在仓库内没有任何登记文件，无法核对，属「定位不到来源」的事实性论断；其中 (a) 的「进程内按线程分离 IO 与多模态处理」仅有 trace 这一条证据｜引文依据：页面行 111 全段；行 195「trace 证据：APIServer 进程 trace（预处理函数分布）与 EngineCore 进程 trace（编码函数分布）」；仓库内 `wiki/vllm-mm-image-two-stage/` 仅含 index.html（`git ls-files` 与目录列举均无 research/）｜修复要求：把三组实测（trace 关键帧、smart_resize 输入输出、合批实验脚本与结果）登记进 research/measured.md 并在页面注明；无法登记的一条（如线程分布）降级为明确标注的「推断」，不得以事实陈述保留｜修复：｜复验：

- [重要·技术] 来源：不适用（来源缺失）｜位置：行 179（§4 为什么这样分）｜问题：页面把设计动机写成定论——「把重活留在离 GPU 最近的地方、轻活留在离请求最近的地方，是这条分界线的动机」。这是对该分界线成因的推断，页面未给任何来源（vLLM 设计文档／PR／注释）支持，按 note 规范「推测、评价和建议必须有依据，并标注其性质」应标注为推断而非结论｜引文依据：页面行 179「……是这条分界线的动机」；同页「来源与范围说明」未列任何关于该动机的官方材料｜修复要求：补官方依据（V1 架构文档／相关 PR），或改写为明确标注的推断（如「按此分界的功能划分，其动机可理解为……」）并去掉「是……动机」的定论语气｜修复：｜复验：

- [重要·技术] 来源：vLLM 仓库 `tests/multimodal/test_utils.py`（v0.16.0 tag 与 main 分支）｜位置：行 196｜问题：页面把单测来源写作「`tests/multimodal/test_utils.py` 的 `TestGroupAndBatchMmItems`」，但该文件在两个版本中都不存在名为 `TestGroupAndBatchMmItems` 的类，而是三个独立 test 函数（main：`test_group_and_batch_mm_items_split_by_fieldset`、`test_group_and_batch_mm_items_split_by_shared_data`、`test_group_and_batch_mm_items_splits_shared_data_by_dtype`）。页面所称「覆盖后两种拆批条件」的内容成立，但给出的标识符定位不到｜引文依据：页面行 196「社区仓库 tests/multimodal/test_utils.py（TestGroupAndBatchMmItems，本地安装包未含 tests，依据仓库路径与测试内容核对）」；仓库实际为独立 test 函数，无该类｜修复要求：把标识符改为实际存在的两个函数名（`test_group_and_batch_mm_items_split_by_fieldset`、`test_group_and_batch_mm_items_split_by_shared_data`），或改为中性表述「该类测试位于 tests/multimodal/test_utils.py」｜修复：｜复验：

- [轻微·表述] 来源：不适用｜位置：行 158–170（合批机制 SVG）｜问题：该图同时承载「合并方式（由字段类型决定）」与「拆批条件（group id 不同才拆）」两组关系，违反 note 规范「每张图只表达一个关系」｜引文依据：guides/note.md「图示……每张图只表达一个关系。图示无法降低理解成本时，改用表格或文字」；图内两组标题文字见行 159、164｜修复要求：拆成两张 SVG（一张画合并方式、一张画拆批条件），或把其中一组改为表格／正文；合并方式已由行 151–155 的表格承担，可将拆批条件单独成一图｜修复：｜复验：

- [轻微·技术] 来源：vLLM `SupportsMultiModal` 协议（vllm/model_executor/models/interfaces.py）｜位置：行 173｜问题：页面把协议能力写成全部结论——「协议只要求对象具有这些多模态方法，不依赖具体模型类，因此任何实现了该组方法的视觉语言模型都能走同一条编码路径」。末句是外推的普适论断，页面未给依据也未标注为推断｜引文依据：页面行 173「……因此任何实现了该组方法的视觉语言模型都能走同一条编码路径」｜修复要求：删去「任何……都能」的普适句，或降级为「按协议约定，实现该组方法的模型即可复用同一编码调用路径（本页仅以 Qwen3.5 走读验证）」｜修复：｜复验：

## 结论

- 统计：阻断 2 / 重要 3 / 轻微 2
- 处置：修复。两个阻断项都在行 141 的 smart_resize 段落：两个示例网格值与换算系数 28 均与被引用的源码相悖，必须用真实 processor 重跑后改写（或删除示例）并纠正「÷28」为「÷patch_size(14)」，同时补出模型 preprocessor 的 min_pixels/max_pixels。三个重要项分别要求登记或降级实测证据、为「分界动机」补来源或标推断、把不存在的单测类名改为真实函数名。图表拆分与普适句收窄为轻微项。改动后重跑 `validate.py`。
- 已核对无误的项（留档）：
  - 合批机制与源码一致：`vllm/multimodal/inputs.py` 的 `MultiModalBatchedField.reduce_data` 在各 item 形状相同时 `torch.stack`、不同时返回 list；`MultiModalSharedField.reduce_data` 返回 `batch[0]`；`PlaceholderRange` 字段为 `offset`/`length`；`_get_group_hash(elem)` 仅对 `MultiModalSharedField` 计算哈希、其余（含 BatchedField）返回 None；`vllm/multimodal/utils.py` 的 `group_and_batch_mm_items` 只合并相邻项，`group_and_batch_mm_kwargs` 先按 modality 分组再批（与页面行 153–156 描述吻合）。
  - 后端链路函数与方法存在且与描述吻合：`_batch_mm_inputs_from_scheduler` 返回 `(list[str] mm_hashes, list[(req_id, MultiModalKwargsItem)], list[(req_id, PlaceholderRange)])`（对应「三个平行列表」）；`encoder_cache` 以 mm_hash 为键；`_gather_mm_embeddings` 存在并按占位符取片段；`SchedulerOutput.scheduled_encoder_inputs` 为 `dict[str, list[int]]`（对应「{请求ID: 图片下标列表}」）；`_execute_mm_encoder` 在 `_preprocess` 中被调用。
  - 前端链路函数名存在：`render_chat_request`（`vllm/entrypoints/serve/render/serving.py`，`OpenAIServingRender`）、`HfRenderer.render_messages_async`、`parse_chat_messages_async`、`_process_multimodal`（`vllm/renderers/base.py`，其内调用 `mm_processor.apply`）、`call_hf_processor_mm_only`（`vllm/transformers_utils/processor.py`，PR #38018 引入）、`fetch_image`、`rgba_to_rgb` 均可在官方仓库/文档中定位到。
  - 结构与机械项：validate.py 通过；标题与导语职责不同；每章单一主题；正文无会话指代与元话语标记；两处内链目标页存在。
