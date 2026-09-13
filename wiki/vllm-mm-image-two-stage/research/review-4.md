<!-- review-meta
round: 4
page: wiki/vllm-mm-image-two-stage/index.html
reviewed_content_sha256: a1618cb8e403a2f3
-->
# 多模态图片的两阶段处理审查记录（第 4 轮）

- 页面版本：25e8b9a6bcf48e05316ad9fa1237550bb694f56c
- 审查时间：2026-09-13 21:26
- 审查者：独立子代理（编排者派发的独立审查者）
- 适用规范：guides/note.md（`<meta name="dojo:type" content="note">`）
- 已完整阅读章节：页面元信息与导语 → 1. 分界在哪里：两段各自做什么 → 2. 前端段：HTTP 请求到 pixel_values → 3. 后端段：_execute_mm_encoder 的视觉编码（收集／合批机制／编码／缓存）→ 4. 为什么这样分 → 来源与范围说明（源码定位／验证数据／边界）→ 页内两处内联 SVG 图与图注 → 全部页面脚本

## 核验依据（本轮实际回源）

- 官方模型配置：抓取 HuggingFace `Qwen/Qwen3.5-0.8B` 的 `preprocessor_config.json`，得 `patch_size=16`、`merge_size=2`、`shortest_edge=65536`、`longest_edge=16777216`、`image_processor_type=Qwen2VLImageProcessorFast`、`processor_class=Qwen3VLProcessor`，与页面第 2 章所述四个参数完全一致；`config.json` 的 `architectures` 为 `Qwen3_5ForConditionalGeneration`、`vision_config.patch_size=16`、`spatial_merge_size=2`，与页面元信息/来源一致。
- 数字复算：以 transformers 4.57.6 本地安装的 `Qwen2VLImageProcessor(patch_size=16, merge_size=2, min_pixels=65536, max_pixels=16777216)` 实跑两张随机图——980×986→网格 `[1, 62, 62]`、480×640→网格 `[1, 30, 40]`，与页面逐字一致；62=992÷16、62×62=3844、30×40=1200、3844÷4=961、1200÷4=300 的算式自洽。`image_processing_qwen2_vl_fast.py` 第 49 行从 `image_processing_qwen2_vl.py` 导入同一 `smart_resize`，第 209 行同样以 `factor=patch_size*merge_size`、第 240 行 `grid=resized//patch_size` 计算，Fast 与慢实现同逻辑。
- smart_resize：`transformers/models/qwen2_vl/image_processing_qwen2_vl.py` 中 `def smart_resize` 确在第 54 行、`factor` 入参默认 28，页面标注正确。
- 符号与文件路径：对照 vLLM `v0.22.1`（`0.22.2.dev` 的前一版本）源码逐一定位——`render_chat_request`（entrypoints/openai/chat_completion/serving.py）、`render_messages_async`（renderers/hf.py:924）、`parse_chat_messages_async`、`fetch_image_async`、`_process_multimodal`（renderers/base.py:666）、`apply`／`_apply_hf_processor_mm_only`（multimodal/processing/processor.py:1663/1214）、`call_hf_processor`（processing/context.py）、`_execute_mm_encoder`（v1/worker/gpu_model_runner.py:2860）、`_batch_mm_inputs_from_scheduler`（:2817）、`_gather_mm_embeddings`（:3071）、`encoder_cache`（:528）、`_get_group_hash`／`_batch_mm_items`／`group_and_batch_mm_items`／`group_and_batch_mm_kwargs`（multimodal/utils.py）、`MultiModalBatchedField`／`MultiModalSharedField`／`PlaceholderRange`（multimodal/inputs.py）、`embed_multimodal`／`_process_image_input`（model_executor/models/qwen3_vl.py，`Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)` 继承而来）、`load_file`／`rgba_to_rgb`（multimodal/media/base.py、image.py）、`SupportsMultiModal`（gpu_model_runner.py:2910 `cast(SupportsMultiModal, self.model)`），全部存在。
- 拆批机制：`_get_group_hash` 对非 `MultiModalSharedField` 返回 `None`、对 shared 字段返回数据哈希；`group_ids` 由按字段名排序的 `(key, hash)` 元组构成、`groupby` 相邻合并——页面「只有 SharedField 参与哈希／字段名集合不同／SharedField 数据不同」三条拆批判定与代码一致；`MultiModalSharedField._reduce_data` 返回 `batch[0]`，`MultiModalBatchedField._reduce_data` 形状相同 `torch.stack`、否则 `return batch`（列表），表格描述一致。
- 单元测试：`tests/multimodal/test_utils.py` 第 187、202 行确有两个函数 `test_group_and_batch_mm_items_split_by_fieldset`、`test_group_and_batch_mm_items_split_by_shared_data`，分别构造字段集合差异与 SharedField 数据差异并断言拆批，页面引用属实。
- 站内链接：`../vllm-mm-unified-embeds-cudagraph/index.html`、`../vllm-v1-two-process-arch/index.html` 两目标页均存在。页面无外链，故无需外部抓取核对正文主张。
- 页面无 `<pre><code>` 可运行代码块；`.dojo/scripts/validate.py wiki/vllm-mm-image-two-stage/index.html` 返回 `validation ok`。

## 问题

- [轻微·可读性] 第 3 章「编码」段末括号（本页以 Qwen3.5 走读验证）：以"本页"为主语的自我指代，属应删除自我／会话指代的表述。｜引文依据：不适用（可读性）｜修复要求：改为不出现"本页"的写法，例如"（以 Qwen3.5 走读验证）"，使正文直接陈述内容。｜修复：｜复验：
- [轻微·技术] 导语"前端 render 链路用 HF image_processor 完成图片加载与 smart_resize"：把"图片加载"归入 HF image_processor 的职责，与第 1、2 章不一致——加载由连接器的 `fetch_image`／`fetch_image_async`（`load_file`、`rgba_to_rgb`）完成，image_processor 只接收已加载的图并做 resize／归一化。｜引文依据：导语"用 HF image_processor 完成图片加载与 smart_resize"；第 2 章"经连接器的 fetch_image_async 加载 → … → 最终由 HF 的 image_processor 完成 resize 与归一化"；第 1 章表格"加载图片、smart_resize、归一化"。｜修复要求：导语改为将加载与处理分属，例如"前端 render 链路经连接器加载图片后，由 HF image_processor 完成 smart_resize 与归一化…"。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布。两条轻微问题均为表述与措辞层面，不涉及数字、来源一致性与主线结论，可带理由接受；本轮未发现事实性错误、两处互相矛盾、算式与结论不符或来源不支持的情况（数字、参数、源码位置、单元测试均已回源核对一致）。