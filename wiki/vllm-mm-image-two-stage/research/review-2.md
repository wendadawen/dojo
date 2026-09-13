<!-- review-meta
round: 2
page: wiki/vllm-mm-image-two-stage/index.html
reviewed_content_sha256: a9fd7e29a7cb1d82
-->
# 多模态图片的两阶段处理审查记录（第 2 轮）

- 页面版本：`3287a741294e813ec7492926643702c0d45a9c31388db2fbe7119b572e710f49`（index.html，mtime 2026-09-13 19:18）
- 审查时间：2026-09-13 19:25
- 审查者：独立子代理（未参与写作与前序轮次；仅使用 index.html、规范与页面引用的外部来源）
- 已完整阅读章节：页首元数据与导语；1. 分界在哪里：两段各自做什么；2. 前端段：HTTP 请求到 pixel_values；3. 后端段：_execute_mm_encoder 的视觉编码（收集 / 合批机制 / 编码 / 缓存）；4. 为什么这样分；来源与范围说明（含两张内联 SVG 与全部表格，逐段通读）

## 核对记录（通过项）

- 分界表与第 1 章：APIServer 侧预处理 / EngineCore 侧编码的进程归属与函数分布，由源码位置与 APIServer、EngineCore 两份 profiler trace 交叉核对一致（trace 事件：`chat_completion/serving.py(229): render_chat_request`、`renderers/hf.py(1254): render_messages_async`、`media/connector.py(425): fetch_image_async`、`media/image.py(82): load_file`、`multimodal/image.py(19): rgba_to_rgb`、`renderers/base.py(672): _process_multimodal`；rank0 侧 `gpu_model_runner.py: _execute_mm_encoder`、`qwen3_vl.py(2676): embed_multimodal`、`qwen3_vl.py(2092): _process_image_input`）。IO 线程（load_file/rgba_to_rgb）与处理线程（_process_multimodal/get_image_processor）确为不同 tid，页面将其标为推算是保守表述、非错误。
- 第 2 章数值：以 transformers 4.57.6 `Qwen2VLImageProcessor`（min_pixels=56*56、max_pixels=28*28*1280）实跑核对——980×986 → 980×980 → grid [1,70,70]（4900 patch，1225 token）；480×640 → 476×644 → grid [1,34,46]（1564 patch，391 token）。`smart_resize(980,986)=(980,980)`、`smart_resize(480,640)=(476,644)`，70=980÷14、34=476÷14、46=644÷14，4900÷4=1225、1564÷4=391 均成立。factor=28=patch_size(14)×merge_size(2)、网格=resized÷patch_size、token=patch÷merge_size² 三项与 `vllm/model_executor/models/qwen3_vl.py`（第 929、941-943 行）一致。
- 第 3 章合批机制：`_get_group_hash`（仅 SharedField 参与哈希，其余返回 None）、`group_and_batch_mm_items` 的排序键与 `groupby` 相邻成组、`group_and_batch_mm_kwargs` 外层按 modality 分组、`MultiModalBatchedField._reduce_data`（同形状 stack / 异形状返回 list）、`MultiModalSharedField._reduce_data`（返回 batch[0]）全部与源码一致；`scheduled_encoder_inputs: dict[str, list[int]]` 与 `_batch_mm_inputs_from_scheduler` 返回的 (mm_hashes, mm_kwargs, mm_lora_refs) 与源码 docstring 一致。
- 第 3 章编码/缓存：`SupportsMultiModal`（Protocol）存在；`_gather_mm_embeddings` 按 `PlaceholderRange.offset/length` 取值、`encoder_cache.get(mm_hash)` 复用；`is_multimodal=is_mm_embed`（gpu_model_runner.py:3633）核对通过。
- 第 4 章动机已按规范标注为推断（"官方设计文档未直接陈述该动机"）。
- `.dojo/scripts/validate.py` 通过。

## 问题

- [重要·来源] 来源与范围说明·源码定位（第 177 行）<code>vllm/multimodal/utils.py</code>（fetch_image），关联正文第 111、140 行：正文称预处理函数 `fetch_image` 位于 APIServer 侧 render 链路、`parse_chat_messages_async` 调 `fetch_image` 加载图片，但来源把该函数定位到 `vllm/multimodal/utils.py`。在线链路的实际调用点是 `MediaConnector.fetch_image_async`（`vllm/multimodal/media/connector.py`），而 utils.py 的同名函数是面向用户代码的独立封装。｜引文依据：`vllm/entrypoints/chat_utils.py:1091` `await self._connector.fetch_image_async(image_url) if image_url else None`；`vllm/multimodal/utils.py:284` `def fetch_image(...)` 文档字符串含 `Warning: This method has direct access to local files and is only intended to be called by user code. Never call this from the online server!`；APIServer 侧 trace 事件名为 `vllm/multimodal/media/connector.py(425): fetch_image_async`。｜修复要求：把正文与来源定位改为 `vllm/multimodal/media/connector.py` 的 `MediaConnector.fetch_image_async`（调用点 `chat_utils.py` 的 `_image_with_uuid_async`），不再引用 utils.py 的 `fetch_image`。｜修复：｜复验：
- [轻微·来源] 来源与范围说明·验证数据（第 187 行）"…图片与音频拆为两批、字段集合不同的两张图拆为两批；后两种拆批条件另有社区单元测试覆盖"：本句自然读法下"后两种"指 modality 拆批与字段集合拆批，与第 189 行列出的两个单元测试不符——两者分别覆盖字段集合与 SharedField 数据，均不覆盖 modality 拆批；第 156 行对同一措辞的用法（指字段集合+SharedField 数据）才是正确的，故本句指代易被误读。｜引文依据：`tests/multimodal/test_utils.py` 中合批相关测试仅 `def test_group_and_batch_mm_items_split_by_fieldset()`（187 行）与 `def test_group_and_batch_mm_items_split_by_shared_data()`（202 行）；modality 分组来自 `group_and_batch_mm_kwargs` 源码 docstring"we add another restriction that the items in a batch must belong to the same modality"。｜修复要求：把该句改为直接写出被覆盖的测试函数名（同第 156 行写法），或删除"后两种"这一指代。｜修复：｜复验：
- [轻微·表述] 第 166 行"（本页以 Qwen3.5 走读验证）"：以"本页"为主语的自我指代。｜引文依据：不适用｜修复要求：删除该括注，或改为"该结论由 Qwen3.5 源码走读验证"。｜修复：｜复验：
- [轻微·表述] 章节标题"1. 分界在哪里：两段各自做什么"（第 94 行）与"4. 为什么这样分"（第 171 行）为提问式/口语化，不符规范"标题短、主题式"。｜引文依据：不适用｜修复要求：改为主题式，如"1. 两段处理的职责划分""4. 分界依据"。｜修复：｜复验：
- [轻微·表述] 第 149 行"机制分两层：怎么合并与什么时候拆批"：口语化措辞。｜引文依据：不适用｜修复要求：改为"机制分两部分：字段合并方式与拆批条件"。｜修复：｜复验：
- [轻微·来源] 第 141 行"image_grid_thw 由 transformers 的 smart_resize 计算"：归因不精确——smart_resize 只返回缩放后的高宽，网格由 image processor 按 resized÷patch_size 导出；正文随后虽给出该推导，首句仍把网格本身归给 smart_resize。｜引文依据：`transformers/models/qwen2_vl/image_processing_qwen2_vl.py` 的 smart_resize docstring 仅声明缩放后满足"两维可被 factor 整除""总像素在 [min_pixels, max_pixels]""尽量保持宽高比"，不产出 grid；grid 由 `preprocessed_size.height // patch_size` 得到（`vllm/model_executor/models/qwen3_vl.py:941-942`）。｜修复要求：首句改为"image_grid_thw 由 HF image processor 基于 smart_resize 的输出计算"。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复
- 说明：本轮开始时工作树内该页曾存在 2026-09-13 17:47 的旧版本（网格数值 [1,62,62]/[1,144,144]、除以 28、"TestGroupAndBatchMmItems" 等），19:18 版本已修正上述内容；本记录针对 19:18 版本（哈希见页首）。