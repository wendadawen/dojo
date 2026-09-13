<!-- review-meta
round: 3
page: wiki/vllm-mm-image-two-stage/index.html
reviewed_content_sha256: a9fd7e29a7cb1d82
-->
# 多模态图片的两阶段处理审查记录（第 3 轮）

- 页面版本：15709a2dcb93815992ee773a78fe06cd8307be67
- 审查时间：2026-09-13 19:23
- 审查者：独立子代理（未参与写作与前序审查）
- 适用规范：guides/note.md（head dojo:type=note）；记录格式按 guides/concept/check.md 第 3、6 节
- 已完整阅读章节（含图注与全部正文，本页无折叠块）：1. 分界在哪里（含 SVG 图注）；2. 前端段；3. 后端段（收集 / 合批机制 / 编码 / 缓存）；4. 为什么这样分；来源与范围说明（源码定位 / 验证数据 / 边界）
- 核对来源：本地 vLLM 检出 /private/tmp/dspark-instr（vllm 0.22.2.dev，含 tests/）、transformers 4.57.6（~/Library/Python/3.9/site-packages，qwen2_vl 与 qwen3_vl）、profiler trace /Users/wendadawen/Desktop/qwen3.5-0.8b-torch-profile/{TENCENT64.site_*.async_llm.*.pt.trace.json.gz, rank0.*.pt.trace.json.gz}；`.dojo/scripts/validate.py` 通过。

## 问题

- [阻断·技术] 第 2 节「前端段」正文及 `<meta name="dojo:summary">`：smart_resize 的取整因子与网格数换算写错。页面写「把图片高宽各自取整到 factor=28 的倍数……再除以 28 得到网格数」，但 Qwen3.5 的视觉配置是 patch_size=16、spatial_merge_size=2，smart_resize 的 factor = patch_size×merge_size = 32，网格 = 缩放后边长 ÷ patch_size = ÷16。按页面所述 factor=28 复算，恰好得不出页面给出的网格数，算式与结论不符。｜引文依据：transformers/models/qwen2_vl/image_processing_qwen2_vl.py:54 `def smart_resize(height, width, factor: int = 28, ...)`；同文件 `_preprocess` 内 `factor=patch_size * merge_size`、`grid_h, grid_w = resized_height // patch_size, resized_width // patch_size`；transformers/models/qwen3_vl/configuration_qwen3_vl.py:37-38 `patch_size=16, spatial_merge_size=2`；vLLM qwen3_vl.py:332 `self.patch_size = vision_config.patch_size`。本机复算：980×986 取 factor=32 → 992×992 → grid [1,62,62]（与页面一致），取 factor=28 → 980×980 → grid [1,70,70]（与页面不符）；2310×2310 取 factor=32 且 max_pixels 足够大 → 2304×2304 → grid [1,144,144]（与页面一致），取 factor=28 → grid [1,70,70]。｜修复要求：将因子改为 32（=patch_size 16 × merge_size 2），把「再除以 28」改为「再除以 patch_size（=16）得到网格数」；dojo:summary 同步修改；并核实并写明该 processor 的 max_pixels 取值（2310×2310 不被下采样需要 max_pixels≥2310²，与 qwen2_vl 默认 28×28×1280 不同，须据模型 preprocessor_config 给出实际值）。｜修复：｜复验：

- [重要·技术] 第 2 节链路与「源码定位」：`call_hf_processor_mm_only` 及所标文件 `vllm/transformers_utils/processor.py` 均不成立。仓库中不存在名为 `call_hf_processor_mm_only` 的函数（transformers_utils/processor.py 只含 get_processor / get_image_processor / cached_* 等）。｜引文依据：`grep -rn "call_hf_processor_mm_only" vllm/` 无结果；实际函数为 vllm/multimodal/processing/processor.py:1207 `def _apply_hf_processor_mm_only(...)`，HF 处理器调用点在 vllm/multimodal/processing/context.py:242 `def call_hf_processor(...)`；profiler 中该跳为 `vllm/multimodal/processing/processor.py(1441): _cached_apply_hf_processor`。｜修复要求：把该步改为 `_apply_hf_processor_mm_only`（vllm/multimodal/processing/processor.py），HF 调用写 `call_hf_processor`（vllm/multimodal/processing/context.py）；「源码定位」一条同步改掉 transformers_utils/processor.py。｜修复：｜复验：

- [重要·表述] 第 4 节末句：把设计动机写成结论，且无来源、未标注为推断。原文「把重活留在离 GPU 最近的地方、轻活留在离请求最近的地方，是这条分界线的动机」是对 vLLM 作者设计意图的断言，全页未给出任何设计文档、PR 或注释来源。｜引文依据：不适用（页面无对应来源）。｜修复要求：或给出可定位的来源（设计文档 / 源码注释 / PR），或在句中标注「（推断）」并说明依据（CPU 预处理与 GPU 编码的事实分布），不得作为既定结论陈述。｜修复：｜复验：

- [轻微·技术] 合批段与「验证数据」：单元测试定位名 `TestGroupAndBatchMmItems` 不存在。该文件中的相关用例是模块级函数，不是类。｜引文依据：tests/multimodal/test_utils.py 无 `class TestGroupAndBatchMmItems`；第 187 行 `def test_group_and_batch_mm_items_split_by_fieldset():`、第 202 行 `def test_group_and_batch_mm_items_split_by_shared_data():`（断言分别为 [2,1,1,1]）。｜修复要求：把「TestGroupAndBatchMmItems」改为这两个测试函数名（或写「tests/multimodal/test_utils.py 中的 group_and_batch_mm_items 拆批用例」）。｜修复：｜复验：

- [轻微·技术] 「源码定位」把 `fetch_image` 归到 `vllm/multimodal/utils.py`，但该链路实际调用的是媒体连接器的方法。｜引文依据：vllm/entrypoints/chat_utils.py:937 `await self._connector.fetch_image_async(image_url)`；profiler APIServer trace 命中 `vllm/multimodal/media/connector.py(425): fetch_image_async`；vllm/multimodal/utils.py:284 的 `fetch_image` 只是连接器包装，不在本链路。｜修复要求：把该步的定位改为 vllm/multimodal/media/connector.py（fetch_image_async），或注明调用的是连接器的 fetch_image/fetch_image_async。｜修复：｜复验：

- [轻微·格式] 两张内联 SVG：代码标识符直接写在 `<text>` 中，未按规范用 `foreignObject` 承载；且第二张图同时承载「合并方式」与「拆批条件」两组关系。｜引文依据：guides/note.md「图示」：「图内公式与代码标识符用 foreignObject 承载」「每张图只表达一个关系」；SVG 中 `<text>` 直接含 `fetch image`、`image processor`、`execute mm encoder`、`embed multimodal`、`BatchedField：同形状 stack / 异形状 list`、`SharedField：取 batch[0]`。｜修复要求：将这些代码标识符移入 `<foreignObject>` 承载；把第二张 SVG 拆成「合并方式」与「拆批条件」两张，或删去冗余的一半只保留一个关系。｜修复：｜复验：

- [轻微·技术] 「源码定位」中 smart_resize 的引文定位「transformers/models/qwen2_vl/image_processing_qwen2_vl.py（62）」无法落到该函数。｜引文依据：该文件 `def smart_resize` 在第 54 行，第 62 行为 docstring 内的空行。｜修复要求：把括号内编号改为函数定义行（54），并据 Qwen3.5 实际的 patch/merge 取值补充「该文件给出的是函数定义；本模型 factor=32」。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 4
- 处置：修复
- 说明：核实通过的部分——两段分界（APIServer 预处理 / EngineCore 编码）、`scheduled_encoder_inputs` 结构、`_batch_mm_inputs_from_scheduler` 的三平行列表（哈希 / 数据 / (req_id, PlaceholderRange)）、`_get_group_hash` 仅对 SharedField 参与哈希、BatchedField 同形状 `torch.stack` 异形状返回列表、SharedField 取 batch[0]、拆批三条件、`embed_multimodal` 对 image 调 `_process_image_input`、`encoder_cache` 按 mm_hash 写入与 `_gather_mm_embeddings` 按 PlaceholderRange 取片、两进程 trace 中函数分布与 APIServer 内 load_file/rgba_to_rgb 与 processor 调用分属不同线程——均已回源确认。上列问题已独立复现，须修复后复验，阻断与重要问题关闭后方可发布。
