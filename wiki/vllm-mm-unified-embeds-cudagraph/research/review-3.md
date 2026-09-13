<!-- review-meta
round: 3
page: wiki/vllm-mm-unified-embeds-cudagraph/index.html
reviewed_content_sha256: 28acfc81a99d1a60
-->
# 多模态统一输入与 cudagraph 审查记录（第 3 轮）

- 页面版本：94d9b05f4c69a69ca99e0a4b3b314ae82cfb88ac
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序轮次）
- 类型：note（依据 `guides/note.md`）
- 已完整阅读章节（按顺序）：1. 问题：两种输入格式与 cudagraph 的固定性矛盾；2. 解法：统一成 inputs_embeds（一个格式，一张图）（含「混合 batch」「全纯文本 batch」「对比：纯文本模型走 input_ids」「v0 双编译到 v1 单图」四个子节）；3. prefill 与 decode 的模式差异；来源与范围说明。图注（内联 SVG）与折叠块内容一并通读。
- 来源获取：官方源码 `vllm`（上游 `github-community/vllm`，工作树 HEAD 5ac268497，`git describe` = v0.26.1rc0-454；同时在 tag v0.22.1 复核），本地路径 `/Users/wendadawen/code/github/github-community/vllm`。页面未引用外部 URL，无需 WebFetch。

## 问题

- 本轮未发现阻断、重要或轻微问题。

## 核对依据（逐条回源的原文片段）

- §1 forward 两分支：页面代码块与 `vllm/model_executor/models/qwen3_next.py:653-657` 逐字一致 —— `if inputs_embeds is not None: hidden_states = inputs_embeds / else: hidden_states = self.embed_input_ids(input_ids)`；`Qwen3_5Model` 在 `vllm/model_executor/models/qwen3_5.py:205-215` 由 `@support_torch_compile(...)` 装饰且 `class Qwen3_5Model(Qwen3NextModel)`，该文件未定义 `forward`，确为继承自 `Qwen3NextModel`。论断成立。
- §2 多模态分支三条件：`vllm/v1/worker/gpu_model_runner.py:3610` —— `if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:`，与页面「模型支持多模态、流水线第一段、非编解码架构」一致。
- §2.1 全部 token 先过 embedding、再掩码覆写：`gpu_model_runner.py:3647-3651` —— `inputs_embeds_scheduled = self.model.embed_input_ids(self.input_ids.gpu[:num_scheduled_tokens], multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed)`；覆写语句在同模型 `vllm/model_executor/models/utils.py:657` —— `inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)`（函数定义 `def _merge_multimodal_embeddings` 起于 `utils.py:636`）。与页面「inputs_embeds[is_multimodal] = 图像特征」「False 位置保持原样」一致。
- §2.1 图像特征按哈希缓存：`gpu_model_runner.py:3042` —— `self.encoder_cache[mm_hash] = output`。论断成立。
- §2.2 全纯文本 batch：`gpu_model_runner.py` `_execute_mm_encoder` 首部 —— `if not mm_kwargs: return []`；`qwen3_5.py:516-517` —— `if multimodal_embeddings is None or len(multimodal_embeddings) == 0: return inputs_embeds`。与页面「没有待编码项直接返回……特征列表为空时跳过覆写、原样返回」一致；同段「额外一次 embedding 计算（推断）」已显式标注推断。
- §2.3 纯文本模型走 input_ids 及注释原文：`gpu_model_runner.py:3688-3692` —— `# For text-only models, we use token ids as input. / While it is possible to use embeddings as input just like the / multimodal models, it is not desirable for performance since / then the embedding layer is not included in the CUDA graph.` 与页面表述一致。
- §2.3 v0 双编译：`gpu_model_runner.py:3670-3672` —— `# engine avoids this by "double compiling" the CUDA graph, once / # with input_ids and again with inputs_embeds, for all num_tokens.` 页面引文为该句的子串，语境（该 `elif self.enable_prompt_embeds and is_first_rank:` 分支）与页面标注「prompt embeds 与 token id 混用」一致。
- §3 dispatch 查表：`vllm/v1/cudagraph_dispatcher.py:274-281` —— `if (not self.keys_initialized or ... or num_tokens > max_size or allowed_modes <= {CUDAGraphMode.NONE}): return CUDAGraphMode.NONE, ...`；`:307-324` 先试 FULL、再试 PIECEWISE，均不命中则 `return CUDAGraphMode.NONE`。与页面「token 数超出最大捕获尺寸、或该形状没有捕获记录时返回 NONE」一致。
- §3 模式定义与默认：`vllm/config/compilation.py:607-632` —— `CUDAGraphMode` 含 `NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE`；docstring `FULL_AND_PIECEWISE. (v1 default)`、`FULL_AND_PIECEWISE mode: Capture full cudagraph for decode batches and piecewise cudagraph for prefill and mixed prefill-decode batches. ... is the default.`、`PIECEWISE mode ... keeping the cudagraph incompatible ops (i.e. some attention ops) outside the cudagraph`。v1 默认值在 `vllm/config/vllm.py:291/314` 落为 `"cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE`。与页面「decode 用 FULL、prefill 与混合批次用 PIECEWISE」「注意力等不适合入图的算子留在图外」一致。
- §3 捕获尺寸上限：`compilation.py:703-704` —— `If not specified, max_cudagraph_capture_size is set to min(max_num_seqs*2, 512) by default.` 与页面 `min(max_num_seqs*2, 512)` 逐字一致。
- 运行环境：`Qwen/Qwen3.5-0.8B` 见 `tests/models/registry.py:1335`（`Qwen3_5ForConditionalGeneration`）。版本「0.22.2.dev」介于 tag v0.22.1 与 v0.23.0 之间，为合理 dev 版本，与上述源码在 v0.22.1 tag 复核结果一致（`git show v0.22.1:vllm/v1/worker/gpu_model_runner.py` 第 3385/3419/3440 行同样含三条件、双编译注释、performance 注释；`compilation.py` 同样含 `min(max_num_seqs*2, 512)`）。
- 页面链接：`../vllm-cudagraph/index.html`、`../vllm-mm-image-two-stage/index.html` 均真实存在且同为 note 页；全文无 research/ 路径引用、无「待生成」占位。
- 机械项：`python3 .dojo/scripts/validate.py wiki/vllm-mm-unified-embeds-cudagraph/index.html` 返回 `validation ok`；`dojo:type=note`、`dojo:topics=推理系统`、`dojo:tag=推理系统` 均在 `catalog_builder.py` 词表内；全页无 Unicode 数学字符（本页无公式，唯一算式 `min(max_num_seqs*2, 512)` 为 ASCII）；无内嵌 `<style>` 外的死资源。
- 渲染：无头 Chrome（`--headless=new`，1200×2400）截图实测，内联 SVG 流程图、Python 代码块、导语与表格均正常渲染；TOC 在 <1400px 按共享 CSS 规则隐藏，属设计行为。
- 表述维度：逐段通读（含 SVG 图注）未发现元话语、会话指代（我/我们/你）、调试复现叙事或临场评价；`本页`（仅「来源与范围说明」范围声明处 1 次）与 `不划算`（性能注释转述）在本仓库既有 note/concept 页中为通行写法（分别见于 75 页与 5 页），不构成问题。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布
