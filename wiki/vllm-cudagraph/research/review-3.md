<!-- review-meta
round: 3
page: wiki/vllm-cudagraph/index.html
reviewed_content_sha256: 8544d447ef8d4497
-->
# torch.compile 图捕获与 CUDA Graph 审查记录（第 3 轮）

- 页面版本：8da65085caff2230e17121864c25f5659eb98459
- 审查时间：2026-09-13 19:52
- 审查者：编排者派发的独立审查者
- 已完整阅读章节：1. torch.compile 的两阶段流程；2. Dynamo 图捕获的处理边界；3. CUDA Graph 的录制与重放；4. 捕获尺寸的确定；来源与范围说明
- 核对来源：vLLM commit 5ac2684（vllm/config/vllm.py、vllm/config/compilation.py、vllm/v1/cudagraph_dispatcher.py、vllm/model_executor/models/config.py）；本地 torch 2.8.0 实跑（`.dojo/scripts/validate.py` 返回 success）
- 已按来源核对通过的部分：两阶段流程（TorchDynamo 捕获 / Inductor 默认后端 + Triton）为 PyTorch 官方通行描述；第 2 节 C 扩展 graph break 行为与警告文本 `Dynamo does not know how to trace the builtin \`_hashlib.openssl_sha1.\`` 经 torch 2.8.0 实跑逐字吻合，`fullgraph=True` 确实抛 `Unsupported`；第 4 节 `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))` 与 `compilation.py` docstring 一致，`decode_query_len = 1 + num_speculative_tokens`、与 `max_num_batched_tokens` 取小、padding 向上补齐、超限/未命中回落 NONE 均与源码一致。

## 问题

- [阻断·技术] 第 4 节「该上限默认取 `min(max_num_seqs × decode_query_len × 2, default_max_graph_size)`…`default_max_graph_size` 默认 512，数据中心 Blackwell 平台为 1024」：一句内含两处来源不支持的内容。(1) `default_max_graph_size` 不是源码中存在的符号——`vllm/config/vllm.py::VllmConfig._set_cudagraph_sizes`（5ac2684）写作 `max_cudagraph_capture_size = min(self.scheduler_config.max_num_seqs * decode_query_len * 2, 512)`，用的是字面量 512，配置字段名为 `max_cudagraph_capture_size`；全仓检索无 `default_max_graph_size`。(2)「数据中心 Blackwell 平台为 1024」与来源不符——该 commit 中唯一把 `max_cudagraph_capture_size` 抬到 1024 的位置是 `vllm/model_executor/models/config.py::GptOssForCausalLMConfig.verify_and_update_config`（`compilation_config.max_cudagraph_capture_size = 1024`，注释「Increase the max capture size from 512 to 1024 for performance.」），这是 gpt-oss 模型专属覆盖，而非平台级默认；该文件也未列入本页来源。｜引文依据：源码原文 `max_cudagraph_capture_size = min(self.scheduler_config.max_num_seqs * decode_query_len * 2, 512)`；`compilation_config.max_cudagraph_capture_size = 1024`（GptOssForCausalLMConfig）｜修复要求：删去 `default_max_graph_size` 命名，按源码写为 `min(max_num_seqs × decode_query_len × 2, 512)`（该上限即配置字段 `max_cudagraph_capture_size`）；「数据中心 Blackwell 平台为 1024」删除，或改为据实表述并将 `vllm/model_executor/models/config.py`（gpt-oss 把上限提高到 1024）补入来源。｜修复：｜复验：

- [重要·技术] 第 4 节末句「据多模态统一输入与 cudagraph 的走读数据，decode 命中 FULL 重放，而 983 token 的 prefill 落到 eager」：被引页面不含 983。`wiki/vllm-mm-unified-embeds-cudagraph/index.html` 全文无 "983"，只支持「大 prefill 单批 token 数超出捕获尺寸上限 NONE，eager 执行」这一定性结论；983 这个具体取值出现在 `wiki/vllm-v1-two-process-arch/index.html`（「同一个 983 token 的 prefill 回合在模型执行阶段退回 eager」）。把另一页的数字挂到被引页名下，沿链接无法核对。｜引文依据：mm 页原文「大 prefill 单批 token 数超出捕获尺寸上限 NONE，eager 执行」；two-process-arch 页原文「同一个 983 token 的 prefill 回合在模型执行阶段退回 eager」｜修复要求：若保留 983，改引 `../vllm-v1-two-process-arch/index.html` 并核对该数值来源；否则删去该具体数字，仅保留被引页支持的定性结论。｜修复：｜复验：

- [轻微·表述] 第 3 节「显存代价：除图本身（每张数十至数百 MB）外…」：该量级以陈述语气写进正文，其「推断」性质仅见于文末「来源与范围说明」（「CUDA Graph 显存占用的具体量级属于推断，未逐一实测」），正文读者不一定读到。｜引文依据：不适用｜修复要求：在该句内联标注为推断，或改写为不给出未实测的具体区间。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 1
- 处置：修复
- 说明：第 4 节的「1024」是唯一阻断项，需删除或据实改写并补来源；983 的来源错挂需订正。其余技术论断（两阶段、C 扩展 graph break 与警告文本、捕获尺寸公式、padding/回落逻辑）经源码与实跑核对均成立。