<!-- review-meta
round: 4
page: wiki/hy4-preview-dataflow/index.html
reviewed_content_sha256: 8ea54e15f1d56756
-->
# Hy4-Preview 前向数据流审查记录（第 4 轮）

- 页面版本：b156f7afd10ce763c656cba3dd73692650eb29c6234cfbda3cad89b1d26fb712
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次，未读取本页 research/ 下任何文件）
- 已完整阅读章节：1. 关键规格；2. 交互式数据流（含 8 个视图 overview/layer/indexer/mla/ihc/moe/mtp/cache 的全部节点、边与悬停提示文字）；3. 要点（整体结构、iHC、DSA 索引器、MLA 注意力、MoE、prefill 与 decode 的等价性、KV cache 两种口径）；4. 实现对照（RoPE 布局、checkpoint 命名）；5. MTP 草稿层；6. 核对方式；来源与范围说明。

核对方式（复验依据）：config 键值与 README 原句取自 huggingface.co/tencent/Hy4-preview（config.json、README.md 原文）；张量形状/dtype/数量取自全部 131 个分片的 safetensors 文件头（HTTP Range 读取，逐张量累加）；小张量实值按 index.json 的 data_offsets 精确 Range 下载后按 F32/BF16 位模式解码；算子语义取自 transformers `src/transformers/models/hy_v4/modeling_hy_v4.py`、`models/deepseek_v2/modeling_deepseek_v2.py`、vLLM `vllm/models/hy_v4/nvidia/{attention,model,mtp}.py`、SGLang `python/sglang/srt/configs/hy_v4.py` 与 `python/sglang/srt/models/hunyuan_v4.py`、以及 hub 内 `finetune/llama_factory_support/hy_v4_patches.py`。

本轮独立复核通过、未报的问题（供下游确认核对面已覆盖）：总参数量 779,960,992,733 与 131 分片逐张量累加完全一致；2006 张量 = 1380 BF16 + 626 F32，626 F32 恰为 iHC 门控 78×2×3 + 主干 sink 78 + 主干路由偏置 77 + hc_head 3；单 token 激活 47.57B 的各分项（注意力 265,685,568、full 索引器 9,371,904、iHC 393,236、hc_head 98,309、MoE 341,311,744、首层 dense 339,738,624）逐项与张量头相符、合计及 49.06B 均复算通过；路由专家 744.10B / 95.40% 复算通过；KV cache 32768/576 = 56.89、87.8 KiB/GiB、4.88 MiB/TiB、90.5 GiB 复算通过；索引器与 MLA 的公式（含 weights_proj×n_heads^-0.5×softmax_scale、qk/√256=0.0625、sink 只进分母、逐元素 sigmoid 门）与源码逐行相符；SwiGLU 截断、路由 sigmoid+偏置+2.827、共享专家无门控与源码一致；全部 checkpoint 小张量实值（hc_base/hc_scale/hc_head/sink/路由偏置/k_norm、MTP sink 与偏置）与页面标注的数值在有效位内一致；RoPE 布局三实现对照与 vLLM/SGLang/transformers 源码一致；`research/measured.md` 存在、站内 7 个前置链接均指向真实页面、alt 中无 `$...$`、`.dojo/scripts/validate.py` 通过。

## 问题

- [阻断·技术] 3 要点 › iHC ›「半区归属的独立印证」一条｜问题：该条写「checkpoint 中 hc_base 前 4 个围绕 −log3、后 4 个围绕 0」，把只在第 77 层成立的后半区观察写成了整份 checkpoint 的无条件性质；同一页紧邻的上一条（「真实权重的门控参数已显著训练」）给出的第 0 层 attn hc_base 后半区为 [−1.518, −1.418, −1.398, −0.486]，明显不在 0 附近。同一页对同一张量的同一半区给出互相矛盾的数值，读者会据此以为 checkpoint 的后半区（post）权重仍停在初值 0 附近，而真实情况只有第 77 层如此（第 0 层已从 0 移出至约 −1.4）｜引文依据：实测 `model.layers.0.hc_attn_layer.hc_pre.hc_base` = [−0.8859, −0.8793, −0.7178, −0.9103, −1.5183, −1.4180, −1.3979, −0.4856]（后半区 ≈ −1.4）；`model.layers.77.hc_attn_layer.hc_pre.hc_base` = [−0.7955, −1.2859, −1.3266, −1.4708, 0.0124, 0.0030, 0.0122, 0.2079]（后半区 ≈ 0）。源码 `modeling_hy_v4.py` `_init_weights`：`base_value = -math.log(max(hc_mult-1,1))`、`base[:hc_mult] = base_value`、其余为 0——「后 4 个初值为 0」是初始化事实，不是 checkpoint 现状｜修复要求：改写该条，改为「前半区停在初始值 −log3 附近、后半区的初值为 0；第 77 层后半区几乎未动（[0.012,0.003,0.012,0.208]），第 0 层后半区已从 0 移出至约 −1.4」，并说明该证据只支持「前半 pre、后半 post」的切分顺序，不得声称 checkpoint 的后半区围绕 0｜修复：｜复验：

- [轻微·技术] 3 要点 › KV cache 的两种口径（正文末段）与视图 8（cache）`ix` 节点｜问题：索引器缓存的 BF16 上界写「5.3 GiB」与本页自身数据不符：5376 B/token × 2^20 token ÷ 2^30 = 5.249 GiB，应为 5.2（或 5.25）；同区间的 FP8 下界 2772 B/token 得 2.707 GiB 写作「2.7」是对的，两端舍入规则不一致｜引文依据：正文「vLLM 用 FP8 存储并按 128 值一组附带 scale，约 2772 B/token；transformers eager 为 BF16，5376 B/token」，以及「1M 上下文下索引器缓存约 2.7–5.3 GiB」；实算 2772×1048576÷1073741824 = 2.7070，5376×1048576÷1073741824 = 5.2494｜修复要求：两处「2.7–5.3 GiB」改为「2.7–5.2 GiB」（或写明 5.25）｜修复：｜复验：

- [轻微·表述] 来源与范围说明（正文末段核对说明）｜问题：以「本页」为主语的自我指代——「本页覆盖语言主干与 MTP 草稿层的完整前向数据流；训练方案、推理调度（如 MTP 的投机采样策略）、量化与部署性能不在范围内」｜引文依据：不适用｜修复要求：改为无主语的覆盖范围陈述，如「覆盖范围：语言主干与 MTP 草稿层的完整前向数据流；训练方案、推理调度（如 MTP 的投机采样策略）、量化与部署性能不在范围内」｜修复：｜复验：

## 结论

- 处置：修复（阻断 1 条须关闭后重跑 validate.py 并复验）
- 统计：阻断 1 / 重要 0 / 轻微 2
