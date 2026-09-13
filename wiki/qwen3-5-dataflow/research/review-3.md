<!-- review-meta
round: 3
page: wiki/qwen3-5-dataflow/index.html
reviewed_content_sha256: fb9195033cdb9c40
-->
# Qwen3.5-397B-A17B 前向数据流审查记录（第 3 轮）

- 页面版本：1783948704c375a35e04a2c12e67a8a79e275149
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：1. 关键规格（表 + 家族变体 + note）、2. 交互式数据流、3. 要点（整体结构 / GDN / 全注意力层 / MoE 路由 / MTP 草稿层 / 长上下文开销）、4. 视觉编码器与多模态融合（4.1–4.8）、5. 与 Qwen3.8-Flash-Next 的架构对比、6. 核对方式、来源与范围说明；并逐节点读完 7 个交互视图（overview/gdn/fa/moe/mtp/vision/fusion）的全部 label、io、公式 f、说明 d、edge 标签与图例。

## 来源核对（本轮已核，均通过，用于支撑「无阻断」判断）

- 官方 config.json（huggingface.co/Qwen/Qwen3.5-397B-A17B，commit 84726181）：hidden_size=4096、num_hidden_layers=60、num_attention_heads=32、num_key_value_heads=2、head_dim=256、num_experts=512、num_experts_per_tok=10、moe_intermediate_size=1024、shared_expert_intermediate_size=1024、router_aux_loss_coef=0.001、full_attention_interval=4、attn_output_gate=true、linear_num_key_heads=16、linear_num_value_heads=64、linear_conv_kernel_dim=4、max_position_embeddings=262144、rope_theta=10000000、partial_rotary_factor=0.25、mrope_section=[11,11,10]、vocab_size=248320、mtp_num_hidden_layers=1、vision_config（27 层 / 宽 1152 / 16 头 / 输出 4096）——第 1 节各行逐项与页面一致。
- 官方模型卡：原文 "Number of Parameters: 397B in total and 17B activated"、"Context Length: 262,144 natively and extensible up to 1,010,000 tokens."、"License: apache-2.0"、"Gated Attention: Number of Attention Heads: 32 for Q and 2 for KV"、"Gated DeltaNet: Number of Linear Attention Heads: 64 for V and 16 for QK"、"MTP: trained with multi-steps"、citation month=February year=2026——与页面「官方称 397B / 17B」「原生 262,144 token、可扩展至 1,010,000」「Apache-2.0」「2026-02-16」「Gated Attention / Gated DeltaNet 头数」逐条一致。
- transformers commit 36deb0b5 `models/qwen3_5/modeling_qwen3_5.py`：`query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)`；`attn_output = attn_output * torch.sigmoid(gate)`；文件内检索 `attn_output_gate` 零命中。`Qwen3_5RMSNormGated` 用 `self.activation="silu"`。=> 页面「输出门无条件、sigmooid、在 o_proj 之前」「attn_output_gate 只是记录键」「GDN 输出门硬编码 silu（与 Qwen3.8 可配 sigmoid 不同）」成立。
- 同 commit `models/qwen3_5_moe/modeling_qwen3_5_moe.py`：类名 `Qwen3_5MoeDecoderLayer`；两处 `hidden_states = residual + hidden_states`；路由 `softmax -> topk -> router_top_value /= router_top_value.sum(...)`（无 norm_topk_prob 分支）。=> 页面「单流 pre-norm 残差」「MoE 先全局 softmax 再对 top-10 无条件重归一化」成立。
- 同 commit `models/qwen3_next/modeling_qwen3_next.py`（组件继承来源）：`l2norm(query/key, dim=-1, eps=1e-6)`、`repeat_interleave(num_v_heads // num_k_heads, dim=2)`（64/16=4）、`scale = 1 / (query.shape[-1] ** 0.5); query = query * scale`、`Qwen3NextRMSNormGated(self.head_v_dim)`、`nn.Conv1d(groups=conv_dim, kernel_size=conv_kernel_size, padding=kernel_size-1)`。=> 页面 GDN 各条（L2 归一、repeat 4 份、o_t=S_tᵀq_t/√d_k、norm.weight [128]、groups=12288/kernel 4）成立。
- vLLM `model_executor/models/qwen3_next_mtp.py`：`torch.cat([inputs_embeds, hidden_states], dim=-1)`，norm 名 `pre_fc_norm_embedding` / `pre_fc_norm_hidden`，注释 "mirroring the Qwen3.5 MTP handling"。=> 页面「拼接顺序 embedding 在前，由 vLLM 实证」成立。
- 家族变体表逐型号拉官方 config 核对：122B-A10B（48 / 3072 / 32Q2KV / 64v / 256 专家 top-8 / I=1024）、35B-A3B（40 / 2048 / 16Q2KV / 32v / 256 top-8 / I=512）、27B（64 / 5120 / 24Q4KV / 48v / dense 17408）、9B（32 / 4096 / 16Q4KV / 32v / dense 12288）、4B（32 / 2560 / 16Q4KV / 32v / dense 9216 / tie_word_embeddings=true，对应「词表两头共享」）——全部一致。
- 对比表与姊妹页 `wiki/qwen3-8-flash-next-dataflow/index.html` 交叉核对：48=36 GDN+12 QSA、hidden 2560、24Q×256/2KV、GDN 16k/48v、I=640、180.0B/6.04B、51.2B 哈希 N-gram、4 流超连接 + 末端 mixer、`full_attention`→`qwen_sparse_attention` 改写、GDN `output_gate_type="sigmoid"`、MTP `fc_embedding`/`fc_hidden`、rope_theta=1e7、mrope_section=[11,11,10]——全部一致；Qwen3.8-Flash-Next 模型卡 "125B with 6B activated, plus 51B n-gram embedding and 4B MTP" / license qwen-community-1.0 / August 2026 与第 5 节一致。
- 页面内算式全部复算通过：403,397,928,944 = 396,346,350,336 + 456,010,480 + 6,595,568,128；激活差 17,349,040,896 − 16,331,922,176 = 248320×4096；MoE 单层激活 125,829,120+12,582,912+2,097,152+4,096 = 140,513,280（占 6,442,450,944 的 2.18%）；60 层路由专家 6,442,450,944×60 = 386,547,056,640（95.82%）；MTP 6,595,568,128 = 6,442,450,944 + 104,857,600 + 33,554,432 + … 逐项吻合，1553 = 512×3 + 17；视觉 456,010,480 = 267,890,544 + 143,451,648 + 40,119,040 + 2,654,208 + 1,770,624 + 124,416，各分组百分比逐项吻合；KV 2048 B/token/layer、GDN 状态 4.00 MiB + 72.00 KiB、45 层 0.179 GiB 与 32K/256K/1M 各列均复算通过；位置轴 8+14=22、(3,8,16)→96 token/推进 24、196=784/4、1764=3136/4 等全部自洽。
- `.dojo/scripts/validate.py wiki/qwen3-5-dataflow/index.html` → validation ok；15 个站内前置概念链接目录全部存在；head 的 description / dojo:summary / dojo:type=dataflow / dojo:topics（模型结构,多模态，均在 ALLOWED_TOPICS 内）/ dojo:tag=数据流（在 ALLOWED_TAGS 内）齐全；无「（待生成）」占位。

## 问题

- [重要·技术] 3. 要点 › MoE 路由（第 2 条）与视图 4「MoE 内部」aux 节点 tooltip：所引源码的辅助损失在完全均匀路由下的取值算错 10 倍｜引文依据：官方 load_balancing_loss_func 为 `tokens_per_expert_sum = tokens_per_expert_sum + torch.bincount(selected_experts.reshape(-1), minlength=num_experts).float()`（对全部 T×top_k 个选中位置计数）与 `tokens_per_expert = tokens_per_expert_sum / total_rows`、`overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)); return overall_loss * num_experts`——即页面定义的 f_i 满足 Σ_i f_i = top_k = 10（不是 1）。完全均匀路由时 f_i = 10/512、p_i = 1/512，Σ_i f_i p_i = 512·(10/512)·(1/512) = 10/512 = 0.019531，E·Σ_i f_i p_i = 512 × 10/512 = 10，乘 router_aux_loss_coef=0.001 得 0.010000。页面写「完全均匀分布下界 0.001000」，该值恰等于系数本身（仅在 Σ_i f_i = 1 的归一化口径下成立，与所引源码不符）；且同段实测值 0.014310 在源码口径下相当于均匀值的 1.4 倍，在 0.001 口径下则相当于 14 倍，两组数字不可能出自同一口径｜修复要求：用产生 0.014310 的同一脚本重算「完全均匀路由」的 𝓛；若按源码定义（Σ_i f_i = top_k），把 0.001000 改为 0.010000，并在 tooltip 公式处写明 Σ_i f_i = top_k；若脚本另做了 f 归一化，则须与 load_balancing_loss_func 的定义对齐，并同步复算 6 token 的实测值｜修复：｜复验：
- [轻微·技术] 3. 要点 › 长上下文开销表：合计 ≠ 分项之和（差值 0.001 GiB）｜引文依据：GDN 列写 0.179 GiB（精确 192,061,440 B / 2^30 = 0.178871 GiB），而合计列按 0.178 算：256K 行为 7.500+0.179=7.679、表列 7.678，1M 行为 30.000+0.179=30.179、表列 30.178；精确合计 7.678871→7.679、30.178871→30.179。32K 行 0.9375+0.178871=1.116371→1.116 与表列一致｜修复要求：统一四舍入口径——把 GDN 列写 0.178（截断）或把合计改为 1.117 / 7.679 / 30.179，二选一后三行自洽｜修复：｜复验：
- [轻微·表述] 来源与范围说明段：「本页覆盖语言主干、视觉编码器、多模态融合与 MTP 结构的完整前向数据流；……不在范围内。」以「本页」作主语的自我指代｜引文依据：不适用｜修复要求：改为不含「本页」主语的表述，例如「覆盖范围：语言主干、视觉编码器、多模态融合与 MTP 结构；训练方案、优化器与后训练流程、MTP 投机解码调度策略以及 FP8 量化仓库不在范围内。」｜修复：｜复验：
- [轻微·表述] 第 3 节 MTP 第 1 条「这一顺序由 vLLM 的 Qwen3NextMultiTokenPredictor.forward 实锤」，以及导语、第 1 节 note、第 5 节对比表中三处「真·全注意力」：口语化 / 评价性措辞（「实锤」「真·」）｜引文依据：不适用｜修复要求：改为中性表述，如「由 vLLM 源码确认（Qwen3NextMultiTokenPredictor.forward）」「标准全注意力（GQA 2 KV 头，纯因果掩码）」｜修复：｜复验：
- [轻微·一致] 第 5 节对比表 Qwen3.8-Flash-Next 行「51.2B 哈希 PLE，仅第 2 层注入」未标层号口径｜引文依据：本页层号惯例为 0 起（第 3 节「第 3、7、11…59 层（0 起）」）；姊妹页 wiki/qwen3-8-flash-next-dataflow/index.html 写「51.2B 参数的哈希 N-gram 查表只在第 1 层（0 起）注入」。此处「第 2 层」按 1 起读才与姊妹页一致，按本页 0 起惯例则会被读成索引 2，与姊妹页矛盾｜修复要求：补口径标注，如「仅第 2 层（1 起，即 0 起第 1 层）注入」｜修复：｜复验：
- [轻微·规范] 2. 交互式数据流：七个视图的节点、边、公式与 tooltip 文本全部定义在页内 `<script>` 的 VIEWS 对象中，脚本失效时 `#cy` 容器为空、视图不可读｜引文依据：guides/model-dataflow.md「视图」节「脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」。正文第 3–6 节仍可读，故影响有限｜修复要求：为每个视图提供无脚本可读的等价内容（视图节点 / 边 / 形状写入 HTML，脚本仅负责切换显隐），或在页面注明该交互为增强项、正文已含全部结论｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复
- 说明：本轮对 config.json、官方模型卡、transformers commit 36deb0b5 的三处源码、vLLM MTP 源码、家族五型号 config、Qwen3.8-Flash-Next 模型卡与姊妹页逐一回源核对，页面全部结构性论断与页面内算式（参数量拆分、MoE 激活、视觉分组、KV/GDN 开销、位置轴推进）均核对通过；未发现无来源支持的结论、构造示例写成来源事实、或把实验条件观察写成无条件论断。唯一「重要」项为辅助损失均匀下界的数字与所引源码的算子定义不符，需用原脚本复核后修正；其余 5 项为口径一致性与表述问题。修正后重新运行 validate.py 并进入复验即可发布。
