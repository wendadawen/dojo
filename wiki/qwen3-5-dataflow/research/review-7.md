<!-- review-meta
round: 7
page: wiki/qwen3-5-dataflow/index.html
reviewed_content_sha256: 05219f93d340c767
-->
# Qwen3.5-397B-A17B 前向数据流审查记录（第 7 轮）

- 页面版本：f499335b9956b9db867a1786368d6577f6f4f3b5（工作树）
- 审查时间：2026-09-13 22:49
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查）
- 适用规范：guides/model-dataflow.md（`dojo:type=dataflow`）；表述与记录格式沿用 guides/concept/check.md
- 已完整阅读章节：1. 关键规格（含家族变体表）→ 2. 交互式数据流（含 noscript 七视图表与全部图注）→ 3. 要点 → 4. 视觉编码器与多模态融合 → 5. 与 Qwen3.8-Flash-Next 的架构对比 → 6. 核对方式 → 来源与范围说明；另逐条通读页内 7 个交互视图的节点 io/f/d 文本、head 元信息（description / dojo:summary）与页内脚本。

## 核对摘要（本轮，供复验引用）

- 官方 config.json（Qwen/Qwen3.5-397B-A17B，sha 84726181…）逐键核对：hidden_size 4096、60 层、full_attention_interval 4、layer_types＝45 linear_attention + 15 full_attention（层号 3,7,…,59）、32Q/2KV/head_dim 256、linear 16k/64v/head_dim 128、conv kernel 4、512 专家 top-10、moe_intermediate_size 1024、shared 1024、router_aux_loss_coef 0.001、max_position_embeddings 262144、mrope_section [11,11,10]、rope_theta 1e7、partial_rotary_factor 0.25、mtp_num_hidden_layers 1、mtp_use_dedicated_embeddings false、vision depth 27/hidden 1152/intermediate 4304/heads 16/num_position_embeddings 2304/out_hidden_size 4096/patch 16/merge 2/temporal 2/deepstack []、token id 248056/248057/248053/248054、vocab 248320、tie_word_embeddings false、mamba_ssm_dtype float32 —— 全部与页面一致。
- 参数量逐项复算：语言主干 396,346,350,336、视觉 456,010,480（27 层 MLP 267,890,544 + attention 143,451,648 + merger 40,119,040 + pos_embed 2,654,208 + patch_embed 1,770,624 + 27 层 LN 124,416）、MTP 6,595,568,128、总参数 403,397,928,944 —— 与页面逐位一致。激活两口径 16,331,922,176 = 15,314,803,456（60 层子层 + 末端 RMSNorm）+ lm_head 1,017,118,720；17,349,040,896 再加词嵌入表，与页面「不含／含查表」的定义一致。
- 张量计数：主干 3 + 45×18 + 15×15 = 1038、视觉 333、MTP 1553，合计 2924 —— 与页面「2924 个张量」「1553 个张量」「2834 BF16 + 90 F32」自洽。
- 公式回源（transformers commit 36deb0b5 与当前 main 双版本比对，结论相同）：Attention 双宽 q_proj + chunk 出 gate + `attn_output * torch.sigmoid(gate)`（无条件，不读 attn_output_gate）、q_norm/k_norm 作用于头维、scaling=head_dim**-0.5、GQA repeat_kv 16 组；GDN conv1d(bias=False, groups=conv_dim=12288)、`beta = b.sigmoid()`、`g = -A_log.exp()*softplus(a+dt_bias)`、q/k repeat_interleave 对齐 64 值头、RMSNormGated 硬编码 silu、delta rule 递归 `S = S*g_t; delta=(v-(S·k).sum(-2))*beta; S = S + k⊗delta`；Router 零初始化、softmax→topk→除法重归一化（无条件）、`load_balancing_loss_func` 的 `E·Σf_i p_i` 且 `Σf_i = top_k`；MTP `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`、vLLM `hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)`（embedding 在前）。
- 家族变体表 6 行逐字段回源 5 个官方 config（122B-A10B / 35B-A3B / 27B / 9B / 4B，含 4B 的 tie_word_embeddings=true）；397B-A17B createdAt 2026-02-16 早于其余成员，页首「系列首个开源旗舰」成立。
- Qwen3.8-Flash-Next 侧：48＝36 GDN+12 QSA、hidden 2560、24Q×256/2KV、GDN 16k/48v、512 专家 top-10 I=640、output_gate_type=sigmoid、27 层 ViT 输出 2560、mrope 全同、180.00B/6.04B、license qwen-community-1.0、createdAt 2026-08-24T08:24:59Z —— 与对比表及姊妹页逐项一致；`Qwen4ExpTextConfig` 把 full_attention 改写为 qwen_sparse_attention 的说法由姊妹页 index.html:152 印证。
- 页内 23 条代表性公式（含 `\mathrm{...\_...}` 与 summary 公式）以本地 libs/katex.min.js 实渲染（throwOnError:true），0 失败；`python3 .dojo/scripts/validate.py wiki/qwen3-5-dataflow/index.html` → validation ok；页面引用的 16 个概念页链接目标均存在；无 `alt` 含 `$...$`；noscript 已给出全部七视图的 HTML 表，属交互视图的无脚本可读版本。
- 数值抽查自洽：KV cache 2048 B/token/层（7.5/30/120 GiB）、GDN 4 MiB + 72 KiB → 0.179 GiB、MRoPE 槽位 11/11/10、最低频 1.66e-7、位置推进 max(h,w)/2（196→14、1764→42、3 帧视频 96→24）、辅助损失均匀值 0.010000 = 0.001×512×(10/512)。

## 问题

- [重要·技术] 第 4.4 节正文、noscript「视觉编码器：27 层 ViT」表、JS vision 视图 out 节点（共 3 处）：1080p 图的视觉 token 数写作 1,980，与官方图像处理管线不符，正确值为 2,040。｜引文依据：官方 preprocessor_config.json（patch_size 16、merge_size 2、size 边界 shortest_edge 65536 / longest_edge 16777216）+ transformers `smart_resize`（src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:78-79「h_bar = round(height / factor) * factor；w_bar = round(width / factor) * factor」，factor = patch_size × merge_size = 32）：1920×1080 → h_bar = round(1080/32)*32 = 1088（68 patch）、w_bar = 1920（120 patch）→ grid (1,68,120) → 68×120/4 = 2040。姊妹页 wiki/qwen3-8-flash-next-dataflow/index.html:343 在同一 processor_class（Qwen3VLProcessor / Qwen2VLImageProcessorFast）下记「1920×1080 图 |(1,68,120)|8,160|2,040」。1,980 对应把 1080 向下取整到 32 的倍数（1056，66 patch），与 round 口径不符。｜修复要求：把 3 处「1080p 图 1,980 个」改为「1080p 图 2,040 个」，或统一按官方 smart_resize 重算并把所得网格一并写出；同段的 448²=196、896²=784、1344²=1764 复算正确，不必改动。｜修复：｜复验：

- [轻微·技术] 第 1 节关键规格表「QK 归一化」行：源码位置写作 `Qwen3_5Attention.__init__`。本页主线模型是 Qwen3_5MoeForConditionalGeneration，其注意力类为 `Qwen3_5MoeAttention`；`Qwen3_5Attention` 只定义在稠密变体模块 qwen3_5/ 内，qwen3_5_moe/ 中不存在该类。｜引文依据：`src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:751 class Qwen3_5MoeAttention(nn.Module)`，其 `__init__` 含 `self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)` 与 `self.k_norm = …`；对照 `src/transformers/models/qwen3_5/modeling_qwen3_5.py:749 class Qwen3_5Attention(nn.Module)`。同页第 123 行已正确写作 `Qwen3_5MoeDecoderLayer.forward`，两处类名命名不一致。｜修复要求：把该类名改为 `Qwen3_5MoeAttention.__init__`；若确要说明与稠密变体同源，另注「稠密变体对应 Qwen3_5Attention」（第 148 行注已说明两者注意力实现相同）。｜修复：｜复验：

- [轻微·技术] 第 5 节对比表「总参 / 激活」行：Qwen3.5 侧取 16.33B、Qwen3.8 侧取 6.04B，但两侧激活口径不同且表内未标注——本页 16.33B 含 lm_head、不含词嵌入查表；姊妹页 6.04B 明确不含 embedding 与 lm_head。按同一口径（不含两头）Qwen3.5 侧应为 15.31B。｜引文依据：本页第 120 行「16,331,922,176（不含词嵌入查表）」；复算 16,331,922,176 = 15,314,803,456（60 层子层 + 末端 RMSNorm）+ 1,017,118,720（lm_head = 248320×4096），即含 lm_head；姊妹页 index.html:135「单 token 激活 | 6.04B（不含 embedding 与 lm_head）| 独立算得 6,035,598,720」。｜修复要求：在该行两个数字上分别标注口径（例：「16.33B（含 lm_head，不含 embedding）/ 6.04B（不含 embedding 与 lm_head）」），或统一换算为「不含 embedding 与 lm_head」口径后并列。｜修复：｜复验：

## 结论

- 处置：修复（1 重要 + 2 轻微，均已给出可直接执行的改法；无阻断）
- 统计：阻断 0 / 重要 1 / 轻微 2