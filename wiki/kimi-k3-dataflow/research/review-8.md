<!-- review-meta
round: 8
page: wiki/kimi-k3-dataflow/index.html
reviewed_content_sha256: 0d9a8abf35b1b1cb
-->
# Kimi K3 前向数据流审查记录（第 8 轮）

- 页面版本：0660adc4f6662230c40890bfe33f1e380a09611c
- 审查时间：2026-09-14 14:39
- 审查者：编排者派发的独立审查者（未参与写作，未读本页 research/）
- 已完整阅读章节：head 元信息与页内脚本 → 1. 关键规格 → 2. 交互式数据流（含 `<noscript>` 回退总表与 8 个视图的 JS 节点/边/提示文本）→ 3. 要点 → 4. 视觉编码器与多模态融合（4.1–4.5）→ 5. 长上下文开销的来源 → 6. 核对方式 → 来源与范围说明

## 外部来源核对（本轮据以判断的材料）

- arXiv:2607.24653v2（Kimi K3 技术报告）
- huggingface.co/moonshotai/Kimi-K3 的 config.json、preprocessor_config.json、modeling_kimi_linear.py、modeling_kimi_k3.py
- 本机 Python 复算、KaTeX 直渲、无头 Chrome 实渲

核对结果（逐条已定位到来源）：

1. 结构数值全部对上 config.json：num_hidden_layers=93、hidden_size=7168、num_attention_heads=96、first_k_dense_replace=1、num_nextn_predict_layers=0、moe_intermediate_size=3072、num_experts=896、num_experts_per_token=16、num_shared_experts=2、routed_expert_hidden_size=3584、moe_router_activation_func="sigmoid"、routed_scaling_factor=1.0、moe_renormalize=true、latent_moe_use_norm=true、vocab_size=163840、tie_word_embeddings=false、q_lora_rank=1536、kv_lora_rank=512、qk_nope_head_dim=128、qk_rope_head_dim=64、v_head_dim=128、mla_use_output_gate=true、mla_use_nope=true、attn_res_block_size=12、linear_attn_config.{num_heads:96, head_dim:128, short_conv_kernel_size:4, gate_lower_bound:-5.0, use_full_rank_gate:true}、intermediate_size(稠密)=33792、rms_norm_eps=1e-05、media_placeholder_token_id=163605、activation_situ_beta=4.0、activation_situ_linear_beta=25.0、hidden_act="situ"。
2. 层分布：full_attn_layers(1-based)=[4,8,…,88,92,93] → 0-based MLA={3,7,…,87,91,92}，与页面「MLA 在 3,7,…,87」「末尾 91、92」完全一致；kda_layers(1-based)=[1,2,3,5,6,7,…] → 0-based KDA={0,1,2,4,…}，共 69，页面层分布与「blk 内 68 + 第 0 层 1」自洽。
3. 视觉 config：patch_embed_proj_bias=false、init_pos_emb_height=64、init_pos_emb_width=64、init_pos_emb_time=4、pos_emb_interpolation_mode="bilinear"、mm_hidden_size=1024、qkv_hidden_size=1536、vt_num_attention_heads=12、vt_hidden_size=1024、vt_intermediate_size=4096、vt_num_hidden_layers=27、merge_type="sd2_tpool"、merge_kernel_size=[2,2]、projector_ln_eps=1e-05、patch_size=14、mm_projector_type="patchmergerv2" —— 页面 §1「视觉」行、§4.1–4.4、视觉视图逐项吻合。
4. 处理器 config：in_patch_limit=65536、patch_limit_on_one_side=512 —— 页面「in_patch_limit=65536、单边 512 patch」吻合；65536=256×256 与报告「最高 3584×3584 像素」自洽。
5. 源码语义：`patch_embed` Conv2d kernel=stride=14 且 bias 由 patch_embed_proj_bias 控制（页面写 false）；位置表非持久 buffer、`assert t <= self.num_frames`（页面「t≤4，源码 assert」）；`Rope2DPosEmbRepeated(qkv_hidden_size//num_heads, 512, 512)`（页面「θ 基 10000、网格上限 512×512」），代码把 x(=flat%max_width，即宽) 放偶数槽、docstring 写反（页面「源码 docstring 把 h/w 标注写反」）；`tpool_patch_merger` 对时间维 `mean(dim=0)`、2×2 的 4 份特征不相加直接送投影（页面「时间全池化 + 2×2 合并、不相加」）；`PatchMergerMLPV2` 两层无 bias Linear + post_norm RMSNorm、V2 无 pre_norm（页面吻合）。
6. 注意力/门控语义：MLA `self.scaling = self.q_head_dim ** (-0.5)`（页面「192^{-1/2}」）、`assert self.use_nope`、`rotary_emb=None`（页面吻合）；参考实现缓存 `key_states=(B,96,·,192)`、`value_states=(B,96,·,128)`，即 96×320×2B=60 KiB/token/层（页面 §5「60 KiB/token/层」成立），潜压缩口径 576×2B=1.125 KiB/token/层成立；KDA 侧 `A_log` 形状 [num_heads]（页面「A_log 每头一个尺度」）、`dt_bias` 形状 [12288]（页面「每头每维」）、`safe_gate=gate_lower_bound is not None`（页面「K3 用 safe_gate」）、`FusedRMSNormGated(head_dim, activation='sigmoid')` 且 `o=self.o_norm(o,g)`（页面吻合）；MoE `scores=logits.sigmoid()`→`scores+e_score_correction_bias` 取 top-k→renormalize→`×routed_scaling_factor`（页面吻合）；共享专家 `intermediate_size = moe_intermediate_size × num_shared_experts = 6144`（页面吻合）；AttnRes `layer_idx % attn_res_block_size == 0` 存快照、`v=cat((block_residual, prefix_sum.unsqueeze(1)))`（顺序为块在前、当前残差在后，页面 `v=[v_1,…,v_8,p]` 一致）、`k=v·rsqrt(var+eps)` 与 `score_weight=norm.weight*proj.weight.squeeze(0)`、`scores=(k*score_weight).sum(-1)`（页面公式逐项吻合）。
7. 融合：`_token_occupation_table` 全 1 后把占位符位改写为 `feature_lengths`、`new_token_positions=cumsum(...)-1`、`max_embed_dim=占用表行和`（=T−#PH+Σn_i）、文本位与视觉位交错、`final_labels` 全填 ignore_index；纯文本（pixel_values 为空）不进融合、decode 单步（`input_ids.shape[1]==1` 且带 cache）只按 cache 修补 position_ids/attention_mask（页面 §4.5 三步与三个边界行为吻合）。
8. 数字复算：§5 开销表逐格可复算（4K 参考 4096×60KiB×24=5.625 GiB；32K=45；128K=180；256K=360；1M=1440；潜压缩 1M=27；KDA 每层 6 MiB(fp32)+288 KiB(bf16)、69 层=0.423 GiB；93 层全 MLA 1M=5.45 TiB、24/93=25.8%；448×448 图 256 token×60KiB×24=360 MiB）；视觉塔 401,214,464=27×14,682,112+602,112+4,194,304+1,024，merger 46,144,512=4096×4096+4096×7168+7168，447.4M 与 0.016% 成立。
9. 渲染：KaTeX 直渲 summary 的 `$2\times2$` 及 SiTU-KDA-AttnRes 等复杂公式全部通过；无头 Chrome（file://）实渲 index 与 #kda 视图正常，节点/边/图例可读，KaTeX 正常落字；`.dojo/scripts/validate.py` 返回 validation ok。
10. 表述：全文（含 noscript 回退表、8 个视图提示文本、图注）无元话语、无「本页」自我指代、无「我们/你」、无调试复现叙事、无临场评价；未见 alt 属性含 `$…$`（页面仅一个 lightbox img，alt 为空）；未见「（待生成）」占位；引用的 8 个站内前置概念页（moonvit-v2/nope/rope/kda/mla/kv-cache/qwen3-8-flash-next-dataflow/block-attnres）均真实存在。正文算术符号（α、β、σ、⊙、Σ 等）全部走 KaTeX，仅 `×`/`→`/`·` 作分隔符直写（与同站其他 dataflow 页一致）。

## 问题

- [轻微·技术] §4.3「3. 要点」前的视觉/语言对照表「量化」行（语言主干单元格，约 file 行 316）：该处把「MoE 路由器保持高精度」与「latent MoE 投影保持高精度」并列，并统一判为「与 config 的量化目标不一致」；而 §1 关键规格「其他」行（约 file 行 105）已明确「MoE 路由器是裸参数、不属 Linear 量化目标，与报告不冲突」，即路由器并不与 config 冲突。两处对同一事实的结论不一致，读者会误以为路由器也被 config 列入量化目标｜引文依据：技术报告 §4.1.4 原文把高精度项列为「attention projections, latent MoE projections, shared experts, and MoE routers」，而 config.json 的 quantization_config ignore 正则只覆盖 self_attn / shared_experts / mlp gate,up,gate_up,down proj / lm_head / vision_tower / mm_projector，不含路由器；故分歧只在 latent MoE 投影一项（与 §1 一致），路由器不构成分歧｜修复要求：把 §4.3 该单元格括号内一句改为与 §1 同口径，例如「报告 §4.1.4 另称 latent MoE 投影与 MoE 路由器保持高精度；其中 latent MoE 投影与 config 的量化目标不一致（路由器为裸参数、不属 Linear 量化目标，不构成分歧）」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 报为 0 / 轻微 1
- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（0 阻断、0 重要；遗留 1 轻微为同页两处口径不一致的表述细节，不推翻任何结论，接受理由为此；建议按上条同步 §4.3 与 §1 的措辞）
