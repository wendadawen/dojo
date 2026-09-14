<!-- review-meta
round: 10
page: wiki/qwen3-5-dataflow/index.html
reviewed_content_sha256: 134a8e2d67278bc2
-->
# Qwen3.5-397B-A17B 前向数据流 审查记录（第 10 轮）

- 页面版本：b7ddae69da4a0b334d6571cae8545d4df27e2883
- 审查时间：2026-09-14
- 审查者：独立审查者（未参与写作与前序轮次；本轮只读页面本身与外部来源）
- 已完整阅读章节：1 关键规格（含家族变体表与 note）/ 2 交互式数据流（含 noscript 七张全量表格与七个视图的每个 tooltip）/ 3 要点（五组）/ 4 视觉编码器与多模态融合（4.1–4.8）/ 5 与 Qwen3.8-Flash-Next 的架构对比 / 6 核对方式 / 来源与范围说明

## 一、本轮实际取得的外部材料

- HF 官方仓库 Qwen/Qwen3.5-397B-A17B：config.json 全文、README.md、LICENSE、model.safetensors-00001..00094-of-00094.safetensors 全部 94 个分片的 JSON 文件头（HTTP Range 读取，未下载权重本体）。仓库 sha = 8472618112abcbd45acbcdc58436aff4233c23f7，即页面来源表所写 commit 84726181；ObjectId 解出的 createdAt = 2026-02-16T04:55:12Z，与页面「Qwen3.5 侧 2026-02-16T04:55:12Z」逐秒一致。
- 家族其余型号 config.json：Qwen3.5-122B-A10B / 35B-A3B / 27B / 9B / 4B。
- 姊妹型号 Qwen3.8-Flash-Next：config.json、LICENSE、仓库元信息（createdAt 2026-08-24T08:24:59Z）。
- 官方实现 huggingface/transformers commit 36deb0b5：src/transformers/models/qwen3_5/{modeling,modular,configuration}_qwen3_5.py、qwen3_5_moe/{modeling,modular}_qwen3_5_moe.py、qwen3_next/modeling_qwen3_next.py、cache_utils.py、vision_utils.py。
- vLLM main 分支 vllm/model_executor/models/qwen3_next_mtp.py。

## 二、回源核对结论（全部通过）

### 1. 逐张量累加（本机按 94 个分片头重跑）

- 张量数 2924，dtype 分布 BF16 2834 + F32 90，参数合计 403,397,928,944 —— 与第 1 节「94 个分片共 2924 个张量头」「2834 个张量 BF16，90 个 F32」「403,397,928,944（403.40B）」三项精确相等。
- 分组：language_model 1038 个张量 396,346,350,336；visual 333 个 456,010,480；mtp 1553 个 6,595,568,128 —— 与页面语言主干 396.346B、视觉 0.456B、MTP 6.596B 逐项相等（396.80B 命名口径亦由此成立）。
- 90 个 F32 实测恰为 45 个 linear_attn.A_log + 45 个 linear_attn.norm.weight，dt_bias 为 BF16 —— 与第 1 节、第 3 节「90 个 F32 张量中的一半」逐项一致。
- 结构计数：gate.weight 实测 61 个（60 主干 + 1 MTP）；indexer 张量 0 个；deepstack 张量 0 个；MTP 侧 mlp.experts.<N>.{gate,up,down}_proj.weight 共 1536 个，主干侧同名逐专家张量 0 个（即打包）——与第 1 节、视图 5、第 3 节 MTP 段落一致。
- 形状抽查（均为页面所列）：q_proj [16384,4096]、k_proj/v_proj [512,4096]、o_proj [4096,8192]、q_norm [256]、in_proj_qkv [12288,4096]、in_proj_z [8192,4096]、in_proj_a [64,4096]、conv1d [12288,1,4]、out_proj [4096,8192]、experts.gate_up_proj [512,2048,4096]、experts.down_proj [512,4096,1024]、shared_expert.gate_proj [1024,4096]、shared_expert_gate [1,4096]、mtp.fc [4096,8192]、mtp.pre_fc_norm_embedding/_hidden [4096]、mtp.norm [4096]、visual.patch_embed.proj.weight [1152,3,2,16,16]、merger.norm.weight [1152]、merger.linear_fc1 [4608,4608]、merger.linear_fc2 [4096,4608]、pos_embed [2304,1152]。
- 层型：linear_attn 45 层、self_attn 15 层，全注意力层号集合 {3,7,11,15,19,23,27,31,35,39,43,47,51,55,59} —— 与第 1 节、第 3 节完全一致。

### 2. 源码语义核对（qwen3_5_moe，commit 36deb0b5）

- Qwen3_5MoeAttention：q_proj 宽度 = num_attention_heads*head_dim*2；`attn_output = attn_output * torch.sigmoid(gate)`；q_norm/k_norm 作用在 head_dim；scaling = head_dim**-0.5；全文件不引用 config.attn_output_gate。→ 页面「attn_output_gate 只是记录键、实现无条件生效」「sigmoid 输出门在 o_proj 之前」「QK-norm 在 RoPE 之前」成立。与稠密变体 Qwen3_5Attention（qwen3_5/modeling_qwen3_5.py 第 632 行起，同样 q_proj 双宽 + torch.sigmoid(gate)）同源，页面「两个变体的注意力实现完全相同」成立。
- Qwen3_5MoeGatedDeltaNet：conv1d(bias=False, groups=conv_dim, kernel_size=linear_conv_kernel_dim)；`beta = b.sigmoid()`；`g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`；q/k 走 repeat_interleave(num_v_heads//num_k_heads) 与 l2norm；update_conv_state 只保留最后 conv_kernel_size 个位置（cache_utils.py：「keep only the last conv_kernel_size tokens」）。→ 页面 g/β 公式、conv 状态 [12288,4] 与 96.00 KiB、repeat_interleave 4 份、L2 归一化、kernel=4 状态全成立。
- Qwen3_5MoeTopKRouter：softmax(float32) → topk(top_k) → `router_top_value /= router_top_value.sum(dim=-1, keepdim=True)` 无条件执行，类内不存在 norm_topk_prob（对照 qwen3_next 的同名类确有 `if self.norm_topk_prob:` 分支）。→ 页面「没有 norm_topk_prob 开关，重归一化总是执行」成立。
- load_balancing_loss_func：tokens_per_expert（按其定义 Σ = top_k）、router_prob_per_expert、`return overall_loss * num_experts`。→ 页面 Σf_i = top-k 的源码口径成立；取 E = num_experts = 512 时均匀下界 = 0.001×512×(10/512) = 0.010000，与页面所写均匀分布下界自洽。
- Qwen3_5MoeRMSNormGated：`self.activation = "silu"` 硬编码，norm 在门之前。→ 页面「输出门激活硬编码 silu」成立。
- Qwen3_5MoeDecoderLayer：两次 `residual + hidden_states`，单流 pre-norm，无任何超连接/mixer 模块。→ 页面「单流 pre-norm」成立。
- `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`：transformers 不实现 MTP 前向、显式忽略 mtp.* 权重。→ 来源表写法成立。
- 视觉侧：Conv3d(kernel_size=stride=(2,16,16), bias=True)；PatchMerger 的 norm 维度取 config.hidden_size（use_postshuffle_norm=False）；VisionAttention 的 qkv bias=True、is_causal=False、head_dim = 1152/16 = 72；VisionBlock 两个 LayerNorm。→ 页面 4.1–4.3 与 vision 视图全成立。modular_qwen3_5_moe.py 的继承关系（视觉组件继承 ../qwen3_5/；Qwen3_5MoeForConditionalGeneration 继承 Qwen3VLMoeForConditionalGeneration、两个 Output 类与 Qwen3_5MoeTopKRouter 继承 Qwen3VLMoeTextTopKRouter）与来源表「视觉侧继承自 qwen3_5/、自 qwen3_vl_moe/ 继承的是多模态装配类、输出类与文本路由器」逐项吻合。
- 位置分配：get_rope_index 中图像/视频段执行 `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`；视频先按 video_grid_thw[:,0] repeat_interleave 并把 T 维置 1 再逐帧处理；get_vision_position_ids 用 arange(h//merge)+s / arange(w//merge)+s，T 维额外加 start_position。→ 页面「位置只推进 max(h,w)/2」「8 文本+(1,28,28) 图后从位置 22 开始而非 204」（8+14=22、8+196=204 自洽）「(1,84,84) 推进 42」「视频按帧拆分、每帧 T 维为帧起始位置」「3 帧 96 token 推进 24」全部成立。
- vLLM qwen3_next_mtp.py：`torch.cat([inputs_embeds, hidden_states], dim=-1)`（嵌入在前）、pre_fc_norm_embedding / pre_fc_norm_hidden、load_weights 只保留 mtp.* 与共享的 embed_tokens / lm_head、注释含 "mirroring the Qwen3.5 MTP handling"。→ 页面 MTP 拼接顺序、共享两头、checkpoint 无 mtp.embed_tokens/mtp.lm_head 全成立。

### 3. 家族与姊妹型号

- 122B-A10B：48 层 / 3072 / 32Q-2KV / GDN 16k-64v / 256 专家 top-8 / I=1024；35B-A3B：40 / 2048 / 16Q-2KV / 32v / 256 top-8 / I=512；27B：64 / 5120 / 24Q-4KV / 48v / 稠密 17408；9B：32 / 4096 / 16Q-4KV / 32v / 稠密 12288；4B：32 / 2560 / 16Q-4KV / 32v / 稠密 9216 且 tie_word_embeddings=True（页面「词表两头共享」）——家族表六行逐项一致。
- 模型卡原文：「Number of Parameters: 397B in total and 17B activated」「MTP: trained with multi-steps」「Context Length: 262,144 natively and extensible up to 1,010,000 tokens.」、license apache-2.0 —— 与第 1 节与来源表逐条一致。
- Qwen3.8-Flash-Next：48 层 / 2560 / 24Q-2KV / GDN 16k-48v / 512 专家 top-10 I=640，LICENSE 首行「Qwen Community License 1.0」，createdAt 2026-08-24T08:24:59Z —— 与第 5 节对比表逐项一致。

### 4. 机械项

- `.dojo/scripts/validate.py` 通过；`.dojo/scripts/check_inline_js.py` 通过（3 个 script 块语法 OK）。
- KaTeX 实测：正文/summary/表格 139 处 $...$ 与七个视图 76 处 tooltip 公式（含 d 字段内联式）以 throwOnError=true 全部渲染成功，0 失败。
- noscript 七张表与 JS VIEWS 逐节点比对（维度/公式/说明）：仅 2 处差异，均为 noscript 写 `$\theta$=1e7` 而 tooltip 写字面 `θ=1e7` 的表现差别，无语义分歧。
- 被引 16 个站内页（deepseek-moe / speculative-decoding / gated-deltanet / linear-attention / vit / positional-encoding / mrope / qwen3-8-flash-next-dataflow / delta-rule / rmsnorm / kv-cache / swiglu / depthwise-conv / residual-connection / hyper-connections / mqa-gqa）均存在于仓库；页内无 <img>（仅 lightbox 空 alt 占位，无 alt 内 $...$）；无 <pre> 代码块，无「声称可运行」的代码需执行。

## 三、问题

- [轻微·技术] 位置：第 3 节「MoE 路由」第 2 条、视图 4（moe）aux 节点、noscript「MoE 内部」表 aux 行（同一公式 3 处）｜问题：负载均衡损失公式 $\mathcal{L}=0.001\cdot E\sum_i f_i p_i$ 中的 $E$ 全页未定义，公式无法仅凭页面复算｜引文依据：qwen3_5_moe/modeling_qwen3_5_moe.py 的 `load_balancing_loss_func` 末行 `return overall_loss * num_experts`，即 E 就是专家数 512；页面给出的「均匀分布下界 0.010000」= 0.001×512×(10/512)，也只有 E=512 才成立｜修复要求：在公式首次出现处补 E 的含义（E 为专家数 512，源于 overal_loss * num_experts），或直接把符号写成 512｜修复：｜复验：
- [轻微·技术] 位置：来源与范围说明表「第 5 节与 Qwen3.8-Flash-Next 的对比（配置数值、许可、发布日期）」一行｜问题：该行称对比结果「与姊妹页 Qwen3.8-Flash-Next 前向数据流逐项一致」，但姊妹页通篇只有配置数值，不含任何许可与发布日期表述，读者按此去姊妹页核对这两项会落空｜引文依据：wiki/qwen3-8-flash-next-dataflow/index.html 全文检索 License/许可/发布/2026-08-24/Apache 均无命中（该页仅第 125 行有自身的「生成于 2026-08-27」）；其配置数值 48=36+12、2560、24Q-2KV、512 专家 top-10 I=640、GDN 16k-48v、ViT 输出 2560、N-gram 51.2B 仅第 1 层、fc_embedding/fc_hidden 与本页对比表确实逐项一致｜修复要求：把「逐项一致」的范围收窄为「配置数值与姊妹页逐项一致」，许可与发布日期只保留「该型号官方模型卡 / HF 仓库元信息」这一来源｜修复：｜复验：
- [轻微·表述] 位置：页面 head 的 description 元数据「……512 专家 top-10 softmax 路由加辅助损失、DeepSeek 式 MTP 草稿层、27 层 ViT 与交错 MRoPE」｜问题：「DeepSeek 式」把 MTP 的设计出处归给 DeepSeek，而本页来源表把 MTP 的来源列为官方模型卡与 vLLM/transformers，页内没有任何材料支持这一归因（本仓库他页对同类归因的处理是降级标注，见下）｜引文依据：README.md 第 69 行「- MTP: trained with multi-steps  」；来源表「MTP 前向语义（拼接顺序、共享两头）｜vLLM vllm/model_executor/models/qwen3_next_mtp.py」；wiki/glm-5-3-flash-dataflow/index.html 同类写法「模块用途按 DeepSeek-V3 系 MTP 的公开做法解读，此处标注为推断」｜修复要求：删去「DeepSeek 式」，或按本仓库惯例写成带出处的推断｜修复：｜复验：

## 四、结论

本轮未发现阻断或重要问题：页面每一个结构数字（张量总数、dtype 分布、三层参数分组、层型分布、各投影与缓存形状、家族六型号与姊妹型号配置）都在本轮取得的外部材料中逐项对上，源码语义、位置推进规则与 MTP 拼接顺序全部与官方实现一致；公式可复算且 KaTeX 全部渲染，noscript 回退可读，站内链接有效。三条轻微问题集中在「一个未定义符号」与「两处归因/来源范围的措辞」，均不影响任何结论与数值。

统计：阻断 0 / 重要 0 / 轻微 3

> 本轮所列问题的处理结果见 `minor-fixes.md`。
