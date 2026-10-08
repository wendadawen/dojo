# Hy4-preview

- 描述：Hy4-preview 一次 prefill 前向的数据流：文本 token 先查词嵌入，复制成 4 份残差流，进 78 层主干；每层用 iHC（identity Hyper-Connections）在注意力和前馈前后混合这 4 份残差流，注意力是门控的 DeepSeek 稀疏注意力，部分层挑出最多 2048 个位置参与计算，其余层复用上一层挑出的位置；第 0 层前馈是稠密 FFN，其余 77 层是 256 选 8 的 MoE；最后把 4 份残差流折叠、归一化并投影成全部位置的 logits。
- 摘要：78 层主干，隐层 6144，残差流复制成 4 份；注意力是 Gated DSA（DeepSeek Sparse Attention），KV 压到 512 维潜向量再加 64 维旋转键，查询压到 2048 维；21 个层跑索引器选出 Top-2048 的位置，其余层复用上一层的结果；第 0 层前馈是稠密 FFN，其余 77 层是 256 个路由专家取 top-8 加 1 个共享专家。
- 主题：模型结构,注意力机制
- 标签：数据流

## 资料

- 源码：Hugging Face `transformers` 仓库的 `src/transformers/models/hy_v4/`，commit `cb33194ad6152bd9fad6305378d92db385dd7b32`（main），用到的文件是 `modeling_hy_v4.py`（956 行）、`configuration_hy_v4.py`（162 行）、`modular_hy_v4.py`（642 行）
- config.json：`tencent/Hy4-preview` 仓库根目录的 `config.json`，副本在 `sources/config.json`，`model_type` 为 `hy_v4`
- 权重文件头：同一仓库 `model-00001-of-00131.safetensors` 到 `model-000131-of-00131.safetensors` 共 131 个分片，按 HTTP Range 只取头部解析，形状表在 `sources/shapes.json`；78 层形状一致时合并成 `layers.{i}` 一条
- 论文或技术报告：arXiv 2512.02556（DeepSeek-V3.2，DSA 与索引器）、arXiv 2603.12201（IndexCache，跨层复用稀疏索引），PDF 在 `sources/paper/`

## 前提

- 这一次前向是 `HYV4ForCausalLM.forward` 的一次调用：推理，不传 `labels`，不算 loss
- 这一次是 prefill：`past_key_values` 不传，`use_cache` 为默认值 True，一次喂进 T 个 token，`position_ids` 不传、由输入长度推出；B 是批大小
- 单卡，不切分张量并行，`input_ids` 在 GPU 上，`inputs_embeds` 不传
- 走 `_attn_implementation` 为 `eager` 这条分支：在参与 softmax 之前，用索引器返回的 Top-2048 下标散出一个加性掩码，把没选中的位置置为极小值
- 层的分支按 `config.json`：`mlp_layer_types` 第 0 层是 `dense`，其余 77 层是 `sparse`；`indexer_types` 第 0、1 层，以及第 5 层起每 4 层（第 5、9…77 层）为 `full`，其余为 `shared`
- `index_topk` 为 2048，本次前向 T 不小于 2048，索引器取满 2048 个位置
- 权重按 `config.json` 的 `dtype`（bfloat16）加载；`e_score_correction_bias`、iHC 的 `fn`、`base`、`scale`、`hc_fn`、`hc_base`、`hc_scale`、索引器的 `weights_proj` 与 `k_norm`、注意力 `sinks` 按源码保持 fp32；`lm_head` 也保持 fp32
- `HYV4ForCausalLM` 的 `_keys_to_ignore_on_load_unexpected` 忽略 `model.mtp_layers.*`，多 token 预测（MTP，Multi-Token Prediction）层不在本次前向的路径上

## config.json

| 组 | 名字 | 值 |
|---|---|---|
| 规模 | vocab_size | 120832 |
| 规模 | hidden_size | 6144 |
| 规模 | num_hidden_layers | 78 |
| 规模 | intermediate_size | 18432 |
| 规模 | max_position_embeddings | 1048576 |
| 注意力 | num_attention_heads | 64 |
| 注意力 | num_key_value_heads | 8（config.json 里的值；源码 `__post_init__` 会在加载时改写成 num_attention_heads 64） |
| 注意力 | q_lora_rank | 2048 |
| 注意力 | kv_lora_rank | 512 |
| 注意力 | qk_nope_head_dim | 192 |
| 注意力 | qk_rope_head_dim | 64 |
| 注意力 | qk_head_dim | 256 |
| 注意力 | v_head_dim | 256 |
| 注意力 | head_dim | 64 |
| 注意力 | attention_bias | false |
| 注意力 | attention_dropout | 0.0 |
| 注意力 | gated_mla | true（注意力带门控） |
| 注意力 | use_mla | true（MLA，Multi-head Latent Attention：KV 压到 512 维潜向量再加 64 维旋转键） |
| 注意力 | use_dsa | true（DSA 指稀疏注意力：索引器挑出最多 2048 个位置） |
| 注意力 | gating_type | elementwise（注意力的输出逐元素乘门控） |
| 注意力 | learnable_sink | true（可学习注意力汇，见注意力图的 sinks） |
| 注意力 | learnable_sink_init | 0.0 |
| 位置编码 | rope_theta | 10000000 |
| 位置编码 | rope_type | default |
| 索引器 | index_n_heads | 32 |
| 索引器 | index_head_dim | 128 |
| 索引器 | index_topk | 2048 |
| 索引器 | indexer_types | 逐层，第 0、1、5、9…77 层为 full，共 21 层，其余为 shared |
| iHC | hc_mult | 4 |
| iHC | hc_magnitude | 2.0 |
| iHC | hc_eps | 1e-06 |
| iHC | enable_ihc | true |
| MoE | n_routed_experts | 256 |
| MoE | n_shared_experts | 1 |
| MoE | num_experts_per_tok | 8 |
| MoE | moe_intermediate_size | 2048 |
| MoE | routed_scaling_factor | 2.827 |
| MoE | norm_topk_prob | true |
| MoE | n_group | 1 |
| MoE | topk_group | 1 |
| MoE | swiglu_limit | 10.0 |
| MoE | mlp_layer_types | 第 0 层 dense，其余 77 层 sparse |
| 量化 | dtype | bfloat16 |
| 量化 | enable_lm_head_fp32 | true |
| 规范化 | rms_norm_eps | 1e-05 |
| 规范化 | hidden_act | silu |
| 其它 | initializer_range | 0.006 |
| 其它 | tie_word_embeddings | false |
| 其它 | num_nextn_predict_layers | 1 |
| 其它 | pad_token_id | 120002 |

## HYV4ForCausalLM.forward（modeling_hy_v4.py:896-952）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ids | 张量 | input_ids | int64 [B, T] | 每个位置都是真实 token | modeling_hy_v4.py:926 |
| model | 算子 | self.model(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache) | int64 [B, T] → bf16 [B, T, 6144] | | modeling_hy_v4.py:925-933 |
| hs | 张量 | hidden_states | bf16 [B, T, 6144] | outputs.last_hidden_state | modeling_hy_v4.py:935 |
| lhead | 算子 | self.lm_head(hidden_states[:, slice_indices, :].to(dtype=self.lm_head.weight.dtype)) | bf16 [B, T, 6144] → fp32 [B, T, 120832] | | modeling_hy_v4.py:939 |
| logits | 张量 | logits | fp32 [B, T, 120832] | logits_to_keep 为默认值 0，slice(-0, None) 就是 slice(0, None)，保留全部位置 | modeling_hy_v4.py:939 |
| lret | 张量 | MoeCausalLMOutputWithPast(loss, logits, past_key_values, ...) | fp32 [B, T, 120832]（logits） | 返回值：本次只用到 logits，loss 为 None | modeling_hy_v4.py:945-952 |
| pkv | 缓存 | past_key_values | 每层压缩后的 KV 潜向量与索引器 key | 各层 self_attn.update 与 update_indexer 写，后续前向读 | modeling_hy_v4.py:945-950 |

### 连线

- ids → model
- model → hs
- model → pkv：写缓存
- hs → lhead
- lhead → logits
- logits → lret
- pkv → lret
- model 点开 → HYV4Model.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| model | 公式 | $hidden\_states, past\_key\_values = model(input\_ids)$，model 是主干，input_ids 是词 id |
| model | 说明 | 主干返回 `last_hidden_state` 与 `past_key_values`；`hidden_states`、`attn`、`router_logits` 等明细字段这次不取 |
| lhead | 公式 | $logits = lm\_head(hidden\_states)$，hidden_states 是主干输出，logits_to_keep 为 0 时保留全部位置 |
| lhead | 说明 | lm_head.weight [120832, 6144]，fp32 权重，权重与 embed_tokens 不共享（tie_word_embeddings 为 false） |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 介绍 | 模型 | Hy4-preview 是腾讯混元团队的 MoE 旗舰模型，770B 总参数、49B 激活参数。主干 78 层，隐层 6144。残差通路（残差流）用 iHC（identity Hyper-Connections）把隐状态复制成 4 份并逐层混合。注意力是 Gated DeepSeek Sparse Attention，KV 只存 512 维潜向量再加 64 维旋转键，查询压到 2048 维；部分层跑一个轻量索引器，为每个查询挑出最多 2048 个位置。第 0 层前馈是稠密 FFN，其余 77 层是 256 个路由专家取 top-8 加 1 个共享专家。这次前向不传缓存、不传 labels，从文本 token 算出全部位置的 logits。 |
| 公式 | 输出 | $logits = lm\_head(hidden\_states)$，hidden_states 是主干输出的隐状态 |
| 配置 | num_hidden_layers | 78 |
| 配置 | hidden_size | 6144 |
| 配置 | vocab_size | 120832 |
| 配置 | hc_mult | 4 |

## HYV4Model.forward（modeling_hy_v4.py:801-861）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ids | 张量 | input_ids | int64 [B, T] | | modeling_hy_v4.py:815 |
| emb | 算子 | self.embed_tokens(input_ids) | int64 [B, T] → bf16 [B, T, 6144] | | modeling_hy_v4.py:815 |
| h0 | 张量 | inputs_embeds | bf16 [B, T, 6144] | | modeling_hy_v4.py:815 |
| pos | 算子 | torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens，再 .unsqueeze(0) | int64 [T] → int64 [1, T] | | modeling_hy_v4.py:822-823 |
| pid | 张量 | position_ids | int64 [1, T] | | modeling_hy_v4.py:822-823 |
| rope | 算子 | self.rotary_emb(hidden_states, position_ids=position_ids) | bf16 [B, T, 6144] 与 int64 [1, T] → bf16 [1, T, 64] × 2 | | modeling_hy_v4.py:838 |
| pe | 张量 | position_embeddings | bf16 [1, T, 64] × 2 | 一对 cos 与 sin，最后一维 64 = qk_rope_head_dim | modeling_hy_v4.py:838 |
| msk | 算子 | create_causal_mask(**mask_kwargs) | bf16 [B, T, 6144] → bf16 [B, 1, T, L] | | modeling_hy_v4.py:835 |
| am | 张量 | causal_mask_mapping["indexed_attention"] | bf16 [B, 1, T, L] | 加性因果掩码，下三角为 0、其余为极小值；L 是键的总长度，本次等于 T；allow_is_causal_skip 为 False，索引器也要用它 | modeling_hy_v4.py:835, 846 |
| hid | 算子 | inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous() | bf16 [B, T, 6144] → bf16 [B, T, 4, 6144] | | modeling_hy_v4.py:840 |
| h | 张量 | hidden_states | bf16 [B, T, 4, 6144] | 4 份残差流，4 = hc_mult | modeling_hy_v4.py:840 |
| ly | 算子 | 78 × HYV4DecoderLayer | bf16 [B, T, 4, 6144] → bf16 [B, T, 4, 6144] | | modeling_hy_v4.py:844 |
| hl | 张量 | hidden_states | bf16 [B, T, 4, 6144] | | modeling_hy_v4.py:844 |
| tk | 张量 | topk_indices | int32 [B, T, 2048] | full 层产出的 Top-2048 位置，随后的 shared 层复用；相对 ly 这个循环，它是出口 | modeling_hy_v4.py:844 |
| ptk | 张量 | prev_topk_indices | int32 [B, T, 2048] | 传给下一次 ly 调用的 Top-2048 位置，进第 0 层时为 None；相对 ly 这个循环，它是入口 | modeling_hy_v4.py:842, 851 |
| hch | 算子 | self.hc_head(hidden_states) | bf16 [B, T, 4, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:856 |
| hc | 张量 | hidden_states | bf16 [B, T, 6144] | | modeling_hy_v4.py:856 |
| nrm | 算子 | self.norm(hidden_states) | bf16 [B, T, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:856 |
| lh | 张量 | last_hidden_state | bf16 [B, T, 6144] | | modeling_hy_v4.py:858-859 |
| pkv | 缓存 | past_key_values | 每层压缩后的 KV 潜向量与索引器 key | 各层写，后续前向读 | modeling_hy_v4.py:817-818, 849 |

### 连线

- ids → emb
- emb → h0
- h0 → pos：形状与 device
- pkv → pos：缓存长度
- pos → pid
- pid → rope
- h0 → rope：device 与 dtype
- rope → pe
- h0 → msk：形状与 device
- pid → msk
- pkv → msk：缓存长度
- msk → am
- h0 → hid
- hid → h
- h → ly
- pe → ly
- am → ly
- pkv → ly
- ly → pkv
- ly → hl
- ly → tk
- ly → ptk
- ptk → ly：prev_topk_indices
- hl → hch
- hch → hc
- hc → nrm
- nrm → lh
- hch 点开 → HYV4HyperHead.forward
- nrm 点开 → HYV4RMSNorm.forward
- ly 点开 → HYV4DecoderLayer.forward
- rope 点开 → HYV4RotaryEmbedding.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| emb | 公式 | $inputs\_embeds = embed\_tokens.weight[input\_ids]$，embed_tokens.weight 是词表矩阵 |
| emb | 说明 | embed_tokens.weight [120832, 6144]，`padding_idx` 为 120002 |
| pos | 公式 | $position\_ids = \mathrm{arange}(T) + past\_seen\_tokens$，past_seen_tokens 是缓存已有的长度，这次为 0 |
| rope | 公式 | $cos, sin = rotary\_emb(hidden\_states, position\_ids)$，内部算式见 HYV4RotaryEmbedding.forward |
| rope | 说明 | 逆频率用 rope_theta 10000000 生成，只加在 qk_rope_head_dim 这 64 维上；rope_type 为 default，attention_scaling 为 1 |
| msk | 公式 | $mask$ 下三角为 0，其余为当前 dtype 的最小有限值 |
| hid | 公式 | $hidden\_states = \mathrm{expand}(inputs\_embeds, hc\_mult)$，把最后一维前面插一个新维再复制 hc_mult 份 |
| hid | 说明 | 复制出的 4 份彼此相等，是 iHC 的 4 条残差流 |
| ly | 公式 | $hidden\_states, topk\_indices = layer(hidden\_states, mask, pe, past\_key\_values, prev\_topk\_indices)$，layer 是这一层的 HYV4DecoderLayer，prev_topk_indices 就是 ptk |
| ly | 说明 | ly 是 78 层的循环：ptk 是进入循环时带的下标，tk 是循环里每层产出的下标，两者在层之间接着传。78 层结构相同，差异只有两处：第 0 层前馈是稠密 FFN，其余 77 层是 MoE；`indexer_types` 为 full 的 21 层（第 0、1、5、9…77 层）跑自己的索引器，其余层复用上一层给出的 topk_indices。第 0 层的 prev_topk_indices 是 None |
| hch | 公式 | $hc = hc\_head(hidden\_states)$，内部算式见 HYV4HyperHead.forward |
| hch | 说明 | hc_fn [4, 24576]、hc_base [4]、hc_scale [1]，都在 fp32 里算 |
| nrm | 公式 | $last\_hidden\_state = norm(hidden\_states)$，内部算式见 HYV4RMSNorm.forward，hidden_states 是同图 `hc` 上的量 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 输入嵌入 | $inputs\_embeds = embed\_tokens.weight[input\_ids]$ |
| 公式 | 残差流复制 | $hidden\_states = \mathrm{expand}(inputs\_embeds, hc\_mult)$，hc_mult 是 4 |
| 公式 | 末尾折叠 | $out = \sum_c pre_c\, x_c$，c 遍历 4 份残差流，pre_c 是每份的折叠系数（见 HYV4HyperHead.forward），x_c 是第 c 份残差流 |
| 配置 | indexer_types | 21 层 full，其余 shared |
| 配置 | max_position_embeddings | 1048576 |

## HYV4DecoderLayer.forward（modeling_hy_v4.py:680-717）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| x | 张量 | hidden_states | bf16 [B, T, 4, 6144] | | modeling_hy_v4.py:691 |
| ahc | 算子 | self.attn_hc(hidden_states) | bf16 [B, T, 4, 6144] → fp32 [B, T, 4] 与 bf16 [B, T, 6144] | | modeling_hy_v4.py:695 |
| pa | 张量 | post | fp32 [B, T, 4] | 注意力支路加回残差流时的混合系数，4 = hc_mult | modeling_hy_v4.py:695 |
| hsa | 张量 | hidden_states | bf16 [B, T, 6144] | 4 份残差流按 pre 系数折叠成的单份 | modeling_hy_v4.py:695 |
| aln | 算子 | self.input_layernorm(hidden_states) | bf16 [B, T, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:697 |
| hsan | 张量 | hidden_states | bf16 [B, T, 6144] | | modeling_hy_v4.py:697 |
| attn | 算子 | self.self_attn(hidden_states, attention_mask, position_ids, past_key_values, use_cache, position_embeddings, prev_topk_indices) | bf16 [B, T, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:698-707 |
| ao | 张量 | hidden_states | bf16 [B, T, 6144] | 注意力输出 | modeling_hy_v4.py:698 |
| tk | 张量 | topk_indices | int32 [B, T, 2048] | 本层算出的 Top-2048 位置，随返回值交给调用方 | modeling_hy_v4.py:698, 717 |
| am | 张量 | attention_mask | bf16 [B, 1, T, L] | 加性因果掩码 | modeling_hy_v4.py:700 |
| pe | 张量 | position_embeddings | bf16 [1, T, 64] × 2 | cos 与 sin | modeling_hy_v4.py:704 |
| ptk | 张量 | prev_topk_indices | int32 [B, T, 2048] | 上一层传来的 Top-2048 位置；full 层不用它 | modeling_hy_v4.py:688, 705 |
| adda | 算子 | post.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2) + residual.float() | fp32 [B, T, 4] 与 bf16 [B, T, 6144] 与 bf16 [B, T, 4, 6144] → bf16 [B, T, 4, 6144] | | modeling_hy_v4.py:708 |
| ha | 张量 | hidden_states | bf16 [B, T, 4, 6144] | | modeling_hy_v4.py:708 |
| fhc | 算子 | self.ffn_hc(hidden_states) | bf16 [B, T, 4, 6144] → fp32 [B, T, 4] 与 bf16 [B, T, 6144] | | modeling_hy_v4.py:711 |
| pf | 张量 | post | fp32 [B, T, 4] | 前馈支路加回残差流时的混合系数 | modeling_hy_v4.py:711 |
| hsf | 张量 | hidden_states | bf16 [B, T, 6144] | | modeling_hy_v4.py:711 |
| fln | 算子 | self.post_attention_layernorm(hidden_states) | bf16 [B, T, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:713 |
| hsfn | 张量 | hidden_states | bf16 [B, T, 6144] | | modeling_hy_v4.py:713 |
| mlp | 算子 | self.mlp(hidden_states) | bf16 [B, T, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:714 |
| mo | 张量 | hidden_states | bf16 [B, T, 6144] | | modeling_hy_v4.py:714 |
| addf | 算子 | post.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2) + residual.float() | fp32 [B, T, 4] 与 bf16 [B, T, 6144] 与 bf16 [B, T, 4, 6144] → bf16 [B, T, 4, 6144] | | modeling_hy_v4.py:715 |
| hf | 张量 | hidden_states | bf16 [B, T, 4, 6144] | | modeling_hy_v4.py:715 |
| ret | 张量 | (hidden_states, topk_indices) | bf16 [B, T, 4, 6144] × int32 [B, T, 2048] | 两个返回值：这一层的输出与它给出的 topk_indices | modeling_hy_v4.py:717 |
| pkv | 缓存 | past_key_values | 每层压缩后的 KV 潜向量与索引器 key | self_attn 里 update 与 update_indexer 写，后续前向读 | modeling_hy_v4.py:702 |

### 连线

- x → ahc
- ahc → pa
- ahc → hsa
- hsa → aln
- aln → hsan
- hsan → attn
- am → attn
- pe → attn
- ptk → attn
- pkv → attn
- attn → pkv
- attn → ao
- attn → tk
- ao → adda
- pa → adda
- x → adda：residual
- adda → ha
- ha → fhc
- ha → addf：residual
- fhc → pf
- fhc → hsf
- hsf → fln
- fln → hsfn
- hsfn → mlp
- mlp → mo
- mo → addf
- pf → addf
- addf → hf
- hf → ret
- tk → ret
- ahc 点开 → HYV4HyperConnection.forward
- fhc 点开 → HYV4HyperConnection.forward
- aln 点开 → HYV4RMSNorm.forward
- fln 点开 → HYV4RMSNorm.forward
- attn 点开 → HYV4Attention.forward
- mlp 点开 → HYV4MoE.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| ahc | 公式 | $pa, hsa = attn\_hc(hidden\_states)$，内部算式见 HYV4HyperConnection.forward |
| ahc | 说明 | 权重 fn [8, 24576]、base [8]、scale [2]；`post` 是后置门的系数，折叠成的单份隐状态在本图叫 `hsa`（在 HYV4HyperConnection.forward 里叫 out） |
| aln | 公式 | $hidden\_states = input\_layernorm(hidden\_states)$，内部算式见 HYV4RMSNorm.forward |
| attn | 公式 | $hidden\_states, attn\_weights, topk\_indices = self\_attn(hidden\_states, mask, pe, past\_key\_values, prev\_topk\_indices)$ |
| attn | 说明 | 第 0 层到第 77 层都调用；`attn_weights` 这次不取；shared 层不跑索引器，直接用 `ptk` 传来的 topk |
| adda | 公式 | $hidden\_states = post \odot hidden\_states + residual$，post 是 [B, T, 4]，沿通道维与单份隐状态相乘，residual 是进入本层的 hidden_states |
| adda | 说明 | 三项都在 fp32 里算，最后转回 bf16；源码里 `residual = hidden_states` 只是取了同一个张量的别名，没有单独的计算 |
| fhc | 公式 | $pf, hsf = ffn\_hc(hidden\_states)$，内部算式见 HYV4HyperConnection.forward |
| fln | 公式 | $hidden\_states = post\_attention\_layernorm(hidden\_states)$，内部算式见 HYV4RMSNorm.forward |
| mlp | 公式 | $mo = mlp(hsfn)$，第 0 层是稠密 FFN，其余层是 MoE |
| mlp | 说明 | 本层的 topk_indices 由 self_attn 返回（同图 `tk`），不经过 mlp；第 0 层是 HYV4MLP（稠密 FFN），其余 77 层是 HYV4MoE；本页 `mlp` 节点点开进 HYV4MoE.forward，HYV4MLP.forward 由 HYV4MoE 图里的 `shp` 节点点开 |
| addf | 公式 | $hidden\_states = post \odot hidden\_states + residual$，post 来自同图的 `fhc`（不是 `adda` 用的那一份） |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 残差混合 | $hidden\_states = post \odot hidden\_states + residual$，前馈那一步的 post 来自 `fhc` |
| 配置 | mlp_layer_types | 第 0 层 dense，其余 77 层 sparse |
| 配置 | hc_mult | 4 |
| 配置 | rms_norm_eps | 1e-05 |

## HYV4Attention.forward（modeling_hy_v4.py:395-481）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hs | 张量 | hidden_states | bf16 [B, T, 6144] | | modeling_hy_v4.py:405 |
| gate | 算子 | self.gate_proj(hidden_states).view(batch_size, seq_length, -1, self.gate_projection_size) | bf16 [B, T, 6144] → bf16 [B, T, 64, 256] | | modeling_hy_v4.py:409 |
| gs | 张量 | gate_states | bf16 [B, T, 64, 256] | 64 = num_attention_heads，256 = v_head_dim | modeling_hy_v4.py:409 |
| qres | 算子 | self.q_a_layernorm(self.q_a_proj(hidden_states)) | bf16 [B, T, 6144] → bf16 [B, T, 2048] | | modeling_hy_v4.py:411 |
| qr | 张量 | q_resid | bf16 [B, T, 2048] | 2048 = q_lora_rank | modeling_hy_v4.py:411 |
| qb | 算子 | self.q_b_proj(q_resid).view(query_shape).transpose(1, 2) | bf16 [B, T, 2048] → bf16 [B, 64, T, 256] | | modeling_hy_v4.py:412 |
| qs | 张量 | q_states | bf16 [B, 64, T, 256] | 256 = qk_head_dim = 192 + 64 | modeling_hy_v4.py:412 |
| qsplit | 算子 | torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) | bf16 [B, 64, T, 256] → bf16 [B, 64, T, 192] 与 bf16 [B, 64, T, 64] | | modeling_hy_v4.py:413 |
| qpass | 张量 | q_pass | bf16 [B, 64, T, 192] | 192 = qk_nope_head_dim | modeling_hy_v4.py:413 |
| qrot | 张量 | q_rot | bf16 [B, 64, T, 64] | 64 = qk_rope_head_dim | modeling_hy_v4.py:413 |
| kva | 算子 | self.kv_a_proj_with_mqa(hidden_states) | bf16 [B, T, 6144] → bf16 [B, T, 576] | | modeling_hy_v4.py:415 |
| ckv | 张量 | compressed_kv | bf16 [B, T, 576] | 576 = kv_lora_rank 512 + qk_rope_head_dim 64 | modeling_hy_v4.py:415 |
| kvs | 算子 | torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) | bf16 [B, T, 576] → bf16 [B, T, 512] 与 bf16 [B, T, 64] | | modeling_hy_v4.py:416 |
| kpass0 | 张量 | kv_pass | bf16 [B, T, 512] | 512 = kv_lora_rank | modeling_hy_v4.py:416 |
| krot0 | 张量 | k_rot | bf16 [B, T, 64] | 64 = qk_rope_head_dim | modeling_hy_v4.py:416 |
| knorm | 算子 | self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank) | bf16 [B, T, 512] → bf16 [B, 1, T, 512] | | modeling_hy_v4.py:418 |
| kpass | 张量 | k_pass | bf16 [B, 1, T, 512] | 单头，1 = 不分头 | modeling_hy_v4.py:418 |
| krotv | 算子 | k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim) | bf16 [B, T, 64] → bf16 [B, 1, T, 64] | | modeling_hy_v4.py:420 |
| krot | 张量 | k_rot | bf16 [B, 1, T, 64] | | modeling_hy_v4.py:420 |
| rope | 算子 | apply_rotary_pos_emb(q_rot, k_rot, cos, sin) | bf16 [B, 64, T, 64] 与 bf16 [B, 1, T, 64] → bf16 [B, 64, T, 64] 与 bf16 [B, 1, T, 64] | | modeling_hy_v4.py:423 |
| qr2 | 张量 | q_rot | bf16 [B, 64, T, 64] | 加过旋转的查询 | modeling_hy_v4.py:423 |
| kr2 | 张量 | k_rot | bf16 [B, 1, T, 64] | 加过旋转的键 | modeling_hy_v4.py:423 |
| pe | 张量 | position_embeddings | bf16 [1, T, 64] × 2 | cos 与 sin | modeling_hy_v4.py:421 |
| cache | 算子 | past_key_values.update(k_pass, k_rot, self.layer_idx) | bf16 [B, 1, T, 512] 与 bf16 [B, 1, T, 64] → bf16 [B, 1, L, 512] 与 bf16 [B, 1, L, 64] | | modeling_hy_v4.py:427 |
| kpassf | 张量 | k_pass | bf16 [B, 1, L, 512] | 含历史，L 是缓存长度 + T | modeling_hy_v4.py:427 |
| krotf | 张量 | k_rot | bf16 [B, 1, L, 64] | 含历史 | modeling_hy_v4.py:427 |
| catq | 算子 | torch.cat((q_pass, q_rot), dim=-1) | bf16 [B, 64, T, 192] 与 bf16 [B, 64, T, 64] → bf16 [B, 64, T, 256] | | modeling_hy_v4.py:429 |
| query | 张量 | query_states | bf16 [B, 64, T, 256] | | modeling_hy_v4.py:429 |
| expkv | 算子 | self.expand_kv(k_pass, k_rot) | bf16 [B, 1, L, 512] 与 bf16 [B, 1, L, 64] → bf16 [B, 64, L, 256] 与 bf16 [B, 64, L, 256] | | modeling_hy_v4.py:431 |
| keys | 张量 | key_states | bf16 [B, 64, L, 256] | 256 = qk_nope_head_dim 192 + qk_rope_head_dim 64 | modeling_hy_v4.py:431 |
| vals | 张量 | value_states | bf16 [B, 64, L, 256] | 256 = v_head_dim | modeling_hy_v4.py:431 |
| idx | 算子 | self.indexer(hidden_states, q_resid, position_embeddings, attention_mask[:, 0, :, :], position_ids, past_key_values) | bf16 [B, T, 6144] 与 bf16 [B, T, 2048] 与 bf16 [1, T, 64] × 2 与 bf16 [B, 1, T, L] → int32 [B, T, 2048]（索引器内部取 [:, 0, :, :] 并切成 [B, T, L]） | | modeling_hy_v4.py:435-442 |
| tkidx | 张量 | topk_indices | int32 [B, T, 2048] | 送给 imask 的 Top-2048 位置下标 | modeling_hy_v4.py:435 |
| ptk | 张量 | prev_topk_indices | int32 [B, T, 2048] | shared 层直接用这组下标，不再跑索引器 | modeling_hy_v4.py:444-446 |
| imask | 算子 | topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool).scatter(-1, topk_indices.long(), False).unsqueeze(1) | int32 [B, T, 2048] 与 bf16 [B, 64, L, 256] → bool [B, 1, T, L] | | modeling_hy_v4.py:451-453 |
| im | 张量 | index_mask | bool [B, 1, T, L] | True 表示被排除的位置 | modeling_hy_v4.py:450 |
| am | 张量 | attention_mask | bf16 [B, 1, T, L] | 加性因果掩码，L 是键的总长度，本次等于 T | modeling_hy_v4.py:399 |
| amask | 算子 | attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min) | bf16 [B, 1, T, L] 与 bool [B, 1, T, L] → bf16 [B, 1, T, L] | | modeling_hy_v4.py:459 |
| amk | 张量 | attention_mask | bf16 [B, 1, T, L] | 加上稀疏掩码后的加性掩码，没选中的位置是极小值 | modeling_hy_v4.py:459 |
| sinks | 张量 | sinks | fp32 [64] | 可学习注意力汇，64 = num_attention_heads | modeling_hy_v4.py:374 |
| attn | 算子 | attention_interface(self, query_states, key_states, value_states, attention_mask, dropout, scaling, indices, s_aux, **kwargs) | bf16 [B, 64, T, 256] 与 bf16 [B, 64, L, 256] 与 bf16 [B, 64, L, 256] 与 bf16 [B, 1, T, L] → bf16 [B, T, 64, 256] | | modeling_hy_v4.py:466-477 |
| aoh | 张量 | attn_output | bf16 [B, T, 64, 256] | | modeling_hy_v4.py:466 |
| ag | 算子 | attn_output * torch.sigmoid(gate_states) | bf16 [B, T, 64, 256] 与 bf16 [B, T, 64, 256] → bf16 [B, T, 64, 256] | | modeling_hy_v4.py:479 |
| agt | 张量 | attn_output | bf16 [B, T, 64, 256] | 门控后的输出 | modeling_hy_v4.py:479 |
| rsh | 算子 | attn_output.reshape(batch_size, seq_length, -1).contiguous() | bf16 [B, T, 64, 256] → bf16 [B, T, 16384] | | modeling_hy_v4.py:480 |
| ar | 张量 | attn_output | bf16 [B, T, 16384] | 16384 = 64 × 256 | modeling_hy_v4.py:480 |
| oproj | 算子 | self.o_proj(attn_output) | bf16 [B, T, 16384] → bf16 [B, T, 6144] | | modeling_hy_v4.py:481 |
| outs | 张量 | (o_proj(attn_output), attn_weights, topk_indices) | bf16 [B, T, 6144] × int32 [B, T, 2048] | 三个返回值，只用第一个和第三个；第 0、1、5、9…层传下去的是自己算的 topk_indices，其余层是上一层传来的 | modeling_hy_v4.py:481 |
| pkv | 缓存 | past_key_values | 压缩后的 KV 潜向量与索引器 key | update 写、update_indexer 写，后续前向读 | modeling_hy_v4.py:427, 441 |

### 连线

- hs → gate
- gate → gs
- hs → qres
- qres → qr
- qr → qb
- qb → qs
- qs → qsplit
- qsplit → qpass
- qsplit → qrot
- hs → kva
- kva → ckv
- ckv → kvs
- kvs → kpass0
- kvs → krot0
- kpass0 → knorm
- knorm → kpass
- krot0 → krotv
- krotv → krot
- pe → rope
- qrot → rope
- krot → rope
- rope → qr2
- rope → kr2
- kpass → cache
- kr2 → cache
- cache → kpassf
- cache → krotf
- cache → pkv
- pkv → cache
- qpass → catq
- qr2 → catq
- catq → query
- kpassf → expkv
- krotf → expkv
- expkv → keys
- expkv → vals
- hs → idx
- qr → idx
- pe → idx
- am → idx
- pkv → idx
- idx → pkv
- idx → tkidx
- tkidx → imask
- ptk → tkidx：shared 层不跑索引器，topk_indices 直接取上一层传来的 prev_topk_indices
- keys → imask：形状
- imask → im
- am → amask
- im → amask
- hs → amask：dtype
- amask → amk
- query → attn
- keys → attn
- vals → attn
- amk → attn
- sinks → attn：s_aux
- attn → aoh
- aoh → ag
- gs → ag
- ag → agt
- agt → rsh
- rsh → ar
- ar → oproj
- oproj → outs
- tkidx → outs
- qres 点开 → HYV4RMSNorm.forward
- knorm 点开 → HYV4RMSNorm.forward
- idx 点开 → HYV4Indexer.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| gate | 公式 | $gate\_states = gate\_proj(hidden\_states)$，gate_proj 把 6144 维映到 64 × 256 维 |
| gate | 说明 | gate_proj.weight [16384, 6144]，16384 = 64 × 256 |
| qres | 公式 | $q\_resid = q\_a\_layernorm(q\_a\_proj(hidden\_states))$，q_a_proj 先降到 q_lora_rank，再 RMSNorm |
| qres | 说明 | q_a_proj.weight [2048, 6144]、q_a_layernorm.weight [2048] |
| qb | 公式 | $q\_states = q\_b\_proj(q\_resid)$ 后 view 成 [B, T, 64, 256] 再转置成 [B, 64, T, 256] |
| qb | 说明 | q_b_proj.weight [16384, 2048]，16384 = 64 × 256 |
| qsplit | 公式 | $q\_pass, q\_rot = \mathrm{split}(q\_states, [192, 64])$，把最后一维切成不加旋转的 192 维和加旋转的 64 维 |
| kva | 公式 | $compressed\_kv = kv\_a\_proj\_with\_mqa(hidden\_states)$，一次投影同时给出 KV 潜向量与旋转键 |
| kva | 说明 | kv_a_proj_with_mqa.weight [576, 6144]，576 = 512 + 64 |
| kvs | 公式 | $kv\_pass, k\_rot = \mathrm{split}(compressed\_kv, [512, 64])$ |
| knorm | 公式 | $k\_pass = kv\_a\_layernorm(kv\_pass)$，再 view 成单头 [B, 1, T, 512] |
| knorm | 说明 | kv_a_layernorm.weight [512] |
| krotv | 公式 | $k\_rot$ 加一个长度为 1 的头维 |
| rope | 公式 | $q\_rot = q\_rot \cos + \mathrm{rotate\_half}(q\_rot) \sin$，$k\_rot = k\_rot \cos + \mathrm{rotate\_half}(k\_rot) \sin$，cos 与 sin 来自 position_embeddings |
| rope | 说明 | 只转 qk_rope_head_dim 这 64 维，是 apply_rotary_pos_emb 这个普通函数，不是类 |
| cache | 公式 | $k\_pass, k\_rot = past\_key\_values.update(k\_pass, k\_rot, layer\_idx)$ |
| cache | 说明 | 缓存里存的是压缩后的潜向量，不是展开后的 key 和 value |
| catq | 公式 | $query\_states = \mathrm{cat}(q\_pass, q\_rot)$，把 192 与 64 维拼回 256 维 |
| expkv | 公式 | $key\_states, value\_states = kv\_b\_proj(k\_pass)$ 切分，$k\_rot$ 广播到 64 个头；kv_b_proj 把 512 维潜向量展开成每头的 192 维键与 256 维值 |
| expkv | 说明 | kv_b_proj.weight [28672, 512]，28672 = 64 × 448，448 = qk_nope_head_dim 192 + v_head_dim 256；这一步是 expand_kv，输出的序列维是 L |
| idx | 公式 | $topk\_indices = indexer(hidden\_states, q\_resid, pe, mask)$，给出每个查询挑出的位置 |
| idx | 说明 | 只在 indexer_types 为 full 的层调用；shared 层不跑索引器，topk_indices 直接取上一层传来的 prev_topk_indices |
| imask | 公式 | $index\_mask = \mathrm{scatter}(\mathrm{ones}(L), topk\_indices, False)$，选中位置为 False，其余为 True |
| amask | 公式 | $attention\_mask = \mathrm{masked\_fill}(attention\_mask, index\_mask, \mathrm{finfo.min})$，被排除的位置加上极小值 |
| amask | 说明 | 本次走 eager 这条分支 |
| attn | 公式 | $attn\_output = \mathrm{softmax}(q k^\top \cdot scaling + mask)\, v$，k、v 是展开后的 key_states、value_states，mask 是加性掩码，scaling 是 qk_head_dim 的 −1/2 次方，即 $256^{-1/2}$ |
| attn | 说明 | 走 eager_attention_forward：key 与 value 不分组复制（num_key_value_groups 为 1）；注意力 logits 与 sinks 拼在一起做 softmax，再去掉 sinks 那一列；sinks [64]。实参 `indices` 就是源码里的 `sparse_indices`，这条路为 None（掩码已在 amask 里散进 attention_mask），只有内核分支才把 Top-2048 下标传下去 |
| ag | 公式 | $attn\_output = attn\_output \odot \sigma(gate\_states)$，逐元素乘门控 |
| rsh | 公式 | $attn\_output = \mathrm{reshape}(attn\_output, [B, T, 64 \times 256])$ |
| oproj | 公式 | $hidden\_states = o\_proj(attn\_output)$ |
| oproj | 说明 | o_proj.weight [6144, 16384] |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 稀疏注意力 | $attn\_output = \mathrm{softmax}(q k^\top \cdot scaling + mask)\, v$，mask 在因果掩码的基础上，再给 Top-2048 外的位置加极小值 |
| 公式 | 门控 | $attn\_output = attn\_output \odot \sigma(gate\_states)$ |
| 配置 | q_lora_rank | 2048 |
| 配置 | kv_lora_rank | 512 |
| 配置 | qk_head_dim | 256 |
| 配置 | v_head_dim | 256 |
| 配置 | index_topk | 2048 |

## HYV4Indexer.forward（modeling_hy_v4.py:188-258）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hs | 张量 | hidden_states | bf16 [B, T, 6144] | | modeling_hy_v4.py:219 |
| qr | 张量 | q_resid | bf16 [B, T, 2048] | 与注意力共用的查询潜向量 | modeling_hy_v4.py:221 |
| wqb | 算子 | self.wq_b(q_resid) | bf16 [B, T, 2048] → bf16 [B, T, 4096] | | modeling_hy_v4.py:221 |
| qv | 张量 | q | bf16 [B, T, 4096] | 4096 = 32 × 128 | modeling_hy_v4.py:221 |
| qview | 算子 | q.view(batch_size, seq_len, self.n_heads, self.head_dim) | bf16 [B, T, 4096] → bf16 [B, T, 32, 128] | | modeling_hy_v4.py:222 |
| qt | 张量 | q | bf16 [B, T, 32, 128] | 32 = index_n_heads，128 = index_head_dim | modeling_hy_v4.py:222 |
| qsplit | 算子 | torch.split(q, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1) | bf16 [B, T, 32, 128] → bf16 [B, T, 32, 64] 与 bf16 [B, T, 32, 64] | | modeling_hy_v4.py:224 |
| qp | 张量 | q_pass | bf16 [B, T, 32, 64] | 不加旋转的部分 | modeling_hy_v4.py:224 |
| qrr | 张量 | q_rot | bf16 [B, T, 32, 64] | 加旋转的部分 | modeling_hy_v4.py:224 |
| wkv | 算子 | self.k_norm(self.wk(hidden_states).to(self.k_norm.weight.dtype)).to(hidden_states.dtype).unsqueeze(2) | bf16 [B, T, 6144] → bf16 [B, T, 1, 128] | | modeling_hy_v4.py:227-229 |
| kv | 张量 | k | bf16 [B, T, 1, 128] | 单头 | modeling_hy_v4.py:227 |
| ksplit | 算子 | torch.split(k, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1) | bf16 [B, T, 1, 128] → bf16 [B, T, 1, 64] 与 bf16 [B, T, 1, 64] | | modeling_hy_v4.py:230 |
| kp | 张量 | k_pass | bf16 [B, T, 1, 64] | | modeling_hy_v4.py:230 |
| kr | 张量 | k_rot | bf16 [B, T, 1, 64] | | modeling_hy_v4.py:230 |
| rope | 算子 | apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2) | bf16 [B, T, 32, 64] 与 bf16 [B, T, 1, 64] → bf16 [B, T, 32, 64] 与 bf16 [B, T, 1, 64] | | modeling_hy_v4.py:232 |
| iqr2 | 张量 | q_rot | bf16 [B, T, 32, 64] | | modeling_hy_v4.py:232 |
| ikr2 | 张量 | k_rot | bf16 [B, T, 1, 64] | | modeling_hy_v4.py:232 |
| pe | 张量 | position_embeddings | bf16 [1, T, 64] × 2 | cos 与 sin | modeling_hy_v4.py:220 |
| catq | 算子 | torch.cat([q_pass, q_rot], dim=-1) | bf16 [B, T, 32, 64] × 2 → bf16 [B, T, 32, 128] | | modeling_hy_v4.py:233 |
| qc | 张量 | q | bf16 [B, T, 32, 128] | | modeling_hy_v4.py:233 |
| catk | 算子 | torch.cat([k_pass, k_rot], dim=-1).squeeze(2) | bf16 [B, T, 1, 64] × 2 → bf16 [B, T, 128] | | modeling_hy_v4.py:234 |
| kc | 张量 | k | bf16 [B, T, 128] | | modeling_hy_v4.py:234 |
| upd | 算子 | past_key_values.update_indexer(k, self.layer_idx) | bf16 [B, T, 128] → bf16 [B, L, 128] | | modeling_hy_v4.py:237 |
| ki | 张量 | k | bf16 [B, L, 128] | 含历史，L 是缓存长度 + T | modeling_hy_v4.py:237 |
| sc | 算子 | torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) | bf16 [B, T, 32, 128] 与 bf16 [B, L, 128] → fp32 [B, T, 32, L] | | modeling_hy_v4.py:239 |
| scores | 张量 | scores | fp32 [B, T, 32, L] | 每个头对每个位置的打分 | modeling_hy_v4.py:239 |
| relu | 算子 | F.relu(scores) | fp32 [B, T, 32, L] → fp32 [B, T, 32, L] | | modeling_hy_v4.py:240 |
| scor | 张量 | scores | fp32 [B, T, 32, L] | 取 ReLU 之后的打分 | modeling_hy_v4.py:240 |
| wp | 算子 | self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5) * self.softmax_scale | bf16 [B, T, 6144] → fp32 [B, T, 32] | | modeling_hy_v4.py:244-248 |
| wts | 张量 | weights | fp32 [B, T, 32] | 每个头的权重，再乘两个常数 | modeling_hy_v4.py:244 |
| matmul | 算子 | torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2) | fp32 [B, T, 1, 32] 与 fp32 [B, T, 32, L] → fp32 [B, T, L] | | modeling_hy_v4.py:249 |
| isco | 张量 | index_scores | fp32 [B, T, L] | 按头加权求和后的位置打分 | modeling_hy_v4.py:249 |
| aadd | 算子 | index_scores + attention_mask | fp32 [B, T, L] 与 bf16 [B, T, L] → fp32 [B, T, L] | | modeling_hy_v4.py:255 |
| iscm | 张量 | index_scores | fp32 [B, T, L] | 加上因果掩码之后 | modeling_hy_v4.py:255 |
| topk | 算子 | index_scores.topk(min(self.index_topk, index_scores.shape[-1]), dim=-1).indices.to(torch.int32) | fp32 [B, T, L] → int32 [B, T, 2048] | | modeling_hy_v4.py:257-258 |
| tki | 张量 | index_scores.topk(min(self.index_topk, index_scores.shape[-1]), dim=-1).indices.to(torch.int32) | int32 [B, T, 2048] | 源码里没有单独的变量名，这是产生它的表达式 | modeling_hy_v4.py:257-258 |
| iam | 张量 | attention_mask | bf16 [B, T, L] | 加性因果掩码；进索引器之前已取 [:, 0, :, :]，L 是缓存长度 + T | modeling_hy_v4.py:255 |
| ikc | 缓存 | past_key_values | 索引器自己的 key，与注意力的 KV 分开存 | update_indexer 写，后续前向读 | modeling_hy_v4.py:237 |

### 连线

- hs → wkv
- qr → wqb
- wqb → qv
- qv → qview
- qview → qt
- qt → qsplit
- qsplit → qp
- qsplit → qrr
- wkv → kv
- kv → ksplit
- ksplit → kp
- ksplit → kr
- pe → rope
- qrr → rope
- kr → rope
- rope → iqr2
- rope → ikr2
- qp → catq
- iqr2 → catq
- catq → qc
- kp → catk
- ikr2 → catk
- catk → kc
- kc → upd
- upd → ki
- upd → ikc
- ikc → upd
- qc → sc
- ki → sc
- sc → scores
- scores → relu
- relu → scor
- hs → wp：输入数据，权重另取 dtype
- wp → wts
- wts → matmul
- scor → matmul
- matmul → isco
- iam → aadd
- isco → aadd
- aadd → iscm
- iscm → topk
- topk → tki

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| wqb | 公式 | $q = wq\_b(q\_resid)$，把查询潜向量映到索引器的 32 个头 |
| wqb | 说明 | wq_b.weight [4096, 2048]，4096 = 32 × 128 |
| qview | 公式 | $q$ view 成 [B, T, 32, 128] |
| qsplit | 公式 | $q\_pass, q\_rot = \mathrm{split}(q, [64, 64])$ |
| qsplit | 说明 | 这里的 `self.head_dim` 是索引器的 head_dim，即 index_head_dim 128，与 config.json 里 head_dim 64 不是同一个值，所以切出的是 64 + 64 而非 192 + 64 |
| wkv | 公式 | $k = k\_norm(wk(hidden\_states))$，先算出 128 维再归一化 |
| wkv | 说明 | wk.weight [128, 6144]、k_norm.weight [128]、k_norm.bias [128]；k_norm 是 PyTorch 的 nn.LayerNorm，在 fp32 里算 |
| ksplit | 公式 | $k\_pass, k\_rot = \mathrm{split}(k, [64, 64])$ |
| rope | 公式 | 同注意力的旋转，只转 64 维，unsqueeze_dim 为 2 |
| catq | 公式 | $q = \mathrm{cat}(q\_pass, q\_rot)$，拼回 128 维 |
| catk | 公式 | $k = \mathrm{cat}(k\_pass, k\_rot)$，去掉长度 1 的头维 |
| upd | 公式 | $k = past\_key\_values.update\_indexer(k, layer\_idx)$ |
| upd | 说明 | 索引器 key 单独一份缓存，与注意力的压缩 KV 分开；输出序列维是 L |
| sc | 公式 | $scores = q k^\top$，在 fp32 里算，q 是 [B, T, 32, 128]，k 是单头 [B, L, 128] |
| relu | 公式 | $scores = \max(scores, 0)$ |
| wp | 公式 | $weights = weights\_proj(hidden\_states) \cdot 32^{-1/2} \cdot 128^{-1/2}$，weights_proj 给出每个头的权重，两个常数分别是 n_heads 与 softmax_scale |
| wp | 说明 | weights_proj.weight [32, 6144]；32 是 index_n_heads，128 是 index_head_dim，softmax_scale 是 index_head_dim 的 −1/2 次方 |
| matmul | 公式 | $index\_scores = \sum_h weights_h \cdot scores_h$，按头加权求和 |
| aadd | 公式 | $index\_scores = index\_scores + attention\_mask$，加性掩码里晚于这个查询的位置是极小值 |
| topk | 公式 | $topk\_indices = \mathrm{topk}(index\_scores, \min(2048, L)).indices$，取打分最高的若干个位置 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 打分 | $index\_scores = \sum_h weights_h \cdot \mathrm{relu}(\sum_d q_{h,d} k_d)$，k 是单头共享、广播到 32 个头的，weights 是每个头的权重 |
| 配置 | index_n_heads | 32 |
| 配置 | index_head_dim | 128 |
| 配置 | index_topk | 2048 |

## HYV4MoE.forward（modeling_hy_v4.py:593-600）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hs | 张量 | hidden_states | bf16 [B, T, 6144] | | modeling_hy_v4.py:594 |
| gate | 算子 | self.gate(hidden_states) | bf16 [B, T, 6144] → fp32 [B×T, 256] 与 fp32 [B×T, 8] 与 int64 [B×T, 8] | | modeling_hy_v4.py:596 |
| tw | 张量 | topk_weights | fp32 [B×T, 8] | 选中专家的权重 | modeling_hy_v4.py:596 |
| ti | 张量 | topk_indices | int64 [B×T, 8] | 选中的专家号 | modeling_hy_v4.py:596 |
| mv | 算子 | hidden_states.view(-1, hidden_states.shape[-1]) | bf16 [B, T, 6144] → bf16 [B×T, 6144] | | modeling_hy_v4.py:597 |
| hfl | 张量 | hidden_states | bf16 [B×T, 6144] | 展平后的 token；展平前叫 `hs`，两者是同一份数据的两个形状 | modeling_hy_v4.py:597 |
| exp | 算子 | self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape) | bf16 [B×T, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:598 |
| he | 张量 | hidden_states | bf16 [B, T, 6144] | 路由专家加权求和的结果 | modeling_hy_v4.py:598 |
| shp | 算子 | self.shared_experts(residuals) | bf16 [B, T, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:599 |
| hsh | 张量 | self.shared_experts(residuals) | bf16 [B, T, 6144] | 共享专家的输出，源码里没有单独的变量名 | modeling_hy_v4.py:599 |
| add | 算子 | hidden_states + self.shared_experts(residuals) | bf16 [B, T, 6144] × 2 → bf16 [B, T, 6144] | | modeling_hy_v4.py:599 |
| out | 张量 | hidden_states | bf16 [B, T, 6144] | | modeling_hy_v4.py:599 |

### 连线

- hs → gate
- gate → tw
- gate → ti
- hs → mv
- mv → hfl
- hfl → exp：hidden_states 本体
- hs → exp：形状（orig_shape，用来 view 回 [B, T, 6144]）
- ti → exp
- tw → exp
- exp → he
- hs → shp：residuals
- shp → hsh
- he → add
- hsh → add
- add → out
- gate 点开 → HYV4TopkRouter.forward
- exp 点开 → HYV4Experts.forward
- shp 点开 → HYV4MLP.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| gate | 公式 | $router\_logits, topk\_weights, topk\_indices = gate(hidden\_states)$ |
| gate | 说明 | 三个返回值里只用 topk_weights 与 topk_indices；router_logits 是每个 token 对 256 个专家的打分，这次前向不取 |
| mv | 公式 | $hidden\_states$ 展平成 [B×T, 6144]，按 token 逐个路由 |
| exp | 公式 | $hidden\_states = experts(hidden\_states, topk\_indices, topk\_weights)$，再 view 回 [B, T, 6144] |
| shp | 公式 | $hsh = shared\_experts(residuals)$，residuals 是进入本层的 hidden_states 别名 |
| add | 公式 | $hidden\_states = he + hsh$，he 是路由专家的输出，hsh 是共享专家的输出 |
| add | 说明 | 相加右边就是同图 `shp` 算出的共享专家输出，没有重复调用 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 混合专家 | $hidden\_states = experts(hidden\_states, topk\_indices, topk\_weights) + shared\_experts(residuals)$ |
| 配置 | n_routed_experts | 256 |
| 配置 | n_shared_experts | 1 |
| 配置 | num_experts_per_tok | 8 |
| 配置 | moe_intermediate_size | 2048 |

## HYV4TopkRouter.forward（modeling_hy_v4.py:513-538）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hs | 张量 | hidden_states | bf16 [B, T, 6144] | 与 HYV4MoE 的 hidden_states 是同一个量，进来时还没展平 | modeling_hy_v4.py:514 |
| view | 算子 | hidden_states.view(-1, self.hidden_dim) | bf16 [B, T, 6144] → bf16 [B×T, 6144] | | modeling_hy_v4.py:514 |
| hfr | 张量 | hidden_states | bf16 [B×T, 6144] | | modeling_hy_v4.py:514 |
| lin | 算子 | F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32)) | bf16 [B×T, 6144] → fp32 [B×T, 256] | | modeling_hy_v4.py:515 |
| rl | 张量 | router_logits | fp32 [B×T, 256] | 每个 token 对 256 个专家的打分 | modeling_hy_v4.py:515 |
| sig | 算子 | router_logits.sigmoid() | fp32 [B×T, 256] → fp32 [B×T, 256] | | modeling_hy_v4.py:516 |
| sc | 张量 | scores | fp32 [B×T, 256] | 打分过 sigmoid | modeling_hy_v4.py:516 |
| corr | 算子 | scores + self.e_score_correction_bias | fp32 [B×T, 256] 与 fp32 [256] → fp32 [B×T, 256] | | modeling_hy_v4.py:517 |
| sfc | 张量 | scores_for_choice | fp32 [B×T, 256] | 只在选专家时用的打分 | modeling_hy_v4.py:517 |
| grp | 算子 | scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group).topk(2, dim=-1)[0].sum(dim=-1) | fp32 [B×T, 256] → fp32 [B×T, 1] | | modeling_hy_v4.py:518-522 |
| grps | 张量 | group_scores | fp32 [B×T, 1] | n_group 为 1，只有一组 | modeling_hy_v4.py:518 |
| gtop | 算子 | torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1] | fp32 [B×T, 1] → int64 [B×T, 1] | | modeling_hy_v4.py:523 |
| gidx | 张量 | group_idx | int64 [B×T, 1] | topk_group 为 1，取打分最高的一组 | modeling_hy_v4.py:523 |
| gmask | 算子 | group_mask.scatter_(1, group_idx, 1) | fp32 [B×T, 1] → fp32 [B×T, 1] | | modeling_hy_v4.py:524-525 |
| gmk | 张量 | group_mask | fp32 [B×T, 1] | 选中的组置 1 | modeling_hy_v4.py:524-525 |
| smask | 算子 | group_mask.unsqueeze(-1).expand(-1, self.num_group, self.num_experts // self.num_group).reshape(-1, self.num_experts) | fp32 [B×T, 1] → fp32 [B×T, 256] | | modeling_hy_v4.py:526-530 |
| smk | 张量 | score_mask | fp32 [B×T, 256] | 组掩码展开到每个专家 | modeling_hy_v4.py:526 |
| mfill | 算子 | scores_for_choice.masked_fill(~score_mask.bool(), float("-inf")) | fp32 [B×T, 256] × 2 → fp32 [B×T, 256] | | modeling_hy_v4.py:531 |
| sfm | 张量 | scores_for_choice | fp32 [B×T, 256] | 组外的专家置为负无穷 | modeling_hy_v4.py:531 |
| topk | 算子 | torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1] | fp32 [B×T, 256] → int64 [B×T, 8] | | modeling_hy_v4.py:532 |
| ti | 张量 | topk_indices | int64 [B×T, 8] | 选中的专家号，8 = num_experts_per_tok | modeling_hy_v4.py:532 |
| gath | 算子 | scores.gather(1, topk_indices) | fp32 [B×T, 256] 与 int64 [B×T, 8] → fp32 [B×T, 8] | | modeling_hy_v4.py:533 |
| tw | 张量 | topk_weights | fp32 [B×T, 8] | 未归一化的专家权重 | modeling_hy_v4.py:533 |
| norm | 算子 | topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20 | fp32 [B×T, 8] → fp32 [B×T, 8] | | modeling_hy_v4.py:535-536 |
| twn | 张量 | topk_weights | fp32 [B×T, 8] | 归一化后的权重 | modeling_hy_v4.py:535-536 |
| scal | 算子 | topk_weights * self.routed_scaling_factor | fp32 [B×T, 8] → fp32 [B×T, 8] | | modeling_hy_v4.py:537 |
| tws | 张量 | topk_weights | fp32 [B×T, 8] | 再乘 2.827 | modeling_hy_v4.py:537 |
| routs | 张量 | (router_logits, topk_weights, topk_indices) | fp32 [B×T, 256] × fp32 [B×T, 8] × int64 [B×T, 8] | 三个返回值 | modeling_hy_v4.py:538 |
| bias | 张量 | e_score_correction_bias | fp32 [256] | 每个专家的偏置，只在选专家时加，不加进权重 | modeling_hy_v4.py:517 |

### 连线

- hs → view
- view → hfr
- hfr → lin
- bias → corr：作为参数
- lin → rl
- rl → sig
- sig → sc
- sc → corr
- corr → sfc
- sfc → grp
- grp → grps
- grps → gtop
- grps → gmask：形状与 dtype
- gtop → gidx
- gidx → gmask
- gmask → gmk
- gmk → smask
- smask → smk
- sfc → mfill
- smk → mfill
- mfill → sfm
- sfm → topk
- topk → ti
- sc → gath
- ti → gath
- gath → tw
- tw → norm
- norm → twn
- twn → scal
- scal → tws
- rl → routs
- tws → routs
- ti → routs

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| view | 公式 | $hidden\_states$ view 成 [B×T, 6144] |
| lin | 公式 | $router\_logits = hidden\_states\, weight^\top$，两边都转 fp32 |
| lin | 说明 | weight [256, 6144]，fp32 里算 |
| sig | 公式 | $scores = \sigma(router\_logits)$ |
| corr | 公式 | $scores\_for\_choice = scores + e\_score\_correction\_bias$，偏置只在选专家时用 |
| grp | 公式 | $group\_scores = \sum_{j \in \mathrm{top2}} scores\_for\_choice_j$，每组取打分最高的两个求和 |
| grp | 说明 | n_group 为 1，全部 256 个专家同属一组 |
| gtop | 公式 | $group\_idx = \mathrm{topk}(group\_scores, topk\_group)$，topk_group 为 1 |
| gmask | 公式 | $group\_mask$ 在选中的组上置 1 |
| smask | 公式 | $score\_mask$ 把组掩码展开到 256 个专家 |
| mfill | 公式 | $scores\_for\_choice = \mathrm{masked\_fill}(scores\_for\_choice, \lnot score\_mask, -\infty)$ |
| topk | 公式 | $topk\_indices = \mathrm{topk}(scores\_for\_choice, 8).indices$ |
| gath | 公式 | $topk\_weights = scores.gather(topk\_indices)$，用的是没过偏置、没置负无穷的 scores |
| norm | 公式 | $topk\_weights = topk\_weights / (\sum topk\_weights + 10^{-20})$，归一化 |
| scal | 公式 | $topk\_weights = topk\_weights \cdot routed\_scaling\_factor$ |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 打分与选择 | $scores = \sigma(hidden\_states\, weight^\top)$，$scores\_for\_choice = scores + bias$，取 top-8 的专家号，权重取自 scores |
| 配置 | n_routed_experts | 256 |
| 配置 | num_experts_per_tok | 8 |
| 配置 | routed_scaling_factor | 2.827 |
| 配置 | n_group | 1 |
| 配置 | topk_group | 1 |

## HYV4Experts.forward（modeling_hy_v4.py:554-569）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hfl | 张量 | hidden_states | bf16 [B×T, 6144] | 与 HYV4MoE 图里展平后的 hidden_states 是同一个量 | modeling_hy_v4.py:557 |
| ti | 张量 | top_k_index | int64 [B×T, 8] | 选中的专家号 | modeling_hy_v4.py:559 |
| tw | 张量 | top_k_weights | fp32 [B×T, 8] | 专家权重 | modeling_hy_v4.py:555 |
| mask | 算子 | F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0) | int64 [B×T, 8] → int64 [256 + 1, 8, B×T] | | modeling_hy_v4.py:559 |
| mk | 张量 | mask | int64 [256 + 1, 8, B×T] | 多出的第 256 号槽承接 one_hot 对齐用的 0，恒为 0 | modeling_hy_v4.py:559 |
| hit | 算子 | torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero() | int64 [256 + 1, 8, B×T] → int64 [n, 1] | | modeling_hy_v4.py:560 |
| hitn | 张量 | hit | int64 [n, 1] | 这次前向真正用到的专家号；n 是命中的专家个数 | modeling_hy_v4.py:560 |
| zero | 算子 | torch.zeros_like(hidden_states) | bf16 [B×T, 6144] → bf16 [B×T, 6144] | | modeling_hy_v4.py:557 |
| final0 | 张量 | final | bf16 [B×T, 6144] | 输出累加器，初值为 0 | modeling_hy_v4.py:557 |
| where | 算子 | torch.where(mask[expert_idx]) | int64 [8, B×T] → int64 [m] 与 int64 [m] | | modeling_hy_v4.py:565 |
| pos | 张量 | top_k_pos | int64 [m] | 命中该专家的权重槽位；m 是命中该专家的 token 数 | modeling_hy_v4.py:565 |
| tok | 张量 | token_idx | int64 [m] | 命中该专家的 token | modeling_hy_v4.py:565 |
| eup | 算子 | self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])) | bf16 [m, 6144] → bf16 [m, 2048] | | modeling_hy_v4.py:566 |
| cur | 张量 | current | bf16 [m, 2048] | 门控后的中间结果 | modeling_hy_v4.py:566 |
| edown | 算子 | F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None] | bf16 [m, 2048] → fp32 [m, 6144] | | modeling_hy_v4.py:567 |
| cur2 | 张量 | current | fp32 [m, 6144] | 乘过权重的专家输出，与 fp32 权重相乘后升为 fp32 | modeling_hy_v4.py:567 |
| iadd | 算子 | final.index_add_(0, token_idx, current.to(final.dtype)) | bf16 [B×T, 6144] 与 int64 [m] 与 fp32 [m, 6144] → bf16 [B×T, 6144] | | modeling_hy_v4.py:568 |
| final | 张量 | final | bf16 [B×T, 6144] | 累加完的输出 | modeling_hy_v4.py:569 |

### 连线

- hfl → zero：形状与 dtype
- zero → final0
- ti → mask
- mask → mk
- mk → hit
- hit → hitn
- hitn → where
- mk → where
- where → pos
- where → tok
- hfl → eup：取行
- tok → eup：取行
- hitn → eup：取专家号
- eup → cur
- cur → edown
- tw → edown
- pos → edown：取权重
- tok → edown：取行
- hitn → edown：取专家号
- edown → cur2
- final0 → iadd
- tok → iadd：累加位置
- cur2 → iadd
- iadd → final

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| zero | 公式 | $final = \mathrm{zeros\_like}(hidden\_states)$ |
| mask | 公式 | $mask = \mathrm{one\_hot}(top\_k\_index, 257)$，再换成 [257, 8, B×T] |
| mask | 说明 | 类别数取 num_experts + 1，专家号只到 255，第 256 类从不命中、恒为 0，统计命中专家时自然被排除 |
| hit | 公式 | $hit = \mathrm{nonzero}(\sum mask > 0)$，列出这次前向命中的专家号 |
| where | 公式 | $top\_k\_pos, token\_idx = \mathrm{where}(mask[expert\_idx])$，找出命中这个专家的位置与 token；expert_idx 取 `hitn` 里的每个专家号，逐个专家处理 |
| where | 说明 | 源码里 expert_idx 是 `for expert_idx in hit` 的循环变量 |
| eup | 公式 | $current = \mathrm{silu}(\mathrm{clamp}(gate)) \odot \mathrm{clamp}(up)$，$gate, up = \mathrm{chunk}(gate\_up\_proj(hidden\_states[token\_idx]), 2)$，gate 只夹上界、up 上下都夹，界限都是 swiglu_limit |
| eup | 说明 | gate_up_proj [256, 4096, 6144]，一次给出 gate 与 up 两段；swiglu_limit 是 10.0 |
| edown | 公式 | $current = down\_proj(current) \cdot top\_k\_weights[token\_idx, top\_k\_pos]$ |
| edown | 说明 | down_proj [256, 6144, 2048] |
| iadd | 公式 | $final = \mathrm{index\_add}(final, token\_idx, current)$，把每个 token 命中的专家输出按权重累加 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 单个专家 | $y = down\_proj(\mathrm{silu}(clamp(gate)) \odot clamp(up))$，再乘权重累加回 token |
| 配置 | moe_intermediate_size | 2048 |
| 配置 | swiglu_limit | 10.0 |
| 配置 | dtype | bfloat16 |

## HYV4MLP.forward（modeling_hy_v4.py:495-497）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| mx0 | 张量 | x | bf16 [B, T, 6144] | 第 0 层的输入形状；shared_experts 用时隐层维相同、中间维是 moe_intermediate_size | modeling_hy_v4.py:496 |
| gp | 算子 | self.gate_proj(x) | bf16 [B, T, 6144] → bf16 [B, T, 18432] | | modeling_hy_v4.py:496 |
| gv | 张量 | gate_proj(x) | bf16 [B, T, 18432] | 门控分支，源码里没有单独的变量名 | modeling_hy_v4.py:496 |
| up | 算子 | self.up_proj(x) | bf16 [B, T, 6144] → bf16 [B, T, 18432] | | modeling_hy_v4.py:496 |
| uv | 张量 | up_proj(x) | bf16 [B, T, 18432] | 上分支，源码里没有单独的变量名 | modeling_hy_v4.py:496 |
| act | 算子 | self.act_fn(gate_proj(x)) | bf16 [B, T, 18432] → bf16 [B, T, 18432] | | modeling_hy_v4.py:496 |
| av | 张量 | act_fn(gate_proj(x)) | bf16 [B, T, 18432] | silu 之后的门控 | modeling_hy_v4.py:496 |
| mul | 算子 | self.act_fn(self.gate_proj(x)) * self.up_proj(x) | bf16 [B, T, 18432] × 2 → bf16 [B, T, 18432] | | modeling_hy_v4.py:496 |
| mv | 张量 | act_fn(gate_proj(x)) * up_proj(x) | bf16 [B, T, 18432] | 两分支逐元素相乘 | modeling_hy_v4.py:496 |
| down | 算子 | self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) | bf16 [B, T, 18432] → bf16 [B, T, 6144] | | modeling_hy_v4.py:496 |
| out | 张量 | down_proj | bf16 [B, T, 6144] | HYV4MLP.forward 的返回值 | modeling_hy_v4.py:497 |

### 连线

- mx0 → gp
- gp → gv
- gv → act
- act → av
- mx0 → up
- up → uv
- av → mul
- uv → mul
- mul → mv
- mv → down
- down → out

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| gp | 公式 | $gate = gate\_proj(x)$，中间维是 intermediate_size 18432 |
| gp | 说明 | gate_proj.weight [18432, 6144] |
| up | 公式 | $up = up\_proj(x)$ |
| up | 说明 | up_proj.weight [18432, 6144] |
| act | 公式 | $act = \mathrm{silu}(gate\_proj(x))$ |
| mul | 公式 | $act \odot up$ |
| down | 公式 | $down\_proj(act\_fn(gate\_proj(x)) \odot up\_proj(x))$，就是作用在同图 `mv` 上的投影 |
| down | 说明 | down_proj.weight [6144, 18432]；第 0 层用这个稠密 FFN，中间维 18432。这个类也被 MoE 的 shared_experts 用，那时隐层维不变、中间维是 moe_intermediate_size × n_shared_experts = 2048，投影结果在 HYV4MoE 图的 `add` 处与路由专家相加 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 稠密 FFN | $y = down\_proj(\mathrm{silu}(gate\_proj(x)) \odot up\_proj(x))$ |
| 配置 | intermediate_size | 18432 |
| 配置 | moe_intermediate_size | 2048（shared_experts 用时） |
| 配置 | hidden_act | silu |

## HYV4HyperConnection.forward（modeling_hy_v4.py:621-640）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hst | 张量 | hidden_streams | bf16 [B, T, 4, 6144] | 4 份残差流 | modeling_hy_v4.py:626 |
| flat | 算子 | hidden_streams.flatten(2).float() | bf16 [B, T, 4, 6144] → fp32 [B, T, 24576] | | modeling_hy_v4.py:626 |
| fl | 张量 | flat | fp32 [B, T, 24576] | 24576 = 4 × 6144 | modeling_hy_v4.py:626 |
| norm | 算子 | self.input_norm(flat) | fp32 [B, T, 24576] → fp32 [B, T, 1] | | modeling_hy_v4.py:628 |
| nrm | 张量 | input_norm(flat) | fp32 [B, T, 1] | 每一行按所有通道整体求逆均方根 | modeling_hy_v4.py:628 |
| lin | 算子 | F.linear(flat, self.fn.float()) | fp32 [B, T, 24576] → fp32 [B, T, 8] | | modeling_hy_v4.py:628 |
| lr | 张量 | F.linear(flat, fn) | fp32 [B, T, 8] | 8 = 2 × hc_mult，前 4 个是前置门 logits，后 4 个是后置门 logits | modeling_hy_v4.py:628 |
| mix | 算子 | F.linear(flat, self.fn.float()) * self.input_norm(flat) | fp32 [B, T, 8] 与 fp32 [B, T, 1] → fp32 [B, T, 8] | | modeling_hy_v4.py:628 |
| mx | 张量 | mixes | fp32 [B, T, 8] | 混合 logits | modeling_hy_v4.py:628 |
| bsplit | 算子 | self.base.split(self.hc_mult, dim=-1) | fp32 [8] → fp32 [4] × 2 | | modeling_hy_v4.py:631 |
| base | 张量 | base | fp32 [8] | 前置门与后置门的偏置，拼成一条 | modeling_hy_v4.py:631 |
| pb | 张量 | pre_b | fp32 [4] | 前置门的偏置 | modeling_hy_v4.py:631 |
| pob | 张量 | post_b | fp32 [4] | 后置门的偏置 | modeling_hy_v4.py:631 |
| ssplit | 算子 | self.scale.unbind(0) | fp32 [2] → fp32 [] × 2 | | modeling_hy_v4.py:632 |
| scale | 张量 | scale | fp32 [2] | 前置门与后置门的缩放 | modeling_hy_v4.py:632 |
| psc | 张量 | pre_scale | fp32 [] | 前置门的缩放 | modeling_hy_v4.py:632 |
| posc | 张量 | post_scale | fp32 [] | 后置门的缩放 | modeling_hy_v4.py:632 |
| msplit | 算子 | mixes.split(self.hc_mult, dim=-1) | fp32 [B, T, 8] → fp32 [B, T, 4] × 2 | | modeling_hy_v4.py:633 |
| prel | 张量 | pre_logits | fp32 [B, T, 4] | | modeling_hy_v4.py:633 |
| postl | 张量 | post_logits | fp32 [B, T, 4] | | modeling_hy_v4.py:633 |
| pre | 算子 | torch.sigmoid(pre_logits * pre_scale + pre_b) + self.hc_eps | fp32 [B, T, 4] → fp32 [B, T, 4] | | modeling_hy_v4.py:635 |
| pv | 张量 | pre | fp32 [B, T, 4] | 4 份残差流的折叠系数 | modeling_hy_v4.py:635 |
| post | 算子 | self.hc_post_magnitude * torch.sigmoid(post_logits * post_scale + post_b) + self.hc_eps | fp32 [B, T, 4] → fp32 [B, T, 4] | | modeling_hy_v4.py:637 |
| pov | 张量 | post | fp32 [B, T, 4] | 后置门系数，乘回残差流；源码名是 hc_post_magnitude，就是配置里的 hc_magnitude | modeling_hy_v4.py:637 |
| out | 算子 | torch.sum(pre.unsqueeze(-1) * hidden_streams, dim=2).to(hidden_streams.dtype) | fp32 [B, T, 4, 1] 与 bf16 [B, T, 4, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:638, 640 |
| ov | 张量 | out | bf16 [B, T, 6144] | 折叠成单份 | modeling_hy_v4.py:638 |
| hret | 张量 | (post, out) | fp32 [B, T, 4] × bf16 [B, T, 6144] | 两个返回值：后置门系数和折叠后的隐状态 | modeling_hy_v4.py:640 |

### 连线

- hst → flat
- flat → fl
- fl → norm
- norm → nrm
- fl → lin
- lin → lr
- lr → mix
- fl → mix：shape 与 fn
- nrm → mix
- mix → mx
- mx → msplit
- base → bsplit
- bsplit → pb
- bsplit → pob
- scale → ssplit
- ssplit → psc
- ssplit → posc
- msplit → prel
- msplit → postl
- prel → pre
- psc → pre
- pb → pre
- pre → pv
- postl → post
- posc → post
- pob → post
- post → pov
- pv → out
- hst → out
- out → ov
- pov → hret
- ov → hret
- norm 点开 → HYV4UnweightedRMSNorm.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| flat | 公式 | $flat = \mathrm{flatten}(hidden\_streams, 2)$，把 4 份残差流展平成一维，再转 fp32 |
| norm | 公式 | $\mathrm{rsqrt}(\mathrm{mean}(flat^2) + \mathit{eps})$，对每一行整体求，不加权重，eps 是 self.eps，取 rms_norm_eps 1e-05 |
| lin | 公式 | $F.linear(flat, fn)$，fn 是混合权重 |
| lin | 说明 | fn [8, 24576]、base [8]、scale [2]，都在 fp32 里算 |
| mix | 公式 | $mixes = (flat\, fn^\top) \odot \mathrm{rsqrt}(\mathrm{mean}(flat^2) + \mathit{eps})$，逐行归一化后的混合 logits |
| bsplit | 公式 | $pre\_b, post\_b = \mathrm{split}(base, hc\_mult)$ |
| ssplit | 公式 | $pre\_scale, post\_scale = \mathrm{unbind}(scale)$ |
| msplit | 公式 | $pre\_logits, post\_logits = \mathrm{split}(mixes, hc\_mult)$ |
| pre | 公式 | $pre = \sigma(pre\_logits \cdot pre\_scale + pre\_b) + hc\_eps$，4 份残差流的折叠系数 |
| post | 公式 | $post = hc\_post\_magnitude \cdot \sigma(post\_logits \cdot post\_scale + post\_b) + hc\_eps$，加回残差流时的系数，hc_post_magnitude 就是配置里的 hc_magnitude |
| out | 公式 | $out = \sum_c pre_c\, hidden\_streams_c$，按折叠系数把 4 份残差流求和 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 折叠 | $out = \sum_c pre_c\, hidden\_streams_c$，$pre = \sigma(pre\_logits \cdot pre\_scale + pre\_b) + hc\_eps$ |
| 配置 | hc_mult | 4 |
| 配置 | hc_magnitude | 2.0 |
| 配置 | hc_eps | 1e-06 |

## HYV4HyperHead.forward（modeling_hy_v4.py:655-664）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hx | 张量 | x | bf16 [B, T, 4, 6144] | 最后一层输出的 4 份残差流 | modeling_hy_v4.py:659 |
| hflat | 算子 | x.flatten(2).float() | bf16 [B, T, 4, 6144] → fp32 [B, T, 24576] | | modeling_hy_v4.py:659 |
| hhfl | 张量 | flat | fp32 [B, T, 24576] | | modeling_hy_v4.py:659 |
| hnorm | 算子 | self.input_norm(flat) | fp32 [B, T, 24576] → fp32 [B, T, 1] | | modeling_hy_v4.py:661 |
| hnrm | 张量 | input_norm(flat) | fp32 [B, T, 1] | | modeling_hy_v4.py:661 |
| hlin | 算子 | F.linear(flat, self.hc_fn.float()) | fp32 [B, T, 24576] → fp32 [B, T, 4] | | modeling_hy_v4.py:661 |
| hlr | 张量 | F.linear(flat, hc_fn) | fp32 [B, T, 4] | 4 = hc_mult | modeling_hy_v4.py:661 |
| hmix | 算子 | F.linear(flat, self.hc_fn.float()) * self.input_norm(flat) | fp32 [B, T, 4] 与 fp32 [B, T, 1] → fp32 [B, T, 4] | | modeling_hy_v4.py:661 |
| hmx | 张量 | mixes | fp32 [B, T, 4] | | modeling_hy_v4.py:661 |
| hpre | 算子 | torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps | fp32 [B, T, 4] → fp32 [B, T, 4] | | modeling_hy_v4.py:662 |
| hpv | 张量 | pre | fp32 [B, T, 4] | 4 份残差流的折叠系数 | modeling_hy_v4.py:662 |
| hout | 算子 | (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype) | fp32 [B, T, 4, 1] 与 bf16 [B, T, 4, 6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:663-664 |
| hov | 张量 | out | bf16 [B, T, 6144] | | modeling_hy_v4.py:663-664 |
| scl | 张量 | hc_scale | fp32 [1] | 折叠的缩放 | modeling_hy_v4.py:662 |
| bas | 张量 | hc_base | fp32 [4] | 折叠的偏置 | modeling_hy_v4.py:662 |

### 连线

- hx → hflat
- hflat → hhfl
- hhfl → hnorm
- hnorm → hnrm
- hhfl → hlin
- hlin → hlr
- hlr → hmix
- hhfl → hmix：shape 与 hc_fn
- hnrm → hmix
- hmix → hmx
- hmx → hpre
- scl → hpre
- bas → hpre
- hpre → hpv
- hpv → hout
- hx → hout
- hout → hov
- hnorm 点开 → HYV4UnweightedRMSNorm.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| hflat | 公式 | $flat = \mathrm{flatten}(x, 2)$，转 fp32 |
| hnorm | 公式 | $\mathrm{rsqrt}(\mathrm{mean}(flat^2) + \mathit{eps})$，不加权重，eps 是 self.eps，取 rms_norm_eps 1e-05 |
| hlin | 公式 | $F.linear(flat, hc\_fn)$ |
| hlin | 说明 | hc_fn [4, 24576]、hc_base [4]、hc_scale [1]，都在 fp32 里算 |
| hmix | 公式 | $mixes = (flat\, hc\_fn^\top) \odot \mathrm{rsqrt}(\mathrm{mean}(flat^2) + \mathit{eps})$ |
| hpre | 公式 | $pre = \sigma(mixes \cdot hc\_scale + hc\_base) + eps$，4 份残差流的折叠系数，eps 是 hc_eps 1e-06 |
| hout | 公式 | $out = \sum_c pre_c\, x_c$，折叠成单份送进最后的 RMSNorm |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 末尾折叠 | $out = \sum_c pre_c\, x_c$，$pre = \sigma(mixes \cdot hc\_scale + hc\_base) + hc\_eps$ |
| 配置 | hc_mult | 4 |
| 配置 | hc_eps | 1e-06 |

## HYV4RMSNorm.forward（modeling_hy_v4.py:55-60）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hs | 张量 | hidden_states | bf16 [B, T, 6144] | 形状随调用点变：q_a_layernorm 用 [B, T, 2048]、kv_a_layernorm 用 [B, T, 512] | modeling_hy_v4.py:56 |
| dt | 算子 | hidden_states.to(torch.float32) | bf16 [B, T, 6144] → fp32 [B, T, 6144] | | modeling_hy_v4.py:57 |
| rmhf | 张量 | hidden_states | fp32 [B, T, 6144] | | modeling_hy_v4.py:57 |
| var | 算子 | hidden_states.pow(2).mean(-1, keepdim=True) | fp32 [B, T, 6144] → fp32 [B, T, 1] | | modeling_hy_v4.py:58 |
| vv | 张量 | variance | fp32 [B, T, 1] | 最后一维的均方 | modeling_hy_v4.py:58 |
| nrm | 算子 | hidden_states * torch.rsqrt(variance + self.variance_epsilon) | fp32 [B, T, 6144] 与 fp32 [B, T, 1] → fp32 [B, T, 6144] | | modeling_hy_v4.py:59 |
| hv | 张量 | hidden_states | fp32 [B, T, 6144] | | modeling_hy_v4.py:59 |
| back | 算子 | self.weight * hidden_states.to(input_dtype) | fp32 [B, T, 6144] 与 bf16 [6144] → bf16 [B, T, 6144] | | modeling_hy_v4.py:60 |
| out | 张量 | self.weight * hidden_states.to(input_dtype) | bf16 [B, T, 6144] | | modeling_hy_v4.py:60 |

### 连线

- hs → dt：数据转 fp32
- dt → rmhf
- rmhf → var
- var → vv
- rmhf → nrm
- vv → nrm
- nrm → hv
- hv → back
- hs → back：dtype
- back → out

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| dt | 公式 | $hidden\_states = hidden\_states.\mathrm{to}(fp32)$ |
| var | 公式 | $variance = \mathrm{mean}(hidden\_states^2)$，在最后一维求 |
| nrm | 公式 | $hidden\_states = hidden\_states \cdot \mathrm{rsqrt}(variance + variance\_epsilon)$ |
| back | 公式 | $out = weight \odot hidden\_states.\mathrm{to}(bf16)$，weight 是学到的缩放 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | RMS 归一化 | $out = weight \odot hidden\_states / \sqrt{\mathrm{mean}(hidden\_states^2) + \mathit{eps}}$ |
| 配置 | hidden_size | 6144 |
| 配置 | rms_norm_eps | 1e-05（input_layernorm、post_attention_layernorm、model.norm 用配置值；q_a_layernorm、kv_a_layernorm 没传 eps，用代码默认值 1e-06） |
| 配置 | 隐层维 | 本图按 6144 画；q_a_layernorm 那条路径是 2048，kv_a_layernorm 是 512 |

## HYV4UnweightedRMSNorm.forward（modeling_hy_v4.py:71-73）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ux | 张量 | x | fp32 [B, T, 24576] | 展平后的 4 份残差流 | modeling_hy_v4.py:73 |
| f32 | 算子 | x.float() | fp32 [B, T, 24576] → fp32 [B, T, 24576] | | modeling_hy_v4.py:73 |
| xf | 张量 | x | fp32 [B, T, 24576] | | modeling_hy_v4.py:73 |
| sq | 算子 | x.float().square().mean(-1, keepdim=True) | fp32 [B, T, 24576] → fp32 [B, T, 1] | | modeling_hy_v4.py:73 |
| umv | 张量 | mean | fp32 [B, T, 1] | 最后一维的均方 | modeling_hy_v4.py:73 |
| rsqrt | 算子 | torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps) | fp32 [B, T, 1] → fp32 [B, T, 1] | | modeling_hy_v4.py:73 |
| rv | 张量 | rsqrt | fp32 [B, T, 1] | 逆均方根 | modeling_hy_v4.py:73 |
| out | 算子 | torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) | fp32 [B, T, 1] → fp32 [B, T, 1] | | modeling_hy_v4.py:73 |
| uov | 张量 | torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps) | fp32 [B, T, 1] | 不加权重、也不乘回 x，直接返回逆均方根 | modeling_hy_v4.py:73 |

### 连线

- ux → f32
- f32 → xf
- xf → sq
- sq → umv
- umv → rsqrt
- rsqrt → rv
- rv → out
- ux → out：dtype
- out → uov

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| f32 | 公式 | $x = x.\mathrm{float()}$ |
| sq | 公式 | $\mathrm{mean}(x^2)$，在最后一维求 |
| rsqrt | 公式 | $\mathrm{rsqrt}(\mathrm{mean}(x^2) + eps)$ |
| out | 公式 | 结果转回输入 dtype。返回的是逆均方根本身，调用方再把它乘回展平后的残差流（调用方里叫 flat，本图里叫 x） |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 逆均方根 | $out = \mathrm{rsqrt}(\mathrm{mean}(x^2) + \mathit{eps})$，不乘权重 |
| 配置 | rms_norm_eps | 1e-05 |

## HYV4RotaryEmbedding.forward（modeling_hy_v4.py:114-120）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| rex | 张量 | x | bf16 [B, T, 6144] | 只用来取得 dtype 与 device | modeling_hy_v4.py:115 |
| pid | 张量 | position_ids | int64 [1, T] | | modeling_hy_v4.py:115 |
| fr | 算子 | position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float) | int64 [1, T] 与 fp32 [32] → fp32 [1, T, 32] | | modeling_hy_v4.py:115 |
| fq | 张量 | freqs | fp32 [1, T, 32] | 32 = 64 / 2 | modeling_hy_v4.py:115 |
| cat | 算子 | torch.cat((freqs, freqs), dim=-1) | fp32 [1, T, 32] × 2 → fp32 [1, T, 64] | | modeling_hy_v4.py:116 |
| emb | 张量 | emb | fp32 [1, T, 64] | 频率按前后两半复制 | modeling_hy_v4.py:116 |
| cos | 算子 | emb.cos() * self.attention_scaling | fp32 [1, T, 64] → bf16 [1, T, 64]（return 时 .to(dtype=x.dtype)） | | modeling_hy_v4.py:117, 120 |
| cv | 张量 | cos | bf16 [1, T, 64] | | modeling_hy_v4.py:117-120 |
| sin | 算子 | emb.sin() * self.attention_scaling | fp32 [1, T, 64] → bf16 [1, T, 64]（return 时 .to(dtype=x.dtype)） | | modeling_hy_v4.py:118, 120 |
| sv | 张量 | sin | bf16 [1, T, 64] | | modeling_hy_v4.py:118-120 |

### 连线

- pid → fr
- rex → fr：device
- fr → fq
- fq → cat
- cat → emb
- emb → cos
- emb → sin
- cos → cv
- rex → cos：dtype
- sin → sv
- rex → sin：dtype

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| fr | 公式 | $freqs = position\_ids \cdot inv\_freq$，inv_freq 是逆频率，长度 32 = 64 / 2 |
| fr | 说明 | 逆频率按 base 的 −2i/d 次方生成，base 是 rope_theta 10000000，d 就是 head_dim（64），逆频率长度 32 = 64 / 2 |
| cat | 公式 | $emb = \mathrm{cat}(freqs, freqs)$，长度翻倍到 64 |
| cos | 公式 | $cos = \cos(emb) \cdot attention\_scaling$，rope_type 为 default 时 attention_scaling 为 1 |
| sin | 公式 | $sin = \sin(emb) \cdot attention\_scaling$ |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 旋转位置编码 | $emb = \mathrm{cat}(freqs, freqs)$，$cos = \cos(emb)$，$sin = \sin(emb)$，freqs 是 position_ids 乘逆频率 |
| 配置 | rope_theta | 10000000 |
| 配置 | qk_rope_head_dim | 64 |
| 配置 | head_dim | 64（就是 qk_rope_head_dim，逆频率按它生成） |
