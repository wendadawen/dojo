# Qwen3-0.6B

- 描述：Qwen3-0.6B 从 token 得到词嵌入，经 28 层注意力和前馈，再经 RMSNorm 和词表投影得到下一个 token 的 logits。
- 摘要：28 层、hidden 1024；16 个查询头共用 8 组 KV，query_states 和 key_states 先按头做 RMSNorm 再做 RoPE；lm_head 与词表矩阵共用权重。
- 主题：模型结构,注意力机制

## 资料

- 源码：transformers v4.57.6，src/transformers/models/qwen3/modeling_qwen3.py；掩码见 src/transformers/masking_utils.py，RoPE 频率见 src/transformers/modeling_rope_utils.py，缓存见 src/transformers/cache_utils.py
- config.json：Hugging Face Qwen/Qwen3-0.6B，commit c1899de289a04d12100db370d81485cdf75e47ca，副本在 sources/config.json
- 权重文件头：同一 commit 的 model.safetensors，311 个张量的形状在 sources/shapes.json
- 论文：Qwen3 Technical Report，arXiv:2505.09388

## 前提

- 推理，不传 labels，不算 loss
- 按 bfloat16 加载（from_pretrained 时传 dtype=torch.bfloat16）
- 注意力实现按 eager 画（transformers 默认是 sdpa）
- 缓存用 DynamicCache，use_cache 为 true
- 调用方传 input_ids，之后的前向再传回上一次的 past_key_values
- 不传 inputs_embeds、attention_mask、position_ids、cache_position，没有 padding

## config.json

| 组 | 名字 | 值 |
|---|---|---|
| 规模 | vocab_size | 151936 |
| 规模 | hidden_size | 1024 |
| 规模 | intermediate_size | 3072 |
| 规模 | num_hidden_layers | 28 |
| 注意力 | num_attention_heads | 16 |
| 注意力 | num_key_value_heads | 8 |
| 注意力 | head_dim | 128 |
| 注意力 | attention_bias | false |
| 注意力 | use_sliding_window | false |
| 位置编码 | rope_theta | 1000000 |
| 位置编码 | rope_scaling | null |
| 其他 | hidden_act | silu |
| 其他 | rms_norm_eps | 1e-06 |
| 其他 | tie_word_embeddings | true |
| 其他 | use_cache | true |
| 其他 | torch_dtype | bfloat16 |

## Qwen3ForCausalLM.forward（modeling_qwen3.py:445-506）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ids | 张量 | input_ids | int64 [B, T] | | modeling_qwen3.py:447 |
| cache | 缓存 | past_key_values | 每层一份 k、v | 第一次前向在 Qwen3Model 里新建，之后由调用方传回 | modeling_qwen3.py:450，484 |
| model | 算子 | model (Qwen3Model) | [B, T] → [B, T, 1024] | 嵌入、28 层 Decoder 和最后的 RMSNorm 都在里面 | modeling_qwen3.py:480-489 |
| last | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:491 |
| head | 算子 | lm_head(hidden_states[:, slice_indices, :]) | 1024 → 151936 | lm_head.weight [151936, 1024]，没有偏置，与 model.embed_tokens.weight 共用（tie_word_embeddings=true）<br>slice_indices 由 logits_to_keep 决定，默认 0，保留全部位置 | modeling_qwen3.py:493-494 |
| logits | 张量 | logits | bf16 [B, T, 151936] | | modeling_qwen3.py:494 |

### 连线

- ids → model
- cache → model：past_key_values
- model → cache：写入各层 k、v
- model → last
- last → head
- head → logits
- model 点开 → Qwen3Model.forward

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | logits | $\mathrm{logits} = \mathrm{hidden\_states}\,W^\top$，$\top$ 是转置，hidden_states 是 Qwen3Model 的 last_hidden_state（已经过最后的 RMSNorm），W 是词表矩阵 model.embed_tokens.weight |
| 配置 | vocab_size | 151936 |
| 配置 | hidden_size | 1024 |
| 配置 | tie_word_embeddings | true |

## Qwen3Model.forward（modeling_qwen3.py:356-425）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ids | 张量 | input_ids | int64 [B, T] | | modeling_qwen3.py:358 |
| emb | 算子 | embed_tokens (nn.Embedding) | 151936 → 1024 | model.embed_tokens.weight [151936, 1024] | modeling_qwen3.py:370-371 |
| embs | 张量 | inputs_embeds | bf16 [B, T, 1024] | | modeling_qwen3.py:371 |
| cache | 缓存 | past_key_values | 每层一份 k、v | 第一次前向用 DynamicCache(config=self.config) 新建，之后用调用方传回的 | modeling_qwen3.py:373-374 |
| arange | 算子 | torch.arange(past_seen_tokens, past_seen_tokens + T) | → [T] | 本次各 token 的位置 P … P+T-1<br>T = inputs_embeds.shape[1]；P 即 past_seen_tokens，是缓存里已有的长度，第一次前向时为 0 | modeling_qwen3.py:376-380 |
| cpos | 张量 | cache_position | int64 [T] | | modeling_qwen3.py:378 |
| unsq | 算子 | cache_position.unsqueeze(0) | [T] → [1, T] | 补出批这一维 | modeling_qwen3.py:382-383 |
| pos | 张量 | position_ids | int64 [1, T] | | modeling_qwen3.py:383 |
| rope | 算子 | rotary_emb (Qwen3RotaryEmbedding) | [1, T] → [1, T, 128] × 2 | 由 position_ids 算出 cos 和 sin，28 层共用这一份 | modeling_qwen3.py:407 |
| pe | 张量 | position_embeddings (cos, sin) | bf16 [1, T, 128] × 2 | | modeling_qwen3.py:407 |
| mask | 算子 | create_causal_mask(**mask_kwargs) | → [B, 1, T, P+T] | 生成因果掩码：可见位置为 0，不可见位置为 dtype 最小值<br>use_sliding_window 是 false，layer_types 全是 full_attention，所以不建 sliding_attention 掩码<br>position_ids 也会传进去，不参与计算 | modeling_qwen3.py:386-398 |
| am | 张量 | causal_mask_mapping["full_attention"] | bf16 [B, 1, T, P+T] | | modeling_qwen3.py:397-398 |
| layers | 算子 | 28 × Qwen3DecoderLayer | [B, T, 1024] 不变 | 28 层结构相同，layer_types 全是 full_attention；每层读写缓存里自己那一层<br>position_ids 也会传进去，不参与计算；cache_position 只进 cache_kwargs，不起作用 | modeling_qwen3.py:409-419 |
| mid | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:410 |
| norm | 算子 | norm (Qwen3RMSNorm) | [B, T, 1024] 不变 | model.norm.weight [1024] | modeling_qwen3.py:421 |
| last | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:421-423 |

### 连线

- ids → emb
- emb → embs
- cache → arange：get_seq_length()
- embs → arange：shape[1]、device
- arange → cpos
- cpos → unsq
- unsq → pos
- pos → rope
- embs → rope：dtype、device
- rope → pe
- cpos → mask
- cache → mask：get_mask_sizes()
- embs → mask：batch、dtype
- pos → mask：position_ids
- mask → am
- embs → layers
- pe → layers
- am → layers
- pos → layers：position_ids
- cpos → layers：cache_position
- cache → layers：past_key_values
- layers → cache：写入各层 k、v
- layers → mid
- mid → norm
- norm → last
- rope 点开 → Qwen3RotaryEmbedding.forward
- layers 点开 → Qwen3DecoderLayer.forward
- norm 点开 → Qwen3RMSNorm.forward

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 位置 | 本次第 $i$ 个 token 的位置是 $P + i$，i 从 0 数起，P 是缓存里已有的长度 |
| 公式 | 因果掩码 | $M_{ij} = 0$（$j \le P + i$），否则取 dtype 最小值；M 是因果掩码，i 是本次第 i 个 token，j 是拼上缓存后的第 j 个位置，从 0 到 P+T-1 |
| 配置 | vocab_size | 151936 |
| 配置 | hidden_size | 1024 |
| 配置 | num_hidden_layers | 28 |
| 配置 | use_sliding_window | false |
| 配置 | use_cache | true |

## Qwen3DecoderLayer.forward（modeling_qwen3.py:246-277）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| din | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:248 |
| pe | 张量 | position_embeddings (cos, sin) | bf16 [1, T, 128] × 2 | | modeling_qwen3.py:254 |
| am | 张量 | attention_mask | bf16 [B, 1, T, P+T] | P 是缓存里已有的长度 | modeling_qwen3.py:249 |
| pos | 张量 | position_ids | int64 [1, T] | | modeling_qwen3.py:250 |
| cpos | 张量 | cache_position | int64 [T] | | modeling_qwen3.py:253 |
| cache | 缓存 | past_key_values | 本层的 k、v | 在 self_attn 里读写 | modeling_qwen3.py:251 |
| ln1 | 算子 | input_layernorm (Qwen3RMSNorm) | [B, T, 1024] 不变 | model.layers.{i}.input_layernorm.weight [1024] | modeling_qwen3.py:258 |
| h1 | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:258 |
| attn | 算子 | self_attn (Qwen3Attention) | [B, T, 1024] 不变 | 第二个返回值是注意力权重，这里不用<br>position_ids 也会传进去，不参与计算；cache_position 只进 cache_kwargs，不起作用 | modeling_qwen3.py:260-269 |
| a | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:260 |
| add1 | 算子 | residual + hidden_states | [B, T, 1024] 不变 | 残差相加，residual 是进入本层时的 hidden_states | modeling_qwen3.py:257，270 |
| h2 | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:270 |
| ln2 | 算子 | post_attention_layernorm (Qwen3RMSNorm) | [B, T, 1024] 不变 | model.layers.{i}.post_attention_layernorm.weight [1024] | modeling_qwen3.py:274 |
| h3 | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:274 |
| mlp | 算子 | mlp (Qwen3MLP) | 1024 → 3072 → 1024 | gate_proj、up_proj 各把 1024 升到 3072，相乘后 down_proj 降回 1024 | modeling_qwen3.py:275 |
| m | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:275 |
| add2 | 算子 | residual + hidden_states | [B, T, 1024] 不变 | 残差相加，residual 是进入 post_attention_layernorm 之前的 hidden_states | modeling_qwen3.py:273，276 |
| hout | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:276-277 |

### 连线

- din → ln1
- ln1 → h1
- h1 → attn
- pe → attn
- am → attn
- pos → attn
- cpos → attn
- cache → attn：past_key_values
- attn → cache：写入本层 k、v
- attn → a
- a → add1
- din → add1：residual
- add1 → h2
- h2 → ln2
- ln2 → h3
- h3 → mlp
- mlp → m
- m → add2
- h2 → add2：residual
- add2 → hout
- ln1 点开 → Qwen3RMSNorm.forward
- attn 点开 → Qwen3Attention.forward
- ln2 点开 → Qwen3RMSNorm.forward
- mlp 点开 → Qwen3MLP.forward

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 注意力子层 | $\mathrm{hidden\_states} \leftarrow \mathrm{hidden\_states} + \mathrm{Attn}(\mathrm{RMSNorm}(\mathrm{hidden\_states}))$，$\leftarrow$ 表示用右边的结果替换 hidden_states，Attn 是 self_attn，RMSNorm 是 input_layernorm |
| 公式 | 前馈子层 | $\mathrm{hidden\_states} \leftarrow \mathrm{hidden\_states} + \mathrm{MLP}(\mathrm{RMSNorm}(\mathrm{hidden\_states}))$，这是注意力子层之后的 hidden_states，MLP 是 mlp，RMSNorm 是 post_attention_layernorm |
| 配置 | hidden_size | 1024 |
| 配置 | rms_norm_eps | 1e-06 |

## Qwen3Attention.forward（modeling_qwen3.py:188-230）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| h1 | 张量 | hidden_states | bf16 [B, T, 1024] | | modeling_qwen3.py:190 |
| pe | 张量 | position_embeddings (cos, sin) | bf16 [1, T, 128] × 2 | | modeling_qwen3.py:191 |
| am | 张量 | attention_mask | bf16 [B, 1, T, P+T] | P 是缓存里已有的长度，第一次前向时为 0 | modeling_qwen3.py:192 |
| cpos | 张量 | cache_position | int64 [T] | | modeling_qwen3.py:194 |
| pos | 张量 | position_ids | int64 [1, T] | 经 **kwargs 传进来，forward 的形参里没有它 | modeling_qwen3.py:195，263 |
| qp | 算子 | q_proj (nn.Linear) | 1024 → 2048 | model.layers.{i}.self_attn.q_proj.weight [2048, 1024]，16 头 × 128，没有偏置 | modeling_qwen3.py:200 |
| qf | 张量 | q_proj(hidden_states) | bf16 [B, T, 2048] | | modeling_qwen3.py:200 |
| qn | 算子 | q_norm(q_proj(hidden_states).view(hidden_shape)).transpose(1, 2) | 2048 → 16 × 128 | hidden_shape 是 (B, T, -1, 128)，由 hidden_states 的形状得到；按头拆开，每头 128 维各做一次 RMSNorm，再把头换到第 1 维<br>model.layers.{i}.self_attn.q_norm.weight [128] | modeling_qwen3.py:197-198，200 |
| q | 张量 | query_states | bf16 [B, 16, T, 128] | | modeling_qwen3.py:200 |
| kp | 算子 | k_proj (nn.Linear) | 1024 → 1024 | model.layers.{i}.self_attn.k_proj.weight [1024, 1024]，8 头 × 128，没有偏置 | modeling_qwen3.py:201 |
| kf | 张量 | k_proj(hidden_states) | bf16 [B, T, 1024] | | modeling_qwen3.py:201 |
| kn | 算子 | k_norm(k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) | 1024 → 8 × 128 | 和 q_norm 一样按头归一化<br>model.layers.{i}.self_attn.k_norm.weight [128] | modeling_qwen3.py:197-198，201 |
| k | 张量 | key_states | bf16 [B, 8, T, 128] | | modeling_qwen3.py:201 |
| vp | 算子 | v_proj (nn.Linear) | 1024 → 1024 | model.layers.{i}.self_attn.v_proj.weight [1024, 1024]，8 头 × 128，没有偏置 | modeling_qwen3.py:202 |
| vf | 张量 | v_proj(hidden_states) | bf16 [B, T, 1024] | | modeling_qwen3.py:202 |
| vt | 算子 | v_proj(hidden_states).view(hidden_shape).transpose(1, 2) | 1024 → 8 × 128 | v 不做归一化 | modeling_qwen3.py:197-198，202 |
| v | 张量 | value_states | bf16 [B, 8, T, 128] | | modeling_qwen3.py:202 |
| apply | 算子 | apply_rotary_pos_emb(query_states, key_states, cos, sin) | 形状不变 | 对 query_states 的每个头 $q' = q \odot \cos + \mathrm{rotate\_half}(q) \odot \sin$，$q$ 是这个头的向量，$q'$ 是旋转后的结果，$\odot$ 是逐元素相乘，key_states 同理<br>cos、sin 先在第 1 维补一维（unsqueeze(1)），广播到各头 | modeling_qwen3.py:204-205，93-117 |
| qr | 张量 | query_states | bf16 [B, 16, T, 128] | | modeling_qwen3.py:205 |
| kr | 张量 | key_states | bf16 [B, 8, T, 128] | | modeling_qwen3.py:205 |
| cache | 缓存 | past_key_values | 本层的 k、v | past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs) 把本次的 key_states、value_states 追加进去，返回拼接后的完整 key、value<br>cache_kwargs（sin、cos、cache_position）对 DynamicCache 不起作用 | modeling_qwen3.py:207-210 |
| kc | 张量 | key_states | bf16 [B, 8, P+T, 128] | | modeling_qwen3.py:210 |
| vc | 张量 | value_states | bf16 [B, 8, P+T, 128] | | modeling_qwen3.py:210 |
| core | 算子 | attention_interface | q [B, 16, T, 128]，k、v [B, 8, P+T, 128] → [B, T, 16, 128] | 即 eager_attention_forward：repeat_kv 把 8 个 key/value 头复制成 16 个，softmax 在 fp32 里算，推理时 dropout 的 p = 0<br>position_ids 从 **kwargs 传进来，不参与计算<br>第二个返回值 attn_weights 是注意力权重，原样返回，本图不画 | modeling_qwen3.py:212-226，132-155，230 |
| ao | 张量 | attn_output | bf16 [B, T, 16, 128] | | modeling_qwen3.py:216 |
| rs | 算子 | attn_output.reshape(*input_shape, -1) | 16 × 128 → 2048 | input_shape 是 hidden_states 的前两维 (B, T) | modeling_qwen3.py:197，228 |
| af | 张量 | attn_output | bf16 [B, T, 2048] | | modeling_qwen3.py:228 |
| op | 算子 | o_proj (nn.Linear) | 2048 → 1024 | model.layers.{i}.self_attn.o_proj.weight [1024, 2048]，没有偏置 | modeling_qwen3.py:229 |
| a | 张量 | attn_output | bf16 [B, T, 1024] | | modeling_qwen3.py:229，260 |

### 连线

- h1 → qp
- qp → qf
- qf → qn
- h1 → qn：shape
- qn → q
- h1 → kp
- kp → kf
- kf → kn
- h1 → kn：shape
- kn → k
- h1 → vp
- vp → vf
- vf → vt
- h1 → vt：shape
- vt → v
- q → apply
- k → apply
- pe → apply
- apply → qr
- apply → kr
- kr → cache
- v → cache
- cpos → cache：cache_kwargs
- pe → cache：cache_kwargs
- cache → kc
- cache → vc
- qr → core
- kc → core
- vc → core
- am → core
- pos → core：**kwargs
- core → ao
- ao → rs
- h1 → rs：shape
- rs → af
- af → op
- op → a
- qn 点开 → Qwen3RMSNorm.forward
- kn 点开 → Qwen3RMSNorm.forward

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 注意力 | $\mathrm{softmax}(q k^\top / \sqrt{128} + M)\,v$，$\top$ 是转置，q、k、v 是 query_states、key_states、value_states，q 和 k 已做过 RoPE，k、v 已拼上缓存、8 个头各复制一份成 16 个，M 是 attention_mask，128 是 head_dim |
| 公式 | RoPE | 对 query_states 的每个头 $q' = q \odot \cos + \mathrm{rotate\_half}(q) \odot \sin$，$q$ 是这个头的向量，$q'$ 是旋转后的结果，$\odot$ 是逐元素相乘，key_states 同理；cos、sin 来自 position_embeddings |
| 公式 | rotate_half | $[q_1, q_2] \mapsto [-q_2, q_1]$，$\mapsto$ 表示映射为，$q_1$、$q_2$ 是一个头的前 64 维和后 64 维 |
| 公式 | 分组查询 | $n_{\mathrm{rep}} = 16 / 8 = 2$，$n_{\mathrm{rep}}$ 是每个 KV 头对应的查询头数，16 是 num_attention_heads，8 是 num_key_value_heads |
| 配置 | num_attention_heads | 16 |
| 配置 | num_key_value_heads | 8 |
| 配置 | head_dim | 128 |
| 配置 | attention_bias | false |
| 配置 | rms_norm_eps | 1e-06 |

## Qwen3MLP.forward（modeling_qwen3.py:81-83）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| h3 | 张量 | x | bf16 [B, T, 1024] | | modeling_qwen3.py:81 |
| gp | 算子 | gate_proj (nn.Linear) | 1024 → 3072 | model.layers.{i}.mlp.gate_proj.weight [3072, 1024]，没有偏置 | modeling_qwen3.py:82，76 |
| g | 张量 | gate_proj(x) | bf16 [B, T, 3072] | | modeling_qwen3.py:82 |
| act | 算子 | act_fn | 形状不变 | hidden_act 是 silu：$\mathrm{silu}(z) = z \cdot \sigma(z)$，z 是输入，$\sigma$ 是 sigmoid | modeling_qwen3.py:82，79 |
| ga | 张量 | act_fn(gate_proj(x)) | bf16 [B, T, 3072] | | modeling_qwen3.py:82 |
| up | 算子 | up_proj (nn.Linear) | 1024 → 3072 | model.layers.{i}.mlp.up_proj.weight [3072, 1024]，没有偏置 | modeling_qwen3.py:82，77 |
| u | 张量 | up_proj(x) | bf16 [B, T, 3072] | | modeling_qwen3.py:82 |
| mul | 算子 | act_fn(gate_proj(x)) * up_proj(x) | 形状不变 | 逐元素相乘 | modeling_qwen3.py:82 |
| hm | 张量 | act_fn(gate_proj(x)) * up_proj(x) | bf16 [B, T, 3072] | | modeling_qwen3.py:82 |
| dp | 算子 | down_proj (nn.Linear) | 3072 → 1024 | model.layers.{i}.mlp.down_proj.weight [1024, 3072]，没有偏置 | modeling_qwen3.py:82，78 |
| m | 张量 | down_proj | bf16 [B, T, 1024] | | modeling_qwen3.py:82-83 |

### 连线

- h3 → gp
- gp → g
- g → act
- act → ga
- h3 → up
- up → u
- ga → mul
- u → mul
- mul → hm
- hm → dp
- dp → m

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | MLP | $\mathrm{down}(\mathrm{silu}(\mathrm{gate}(x)) \odot \mathrm{up}(x))$，x 是输入，gate、up、down 是三个没有偏置的线性层，$\odot$ 是逐元素相乘 |
| 公式 | silu | $\mathrm{silu}(z) = z \cdot \sigma(z)$，z 是输入，$\sigma$ 是 sigmoid 函数 |
| 配置 | hidden_size | 1024 |
| 配置 | intermediate_size | 3072 |
| 配置 | hidden_act | silu |

## Qwen3RMSNorm.forward（modeling_qwen3.py:59-64）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| nin | 张量 | hidden_states | bf16 [..., d] | d 是被归一化的最后一维：norm、input_layernorm、post_attention_layernorm 里是 1024，q_norm、k_norm 里是 128 | modeling_qwen3.py:59 |
| f32 | 算子 | hidden_states.to(torch.float32) | 形状不变 | 先记下输入的 dtype（input_dtype），最后转回去 | modeling_qwen3.py:60-61 |
| xf | 张量 | hidden_states | fp32 [..., d] | | modeling_qwen3.py:61 |
| var | 算子 | hidden_states.pow(2).mean(-1, keepdim=True) | d → 1 | 最后一维上的平方均值 | modeling_qwen3.py:62 |
| vr | 张量 | variance | fp32 [..., 1] | | modeling_qwen3.py:62 |
| nr | 算子 | hidden_states * torch.rsqrt(variance + self.variance_epsilon) | 形状不变 | rsqrt 是平方根的倒数；variance_epsilon 即 rms_norm_eps = 1e-06 | modeling_qwen3.py:63 |
| xn | 张量 | hidden_states | fp32 [..., d] | | modeling_qwen3.py:63 |
| back | 算子 | hidden_states.to(input_dtype) | 形状不变 | 转回输入的 dtype | modeling_qwen3.py:64 |
| xb | 张量 | hidden_states.to(input_dtype) | bf16 [..., d] | | modeling_qwen3.py:64 |
| mw | 算子 | self.weight * hidden_states.to(input_dtype) | 形状不变 | weight [d]，每个实例一份，如 model.norm.weight [1024]、model.layers.{i}.self_attn.q_norm.weight [128] | modeling_qwen3.py:64，56 |
| nout | 张量 | self.weight * hidden_states.to(input_dtype) | bf16 [..., d] | | modeling_qwen3.py:64 |

### 连线

- nin → f32
- f32 → xf
- xf → var
- var → vr
- xf → nr
- vr → nr
- nr → xn
- xn → back
- nin → back：dtype
- back → xb
- xb → mw
- mw → nout

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | RMSNorm | $y = w \odot h / \sqrt{\mathrm{mean}(h^2) + \epsilon}$，h 是 hidden_states，y 是返回值，mean 在最后一维上取，$\epsilon$ 是 rms_norm_eps，w 是 weight，$\odot$ 是逐元素相乘 |
| 配置 | rms_norm_eps | 1e-06 |
| 配置 | hidden_size | 1024 |
| 配置 | head_dim | 128 |

## Qwen3RotaryEmbedding.forward（modeling_qwen3.py:321-332）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| rx | 张量 | x | bf16 [B, T, 1024] | 只用它的 device 和 dtype | modeling_qwen3.py:321 |
| pos | 张量 | position_ids | int64 [1, T] | | modeling_qwen3.py:321 |
| inv | 张量 | self.inv_freq | fp32 [64] | 初始化时按默认 RoPE 算成 fp32：$\theta_j = (10^6)^{-2j/128}$，j = 0 … 63，$10^6$ 是 rope_theta，128 是 dim<br>dim = head_dim × partial_rotary_factor = 128，partial_rotary_factor 默认 1.0，config.json 里没有<br>register_buffer 注册，persistent=False，不进权重文件。初始化时用 torch.float 显式算成 fp32，所以不随 from_pretrained 的 dtype 变成 bf16；forward 里的 .float() 不改变 dtype<br>rope_scaling 是 null，dynamic_rope_update 不起作用 | modeling_qwen3.py:315-316；modeling_rope_utils.py:131 |
| ie | 算子 | self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) | [64] → [1, 64, 1] | 第 0 维取 position_ids 的批大小，这里是 1 | modeling_qwen3.py:322 |
| iet | 张量 | inv_freq_expanded | fp32 [1, 64, 1] | | modeling_qwen3.py:322 |
| pexp | 算子 | position_ids[:, None, :].float() | [1, T] → [1, 1, T] | 转成 fp32 | modeling_qwen3.py:323 |
| pet | 张量 | position_ids_expanded | fp32 [1, 1, T] | | modeling_qwen3.py:323 |
| mm | 算子 | (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) | [1, 64, 1] @ [1, 1, T] → [1, T, 64] | 每个位置乘每个频率；关掉 autocast，保证在 fp32 里算，autocast 的 device_type 取自 x 的 device | modeling_qwen3.py:325-327 |
| fr | 张量 | freqs | fp32 [1, T, 64] | | modeling_qwen3.py:327 |
| cat | 算子 | torch.cat((freqs, freqs), dim=-1) | 64 → 128 | 64 个角度（$p\,\theta_j$）拼两遍，前后两半对应 rotate_half 的两半 | modeling_qwen3.py:328 |
| embt | 张量 | emb | fp32 [1, T, 128] | | modeling_qwen3.py:328 |
| cs | 算子 | emb.cos() * self.attention_scaling，emb.sin() * self.attention_scaling | 形状不变 | 默认 RoPE 的 attention_scaling 是 1.0 | modeling_qwen3.py:329-330 |
| csf | 张量 | cos, sin | fp32 [1, T, 128] × 2 | | modeling_qwen3.py:329-330 |
| tod | 算子 | cos.to(dtype=x.dtype), sin.to(dtype=x.dtype) | 形状不变 | 转成 x 的 dtype | modeling_qwen3.py:332 |
| pe | 张量 | (cos, sin) | bf16 [1, T, 128] × 2 | | modeling_qwen3.py:332 |

### 连线

- inv → ie
- pos → ie：shape[0]
- rx → ie：device
- ie → iet
- pos → pexp
- pexp → pet
- iet → mm
- pet → mm
- rx → mm：device
- mm → fr
- fr → cat
- cat → embt
- embt → cs
- cs → csf
- csf → tod
- rx → tod：dtype
- tod → pe

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 频率 | $\theta_j = (10^6)^{-2j/128}$，j = 0 … 63，$10^6$ 是 rope_theta，128 是 dim = head_dim × partial_rotary_factor |
| 公式 | 角度 | $p\,\theta_j$，p 是 token 的位置，即 position_ids 里的值 |
| 公式 | cos 和 sin | $\cos(p\,\theta_j)$、$\sin(p\,\theta_j)$，64 个角度拼两遍得到 128 维 |
| 配置 | rope_theta | 1000000 |
| 配置 | head_dim | 128 |
| 配置 | partial_rotary_factor | 1.0（默认值，config.json 里没有） |
| 配置 | rope_scaling | null |
