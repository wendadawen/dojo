# DeepSeek-V4.1-Flash

- 描述：DeepSeek-V4.1-Flash 一次 prefill 前向的数据流：文本 token 先查出每个位置的 n-gram 哈希行号，再取词嵌入，进 40 层主干；压缩比大于 0 的层在 128 个位置的滑动窗口之外，另加索引器选出的压缩 KV；前馈是 384 选 6 的 MoE；最后折叠、归一化并投影成下一个 token 的 logits。
- 摘要：40 层主干；压缩 KV 只由 4 个 source 层（2、8、14、20）产出并按层序共享；窗口 128 个位置另加索引器选出的 Top-512 压缩条目；窗口 KV 与压缩 KV 都按 FP8、FP4 精度量化再反量化之后才参与注意力；MoE 384 个路由专家取 top-6 加 1 个共享专家，路由专家存 fp4、共享专家存 fp8；残差流按 4 份复制，混合系数由 Sinkhorn 迭代 20 轮给出；Engram 插在层 1、14。
- 主题：模型结构,注意力机制

## 资料

- 源码：Hugging Face 仓库 `deepseek-ai/DeepSeek-V4.1-Flash`，修订 `dba1be0a40aa45a94ad051997016db3960a90277`（2026-09-10）的 `inference/` 参考实现，用到的文件是 `model.py`（1309 行）、`kernel.py`（591 行）、`engram.py`（184 行）
- transformers v4.57.6 只带 `deepseek_v2` 与 `deepseek_v3`，没有这个模型，故按官方仓库源码画
- config.json：`inference/config.json`，副本在 `sources/config.json`，字段名与 `ModelArgs` 一一对应
- 权重文件头：同一修订下 `model-00001-of-00048.safetensors` 到 `model-00048-of-00048.safetensors` 共 48 个分片，按 HTTP Range 只取头部解析，张量的形状表在 `sources/shapes.json`；层号、块号、专家号相同的张量形状一致时合并成 `<名字>.{i}` 一条，Engram 两张词表形状不同，按具体层号列出
- 论文或技术报告：同一修订下的 `DeepSeek_V41_Tech_Report.pdf`

## 前提

- 这一次前向是 `Transformer.forward` 的一次调用：推理，不算 loss
- 这一次是 prefill：`start_pos` 为 0，一次喂进 T 个 token，T 不小于 1024，且能被各层非零的 `compress_ratio` 整除
- 文本输入：`images` 和 `token_types` 都不传，于是 `image_mask` 与 `engram_mask` 都是 None
- 单卡，`world_size` 为 1
- 权重按 `config.json` 的 `dtype` 与 `expert_dtype` 加载：用 `Linear`、`ColumnParallelLinear`、`RowParallelLinear` 建、没写 dtype 的权重取 `dtype`（fp8），路由专家取 `expert_dtype`（fp4），压缩器在 `compress_ratio` 大于 1 时把 `wkv`、`wgate` 提升到 fp32；源码里显式指定了别的精度的按源码，如 wo_a、ParallelHead 的 weight、attn_sink、hc_* 六组参数、索引器的 wk.weight 与 weights_proj.weight、压缩比不大于 1 的压缩器的 wkv
- 注意力、压缩器与索引器走的是 compress_ratio 为 2 这条分支：source 层 2、8、14 的结构相同；层 20 的 compress_ratio 是 1，压缩器只有一次投影
- 注意力只有 `sparse_attn` 一个 kernel，窗口 KV 与压缩 KV 拼成一份 KV 之后一次算完
- 调用的是 `Transformer.forward`，不是 `forward_spec`（MTP/DSpark 草稿路径）
- 缓存按 `max_batch_size` 分配，`config.json` 里没有这一项，取 `ModelArgs` 默认值 4；它是缓存容量，与本次前向的批大小 B 不是同一个量

## config.json

| 组 | 名字 | 值 |
|---|---|---|
| 规模 | vocab_size | 129280 |
| 规模 | dim | 5120 |
| 规模 | n_layers | 40 |
| 规模 | n_heads | 64 |
| 规模 | head_dim | 512 |
| 规模 | rope_head_dim | 64 |
| 规模 | norm_eps | 1e-20 |
| 注意力 | q_lora_rank | 1280 |
| 注意力 | o_lora_rank | 1024 |
| 注意力 | o_groups | 8 |
| 注意力 | window_size | 128 |
| 注意力 | compress_ratios | 前 40 项是主干：层 0、1 为 0，层 2–19 为 2，层 20–39 为 1；末 3 项是 MTP 层，为 0 |
| 注意力 | kv_source_layers | [2, 8, 14, 20] |
| 注意力 | index_source_layers | [2, 8, 14, 20, 24, 28, 32, 36] |
| 注意力 | compress_rope_theta | 160000 |
| 位置编码 | rope_theta | 10000 |
| 位置编码 | rope_factor | 16 |
| 位置编码 | original_seq_len | 65536 |
| 位置编码 | beta_fast | 32 |
| 位置编码 | beta_slow | 1 |
| 索引器 | index_n_heads | 32 |
| 索引器 | index_head_dim | 128 |
| 索引器 | index_topk | 512 |
| 索引器 | candidate_source_layer | 20 |
| 索引器 | candidate_topk_blocks | 2048 |
| 索引器 | candidate_block_size | 8 |
| Hyper-Connections | hc_mult | 4 |
| Hyper-Connections | hc_sinkhorn_iters | 20 |
| Hyper-Connections | hc_eps | 1e-06 |
| MoE | n_routed_experts | 384 |
| MoE | n_shared_experts | 1 |
| MoE | n_activated_experts | 6 |
| MoE | moe_inter_dim | 2304 |
| MoE | score_func | sqrtsoftplus |
| MoE | route_scale | 1.5 |
| MoE | norm_topk_prob | true（默认值，config.json 里没有这一项） |
| MoE | swiglu_limit | 10.0 |
| Engram | engram_layer_ids | [1, 14] |
| Engram | engram_max_ngram_size | 4 |
| Engram | engram_n_heads | 8 |
| Engram | engram_head_dim | 256 |
| Engram | engram_vocab_size | 16000000 |
| Engram | engram_num_embeddings | [384006168, 384016682] |
| Engram | engram_compressed_vocab_size | 99092 |
| Engram | engram_pad_id | 2 |
| DSpark | dspark_target_layer_ids | [37, 38, 39] |
| 量化 | dtype | fp8 |
| 量化 | expert_dtype | fp4 |

## Transformer.forward（model.py:1242-1272）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ids | 张量 | input_ids | int64 [B, T] | 每个位置都是真实 token，填充不会传进来 | model.py:1243 |
| ngram | 算子 | self.engram_hash(input_ids, start_pos, engram_mask) | [B, T] → [B, T, 2, 24] |  | model.py:1252 |
| hash | 张量 | engram_hashes | int64 [B, T, 2, 24] | | model.py:1252 |
| emb | 算子 | self.embed(input_ids) | 129280 → 5120 |  | model.py:1253 |
| h0 | 张量 | h | bf16 [B, T, 5120] | | model.py:1253 |
| hcexp | 算子 | h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1) | [B, T, 5120] → [B, T, 4, 5120] |  | model.py:1258 |
| hc | 张量 | h | bf16 [B, T, 4, 5120] | | model.py:1258 |
| eng | 算子 | layer.engram(h, engram_hashes[:, :,<br>layer.engram.layer_hash_index, :], engram_mask) | [B, T, 4, 5120] 不变 |  | model.py:1262-1263 |
| pm0 | 算子 | make_identity_pre_mix(h, self.hc_mult) | [B, T, 4, 5120] → fp32 [B, T, 4] |  | model.py:1260 |
| pre | 张量 | pre_mix | fp32 [B, T, 4] | 进入第 0 层时的初值，此后每层返回的 ffn_pre 都覆盖它 | model.py:1260 |
| layers | 算子 | 40 × Block | [B, T, 4, 5120] 不变 |  | model.py:1204，1261-1267 |
| hl | 张量 | h | bf16 [B, T, 4, 5120] | | model.py:1267 |
| avg | 算子 | h.mean(dim=2) | [B, T, 4, 5120] → [B, T, 5120] |  | model.py:1265-1266 |
| mh | 张量 | main_hidden | bf16 [B, T, 15360] | 由 main_hiddens 里 3 个目标层各 [B, T, 5120] 沿最后一维拼接而来，作为第三个返回值交给调用方 | model.py:1259，1266，1271 |
| col | 算子 | layer.hc_pre(h, pre_mix) | [B, T, 4, 5120] → [B, T, 5120] |  | model.py:1268 |
| hc2 | 张量 | h | bf16 [B, T, 5120] | | model.py:1268 |
| fnorm | 算子 | self.norm (RMSNorm) | [B, T, 5120] 不变 |  | model.py:1269 |
| hn | 张量 | h | bf16 [B, T, 5120] | | model.py:1269 |
| head | 算子 | self.head (ParallelHead) | 5120 → 129280 |  | model.py:1269 |
| logits | 张量 | logits | fp32 [B, 129280] | | model.py:1269 |
| samp | 算子 | sample(logits, self.temperature) | fp32 [B, 129280] → int64 [B] |  | model.py:1270 |
| oids | 张量 | output_ids | int64 [B] | | model.py:1270 |
| shar | 缓存 | shared_attn | compress_kv、index_k、topk_idxs | 各层之间传递的共享量，每个前向里由靠前的层写、靠后的层读，不重置 | model.py:1166-1180 |
| tret | 张量 | (output_ids, logits, main_hidden) | — | 三个返回值：采样出的 id、logits、拼接后的 main_hidden | model.py:1272 |

### 连线

- ids → emb
- emb → h0
- ids → ngram
- ngram → hash
- h0 → hcexp
- hcexp → hc
- hc → pm0：形状与 device
- pm0 → pre
- hash → eng
- layers → eng：上一层 Block 的输出
- eng → layers：层 1、14 的输入
- hc → layers：层 0 的输入
- pre → layers
- shar → layers
- layers → shar
- layers → hl
- layers → avg：第 37、38、39 层的输入
- avg → mh
- hl → col
- layers → col：最后一层返回的 pre_mix
- col → hc2
- hc2 → fnorm
- fnorm → hn
- hn → head
- head → logits
- logits → samp
- samp → oids
- oids → tret
- logits → tret
- mh → tret
- ngram 点开 → NgramHashState.forward
- eng 点开 → Engram.forward
- layers 点开 → Block.forward
- fnorm 点开 → RMSNorm.forward
- head 点开 → ParallelHead.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| ngram | 说明 | NgramHashState，把每个位置结尾的 n-gram 哈希成表行号<br>倒数第二维是 Engram 层序号，对应 engram_layer_ids 的 [1, 14]；24 = (4 − 1) × 8 |
| emb | 公式 | $h = embed.weight[input\_ids]$，input_ids 是词 id，embed.weight 是词表矩阵。单卡 world_size 为 1，按行查 |
| emb | 说明 | ParallelEmbedding：embed.weight [129280, 5120]，按 rank 沿词表维切分，单卡即整张 |
| hcexp | 公式 | $h$ 沿新的一维复制 $hc\_mult$ 份，每一份等于原来的 $h$ |
| hcexp | 说明 | 把残差流复制成 hc_mult 份，供 Hyper-Connections 用 |
| eng | 说明 | Engram：层 1、14 在进入本层之前跑，吃的是上一层 Block 的输出，把 n-gram 查表的结果按门控加进残差流 |
| pm0 | 公式 | $pre\_mix_{:,:,0} = 1$，其余为 0。h 用来取 batch 和序列长度，hc_mult 是份数 |
| pm0 | 说明 | 第 0 份置 1、其余置 0，作为进入第 0 层时的 pre_mix |
| layers | 说明 | 层 0 的输入是 hc；层 1、14 在进本层之前先跑 Engram 改写残差流；其余层的输入是上一层 Block 的输出<br>压缩 KV 只在 4 个 source 层（2、8、14、20）算一次，其余 compress_ratio 大于 0 的层从 shared_attn 读。层 0、1 的 compress_ratio 为 0，只有滑动窗口 |
| avg | 公式 | $\mathrm{mean}(h, \dim=2)$，h 是进入目标层之前的残差流，求和的那一维是 hc 份 |
| avg | 说明 | 进入第 37、38、39 层之前的残差流，按 hc 维取均值 |
| col | 公式 | $h = \sum_{c=0}^{hc\_mult-1} pre\_mix_c\, h_c$，h_c 是第 c 份残差，pre_mix 是最后一层返回的系数 |
| col | 说明 | 用最后一层返回的 pre_mix 把 4 份折叠成一份 |
| fnorm | 公式 | $h = norm.weight \odot h / \sqrt{\mathrm{mean}(h^2) + norm.eps}$，h 是送进来的张量，norm.weight 是权重，norm.eps 是 eps，mean 在最后一维 |
| fnorm | 说明 | norm.weight [5120] |
| head | 公式 | $logits = h_{:,-1}\, head.weight^\top$，h 是送进 head 的隐状态，head.weight 是词表矩阵，只取最后一个位置 |
| head | 说明 | head.weight [129280, 5120]，只取最后一个位置 |
| samp | 公式 | $logits = logits / temperature$，$probs = \mathrm{softmax}(logits)$，$output\_ids = \mathrm{argmax}(probs / \mathrm{exponential})$。temperature 是 self.temperature，exponential 是同形状的指数分布随机数 |
| samp | 说明 | Gumbel-max：概率除以指数分布随机数后取 argmax，等价于按概率采样 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 介绍 | 模型 | DeepSeek-V4.1-Flash 是一个文本解码模型。一次前向从 token 算到下一个 token 的 logits。主干 40 层。残差流复制成 4 份，每层用 Hyper-Connections 在注意力和前馈前后混合这 4 份。注意力除了最近 128 个位置的窗口，还用索引器从压缩 KV 里再取最多 512 条。前馈从 384 个路由专家里取 6 个，再加 1 个共享专家。Engram 加在第 1 层和第 14 层进入之前。 |
| 公式 | 输入嵌入 | $h = embed.weight[input\_ids]$，input_ids 是词 id，embed.weight 是词表矩阵。单卡按行查 |
| 公式 | 残差流复制 | $h$ 沿新的一维复制 $hc\_mult$ 份，每一份等于原来的 $h$，hc_mult 是 4 |
| 公式 | 输出 | $logits = h_{:,-1}\, head.weight^\top$，h 是送进 head 的隐状态，head.weight 是词表矩阵，只取最后一个位置 |
| 配置 | vocab_size | 129280 |
| 配置 | dim | 5120 |
| 配置 | n_layers | 40 |
| 配置 | hc_mult | 4 |
| 配置 | dspark_target_layer_ids | [37, 38, 39] |
| 配置 | engram_layer_ids | [1, 14] |

## Block.forward（model.py:968-994）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hin | 张量 | x | bf16 [B, T, 4, 5120] | 进入本层的残差流，4 份 | model.py:970 |
| bpre | 张量 | pre_mix | fp32 [B, T, 4] | 上一层返回的 pre_mix，把 4 份折叠成一份时用 | model.py:972 |
| resid0 | 张量 | residual | bf16 [B, T, 4, 5120] | 进注意力子层之前的残差流 | model.py:981 |
| mixes_a | 算子 | self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base) | [B, T, 4, 5120] → 三组 fp32 系数 |  | model.py:982 |
| apre | 张量 | attn_pre | fp32 [B, T, 4] | | model.py:982 |
| apost | 张量 | attn_post | fp32 [B, T, 4] | | model.py:982 |
| acomb | 张量 | attn_comb | fp32 [B, T, 4, 4] | | model.py:982 |
| cpre | 算子 | self.hc_pre(x, pre_mix) | [B, T, 4, 5120] → [B, T, 5120] |  | model.py:983 |
| x1 | 张量 | x | bf16 [B, T, 5120] | | model.py:983 |
| an | 算子 | self.attn_norm (RMSNorm) | [B, T, 5120] 不变 |  | model.py:984 |
| x2 | 张量 | x | bf16 [B, T, 5120] | | model.py:984 |
| attn | 算子 | self.attn(x, start_pos, *attn_args) | [B, T, 5120] 不变 |  | model.py:985 |
| x3 | 张量 | x | bf16 [B, T, 5120] | | model.py:985 |
| cpost | 算子 | self.hc_post(x, residual, attn_post, attn_comb) | [B, T, 5120] → [B, T, 4, 5120] |  | model.py:986 |
| x4 | 张量 | x | bf16 [B, T, 4, 5120] | | model.py:986 |
| resid1 | 张量 | residual | bf16 [B, T, 4, 5120] | 进前馈子层之前的残差流 | model.py:988 |
| mixes_f | 算子 | self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base) | [B, T, 4, 5120] → 三组 fp32 系数 |  | model.py:989 |
| fpre | 张量 | ffn_pre | fp32 [B, T, 4] | 作为返回值交给下一层当 pre_mix | model.py:989 |
| fpost | 张量 | ffn_post | fp32 [B, T, 4] | | model.py:989 |
| fcomb | 张量 | ffn_comb | fp32 [B, T, 4, 4] | | model.py:989 |
| cpre2 | 算子 | self.hc_pre(x, attn_pre) | [B, T, 4, 5120] → [B, T, 5120] |  | model.py:990 |
| x5 | 张量 | x | bf16 [B, T, 5120] | | model.py:990 |
| fn | 算子 | self.ffn_norm (RMSNorm) | [B, T, 5120] 不变 |  | model.py:991 |
| x6 | 张量 | x | bf16 [B, T, 5120] | | model.py:991 |
| moe_op | 算子 | self.ffn(x, image_mask) | [B, T, 5120] 不变 |  | model.py:992 |
| x7 | 张量 | x | bf16 [B, T, 5120] | | model.py:992 |
| cpost2 | 算子 | self.hc_post(x, residual, ffn_post, ffn_comb) | [B, T, 5120] → [B, T, 4, 5120] |  | model.py:993 |
| hout | 张量 | x | bf16 [B, T, 4, 5120] | | model.py:993 |
| bret | 张量 | (x, ffn_pre) | — | 第二个返回值是下一层的 pre_mix | model.py:994 |

### 连线

- hin → resid0
- hin → mixes_a
- mixes_a → apre
- mixes_a → apost
- mixes_a → acomb
- hin → cpre
- bpre → cpre
- cpre → x1
- x1 → an
- an → x2
- x2 → attn
- attn → x3
- x3 → cpost
- resid0 → cpost
- apost → cpost
- acomb → cpost
- cpost → x4
- x4 → resid1
- x4 → mixes_f
- mixes_f → fpre
- mixes_f → fpost
- mixes_f → fcomb
- x4 → cpre2
- apre → cpre2
- cpre2 → x5
- x5 → fn
- fn → x6
- x6 → moe_op
- moe_op → x7
- x7 → cpost2
- resid1 → cpost2
- fpost → cpost2
- fcomb → cpost2
- cpost2 → hout
- hout → bret
- fpre → bret
- an 点开 → RMSNorm.forward
- attn 点开 → Attention.forward
- fn 点开 → RMSNorm.forward
- moe_op 点开 → MoE.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| mixes_a | 公式 | $mixes = \mathrm{flatten}(x)\, hc\_attn\_fn^\top \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x)^2) + norm\_eps)$<br>$pre_j = \mathrm{sigmoid}(mixes_j \cdot hc\_attn\_scale_0 + hc\_attn\_base_j) + hc\_eps$<br>$post_j = 2\,\mathrm{sigmoid}(mixes_{j+hc\_mult} \cdot hc\_attn\_scale_1 + hc\_attn\_base_{j+hc\_mult})$<br>$comb_{j,k} = \mathrm{Sinkhorn}(\mathrm{softmax}(mixes_{j \cdot hc\_mult + k + 2 hc\_mult} \cdot hc\_attn\_scale_2 + hc\_attn\_base_{j \cdot hc\_mult + k + 2 hc\_mult}) + hc\_eps)$<br>x 是残差流，hc\_attn\_fn、hc\_attn\_scale、hc\_attn\_base 是这一调用的参数，norm_eps 与 hc_eps 是 Block 上的 eps。j、k 从 0 到 hc_mult−1。mixes 与 base 按顺序切成三段：前 hc_mult 个给 pre，接着 hc_mult 个给 post，最后 hc_mult × hc_mult 个给 comb。Sinkhorn 先做一次行归一化（softmax 后加 hc_eps），再做一次列归一化，然后交替做行、列各 19 次，共行 20 次、列 20 次，最后一次落在列方向 |
| mixes_a | 说明 | 一次投影同时给出 pre、post、comb 三套系数<br>hc_attn_fn [24, 20480]、hc_attn_base [24]、hc_attn_scale [3]，都是 fp32；24 = (2 + 4) × 4，20480 = 4 × 5120 |
| cpre | 公式 | $x = \sum_{c=0}^{hc\_mult-1} pre\_mix_c\, x_c$，x_c 是第 c 份残差，pre_mix 是上一层返回的系数 |
| cpre | 说明 | 按 pre_mix 把 4 份加权求和成 1 份 |
| an | 公式 | $x = attn\_norm.weight \odot x / \sqrt{\mathrm{mean}(x^2) + attn\_norm.eps}$，x 是送进来的张量，attn\_norm.weight 是权重，attn\_norm.eps 是 eps，mean 在最后一维 |
| an | 说明 | attn_norm.weight [5120] |
| attn | 说明 | Attention，本层的窗口 KV 与压缩 KV 都在里面读写；本路径 attn_args 为空 |
| cpost | 公式 | $x_c = attn\_post_c\, x + \sum_{c'} attn\_comb_{c',c}\, residual_{c'}$，右边的 x 是注意力的输出，residual 是进注意力前的残差，attn_post 与 attn_comb 是 hc_mixes 的系数。c' 是 residual 的份号，c 是写回的份号 |
| cpost | 说明 | 把子层输出按 post 展开成 4 份，再把 residual 按 comb 混进去 |
| mixes_f | 公式 | $mixes = \mathrm{flatten}(x)\, hc\_ffn\_fn^\top \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x)^2) + norm\_eps)$<br>$pre_j = \mathrm{sigmoid}(mixes_j \cdot hc\_ffn\_scale_0 + hc\_ffn\_base_j) + hc\_eps$<br>$post_j = 2\,\mathrm{sigmoid}(mixes_{j+hc\_mult} \cdot hc\_ffn\_scale_1 + hc\_ffn\_base_{j+hc\_mult})$<br>$comb_{j,k} = \mathrm{Sinkhorn}(\mathrm{softmax}(mixes_{j \cdot hc\_mult + k + 2 hc\_mult} \cdot hc\_ffn\_scale_2 + hc\_ffn\_base_{j \cdot hc\_mult + k + 2 hc\_mult}) + hc\_eps)$<br>x 是残差流，hc\_ffn\_fn、hc\_ffn\_scale、hc\_ffn\_base 是这一调用的参数，norm_eps 与 hc_eps 是 Block 上的 eps。j、k 从 0 到 hc_mult−1。mixes 与 base 按顺序切成三段：前 hc_mult 个给 pre，接着 hc_mult 个给 post，最后 hc_mult × hc_mult 个给 comb。Sinkhorn 先做一次行归一化（softmax 后加 hc_eps），再做一次列归一化，然后交替做行、列各 19 次，共行 20 次、列 20 次，最后一次落在列方向 |
| mixes_f | 说明 | hc_ffn_fn [24, 20480]、hc_ffn_base [24]、hc_ffn_scale [3]，形状与注意力那组相同 |
| cpre2 | 公式 | $x = \sum_{c=0}^{hc\_mult-1} attn\_pre_c\, x_c$，x_c 是第 c 份残差，attn_pre 是本层注意力 hc_mixes 给出的 pre |
| cpre2 | 说明 | 用注意力子层算出的 attn_pre |
| fn | 公式 | $x = ffn\_norm.weight \odot x / \sqrt{\mathrm{mean}(x^2) + ffn\_norm.eps}$，x 是送进来的张量，ffn\_norm.weight 是权重，ffn\_norm.eps 是 eps，mean 在最后一维 |
| fn | 说明 | ffn_norm.weight [5120] |
| moe_op | 说明 | MoE，image_mask 为 None |
| cpost2 | 公式 | $x_c = ffn\_post_c\, x + \sum_{c'} ffn\_comb_{c',c}\, residual_{c'}$，右边的 x 是前馈的输出，residual 是进前馈前的残差，ffn_post 与 ffn_comb 是 hc_mixes 的系数。c' 是 residual 的份号，c 是写回的份号 |
| cpost2 | 说明 | 把前馈输出按 ffn_post 展开成 4 份，再把 residual 按 ffn_comb 混进去 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 混合系数 | $mixes = \mathrm{flatten}(x)\, hc\_fn^\top \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x)^2) + norm\_eps)$，x 是残差流。注意力那次 hc_fn、hc_scale、hc_base 是 hc_attn_fn、hc_attn_scale、hc_attn_base，前馈那次是 hc_ffn_fn、hc_ffn_scale、hc_ffn_base |
| 公式 | pre、post、comb | $pre_j = \mathrm{sigmoid}(mixes_j \cdot hc\_scale_0 + hc\_base_j) + hc\_eps$，$post_j = 2\,\mathrm{sigmoid}(mixes_{j+hc\_mult} \cdot hc\_scale_1 + hc\_base_{j+hc\_mult})$，$comb_{j,k} = \mathrm{Sinkhorn}(\mathrm{softmax}(mixes_{j \cdot hc\_mult + k + 2 hc\_mult} \cdot hc\_scale_2 + hc\_base_{j \cdot hc\_mult + k + 2 hc\_mult}) + hc\_eps)$。hc_scale、hc_base 在注意力是 hc_attn_scale、hc_attn_base，在前馈是 hc_ffn_scale、hc_ffn_base。j、k 从 0 到 hc_mult−1。Sinkhorn 先行归一化、再列归一化，然后交替各 19 次，共行 20 次、列 20 次，最后一次落在列方向 |
| 公式 | 折叠 | 注意力前 $x = \sum_c pre\_mix_c\, x_c$，前馈前 $x = \sum_c attn\_pre_c\, x_c$。c 从 0 到 hc_mult−1 |
| 公式 | 展开 | 注意力后用 attn_post、attn_comb，前馈后用 ffn_post、ffn_comb：$x_c = post_c\, x + \sum_{c'} comb_{c',c}\, residual_{c'}$。右边的 x 是子层输出，c' 是 residual 的份号，c 是写回的份号 |
| 配置 | dim | 5120 |
| 配置 | hc_mult | 4 |
| 配置 | hc_sinkhorn_iters | 20 |
| 配置 | hc_eps | 1e-06 |
| 配置 | norm_eps | 1e-20 |

## Attention.forward（model.py:765-789）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ax | 张量 | x | bf16 [B, T, 5120] | 已经过 attn_norm 的层输入 | model.py:766 |
| fcall | 张量 | self.freqs_cis | 复数 fp32 [max_seq_len, 32] | 构造时按 max_seq_len 预生成的整张频率表 | model.py:688-698 |
| fc | 张量 | freqs_cis | 复数 fp32 [T, 32] | 从 self.freqs_cis 里取本次这一段，start_pos 为 0 所以是前 T 行，构造时按 max_seq_len 预生成<br>压缩比为 0 的层用 rope_theta 且不做 YaRN，其余层用 compress_rope_theta 并按 original_seq_len 做 YaRN<br>32 = rope_head_dim / 2，每个位置一行 | model.py:767 |
| qr_op | 算子 | self.q_norm(self.wq_a(x)) | 5120 → 1280 |  | model.py:770 |
| qr | 张量 | qr | bf16 [B, T, 1280] | 索引器拿它当 query 的输入 | model.py:770 |
| qb | 算子 | self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim)) | 1280 → 64 × 512 |  | model.py:771 |
| qpre | 张量 | q | bf16 [B, T, 64, 512] | | model.py:771 |
| rope_q | 算子 | apply_rotary_emb(q[..., -rd:], freqs_cis) | 末 64 维，形状不变 |  | model.py:772 |
| q | 张量 | q | bf16 [B, T, 64, 512] | | model.py:772 |
| wkv_win | 算子 | self.kv_norm(self.wkv(x)) | 5120 → 512 |  | model.py:705 |
| wkv_o | 张量 | kv | bf16 [B, T, 512] | | model.py:705 |
| wrope | 算子 | apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis) | 末 64 维，形状不变 |  | model.py:706 |
| wkv_r | 张量 | kv | bf16 [B, T, 512] | | model.py:706 |
| wquant | 算子 | act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True) | 形状不变 |  | model.py:707 |
| wkv_q | 张量 | kv | bf16 [B, T, 512] | 存的是 fp8 量化后再反量化的值 | model.py:707 |
| wwrite | 算子 | self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1) | 写环形缓冲 |  | model.py:712-715 |
| wchunk | 张量 | window_kv | bf16 [B, T, 512] | prefill 就是本次这块 KV | model.py:716 |
| widx | 算子 | get_window_topk_idxs(win, bsz, seqlen, start_pos) | T, window_size → int32 [B, T, 128] |  | model.py:720 |
| widxs | 张量 | topk_idxs | int32 [B, T, 128] | | model.py:720 |
| wcache | 缓存 | window_kv_cache | [max_batch_size, 128, 512] | 每层一份，按 max_batch_size 分配；prefill 整块覆盖，decode 时改写 start_pos % 128 那一槽 | model.py:663-668 |
| comp_op | 算子 | self.compressor(x, start_pos) | [B, T, 5120] → [B, T/r, 512] |  | model.py:747 |
| lat | 张量 | latent | bf16 [B, T/r, 512] | 压缩器输出的潜在量，还没加 RoPE；r 是这一层的 compress_ratio | model.py:747 |
| cidx_op | 算子 | self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len) | [B, T, 5120] → int32 [B, T, 512] |  | model.py:750 |
| cidxs | 张量 | idxs | int32 [B, T, 512] | 压缩条目下标，已经偏移到拼起来的 KV 里的位置 | model.py:750 |
| cf | 算子 | self.freqs_cis[: seqlen - seqlen % ratio : ratio] | [max_seq_len, 32] → [T/r, 32] |  | model.py:753-757 |
| cfq | 张量 | freqs | 复数 fp32 [T/r, 32] | | model.py:753-757 |
| crope | 算子 | apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs) | 末 64 维，形状不变 |  | model.py:758 |
| lat_r | 张量 | latent | bf16 [B, T/r, 512] | | model.py:758 |
| cquant | 算子 | fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn) | 形状不变 |  | model.py:760 |
| lat_q | 张量 | latent | bf16 [B, T/r, 512] | 存的是 fp4 量化后再反量化的值 | model.py:760 |
| cwrite | 算子 | self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent | 写压缩缓存 |  | model.py:761 |
| ccache | 缓存 | compress_kv_cache | [max_batch_size, max_seq_len / r, 512] | 2、8、14、20 各有一份，按 max_batch_size 分配；每个 source 层写自己这一份并把它发布给 shared_attn，它之后到下一个 kv source 层之前的层读的是这一份；decode 时每组填满才写一行 | model.py:669-679 |
| ckv | 张量 | compress_kv | bf16 [B, T/r, 512] | 从 compress_kv_cache 读出的、本次可见的部分 | model.py:763 |
| cat | 算子 | torch.cat([kv, compress_kv], dim=1) | 两份 KV 沿位置维拼起来 |  | model.py:777 |
| kvc | 张量 | kv | bf16 [B, T + T/r, 512] | | model.py:777 |
| cat2 | 算子 | torch.cat([topk_idxs, compress_idxs], dim=-1) | 两份下标拼起来 |  | model.py:778 |
| tki | 张量 | topk_idxs | int32 [B, T, 128 + 512] | | model.py:778 |
| core | 算子 | sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale) | q [B, T, 64, 512] 对 kv [B, T + T/r, 512] → [B, T, 64, 512] |  | model.py:780 |
| o0 | 张量 | o | bf16 [B, T, 64, 512] | | model.py:780 |
| unrope | 算子 | apply_rotary_emb(o[..., -rd:], freqs_cis, True) | 末 64 维，形状不变 |  | model.py:781 |
| o1 | 张量 | o | bf16 [B, T, 64, 512] | | model.py:781 |
| grp | 算子 | o.view(bsz, seqlen, self.n_local_groups, -1) | [B, T, 64, 512] → [B, T, 8, 4096] |  | model.py:785 |
| og | 张量 | o | bf16 [B, T, 8, 4096] | | model.py:785 |
| woa | 算子 | self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1) | [8192, 4096] → [8, 1024, 4096] |  | model.py:786 |
| wav | 张量 | wo_a | bf16 [8, 1024, 4096] | | model.py:786 |
| eins | 算子 | torch.einsum("bsgd,grd->bsgr", o, wo_a) | [B, T, 8, 4096] 与 [8, 1024, 4096] → [B, T, 8, 1024] |  | model.py:787 |
| ol | 张量 | o | bf16 [B, T, 8, 1024] | | model.py:787 |
| wb | 算子 | self.wo_b(o.flatten(2)) | 8192 → 5120 |  | model.py:788 |
| axout | 张量 | x | bf16 [B, T, 5120] | | model.py:788-789 |

### 连线

- ax → qr_op
- qr_op → qr
- qr → qb
- qb → qpre
- qpre → rope_q
- fcall → fc
- fc → rope_q
- rope_q → q
- ax → wkv_win
- wkv_win → wkv_o
- wkv_o → wrope
- fc → wrope
- wrope → wkv_r
- wkv_r → wquant
- wquant → wkv_q
- wkv_q → wwrite
- wwrite → wcache
- wkv_q → wchunk
- ax → widx：形状（bsz、seqlen）
- widx → widxs
- ax → comp_op
- comp_op → lat
- lat → cidx_op
- qr → cidx_op
- ax → cidx_op
- cidx_op → cidxs
- lat → crope
- fcall → cf
- cf → cfq
- cfq → crope
- crope → lat_r
- lat_r → cquant
- cquant → lat_q
- lat_q → cwrite
- cwrite → ccache
- ccache → ckv
- wchunk → cat
- ckv → cat
- cat → kvc
- widxs → cat2
- cidxs → cat2
- cat2 → tki
- q → core
- kvc → core
- tki → core
- core → o0
- o0 → unrope
- fc → unrope
- unrope → o1
- o1 → grp
- grp → og
- og → eins
- wav → eins
- eins → ol
- ol → wb
- wb → axout
- comp_op 点开 → Compressor.forward
- cidx_op 点开 → Indexer.forward
- qr_op 点开 → RMSNorm.forward
- wkv_win 点开 → RMSNorm.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| qr_op | 公式 | $qr = q\_norm.weight \odot \dfrac{x\, wq\_a.weight^\top}{\sqrt{\mathrm{mean}((x\, wq\_a.weight^\top)^2) + q\_norm.eps}}$，x 是本层输入，wq_a.weight 与 q_norm.weight 是这一行的两个权重，q_norm.eps 是 eps，mean 在最后一维 |
| qr_op | 说明 | wq_a.weight [1280, 5120]、q_norm.weight [1280] |
| qb | 公式 | $q = \mathrm{unflatten}(qr\, wq\_b.weight^\top, (n\_local\_heads, head\_dim))$，qr 是上一行的输出，wq_b.weight 是第二层投影 |
| qb | 说明 | wq_b.weight [32768, 1280]，64 × 512 = 32768 |
| rope_q | 公式 | $q_{...,-rd:} = \mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(q_{...,-rd:}) \odot freqs\_cis)$，q 是 query，freqs_cis 是这一段的频率，rd 是 rope_head_dim，写在原地 |
| rope_q | 说明 | 只旋转每头 512 维里的后 64 维，前 448 维不动；写在原地 |
| wkv_win | 公式 | $kv = kv\_norm.weight \odot \dfrac{x\, wkv.weight^\top}{\sqrt{\mathrm{mean}((x\, wkv.weight^\top)^2) + kv\_norm.eps}}$，x 是本层输入，wkv.weight 与 kv_norm.weight 是这一行的两个权重，kv_norm.eps 是 eps |
| wkv_win | 说明 | wkv.weight [512, 5120]、kv_norm.weight [512] |
| wrope | 公式 | $kv_{...,-rope\_head\_dim:} = \mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(kv_{...,-rope\_head\_dim:}) \odot freqs\_cis)$，kv 是窗口键，freqs_cis 与 query 用的是同一份 |
| wrope | 说明 | 和 query 用同一份 freqs_cis |
| wquant | 说明 | 每 32 个通道一个 E8M0 scale，原地量化再反量化；量化按 fp8 精度（压缩 KV 走 fp4）<br>量化覆盖整条 512 维，含已经加过 RoPE 的末 64 维 |
| wwrite | 说明 | cutoff = seqlen % win，只留最后 win 个 token，尾巴绕回环首 |
| widx | 说明 | 每个 query 能看的窗口槽位，−1 是空槽；每个 query 各一行，只看到自己的因果窗口 |
| comp_op | 说明 | 只有 4 个 source 层（2、8、14、20）算；其余 compress_ratio 大于 0 的层 latent 为 None，直接读已有缓存；层 0、1 的压缩比为 0，整段压缩都不走 |
| cidx_op | 说明 | 本层是 index source（2、8、14、20、24、28、32、36）就跑自己的 Indexer，其余 compress_ratio 大于 0 的层直接复用 shared_attn.topk_idxs<br>offset 取窗口 KV 的长度（本次是 T），compress_len 是 (start_pos + T) / r |
| cf | 公式 | $freqs = freqs\_cis[:seqlen - seqlen \% ratio: ratio]$，第 g 组取位置 $g \times ratio$，ratio 是这一层的 compress_ratio |
| cf | 说明 | 取压缩位置的频率，一个 latent 顶替一整组，第 g 组取位置 g × r |
| crope | 公式 | $latent_{...,-rope\_head\_dim:} = \mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(latent_{...,-rope\_head\_dim:}) \odot freqs)$，latent 是压缩潜在量，freqs 是压缩位置的频率 |
| crope | 说明 | 乘压缩位置的 freqs |
| cquant | 说明 | E2M1 格式，每 16 个通道一个 E4M3 scale，原地量化再反量化 |
| cwrite | 说明 | start_pos 为 0，所以从第 0 行起整块写 |
| cat | 说明 | 窗口在前、压缩在后 |
| cat2 | 说明 | 压缩下标在 _compress_kv 里已经加过窗口长度 T |
| core | 公式 | $o = \dfrac{\sum_{t} \exp(q \cdot kv_t \cdot softmax\_scale - \max) \, kv_t}{\sum_{t} \exp(q \cdot kv_t \cdot softmax\_scale - \max) + \exp(attn\_sink - \max)}$，t 取 topk_idxs 里的槽位，kv 同时充当 k 和 v，softmax_scale 是 head_dim 的 -1/2 次方，attn_sink 是可学习参数，$\max$ 是这些槽位上 $q \cdot kv_t \cdot softmax\_scale$ 的最大值 |
| core | 说明 | 单 KV 头，64 个查询头看同一份 KV；attn_sink 是可学习参数 [64]<br>下标为 −1 的槽位分子分母都不贡献 |
| unrope | 公式 | $o_{...,-rd:} = \mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(o_{...,-rd:}) \odot freqs\_cis^*)$，o 是注意力输出，freqs_cis 取共轭，rd 是 rope_head_dim |
| unrope | 说明 | 用共轭频率对输出做一次逆旋转，抵消 KV 里已加 RoPE 的键带进注意力输出的那部分旋转，使输出回到与键缓存同一套表示下 |
| grp | 说明 | 每 8 个头一组，8 × 512 = 4096 |
| woa | 说明 | wo_a.weight [8192, 4096]，8192 = 8 × 1024 |
| eins | 公式 | $o_{b,s,g,r} = \sum_d o_{b,s,g,d}\, wo\_a_{g,r,d}$，o 是分组后的注意力输出，wo_a 是 wo_a.weight 按组摊开的结果 |
| eins | 说明 | 每组只和本组的权重相乘，等效块对角矩阵 |
| wb | 公式 | $x = \mathrm{flatten}(o)\, wo\_b.weight^\top$，o 是 eins 的输出，wo_b.weight 是输出投影 |
| wb | 说明 | wo_b.weight [5120, 8192] |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 频率 | $\theta_j = \mathrm{base}^{-2j/64}$，j = 0 … 31，base 在压缩比为 0 的层是 rope_theta、其余层是 compress_rope_theta，64 是 rope_head_dim |
| 公式 | YaRN | $\theta_j' = (\theta_j/16)\,\rho_j + \theta_j\,(1 - \rho_j)$，16 是 rope_factor，$\rho_j = \mathrm{clamp}\big((j - low)/(high - low), 0, 1\big)$ 是随 j 递增的斜坡，j 不大时 $\rho_j$ 为 0、保留原频率，j 变大后逐步除以 16，low 是 corrected_dim(beta_fast) 向下取整、high 是 corrected_dim(beta_slow) 向上取整；只对压缩比大于 0 的层生效 |
| 公式 | query 低秩投影 | $qr = q\_norm.weight \odot \dfrac{x\, wq\_a.weight^\top}{\sqrt{\mathrm{mean}((x\, wq\_a.weight^\top)^2) + q\_norm.eps}}$，再 $q = \mathrm{unflatten}(qr\, wq\_b.weight^\top, (n\_local\_heads, head\_dim))$。x 是本层输入 |
| 公式 | 窗口 KV | $kv = kv\_norm.weight \odot \dfrac{x\, wkv.weight^\top}{\sqrt{\mathrm{mean}((x\, wkv.weight^\top)^2) + kv\_norm.eps}}$，x 是本层输入。后 rope_head_dim 维再乘 freqs_cis |
| 公式 | 压缩 KV 的位置 | 第 g 个 latent 顶替第 $g r$ 到 $(g+1)r - 1$ 个 token，所以它的 RoPE 用位置 $g r$，r 是 compress_ratio |
| 公式 | 稀疏注意力 | $o = \dfrac{\sum_{t} \exp(q \cdot kv_t \cdot softmax\_scale - \max)\, kv_t}{\sum_{t} \exp(q \cdot kv_t \cdot softmax\_scale - \max) + \exp(attn\_sink - \max)}$，t 取 topk_idxs 里的槽位，kv 同时充当 k 和 v，softmax_scale 是 head_dim 的 -1/2 次方，$\max$ 是这些分数的最大值，attn_sink 是可学习参数 |
| 公式 | 输出低秩投影 | $o_{b,s,g,r} = \sum_d o_{b,s,g,d}\, wo\_a_{g,r,d}$，再 $x = \mathrm{flatten}(o)\, wo\_b.weight^\top$。wo_a 是 wo_a.weight 按组摊开的结果 |
| 配置 | dim | 5120 |
| 配置 | n_heads | 64 |
| 配置 | head_dim | 512 |
| 配置 | rope_head_dim | 64 |
| 配置 | q_lora_rank | 1280 |
| 配置 | o_lora_rank | 1024 |
| 配置 | o_groups | 8 |
| 配置 | window_size | 128 |
| 配置 | index_topk | 512 |
| 配置 | rope_theta | 10000 |
| 配置 | compress_rope_theta | 160000 |

## Compressor.forward（model.py:458-485）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ax | 张量 | x | bf16 [B, T, 5120] | source 层的注意力输入，与注意力、索引器两张图里的 `ax` 是同一份 | model.py:459 |
| cf32 | 算子 | x.float() | [B, T, 5120] 不变 |  | model.py:464 |
| cxf | 张量 | x | fp32 [B, T, 5120] | | model.py:464 |
| proj | 算子 | self.wkv(x), self.wgate(x) | [B, T, 5120] → [B, T, 512] × 2 |  | model.py:465 |
| kv0 | 张量 | kv | fp32 [B, T, 512] | 逐 token 的值投影 | model.py:465 |
| csc | 张量 | score | fp32 [B, T, 512] | 逐 token 的门控打分 | model.py:465 |
| pool | 算子 | kv.unflatten(1, (-1, ratio))、score.unflatten(1, (-1, ratio)) 与 (kv * score.softmax(dim=2)).sum(dim=2) | [B, T, 512] → [B, T/r, 512] |  | model.py:473-475 |
| kvg | 张量 | kv | fp32 [B, T/r, 512] | | model.py:475 |
| cnorm | 算子 | self.norm(kv.to(dtype)) | 形状不变 |  | model.py:485 |
| cout | 张量 | self.norm(kv.to(dtype)) | bf16 [B, T/r, 512] | 返回值是还没加 RoPE 的压缩潜在量 | model.py:485 |

### 连线

- ax → cf32
- cf32 → cxf
- cxf → proj
- proj → kv0
- proj → csc
- kv0 → pool
- csc → pool
- pool → kvg
- kvg → cnorm
- cnorm → cout
- cnorm 点开 → RMSNorm.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| cf32 | 说明 | 池化与门控都在 fp32 里算 |
| proj | 公式 | $kv = x\, wkv.weight^\top$，$score = x\, wgate.weight^\top$，x 是本层输入 |
| proj | 说明 | wkv.weight [512, 5120] 与 wgate.weight [512, 5120]，压缩比大于 1 时两个权重都是 fp32 |
| pool | 公式 | 按 ratio 分组后 $kv = \sum_t \mathrm{softmax}(score)_t \, kv_t$，softmax 在这一组的 ratio 个 token 上，kv 与 score 是上一行的两个投影 |
| pool | 说明 | 把相邻 r 个 token 当成一组，组内用 score 的 softmax 做加权求和 |
| cnorm | 公式 | $kv = norm.weight \odot kv / \sqrt{\mathrm{mean}(kv^2) + norm.eps}$，kv 是送进来的张量，norm.weight 是权重，norm.eps 是 eps，mean 在最后一维 |
| cnorm | 说明 | 池化结果先转回 bf16 再 RMSNorm，layers.{i}.attn.compressor.norm.weight [512] |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 池化 | 按 ratio 分组后 $kv = \sum_t \mathrm{softmax}(score)_t \, kv_t$，softmax 在这一组的 ratio 个 token 上，kv 与 score 是 wkv、wgate 的输出 |
| 公式 | 输出 | $kv = norm.weight \odot kv / \sqrt{\mathrm{mean}(kv^2) + norm.eps}$，kv 是池化结果，先转回 bf16，norm.weight 是 compressor.norm.weight，mean 在最后一维 |
| 配置 | dim | 5120 |
| 配置 | head_dim | 512 |
| 配置 | norm_eps | 1e-20 |
| 配置 | compress_ratios | 前 40 项是主干：层 0、1 为 0，层 2–19 为 2，层 20–39 为 1；末 3 项是 MTP 层，为 0 |

## Indexer.forward（model.py:527-580）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ax | 张量 | x | bf16 [B, T, 5120] | source 层的注意力输入，与注意力、压缩器两张图里的 `ax` 是同一份 | model.py:533 |
| qr | 张量 | qr | bf16 [B, T, 1280] | 注意力那边算出的低秩 query，索引器直接拿它当输入 | model.py:527 |
| lat | 张量 | latent | bf16 [B, T/r, 512] | 压缩器的输出，还没加 RoPE | model.py:527，537 |
| fcr | 张量 | self.freqs_cis | 复数 fp32 [max_seq_len, 32] | 注意力那边传进来的整张频率表 | model.py:733-734 |
| fpos | 算子 | self.freqs_cis[: seqlen - seqlen % ratio : ratio] | [max_seq_len, 32] → [T/r, 32] |  | model.py:539-543 |
| fq | 张量 | freqs | 复数 fp32 [T/r, 32] | | model.py:540-543 |
| kw | 算子 | self.k_norm(self.wk(latent)) | 512 → 128 |  | model.py:544 |
| k0 | 张量 | k | bf16 [B, T/r, 128] | | model.py:544 |
| krope | 算子 | apply_rotary_emb(k[..., -rd:], freqs) | 末 64 维，形状不变 |  | model.py:545 |
| kr | 张量 | k | bf16 [B, T/r, 128] | | model.py:545 |
| kq | 算子 | fp4_act_quant(k, fp4_block_size, True) | 形状不变 |  | model.py:546 |
| kf | 张量 | k | bf16 [B, T/r, 128] | 存的是 fp4 量化后再反量化的值 | model.py:546 |
| kwrite | 算子 | self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k | 写入索引键缓存 |  | model.py:547-548 |
| kcache | 缓存 | k_cache | [max_batch_size, max_seq_len / r, 128] | 2、8、14、20 各有一份，按 max_batch_size 分配；写完发布给 shared_attn，它之后到下一个 kv source 层之前的索引器读的是这一份 | model.py:520-525 |
| qop | 算子 | self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim)) | 1280 → 32 × 128 |  | model.py:550 |
| iq0 | 张量 | q | bf16 [B, T, 32, 128] | | model.py:550 |
| qrope | 算子 | apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos]) | 末 64 维，形状不变 |  | model.py:551 |
| iqr | 张量 | q | bf16 [B, T, 32, 128] | | model.py:551 |
| iq_q | 算子 | fp4_act_quant(q, fp4_block_size, True) | 形状不变 |  | model.py:552 |
| iq | 张量 | q | bf16 [B, T, 32, 128] | 存的是 fp4 量化后再反量化的值 | model.py:552 |
| ik | 张量 | index_k | bf16 [B, T/r, 128] | 从 shared_attn.index_k 读出的、本次可见的部分 | model.py:554 |
| wproj | 算子 | self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5) | 5120 → 32 |  | model.py:555 |
| iwt | 张量 | weights | bf16 [B, T, 32] | | model.py:555 |
| idot | 算子 | torch.einsum("bshd,btd->bsht", q, index_k) | [B, T, 32, 128] 与 [B, T/r, 128] → [B, T, 32, T/r] |  | model.py:556 |
| sc0 | 张量 | index_score | bf16 [B, T, 32, T/r] | | model.py:556 |
| red | 算子 | (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2) | [B, T, 32, T/r] → [B, T, T/r] |  | model.py:557 |
| isc | 张量 | index_score | bf16 [B, T, T/r] | | model.py:557 |
| lens | 算子 | (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1) | T → int64 [T, 1] |  | model.py:563-564 |
| lv | 张量 | compress_lens | int64 [T, 1] | | model.py:563-564 |
| imask | 算子 | index_score.masked_fill_(torch.arange(seqlen // ratio) >= compress_lens, -torch.inf) | 形状不变 |  | model.py:565 |
| iscm | 张量 | index_score | bf16 [B, T, T/r] | | model.py:565 |
| topk | 算子 | index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values | bf16 [B, T, T/r] → int64 [B, T, 512] |  | model.py:578-579 |
| isel | 张量 | idxs | int64 [B, T, 512] | | model.py:579 |
| shift | 算子 | torch.where(idxs < compress_lens, idxs + offset, -1) | 形状不变 |  | model.py:580 |
| cidxs | 张量 | idxs | int32 [B, T, 512] | 交给注意力那边拼进 topk_idxs | model.py:580 |

### 连线

- qr → qop
- qop → iq0
- iq0 → qrope
- fcr → qrope
- qrope → iqr
- iqr → iq_q
- iq_q → iq
- fcr → fpos
- fpos → fq
- lat → kw
- kw → k0
- k0 → krope
- fq → krope
- krope → kr
- kr → kq
- kq → kf
- kf → kwrite
- kwrite → kcache
- kcache → ik
- ax → wproj
- wproj → iwt
- iq → idot
- ik → idot
- idot → sc0
- sc0 → red
- iwt → red
- red → isc
- ax → lens：形状与 device
- lens → lv
- isc → imask
- lv → imask
- imask → iscm
- iscm → topk
- topk → isel
- isel → shift
- lv → shift
- shift → cidxs
- kw 点开 → RMSNorm.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| fpos | 公式 | $freqs = freqs\_cis[:seqlen - seqlen \% ratio: ratio]$，第 g 组取位置 $g \times ratio$ |
| fpos | 说明 | 取压缩位置的频率，第 g 组取位置 g × r |
| kw | 公式 | $k = k\_norm.weight \odot \dfrac{latent\, wk.weight^\top}{\sqrt{\mathrm{mean}((latent\, wk.weight^\top)^2) + k\_norm.eps}}$，latent 是压缩器的输出，wk.weight 与 k_norm.weight 是这一行的两个权重 |
| kw | 说明 | wk.weight [128, 512]、k_norm.weight [128] |
| krope | 公式 | $k_{...,-rd:} = \mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(k_{...,-rd:}) \odot freqs)$，k 是索引键，freqs 是压缩位置的频率，rd 是 rope_head_dim |
| krope | 说明 | 128 维里的最后 64 维加 RoPE |
| kq | 说明 | E2M1 格式，每 32 个通道一个 E8M0 scale，原地量化再反量化 |
| kwrite | 说明 | start_pos 为 0，从第 0 行起整块写；写完把这份缓存发布到 shared_attn.index_k，后面的层直接读 |
| qop | 公式 | $q = \mathrm{unflatten}(qr\, wq\_b.weight^\top, (n\_local\_heads, index\_head\_dim))$，qr 是注意力低秩 query，wq_b.weight 是索引器的查询投影 |
| qop | 说明 | wq_b.weight [4096, 1280]，32 × 128 = 4096 |
| qrope | 公式 | $q_{...,-rd:} = \mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(q_{...,-rd:}) \odot freqs\_cis)$，q 是索引 query，freqs_cis 从 start_pos 取到 end_pos，rd 是 rope_head_dim |
| qrope | 说明 | 乘从 start_pos 取到 end_pos 的 freqs_cis |
| iq_q | 说明 | 索引器的 query 也量化成 fp4 |
| wproj | 公式 | $weights = (x\, weights\_proj.weight^\top) \cdot softmax\_scale \cdot n\_heads^{-0.5}$，x 是本层输入，softmax_scale 是 index_head_dim 的 -1/2 次方，n_heads 是 index_n_heads |
| wproj | 说明 | weights_proj.weight [32, 5120]；乘的常数是 $128^{-1/2} \cdot 32^{-1/2}$ |
| idot | 公式 | $index\_score_{b,s,h,t} = \sum_d q_{b,s,h,d}\, index\_k_{b,t,d}$，q 是索引 query，index_k 是索引键，h 是 einsum 里的头维，d 是 index_head_dim |
| idot | 说明 | 每个索引头对每个压缩位置记一个点积 |
| red | 公式 | $index\_score = \sum_i \mathrm{relu}(index\_score_i) \cdot weights_i$，i 是 head 维，weights 是 weights_proj 给出的系数，求和前的 index_score 是上一行的点积 |
| red | 说明 | 先过 ReLU，再按头用 weights 加权求和 |
| lens | 公式 | $compress\_lens_i = \lfloor (i+1) / ratio \rfloor$，i 从 0 到 seqlen−1，ratio 是 compress_ratio |
| lens | 说明 | 第 i 个 query 能看到 ⌊(i+1)/r⌋ 个压缩条目 |
| imask | 公式 | 把 $arange(seqlen / ratio) \ge compress\_lens$ 的位置写成 $-\infty$，改的是 index_score |
| imask | 说明 | 把本次还看不到的压缩条目在 index_score 上原地写成 −inf |
| topk | 公式 | $idxs = \mathrm{sort}(\mathrm{topk}(index\_score, topk))$，topk 取 min(512, T/ratio)，sort 按位置排回 |
| topk | 说明 | topk = min(512, T/r)，前提里 T 足够长所以就是 512；取分数最高的 512 个，再按位置顺序排回来 |
| shift | 公式 | $idxs = \mathrm{where}(idxs < compress\_lens,\ idxs + offset,\ -1)$，idxs 是 topk 的结果，offset 是窗口 KV 的长度 |
| shift | 说明 | 不小于可达数的置 −1，其余加上 offset 变成拼起来的 KV 里的下标 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 索引打分 | $index\_score = \sum_i \mathrm{relu}(q_i \cdot index\_k) \cdot weights_i$，weights 来自 $weights = (x\, weights\_proj.weight^\top) \cdot softmax\_scale \cdot n\_heads^{-0.5}$。softmax_scale 是 index_head_dim 的 -1/2 次方，n_heads 是 index_n_heads，i 是头维 |
| 公式 | 可达数 | 第 i 个 query 能看到 $\lfloor (i+1)/r \rfloor$ 个压缩条目，r 是 compress_ratio；还看不到的那些置 $-\infty$ |
| 公式 | 输出 | $idxs = \mathrm{sort}(\mathrm{topk}(index\_score, topk))$，再 $idxs = \mathrm{where}(idxs < compress\_lens, idxs + offset, -1)$。topk 取 min(512, T/ratio)，offset 是窗口 KV 的长度 |
| 配置 | index_n_heads | 32 |
| 配置 | index_head_dim | 128 |
| 配置 | index_topk | 512 |
| 配置 | rope_head_dim | 64 |

## MoE.forward（model.py:889-904）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| mx | 张量 | x | bf16 [B, T, 5120] | 前馈子层的输入，已经过 ffn_norm | model.py:890 |
| flat | 算子 | x.view(-1, self.dim) | [B, T, 5120] → [B×T, 5120] |  | model.py:891 |
| mxf | 张量 | x | bf16 [B×T, 5120] | | model.py:891 |
| gate_op | 算子 | self.gate(x, None) | [B×T, 5120] → 权重 [B×T, 6] 与专家下标 [B×T, 6] |  | model.py:892 |
| gw | 张量 | weights | fp32 [B×T, 6] | | model.py:892 |
| gi | 张量 | indices | int64 [B×T, 6] | | model.py:892 |
| y0 | 算子 | torch.zeros_like(x, dtype=torch.float32) | bf16 [B×T, 5120] → fp32 [B×T, 5120] |  | model.py:893 |
| y | 张量 | y | fp32 [B×T, 5120] | | model.py:893 |
| mpick | 算子 | torch.where(indices == i) | int64 [B×T, 6] → int64 [n] × 2 |  | model.py:899 |
| pidx | 张量 | idx, top | int64 [n] 与 int64 [n] | n 是分到第 i 个专家的 token 数 | model.py:899 |
| mexp | 算子 | expert(x[idx], weights[idx, top, None]) | 5120 → 5120 |  | model.py:900 |
| eout | 张量 | expert(x[idx], weights[idx, top, None]) | bf16 [n, 5120] | | model.py:900 |
| acc | 算子 | y[idx] += expert(...) | fp32 [B×T, 5120] 上累加 |  | model.py:900 |
| y2 | 张量 | y | fp32 [B×T, 5120] | | model.py:900 |
| shared | 算子 | self.shared_experts(x) | 5120 → 5120 |  | model.py:903 |
| sv | 张量 | self.shared_experts(x) | bf16 [B×T, 5120] | | model.py:903 |
| madd | 算子 | y += self.shared_experts(x) | 逐元素相加 |  | model.py:903 |
| y3 | 张量 | y | fp32 [B×T, 5120] | | model.py:903 |
| mback | 算子 | y.type_as(x).view(shape) | [B×T, 5120] → bf16 [B, T, 5120] |  | model.py:904 |
| mout | 张量 | y | bf16 [B, T, 5120] | | model.py:904 |

### 连线

- mx → flat
- flat → mxf
- mxf → gate_op
- gate_op → gw
- gate_op → gi
- mxf → y0：形状与 device
- y0 → y
- gi → mpick
- mpick → pidx
- pidx → mexp：行下标与 top 位置
- mxf → mexp：x[idx]
- gw → mexp：weights[idx, top]
- mexp → eout
- eout → acc
- y → acc
- acc → y2
- mxf → shared
- shared → sv
- y2 → madd
- sv → madd
- madd → y3
- y3 → mback
- mback → mout
- gate_op 点开 → Gate.forward
- mexp 点开 → Expert.forward
- shared 点开 → Expert.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| flat | 说明 | 把批次与序列两维拉平，MoE 按 token 处理 |
| gate_op | 说明 | Gate：打分、加偏置选专家、在 top-6 内归一化、乘 route_scale；image_mask 为 None，源码在这个分支上就传 None |
| y0 | 说明 | 累加器，路由专家的结果都往它上面加 |
| mpick | 说明 | 挑出分给第 i 个专家的 token，384 个专家各跑一遍；没分到 token 的专家整个跳过 |
| mexp | 说明 | 每个 token 只过它被分到的那个专家，路由权重按自己的 top 位置取 |
| acc | 公式 | $y[idx] = y[idx] + expert(x[idx], weights[idx, top])$，idx 与 top 标明这一 token 分到的专家和它在 top-6 里的位置 |
| acc | 说明 | 一个 token 落在 top-6 的哪几个专家上就加几次 |
| shared | 说明 | shared_experts 是一个 Expert，每个 token 都过，它的三个权重按主干精度 fp8 存储 |
| madd | 公式 | $y = y + shared\_experts(x)$，y 是路由专家已经累加的结果，x 是拉平后的隐状态 |
| madd | 说明 | 路由专家的和再加上共享专家的输出 |
| mback | 说明 | 转回输入的 dtype 与形状 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 前馈 | $y[idx] = y[idx] + expert(x[idx], weights[idx, top])$，然后 $y = y + shared\_experts(x)$。x 是拉平后的隐状态，weights 是路由权重 |
| 配置 | dim | 5120 |
| 配置 | moe_inter_dim | 2304 |
| 配置 | n_routed_experts | 384 |
| 配置 | n_shared_experts | 1 |
| 配置 | n_activated_experts | 6 |

## Gate.forward（model.py:809-827）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| gx | 张量 | x | bf16 [B×T, 5120] | 拉平后的隐状态 | model.py:811 |
| lg | 算子 | linear(x.float(), self.weight.float()) / self.gate_temp | 5120 → 384 |  | model.py:811 |
| s0 | 张量 | scores | fp32 [B×T, 384] | | model.py:811 |
| act | 算子 | F.softplus(scores).sqrt() | 形状不变 |  | model.py:817 |
| s | 张量 | scores | fp32 [B×T, 384] | | model.py:817 |
| bv | 张量 | self.bias | fp32 [384] | 可学习参数，只在挑专家时加到分数上，不改权重的值；image_mask 为 None 时源码直接用 `self.bias` | model.py:818 |
| gsel | 算子 | (scores + bias).topk(self.topk, dim=-1)[1] | fp32 [B×T, 384] → int64 [B×T, 6] |  | model.py:822 |
| gi | 张量 | indices | int64 [B×T, 6] | | model.py:822 |
| gth | 算子 | scores.gather(1, indices) | fp32 [B×T, 384] → fp32 [B×T, 6] |  | model.py:823 |
| w0 | 张量 | weights | fp32 [B×T, 6] | | model.py:823 |
| nrm | 算子 | weights /= weights.sum(dim=-1, keepdim=True) + 1e-20 | 形状不变 |  | model.py:824-825 |
| w1 | 张量 | weights | fp32 [B×T, 6] | | model.py:825 |
| grs | 算子 | weights *= self.route_scale | 形状不变 |  | model.py:826 |
| gw | 张量 | weights | fp32 [B×T, 6] | | model.py:826 |
| gret | 张量 | (weights, indices) | — | 两个返回值都交给 MoE | model.py:827 |

### 连线

- gx → lg
- lg → s0
- s0 → act
- act → s
- s → gsel
- bv → gsel
- gsel → gi
- s → gth
- gi → gth
- gth → w0
- w0 → nrm
- nrm → w1
- w1 → grs
- grs → gw
- gw → gret
- gi → gret

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| lg | 公式 | $scores = \mathrm{linear}(x, weight) / gate\_temp$，x 是拉平后的隐状态，weight 是 gate.weight，gate_temp 是 1.0 |
| lg | 说明 | weight [384, 5120]，打分在 fp32 里算<br>gate_temp 取默认值 1.0，config.json 里没有这一项 |
| act | 公式 | $scores = \sqrt{\mathrm{softplus}(scores)}$，$\mathrm{softplus}(scores) = \ln(1+e^{scores})$ |
| act | 说明 | score_func 是 sqrtsoftplus：先 softplus 再开方<br>$\mathrm{softplus}(z) = \ln(1 + e^z)$ |
| gsel | 公式 | $indices = \mathrm{topk}(scores + bias, topk)$，bias 是 self.bias，topk 是 n_activated_experts |
| gsel | 说明 | 偏置只参与挑专家，不改权重 |
| gth | 公式 | $weights = \mathrm{gather}(scores, indices)$，取得的是没加 bias 的 scores |
| gth | 说明 | 权重取自没加偏置的 scores |
| nrm | 公式 | $weights = weights / (\mathrm{sum}(weights) + 10^{-20})$，求和在这 6 个专家上 |
| nrm | 说明 | 在 top-6 内除以这 6 个权重的和，分母加 1e-20 |
| grs | 公式 | $weights = weights \cdot route\_scale$，route_scale 是 1.5 |
| grs | 说明 | route_scale 是 1.5 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 打分 | $scores = \mathrm{linear}(x, weight) / gate\_temp$，再 $scores = \sqrt{\mathrm{softplus}(scores)}$。x 是拉平后的隐状态，weight 是 gate.weight，gate_temp 是 1.0，$\mathrm{softplus}(scores) = \ln(1+e^{scores})$ |
| 公式 | 选专家 | $indices = \mathrm{topk}(scores + bias, topk)$，bias 是 self.bias，topk 是 6。bias 只加在挑选上，weights 从原来的 scores 里 gather |
| 公式 | 权重 | $weights = \mathrm{gather}(scores, indices)$，再 $weights = weights / (\mathrm{sum}(weights) + 10^{-20})$，再 $weights = weights \cdot route\_scale$。route_scale 是 1.5，求和在这 6 个专家上 |
| 配置 | dim | 5120 |
| 配置 | n_routed_experts | 384 |
| 配置 | n_activated_experts | 6 |
| 配置 | score_func | sqrtsoftplus |
| 配置 | route_scale | 1.5 |
| 配置 | norm_topk_prob | true（默认值） |

## Expert.forward（model.py:841-851）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ex | 张量 | x | bf16 [n, 5120] | n 是分给这个专家的 token 数；共享专家那里 n 就是全部 token | model.py:842 |
| wts | 张量 | weights | fp32 [n, 1] | 只有路由专家传这个，共享专家不传 | model.py:841 |
| g1 | 算子 | self.w1(x).float() | 5120 → 2304 |  | model.py:843 |
| eact | 张量 | gate | fp32 [n, 2304] | | model.py:843 |
| u1 | 算子 | self.w3(x).float() | 5120 → 2304 |  | model.py:844 |
| up | 张量 | up | fp32 [n, 2304] | | model.py:844 |
| cl | 算子 | torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit) | 形状不变 |  | model.py:846 |
| up2 | 张量 | up | fp32 [n, 2304] | | model.py:846 |
| cl2 | 算子 | torch.clamp(gate, max=self.swiglu_limit) | 形状不变 |  | model.py:847 |
| eact2 | 张量 | gate | fp32 [n, 2304] | | model.py:847 |
| silu | 算子 | F.silu(gate) * up | 形状不变 |  | model.py:848 |
| hm | 张量 | F.silu(gate) * up | fp32 [n, 2304] | | model.py:848 |
| emw | 算子 | weights * x | 形状不变 |  | model.py:850 |
| hw | 张量 | weights * x | fp32 [n, 2304] | | model.py:850 |
| w2 | 算子 | self.w2(x.to(dtype)) | 2304 → 5120 |  | model.py:851 |
| eout | 张量 | self.w2(x.to(dtype)) | bf16 [n, 5120] | | model.py:851 |

### 连线

- ex → g1
- g1 → eact
- ex → u1
- u1 → up
- up → cl
- cl → up2
- eact → cl2
- cl2 → eact2
- eact2 → silu
- up2 → silu
- silu → hm
- hm → emw
- wts → emw
- emw → hw
- hw → w2
- w2 → eout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| g1 | 公式 | $gate = x\, w1.weight^\top$，x 是输入，结果转成 fp32 |
| g1 | 说明 | 路由专家的 w1.weight 存成 [2304, 2560] 的 fp4，每 32 个通道一个 E8M0 scale；共享专家走这张图时是 [2304, 5120] 的 fp8<br>先转 fp32 再夹 |
| u1 | 公式 | $up = x\, w3.weight^\top$，x 是输入，结果转成 fp32 |
| u1 | 说明 | w3.weight 与 w1 同一种存法 |
| cl | 公式 | $up = \mathrm{clamp}(up, -swiglu\_limit, swiglu\_limit)$，swiglu_limit 是 10 |
| cl | 说明 | up 两头都夹；swiglu_limit 是 10.0<br>夹是为了让后面的 fp8、fp4 激活不越界 |
| cl2 | 公式 | $gate = \mathrm{clamp}(gate, \max = swiglu\_limit)$，swiglu_limit 是 10 |
| cl2 | 说明 | gate 只夹上界 |
| silu | 公式 | $x = \mathrm{silu}(gate) \odot up$，gate 与 up 是夹过界的两路 |
| silu | 说明 | SiLU 的结果逐元素乘 up |
| emw | 公式 | $x = weights \odot x$，weights 是路由权重 |
| emw | 说明 | 路由权重逐 token 乘上去 |
| w2 | 公式 | $x = x\, w2.weight^\top$，x 是乘过路由权重并转回存储 dtype 的隐状态 |
| w2 | 说明 | 路由专家的 w2.weight 存成 [5120, 1152] 的 fp4；共享专家是 [5120, 2304] 的 fp8 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | SwiGLU | $gate = x\, w1.weight^\top$，$up = x\, w3.weight^\top$，夹到 swiglu_limit 以后 $x = weights \odot \mathrm{silu}(gate) \odot up$，再 $x = x\, w2.weight^\top$。x 是输入，weights 是路由权重 |
| 公式 | 夹取 | $up = \mathrm{clamp}(up, -swiglu\_limit, swiglu\_limit)$，$gate = \mathrm{clamp}(gate, \max=swiglu\_limit)$，swiglu_limit 是 10 |
| 配置 | dim | 5120 |
| 配置 | moe_inter_dim | 2304 |
| 配置 | swiglu_limit | 10.0 |

## Engram.forward（model.py:350-365）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| nx | 张量 | x | bf16 [B, T, 4, 5120] | 本层的残差流，4 份 | model.py:350 |
| hash_ids | 张量 | hash_ids | int64 [B, T, 24] | 本层的 n-gram 哈希行号，24 = (4 − 1) × 8 | model.py:350 |
| nemb | 算子 | self.embed(hash_ids).flatten(-2) | [B, T, 24] → [B, T, 6144] |  | model.py:353，296-325 |
| ev | 张量 | self.embed(hash_ids).flatten(-2) | bf16 [B, T, 6144] | | model.py:353 |
| nwkv | 算子 | self.wkv(...) | 6144 → 25600 |  | model.py:353-354 |
| nkv | 张量 | kv | bf16 [B, T, 25600] | | model.py:353 |
| key0 | 张量 | key | bf16 [B, T, 20480] | 20480 = 4 × 5120，每份残差流一个键 | model.py:354 |
| value | 张量 | value | bf16 [B, T, 5120] | 4 份残差流共用同一个值 | model.py:354 |
| ekey | 算子 | key.float().unflatten(-1, (self.hc_mult, self.dim)) | [B, T, 20480] → fp32 [B, T, 4, 5120] |  | model.py:355 |
| key | 张量 | key | fp32 [B, T, 4, 5120] | | model.py:355 |
| ewm | 算子 | self.q_weight.float() * self.k_weight.float() | [4, 5120] 逐元素相乘 |  | model.py:356 |
| ew | 张量 | weight | fp32 [4, 5120] | | model.py:356 |
| nhf | 算子 | x.float() | [B, T, 4, 5120] 不变 |  | model.py:357 |
| nh | 张量 | h | fp32 [B, T, 4, 5120] | | model.py:357 |
| rstd | 算子 | torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps) | [B, T, 4, 5120] → fp32 [B, T, 4] |  | model.py:359 |
| nrs | 张量 | rstd | fp32 [B, T, 4] | | model.py:359 |
| ndot | 算子 | (h * weight * key).sum(-1) * rstd * self.dim**-0.5 | [B, T, 4, 5120] → [B, T, 4] |  | model.py:360 |
| d | 张量 | dot | fp32 [B, T, 4] | | model.py:360 |
| ngate | 算子 | torch.sigmoid(<br>torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)) | 形状不变 |  | model.py:362 |
| g | 张量 | gate | fp32 [B, T, 4] | | model.py:362 |
| vf | 算子 | value.float().unsqueeze(-2) | [B, T, 5120] → fp32 [B, T, 1, 5120] |  | model.py:365 |
| v | 张量 | value.float().unsqueeze(-2) | fp32 [B, T, 1, 5120] | | model.py:365 |
| nadd | 算子 | h + gate.unsqueeze(-1) * value.float().unsqueeze(-2) | 形状不变 |  | model.py:365 |
| sum | 张量 | h + gate.unsqueeze(-1) * value.float().unsqueeze(-2) | fp32 [B, T, 4, 5120] | | model.py:365 |
| ncast | 算子 | .to(x.dtype) | 形状不变 |  | model.py:365 |
| nout | 张量 | (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype) | bf16 [B, T, 4, 5120] | | model.py:365 |

### 连线

- nx → nhf
- nhf → nh
- hash_ids → nemb
- nemb → ev
- ev → nwkv
- nwkv → nkv
- nkv → key0
- nkv → value
- key0 → ekey
- ekey → key
- ewm → ew
- nh → rstd
- key → rstd
- rstd → nrs
- nh → ndot
- ew → ndot
- key → ndot
- nrs → ndot
- ndot → d
- d → ngate
- ngate → g
- value → vf
- vf → v
- g → nadd
- v → nadd
- nh → nadd
- nadd → sum
- sum → ncast
- ncast → nout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| nemb | 说明 | ParallelEngramEmbedding：表按行分片，查出 fp8 的行再用 scale 反量化成 bf16<br>24 × 256 = 6144；层 1 的表有 384006168 行，层 14 有 384016682 行 |
| nwkv | 说明 | wkv.weight [25600, 6144]；输出切成 key 与 value 两段 |
| ekey | 说明 | key 转成 fp32，再拆成 hc_mult 份 |
| ewm | 公式 | $weight = q\_weight \odot k\_weight$，两个参数都是 [hc_mult, dim] |
| ewm | 说明 | q_weight 与 k_weight 都是 [4, 5120] 的参数，代码里只以乘积形式出现 |
| nhf | 说明 | 归一化与点积都在 fp32 里算 |
| rstd | 公式 | $rstd = \mathrm{rsqrt}(\mathrm{mean}(h^2) + eps) \cdot \mathrm{rsqrt}(\mathrm{mean}(key^2) + eps)$，h 是 x 转成 fp32 的残差，key 是查表得到的键，eps 是 norm_eps，mean 在最后一维 |
| rstd | 说明 | 两边各按最后一维求均方，逐 token、逐份归一化 |
| ndot | 公式 | $dot = \mathrm{sum}(h \odot weight \odot key, -1) \cdot rstd \cdot dim^{-0.5}$，h、weight、key、rstd 是送进来的张量，dim 是 5120 |
| ndot | 说明 | 归一化点积，再乘 $5120^{-1/2}$ |
| ngate | 公式 | $gate = \mathrm{sigmoid}(\mathrm{sign}(dot)\sqrt{\max(\|dot\|, clamp\_value)})$，dot 是归一化点积，clamp_value 是 1e-6 |
| ngate | 说明 | 先带符号开方再进 sigmoid，与训练 kernel 一致 |
| vf | 说明 | 补一维好按份广播 |
| nadd | 公式 | $h = h + gate \odot value$，h 是残差流，gate 逐份，value 是 4 份共用的值 |
| nadd | 说明 | 门控乘值再加回残差流 |
| ncast | 说明 | 转回输入时的 dtype |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 归一化点积 | $rstd = \mathrm{rsqrt}(\mathrm{mean}(h^2)+eps)\cdot\mathrm{rsqrt}(\mathrm{mean}(key^2)+eps)$，$dot = \mathrm{sum}(h \odot weight \odot key, -1) \cdot rstd \cdot dim^{-0.5}$。h 是 x 转成 fp32 的残差，weight 是 q_weight 与 k_weight 的乘积，key 是查表得到的键，eps 是 norm_eps，dim 是 5120 |
| 公式 | 门控 | $gate = \mathrm{sigmoid}(\mathrm{sign}(dot)\sqrt{\max(|dot|, clamp\_value)})$，dot 是归一化点积，clamp_value 是 1e-6 |
| 公式 | 写入 | $h = h + gate \odot value$，h 是残差流，gate 逐份，value 是 4 份共用的值 |
| 配置 | dim | 5120 |
| 配置 | hc_mult | 4 |
| 配置 | engram_max_ngram_size | 4 |
| 配置 | engram_n_heads | 8 |
| 配置 | engram_head_dim | 256 |
| 配置 | engram_layer_ids | [1, 14] |
| 配置 | engram_num_embeddings | [384006168, 384016682] |
| 配置 | norm_eps | 1e-20 |

## NgramHashState.forward（engram.py:160-184）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ids | 张量 | input_ids | int64 [B, T] | | engram.py:160 |
| cmap | 算子 | self.token_map[input_ids] | int64 [B, T] 不变 |  | engram.py:164 |
| cids | 张量 | compressed | int64 [B, T] | | engram.py:164 |
| wr | 算子 | self.cache[:batch, start_pos : start_pos + seqlen] = compressed | 写压缩 id 缓存 |  | engram.py:167 |
| cache | 缓存 | cache | int64 [max_batch_size, max_seq_len] | 按 max_batch_size 分配，取代码默认值 4；存的是压缩后的 id，不是原始 token id<br>decode 时按 start_pos 往后接 | engram.py:155-157 |
| pos | 算子 | torch.arange(start_pos, start_pos + seqlen).expand(batch, seqlen) | batch, seqlen → int64 [B, T] |  | engram.py:169 |
| p | 张量 | positions | int64 [B, T] | | engram.py:169 |
| blk0 | 算子 | torch.zeros_like(positions, dtype=torch.bool) | int64 [B, T] → bool [B, T] |  | engram.py:170 |
| blk | 张量 | blocked | bool [B, T] | | engram.py:170 |
| gath | 算子 | self.cache[:batch].gather(1, (positions - shift).clamp_min(0)) | [max_batch_size, max_seq_len] → int64 [B, T] |  | engram.py:172 |
| src | 张量 | source | int64 [B, T] | | engram.py:172 |
| upd | 算子 | blocked \| (positions < shift) \| (source == self.DEAD) | 形状不变 |  | engram.py:173 |
| updt | 张量 | blocked | bool [B, T] | | engram.py:173 |
| rep | 算子 | torch.where(blocked, self.pad_id, source) | 形状不变 |  | engram.py:174 |
| tk | 张量 | tokens | int64 [B, T, 4] | 第 k 列是往前看 k 个位置的那个 token | engram.py:174-175 |
| prod | 算子 | tokens.unsqueeze(2) * self.multipliers | [B, T, 4] 与 [2, 4] → [B, T, 2, 4] |  | engram.py:179 |
| pr | 张量 | products | int64 [B, T, 2, 4] | | engram.py:179 |
| roll | 算子 | torch.bitwise_xor(rolling, products[..., i]) | 形状不变 |  | engram.py:180-182 |
| rl | 张量 | rolling | int64 [B, T, 2] | | engram.py:182 |
| mod | 算子 | rolling.unsqueeze(-1) % self.primes[:, i - 1] | int64 [B, T, 2] → int64 [B, T, 2, 8] |  | engram.py:183 |
| hs | 张量 | torch.cat(hashes, dim=-1) | int64 [B, T, 2, 24] | hashes 是 3 个桶组各 [B, T, 2, 8]，沿最后一维拼接；24 = (4 − 1) × 8 | engram.py:183-184 |
| off | 算子 | torch.cat(hashes, dim=-1) + self.offsets | 形状不变 |  | engram.py:184 |
| hashout | 张量 | torch.cat(hashes, dim=-1) + self.offsets | int64 [B, T, 2, 24] | | engram.py:184 |

### 连线

- ids → cmap
- cmap → cids
- cids → wr
- wr → cache
- ids → pos：形状与 device
- pos → p
- p → blk0
- blk0 → blk
- cache → gath
- p → gath
- gath → src
- src → upd
- p → upd
- blk → upd
- upd → updt
- updt → rep
- src → rep
- rep → tk
- tk → prod
- prod → pr
- pr → roll
- roll → rl
- rl → mod
- mod → hs
- hs → off
- off → hashout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| cmap | 公式 | $compressed = token\_map[input\_ids]$，input_ids 是词 id，token_map 是压缩词表的行号 |
| cmap | 说明 | token_map 把每个 token id 映到压缩词表的 id，大小写与空白差异都归一到一个 id<br>压缩词表 99092 项，与 config 的 engram_compressed_vocab_size 一致，所有哈希乘数都由它推出 |
| wr | 说明 | start_pos 为 0，所以从第 0 列起整块写 |
| pos | 说明 | start_pos 到 start_pos + seqlen 的位置，扩成 batch 行 |
| blk0 | 说明 | 标记哪些位置已经不能往前看 |
| gath | 说明 | 往前看 shift 个位置，越界处夹到 0 |
| upd | 说明 | 位置在序列开头之前就算断掉，之后的回看步也不再往前 |
| rep | 说明 | 断掉的位置填 pad_id，即 engram_pad_id（2）经 token_map 映出的那个压缩 id，与训练一致 |
| prod | 公式 | $products = tokens \odot multipliers$，tokens 是各回看步的压缩 id，multipliers 是每层每个回看步的奇数乘数 |
| prod | 说明 | multipliers 每个（Engram 层，回看步）一个奇数乘数，各层用自己的随机种子生成 |
| roll | 公式 | $rolling = rolling \oplus products$，products 是上一行的乘积，$\oplus$ 是按位异或 |
| roll | 说明 | 一次异或一个回看步，第 i 步之后得到的是 i+1 元 n-gram 的哈希 |
| mod | 公式 | $rolling \bmod primes$，primes 是每个（层，桶组，头）的质数 |
| mod | 说明 | 每个（Engram 层，桶组，头）各有一个质数，互不重复；8 是 engram_n_heads，3 个桶组的结果沿最后一维拼成 24 |
| off | 公式 | $hash = (rolling \bmod primes) + offsets$，offsets 是该桶在表里的起始行 |
| off | 说明 | 加上各桶在表里的起始偏移，得到整张表的行号 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 压缩映射 | $compressed = token\_map[input\_ids]$，input_ids 是词 id，token_map 把同形 token 映到同一个压缩 id，压缩词表 99092 项 |
| 公式 | n-gram | 第 t 个位置的回看取 $c_{t-k}$，$k = 0 \ldots 3$；越界就填 pad_id |
| 公式 | 哈希 | $products = tokens \odot multipliers$，再 $rolling = rolling \oplus products$。tokens 是各回看步的压缩 id，multipliers 是乘数，$\oplus$ 是按位异或 |
| 公式 | 桶 | $hash = (rolling \bmod primes) + offsets$，primes 是每个（层，桶组，头）的质数，offsets 是该桶在表里的起始行 |
| 配置 | engram_max_ngram_size | 4 |
| 配置 | engram_n_heads | 8 |
| 配置 | engram_head_dim | 256 |
| 配置 | engram_vocab_size | 16000000 |
| 配置 | engram_compressed_vocab_size | 99092 |
| 配置 | engram_pad_id | 2 |
| 配置 | engram_layer_ids | [1, 14] |

## RMSNorm.forward（model.py:288-293）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| rx | 张量 | x | bf16 [..., d] | d 是被归一化的那一维：主干里 norm、attn_norm、ffn_norm 是 5120，q_norm 是 1280，kv_norm 是 512；压缩器的 norm 是 512；索引器的 k_norm 是 128 | model.py:288 |
| rf32 | 算子 | x.float() | 形状不变 |  | model.py:289-290 |
| rxf | 张量 | x | fp32 [..., d] | | model.py:290 |
| var | 算子 | x.square().mean(-1, keepdim=True) | [..., d] → [..., 1] |  | model.py:291 |
| vr | 张量 | var | fp32 [..., 1] | | model.py:291 |
| nr | 算子 | x * torch.rsqrt(var + self.eps) | 形状不变 |  | model.py:292 |
| xn | 张量 | x | fp32 [..., d] | | model.py:292 |
| rmw | 算子 | self.weight * x | 形状不变 |  | model.py:293 |
| rw | 张量 | self.weight * x | fp32 [..., d] | | model.py:293 |
| rback | 算子 | .to(dtype) | 形状不变 |  | model.py:293 |
| rout | 张量 | (self.weight * x).to(dtype) | bf16 [..., d] | | model.py:293 |

### 连线

- rx → rf32
- rf32 → rxf
- rxf → var
- var → vr
- rxf → nr
- vr → nr
- nr → xn
- xn → rmw
- rmw → rw
- rw → rback
- rback → rout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| rf32 | 说明 | 先记下输入的 dtype，最后转回去 |
| var | 公式 | $var = \mathrm{mean}(x^2)$，x 是转成 fp32 的输入，mean 在最后一维 |
| var | 说明 | 最后一维上的平方均值 |
| nr | 公式 | $x = x \cdot \mathrm{rsqrt}(var + eps)$，var 是上一行的平方均值，eps 是 self.eps |
| nr | 说明 | rsqrt 是平方根的倒数，eps 是 norm_eps |
| rmw | 公式 | $x = weight \odot x$，weight 是 self.weight，x 是上一行归一化后的结果 |
| rmw | 说明 | weight [d]，每个实例一份，如 norm.weight [5120]、q_norm.weight [1280] |
| rback | 说明 | 转回输入时的 dtype |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | RMSNorm | $x = weight \odot x / \sqrt{\mathrm{mean}(x^2) + eps}$，x 是输入，weight 是 self.weight，eps 是 self.eps，mean 在最后一维 |
| 配置 | dim | 5120 |
| 配置 | head_dim | 512 |
| 配置 | index_head_dim | 128 |
| 配置 | norm_eps | 1e-20 |

## ParallelHead.forward（model.py:1008-1017）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| px | 张量 | x | bf16 [B, T, 5120] | 已经过最后的 RMSNorm | model.py:1008 |
| plast | 算子 | x[:, -1] | [B, T, 5120] → [B, 5120] |  | model.py:1010-1011 |
| xp | 张量 | x | bf16 [B, 5120] | | model.py:1011 |
| lin | 算子 | F.linear(x.float(), self.weight) | 5120 → 129280 |  | model.py:1012 |
| logits | 张量 | logits | fp32 [B, 129280] | | model.py:1012 |

### 连线

- px → plast
- plast → xp
- xp → lin
- lin → logits

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| plast | 公式 | $x = x_{:,-1}$，x 是送进 head 的隐状态 |
| plast | 说明 | full_logits 默认 false，只留最后一个位置 |
| lin | 公式 | $logits = x\, weight^\top$，x 是最后一个位置，weight 是 head.weight |
| lin | 说明 | weight [129280, 5120]，fp32 参数，打分在 fp32 里算 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 词表投影 | $logits = x\, weight^\top$，x 是最后一个位置，weight 是 head.weight |
| 配置 | dim | 5120 |
| 配置 | vocab_size | 129280 |
