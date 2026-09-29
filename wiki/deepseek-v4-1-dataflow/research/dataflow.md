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
| ngram | 算子 | self.engram_hash(input_ids, start_pos, engram_mask) | [B, T] → [B, T, 2, 24] | NgramHashState，把每个位置结尾的 n-gram 哈希成表行号<br>倒数第二维是 Engram 层序号，对应 `engram_layer_ids` 的 [1, 14]；24 = (4 − 1) × 8 | model.py:1252 |
| hash | 张量 | engram_hashes | int64 [B, T, 2, 24] | | model.py:1252 |
| emb | 算子 | self.embed(input_ids) | 129280 → 5120 | ParallelEmbedding：embed.weight [129280, 5120]，按 rank 沿词表维切分，单卡即整张 | model.py:1253 |
| h0 | 张量 | h | bf16 [B, T, 5120] | | model.py:1253 |
| hcexp | 算子 | h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1) | [B, T, 5120] → [B, T, 4, 5120] | 把残差流复制成 hc_mult 份，供 Hyper-Connections 用 | model.py:1258 |
| hc | 张量 | h | bf16 [B, T, 4, 5120] | | model.py:1258 |
| eng | 算子 | layer.engram(h, engram_hashes[:, :,<br>layer.engram.layer_hash_index, :], engram_mask) | [B, T, 4, 5120] 不变 | Engram：层 1、14 在进入本层之前跑，吃的是上一层 Block 的输出，把 n-gram 查表的结果按门控加进残差流 | model.py:1262-1263 |
| pm0 | 算子 | make_identity_pre_mix(h, self.hc_mult) | → fp32 [B, T, 4] | 第 0 份置 1、其余置 0，作为进入第 0 层时的 pre_mix | model.py:1260 |
| pre | 张量 | pre_mix | fp32 [B, T, 4] | 进入第 0 层时的初值，此后每层返回的 ffn_pre 都覆盖它 | model.py:1260 |
| layers | 算子 | 40 × Block | [B, T, 4, 5120] 不变 | 层 0 的输入是 hc；层 1、14 在进本层之前先跑 Engram 改写残差流；其余层的输入是上一层 Block 的输出<br>压缩 KV 只在 4 个 source 层（2、8、14、20）算一次，其余 compress_ratio 大于 0 的层从 shared_attn 读<br>层 0、1 的 compress_ratio 为 0，只有滑动窗口 | model.py:1204，1261-1267 |
| hl | 张量 | h | bf16 [B, T, 4, 5120] | | model.py:1267 |
| avg | 算子 | h.mean(dim=2) | [B, T, 4, 5120] → [B, T, 5120] | 进入第 37、38、39 层之前的残差流，按 hc 维取均值 | model.py:1265-1266 |
| mh | 张量 | main_hidden | bf16 [B, T, 15360] | 由 main_hiddens 里 3 个目标层各 [B, T, 5120] 沿最后一维拼接而来，作为第三个返回值交给调用方 | model.py:1259，1266，1271 |
| col | 算子 | layer.hc_pre(h, pre_mix) | [B, T, 4, 5120] → [B, T, 5120] | 用最后一层返回的 pre_mix 把 4 份折叠成一份 | model.py:1268 |
| hc2 | 张量 | h | bf16 [B, T, 5120] | | model.py:1268 |
| fnorm | 算子 | self.norm (RMSNorm) | [B, T, 5120] 不变 | norm.weight [5120] | model.py:1269 |
| hn | 张量 | h | bf16 [B, T, 5120] | | model.py:1269 |
| head | 算子 | self.head (ParallelHead) | 5120 → 129280 | head.weight [129280, 5120]，只取最后一个位置 | model.py:1269 |
| logits | 张量 | logits | fp32 [B, 129280] | | model.py:1269 |
| samp | 算子 | sample(logits, self.temperature) | → [B] | Gumbel-max：概率除以指数分布随机数后取 argmax，等价于按概率采样 | model.py:1270 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 输入嵌入 | $h = \mathrm{Embed}(input\_ids)$，h 是词嵌入，Embed 是 embed，词表 129280、维度 5120 |
| 公式 | 残差流复制 | $h \in \mathbb{R}^{B \times T \times d} \mapsto h \in \mathbb{R}^{B \times T \times 4 \times d}$，把一份残差流复制成 hc_mult = 4 份，4 是 `hc_mult`，d 是 5120 |
| 公式 | 输出 | $\mathrm{logits} = \mathrm{fnorm}(h)\,W^\top$，h 是用最后一层返回的 pre_mix 折叠后的隐状态，fnorm 是最后的 RMSNorm，W 是 head.weight |
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
| mixes_a | 算子 | self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base) | [B, T, 4, 5120] → 三组 fp32 系数 | 一次投影同时给出 pre、post、comb 三套系数<br>hc_attn_fn [24, 20480]、hc_attn_base [24]、hc_attn_scale [3]，都是 fp32；24 = (2 + 4) × 4，20480 = 4 × 5120 | model.py:982 |
| apre | 张量 | attn_pre | fp32 [B, T, 4] | | model.py:982 |
| apost | 张量 | attn_post | fp32 [B, T, 4] | | model.py:982 |
| acomb | 张量 | attn_comb | fp32 [B, T, 4, 4] | | model.py:982 |
| cpre | 算子 | self.hc_pre(x, pre_mix) | [B, T, 4, 5120] → [B, T, 5120] | 按 pre_mix 把 4 份加权求和成 1 份 | model.py:983 |
| x1 | 张量 | x | bf16 [B, T, 5120] | | model.py:983 |
| an | 算子 | self.attn_norm (RMSNorm) | [B, T, 5120] 不变 | attn_norm.weight [5120] | model.py:984 |
| x2 | 张量 | x | bf16 [B, T, 5120] | | model.py:984 |
| attn | 算子 | self.attn(x, start_pos, *attn_args) | [B, T, 5120] 不变 | Attention，本层的窗口 KV 与压缩 KV 都在里面读写；本路径 attn_args 为空 | model.py:985 |
| x3 | 张量 | x | bf16 [B, T, 5120] | | model.py:985 |
| cpost | 算子 | self.hc_post(x, residual, attn_post, attn_comb) | [B, T, 5120] → [B, T, 4, 5120] | 把子层输出按 post 展开成 4 份，再把 residual 按 comb 混进去 | model.py:986 |
| x4 | 张量 | x | bf16 [B, T, 4, 5120] | | model.py:986 |
| resid1 | 张量 | residual | bf16 [B, T, 4, 5120] | 进前馈子层之前的残差流 | model.py:988 |
| mixes_f | 算子 | self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base) | [B, T, 4, 5120] → 三组 fp32 系数 | hc_ffn_fn [24, 20480]、hc_ffn_base [24]、hc_ffn_scale [3]，形状与注意力那组相同 | model.py:989 |
| fpre | 张量 | ffn_pre | fp32 [B, T, 4] | 作为返回值交给下一层当 pre_mix | model.py:989 |
| fpost | 张量 | ffn_post | fp32 [B, T, 4] | | model.py:989 |
| fcomb | 张量 | ffn_comb | fp32 [B, T, 4, 4] | | model.py:989 |
| cpre2 | 算子 | self.hc_pre(x, attn_pre) | [B, T, 4, 5120] → [B, T, 5120] | 用的是注意力子层算出的 pre_mix，不是上一层传来的 | model.py:990 |
| x5 | 张量 | x | bf16 [B, T, 5120] | | model.py:990 |
| fn | 算子 | self.ffn_norm (RMSNorm) | [B, T, 5120] 不变 | ffn_norm.weight [5120] | model.py:991 |
| x6 | 张量 | x | bf16 [B, T, 5120] | | model.py:991 |
| moe_op | 算子 | self.ffn(x, image_mask) | [B, T, 5120] 不变 | MoE，image_mask 为 None | model.py:992 |
| x7 | 张量 | x | bf16 [B, T, 5120] | | model.py:992 |
| cpost2 | 算子 | self.hc_post(x, residual, ffn_post, ffn_comb) | [B, T, 5120] → [B, T, 4, 5120] | | model.py:993 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 混合系数 | $\mathrm{mixes} = W\, \hat{x}$，$\hat{x} = \mathrm{flatten}(x) \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x)^2) + \epsilon)$，x 是本子层前的残差流，flatten 把最后两维拉平，W 是 hc_attn_fn 或 hc_ffn_fn，$\epsilon$ 是 norm_eps |
| 公式 | pre、post、comb | $pre_j = \sigma(m_j s_0 + b_j) + \epsilon$、$post_j = 2\sigma(m_{j+hc} s_1 + b_{j+hc})$、$comb_{j,k} = \mathrm{Sinkhorn}\big(\mathrm{softmax}(m_{j\,hc + k + 2hc}\, s_2 + b_{j\,hc + k + 2hc}) + \epsilon\big)$，hc 是 hc_mult，m 是 mixes，s 是 hc_attn_scale 或 hc_ffn_scale，b 是 hc_attn_base 或 hc_ffn_base，$\sigma$ 是 sigmoid，$\epsilon$ 是 hc_eps；mixes 与 base 都按顺序切成三段用：前 hc 个给 pre、接着 hc 个给 post、最后 hc × hc 个给 comb；Sinkhorn 先做一次行归一化（softmax 后加 $\epsilon$），再做一次列归一化，然后交替做行、列各 19 次，共行 20 次、列 20 次，最后一次落在列方向 |
| 公式 | 折叠 | $x_{out} = \sum_{c} pre_c x_c$，$x_c$ 是 4 份里的第 c 份，pre 是上面算出的 pre |
| 公式 | 展开 | $y_c = post_c\,x + \sum_{c'} comb_{c',c}\,residual_{c'}$，x 是子层输出，residual 是进本子层前的残差流，$comb_{c',c}$ 是第 c' 份残差加给第 c 份输出的权重，求和在第一维上 |
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
| qr_op | 算子 | self.q_norm(self.wq_a(x)) | 5120 → 1280 | wq_a.weight [1280, 5120]、q_norm.weight [1280] | model.py:770 |
| qr | 张量 | qr | bf16 [B, T, 1280] | 索引器拿它当 query 的输入 | model.py:770 |
| qb | 算子 | self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim)) | 1280 → 64 × 512 | wq_b.weight [32768, 1280]，64 × 512 = 32768 | model.py:771 |
| qpre | 张量 | q | bf16 [B, T, 64, 512] | | model.py:771 |
| rope_q | 算子 | apply_rotary_emb(q[..., -rd:], freqs_cis) | 末 64 维，形状不变 | 只旋转每头 512 维里的后 64 维，前 448 维不动；写在原地 | model.py:772 |
| q | 张量 | q | bf16 [B, T, 64, 512] | | model.py:772 |
| wkv_win | 算子 | self.kv_norm(self.wkv(x)) | 5120 → 512 | wkv.weight [512, 5120]、kv_norm.weight [512] | model.py:705 |
| wkv_o | 张量 | kv | bf16 [B, T, 512] | | model.py:705 |
| wrope | 算子 | apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis) | 末 64 维，形状不变 | 和 query 用同一份 freqs_cis | model.py:706 |
| wkv_r | 张量 | kv | bf16 [B, T, 512] | | model.py:706 |
| wquant | 算子 | act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True) | 形状不变 | 每 32 个通道一个 E8M0 scale，原地量化再反量化；量化按 fp8 精度（压缩 KV 走 fp4）<br>量化覆盖整条 512 维，含已经加过 RoPE 的末 64 维 | model.py:707 |
| wkv_q | 张量 | kv | bf16 [B, T, 512] | 存的是 fp8 量化后再反量化的值 | model.py:707 |
| wwrite | 算子 | self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1) | 写环形缓冲 | cutoff = seqlen % win，只留最后 win 个 token，尾巴绕回环首 | model.py:712-715 |
| wchunk | 张量 | window_kv | bf16 [B, T, 512] | prefill 就是本次这块 KV | model.py:716 |
| widx | 算子 | get_window_topk_idxs(win, bsz, seqlen, start_pos) | → int32 [B, T, 128] | 每个 query 能看的窗口槽位，−1 是空槽；每个 query 各一行，只看到自己的因果窗口 | model.py:720 |
| widxs | 张量 | topk_idxs | int32 [B, T, 128] | | model.py:720 |
| wcache | 缓存 | window_kv_cache | [max_batch_size, 128, 512] | 每层一份，按 max_batch_size 分配；prefill 整块覆盖，decode 时改写 start_pos % 128 那一槽 | model.py:663-668 |
| comp_op | 算子 | self.compressor(x, start_pos) | 5120 → [B, T/r, 512] | 只有 4 个 source 层（2、8、14、20）算；其余 compress_ratio 大于 0 的层 latent 为 None，直接读已有缓存；层 0、1 的压缩比为 0，整段压缩都不走 | model.py:747 |
| lat | 张量 | latent | bf16 [B, T/r, 512] | 压缩器输出的潜在量，还没加 RoPE；r 是这一层的 compress_ratio | model.py:747 |
| cidx_op | 算子 | self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len) | → int32 [B, T, 512] | 本层是 index source（2、8、14、20、24、28、32、36）就跑自己的 Indexer，其余 compress_ratio 大于 0 的层直接复用 shared_attn.topk_idxs<br>offset 取窗口 KV 的长度（本次是 T），compress_len 是 (start_pos + T) / r | model.py:750 |
| cidxs | 张量 | idxs | int32 [B, T, 512] | 压缩条目下标，已经偏移到拼起来的 KV 里的位置 | model.py:750 |
| cf | 算子 | self.freqs_cis[: seqlen - seqlen % ratio : ratio] | → [T/r, 32] | 取压缩位置的频率，一个 latent 顶替一整组，第 g 组取位置 g × r | model.py:753-757 |
| cfq | 张量 | freqs | 复数 fp32 [T/r, 32] | | model.py:753-757 |
| crope | 算子 | apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs) | 末 64 维，形状不变 | 用的是压缩位置的频率，不是普通位置 | model.py:758 |
| lat_r | 张量 | latent | bf16 [B, T/r, 512] | | model.py:758 |
| cquant | 算子 | fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn) | 形状不变 | E2M1 格式，每 16 个通道一个 E4M3 scale，原地量化再反量化 | model.py:760 |
| lat_q | 张量 | latent | bf16 [B, T/r, 512] | 存的是 fp4 量化后再反量化的值 | model.py:760 |
| cwrite | 算子 | self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent | 写压缩缓存 | start_pos 为 0，所以从第 0 行起整块写 | model.py:761 |
| ccache | 缓存 | compress_kv_cache | [max_batch_size, max_seq_len / r, 512] | 2、8、14、20 各有一份，按 max_batch_size 分配；每个 source 层写自己这一份并把它发布给 shared_attn，它之后到下一个 kv source 层之前的层读的是这一份；decode 时每组填满才写一行 | model.py:669-679 |
| ckv | 张量 | compress_kv | bf16 [B, T/r, 512] | 从 compress_kv_cache 读出的、本次可见的部分 | model.py:763 |
| cat | 算子 | torch.cat([kv, compress_kv], dim=1) | 两份 KV 沿位置维拼起来 | 窗口在前、压缩在后 | model.py:777 |
| kvc | 张量 | kv | bf16 [B, T + T/r, 512] | | model.py:777 |
| cat2 | 算子 | torch.cat([topk_idxs, compress_idxs], dim=-1) | 两份下标拼起来 | 压缩下标在 _compress_kv 里已经加过窗口长度 T | model.py:778 |
| tki | 张量 | topk_idxs | int32 [B, T, 128 + 512] | | model.py:778 |
| core | 算子 | sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale) | q [B, T, 64, 512] 对 kv [B, T + T/r, 512] → [B, T, 64, 512] | 单 KV 头，64 个查询头看同一份 KV<br>分母多一项 $e^{\mathrm{attn\_sink} - m}$，attn_sink 是可学习参数 [64]<br>下标为 −1 的槽位分子分母都不贡献 | model.py:780 |
| o0 | 张量 | o | bf16 [B, T, 64, 512] | | model.py:780 |
| unrope | 算子 | apply_rotary_emb(o[..., -rd:], freqs_cis, True) | 末 64 维，形状不变 | 用共轭频率对输出做一次逆旋转，抵消 KV 里已加 RoPE 的键带进注意力输出的那部分旋转，使输出回到与键缓存同一套表示下 | model.py:781 |
| o1 | 张量 | o | bf16 [B, T, 64, 512] | | model.py:781 |
| grp | 算子 | o.view(bsz, seqlen, self.n_local_groups, -1) | [B, T, 64, 512] → [B, T, 8, 4096] | 每 8 个头一组，8 × 512 = 4096 | model.py:785 |
| og | 张量 | o | bf16 [B, T, 8, 4096] | | model.py:785 |
| woa | 算子 | self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1) | → [8, 1024, 4096] | wo_a.weight [8192, 4096]，8192 = 8 × 1024 | model.py:786 |
| wav | 张量 | wo_a | bf16 [8, 1024, 4096] | | model.py:786 |
| eins | 算子 | torch.einsum("bsgd,grd->bsgr", o, wo_a) | [B, T, 8, 4096] 与 [8, 1024, 4096] → [B, T, 8, 1024] | 每组只和本组的权重相乘，等效块对角矩阵 | model.py:787 |
| ol | 张量 | o | bf16 [B, T, 8, 1024] | | model.py:787 |
| wb | 算子 | self.wo_b(o.flatten(2)) | 8192 → 5120 | wo_b.weight [5120, 8192] | model.py:788 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 频率 | $\theta_j = \mathrm{base}^{-2j/64}$，j = 0 … 31，base 在压缩比为 0 的层是 rope_theta、其余层是 compress_rope_theta，64 是 rope_head_dim |
| 公式 | YaRN | $\theta_j' = (\theta_j/16)\,\rho_j + \theta_j\,(1 - \rho_j)$，16 是 rope_factor，$\rho_j = \mathrm{clamp}\big((j - low)/(high - low), 0, 1\big)$ 是随 j 递增的斜坡，j 不大时 $\rho_j$ 为 0、保留原频率，j 变大后逐步除以 16，low 是 corrected_dim(beta_fast) 向下取整、high 是 corrected_dim(beta_slow) 向上取整；只对压缩比大于 0 的层生效 |
| 公式 | query 低秩投影 | $q = W_{qb}\,\mathrm{RMSNorm}(W_{qa}\,x)$，x 是本层输入，$W_{qa}$ 是 wq_a、$W_{qb}$ 是 wq_b，RMSNorm 是 q_norm |
| 公式 | 窗口 KV | $k = \mathrm{RMSNorm}(W_{kv}\,x)$，$W_{kv}$ 是 wkv，RMSNorm 是 kv_norm；x 先过 wkv 再过 kv_norm；每头 512 维里只给后 64 维加 RoPE |
| 公式 | 压缩 KV 的位置 | 第 g 个 latent 顶替第 $g r$ 到 $(g+1)r - 1$ 个 token，所以它的 RoPE 用位置 $g r$，r 是 compress_ratio |
| 公式 | 稀疏注意力 | $o = \dfrac{\sum_{t \in \mathcal{T}} e^{\,q \cdot k_t / \sqrt{512} - m}\, v_t}{\sum_{t \in \mathcal{T}} e^{\,q \cdot k_t / \sqrt{512} - m} + e^{\,\mathrm{sink} - m}}$，$\mathcal{T}$ 是选中的槽位，单 KV 头所以 k 和 v 是同一份，m 是选中槽位里的最大分数，sink 是 attn_sink，512 是 head_dim；分子、分母里每一项减的都是同一个 m |
| 公式 | 输出低秩投影 | $x_{out} = W_{ob}\,\mathrm{einsum}(o, W_{oa})$，$W_{oa}$ 按 8 组摊成块对角，$W_{ob}$ 是 wo_b，o 是注意力输出 |
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
| cf32 | 算子 | x.float() | [B, T, 5120] 不变 | 池化与门控都在 fp32 里算 | model.py:464 |
| cxf | 张量 | x | fp32 [B, T, 5120] | | model.py:464 |
| proj | 算子 | self.wkv(x), self.wgate(x) | 5120 → 2 × [B, T, 512] | wkv.weight [512, 5120] 与 wgate.weight [512, 5120]，压缩比大于 1 时两个权重都是 fp32 | model.py:465 |
| kv0 | 张量 | kv | fp32 [B, T, 512] | 逐 token 的值投影 | model.py:465 |
| csc | 张量 | score | fp32 [B, T, 512] | 逐 token 的门控打分 | model.py:465 |
| pool | 算子 | kv.unflatten(1, (-1, ratio))、score.unflatten(1, (-1, ratio)) 与 (kv * score.softmax(dim=2)).sum(dim=2) | [B, T, 512] → [B, T/r, 512] | 把相邻 r 个 token 当成一组，组内用 score 的 softmax 做加权求和 | model.py:473-475 |
| kvg | 张量 | kv | fp32 [B, T/r, 512] | | model.py:475 |
| cnorm | 算子 | self.norm(kv.to(dtype)) | 形状不变 | 池化结果先转回 bf16 再 RMSNorm，layers.{i}.attn.compressor.norm.weight [512] | model.py:485 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 池化 | $c_j = \sum_{t=jr}^{(j+1)r-1} \mathrm{softmax}_{t'}\big(s_{t'}\big)\, v_t$，j 是第 j 组，r 是 compress_ratio，$v_t$ 是 kv 里第 t 个 token 的值投影，$s_t$ 是 score 里第 t 个，softmax 在这一组的 r 个 token 上取 |
| 公式 | 输出 | $\mathrm{RMSNorm}(c)$，c 是池化结果，先转回 bf16 再算；RMSNorm 是 self.norm |
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
| fpos | 算子 | self.freqs_cis[: seqlen - seqlen % ratio : ratio] | → [T/r, 32] | 取压缩位置的频率，第 g 组取位置 g × r | model.py:539-543 |
| fq | 张量 | freqs | 复数 fp32 [T/r, 32] | | model.py:540-543 |
| kw | 算子 | self.k_norm(self.wk(latent)) | 512 → 128 | wk.weight [128, 512]、k_norm.weight [128] | model.py:544 |
| k0 | 张量 | k | bf16 [B, T/r, 128] | | model.py:544 |
| krope | 算子 | apply_rotary_emb(k[..., -rd:], freqs) | 末 64 维，形状不变 | 128 维里的最后 64 维加 RoPE | model.py:545 |
| kr | 张量 | k | bf16 [B, T/r, 128] | | model.py:545 |
| kq | 算子 | fp4_act_quant(k, fp4_block_size, True) | 形状不变 | E2M1 格式，每 32 个通道一个 E8M0 scale，原地量化再反量化 | model.py:546 |
| kf | 张量 | k | bf16 [B, T/r, 128] | 存的是 fp4 量化后再反量化的值 | model.py:546 |
| kwrite | 算子 | self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k | 写入索引键缓存 | start_pos 为 0，从第 0 行起整块写；写完把这份缓存发布到 shared_attn.index_k，后面的层直接读 | model.py:547-548 |
| kcache | 缓存 | k_cache | [max_batch_size, max_seq_len / r, 128] | 2、8、14、20 各有一份，按 max_batch_size 分配；写完发布给 shared_attn，它之后到下一个 kv source 层之前的索引器读的是这一份 | model.py:520-525 |
| qop | 算子 | self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim)) | 1280 → 32 × 128 | wq_b.weight [4096, 1280]，32 × 128 = 4096 | model.py:550 |
| iq0 | 张量 | q | bf16 [B, T, 32, 128] | | model.py:550 |
| qrope | 算子 | apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos]) | 末 64 维，形状不变 | 用的是普通位置，不是压缩位置 | model.py:551 |
| iqr | 张量 | q | bf16 [B, T, 32, 128] | | model.py:551 |
| iq_q | 算子 | fp4_act_quant(q, fp4_block_size, True) | 形状不变 | 索引器的 query 也量化成 fp4 | model.py:552 |
| iq | 张量 | q | bf16 [B, T, 32, 128] | 存的是 fp4 量化后再反量化的值 | model.py:552 |
| ik | 张量 | index_k | bf16 [B, T/r, 128] | 从 shared_attn.index_k 读出的、本次可见的部分 | model.py:554 |
| wproj | 算子 | self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5) | 5120 → 32 | weights_proj.weight [32, 5120]；乘的常数是 $128^{-1/2} \cdot 32^{-1/2}$ | model.py:555 |
| iwt | 张量 | weights | bf16 [B, T, 32] | | model.py:555 |
| idot | 算子 | torch.einsum("bshd,btd->bsht", q, index_k) | [B, T, 32, 128] 与 [B, T/r, 128] → [B, T, 32, T/r] | 每个索引头对每个压缩位置记一个点积 | model.py:556 |
| sc0 | 张量 | index_score | bf16 [B, T, 32, T/r] | | model.py:556 |
| red | 算子 | (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2) | [B, T, 32, T/r] → [B, T, T/r] | 先过 ReLU，再按头用 weights 加权求和 | model.py:557 |
| isc | 张量 | index_score | bf16 [B, T, T/r] | | model.py:557 |
| lens | 算子 | (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1) | → [T, 1] | 第 i 个 query 能看到 ⌊(i+1)/r⌋ 个压缩条目 | model.py:563-564 |
| lv | 张量 | compress_lens | int64 [T, 1] | | model.py:563-564 |
| imask | 算子 | index_score.masked_fill_(torch.arange(seqlen // ratio) >= compress_lens, -torch.inf) | 形状不变 | 本次还看不到的压缩条目置 −inf，这一步原地改 | model.py:565 |
| iscm | 张量 | index_score | bf16 [B, T, T/r] | | model.py:565 |
| topk | 算子 | index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values | → [B, T, 512] | topk = min(512, T/r)，前提里 T 足够长所以就是 512；取分数最高的 512 个，再按位置顺序排回来 | model.py:578-579 |
| isel | 张量 | idxs | int64 [B, T, 512] | | model.py:579 |
| shift | 算子 | torch.where(idxs < compress_lens, idxs + offset, -1) | 形状不变 | 不小于可达数的置 −1，其余加上 offset 变成拼起来的 KV 里的下标 | model.py:580 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 索引打分 | $I_{q,j} = \sum_h w_{q,h} \cdot \mathrm{ReLU}(q_h \cdot k_j)$，h 从 0 到 31 是索引头，$q_h$ 是第 h 个量化成 fp4 的索引 query，$k_j$ 是第 j 个压缩条目的 fp4 索引键，$w_{q,h}$ 由 weights_proj 给出并乘上 $128^{-1/2} \cdot 32^{-1/2}$ |
| 公式 | 可达数 | 第 i 个 query 能看到 $\lfloor (i+1)/r \rfloor$ 个压缩条目，r 是 compress_ratio；还看不到的那些置 $-\infty$ |
| 公式 | 输出 | $idxs = \mathrm{sort}\big(\mathrm{topk}_{512}(I)\big)$，不小于可达数的置 −1，其余加上 offset 变成拼起来的 KV 里的下标 |
| 配置 | index_n_heads | 32 |
| 配置 | index_head_dim | 128 |
| 配置 | index_topk | 512 |
| 配置 | rope_head_dim | 64 |

## MoE.forward（model.py:889-904）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| mx | 张量 | x | bf16 [B, T, 5120] | 前馈子层的输入，已经过 ffn_norm | model.py:890 |
| flat | 算子 | x.view(-1, self.dim) | [B, T, 5120] → [B×T, 5120] | 把批次与序列两维拉平，MoE 按 token 处理 | model.py:891 |
| mxf | 张量 | x | bf16 [B×T, 5120] | | model.py:891 |
| gate_op | 算子 | self.gate(x, None) | [B×T, 5120] → 权重 [B×T, 6] 与专家下标 [B×T, 6] | Gate：打分、加偏置选专家、在 top-6 内归一化、乘 route_scale；image_mask 为 None，源码在这个分支上就传 None | model.py:892 |
| gw | 张量 | weights | fp32 [B×T, 6] | | model.py:892 |
| gi | 张量 | indices | int64 [B×T, 6] | | model.py:892 |
| y0 | 算子 | torch.zeros_like(x, dtype=torch.float32) | → fp32 [B×T, 5120] | 累加器，路由专家的结果都往它上面加 | model.py:893 |
| y | 张量 | y | fp32 [B×T, 5120] | | model.py:893 |
| mpick | 算子 | torch.where(indices == i) | → 行下标与 top 位置 | 挑出分给第 i 个专家的 token，384 个专家各跑一遍；没分到 token 的专家整个跳过 | model.py:899 |
| pidx | 张量 | idx, top | int64 [n] 与 int64 [n] | n 是分到第 i 个专家的 token 数 | model.py:899 |
| mexp | 算子 | expert(x[idx], weights[idx, top, None]) | 5120 → 5120 | 每个 token 只过它被分到的那个专家，路由权重按自己的 top 位置取 | model.py:900 |
| eout | 张量 | expert(x[idx], weights[idx, top, None]) | bf16 [n, 5120] | | model.py:900 |
| acc | 算子 | y[idx] += expert(...) | fp32 [B×T, 5120] 上累加 | 一个 token 落在 top-6 的哪几个专家上就加几次 | model.py:900 |
| y2 | 张量 | y | fp32 [B×T, 5120] | | model.py:900 |
| shared | 算子 | self.shared_experts(x) | 5120 → 5120 | shared_experts 是一个 Expert，每个 token 都过，它的三个权重按主干精度 fp8 存储 | model.py:903 |
| sv | 张量 | self.shared_experts(x) | bf16 [B×T, 5120] | | model.py:903 |
| madd | 算子 | y += self.shared_experts(x) | 逐元素相加 | 路由专家的和再加上共享专家的输出 | model.py:903 |
| y3 | 张量 | y | fp32 [B×T, 5120] | | model.py:903 |
| mback | 算子 | y.type_as(x).view(shape) | [B×T, 5120] → bf16 [B, T, 5120] | 转回输入的 dtype 与形状 | model.py:904 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 前馈 | $y = \sum_{e \in \mathrm{top6}} \mathrm{Expert}_e(x) + \mathrm{Shared}(x)$，x 是拉平后的隐状态，Expert 在内部再乘路由权重 w，Shared 是共享专家 |
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
| lg | 算子 | linear(x.float(), self.weight.float()) / self.gate_temp | 5120 → 384 | weight [384, 5120]，打分在 fp32 里算<br>gate_temp 取默认值 1.0，config.json 里没有这一项 | model.py:811 |
| s0 | 张量 | scores | fp32 [B×T, 384] | | model.py:811 |
| act | 算子 | F.softplus(scores).sqrt() | 形状不变 | score_func 是 sqrtsoftplus：先 softplus 再开方<br>$\mathrm{softplus}(z) = \ln(1 + e^z)$ | model.py:817 |
| s | 张量 | scores | fp32 [B×T, 384] | | model.py:817 |
| bv | 张量 | self.bias | fp32 [384] | 可学习参数，只在挑专家时加到分数上，不改权重的值；image_mask 为 None 时源码直接用 `self.bias` | model.py:818 |
| gsel | 算子 | (scores + bias).topk(self.topk, dim=-1)[1] | → [B×T, 6] | 偏置只参与挑专家，不改权重 | model.py:822 |
| gi | 张量 | indices | int64 [B×T, 6] | | model.py:822 |
| gth | 算子 | scores.gather(1, indices) | → [B×T, 6] | 权重取自没加偏置的 scores | model.py:823 |
| w0 | 张量 | weights | fp32 [B×T, 6] | | model.py:823 |
| nrm | 算子 | weights /= weights.sum(dim=-1, keepdim=True) + 1e-20 | 形状不变 | norm_topk_prob 为 true 时在 top-6 内归一化；加的是 1e-20，不是 norm_eps | model.py:824-825 |
| w1 | 张量 | weights | fp32 [B×T, 6] | | model.py:825 |
| grs | 算子 | weights *= self.route_scale | 形状不变 | route_scale 是 1.5 | model.py:826 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 打分 | $\mathrm{score} = \sqrt{\mathrm{softplus}(x W^\top)}$，x 是拉平后的隐状态，W 是 gate 的 weight，$\mathrm{softplus}(z) = \ln(1 + e^z)$ |
| 公式 | 选专家 | $\mathrm{idx} = \mathrm{topk}_6(\mathrm{score} + b)$，b 是 bias，它只影响选哪几个专家 |
| 公式 | 权重 | $w = \dfrac{\mathrm{score}_{\mathrm{idx}}}{\sum \mathrm{score}_{\mathrm{idx}} + 10^{-20}} \times 1.5$，求和只在这 6 个专家上，1.5 是 route_scale |
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
| g1 | 算子 | self.w1(x).float() | 5120 → 2304 | 路由专家的 w1.weight 存成 [2304, 2560] 的 fp4，每 32 个通道一个 E8M0 scale；共享专家走这张图时是 [2304, 5120] 的 fp8<br>先转 fp32 再夹 | model.py:843 |
| eact | 张量 | gate | fp32 [n, 2304] | | model.py:843 |
| u1 | 算子 | self.w3(x).float() | 5120 → 2304 | w3.weight 与 w1 同一种存法 | model.py:844 |
| up | 张量 | up | fp32 [n, 2304] | | model.py:844 |
| cl | 算子 | torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit) | 形状不变 | up 两头都夹；swiglu_limit 是 10.0<br>夹是为了让后面的 fp8、fp4 激活不越界 | model.py:846 |
| up2 | 张量 | up | fp32 [n, 2304] | | model.py:846 |
| cl2 | 算子 | torch.clamp(gate, max=self.swiglu_limit) | 形状不变 | gate 只夹上界 | model.py:847 |
| eact2 | 张量 | gate | fp32 [n, 2304] | | model.py:847 |
| silu | 算子 | F.silu(gate) * up | 形状不变 | SiLU 的结果逐元素乘 up | model.py:848 |
| hm | 张量 | F.silu(gate) * up | fp32 [n, 2304] | | model.py:848 |
| emw | 算子 | weights * x | 形状不变 | 路由权重逐 token 乘上去 | model.py:850 |
| hw | 张量 | weights * x | fp32 [n, 2304] | | model.py:850 |
| w2 | 算子 | self.w2(x.to(dtype)) | 2304 → 5120 | 路由专家的 w2.weight 存成 [5120, 1152] 的 fp4；共享专家是 [5120, 2304] 的 fp8 | model.py:851 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | SwiGLU | $\mathrm{Expert}(x) = W_2\Big(w \odot \big(\mathrm{silu}(\mathrm{clamp}_{上}(W_1 x)) \odot \mathrm{clamp}_{\pm}(W_3 x)\big)\Big)$，x 是输入，$W_1$ 是 w1、$W_2$ 是 w2、$W_3$ 是 w3，$w$ 是路由权重，$\odot$ 是逐元素相乘 |
| 公式 | 夹取 | $\mathrm{clamp}_{\pm}(z) = \min(\max(z, -10), 10)$、$\mathrm{clamp}_{上}(z) = \min(z, 10)$，10 是 swiglu_limit；共享专家不乘 $w$ |
| 配置 | dim | 5120 |
| 配置 | moe_inter_dim | 2304 |
| 配置 | swiglu_limit | 10.0 |

## Engram.forward（model.py:350-365）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| nx | 张量 | x | bf16 [B, T, 4, 5120] | 本层的残差流，4 份 | model.py:350 |
| hash_ids | 张量 | hash_ids | int64 [B, T, 24] | 本层的 n-gram 哈希行号，24 = (4 − 1) × 8 | model.py:350 |
| nemb | 算子 | self.embed(hash_ids).flatten(-2) | [B, T, 24] → [B, T, 6144] | ParallelEngramEmbedding：表按行分片，查出 fp8 的行再用 scale 反量化成 bf16<br>24 × 256 = 6144；层 1 的表有 384006168 行，层 14 有 384016682 行 | model.py:353，296-325 |
| ev | 张量 | self.embed(hash_ids).flatten(-2) | bf16 [B, T, 6144] | | model.py:353 |
| nwkv | 算子 | self.wkv(...) | 6144 → 25600 | wkv.weight [25600, 6144]；输出切成 key 与 value 两段 | model.py:353-354 |
| nkv | 张量 | kv | bf16 [B, T, 25600] | | model.py:353 |
| key0 | 张量 | key | bf16 [B, T, 20480] | 20480 = 4 × 5120，每份残差流一个键 | model.py:354 |
| value | 张量 | value | bf16 [B, T, 5120] | 4 份残差流共用同一个值 | model.py:354 |
| ekey | 算子 | key.float().unflatten(-1, (self.hc_mult, self.dim)) | [B, T, 20480] → fp32 [B, T, 4, 5120] | | model.py:355 |
| key | 张量 | key | fp32 [B, T, 4, 5120] | | model.py:355 |
| ewm | 算子 | self.q_weight.float() * self.k_weight.float() | [4, 5120] 逐元素相乘 | q_weight 与 k_weight 都是 [4, 5120] 的参数，代码里只以乘积形式出现 | model.py:356 |
| ew | 张量 | weight | fp32 [4, 5120] | | model.py:356 |
| nhf | 算子 | x.float() | [B, T, 4, 5120] 不变 | 归一化与点积都在 fp32 里算 | model.py:357 |
| nh | 张量 | h | fp32 [B, T, 4, 5120] | | model.py:357 |
| rstd | 算子 | torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps) | → fp32 [B, T, 4] | 两边各按最后一维求均方，逐 token、逐份归一化 | model.py:359 |
| nrs | 张量 | rstd | fp32 [B, T, 4] | | model.py:359 |
| ndot | 算子 | (h * weight * key).sum(-1) * rstd * self.dim**-0.5 | [B, T, 4] | 归一化点积，再乘 $5120^{-1/2}$ | model.py:360 |
| d | 张量 | dot | fp32 [B, T, 4] | | model.py:360 |
| ngate | 算子 | torch.sigmoid(<br>torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)) | 形状不变 | 先带符号开方再进 sigmoid，与训练 kernel 一致 | model.py:362 |
| g | 张量 | gate | fp32 [B, T, 4] | | model.py:362 |
| vf | 算子 | value.float().unsqueeze(-2) | [B, T, 5120] → fp32 [B, T, 1, 5120] | 补一维好按份广播 | model.py:365 |
| v | 张量 | value.float().unsqueeze(-2) | fp32 [B, T, 1, 5120] | | model.py:365 |
| nadd | 算子 | h + gate.unsqueeze(-1) * value.float().unsqueeze(-2) | 形状不变 | 门控乘值再加回残差流 | model.py:365 |
| sum | 张量 | h + gate.unsqueeze(-1) * value.float().unsqueeze(-2) | fp32 [B, T, 4, 5120] | | model.py:365 |
| ncast | 算子 | .to(x.dtype) | 形状不变 | 转回输入时的 dtype | model.py:365 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 归一化点积 | $d = \big(\sum_j h_j w_j key_j\big) \cdot \mathrm{rstd} \cdot 5120^{-1/2}$，$\mathrm{rstd} = \big(\mathrm{mean}_j h_j^2 + \epsilon\big)^{-1/2}\big(\mathrm{mean}_j key_j^2 + \epsilon\big)^{-1/2}$，h 是本层残差流，key 是哈希查表投影出的键，w 是 q_weight 与 k_weight 的逐元素积，$\epsilon$ 是 norm_eps，5120 是 dim |
| 公式 | 门控 | $gate = \sigma\big(\mathrm{sign}(d)\sqrt{\max(|d|, 10^{-6})}\big)$，$\sigma$ 是 sigmoid，$10^{-6}$ 是 clamp_value |
| 公式 | 写入 | $x \leftarrow h + gate \odot value$，value 是投影出的值，4 份残差流共用同一个 value，只有 gate 逐份不同 |
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
| cmap | 算子 | self.token_map[input_ids] | → int64 [B, T] | token_map 把每个 token id 映到压缩词表的 id，大小写与空白差异都归一到一个 id<br>压缩词表 99092 项，与 config 的 engram_compressed_vocab_size 一致，所有哈希乘数都由它推出 | engram.py:164 |
| cids | 张量 | compressed | int64 [B, T] | | engram.py:164 |
| wr | 算子 | self.cache[:batch, start_pos : start_pos + seqlen] = compressed | 写压缩 id 缓存 | start_pos 为 0，所以从第 0 列起整块写 | engram.py:167 |
| cache | 缓存 | cache | int64 [max_batch_size, max_seq_len] | 按 max_batch_size 分配，取代码默认值 4；存的是压缩后的 id，不是原始 token id<br>decode 时按 start_pos 往后接 | engram.py:155-157 |
| pos | 算子 | torch.arange(start_pos, start_pos + seqlen).expand(batch, seqlen) | → int64 [B, T] | | engram.py:169 |
| p | 张量 | positions | int64 [B, T] | | engram.py:169 |
| blk0 | 算子 | torch.zeros_like(positions, dtype=torch.bool) | → bool [B, T] | 标记哪些位置已经不能往前看 | engram.py:170 |
| blk | 张量 | blocked | bool [B, T] | | engram.py:170 |
| gath | 算子 | self.cache[:batch].gather(1, (positions - shift).clamp_min(0)) | → int64 [B, T] | 往前看 shift 个位置，越界处夹到 0 | engram.py:172 |
| src | 张量 | source | int64 [B, T] | | engram.py:172 |
| upd | 算子 | blocked \| (positions < shift) \| (source == self.DEAD) | 形状不变 | 位置在序列开头之前就算断掉，之后的回看步也不再往前 | engram.py:173 |
| updt | 张量 | blocked | bool [B, T] | | engram.py:173 |
| rep | 算子 | torch.where(blocked, self.pad_id, source) | 形状不变 | 断掉的位置填 pad_id，即 engram_pad_id（2）经 token_map 映出的那个压缩 id，与训练一致 | engram.py:174 |
| tk | 张量 | tokens | int64 [B, T, 4] | 第 k 列是往前看 k 个位置的那个 token | engram.py:174-175 |
| prod | 算子 | tokens.unsqueeze(2) * self.multipliers | [B, T, 4] 与 [2, 4] → [B, T, 2, 4] | multipliers 每个（Engram 层，回看步）一个奇数乘数，各层用自己的随机种子生成 | engram.py:179 |
| pr | 张量 | products | int64 [B, T, 2, 4] | | engram.py:179 |
| roll | 算子 | torch.bitwise_xor(rolling, products[..., i]) | 形状不变 | 一次异或一个回看步，第 i 步之后得到的是 i+1 元 n-gram 的哈希 | engram.py:180-182 |
| rl | 张量 | rolling | int64 [B, T, 2] | | engram.py:182 |
| mod | 算子 | rolling.unsqueeze(-1) % self.primes[:, i - 1] | → [B, T, 2, 8] | 每个（Engram 层，桶组，头）各有一个质数，互不重复；8 是 engram_n_heads，3 个桶组的结果沿最后一维拼成 24 | engram.py:183 |
| hs | 张量 | torch.cat(hashes, dim=-1) | int64 [B, T, 2, 24] | hashes 是 3 个桶组各 [B, T, 2, 8]，沿最后一维拼接；24 = (4 − 1) × 8 | engram.py:183-184 |
| off | 算子 | torch.cat(hashes, dim=-1) + self.offsets | 形状不变 | 加上各桶在表里的起始偏移，得到整张表的行号 | engram.py:184 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 压缩映射 | $c_t = \mathrm{map}(input\_ids_t)$，map 是 token_map；归一化后同形的 token 映到同一个 id，压缩词表有 99092 项 |
| 公式 | n-gram | 第 t 个位置的回看取 $c_{t-k}$，$k = 0 \ldots 3$；越界就填 pad_id |
| 公式 | 哈希 | $h^{(k)} = c_t m_0 \oplus c_{t-1} m_1 \oplus \cdots \oplus c_{t-k} m_k$，$\oplus$ 是按位异或，$m$ 是 multipliers，第 k 步（k = 1、2、3）得到的是 k + 1 元 n-gram，落在第 k − 1 个桶组里（从 0 数） |
| 公式 | 桶 | $id = h^{(k)} \bmod p_{k-1,\,head} + \mathrm{offset}$，每个（Engram 层，桶组，头）各有一个质数 $p$，互不重复，$p_{k-1,\,head}$ 取第 k − 1 个桶组里第 head 个头的那个，offset 是该桶在表里的起始行 |
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
| rf32 | 算子 | x.float() | 形状不变 | 先记下输入的 dtype，最后转回去 | model.py:289-290 |
| rxf | 张量 | x | fp32 [..., d] | | model.py:290 |
| var | 算子 | x.square().mean(-1, keepdim=True) | d → 1 | 最后一维上的平方均值 | model.py:291 |
| vr | 张量 | var | fp32 [..., 1] | | model.py:291 |
| nr | 算子 | x * torch.rsqrt(var + self.eps) | 形状不变 | rsqrt 是平方根的倒数，eps 是 norm_eps | model.py:292 |
| xn | 张量 | x | fp32 [..., d] | | model.py:292 |
| rmw | 算子 | self.weight * x | 形状不变 | weight [d]，每个实例一份，如 norm.weight [5120]、q_norm.weight [1280] | model.py:293 |
| rw | 张量 | self.weight * x | fp32 [..., d] | | model.py:293 |
| rback | 算子 | .to(dtype) | 形状不变 | 转回输入时的 dtype | model.py:293 |
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

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | RMSNorm | $y = w \odot h / \sqrt{\mathrm{mean}(h^2) + \epsilon}$，h 是输入，y 是返回值，mean 在最后一维上取，$\epsilon$ 是 norm_eps，w 是 weight，$\odot$ 是逐元素相乘 |
| 配置 | dim | 5120 |
| 配置 | head_dim | 512 |
| 配置 | index_head_dim | 128 |
| 配置 | norm_eps | 1e-20 |

## ParallelHead.forward（model.py:1008-1017）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| px | 张量 | x | bf16 [B, T, 5120] | 已经过最后的 RMSNorm | model.py:1008 |
| plast | 算子 | x[:, -1] | [B, T, 5120] → [B, 5120] | full_logits 默认 false，只留最后一个位置 | model.py:1010-1011 |
| xp | 张量 | x | bf16 [B, 5120] | | model.py:1011 |
| lin | 算子 | F.linear(x.float(), self.weight) | 5120 → 129280 | weight [129280, 5120]，fp32 参数，打分在 fp32 里算 | model.py:1012 |
| logits | 张量 | logits | fp32 [B, 129280] | | model.py:1012 |

### 连线

- px → plast
- plast → xp
- xp → lin
- lin → logits

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 词表投影 | $\mathrm{logits} = h W^\top$，h 是最后一个位置的隐状态（已经过最后的 RMSNorm），W 是 head.weight |
| 配置 | dim | 5120 |
| 配置 | vocab_size | 129280 |
