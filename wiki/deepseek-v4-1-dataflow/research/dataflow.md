# DeepSeek-V4.1-Flash

- 描述：DeepSeek-V4.1-Flash 一次带图的 prefill 前向的数据流：一张图像先切成 patch，过 32 层视觉 Transformer（ViT，Vision Transformer）编码，再由对齐器按 3×3 降采样并投到主干维度，写回序列里图像 span 那一段的隐状态；文本 token 一边查出每个位置的 n-gram 哈希行号，一边取词嵌入，整段一起进 40 层主干；压缩比大于 0 的层，除 128 个位置的滑动窗口外，另加索引器选出的压缩 KV（压缩 KV 只由 4 个 kv source 层产出并按层序共享，索引器在 2、8、14、20 四层以及其后的 24、28、32、36 层各自选出参与注意力的 top-512 条压缩 KV）；前馈是 384 选 6 的专家混合（MoE，Mixture-of-Experts），图像 span 内用另一份路由偏置；最后折叠、归一化并投影成下一个 token 的 logits。
- 摘要：32 层 ViT 对一张图的 patch 做全双向注意力，位置编码是二维旋转位置编码（RoPE，Rotary Position Embedding）；对齐器按 3×3 降采样后投到 5120 维，对齐器的输出按行序写进图像 span 里的 IMAGE 位置，起止与换行位置写入学到的嵌入；主干 40 层，压缩 KV 只由 4 个 kv source 层（2、8、14、20）产出并按层序共享；窗口 128 个位置另加索引器选出的 top-512 压缩条目；窗口 KV 按 fp8、压缩 KV 按 fp4 量化再反量化之后才参与注意力；MoE 384 个路由专家取 top-6 加 1 个共享专家，路由专家存 fp4、共享专家存 fp8；Engram（按 n-gram 查表、再把结果按门控写回残差流的模块）的 n-gram 回看遇到图像 span 就断掉，路由挑专家时图像 span 的位置换成视觉语言（VL）那份偏置；残差流按 4 份复制，混合矩阵 comb 由 Sinkhorn 迭代 20 轮给出，pre、post 来自 sigmoid；Engram 插在层 1、14。
- 主题：模型结构,注意力机制
- 标签：数据流

## 资料

- 源码：Hugging Face 仓库 `deepseek-ai/DeepSeek-V4.1-Flash`，修订 `dba1be0a40aa45a94ad051997016db3960a90277`（2026-09-10）的 `inference/` 参考实现，用到的文件是 `model.py`（1309 行）、`vision.py`（119 行）、`kernel.py`（591 行）、`engram.py`（184 行）
- 视觉支路用到 `image_processor.py`（173 行）：patch 从原图怎么切出来（resize、归一化、按 patch 重排）写在它的 `load_image` 里，本页只取它的结果当输入，图里不画这一步
- transformers v4.57.6 只带 `deepseek_v2` 与 `deepseek_v3`，没有这个模型，故按官方仓库源码画
- config.json：`inference/config.json`，副本在 `sources/config.json`，字段名与 `ModelArgs` 一一对应
- 权重文件头：同一修订下 `model-00001-of-00048.safetensors` 到 `model-00048-of-00048.safetensors` 共 48 个分片，按 HTTP Range 只取头部解析，张量的形状表在 `sources/shapes.json`；同一层内同一子模块的编号相同的张量形状一致时合并成 `<名字>.{i}` 一条，Engram 两张词表形状不同，按具体层号列出
- 模型卡：同一仓库的 `README.md`，副本在 `sources/model-card.md`（2026-10-08 取回）；参数量、压缩稀疏注意力的指标（模型卡里叫 CSA2）、KV 每 token 字节数、Engram 与 DSpark 的参数都取自它
- 论文或技术报告：同一修订下的 `DeepSeek_V41_Tech_Report.pdf`

## 前提

- 这一次前向是 `Transformer.forward` 的一次调用：推理，不算 loss
- 这一次是 prefill：`start_pos` 为 0，一次喂进 T 个 token，T 不小于 1024，且能被各层非零的 `compress_ratio` 整除
- 带图输入：`images` 传一张图、`token_types` 按 `image_processor.prepare_vl_inputs` 给出，于是 `image_mask` 是 `token_types >= 0`、`engram_mask` 是 `~image_mask`，两者都是实际张量。源码要求图像 span 落在这一次 chunk 里，`start_pos` 不为 0 时直接断言失败
- 这张图经 `image_processor.load_image` 变成 patch 网格：`patches` 是 bf16 [S, 3, 14, 14]，patch 数 S 是 n_vit_h × n_vit_w；对齐器按 3×3 降采样后的 token 网格是 [n_llm_h, n_llm_w]，图像在序列里占 T_v = n_llm_h × (n_llm_w + 1) + 2 个位置（每行 n_llm_w 个 IMAGE 再加一个换行，首尾各一）。本页按一张图、一个 patch 网格画，T_v 与 S 都是这张图自己的大小
- 图像 span 的排布取 `image_processor.image_token_types` 的默认式样：首尾各一个 IMAGE_START、IMAGE_END，中间 n_llm_h 行，每行 n_llm_w 个 IMAGE 加一个 IMAGE_NEW_LINE。span 内所有位置在 `input_ids` 里都是 `image_token_id`，只靠 `token_types` 区分
- 单卡，`world_size` 为 1
- Expert.forward 这张图画路由专家：调用时传入 weights。共享专家同一次前向也会调用 Expert，不传 weights，不乘路由权重
- 权重按 `config.json` 的 `dtype` 与 `expert_dtype` 加载：用 `Linear`、`ColumnParallelLinear`、`RowParallelLinear` 建；这些层里没写 dtype 的权重取 `dtype`（fp8），路由专家取 `expert_dtype`（fp4）；压缩器在 `compress_ratio` 大于 1 时把 `wkv`、`wgate` 提升到 fp32；激活在做 fp8/fp4 GEMM 前按块量化（`linear()` 里的 `act_quant`），见各图用 `Linear` 的节点的折叠。源码里显式指定了别的精度的按源码，如 wo_a、ParallelHead 的 weight、attn_sink、hc_* 六组参数、Gate 的 weight/bias/bias_vl、索引器的 wk.weight 与 weights_proj.weight、压缩比不大于 1 的压缩器的 wkv
- 注意力、压缩器与索引器走的是 compress_ratio 大于 0 这条分支：kv source 层 2、8、14 的 compress_ratio 是 2、结构相同；层 20 的 compress_ratio 是 1，压缩器只有压缩那一份的投影（不做逐 token 的池化，只留 wkv 一路），索引器仍由它选出候选块
- 索引器在 8 个 index source 层（2、8、14、20、24、28、32、36）各跑一次：2、8、14、20 自己把压缩潜在量变成索引键，层 20 另选出候选块；24、28、32、36 不建键、只从前面发布的那份 index_k 读键，并在层 20 选出的候选块里取 top-512。这两件事互斥：「自己把压缩潜在量变成索引键」的 2、8、14、20 不做候选块过滤，「从已发布的索引键读键」的 24、28、32、36 也不选候选块（只有层 20 选，供它们用）
- 注意力只有 `sparse_attn` 一个 kernel，窗口 KV 与压缩 KV 拼成一份 KV 之后一次算完
- 调用的是 `Transformer.forward`，不是 `forward_spec`（草稿路径）。config.json 里叫 dspark_target_layer_ids 的那 3 个层号 [37, 38, 39] 在这里是 `target_layer_ids`，用来在进第 37、38、39 层之前收集 main_hidden，属于本次前向；真正走草稿路径的是 forward_spec 与那 3 个多 token 预测（MTP，Multi-Token Prediction）层，草稿路径与本次前向的 tap 不是一回事
- 缓存按 `max_batch_size` 分配，`config.json` 里没有这一项，取 `ModelArgs` 默认值 4；它是缓存容量，与本次前向的批大小 B 不是同一个量。频率表与压缩 KV／索引键缓存的序列维取 `max_seq_len` 默认值 4096，`config.json` 里也没有这一项；窗口 KV 缓存只有 `window_size` 个槽位

## config.json

| 组 | 名字 | 值 |
|---|---|---|
| 注意力 | window_size | 128 |
| 注意力 | compress_ratios | 前 40 项是主干：层 0、1 为 0，层 2–19 为 2，层 20–39 为 1；末 3 项是多 token 预测（MTP，Multi-Token Prediction）层，为 0 |
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
| Hyper-Connections | hc_mult | 4（Hyper-Connections 把残差流复制成多份，每层学一组 pre/post/comb 系数来折叠与展开；hc_mult 是份数） |
| Hyper-Connections | hc_sinkhorn_iters | 20 |
| Hyper-Connections | hc_eps | 1e-06 |
| MoE | n_routed_experts | 384 |
| MoE | n_shared_experts | 1 |
| MoE | n_activated_experts | 6 |
| MoE | score_func | sqrtsoftplus |
| MoE | route_scale | 1.5 |
| MoE | norm_topk_prob | true（默认值，config.json 里没有这一项） |
| MoE | swiglu_limit | 10.0 |
| Engram | engram_layer_ids | [1, 14] |
| Engram | engram_max_ngram_size | 4 |
| Engram | engram_n_heads | 8 |
| Engram | engram_head_dim | 256 |
| Engram | engram_vocab_size | 16000000 |
| Engram | engram_compressed_vocab_size | 99092 |
| Engram | engram_pad_id | 2 |
| DSpark | dspark_target_layer_ids | [37, 38, 39]（DSpark 指半自回归草稿 + 置信度调度验证的推测解码那一套；本页用的这 3 个层号是收集 main_hidden 的 tap，与草稿路径无关） |
| 视觉 | vision_n_layers | 32 |
| 视觉 | vision_rope_theta | 10000 |
| 视觉 | vision_max_n_token | 1024 |
| 视觉 | vision_max_wh_ratio | null（config.json 里的值就是 null，图像不按长宽比上限截断） |
| 量化 | dtype | fp8 |
| 量化 | expert_dtype | fp4 |

## Transformer.forward（model.py:1242-1272）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ids | 张量 | input_ids | int64 [B, T] | B 是批大小（本次 1）；T 是这次喂进的 token 数；每个位置都是真实 token，填充不会传进来；图像 span 内每个位置都是 image_token_id | model.py:1243 |
| tt | 张量 | token_types | int64 [B, T] | 每个位置的类型，文本是 TEXT（−1），图像 span 内是 IMAGE_START、IMAGE、IMAGE_NEW_LINE、IMAGE_END 里的一个；由 image_processor.prepare_vl_inputs 给出 | model.py:1243 |
| cim | 算子 | token_types >= 0 | int64 [B, T] → bool [B, T] |  | model.py:1249 |
| im | 张量 | image_mask | bool [B, T] | 图像 span 的位置为 True；Block、MoE、Gate 各图里用的是同一个量 | model.py:1249 |
| cem | 算子 | ~image_mask | bool [B, T] → bool [B, T] |  | model.py:1251 |
| emv | 张量 | engram_mask | bool [B, T] | 图像 span 的位置为 False，与 image_mask 相反；Engram 与 NgramHashState 两图里用的是同一个量 | model.py:1251 |
| patches | 张量 | img.patches | bf16 [S, 3, 14, 14] | S 是这张图的 patch 数，等于 n_vit_h × n_vit_w；14 是 vision_patch_size | model.py:1238 |
| venc | 算子 | self.vision(patches, n_vit_h, n_vit_w) | bf16 [S, 3, 14, 14] → bf16 [S, 1024] |  | model.py:1226 |
| vla | 张量 | self.vision(patches, n_vit_h, n_vit_w) | bf16 [S, 1024] | ViT 的输出；源码里嵌在 aligner 的实参位置，写作 x，没有单独的变量名 | model.py:1226 |
| alen | 算子 | self.aligner(vla, n_vit_h, n_vit_w) | bf16 [S, 1024] → bf16 [n_llm_h × n_llm_w, 5120] |  | model.py:1226 |
| vemb | 张量 | self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w) | bf16 [n_llm_h × n_llm_w, 5120] | encode_image 的返回值，就是写回图像 span 的那份图像 token 嵌入 | model.py:1226 |
| ngram | 算子 | self.engram_hash(input_ids, start_pos, engram_mask) | [B, T] → [B, T, 2, 24] |  | model.py:1252 |
| hash | 张量 | engram_hashes | int64 [B, T, 2, 24] | | model.py:1252 |
| emb | 算子 | self.embed(input_ids) | int64 [B, T] → bf16 [B, T, 5120] |  | model.py:1253 |
| h0 | 张量 | h | bf16 [B, T, 5120] | | model.py:1253 |
| imgp | 张量 | self.image_start、self.image_end、self.image_newline | bf16 [5120] × 3 | Transformer 上三个学到的嵌入参数，写回图像 span 的起止与换行位置各一份 | model.py:1220-1222 |
| mrg | 算子 | self.merge_image_embeddings(images, h) | bf16 [B, T, 5120] 与 bf16 [n_llm_h × n_llm_w, 5120] → bf16 [B, T, 5120] |  | model.py:1256 |
| himg | 张量 | h | bf16 [B, T, 5120] | 图像 span 里 IMAGE 位置已换成对齐器的行，起止、换行位置换成学到的嵌入 | model.py:1256 |
| hcexp | 算子 | h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1) | [B, T, 5120] → [B, T, 4, 5120] |  | model.py:1258 |
| hc | 张量 | h | bf16 [B, T, 4, 5120] | 4 = hc_mult；Hyper-Connections 的 4 份并行残差流，主干从头到尾都在这 4 份上算 | model.py:1258 |
| pm0 | 算子 | make_identity_pre_mix(h, self.hc_mult) | [B, T, 4, 5120] → fp32 [B, T, 4] |  | model.py:1260 |
| pre | 张量 | pre_mix | fp32 [B, T, 4] | 进入第 0 层时的初值，此后每层返回的 ffn_pre 都覆盖它 | model.py:1260 |
| eng | 算子 | layer.engram(h, engram_hashes[:, :,<br>layer.engram.layer_hash_index, :], engram_mask) | [B, T, 4, 5120] → [B, T, 4, 5120] |  | model.py:1262-1263 |
| eh | 张量 | h | bf16 [B, T, 4, 5120] | 层 1、14 在进 Block 之前，Engram 写回的残差 | model.py:1263 |
| layers | 算子 | 40 × Block | [B, T, 4, 5120] → [B, T, 4, 5120] |  | model.py:1267 |
| hl | 张量 | h | bf16 [B, T, 4, 5120] | 循环里每一层 Block 返回、又喂给下一层的那份残差流（本图合并成同一个量看）；层 1、14 进层前先过 Engram 的是它，第 37、38、39 层进层前取均值的是它，循环结束后交给最后一次 hc_pre 的也是它 | model.py:1261-1267 |
| lpm | 张量 | pre_mix | fp32 [B, T, 4] | 每一层返回的 pre_mix，喂给下一层；最后一层的送给最后一次 hc_pre | model.py:1267 |
| avg | 算子 | h.mean(dim=2) | [B, T, 4, 5120] → [B, T, 5120] |  | model.py:1265-1266 |
| am | 张量 | h.mean(dim=2) | bf16 [B, T, 5120] | 3 个目标层各一份，逐层追加进 main_hiddens | model.py:1266 |
| mcat | 算子 | torch.cat(main_hiddens, dim=-1) | [B, T, 5120] × 3 → [B, T, 15360] |  | model.py:1271 |
| mh | 张量 | main_hidden | bf16 [B, T, 15360] | 作为第三个返回值交给调用方 | model.py:1271 |
| col | 算子 | layer.hc_pre(h, pre_mix) | [B, T, 4, 5120] → [B, T, 5120] |  | model.py:1268 |
| hc2 | 张量 | h | bf16 [B, T, 5120] | | model.py:1268 |
| fnorm | 算子 | self.norm(h) | bf16 [B, T, 5120] → bf16 [B, T, 5120] |  | model.py:1269 |
| hn | 张量 | h | bf16 [B, T, 5120] | | model.py:1269 |
| head | 算子 | self.head(h) | bf16 [B, T, 5120] → fp32 [B, 129280] |  | model.py:1269 |
| logits | 张量 | logits | fp32 [B, 129280] | | model.py:1269 |
| samp | 算子 | sample(logits, self.temperature) | fp32 [B, 129280] → int64 [B] |  | model.py:1270 |
| oids | 张量 | output_ids | int64 [B] | | model.py:1270 |
| shar | 缓存 | shared_attn | compress_kv、index_k、topk_idxs、candidates | 各层之间传递的共享量，每个前向里由靠前的层写、靠后的层读，不重置。压缩 KV 与索引键由 4 个 kv source 层各写自己那一份，写入后发布到这里，下一层起读到下一个 source 层；candidates 由层 20 写，层 24、28、32、36 读 | model.py:1180 |
| tret | 张量 | (output_ids, logits, main_hidden) | int64 [B] × fp32 [B, 129280] × bf16 [B, 15360] | 三个返回值：采样出的 id、logits、拼接后的 main_hidden | model.py:1272 |

### 连线

- ids → emb
- emb → h0
- tt → cim
- cim → im
- im → cem
- cem → emv
- emv → ngram
- ids → ngram
- ngram → hash
- patches → venc：n_vit_h、n_vit_w
- venc → vla
- vla → alen：n_vit_h、n_vit_w
- alen → vemb
- vemb → mrg
- h0 → mrg
- imgp → mrg：三个学到的嵌入
- mrg → himg
- himg → hcexp
- hcexp → hc
- hc → pm0：形状与 device
- pm0 → pre
- pre → layers：层 0 的 pre_mix
- hl → eng：层 1、14 进层前的输入
- hash → eng
- emv → eng：engram_mask
- eng → eh
- eh → layers：层 1、14 的输入
- hc → layers：层 0 的输入
- hl → layers：层 2–39（除层 14 外）的输入，循环里上一层 Block 返回的残差流
- lpm → layers：上一层返回的 pre_mix
- im → layers：image_mask
- shar → layers
- layers → shar
- layers → hl
- layers → lpm
- hl → avg：第 37、38、39 层进层前的输入
- avg → am
- am → mcat
- mcat → mh
- hl → col
- lpm → col
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
- venc 点开 → ViT.forward
- alen 点开 → Aligner.forward
- ngram 点开 → NgramHashState.forward
- eng 点开 → Engram.forward
- layers 点开 → Block.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| cim | 公式 | $\mathbf{M} = \mathbf{t} \ge 0$ |
| cim | 符号 | $\mathbf{M}$：image_mask，图像 span 的位置为 True<br>$\mathbf{t}$：token_types，每个位置的类型，非负的都是图像 span 里的位置 |
| cim | 说明 | TEXT 是 −1，所以非负的都是图像 span 里的位置；图像 span 内所有位置在 $I$ 里都是 image_token_id，只有 $\mathbf{t}$ 能区分它们与文本 |
| cem | 公式 | $\mathbf{E} = \lnot \mathbf{M}$ |
| cem | 符号 | $\mathbf{E}$：engram_mask，与 image_mask 相反<br>$\mathbf{M}$：上一步由 token_types 得出的布尔掩码 |
| cem | 说明 | NgramHashState 拿它当 token_mask，False 的位置不参与 n-gram，在 cache 里记成 DEAD |
| ngram | 公式 | $\mathbf{H} = \mathrm{hash}(I, s, \mathbf{E})$ |
| ngram | 符号 | $\mathbf{H}$：engram_hashes，每个位置结尾的 n-gram 哈希行号<br>$I$：input_ids，词 id<br>$s$：start_pos，0<br>$\mathbf{E}$：这次传进来的 engram_mask，图像 span 的位置为 False |
| ngram | 说明 | NgramHashState，把每个位置结尾的 n-gram 哈希成表行号<br>倒数第二维是 Engram 层序号，对应 engram_layer_ids 的 [1, 14]；24 = (4 − 1) × 8 |
| emb | 公式 | $h = W_e[I]$ |
| emb | 符号 | $h$：查出的词嵌入<br>$W_e$：词表矩阵，源码里叫 embed.weight<br>$I$：input_ids，词 id |
| emb | 说明 | ParallelEmbedding：$W_e$ [129280, 5120]（源码里叫 embed.weight），按 rank 沿词表维切分，单卡即整张。图像 span 内的位置也照常查表，查出来的结果随后被覆盖 |
| venc | 公式 | $x_v = \mathrm{vision}(\mathbf{P}, u, v)$ |
| venc | 符号 | $x_v$：ViT 的输出<br>$\mathbf{P}$：patches，这张图的 patch 网格<br>$u$：n_vit_h，patch 网格的高<br>$v$：n_vit_w，patch 网格的宽 |
| venc | 说明 | 只对图算一次，与文本的序列长度无关。ViT 的输出交给同一图的 alen |
| alen | 公式 | $\mathbf{F} = \mathrm{aligner}(x_v, u, v)$ |
| alen | 符号 | $\mathbf{F}$：embeds，图像 token 嵌入<br>$x_v$：ViT 的输出<br>$u$：n_vit_h，patch 网格的高<br>$v$：n_vit_w，patch 网格的宽 |
| alen | 说明 | 对齐器把 ViT 的输出按 3×3 降采样再投到主干维度。encode_image 是 Transformer 上的一个方法，把 self.vision 与 self.aligner 接起来 |
| mrg | 公式 | $\mathbf{S} = h_{i,\ a : a + \lvert \mathbf{t} \rvert}$<br>$\mathbf{S}[\mathbf{t} = \mathrm{IMAGE\_START}] = \mathbf{r}_1$<br>$\mathbf{S}[\mathbf{t} = \mathrm{IMAGE\_END}] = \mathbf{r}_2$<br>$\mathbf{S}[\mathbf{t} = \mathrm{IMAGE\_NEW\_LINE}] = \mathbf{r}_3$<br>$\mathbf{S}[\mathbf{t} = \mathrm{IMAGE}] = \mathbf{F}$ |
| mrg | 符号 | $\mathbf{S}$：span，h 里图像 span 那一段的切片，写回的就是 h 本身<br>$h$：进 merge 的隐状态<br>$a$：img.start，图像 span 在序列里的起始位置<br>$\mathbf{t}$：types，img.types 移到 h 的 device 之后的那一份<br>$\mathbf{F}$：embeds，对齐器的输出<br>$\mathbf{r}_1$、$\mathbf{r}_2$、$\mathbf{r}_3$：image_start、image_end、image_newline，Transformer 上三个学到的嵌入参数 |
| mrg | 说明 | 大写的是 token 类型（IMAGE_START 等），小写的 image_start [5120]、image_end [5120]、image_newline [5120] 是 Transformer 上三个学到的嵌入参数，to($h$.dtype) 之后写进去；按行序把 $\mathbf{F}$ 的行写进 IMAGE 位置 |
| hcexp | 公式 | $h = h.\mathrm{unsqueeze}(2).\mathrm{repeat}(1, 1, C, 1)$ |
| hcexp | 符号 | $h$：残差流<br>$C$：hc_mult，份数 |
| hcexp | 说明 | Hyper-Connections（每一层学一组 pre/post/comb 系数，把残差流折叠成一份送进子层、再展开回多份）的起点：把一份残差流沿新的一维复制成 $C$ 份，主干从头到尾都在 $C$ 份上算，层间只改这 $C$ 份的混合方式 |
| eng | 公式 | $h = \mathrm{engram}(h, \mathbf{H}_{:,:,l,:}, \mathbf{E})$ |
| eng | 符号 | $h$：残差流<br>$\mathbf{H}$：engram_hashes，n-gram 哈希行号<br>$l$：layer_hash_index，本层的 Engram 层序号<br>$\mathbf{E}$：engram_mask，这次传进来的张量 |
| eng | 说明 | Engram：层 1、14 在进入本层之前跑，吃的是上一层 Block 的输出，把 n-gram 查表的结果按门控加进残差流 |
| pm0 | 公式 | $\pi_{:,:,0} = 1$ |
| pm0 | 符号 | $\pi$：pre_mix，进入第 0 层时的折叠系数<br>$h$：用来取 batch 和序列长度<br>$C$：hc_mult，份数 |
| pm0 | 说明 | 第 0 份置 1、其余置 0，作为进入第 0 层时的 pre_mix |
| layers | 公式 | $h', \pi' = \mathrm{layer}(h, s, \pi, \mathbf{M})$ |
| layers | 符号 | $h$：残差流<br>$\pi$：pre_mix，折叠系数<br>$\mathrm{layer}$：这一层的 Block<br>$s$：start_pos，0<br>$\mathbf{M}$：image_mask，这次传进来的张量，图像 span 的位置为 True |
| layers | 说明 | 层 0 的输入是 hc；层 1、14 在进本层之前先跑 Engram 改写残差流；其余层的输入是上一层 Block 的输出<br>压缩 KV 只在 4 个 kv source 层（2、8、14、20）算一次，其余 compress_ratio 大于 0 的层从 shared_attn 读。层 0、1 的 compress_ratio 为 0，只有滑动窗口。层 20–39 的 compress_ratio 是 1。层 20 的索引器写出 candidates，层 24、28、32、36 只在这些候选块里取 top-512<br>$\mathbf{M}$ 一路传到每层的 MoE，图像 span 的路由走 VL 偏置 |
| avg | 公式 | $\bar{h} = \mathrm{mean}(h, \dim=2)$ |
| avg | 符号 | $\bar{h}$：取均值后的隐状态<br>$h$：进入目标层之前的残差流<br>$2$：hc 那一维 |
| avg | 说明 | 进入第 37、38、39 层之前的残差流，按 hc 维（4 份）取均值 |
| mcat | 公式 | $\mathbf{D} = \mathrm{cat}(\{\mathbf{D}_j\}, \dim=-1)$ |
| mcat | 符号 | $\mathbf{D}$：main_hidden，拼接后的隐状态<br>$\mathbf{D}_j$：main_hiddens，3 个目标层各一份 [B, T, 5120]<br>$j$：目标层序号（37、38、39） |
| mcat | 说明 | 3 个目标层（37、38、39）各一份，逐层追加 |
| col | 公式 | $h = \sum_{c=0}^{C-1} \pi_c\, h_c$ |
| col | 符号 | $h$：折叠后的残差流<br>$\pi$：pre_mix，最后一层返回的系数<br>$h_c$：第 c 份残差<br>$C$：hc_mult |
| col | 说明 | 用最后一层返回的 pre_mix 把 4 份折叠成一份 |
| fnorm | 公式 | $h = w \odot \dfrac{h}{\sqrt{\mathrm{mean}(h^2) + \epsilon}}$ |
| fnorm | 符号 | $h$：送进 norm 的隐状态<br>$w$：norm 的权重，源码里叫 norm.weight<br>$\epsilon$：1e-20，源码里叫 norm.eps<br>$\mathrm{mean}$：在最后一维 |
| fnorm | 说明 | $w$ [5120]（源码里叫 norm.weight） |
| head | 公式 | $\mathbf{L} = h_{:,-1}\, W_h^\top$ |
| head | 符号 | $\mathbf{L}$：logits，下一个 token 的词表打分<br>$h$：送进 head 的隐状态，只取最后一个位置 $h_{:,-1}$<br>$W_h$：词表矩阵，源码里叫 head.weight |
| head | 说明 | $W_h$ [129280, 5120]（源码里叫 head.weight），fp32 |
| samp | 公式 | $\mathbf{L} = \mathbf{L} / \max(\tau, 10^{-5})$；$\mathbf{O} = \mathrm{argmax}(\mathrm{softmax}(\mathbf{L}) / \xi)$ |
| samp | 符号 | $\mathbf{L}$：logits，词表打分，按温度缩放后原地覆盖<br>$\tau$：temperature，取默认值 1<br>$\xi$：exponential，同形状的指数分布随机数<br>$\mathbf{O}$：output_ids，采样出的 id |
| samp | 说明 | Gumbel-max：概率除以指数分布随机数后取 argmax，等价于按概率采样 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 介绍 | 模型 | DeepSeek-V4.1-Flash 是 DeepSeek 的多模态 MoE 模型，主干 552B 参数、每个 token 激活 8B（prefill）/ 16B（decode），上下文最长 100 万 token。主干 40 层，注意力用压缩稀疏注意力（CSA2，Compressed Sparse Attention 2），KV 压到每 token 890 字节。每层 MoE 是 384 个路由专家取 6 个、加 1 个共享专家；另有 Engram 条件记忆和 DSpark 推测解码。视觉侧是 DeepSeek-ViT 加一个两层 MLP 投影。 |
| 介绍 | 这次前向 | 一次带图的 prefill：`start_pos` 为 0，一张图先过 32 层 ViT 与对齐器，把输出写进序列里图像 span 的隐状态；文本位置照常取词嵌入与 n-gram 哈希；整段一起过 40 层主干，最后取最后一个位置投影出下一个 token 的 logits。 |
| 公式 | 输入嵌入 | $h = W_e[I]$<br>查词嵌入 |
| 公式 | 图像嵌入 | $\mathbf{F} = \mathrm{encode\_image}(\mathbf{P}, u, v)$<br>编码图像得到 token 嵌入 |
| 公式 | 图像写回 | $\mathbf{S}[\mathbf{t} = \mathrm{IMAGE}] = \mathbf{F}$<br>把图像嵌入按行写回序列 |
| 公式 | 残差流复制 | $h = h.\mathrm{unsqueeze}(2).\mathrm{repeat}(1, 1, C, 1)$<br>把残差流复制成 hc_mult 份 |
| 公式 | 输出 | $\mathbf{L} = h_{:,-1}\, W_h^\top$<br>取末位隐状态投出词表 logits |
| 配置 | hc_mult | 4 |
| 配置 | dspark_target_layer_ids | [37, 38, 39] |
| 配置 | engram_layer_ids | [1, 14] |
| 配置 | vision_n_layers | 32 |

## ViT.forward（vision.py:98-103）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| patches | 张量 | patches | bf16 [S, 3, 14, 14] | S 是这张图的 patch 数，等于 n_vit_h × n_vit_w；14 是 vision_patch_size；与 Transformer 图的 patches 是同一份 | vision.py:98 |
| vpe | 算子 | self.patch_embed(patches) | bf16 [S, 3, 14, 14] → bf16 [S, 1024] |  | vision.py:99 |
| vx | 张量 | x | bf16 [S, 1024] | patch 嵌入，1024 是 vision_dim | vision.py:99 |
| vdim | 张量 | self.rope_dim | int，值为 32 | 构造时算出 vision_dim // vision_n_heads // 2，本图只取它的值 | vision.py:92 |
| vth | 张量 | self.rope_theta | float，值为 10000 | 构造时取 vision_rope_theta 10000 | vision.py:93 |
| vcs_op | 算子 | get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta) | int n_vit_h 与 int n_vit_w → fp32 [n_vit_h × n_vit_w, 1, 32] × 2 |  | vision.py:100 |
| vcs | 张量 | cos, sin | fp32 [n_vit_h × n_vit_w, 1, 32] × 2 | 一对二维频率表，整张图共用；32 = rope_dim | vision.py:100 |
| vblocks | 算子 | 32 × vision.Block | bf16 [S, 1024] → bf16 [S, 1024] |  | vision.py:95，101-102 |
| vx2 | 张量 | x | bf16 [S, 1024] | | vision.py:102 |
| vnorm | 算子 | self.norm(x) | bf16 [S, 1024] → bf16 [S, 1024] |  | vision.py:103 |
| vout | 张量 | self.norm(x) | bf16 [S, 1024] | | vision.py:103 |

### 连线

- patches → vpe
- vpe → vx
- vdim → vcs_op
- vth → vcs_op
- patches → vcs_op：n_vit_h、n_vit_w
- vcs_op → vcs
- vx → vblocks
- vcs → vblocks
- vblocks → vx2
- vx2 → vnorm
- vnorm → vout
- vblocks 点开 → vision.Block.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| vpe | 公式 | $x = \mathrm{flatten}(\mathbf{P}, 1)\, W_p^\top + b_p$ |
| vpe | 符号 | $x$：patch 嵌入<br>$\mathbf{P}$：patches，这张图的 patch 网格<br>$\mathrm{flatten}$：把每个 patch 的 3 × 14 × 14 摊平<br>$W_p$、$b_p$：proj 的权重与偏置，源码里叫 proj.weight、proj.bias |
| vpe | 说明 | PatchEmbed：patch_embed 就是一次线性投影，proj 是 nn.Linear（vision.py:40），$W_p$ [1024, 588]、$b_p$ [1024]（源码里叫 proj.weight、proj.bias），588 = 3 × 14 × 14 |
| vcs_op | 公式 | $f = \mathrm{flatten}([p_h, p_w] \odot \nu)$<br>$\mathbf{c} = \cos(f)$<br>$\mathbf{s} = \sin(f)$<br>$\nu_j = \theta^{-2j/d_r}$ |
| vcs_op | 符号 | $f$：freqs，位置频率<br>$p_h$、$p_w$：hpos、wpos，每个 patch 在网格里的高、宽坐标<br>$\nu$：inv_freq，各维的逆频率<br>$\mathbf{c}$、$\mathbf{s}$：cos、sin，位置正余弦<br>$d_r$：rope_dim，取频率用的维度（vision_dim // vision_n_heads // 2 = 32）<br>$\theta$：vision_rope_theta |
| vcs_op | 说明 | stack 把高坐标、宽坐标配成对（第 1 维是高、第 2 维是宽），再摊平成 32 维：相邻两维一对、交替排列，偶维配高坐标、奇维配宽坐标，同一对内两维共用一个逆频率；$\mathbf{c}$、$\mathbf{s}$ 各补一个长度为 1 的维度；整张图只算一次，32 层共用 |
| vblocks | 公式 | $x = \mathrm{block}(x, \mathbf{c}, \mathbf{s})$ |
| vblocks | 符号 | $x$：隐状态<br>$\mathrm{block}$：这一层的 vision.Block<br>$\mathbf{c}$、$\mathbf{s}$：cos、sin，整张图共用的二维频率表 |
| vblocks | 说明 | 32 层结构相同，都拿同一份 $\mathbf{c}$、$\mathbf{s}$；差异只有各自那一份权重 |
| vnorm | 公式 | $x = w \odot \dfrac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}}$ |
| vnorm | 符号 | $x$：最后一层 Block 的输出<br>$w$：vision.norm 的权重，源码里叫 norm.weight<br>$\epsilon$：1e-6，源码里叫 norm.eps<br>$\mathrm{mean}$：在最后一维 |
| vnorm | 说明 | $w$ [1024]（源码里叫 vision.norm.weight） |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 位置频率 | $f = \mathrm{flatten}([p_h, p_w] \odot \nu)$<br>按 patch 高宽坐标算二维位置频率 |
| 公式 | 位置正余弦 | $\mathbf{c}, \mathbf{s} = \cos(f), \sin(f)$<br>把位置频率拆成余弦与正弦 |
| 配置 | vision_n_layers | 32 |
| 配置 | vision_rope_theta | 10000 |

## vision.Block.forward（vision.py:82-84）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| vbx | 张量 | x | bf16 [S, 1024] | 本层的输入，ViT 里 32 层结构相同 | vision.py:82 |
| vcs | 张量 | cos, sin | fp32 [n_vit_h × n_vit_w, 1, 32] × 2 | ViT 一次算好传给每一层，与 ViT 图里的 vcs 是同一对（沿用产出它的 id） | vision.py:82 |
| vbn1 | 算子 | self.norm1(x) | bf16 [S, 1024] → bf16 [S, 1024] |  | vision.py:83 |
| vbx1 | 张量 | x | bf16 [S, 1024] | | vision.py:83 |
| vba | 算子 | self.attn(x, cos, sin) | bf16 [S, 1024] 与 fp32 [n_vit_h × n_vit_w, 1, 32] × 2 → bf16 [S, 1024] |  | vision.py:83 |
| vba1 | 张量 | self.attn(x, cos, sin) | bf16 [S, 1024] | 源码里这一项嵌在加法里，没有单独的变量名；实参 x 是上一节点 norm1 的输出 | vision.py:83 |
| vbad | 算子 | x + self.attn(x, cos, sin) | bf16 [S, 1024] 与 bf16 [S, 1024] → bf16 [S, 1024] |  | vision.py:83 |
| vby | 张量 | x | bf16 [S, 1024] | 注意力那一步之后的残差 | vision.py:83 |
| vbn2 | 算子 | self.norm2(x) | bf16 [S, 1024] → bf16 [S, 1024] |  | vision.py:84 |
| vby1 | 张量 | x | bf16 [S, 1024] | | vision.py:84 |
| vmlp | 算子 | self.mlp(x) | bf16 [S, 1024] → bf16 [S, 1024] |  | vision.py:84 |
| vbm1 | 张量 | self.mlp(x) | bf16 [S, 1024] | 源码里这一项嵌在加法里，没有单独的变量名 | vision.py:84 |
| vbad2 | 算子 | x + self.mlp(x) | bf16 [S, 1024] 与 bf16 [S, 1024] → bf16 [S, 1024] |  | vision.py:84 |
| vbout | 张量 | x | bf16 [S, 1024] | | vision.py:84 |

### 连线

- vbx → vbn1
- vbn1 → vbx1
- vbx1 → vba
- vcs → vba
- vba → vba1
- vbx → vbad：residual（本层输入）
- vba1 → vbad：注意力子层的输出
- vbad → vby
- vby → vbn2
- vbn2 → vby1
- vby1 → vmlp
- vmlp → vbm1
- vby → vbad2：residual（注意力那一步的输出）
- vbm1 → vbad2：前馈子层的输出
- vbad2 → vbout
- vba 点开 → vision.Attention.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| vbn1 | 公式 | $x = w_1 \odot x / \sqrt{\mathrm{mean}(x^2) + \epsilon}$ |
| vbn1 | 符号 | $x$：本层输入<br>$w_1$：norm1 的权重，源码里叫 norm1.weight<br>$\epsilon$：1e-6，源码里叫 norm1.eps<br>$\mathrm{mean}$：在最后一维 |
| vbn1 | 说明 | $w_1$ [1024]（源码里叫 norm1.weight），norm1 是 vision.RMSNorm，vision 支路统一 eps = 1e-6 |
| vba | 公式 | $x = \mathrm{attn}(x, \mathbf{c}, \mathbf{s})$ |
| vba | 符号 | $x$：norm1 之后的隐状态<br>$\mathbf{c}$、$\mathbf{s}$：cos、sin，整张图共用的二维频率表 |
| vba | 说明 | Attention，全双向，没有因果掩码；这一层用自己那份 wqkv、wo |
| vbad | 公式 | $x_{out} = x + \mathrm{attn}(x_1, \mathbf{c}, \mathbf{s})$ |
| vbad | 符号 | $x$：本层输入，进注意力之前的残差<br>$x_1$：norm1 的输出，喂给注意力那份<br>$\mathrm{attn}(\cdot)$：注意力的输出<br>$\mathbf{c}$、$\mathbf{s}$：cos、sin，整张图共用的二维频率表 |
| vbn2 | 公式 | $x = w_2 \odot x / \sqrt{\mathrm{mean}(x^2) + \epsilon}$ |
| vbn2 | 符号 | $x$：注意力那一步的输出<br>$w_2$：norm2 的权重，源码里叫 norm2.weight<br>$\epsilon$：1e-6，源码里叫 norm2.eps<br>$\mathrm{mean}$：在最后一维 |
| vbn2 | 说明 | $w_2$ [1024]（源码里叫 norm2.weight），norm2 是 vision.RMSNorm，vision 支路统一 eps = 1e-6 |
| vmlp | 公式 | $g, u = \mathrm{chunk}(x\, W_1^\top, 2)$<br>$x = (\mathrm{silu}(g) \odot u)\, W_2^\top$ |
| vmlp | 符号 | $x$：norm2 之后的隐状态<br>$g$、$u$：gate、up，投影后切成两路<br>$W_1$、$W_2$：w1、w2 的权重，源码里叫 w1.weight、w2.weight，无 bias |
| vmlp | 说明 | 通用 SwiGLU 的 MLP（vision.MLP），$W_1$ [5632, 1024]、$W_2$ [1024, 2816]（源码里叫 w1.weight、w2.weight），5632 = 2 × vision_inter_dim，2816 是 vision_inter_dim；w1、w2 是 nn.Linear（PyTorch 自带） |
| vbad2 | 公式 | $x_{out} = x + \mathrm{mlp}(x_2)$ |
| vbad2 | 符号 | $x$：注意力那一步的输出，进前馈之前的残差<br>$x_2$：norm2 的输出，喂给前馈那份<br>$\mathrm{mlp}(\cdot)$：前馈的输出 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 注意力残差 | $x = x + \mathrm{attn}(\mathrm{norm1}(x), \mathbf{c}, \mathbf{s})$<br>注意力的输出加回本层输入 |
| 公式 | 前馈残差 | $x = x + \mathrm{mlp}(\mathrm{norm2}(x))$<br>前馈的输出加回注意力那一步 |
| 配置 | vision_n_layers | 32 |

## vision.Attention.forward（vision.py:54-60）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| vax | 张量 | x | bf16 [S, 1024] | 已经过 norm1 的隐状态 | vision.py:54 |
| vcs | 张量 | cos, sin | fp32 [n_vit_h × n_vit_w, 1, 32] × 2 | 与 ViT 图里的 vcs 是同一对，逐层传进来 | vision.py:54 |
| vnh | 张量 | self.n_heads | int，值为 16 | 构造时取 vision_n_heads 16 | vision.py:49 |
| vhdm | 张量 | self.head_dim | int，值为 64 | 视觉支路自己的 head_dim，等于 vision_dim // vision_n_heads，与主干的 head_dim 512 不是同一个量 | vision.py:50 |
| vwqkv | 算子 | self.wqkv(x) | bf16 [S, 1024] → bf16 [S, 3072] |  | vision.py:56 |
| vqkv | 张量 | self.wqkv(x) | bf16 [S, 3072] | 3072 = 3 × vision_dim，一次投影给出 q、k、v 三段；下一步按三等分切成 q、k、v，再各自 view 成 (S, 16, 64) | vision.py:56 |
| vq | 张量 | q, k, v | bf16 [S, 16, 64] × 3 | 64 = head_dim，头维插在序列维之后 | vision.py:56 |
| vcq | 算子 | apply_rotary(q, cos, sin) | bf16 [S, 16, 64] 与 fp32 [n_vit_h × n_vit_w, 1, 32] × 2 → bf16 [S, 16, 64] |  | vision.py:57 |
| vqr | 张量 | q | bf16 [S, 16, 64] | 加过二维旋转的查询 | vision.py:57 |
| vck | 算子 | apply_rotary(k, cos, sin) | bf16 [S, 16, 64] 与 fp32 [n_vit_h × n_vit_w, 1, 32] × 2 → bf16 [S, 16, 64] |  | vision.py:58 |
| vkr | 张量 | k | bf16 [S, 16, 64] | 加过二维旋转的键 | vision.py:58 |
| vsdpa | 算子 | F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)) | bf16 [S, 16, 64] × 3 → bf16 [16, S, 64] |  | vision.py:59 |
| vo | 张量 | o | bf16 [16, S, 64] | | vision.py:59 |
| vwo | 算子 | self.wo(o.transpose(0, 1).reshape(n, -1)) | bf16 [16, S, 64] → bf16 [S, 1024] |  | vision.py:60 |
| vaout | 张量 | self.wo(o.transpose(0, 1).reshape(n, -1)) | bf16 [S, 1024] | | vision.py:60 |

### 连线

- vax → vwqkv
- vwqkv → vqkv
- vax → vqkv：形状（S）
- vnh → vqkv
- vhdm → vqkv
- vqkv → vq
- vq → vcq
- vcs → vcq
- vcq → vqr
- vq → vck
- vcs → vck
- vck → vkr
- vqr → vsdpa
- vkr → vsdpa
- vq → vsdpa：v
- vsdpa → vo
- vo → vwo
- vax → vwo：形状（S），源码里的 n
- vwo → vaout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| vwqkv | 公式 | $qkv = x\, W_{qkv}^\top + b_{qkv}$ |
| vwqkv | 符号 | $qkv$：q、k、v 拼在一起的投影<br>$x$：norm1 之后的隐状态<br>$W_{qkv}$、$b_{qkv}$：wqkv 的权重与偏置，源码里叫 wqkv.weight、wqkv.bias |
| vwqkv | 说明 | $W_{qkv}$ [3072, 1024]、$b_{qkv}$ [3072]（源码里叫 wqkv.weight、wqkv.bias），3072 = 3 × 1024；wqkv 是 nn.Linear（PyTorch 自带，与主干用的 Linear 类不是一类）。Step 2 按三等分切成 q、k、v，再各自 view 成 (S, 16, 64)，头维插在序列维之后 |
| vcq | 公式 | $x_1, x_2 = \mathrm{chunk}(q.\mathrm{float}(), 2, \dim=-1)$<br>$q = \mathrm{cat}(x_1 \mathbf{c} - x_2 \mathbf{s},\ x_2 \mathbf{c} + x_1 \mathbf{s})$ |
| vcq | 符号 | $q$：查询<br>$x_1$、$x_2$：每头前后两半<br>$\mathbf{c}$、$\mathbf{s}$：cos、sin，二维频率表 |
| vcq | 说明 | 每个头 64 维的前一半与后一半配成一对（第 i 维配第 i+32 维），位置信息来自 patch 在网格里的高、宽；返回值再转回输入 dtype（vision.py:18-21） |
| vck | 公式 | $x_1, x_2 = \mathrm{chunk}(k.\mathrm{float}(), 2, \dim=-1)$<br>$k = \mathrm{cat}(x_1 \mathbf{c} - x_2 \mathbf{s},\ x_2 \mathbf{c} + x_1 \mathbf{s})$ |
| vck | 符号 | $k$：键<br>$x_1$、$x_2$：每头前后两半<br>$\mathbf{c}$、$\mathbf{s}$：cos、sin，与查询同一对二维频率表 |
| vck | 说明 | 返回值再转回输入 dtype |
| vsdpa | 公式 | $o = \mathrm{softmax}(q k^\top / \sqrt{d_h})\, v$ |
| vsdpa | 符号 | $o$：注意力输出<br>$q$、$k$、$v$：都先把头维挪到最前<br>$d_h$：head_dim，64 |
| vsdpa | 说明 | PyTorch 的 F.scaled_dot_product_attention，不给掩码、不做 dropout，整张图里 patch 之间全双向 |
| vwo | 公式 | $x = o.\mathrm{transpose}(0, 1).\mathrm{reshape}(S, -1)\, W_o^\top + b_o$ |
| vwo | 符号 | $x$：输出<br>$o$：注意力输出<br>$W_o$、$b_o$：wo 的权重与偏置，源码里叫 wo.weight、wo.bias |
| vwo | 说明 | $W_o$ [1024, 1024]、$b_o$ [1024]（源码里叫 wo.weight、wo.bias）；wo 也是 nn.Linear（PyTorch 自带） |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 注意力 | $o = \mathrm{softmax}(q k^\top / \sqrt{d_h})\, v$<br>整张图 patch 做无掩码双向注意力 |
| 配置 | vision_rope_theta | 10000 |

## Aligner.forward（vision.py:114-119）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| vout | 张量 | x | bf16 [S, 1024] | ViT 的输出，沿用 ViT 图的 vout，S = n_vit_h × n_vit_w | vision.py:114 |
| vrd | 张量 | self.downsample_ratio | int，值为 3 | 构造时取 vision_downsample_ratio 3 | vision.py:109 |
| vpad | 算子 | F.pad(x, (0, -n_w % r, 0, -n_h % r)) | bf16 [1024, n_vit_h, n_vit_w] → bf16 [1024, n_vit_h + (-n_vit_h % r), n_vit_w + (-n_vit_w % r)] |  | vision.py:117 |
| vpv | 张量 | x | bf16 [1024, n_vit_h + (-n_vit_h % r), n_vit_w + (-n_vit_w % r)] | 高、宽都补到 r 的整数倍；n_vit_h、n_vit_w 已经是 r 的整数倍时补 0 | vision.py:117 |
| vunf | 算子 | F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1) | bf16 [1024, n_vit_h + (-n_vit_h % r), n_vit_w + (-n_vit_w % r)] → bf16 [n_llm_h × n_llm_w, 9216] |  | vision.py:118 |
| vuv | 张量 | F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1) | bf16 [n_llm_h × n_llm_w, 9216] | 9216 = 3 × 3 × 1024；n_llm_h = ceil(n_vit_h / r)、n_llm_w = ceil(n_vit_w / r)，每行是一个 3×3 块的展开 | vision.py:118 |
| aw1 | 算子 | self.w1(x) | bf16 [n_llm_h × n_llm_w, 9216] → bf16 [n_llm_h × n_llm_w, 5120] |  | vision.py:119 |
| alg | 张量 | F.gelu(self.w1(x)) | bf16 [n_llm_h × n_llm_w, 5120] | w1 的输出再过一层 gelu（普通函数不单独成节点，跟它作用的值同节点）；源码里这一步嵌在下一行的调用里，没有单独的变量名 | vision.py:119 |
| vlin | 算子 | self.w2(alg) | bf16 [n_llm_h × n_llm_w, 5120] → bf16 [n_llm_h × n_llm_w, 5120] |  | vision.py:119 |
| vemb | 张量 | self.w2(F.gelu(self.w1(x))) | bf16 [n_llm_h × n_llm_w, 5120] | 对齐器的输出，与主干同一个维度，可以按行写回序列；与 Transformer 图的 vemb 是同一个量 | vision.py:119 |

### 连线

- vout → vpad：n_vit_h、n_vit_w（源码里的 n_h、n_w）；先 view(n_h, n_w, -1).permute(2, 0, 1) 把 patch 维拆回高宽、通道挪到最前
- vrd → vpad：r
- vrd → vunf：块大小与步长
- vpad → vpv
- vpv → vunf
- vunf → vuv
- vuv → aw1
- aw1 → alg
- alg → vlin
- vlin → vemb

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| vpad | 公式 | $x = \mathrm{permute}(\mathrm{view}(x, (u, v, -1)), (2, 0, 1))$<br>$x = \mathrm{pad}(x, (0, -v \bmod r,\ 0, -u \bmod r))$ |
| vpad | 符号 | $x$：ViT 的输出<br>$u$、$v$：n_h、n_w，就是 n_vit_h、n_vit_w<br>$k$：downsample_ratio，负数表示在右边、下边补 |
| vpad | 说明 | 先把 patch 维拆回高、宽，通道挪到最前（1024 是 vision_dim），再把高、宽各补到 $k$ 的整数倍，补 0。vision_max_n_token 1024 限住了总 token 数，n_vit_h、n_vit_w 不是 $k$ 的整数倍时才需要补 |
| vunf | 公式 | $x = \mathrm{unfold}(x.\mathrm{unsqueeze}(0), r, \mathrm{stride}=r).\mathrm{squeeze}(0).\mathrm{transpose}(0, 1)$ |
| vunf | 符号 | $x$：补过边的 patch 网格<br>$k$：视觉的 downsample_ratio（3），与注意力里的 compress_ratio 同形不同义 |
| vunf | 说明 | 把高宽平面切成不重叠的 $k$×$k$ 块，每块摊成一行；一块 $k$ × $k$ × 1024 摊平成 9216 维；输出行数是 ceil(n_vit_h / $k$) × ceil(n_vit_w / $k$)，就是图像在序列里占的 token 网格 |
| aw1 | 公式 | $out = x\, W_1^\top + b_1$ |
| aw1 | 符号 | $out$：第一层 MLP 的输出<br>$x$：展开后的 3×3 块，9216 维<br>$W_1$、$b_1$：w1 的权重与偏置，源码里叫 w1.weight、w1.bias |
| aw1 | 说明 | $W_1$ [5120, 9216]、$b_1$ [5120]（源码里叫 w1.weight、w1.bias），把 9216 维映到 5120；w1 是 nn.Linear（PyTorch 自带） |
| vlin | 公式 | $\mathbf{F} = g\, W_2^\top + b_2$ |
| vlin | 符号 | $\mathbf{F}$：embeds，对齐器的输出<br>$g$：上一节点（w1 的输出过 gelu）的值<br>$W_2$、$b_2$：w2 的权重与偏置，源码里叫 w2.weight、w2.bias |
| vlin | 说明 | $W_2$ [5120, 5120]、$b_2$ [5120]（源码里叫 w2.weight、w2.bias）；w2 是 nn.Linear（PyTorch 自带） |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 降采样 | $out = \mathrm{gelu}(\mathrm{unfold}(x)\, W_1^\top + b_1)\, W_2^\top + b_2$<br>按 3×3 切块摊平后过两层 MLP 降采样 |
| 配置 | vision_downsample_ratio | 3 |
| 配置 | vision_max_n_token | 1024 |

## Block.forward（model.py:968-994）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hin | 张量 | x | bf16 [B, T, 4, 5120] | 进入本层的残差流，4 份；进注意力前的 `residual = x` 就是它（同一份重新绑定，不另立节点） | model.py:970 |
| bpre | 张量 | pre_mix | fp32 [B, T, 4] | 上一层返回的 pre_mix，把 4 份折叠成一份时用 | model.py:972 |
| im | 张量 | image_mask | bool [B, T] | 这一层前向的参数，由外层一路传进 MoE；图像 span 的位置为 True；与 Transformer 图的 im 是同一个量 | model.py:973 |
| afn | 张量 | self.hc_attn_fn、self.hc_ffn_fn | fp32 [24, 20480] × 2 | 注意力与前馈两次混合投影的权重，各一份；24 = (2 + 4) × 4，20480 = 4 × 5120 | model.py:941-942 |
| asc | 张量 | self.hc_attn_scale、self.hc_ffn_scale | fp32 [3] × 2 | 两次混合各自的三个缩放系数 | model.py:945-946 |
| abs | 张量 | self.hc_attn_base、self.hc_ffn_base | fp32 [24] × 2 | 两次混合各自的偏置 | model.py:943-944 |
| mixes_a | 算子 | self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base) | [B, T, 4, 5120] → fp32 [B, T, 4] × 2 与 fp32 [B, T, 4, 4] |  | model.py:982 |
| apre | 张量 | attn_pre | fp32 [B, T, 4] | | model.py:982 |
| apost | 张量 | attn_post | fp32 [B, T, 4] | | model.py:982 |
| acomb | 张量 | attn_comb | fp32 [B, T, 4, 4] | | model.py:982 |
| cpre | 算子 | self.hc_pre(x, pre_mix) | [B, T, 4, 5120] → [B, T, 5120] |  | model.py:983 |
| x1 | 张量 | x | bf16 [B, T, 5120] | | model.py:983 |
| an | 算子 | self.attn_norm(x) | bf16 [B, T, 5120] → bf16 [B, T, 5120] |  | model.py:984 |
| x2 | 张量 | x | bf16 [B, T, 5120] | | model.py:984 |
| attn | 算子 | self.attn(x, start_pos, *attn_args) | bf16 [B, T, 5120] → bf16 [B, T, 5120] |  | model.py:985 |
| x3 | 张量 | x | bf16 [B, T, 5120] | | model.py:985 |
| cpost | 算子 | self.hc_post(x, residual, attn_post, attn_comb) | [B, T, 5120] → [B, T, 4, 5120] |  | model.py:986 |
| x4 | 张量 | x | bf16 [B, T, 4, 5120] | 注意力之后、进前馈之前的残差流；进前馈前的 `residual = x` 就是它（同一份重新绑定，不另立节点） | model.py:986 |
| mixes_f | 算子 | self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base) | [B, T, 4, 5120] → fp32 [B, T, 4] × 2 与 fp32 [B, T, 4, 4] |  | model.py:989 |
| fpre | 张量 | ffn_pre | fp32 [B, T, 4] | 作为返回值交给下一层当 pre_mix | model.py:989 |
| fpost | 张量 | ffn_post | fp32 [B, T, 4] | | model.py:989 |
| fcomb | 张量 | ffn_comb | fp32 [B, T, 4, 4] | | model.py:989 |
| cpre2 | 算子 | self.hc_pre(x, attn_pre) | [B, T, 4, 5120] → [B, T, 5120] |  | model.py:990 |
| x5 | 张量 | x | bf16 [B, T, 5120] | | model.py:990 |
| fn | 算子 | self.ffn_norm(x) | bf16 [B, T, 5120] → bf16 [B, T, 5120] |  | model.py:991 |
| x6 | 张量 | x | bf16 [B, T, 5120] | | model.py:991 |
| moe_op | 算子 | self.ffn(x, image_mask) | bf16 [B, T, 5120] → bf16 [B, T, 5120] |  | model.py:992 |
| x7 | 张量 | x | bf16 [B, T, 5120] | | model.py:992 |
| cpost2 | 算子 | self.hc_post(x, residual, ffn_post, ffn_comb) | [B, T, 5120] → [B, T, 4, 5120] |  | model.py:993 |
| hout | 张量 | x | bf16 [B, T, 4, 5120] | | model.py:993 |
| bret | 张量 | (x, ffn_pre) | bf16 [B, T, 4, 5120] × fp32 [B, T, 4] | 第二个返回值是下一层的 pre_mix | model.py:994 |

### 连线

- hin → mixes_a
- afn → mixes_a
- asc → mixes_a
- abs → mixes_a
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
- hin → cpost：residual
- apost → cpost
- acomb → cpost
- cpost → x4
- x4 → mixes_f
- afn → mixes_f：hc_ffn 那一份
- asc → mixes_f：hc_ffn 那一份
- abs → mixes_f：hc_ffn 那一份
- mixes_f → fpre
- mixes_f → fpost
- mixes_f → fcomb
- x4 → cpre2
- apre → cpre2
- cpre2 → x5
- x5 → fn
- fn → x6
- x6 → moe_op
- im → moe_op：image_mask
- moe_op → x7
- x7 → cpost2
- x4 → cpost2：residual
- fpost → cpost2
- fcomb → cpost2
- cpost2 → hout
- hout → bret
- fpre → bret
- attn 点开 → Attention.forward
- moe_op 点开 → MoE.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| mixes_a | 公式 | $m = \mathrm{flatten}(x, 2)\, W^\top \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x, 2)^2) + \epsilon)$<br>$a_j = \mathrm{sigmoid}(m_j \cdot s_0 + b_j) + \epsilon$<br>$p_j = 2\,\mathrm{sigmoid}(m_{j+C} \cdot s_1 + b_{j+C})$<br>$u_{j,k} = m_{j \cdot C + k + 2 C} \cdot s_2 + b_{j \cdot C + k + 2 C}$<br>$\mathbf{J} = \mathrm{Sinkhorn}(u)$ |
| mixes_a | 符号 | $m$：mixes，混合系数<br>$a$、$p$、$\mathbf{J}$：attn_pre、attn_post、attn_comb，折叠、展开、混合用的系数<br>$W$、$s$、$b$：注意力那次的参数，源码里叫 hc_attn_fn、hc_attn_scale、hc_attn_base<br>$C$：hc_mult<br>$\epsilon$：Block 上的 norm_eps 与 hc_eps<br>$j$、$k$：从 0 到 C−1<br>$u$：m 里填进 J、softmax 之前的那一块，源码里没有单独的名字 |
| mixes_a | 说明 | 一次投影同时给出 attn_pre、attn_post、attn_comb 三套系数<br>$W$ [24, 20480]、$b$ [24]、$s$ [3]（源码里叫 hc_attn_fn、hc_attn_base、hc_attn_scale），都是 fp32；24 = (2 + 4) × 4，20480 = 4 × 5120<br>mixes 与 $b$ 按顺序切成三段：前 $C$ 个给 pre，接着 $C$ 个给 post，最后 $C$ × $C$ 个给 comb；Sinkhorn 作用在整张 $u$ 上<br>Sinkhorn 先对 $u$ 的每一行做 softmax，再加 hc_eps；再按列除以（列和 + hc_eps）；然后行、列交替各再做 19 次，每次除以（和 + hc_eps）。行 20 次、列 20 次，最后一次是列 |
| cpre | 公式 | $x = \sum_{c=0}^{C-1} \pi_c\, x_c$ |
| cpre | 符号 | $x$：折叠后的一份<br>$\pi$：pre_mix，上一层返回的系数<br>$x_c$：第 c 份残差<br>$C$：hc_mult |
| cpre | 说明 | 按上一层返回的 pre_mix 把 4 份加权求和成 1 份 |
| an | 公式 | $x = w_a \odot \dfrac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}}$ |
| an | 符号 | $x$：送进来的张量<br>$w_a$：attn_norm 的权重，源码里叫 attn_norm.weight<br>$\epsilon$：1e-20<br>$\mathrm{mean}$：在最后一维 |
| an | 说明 | $w_a$ [5120]（源码里叫 attn_norm.weight） |
| attn | 公式 | $x = \mathrm{attn}(x, t)$ |
| attn | 符号 | $x$：attn_norm 之后的隐状态<br>$t$：start_pos，0<br>$attn\_args$：这次为空 |
| attn | 说明 | Attention，本层的窗口 KV 与压缩 KV 都在里面读写；本路径 attn_args 为空 |
| cpost | 公式 | $x_c = p_c\, x + \sum_{c'} \mathbf{J}_{c',c}\, \rho_{c'}$ |
| cpost | 符号 | $x$：注意力的输出<br>$\rho$：residual，同图的 hin，进注意力前的残差<br>$p$、$\mathbf{J}$：attn_post、attn_comb，hc_mixes 的系数<br>$c'$：$\rho$ 的份号<br>$c$：写回的份号 |
| cpost | 说明 | 把子层输出按 post 展开成 4 份，再把 residual 按 comb 混进去 |
| mixes_f | 公式 | $m' = \mathrm{flatten}(x, 2)\, W'^\top \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x, 2)^2) + \epsilon)$<br>$\pi_j = \mathrm{sigmoid}(m'_j \cdot s'_0 + b'_j) + \epsilon$<br>$p_j = 2\,\mathrm{sigmoid}(m'_{j+C} \cdot s'_1 + b'_{j+C})$<br>$u'_{j,k} = m'_{j \cdot C + k + 2 C} \cdot s'_2 + b'_{j \cdot C + k + 2 C}$<br>$\mathbf{J}' = \mathrm{Sinkhorn}(u')$ |
| mixes_f | 符号 | $m'$：mixes，混合系数<br>$\pi$、$p$、$\mathbf{J}'$：ffn_pre、ffn_post、ffn_comb，折叠、展开、混合用的系数<br>$W'$、$s'$、$b'$：前馈那次的参数，源码里叫 hc_ffn_fn、hc_ffn_scale、hc_ffn_base<br>$C$：hc_mult<br>$\epsilon$：Block 上的 norm_eps 与 hc_eps<br>$j$、$k$：从 0 到 C−1<br>$u'$：m′ 里填进 J′、softmax 之前的那一块，源码里没有单独的名字 |
| mixes_f | 说明 | $W'$ [24, 20480]、$b'$ [24]、$s'$ [3]（源码里叫 hc_ffn_fn、hc_ffn_base、hc_ffn_scale），形状与注意力那组相同<br>mixes 与 $b'$ 按顺序切成三段：前 $C$ 个给 pre，接着 $C$ 个给 post，最后 $C$ × $C$ 个给 comb；Sinkhorn 作用在整张 $u'$ 上<br>Sinkhorn 的做法与注意力那次相同：行 softmax 后加 hc_eps，再列归一化，然后交替各 19 次 |
| cpre2 | 公式 | $x = \sum_{c=0}^{C-1} a_c\, x_c$ |
| cpre2 | 符号 | $x$：折叠后的一份<br>$a$：attn_pre，本层注意力 hc_mixes 给出的 pre<br>$x_c$：第 c 份残差<br>$C$：hc_mult |
| cpre2 | 说明 | 用注意力子层算出的 $a$ |
| fn | 公式 | $x = w_f \odot \dfrac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}}$ |
| fn | 符号 | $x$：送进来的张量<br>$w_f$：ffn_norm 的权重，源码里叫 ffn_norm.weight<br>$\epsilon$：1e-20<br>$\mathrm{mean}$：在最后一维 |
| fn | 说明 | $w_f$ [5120]（源码里叫 ffn_norm.weight） |
| moe_op | 公式 | $x = \mathrm{ffn}(x, \mathbf{M})$ |
| moe_op | 符号 | $x$：ffn_norm 之后的隐状态<br>$\mathrm{ffn}$：本层的 MoE<br>$\mathbf{M}$：image_mask，这次传进来的张量，图像 span 的位置为 True |
| moe_op | 说明 | MoE，$\mathbf{M}$ 一路传到 Gate，用来在图像 span 的位置上换一份路由偏置 |
| cpost2 | 公式 | $x_c = p'_c\, x + \sum_{c'} \mathbf{J}'_{c',c}\, \rho'_{c'}$ |
| cpost2 | 符号 | $x$：前馈的输出<br>$\rho'$：residual，同图的 x4，进前馈前的残差<br>$p'$、$\mathbf{J}'$：ffn_post、ffn_comb，hc_mixes 的系数<br>$c'$：$\rho'$ 的份号<br>$c$：写回的份号 |
| cpost2 | 说明 | 把前馈输出按 ffn_post 展开成 4 份，再把 residual 按 ffn_comb 混进去 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 归一化因子 | $n = \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x, 2)^2) + \epsilon)$<br>算残差流的均方根缩放 |
| 公式 | 混合系数 | $m = \mathrm{flatten}(x, 2)\, W^\top \odot n$<br>投影出各份的混合系数 |
| 公式 | pre | $a_j = \mathrm{sigmoid}(m_j \cdot s_0 + b_j) + \epsilon$<br>由混合系数算折叠用的系数 |
| 公式 | post | $p_j = 2\,\mathrm{sigmoid}(m_{j+C} \cdot s_1 + b_{j+C})$<br>由混合系数算展开用的系数 |
| 公式 | comb | $\mathbf{J} = \mathrm{Sinkhorn}(u)$<br>把 u 迭代成混合矩阵 |
| 公式 | 注意力折叠 | $x = \sum_c \pi_c\, x_c$<br>按上一层给的 pre 把残差折成一份 |
| 公式 | 前馈折叠 | $x = \sum_c a_c\, x_c$<br>按注意力算的 pre 把残差折成一份 |
| 公式 | 展开 | $x_c = p_c\, x + \sum_{c'} \mathbf{J}_{c',c}\, \rho_{c'}$<br>按 post、comb 把子层输出写回多份残差 |
| 配置 | hc_mult | 4 |
| 配置 | hc_sinkhorn_iters | 20 |
| 配置 | hc_eps | 1e-06 |

## Attention.forward（model.py:765-789）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ax | 张量 | x | bf16 [B, T, 5120] | 已经过 attn_norm 的层输入 | model.py:766 |
| fcall | 张量 | self.freqs_cis | 复数 fp32 [max_seq_len, 32] | 构造时按 max_seq_len 预生成的整张频率表。32 = rope_head_dim / 2；频率基数按本层 compress_ratio 选：大于 0 用 compress_rope_theta、否则用 rope_theta | model.py:680-698 |
| fc | 张量 | freqs_cis | 复数 fp32 [T, 32] | 本次这一段。本路径 start_pos 为 0，所以是前 T 行 | model.py:767 |
| wqa | 算子 | self.wq_a(x) | bf16 [B, T, 5120] → bf16 [B, T, 1280] |  | model.py:770 |
| qra | 张量 | self.wq_a(x) | bf16 [B, T, 1280] | wq_a 的输出，源码里嵌在下一行的调用里，没有单独的变量名 | model.py:770 |
| qr_op | 算子 | self.q_norm(qra) | bf16 [B, T, 1280] → bf16 [B, T, 1280] |  | model.py:770 |
| qr | 张量 | qr | bf16 [B, T, 1280] | 索引器拿它当 query 的输入 | model.py:770 |
| qb | 算子 | self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim)) | bf16 [B, T, 1280] → bf16 [B, T, 64, 512] |  | model.py:771 |
| q | 张量 | q | bf16 [B, T, 64, 512] | | model.py:771 |
| rope_q | 算子 | apply_rotary_emb(q[..., -rd:], freqs_cis) | bf16 [B, T, 64, 512] → bf16 [B, T, 64, 512] |  | model.py:772 |
| q2 | 张量 | q | bf16 [B, T, 64, 512] | | model.py:772 |
| wkv_proj | 算子 | self.wkv(x) | bf16 [B, T, 5120] → bf16 [B, T, 512] |  | model.py:705 |
| wkva | 张量 | self.wkv(x) | bf16 [B, T, 512] | wkv 的输出，源码里嵌在下一行的调用里，没有单独的变量名 | model.py:705 |
| wkv_win | 算子 | self.kv_norm(wkva) | bf16 [B, T, 512] → bf16 [B, T, 512] |  | model.py:705 |
| wkv_o | 张量 | kv | bf16 [B, T, 512] | | model.py:705 |
| wrope | 算子 | apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis) | bf16 [B, T, 512] → bf16 [B, T, 512] |  | model.py:706 |
| wkv_r | 张量 | kv | bf16 [B, T, 512] | | model.py:706 |
| wquant | 算子 | act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True) | bf16 [B, T, 512] → bf16 [B, T, 512] |  | model.py:707 |
| wkv_q | 张量 | kv | bf16 [B, T, 512] | 存的是 fp8 量化后再反量化的值 | model.py:707 |
| wwrite | 算子 | self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1) | bf16 [B, T, 512] → [max_batch_size, 128, 512] |  | model.py:712-715 |
| wcache | 缓存 | window_kv_cache | [max_batch_size, 128, 512] | 每层一份；第一维是缓存容量（`max_batch_size` 默认 4，不是本次批大小 $B$），第二维是 128 个槽位的环；prefill 把最后 128 个 token 按 cutoff 绕进环里；decode 时改写 start_pos % 128 那一槽 | model.py:663-668 |
| wchunk | 张量 | window_kv | bf16 [B, T, 512] | prefill 就是本次这块 KV | model.py:716 |
| widx | 算子 | get_window_topk_idxs(win, bsz, seqlen, start_pos) | int win 与 int bsz 与 int seqlen 与 int start_pos → int32 [B, T, 128] |  | model.py:720 |
| widxs | 张量 | topk_idxs | int32 [B, T, 128] | | model.py:720 |
| comp_op | 算子 | self.compressor(x, start_pos) | [B, T, 5120] → [B, T/r, 512] |  | model.py:747 |
| lat | 张量 | latent | bf16 [B, T/r, 512] | 压缩器输出的潜在量，还没加 RoPE；r 是这一层的 compress_ratio（压缩比，每 r 个 token 池成一个压缩条目） | model.py:747 |
| cidx_op | 算子 | self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len) | [B, T, 5120] → int32 [B, T, 512] |  | model.py:750 |
| cidxs | 张量 | idxs | int32 [B, T, 512] | 压缩条目下标，已经偏移到拼起来的 KV 里的位置 | model.py:750 |
| crope | 算子 | apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs) | bf16 [B, T/r, 512] → bf16 [B, T/r, 512] |  | model.py:753-758 |
| lat_r | 张量 | latent | bf16 [B, T/r, 512] | | model.py:758 |
| cquant | 算子 | fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn) | bf16 [B, T/r, 512] → bf16 [B, T/r, 512] |  | model.py:760 |
| lat_q | 张量 | latent | bf16 [B, T/r, 512] | 存的是 fp4 量化后再反量化的值 | model.py:760 |
| cwrite | 算子 | self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent | bf16 [B, T/r, 512] → [max_batch_size, max_seq_len / r, 512] |  | model.py:761 |
| ccache | 缓存 | compress_kv_cache | [max_batch_size, max_seq_len / r, 512] | 2、8、14、20 各有一份；第一维是缓存容量（`max_batch_size` 默认 4，不是本次批大小 $B$），第二维是 max_seq_len / r 个槽；每个 kv source 层写自己这一份并把它发布给 shared_attn，它之后到下一个 kv source 层之前的层读的是这一份；decode 时每组填满才写一行 | model.py:669-679 |
| ckv | 张量 | compress_kv | bf16 [B, T/r, 512] | 从 compress_kv_cache 读出的、本次可见的部分 | model.py:763 |
| cat | 算子 | torch.cat([kv, compress_kv], dim=1) | [B, T, 512] 与 [B, T/r, 512] → [B, T + T/r, 512] |  | model.py:777 |
| kvc | 张量 | kv | bf16 [B, T + T/r, 512] | | model.py:777 |
| cat2 | 算子 | torch.cat([topk_idxs, compress_idxs], dim=-1) | int32 [B, T, 128] 与 int32 [B, T, 512] → int32 [B, T, 640] |  | model.py:778 |
| tki | 张量 | topk_idxs | int32 [B, T, 128 + 512] | | model.py:778 |
| attn_sink | 张量 | self.attn_sink | fp32 [64] | 可学习注意力汇点，每个头一个 | model.py:780 |
| core | 算子 | sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale) | q [B, T, 64, 512] 对 kv [B, T + T/r, 512] → [B, T, 64, 512] |  | model.py:780 |
| o0 | 张量 | o | bf16 [B, T, 64, 512] | | model.py:780 |
| unrope | 算子 | apply_rotary_emb(o[..., -rd:], freqs_cis, True) | bf16 [B, T, 64, 512] → bf16 [B, T, 64, 512] |  | model.py:781 |
| o1 | 张量 | o | bf16 [B, T, 64, 512] | | model.py:781 |
| eins | 算子 | torch.einsum("bsgd,grd->bsgr", o.view(bsz, seqlen, self.n_local_groups, -1), self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)) | [B, T, 64, 512] → [B, T, 8, 1024] |  | model.py:785-787 |
| wav | 张量 | self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1) | bf16 [8, 1024, 4096] | wo_a 的权重按组摊开（纯 view），当 einsum 的第二个操作数 | model.py:786 |
| ol | 张量 | o | bf16 [B, T, 8, 1024] | | model.py:787 |
| wb | 算子 | self.wo_b(o.flatten(2)) | bf16 [B, T, 8192] → bf16 [B, T, 5120] |  | model.py:788 |
| axout | 张量 | x | bf16 [B, T, 5120] | | model.py:788-789 |

### 连线

- ax → wqa
- wqa → qra
- qra → qr_op
- qr_op → qr
- qr → qb
- qb → q
- q → rope_q
- fcall → fc：切出本次这一段
- fc → rope_q
- rope_q → q2
- ax → wkv_proj
- wkv_proj → wkva
- wkva → wkv_win
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
- wchunk → cidx_op：长度（offset）
- cidx_op → cidxs
- lat → crope
- fcall → crope：切出压缩位置的频率
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
- q2 → core
- kvc → core
- tki → core
- attn_sink → core
- core → o0
- o0 → unrope
- fc → unrope
- unrope → o1
- o1 → eins
- wav → eins：wo_a.weight 按组摊开（源码里 einsum 的第二操作数就是它）
- eins → ol
- ol → wb
- wb → axout
- comp_op 点开 → Compressor.forward
- cidx_op 点开 → Indexer.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| wqa | 公式 | $u_q = x\, W_q^\top$ |
| wqa | 符号 | $u_q$：第一层低秩投影的输出<br>$x$：本层输入<br>$W_q$：它的权重，源码里叫 wq_a.weight |
| wqa | 说明 | $W_q$ [1280, 5120]（源码里叫 wq_a.weight）。权重是 fp8，进入 GEMM 前先把激活 $x$ 按 32 通道一块量化成 fp8（`linear()` 里的 `act_quant`，model.py:186-197） |
| qr_op | 公式 | $qr = w_q \odot \dfrac{u_q}{\sqrt{\mathrm{mean}(u_q^2) + \epsilon}}$ |
| qr_op | 符号 | $qr$：归一化后的低秩 query<br>$u_q$：上一节点的输出<br>$w_q$：q_norm 的权重，源码里叫 q_norm.weight<br>$\mathrm{mean}$：在最后一维 |
| qr_op | 说明 | $w_q$ [1280]（源码里叫 q_norm.weight），q_norm 是 RMSNorm，eps 取 1e-20 |
| wkv_proj | 公式 | $u_{kv} = x\, W_{kv}^\top$ |
| wkv_proj | 符号 | $u_{kv}$：窗口 KV 投影的输出<br>$x$：本层输入<br>$W_{kv}$：它的权重，源码里叫 wkv.weight |
| wkv_proj | 说明 | $W_{kv}$ [512, 5120]（源码里叫 wkv.weight）；fp8 权重，同样是激活先量化成 fp8 再进 GEMM |
| wkv_win | 公式 | $u_{kv} = w_{kv} \odot \dfrac{u_{kv}}{\sqrt{\mathrm{mean}(u_{kv}^2) + \epsilon}}$ |
| wkv_win | 符号 | $u_{kv}$：归一化后的窗口 KV<br>$w_{kv}$：kv_norm 的权重，源码里叫 kv_norm.weight<br>$\mathrm{mean}$：在最后一维 |
| wkv_win | 说明 | $w_{kv}$ [512]（源码里叫 kv_norm.weight），kv_norm 是 RMSNorm，eps 取 1e-20 |
| qb | 公式 | $q = \mathrm{unflatten}(qr\, W_q'^\top, (H, d_h))$ |
| qb | 符号 | $q$：展开成各头的 query<br>$qr$：上一行的输出<br>$W_q'$：第二层投影，源码里叫 wq_b.weight<br>$H$：n_local_heads，本卡的头数，单卡下等于 n_heads 64<br>$d_h$：head_dim，512 |
| qb | 说明 | $W_q'$ [32768, 1280]（源码里叫 wq_b.weight），64 × 512 = 32768；fp8 权重，激活先量化 |
| rope_q | 公式 | $q$ 的末 $d_r$ 维乘一个旋转因子（复数形式 $e^{i \varphi_t}$，逐位置） |
| rope_q | 符号 | $q$：query<br>$d_r$：rd（rope_head_dim），64<br>$\varphi_t$：第 $t$ 个位置的旋转角<br>源码写法：$\mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(q.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot \mathcal{F}_t))$ |
| rope_q | 说明 | 只旋转每头 512 维里的后 64 维（相邻两维配成一对复数），前 448 维不动；写在原地 |
| wrope | 公式 | $\mathbf{K}$ 的末 $d_r$ 维乘同一个旋转因子 $e^{i \varphi_t}$ |
| wrope | 符号 | $\mathbf{K}$：窗口 KV<br>$d_r$：rope_head_dim，64<br>$\varphi_t$：与 query 同一份频率给出的旋转角<br>源码写法：$\mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(\mathbf{K}.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot \mathcal{F}_t))$ |
| wrope | 说明 | 和 query 用同一份 freqs_cis；保留的未旋转部分就是 nope（不加旋转位置编码的一段，no position embedding）那 448 维 |
| wquant | 公式 | $\sigma_K = \mathrm{fast\_round\_scale}(\max(\mathrm{absmax}(\mathbf{K}), 10^{-4}), 1/448)$<br>$\mathbf{K} = \mathrm{bf16}(\mathrm{fp8}(\mathrm{clamp}(\mathbf{K} / \sigma_K, -448, 448)) \cdot \sigma_K)$ |
| wquant | 符号 | $\sigma_K$：量化 scale，窗口 KV 的<br>$\mathbf{K}$：加过 RoPE 的窗口 KV<br>$\mathrm{absmax}$：按每 32 个数一组取，不是逐元素<br>$\mathrm{fast\_round\_scale}$：把 scale 收成 2 的幂 |
| wquant | 说明 | 每 32 个通道一个 E8M0 scale（E8M0 是只有 8 位指数、没有尾数的缩放格式），原地量化再反量化；量化按 fp8 精度（压缩 KV 走 fp4）<br>量化覆盖整条 512 维，含已经加过 RoPE 的末 64 维 |
| wwrite | 公式 | $b = T \bmod w$<br>$\mathbf{W}_{:,b:w},\ \mathbf{W}_{:,:b} = \mathrm{split}(\mathbf{K}_{:,-w:}, [w-b,\,b])$ |
| wwrite | 符号 | $b$：cutoff，seqlen % win<br>$T$：seqlen<br>$w$：win（window_size），128<br>$\mathbf{W}$：window_kv_cache，窗口 KV 缓存<br>$\mathbf{K}$：量化后的窗口 KV |
| wwrite | 说明 | 按 $b$ 把最后 $w$ 个 token 切成两段，写进 $\mathbf{W}$ 的环首和环尾，尾巴绕回环首 |
| widx | 公式 | $\mathbf{X} = \mathrm{clamp}(e - w + 1, 0) + \{0,\ldots,w-1\}$<br>$\mathbf{G}_w = \mathrm{where}(\mathbf{X} > e, -1, \mathbf{X})$ |
| widx | 符号 | $\mathbf{X}$：idx_base，候选下标<br>$\mathbf{G}_w$：idxs，最终下标<br>$e$：end，query 的位置<br>$w$：window_size，128 |
| widx | 说明 | 只吃 win、bsz、seqlen、start_pos 四个标量，不接 $x$。晚于这个 query 的位置写成 −1；再扩成 batch 并转成 int32。prefill 时每个 query 一行，下标是这次 window_kv（长度 $T$）里的位置 |
| comp_op | 公式 | $\mathbf{Z} = \mathrm{compressor}(x, s)$ |
| comp_op | 符号 | $\mathbf{Z}$：latent，压缩器的输出<br>$\mathrm{compressor}$：本层的压缩器<br>$x$：本层输入<br>$s$：start_pos，0 |
| comp_op | 说明 | 只有 4 个 kv source 层（2、8、14、20）算；其余 compress_ratio 大于 0 的层 $\mathbf{Z}$ 为 None，直接读已有缓存；层 0、1 的压缩比为 0，整段压缩都不走 |
| cidx_op | 公式 | $\mathbf{G}_c = \mathrm{compress\_topk\_idxs}(x, qr, \mathbf{Z}, s, L_w, L_c)$ |
| cidx_op | 符号 | $\mathbf{G}_c$：idxs，压缩条目下标<br>$x$：本层输入<br>$qr$：低秩 query<br>$\mathbf{Z}$：latent，压缩器的输出<br>$s$：start_pos，0<br>$L_w$：offset，窗口 KV 的长度<br>$L_c$：compress_len，(start_pos + T) / r |
| cidx_op | 说明 | 本层是 index source（2、8、14、20、24、28、32、36）就跑自己的 Indexer，其余 compress_ratio 大于 0 的层直接复用 shared_attn.topk_idxs |
| crope | 公式 | $\mathbf{Z}$ 的末 $d_r$ 维乘一个旋转因子 $e^{i \varphi_j}$（压缩位置 $j$ 的频率） |
| crope | 符号 | $\mathbf{Z}$：latent，压缩潜在量<br>$d_r$：rope_head_dim，64<br>$\varphi_j$：压缩位置 $j$ 的旋转角，取自 freqs_cis 的第 $j \cdot r$ 行<br>源码写法：$\mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(\mathbf{Z}.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot f))$ |
| crope | 说明 | 一个 $\mathbf{Z}$ 顶替一整组，第 j 组取位置 j × $r$ 的频率 |
| cquant | 公式 | $\sigma_Z = \mathrm{fp8}(\max(\mathrm{absmax}(\mathbf{Z}), 6 \cdot 2^{-9}) / 6)$<br>$\mathbf{Z} = \mathrm{bf16}(\mathrm{fp4}(\mathrm{clamp}(\mathbf{Z} / \sigma_Z, -6, 6)) \cdot \sigma_Z)$ |
| cquant | 符号 | $\sigma_Z$：量化 scale，压缩 KV 的，存成 E4M3<br>$\mathbf{Z}$：latent，加过 RoPE 的压缩 KV<br>$\mathrm{absmax}$：按每 16 个数一组取 |
| cquant | 说明 | E2M1 格式（1 符号位、2 指数位、1 尾数位），每 16 个通道一个 E4M3 scale（4 指数位、3 尾数位），原地量化再反量化 |
| cwrite | 公式 | $\mathbf{N}_{:B,\ 0:T/r} = \mathbf{Z}$ |
| cwrite | 符号 | $\mathbf{N}$：compress_kv_cache，压缩 KV 缓存<br>$\mathbf{Z}$：latent，量化后的压缩 KV<br>$s$：start_pos，0 |
| cwrite | 说明 | $s$ 为 0，所以从第 0 行起整块写 |
| cat | 公式 | $\mathbf{K}_\mathrm{all} = \mathrm{cat}([\mathbf{K}, \mathbf{N}], \dim=1)$ |
| cat | 符号 | $\mathbf{K}_\mathrm{all}$：kv，拼起来的 KV<br>$\mathbf{K}$：窗口 KV，同时充当 k 和 v<br>$\mathbf{N}$：compress_kv，压缩 KV |
| cat | 说明 | 窗口在前、压缩在后 |
| cat2 | 公式 | $\mathbf{G} = \mathrm{cat}([\mathbf{G}_w, \mathbf{G}_c], \dim=-1)$ |
| cat2 | 符号 | $\mathbf{G}$：topk_idxs，拼起来的下标<br>$\mathbf{G}_w$：窗口 topk_idxs，窗口下标<br>$\mathbf{G}_c$：compress_idxs，压缩下标 |
| cat2 | 说明 | 压缩下标在 Indexer.forward 的 shift 里已经加上窗口 KV 的长度 $T$ |
| core | 公式 | $o = \dfrac{\sum_{\iota} \exp(q \cdot \mathbf{K}_{\mathrm{all},\iota} \cdot \mu - M) \, \mathbf{K}_{\mathrm{all},\iota}}{\sum_{\iota} \exp(q \cdot \mathbf{K}_{\mathrm{all},\iota} \cdot \mu - M) + \exp(\kappa - M)}$ |
| core | 符号 | $o$：注意力输出<br>$q$：查询<br>$\mathbf{K}_\mathrm{all}$：同时充当 k 和 v<br>$\iota$：$\mathbf{G}$ 里的槽位（不是位置）<br>$\mu$：softmax_scale，注意力自己的，取 head_dim（512）的 −1/2 次方<br>$\kappa$：可学习注意力汇点，源码里叫 attn_sink<br>$M$：这些槽位上 $q \cdot \mathbf{K}_{\mathrm{all},\iota} \cdot \mu$ 的最大值 |
| core | 说明 | 单 KV 头，64 个查询头看同一份 KV；公式里的注意力汇点是可学习参数 [64]（源码里叫 attn_sink）<br>下标为 −1 的槽位分子分母都不贡献 |
| unrope | 公式 | $o$ 的末 $d_r$ 维乘旋转因子 $e^{-i \varphi_t}$（query 那个旋转角的共轭） |
| unrope | 符号 | $o$：注意力输出<br>$d_r$：rd（rope_head_dim）<br>$\varphi_t$：query 所在位置的旋转角，取共轭<br>源码写法：$\mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(o.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot \mathcal{F}_t^*))$ |
| unrope | 说明 | 用 query 所在位置的共轭频率再转输出的末 64 维，去掉 query 的旋转。值里各键位置的旋转变成相对 query 的位置差 |
| eins | 公式 | $o_{b,t,g,m} = \sum_d o_{b,t,g,d}\, \mathbf{A}_{g,m,d}$ |
| eins | 符号 | $o$：按组摊开的注意力输出<br>$\mathbf{A}$：W_a 按组摊开的结果，源码里叫 wo_a.weight<br>$g$：o_groups 的组号（8）<br>$m$：o_lora_rank 维（1024）<br>$d$：每组自己的那 4096 维 |
| eins | 说明 | 每组只和本组的权重相乘，等效块对角矩阵。源码里 $o$.view(bsz, seqlen, 8, -1)（64 个头按 8 个一组摊开）与 wo_a.weight.view(8, 1024, -1)（权重按组摊开）都是纯 view，并进这一步 |
| wb | 公式 | $x = \mathrm{flatten}(o)\, W_b^\top$ |
| wb | 符号 | $x$：输出<br>$o$：eins 的输出<br>$W_b$：输出投影，源码里叫 wo_b.weight |
| wb | 说明 | $W_b$ [5120, 8192]（源码里叫 wo_b.weight），8192 = 8 × 1024；fp8 权重，激活先量化 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 频率 | $\theta_j = \mathrm{base}^{-2j/64}$<br>算 RoPE 各维的基础频率 |
| 公式 | 频率基数 | $\mathrm{base} = \begin{cases} \theta_c & r > 0 \\ \theta_0 & r = 0 \end{cases}$<br>走压缩的层用压缩基数，纯窗口层用 rope_theta |
| 公式 | YaRN | $\theta_j' = (\theta_j/16)\,\rho_j + \theta_j\,(1 - \rho_j)$<br>按 YaRN 斜坡缩放各维频率做长上下文扩展 |
| 公式 | query 低秩投影 | $u_q = x\, W_q^\top$<br>把本层输入投成低秩 query |
| 公式 | query 展开 | $q = \mathrm{unflatten}(qr\, W_q'^\top, (H, d_h))$<br>把低秩 query 展开成各头 |
| 公式 | query 归一化 | $qr = w_q \odot u_q / \sqrt{\mathrm{mean}(u_q^2) + \epsilon}$<br>对低秩 query 做 RMSNorm |
| 公式 | 窗口 KV | $u_{kv} = x\, W_{kv}^\top$<br>把本层输入投成窗口 KV |
| 公式 | 窗口 KV 归一化 | $u_{kv} = w_{kv} \odot u_{kv} / \sqrt{\mathrm{mean}(u_{kv}^2) + \epsilon}$<br>对窗口 KV 做 RMSNorm |
| 公式 | 压缩 KV 的位置 | $j r \le t \le (j+1)r - 1$<br>第 j 个 latent 顶替这一段 token |
| 公式 | 稀疏注意力 | $o = \dfrac{\sum_{\iota} \exp(q \cdot \mathbf{K}_{\mathrm{all},\iota} \cdot \mu - M)\, \mathbf{K}_{\mathrm{all},\iota}}{\sum_{\iota} \exp(q \cdot \mathbf{K}_{\mathrm{all},\iota} \cdot \mu - M) + \exp(\kappa - M)}$<br>对窗口与压缩 KV 做带 sink 的注意力 |
| 公式 | 分组低秩投影 | $o_{b,t,g,m} = \sum_d o_{b,t,g,d}\, \mathbf{A}_{g,m,d}$<br>按组对注意力输出做低秩投影 |
| 公式 | 输出投影 | $x = \mathrm{flatten}(o)\, W_b^\top$<br>把各组投影拼起来投到主干维度 |
| 配置 | window_size | 128 |
| 配置 | index_topk | 512 |
| 配置 | rope_theta | 10000 |
| 配置 | compress_rope_theta | 160000 |
| 配置 | rope_factor | 16 |
| 配置 | original_seq_len | 65536 |
| 配置 | beta_fast | 32 |
| 配置 | beta_slow | 1 |

## Compressor.forward（model.py:458-485）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ax | 张量 | x | bf16 [B, T, 5120] | kv source 层的注意力输入，与注意力、索引器两张图里的 `ax` 是同一份 | model.py:459 |
| projk | 算子 | self.wkv(x) | fp32 [B, T, 5120] → fp32 [B, T, 512] |  | model.py:465 |
| kv0 | 张量 | kv | fp32 [B, T, 512] | 逐 token 的值投影 | model.py:465 |
| projg | 算子 | self.wgate(x) | [B, T, 5120] → [B, T, 512] |  | model.py:465 |
| csc | 张量 | score | fp32 [B, T, 512] | 逐 token 的门控打分 | model.py:465 |
| pool | 算子 | (kv * score.softmax(dim=2)).sum(dim=2) | [B, T/r, r, 512] → [B, T/r, 512] |  | model.py:475 |
| kvg | 张量 | kv | fp32 [B, T/r, 512] | 组内按 score 加权求和后的 kv | model.py:475 |
| cnorm | 算子 | self.norm(kv.to(dtype)) | fp32 [B, T/r, 512] → bf16 [B, T/r, 512] |  | model.py:485 |
| lat | 张量 | self.norm(kv.to(dtype)) | bf16 [B, T/r, 512] | 返回值是还没加 RoPE 的压缩潜在量；层 2、8、14 汇自 cnorm，层 20 汇自 cnorm1 | model.py:485 |
| projk1 | 算子 | self.wkv(x) | bf16 [B, T, 5120] → bf16 [B, T, 512] |  | model.py:462 |
| kv1 | 张量 | self.wkv(x) | bf16 [B, T, 512] | 层 20 的值投影，在 bf16 里算，没有 fp32 上转与 wgate | model.py:462 |
| cnorm1 | 算子 | self.norm(kv1) | bf16 [B, T, 512] → bf16 [B, T, 512] |  | model.py:462 |

### 连线

- ax → projk：先 x.float() 上转 fp32，池化与门控都在 fp32 里算（层 2、8、14）
- ax → projg：同样先上转 fp32
- projk → kv0
- projg → csc
- kv0 → pool：按 r 个一组重排成 [B, T/r, r, 512]
- csc → pool：同样重排成 [B, T/r, r, 512]
- pool → kvg
- kvg → cnorm
- cnorm → lat
- ax → projk1：层 20 的输入（层 20 ratio 为 1，不转 fp32）
- projk1 → kv1
- kv1 → cnorm1
- cnorm1 → lat：层 20 的返回值

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| projk | 公式 | $\mathbf{U} = x\, W_{kv}^\top$ |
| projk | 符号 | $\mathbf{U}$：kv，逐 token 的值投影<br>$x$：本层输入<br>$W_{kv}$：值投影的权重，源码里叫 wkv.weight |
| projk | 说明 | $W_{kv}$ [512, 5120]（源码里叫 wkv.weight），压缩比大于 1 时是 fp32。进 GEMM 前先把 $x$ 上转 fp32（`x.float()`），池化与门控都在 fp32 里算；只有 compress_ratio 大于 1 的 kv source 层（2、8、14）走这一支 |
| projg | 公式 | $\mathbf{g} = x\, W_g^\top$ |
| projg | 符号 | $\mathbf{g}$：score，逐 token 的门控打分<br>$x$：本层输入<br>$W_g$：门控投影的权重，源码里叫 wgate.weight |
| projg | 说明 | $W_g$ [512, 5120]（源码里叫 wgate.weight），压缩比大于 1 时是 fp32 |
| pool | 公式 | $\mathbf{U} = \sum_t \mathrm{softmax}(\mathbf{G}_g)_t \, \mathbf{U}_{g,t}$ |
| pool | 符号 | $\mathbf{U}$：kv，组内按 score 加权求和后的张量<br>$\mathbf{G}_g$：分组后的 score，按 $r$ 个一组重排<br>$\mathbf{U}_{g,t}$：组内第 $t$ 个 token 的值<br>$t$：组内位置，从 0 到 $r-1$ |
| pool | 说明 | 组内用 score 的 softmax 做加权求和，把 $r$ 个 token 池成一个压缩条目；$r$ 就是这一层的 compress_ratio；源码写法 `(kv * score.softmax(dim=2)).sum(dim=2)` |
| cnorm | 公式 | $\mathbf{Z} = w \odot \dfrac{\mathbf{U}.\mathrm{to}(dtype)}{\sqrt{\mathrm{mean}(\mathbf{U}^2) + \epsilon}}$ |
| cnorm | 符号 | $\mathbf{Z}$：latent，压缩器的输出<br>$\mathbf{U}$：kv，池化后的张量<br>$w$：compressor.norm 的权重，源码里叫 norm.weight<br>$\epsilon$：1e-20<br>$\mathrm{mean}$：在最后一维<br>$\mathrm{to}(dtype)$：转回 bf16 |
| cnorm | 说明 | 池化结果先转回 bf16 再 RMSNorm，$w$ [512]（源码里叫 layers.{i}.attn.compressor.norm.weight） |
| projk1 | 公式 | $\mathbf{U} = x\, W_{kv}^\top$ |
| projk1 | 符号 | $\mathbf{U}$：kv，逐 token 的值投影<br>$x$：本层输入<br>$W_{kv}$：值投影的权重，源码里叫 wkv.weight |
| projk1 | 说明 | 层 20 的 $W_{kv}$ 是 bf16（compress_ratio 不大于 1），没有 $x$.float() 与 wgate；层 20 的 ratio 为 1，没有池化、没有门控 |
| cnorm1 | 公式 | $\mathbf{Z} = w \odot \dfrac{\mathbf{U}}{\sqrt{\mathrm{mean}(\mathbf{U}^2) + \epsilon}}$ |
| cnorm1 | 符号 | $\mathbf{Z}$：latent，压缩器的输出<br>$\mathbf{U}$：kv，projk1 的输出（层 20 的 wkv，在 bf16 里算）<br>$w$：compressor.norm 的权重，源码里叫 norm.weight<br>$\epsilon$：源码里叫 norm.eps<br>$\mathrm{mean}$：在最后一维 |
| cnorm1 | 说明 | $w$ [512]（源码里叫 layers.20.attn.compressor.norm.weight） |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 池化 | $\mathbf{U} = \sum_t \mathrm{softmax}(\mathbf{G}_g)_t \, \mathbf{U}_{g,t}$<br>按 ratio 分组对 token 做加权池化 |
| 公式 | 输出 | $\mathbf{Z} = w \odot \mathbf{U} / \sqrt{\mathrm{mean}(\mathbf{U}^2) + \epsilon}$<br>对池化结果做归一化后输出 |
| 配置 | compress_ratios | 前 40 项是主干：层 0、1 为 0，层 2–19 为 2，层 20–39 为 1；末 3 项是 MTP 层，为 0 |

## Indexer.forward（model.py:527-580）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ax | 张量 | x | bf16 [B, T, 5120] | kv source 层的注意力输入，与注意力、压缩器两张图里的 `ax` 是同一份 | model.py:533 |
| qr | 张量 | qr | bf16 [B, T, 1280] | 注意力那边算出的低秩 query，索引器直接拿它当输入 | model.py:527 |
| lat | 张量 | latent | bf16 [B, T/r, 512] | 压缩器的输出，还没加 RoPE；只有 2、8、14、20 这四个 kv source 层是实际张量，其余 index source 层（24、28、32、36）不建键、传进来的是 None，图里这一段整个跳过，只从 shared_attn.index_k 读键 | model.py:527、537、500 |
| fcall | 张量 | self.freqs_cis | 复数 fp32 [max_seq_len, 32] | 注意力那边传进来的整张频率表，与注意力图里的 fcall 是同一份 | model.py:733-734 |
| wka | 张量 | self.wk(latent) | bf16 [B, T/r, 128] | wk 的输出（未归一化），源码里嵌在下一行的调用里，没有单独的变量名 | model.py:544 |
| kw | 算子 | self.k_norm(wka) | bf16 [B, T/r, 128] → bf16 [B, T/r, 128] |  | model.py:544 |
| k0 | 张量 | k | bf16 [B, T/r, 128] | | model.py:544 |
| krope | 算子 | apply_rotary_emb(k[..., -rd:], freqs) | bf16 [B, T/r, 128] → bf16 [B, T/r, 128] |  | model.py:545 |
| kr | 张量 | k | bf16 [B, T/r, 128] | | model.py:545 |
| kq | 算子 | fp4_act_quant(k, fp4_block_size, True) | bf16 [B, T/r, 128] → bf16 [B, T/r, 128] |  | model.py:546 |
| kf | 张量 | k | bf16 [B, T/r, 128] | 存的是 fp4 量化后再反量化的值 | model.py:546 |
| kwrite | 算子 | self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k | bf16 [B, T/r, 128] → [max_batch_size, max_seq_len / r, 128] |  | model.py:547-548 |
| kcache | 缓存 | k_cache | [max_batch_size, max_seq_len / r, 128] | 2、8、14、20 各有一份；第一维是缓存容量（`max_batch_size` 默认 4，不是本次批大小 $B$），第二维是 max_seq_len / r 个槽。`self.owns_k` 的层（2、8、14、20）写完发布给 shared_attn.index_k，跳过上面的建键段的层（24、28、32、36）直接读这一份 | model.py:520-525 |
| qop | 算子 | self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim)) | bf16 [B, T, 1280] → bf16 [B, T, 32, 128] |  | model.py:550 |
| iq0 | 张量 | q | bf16 [B, T, 32, 128] | | model.py:550 |
| qrope | 算子 | apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos]) | bf16 [B, T, 32, 128] → bf16 [B, T, 32, 128] |  | model.py:551 |
| iqr | 张量 | q | bf16 [B, T, 32, 128] | | model.py:551 |
| iq_q | 算子 | fp4_act_quant(q, fp4_block_size, True) | bf16 [B, T, 32, 128] → bf16 [B, T, 32, 128] |  | model.py:552 |
| iq | 张量 | q | bf16 [B, T, 32, 128] | 存的是 fp4 量化后再反量化的值 | model.py:552 |
| ik | 张量 | index_k | bf16 [B, T/r, 128] | 从 shared_attn.index_k 读出的、本次可见的部分 | model.py:554 |
| wproj | 算子 | self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5) | bf16 [B, T, 5120] → bf16 [B, T, 32] |  | model.py:555 |
| iwt | 张量 | weights | bf16 [B, T, 32] | | model.py:555 |
| idot | 算子 | torch.einsum("bshd,btd->bsht", q, index_k) | [B, T, 32, 128] 与 [B, T/r, 128] → [B, T, 32, T/r] |  | model.py:556 |
| sc0 | 张量 | index_score | bf16 [B, T, 32, T/r] | | model.py:556 |
| red | 算子 | (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2) | [B, T, 32, T/r] → [B, T, T/r] |  | model.py:557 |
| isc | 张量 | index_score | bf16 [B, T, T/r] | | model.py:557 |
| imask | 算子 | index_score.masked_fill_(torch.arange(seqlen // ratio) >= compress_lens, -torch.inf) | [B, T, T/r] → [B, T, T/r] |  | model.py:565 |
| iscm | 张量 | index_score | bf16 [B, T, T/r] | 掩掉看不到的位置之后的打分 | model.py:565 |
| csel | 算子 | select_candidate_blocks(index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size) | [B, T, T/r] → bool [B, T, T/r] |  | model.py:569-571 |
| cand | 张量 | shared_attn.candidates | bool [B, T, T/r] | 两个名字是同一个 shared_attn.candidates：层 20 里 select_candidate_blocks 的返回值（源码里没有赋给局部变量），也就是 published 给共享状态的那一份；已按块展开成逐个位置的 True/False | model.py:569-571 |
| cmf | 算子 | index_score.masked_fill(~shared_attn.candidates, -torch.inf) | [B, T, T/r] 与 bool [B, T, T/r] → [B, T, T/r] |  | model.py:575 |
| cmfd | 张量 | index_score | bf16 [B, T, T/r] | 候选块外的位置置成 −∞ 之后的打分 | model.py:575 |
| topk | 算子 | index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values | bf16 [B, T, T/r] → int64 [B, T, 512] |  | model.py:578-579 |
| isel | 张量 | idxs | int64 [B, T, 512] | | model.py:579 |
| shift | 算子 | torch.where(idxs < compress_lens, idxs + offset, -1).int() | int64 [B, T, 512] → int32 [B, T, 512] |  | model.py:580 |
| cidxs | 张量 | idxs | int32 [B, T, 512] | 交给注意力那边拼进 topk_idxs | model.py:580 |

### 连线

- qr → qop
- qop → iq0
- iq0 → qrope
- fcall → qrope
- qrope → iqr
- iqr → iq_q
- iq_q → iq
- lat → kw：wk 投影（键向量 $u_k$）+ $k$ 归一化；只在本层 owns_k、latent 非空时走（2、8、14、20）
- kw → k0
- k0 → krope
- fcall → krope：切出压缩位置的频率
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
- ax → imask：形状（seqlen）与 device
- isc → imask
- imask → iscm
- iscm → topk：不走候选块过滤的层（2、8、14、20），从这个打分直接取 top-512
- iscm → csel：只在层 20，走候选块过滤这一支
- csel → cand
- cand → cmf
- iscm → cmf
- cmf → cmfd
- cmfd → topk：用候选块的层（24、28、32、36），在掩过候选块后的打分上取 top-512
- topk → isel
- isel → shift
- ax → shift：compress_lens（由 seqlen 与 r 算出，与 imask 用的是同一个标量）
- shift → cidxs

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| kw | 公式 | $\mathbf{J} = w_k \odot \dfrac{u_k}{\sqrt{\mathrm{mean}(u_k^2) + \epsilon}}$ |
| kw | 符号 | $\mathbf{J}$：k，归一化后的索引键<br>$u_k$：wk(latent) 的输出，本图把投影与归一化并成这一步<br>$w_k$：k_norm 的权重，源码里叫 k_norm.weight<br>$\mathrm{mean}$：在最后一维 |
| kw | 说明 | $w_k$ [128]（源码里叫 k_norm.weight），k_norm 是 RMSNorm，eps 取 1e-20；$W_k$ [128, 512]（源码里叫 wk.weight），bf16。只在本层 owns_k 且 latent 非空时执行 |
| krope | 公式 | $\mathbf{J}$ 的末 $d_r$ 维乘旋转因子 $e^{i \varphi_j}$（压缩位置 $j$ 的频率） |
| krope | 符号 | $\mathbf{J}$：索引键<br>$d_r$：rd（rope_head_dim）<br>$\varphi_j$：压缩位置 $j$ 的旋转角<br>源码写法：$\mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(\mathbf{J}.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot f))$ |
| krope | 说明 | 128 维里的最后 64 维加 RoPE；第 $j$ 组取位置 $j$ × $r$ 的频率 |
| kq | 公式 | $\sigma_K = \mathrm{fast\_round\_scale}(\max(\mathrm{absmax}(\mathbf{J}), 6 \cdot 2^{-126}), 1/6)$<br>$\mathbf{J} = \mathrm{bf16}(\mathrm{fp4}(\mathrm{clamp}(\mathbf{J} / \sigma_K, -6, 6)) \cdot \sigma_K)$ |
| kq | 符号 | $\mathbf{J}$：加过 RoPE 的索引键<br>$\sigma_K$：量化 scale，索引键的<br>$\mathrm{absmax}$：按每 32 个数一组取 |
| kq | 说明 | E2M1 格式，每 32 个通道一个 E8M0 scale，原地量化再反量化 |
| kwrite | 公式 | $\mathbf{J}^I_{:B,\ 0:T/r} = \mathbf{J}$ |
| kwrite | 符号 | $\mathbf{J}^I$：k_cache，索引键缓存<br>$\mathbf{J}$：量化后的索引键<br>$s$：start_pos，0 |
| kwrite | 说明 | $s$ 为 0，从第 0 行起整块写；写完把这份缓存发布到 shared_attn.index_k，后面的层直接读。只在 owns_k 的层（2、8、14、20）执行；$q$ 前面的 wq_b 投影与展开 unflatten(-1, (n_local_heads, index_head_dim)) 一起并进 qop |
| qrope | 公式 | $q$ 的末 $d_r$ 维乘旋转因子 $e^{i \varphi_t}$（第 $t$ 个位置的频率，从 start_pos 取到 end_pos） |
| qrope | 符号 | $q$：索引 query<br>$d_r$：rd（rope_head_dim）<br>$\varphi_t$：位置 $t$ 的旋转角<br>源码写法：$\mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(q.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot \mathcal{F}_t))$ |
| qrope | 说明 | 乘从 start_pos 取到 end_pos 的 $\mathcal{F}_t$ |
| iq_q | 公式 | $\sigma_q = \mathrm{fast\_round\_scale}(\max(\mathrm{absmax}(q), 6 \cdot 2^{-126}), 1/6)$<br>$q = \mathrm{bf16}(\mathrm{fp4}(\mathrm{clamp}(q / \sigma_q, -6, 6)) \cdot \sigma_q)$ |
| iq_q | 符号 | $q$：加过 RoPE 的索引 query<br>$\sigma_q$：量化 scale，索引 query 的<br>$\mathrm{absmax}$：按每 32 个数一组取 |
| iq_q | 说明 | 索引器的 query 也量化成 fp4 |
| wproj | 公式 | $\mathbf{w} = (x\, W_w^\top) \cdot \mu_i \cdot H_i^{-0.5}$ |
| wproj | 符号 | $\mathbf{w}$：weights，各索引头的系数<br>$x$：本层输入<br>$W_w$：weights_proj 的权重，源码里叫 weights_proj.weight<br>$\mu_i$：softmax_scale，索引器自己的，取 index_head_dim（128）的 −1/2 次方，与注意力那次不是同一个量<br>$H_i$：n_heads（index_n_heads），32 |
| wproj | 说明 | $W_w$ [32, 5120]（源码里叫 weights_proj.weight） |
| idot | 公式 | $\mathbf{Q}_{b,t,i,u} = \sum_d q_{b,t,i,d}\, \mathbf{J}_{b,u,d}$ |
| idot | 符号 | $\mathbf{Q}$：index_score，索引打分<br>$q$：索引 query<br>$\mathbf{J}$：index_k，索引键<br>$i$：einsum 里的头维<br>$d$：index_head_dim |
| idot | 说明 | 每个索引头对每个压缩位置记一个点积 |
| red | 公式 | $\mathbf{Q} = \sum_i \mathrm{relu}(\mathbf{Q}_i) \cdot \mathbf{w}_i$ |
| red | 符号 | $\mathbf{Q}$：index_score，索引打分<br>$\mathbf{Q}_i$：上一行按头 $i$ 的点积<br>$i$：head 维<br>$\mathbf{w}$：weights_proj 给出的系数 |
| red | 说明 | 先过 ReLU，再按头用 weights 加权求和 |
| imask | 公式 | $\mathbf{Q} = \mathbf{Q}.\mathrm{masked\_fill}(\mathrm{arange}(T /\! / r) \ge \ell, -\infty)$ |
| imask | 符号 | $\mathbf{Q}$：index_score，索引打分<br>$\ell$：compress_lens，每个 query 能看到的压缩条目数，第 $j$ 个 query 能看到 ⌊($j$+1)/$r$⌋ 个<br>$j$：query 的位置下标（不是同图 red 里的头维 $i$）<br>$r$：compress_ratio |
| imask | 说明 | 把本次还看不到的压缩条目在 $\mathbf{Q}$ 上原地写成 −inf |
| csel | 公式 | $\mathbf{c} = \mathrm{select\_candidate\_blocks}(\mathbf{Q}, \ell, k_b, \beta)$ |
| csel | 符号 | $\mathbf{c}$：candidates，候选块掩码<br>$\mathbf{Q}$：index_score，掩过不可达位置之后的索引打分<br>$\ell$：compress_lens，每个 query 能看到的压缩条目数<br>$k_b$：candidate_topk_blocks，2048<br>$\beta$：candidate_block_size，8 |
| csel | 说明 | 先把打分按块切开、取块内最大，把最新那一块钉成正无穷，再取前 candidate_topk_blocks 块，最后按块展开回逐个位置。只有层 20 是 candidate source（config 的 candidate_source_layer），由它写出这份候选块掩码；其余 index source 层不做这一步 |
| cmf | 公式 | $\mathbf{Q} = \mathbf{Q}.\mathrm{masked\_fill}(\lnot \mathbf{c}, -\infty)$ |
| cmf | 符号 | $\mathbf{Q}$：index_score，索引打分<br>$\mathbf{c}$：candidates，层 20 写出的候选块掩码 |
| cmf | 说明 | 候选块外的位置置成负无穷。只有层 24、28、32、36（uses_candidates）走这一步，它们做完再取 top-512；层 2、8、14、20 不做候选块过滤，直接在掩过不可达位置的打分上取 |
| topk | 公式 | $\mathbf{G}_c = \mathrm{sort}(\mathrm{topk}(\mathbf{Q}, k_t))$ |
| topk | 符号 | $\mathbf{G}_c$：idxs，选出的压缩条目下标<br>$\mathbf{Q}$：index_score，索引打分<br>$k_t$：topk，取 min(512, T/ratio)，前提里 T 足够长所以就是 512 |
| topk | 说明 | 取分数最高的 512 个，再按位置顺序排回来 |
| shift | 公式 | $\mathbf{G}_c = \mathrm{where}(\mathbf{G}_c < \ell,\ \mathbf{G}_c + L_w,\ -1)$ |
| shift | 符号 | $\mathbf{G}_c$：idxs，topk 的结果<br>$\ell$：compress_lens，每个 query 能看到的压缩条目数<br>$L_w$：offset，窗口 KV 的长度 |
| shift | 说明 | 不小于可达数的置 −1，其余加上 offset 变成拼起来的 KV 里的下标。offset 由注意力传入，是这次窗口 KV 的长度 $T$ |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 索引打分 | $\mathbf{Q} = \sum_i \mathrm{relu}(q_i \cdot \mathbf{J}) \cdot \mathbf{w}_i$<br>给每个压缩条目算一个索引分 |
| 公式 | 打分权重 | $\mathbf{w} = (x\, W_w^\top) \cdot \mu_i \cdot H_i^{-0.5}$<br>投影出各索引头的权重 |
| 公式 | 可达数 | $\ell_j = \lfloor (j+1)/r \rfloor$<br>第 j 个 query 能看的压缩条目数 |
| 公式 | 输出 | $\mathbf{G}_c = \mathrm{sort}(\mathrm{topk}(\mathbf{Q}, k_t))$<br>选出分最高的 topk 并排回位置序 |
| 公式 | 下标偏移 | $\mathbf{G}_c = \mathrm{where}(\mathbf{G}_c < \ell, \mathbf{G}_c + L_w, -1)$<br>加窗口偏移变成拼起来的 KV 下标 |
| 配置 | index_n_heads | 32 |
| 配置 | index_head_dim | 128 |
| 配置 | index_topk | 512 |

## MoE.forward（model.py:889-904）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| mx | 张量 | x | bf16 [B, T, 5120] | 前馈子层的输入，已经过 ffn_norm | model.py:890 |
| mxf | 张量 | x | bf16 [B×T, 5120] | 拉平后的隐状态，源码里 x.view(-1, self.dim) 的结果；MoE 按 token 处理 | model.py:891 |
| im | 张量 | image_mask | bool [B, T] | MoE.forward 的参数，由 Block 传进来；图像 span 的位置为 True，与 Transformer 图的 im 是同一个量 | model.py:889 |
| gate_op | 算子 | self.gate(x, None if image_mask is None else image_mask.flatten()) | [B×T, 5120] 与 bool [B×T] → 权重 [B×T, 6] 与专家下标 [B×T, 6] |  | model.py:892 |
| gw | 张量 | weights | fp32 [B×T, 6] | | model.py:892 |
| gi | 张量 | indices | int64 [B×T, 6] | | model.py:892 |
| bc | 算子 | torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist() | int64 [B×T, 6] → list，长度 384 |  | model.py:894 |
| counts | 张量 | counts | list，长度 384 | | model.py:894 |
| y | 张量 | y | fp32 [B×T, 5120] | 累加器，源码里 torch.zeros_like(x, dtype=torch.float32)；先记下 x 的形状与 device | model.py:893 |
| mpick | 算子 | torch.where(indices == i) | int64 [B×T, 6] → int64 [n] × 2 |  | model.py:899 |
| pidx | 张量 | idx, top | int64 [n] 与 int64 [n] | n 是分到第 i 个专家的 token 数 | model.py:899 |
| mexp | 算子 | expert(x[idx], weights[idx, top, None]) | bf16 [n, 5120] → bf16 [n, 5120] |  | model.py:900 |
| eout | 张量 | expert(x[idx], weights[idx, top, None]) | bf16 [n, 5120] | | model.py:900 |
| acc | 算子 | y[idx] += expert(x[idx], weights[idx, top, None]) | fp32 [B×T, 5120] 与 bf16 [n, 5120] → fp32 [B×T, 5120] |  | model.py:900 |
| y2 | 张量 | y | fp32 [B×T, 5120] | | model.py:900 |
| shared | 算子 | self.shared_experts(x) | bf16 [B×T, 5120] → bf16 [B×T, 5120] |  | model.py:903 |
| sv | 张量 | self.shared_experts(x) | bf16 [B×T, 5120] | | model.py:903 |
| madd | 算子 | y += self.shared_experts(x) | fp32 [B×T, 5120] 与 bf16 [B×T, 5120] → fp32 [B×T, 5120] |  | model.py:903 |
| mout | 张量 | y | bf16 [B, T, 5120] | 源码里 y.type_as(x).view(shape) 的结果：转回输入的 dtype 与原形状 | model.py:904 |

### 连线

- mx → mxf：view(-1, dim)
- mxf → gate_op
- im → gate_op：image_mask 压平成按 token 的一维
- gate_op → gw
- gate_op → gi
- mxf → y：形状与 device
- gi → bc
- bc → counts
- gi → mpick
- counts → mpick：counts[i] 为 0 则跳过
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
- madd → mout
- mx → mout：dtype 与 shape
- gate_op 点开 → Gate.forward
- mexp 点开 → Expert.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| gate_op | 公式 | $\mathbf{w}, \mathbf{G} = \mathrm{gate}(x, \mathbf{M})$ |
| gate_op | 符号 | $\mathbf{w}$：weights，路由权重<br>$\mathbf{G}$：indices，专家下标<br>$x$：拉平后的隐状态<br>$\mathbf{M}$：image_mask 压平成 [B×T] 之后的那一维 |
| gate_op | 说明 | Gate：打分、加偏置选专家、在 top-6 内归一化、乘 route_scale；源码里实参写作 `None if image_mask is None else image_mask.flatten()`，即把压平后的掩码传进去（压平本身在线上注明）。图像 span 的位置换一份偏置，见 Gate 图 |
| bc | 公式 | $\mathbf{n} = \mathrm{bincount}(\mathrm{flatten}(\mathbf{G}), \mathrm{minlength}=N_e).\mathrm{tolist}()$ |
| bc | 符号 | $\mathbf{n}$：counts，每个专家分到的 token 数<br>$\mathbf{G}$：indices，专家下标<br>$N_e$：n_routed_experts，384 |
| bc | 说明 | $\mathbf{n}$ 是 Python list，长度 384。$\mathbf{n}$[$i$] 为 0 的专家后面直接跳过 |
| mpick | 公式 | $r, \tau = \mathrm{where}(\mathbf{G} = i)$ |
| mpick | 符号 | $r$：idx，分到第 i 个专家的 token 行号<br>$\tau$：top，它在 top-6 里的位置<br>$\mathbf{G}$：每个 token 的 top-6 专家号<br>$i$：当前专家 |
| mpick | 说明 | 挑出分给第 $i$ 个专家的 token，384 个专家各跑一遍；$\mathbf{n}$[$i$] 为 0 的专家整个跳过 |
| mexp | 公式 | $\mathbf{B} = \mathrm{expert}(x[r], \mathbf{w}[r, \tau, \mathrm{None}])$ |
| mexp | 符号 | $\mathbf{B}$：eout，专家输出<br>$x$：拉平后的隐状态<br>$\mathbf{w}$：weights，路由权重<br>$r$、$\tau$：分到当前专家的 token 与 top 位置 |
| mexp | 说明 | 这一步只处理分给当前专家 $i$ 的 token。一个 token 可以出现在多个专家的 top-6 里，路由权重按它在 top-6 里的位置取 |
| acc | 公式 | $\mathbf{Y}[r] = \mathbf{Y}[r] + \mathbf{B}$ |
| acc | 符号 | $\mathbf{Y}$：y，累加器<br>$\mathbf{B}$：eout，专家输出<br>$r$：idx，分到这个专家的 token |
| acc | 说明 | 一个 token 落在 top-6 的哪几个专家上就加几次 |
| shared | 公式 | $\mathbf{V}_s = \mathrm{shared\_experts}(x)$ |
| shared | 符号 | $\mathbf{V}_s$：sv，共享专家输出<br>$x$：拉平后的隐状态 |
| shared | 说明 | shared_experts 是一个 Expert，每个 token 都过。不传 weights，所以不走 Expert.forward 里的 emw。三个权重按主干精度 fp8 存储 |
| madd | 公式 | $\mathbf{Y} = \mathbf{Y} + \mathbf{V}_s$ |
| madd | 符号 | $\mathbf{Y}$：y，路由专家已经累加的结果<br>$\mathbf{V}_s$：sv，共享专家的输出 |
| madd | 说明 | 路由专家的和再加上共享专家的输出 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 专家累加 | $\mathbf{Y}[r] = \mathbf{Y}[r] + \mathbf{B}$<br>把路由专家的输出加回各 token |
| 公式 | 共享专家 | $\mathbf{Y} = \mathbf{Y} + \mathbf{V}_s$<br>把共享专家的输出加到累加器 |
| 配置 | n_routed_experts | 384 |
| 配置 | n_shared_experts | 1 |
| 配置 | n_activated_experts | 6 |

## Gate.forward（model.py:809-827）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| gx | 张量 | x | bf16 [B×T, 5120] | 拉平后的隐状态 | model.py:811 |
| gwt | 张量 | self.weight | bf16 [384, 5120] | 专家打分矩阵，gate.weight；构造时未指定 dtype，取参考实现的默认 bf16 | model.py:805 |
| lg | 算子 | linear(x.float(), self.weight.float()) / self.gate_temp | bf16 [B×T, 5120] → fp32 [B×T, 384] |  | model.py:811 |
| s0 | 张量 | scores | fp32 [B×T, 384] | | model.py:811 |
| act | 算子 | F.softplus(scores).sqrt() | fp32 [B×T, 384] → fp32 [B×T, 384] |  | model.py:817 |
| s | 张量 | scores | fp32 [B×T, 384] | | model.py:817 |
| bv | 张量 | self.bias | fp32 [384] | 可学习参数，只在挑专家时加到分数上，不改权重的值；文本位置用这一份 | model.py:818 |
| bvl | 张量 | self.bias_vl | fp32 [384] | 另一份偏置，图像 span 的位置换成它。Transformer 在 vision_n_layers 大于 0（vision_enabled 为真）时建 ViT 与 Aligner，同一条件下建这份偏置，否则是 None | model.py:807 |
| mfm | 张量 | image_mask | bool [B×T] | Gate 收到的那个 image_mask，已经压成按 token 的一维；图像 span 的位置为 True | model.py:819 |
| bsel | 算子 | torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias) | fp32 [384] 与 fp32 [384] 与 bool [B×T] → fp32 [B×T, 384] |  | model.py:819-820 |
| bbias | 张量 | bias | fp32 [B×T, 384] | 按 token 选出来的那一份偏置，每个 token 整行取同一份 | model.py:819-820 |
| gsel | 算子 | (scores + bias).topk(self.topk, dim=-1)[1] | fp32 [B×T, 384] → int64 [B×T, 6] |  | model.py:822 |
| gi | 张量 | indices | int64 [B×T, 6] | | model.py:822 |
| gth | 算子 | scores.gather(1, indices) | fp32 [B×T, 384] → fp32 [B×T, 6] |  | model.py:823 |
| w0 | 张量 | weights | fp32 [B×T, 6] | | model.py:823 |
| nrm | 算子 | weights /= weights.sum(dim=-1, keepdim=True) + 1e-20 | fp32 [B×T, 6] → fp32 [B×T, 6] |  | model.py:824-825 |
| w1 | 张量 | weights | fp32 [B×T, 6] | | model.py:825 |
| grs | 算子 | weights *= self.route_scale | fp32 [B×T, 6] → fp32 [B×T, 6] |  | model.py:826 |
| gw | 张量 | weights | fp32 [B×T, 6] | | model.py:826 |
| gret | 张量 | (weights, indices) | fp32 [B×T, 6] × int64 [B×T, 6] | 两个返回值都交给 MoE | model.py:827 |

### 连线

- gx → lg
- gwt → lg
- lg → s0
- s0 → act
- act → s
- s → gsel
- bsel → bbias
- bbias → gsel：选出来的那份偏置
- mfm → bsel：image_mask
- bv → bsel
- bvl → bsel
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
| lg | 公式 | $\mathbf{R} = \mathrm{linear}(x, W_g) / \tau_g$ |
| lg | 符号 | $\mathbf{R}$：scores，专家打分<br>$x$：拉平后的隐状态<br>$W_g$：gate.weight<br>$\tau_g$：gate_temp，1.0 |
| lg | 说明 | $W_g$ 构造时未指定 dtype，取参考实现的默认 bf16（model.py:1296），打分里再上转 fp32（源码写作 x.float()、self.weight.float()）；gate_temp 取默认值 1.0，config.json 里没有这一项 |
| act | 公式 | $\mathbf{R} = \sqrt{\mathrm{softplus}(\mathbf{R})}$ |
| act | 符号 | $\mathbf{R}$：scores，专家打分<br>$\mathrm{softplus}$：$\mathrm{softplus}(\mathbf{R}) = \ln(1+e^{\mathbf{R}})$ |
| act | 说明 | score_func 是 sqrtsoftplus：先 softplus 再开方 |
| bsel | 公式 | $\mathbf{b} = \mathrm{where}(\mathbf{M}.\mathrm{unsqueeze}(-1), \mathbf{b}_{vl}, \mathbf{b}_{txt})$ |
| bsel | 符号 | $\mathbf{b}$：选出来的偏置<br>$\mathbf{M}$：image_mask，由 [B×T] 扩成 [B×T, 1] 再广播到 384<br>$\mathbf{b}_{vl}$：另一份偏置，源码里叫 bias_vl<br>$\mathbf{b}_{txt}$：文本位置的偏置，源码里叫 bias |
| bsel | 说明 | 两份偏置都只在挑专家时用，不改权重的值；每个 token 整行要么全取 $\mathbf{b}_{vl}$、要么全取 $\mathbf{b}_{txt}$ |
| gsel | 公式 | $\mathbf{G} = \mathrm{topk}(\mathbf{R} + \mathbf{b}, k)$ |
| gsel | 符号 | $\mathbf{G}$：indices，选中的专家号<br>$\mathbf{R}$：scores，专家打分<br>$\mathbf{b}$：bsel 选出来、存在 bbias 里的那一份<br>$k$：topk（n_activated_experts） |
| gsel | 说明 | 偏置只参与挑专家，不改权重 |
| gth | 公式 | $\mathbf{w} = \mathrm{gather}(\mathbf{R}, \mathbf{G})$ |
| gth | 符号 | $\mathbf{w}$：weights，路由权重<br>$\mathbf{R}$：scores，专家打分，取得的是没加 b 的那份<br>$\mathbf{G}$：indices，选中的专家号 |
| gth | 说明 | 权重取自没加偏置的 scores |
| nrm | 公式 | $\mathbf{w} = \mathbf{w} / (\mathrm{sum}(\mathbf{w}) + 10^{-20})$ |
| nrm | 符号 | $\mathbf{w}$：weights，路由权重<br>$\mathrm{sum}$：在这 6 个专家上求和 |
| nrm | 说明 | 在 top-6 内除以这 6 个权重的和，分母加 1e-20 |
| grs | 公式 | $\mathbf{w} = \mathbf{w} \cdot \gamma$ |
| grs | 符号 | $\mathbf{w}$：weights，路由权重<br>$\gamma$：route_scale，1.5 |
| grs | 说明 | route_scale 是 1.5 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 打分 | $\mathbf{R} = \mathrm{linear}(x, W_g) / \tau_g$<br>给每个专家算一个原始分 |
| 公式 | 激活函数 | $\mathbf{R} = \sqrt{\mathrm{softplus}(\mathbf{R})}$<br>把原始分过 sqrtsoftplus |
| 公式 | 选专家 | $\mathbf{G} = \mathrm{topk}(\mathbf{R} + \mathbf{b}, k)$<br>按偏置加过分后选 topk 专家 |
| 公式 | 偏置选择 | $\mathbf{b} = \mathrm{where}(\mathbf{M}.\mathrm{unsqueeze}(-1), \mathbf{b}_{vl}, \mathbf{b}_{txt})$<br>按图像掩码选路由偏置 |
| 公式 | 取权重 | $\mathbf{w} = \mathrm{gather}(\mathbf{R}, \mathbf{G})$<br>取选中专家的分当路由权重 |
| 公式 | 归一化 | $\mathbf{w} = \mathbf{w} / (\mathrm{sum}(\mathbf{w}) + 10^{-20})$<br>在 top-6 内归一化权重 |
| 公式 | 缩放 | $\mathbf{w} = \mathbf{w} \cdot \gamma$<br>权重乘路由缩放 |
| 配置 | n_routed_experts | 384 |
| 配置 | n_activated_experts | 6 |
| 配置 | score_func | sqrtsoftplus |
| 配置 | route_scale | 1.5 |
| 配置 | norm_topk_prob | true（默认值） |

## Expert.forward（model.py:841-851）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ex | 张量 | x | bf16 [n, 5120] | n 是分给这个路由专家的 token 数 | model.py:842 |
| wts | 张量 | weights | fp32 [n, 1] | 只有路由专家传这个，共享专家不传 | model.py:841 |
| g1 | 算子 | self.w1(x).float() | bf16 [n, 5120] → fp32 [n, 2304] |  | model.py:843 |
| eact | 张量 | gate | fp32 [n, 2304] | | model.py:843 |
| u1 | 算子 | self.w3(x).float() | bf16 [n, 5120] → fp32 [n, 2304] |  | model.py:844 |
| up | 张量 | up | fp32 [n, 2304] | | model.py:844 |
| cl | 算子 | torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit) | fp32 [n, 2304] → fp32 [n, 2304] |  | model.py:846 |
| up2 | 张量 | up | fp32 [n, 2304] | | model.py:846 |
| cl2 | 算子 | torch.clamp(gate, max=self.swiglu_limit) | fp32 [n, 2304] → fp32 [n, 2304] |  | model.py:847 |
| eact2 | 张量 | gate | fp32 [n, 2304] | | model.py:847 |
| silu | 算子 | F.silu(gate) * up | fp32 [n, 2304] → fp32 [n, 2304] |  | model.py:848 |
| hm | 张量 | F.silu(gate) * up | fp32 [n, 2304] | | model.py:848 |
| emw | 算子 | weights * x | fp32 [n, 2304] → fp32 [n, 2304] |  | model.py:850 |
| hw | 张量 | weights * x | fp32 [n, 2304] | | model.py:850 |
| w2 | 算子 | self.w2(x.to(dtype)) | fp32 [n, 2304] → bf16 [n, 5120] |  | model.py:851 |
| eout | 张量 | self.w2(x.to(dtype)) | bf16 [n, 5120] | | model.py:851 |

### 连线

- ex → g1
- g1 → eact
- ex → u1：激活先按 32 通道一块量化成 fp8（`linear()` 里的 `act_quant`）再进 GEMM
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
- ex → w2：dtype
- w2 → eout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| g1 | 公式 | $g = x\, W_1^\top$ |
| g1 | 符号 | $g$：gate，门分支<br>$x$：输入<br>$W_1$：w1 的权重，源码里叫 w1.weight |
| g1 | 说明 | 路由专家的 $W_1$ 存成 [2304, 2560]（源码里叫 w1.weight） 的 fp4：末维是逻辑宽度 5120 的一半，因为 fp4 每个字节装两个值；每 32 个通道一个 E8M0 scale；激活也按 32 通道一块量化成 fp8 再进 GEMM（`linear()` 里的 `act_quant`）。结果转成 fp32，夹取在后面的 cl2 |
| u1 | 公式 | $u = x\, W_3^\top$ |
| u1 | 符号 | $u$：up，上分支<br>$x$：输入<br>$W_3$：w3 的权重，源码里叫 w3.weight |
| u1 | 说明 | $W_3$ 与 $W_1$ 同一种存法（源码里叫 w3.weight、w1.weight），激活同样先量化成 fp8；结果转成 fp32 |
| cl | 公式 | $u = \mathrm{clamp}(u, -\lambda, \lambda)$ |
| cl | 符号 | $u$：up，上分支<br>$\lambda$：swiglu_limit，10 |
| cl | 说明 | up 两头都夹；swiglu_limit 是 10.0<br>夹是为了让后面的 fp8、fp4 激活不越界 |
| cl2 | 公式 | $g = \mathrm{clamp}(g, \max = \lambda)$ |
| cl2 | 符号 | $g$：gate，门分支<br>$\lambda$：swiglu_limit，10 |
| cl2 | 说明 | $g$ 只夹上界 |
| silu | 公式 | $x = \mathrm{silu}(g) \odot u$ |
| silu | 符号 | $x$：激活后的隐状态<br>$g$、$u$：夹过界的两路 |
| silu | 说明 | SiLU 的结果逐元素乘 up |
| emw | 公式 | $x = w \odot x$ |
| emw | 符号 | $x$：SiLU 之后的隐状态<br>$w$：weights，路由权重 |
| emw | 说明 | 只在传入 $w$ 时执行。这张图是路由专家，$w$ 来自 top-6 里该专家的权重 |
| w2 | 公式 | $x = x.\mathrm{to}(dtype)\, W_2^\top$ |
| w2 | 符号 | $x$：乘过路由权重的隐状态<br>$dtype$：进入 Expert 时 x 的 dtype（bf16）<br>$W_2$：w2 的权重，源码里叫 w2.weight |
| w2 | 说明 | 路由专家的 $W_2$ 存成 [5120, 1152]（源码里叫 w2.weight） 的 fp4（末维 1152 = 2304 / 2，同样是一格装两个值），激活先量化成 fp8；共享专家是 [5120, 2304] 的 fp8，激活同样量化成 fp8 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 门投影 | $g = x\, W_1^\top$<br>把输入投成门分支 |
| 公式 | 上分支投影 | $u = x\, W_3^\top$<br>把输入投成上分支 |
| 公式 | 激活相乘 | $x = \mathrm{silu}(g) \odot u$<br>门分支过 SiLU 后乘上分支 |
| 公式 | 路由加权 | $x = w \odot x$<br>路由专家乘上本专家的路由权重 |
| 公式 | 输出投影 | $x = x.\mathrm{to}(dtype)\, W_2^\top$<br>转回 dtype 后投回主干维度 |
| 公式 | 上分支夹取 | $u = \mathrm{clamp}(u, -\lambda, \lambda)$<br>上分支两头夹到 swiglu_limit |
| 公式 | 门分支夹取 | $g = \mathrm{clamp}(g, \max=\lambda)$<br>门分支只夹上界 |
| 配置 | swiglu_limit | 10.0 |

## Engram.forward（model.py:350-365）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| nx | 张量 | x | bf16 [B, T, 4, 5120] | 本层的残差流，4 份 | model.py:350 |
| hash_ids | 张量 | hash_ids | int64 [B, T, 24] | 本层的 n-gram 哈希行号，24 = (4 − 1) × 8 | model.py:350 |
| nemb | 算子 | self.embed(hash_ids).flatten(-2) | [B, T, 24] → [B, T, 6144] |  | model.py:353，296-325 |
| ev | 张量 | self.embed(hash_ids).flatten(-2) | bf16 [B, T, 6144] | | model.py:353 |
| nwkv | 算子 | self.wkv(ev) | [B, T, 6144] → [B, T, 25600] |  | model.py:353 |
| nkv | 张量 | kv | bf16 [B, T, 25600] | | model.py:353 |
| nsplit | 算子 | kv.split([self.hc_mult * self.dim, self.dim], dim=-1) | [B, T, 25600] → [B, T, 20480] 与 [B, T, 5120] |  | model.py:354 |
| key0 | 张量 | key | bf16 [B, T, 20480] | 20480 = 4 × 5120，每份残差流一个键 | model.py:354 |
| value | 张量 | value | bf16 [B, T, 5120] | 4 份残差流共用同一个值 | model.py:354 |
| ekw | 张量 | self.k_weight | bf16 [4, 5120] | 键侧的缩放参数，可学习；构造时未指定 dtype，取参考实现的默认 bf16 | model.py:348 |
| ewm | 算子 | self.q_weight.float() * self.k_weight.float() | [4, 5120] → fp32 [4, 5120] |  | model.py:356 |
| ew | 张量 | weight | fp32 [4, 5120] | | model.py:356 |
| nkey | 算子 | key.float().unflatten(-1, (self.hc_mult, self.dim)) | [B, T, 20480] → fp32 [B, T, 4, 5120] |  | model.py:355 |
| key | 张量 | key | fp32 [B, T, 4, 5120] | | model.py:355 |
| nh | 张量 | h | fp32 [B, T, 4, 5120] | 本层残差流上转 fp32 的结果，源码里 x.float() | model.py:357 |
| rstd | 算子 | torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps) | fp32 [B, T, 4, 5120] 与 fp32 [B, T, 4, 5120] → fp32 [B, T, 4] |  | model.py:359 |
| nrs | 张量 | rstd | fp32 [B, T, 4] | | model.py:359 |
| ndot | 算子 | (h * weight * key).sum(-1) * rstd * self.dim**-0.5 | fp32 [B, T, 4, 5120] → fp32 [B, T, 4] |  | model.py:360 |
| d | 张量 | dot | fp32 [B, T, 4] | | model.py:360 |
| ngate | 算子 | torch.sigmoid(<br>torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)) | fp32 [B, T, 4] → fp32 [B, T, 4] |  | model.py:362 |
| g | 张量 | gate | fp32 [B, T, 4] | | model.py:362 |
| emv | 张量 | token_mask | bool [B, T] | Engram.forward 的参数，图像 span 的位置为 False；与 Transformer 图的 emv 是同一个量 | model.py:350 |
| gmask | 算子 | gate.masked_fill(~token_mask.unsqueeze(-1), 0) | fp32 [B, T, 4] 与 bool [B, T] → fp32 [B, T, 4] |  | model.py:363-364 |
| gm | 张量 | gate | fp32 [B, T, 4] | 图像 span 的位置置 0，这些位置不加 Engram 的写入 | model.py:364 |
| nadd | 算子 | h + gate.unsqueeze(-1) * v | fp32 [B, T, 4, 5120] 与 fp32 [B, T, 4] 与 fp32 [B, T, 1, 5120] → fp32 [B, T, 4, 5120] |  | model.py:365 |
| sum | 张量 | h + gate.unsqueeze(-1) * value.float().unsqueeze(-2) | fp32 [B, T, 4, 5120] | | model.py:365 |
| ncast | 算子 | .to(x.dtype) | fp32 [B, T, 4, 5120] → bf16 [B, T, 4, 5120] |  | model.py:365 |
| nout | 张量 | (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype) | bf16 [B, T, 4, 5120] | | model.py:365 |

### 连线

- nx → nh：x.float() 上转 fp32
- hash_ids → nemb
- nemb → ev
- nemb 点开 → ParallelEngramEmbedding.forward
- ev → nwkv
- nwkv → nkv
- nkv → nsplit
- nsplit → key0
- nsplit → value
- key0 → nkey
- nkey → key
- ekw → ewm
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
- g → gmask
- emv → gmask
- gmask → gm
- value → nadd：value.float().unsqueeze(-2)，转 fp32 并在份这一维补成 1
- gm → nadd
- nh → nadd
- nadd → sum
- nx → ncast：dtype
- sum → ncast
- ncast → nout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| nemb | 公式 | $\mathbf{e} = \mathrm{embed}(\mathbf{H}).\mathrm{flatten}(-2)$ |
| nemb | 符号 | $\mathbf{e}$：ev，查表并摊平后的向量<br>$\mathbf{H}$：hash_ids，哈希行号<br>$\mathrm{embed}$：ParallelEngramEmbedding，源码里叫 self.embed |
| nemb | 说明 | ParallelEngramEmbedding：表按行分片，查出 fp8 的行再用 scale 反量化成 bf16，见点开的那张图；外层 flatten 把 24 与 256 合成 6144<br>24 × 256 = 6144；层 1 的表有 384006168 行，层 14 有 384016682 行 |
| nwkv | 公式 | $\mathbf{z} = \mathbf{e}\, W_{kv}^\top$ |
| nwkv | 符号 | $\mathbf{z}$：kv，投影输出<br>$\mathbf{e}$：查表并 flatten 后的向量<br>$W_{kv}$：wkv 的权重，源码里叫 wkv.weight |
| nwkv | 说明 | $W_{kv}$ [25600, 6144]（源码里叫 wkv.weight）；输出切成 key 与 value 两段 |
| nsplit | 公式 | $\mathbf{K}_e, \mathbf{R}_e = \mathrm{split}(\mathbf{z}, [C \cdot d, d])$ |
| nsplit | 符号 | $\mathbf{K}_e$、$\mathbf{R}_e$：key、value，切出的两段<br>$\mathbf{z}$：W_kv 的输出<br>$C$：hc_mult（4）<br>$d$：dim，5120 |
| nkey | 公式 | $\mathbf{K}_e = \mathbf{K}_e.\mathrm{float}().\mathrm{unflatten}(-1, (C, d))$ |
| nkey | 符号 | $\mathbf{K}_e$：key，切出来的前 20480 维<br>$C$：hc_mult（4）<br>$d$：dim，5120 |
| nkey | 说明 | key 转成 fp32，再拆成 hc_mult 份 |
| ewm | 公式 | $\mathbf{W} = w_q \odot w_k$ |
| ewm | 符号 | $\mathbf{W}$：weight，键侧的缩放<br>$w_q$、$w_k$：两个参数，源码里叫 q_weight、k_weight |
| ewm | 说明 | $w_q$ 与 $w_k$ 都是 [4, 5120] 的参数（源文件里 q_weight 在 model.py:347、k_weight 在 model.py:348），构造时未指定 dtype，取参考实现的默认 bf16；源码里在 fp32 里相乘（self.q_weight.float() * self.k_weight.float()），本图把 q_weight 的取用也并进这一步 |
| rstd | 公式 | $\nu = \mathrm{rsqrt}(\mathrm{mean}(x^2) + \epsilon) \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathbf{K}_e^2) + \epsilon)$ |
| rstd | 符号 | $\nu$：rstd，归一化因子<br>$x$：上转 fp32 的残差（同图的 nh）<br>$\mathbf{K}_e$：key，查表得到的键<br>$\epsilon$：Engram 上的 norm_eps<br>$\mathrm{mean}$：在最后一维 |
| rstd | 说明 | 两边各按最后一维求均方，逐 token、逐份归一化 |
| ndot | 公式 | $\delta = \mathrm{sum}(x \odot \mathbf{W} \odot \mathbf{K}_e, -1) \cdot \nu \cdot d^{-0.5}$ |
| ndot | 符号 | $\delta$：dot，归一化点积<br>$x$、$\mathbf{W}$、$\mathbf{K}_e$、$\nu$：送进来的张量<br>$\mathbf{W}$：w_q 与 w_k 的逐元素积<br>$d$：dim，5120 |
| ndot | 说明 | 归一化点积，再乘 dim 的 −1/2 次方 |
| ngate | 公式 | $g = \mathrm{sigmoid}(\mathrm{copysign}(\sqrt{\max(\lvert \delta \rvert, j)}, \delta))$ |
| ngate | 符号 | $g$：gate，写入门控<br>$\delta$：dot，归一化点积<br>$j$：1e-6，源码里叫 clamp_value |
| ngate | 说明 | 先带符号开方再进 sigmoid，与训练 kernel 一致 |
| gmask | 公式 | $g = g.\mathrm{masked\_fill}(\lnot \mathbf{E}.\mathrm{unsqueeze}(-1), 0)$ |
| gmask | 符号 | $g$：gate，写入门控<br>$\mathbf{E}$：token_mask，为 False 的位置把门控置 0 |
| gmask | 说明 | 图像 span 的位置不做门控写入，残差流原样穿过。$\mathbf{E}$ 就是这一层的 engram_mask |
| nadd | 公式 | $x = x + \mathrm{unsqueeze}(g, -1) \odot \mathbf{R}_e$ |
| nadd | 符号 | $x$：上转 fp32 的残差流<br>$g$：gmask 之后的门控、逐份<br>$\mathbf{R}_e$：value 转成 fp32 并在份这一维扩成 1 |
| nadd | 说明 | 门控乘值再加回残差流；公式里的 $\mathbf{R}_e$ 就是源码里 value.float().unsqueeze(-2) 的结果 |
| ncast | 公式 | $x = x.\mathrm{to}(dtype)$ |
| ncast | 符号 | $x$：写回后的残差<br>$dtype$：进入 Engram 时残差的 dtype |
| ncast | 说明 | 转回输入时的 dtype |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 归一化因子 | $\nu = \mathrm{rsqrt}(\mathrm{mean}(x^2)+\epsilon)\cdot\mathrm{rsqrt}(\mathrm{mean}(\mathbf{K}_e^2)+\epsilon)$<br>算残差与键各自的均方根倒数 |
| 公式 | 归一化点积 | $\delta = \mathrm{sum}(x \odot \mathbf{W} \odot \mathbf{K}_e, -1) \cdot \nu \cdot d^{-0.5}$<br>算残差与查表键的归一化点积 |
| 公式 | 门控 | $g = \mathrm{sigmoid}(\mathrm{copysign}(\sqrt{\max(\lvert \delta \rvert, j)}, \delta))$<br>把点积变成写入门 |
| 公式 | 写入 | $x = x + \mathrm{unsqueeze}(g, -1) \odot \mathbf{R}_e$<br>按门控把 value 写回残差流 |
| 配置 | hc_mult | 4 |
| 配置 | engram_max_ngram_size | 4 |
| 配置 | engram_n_heads | 8 |
| 配置 | engram_head_dim | 256 |
| 配置 | engram_layer_ids | [1, 14] |
| 配置 | engram_num_embeddings | [384006168, 384016682] |

## ParallelEngramEmbedding.forward（model.py:312-325）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hash_ids | 张量 | indices | int64 [B, T, 24] | 参数 indices，就是 Engram 图里的 hash_ids | model.py:312 |
| peu | 张量 | self.weight | fp8_e4m3 [384016682, 256] | n-gram 哈希表，fp8 存，按 rank 沿行切分，单卡即整张 | model.py:309 |
| pes | 张量 | self.scale | e8m0 [384016682, 8] | 每 32 个通道（256 那一维）一组一个 scale，每行 8 个（scale_dtype 是 float8_e8m0fnu），8 = 256 / 32 | model.py:310 |
| peuq | 算子 | F.embedding(local_indices, self.weight) | int64 [B, T, 24] → fp8 [B, T, 24, 256] |  | model.py:317 |
| pev | 张量 | values | fp8 [B, T, 24, 256] | | model.py:317 |
| pesq | 算子 | F.embedding(local_indices, self.scale) | int64 [B, T, 24] → e8m0 [B, T, 24, 8] |  | model.py:318 |
| pesc | 张量 | scales | e8m0 [B, T, 24, 8] | | model.py:318 |
| pedq | 算子 | (values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16) | fp8 [B, T, 24, 256] 与 e8m0 [B, T, 24, 8] → bf16 [B, T, 24, 256] |  | model.py:319-320 |
| ped | 张量 | values | bf16 [B, T, 24, 256] | 反量化后的行，每 32 个通道共用一个 scale | model.py:320 |
| peout | 张量 | values | bf16 [B, T, 24, 256] | 返回值；Engram 图里接着做 flatten(-2) | model.py:325 |

### 连线

- hash_ids → peuq
- hash_ids → pesq
- peu → peuq
- pes → pesq
- peuq → pev
- pesq → pesc
- pev → pedq
- pesc → pedq
- pedq → ped
- ped → peout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| peuq | 公式 | $\mathbf{R}_v = \mathrm{embedding}(I', W)$ |
| peuq | 符号 | $\mathbf{R}_v$：values，查出的 fp8 行<br>$I'$：local_indices，hash_ids 减去本卡起始行<br>$W$：fp8 表，源码里叫 weight |
| peuq | 说明 | $W$ [384016682, 256]（源码里叫 weight），按 rank 沿行切分，单卡即整张；层 1 的表有 384006168 行，层 14 有 384016682 行。这一行同时按行号取出缩放表那一份（源码把两次 lookup 写在一行），取出的 fp8 行在 pedq 里按 scale 反量化 |
| pesq | 公式 | $\mathbf{s} = \mathrm{embedding}(I', \mathbf{T})$ |
| pesq | 符号 | $\mathbf{s}$：scales，每 32 个通道一组的缩放<br>$I'$：local_indices，hash_ids 减去本卡起始行<br>$\mathbf{T}$：缩放表，源码里叫 scale |
| pesq | 说明 | $\mathbf{T}$ [384016682, 8]（源码里叫 scale） |
| pedq | 公式 | $\mathbf{R}_v = \mathrm{flatten}(\mathbf{R}_v.\mathrm{float}().\mathrm{unflatten}(-1, (-1, 32)) \odot \mathbf{s}.\mathrm{float}().\mathrm{unsqueeze}(-1)).\mathrm{to}(bf16)$ |
| pedq | 符号 | $\mathbf{R}_v$：values，反量化后的行<br>$\mathbf{s}$：scales，查出的缩放<br>$32$：fp8_block_size，256 / 32 = 8 |
| pedq | 说明 | 查出的 fp8 行按每 32 个通道一个 scale 反量化成 bf16；单卡下源码里的 mask 与 all_reduce 不执行 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 配置 | engram_head_dim | 256 |
| 配置 | engram_num_embeddings | [384006168, 384016682] |

## NgramHashState.forward（engram.py:160-184）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ids | 张量 | input_ids | int64 [B, T] | | engram.py:160 |
| tmap | 张量 | self.token_map | int64，长度 129280 | 压缩词表，把每个原始 token id 映到压缩词表的行号；长度等于 tokenizer 词表大小（vocab_size 129280），码位的取值上界才是 engram_compressed_vocab_size（99092）；缓存 | engram.py:45，154 |
| cmap | 算子 | self.token_map[input_ids] | int64 [B, T] → int64 [B, T] |  | engram.py:164 |
| cids | 张量 | compressed | int64 [B, T] | | engram.py:164 |
| emv | 张量 | token_mask | bool [B, T] | 调用的参数，图像 span 的位置为 False；与 Transformer 图的 emv 是同一个量 | engram.py:160 |
| cgm | 算子 | torch.where(token_mask, compressed, self.DEAD) | int64 [B, T] 与 bool [B, T] → int64 [B, T] |  | engram.py:166 |
| cdd | 张量 | compressed | int64 [B, T] | 重写后的压缩 id，图像 span 的位置记成 DEAD（−1） | engram.py:166 |
| wr | 算子 | self.cache[:batch, start_pos : start_pos + seqlen] = compressed | int64 [B, T] → [max_batch_size, max_seq_len] |  | engram.py:167 |
| cache | 缓存 | cache | int64 [max_batch_size, max_seq_len] | 按 `max_batch_size` 分配（默认 4，是缓存容量，不是本次批大小 $B$），序列维取 `max_seq_len`（默认 4096）。存的是压缩后的 id，不是原始 token id；decode 时按 start_pos 往后接 | engram.py:155-157 |
| p | 张量 | positions | int64 [B, T] | 源码里 torch.arange(start_pos, start_pos + seqlen).expand(batch, seqlen) 的结果，start_pos 到 start_pos + seqlen 的位置、扩成 batch 行 | engram.py:169 |
| gath | 算子 | self.cache[:batch].gather(1, (positions - shift).clamp_min(0)) | [max_batch_size, max_seq_len] → int64 [B, T] |  | engram.py:172 |
| src | 张量 | source | int64 [B, T] | | engram.py:172 |
| blocked | 张量 | blocked | bool [B, T] | 标记哪些位置已经不能往前看，源码里 torch.zeros_like(positions, dtype=torch.bool) 的初值 | engram.py:170 |
| upd | 算子 | blocked \| (positions < shift) \| (source == self.DEAD) | bool [B, T] → bool [B, T] |  | engram.py:173 |
| updt | 张量 | blocked | bool [B, T] | | engram.py:173 |
| rep | 算子 | torch.where(blocked, self.pad_id, source) | int64 [B, T] → int64 [B, T] |  | engram.py:174 |
| tok | 张量 | torch.where(blocked, self.pad_id, source) | int64 [B, T] | 每个回看步一份 | engram.py:174 |
| stk | 算子 | torch.stack(tokens, dim=-1) | int64 [B, T] × 4 → int64 [B, T, 4] |  | engram.py:175 |
| tk | 张量 | tokens | int64 [B, T, 4] | 第 k 列对应 shift=k，k=0 是当前位置 | engram.py:175 |
| mul | 张量 | self.multipliers | int64 [2, 4] | 每个（Engram 层，回看步）一个奇数乘数 | engram.py:153 |
| prod | 算子 | tokens.unsqueeze(2) * self.multipliers | [B, T, 4] 与 [2, 4] → [B, T, 2, 4] |  | engram.py:179 |
| pr | 张量 | products | int64 [B, T, 2, 4] | | engram.py:179 |
| r0 | 算子 | products[..., 0] | [B, T, 2, 4] → [B, T, 2] |  | engram.py:180 |
| rinit | 张量 | rolling | int64 [B, T, 2] | 异或循环的初值 | engram.py:180 |
| roll | 算子 | torch.bitwise_xor(rolling, products[..., i]) | int64 [B, T, 2] → int64 [B, T, 2] |  | engram.py:181-182 |
| rl | 张量 | rolling | int64 [B, T, 2] | | engram.py:182 |
| prm | 张量 | self.primes | int64 [2, 3, 8] | 每个（层，桶组，头）的质数模数 | engram.py:151 |
| mod | 算子 | rolling.unsqueeze(-1) % self.primes[:, i - 1] | int64 [B, T, 2] → int64 [B, T, 2, 8] |  | engram.py:183 |
| hpart | 张量 | rolling.unsqueeze(-1) % self.primes[:, i - 1] | int64 [B, T, 2, 8] | 3 个桶组各一份 | engram.py:183 |
| hcat | 算子 | torch.cat(hashes, dim=-1) | [B, T, 2, 8] × 3 → int64 [B, T, 2, 24] |  | engram.py:184 |
| hs | 张量 | torch.cat(hashes, dim=-1) | int64 [B, T, 2, 24] | 24 = (4 − 1) × 8 | engram.py:184 |
| ofs | 张量 | self.offsets | int64 [2, 3, 8] | 每个桶在表里的起始行 | engram.py:152 |
| off | 算子 | torch.cat(hashes, dim=-1) + self.offsets | int64 [B, T, 2, 24] → int64 [B, T, 2, 24] |  | engram.py:184 |
| hash | 张量 | torch.cat(hashes, dim=-1) + self.offsets | int64 [B, T, 2, 24] | | engram.py:184 |

### 连线

- ids → cmap
- tmap → cmap
- cmap → cids
- cids → cgm
- emv → cgm
- cgm → cdd
- cdd → wr
- wr → cache
- ids → p：形状与 device
- cache → gath
- p → gath
- gath → src
- src → upd
- p → upd
- blocked → upd
- upd → updt
- updt → rep
- src → rep
- rep → tok
- tok → stk
- stk → tk
- tk → prod
- mul → prod
- prod → pr
- pr → r0
- r0 → rinit
- rinit → roll
- pr → roll
- roll → rl
- rl → mod
- prm → mod
- mod → hpart
- hpart → hcat
- hcat → hs
- hs → off
- ofs → off
- off → hash

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| cmap | 公式 | $\mathbf{C} = T_m[I]$ |
| cmap | 符号 | $\mathbf{C}$：compressed，压缩词表 id<br>$I$：input_ids，词 id<br>$T_m$：token_map，压缩词表的行号 |
| cmap | 说明 | $T_m$ 把每个 token id 映到压缩词表的 id，大小写与空白差异都归一到一个 id<br>$T_m$ 长度 129280，等于 tokenizer 词表大小（也是 vocab_size）；压缩后的 id 取值范围上界是 99092，即 config 的 engram_compressed_vocab_size，所有哈希乘数都由它推出 |
| cgm | 公式 | $\mathbf{C} = \mathrm{where}(\mathbf{M}, \mathbf{C}, \mathrm{DEAD})$ |
| cgm | 符号 | $\mathbf{C}$：compressed，重写后的压缩 id<br>$\mathbf{M}$：token_mask，为 False 的位置（图像 span）盖成 DEAD<br>$\mathrm{DEAD}$：−1 |
| cgm | 说明 | 图像 span 的位置不参与任何 n-gram，后面的回看遇到它就断掉 |
| wr | 公式 | $\mathbf{X}_{:B,\ 0:T} = \mathbf{C}$ |
| wr | 符号 | $\mathbf{X}$：cache，n-gram 缓存<br>$\mathbf{C}$：compressed，压成 DEAD 之后的 id<br>$s$：start_pos，0 |
| wr | 说明 | $s$ 为 0，从第 0 列起整块写；图像 span 的 DEAD 也写进 cache，供后面的回看读到 |
| gath | 公式 | $\mathbf{u} = \mathrm{gather}(\mathbf{X}_{:B}, (\mathbf{p} - \delta).\mathrm{clamp}(min=0))$ |
| gath | 符号 | $\mathbf{u}$：source，回看取到的 id<br>$\mathbf{X}$：cache，n-gram 缓存<br>$\mathbf{p}$：positions，位置下标<br>$\delta$：shift，往前看的步数 |
| gath | 说明 | 往前看 shift 个位置，越界处夹到 0 |
| upd | 公式 | $\mathbf{d} = \mathbf{d} \lor (\mathbf{p} < \delta) \lor (\mathbf{u} = \mathrm{DEAD})$ |
| upd | 符号 | $\mathbf{d}$：blocked，更新后的断点标记<br>$\mathbf{p}$：positions，位置下标<br>$\delta$：shift，往前看的步数<br>$\mathbf{u}$：source，回看取到的 id<br>$\mathrm{DEAD}$：−1 |
| upd | 说明 | 位置在序列开头之前、或往前看到 DEAD，都算断掉，之后的回看步也不再往前。图像 span 的位置在 cache 里就是 DEAD，回看跨过它们就断。DEAD 只影响后面的回看，图像位置自己不做 Engram 写入是另外一步，见 Engram 图的 gmask |
| rep | 公式 | $\mathbf{q} = \mathrm{where}(\mathbf{d}, \mathrm{pad\_id}, \mathbf{u})$ |
| rep | 符号 | $\mathbf{q}$：tokens，每个回看步一份<br>$\mathbf{d}$：blocked，断点标记<br>$\mathrm{pad\_id}$：断掉处填的压缩 id<br>$\mathbf{u}$：source，回看取到的 id |
| rep | 说明 | 断掉的位置填 pad_id，即 engram_pad_id（2）经 token_map 映出的那个压缩 id，与训练一致 |
| prod | 公式 | $\mathbf{o} = \mathbf{q}.\mathrm{unsqueeze}(2) \odot \mathbf{m}$ |
| prod | 符号 | $\mathbf{o}$：products，乘上乘数后的值<br>$\mathbf{q}$：tokens，各回看步的压缩 id<br>$\mathbf{m}$：multipliers，每层每个回看步的奇数乘数，源码里叫 self.multipliers |
| prod | 说明 | unsqueeze 把层这一维扩出来；multipliers 每个（Engram 层，回看步）一个奇数乘数，各层用自己的随机种子生成 |
| r0 | 公式 | $\mathbf{r} = \mathbf{o}_{...,0}$ |
| r0 | 符号 | $\mathbf{r}$：rolling，异或循环的初值<br>$\mathbf{o}$：products，tokens 乘 multipliers 的结果 |
| r0 | 说明 | 取出异或循环的初值；源码里这一步与乘积同属一行 |
| stk | 公式 | $\mathbf{q} = \mathrm{stack}(\mathbf{q}, \dim=-1)$ |
| stk | 符号 | $\mathbf{q}$：tokens，各回看步的压缩 id<br>$4$：回看步数，shift 从 0 到 3 |
| stk | 说明 | 4 个回看步各一份 [$B$, $T$] |
| hcat | 公式 | $\mathbf{h} = \mathrm{cat}(\mathbf{h}, \dim=-1)$ |
| hcat | 符号 | $\mathbf{h}$：hashes，拼接后的桶下标<br>$3$：桶组数 |
| hcat | 说明 | 3 个桶组各 [$B$, $T$, 2, 8] |
| roll | 公式 | $\mathbf{r} = \mathbf{r} \oplus \mathbf{o}_{...,i}$ |
| roll | 符号 | $\mathbf{r}$：rolling，滚动异或值<br>$\mathbf{o}$：products，上一行的乘积<br>$i$：回看步，从 1 到 3<br>$\oplus$：按位异或 |
| roll | 说明 | 一次异或一个回看步，第 $i$ 步之后得到的是 $i$+1 元 n-gram 的哈希 |
| mod | 公式 | $\mathbf{r} \bmod \mathbf{\Pi}$ |
| mod | 符号 | $\mathbf{r}$：rolling，滚动异或值<br>$\mathbf{\Pi}$：primes，每个（层、桶组、头）的质数 |
| mod | 说明 | 每个（Engram 层，桶组，头）各有一个质数，互不重复；8 是 engram_n_heads，3 个桶组的结果沿最后一维拼成 24 |
| off | 公式 | $\mathbf{H} = \mathbf{h} + \mathbf{O}_b$ |
| off | 符号 | $\mathbf{H}$：hash，整张表的行号<br>$\mathbf{h}$：hs，拼接后的桶下标<br>$\mathbf{O}_b$：offsets，各桶在表里的起始行 |
| off | 说明 | 加上各桶在表里的起始偏移，得到整张表的行号 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 压缩映射 | $\mathbf{C} = T_m[I]$<br>把词 id 映到压缩 id |
| 公式 | 回看取值 | $\mathbf{C}_{t-i}$<br>第 t 个位置往前看第 i 步取到的压缩 id，i 从 0 到 3 |
| 公式 | 乘数 | $\mathbf{o} = \mathbf{q} \odot \mathbf{m}$<br>每个回看步的压缩 id 乘上该步的乘数 |
| 公式 | 滚动异或 | $\mathbf{r} = \mathbf{r} \oplus \mathbf{o}_{...,i}$<br>逐个回看步异或进滚动值 |
| 公式 | 分桶 | $\mathbf{h} = \mathbf{r} \bmod \mathbf{\Pi}$<br>每次异或后对质数取模得分桶下标 |
| 公式 | 行号 | $\mathbf{H} = \mathbf{h} + \mathbf{O}_b$<br>加分桶偏移得到整张表的行号 |
| 配置 | engram_max_ngram_size | 4 |
| 配置 | engram_n_heads | 8 |
| 配置 | engram_compressed_vocab_size | 99092 |
| 配置 | engram_pad_id | 2 |
| 配置 | engram_layer_ids | [1, 14] |
