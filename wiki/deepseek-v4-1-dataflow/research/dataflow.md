# DeepSeek-V4.1-Flash

- 描述：DeepSeek-V4.1-Flash 一次带图的 prefill 前向的数据流：一张图像先切成 patch，过 32 层视觉 Transformer（ViT，Vision Transformer）编码，再由对齐器按 3×3 降采样并投到主干维度，写回序列里图像占位那一段的隐状态；文本 token 一边查出每个位置的 n-gram 哈希行号，一边取词嵌入，整段一起进 40 层主干；压缩比大于 0 的层，除 128 个位置的滑动窗口外，另加索引器选出的压缩 KV（压缩 KV 只由 4 个 kv source 层产出并按层序共享，索引器在 2、8、14、20 四层，以及其后的 24、28、32、36 层各自用索引器选出参与注意力的 top-512 条压缩 KV）；前馈是 384 选 6 的专家混合（MoE，Mixture-of-Experts），图像 span 内用另一份路由偏置；最后折叠、归一化并投影成下一个 token 的 logits。
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
- 权重按 `config.json` 的 `dtype` 与 `expert_dtype` 加载：用 `Linear`、`ColumnParallelLinear`、`RowParallelLinear` 建；这些层里没写 dtype 的权重取 `dtype`（fp8），路由专家取 `expert_dtype`（fp4），压缩器在 `compress_ratio` 大于 1 时把 `wkv`、`wgate` 提升到 fp32；源码里显式指定了别的精度的按源码，如 wo_a、ParallelHead 的 weight、attn_sink、hc_* 六组参数、索引器的 wk.weight 与 weights_proj.weight、压缩比不大于 1 的压缩器的 wkv
- 注意力、压缩器与索引器走的是 compress_ratio 大于 0 这条分支：kv source 层 2、8、14 的 compress_ratio 是 2、结构相同；层 20 的 compress_ratio 是 1，压缩器只有压缩那一份的投影（不做逐 token 的池化，只留 wkv 一路），索引器仍由它选出候选块
- 索引器在 8 个 index source 层（2、8、14、20、24、28、32、36）各跑一次，这 8 层画成同一张图：2、8、14、20 自己把压缩潜在量变成索引键，层 20 另选出候选块，24、28、32、36 不建键、只从前面发布的那份 index_k 读键，并在层 20 选出的候选块里取 top-512
- 注意力只有 `sparse_attn` 一个 kernel，窗口 KV 与压缩 KV 拼成一份 KV 之后一次算完
- 调用的是 `Transformer.forward`，不是 `forward_spec`（草稿路径）。config.json 里叫 dspark_target_layer_ids 的那 3 个层号 [37, 38, 39] 在这里是 `target_layer_ids`，用来在进第 37、38、39 层之前收集 main_hidden，属于本次前向；真正走草稿路径的是 forward_spec 与那 3 个多 token 预测（MTP，Multi-Token Prediction）层，二者不是一回事
- 缓存按 `max_batch_size` 分配，`config.json` 里没有这一项，取 `ModelArgs` 默认值 4；它是缓存容量，与本次前向的批大小 B 不是同一个量。频率表与压缩 KV／索引键缓存的序列维取 `max_seq_len` 默认值 4096，`config.json` 里也没有这一项；窗口 KV 缓存只有 `window_size` 个槽位

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
| DSpark | dspark_target_layer_ids | [37, 38, 39]（DSpark 指半自回归草稿 + 置信度调度验证的推测解码那一套；本页用的这 3 个层号是收集 main_hidden 的 tap，与草稿路径无关） |
| 视觉 | vision_n_layers | 32 |
| 视觉 | vision_dim | 1024 |
| 视觉 | vision_n_heads | 16 |
| 视觉 | vision_inter_dim | 2816 |
| 视觉 | vision_patch_size | 14 |
| 视觉 | vision_rope_theta | 10000 |
| 视觉 | vision_downsample_ratio | 3 |
| 视觉 | vision_max_n_token | 1024 |
| 视觉 | vision_min_pixels | 295936 |
| 视觉 | vision_max_wh_ratio | null（config.json 里的值就是 null，图像不按长宽比上限截断） |
| 视觉 | image_token_id | 129264 |
| 量化 | dtype | fp8 |
| 量化 | expert_dtype | fp4 |

## Transformer.forward（model.py:1242-1272）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| ids | 张量 | input_ids | int64 [B, T] | 每个位置都是真实 token，填充不会传进来；图像 span 内每个位置都是 image_token_id | model.py:1243 |
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
| mrg | 算子 | self.merge_image_embeddings(images, h) | bf16 [B, T, 5120] 与 bf16 [n_llm_h × n_llm_w, 5120] → bf16 [B, T, 5120] |  | model.py:1256 |
| himg | 张量 | h | bf16 [B, T, 5120] | 图像 span 里 IMAGE 位置已换成对齐器的行，起止、换行位置换成学到的嵌入 | model.py:1256 |
| hcexp | 算子 | h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1) | [B, T, 5120] → [B, T, 4, 5120] |  | model.py:1258 |
| hc | 张量 | h | bf16 [B, T, 4, 5120] | | model.py:1258 |
| eng | 算子 | layer.engram(h, engram_hashes[:, :,<br>layer.engram.layer_hash_index, :], engram_mask) | [B, T, 4, 5120] → [B, T, 4, 5120] |  | model.py:1262-1263 |
| eh | 张量 | h | bf16 [B, T, 4, 5120] | 层 1、14 在进 Block 之前，Engram 写回的残差 | model.py:1263 |
| pm0 | 算子 | make_identity_pre_mix(h, self.hc_mult) | [B, T, 4, 5120] → fp32 [B, T, 4] |  | model.py:1260 |
| pre | 张量 | pre_mix | fp32 [B, T, 4] | 进入第 0 层时的初值，此后每层返回的 ffn_pre 都覆盖它 | model.py:1260 |
| layers | 算子 | 40 × Block | [B, T, 4, 5120] → [B, T, 4, 5120] |  | model.py:1267 |
| lpm | 张量 | pre_mix | fp32 [B, T, 4] | 最后一层返回的 pre，送给最后一次 hc_pre | model.py:1267 |
| hl | 张量 | h | bf16 [B, T, 4, 5120] | 循环里每一层 Block 返回、又喂给下一层的那份残差流（本图合并成同一个量看）；层 1、14 进层前先过 Engram 的是它，第 37、38、39 层进层前取均值的是它，循环结束后交给最后一次 hc_pre 的也是它 | model.py:1261-1267 |
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
| shar | 缓存 | shared_attn | compress_kv、index_k、topk_idxs、candidates | 各层之间传递的共享量，每个前向里由靠前的层写、靠后的层读，不重置。candidates 由层 20 写，层 24、28、32、36 读 | model.py:1180 |
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
- mrg → himg
- himg → hcexp
- hcexp → hc
- hc → pm0：形状与 device
- pm0 → pre
- layers → hl：循环里每层返回的残差流（合并成同一份看）
- hl → eng：层 1、14 进层前的输入
- hash → eng
- emv → eng：engram_mask
- eng → eh
- eh → layers：层 1、14 的输入
- hc → layers：层 0 的输入
- pre → layers
- hl → layers：层 2–39（除层 1、14 外）的输入，循环里上一层 Block 返回的残差流
- lpm → layers：上一层返回的 pre_mix
- im → layers：image_mask
- shar → layers
- layers → shar
- hl → avg：第 37、38、39 层进层前的输入
- avg → am
- am → mcat
- mcat → mh
- hl → col
- layers → lpm
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
| cim | 公式 | $image\_mask = token\_types \ge 0$，token_types 是每个位置的类型，TEXT 是 −1，所以非负的都是图像 span 里的位置 |
| cim | 说明 | 图像 span 内所有位置在 input_ids 里都是 image_token_id，只有 token_types 能区分它们与文本 |
| cem | 公式 | $engram\_mask = \lnot image\_mask$，图像 span 的位置为 False |
| cem | 说明 | NgramHashState 拿它当 token_mask，False 的位置不参与 n-gram，在 cache 里记成 DEAD |
| ngram | 公式 | $engram\_hashes = engram\_hash(input\_ids, start\_pos, engram\_mask)$，input_ids 是词 id，start_pos 为 0，engram_mask 是这次传进来的张量，图像 span 的位置为 False |
| ngram | 说明 | NgramHashState，把每个位置结尾的 n-gram 哈希成表行号<br>倒数第二维是 Engram 层序号，对应 engram_layer_ids 的 [1, 14]；24 = (4 − 1) × 8 |
| emb | 公式 | $h = embed.weight[input\_ids]$，input_ids 是词 id，embed.weight 是词表矩阵，单卡下按行查 |
| emb | 说明 | ParallelEmbedding：embed.weight [129280, 5120]，按 rank 沿词表维切分，单卡即整张。图像 span 内的位置也照常查表，查出来的结果随后被覆盖 |
| venc | 公式 | $x = vision(patches, n\_vit\_h, n\_vit\_w)$，patches 是这张图的 patch 网格，n_vit_h、n_vit_w 是它的高宽 |
| venc | 说明 | 只对图算一次，与文本的序列长度无关。ViT 的输出交给同一图的 alen |
| alen | 公式 | $embeds = aligner(x, n\_vit\_h, n\_vit\_w)$，x 是 ViT 的输出，n_vit_h、n_vit_w 是 patch 网格的高宽 |
| alen | 说明 | 对齐器把 ViT 的输出按 3×3 降采样再投到主干维度。encode_image 是 Transformer 上的一个方法，把 self.vision 与 self.aligner 接起来 |
| mrg | 公式 | $span = h_{i,\ img.start : img.start + numel(types)}$；按行序把 embeds 的行写进 IMAGE 位置，另三个位置换成学到的嵌入：$span[types = IMAGE\_START] = image\_start$、$span[types = IMAGE\_END] = image\_end$、$span[types = IMAGE\_NEW\_LINE] = image\_newline$、$span[types = IMAGE] = embeds$。span 只是切片的别名，写回的就是 h 本身；types 是 `img.types` 移到 h 的 device 之后的那一份 |
| mrg | 说明 | 大写的是 token 类型（IMAGE_START 等），小写的 image_start [5120]、image_end [5120]、image_newline [5120] 是 Transformer 上三个学到的嵌入参数，to(h.dtype) 之后写进去 |
| hcexp | 公式 | $h$ 沿新的一维复制 $hc\_mult$ 份，每一份等于原来的 $h$ |
| hcexp | 说明 | 把残差流复制成 hc_mult 份，供 Hyper-Connections 用 |
| eng | 公式 | $h = engram(h, engram\_hashes_{:,:,layer\_hash\_index,:}, engram\_mask)$，h 是残差流，engram_mask 是这次传进来的张量 |
| eng | 说明 | Engram：层 1、14 在进入本层之前跑，吃的是上一层 Block 的输出，把 n-gram 查表的结果按门控加进残差流 |
| pm0 | 公式 | $pre\_mix_{:,:,0} = 1$，其余为 0。h 用来取 batch 和序列长度，hc_mult 是份数 |
| pm0 | 说明 | 第 0 份置 1、其余置 0，作为进入第 0 层时的 pre_mix |
| layers | 公式 | $h, pre\_mix = layer(h, start\_pos, pre\_mix, image\_mask)$，layer 是这一层的 Block，image_mask 是这次传进来的张量，图像 span 的位置为 True |
| layers | 说明 | 层 0 的输入是 hc；层 1、14 在进本层之前先跑 Engram 改写残差流；其余层的输入是上一层 Block 的输出<br>压缩 KV 只在 4 个 kv source 层（2、8、14、20）算一次，其余 compress_ratio 大于 0 的层从 shared_attn 读。层 0、1 的 compress_ratio 为 0，只有滑动窗口。层 20–39 的 compress_ratio 是 1。层 20 的索引器写出 candidates，层 24、28、32、36 只在这些候选块里取 top-512<br>image_mask 一路传到每层的 MoE，图像 span 的路由走 VL 偏置 |
| avg | 公式 | $\mathrm{mean}(h, \dim=2)$，h 是进入目标层之前的残差流，取均值的那一维是 hc 份 |
| mcat | 公式 | $main\_hidden = \mathrm{cat}(main\_hiddens, \dim=-1)$，main_hiddens 是 3 个目标层各一份 [B, T, 5120] |
| avg | 说明 | 进入第 37、38、39 层之前的残差流，按 hc 维取均值 |
| col | 公式 | $h = \sum_{c=0}^{hc\_mult-1} pre\_mix_c\, h_c$，h_c 是第 c 份残差，pre_mix 是最后一层返回的系数 |
| col | 说明 | 用最后一层返回的 pre_mix 把 4 份折叠成一份 |
| fnorm | 公式 | $h = norm.weight \odot \dfrac{h}{\sqrt{\mathrm{mean}(h^2) + norm.eps}}$，h 是送进 norm 的隐状态，norm 是 RMSNorm，mean 在最后一维，norm.eps 取 1e-20 |
| fnorm | 说明 | norm.weight [5120] |
| head | 公式 | $logits = h_{:,-1}\, head.weight^\top$，h 是送进 head 的隐状态，只取最后一个位置 $h_{:,-1}$，head.weight 是词表矩阵 |
| head | 说明 | head.weight [129280, 5120]，fp32 |
| samp | 公式 | $logits = logits / \max(temperature, 10^{-5})$，$probs = \mathrm{softmax}(logits)$，$output\_ids = \mathrm{argmax}(probs / \mathrm{exponential})$。temperature 取默认值 1，exponential 是同形状的指数分布随机数 |
| samp | 说明 | Gumbel-max：概率除以指数分布随机数后取 argmax，等价于按概率采样 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 介绍 | 模型 | DeepSeek-V4.1-Flash 是 DeepSeek 的多模态 MoE 模型，主干 552B 参数、每个 token 激活 8B（prefill）/ 16B（decode），上下文最长 100 万 token。主干 40 层，注意力用压缩稀疏注意力（CSA2），KV 压到每 token 890 字节。每层 MoE 是 384 个路由专家取 6 个、加 1 个共享专家；另有 Engram 条件记忆和 DSpark 推测解码。视觉侧是 DeepSeek-ViT 加一个两层 MLP 投影。 |
| 介绍 | 这次前向 | 一次带图的 prefill：`start_pos` 为 0，一张图先过 32 层 ViT 与对齐器，把输出写进序列里图像 span 的隐状态；文本位置照常取词嵌入与 n-gram 哈希；整段一起过 40 层主干，最后取最后一个位置投影出下一个 token 的 logits。 |
| 公式 | 输入嵌入 | $h = embed.weight[input\_ids]$<br>input_ids 是词 id，embed.weight 是词表矩阵，单卡按行查 |
| 公式 | 图像嵌入 | $embeds = encode\_image(patches, n\_vit\_h, n\_vit\_w)$，再按行序把 embeds 的行写进图像占位那一段 $h_{i,\ img.start : img.start + numel(types)}$ 里的 IMAGE 位置<br>img 是这张图、img.start 是这段在序列里的起点、types 是该图的 token_types、i 是批内序号；起止与换行位置换成 Transformer 的 image_start、image_end、image_newline 三个可学习参数 |
| 公式 | 残差流复制 | $h$ 沿新的一维复制 $hc\_mult$ 份，每一份等于原来的 $h$<br>hc_mult 是 4 |
| 公式 | 输出 | $logits = h_{:,-1}\, head.weight^\top$<br>h 是送进 head 的隐状态，head.weight 是词表矩阵，只取最后一个位置 |
| 配置 | vocab_size | 129280 |
| 配置 | dim | 5120 |
| 配置 | n_layers | 40 |
| 配置 | hc_mult | 4 |
| 配置 | dspark_target_layer_ids | [37, 38, 39] |
| 配置 | engram_layer_ids | [1, 14] |
| 配置 | vision_n_layers | 32 |
| 配置 | vision_dim | 1024 |
| 配置 | vision_patch_size | 14 |
| 配置 | vision_downsample_ratio | 3 |
| 配置 | image_token_id | 129264 |

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
| vblocks | 算子 | 32 × vision.Block | bf16 [S, 1024] → bf16 [S, 1024] |  | vision.py:101-102 |
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
| vpe | 公式 | $x = \mathrm{flatten}(patches, 1)\, proj.weight^\top + proj.bias$，patches 是这张图的 patch 网格，flatten 把每个 patch 的 3 × 14 × 14 摊平，patch_embed 是 PatchEmbed，就是一次线性投影 |
| vpe | 说明 | PatchEmbed：proj 是 nn.Linear（vision.py:40），proj.weight [1024, 588]、proj.bias [1024]，588 = 3 × 14 × 14 |
| vcs_op | 公式 | $hpos$、$wpos$ 是每个 patch 在网格里的高、宽坐标；$freqs = \mathrm{flatten}([hpos, wpos] \odot inv\_freq)$，$cos = \cos(freqs)$，$sin = \sin(freqs)$，各补一个长度为 1 的维度。$inv\_freq_j = \theta^{-2j/rope\_dim}$，rope_dim 是取频率用的维度（vision_dim // vision_n_heads // 2 = 32），$\theta$ 是 vision_rope_theta |
| vcs_op | 说明 | 频率长度 32 = rope_dim = 每个头宽度的一半，一个头里的前半段配高坐标、后半段配宽坐标；整张图只算一次，32 层共用 |
| vblocks | 公式 | $x = block(x, cos, sin)$，block 是这一层的 vision.Block |
| vblocks | 说明 | 32 层结构相同，都拿同一份 cos、sin；差异只有各自那一份权重 |
| vnorm | 公式 | $x = norm.weight \odot \dfrac{x}{\sqrt{\mathrm{mean}(x^2) + norm.eps}}$，x 是最后一层 Block 的输出，norm 是 vision.RMSNorm，mean 在最后一维，eps 取 1e-6 |
| vnorm | 说明 | vision.norm.weight [1024] |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 二维位置编码 | $freqs = \mathrm{flatten}([hpos, wpos] \odot inv\_freq)$，$cos = \cos(freqs)$，$sin = \sin(freqs)$。hpos、wpos 是 patch 在网格里的高、宽坐标，$inv\_freq_j = \theta^{-2j/rope\_dim}$，rope_dim = vision_dim // vision_n_heads // 2 = 32，$\theta$ 是 vision_rope_theta，频率长度 32 = rope_dim = 每个头宽度的一半 |
| 配置 | vision_n_layers | 32 |
| 配置 | vision_dim | 1024 |
| 配置 | vision_n_heads | 16 |
| 配置 | vision_patch_size | 14 |
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
| vbn1 | 公式 | $x = weight \odot x / \sqrt{\mathrm{mean}(x^2) + eps}$，x 是本层输入，weight 是 norm1.weight，mean 在最后一维，eps 取 1e-6 |
| vbn1 | 说明 | norm1.weight [1024]，norm1 是 vision.RMSNorm，vision 支路统一 eps = 1e-6 |
| vba | 公式 | $x = attn(x, cos, sin)$，x 是 norm1 之后的隐状态，cos、sin 是整张图共用的二维频率表 |
| vba | 说明 | Attention，全双向，没有因果掩码；这一层用自己那份 wqkv、wo |
| vbad | 公式 | $x_{out} = x + attn(x_1, cos, sin)$，加号左边的 x 是本层输入（进注意力之前的残差），$x_1$ 是 norm1 的输出（喂给注意力那份），attn(...) 是注意力的输出 |
| vbn2 | 公式 | $x = weight \odot x / \sqrt{\mathrm{mean}(x^2) + eps}$，x 是注意力那一步的输出，weight 是 norm2.weight，mean 在最后一维，eps 取 1e-6 |
| vbn2 | 说明 | norm2.weight [1024]，norm2 是 vision.RMSNorm，vision 支路统一 eps = 1e-6 |
| vmlp | 公式 | $x = (\mathrm{silu}(gate) \odot up)\, w2.weight^\top$，$gate, up = \mathrm{chunk}(x\, w1.weight^\top, 2)$，x 是 norm2 之后的隐状态，w1、w2 无 bias |
| vmlp | 说明 | 通用 SwiGLU 的 MLP（vision.MLP），w1.weight [5632, 1024]、w2.weight [1024, 2816]，5632 = 2 × vision_inter_dim，2816 是 vision_inter_dim；w1、w2 是 nn.Linear（PyTorch 自带） |
| vbad2 | 公式 | $x_{out} = x + mlp(x_2)$，加号左边的 x 是注意力那一步的输出（进前馈之前的残差），$x_2$ 是 norm2 的输出（喂给前馈那份），mlp(...) 是前馈的输出 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 残差 | 注意力和前馈各包一层 pre-norm 的残差：$x = x + attn(norm1(x), cos, sin)$，$x = x + mlp(norm2(x))$。norm1、norm2 是 RMSNorm |
| 配置 | vision_dim | 1024 |
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
| vqkv | 张量 | self.wqkv(x) | bf16 [S, 3072] | 3072 = 3 × vision_dim，一次投影给出 q、k、v 三段 | vision.py:56 |
| vsplit | 算子 | self.wqkv(x).chunk(3, dim=-1) | bf16 [S, 3072] → bf16 [S, 1024] × 3 |  | vision.py:56 |
| vqkv2 | 张量 | q, k, v | bf16 [S, 1024] × 3 | 三份各 vision_dim 维 | vision.py:56 |
| vview | 算子 | (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1)) | bf16 [S, 1024] × 3 → bf16 [S, 16, 64] × 3 |  | vision.py:56 |
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
- vqkv → vsplit
- vsplit → vqkv2
- vqkv2 → vview
- vax → vview：形状（S）
- vnh → vview
- vhdm → vview
- vview → vq
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
| vwqkv | 公式 | $qkv = x\, wqkv.weight^\top + wqkv.bias$，x 是 norm1 之后的隐状态 |
| vwqkv | 说明 | wqkv.weight [3072, 1024]、wqkv.bias [3072]，3072 = 3 × 1024；wqkv 是 nn.Linear（PyTorch 自带，与主干用的 Linear 类不是一类） |
| vsplit | 公式 | $q, k, v = \mathrm{chunk}(qkv, 3, \dim=-1)$ |
| vview | 公式 | $q, k, v$ 各 view 成 [S, 16, 64]，16 是 n_heads，64 是 head_dim |
| vview | 说明 | 16 × 64 = 1024，头维插在序列维之后，好按头做注意力 |
| vcq | 公式 | $x_1, x_2 = \mathrm{chunk}(q.\mathrm{float}(), 2, \dim=-1)$，返回 $\mathrm{cat}(x_1 \cos - x_2 \sin,\ x_2 \cos + x_1 \sin)$ 再转回输入 dtype。cos、sin 是二维频率表 |
| vcq | 说明 | 每个头 64 维的前一半与后一半配成一对（第 i 维配第 i+32 维），位置信息来自 patch 在网格里的高、宽 |
| vck | 公式 | $x_1, x_2 = \mathrm{chunk}(k.\mathrm{float}(), 2, \dim=-1)$，返回 $\mathrm{cat}(x_1 \cos - x_2 \sin,\ x_2 \cos + x_1 \sin)$ 再转回输入 dtype |
| vck | 说明 | 与查询用同一对 cos、sin |
| vsdpa | 公式 | $o = \mathrm{softmax}(q k^\top / \sqrt{64})\, v$，q、k、v 都先把头维挪到最前，64 是 head_dim |
| vsdpa | 说明 | PyTorch 的 F.scaled_dot_product_attention，不给掩码、不做 dropout，整张图里 patch 之间全双向 |
| vwo | 公式 | $x = o.\mathrm{transpose}(0, 1).\mathrm{reshape}(S, -1)\, wo.weight^\top + wo.bias$ |
| vwo | 说明 | wo.weight [1024, 1024]、wo.bias [1024]；wo 也是 nn.Linear（PyTorch 自带） |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 注意力 | $o = \mathrm{softmax}(q k^\top / \sqrt{head\_dim})\, v$，缩放系数是 head_dim 的 −1/2 次方，这里的 head_dim 是视觉头的宽度 64；不给掩码，整张图的 patch 互相都看 |
| 配置 | vision_dim | 1024 |
| 配置 | vision_n_heads | 16 |
| 配置 | vision_rope_theta | 10000 |

## Aligner.forward（vision.py:114-119）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| vout | 张量 | x | bf16 [S, 1024] | ViT 的输出，沿用 ViT 图的 vout，S = n_vit_h × n_vit_w | vision.py:114 |
| vv | 算子 | x.view(n_h, n_w, -1).permute(2, 0, 1) | bf16 [S, 1024] → bf16 [1024, n_vit_h, n_vit_w] |  | vision.py:116 |
| vv2 | 张量 | x | bf16 [1024, n_vit_h, n_vit_w] | 通道维挪到最前 | vision.py:116 |
| vpad | 算子 | F.pad(x, (0, -n_w % r, 0, -n_h % r)) | bf16 [1024, n_vit_h, n_vit_w] → bf16 [1024, n_vit_h + (-n_vit_h % r), n_vit_w + (-n_vit_w % r)] |  | vision.py:117 |
| vpv | 张量 | x | bf16 [1024, n_vit_h + (-n_vit_h % r), n_vit_w + (-n_vit_w % r)] | 高、宽都补到 r 的整数倍；n_vit_h、n_vit_w 已经是 r 的整数倍时补 0 | vision.py:117 |
| vunf | 算子 | F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1) | bf16 [1024, n_vit_h + (-n_vit_h % r), n_vit_w + (-n_vit_w % r)] → bf16 [n_llm_h × n_llm_w, 9216] |  | vision.py:118 |
| vuv | 张量 | F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1) | bf16 [n_llm_h × n_llm_w, 9216] | 9216 = 3 × 3 × 1024；n_llm_h = ceil(n_vit_h / r)、n_llm_w = ceil(n_vit_w / r)，每行是一个 3×3 块的展开 | vision.py:118 |
| aw1 | 算子 | self.w1(x) | bf16 [n_llm_h × n_llm_w, 9216] → bf16 [n_llm_h × n_llm_w, 5120] |  | vision.py:119 |
| alg | 张量 | F.gelu(self.w1(x)) | bf16 [n_llm_h × n_llm_w, 5120] | w1 的输出再过一层 gelu（普通函数不单独成节点，跟它作用的值同节点）；源码里这一步嵌在下一行的调用里，没有单独的变量名 | vision.py:119 |
| vlin | 算子 | self.w2(alg) | bf16 [n_llm_h × n_llm_w, 5120] → bf16 [n_llm_h × n_llm_w, 5120] |  | vision.py:119 |
| vemb | 张量 | self.w2(F.gelu(self.w1(x))) | bf16 [n_llm_h × n_llm_w, 5120] | 对齐器的输出，与主干同一个维度，可以按行写回序列；与 Transformer 图的 vemb 是同一个量 | vision.py:119 |
| vrd | 张量 | self.downsample_ratio | int，值为 3 | 构造时取 vision_downsample_ratio 3 | vision.py:109 |

### 连线

- vout → vv：n_vit_h、n_vit_w（源码里的 n_h、n_w）
- vv → vv2
- vv2 → vpad
- vpad → vpv
- vpv → vunf
- vunf → vuv
- vrd → vunf：块大小与步长
- vrd → vpad：r
- vout → vpad：n_vit_h、n_vit_w
- vuv → aw1
- aw1 → alg
- alg → vlin
- vlin → vemb

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| vv | 公式 | $x = x.\mathrm{view}(n\_h, n\_w, -1).\mathrm{permute}(2, 0, 1)$，n_h、n_w 就是 n_vit_h、n_vit_w |
| vv | 说明 | 把 patch 维拆回高、宽，通道挪到最前，好按空间切块；1024 是 vision_dim |
| vpad | 公式 | $x = \mathrm{pad}(x, (0, -n\_w \% r,\ 0, -n\_h \% r))$，r 是 downsample_ratio，负数表示在右边、下边补 |
| vpad | 说明 | 高、宽各补到 r 的整数倍，补 0。vision_max_n_token 1024 限住了总 token 数，n_vit_h、n_vit_w 不是 r 的整数倍时才需要补 |
| vunf | 公式 | $x = \mathrm{unfold}(x.\mathrm{unsqueeze}(0), r, \mathrm{stride}=r).\mathrm{squeeze}(0).\mathrm{transpose}(0, 1)$，把高宽平面切成不重叠的 r×r 块，每块摊成一行。这里的 r 是视觉的 downsample_ratio（3），与注意力里的 compress_ratio 同形不同义 |
| vunf | 说明 | 一块 r × r × 1024 摊平成 9216 维；输出行数是 ceil(n_vit_h / r) × ceil(n_vit_w / r)，就是图像在序列里占的 token 网格 |
| aw1 | 公式 | $out = x\, w1.weight^\top + w1.bias$，x 是展开后的 3×3 块，w1 把 9216 维映到 5120 |
| aw1 | 说明 | w1.weight [5120, 9216]、w1.bias [5120]；w1 是 nn.Linear（PyTorch 自带） |
| vlin | 公式 | $embeds = \mathrm{gelu}(w1(x))\, w2.weight^\top + w2.bias$，gelu(w1(x)) 是上一节点的输出，w2 保持 5120 维 |
| vlin | 说明 | w2.weight [5120, 5120]、w2.bias [5120]；w2 是 nn.Linear（PyTorch 自带） |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 降采样 | 把 patch 网格按 r × r 不重叠地切成块，每块摊平后过一次两层的 MLP：$out = w2(\mathrm{gelu}(w1(\mathrm{unfold}(x))))$，r 是 vision_downsample_ratio，展平后一块是 3 × 3 × 1024 = 9216 维 |
| 配置 | vision_downsample_ratio | 3 |
| 配置 | vision_dim | 1024 |
| 配置 | dim | 5120 |
| 配置 | vision_max_n_token | 1024 |

## Block.forward（model.py:968-994）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hin | 张量 | x | bf16 [B, T, 4, 5120] | 进入本层的残差流，4 份；进注意力前的 `residual = x` 就是它（同一份重新绑定，不另立节点） | model.py:970 |
| bpre | 张量 | pre_mix | fp32 [B, T, 4] | 上一层返回的 pre_mix，把 4 份折叠成一份时用 | model.py:972 |
| im | 张量 | image_mask | bool [B, T] | 这一层前向的参数，由外层一路传进 MoE；图像 span 的位置为 True；与 Transformer 图的 im 是同一个量 | model.py:973 |
| afn | 张量 | self.hc_attn_fn | fp32 [24, 20480] | 注意力那次的混合投影权重，24 = (2 + 4) × 4，20480 = 4 × 5120 | model.py:941 |
| asc | 张量 | self.hc_attn_scale | fp32 [3] | 注意力那次的三个缩放系数 | model.py:945 |
| abs | 张量 | self.hc_attn_base | fp32 [24] | 注意力那次的偏置 | model.py:943 |
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
| ffn_a | 张量 | self.hc_ffn_fn | fp32 [24, 20480] | 前馈那次的混合投影权重，形状同注意力那组 | model.py:942 |
| fsc | 张量 | self.hc_ffn_scale | fp32 [3] | 前馈那次的三个缩放系数 | model.py:946 |
| fbs | 张量 | self.hc_ffn_base | fp32 [24] | 前馈那次的偏置 | model.py:944 |
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
- ffn_a → mixes_f
- fsc → mixes_f
- fbs → mixes_f
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
| mixes_a | 公式 | $mixes = \mathrm{flatten}(x, 2)\, hc\_attn\_fn^\top \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x, 2)^2) + norm\_eps)$<br>$pre_j = \mathrm{sigmoid}(mixes_j \cdot hc\_attn\_scale_0 + hc\_attn\_base_j) + hc\_eps$<br>$post_j = 2\,\mathrm{sigmoid}(mixes_{j+hc\_mult} \cdot hc\_attn\_scale_1 + hc\_attn\_base_{j+hc\_mult})$<br>$u_{j,k} = mixes_{j \cdot hc\_mult + k + 2 hc\_mult} \cdot hc\_attn\_scale_2 + hc\_attn\_base_{j \cdot hc\_mult + k + 2 hc\_mult}$<br>$comb = \mathrm{Sinkhorn}(u)$<br>x 是残差流，hc_attn_fn、hc_attn_scale、hc_attn_base 是这一调用的参数，norm_eps 与 hc_eps 是 Block 上的 eps。j、k 从 0 到 hc_mult−1。mixes 与 base 按顺序切成三段：前 hc_mult 个给 pre，接着 hc_mult 个给 post，最后 hc_mult × hc_mult 个给 comb。u 是 mixes 里填进 comb、softmax 之前的那一块，源码里没有单独的名字，Sinkhorn 作用在整张 u 上 |
| mixes_a | 说明 | 一次投影同时给出 pre、post、comb 三套系数<br>hc_attn_fn [24, 20480]、hc_attn_base [24]、hc_attn_scale [3]，都是 fp32；24 = (2 + 4) × 4，20480 = 4 × 5120<br>Sinkhorn 先对 u 的每一行做 softmax，再加 hc_eps；再按列除以（列和 + hc_eps）；然后行、列交替各再做 19 次，每次除以（和 + hc_eps）。行 20 次、列 20 次，最后一次是列 |
| cpre | 公式 | $x = \sum_{c=0}^{hc\_mult-1} pre\_mix_c\, x_c$，x_c 是第 c 份残差，pre_mix 是上一层返回的系数 |
| cpre | 说明 | 按 pre_mix 把 4 份加权求和成 1 份 |
| an | 公式 | $x = attn\_norm.weight \odot \dfrac{x}{\sqrt{\mathrm{mean}(x^2) + attn\_norm.eps}}$，x 是送进来的张量，attn_norm 是 RMSNorm，mean 在最后一维，eps 取 1e-20 |
| an | 说明 | attn_norm.weight [5120] |
| attn | 公式 | $x = attn(x, start\_pos)$，x 是 attn_norm 之后的隐状态，这次 attn_args 为空 |
| attn | 说明 | Attention，本层的窗口 KV 与压缩 KV 都在里面读写；本路径 attn_args 为空 |
| cpost | 公式 | $x_c = attn\_post_c\, x + \sum_{c'} attn\_comb_{c',c}\, residual_{c'}$，右边的 x 是注意力的输出，residual 就是同图的 hin（进注意力前的残差），attn_post 与 attn_comb 是 hc_mixes 的系数。c' 是 residual 的份号，c 是写回的份号 |
| cpost | 说明 | 把子层输出按 post 展开成 4 份，再把 residual 按 comb 混进去 |
| mixes_f | 公式 | $mixes = \mathrm{flatten}(x, 2)\, hc\_ffn\_fn^\top \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x, 2)^2) + norm\_eps)$<br>$pre_j = \mathrm{sigmoid}(mixes_j \cdot hc\_ffn\_scale_0 + hc\_ffn\_base_j) + hc\_eps$<br>$post_j = 2\,\mathrm{sigmoid}(mixes_{j+hc\_mult} \cdot hc\_ffn\_scale_1 + hc\_ffn\_base_{j+hc\_mult})$<br>$u_{j,k} = mixes_{j \cdot hc\_mult + k + 2 hc\_mult} \cdot hc\_ffn\_scale_2 + hc\_ffn\_base_{j \cdot hc\_mult + k + 2 hc\_mult}$<br>$comb = \mathrm{Sinkhorn}(u)$<br>x 是残差流，hc_ffn_fn、hc_ffn_scale、hc_ffn_base 是这一调用的参数，norm_eps 与 hc_eps 是 Block 上的 eps。j、k 从 0 到 hc_mult−1。mixes 与 base 按顺序切成三段：前 hc_mult 个给 pre，接着 hc_mult 个给 post，最后 hc_mult × hc_mult 个给 comb。u 是 mixes 里填进 comb、softmax 之前的那一块，源码里没有单独的名字，Sinkhorn 作用在整张 u 上 |
| mixes_f | 说明 | hc_ffn_fn [24, 20480]、hc_ffn_base [24]、hc_ffn_scale [3]，形状与注意力那组相同<br>Sinkhorn 的做法与注意力那次相同：行 softmax 后加 hc_eps，再列归一化，然后交替各 19 次 |
| cpre2 | 公式 | $x = \sum_{c=0}^{hc\_mult-1} attn\_pre_c\, x_c$，x_c 是第 c 份残差，attn_pre 是本层注意力 hc_mixes 给出的 pre |
| cpre2 | 说明 | 用注意力子层算出的 attn_pre |
| fn | 公式 | $x = ffn\_norm.weight \odot \dfrac{x}{\sqrt{\mathrm{mean}(x^2) + ffn\_norm.eps}}$，x 是送进来的张量，ffn_norm 是 RMSNorm，mean 在最后一维，eps 取 1e-20 |
| fn | 说明 | ffn_norm.weight [5120] |
| moe_op | 公式 | $x = ffn(x, image\_mask)$，x 是 ffn_norm 之后的隐状态，image_mask 是这次传进来的张量，图像 span 的位置为 True |
| moe_op | 说明 | MoE，image_mask 一路传到 Gate，用来在图像 span 的位置上换一份路由偏置 |
| cpost2 | 公式 | $x_c = ffn\_post_c\, x + \sum_{c'} ffn\_comb_{c',c}\, residual_{c'}$，右边的 x 是前馈的输出，residual 就是同图的 x4（进前馈前的残差），ffn_post 与 ffn_comb 是 hc_mixes 的系数。c' 是 residual 的份号，c 是写回的份号 |
| cpost2 | 说明 | 把前馈输出按 ffn_post 展开成 4 份，再把 residual 按 ffn_comb 混进去 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 混合系数 | $mixes = \mathrm{flatten}(x, 2)\, hc\_fn^\top \cdot \mathrm{rsqrt}(\mathrm{mean}(\mathrm{flatten}(x, 2)^2) + norm\_eps)$，x 是残差流。注意力那次 hc_fn、hc_scale、hc_base 是 hc_attn_fn、hc_attn_scale、hc_attn_base，前馈那次是 hc_ffn_fn、hc_ffn_scale、hc_ffn_base |
| 公式 | pre、post、comb | $pre_j = \mathrm{sigmoid}(mixes_j \cdot hc\_scale_0 + hc\_base_j) + hc\_eps$，$post_j = 2\,\mathrm{sigmoid}(mixes_{j+hc\_mult} \cdot hc\_scale_1 + hc\_base_{j+hc\_mult})$，$u_{j,k} = mixes_{j \cdot hc\_mult + k + 2 hc\_mult} \cdot hc\_scale_2 + hc\_base_{j \cdot hc\_mult + k + 2 hc\_mult}$，再 $comb = \mathrm{Sinkhorn}(u)$。hc_scale、hc_base 在注意力是 hc_attn_scale、hc_attn_base，在前馈是 hc_ffn_scale、hc_ffn_base。j、k 从 0 到 hc_mult−1。Sinkhorn 先对每一行做 softmax 再加 hc_eps，再按列除以（列和 + hc_eps），然后行、列交替各 19 次 |
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
| fcall | 张量 | self.freqs_cis | 复数 fp32 [max_seq_len, 32] | 构造时按 max_seq_len 预生成的整张频率表。32 = rope_head_dim / 2 | model.py:688-698 |
| fsl | 算子 | self.freqs_cis[start_pos : start_pos + seqlen] | [max_seq_len, 32] → [T, 32] |  | model.py:767 |
| fc | 张量 | freqs_cis | 复数 fp32 [T, 32] | 本次这一段。本路径 start_pos 为 0，所以是前 T 行 | model.py:767 |
| wqa | 算子 | self.wq_a(x) | bf16 [B, T, 5120] → bf16 [B, T, 1280] |  | model.py:770 |
| qra | 张量 | self.wq_a(x) | bf16 [B, T, 1280] | wq_a 的输出，源码里嵌在下一行的调用里，没有单独的变量名 | model.py:770 |
| qr_op | 算子 | self.q_norm(qra) | bf16 [B, T, 1280] → bf16 [B, T, 1280] |  | model.py:770 |
| qr | 张量 | qr | bf16 [B, T, 1280] | 索引器拿它当 query 的输入 | model.py:770 |
| qb | 算子 | self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim)) | bf16 [B, T, 1280] → bf16 [B, T, 64, 512] |  | model.py:771 |
| qpre | 张量 | q | bf16 [B, T, 64, 512] | | model.py:771 |
| rope_q | 算子 | apply_rotary_emb(q[..., -rd:], freqs_cis) | bf16 [B, T, 64, 512] → bf16 [B, T, 64, 512] |  | model.py:772 |
| q | 张量 | q | bf16 [B, T, 64, 512] | | model.py:772 |
| wkv_proj | 算子 | self.wkv(x) | bf16 [B, T, 5120] → bf16 [B, T, 512] |  | model.py:705 |
| wkva | 张量 | self.wkv(x) | bf16 [B, T, 512] | wkv 的输出，源码里嵌在下一行的调用里，没有单独的变量名 | model.py:705 |
| wkv_win | 算子 | self.kv_norm(wkva) | bf16 [B, T, 512] → bf16 [B, T, 512] |  | model.py:705 |
| wkv_o | 张量 | kv | bf16 [B, T, 512] | | model.py:705 |
| wrope | 算子 | apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis) | bf16 [B, T, 512] → bf16 [B, T, 512] |  | model.py:706 |
| wkv_r | 张量 | kv | bf16 [B, T, 512] | | model.py:706 |
| wquant | 算子 | act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True) | bf16 [B, T, 512] → bf16 [B, T, 512] |  | model.py:707 |
| wkv_q | 张量 | kv | bf16 [B, T, 512] | 存的是 fp8 量化后再反量化的值 | model.py:707 |
| wwrite | 算子 | self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1) | bf16 [B, T, 512] → [max_batch_size, 128, 512] |  | model.py:712-715 |
| wchunk | 张量 | window_kv | bf16 [B, T, 512] | prefill 就是本次这块 KV | model.py:716 |
| widx | 算子 | get_window_topk_idxs(win, bsz, seqlen, start_pos) | [B, T, 5120] → int32 [B, T, 128] |  | model.py:720 |
| widxs | 张量 | topk_idxs | int32 [B, T, 128] | | model.py:720 |
| wcache | 缓存 | window_kv_cache | [max_batch_size, 128, 512] | 每层一份，按 max_batch_size 分配；prefill 把最后 128 个 token 按 cutoff 绕进环里；decode 时改写 start_pos % 128 那一槽 | model.py:663-668 |
| comp_op | 算子 | self.compressor(x, start_pos) | [B, T, 5120] → [B, T/r, 512] |  | model.py:747 |
| lat | 张量 | latent | bf16 [B, T/r, 512] | 压缩器输出的潜在量，还没加 RoPE；r 是这一层的 compress_ratio | model.py:747 |
| cidx_op | 算子 | self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len) | [B, T, 5120] → int32 [B, T, 512] |  | model.py:750 |
| cidxs | 张量 | idxs | int32 [B, T, 512] | 压缩条目下标，已经偏移到拼起来的 KV 里的位置 | model.py:750 |
| cf | 算子 | self.freqs_cis[: seqlen - seqlen % ratio : ratio] | [max_seq_len, 32] → [T/r, 32] |  | model.py:753-757 |
| cfq | 张量 | freqs | 复数 fp32 [T/r, 32] | | model.py:753-757 |
| crope | 算子 | apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs) | bf16 [B, T/r, 512] → bf16 [B, T/r, 512] |  | model.py:758 |
| lat_r | 张量 | latent | bf16 [B, T/r, 512] | | model.py:758 |
| cquant | 算子 | fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn) | bf16 [B, T/r, 512] → bf16 [B, T/r, 512] |  | model.py:760 |
| lat_q | 张量 | latent | bf16 [B, T/r, 512] | 存的是 fp4 量化后再反量化的值 | model.py:760 |
| cwrite | 算子 | self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent | bf16 [B, T/r, 512] → [max_batch_size, max_seq_len / r, 512] |  | model.py:761 |
| ccache | 缓存 | compress_kv_cache | [max_batch_size, max_seq_len / r, 512] | 2、8、14、20 各有一份，按 max_batch_size 分配；每个 kv source 层写自己这一份并把它发布给 shared_attn，它之后到下一个 kv source 层之前的层读的是这一份；decode 时每组填满才写一行 | model.py:669-679 |
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
| grp | 算子 | o.view(bsz, seqlen, self.n_local_groups, -1) | [B, T, 64, 512] → [B, T, 8, 4096] |  | model.py:785 |
| og | 张量 | o | bf16 [B, T, 8, 4096] | | model.py:785 |
| woa | 算子 | self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1) | [8192, 4096] → [8, 1024, 4096] |  | model.py:786 |
| wav | 张量 | wo_a | bf16 [8, 1024, 4096] | | model.py:786 |
| eins | 算子 | torch.einsum("bsgd,grd->bsgr", o, wo_a) | [B, T, 8, 4096] 与 [8, 1024, 4096] → [B, T, 8, 1024] |  | model.py:787 |
| ol | 张量 | o | bf16 [B, T, 8, 1024] | | model.py:787 |
| wb | 算子 | self.wo_b(o.flatten(2)) | bf16 [B, T, 8192] → bf16 [B, T, 5120] |  | model.py:788 |
| axout | 张量 | x | bf16 [B, T, 5120] | | model.py:788-789 |

### 连线

- ax → wqa
- wqa → qra
- qra → qr_op
- qr_op → qr
- qr → qb
- qb → qpre
- qpre → rope_q
- fcall → fsl
- ax → fsl：形状（seqlen）
- fsl → fc
- fc → rope_q
- rope_q → q
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
- fcall → cf
- ax → cf：形状（seqlen）
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
- attn_sink → core
- core → o0
- o0 → unrope
- fc → unrope
- unrope → o1
- o1 → grp
- grp → og
- og → eins
- woa → wav
- wav → eins
- eins → ol
- ol → wb
- wb → axout
- comp_op 点开 → Compressor.forward
- cidx_op 点开 → Indexer.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| wqa | 公式 | $wq\_a(x) = x\, wq\_a.weight^\top$，x 是本层输入，wq_a.weight 是第一层低秩投影的权重 |
| wqa | 说明 | wq_a.weight [1280, 5120] |
| qr_op | 公式 | $qr = q\_norm.weight \odot \dfrac{wq\_a(x)}{\sqrt{\mathrm{mean}(wq\_a(x)^2) + q\_norm.eps}}$，wq_a(x) 是上一节点的输出，mean 在最后一维 |
| qr_op | 说明 | q_norm.weight [1280]，q_norm 是 RMSNorm，q_norm.eps 取 1e-20 |
| wkv_proj | 公式 | $wkv(x) = x\, wkv.weight^\top$，x 是本层输入，wkv.weight 是窗口 KV 投影的权重 |
| wkv_proj | 说明 | wkv.weight [512, 5120] |
| wkv_win | 公式 | $kv = kv\_norm.weight \odot \dfrac{wkv(x)}{\sqrt{\mathrm{mean}(wkv(x)^2) + kv\_norm.eps}}$，wkv(x) 是上一节点的输出，mean 在最后一维 |
| wkv_win | 说明 | kv_norm.weight [512]，kv_norm 是 RMSNorm，kv_norm.eps 取 1e-20 |
| qb | 公式 | $q = \mathrm{unflatten}(qr\, wq\_b.weight^\top, (n\_local\_heads, head\_dim))$，qr 是上一行的输出，wq_b.weight 是第二层投影 |
| qb | 说明 | wq_b.weight [32768, 1280]，64 × 512 = 32768。n_local_heads 是本卡的头数，单卡下等于 n_heads 64 |
| fsl | 公式 | $freqs\_cis = self.freqs\_cis[start\_pos : start\_pos + seqlen]$，self.freqs_cis 是整张频率表，左边是切出来的这一段，start_pos 为 0，长度是 T |
| rope_q | 公式 | $q_{...,-rd:} = \mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(q_{...,-rd:}.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot freqs\_cis))$，q 是 query，freqs_cis 是这一段的频率，rd 是 rope_head_dim，相邻两维配成一对复数，写回原地 |
| rope_q | 说明 | 只旋转每头 512 维里的后 64 维，前 448 维不动；写在原地 |
| wrope | 公式 | $kv_{...,-rope\_head\_dim:} = \mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(kv_{...,-rope\_head\_dim:}.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot freqs\_cis))$，kv 是窗口 KV，freqs_cis 与 query 用的是同一份，相邻两维配成一对复数 |
| wrope | 说明 | 和 query 用同一份 freqs_cis |
| wquant | 公式 | $s = \mathrm{fast\_round\_scale}(\max(\mathrm{absmax}(kv), 10^{-4}), 1/448)$，$kv = \mathrm{bf16}(\mathrm{fp8}(\mathrm{clamp}(kv / s, -448, 448)) \cdot s)$。absmax 按每 32 个数一组取，不是逐元素。kv 是加过 RoPE 的窗口 KV，fast_round_scale 把 scale 收成 2 的幂 |
| wquant | 说明 | 每 32 个通道一个 E8M0 scale（E8M0 是只有 8 位指数、没有尾数的缩放格式），原地量化再反量化；量化按 fp8 精度（压缩 KV 走 fp4）<br>量化覆盖整条 512 维，含已经加过 RoPE 的末 64 维 |
| wwrite | 公式 | $cutoff = seqlen \bmod window\_size$，再把 $kv_{:,-window\_size:}$ 按 cutoff 切成两段，写进 window_kv_cache 的环首和环尾。kv 是量化后的窗口 KV |
| wwrite | 说明 | cutoff = seqlen % win，只留最后 win 个 token，尾巴绕回环首 |
| widx | 公式 | $idx\_base = \mathrm{clamp}(end - window\_size + 1, 0) + \{0,\ldots,window\_size-1\}$，$idxs = \mathrm{where}(idx\_base > end, -1, idx\_base)$。end 是 query 的位置（源码里的变量名），window_size 是 128。再扩成 batch 并转成 int32 |
| widx | 说明 | prefill 时每个 query 一行，下标是这次 window_kv（长度 T）里的位置；晚于这个 query 的位置写成 −1 |
| comp_op | 公式 | $latent = compressor(x, start\_pos)$，x 是本层输入，start_pos 为 0 |
| comp_op | 说明 | 只有 4 个 kv source 层（2、8、14、20）算；其余 compress_ratio 大于 0 的层 latent 为 None，直接读已有缓存；层 0、1 的压缩比为 0，整段压缩都不走 |
| cidx_op | 公式 | $idxs = \_compress\_topk\_idxs(x, qr, latent, start\_pos, offset, compress\_len)$，offset 是窗口 KV 的长度 |
| cidx_op | 说明 | 本层是 index source（2、8、14、20、24、28、32、36）就跑自己的 Indexer，其余 compress_ratio 大于 0 的层直接复用 shared_attn.topk_idxs<br>offset 是 window_kv 的长度，本次是 T；compress_len 是 (start_pos + T) / r |
| cf | 公式 | $freqs = freqs\_cis[:seqlen - seqlen \% ratio: ratio]$，第 j 组取位置 $j \times ratio$，ratio 是这一层的 compress_ratio |
| cf | 说明 | 取压缩位置的频率，一个 latent 顶替一整组，第 j 组取位置 j × r |
| crope | 公式 | $latent_{...,-rope\_head\_dim:} = \mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(latent_{...,-rope\_head\_dim:}.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot freqs))$，latent 是压缩潜在量，freqs 是压缩位置的频率，相邻两维配成一对复数 |
| crope | 说明 | 乘压缩位置的 freqs |
| cquant | 公式 | $s = \mathrm{fp8}(\max(\mathrm{absmax}(latent), 6 \cdot 2^{-9}) / 6)$，$latent = \mathrm{bf16}(\mathrm{fp4}(\mathrm{clamp}(latent / s, -6, 6)) \cdot s)$。absmax 按每 16 个数一组取。latent 是加过 RoPE 的压缩 KV，scale 存成 E4M3 |
| cquant | 说明 | E2M1 格式（1 符号位、2 指数位、1 尾数位），每 16 个通道一个 E4M3 scale（4 指数位、3 尾数位），原地量化再反量化 |
| cwrite | 公式 | $compress\_kv\_cache_{:B,\ 0:T/r} = latent$，latent 是量化后的压缩 KV，start_pos 为 0 |
| cwrite | 说明 | start_pos 为 0，所以从第 0 行起整块写 |
| cat | 公式 | $kv = \mathrm{cat}([kv, compress\_kv], \dim=1)$，kv 是窗口 KV，compress_kv 是压缩 KV。窗口 KV 同时充当 k 和 v |
| cat | 说明 | 窗口在前、压缩在后 |
| cat2 | 公式 | $topk\_idxs = \mathrm{cat}([topk\_idxs, compress\_idxs], \dim=-1)$，topk_idxs 是窗口下标，compress_idxs 是压缩下标 |
| cat2 | 说明 | 压缩下标在 Indexer.forward 的 shift 里已经加上窗口 KV 的长度 T |
| core | 公式 | $o = \dfrac{\sum_{s} \exp(q \cdot kv_s \cdot softmax\_scale - \max) \, kv_s}{\sum_{s} \exp(q \cdot kv_s \cdot softmax\_scale - \max) + \exp(attn\_sink - \max)}$，s 取 topk_idxs 里的槽位（不是位置），kv 同时充当 k 和 v，这里是注意力自己的 softmax_scale，取 head_dim（512）的 -1/2 次方，attn_sink 是可学习参数，$\max$ 是这些槽位上 $q \cdot kv_s \cdot softmax\_scale$ 的最大值 |
| core | 说明 | 单 KV 头，64 个查询头看同一份 KV；attn_sink 是可学习参数 [64]<br>下标为 −1 的槽位分子分母都不贡献 |
| unrope | 公式 | $o_{...,-rd:} = \mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(o_{...,-rd:}.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot freqs\_cis^*))$，o 是注意力输出，freqs_cis 取共轭，rd 是 rope_head_dim，相邻两维配成一对复数 |
| unrope | 说明 | 用 query 所在位置的共轭频率再转输出的末 64 维，去掉 query 的旋转。值里各键位置的旋转变成相对 query 的位置差 |
| grp | 公式 | $o = o.\mathrm{view}(bsz, seqlen, n\_local\_groups, -1)$，o 是注意力输出，n_local_groups 是 8 |
| grp | 说明 | 每 8 个头一组，8 × 512 = 4096。n_local_groups 是本卡的组数，单卡下等于 o_groups 8 |
| woa | 公式 | $wo\_a = wo\_a.weight.\mathrm{view}(n\_local\_groups, o\_lora\_rank, -1)$，wo_a.weight 是 [8192, 4096]，n_local_groups 是 8，o_lora_rank 是 1024 |
| woa | 说明 | wo_a.weight [8192, 4096]，8192 = 8 × 1024 |
| eins | 公式 | $o_{b,s,g,m} = \sum_d o_{b,s,g,d}\, wo\_a_{g,m,d}$，o 是分组后的注意力输出，wo_a 是 wo_a.weight 按组摊开的结果，g 是 o_groups 的组号、m 是 o_lora_rank 维 |
| eins | 说明 | 每组只和本组的权重相乘，等效块对角矩阵 |
| wb | 公式 | $x = \mathrm{flatten}(o)\, wo\_b.weight^\top$，o 是 eins 的输出，wo_b.weight 是输出投影 |
| wb | 说明 | wo_b.weight [5120, 8192] |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 频率 | $\theta_j = \mathrm{base}^{-2j/64}$，j = 0 … 31，base 在压缩比为 0 的层是 rope_theta、其余层是 compress_rope_theta，64 是 rope_head_dim |
| 公式 | YaRN | $\theta_j' = (\theta_j/16)\,\rho_j + \theta_j\,(1 - \rho_j)$，16 是 rope_factor，$\rho_j = \mathrm{clamp}\big((j - low)/(high - low), 0, 1\big)$ 是随 j 递增的斜坡，j 不大时 $\rho_j$ 为 0、保留原频率，j 变大后逐步除以 16，low 是 corrected_dim(beta_fast) 向下取整、high 是 corrected_dim(beta_slow) 向上取整。YaRN 是长上下文的频率缩放，corrected_dim(r) 由 beta_fast/beta_slow 反推出「波长恰好铺满训练长度的那个频率维下标」；只对压缩比大于 0 的层生效 |
| 公式 | query 低秩投影 | $qr = q\_norm.weight \odot \dfrac{x\, wq\_a.weight^\top}{\sqrt{\mathrm{mean}((x\, wq\_a.weight^\top)^2) + q\_norm.eps}}$，再 $q = \mathrm{unflatten}(qr\, wq\_b.weight^\top, (n\_local\_heads, head\_dim))$。x 是本层输入 |
| 公式 | 窗口 KV | $kv = kv\_norm.weight \odot \dfrac{x\, wkv.weight^\top}{\sqrt{\mathrm{mean}((x\, wkv.weight^\top)^2) + kv\_norm.eps}}$，x 是本层输入。后 rope_head_dim 维再乘 freqs_cis |
| 公式 | 压缩 KV 的位置 | 第 j 个 latent 顶替第 $j r$ 到 $(j+1)r - 1$ 个 token，所以它的 RoPE 用位置 $j r$，r 是 compress_ratio，j 是这个压缩组的序号（与「输出低秩投影」里的 g 不是一回事） |
| 公式 | 稀疏注意力 | $o = \dfrac{\sum_{s} \exp(q \cdot kv_s \cdot softmax\_scale - \max)\, kv_s}{\sum_{s} \exp(q \cdot kv_s \cdot softmax\_scale - \max) + \exp(attn\_sink - \max)}$，s 取 topk_idxs 里的槽位（不是位置），kv 同时充当 k 和 v，这里的 softmax_scale 是注意力自己的，取 head_dim（512）的 -1/2 次方；$\max$ 是这些分数的最大值，attn_sink 是可学习参数 |
| 公式 | 输出低秩投影 | $o_{b,s,g,m} = \sum_d o_{b,s,g,d}\, wo\_a_{g,m,d}$，再 $x = \mathrm{flatten}(o)\, wo\_b.weight^\top$。wo_a 是 wo_a.weight 按组摊开的结果，g 是 o_groups 的组号、m 是 o_lora_rank 维，与上一式的 r（compress_ratio）不是一回事 |
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
| ax | 张量 | x | bf16 [B, T, 5120] | kv source 层的注意力输入，与注意力、索引器两张图里的 `ax` 是同一份 | model.py:459 |
| cf32 | 算子 | x.float() | bf16 [B, T, 5120] → fp32 [B, T, 5120] |  | model.py:464 |
| cxf | 张量 | x | fp32 [B, T, 5120] | | model.py:464 |
| projk | 算子 | self.wkv(x) | [B, T, 5120] → [B, T, 512] |  | model.py:465 |
| kv0 | 张量 | kv | fp32 [B, T, 512] | 逐 token 的值投影 | model.py:465 |
| projg | 算子 | self.wgate(x) | [B, T, 5120] → [B, T, 512] |  | model.py:465 |
| csc | 张量 | score | fp32 [B, T, 512] | 逐 token 的门控打分 | model.py:465 |
| pool | 算子 | kv.unflatten(1, (-1, ratio))、score.unflatten(1, (-1, ratio)) 与 (kv * score.softmax(dim=2)).sum(dim=2) | [B, T, 512] → [B, T/r, 512] |  | model.py:473-475 |
| kvg | 张量 | kv | fp32 [B, T/r, 512] | | model.py:475 |
| cnorm | 算子 | self.norm(kv.to(dtype)) | fp32 [B, T/r, 512] → bf16 [B, T/r, 512] |  | model.py:485 |
| lat | 张量 | self.norm(kv.to(dtype)) | bf16 [B, T/r, 512] | 返回值是还没加 RoPE 的压缩潜在量；层 2、8、14 汇自 cnorm，层 20 汇自 cnorm1 | model.py:485 |
| projk1 | 算子 | self.wkv(x) | bf16 [B, T, 5120] → bf16 [B, T, 512] |  | model.py:462 |
| kv1 | 张量 | self.wkv(x) | bf16 [B, T, 512] | 层 20 的值投影，在 bf16 里算，没有 x.float() 与 wgate | model.py:462 |
| cnorm1 | 算子 | self.norm(kv1) | bf16 [B, T, 512] → bf16 [B, T, 512] |  | model.py:462 |

### 连线

- ax → cf32
- ax → cnorm：dtype
- cf32 → cxf
- cxf → projk
- cxf → projg
- projk → kv0
- projg → csc
- kv0 → pool
- csc → pool
- pool → kvg
- kvg → cnorm
- cnorm → lat
- ax → projk1：层 20 的输入（层 20 ratio 为 1）
- projk1 → kv1
- kv1 → cnorm1
- cnorm1 → lat：层 20 的返回值

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| cf32 | 公式 | $x = x.\mathrm{float}()$，x 是送进压缩器的隐状态 |
| cf32 | 说明 | 池化与门控都在 fp32 里算；只有 compress_ratio 大于 1 的 kv source 层（2、8、14）走这一支；层 20 的 ratio 为 1，直接走 projk1 那一支，不做 x.float() |
| projk | 公式 | $kv = x\, wkv.weight^\top$，x 是本层输入，wkv.weight 是值投影的权重 |
| projk | 说明 | wkv.weight [512, 5120]，压缩比大于 1 时是 fp32 |
| projg | 公式 | $score = x\, wgate.weight^\top$，x 是本层输入，wgate.weight 是门控投影的权重 |
| projg | 说明 | wgate.weight [512, 5120]，压缩比大于 1 时是 fp32 |
| cnorm1 | 公式 | $latent = norm(kv)$，kv 是 projk1 的输出（层 20 的 wkv，在 bf16 里算）；层 20 的 ratio 为 1，没有池化、没有门控 |
| cnorm1 | 说明 | layers.20.attn.compressor.norm.weight [512]；层 20 的 wkv.weight 是 bf16（compress_ratio 不大于 1） |
| pool | 公式 | 按 ratio 分组后 $kv = \sum_t \mathrm{softmax}(score)_t \, kv_t$，softmax 在这一组的 ratio 个 token 上，kv 与 score 是 projk 与 projg 两个投影的输出 |
| pool | 说明 | 把相邻 r 个 token 当成一组，组内用 score 的 softmax 做加权求和 |
| cnorm | 公式 | $kv = norm.weight \odot \dfrac{kv.\mathrm{to}(dtype)}{\sqrt{\mathrm{mean}(kv^2) + norm.eps}}$，kv 是池化后的张量，norm 是 compressor.norm（RMSNorm），mean 在最后一维，eps 取 1e-20，to(dtype) 转回 bf16 |
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
| ax | 张量 | x | bf16 [B, T, 5120] | kv source 层的注意力输入，与注意力、压缩器两张图里的 `ax` 是同一份 | model.py:533 |
| qr | 张量 | qr | bf16 [B, T, 1280] | 注意力那边算出的低秩 query，索引器直接拿它当输入 | model.py:527 |
| lat | 张量 | latent | bf16 [B, T/r, 512] | 压缩器的输出，还没加 RoPE；只有 2、8、14、20 这四个 kv source 层是实际张量，其余 index source 层（24、28、32、36）不建键、传进来的是 None，图里这一段整个跳过，只从 shared_attn.index_k 读键 | model.py:527、537、500 |
| fcall | 张量 | self.freqs_cis | 复数 fp32 [max_seq_len, 32] | 注意力那边传进来的整张频率表，与注意力图里的 fcall 是同一份 | model.py:733-734 |
| fpos | 算子 | self.freqs_cis[: seqlen - seqlen % ratio : ratio] | [max_seq_len, 32] → [T/r, 32] |  | model.py:539-543 |
| fq | 张量 | freqs | 复数 fp32 [T/r, 32] | | model.py:540-543 |
| wk_p | 算子 | self.wk(latent) | bf16 [B, T/r, 512] → bf16 [B, T/r, 128] |  | model.py:544 |
| wka | 张量 | self.wk(latent) | bf16 [B, T/r, 128] | wk 的输出，源码里嵌在下一行的调用里，没有单独的变量名 | model.py:544 |
| kw | 算子 | self.k_norm(wka) | bf16 [B, T/r, 128] → bf16 [B, T/r, 128] |  | model.py:544 |
| k0 | 张量 | k | bf16 [B, T/r, 128] | | model.py:544 |
| krope | 算子 | apply_rotary_emb(k[..., -rd:], freqs) | bf16 [B, T/r, 128] → bf16 [B, T/r, 128] |  | model.py:545 |
| kr | 张量 | k | bf16 [B, T/r, 128] | | model.py:545 |
| kq | 算子 | fp4_act_quant(k, fp4_block_size, True) | bf16 [B, T/r, 128] → bf16 [B, T/r, 128] |  | model.py:546 |
| kf | 张量 | k | bf16 [B, T/r, 128] | 存的是 fp4 量化后再反量化的值 | model.py:546 |
| kwrite | 算子 | self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k | bf16 [B, T/r, 128] → [max_batch_size, max_seq_len / r, 128] |  | model.py:547-548 |
| kcache | 缓存 | k_cache | [max_batch_size, max_seq_len / r, 128] | 2、8、14、20 各有一份，按 max_batch_size 分配；写完发布给 shared_attn，它之后到下一个 kv source 层之前的索引器读的是这一份 | model.py:520-525 |
| shar_cand | 缓存 | shared_attn.candidates | 层 20 选出的候选块掩码 bool [B, T, T/r] | 层 20 写，层 24、28、32、36 读；是 Transformer 图 shar（shared_attn）里的 candidates 一项 | model.py:569-575 |
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
| lens | 算子 | (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1) | [B, T, 5120] → int64 [T, 1] |  | model.py:563-564 |
| lv | 张量 | compress_lens | int64 [T, 1] | | model.py:563-564 |
| imask | 算子 | index_score.masked_fill_(torch.arange(seqlen // ratio) >= compress_lens, -torch.inf) | bf16 [B, T, T/r] → bf16 [B, T, T/r] |  | model.py:565 |
| iscm | 张量 | index_score | bf16 [B, T, T/r] | 掩掉看不到的位置之后的打分 | model.py:565 |
| csel | 算子 | select_candidate_blocks(index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size) | [B, T, T/r] → bool [B, T, T/r] |  | model.py:569-571 |
| cmask | 张量 | shared_attn.candidates | bool [B, T, T/r] | 层 20 选出的候选块，已按块展开成逐个位置的 True/False | model.py:569-571 |
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
- fcall → fpos
- ax → fpos：形状（seqlen）
- fpos → fq
- lat → wk_p
- wk_p → wka
- wka → kw
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
- iscm → csel
- lv → csel：compress_lens、块数上限与块大小
- csel → cmask
- cmask → shar_cand
- shar_cand → cmf
- iscm → cmf
- cmf → cmfd
- cmfd → topk
- topk → isel
- isel → shift
- lv → shift
- shift → cidxs

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| fpos | 公式 | $freqs = freqs\_cis[:seqlen - seqlen \% ratio: ratio]$，第 j 组取位置 $j \times ratio$ |
| fpos | 说明 | 取压缩位置的频率，第 j 组取位置 j × r |
| wk_p | 公式 | $wk(latent) = latent\, wk.weight^\top$，latent 是压缩器的输出，wk.weight 是索引键投影的权重 |
| wk_p | 说明 | wk.weight [128, 512]，bf16 |
| kw | 公式 | $k = k\_norm.weight \odot \dfrac{wk(latent)}{\sqrt{\mathrm{mean}(wk(latent)^2) + k\_norm.eps}}$，wk(latent) 是上一节点的输出，mean 在最后一维 |
| kw | 说明 | k_norm.weight [128]，k_norm 是 RMSNorm，k_norm.eps 取 1e-20 |
| krope | 公式 | $k_{...,-rd:} = \mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(k_{...,-rd:}.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot freqs))$，k 是索引键，freqs 是压缩位置的频率，rd 是 rope_head_dim，相邻两维配成一对复数 |
| krope | 说明 | 128 维里的最后 64 维加 RoPE |
| kq | 公式 | $s = \mathrm{fast\_round\_scale}(\max(\mathrm{absmax}(k), 6 \cdot 2^{-126}), 1/6)$，$k = \mathrm{bf16}(\mathrm{fp4}(\mathrm{clamp}(k / s, -6, 6)) \cdot s)$。absmax 按每 32 个数一组取。k 是加过 RoPE 的索引键 |
| kq | 说明 | E2M1 格式，每 32 个通道一个 E8M0 scale，原地量化再反量化 |
| kwrite | 公式 | $k\_cache_{:B,\ 0:T/r} = k$，k 是量化后的索引键，start_pos 为 0 |
| kwrite | 说明 | start_pos 为 0，从第 0 行起整块写；写完把这份缓存发布到 shared_attn.index_k，后面的层直接读 |
| qop | 公式 | $q = \mathrm{unflatten}(qr\, wq\_b.weight^\top, (n\_local\_heads, index\_head\_dim))$，qr 是注意力低秩 query，wq_b.weight 是索引器的查询投影 |
| qop | 说明 | wq_b.weight [4096, 1280]，32 × 128 = 4096 |
| qrope | 公式 | $q_{...,-rd:} = \mathrm{flatten}(\mathrm{view\_as\_real}(\mathrm{view\_as\_complex}(q_{...,-rd:}.\mathrm{float}().\mathrm{unflatten}(-1,(-1,2))) \odot freqs\_cis))$，q 是索引 query，freqs_cis 从 start_pos 取到 end_pos，rd 是 rope_head_dim，相邻两维配成一对复数 |
| qrope | 说明 | 乘从 start_pos 取到 end_pos 的 freqs_cis |
| iq_q | 公式 | $s = \mathrm{fast\_round\_scale}(\max(\mathrm{absmax}(q), 6 \cdot 2^{-126}), 1/6)$，$q = \mathrm{bf16}(\mathrm{fp4}(\mathrm{clamp}(q / s, -6, 6)) \cdot s)$。absmax 按每 32 个数一组取。q 是加过 RoPE 的索引 query |
| iq_q | 说明 | 索引器的 query 也量化成 fp4 |
| wproj | 公式 | $weights = (x\, weights\_proj.weight^\top) \cdot softmax\_scale \cdot n\_heads^{-0.5}$，x 是本层输入，这里是索引器自己的 softmax_scale，取 index_head_dim（128）的 -1/2 次方，与注意力那次不是同一个量，n_heads 是 index_n_heads |
| wproj | 说明 | weights_proj.weight [32, 5120] |
| idot | 公式 | $index\_score_{b,s,h,t} = \sum_d q_{b,s,h,d}\, index\_k_{b,t,d}$，q 是索引 query，index_k 是索引键，h 是 einsum 里的头维，d 是 index_head_dim |
| idot | 说明 | 每个索引头对每个压缩位置记一个点积 |
| red | 公式 | $index\_score = \sum_i \mathrm{relu}(s_i) \cdot weights_i$，s 是上一行按头的点积，i 是 head 维，weights 是 weights_proj 给出的系数 |
| red | 说明 | 先过 ReLU，再按头用 weights 加权求和 |
| lens | 公式 | $compress\_lens_s = \lfloor (s+1) / ratio \rfloor$，s 从 0 到 seqlen−1 是 query 的位置下标（不是同图 red 里的头维 i），ratio 是 compress_ratio |
| lens | 说明 | 第 s 个 query 能看到 ⌊(s+1)/r⌋ 个压缩条目 |
| imask | 公式 | $index\_score = index\_score.\mathrm{masked\_fill}(\mathrm{arange}(seqlen /\! / ratio) \ge compress\_lens, -\infty)$，compress_lens 是每个 query 能看到多少个压缩位置；够不到的那几个位置写成 $-\infty$ |
| imask | 说明 | 把本次还看不到的压缩条目在 index_score 上原地写成 −inf |
| csel | 公式 | $candidates = select\_candidate\_blocks(index\_score, compress\_lens, candidate\_topk\_blocks, candidate\_block\_size)$：先把打分按块切开、取块内最大，把最新那一块钉成 $+\infty$，再取前 candidate_topk_blocks 块，最后按块展开回逐个位置 |
| csel | 说明 | 只有层 20 是 candidate source（config 的 candidate_source_layer），由它写出这份候选块掩码 |
| cmf | 公式 | $index\_score = index\_score.\mathrm{masked\_fill}(\lnot candidates, -\infty)$，候选块外的位置置 $-\infty$ |
| cmf | 说明 | 层 24、28、32、36 是 uses_candidates，用层 20 写出的候选块把自己看不到的位置排掉，再取 top-512 |
| topk | 公式 | $idxs = \mathrm{sort}(\mathrm{topk}(index\_score, topk))$，topk 取 min(512, T/ratio)，sort 按位置排回 |
| topk | 说明 | topk = min(512, T/r)，前提里 T 足够长所以就是 512；取分数最高的 512 个，再按位置顺序排回来 |
| shift | 公式 | $idxs = \mathrm{where}(idxs < compress\_lens,\ idxs + offset,\ -1)$，idxs 是 topk 的结果，offset 是窗口 KV 的长度 |
| shift | 说明 | 不小于可达数的置 −1，其余加上 offset 变成拼起来的 KV 里的下标。offset 由注意力传入，是这次窗口 KV 的长度 T |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 索引打分 | $index\_score = \sum_i \mathrm{relu}(q_i \cdot index\_k) \cdot weights_i$，weights 来自 $weights = (x\, weights\_proj.weight^\top) \cdot softmax\_scale \cdot n\_heads^{-0.5}$。这里是索引器自己的 softmax_scale，取 index_head_dim（128）的 -1/2 次方，与注意力那次不是同一个量，n_heads 是 index_n_heads，i 是头维 |
| 公式 | 可达数 | 第 s 个 query 能看到 $\lfloor (s+1)/r \rfloor$ 个压缩条目，s 是 query 的位置，r 是 compress_ratio；还看不到的那些置 $-\infty$ |
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
| im | 张量 | image_mask | bool [B, T] | MoE.forward 的参数，由 Block 传进来；图像 span 的位置为 True，与 Transformer 图的 im 是同一个量 | model.py:889 |
| mmask | 算子 | image_mask.flatten() | bool [B, T] → bool [B×T] |  | model.py:892 |
| mfm | 张量 | image_mask.flatten() | bool [B×T] | image_mask 是这一层前向收进来的那个张量，压平到按 token 的位置；图像 span 的位置为 True | model.py:892 |
| gate_op | 算子 | self.gate(x, image_mask) | [B×T, 5120] 与 bool [B×T] → 权重 [B×T, 6] 与专家下标 [B×T, 6] |  | model.py:892 |
| gw | 张量 | weights | fp32 [B×T, 6] | | model.py:892 |
| gi | 张量 | indices | int64 [B×T, 6] | | model.py:892 |
| bc | 算子 | torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist() | int64 [B×T, 6] → list，长度 384 |  | model.py:894 |
| counts | 张量 | counts | list，长度 384 | | model.py:894 |
| y0 | 算子 | torch.zeros_like(x, dtype=torch.float32) | bf16 [B×T, 5120] → fp32 [B×T, 5120] |  | model.py:893 |
| y | 张量 | y | fp32 [B×T, 5120] | | model.py:893 |
| mpick | 算子 | torch.where(indices == i) | int64 [B×T, 6] → int64 [n] × 2 |  | model.py:899 |
| pidx | 张量 | idx, top | int64 [n] 与 int64 [n] | n 是分到第 i 个专家的 token 数 | model.py:899 |
| mexp | 算子 | expert(x[idx], weights[idx, top, None]) | bf16 [n, 5120] → bf16 [n, 5120] |  | model.py:900 |
| eout | 张量 | expert(x[idx], weights[idx, top, None]) | bf16 [n, 5120] | | model.py:900 |
| acc | 算子 | y[idx] += expert(x[idx], weights[idx, top, None]) | fp32 [B×T, 5120] 与 bf16 [n, 5120] → fp32 [B×T, 5120] |  | model.py:900 |
| y2 | 张量 | y | fp32 [B×T, 5120] | | model.py:900 |
| shared | 算子 | self.shared_experts(x) | bf16 [B×T, 5120] → bf16 [B×T, 5120] |  | model.py:903 |
| sv | 张量 | self.shared_experts(x) | bf16 [B×T, 5120] | | model.py:903 |
| madd | 算子 | y += self.shared_experts(x) | fp32 [B×T, 5120] 与 bf16 [B×T, 5120] → fp32 [B×T, 5120] |  | model.py:903 |
| y3 | 张量 | y | fp32 [B×T, 5120] | | model.py:903 |
| mback | 算子 | y.type_as(x).view(shape) | [B×T, 5120] → bf16 [B, T, 5120] |  | model.py:904 |
| mout | 张量 | y | bf16 [B, T, 5120] | | model.py:904 |

### 连线

- mx → flat
- flat → mxf
- mxf → gate_op
- im → mmask
- mmask → mfm
- mfm → gate_op：image_mask 压平后的那一维
- gate_op → gw
- gate_op → gi
- mxf → y0：形状与 device
- y0 → y
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
- madd → y3
- y3 → mback
- mx → mback：dtype 与 shape
- mback → mout
- gate_op 点开 → Gate.forward
- mexp 点开 → Expert.forward

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| flat | 公式 | $x = x.\mathrm{view}(-1, dim)$，x 是前馈输入，dim 是 5120 |
| flat | 说明 | 把批次与序列两维拉平，MoE 按 token 处理 |
| mmask | 公式 | $image\_mask = image\_mask.\mathrm{flatten}()$，进 MoE 时是 [B, T]，压成按 token 的一维交给 gate_op |
| mmask | 说明 | 这个 image_mask 就是这一层前向的参数，Block 图里的 im；图像 span 的位置为 True |
| gate_op | 公式 | $weights, indices = gate(x, image\_mask)$，x 是拉平后的隐状态，image_mask 是压平后的那一维 |
| gate_op | 说明 | Gate：打分、加偏置选专家、在 top-6 内归一化、乘 route_scale；图像 span 的位置换一份偏置，见 Gate 图 |
| y0 | 公式 | $y = \mathrm{zeros\_like}(x)$，x 是拉平后的隐状态，dtype 取 fp32 |
| y0 | 说明 | 累加器，路由专家的结果都往它上面加 |
| bc | 公式 | $counts = \mathrm{bincount}(indices.\mathrm{flatten}(), minlength=n\_routed\_experts).\mathrm{tolist}()$，indices 是专家下标，n_routed_experts 是 384 |
| bc | 说明 | counts 是 Python list，长度 384。counts[i] 为 0 的专家后面直接跳过 |
| mpick | 公式 | $idx, top = \mathrm{where}(indices = i)$，indices 是每个 token 的 top-6 专家号，i 是当前专家 |
| mpick | 说明 | 挑出分给第 i 个专家的 token，384 个专家各跑一遍；counts[i] 为 0 的专家整个跳过 |
| mexp | 公式 | $eout = expert(x[idx], weights[idx, top, None])$，x 是拉平后的隐状态，weights 是路由权重 |
| mexp | 说明 | 这一步只处理分给当前专家 i 的 token。一个 token 可以出现在多个专家的 top-6 里，路由权重按它在 top-6 里的位置取 |
| acc | 公式 | $y[idx] = y[idx] + eout$，y 是累加器，eout 是 expert(x[idx], weights[idx, top, None]) 的返回值，idx 是分到这个专家的 token |
| acc | 说明 | 一个 token 落在 top-6 的哪几个专家上就加几次 |
| shared | 公式 | $sv = shared\_experts(x)$，x 是拉平后的隐状态。这次调用不传 weights |
| shared | 说明 | shared_experts 是一个 Expert，每个 token 都过。不传 weights，所以不走 Expert.forward 里的 emw。三个权重按主干精度 fp8 存储 |
| madd | 公式 | $y = y + sv$，y 是路由专家已经累加的结果，sv 是共享专家的输出 |
| madd | 说明 | 路由专家的和再加上共享专家的输出 |
| mback | 公式 | $y = y.\mathrm{type\_as}(x).\mathrm{view}(shape)$，x 是拉平后的隐状态，dtype 与进入时相同，shape 是拉平前记下的 [B, T, 5120] |
| mback | 说明 | 转回输入的 dtype 与形状 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 前馈 | $y[idx] = y[idx] + eout$，然后 $y = y + sv$。eout 是路由专家的输出，sv 是共享专家的输出。共享专家不传 weights |
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
| gwt | 张量 | self.weight | fp32 [384, 5120] | 专家打分矩阵，gate.weight | model.py:805 |
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
| lg | 公式 | $scores = \mathrm{linear}(x, weight) / gate\_temp$，x 是拉平后的隐状态，weight 是 gate.weight，gate_temp 是 1.0 |
| lg | 说明 | weight [384, 5120]，打分在 fp32 里算<br>gate_temp 取默认值 1.0，config.json 里没有这一项 |
| act | 公式 | $scores = \sqrt{\mathrm{softplus}(scores)}$，$\mathrm{softplus}(scores) = \ln(1+e^{scores})$ |
| act | 说明 | score_func 是 sqrtsoftplus：先 softplus 再开方 |
| bsel | 公式 | $bias = \mathrm{where}(image\_mask.\mathrm{unsqueeze}(-1), bias\_vl, bias)$，image_mask 由 [B×T] 扩成 [B×T, 1] 再广播到 384；每个 token 整行要么全取 bias_vl、要么全取 bias |
| bsel | 说明 | 两份偏置都只在挑专家时用，不改权重的值 |
| gsel | 公式 | $indices = \mathrm{topk}(scores + bias, topk)$，bias 是 bsel 选出来、存在 bbias 里的那一份，topk 是 n_activated_experts |
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
| 公式 | 选专家 | $bias = \mathrm{where}(image\_mask.\mathrm{unsqueeze}(-1), bias\_vl, bias)$，再 $indices = \mathrm{topk}(scores + bias, topk)$，topk 是 6。image_mask 由 [B×T] 广播到 384 维，每个 token 整行取同一份偏置；偏置只加在挑选上，weights 从原来的 scores 里 gather |
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
- ex → w2：dtype
- w2 → eout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| g1 | 公式 | $gate = x\, w1.weight^\top$，x 是输入，结果转成 fp32 |
| g1 | 说明 | 路由专家的 w1.weight 存成 [2304, 2560] 的 fp4：末维是逻辑宽度 5120 的一半，因为 fp4 每个字节装两个值；每 32 个通道一个 E8M0 scale。结果转成 fp32，夹取在后面的 cl2 |
| u1 | 公式 | $up = x\, w3.weight^\top$，x 是输入，结果转成 fp32 |
| u1 | 说明 | w3.weight 与 w1 同一种存法 |
| cl | 公式 | $up = \mathrm{clamp}(up, -swiglu\_limit, swiglu\_limit)$，swiglu_limit 是 10 |
| cl | 说明 | up 两头都夹；swiglu_limit 是 10.0<br>夹是为了让后面的 fp8、fp4 激活不越界 |
| cl2 | 公式 | $gate = \mathrm{clamp}(gate, \max = swiglu\_limit)$，swiglu_limit 是 10 |
| cl2 | 说明 | gate 只夹上界 |
| silu | 公式 | $x = \mathrm{silu}(gate) \odot up$，gate 与 up 是夹过界的两路 |
| silu | 说明 | SiLU 的结果逐元素乘 up |
| emw | 公式 | $x = weights \odot x$，weights 是路由权重 |
| emw | 说明 | 只在传入 weights 时执行。这张图是路由专家，weights 来自 top-6 里该专家的权重 |
| w2 | 公式 | $x = x.\mathrm{to}(dtype)\, w2.weight^\top$，dtype 是进入 Expert 时 x 的 dtype（bf16），x 是乘过路由权重的隐状态 |
| w2 | 说明 | 路由专家的 w2.weight 存成 [5120, 1152] 的 fp4（末维 1152 = 2304 / 2，同样是一格装两个值）；共享专家是 [5120, 2304] 的 fp8 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | SwiGLU | $gate = x\, w1.weight^\top$，$up = x\, w3.weight^\top$，夹到 swiglu_limit 以后 $x = \mathrm{silu}(gate) \odot up$，路由专家再 $x = weights \odot x$，然后 $x = x.\mathrm{to}(dtype)\, w2.weight^\top$。x 是输入，weights 是路由权重，dtype 是进入时 x 的 dtype |
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
| nwkv | 算子 | self.wkv(ev) | [B, T, 6144] → [B, T, 25600] |  | model.py:353 |
| nkv | 张量 | kv | bf16 [B, T, 25600] | | model.py:353 |
| spl | 算子 | kv.split([self.hc_mult * self.dim, self.dim], dim=-1) | [B, T, 25600] → [B, T, 20480] 与 [B, T, 5120] |  | model.py:354 |
| key0 | 张量 | key | bf16 [B, T, 20480] | 20480 = 4 × 5120，每份残差流一个键 | model.py:354 |
| value | 张量 | value | bf16 [B, T, 5120] | 4 份残差流共用同一个值 | model.py:354 |
| ekey | 算子 | key.float().unflatten(-1, (self.hc_mult, self.dim)) | [B, T, 20480] → fp32 [B, T, 4, 5120] |  | model.py:355 |
| key | 张量 | key | fp32 [B, T, 4, 5120] | | model.py:355 |
| eqw | 张量 | self.q_weight | fp32 [4, 5120] | 查询侧的缩放参数，可学习，代码里只以乘积形式出现 | model.py:347 |
| ekw | 张量 | self.k_weight | fp32 [4, 5120] | 键侧的缩放参数，可学习，代码里只以乘积形式出现 | model.py:348 |
| ewm | 算子 | self.q_weight.float() * self.k_weight.float() | [4, 5120] → fp32 [4, 5120] |  | model.py:356 |
| ew | 张量 | weight | fp32 [4, 5120] | | model.py:356 |
| nhf | 算子 | x.float() | bf16 [B, T, 4, 5120] → fp32 [B, T, 4, 5120] |  | model.py:357 |
| nh | 张量 | h | fp32 [B, T, 4, 5120] | | model.py:357 |
| rstd | 算子 | torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps) | [B, T, 4, 5120] → fp32 [B, T, 4] |  | model.py:359 |
| nrs | 张量 | rstd | fp32 [B, T, 4] | | model.py:359 |
| ndot | 算子 | (h * weight * key).sum(-1) * rstd * self.dim**-0.5 | [B, T, 4, 5120] → [B, T, 4] |  | model.py:360 |
| d | 张量 | dot | fp32 [B, T, 4] | | model.py:360 |
| ngate | 算子 | torch.sigmoid(<br>torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)) | fp32 [B, T, 4] → fp32 [B, T, 4] |  | model.py:362 |
| g | 张量 | gate | fp32 [B, T, 4] | | model.py:362 |
| emv | 张量 | token_mask | bool [B, T] | Engram.forward 的参数，图像 span 的位置为 False；与 Transformer 图的 emv 是同一个量 | model.py:350 |
| gmask | 算子 | gate.masked_fill(~token_mask.unsqueeze(-1), 0) | fp32 [B, T, 4] 与 bool [B, T] → fp32 [B, T, 4] |  | model.py:363-364 |
| gm | 张量 | gate | fp32 [B, T, 4] | 图像 span 的位置置 0，这些位置不加 Engram 的写入 | model.py:364 |
| vf | 算子 | value.float().unsqueeze(-2) | [B, T, 5120] → fp32 [B, T, 1, 5120] |  | model.py:365 |
| v | 张量 | value.float().unsqueeze(-2) | fp32 [B, T, 1, 5120] | | model.py:365 |
| nadd | 算子 | h + gate.unsqueeze(-1) * v | fp32 [B, T, 4, 5120] → fp32 [B, T, 4, 5120] |  | model.py:365 |
| sum | 张量 | h + gate.unsqueeze(-1) * value.float().unsqueeze(-2) | fp32 [B, T, 4, 5120] | | model.py:365 |
| ncast | 算子 | .to(x.dtype) | fp32 [B, T, 4, 5120] → bf16 [B, T, 4, 5120] |  | model.py:365 |
| nout | 张量 | (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype) | bf16 [B, T, 4, 5120] | | model.py:365 |

### 连线

- nx → nhf
- nhf → nh
- hash_ids → nemb
- nemb → ev
- nemb 点开 → ParallelEngramEmbedding.forward
- ev → nwkv
- nwkv → nkv
- nkv → spl
- spl → key0
- spl → value
- key0 → ekey
- ekey → key
- eqw → ewm
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
- value → vf
- vf → v
- gm → nadd
- v → nadd
- nh → nadd
- nadd → sum
- sum → ncast
- nx → ncast：dtype
- ncast → nout

### 折叠

| id | 名字 | 内容 |
|---|---|---|
| nemb | 公式 | $ev = embed(hash\_ids).\mathrm{flatten}(-2)$，hash_ids 是哈希行号，embed 是 ParallelEngramEmbedding，查表与反量化见点开的那张图，外层 flatten 把 24 与 256 合成 6144 |
| nemb | 说明 | ParallelEngramEmbedding：表按行分片，查出 fp8 的行再用 scale 反量化成 bf16<br>24 × 256 = 6144；层 1 的表有 384006168 行，层 14 有 384016682 行 |
| nwkv | 公式 | $kv = ev\, wkv.weight^\top$，ev 是查表并 flatten 后的向量，wkv.weight 是 [25600, 6144] |
| nwkv | 说明 | wkv.weight [25600, 6144]；输出切成 key 与 value 两段 |
| spl | 公式 | $key, value = \mathrm{split}(kv, [hc\_mult \cdot dim, dim])$，kv 是 wkv 的输出，hc_mult 是 4，dim 是 5120 |
| ekey | 公式 | $key = key.\mathrm{float}().\mathrm{unflatten}(-1, (hc\_mult, dim))$，key 是切出来的前 20480 维，hc_mult 是 4，dim 是 5120 |
| ekey | 说明 | key 转成 fp32，再拆成 hc_mult 份 |
| ewm | 公式 | $weight = q\_weight \odot k\_weight$，两个参数都是 [hc_mult, dim] |
| ewm | 说明 | q_weight 与 k_weight 都是 [4, 5120] 的参数，代码里只以乘积形式出现 |
| nhf | 公式 | $h = x.\mathrm{float}()$，x 是本层残差流 |
| nhf | 说明 | 归一化与点积都在 fp32 里算 |
| rstd | 公式 | $rstd = \mathrm{rsqrt}(\mathrm{mean}(h^2) + eps) \cdot \mathrm{rsqrt}(\mathrm{mean}(key^2) + eps)$，h 是 x 转成 fp32 的残差，key 是查表得到的键，eps 是 norm_eps，mean 在最后一维 |
| rstd | 说明 | 两边各按最后一维求均方，逐 token、逐份归一化 |
| ndot | 公式 | $dot = \mathrm{sum}(h \odot weight \odot key, -1) \cdot rstd \cdot dim^{-0.5}$，h、weight、key、rstd 是送进来的张量，dim 是 5120 |
| ndot | 说明 | 归一化点积，再乘 dim 的 −1/2 次方 |
| ngate | 公式 | $gate = \mathrm{sigmoid}(\mathrm{copysign}(\sqrt{\max(\lvert dot \rvert, clamp\_value)}, dot))$，dot 是归一化点积，clamp_value 是 1e-6 |
| ngate | 说明 | 先带符号开方再进 sigmoid，与训练 kernel 一致 |
| gmask | 公式 | $gate = gate.\mathrm{masked\_fill}(\lnot token\_mask.\mathrm{unsqueeze}(-1), 0)$，token_mask 为 False 的位置把门控置 0 |
| gmask | 说明 | 图像 span 的位置不做门控写入，残差流原样穿过。token_mask 就是这一层的 engram_mask |
| vf | 公式 | $v = value.\mathrm{float}().\mathrm{unsqueeze}(-2)$，value 是查表得到的值，补一维好按份广播 |
| nadd | 公式 | $h = h + \mathrm{unsqueeze}(gate, -1) \odot v$，h 是残差流，gate 是 gmask 之后的门控、逐份，v 是 value 转成 fp32 并在份这一维扩成 1 的结果 |
| nadd | 说明 | 门控乘值再加回残差流。折叠里的 gate 是 gmask 之后的门控，h 是 Engram 入口的残差流 |
| ncast | 公式 | $h = h.\mathrm{to}(x.dtype)$，x 是进入 Engram 时的残差 |
| ncast | 说明 | 转回输入时的 dtype |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 归一化点积 | $rstd = \mathrm{rsqrt}(\mathrm{mean}(h^2)+eps)\cdot\mathrm{rsqrt}(\mathrm{mean}(key^2)+eps)$，$dot = \mathrm{sum}(h \odot weight \odot key, -1) \cdot rstd \cdot dim^{-0.5}$。h 是 x 转成 fp32 的残差，weight 是 q_weight 与 k_weight 的乘积，key 是查表得到的键，eps 是 norm_eps，dim 是 5120 |
| 公式 | 门控 | $gate = \mathrm{sigmoid}(\mathrm{copysign}(\sqrt{\max(\lvert dot \rvert, clamp\_value)}, dot))$，dot 是归一化点积，clamp_value 是 1e-6 |
| 公式 | 写入 | $h = h + \mathrm{unsqueeze}(gate, -1) \odot value.\mathrm{float}().\mathrm{unsqueeze}(-2)$，h 是残差流，gate 是 [B, T, 4]，value 是 [B, T, 5120]，按份广播 |
| 配置 | dim | 5120 |
| 配置 | hc_mult | 4 |
| 配置 | engram_max_ngram_size | 4 |
| 配置 | engram_n_heads | 8 |
| 配置 | engram_head_dim | 256 |
| 配置 | engram_layer_ids | [1, 14] |
| 配置 | engram_num_embeddings | [384006168, 384016682] |
| 配置 | norm_eps | 1e-20 |

## ParallelEngramEmbedding.forward（model.py:312-325）

### 节点

| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |
|---|---|---|---|---|---|
| hash_ids | 张量 | indices | int64 [B, T, 24] | 参数 indices，就是 Engram 图里的 hash_ids | model.py:312 |
| peu | 张量 | self.weight | fp8_e4m3 [384016682, 256] | n-gram 哈希表，fp8 存，按 rank 沿行切分，单卡即整张 | model.py:309 |
| pes | 张量 | self.scale | e8m0 [384016682, 8] | 每 32 行一组一个 scale（scale_dtype 是 float8_e8m0fnu），8 = 256 / 32 | model.py:310 |
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
| peuq | 公式 | $values = \mathrm{embedding}(local\_indices, weight)$，local_indices 是 hash_ids 减去本卡起始行，weight 是 fp8 表 |
| peuq | 说明 | weight [384016682, 256]，按 rank 沿行切分，单卡即整张；层 1 的表有 384006168 行，层 14 有 384016682 行 |
| pesq | 公式 | $scales = \mathrm{embedding}(local\_indices, scale)$，scale 是每 32 行一组的缩放 |
| pedq | 公式 | $values = \mathrm{flatten}(values.\mathrm{float}().\mathrm{unflatten}(-1, (-1, 32)) \odot scales.\mathrm{float}().\mathrm{unsqueeze}(-1)).\mathrm{to}(bf16)$，32 是 fp8_block_size，256 / 32 = 8 |
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
| tmap | 张量 | self.token_map | int64，长度 99092 | 压缩词表，把原始 token id 映到压缩词表的行号；缓存，按 engram_compressed_vocab_size 建 | engram.py:154 |
| cmap | 算子 | self.token_map[input_ids] | int64 [B, T] → int64 [B, T] |  | engram.py:164 |
| cids | 张量 | compressed | int64 [B, T] | | engram.py:164 |
| emv | 张量 | token_mask | bool [B, T] | 调用的参数，图像 span 的位置为 False；与 Transformer 图的 emv 是同一个量 | engram.py:160 |
| cgm | 算子 | torch.where(token_mask, compressed, self.DEAD) | int64 [B, T] 与 bool [B, T] → int64 [B, T] |  | engram.py:166 |
| cdd | 张量 | compressed | int64 [B, T] | 重写后的压缩 id，图像 span 的位置记成 DEAD（−1） | engram.py:166 |
| wr | 算子 | self.cache[:batch, start_pos : start_pos + seqlen] = compressed | int64 [B, T] → [max_batch_size, max_seq_len] |  | engram.py:167 |
| cache | 缓存 | cache | int64 [max_batch_size, max_seq_len] | 按 max_batch_size 分配，取代码默认值 4；存的是压缩后的 id，不是原始 token id<br>decode 时按 start_pos 往后接 | engram.py:155-157 |
| pos | 算子 | torch.arange(start_pos, start_pos + seqlen).expand(batch, seqlen) | int64 [B, T] → int64 [B, T] |  | engram.py:169 |
| p | 张量 | positions | int64 [B, T] | | engram.py:169 |
| blk0 | 算子 | torch.zeros_like(positions, dtype=torch.bool) | int64 [B, T] → bool [B, T] |  | engram.py:170 |
| blk | 张量 | blocked | bool [B, T] | | engram.py:170 |
| gath | 算子 | self.cache[:batch].gather(1, (positions - shift).clamp_min(0)) | [max_batch_size, max_seq_len] → int64 [B, T] |  | engram.py:172 |
| src | 张量 | source | int64 [B, T] | | engram.py:172 |
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
| cmap | 公式 | $compressed = token\_map[input\_ids]$，input_ids 是词 id，token_map 是压缩词表的行号 |
| cmap | 说明 | token_map 把每个 token id 映到压缩词表的 id，大小写与空白差异都归一到一个 id<br>压缩词表 99092 项，与 config 的 engram_compressed_vocab_size 一致，所有哈希乘数都由它推出 |
| cgm | 公式 | $compressed = \mathrm{where}(token\_mask, compressed, DEAD)$，token_mask 为 False 的位置（图像 span）盖成 DEAD |
| cgm | 说明 | DEAD 是 −1。图像 span 的位置不参与任何 n-gram，后面的回看遇到它就断掉 |
| wr | 公式 | $cache_{:B,\ 0:T} = compressed$，compressed 是压成 DEAD 之后的 id，start_pos 为 0 |
| wr | 说明 | start_pos 为 0，所以从第 0 列起整块写；图像 span 的 DEAD 也写进 cache，供后面的回看读到 |
| pos | 公式 | $positions = \mathrm{arange}(start\_pos, start\_pos + seqlen).\mathrm{expand}(batch, seqlen)$，start_pos 为 0 |
| pos | 说明 | start_pos 到 start_pos + seqlen 的位置，扩成 batch 行 |
| blk0 | 公式 | $blocked = \mathrm{zeros\_like}(positions)$，dtype 是 bool |
| blk0 | 说明 | 标记哪些位置已经不能往前看 |
| gath | 公式 | $source = \mathrm{gather}(cache_{:B}, (positions - shift).\mathrm{clamp}(min=0))$，shift 是往前看的步数 |
| gath | 说明 | 往前看 shift 个位置，越界处夹到 0 |
| upd | 公式 | $blocked = blocked \lor (positions < shift) \lor (source = DEAD)$，DEAD 是 −1，标记不参与 n-gram 的位置。图像 span 的位置在 cache 里就是 DEAD，回看跨过它们就断 |
| upd | 说明 | 位置在序列开头之前、或往前看到 DEAD，都算断掉，之后的回看步也不再往前。DEAD 只影响后面的回看，图像位置自己不做 Engram 写入是另外一步，见 Engram 图的 gmask |
| rep | 公式 | $tokens = \mathrm{where}(blocked, pad\_id, source)$，pad_id 是断掉处填的压缩 id |
| rep | 说明 | 断掉的位置填 pad_id，即 engram_pad_id（2）经 token_map 映出的那个压缩 id，与训练一致 |
| prod | 公式 | $products = tokens.\mathrm{unsqueeze}(2) \odot multipliers$，tokens 是各回看步的压缩 id，unsqueeze 把层这一维扩出来，multipliers 是每层每个回看步的奇数乘数 |
| prod | 说明 | multipliers 每个（Engram 层，回看步）一个奇数乘数，各层用自己的随机种子生成 |
| r0 | 公式 | $rolling = products_{...,0}$，products 是 tokens 乘 multipliers 的结果，这一步取出异或循环的初值 |
| stk | 公式 | $tokens = \mathrm{stack}(tokens, \dim=-1)$，4 个回看步各一份 [B, T]，shift 从 0 到 3 |
| hcat | 公式 | $hashes = \mathrm{cat}(hashes, \dim=-1)$，3 个桶组各 [B, T, 2, 8] |
| roll | 公式 | $rolling = rolling \oplus products_{...,i}$，i 从 1 到 3，products 是上一行的乘积，$\oplus$ 是按位异或 |
| roll | 说明 | 一次异或一个回看步，第 i 步之后得到的是 i+1 元 n-gram 的哈希 |
| mod | 公式 | $rolling \bmod primes$，primes 是每个（层，桶组，头）的质数 |
| mod | 说明 | 每个（Engram 层，桶组，头）各有一个质数，互不重复；8 是 engram_n_heads，3 个桶组的结果沿最后一维拼成 24 |
| off | 公式 | $hash = hs + offsets$，hs 是拼接后的桶下标，offsets 是各桶在表里的起始行 |
| off | 说明 | 加上各桶在表里的起始偏移，得到整张表的行号 |

### 面板

| 组 | 名字 | 值 |
|---|---|---|
| 公式 | 压缩映射 | $compressed = token\_map[input\_ids]$，input_ids 是词 id，token_map 把同形 token 映到同一个压缩 id，压缩词表 99092 项 |
| 公式 | n-gram | 第 t 个位置的回看取 $c_{t-k}$，$k = 0 \ldots 3$；越界就填 pad_id |
| 公式 | 哈希 | $products = tokens \odot multipliers$。对 $i = 1 \ldots 3$，先 $rolling = rolling \oplus products_{...,i}$，再立刻对这次 rolling 取模。tokens 是各回看步的压缩 id，multipliers 是乘数，$\oplus$ 是按位异或 |
| 公式 | 桶 | 每一步异或之后 $hashes = rolling \bmod primes$，三步拼起来再 $hash = hs + offsets$。primes 是每个（层，桶组，头）的质数，hs 是拼起来的桶下标，offsets 是该桶在表里的起始行 |
| 配置 | engram_max_ngram_size | 4 |
| 配置 | engram_n_heads | 8 |
| 配置 | engram_head_dim | 256 |
| 配置 | engram_vocab_size | 16000000 |
| 配置 | engram_compressed_vocab_size | 99092 |
| 配置 | engram_pad_id | 2 |
| 配置 | engram_layer_ids | [1, 14] |
