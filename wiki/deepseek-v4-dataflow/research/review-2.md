<!-- review-meta
round: 2
page: wiki/deepseek-v4-dataflow/index.html
reviewed_content_sha256: 2af7a848901fe4c3
-->
# DeepSeek-V4-Pro 前向数据流审查记录（第 2 轮）

- 页面版本：b0c29a851999abd9351eef850740740d384c2da8324820f442ee05af0d946206
- 审查时间：2026-09-13 19:01
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节（含三个内联脚本里的 8 个交互视图节点说明、公式与图注）：页面说明与 §1 关键规格、§2 交互式数据流（overview / csa / hca / compressor / indexer / moe / mhc / mtp 全部视图）、§3 要点（整体结构、CSA、HCA、两种层共有的四个细节、mHC、MoE 路由、长上下文效率的来源）、§4 本机实测与核对、来源与范围说明
- 核对来源：官方 config.json 与 inference/config.json（huggingface.co/deepseek-ai/DeepSeek-V4-Pro）、inference/model.py、inference/kernel.py、64 个 safetensors 文件头（HTTP Range 读取，145116 个张量全部枚举）、技术报告 arXiv:2606.19348v1；`.dojo/scripts/validate.py` 通过（validation ok）。

## 已核对并成立的条目（供复验参照，不计入问题）

- config 全部数值与页面 §1 一致：num_hidden_layers=61、hidden_size=7168、head_dim=512、qk_rope_head_dim=64、num_attention_heads=128、num_key_value_heads=1、sliding_window=128、index_topk=1024、index_n_heads=64、index_head_dim=128、n_routed_experts=384、num_experts_per_tok=6、n_shared_experts=1、moe_intermediate_size=3072、scoring_func=sqrtsoftplus、routed_scaling_factor=2.5、num_hash_layers=3、hc_mult=4、hc_sinkhorn_iters=20、max_position_embeddings=1048576、rope factor=16 / original 65536、expert_dtype=fp4、q_lora_rank=1536、o_groups=16、o_lora_rank=1024、vocab_size=129280、tie_word_embeddings=false、swiglu_limit=10.0、compress_rope_theta=160000、rope_theta=10000。
- checkpoint 头核对：张量总数 145116；含 attn.indexer.* 的层恰为偶数层 2,4,…,60（30 层）；含 gate.tid2eid 的层恰为 0,1,2（I64 [129280,6]）；含 gate.bias 的层为 3..60（58 层）；hc_attn_fn/hc_ffn_fn 全 61 层（F32 [24,28672]）；layers.2 compressor.wkv BF16 [1024,7168]、ape F32 [4,1024]；layers.0 compressor.wkv BF16 [512,7168]、ape F32 [128,512]；专家 I8 [3072,3584]+scale F8_E8M0 [3072,224]；共享专家 F8_E4M3 [3072,7168]；indexer.wq_b F8_E4M3 [8192,1536]、weights_proj BF16 [64,7168]、compressor.wkv BF16 [256,7168]；attn_sink F32 [128]；wo_a [16384,4096]、wo_b [7168,16384]；embed/head BF16 [129280,7168]；mtp.0.e_proj/h_proj [7168,7168]、mtp.0.hc_head_fn F32 [4,28672]。
- 源码核对：model.py `self.overlap = compress_ratio == 4`；`topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)` 后单次 sparse_attn；`apply_rotary_emb(o[..., -rd:], freqs_cis, True)`；kv 非 RoPE 维 `act_quant(kv[..., :-rd], 64, …)`；compress_ratio 为 0 时 `original_seq_len, rope_theta = 0, args.rope_theta`（关闭 YaRN）；Gate `scores = F.softplus(scores).sqrt()`、`scores = scores + self.bias` 后 `weights = original_scores.gather(1, indices)`；hash 层 `indices = self.tid2eid[input_ids]`；Compressor `(kv * score.softmax(dim=2)).sum(dim=2)`、`should_compress = (start_pos + 1) % self.compress_ratio == 0`；Indexer `(index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)`、`weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)`、`rotate_activation` 后再 `fp4_act_quant`。kernel.py 用 tilelang，`hc_split_sinkhorn` 顺序为 softmax(-1)+eps → 列归一化 → (iters-1) 轮(行归一化→列归一化)，与页面所述一致。
- 报告核对：arXiv:2606.19348v1 确有「27% of single-token inference FLOPs」、「reduces the KV cache size by nearly half compared with pure BF16 storage」（§2.3.4）、「approximately 2% times of that baseline …（BF16 GQA8, head dimension 128）」（§2.3.4）、「constrain the residual mapping matrix Bl to the manifold of doubly stochastic matrices」(§2.2)、「This normalization avoids exploding attention logits」(§2.3.3)、「attention computation within the lightning indexer is performed in FP4 precision」(§2.3.4)。页面对报告各处的引用与归因准确。
- 复算通过：576 B/条目 = 64×2+448×1，为纯 BF16 的 56.25%（页 56.2%）；1M 主注意力 KV cache ≈ 4.36 GiB，对 244 GiB 基线为 1.79%；候选集表 4K/64K/1M 的 160/640/8320 与 128+seq/128 一致；128×512=65536、16×1024=16384、4096=65536/16；mHC 每层 2×24×28672=1.38M，61 层 0.084B；logit=12 时 sigmoid=0.9999938、sqrt(softplus(12))=3.4641。

## 问题

- [阻断·技术] §1 关键规格表「Indexer 选中条目数」行（index.html:125）：数值括注 `1024（$64 \times 128$）` 的算式与结论不符。｜引文依据：官方 config.json `"index_topk": 1024, "index_n_heads": 64, "index_head_dim": 128`；64×128 = 8192 ≠ 1024。｜修复要求：删去该括注或替换为正确来源（如「= index_topk，取自 config.json」）；若意在指出 64 头 ×128 维是 Indexer 的 q 投影维度（8192），须写明其为投影维度而非选中条目数。

- [阻断·技术] page-lead（index.html:111）、§1 表「压缩率逐层配置」行（index.html:123），及 meta description/summary（index.html:6、:7）：逐层压缩率写成「奇数层 HCA（128）、偶数层 CSA（4）」，与官方 config 及本页 §3 自相矛盾；该行自身也不自洽。｜引文依据：config `"compress_ratios": [128, 128, 4, 128, 4, …, 4, 0]`——第 0 层为偶数层却 ratio=128（HCA）；「奇数层 128」只对应 30 个奇数层，行内却写「共 31 + 30 层」；§3（index.html:151）与视图说明（index.html:457）明确「第 0、1 层连续两层 HCA……HCA 31 层、CSA 30 层」。｜修复要求：lead 与 §1 该行改为「第 0、1 层 HCA；自第 2 层起偶数层 ratio=4（CSA）、奇数层 ratio=128（HCA），HCA 31 层、CSA 30 层」，删去与 §3 相冲突的「奇数层 128、偶数层 4」写法；description/summary 中「逐层交替」同样补上开头两层的例外。

- [阻断·技术] §4 参数量表（index.html:219-229）：各行之和与「合计」不符。｜引文依据：1547.00+19.47+4.03+0.93+0.93+0.17+0.08 = 1572.61 B，而合计行写 1598.84 B，差约 26.2 B；差额对应未列出的 MTP 层（1 层路由专家 384×66.06M≈25.4 B，加注意力/融合投影/norm）。config `num_nextn_predict_layers=1`，checkpoint 含完整 mtp.0.ffn.experts.* 与 mtp.0.attn.*；61 层路由专家 = 61×384×66.06M≈1547.4 B，与表中 1547.00 B 相符，说明合计 1598.84 B 含 MTP 而表中无此行。｜修复要求：补一行「MTP（1 层）」并给出其参数量，使各行之和等于合计；或注明合计口径包含 MTP、表中为 61 层主干分解。

- [重要·来源] §4 末尾 note（index.html:231）与「来源与范围说明」表（index.html:240）：正文把实测产物指向已移除的文件路径。｜引文依据：research/ 目录现仅存 measured.md 与 review-1.md；measured.md 自述「原先存放在本目录下，现已从仓库移除」，而页面仍写「实测脚本与完整输出存于 research/ 目录：kernel_ref.py … t1~t11 … all_results.txt」「本机实测，脚本见 research/」；规范 model-dataflow.md:38 要求「页面正文引用实测结论时写清『实测得到』而不是指向已移除的文件路径」。此外页内各条实测数字（均值池化相对差异 0.76、sink=5 时权重和 0.26、正反 RoPE 误差 2.4e-7、双随机 L2 比 0.83、尺度 3 时约 231 轮、压缩索引拼接后输出变化 1.57 等）均未给出脚本与运行条件。｜修复要求：删除对 research/ 下具体文件的指向，改为「本机实测」；并补注可复算所需的运行条件（缩小配置的层数/维度、算子等价复现方式），使每条实测数字可判读；无法复现者明确标注为不可复算的实测记录。

- [轻微·技术] §4（index.html:230）与 §1（index.html:118）：算式与标注值差 0.01。「24.18 B + 24.68 B」实为 48.86 B，同句及 §1 写 48.85 B。｜引文依据：句内数字。｜修复要求：统一为 48.86，或注明取整口径。

- [轻微·技术] §3「长上下文效率的来源」末段（index.html:206）：Indexer cache「1.88 GiB（30 个 CSA 层各 262144 × 128）」隐含每元素 2 字节，未说明该口径的存储精度，且与页内「Indexer 压缩器先 Hadamard 旋转再 FP4 量化模拟」的精度描述不一致。｜引文依据：1.88 GiB = 30×262144×128×2 B；model.py Indexer 的 Compressor 以 rotate=True 调用，随后 `fp4_act_quant`。｜修复要求：补注该 1.88 GiB 所假定的每元素字节数（精度），或按实际精度重算。

- [轻微·技术] §3 HCA 小节（index.html:168）：「正是因为压得足够狠，才可以不做稀疏化」是设计动机推断，页面以结论口吻写出。｜引文依据：报告 §2.3.2 仅称「HCA compresses the KV cache in a heavier manner, but does not employ sparse attention」，未给出该因果关系。｜修复要求：改为「报告称 HCA 不做稀疏注意力」，或明确标注为本文推断。

- [轻微·表述] 正文与视图节点说明（index.html:161「——注意是先 ReLU 再按头加权求和」、index.html:461「注意这里只有 pre 权重」、index.html:567「注意没有 softmax」、index.html:492/:539「关键：」、index.html:184「实测发现：」）：含提示语/临场评价式元话语。｜引文依据：不适用。｜修复要求：删去提示语直接陈述：「打分是先 ReLU 再按头加权求和」「该头只有 pre 权重，不做 comb」「此处无 softmax」「两条分支共享同一 softmax 分母」「官方 kernel 的迭代顺序为…」。

## 结论

- 统计：阻断 3 / 重要 1 / 轻微 4
- 处置：修复（阻断与重要问题全部关闭后进入下一轮审查）
- 说明：页面的结构性论断（形状、层数、路由、压缩、mHC、KV cache 口径）与官方 config.json、inference/model.py、inference/kernel.py、64 个 safetensors 文件头、arXiv:2606.19348v1 逐项核对后成立，报告引用与归因准确；本轮阻断项均为可复算的数字/派述不一致与一处来源路径失效，修复不涉及结论范围变更。
