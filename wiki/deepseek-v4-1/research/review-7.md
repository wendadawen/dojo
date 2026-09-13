<!-- review-meta
round: 7
page: wiki/deepseek-v4-1/index.html
reviewed_content_sha256: 15a2431aac7d81b2
-->
# DeepSeek-V4.1-Flash（890 字节 KV 缓存）审查记录（第 7 轮）

- 页面版本：index.html 工作树哈希 b10960b23c7a17653cba280bb3146dbd23bd2e75
- 审查时间：2026-09-13 21:47
- 审查者：独立子代理（未参与写作与前序轮次审查）
- 已完整阅读章节：核心问题、常见误解；1. 890 字节的账（1.1 压缩条目 / 1.2 一条条目 288 字节 / 1.3 只有四个层生产主 KV / 本章问题）；2. prefill 为什么只跑一半（2.1 主干被切成编码器与解码器两半 / 2.2 解码器的全局 KV 由 H_{L/2} 投影而来 / 2.3 滑动窗口 KV 不能一起省 / 本章问题）；3. 一个 query 看到哪些位置（3.1 可见集 / 3.2 可达条目数 / 3.3 两级筛选 / 3.4 三模式 / 3.5 选中的槽位怎么参与计算 / 3.6 可达性与 Top-K 语义的复算 / 本章问题）；4. 8B 与 16B（4.1 单层 377M / 4.2 半栈加总 / 本章问题）；5. Engram 与 DSpark（5.1 / 5.2 / 本章问题）；来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）。含全部折叠块与图注。

## 来源获取方式

- 技术报告 `DeepSeek_V41_Tech_Report.pdf`、`inference/config.json`、`config.json`、`inference/model.py`、`inference/kernel.py`、`inference/engram.py`、`model.safetensors.index.json`：自官方公开仓库 deepseek-ai/DeepSeek-V4.1-Flash 通过 HTTP 抓取原文并逐行/逐字段核对（报告以 pdftotext -layout 还原后按行号定位）。
- 未读取本页 research/ 下的规划与审查记录；下文中「实测」类条目的原始清单在 research/measured.md 内，本轮不据其判断，凡能由 config/源码/报告核对的一律回到原始来源核对。

## 核对摘要（回源记录）

以下为逐条回源核对中关键数值的原文片段，用于支撑结论：

- [C1] 报告行 314–319（与页引完全一致）：「Its language backbone comprises 40 causal Transformer layers, organized into a 20-layer causal encoder followed by a 20-layer decoder. Each layer incorporates both global attention and sliding window attention (SWA), except for the first two layers, which use SWA only.」`inference/config.json` `"compress_ratios": [0,0,2×18,1×20,0×3]`、`"kv_source_layers":[2,8,14,20]`、`"index_source_layers":[2,8,14,20,24,28,32,36]`、`"candidate_source_layer":20` 与该行结论一致。
- [C6] 报告行 388–393：「For the upper half layers (i.e., the decoder, l > L/2), the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H_{L/2}, using layer-dependent projection weights」；行 544–546：「the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer, i.e. the last layer of the causal encoder.」
- [C3] 报告行 689–691：「we select E2M1 with one E4M3 scale per 16 channels, following NVFP4 …, but omitting its second-level global scale」；行 703–704：「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」`model.py` L759 注释「Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.」，L760 `fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`，L707 窗口 KV `act_quant(...)` —— 支撑「主 KV 每 16 通道 E4M3、索引器每 32 通道 E8M0、窗口 KV 保持 FP8」。
- [C2] 报告 §2.3.1（行 490–520）：「Full Mode. The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices.」`model.py` L498–500 注释「the index keys are derived from the compressor's latent, so only a layer that compresses its own KV can produce them; every other indexer reads them from that layer's cache」——支撑「索引器 K 只有 4 份」。
- [C7] 报告 §2.3.2（行 542–570）：「selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions」「the candidate pool is shared across indexing layers, while their final selections can differ」；`model.py` L602–605 注释「the block with this query's newest position is only partly filled, so pin it in」——支撑两级筛选与「钉住最新块」。
- [N1/N9] 报告行 18–19「reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 …」；行 35–38「approximately 4-fold and 437-fold reductions … relative to DeepSeek-V4-Flash and DeepSeek-V1」；行 139–140「approximately 1/4 as much runtime KV cache storage and 1/8 as much persistent KV cache storage」。
- [N2/N3] 报告行 320–322：「has 552B backbone parameters and 196B Engram parameters, activating 8B parameters per token during prefill and 16B during decode.」
- 字节账复算：512/2=256、512/16=32、256+32=288；128/2+128/32=64+4=68；3×144+288=720、3×34+68=170、720+170=890 —— 与报告行 18 的 890 精确一致。
- 反事实复算：18×144+20×288=2592+5760=8352，8352/720=11.6；720/890=80.9%≈81%，170/890=19.1%≈19%；288/720=40% —— 全部复算一致。
- [N6] 单层激活参数按 config + 真实张量头复算：注意力 126,617,408（wq_a 6,553,600 + q_norm 1,280 + wq_b 41,943,040 + wkv 2,621,440 + kv_norm 512 + wo_a 33,554,432 + wo_b 41,943,040 + sink 64）≈126.62M；门控 384×5120+384+384=1,966,848≈1.97M；共享专家 3×5120×2304=35,389,440≈35.39M；top-6 路由 6×35,389,440=212,336,640≈212.34M；超连接 2×24×20480=983,040≈0.98M；归一化 2×5120=10,240≈0.010M。合计 377.30M，六项两位小数之和 377.31M（页面即按此法标注，一致）。
- [N2] 半栈加总复算：20×377,303,670 + 3×(5,243,392+5,472,384) + 2×157,327,360 = 7,892,875,448 ≈ 7.8929B（编码器，与页一致）；20×377,303,670 + (2,621,952+5,472,384) + 4×5,406,720 = 7,575,794,616 ≈ 7.5758B（解码器，与页一致）。Full 层（ratio 2）注意力 137.33M、Full 层（ratio 1）134.71M、Reindex 层 132.02M 均由张量头形状复算得同一值。
- 嵌入口径复算：129280×5120=661,913,600=0.6619B；7.8929+0.6619=8.5548；7.8929+7.5758=15.4687；15.4687+0.6619+0.6619=16.7925；15.4687/7.8929=1.960 —— 全部一致。
- [C4] `engram.py` L79/注释「A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max_ngram_size-gram)」；`model.py` L344 `n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads` → 3×8=24；L362 `gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))`；L1251 `engram_mask = None if image_mask is None else ~image_mask` —— 与 5.1 节描述逐点一致。
- [C5] 配置 `n_mtp_layers=3`、`dspark_block_size=5`、`dspark_noise_token_id=128799`、`dspark_target_layer_ids=[37,38,39]`、`dspark_markov_rank=256`、`dspark_n_routed_experts=128`、`dspark_n_activated_experts=3`；`model.py` L1131–1132、L1265–1266 与页引一致；checkpoint 索引中 `mtp.*` 张量恰好 2401 个，张量总数恰好 96085 个 —— 与 [N3]、[C5] 的数字完全对上。
- [F3] `model.py` L563–567：prefill 分支 `compress_lens = (torch.arange(1, seqlen + 1) // ratio)` 即 ⌊(i+1)/r⌋，decode 分支 `compress_lens = end_pos // ratio` 即 ⌊(s+1)/r⌋ —— 与 3.2 节公式一致。
- [F8] `kernel.py` L310–389：`scale=(1.0/d)**0.5`、`acc_s = q·k * scale`、`scores_max` 仅由选中槽位取得、`sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])` 只进分母、`idxs==-1` 时 `acc_s=-inf` 且 `kv=0` —— 与 3.5 节公式及三条说明逐条一致。
- [F6]/[C9] `kernel.py` L426–460：`pre=sigmoid(m0*scale0+base0)+eps`、`post=2*sigmoid(m1*scale1+base1)`、`comb` 先 softmax+eps 再 20 轮行列交替（末步列方向）—— 与页一致。
- [C10] `model.py` L537 `if self.owns_k and latent is not None:`（发布在条件分支内）、L554 `index_k = shared_attn.index_k[:bsz, : end_pos // ratio]`（读取无条件）—— 与 3.4 节补充一致。
- [C11] `model.py` L1261 `for i, layer in enumerate(self.layers)` 无跳过条件，`self.layers` 恰为 40 层（`self.mtp` 另存 3 个 DSparkBlock）—— 与 2.2 节补充「对任何 prefill 都逐层跑满 40 层」一致。
- 3.6 节代码：从页面提取（HTML 反转义）后实机执行，输出与页面「预期输出」逐行完全一致（`不符 0 处`、`i=4999 r=2: 可达 2500 条`、`屏蔽后前 4 名: [0, 2, 3, 4]`、`True`）。
- 机械项：15 个前置概念链接全部指向真实存在的 `wiki/<name>/index.html`；overview.html 与 index.html 双向链接；`.dojo/scripts/validate.py` 返回 `validation ok`；全文公式定界符之外的 Unicode 数学字符仅 1 处 `×`，出现在 `<code>compress_ratios = [0,0,2×18,1×20,0×3]</code>`（风格规范明确豁免 code 内字符）；无 `alt` 含 `$...$`、无 `（待生成）` 占位、无 `<img>` 正文图。
- 用词：自称全部为「本页/本文」（正文 14 处「本页」均为范围声明，无「本页将…」式前向元话语）；全文无「我们/你/我」；与 style-guide §12 一致。

## 问题

- [轻微·技术] 4.2 节「半栈差额分解」表第三行：<td>其余组件（门控、专家、超连接、归一化）逐层差</td><td>+0.02M</td>｜问题：这四项在编码器/解码器两层之间逐层同形，真实差值为 0；该行为弥合两个半栈取四位小数（7.8929−7.5758=0.3171）与按两位小数分项相加（314.65+2.43=317.08）之间的 0.02M 取整缺口而设，属把四舍五入残差写成实体分量，并与同一行说明「各层同形，可忽略」自相矛盾。｜引文依据：checkpoint 张量头 `model.safetensors.index.json` 中 layers.0（2334 张量）、layers.2、layers.20、layers.24 的 `ffn.gate.{weight,bias,bias_vl}`、`ffn.experts.{0..383}.{w1,w2,w3}.{weight,scale}`（各层均 384×6）、`ffn.shared_experts.{w1,w2,w3}.*`、`ffn_norm.weight`、`attn_norm.weight`、`hc_attn_fn/hc_ffn_fn/hc_attn_base/hc_ffn_base/hc_attn_scale/hc_ffn_scale` 名称与数量逐层完全一致；`model.py` L935–945（`mix_hc = (2 + hc_mult) * hc_mult`、`hc_dim = hc_mult * args.dim`）与 L877–887（共享专家/专家构造不随层模式变化）亦表明此四项与压缩比、编解码段无关。按张量头精确求和，两半栈之差 = 314,654,720 + 2,426,112 = 317,080,832 ≈ 317.08M，而表内合计写作 317.10M。｜修复要求：删除该行并把合计改为 +317.08M，或将该行改名为「两半栈取整尾差」并注明其非实体分量；改后须保证合计等于表中各行之算术和。｜修复：｜复验：
- [轻微·表述] 1.1 节末段、3.5 节末段、3.6 节「观察重点」、5.1 节中段：出现临场评价与口语化措辞——「池化的数值路径也有一处实测细节」「这一安排不是多余的——它把舍入推迟到最后一步」（1.1 节）、「谁的分高谁拿得多」（3.5 节）、「这说明『置 $-\infty$』不是可有可无的收尾」（3.6 节）、「表很贵、激活很便宜但并非免费」（5.1 节）。｜引文依据：不适用（表述类）。｜修复要求：把这些句子改为不含评价与口语的陈述，例如「……内部用 fp32 计算、输出再转回 bf16」省略「不是多余的」这一评价，「未生成条目仍会参与竞争并被选中」替换「不是可有可无的收尾」，「取回的只有 24 行，写回的投影是稠密的」替换「表很贵、激活很便宜但并非免费」；改后全页不得再出现评价性形容词与口语短句。｜修复：｜复验：
- [轻微·表述] 3.2 节末句与 [N7]：「实测在等比缩小模型上，修正后（fp32）两种形态逐位置一致[N7]」；[N7] 记「原始最大差 8.9e-3（从层 2 起，首个差异出现在位置 10），修正后 bf16 残差 5.0e-3、fp32 为 1.5e-8 至 3e-8」。｜问题：「修正」未说明修正对象（是缩小模型的实现缺陷、还是比对口径），读者无法判断 8.9e-3 到 1.5e-8 的落差来自何处，读作未交代的复现过程叙事。｜引文依据：[N7] 原文片段如上。｜修复要求：在 [N7] 内用一句说明「修正」指向的具体对象（例如「缩小模型中某处张量并行切分与真实实现不一致，修正该处后……」），使该条证据可被追溯；或在正文改为只陈述「两种形态在 fp32 下逐位置一致」而删去未经说明的修正叙事。改后「修正」一词须有可定位的所指。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（提交上述 3 条轻微修复；本轮未发现阻断或重要问题）

## 本轮核对结论说明

本页核心结论均可回源：890 = 主 KV 720 + 索引器 K 170 的构成与报告行 18–19 的 890 B/token 精确一致，且每一项（512 维 FP4 + 每 16 通道 E4M3 = 288 B；128 维索引器 K + 每 32 通道 E8M0 = 68 B；压缩比 2/2/2/1 的四个 source 层）都能在 config、model.py、kernel.py 的常量与注释中逐一定位；CED 的 H_{L/2} 投影、三模式分工、两级筛选与钉住最新块、MoE 路由公式、mHC/Sinkhorn、稀疏注意力汇聚点、Engram 与 DSpark 的结构描述均与报告原文或参考实现源码逐行对应。8B/16B 口径、377.31M 单层构成、7.8929B/7.5758B 半栈加总、317.1M 差额、0.6619B 嵌入层、比值 1.960 等全部可由 config + 真实张量头复算且与页面一致。第 3.6 节代码实机运行输出与页面预期输出逐行相同。唯一发现的实质缺陷是 4.2 表内一个把取整尾差写成实体分量的 0.02M 行，数值量级不影响任何结论。
