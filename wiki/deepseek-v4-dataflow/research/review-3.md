<!-- review-meta
round: 3
page: wiki/deepseek-v4-dataflow/index.html
reviewed_content_sha256: 79047716b1084634
-->
# DeepSeek-V4-Pro 前向数据流审查记录（第 3 轮）

- 页面版本：index.html git 对象 98a866ea214c4b092136a7807857076d9faab4e5（sha256 e5a8e04a…b893e6d）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取 research/ 下规划、修复与前序审查记录）
- 规范：guides/model-dataflow.md（dojo:type=dataflow），表述维度按 guides/concept/check.md §2.2-12
- 已完整阅读章节：head 与页脚；1 关键规格；2 交互式数据流（放大图注，并逐条读完 8 个视图 VIEWS 的全部节点 label/io/公式/说明与连线）；3 要点（整体结构 / CSA / HCA / 两种层共有的四个细节 / mHC / MoE 路由 / 长上下文效率的来源）；4 本机实测与核对；来源与范围说明。来源侧完整读了官方 inference/model.py、inference/kernel.py、inference/config.json、inference/convert.py、HF config.json，并用 HTTP Range 读取了 64 个 safetensors 分片的 JSON 头逐张量复算；报告读了 arXiv:2606.19348v1 全文（定位到 2.3.3 / 2.3.4 各句）。
- 已核对通过（不列为问题）：config 全部结构参数（层数 61+MTP1、hidden 7168、128/1 头、head_dim 512、compress_ratios 62 项 [128,128,4,…] HCA31/CSA30、window 128、index_topk 1024/n_heads 64/head_dim 128、384 专家 top-6+moe_inter_dim 3072、sqrtsoftplus+route_scale 2.5、num_hash_layers 3、hc_mult 4/hc_sinkhorn_iters 20、1M+YaRN16、compress_rope_theta 160000）；张量头逐条相符（embed.weight BF16[129280,7168]；gate.tid2eid I64[129280,6] 只在第 0/1/2 层、这三层无 gate.bias，第 3 层起为 F32[384]；attn_sink F32[128]；hc_attn_fn F32[24,28672]；compressor.ape F32[128,512]/[4,1024]；compressor.wkv BF16[512,7168]/[1024,7168]；indexer.compressor.wkv BF16[256,7168]；专家 I8[3072,3584]+F8_E8M0[3072,224]；共享专家 F8_E4M3[3072,7168]）；张量总数 145116 与报告 2.3.4「approx 2% vs BF16 GQA8 head_dim 128」「nearly half」「27% FLOPs」「indexer FP4 精度」各句均能定位支持。

## 问题

- [阻断·技术] §1 关键规格表「总参数 / 每 token 激活」行 ↔ §4 末段：同一量同页两处取值不一致。规格表写 48.85B，§4 实算写 48.86B。｜引文依据：规格表「1.599T / 48.85B」；§4「61 层 × 6 个路由专家（24.18 B）+ 非专家部分（…19.47+4.03+0.93+0.17+0.08 = 24.68 B，不含 lm_head）= 48.86 B」（24.18+24.68=48.86）。｜修复要求：两处统一（实测合算为 48.86B；若采官方 49B 口径须注明），不得一处 48.85 一处 48.86。｜修复：｜复验：

- [阻断·技术] §4 参数量表「路由专家（61 层）」与「MTP（1 层…）」两行：分项与官方 checkpoint 不符。按 HF 仓库 64 个 safetensors 文件头逐张量复算，61 层路由专家应为 1547.40 B，页面写 1547.00 B；MTP（含路由专家+注意力+融合投影+mHC+共享专家+gate）应为 25.84 B，页面写 26.23 B。两处误差约 ±0.4 B 恰好互抵，故合计 1598.84 B 与占比仍成立，但分项数字与来源不一致。｜引文依据：单专家逻辑参数 w1 I8[3072,3584]→22,020,096、w2 I8[7168,1536]→22,020,096、w3 同 w1，合 66,060,288；61×384×66,060,288 = 1547.396 B；按 "mtp." 前缀权重逐张量求和 = 25.839 B。｜修复要求：按文件头重算并改为 1547.40 B / 25.84 B（或注明分项口径），使两行与来源一致、合计仍为 1598.84 B。｜修复：｜复验：

- [重要·技术] §3 HCA 小节第 3 条：机制描述与 model.py 不符。原文称压缩条目「位置索引是压缩后的块号，跨度被压掉，因此需要单独的 theta」；源码取的是压缩块首 token 的位置而非块号，位置跨度并未被压缩。｜引文依据：model.py Compressor.forward `freqs_cis = self.freqs_cis[:cutoff:ratio]`（prefill，取 0,ratio,2·ratio,…）与 `freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)`（decode，取块首位置），最大位置仍约达 max_seq_len，故「跨度被压掉」及其引出的「需要单独 theta」之理由缺少源码支持。｜修复要求：改为源码可支持的表述（如「压缩条目取所在块首 token 的位置」，或只说带压缩层用 compress_rope_theta=160000 而不给跨度被压的归因），不得保留无来源支持的机制归因。｜修复：｜复验：

- [轻微·表述] §2 交互式数据流各视图节点说明：多处节点说明以操作指引收尾（hcexp「点击下钻看 mHC 内部」、l01/csa/hca/moe/mtp「点击下钻。」、comp「点击下钻看压缩机制。」、idx「点击下钻。」），与悬停提示自动追加的「▸ 点击此节点下钻进内部」重复，属引导语；同节开头「节点标签、连线与悬停说明用「→」「×」等图形记号书写形状与运算过程（悬停说明中的数学式仍走 KaTeX）」为对页面自身记号的说明。｜引文依据：不适用。｜修复要求：删去节点说明中的「点击下钻…」句与对自身记号的说明，直接写形状与运算。｜修复：｜复验：

- [轻微·技术] §2 MoE 视图 exp 节点公式：公式写作 $w_2(\mathrm{SiLU}(w_1x)\odot\mathrm{clamp}(w_3x))$，漏掉 gate 支的上侧截断，与源码不符。｜引文依据：model.py Expert.forward `up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit); gate = torch.clamp(gate, max=self.swiglu_limit); x = F.silu(gate) * up`（同段文字已说明「对 gate 支上侧截断」）。｜修复要求：公式补上 gate 支截断（如 $\mathrm{SiLU}(\mathrm{clamp}(w_1x))\odot\mathrm{clamp}(w_3x)$），与文字一致。｜修复：｜复验：

- [轻微·表述] §2 Indexer 视图 qr 节点说明：末句「与主注意力共享同一个潜向量——这是「lightning」的一部分」为无来源的命名归因。｜引文依据：不适用。｜修复要求：删去该判断句，只保留「共享 qr 潜向量」的事实描述。｜修复：｜复验：

## 结论

- 统计：阻断 2 / 重要 1 / 轻微 3
- 处置：修复（阻断与重要问题修复并复验后重跑 .dojo/scripts/validate.py；本轮 validate.py 已通过）
