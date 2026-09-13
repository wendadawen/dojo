<!-- review-meta
round: 1
page: wiki/deepseek-v4-dataflow/index.html
reviewed_content_sha256: 2af7a848901fe4c3
-->
# DeepSeek-V4-Pro 前向数据流审查记录（第 1 轮）

- 页面版本：c8e14e6a305584838a8fb7a5f5a7fc1a8cec0c85
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作）
- 页面类型：dataflow（`dojo:type=dataflow`），依据 `guides/model-dataflow.md`；表述维度按该文件「与 concept 页同一套规范」引用 `guides/concept/check.md`
- 来源获取：本页 `research/official/` 不存在，官方材料改为直接抓外部来源核对——`huggingface.co/deepseek-ai/DeepSeek-V4-Pro` 的 `config.json`、`inference/config.json`、`inference/model.py`、`inference/kernel.py` 与仓库文件列表，arXiv:2606.19348v1 的 HTML 全文
- 已完整阅读章节（含全部折叠块、图注与 8 个交互视图的节点说明）：1. 关键规格｜2. 交互式数据流（overview / csa / hca / compressor / indexer / moe / mhc / mtp 视图）｜3. 要点（整体结构、CSA、HCA、两种层共有的四个细节、mHC、MoE 路由、长上下文效率的来源）｜4. 本机实测与核对｜来源与范围说明

## 问题

- [阻断·技术] index.html 第 123 行（§1 关键规格表「压缩率逐层配置」行）、第 111 行（page-lead）、第 457 行（overview 视图 `blk` 节点）：把逐层压缩率总结成「奇数层 128、偶数层 4」，与官方 config 的实际序列不符，且由该规则推出的层数与同一行给出的「31 + 30 层」相反（按规则：奇数层 1/3/…/59 共 30 层为 128，偶数层 0/2/…/60 共 31 层为 4，即 30 层 HCA、31 层 CSA，与页面自称的 31 层 HCA / 30 层 CSA 正好相反），句子自相矛盾；page-lead 的「奇数层 HCA…偶数层 CSA」同样漏掉第 0 层例外。｜引文依据：`config.json` `"compress_ratios": [128, 128, 4, 128, 4, 128, …, 4, 0]`（第 0 项 128、第 1 项 128、第 2 项 4）；页面第 151 行自称「第 0、1 层连续两层 HCA，此后严格奇偶交替」；实算 HCA=31（{0,1,3,…,59}）、CSA=30（{2,4,…,60}）。｜修复要求：把该行改为「第 0、1 层 128（HCA），第 2 层起奇数层 128、偶数层 4；HCA 31 层、CSA 30 层」，并同步修正第 111 行 page-lead 的「奇数层 HCA / 偶数层 CSA」。｜修复：｜复验：

- [重要·技术] index.html 第 125 行（§1 关键规格表「Indexer 选中条目数」行）：数值 1024 后括注 `$64 \times 128$`，但 64×128 = 8192 ≠ 1024，该等式不成立，页面对应的三个来源字段也不支持这个乘积关系。｜引文依据：`config.json` `"index_topk": 1024`、`"index_n_heads": 64`、`"index_head_dim": 128`；8192 实为 indexer 的 `wq_b` 输出维（与页面第 562 行 `indexer.wq_b 1536 → 8192` 一致），与 1024 无关。｜修复要求：删去该乘积括注，或改写为「1024（index_topk）；索引头 64、头维 128」。｜修复：｜复验：

- [重要·技术] index.html 第 219–229 行（§4 参数量表）：各分项之和与「合计」不符——1547.00 + 19.47 + 4.03 + 0.93 + 0.93 + 0.17 + 0.08 = 1572.61 B，表格却给出「合计 1598.84 B」，差 26.23 B。各行标注「61 层」，而合计对应的是含 MTP 的 62 个 block（按每专家 66.06M 参数算：61 层 1547.40 B、62 层 1572.76 B），口径不一致，与正文第 218 行「逐一累加」的自述冲突。｜引文依据：表内数字 1547.00 / 19.47 / 4.03 / 0.93 / 0.93 / 0.17 / 0.08 与合计行 1598.84；`config.json` `"num_hidden_layers": 61`、`"num_nextn_predict_layers": 1`；66,060,288 × 384 × 61 = 1547.40 B。｜修复要求：统一口径——或把各行改为含 MTP 的 62 层数值，或单列「MTP（1 层）」一行，使分项相加等于合计，并在表头写明统计范围。｜修复：｜复验：

- [重要·技术] index.html 第 231 行（§4 结尾 note）、第 240 行（「来源与范围说明」表末行）：正文把实测结论指向本页 `research/` 下的脚本与输出（`kernel_ref.py`、`t1`~`t11`、`all_results.txt`），但这些实测产物已从仓库移除，目录中只剩 `measured.md`。dataflow 规范明令引用实测结论时写清「实测得到」而不指向已移除的文件路径。｜引文依据：页面第 231 行「实测脚本与完整输出存于 research/ 目录：kernel_ref.py 为算子等价实现，t1~t11 为各项验证，all_results.txt 为…原始输出」；`research/measured.md`「本页的实测产物原先存放在本目录下，现已从仓库移除」；目录实况 `wiki/deepseek-v4-dataflow/research/` 仅含 `measured.md`。｜修复要求：删除对具体已移除文件名的指向，改为「上述结论由本机实测得到，实测清单见 research/measured.md」；第 240 行「脚本见 research/」同样改为指向 measured.md。｜修复：｜复验：

- [重要·技术] index.html 第 211–217 行（§4 候选集表）与其引言第 210 行：表中第 1 层 compress_ratio 标为 4、第 2 层标为 128，与官方 `compress_ratios` 的前四项 `[128, 128, 4, 128]` 冲突，也与本页第 151、456 行「第 0、1 层连续两层 HCA，此后严格奇偶交替」矛盾；据此第 210 行「结构与官方同构：HCA/CSA 交替」的说法不成立——表中所列的缩小配置实际用的是 128,4,128,4 的交错序列。｜引文依据：`config.json` `compress_ratios` 前四项 `[128, 128, 4, 128]`；页面第 213–216 行「层 0｜128 / 层 1｜4 / 层 2｜128 / 层 3｜4」。｜修复要求：写明缩小配置实际采用的 `compress_ratios` 序列并去掉「与官方同构」的表述，或把缩小配置改为官方序列后重跑，使表中层号与压缩率不再与官方 config 冲突。｜修复：｜复验：

- [重要·技术] index.html 第 168 行、第 177 行、第 184 行：三处把没有来源支持的归因/动机写成结论——第 168 行「正是因为压得足够狠，才可以不做稀疏化」；第 177 行「$128 \times 512 = 65536$ 维直接投回 7168 代价过高，故按 o_groups=16 分组」；第 184 行「hc_sinkhorn_iters=20 是与『门控参数初始化较小、矩阵接近均匀』相匹配的取值」。报告与 config/源码均未陈述这些因果或设计理由。｜引文依据：报告 2.3.2 关于 HCA 仅陈述「more aggressive compression…but keeps dense attention」，未给因果；`config.json` 只有 `o_groups: 16`、`hc_sinkhorn_iters: 20` 两个取值，无动机说明（另：第 184 行的 Sinkhorn 迭代顺序本身已核对无误，kernel.py `hc_split_sinkhorn` 确为 `comb.softmax(-1)+eps` → 列归一 → `T.serial(sinkhorn_iters-1)` 轮「行→列」，末步为列归一）。｜修复要求：删去「正因为…才」「代价过高，故」「是与…相匹配的取值」等因果措辞，改为可核查的陈述；确需保留则明确标注为推断。｜修复：｜复验：

- [轻微·表述] index.html 第 135 行、第 161/461/491/539/567/592 行：第 135 行「下图为实际前向路径。…」属引导语，按 dataflow 规范应直接进入路径；节点说明与列表多处「注意…」「关键：」（如第 161 行「——注意是先 ReLU 再按头加权求和」、第 461 行「注意这里只有 pre 权重」、第 491 行「注意打分是 ReLU 后按头加权求和」、第 539 行「关键：权重由内容算出」、第 567 行「注意没有 softmax」、第 592 行「注意精度与路由专家不同」）属元话语式提示。｜引文依据：不适用｜修复要求：删去「下图为…」引导句；把「注意」「关键：」改为直接陈述。｜修复：｜复验：

- [轻微·格式] index.html 第 184 行（`≤` 与三处 `→`）、第 230 行（`×`）、第 569 行（`≥`）：正文直接使用 Unicode 数学符号，未走 KaTeX（全页正文与图注计 34 个 `×`、43 个 `→`、1 个 `≤`、1 个 `≥`），与同页公式已用 KaTeX 的写法混用。｜引文依据：不适用｜修复要求：正文中的 `≤`/`≥` 改写为 `$\le$`/`$\ge$`，`→` 按语义写成 `\to` 或改为文字「得」；canvas 节点标签内的 `×`/`→` 若确无法走 KaTeX，需在页面注明该处为图内记号且写法统一。｜修复：｜复验：

- [轻微·技术] index.html 第 230 行：「每 token 激活 = 61 层 × 6 专家（24.18 B）+ 非专家部分（24.68 B）= 48.85 B」，24.18 + 24.68 = 48.86，与页面自称的 48.85 B（同见第 118 行规格表）差 0.01 B；且「非专家部分 24.68 B」未计入 lm_head 的 0.93 B，口径未说明（19.47+4.03+0.93+0.17+0.08 = 24.68，恰好排除 lm_head）。｜引文依据：页面自身数字 24.18 与 24.68；规格表 48.85B；报告与 HF README 作「49B」。｜修复要求：统一舍入使其自洽，并说明「非专家部分」包含哪些组件（是否含 lm_head）。｜修复：｜复验：

## 已核对并确认无误的要点（留档）

- 结构参数全部与 `config.json` / `inference/config.json` 一致：`num_hidden_layers` 61、`num_nextn_predict_layers` 1、`hidden_size` 7168、`num_attention_heads` 128、`num_key_value_heads` 1、`head_dim` 512、`qk_rope_head_dim` 64、`sliding_window` 128、`index_topk` 1024、`index_n_heads` 64、`index_head_dim` 128、`n_routed_experts` 384、`num_experts_per_tok` 6、`n_shared_experts` 1、`moe_intermediate_size` 3072、`scoring_func: sqrtsoftplus`、`routed_scaling_factor` 2.5、`num_hash_layers` 3、`hc_mult` 4、`hc_sinkhorn_iters` 20、`hc_eps` 1e-6、`max_position_embeddings` 1048576、`rope_scaling.factor` 16、`original_max_position_embeddings` 65536、`rope_theta` 10000、`compress_rope_theta` 160000、`o_groups` 16、`o_lora_rank` 1024、`q_lora_rank` 1536、`swiglu_limit` 10.0、`norm_topk_prob` true、`tie_word_embeddings` false、`vocab_size` 129280、`expert_dtype` fp4、量化 `fmt e4m3` / `scale_fmt ue8m0`。
- 源码级断言核对无误：`inference/model.py` 中 `self.overlap = compress_ratio == 4`；`indices = self.tid2eid[input_ids]`；`topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)`；`apply_rotary_emb(o[..., -rd:], freqs_cis, True)`；`if self.compress_ratio:` 时取 `compress_rope_theta`、否则取 `rope_theta`。`inference/kernel.py` 的 `hc_split_sinkhorn` 迭代顺序与页面第 184 行完全一致。
- 长上下文 KV 数字复算无误：每条目 64×2 + 448×1 = 576 B，为纯 BF16（1024 B）的 56.2%；1M 下主注意力合计 (30×262144 + 31×8192)×576 B = 4.355 GiB；基线 BF16 GQA8（head_dim 128，K+V）61×8×128×2×2×1048576 B = 244.0 GiB；比值 1.785% ≈ 1.79%，与报告「approximately 2% times of that baseline in the 1M-context setting」一致；indexer 30×262144×128×2 B = 1.875 GiB。报告「reduces the KV cache size by nearly half compared with pure BF16 storage」「27% of the single-token FLOPs … relative to DeepSeek-V3.2」、Eq.6 `A_l = σ(Ã_l)`、Eq.7 `C_l = 2σ(C̃_l)`、2.3.3「avoids exploding attention logits」、2.3.4「attention computation within the lightning indexer is performed in FP4 precision」均已定位到原文，引用节次正确。
- 仓库 64 个 `model-00001-of-00064.safetensors` 分片与页面「64 个 safetensors」一致。
- 表述维度：全文（含折叠块与 8 个视图节点说明）未出现「我/我们/你」会话指代，也未出现「本页将」「场景」等 AI 拼接腔词；元话语仅第 135 行引导句与「注意/关键」提示语（见上）。

## 结论

- 统计：阻断 1 / 重要 5 / 轻微 3
- 处置：修复（阻断项与重要项须全部关闭后再复验）