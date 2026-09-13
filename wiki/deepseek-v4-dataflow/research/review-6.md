<!-- review-meta
round: 6
page: wiki/deepseek-v4-dataflow/index.html
reviewed_content_sha256: 70487dfecffe104c
-->
# DeepSeek-V4-Pro 前向数据流审查记录（第 6 轮）

- 页面版本：a0cc100f85b0667a94ffe332f3e736e003631bbf（index.html 工作树哈希）
- 审查时间：2026-09-13 21:46
- 审查者：独立子代理（第 6 轮；未参与写作与前序审查，未读取本页 research/）
- 已完整阅读章节：1. 关键规格 / 2. 交互式数据流（含 noscript 七张表与全部视图节点说明、边列表）/ 3. 要点 / 4. 本机实测与核对 / 来源与范围说明

## 一、来源核对（本轮实际抓取并逐条比对）

来源全部可获取并已抓原文：arXiv:2606.19348v1 的 HTML 版；huggingface.co/deepseek-ai/DeepSeek-V4-Pro 的 config.json、inference/config.json、inference/model.py、inference/kernel.py、model.safetensors.index.json 与各分片 safetensors 文件头（HTTP Range）。

1. 结构参数（config.json / inference/config.json）：n_layers=61、dim=7168、n_heads=128、head_dim=512、rope_head_dim=64、q_lora_rank=1536、o_groups=16、o_lora_rank=1024、window_size=128、n_routed_experts=384、n_activated_experts=6、n_shared_experts=1、moe_inter_dim=3072、n_hash_layers=3、score_func=sqrtsoftplus、route_scale=2.5、index_n_heads=64、index_head_dim=128、index_topk=1024、hc_mult=4、hc_sinkhorn_iters=20、original_seq_len=65536、rope_factor=16、max_position_embeddings=1048576、vocab_size=129280、expert_dtype=fp4 —— 与「1. 关键规格」表逐行相符。
2. compress_ratios 实为 `[128,128,4,128,4,…,128,4,0]`（62 项，末项 0）。逐项计数得 128 共 31 个、4 共 30 个：与「HCA 31 层、CSA 30 层」「第 0、1 层连续两层 HCA，此后严格奇偶交替（偶数层 4 / 奇数层 128）」「前 12 项 [128,128,4,128,4,128,4,128,4,128,4,128]」「第 62 项为 0 即 MTP 不做压缩」全部相符。
3. checkpoint 计数（index.json + 文件头）：张量总数 145116、分片 64 —— 与正文「64 个分片、145116 个张量」相符。含 `attn.indexer.*` 的层恰为偶数层 2,4,…,60 共 30 层；`ffn.gate.tid2eid` 仅第 0,1,2 层；`ffn.gate.bias` 61 层中 58 层有、缺的正是 0,1,2；mtp.0 亦为 384 路由专家 —— 与正文两处互相印证的表述相符。
4. 张量形状/dtype 逐个读取文件头核对，全部相符：embed.weight BF16[129280,7168]；head.weight BF16[129280,7168]；layers.0.attn.compressor.wkv BF16[512,7168] 与 layers.2 BF16[1024,7168]（HCA 的一半）；ape F32[128,512]（l0，HCA）与 F32[4,1024]（l2，CSA）；indexer.compressor.wkv BF16[256,7168]；indexer.wq_b F8_E4M3[8192,1536]；indexer.weights_proj BF16[64,7168]；wq_a F8_E4M3[1536,7168]、wq_b F8_E4M3[65536,1536]、wkv F8_E4M3[512,7168]、wo_a F8_E4M3[16384,4096]、wo_b F8_E4M3[7168,16384]；attn_sink F32[128]；hc_attn_fn F32[24,28672]；hc_head_fn F32[4,28672]；路由专家 I8[3072,3584] + scale F8_E8M0[3072,224]（fp4_block_size=32，7168/32=224）；共享专家 F8_E4M3[3072,7168]；mtp.0.e_proj/h_proj F8_E4M3[7168,7168]；norm.weight BF16[7168]。
5. 源码语义（inference/model.py）：`h = h.unsqueeze(2).repeat(1,1,self.hc_mult,1)`（hc_mult=4）、`get_logits` 取 `x[:, -1]`、`self.overlap = compress_ratio == 4`、`self.mtp[-1].embed = self.embed`、`compress_ratio=0` 时 `original_seq_len, rope_theta = 0, args.rope_theta`(10000)、`should_compress = (start_pos+1) % self.compress_ratio == 0`、`indices = self.tid2eid[input_ids]`、`q *= torch.rsqrt(q.square().mean(-1,keepdim=True)+eps)`（逐头无权重 RMS）、`weights = original_scores.gather(...)`（bias 只进 topk）、`F.softplus(scores).sqrt()`、`weights /= weights.sum(...)` 后 `*= route_scale`、`apply_rotary_emb(o[...,-rd:], freqs_cis, True)`、`act_quant(kv[..., :-rd], 64, ...)`（前 448 维、block=64）、Indexer 的 `rotate_activation` + `fp4_act_quant`、`weights_proj(x) * (softmax_scale * n_heads ** -0.5)`、`topk(min(self.index_topk, end_pos // ratio))`、`mix_hc = (2 + hc_mult) * hc_mult = 24`、pre/post/comb 切分、hc_head 仅 `sigmoid(...)+hc_eps` 无 comb、Expert 中 `up=clamp(up,±lim)` 与 `gate=clamp(gate,max=lim)` —— 与正文及全部视图节点说明逐条相符。
6. kernel.py：`hc_split_sinkhorn` 的迭代顺序实为「softmax(-1)+ε → 列归一化 → 19 轮(行归一化→列归一化)」，末步为列归一化；`sparse_attn` 把 `exp(attn_sink[i]-max)` 加进分母、`idx==-1` 记为 −inf。与正文「列和精确为 1、报告式(8) 𝒯_r∘𝒯_c 末步为行归一化、与 kernel 顺序相反」「sink 使权重和可小于 1」「索引 -1 完全不参与」相符；报告 Eq.8 原文确为 M^(t)=T_r(T_c(M^(t-1)))，页面表述正确。
7. 报告（arXiv:2606.19348v1，第 2 节）：2.3.3「avoids exploding attention logits」、2.3.4「attention computation within the lightning indexer is performed in FP4 precision」、Eq.6 A_l=σ(Ã_l)、Eq.7 C_l=2σ(C̃_l)、Eq.8 如上、"reduces the KV cache size by nearly half compared with pure BF16 storage"、"approximately 2% times of that baseline in the 1M-context setting"、"27% of the single-token FLOPs … and 10% of the KV cache size relative to DeepSeek-V3.2"、CSA「compresses the sequence length to 1/m times」且每 C_i^Comp 由 2m 条组成、HCA「keeps dense attention」—— 页面引用的节次、公式与数字全部定位到且相符。
8. 算术复算全部自洽：参量表 1547.396+19.465+4.030+0.927+0.927+0.168+0.084+25.840 = 1598.837 B = 1.599 T（占比相加 100.1%，页面已说明取整余差）；路由专家 61×384=23424 个，1547.396B/23424 = 66.06M/个；24.18+24.68 = 48.86 B；注意力分项 30×CSA + 31×HCA ≈ 19.46 B；mHC 2×24×28672 = 1.376M/层 ×61 = 0.084B（占 0.005%）；测试表 132 = 128+512/128、144 = 128+min(16,512/4)、640 = 512+128、516 = 512+4；候选集表 4096/128=32、65536/128=512、1048576/128=8192，1152 与 8320 均对；KV cache 576 B/条 = 64×2+448，576/1024 = 56.2%；4.36 GiB = (30×262144+31×8192)×576 B；4.36/244 = 1.79%；Indexer 30×262144×128×2 B = 1.88 GiB、×0.5 B = 0.47 GiB。
9. 页面功能：`../hyper-connections/index.html`、`../deepseek-moe/index.html` 均存在；无「（待生成）」占位；`.dojo/scripts/validate.py` 返回 `validation ok`；无脚本时 noscript 提供七张纯 HTML 表格，交互视图仍可读；图中 `$...$` 由 KaTeX 渲染，无 `<img>` 依赖（唯一天箱图 `lightboxImg` 的 alt 为空，无 `$...$`）。

结论：本轮未发现来源不符、数字矛盾、算式与结论不符、引文错位或把推断包装成来源结论的条目；未发现元话语、会话指代、调试叙事或占位表述。

## 二、问题

- [轻微·格式] index.html 行 241（2. 交互式数据流 noscript「mHC 内部」表，sink 行「说明」格）与行 732（mhc 视图 sink 节点 `d` 字段）；同类见行 730–731（pre/post 节点 `label`）：公式定界符之外直接出现 Unicode 数学字符，且同一符号全页两种写法｜引文依据：规范原文（guides/concept/style-guide.md 行 107）「页面任何位置出现的数学变量、希腊字母、上下标、数学运算符和关系符都必须包在 `$...$` 或 `$$...$$` 中……禁止直接使用 Unicode 数学字符替代」；guides/concept/check.md 第 2.2 节第 9 条「标题、summary、正文、列表和表格中无 Unicode 数学字符直接出现；同一变量全页写法一致」。页内原文：同一 `<tr>` 第 3 格为 `$M^{(t)} = \mathcal{T}_c(\mathcal{T}_r(M^{(t-1)}))$`，第 4 格说明却写作 `报告式(8)把该复合写成 𝒯_r∘𝒯_c(M)（末步为行归一化）`（𝒯 = U+1D4AF 数学手写体大写 T，∘ = U+2218 环运算符）；节点 `label` 亦含裸 `σ`（U+03C3）、`ε`（U+03B5）｜修复要求：把说明中的 `𝒯_r∘𝒯_c(M)` 改为 `$\mathcal{T}_r\circ\mathcal{T}_c(M)$`（或 `$\mathcal{T}_r(\mathcal{T}_c(M))$`），与同格公式写法统一；canvas 节点 label 中的 `σ(·)`／`2σ(·)` 改为不含裸数学字符的写法，使正文与图中都不出现裸 Unicode 数学字符｜修复结果：（未修复）｜复验结果：（待复验）

## 三、结论

- 处置：可发布。阻断与重要问题均为 0；唯一轻微问题为上述公式书写一致性，不影响正确性与主线理解，记录在案。若修复，仅需改动说明文本与两处节点 label，不涉及任何数字或结论。
- 说明：本轮所有来源论断均已给出来源原文片段或关键数值（见第一节），无未核对条目。

统计：阻断 0 / 重要 0 / 轻微 1