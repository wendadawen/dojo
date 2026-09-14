<!-- review-meta
round: 8
page: wiki/deepseek-v4-1/index.html
reviewed_content_sha256: 0d8c7a83f77975da
-->
# DeepSeek-V4.1-Flash：把每 token 全局 KV 缓存压到 890 字节的三项叠加 审查记录（第 8 轮）

- 页面版本：36f2cb6170a88eb1037a698a5564bdfc10db78fa
- 审查时间：2026-09-14 16:50
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查与修复）
- 适用规范：dojo:type=concept → `guides/concept/check.md`（并据 `guides/concept/style-guide.md` 判定格式与用词）
- 已完整阅读章节：核心问题 · 常见误解 · 1. 890 字节的账（1.1 / 1.2 / 1.3 / 本章问题）· 2. prefill 为什么只跑一半（2.1 / 2.2 / 2.3 / 本章问题）· 3. 一个 query 看到哪些位置（3.1–3.6 / 本章问题）· 4. 8B 与 16B（4.1 / 4.2 / 本章问题）· 5. Engram 与 DSpark（5.1 / 5.2 / 本章问题）· 来源与范围说明

## 来源核对（本轮抽验，全部通过）

来源获取方式：`https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/...` 取 `inference/config.json`、`inference/model.py`、`inference/kernel.py`、`inference/engram.py`；`resolve/main/DeepSeek_V41_Tech_Report.pdf` 取技术报告（pdftotext -layout，行号与页面标注一致）；HF 模型卡与 `resolve`/`tree` API 核对文件清单。

- [C1] 报告行 314–317「Its language backbone comprises 40 causal Transformer layers, organized into a 20-layer causal encoder followed by a 20-layer decoder. Each layer incorporates both global attention and sliding window attention (SWA), except for the first two layers, which use SWA only.」；`compress_ratios` 实际 43 项 = `0,0 + 2×18 + 1×20 + 0×3`，页面记号 [0,0,2×18,1×20,0×3] 计数吻合（18 个 2 在层 2–19、20 个 1 在层 20–39）。
- [C2] 报告行 497–498「Full Mode. The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices.」；model.py L498–499 注释「the index keys are derived from the compressor's latent, so only a layer that compresses its own KV can produce them; every other indexer reads them from that layer's cache」。`kv_source_layers=[2,8,14,20]`、`index_source_layers=[2,8,14,20,24,28,32,36]` → Full 4 / Reindex 4 / 其余 30 层 Reuse，与页面一致。
- [C3] 报告行 689–691「we select E2M1 with one E4M3 scale per 16 channels, following NVFP4 ... but omitting its second-level global scale」、行 703–704「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」；model.py L759 注释「Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.」、L760 `fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`、L707 窗口 KV `act_quant(kv, fp8_block_size, ...)`、L28 `fp4_block_size = 32`、L546/L552 索引器 K/Q。
- [C4] `engram_layer_ids=[1,14]`、`engram_max_ngram_size=4`、`engram_n_heads=8`、`engram_head_dim=256`；model.py L344 `n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads`（=24）；engram.py L171–184 按 2..max_ngram_size 的 n-gram、每段 n_heads 个素桶求哈希；L362 门控 `sigmoid(copysign(abs(dot).clamp_min(1e-6).sqrt(), dot))`。
- [C5] `n_mtp_layers=3`、`dspark_block_size=5`、`dspark_noise_token_id=128799`、`dspark_target_layer_ids=[37,38,39]`、`dspark_markov_rank=256`、`dspark_n_routed_experts=128`、`dspark_n_activated_experts=3`；报告行 664–667「The drafter comprises three Transformer blocks... computes base logits for five draft positions in parallel」；model.py L1131–1132 `draft_input_ids = input_ids.new_full([..., block_size], noise_token_id)`、L1265–1266 `if i in self.target_layer_ids: main_hiddens.append(h.mean(dim=2))`。
- [C6]/[F7] 报告 §2.2（行 379 起）行 389–392 原文同页面引用；行 394–396 为 Eq.(1) `C_l = H_{L/2}W_l^{KV}, Z_l = H_{L/2}W_l^{Z}, l > L/2`；行 544–546「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer, i.e. the last layer of the causal encoder.」
- [C7]/[F4] 报告行 565–570「blockwise candidate selection: each block is assigned the maximum index score among its positions ... For example, selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions.」；model.py L596–610：`amax` 块分（最大，非平均）、L604–605 用 `token.inf` 钉住最新块。
- [C8]/[F5] model.py L817 `scores = F.softplus(scores).sqrt()`、L821–822 注释「the bias picks experts but does not scale them: weights come from the raw scores」、L822–826 `indices=(scores+bias).topk(...)`、`weights /= weights.sum(...)+1e-20`、`weights *= self.route_scale`；`route_scale=1.5`、`n_routed_experts=384`、`n_activated_experts=6`、`norm_topk_prob=true`。
- [C9]/[F6] kernel.py L426–460：`pre = sigmoid(mix*scale0+base)+eps`、`post = 2*sigmoid(mix*scale1+base)`、`comb = softmax(-1)+eps` 后做 1 次列归一 + 19 轮「行→列」，即行、列各 20 次、末步列方向；L409 `mix_hc = (2 + hc) * hc`（=24）。
- [C10] model.py L537 `if self.owns_k and latent is not None:`（发布在分支内）、L554 读取无条件执行。
- [C11] model.py L1261–1267 `for i, layer in enumerate(self.layers)` 无跳过条件；`grep` 全 inference/ 无 `replay`/`bounded`/encoder-decoder 分支。
- [F1] model.py L458–485 池化：`(kv * score.softmax(dim=2)).sum(dim=2)`、`ratio == 1` 时 `return self.norm(self.wkv(x))`（无门控）；L538 注释「a latent stands for the first token of its group, so group j takes position j * ratio」。
- [F2] model.py L550–557 `einsum("bshd,btd->bsht", q, index_k)` → `(index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)`，与页面 $\sum_h w_{q,h}\mathrm{ReLU}(q_h\cdot k_j)$ 一致。
- [F3] model.py L563–567 `compress_lens = (arange(1, seqlen+1)//ratio)`、`masked_fill_(arange(seqlen//ratio) >= compress_lens, -inf)`（即 $\lfloor(i+1)/r\rfloor$，且为「不小于」屏蔽）。
- [F8] kernel.py L310–403 `sparse_attn`：`sum_exp[i] += exp(attn_sink[i] - scores_max[i])`（汇聚点只进分母）、`m` 为选中槽位行最大、`idxs==-1` 置 `-inf` 不贡献。
- [N1]/[N9] 报告行 18–19「reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of ... DeepSeek-V4-Flash」；行 35–38「approximately 4-fold and 437-fold reductions ... relative to DeepSeek-V4-Flash and DeepSeek-V1, respectively」；行 139–140「1/4 as much runtime KV cache storage and 1/8 as much persistent KV cache storage」。
- [N2]/[N3] 报告行 320–322「552B backbone parameters and 196B Engram parameters, activating 8B parameters per token during prefill and 16B during decode」。
- [N4] `window_size=128`、`index_topk=512`、`candidate_topk_blocks=2048`、`candidate_block_size=8`、`index_n_heads=32`、`index_head_dim=128`、`num_key_value_heads=1`、`head_dim=512`。
- [N5] 报告行 692「the format supports magnitudes up to 448 × 6 = 2688」、行 693–698「the largest trained RMSNorm weight magnitude is approximately 1 ... the L2 norm of the 512-channel KV latent is at most approximately 512 ... maximum magnitude observed during training is around 10」；$\sqrt{512}=22.6274$、$9.5/2.65=3.585$、$2688/10\approx269$ 均可复算（9.5%/2.65% 属 [N5] 自陈实测，非报告数值，标注正确）。
- 复算：288=512/2+512/16；68=128/2+128/32；720=3×144+288；170=3×34+68；890=720+170；8352=18×144+20×288，8352/720=11.6；720/890=80.9%→81%、170/890=19.1%→19%；单层 126.62+1.97+35.39+212.34+0.98+0.010=377.31M、层 20=385.40M；137.33/134.71/132.02 与按 model.py 逐模块数出的参数量吻合（126.62M 由 wq_a 6.554M + wq_b 41.943M + wkv 2.621M + wo_a 33.554M + wo_b 41.943M + 归一/汇聚 0.002M 得）；7.8929B/7.5758B/15.4687B/16.7925B 与页面各步加总自洽。
- 代码：抽出 3.6 节 `<pre><code class="language-python">` 去转义后实跑，输出与页面「预期输出」逐行一致（`不符 0 处`、`i=4999 r=2: 可达 2500 条`、`i=4999 r=1: 可达 5000 条`、`i=4 r=2: 可达 2 条`、`未屏蔽时前 4 名: [0, 2, 6, 7]`、`屏蔽后前 4 名: [0, 2, 3, 4]`、`True`）。
- 机械项：`.dojo/scripts/validate.py` 返回 `validation ok`（退出码 0）；被引前置概念页 kv-cache / mla / mxfp4-qat / mixed-precision-quant / rmsnorm / sliding-window-attention / residual-connection / rope / dsa / cross-layer-kv-sharing / attention-sink / hyper-connections / deepseek-moe / aux-loss-free-routing / ngram / speculative-decoding / deepseek-v4-1-dataflow 全部存在；无「（待生成）」占位；页面引用的 `research/measured.md`、`inference/model.py`、`inference/kernel.py`、`inference/engram.py`、`inference/config.json`、`DeepSeek_V41_Tech_Report.pdf` 均真实存在；两版 config.json 关键字段取值一致、字段名不同，与页头说明相符；`overview.html` 与 `index.html` 双向链接；图表为 HTML 结构（`.dg-stack`/`.dg-flow`）非等宽字符框线，图内公式写在 div 中由 auto-render 渲染，`<text>` 未出现 ASCII 近似；无 `<img>` 带 `$...$` 的 alt；无 Unicode 数学字符直接出现（validate.py 通过）。

## 问题

- [轻微·技术] 核心问题第 3 题解答（可见集）/ 正文 3.2 节：两处对同一可达性边界的表述不一致——解答写「尚未产生的条目（编号超出 $\lfloor(i+1)/r\rfloor$）」，按字面读为「编号 > $\lfloor(i+1)/r\rfloor$」，会把编号恰好等于 $\lfloor(i+1)/r\rfloor$ 的第一条未生成条目算成已生成（$i=4999,r=2$ 时该值为 2500，编号 2500 的条目尚未产生）。｜引文依据：正文 3.2 节「$n_{\text{reach}}$：从条目 0 起连续可达的条目条数。编号不小于该值的条目在打分阶段被置为 $-\infty$。」；model.py L565 `masked_fill_(torch.arange(seqlen // ratio, ...) >= compress_lens, -torch.inf)`（判据为「不小于」）。｜修复要求：把该处「编号超出 $\lfloor(i+1)/r\rfloor$」改为「编号不小于 $\lfloor(i+1)/r\rfloor$」，与正文 3.2 节及源码 `>=` 判据一致。｜修复：｜复验：
- [轻微·表述] §1.1「压缩条目：把 $r$ 个 token 池化成一条 KV」段「池化的数值路径也有一处实测细节：同一条池化在 fp32 层面对照时差异为 4.5e-6（相对 2.2e-6），转成 bf16 后 4096 个元素中有 5 个落在不同 ulp」：这是一次实现级数值比对记录，不支撑本章任何结论，并以元话语「也有一处实测细节」引入，属实现细节旁白而非机制解释（其后的「压缩器内部用 fp32 计算、输出再转回 bf16」才是与机制相关的结论，可保留）。｜引文依据：不适用（表述类；其数值与 [N8] 登记一致，非事实错误）。｜修复要求：删除该句中 4.5e-6 / 2.2e-6 / 4096 个元素中 5 个 ulp 的比对细节，仅保留 fp32 计算后转回 bf16 的机制说明；如确需留存验证记录，[N8] 已登记同一内容，无需在正文复述。｜修复：｜复验：
- [轻微·表述] §4 章首「这一章把这个数字从宣传口径拆成可复算的加法」：「宣传口径」带口语与评价色彩，与全文其余位置一致的「报告宣称」「登记为报告宣称」中性表述不同一格。｜引文依据：不适用。｜修复要求：改为中性表述，如「这一章把这个数字拆成可复算的加法」或「这一章把报告给出的口径拆成可复算的加法」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（3 项均为轻微，逐条修复后即可；本轮无阻断、无重要问题）
