<!-- review-meta
round: 1
page: wiki/sliding-window-attention/index.html
reviewed_content_sha256: be6c4fd7d61fcaed
-->
# 滑动窗口注意力审查记录（第 1 轮）

- 页面版本：6e21c923b42e4e65771c2821cc286c1737d24a11（index.html 工作树哈希）
- 审查时间：2026-09-10 16:14
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节：核心问题（4 条解答折叠块）→ 常见误解 → 1. 窗口把可见集切成「最近 $W$ 个位置」（含 1.1、补充折叠块、本章问题）→ 2. 省下的是打分次数与缓存字节（2.1–2.3、展开折叠块、本章问题）→ 3. 层堆叠接力：感受野 $k\times W$ 与它的天花板（3.1–3.3、补充折叠块、本章问题）→ 4. 环形缓冲：槽位、覆盖与位置编码（4.1–4.3、代码折叠块、本章问题）→ 来源与范围说明（6 个 h3）→ overview.html 全文

## 机械验证结果

- `.dojo/scripts/validate.py wiki/sliding-window-attention/index.html` → `validation ok`
- 页面可运行代码块（4.1 环形缓冲）复制执行，输出与页面「预期输出」逐行一致（槽位序列 `[4, 5, 6, 7, 0, 1, 2, 3]`、映射回位置 `[12..19]`）
- `research/verify_swa_accounts.py` 实跑输出与 `verify_swa_accounts.out` 逐行一致
- `wiki/deepseek-v4-1/research/ckpt/verify_sparse_attn_window.out` 中窗口环形缓冲（W=8、序列长 20）20 步全一致，与 [N7] 表述相符
- `overview.html` 与 `index.html` 相互链接；5 个前置概念页（standard-attention、causal-mask、kv-cache、rope、attention-sink）均存在
- `dojo:topics` = 注意力机制,内存与缓存，均在 AGENTS.md 固定大类内；`<head>` 五个 meta 齐备

## 问题

- [重要·技术] index.html L764、L803、L841、L849–850、L869：第 1 章主文与对照表把可见集写成 $[i-W, i]$，同一处又写「最多 $W$ 个槽位」/「$\min(i+1, W)$」；区间 $[i-W,i]$ 含 $W+1$ 个整数位置，与 $W$ 个槽位自相矛盾，第 3 章 $\text{span}(k)=k\times W$ 沿用同一口径，唯一澄清落在 1 章本章问题 Q1 的折叠块内（收起折叠块后正文只剩矛盾表述）｜引文依据：Mistral §2「attends to all hidden states from the previous layer with positions between i−W and i」，同节 Figure 1 caption「each token can attend to at most W tokens from the previous layer」；参考实现 `get_window_topk_idxs` 取 `(end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))`（即 $W$ 个槽位）｜修复要求：在主文与对照表统一口径，写明「实现取最近 $W$ 个位置即 $[i-W+1, i]$，来源论文写作 $[i-W, i]$」，并让可见集、单层槽位数、$\text{span}(k)$ 三处一致（可在第 1 章正文给出该说明，不只放在折叠答案里）｜修复：｜复验：
- [重要·技术] index.html L771、L926、L946、L1138、L1150；overview.html L54：40 层窗口缓存总量未声明统计范围。主干 40 层之外，模型还有 3 个 DSpark/MTP 草稿层，各自注册一份同尺寸 `window_kv_cache`，合计 43 份；页面把核心问题「是每层还是整个模型」答成「整个模型 2.50 MiB」｜引文依据：`official/inference/config.json` 中 `n_mtp_layers=3`、`dspark_block_size=5`；`model.py` 中 `class DSparkBlock(Block)`、`attention_cls = DSparkAttention`，`DSparkAttention(Attention)` 继承 `Attention.__init__` 注册 `torch.zeros(args.max_batch_size, args.window_size, self.head_dim)`；`Model.__init__` 中 `if args.dspark_block_size: for layer_id in range(args.n_mtp_layers): self.mtp.append(DSparkBlock(args.n_layers + layer_id, args))`；同文件注释「one entry per layer, MTP layers included」｜修复要求：在 2.2、核心问题 Q2 答案、[N5] 与「简化条件及其限制」中把 2.50 MiB 明确限定为「主干 40 层」，并说明另有 3 个草稿层注册同类缓存（约 0.19 MiB）；overview.html 同步｜修复：｜复验：
- [轻微·技术] index.html L1124（[C3]）、L750（blockquote.meta）：[C3] 第三句引文的来源位置标注错误，标为 §3.1、§3.2｜引文依据：该句「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV」位于 §1 Introduction；§3.2 的对应表述为「Alongside the current sliding window tokens, we reintroduce a few starting tokens' KV in the attention computation.」，§3.1/§3.2 均不含该原句｜修复要求：把该句标注改为 §1（或改用 §3.2 的原句），并同步修正 blockquote.meta 的「StreamingLLM … §3」｜修复：｜复验：
- [轻微·格式] index.html L1121、L1128：来源章节 h3 命名与 style-guide §1 规定的固定命名不符，页面用「核心论断与来源」「核心公式与来源」，规定为「论断与来源（C）」「公式与来源（F）」｜引文依据：不适用｜修复要求：改为规定命名（「外部数字与实验条件」「构造示例」「辅助解释与类比边界」「简化条件及其限制」已符合，无需改动）｜修复：｜复验：
- [轻微·格式] index.html L1143：构造示例说明引用了页面中不存在的「$W=3$ 的窗口示意」；页面仅有 L807–837 一处 6 个位置的示意图，且图中未标注 $W=6$｜引文依据：不适用｜修复要求：改为与实际图示一致的描述（如「6 个位置的窗口示意」），或按说明补出 $W=3$/$W=6$ 两个示意｜修复：｜复验：
- [轻微·格式] index.html L1129（[F1]）、L1135（[N2]）：[F1]、[N2] 仅出现在来源章节，正文没有对应 `<sup>` 引用，来源编号未双向对应｜引文依据：不适用｜修复要求：在正文相应位置补引用（如 $\text{span}(k)=k\times W$ 处引 [F1]），或删除正文未使用的来源条目｜修复：｜复验：
- [轻微·技术] index.html L889、L899–903、L908、L1151：打分次数计数口径不一致——全注意力按非因果全对 $N^2$ 计，窗口按因果 $N\cdot W$ 计，比值被写成「恰好等于 $N/W$」；若两者都按因果计（全注意力每 query 可见 $i+1$ 个槽位，合计 $N(N+1)/2$），比值约为 $N/(2W)$｜引文依据：页面自身定义 $n_{\text{score}}(N)$ =「query 数 × 每个 query 的可见槽位数」；因果全注意力的可见槽位数是 $i+1$ 而非 $N$；`verify_swa_accounts.out` 中 N=1048576 的 8192× 由 $N^2/(N\cdot W)$ 得出｜修复要求：在「简化条件及其限制」中写明全注意力按 $N^2$（全部 query-key 对）计数，或把「恰好」改为「约」并注明口径｜修复：｜复验：
- [轻微·技术] index.html L1000：「给每层再挂一路压缩后的全局 KV」与源码不符｜引文依据：`config.json` 中 `compress_ratios` 主干前两位为 0（第 0、1 层为纯窗口，无压缩 KV），`kv_source_layers=[2,8,14,20]`；`model.py` 注释「layers sharing a ratio also share one compressed KV and one indexer, produced by the first」｜修复要求：改为「除最前两层外，同压缩比的层共享一路压缩后的全局 KV 与一个索引器」，或删去「每层」这一限定｜修复：｜复验：
- [轻微·技术] index.html L1004、L1136：「把开头 4 个 token 换成换行符，模型仍然依赖它们」未标注来源，[N3] 也未收录对应数字｜引文依据：StreamingLLM Table 1 中 `4"\n" + 1020` 行困惑度 5.60（页面 [N3] 只收录 0+1024=5158.07 与 4+1020=5.40）｜修复要求：该句补 `<sup>[N3]</sup>`，并在 [N3] 中补入 4"\n"+1020=5.60 一行｜修复：｜复验：
- [轻微·技术] index.html L914–921、L1125（[C4]）、L1138（[N5]）：64 KiB 的「每元素 1 字节」口径来源未说明。该数字由技术报告 §2.4.4 的部署口径支持，但页面 [C4] 同时引用的参考实现 `act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)` 按 kernel 文档是 quant+dequant 回 BF16，其缓冲区为 2 B/元素（128 KiB/层）｜引文依据：`kernel.py` 中 `act_quant` docstring「inplace=True does fused quant+dequant back to BF16」及 `y = torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)`；`tech_report.txt` §2.4.4「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」｜修复要求：在 [N5] 或「简化条件及其限制」中注明 1 B/元素取自技术报告的部署口径，并说明参考实现该调用为 quant+dequant（缓冲区非 FP8 字节布局）｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 8
- 处置：修复（重要问题需全部关闭后进入第 2 轮审查；轻微问题逐条修复或写明接受理由）
