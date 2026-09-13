<!-- review-meta
round: 5
page: wiki/deepseek-v4-1/index.html
reviewed_content_sha256: 24dbcf68e4195998
-->
# DeepSeek-V4.1-Flash（890 字节全局 KV 缓存）审查记录（第 5 轮）

- 页面版本：工作树 sha256 `4df37e2e74a248639729ddd437687266765f82e32784a017315135c7e61c2fe0`（git blob `6931d8f9b1f130abde152f29d32628dece112333`）
- 审查时间：2026-09-13 20:13
- 审查者：独立子代理（未参与写作与前序轮次的审查与修复）
- 已完整阅读章节：核心问题（5 条）/ 常见误解（5 条）/ 1. 890 字节的账（1.1、1.2、1.3、本章问题）/ 2. prefill 为什么只跑一半（2.1、2.2、2.3、本章问题）/ 3. 一个 query 看到哪些位置（3.1–3.6、本章问题）/ 4. 8B 与 16B（4.1、4.2、本章问题）/ 5. Engram 与 DSpark（5.1、5.2、本章问题）/ 来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）。正文含全部 `<details>` 折叠块与两处图注。
- 来源获取：官方仓库 `deepseek-ai/DeepSeek-V4.1-Flash`（Hugging Face）——`DeepSeek_V41_Tech_Report.pdf`、`config.json`、`inference/config.json`、`inference/model.py`、`inference/kernel.py`、`inference/engram.py`，均以本机 HTTP 抓取原文并按行核对。仓库中不存在外部 URL，页面所有外链均为站内概念页。

## 已核机械项（全部通过）

- `.dojo/scripts/validate.py wiki/deepseek-v4-1/index.html` → `validation ok`。
- head 字段齐全：纯文本 `description`、`dojo:summary`、`dojo:type=concept`、`dojo:topics=推理系统,内存与缓存`（词表内）、`dojo:tag=模型架构`（`ALLOWED_TAGS` 内）。
- `index.html` 与 `overview.html` 互链；15 个前置概念链接（kv-cache、mla、rmsnorm、rope、sliding-window-attention、cross-layer-kv-sharing、dsa、attention-sink、hyper-connections、deepseek-moe、aux-loss-free-routing、mxfp4-qat、mixed-precision-quant、residual-connection、deepseek-v4-1-dataflow）均存在。
- 来源编号 1:1 双向对应：正文 `<sup>[…]</sup>` 用到的 C1–C11 / F1–F8 / N1–N9 与来源章节定义完全一致，无未定义引用、无未引用条目；`research/measured.md` 与页面引用的所有仓库路径（`config.json`、`inference/config.json`、`inference/model.py`、`inference/kernel.py`、`engram.py`、技术报告 PDF）都存在。
- 公式定界符外的裸数学字符：仅 `×` 一处，位于 `<code>compress_ratios = [0,0,2×18,1×20,0×3]</code>` 代码块内（style-guide.md §11 代码块豁免，validate.py 亦排除 `×`）。结构图均为 HTML `div`（`dg-stack`/`dg-flow`），无等宽框线图，公式在 `div` 内由 KaTeX 渲染。
- §3.6 代码块实际执行（纯 Python，无第三方依赖），输出与页面「预期输出」逐行一致（`不符 0 处`、`i=4999 r=2: 可达 2500 条`、`未屏蔽时前 4 名: [0, 2, 6, 7]`、`屏蔽后前 4 名  : [0, 2, 3, 4]`、`True`）。

## 复算结论（数值与来源核对，抽查全部成立）

- 890 账：主 KV `3×144+288=720`、索引器 K `3×34+68=170`，合计 890。每条 288 = `512/2 + 512/16`；每条 68 = `128/2 + 128/32`；来源：config `kv_source_layer_ids=[2,8,14,20]`、`index_source_layer_ids=[2,8,14,20,24,28,32,36]`、`index_head_dim=128`、`head_dim=512`；`model.py` L760 `fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`（主 KV 每 16 通道 E4M3）与 L546 `fp4_act_quant(k, fp4_block_size=32, True)`（索引器每 32 通道 E8M0，`kernel.py` fp4_act_quant 默认 `scale_dtype=torch.float8_e8m0fnu`，docstring「FP8 with E8M0 scales for the indexer or E4M3 scales for compressed KV」）。
- §4.1 单层 377.31M 与 §4.2 半栈 7.8929B/7.5758B：按 config 形状逐项复算——层 0 注意力 `126,617,344`；层 20 注意力 `126,617,344 + 2,621,952(压缩器 r1) + 5,472,384(索引器含 wk/k_norm) = 134,711,680`；Reindex 注意力 `126,617,344 + 5,406,720 = 132,024,064`；压缩比 2 的 Full 层 `137,333,120`；门控 1,966,848；单专家 35,389,440（top-6 = 212,336,640）；超连接 `2×24×20480=983,040`；Engram 稠密投影 `24×256×25600 + 2×4×5120 = 157,327,360`（≈157.33M，两层 314.65M）。与页面表内数值逐项吻合。
- 引用行号抽查：报告行 314–319（40 层 / 前两层 SWA only）、388–396（Eq.1 `C_l=H_{L/2}W_l^{KV}`）、497–499（Full Mode 定义）、544–546（CSA2+CED）、561–562（2048×8=16384）、689–691（E2M1 + E4M3/16ch）、703–704（SWA 保持 FP8）、647–652（Engram 196B/8 heads/2048 per order）、661–668（DSpark 三块/5 草稿）；`model.py` L498–500、L537–548、L554、L583–610、L809–827、L1100–1156、L1261–1267；`kernel.py` L310–403、L426–462 —— 均与页面所述一致。

## 问题

- [重要·技术] `index.html:386`（§3.3 两级筛选末段）：「两级用的是同一套索引分——每个 query 对每条压缩条目的打分 $\sum_h w_{q,h}\mathrm{ReLU}(q_h\cdot k_j)$，区别只在聚合粒度」把第二级（位置级 Top-512）的打分说成与第一级（块级）共用同一份分数，且断言两级「区别只在聚合粒度」。来源相反：候选池（块级）由建池层用其自身索引分选出并共享，而位置级由各索引层用**各自的 query 重新打分**，故分数逐层不同——这也是 Reindex 与 Reuse 的区别所在。该句与同页 §3.4 三模式表「Reindex……Top-K 选择：自己算（在候选池内）」「Reuse……不算索引分」自相矛盾。｜引文依据：报告 §2.3.2「Subsequent layers in Reindex Mode score only the candidate positions for the corresponding query and select their own Top-K entries within that pool.」；`model.py` L556 `index_score = torch.einsum("bshd,btd->bsht", q, index_k)`、L573–575 `elif self.uses_candidates: index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)`。｜修复要求：删去「同一套索引分……区别只在聚合粒度」，改为「块级候选池由建池层用自身索引分选出并被后续层共享；位置级 Top-512 由各索引层在池内用各自的索引分重算（打分公式相同，分数逐层不同）」。｜修复：｜复验：
- [重要·技术] `index.html:705`（§5.2 第二段首句）：「3 个 MTP 块……每块一次前向产出 5 个草稿位置（块大小 5）」按字面读是每块各产出 5 个（3×5=15），与同页「核心问题」第 5 条「用 3 个 MTP 块一次前向产出 5 个草稿位置」矛盾，也与来源不符：3 个 DSparkBlock 顺序作用在**同一批** block_size=5 的位置上逐层细化，一次前向最终只产出 5 个草稿位置。｜引文依据：报告 §2.4.3「The drafter comprises three Transformer blocks … A single forward pass through these blocks computes base logits for five draft positions in parallel」；`model.py` L1278–1282 `for layer in self.mtp: h, pre_mix = layer(...)` 后 `return self.mtp[-1].forward_head(h, pre_mix, input_ids)`，而 `forward_head` 输出 `input_ids.new_empty(b, self.block_size + 1)`。｜修复要求：改为「3 个 MTP 块顺序作用在同一批 5 个位置（块大小 5）上，一次前向产出 5 个草稿位置」。｜修复：｜复验：
- [轻微·技术] `index.html:248`（§2.1 正文）与 `index.html:264`（同节结构图图层注）：「编码器各层照常从自己的隐状态算全局 KV」/「层 0–19（因果编码器）各层从本层隐状态算自己的全局 KV」把编码器说成每层都有全局 KV；实际层 0、1 压缩比为 0，是纯滑动窗口层，没有任何全局 KV，同页 §3.4 已写明「层 0、1 没有全局 KV，属于纯滑动窗口层」。｜引文依据：`config.json` `compress_ratios = [0,0,2,…,1,…,0,0,0]`；报告行 316–317「Each layer incorporates both global attention and sliding window attention (SWA), except for the first two layers, which use SWA only.」｜修复要求：两处各限定为「编码器中带全局 KV 的各层（层 0、1 除外）」或「编码器的 CSA2 层」。｜修复：｜复验：
- [轻微·技术] `index.html:355`（§3.2 末句）：「实测在等比缩小模型上逐位置一致[N7]」与所引 [N7] 的记录不一致：[N7] 记「原始最大差 8.9e-3……首个差异出现在位置 10」，只在修正后 fp32 下才一致；照字面读，[N7] 并不支持「逐位置一致」。｜引文依据：页面 [N7]（`index.html:763`）「prefill 与 decode 的一致性实测：原始最大差 8.9e-3（从层 2 起，首个差异出现在位置 10），修正后 bf16 残差 5.0e-3、fp32 为 1.5e-8 至 3e-8」。｜修复要求：把「逐位置一致」限定为「可达条目数的判据一致」，或改为「修正后（fp32）两种形态一致」，以与 [N7] 的原始差值表述相容。｜修复：｜复验：
- [轻微·表述] `index.html:155`（§1.2）、`index.html:309`（§2.2 解答）、`index.html:455`（§3.5）、`index.html:461`（§3.6）：含元话语与临场评价——「接下来算每条条目占多少字节。」「需要注意「共享的是输入而非投影结果」」（check.md 明列「需要注意的是」式元话语）、「三点值得单独指出。」「下面这段代码验证两件事：」。｜引文依据：不适用。｜修复要求：删去引导性套语或改为直接陈述（如「共享的是输入而非投影结果」；「该段代码验证……」「三点」改为直接列出）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：修复
- 说明：核心结论（890 = 720+170、每条 288/68 字节、prefill 只跑编码器半栈、8B/16B 加总、CED 由 $H_{L/2}$ 投影、三模式分工、可达性 $\lfloor(i+1)/r\rfloor$、Engram/DSpark 不改缓存账）经逐项回源与复算全部成立；8B/16B 单层与半栈的每个参数分量、890 的每一项均与官方 config/源码/报告吻合，页面内数值无第二处矛盾。本轮无阻断项；两条重要项为机制表述与草稿数表述，修复后即可发布。表述维度除上列四处外，全页无第一/第二人称、无调试叙事、无来源不支持的判断被写成结论（未支持的三条动机说明已在「辅助解释与类比边界」显式降级）。