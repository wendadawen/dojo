<!-- review-meta
round: 7
page: wiki/mqa-gqa/index.html
reviewed_content_sha256: a9028104b1859907
-->
# MQA 与 GQA 审查记录（第 7 轮）

- 页面版本：27c7d848a166dafe11c2cab0476718c350fa2844（wiki/mqa-gqa/index.html 工作树）
- 审查时间：2026-09-14
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节（按顺序）：核心问题 / 最容易误解 / 1. 为什么 MHA 推理受内存带宽限制——KV cache 如何随头数与序列长度增长 / 2. MQA——所有 query 头共享一组 K/V，cache 减到 1/h / 3. GQA——在 MHA 与 MQA 之间插值，G=8 是折中点 / 4. 手算对比——4 头 MHA / 2 组 GQA / 1 组 MQA 的 KV cache 连续谱 / 5. 边界与后续——MQA/GQA 不解决什么，与 MLA 的根本区别 / 来源与范围说明（含全部折叠块、图注与 callout）；另读 overview.html。

- 来源核对版本（逐条回源）：Shazeer 2019, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150（仅 v1）；Ainslie 2023, "GQA: ...", arXiv:2305.13245 v3（EMNLP 2023 版，ar5iv 最新渲染）；DeepSeek-V2, arXiv:2405.04434（最新版）。核对结果：Shazeer abstract "incur only minor quality degradation from the baseline"/"much faster to decode"、§3 "Multi-query attention is identical except that the different heads share a single set of keys and values."、§2.4.1 "the ratio of memory access to arithmetic operations is Θ(n/d + 1/b). When n≈d or b≈1, the ratio is close to 1"、§3.1 "Θ(1/d + n/(dh) + 1/b) ... We have reduced the offensive n/d by a factor of h"、实验任务 WMT14 En-De 与 Billion-Word LM 均与页面一致；Ainslie §2 "Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head."、§2.1 mean-pool 优于 single head/random、§3.1 "For α=0.05, training took approximately 600 TPUv3 chip-days"、§3.1 "We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder self-attention"、§3.3 Figure 4 "ordered by the degree to which information is preserved from the pre-trained model"、Figure 5 "gain from 5% uptraining with diminishing returns from 10%"、Figure 6 "We selected 8 groups as a favorable middle ground"、Appendix A 训练稳定性整句、Table 1（0.37/46.0、1.51/47.2、0.24/46.6、0.28/47.1，7 个数据集的平均）均一致；DeepSeek-V2 §2.1.2 Eq.(9)–(11)（c_t^{KV}=W^{DKV}h_t 等）、§2.1.4 Table 1（MHA 2n_h d_h l / GQA 2n_g d_h l / MQA 2d_h l / MLA (d_c+d_h^R)l）、§3 配置 n_h=128,d_h=128,d_c=512,d_h^R=64、abstract "Compared with DeepSeek 67B ... reduces the KV cache by 93.3%" 均一致（DeepSeek 67B 使用 GQA 见 arXiv:2401.02954 §2）。全页 154 段公式经本地 KaTeX 渲染 0 失败；`python3 .dojo/scripts/validate.py wiki/mqa-gqa/index.html` 通过；图内无 ASCII 近似公式、无 `$...$` 进入 alt、无 `<text>` 数学；引用的 standard-attention / mla 前置页真实存在。数值复算：2·128·128·4096·80·2=2.147e10 B≈21.5 GB、GQA-8≈1.34 GB、MQA≈168 MB、32768/576≈57、512/256/128、1.51/0.28≈5.4×、1.51/0.24≈6.3× 全部自洽。

## 问题

- [阻断·技术] §5 正文（index.html 第 460 行）与 §5「本章问题」第 1 题答案（第 494 行）：「当 batch $b$ 极大使 $1/b$ 项主导 $\Theta(n/d+1/b)$ 时，减 $n/d$ 项的收益被稀释」把条件写反——$1/b$ 项只在 batch **小**（$b\to1$）时主导；$b$ 极大时 $1/b\to0$、几乎不贡献，与该式及同页三处叙述相矛盾。同页第 74、194、172 行均写「$n$ 接近 $d$ 或 batch $b$ 小时[比值]接近 1」，overview.html 第 30 行同；第 201 行写「当 $1/b$ 项主导时应增大 batch 而非改注意力」（隐含当时 $b$ 小）。第 460/494 行与上述四处及出处公式互相矛盾。｜引文依据：Shazeer 2019（arXiv:1911.02150v1）§2.4.1 原文 "the ratio of memory access to arithmetic operations is Θ(n/d + 1/b). When n≈d or b≈1, the ratio is close to 1 ... The 1/b term is the easier one - we can just use a larger batch size."｜修复要求：把第 460 行与第 494 行的「batch $b$ 极大使 $1/b$ 项主导」改为「batch $b$ 极小（$1/b$ 项主导）」（或等价表述），与第 74/172/194/201 行统一；不得保留「极大 + 1/b 主导」的组合。｜修复：｜复验：
- [轻微·技术] §4 折叠块「推广到真实规模的量级感」（第 428 行）：「这就是长上下文模型的核心瓶颈」是无来源支撑的判断被写成结论，且未像本页其他推断（第 176/257/460 行）那样标注来源边界。｜引文依据：不适用（该句为归纳外推，非所引论文原文）｜修复要求：改为「在本例规模下……可解释长上下文模型为何受显存制约」一类的推断性表述，或补「以上为教学推断」边界说明。｜修复：｜复验：
- [轻微·技术] §1 构造示例（第 165 行）「单条请求就占满可观显存」与 overview.html 第 31 行「占满显存」：21.5 GB 的单请求 cache 在常见 40–80 GB 卡上只占一部分，「占满」用词过度。｜引文依据：不适用（21.5 GB 为本页构造推算 `2×128×128×4096×80×2≈2.15×10^10` 字节）｜修复要求：改为「占用可观显存 / 占去相当一部分显存」，避免「占满」。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 2
- 处置：修复
