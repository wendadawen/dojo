<!-- review-meta
round: 9
page: wiki/mqa-gqa/index.html
reviewed_content_sha256: d7d7e752fc9c7d92
-->
# MQA 与 GQA 审查记录（第 9 轮）

- 页面版本：index.html `d7d3ffe090f5ddab77e3a89d1a5e8e31338562c0`（overview.html `4abd8960fbb242211edd901af123eef7c6ee7bc4`）
- 审查时间：2026-09-14 17:54
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查）
- 已完整阅读章节：核心问题（5 问 5 答）、最容易误解、1. 为什么 MHA 推理受内存带宽限制、2. MQA、3. GQA、4. 手算对比、5. 边界与后续、来源与范围说明；含全部折叠块、图注与表格

## 核对方式与版本

- 来源版本：Shazeer 2019 arXiv:1911.02150（ar5iv 全文，章节号已逐一对齐）；Ainslie 2023 arXiv:2305.13245（EMNLP 2023，ar5iv 全文）；DeepSeek-V2 arXiv:2405.04434（ar5iv 全文）；DeepSeek LLM arXiv:2401.02954（ar5iv 全文）；NVIDIA Data Center GPU Line Card。
- 图内数值：本页结构图均为 HTML 表格/节点（无 `<img>` 位图），无数值需像素测量；公式与表格数字均逐一复算。

## 逐条来源核对（要点）

- Shazeer 2019 §2.4.1 ：「the ratio of memory access to arithmetic operations is Θ(n/d + 1/b)」「When n≈d or b≈1, the ratio is close to 1」——与正文 [F5]、本章问题答案一致。§3.1 ：「Θ(1/d+n/(dh)+1/b)」「We have reduced the offensive n/d by a factor of h.」——与 [C2] 逐字一致。§3 ：「the different heads share a single set of keys and values」——[C3] 逐字一致。abstract：「incur only minor quality degradation from the baseline」——[C5]/[N1] 一致。投影张量形状：论文 `P_q [h,d,k]`、`P_k [h,d,k]`、`P_v [h,d,v]`、`P_o [h,d,v]`，MQA 下 `P_k [d,k]`、`P_v [d,v]`——与正文第 212–221 行表格完全一致。
- Ainslie 2023 §2 ：「Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head.」——[C6] 逐字一致。§2.1 均值池化与「600 TPUv3 chip-days」在 §3.1——[C7]/[N4] 一致。§3.3（Ablations）含 Figure 4/5/6，Figure 4 理由句「results are ordered by the degree to which information is preserved from the pre-trained model」——与第 316–323 行一致。Appendix A：「multi-query attention can lead to training instability during fine-tuning, in particular combined with long input tasks... Uptrained grouped-query attention models, however, appear to be stable.」——一致。Table 1：MHA-Large 0.37/46.0、MHA-XXL 1.51/47.2、MQA-XXL 0.24/46.6、GQA-8-XXL 0.28/47.1——与第 333–336 行逐一相符；换算 1.51/0.24=6.3×、1.51/0.28=5.4×、47.2−46.6=0.6、47.2−47.1=0.1 均复核正确。Figure 6：「We selected 8 groups as a favorable middle ground.」「increasing the number of groups from MQA only results in modest slowdowns initially」——[N3] 一致。应用范围：「We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder self-attention.」——第 340 行一致。
- DeepSeek-V2 §2.1.4 Table 1：MHA `2n_h d_h l`、GQA `2n_g d_h l`、MQA `2d_h l`、MLA `(d_c+d_h^R)l≈9/2 d_h l`；超参 n_h=128、d_h=128、d_c=512、d_h^R=64。正文在明确标注「$n_h=h$、$d_h=d_k$，沿用 DeepSeek-V2 记法」并按「每 token 每 layer」归一（隐去 l），四行口径一致：MHA 2·128·128=32768、MLA 512+64=576、32768/576≈57、MQA 2·128=256——数字全部复算正确。
- 构造示例复算：512/256/128 元素、5120/2560/1280（10 token）、10240/5120/2560 字节（10 KB/5 KB/2.5 KB）全部正确；真实规模 2·128·128·4096·80·2≈2.15e10 B≈21.5 GB、GQA-8≈1.34 GB、MQA≈168 MB 复算正确；21.5 GB/168 MB≈128≈h 正确。
- NVIDIA 数字：V100 SXM2 FP32 15.7 TFLOPS/900 GB/s、A100 80GB SXM 19.5/2039、H100 SXM 67/3352——与官方口径一致；67/15.7≈4.3×、3352/900≈3.7× 复算正确。
- 机械项：`validate.py` 返回 `validation ok`；无 Unicode 数学字符；无「待生成/TODO」占位；`body > img` 仅 lightbox 占位（`alt=""`），无 `$...$` 出现在 alt；链接 `../standard-attention/index.html`、`../mla/index.html`、`../../index.html`、`overview.html` 与 libs 资源均真实存在；overview.html 与 index.html 互相链接；两级问题（核心问题 + 每章本章问题）每题均有折叠解答，核心问题答案均指明完整论证所在章节。术语/符号单义，折叠块收起后正文仍可建立结论。

## 问题

- [轻微·技术] 「5. 边界与后续」，index.html 第 484 行：把「DeepSeek 67B 的 GQA 基线」这一识别写成 DeepSeek-V2 abstract 的内容，属引文归属不精确（论断本身为真）｜引文依据：DeepSeek-V2 abstract 仅 "Compared with DeepSeek 67B, DeepSeek-V2 … reduces the KV cache by 93.3%"，无「GQA」字样；GQA 识别出自 DeepSeek LLM（arXiv:2401.02954）"To optimize inference cost, the 67B model uses Grouped-Query Attention (GQA)"（64 heads / 8 KV groups）｜修复要求：将该括号内出处的「GQA 基线」改标为 DeepSeek LLM 67B，或删去「GQA」只写「对比 DeepSeek 67B」｜修复：｜复验：
- [轻微·表述] index.html 第 65、74、132、172、176 行：口语化短句「GPU 在等数据/等内存而非算」共 5 处重复，构成固定句式｜引文依据：不适用｜修复要求：保留 1–2 处，其余改为「程序受内存带宽限制」等中性表述｜修复：｜复验：
- [轻微·可读性] index.html 第 65 行（引言）：HBM 与 fp16 在首次出现处未解释，均延后数十行才给出全称/字节数（HBM 定义在第 122 行、fp16「每元素 2 字节」在第 165 行）｜引文依据：不适用｜修复要求：把 HBM 中文全称、fp16 字节数移到各自首次出现处，或首次出现时括注｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（仅轻微项，阻断与重要均为 0）
