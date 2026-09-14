<!-- review-meta
round: 7
page: wiki/dflash2/index.html
reviewed_content_sha256: f8a04cab8f70b61f
-->
# DFlash 2 审查记录（第 7 轮）

- 页面版本：7de3fb5d2d5fb02da23ed7dbf2208d1b19c485d7
- 审查时间：2026-09-14 14:35
- 审查者：独立子代理（未参与写作，也未参与前序轮次）
- 已完整阅读章节：引言与「核心问题」（5 个解答折叠块）→ 1. 两个剩余问题——候选池里有答案、块末端在漏气（含构造网格与本章问题）→ 2. 路径选择器（2.1 打分公式、2.2 全并行、2.3 与 DSpark 对比、2.4 最小例子，含本章问题）→ 3. 两抽头卷积（3.1 证据链、3.2 公式与 SVG 图示、3.3 效果表与折叠块，含本章问题）→ 4. 组合效果（4.1 Qwen3.5-4B、4.2 Qwen3.8-27B 与 Muse Glimmer、4.3 成本总账，含本章问题）→ 5. 端到端与边界（5.1 并发×任务表、5.2 机制、5.3 部署注意，含本章问题）→ 来源与范围说明（C/F/N 表、构造示例、类比边界、简化条件）。`overview.html` 一并核对。

## 来源核对（本轮实际打开的来源）

- Inco AI 博客《DFlash 2: Keep Drafting Parallel》2026-08-18 全文（正文 + Table 1/2/3/4/5 + Figure 2/3/5 数据表 + Citation `@misc{inco2026dflash2}` + Run It Now 代码块）。
- HF 模型卡 `incoai/Qwen3.8-27B-DFlash2` README（YAML：`license: apache-2.0`、`library_name: transformers`、`base_model: Qwen/Qwen3.8-27B`；正文 "It is not a standalone language model"；"Decoding is lossless: greedy output matches the target model exactly, and sampling preserves its distribution"；Evaluation 条件含 `xhigh` reasoning effort / 4096 max new tokens；Concurrency 1/8/32 三张吞吐表）。
- HF 模型卡 `incoai/Muse-Glimmer-30B-DFlash2` README（YAML `license: apache-2.0`、`base_model: meta-models/Muse-Glimmer-30B`）。
- arXiv:2602.06036v2（DFlash: Block Diffusion for Flash Speculative Decoding，Chen / Liang / Liu，ICML 2026；§5.5.2 Number of Draft Layers：Table 6 caption "5-layer draft model has the best average speedup"）——用于核对正文「5 层是经验最优，论文另有层数消融」。
- 机械项：`.dojo/scripts/validate.py wiki/dflash2/index.html` → `validation ok`；`dojo:topics=训练与优化`、`dojo:tag=推理加速` 均在词表内；`#sources-and-scope-notes`、`../dflash/index.html#inference-pipeline`（对端 h2 `2. 推理管线——KV 注入与单步并行起草` 存在）、`../block-diffusion/`、`../speculative-decoding/` 链接全部有效；C1–C17 / F1–F2 / N1–N10 与正文 `<sup>` 引用双向一一对应（无孤儿、无未定义）；`alt` 中无 `$...$`；无 Unicode 数学字符（`×` 按 validate.py 第 53 行规则属散文排版字符）。

逐项回源的结论：Table 1（85.4/80.3/79.4/78.3/77.5/75.9/72.9 与 99.5/97.3/94.8/92.6/90.8/89.4/87.8、4.27→6.79）、Figure 2（85.21/85.39/86.42/85.83 与位置 3 的 75.75/78.27/80.34/79.68、末位 64.97/72.86/78.73/77.61）、Table 2（+77.8M/+9.6%/4.49/4.08 与 +2.0M/+0.6%/4.61/4.25）、Table 3（4.78/4.99/5.69/6.20 各列与均值 4.54/4.92/5.49/5.97）、Table 4（4.28/3.62/4.80）、Table 5（4.44/4.48/5.70）、Figure 5（DFlash 2 88.3→86.48，末位 MTP 77.85 / DFlash 77.48 / DSpark 79.86）、模型卡三张吞吐表全部 30 个倍数格与自回归基线 68.9/69.0/69.0/69.0/68.9、机制数字（30%→8%、9.4%→0.5%、16.5M/3%、15.2%、1.3%、16–25%、40×/16×、+1.18/+1.22、+21%/+1.05/+0.48/+1.43）均与来源逐字一致；Qwen3-4B（Table 1/2/Figure 2）与 Qwen3.5-4B（Table 3）两款目标模型全页未混用；未发现阻断或重要级事实问题。

## 问题

- [轻微·技术] 第 1 章末段（构造示例小结）：「真实数据里位置 1、2 的 Recall@1 已从首位下降（80.3%/79.4% vs 85.4%，见 N2），说明后段候选池本身在变差——靠选择器救不回来」——用 Recall@1 下降支撑「候选池本身变差」这一结论，两者不是同一个量：Recall@1 是 draft 独立 top-1 命中率，候选池覆盖由 Recall@16/oracle 度量；本节前段与「本章问题 2」正确使用的正是 Recall@16 99.5%→87.8% 与 oracle 衰减。证据与结论不匹配。｜引文依据：博客 Table 1「Recall@1 85.4% … 72.9%」「Recall@16 99.5% … 87.8%」；博客「Even the oracle decays: with perfect selection, accuracy still falls from 99.5% at the first position to 87.8% by the last. No selector can fix that」。｜修复要求：把该句的证据换成 Recall@16 沿块下降（99.5%→87.8%）或 oracle 自身衰减，或删去「说明后段候选池本身在变差」这一归纳，只保留「位置 1、2 的 Recall@1 已低于首位」。｜修复：｜复验：
- [轻微·技术] 5.1 要点第三条：「并发 32 时 MTP 与 DSpark 多数任务降到 1× 以下（最差的 MT-Bench 0.74–0.77×）——这些方法在批处理填满空闲算力后变成纯浪费，不能用」——前半句有模型卡支撑，后半句的机制与「不能用」的行动判断在本页 C14 与 5.2 节被明确定义为分析性推断，此处却以事实句给出，未带推断标注。｜引文依据：HF 模型卡 Throughput 表仅给倍数（并发 32：MTP 1.04/0.94/0.84/0.87/0.77×，DSpark 1.13/0.95/0.86/0.90/0.74×），无因果分析；博客亦未给出该机制。｜修复要求：在该要点内补「（分析性推断，见 C14）」标注，或删去「变成纯浪费，不能用」只保留「多数任务降到 1× 以下」这一可核对事实。｜修复：｜复验：
- [轻微·可读性] 2.4 节走路径列表前的括注：「（bonus 即验证顺带产出的下一 token；本节不区分验证 bonus 与拒绝修正 token 的来源）」——「bonus」全页仅此一处出现，构造示例（3 位置、无已验证前缀、只走 draft 位置）中不存在「验证 bonus 与拒绝修正 token」的区分对象，读者无法据此判断该步取值范围。｜引文依据：不适用｜修复要求：删去该括注，或改写为「本示例只覆盖 draft 块内走路径，不含验证阶段产出的 token」。｜修复：｜复验：
- [轻微·表述] 元话语与自我指代：引言段「以下按这两个组件的机制、证据和边界展开」（元话语）；5.2 节「本节对该方向性事实给出机制解释，明确标注为分析性推断」（以「本节」为主语陈述写作安排）；来源与范围说明中 C5「本文按作者设计论证引用」与简化条件⑥「本页面所有「分析性推断」均已显式标注为推断」（以「本文/本页面」为主语的自我指代）。｜引文依据：不适用｜修复要求：把「以下按…展开」并入直接陈述（如「两个组件的机制、证据和边界分别见第 2、3 章」）；C5 的「本文按作者设计论证引用」并入来源条目本身（如「该句为作者的设计论证，非实测结论」）；⑥ 改为无主语句「页面内的分析性推断均已标注为推断」。｜修复：｜复验：
- [轻微·表述] 5.3 节末段以「生态一句：」起句，为标题式残句＋口语化标签，与全页技术散文语体不一致。｜引文依据：不适用｜修复要求：改为完整句，如「DFlash（一代）的生态与厂商自测数字：…」。｜修复：｜复验：

## 结论

- 处置：修复（仅 5 项轻微；阻断与重要均为 0，不阻碍发布）
- 统计：阻断 0 / 重要 0 / 轻微 5