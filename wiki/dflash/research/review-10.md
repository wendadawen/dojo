<!-- review-meta
round: 10
page: wiki/dflash/index.html
reviewed_content_sha256: 765421beb83cc1a3
-->
# DFlash 审查记录（第 10 轮）

- 页面版本：87b66aff029d360681ab00229c36e7f964ffc186（工作树 index.html）
- 论文版本：arXiv:2602.06036v2（2026-05-28 修订，ICML 2026 accepted）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：head 元信息 / 核心问题（5 题）/ 1. 起草为什么慢 / 本章问题 / 2. 推理管线 / 本章问题（含折叠块）/ 3. 训练 3.1–3.5 / 本章问题（含 loss decay 权重表）/ 4. 实验 4.1–4.6 / 本章问题 / 5. 方法评价（9 行维度表）/ 本章问题 / 来源与范围说明（C/F/N 全表、构造示例、简化条件）/ 全部图注与 alt / 全部脚本。overview.html 一并阅读。

## 核对方式

- 论文原文：`https://arxiv.org/html/2602.06036v2` 全文 HTML（已下载，逐表逐式核对）；生态数字取 Inco AI 官方博客 `https://inco.ai/blog/dflash2/`。
- 图内数值：对 assets/img-01.webp、img-04.webp 做像素测量（用 y 轴刻度标签行标定，img-04 为 8.5 px/单位）。
- 机械项：`python3 .dojo/scripts/validate.py wiki/dflash/index.html` → `validation ok`。

### 关键核对依据（逐条对照原文片段/数值）

- 公式 F1/F2/F3：Eq.(1) `L=(T_draft+T_verify)/τ, τ∈[1,γ+1]`；Eq.(2) `T_draft=γ·t_step`；Eq.(3) `T_draft=t_parallel, t_parallel≪γ·t_step`。符号与取值范围与页面一致。
- F4/F5：A.3 原文 `H_t=RMSNorm(W_c[H^(l1);…;H^(l5)])`、`Q_i=W_i^Q H_d`、`K_i=[W_i^K H_t; W_i^K H_d]_seq`、`V_i=[W_i^V H_t; W_i^V H_d]_seq`；`only serve as additional KV entries … bypass the draft model's Q projection, output projection, self-attention update, and FFN`。与页面逐字吻合。
- F6：Eq.(4) `w_k=exp(-(k-1)/γ)`；A.1 `γ … 7 for block size 16, 5 for block size 10, and 4 for block size 8`。页面 loss decay 权重表 16 个值手算复算全部吻合（1.000…0.117，位置 1/16 ≈ 8.5 倍）。
- 主表（Tab. 1）逐行核对：Q3-4B T=0 EAGLE-3(16) 1.81×/3.05、EAGLE-3(60) 2.08×/3.48、DFlash 4.91×/6.54（Math500 6.09）；Q3-8B T=0 1.76×/2.96、2.02×/3.40、4.86×/6.49（Math500 6.08、MT-Bench 2.75）；T=1 四行 4.24/4.03、EAGLE-3 1.72/1.93、1.68/1.88 等全部与页面表格一致。折算列 4.91/1.81=2.71、4.86/1.76=2.76、4.24/1.72=2.46、4.03/1.68=2.40 → 页面「2.4–2.8×」成立。
- Tab. 3（SGLang）：基线 316/312/230/229/229/220/228、τ 8.01/6.63/8.01/6.50/8.09/6.42/7.23、conc1 与 conc32 各值全部吻合；论文 `single B200 GPU with the FlashAttention-4 (FA4) backend … Spec-v2 scheduling overlap`。
- Tab. 5（LLaMA-3.1-8B）：9 行 conc1–32 与 τ 全部吻合；论文 `using the exactly same training data as EAGLE-3`、`Spec-v2 does not support tree-based drafting for EAGLE-3`、`7 draft steps with top-k=10`。
- Tab. 6/8/9/10/12/13：3L 4.69/5.64、5L 4.71/5.99、8L 4.64/6.33；b16→b16 6.33、b16→b8 5.09、b8→b8 5.21、b8→b16 5.02；消融四行 τ/加速 4.2/2.1…4.2/3.3；Tab.10 T=0 2.83/3.73/3.43/3.35；vLLM 并发 1–32 的 3.0–4.6→1.3；anchor Sample vs Standard 5.64/4.94、4.61/3.86、3.18/2.80。页面全部一致。
- Tab. 4（长上下文）：hotpotqa 4K Base 4.91 → 16K Base 3.61、16K Long 6.05；`fine-tune … 1.6K samples from LongAlign-10K for 3 epochs`。
- Tab. 2（thinking）：6 个 T=0 值均值 4.42、6 个 T=1 值均值 3.79 → 页面「约 4.4×/3.8×，论文概述 roughly 4.5× and 3.9×」成立。
- 显存（N12）：A.3 `5×2048×2048×2 ≈ 42 MB … roughly 70 GB target model`、`about 40 MB and 8 MB`、`below 400 KB`。
- 生态（N16）：博客 `up to 15× throughput`（Blackwell）、`3× more tokens per second`（TPU）、`more than 3.5 million times`（下载）、`runs in SGLang, vLLM, TensorRT-LLM, and llama.cpp`；DFlash 2 的 path selector 与 two-tap dynamic convolution。均标为厂商宣称。
- 硬件：论文 `We conduct all experiments on NVIDIA H200 GPUs unless otherwise specified.` + `single H200 GPU`、`single B200 GPU` → 页面「单卡 H200/B200」成立。
- 原图：Figure 1 各柱标注值（5.15/6.08/5.62/5.14/4.65/5.51/2.75 与 EAGLE-3 2.23…1.90）与页面 alt/图注逐项吻合，EAGLE-3 行＝Tab. 1 EAGLE-3(60) 行；Figure 2/4 的色块图例与图注一致；Figure 3 像素测量 EAGLE-3 ≈6.2/11.3/25.5 ms、DFlash ≈1.5–5.3 ms，与 alt「约 6 ms→约 25.6 ms」「约 1.7–5.5 ms」在 1–2 px（约 0.1–0.2 ms）内一致，视为读数容差内的近似，不记为问题。
- 链接：`..//speculative-decoding|standard-attention|block-diffusion|eagle-speculative|dflash2/index.html` 五个前置概念页真实存在；github.com/z-lab/dflash、hf.co/collections/z-lab/dflash、inco.ai/blog/dflash2/ 均 200；无「（待生成）」占位。alt 属性无 `$…$`；无 Unicode 数学字符（validate.py 通过）。
- 表述维度：通读正文、全部折叠块与图注，未见「本页将…」「下面来看…」「需要注意的是」类元话语，无第一/第二人称会话指代，无调试叙事与临场评价；「本文」仅出现在「来源与范围说明」（用于区分论文与解析），与站内 44/98 页的既有用法一致，不记为问题。

## 问题

- [轻微·技术] 「来源与范围说明·核心论断与来源」C5 条目（index.html:539）与正文 §2 末段（:219）、§5 评价表负结果行（:493）：同一量（无 KV 注入的纯块扩散草稿器加速比）在正文/评价表写作「约 2.8–3.7×」，C5 条目却写作「2–3×」，两者区间不重合（3.7 落在「2–3」之外）。｜引文依据：论文 Table 10（Temp 0）GSM8K 2.83 / Math500 3.73 / AIME24 3.43 / AIME25 3.35；§4.1 段尾 "the resulting speedups are modest, typically around 2–3×"。页面 C5 沿用了论文正文概述、正文用了表值，两处标法未统一。｜修复要求：把 C5 条目的数值改为与正文一致的「约 2.8–3.7×（论文正文概述为 2–3×）」，或反之统一为「2–3×」并在正文标注为论文概述。｜修复：｜复验：
- [轻微·技术] 「5. 方法评价」未做行的「依据」列（index.html:494）：把「未与 TiDAR / DiffuSpec / SpecDiff-2 实验对比（开源实现缺失）」的出处标为「§5.1 Baselines」，但该 Baselines 段位于 §5 Experiments 引言（在「5.1 Instruct Models」标题之前），不在 §5.1。｜引文依据：§5 引言 "Baselines. We compare DFlash with the vanilla autoregressive decoding … We did not include comparisons with other dLLM-based speculative decoding methods (Liu et al., 2025; Samragh et al., 2025; Li et al., 2025a; Sandler et al., 2025) due to lack of open-source implementation."；其后才出现 "5.1 Instruct Models" 标题。｜修复要求：该行（含同单元格正文里的 `<sup>[§5.1]</sup>`）改为「§5 Baselines 段」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（两条轻微问题不影响正确性与主线理解，建议随下一轮一并清理）
