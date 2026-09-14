<!-- review-meta
round: 7
page: wiki/gated-deltanet/index.html
reviewed_content_sha256: 3db03cfe92d6ecb1
-->
# Gated DeltaNet 审查记录（第 7 轮）

- 页面版本：index.html 5e9bfeaac8526143ba3734da6f969f8e6e4efde7；overview.html e4e47959d0d5b86c28184e21f59203a2688b580f
- 审查时间：2026-09-14 16:55
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节（按顺序）：核心问题 / 最容易误解 → 1. DeltaNet 与 Mamba2 各自缺什么——gating 与 delta 规则的互补性 → 2. Gated DeltaNet 的公式与符号——α_t 衰减与 β_t 擦写如何协同（含 2.1、2.2 与图示） → 3. 手算 Gated DeltaNet 一步更新——三模型在同一序列上并排对比（含 3.1、折叠块、Python 代码块） → 4. 退化关系与并行训练算法——为什么训练不能直接串行递归（含 4.1、4.2、代码折叠块） → 5. 与 KDA 的关系及实验效果——α_t 从标量到 channel-wise（含 5.1、5.2、5.3） → 来源与范围说明

## 核对来源留痕（本轮实际打开的原件）

- arXiv:2412.06464v1 全文 HTML（https://arxiv.org/html/2412.06464v1）：摘要、"gating enables rapid memory erasure while the delta rule facilitates targeted updates"（全文仅出现 1 次，位于摘要；§1 只有等义的 "Recognizing the complementary advantages of the gated update rule and the delta rule"）；§1 "the model lacks the ability to rapidly clear outdated or irrelevant information, especially during context switches where previous data needs to be erased"、"uniformly decays all key-value associations at each time step by a dynamic ratio"、"this approach does not account for the varying importance of different key-value associations"；§3.1 式 (8) `S_t = S_{t-1}(α_t(I − β_t k_t k_t^T)) + β_t v_t k_t^T` 及其后 "where the data-dependent gating term α_t∈(0,1) controls state decay"；§3.2 "Similar to Mamba2, the gating term ... only performs elementwise multiplication with (intermediate) variables without affecting matrix multiply structures, enabling tensor core GPU optimization"；§3.2 式 (9) `P^r_[t] = γ^r_[t](I − Σ w^i_[t] k^iT_[t])`；§3.3 脚注原文仅 "We use Mamba2's parameterization for α but omit it for brevity."；§4 设定 "1.3B parameters on 100B tokens sampled from the FineWeb-Edu dataset"；Table 2 全部数值（Transformer++ 18.53/18.32/52.25；Mamba2 16.56/12.56/54.89；DeltaNet 17.71/16.88/52.14；GDN 16.42/12.17/55.32；H1 16.07/12.12/56.40；H2 15.91/12.55/56.18）；Table 3（S-NIAH-2 4K: 18.6/56.2/92.2；S-NIAH-3 4K: 22.4/4.6/27.6；S-NIAH-1 4K DeltaNet 99.0）；§2.1 "Typical C = 64 (FLA)" 与 "C 取 16 的倍数以用 tensor cores"
- 复核 Avg. 定义：GDN 行 8 个准确率列 (46.65+72.25+55.76+57.45+71.21+38.39+40.63+60.24)/8 = 55.3225 ≈ 55.32，与页面 N3 所述"LAMBADA + 7 项常识共 8 项平均"一致
- arXiv:2412.06464v3 全文 HTML：gated delta rule 为式 (10)，证实页面"v1 编号 ≠ v3 编号"的说明成立
- arXiv:2406.06484v6 全文 HTML：§2.2 标题 "DeltaNet: Linear Transformers with the Delta Update Rule"（F4 定位成立）；作者 Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim（页面 F4 作者串 "Yang, Wang, Zhang, Shen & Kim" 正确）
- Qwen3-Next 官方博客（qwen.ai，标题 "Qwen3-Next: Towards Ultimate Training & Inference Efficiency"）：3:1 混合、Gated DeltaNet 占 75%、80B 总参数 / ~3B 激活
- https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json：full_attention_interval=4、num_hidden_layers=60、layer_types 为 45 linear_attention + 15 full_attention、attn_output_gate=true
- https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json：num_hidden_layers=93、69 层 KDA + 24 层 MLA（mla_use_output_gate=true）、linear_attn_config.gate_lower_bound=−5.0、use_full_rank_gate=true
- 手算与代码复算：按页面 F1/F3/F4 用 numpy 复算，S_3 三模型矩阵与三条查询向量全部吻合；抽取页面 Python 代码块实跑，退化验证两行均为 True

## 问题

- [轻微·来源定位] overview.html「为什么需要它」第 3 条：同一句引文在 index.html 正文标注为"摘要原文"（正确），在本页却标注为"论文 §1 原文"，两处来源位置标注互相矛盾。｜引文依据：v1 全文 HTML 中 "gating enables rapid memory erasure" 仅出现 1 次，位置在摘要（字符偏移 3985，位于 Abstract 段内）；§1 对应句为 "Recognizing the complementary advantages of the gated update rule and the delta rule in memory management, we propose the gated delta rule"，并非该引文原句。｜修复要求：把 overview.html 中该引文的出处由"论文 §1 原文"改为"论文摘要原文"（与 index.html 一致）。｜修复：｜复验：
- [轻微·来源定位] index.html §4.2「并行训练算法」："Yang 2025 ICLR §3.2 给出 chunkwise 并行算法……（$C$ 通常取 64，且为 16 的倍数以用 Tensor Core）"——括号内两个具体数值实际出自该文 §2.1（线性注意力 chunkwise 一节），§3.2 只给出针对 gated delta rule 的算法而未复述这两个数值。｜引文依据：v1 HTML 中 "by setting C to a multiple of 16, one can take advantage of tensor cores" 与 "Typically, C is set to a small constant (e.g., 64 as implemented in FLA)" 均位于 §2.1 的 "Chunkwise parallel form"；§3.2 正文无 "64"／"multiple of 16"。｜修复要求：将括号内出处明确为"§2.1"，或把该括注与 §3.2 的归属句拆开。｜修复：｜复验：
- [轻微·来源支持] index.html「最容易误解」第 4 条与 overview.html「关键结论与边界」："这是它相对早期 data-independent decay（如 RetNet 的固定 $\gamma$）的区别"——对 RetNet 衰减机制的描述未给出任何来源，而本页对 Qwen3-Next、Qwen3.5、Kimi K3、flash-linear-attention 等其他模型／实现的事实均带 [N4]–[N7] 引文。｜引文依据：不适用（该论断未标注来源；RetNet 固定 $\gamma$ 的说法本身正确，但页面无对应引文可定位）。｜修复要求：为 RetNet 固定 $\gamma$ 补一条来源引文，或删去该括注只保留"data-independent decay"的一般说法。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布。本轮未发现阻断或重要问题：核心公式 F1 与 v1 式 (8) 逐字一致；§1/摘要引文可定位且与原文相符；Table 2 / Table 3 全部数值与官方表格一致，N3 的 Avg. 定义经 8 项求平均复算成立；v1/v3 编号差异说明经 v3 原件证实；手算与 Python 代码块实跑结果一致（仅 numpy 版本导致的空白格式差异，数值全同）；KDA 的 channel-wise、下界 −5、full-rank gate、93 层中 69 层 KDA / 24 层 Gated MLA 均经官方 config.json 核实；图示为 HTML 结构、无 Unicode 数学字符、alt 无 `$…$`；表述维度通读未见元话语、会话指代、调试叙事或 AI 拼接腔；`validate.py` 返回 validation ok。3 条轻微问题均为来源定位／引用完备性层面的小瑕疵，不影响学习目标达成与主线理解，可在后续轮次修复。