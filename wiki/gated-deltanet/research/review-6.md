<!-- review-meta
round: 6
page: wiki/gated-deltanet/index.html
reviewed_content_sha256: d1ff30a3cf404075
-->
# Gated DeltaNet 审查记录（第 6 轮）

- 页面版本：825aeaef71766e0b2386a4b1dae65589d507144a
- 审查时间：2026-09-13 21:13
- 审查者：独立子代理（未参与写作与前序轮次审查；本轮仅读取 index.html、overview.html、页面引用的外部来源与本规范）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. DeltaNet 与 Mamba2 各自缺什么 / 2. Gated DeltaNet 的公式与符号（含 2.1、2.2）/ 3. 手算 Gated DeltaNet 一步更新（含 3.1、图注、展开折叠块）/ 4. 退化关系与并行训练算法（含 4.1、4.2、补充折叠块、代码折叠块）/ 5. 与 KDA 的关系及实验效果（含 5.1、5.2、5.3）/ 来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制），全部折叠块与图注逐段读完

## 来源核对（每条定位到原文片段，作为核对依据）

- C1（§1）：原文 "However, since this process only modifies a single key-value pair at a time, the model lacks the ability to rapidly clear outdated or irrelevant information, especially during context switches where previous data needs to be erased." 与页面引文逐字一致（含主语 "the model"）。通过。
- C2（§1）：原文 "which uniformly decays all key-value associations at each time step by a dynamic ratio, αt" 与 "However, this approach does not account for the varying importance of different key-value associations, potentially leading to inefficient memory utilization." 页面引文一致（后半句截断于 "associations"，未改写原意）。通过。
- C3 / F1（§3.1 Eq.8，v1）：原文 "The proposed gated delta rule is simple yet effective: 𝐒t = 𝐒t−1(αt(𝐈−βt𝒌t𝒌t⊺)) + βt𝒗t𝒌t⊺, where the data-dependent gating term αt∈(0,1) controls state decay." 页面公式与此完全相同。v3（ICLR 定稿）同一公式编号为 Eq.10、仍在 §3.1 —— 页面 meta 说明"与 v3 编号不同"属实。通过。
- C4（§3.3 脚注）：原文脚注 "We use Mamba2's parameterization for α but omit it for brevity."；§3.3 标题为 "Gated Delta Networks and Hybrid Models"。页面"§3.3 脚注说明 α_t 参数化沿用 Mamba2，正文未展开"属实。通过。
- C6（§3.2 Eq.9-12）：原文 "the gating term (colored in blue) only performs elementwise multiplication with (intermediate) variables without affecting matrix multiply structures, enabling tensor core GPU optimization"；Eq.9 原文 "𝐏_[t]^r = γ_[t]^r (𝐈 − Σ_{i=1}^r 𝐰_[t]^i 𝒌_[t]^i⊺)"、γ 为 chunk 内累积衰减。页面折叠块的 P 公式与 γ 定义一致。通过。
- C7（摘要/§3.1）：摘要原文 "We observe that these mechanisms are complementary—gating enables rapid memory erasure while the delta rule facilitates targeted updates."（破折号，与页面一致）。通过。
- C8（K3 报告 §2.1.1）：Eq.1 原文 "𝐒t=(𝐈−βt 𝒌t𝒌t⊤) Diag(𝜶t) 𝐒t−1+βt 𝒌t𝒗t⊤, 𝒐~t=𝐒t⊤𝒒t"；Eq.5 原文 "𝒈th=gmin Sigmoid(e^{Ah}𝒛th)∈(gmin,0)^{dk}, 𝜶th=exp(𝒈th)∈(e^{gmin},1)^{dk}"，并给出 "αt,j > e−5 ≈ 6.7×10−3"、BF16 动态范围论证。页面 KDA 公式、e^{−5}≈0.0067 与 g_min=−5 均一致。通过。
- F3（§2.2）：该节标题 "Mamba2: Linear attention with scalar-valued data-dependent decay"，原文 "𝐒_t = α_t 𝐒_{t−1} + 𝒗_t 𝒌_t^⊺"。页面 F3 的 §2.2 定位与公式一致。通过。
- F4（arXiv:2406.06484 §2.2）：该节标题 "DeltaNet: Linear Transformers with the Delta Update Rule"，β_t 定义 "βt = σ(𝐖β𝒙t) ∈ (0,1)"。页面 F4 的 §2.2 定位与公式形式一致。通过。
- N1 / N3（§4 Table 2）：Wiki/LMB PPL 与 Avg. 逐一核对：Transformer++ 18.53/18.32/52.25；Mamba2 16.56/12.56/54.89；DeltaNet 17.71/16.88/52.14；Gated DeltaNet 16.42/12.17/55.32；H1 16.07/12.12/56.40；H2 15.91/12.55/56.18 —— 与页面正文表及 N1/N3 完全一致。表题 "Performance comparison on language modeling and zero-shot common-sense reasoning."，Avg. 为 LAMBADA 加 7 项常识（PIQA/HellaSwag/WinoGrande/ARC-e/ARC-c/SIQA/BoolQ）共 8 项平均，与 N3 描述一致。通过。
- N2（§4 Table 3）：S-NIAH-2 4K：DeltaNet 18.6 / Mamba2 56.2 / Gated DeltaNet 92.2；S-NIAH-3 4K：22.4 / 4.6 / 27.6 —— 与页面一致。论文原文确认 S-NIAH-1 为 passkey（合成）、S-NIAH-2 "number in haystack"、S-NIAH-3 "word in haystack"（后两者 needle 落在真实文本），页面 §5.2 的 "number/word 针、needle 落在真实文本中" 与 "DeltaNet 在 S-NIAH-1 上接近完美、S-NIAH-2/3 显著下降" 均有原文支持。通过。
- 关于 H1/H2：原文 "combine linear recurrent layers with sliding window attention (SWA), resulting in GatedDeltaNet-H1" 与 "We also stack Mamba2, GatedDeltaNet and SWA, resulting in GatedDeltaNet-H2"。页面 "H1=GDN+SWA、H2=GDN+Mamba2+SWA" 属实。通过。
- N4（Qwen3-Next）：官方资料确认 3:1 混合（75% 层 Gated DeltaNet / 25% 层 Gated Attention）、引用 Yang et al. 2025、Qwen3-Next-80B-A3B 总参数 80B / 激活约 3B。通过。
- N6（Qwen3.5-397B-A17B config.json）：字段 full_attention_interval=4、num_hidden_layers=60、layer_types 每 4 层为 3×linear_attention + 1×full_attention（重复 15 次）→ 45+15。页面数字一致。通过。
- N7（moonshotai/Kimi-K3 config.json）：num_hidden_layers=93、kda_layers 69 个、full_attn_layers 24 个、gate_lower_bound=−5.0、use_full_rank_gate=true。页面 "93 层中 69 层 KDA、24 层 Gated MLA" 与 g_min=−5 一致。通过。
- N5：NVlabs/GatedDeltaNet 仓库存在，为论文官方（NVIDIA）实现，README 推荐 flash-linear-attention 库；页面表述一致。通过。
- 代码（§4 折叠块）：实际执行页面所列 Python（numpy 标量循环）。输出逐行核对：DeltaNet S_3=[[0,0],[1,1]]、@k_3=[0,1]、@k_2=[0,1]；Gated DeltaNet S_3=[[0,0],[1,0.5]]、@k_3=[0,1]、@k_2=[0,0.5]；Mamba2 S_3=[[0.5,0],[0,0.5]]、@k_3=[0.5,0]、@k_2=[0,0.5]；两条退化验证均 True。与"预期输出"块逐字一致。通过。
- 手算（§3 及 3.1 对比表、§3 三条本章问题、展开折叠块）：逐步复算 S_1=[[1,0],[0,0]]、S_2=[[1,0],[0,1]]、S_3=[[0,0],[1,0.5]]；三模型 S_3 及 S_3 k_3 / S_3 k_2 全部复算相符；本章问题解答中 S_t=[[0.5,1],[0,0]] 及两次查询结果均正确。通过。
- 退化（§4.1、§2.2 边界检查）：α_t→1 得 DeltaNet、β_t→0 得 α_t S_{t−1}、α_t→0 得 β_t v_t k_t^⊺，代入均正确。通过。
- 页面功能与格式：validate.py 返回 "validation ok"；链接 ../delta-rule/、../linear-attention/、../kda/、../qwen3-5-dataflow/、../../index.html、overview.html 均真实存在；无"（待生成）"占位；结构图为 HTML（dg-flow），非等宽字符框线；无 img（唯一 img 为 lightbox 空 alt），alt 中无 `$...$`；dojo:type=concept、dojo:topics=注意力机制、dojo:tag、dojo:summary（LaTeX 可渲染）、description（纯文本）齐备；overview.html 与 index.html 双向互链。通过。

## 问题

- [轻微·表述] 章节衔接（第 178、289、393、567 行四处章末过渡段）：四处过渡共用同一模板"本章〔动词〕了 X。但 Y——下一章〔动词〕 Z。"，属 style-guide §8 所禁的固定句式，且以"本章"为主语作自我指代。｜引文依据：不适用｜修复要求：至少改写其中两处，去掉"本章…了…"的固定起句，改为直接给出结论与下一问题，四处不再共用同一句式骨架。｜修复：｜复验：
- [轻微·表述] 第 186 行："公式出现前先说清它解决的问题：上一章看到 DeltaNet 缺全局遗忘、Mamba2 缺方向擦除。"｜该句以版面组织顺序（"公式出现前"）为主语作元话语，且实际位于 F1 公式（第 184 行）之后，与所述顺序不符。｜引文依据：不适用｜修复要求：删去"公式出现前先说清它解决的问题："这一元话语引导语，直接陈述 Gated DeltaNet 的设计动因（把标量 α_t 嵌进 delta 擦除项前）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布。全部来源论断（C1–C8、F1–F4、N1–N7）均已定位原文片段核对通过；全部数字（Table 2 的 Wiki/LMB PPL 与 Avg.、Table 3 的 S-NIAH-2/3、Qwen3-Next 与 Qwen3.5/Kimi K3 config 字段）与官方材料一致；手算、代码输出、退化关系逐一复算相符；无页内自相矛盾、无无来源结论、无失效链接、无占位。余 2 条轻微问题不推翻任何核心结论与主线理解，修复或标注接受理由后即可发布。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
