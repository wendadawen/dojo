<!-- review-meta
round: 5
page: wiki/kda/index.html
reviewed_content_sha256: 1bf0e68faf4e3d45
-->
# Kimi Delta Attention（KDA）审查记录（第 5 轮）

- 页面版本：c51b044a389979012b5b34c3172f0620507ff0e4（git hash-object wiki/kda/index.html）
- 审查时间：2026-09-13 20:16
- 审查者：独立子代理（编排者派发，未参与写作与前序轮次审查）
- 来源核对方式：以 curl/WebFetch 取回原始来源——Kimi K3 Technical Report（arXiv:2607.24653v1 的 HTML 与 PDF 两种版本，核对 §2.1「Hybrid Attention」、§2.1.1「Kimi Delta Attention」Eq.1–6 与 Fig.3、§2.1.2、§5.1.1/§5.1.2，以及文末参考文献表）与官方 config.json（https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json，逐字段解析）。仅依据当前 index.html、overview.html 与本规范判断；未读取本页 research/ 下任何文件（含前序审查记录）。
- 已完整阅读章节（按顺序，含折叠块与图注）：核心问题 / 最容易误解 / 1. 为什么 K3 需要 KDA——KV cache 爆炸与线性注意力的"记不清" / 2. KDA 的递归核心——delta rule 加 channel-wise forget gate / 3. K3 的关键改动——lower-bounded decay（3.1–3.4）/ 4. 参数化与 full-rank output gate——把递归包成可训练的一层（4.1）/ 5. chunkwise 并行形式——chunk 内并行 + chunk 间递归（5.1–5.3）/ 6. KDA 在 K3 中的配置与边界——69+24 层的 3:1 交替（6.1–6.4）/ 来源与范围说明

## 回源核对通过的关键项

- Eq.1（行 239）：报告原文 `St = (I - βt kt kt⊤) Diag(αt)St-1 + βt kt vt⊤, õt = St⊤qt`，与页面逐字一致（含 Diag(αt) 在擦除项右侧、先于 delta 擦写的次序）。
- Eq.2 参数化链（行 453–457）、Eq.3 累积衰减（行 539）、Eq.4 chunkwise（行 557–559）、Eq.5 scaled sigmoid / g_min=-5（行 354）、Eq.6 full-rank gate `yt = Wo[Sigmoid(Wg xt) ⊙ RMSNorm(õt)]`（行 496）均与报告 §2.1.1 逐字一致。
- config.json 逐字段与 §6.2 表（行 641–656）一致：head_dim=128、num_heads=96、short_conv_kernel_size=4、gate_lower_bound=-5.0、use_full_rank_gate=true、hidden_size=7168、max_position_embeddings=1048576、num_hidden_layers=93；kda_layers 恰 69 项、full_attn_layers=[4,8,…,88,92,93] 恰 24 项。层布局「22 个完整 3:1 块（88 层）+ 末尾 3 KDA（89-91）+ 末尾 2 Gated MLA（92,93）」与 config 一致；报告 §2.1 原文亦为「Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3:1 mixing ratio … An additional Gated MLA layer is placed at the end of the backbone」。
- 数值全部可复算：5.15×10^10 B≈48 GB（2^20×96×128×4）、128×128×2≈32 KB、单层 96 头≈3 MB、e^-5≈6.7×10^-3、e^80≈5.54×10^34、e^160≈3×10^69、Softplus(1)≈1.313→α≈0.269、Sigmoid(1)≈0.731→g≈-3.655→α≈0.0259；§2 手算 S_1/S_2 三步与 §5 C=3 chunk 的 Γ 连乘（0.35/0.48/0.855/0.12）均与标注一致。
- §5.1.1（FlashKDA）/§5.1.2（KCP）小节号、Fig.3(a)(b) 两面板含义、行 344「Kimi Linear 沿用 GDN / Mamba-2 的映射」均有报告原文支撑（`Following GDN and Mamba-2, Kimi Linear uses the negative-Softplus mapping … [138, 24, 63]`）。
- 链接有效：wiki/delta-rule/index.html、wiki/linear-attention/index.html 均存在；无「（待生成）」占位；正文与来源说明未指向 research/ 下文件。validate.py 返回 validation ok。overview.html 与 index.html 双向互链。

## 问题

- [重要·技术] 主要依据（行 105）与正文（行 485、533、719、749）：全页 5 处把「Kimi Linear 论文」标注为 [64]，但 K3 报告参考文献表中 [64] 是 MLIR，Kimi Linear 是 [63]。该编号出现于核心对比来源（KDA 相对 Kimi Linear 的两处改动），读者按 [64] 会定位到无关的编译器论文。｜引文依据：报告参考文献表 `[63] Kimi Team et al. Kimi Linear: An Expressive, Efficient Attention Architecture. 2025. arXiv: 2510.26692 [cs.CL].` 与 `[64] Chris Lattner et al. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation". In: 2021 IEEE/ACM CGO. 2021, pp. 2–14.`；正文对应处亦为 `Following Kimi Linear [63]`、`the low-rank parameterization used by Kimi Linear [63]`。｜修复要求：把全页（行 105、485、533、719、749）的「[64]」改为「[63]」。｜修复：已把全页指向 Kimi Linear 的引用编号 [64] 一律改为 [63]（主要依据、ShortConv 继承、chunkwise 借用、$V_e$ 来源、[F4]、UT 变换未展开，共 6 处 7 个字面；行 565 的「来自 [64]」同属 Kimi Linear 引用，一并改正）。｜复验：
- [轻微·表述] 行 108、172、209、252、533、600、663、664、753：以「本页/本文」作主语的自我指代与元话语（如「本文回答：KDA 的递归长什么样…」「本页直接用它的结论」「本文聚焦 KDA 的机制与 K3 的两处改动…本页不展开」「本页按 K3 报告约定用…」「本页只讲形式」「本页不展开」「本文从 KV cache 爆炸问题出发」）。｜引文依据：不适用｜修复要求：逐处删除或改写为无主语的客观表述（如「KDA 的递归长什么样」「这里直接用该结论」「Gated MLA、Attention Residuals、FlashKDA kernel 不展开」）。｜修复：已逐处改写：行 108「本文回答：」改为「要回答的问题有三个：」；行 172「本页直接用它的结论」改为「这里直接引用该页结论」；行 209「本文聚焦…都不在本页展开」改为「Gated MLA、Attention Residuals 与 FlashKDA kernel 属于其它机制，不展开」；行 252「本页按 K3 报告约定用」改为「按 K3 报告约定则用」；行 533「本页只讲形式」改为「这里只讲形式」；行 600、663、664「本页不展开」改为「不展开」；行 753「本文从 KV cache 爆炸问题出发，拆解了」改为「从 KV cache 爆炸问题出发，KDA 的机制可拆解为」。｜复验：
- [轻微·技术] 行 108：「K3 相对直接前身 Kimi Linear 做了两处改动」把 Kimi Linear 说成 K3 的前身模型；Kimi Linear 是 KDA 机制的设计来源，报告把 K3 的（模型）前身写作 Kimi K2 / K2.5。｜引文依据：报告 §2.1.2 原文 `Unlike Kimi K2 and Kimi K2.5, Kimi K3 follows the hybrid design of Kimi Linear [63]`。｜修复要求：改为「K3 的 KDA 相对其来源 Kimi Linear 做了两处改动」或同等表述。｜修复：已改为「K3 的 KDA 相对其来源 Kimi Linear 做了两处改动」。｜复验：
- [轻微·技术] 行 600：「比纯串行快 $C$ 倍」把并行度写成无来源的整体加速结论；报告只说 chunkwise 形式 chunk 内并行、chunk 间串行，未给出 C 倍加速，且 chunk 间仍是串行依赖，整体加速 < C。｜引文依据：报告 §2.1.1 `KDA is recurrent across chunks and parallel within each chunk`（无加速倍数表述）。｜修复要求：删去「快 $C$ 倍」，改为「chunk 内可并行、chunk 间仍串行」或标注为并行度上界并说明整体加速受 chunk 间依赖限制。｜修复：已删去「比纯串行快 $C$ 倍」，改为「单 chunk 内 $C$ 个位置可并行计算，但 chunk 间仍存在串行依赖，整体加速受此限制」。｜复验：
- [轻微·格式] 行 704（[C2]）：正文 `<code>` 内直接出现 Unicode 数学字符（β、⊤、−），写法与全页 KaTeX 记法不一致（`St = (I − βt kt kt⊤) Diag(αt)St−1 + βt kt vt⊤`）。｜引文依据：不适用（格式一致性，规范 §2.2 第 9 条要求正文无 Unicode 数学字符直接出现）。｜修复要求：改为 KaTeX 渲染该式，或改为不含 Unicode 数学字符的纯文字描述（如「擦除项 (I − β_t k_t k_tᵀ) 作用于 Diag(α_t) S_{t−1}」的纯文字化写法）。｜修复：已将 [C2] 中 <code> 内的 Unicode 数学字符改为 KaTeX 内联公式 $S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top$。｜复验：
- [轻微·格式] 行 286、380、543：「构造示例。 」句后带一个多余空格。｜引文依据：不适用｜修复要求：删除「构造示例。」后的空格（三处）。｜修复：已删除三处「构造示例。」后的多余空格（行 286、380、543）。｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复（重要问题 1 条待关闭；核心公式、数字、配置与手算示例均回源一致，无阻断级问题）。修复后按规范第 4 节运行 `.dojo/scripts/validate.py wiki/kda/index.html` 并复验。