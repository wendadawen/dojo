<!-- review-meta
round: 4
page: wiki/low-rank-projection/index.html
reviewed_content_sha256: 94cd688b3e68537d
-->
# 低秩分解审查记录（第 4 轮）

- 页面版本：f38b2d71fb51a0aac84c2dd2883038b7f668d36b
- 审查时间：2026-09-13 19:44
- 审查者：独立子代理（编排者派发；未参与写作与前三轮审查）
- 已完整阅读章节：核心问题（5 条）；1. 大矩阵为什么贵——参数与存储的成本｜本章问题；2. 矩阵的秩——能被压缩多少由什么决定｜本章问题；3. SVD 与最优低秩近似——误差由奇异值决定｜补充（直觉说明）｜展开（3×3 对角矩阵示例）｜本章问题；4. LoRA——把权重更新参数化为低秩｜本章问题；5. MLA——把 KV 压进低维潜向量｜本章问题；6. 适用边界——低秩不是万能的｜本章问题；来源与范围说明（论断与来源 C／公式与来源 F／外部数字与实验条件 N／构造示例／辅助解释与类比边界／简化条件及其限制）；overview.html。

## 来源核对摘要（作为核对依据，均定位到原文）

- LoRA §1（arXiv:2106.09685v2）："a very low rank (i.e., r in Figure 1 can be one or two) suffices even when the full rank (i.e., d) is as high as 12,288"——支持 [N2]。§1 同处 "which show that the learned over-parametrized models in fact reside on a low intrinsic dimension. We hypothesize that the change in weights during model adaptation also has a low "intrinsic rank""——支持 [C1]。§1 "reduce the number of trainable parameters by 10,000 times"——支持 [N1] 的 §1。§2 "|\Theta| can be as small as 0.01% of |\Phi_0|"——支持 [N1] 的 §2。§4.1 公式 (3) "h = W_0 x + BAx"，随后 "We use a random Gaussian initialization for A and zero for B, so ΔW=BA is zero at the beginning of training"；Figure 1 题注 "Our reparametrization. We only train A and B"——支持 [F1]。
- DeepSeek-V2 §2.1.2（arXiv:2405.04434v2）：Eq.(9) "𝐜_t^KV = W^DKV 𝐡_t"、Eq.(10) "𝐤_t^C = W^UK 𝐜_t^KV"、Eq.(11) "𝐯_t^C = W^UV 𝐜_t^KV"——支持 [C3]、[F4]。§2.1.4 Table 1 有 "MHA 2n_h d_h l" 与 "MLA (d_c + d_h^R)l"——支持 [F6]。§3.1.2 "we set the number of attention heads n_h to 128 and the per-head dimension d_h to 128. The KV compression dimension d_c is set to 512"——支持 [N3]。摘要 "Compared with DeepSeek 67B, DeepSeek-V2 ... reduces the KV cache by 93.3%"——支持 [N4] 的 93.3%。附录 C.2 "Comparison Between MLA and MHA" 的 Table 7 逐格为 "KV Cache per Token (# Element): Small MoE w/ MHA 110.6K、w/ MLA 15.6K；Large MoE w/ MHA 860.2K、w/ MLA 34.6K"，正文 "MLA requires a significantly smaller amount of KV cache (14% for small MoE models and 4% for large MoE models) than MHA"，并说明四模型 "share the same architecture except for the attention mechanisms"——支持 [N4] 的表号（C.2 / Table 7）与全部数字及定性描述。
- Wikipedia：SVD 条目 "is the best approximation of M by any matrix of rank less than or equal to t, under the Frobenius norm ... This is known as the Eckart–Young theorem, as it was proved by those two authors in 1936."；Low-rank approximation 条目 "The result is referred to as the matrix approximation lemma or Eckart–Young–Mirsky theorem. This problem was originally solved by Erhard Schmidt"、"L. Mirsky generalized the result to arbitrary unitarily invariant norms. (Q.J. Math. 11 (1960), 50-59)"，且该条目第 4、5 节标题为 "Proof of Eckart–Young–Mirsky theorem (for spectral norm / for Frobenius norm)"——支持 [C2]、[F3] 的引文与"谱范数与 Frobenius 范数证明小节"。
- 可复算数值全部复核通过：12288² = 150,994,944；2×24576 = 49,152；49,152/150,994,944 ≈ 0.033%；2×128×128 = 32768，512/32768 = 1/64（98.4% 减少、残留 1.5625%）；15.6/110.6 ≈ 14.1%（减少 85.9%）、34.6/860.2 ≈ 4.02%（减少 96.0%）；2×4096×8 = 65536、16/4096 ≈ 0.39%；√1.25 ≈ 1.118、√4.01 ≈ 2.002、√(2.9²+2.8²) ≈ 4.03；A=U_kΣ_k^{1/2}(m×k)、B=Σ_k^{1/2}V_k^T(k×n) 相乘等于 W_k。
- 页面 <head> 四项元数据齐备、dojo:type=concept、dojo:topics=数学基础（词表内）；`.dojo/scripts/validate.py wiki/low-rank-projection/index.html` 返回 "validation ok"（无模板残留、重复 id、锚点/本地资源断链）；正文与数学内容中无 Unicode 数学字符（validate.py 的 bare-math 检查通过），折叠块内的 ×、→ 属 validate.py 明示豁免的排版字符；index.html 与 overview.html 相互链接，standard-attention / mla / kda 概念页均真实存在。

## 问题

- [轻微·技术] 5. MLA ›「对比项」表格「等价权重秩」行 MHA 列：写成「满秩（$\le \min(d_h n_h, d)$）」，「满秩」应指秩恰等于 $\min(d_h n_h, d)$，与「$\le$」自相矛盾，读者无法判断 MHA 一侧是否有上界约束。｜引文依据：arXiv:2405.04434v2 §2.1.2 Eq.(10)「𝐤_t^C = W^UK 𝐜_t^KV」；同表 MLA 列为「秩 $\le d_c$（经过 $d_c$ 维瓶颈）」，两列对照意在区分「无低秩约束」与「秩 ≤ d_c」。｜修复要求：MHA 列改为「满秩（一般情形秩 $= \min(d_h n_h, d)$）」或「无低秩约束（秩上限 $\min(d_h n_h, d)$）」。｜修复：｜复验：
- [轻微·技术] 6. 适用边界 › K3 实例段与来源 [C4]：K3 论断的来源只写「KDA 概念页已核对的 Kimi K3 config.json use_full_rank_gate = true（Kimi K3 技术报告）」，未给出技术报告的章节/引文定位，页首「主要依据」也未列 K3 报告；核对依赖于另一 wiki 页面而非一手出处。｜引文依据：页面 [C4] 原文「KDA 概念页已核对的 Kimi K3 config.json use_full_rank_gate = true（Kimi K3 技术报告）」；报告关于该改动的原文句（low-rank 参数化 → full-rank 投影）在本页缺失。｜修复要求：把来源改为可定位的一手出处（写明 K3 报告章节号并给出原文句），并同步补入页首「主要依据」。｜修复：｜复验：
- [轻微·格式] 来源与范围说明 › 构造示例小节：第 1 条冠以 [N5]，第 2 条无编号；[N5] 在正文无任何 `<sup>[N5]</sup>` 引用。style-guide §6 要求来源编号与来源章节「双向对应」，且 [N] 小节定义为「外部数字与实验条件」，构造示例不属于外部数字。｜引文依据：不适用。｜修复要求：去掉该 [N5] 编号（构造示例节按规范只作说明、不挂来源码），或使该节编号方式统一且正文存在对应引用。｜修复：｜复验：
- [轻微·技术] overview.html ›「4. 关键结论与边界」第 3 条：「K3 把 Kimi Linear 的 low-rank 门控改为 full-rank，就是因为低秩门控表达力不足」——把 index.html 明确标注为推断的动机写成来源事实，两页对同一命题的确定性不一致。｜引文依据：index.html §6 原文「本页推断改动动机是低秩门控表达力不足……配置项本身只表明改用了 full-rank」；overview.html 原文「就是因为低秩门控表达力不足」。｜修复要求：overview 改为与 index 一致的推断标注（如「K3 改用 full-rank；动机未在来源中说明」）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：修复（四条轻微项逐条修复后即可发布；核心结论、公式、数字与来源核对均无阻断或重要问题）
