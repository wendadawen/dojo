<!-- review-meta
round: 8
page: wiki/positional-encoding/index.html
reviewed_content_sha256: bea8dd3efcbd5021
-->
# 位置编码基础审查记录（第 8 轮）

- 页面版本：index.html 工作树哈希 sha256:fcd9846ac421a6c54ba44740e6b9288d3e836384679eeab92846c594fc01e022
- 审查时间：2026-09-13 22:40
- 审查者：独立子代理（编排者派发的独立审查者，未参与写作，未参与前序轮次审查；未读取本页 research/）
- 已完整阅读章节（按顺序，含折叠块与图注）：引言与「核心问题」→「常见误解」→ 1. 为什么 Transformer 需要位置编码——注意力的排列等变性 → 2. 绝对正弦位置编码——Vaswani 2017 的 sin/cos 公式与手算 → 3. 可学习绝对位置编码——把固定向量换成可学习参数 → 4. 相对位置编码——T5 的 bias 机制与「加在分数上」的本质区别 → 5. 四类方案对比与 NoPE 选择——为什么 K3 选 NoPE（含 5.1）→ 来源与范围说明

## 来源核对（本轮通过项，附引文依据）

- 正弦公式（F1/C2）：Vaswani et al. 2017 §3.5 原文为 PE(pos,2i)=sin(pos/10000^(2i/d_model))、PE(pos,2i+1)=cos(pos/10000^(2i/d_model))，与页面一致。波长「geometric progression from 2π to 10000·2π」、base 模型 d_model=512 均在 §3.5/§3.1 出现（C3/N2/N3）。
- 线性性质（C4/F3）：§3.5 原文 "for any fixed offset k, PE_{pos+k} can be represented as a linear function of PE_{pos}"；选 sin/cos 的理由 "we hypothesized it would allow the model to easily learn to attend by relative positions"。页面 (240)(241) 的和角展开与旋转角 −Δω_i 经手算复核一致，(253) 的 2sin1cos1=sin2≈0.9093 复算正确。
- Table 3 row (e)（C5/N1）：ar5iv 全文表格中 base 行 BLEU 25.8、row (E)「positional embedding instead of sinusoids」BLEU 25.7，论文 §3.5 原文 "the two versions produced nearly identical results (see Table 3 row (E))"、"may allow the model to extrapolate to sequence lengths longer than the ones encountered during training"。页面 25.8/25.7、"nearly identical"、"may allow" 无外推实测的表述均与来源相符；big=28.4（Table 2，test newstest2014）无误。
- 手算数值：d_model=4、ω0=1、ω1=0.01；sin1≈0.8415、cos1≈0.5403、sin0.01≈0.0100、cos0.01≈0.99995≈1.0000、sin2≈0.9093、cos2≈−0.4161、cos0.02≈0.9998；dim0 差 0.0678、dim1 差 −0.9564、pos=1→100 dim2 差约 0.83；L=d=512 时 L·d=262144≈26 万。全部复算通过。
- T5（C7/F4）：Raffel 2019 §2.1 原文 "we use 32 embeddings"、"up to an offset of 128"、"each attention head uses a different learned position embedding"、"share the position embedding parameters across all layers"。页面「32 桶/每桶一标量、每头独立、各层共享、超 128 clamp」一致。
- ALiBi（C11）：arXiv:2108.12409 原文 "scalar m is a head-specific slope fixed before training"，penalty proportional to distance，作者明确 experiment 使 slope 可训练未见外推收益，故固定不学习；页面 −m_h·|i−j|、不学习一致。
- RoPE（C9）：arXiv:2104.09864 Eq. 11 ⟨f_q(x_m,m),f_k(x_n,n)⟩=g(x_m,x_n,m−n)，Eq. 16 内积只依赖 n−m（相对位置）；「绝对构造、相对效果」成立（该性质 §3.1 提出、§3.2 给出旋转矩阵，页面标注 §3.1 可接受）。
- DeepSeek-V2（C13）：arXiv:2405.04434 §2.1.3「RoPE is incompatible with low-rank KV compression」、「WUK cannot be absorbed into WQ any more during inference」，故提出 decoupled RoPE；页面「位置敏感的旋转破坏矩阵吸收」一致。
- Kimi K3（C10）：arXiv:2607.24653v2 §2.1.2 "applies No Position Encoding (NoPE) to all MLA layers"、"the intervening KDA layers provide position-sensitive and recency-aware sequence mixing"、"the MLA layers provide unrestricted global content interaction"、"avoids modifying positional-encoding parameters when extending the context length, such as retuning a RoPE frequency base or applying YaRN"；§3.4 "extrapolates directly to 1M-token contexts"。官方 config.json 核对：mla_use_nope=true，且 qk_rope_head_dim=64（即保留 rot 分量接口）。页面 457/459/461/487 各点均有来源支持。
- NoPE（C12）：arXiv:2305.19466 摘要 "Transformers without positional encoding (NoPE)"，作者 Kazemnejad 等，NeurIPS 2023。
- 机械项：validate.py 返回 "validation ok"；公式全部为 LaTeX（无未被 KaTeX 渲染的 Unicode 数学字符，`桶数×头数` 中的 × 由规范明列为普通排版字符）；结构图为 HTML（dg-stack/dg-layer，CSS 已定义）；前置概念页 rope/nope/kimi-k3/mla/standard-attention 均真实存在；overview.html 与 index.html 相互链接；alt 属性无 `$...$`。

## 问题

- [轻微·表述] index.html 引言（第 65 行）：开篇使用口语化措辞「喂给」「模型能分清谁打谁吗？答案是：不能」，与全文其余部分的书面对比语体不一致｜引文依据：不适用｜修复要求：改为书面表述，例如「把『我打你』和『你打我』输入一个不含位置编码的 Transformer，模型无法区分二者」｜修复：｜复验：
- [轻微·表述] index.html 第 385 行（4. 本质区别「作用点不同」）与第 422 行（4. 本章问题第 2 题答案）：句末「……每个注意力层、每个头都直接把位置信号注入分数——模型对位置更敏感」，把无来源支持的效果判断写成结论，且在正文与章末答案两处重复｜引文依据：该处所引 C8 的来源为 Vaswani 2017 §3.5 与 Raffel 2019 §2.1，两者只支持「编码对象」与「作用点」两项事实，均未出现「相对偏置使模型对位置更敏感」一类效果陈述｜修复要求：删去「模型对位置更敏感」这一效果判断，或改写为标注清楚的推断并给出依据｜修复：｜复验：
- [轻微·表述] overview.html「1. 为什么需要它」（第 26 行）：「它对输入 token 的行重排不变（排列等变）」——主句「不变」与同句括注「排列等变」自相矛盾；自注意力对输入重排是等变（输出随之置换）而非不变，且与 index.html 第 135 行「对行重排等变（输出随重排一同置换）」表述不一致｜引文依据：不适用｜修复要求：将「不变」改为「等变」，与 index.html 第 135 行保持一致｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布

统计：阻断 0 / 重要 0 / 轻微 3