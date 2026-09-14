<!-- review-meta
round: 5
page: wiki/low-rank-projection/index.html
reviewed_content_sha256: 4003e69dc4c85144
-->
# 低秩分解审查记录（第 5 轮）

- 页面版本：756076e7d588084513039c220efe4db3cc41578e（wiki/low-rank-projection/index.html）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题；1. 大矩阵为什么贵——参数与存储的成本；2. 矩阵的秩——能被压缩多少由什么决定；3. SVD 与最优低秩近似——误差由奇异值决定；4. LoRA——把权重更新参数化为低秩；5. MLA——把 KV 压进低维潜向量；6. 适用边界——低秩不是万能的；来源与范围说明（含全部 `<details>` 折叠块、HTML 结构图图注，以及 overview.html）

核对来源与方式：arXiv:2106.09685v2（LoRA）HTML 全文逐节比对；arXiv:2405.04434v2（DeepSeek-V2）PDF 全文（§2.1.2 Eq.(9)(10)(11)、§2.1.4 Table 1、§3.1.2、附录 C.2 Table 7）逐字比对；Wikipedia "Low-rank approximation" 与 "Singular value decomposition"；arXiv:2607.24653v2（Kimi K3）§2.1.1 Eq.(6)；huggingface.co/moonshotai/Kimi-K3 config.json。本页无图片/坐标图，无需像素测量；无声称可运行的代码块。

回源核对结论（均一致，未复述全部）：12288²=150,994,944；49152/150,994,944≈0.033%；r(m+n)=2×24576=49152；512/32768=1/64（1/64 对应 98.4% 减少、残留 1.56%）；2 n_h d_h=2×128×128=32768；√1.25≈1.118、√4.01≈2.002、√(2.9²+2.8²)≈4.03；2dr/d²=16/4096≈0.39%；MLA 损耗 d_h n_h−d_c=16384−512=15872。LoRA 侧：§1 的「reside on a low intrinsic dimension」「the change in weights during model adaptation also has a low 'intrinsic rank'」、§4 Eq.(3)「W0+ΔW=W0+BA」与「h=W0x+ΔWx=W0x+BAx」、「We use a random Gaussian initialization for A and zero for B, so ΔW=BA is zero at the beginning of training」「we do not introduce any additional latency」、§1「a very low rank (i.e., r in Figure 1 can be one or two) suffices even when the full rank (i.e., d) is as high as 12,288」、§2「can be as small as 0.01% of |Φ0|」——全部逐字吻合。DeepSeek-V2 侧：§2.1.2「The core of MLA is the low-rank joint compression for keys and values to reduce KV cache」与 Eq.(9)(10)(11) 的 W^DKV∈R^{d_c×d}、W^UK,W^UV∈R^{d_h n_h×d_c}、§2.1.4 Table 1「MHA 2n_h d_h l / MLA (d_c+d_h^R)l≈9/2 d_h l」、§3.1.2「n_h=128, d_h=128, d_c=512」、摘要 93.3%、附录 C.2 Table 7「KV Cache per Token：Small MoE 110.6K→15.6K、Large MoE 860.2K→34.6K」——逐字吻合。K3 侧：§2.1.1「Full-rank gate」Eq.(6)「Kimi K3 changes KDA's output gate from the low-rank parameterization used by Kimi Linear … to an input-dependent full-rank projection」与 config.json `linear_attn_config.use_full_rank_gate=true`——吻合。Wikipedia：Eckart-Young 定理、‖D−D̂*‖_F=√(σ²_{k+1}+⋯+σ²_r)、‖A−A_k‖₂=σ_{k+1}、Schmidt/Eckart-Young 1936/Mirsky 1960——吻合。功能性：KaTeX 渲染、`<details>` 折叠、目录锚点、本地资源与前置概念页链接（standard-attention / mla / kda，均真实存在）正常；`.dojo/scripts/validate.py` 返回 validation ok。表述维度逐段通读：未见会话指代（我/我们/你）、调试叙事、临场评价或 AI 拼接腔；「本页/本章」用法符合 style-guide §12（自称可用「本页」）与 §8（章节衔接），未计为问题。

## 问题

- [轻微·技术] 「来源与范围说明」[N1]（行 554）：把 LoRA「10,000× 参数减少」的出处标为「§1」，该数字不在 §1（Introduction）。｜引文依据：§1 全文仅见 "LoRA makes training more efficient and lowers the hardware barrier to entry by up to 3 times when using adaptive optimizers"；"10,000 times" 实际出现在 Abstract（"Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times"）、§4.2（"the checkpoint size is reduced by roughly 10,000× (from 350GB to 35MB)"）与 §7（"we achieved the largest reduction of trainable parameters (up to 10,000×)"）。数字本身正确，仅章节标注有误。｜修复要求：将 [N1] 中「§1（10,000 times）」改为「Abstract（10,000 times；§4.2、§7 亦述及）」；§2（0.01%）标注正确无需改。｜修复：｜复验：
- [轻微·技术] §5（行 443）：括注「MHA 变体的注意力头配置与 MLA 不同」在标注的出处（附录 C.2 / Table 7）无依据，且用 Table 7 自身数值判断只对 Large MoE 成立。｜引文依据：C.2 原文只说 "two small MoE models and two large MoE models respectively share the same architecture except for the attention mechanisms"，Table 7 只给 "# Activated/# Total Params" 与 "KV Cache per Token (# Element)"（110.6K/15.6K/860.2K/34.6K），未给任何注意力头数或头维度；由 Table 7 反推，Small MoE 的 MHA 缓存 110.6K=2×(16×128)×27，与其 MLA 变体的 15.6K=576×27（d_c+d_h^R=512+64=576）对应同一 n_h·d_h=2048，即 Small MoE 两变体头配置一致；仅 Large MoE 可佐证配置不同（860.2K≠2×128×128×60=1966080）。｜修复要求：删除该括注中的「MHA 变体的注意力头配置与 MLA 不同」，或限定为「Large MoE 的 MHA 变体注意力头配置与 MLA 不同」并标注为据 Table 7 反推的推断。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（两条轻微问题建议按上述要求修复；均不构成发布阻断，非核心结论错误、无同页数字矛盾、无来源事实缺漏）

统计：阻断 0 / 重要 0 / 轻微 2