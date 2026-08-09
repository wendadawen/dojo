# MQA 与 GQA 独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源）
- 页面版本：index.html ac5b744、overview.html ac5b744
- 时间：2026-08-09
- 来源：Shazeer 2019, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150, §2.2/§2.4.1/§3/§3.1；Ainslie 2023, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023, arXiv:2305.13245, §1/§2/§2.1/§3.1/§3.3/Table 1/Figure 6/Appendix A。均经 ar5iv HTML 版逐条核对。

## 问题

- [轻微·盲读] "手算对比" 章开头写"把 S1–S3 的三次手算合并成一张表"；"MQA" 章教学示例写"复用 S1 的 $h=4,d_k=64,l=1,n=10$"；"GQA" 章教学示例写"复用 $h=4,d_k=64,l=1,n=10$"。页面章节标题为中文（"为什么 MHA 推理受内存带宽限制""MQA——所有 query 头共享一组 K/V""GQA——在 MHA 与 MQA 之间插值"等），无 S1/S2/S3 编号标记；小白读者无法定位"S1"指哪一节。修法：将"S1–S3"改为章节标题或锚点链接（如"把<a href='#why-mha-bandwidth-bound'>带宽瓶颈</a>至<a href='#gqa-interpolation'>GQA</a>的三次手算合并成一张表"），或去掉内部编号。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：进入修复（轻微不阻断；建议修复以提升首读体验）

## 段 A 盲读小结

扮演完全小白读者按页面顺序阅读。主线理解顺畅：从训练（并行）vs 推理（串行）的对比引出 KV cache 瓶颈；cache 公式 $2hd_k$ 逐项拆解（$h$、$d_k$、$n$、$l$、常数 2）清晰；roofline 模型折叠块把"算术强度"解释到小白可懂；MQA 的投影形状对比表（MHA $P^K\in\mathbb{R}^{h\times d\times d_k}$ vs MQA $P^K\in\mathbb{R}^{d\times d_k}$）直白；"multi-query 指 query 头保持多头"的误解澄清到位；GQA 的连续谱旋钮（$G=h$→MHA、$G=1$→MQA）直觉清晰；uptraining 两步配方（均值池化 + 5% 继续预训练）具体可操作；Table 1 四行实验数据支撑"GQA-8 质量接近 MHA、速度接近 MQA"的结论；手算对比表把三种机制的 cache 数字（512/256/128）并排让连续谱落到可复算程度；真实规模推广（21.5 GB / 1.34 GB / 168 MB）给出量级感；与 MLA 的"共享头 vs 低秩压缩"区别点明，作为 MLA 前置的逻辑闭合。学习目标五条均由正文章节完整回答。卡点仅上述一条轻微项，不阻断主线。

## 段 B 对照来源小结

逐条核对 Shazeer 2019 §2.2/§2.4.1/§3/§3.1 与 Ainslie 2023 §1/§2/§2.1/§3.1/§3.3/Table 1/Appendix A：

1. 定义与机制：MQA 定义"the different heads share a single set of keys and values"（Shazeer 2019 §3）逐字一致；GQA 定义"Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head"（Ainslie 2023 §2.2）逐字一致；MHA 投影形状 $P^Q\in\mathbb{R}^{h\times d\times d_k}$ 等（Shazeer 2019 §2.2 einsum 记法）一致；MQA 把 $P^K,P^V$ 去头维度变 $\mathbb{R}^{d\times d_k}$、$\mathbb{R}^{d\times d_v}$ 一致；"multi-query 指 query 头保持多头"（Shazeer 2019 §2.4.1 "while maintaining the 'heads' dimension in the queries"）一致。
2. 公式与推导：cache 公式 MHA=$2hd_k$、MQA=$2d_k$、GQA=$2Gd_k$ 由定义直接推出，复算一致；性能比值 $\Theta(n/d+1/b)$（Shazeer 2019 §2.4.1）一致；"We have reduced the offensive $n/d$ by a factor of $h$"（§3.1）逐字一致；教学手算 $h=4,d_k=64$：MHA=512、GQA-2=256、MQA=128 复算一致；真实规模 $h=128,d_k=128,l=80,n=4096$：MHA≈21.5 GB、GQA-8≈1.34 GB、MQA≈168 MB 复算一致；MLA cache=$d_c+d_h^R=512+64=576$、MHA=$2\times128\times128=32768$、比值 1/57 复算一致；GPU 规格 V100/A100/H100 FP32 15.7/19.5/67 TFLOPS、带宽 900/2039/3352 GB/s 核对一致。
3. 可运行代码：页面无可运行代码块，不适用。
4. 事实与推断：Table 1 四行（MHA-Large 0.37s/46.0、MHA-XXL 1.51s/47.2、MQA-XXL 0.24s/46.6、GQA-8-XXL 0.28s/47.1）逐行核对一致；速度倍数 MQA 6.3×（1.51/0.24=6.29）、GQA-8 5.4×（1.51/0.28=5.39）复算一致；质量差 MQA −0.6（47.2−46.6）、GQA-8 −0.1（47.2−47.1）一致；"We selected 8 groups as a favorable middle ground"（§3.3/Figure 6）逐字一致；uptraining 均值池化优于选第一个头与随机初始化（§2.1 原文"works better than selecting a single key and value head or randomly initializing"）一致；α=0.05、600 TPUv3 chip-days（§3.1）一致；训练不稳定"multi-query attention can lead to training instability during fine-tuning, in particular combined with long input tasks"与"Uptrained grouped-query attention models, however, appear to be stable"（Appendix A）逐字一致；"MQA can lead to quality degradation"（Abstract/§1）一致；"much faster to decode"与"minor quality degradation"（Shazeer 2019 abstract）逐字一致。教学构造均标注"数字为便于手算构造"。
5. 前置知识引用：标准注意力概念页链接 `../standard-attention/index.html` 有效；MLA 概念页链接 `../mla/index.html` 有效。
6. 教学简化：手算例子用 $h=4$、真实规模用 $h=128$、不展开 Shazeer 2019 性能比值完整推导、不展开 uptraining 消融细节、不展开 MLA 机制、不引用 Shazeer 2019 具体速度倍数（因 PDF 一手数据未取到精确值，只引定性结论）——均标注简化理由与可/不可推出边界，未发现简化导致核心结论失真。
7. 页面功能：KaTeX 公式渲染正常；details 折叠交互正常；侧边目录锚点（why-mha-bandwidth-bound / mqa-shared-kv / gqa-interpolation / handcalc-spectrum / boundary-and-mla / sources-and-teaching-notes）有效。

未发现来源不一致项。
