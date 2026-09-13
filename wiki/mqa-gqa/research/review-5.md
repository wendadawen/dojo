<!-- review-meta
round: 5
page: wiki/mqa-gqa/index.html
reviewed_content_sha256: 4c9973dba605b10d
-->
# MQA 与 GQA 审查记录（第 5 轮）

- 页面版本：b7b0f70f48a322cddbaac43fc98c65d8fc24d657
- 审查时间：2026-09-13 20:18
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：引言（主要依据 + 开篇）、核心问题、最容易误解、1. 为什么 MHA 推理受内存带宽限制、2. MQA——所有 query 头共享一组 K/V、3. GQA——在 MHA 与 MQA 之间插值、4. 手算对比、5. 边界与后续、来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）
- 外部来源核对方式：Shazeer 2019（ar5iv/arXiv:1911.02150）、Ainslie 2023（ar5iv + arxiv.org/html/2305.13245，arXiv:2305.13245）、DeepSeek-V2（arXiv:2405.04434）、DeepSeek LLM 67B（ar5iv/arXiv:2401.02954）；机械项运行 `.dojo/scripts/validate.py`。

## 问题

- [轻微·表述] L484：以祈使元话语"注意"起句（与规范列举的"需要注意的是"同族）。｜引文依据：不适用｜修复要求：去掉"注意"，直接陈述为"MLA 的 cache（576 元素/token/layer）大于 MQA（256）"。｜修复：｜复验：
- [轻微·表述] L369："把它们并排放到手算里，连续谱关系随之显形"含隐含第二人称的读者指令与临场评价（"随之显形"）。｜引文依据：不适用｜修复要求：改为陈述句，例如"三种机制的 cache 公式可合并为一张表，连续谱关系由该表直接给出"。｜修复：｜复验：
- [轻微·表述] L176/179/201/206：同一比喻"切入点"全页出现 4 次，其中 L179 与 L201 为完全相同的句子"这是 MQA 与 GQA 的共同切入点"。｜引文依据：不适用｜修复要求：保留 1 处，其余改写为直陈式（如"这是 MQA 与 GQA 要解决的问题"）。｜修复：｜复验：
- [轻微·技术·符号] L475/484/545：MLA 段引入 $n_h$、$d_h$ 表示头数与每头维度（沿用 DeepSeek-V2 记法），与全页其余位置的 $h$、$d_k$ 不统一；且只定义了 $d_c$、$d_h^R$，$n_h$、$d_h$ 未定义。｜引文依据：DeepSeek-V2 §2.1.4 用 $n_h,d_h$ 记头数与每头维度；页面 L150 用 $h$、L151 用 $d_k$ 表示同一量。｜修复要求：在首次出现处注明"$n_h=h$、$d_h=d_k$（DeepSeek-V2 记法）"，或统一改写为 $h$、$d_k$。｜修复：｜复验：
- [轻微·技术·来源] L546（N6）：算力与带宽数字标注"源自 NVIDIA 官方数据手册"，但未给出可定位的手册名称/版本/链接，按 §2.2 第 1 条无法定位到具体位置。｜引文依据：数值本身与 NVIDIA SXM 规格一致（V100 15.7 TFLOPS/900 GB/s、A100 19.5/2039、H100 67/3352），但页面未提供出处。｜修复要求：补具体数据手册名与链接（V100/A100/H100 SXM datasheet）。｜修复：｜复验：
- [轻微·技术] L327：表引言称"Ainslie 2023 Table 1 在 T5-XXL 上的实验数据"，但紧随的表内含 MHA-Large 行（T5-Large，非 XXL）。｜引文依据：Ainslie 2023 Table 1 同时给出 T5-Large 与 T5-XXL 两档（MHA-Large 0.37s/46.0、MHA-XXL 1.51s/47.2）。｜修复要求：引言改为"T5-Large 与 T5-XXL"或删去 MHA-Large 行。｜修复：｜复验：
- [轻微·技术] L460："它们也不解决训练时的激活内存，也不引入位置信息"等边界断言无来源，且未按页面既有做法标注为通用背景或推断（对照 L176"以上为通用背景知识…不是上述论文的原文结论"、L257"这一组描述为工程实现层面的推断"）。｜引文依据：不适用｜修复要求：按同页既有做法加"以上为通用背景/范围判断"标注，或补来源。｜修复：｜复验：

## 核对记录（无问题项，供复验）

- C1 §1 引言："the speed of incremental Transformer inference on modern computing hardware is limited by the memory bandwidth necessary to reload the large "keys" and "values" tensors"——与 L519 一致（仅定位写"§1 第 2 段"，ar5iv 该引言并作一段，属可接受的粗定位）。
- C2/F5 §2.4.1 MHA 比值 Θ(n/d + 1/b)、§3.1 MQA 比值 Θ(1/d + n/(dh) + 1/b)——与页面 L170/L245/L201 一致。
- §2.4.1"当 n≈d 或 b≈1 时比值接近 1"——支持 L74/L172/L194 的"在 n 接近 d 或 batch 小时接近 1"。
- C3 §3 "Multi-query attention is identical except that the different heads share a single set of keys and values."——逐字一致。
- §2.2 einsum 形状 P_q/P_k=[h,d,k]、P_v/P_o=[h,d,v]；§3 去掉 h 维为 [d,k]/[d,v]——与 L212/L218-221 一致（页面把 k/v 写作 $d_k/d_v$，L142 已说明统一记法）。
- C5 质量与稳定性：Shazeer abstract"incur only minor quality degradation from the baseline"；Ainslie abstract/§1"MQA can lead to quality degradation"；Appendix A"multi-query attention can lead to training instability during fine-tuning, in particular combined with long input tasks…Uptrained grouped-query attention models, however, appear to be stable."——逐字一致。
- C6 §2/§2.2 "Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head."——一致。
- C7 §2.1 均值池化"works better than selecting a single key and value head or randomly initializing…"；§3.1"For α=0.05, training took approximately 600 TPUv3 chip-days."——一致。
- §3.3 Figure 4"Intuitively, results are ordered by the degree to which information is preserved from the pre-trained model."；Figure 4 讨论"Mean pooling appears to work best, followed by selecting a single head and then random initialization."（L322"选第一个头"为图例近似，不单列问题）。
- §3.3 Figure 5 题注"Performance as a function of uptraining proportion for T5 XXL models with MQA and GQA-8."、"Both MQA and GQA gain from 5% uptraining with diminishing returns from 10%."、"GQA already achieves reasonable performance after conversion while MQA requires uptraining to be useful."——支持 L317/L323。
- N3/Figure 6：arxiv.org/html 版本含"increasing the number of groups from MQA only results in modest slowdowns initially"与"We selected 8 groups as a favorable middle ground."——L543 引文属实（非同稿的 ar5iv 渲染为"adds modest inference overhead, with increasing cost to adding more groups"，属版本差异，不判为误引）。
- C8/N2 Table 1：MHA-Large 0.37/46.0、MHA-XXL 1.51/47.2、MQA-XXL 0.24/46.6、GQA-8-XXL 0.28/47.1——逐字一致；派生倍数 1.51/0.24=6.3×、1.51/0.28=5.4× 复算正确；七数据集（CNN/DailyMail、arXiv、PubMed、MediaSum、MultiNews、WMT EnDe、TriviaQA）与"We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder self-attention."均一致（支持 L340）。
- C9/F4/N5：DeepSeek-V2 §2.1.4 Table 1（MHA $2n_hd_h$、MQA $2d_h$、MLA $d_c+d_h^R$）、配置 $n_h{=}128,d_h{=}128,d_c{=}512,d_h^R{=}64$、576 元素/token/layer、32768/576≈1/57、abstract"reduces the KV cache by 93.3%"——一致；基线"DeepSeek 67B 的 GQA"经 DeepSeek LLM 67B 论文核实（"the 67B model uses Grouped-Query Attention (GQA)"，95 层/64 头/8 KV 头），属实。
- 全部手算：512/256/128 每 token、5120/2560/1280 共 10 token、10240/5120/2560 字节、21.5 GB/1.34 GB/168 MB、$2 d_k{=}128$ 递减——逐项复算无误，无分项之和与合计不符。
- 页面级与章节级问题块均有两级解答折叠块，核心问题答案均指明完整论证所在章节；无"（待生成）"占位；`../standard-attention/index.html` 与 `../mla/index.html` 均真实存在；head 中 `dojo:type=concept`、`dojo:topics=注意力机制`（在 ALLOWED_TOPICS 内）、`dojo:tag=注意力`（在 ALLOWED_TAGS 内）；正文与来源说明未引用 `research/` 下任何路径。`.dojo/scripts/validate.py wiki/mqa-gqa/index.html` 返回 "validation ok"。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 7
- 处置：可发布（7 项轻微，建议按修复要求逐条处理；无阻断与重要问题）
