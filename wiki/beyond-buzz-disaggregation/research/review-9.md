<!-- review-meta
round: 9
page: wiki/beyond-buzz-disaggregation/index.html
reviewed_content_sha256: 2c15fc7ba9bc4141
-->
# Beyond the Buzz 审查记录（第 9 轮）

- 页面版本：index.html 工作树哈希 9c914d64809b13dddb9456ab31aab64294e9f60a
- 论文版本：arXiv:2506.05508v1（2025-06-05 提交；TeX 源码 tarball 内文件日期 2025-06-06，使用 neurips_2025.sty；arXiv 未列 Comments/录用信息）
- 审查时间：2026-09-14
- 审查者：独立子代理（编排者派发，未参与写作与既往轮次）
- 材料：index.html、overview.html、arXiv abs/html/2506.05508v1、arXiv e-print 源码包（design_principles.tex / disaggregation_in_practice.tex / system_considerations.tex / appendixB.tex / appendixD.tex / appendix.tex / abstract.tex / introduction.tex / conclusions.tex / future_work.tex）、assets 原图（像素测量）。未读取 research/ 任何文件。
- 已完整阅读章节（按序）：术语速查 → 核心问题（5 题及解答）→ 1. 方法：模拟器与设计空间（含本章问题）→ 2. 什么条件下分离收益最大（2.1 流量 / 2.2 模型大小 / 2.3 架构，含本章问题）→ 3. 配套机制：切分策略与动态 rate matching（3.1 / 3.2 / 3.3，含本章问题）→ 4. 分离的代价：KV cache 传输带宽（4.1 / 4.2 / 4.3，含本章问题）→ 5. 方法评价：可操作结论与边界（5.1 清单 / 5.2 边界，含本章问题）→ 来源与范围说明（含 collapse 块与全部图注、alt）。

## 回源核对结论（无问题项，列关键依据）

- 摘要/intro「首次系统性研究、数十万设计点、prefill-heavy + >10B 受益最大、dynamic rate matching 与 elastic scaling 关键」：abstract.tex、introduction.tex 逐句一致。
- FTL 约束只作用于 prefill、TTL 作用于 decode、两步 rate matching、integer solver、tolerance=0.03：design_principles.tex §3.2 与 appendixB.tex Algorithm 1/2 一致。
- 页面对 Algorithm 2 的复述「decode_request_throughput = decode_throughput/(OSL−1)」「α = best_prefill_throughput/decode_request_throughput」「num×G_dec 个 prefill GPU 与 den×G_prefill 个 decode GPU」与 appendixB.tex 第 49–50 行 `num_prefill_gpus = numerator(α)×G`、`num_decode_gpus = denominator(α)×best_prefill_config.num_gpus` 完全一致（正文侧 [App.B, line 46] 指向 appendixB.tex 第 46 行 `decode_request_throughput ← decode_throughput/(OSL−1)`，行号核对无误）。
- FTL>10s 排除：design_principles.tex「All design points with an FTL $>10$ seconds ... are excluded」。Blackwell+FP4、归一化呈现、逐层即时传输假设亦逐句一致。
- 带宽公式 F1/F2 与 system_considerations.tex Eq.(1)/(2) 逐项一致；「现有数据中心带宽足够」引文与原文一致；KV 复制因子与「唯一切分 KV 的 GPU」表述一致。
- 图内数值像素测量核对（img-01/02/03/04/05/06/07/08/10）：Fig.12 蓝 (ISL 16k/OSL 2k) ≈0.4–1.23、红 (ISL 1M/OSL 2k) ≈0.98–1.82，与页面「0.4–1.2 / 1.0–1.8」「0.4–1.8 GB/s/GPU」相符；Fig.9 DeepSeek 3.64→0.06、70B 0.95→0.45、405B 2.07→0.21、8B 0.41→0.35，与页面「3.6→0.05 / 0.95→0.45 / 2.1→0.2 / ≈0.4」相符；Fig.5 红线 log2(FTL) 由 ≈6.52 降到 ≈1.59（页写 6.5→1.5）、蓝线归一吞吐 ≈1.0；Fig.10 绿 (ratio 0.5) 平于 ≈0.42；Fig.1/6/7/8/14 标题读出的 ISL/OSL 与模型与页面图注逐字相符。
- 附录标签：appendix.tex 依序 \input{appendixA,appendixB,appendixD}，故 appendixD.tex 编译为 Appendix C；页面「附录 C（源文件名 appendixD.tex）」正确。
- 概念链接 ../moe-serving、../model-parallelism、../chunked-prefill、../mla、../mqa-gqa、../gpu-communication 均真实存在且确含被引内容；无「（待生成）」占位。
- 全部 alt 无 `$...$`；`.dojo/scripts/validate.py` 通过；核心问题（5）与各章本章问题（2/3/2/3/2）均有解答折叠块，核心问题答案均指明「完整论证见第 N 章」。
- 全文 grep 无「本页/本解析/我们/你/下面/接下来/需要注意的是」等元话语与会话指代。

## 问题

- [轻微·可读性] 第 4 章本章问题 Q1 解答（约 line 341）｜问题：「两者相乘 = 可用于传输的总时间-卡数资源，再除以 KV 总量得到每卡每单位时间字节率」一句，把除法方向写反；按字面（FTL×NumGPU）÷KV 是带宽的倒数，与正文 §4.1「KV 总产出除以（时间×卡数）= 带宽」及公式 F1 不符，折叠块内单独阅读易被读成反比关系｜引文依据：system_considerations.tex Eq.(1) `BW_egress = N_layers×BS_prefill×ISL×d_head×N_kv_heads×bytes_element /（FTL×NumGPU_prefill）`；页面同页正文「直觉：KV 总产出除以（时间 × 卡数）= 每卡每单位时间的字节率 = 带宽」｜修复要求：把该句改为「KV 总量除以（FTL × NumGPU_prefill）得到每卡每单位时间字节率」，使其与正文 §4.1 及公式方向一致｜修复：｜复验：
- [轻微·技术] §4.1 符号表（line 295）｜问题：$bytes_{element}$ 定义为「每个 KV 元素的字节数（FP4 下为 0.5 等）」，与论文原文对该符号的定义不一致｜引文依据：system_considerations.tex「$bytes_{{element}}$ indicates the number of KV cache bytes per token」；论文自身 prose 与公式相乘项（d_head×N_kv_heads×bytes_element）不自洽，页面取了与公式量纲一致的一种读法｜修复要求：在符号说明或来源与范围说明中加一句注明——论文原文写作“bytes per token”，本页按公式量纲读作“每 KV 元素的字节数”（即 d_head×N_kv_heads×bytes_element 为每 token 每层 KV 字节数）｜修复：｜复验：
- [轻微·技术] §2.1 流量（line 173）｜问题：四种 (ISL,OSL) 组合中，(16k,16k) 与 (2k,2k) 的 ISL:OSL 均为 1:1，页面却分别标为「平衡偏 prefill」「平衡偏 decode」，该倾向标签来源与依据未给出，二者并列易被误认为比例不同｜引文依据：disaggregation_in_practice.tex §4.2 仅写「four distinct traffic patterns」，未对任一组合给出偏 prefill/偏 decode 的定性；原图（img-07）四个面板标题为 16384/2048、16384/16384、2048/2048、2048/16384，无标签｜修复要求：改为按 ISL:OSL 比值的客观描述（如「(16k,16k) 与 (2k,2k) 均为 1:1 平衡」），或明确注明该倾向为页面推断并给出依据（如按序列绝对长度导致的 prefill 二次算力占比）｜修复：｜复验：
- [轻微·技术] §3.1 末段（line 236）｜问题：「对常见 ISL（16k、4k）该结论仍成立但压低 FTL 的空间更小」把图 5（仅 ISL 256K）的 CPP 结论外推到未测的短序列，且未标注为推断、无来源支撑｜引文依据：disaggregation_in_practice.tex 仅给 Fig.5「DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP×PP=64)」，论文未在 16k/4k 上验证 CPP｜修复要求：在句首加「（页面推断）」标记并说明依据（单层 FTL 已足够低），或删去该泛化改述为论文未覆盖短 ISL 验证｜修复：｜复验：
- [轻微·技术] §5.1 部署者清单 ③（line 371）｜问题：弹性伸缩条目引 [C3, C25]，但 C25 定义为 §7 future work，该节只列「KV cache reuse / speculation / inference-time compute / model architecture evolution」，不含弹性伸缩｜引文依据：future_work.tex 全文；弹性伸缩仅见于 abstract.tex 与 introduction.tex（C1/C3）｜修复要求：把该条引文改为 [C1, C3]（或 [C3]），去掉 C25｜修复：｜复验：
- [轻微·技术] 「简化条件及其限制」Blackwell+FP4 条目（line 488）｜问题：「FP4 字节数是 mlp 量化精度的代表」中的「mlp（FFN）级量化」为论文未出现的具体化断言，论文只泛称 FP4 precision，未限定到 MLP 层｜引文依据：design_principles.tex「Our analysis focuses on modern Blackwell systems using FP4 precision, which represent the state of the art in LLM inference infrastructure.」｜修复要求：删去「mlp」限定，改为「FP4 精度是论文所用精度设定，论文未给其他精度的对照数字」｜修复：｜复验：

## 复验说明（非问题项，说明本轮为何未报）

- 正文与 summary/overview 的三处关键数字（>10B、3.6→0.05、0.4–1.8 GBps/GPU）交叉一致；CPP、动态 rate matching、MLA chunking 开销等论断均回源核对通过。
- 开头段的叙事化措辞（「社区热度很高」「没人说得清」）与 pages 内其他论文页的设问式开头属同一文风约定，未按口语化问题计。
- 单位写法混用 GB/s 与 GBps（4 次 / 10 次）为符号一致性轻微项，因两写法同义且不影响读数，记录但不单列。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：可发布（阻断与重要问题为 0；6 条轻微问题不影响核心结论与原文一致性，建议按「修复要求」在各条位置一并收口）
