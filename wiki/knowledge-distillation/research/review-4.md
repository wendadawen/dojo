<!-- review-meta
round: 4
page: wiki/knowledge-distillation/index.html
reviewed_content_sha256: e5d3dd2669e47a2d
-->
# 知识蒸馏审查记录（第 4 轮）

- 页面版本：8e9ca1372c7063727d19907731c1d8f9396b2a43
- 审查时间：2026-09-13 19:40
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题 → 1. 硬标签丢掉了什么——非目标类概率携带的暗知识 → 2. 温度 softmax——把分布软化 → 3. 手算温度 softmax——3 类 logits 在 T=1 与 T=5 下的分布 → 4. KD 总损失——软项 + 硬项 + T² 缩放（含 4.1 为什么必须乘 T²）→ 5. 边界——KD 解决什么、不解决什么、与 MOPD 的关系（含 5.1 适用条件、5.2 与 K3 MOPD 的关系）→ 来源与范围说明

来源核对方式：下载 arXiv:1503.02531 原文（9 页）并提取全文，逐条定位；代码在本地执行；`overview.html` 一并通读。

已核对通过（无问题）：MNIST 67/146/74、T=20、2×1200 教师 / 2×800 学生、30 单元学生 T≈2.5–4（§3 原文逐字命中）；「无数字 3」实验 206/133 与偏差 +3.5 后 109/14→996/1010=98.6%（§3 逐字命中，页面「877/86.8%」换算正确）；语音 58.9/10.9、61.1/10.7、60.8/10.7、8×2560、85M、2000 小时（§4 与 Table 1 逐字命中）；C6 引文与原文 "Since the magnitudes of the gradients produced by the soft targets scale as 1/T²" 逐字一致；T² 推导（链式 1/T × 分布压缩 1/T）、高温零均值等价于最小化 (1/2N)‖z−v‖²（Eq.2–4）可复算；手算 (0.665,0.245,0.090) / (0.402,0.329,0.269) 与差值 0.263/0.084/0.179 可复算；页面 Python 代码实跑输出与「预期输出」逐字符一致；`validate.py` 通过；无 research/ 路径引用；mopd/opd 链接页面真实存在；核心问题与 5 个本章问题均有解答折叠块且指认章节；C1–C12/F1–F4/N1–N5 与正文 `<sup>` 双向对应完整。

## 问题

- [重要·技术] §4 权重条目（正文）与 N5（来源与范围说明）：页面称 HVD15「报告 α 通常接近 1（软项主导）」，但 HVD15 未给出 α 的任何取值；§2 只说硬项权重低得多（未量化），而 §4.1 明确以 0.5 作为硬标签交叉熵的相对权重（按本页 α=软项权重的约定即 α=0.5）。「接近 1」是对来源的量化夸大，且与论文唯一给出的具体权重相矛盾。｜引文依据：§2 "We found that the best results were generally obtained by using a condiderably lower weight on the second objective function."；§4.1 "we tried temperatures of [1, 2, 5, 10] and used a relative weight of 0.5 on the cross-entropy for the hard targets"。｜修复要求：将「α 通常接近 1」改为与原文一致的「硬项权重明显低于软项（论文未给出统一最优值）」，或补注 §4.1 的硬项权重 0.5 以吻合来源；不得保留「接近 1」这一无来源的量化表述。｜修复：｜复验：

- [轻微·技术] §1 callout 与 C1：「暗知识（dark knowledge）……这一称谓源自 Hinton 的报告」为无出处的来源论断，页面（及 C1 条目）均未给出「Hinton 的报告」的可定位出处。｜引文依据：HVD15 全文检索无 "dark knowledge"（无命中）；页面未提供任何可定位的「报告」文献。｜修复要求：删除「源自 Hinton 的报告」或补上可定位出处；否则改写为「业界流传的称谓，HVD15 正文未使用该词」。｜修复：｜复验：

- [轻微·技术] §5.1 适用条件：前两条「师生共享输出空间」「教师已训练到收敛」未标来源，来源章节 C/F/N 也无对应条目，属无来源支持的判断直接写成结论。｜引文依据：不适用（来源章节无对应条目）。｜修复要求：为两条补可定位来源，或明确标注为一般性适用条件推断。｜修复：｜复验：

- [轻微·技术] C7（来源与范围说明）：C7 记作「HVD15 §3 表格」，但 67/146/74 三个数字出自 §3 正文，§3 全文无表格（表格出现在 §4 Table 1 与 §5 Table 4/5）。｜引文依据：§3 "This net achieved 67 test errors whereas a smaller net ... achieved 146 errors. But if the smaller net was regularized solely by adding the additional task of matching the soft targets ... it achieved 74 test errors."；§3 内 "Table" 出现 0 次。｜修复要求：将「§3 表格」改为「§3 正文」。｜修复：｜复验：

- [轻微·格式] §4 KD 总损失公式后的符号定义：该公式（与折叠块中的近似梯度公式）后的符号说明使用段落并以「；」分隔，未按 style-guide §11「公式后紧跟 `<ul>` 逐项定义每个符号」用 `<ul>` 逐项列出（对比：§2 温度 softmax 公式已按规范用 `<ul>`）。｜引文依据：不适用。｜修复要求：将 §4 公式后的符号定义改为 `<ul>` 逐项列出，与 §2 写法一致。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（关闭「α 接近 1」这一来源不符项后再行复验；4 条轻微项一并修正）