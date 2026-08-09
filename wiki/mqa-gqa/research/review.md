# MQA 与 GQA 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 强推理模型对照来源）
- 审查对象：`wiki/mqa-gqa/index.html`（工作树哈希 `00fed622a0256239188ded1b6ef3dfe1a824b048`）、`wiki/mqa-gqa/overview.html`（工作树哈希 `88783100c9939e7911731ac83af592f69ddf315d`）
- 时间：2026-08-09
- 审查依据：`guides/concept/check.md` 段 A 盲读 + 段 B 对照来源
- 对照来源：Shazeer 2019 "Fast Transformer Decoding: One Write-Head is All You Need"（arXiv:1911.02150）；Ainslie 2023 "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"（arXiv:2305.13245, EMNLP 2023）；页面另引用 DeepSeek-V2 作为 MLA 对照来源（补搜核对）
- 隔离说明：未读取 `research/` 目录任何已有文件；`scope.md` 学习目标因隔离不可读，学习目标闭环核对以页面自身"读完你能回答"声明的 5 条目标为准

## 段 A 盲读小结

按页面顺序以小白视角通读 index.html 五章 + overview.html。主线（训练并行 vs 推理串行 → KV cache 随 h、n 线性增长 → MQA 共享 1 组 → GQA 分 G 组插值 → 手算连续谱 → 与 MLA 边界）总体清晰，教学数字 h=4/d_k=64/l=1/n=10 的三次手算（512/256/128 元素每 token）自洽，检查问题与章节内容对应。盲读中卡点集中在两处硬伤：第一章教学示例推广到真实规模时 cache 字节数与同页 S4 折叠块及 overview 直接打架（10.7 GB vs 21.5 GB）；第五章把 MLA cache 与 MQA cache 比大小，方向读反。另记录若干术语首现未解释、来源未标的小卡点，见下。

## 段 B 对照来源小结

逐条核对页面"来源与教学说明"中 C1–C9、F1–F5、N1–N5 与外部来源一致性：

- **与来源一致**：C1（Shazeer §1 "limited by the memory bandwidth necessary to reload the large 'keys' and 'values' tensors"）、C3（Shazeer §3 "the different heads share a single set of keys and values"）、C5（Shazeer abstract "incur only minor quality degradation"；Ainslie Appendix A 训练不稳定）、C6（Ainslie §2 "divides query heads into G groups, each of which shares a single key head and value head"）、C7（Ainslie §2.1 均值池化 + §3.1 α=5%）、C8（Ainslie Table 1：MHA-XXL 1.51s/47.2、MQA-XXL 0.24s/46.6、GQA-8-XXL 0.28s/47.1，多源交叉确认）、F1/F2/F3（cache 公式与 aiwiki KV cache 公式一致）、F5（Θ(n/d + 1/b) 与 adrian.idv.hk 笔记 `1/b` 一致；papercache 译作 `d/(bn)` 系误译，不判错）、N2/N3/N4（Table 1 数据、G=8 折中点、600 TPUv3 chip-days 均确认）。
- **发现不一致**：C9/第五章 MLA 对照（详见问题 2）、第一章真实规模字节数（详见问题 1）。
- **超出指定两来源**：DeepSeek-V2 MLA 部分（C9/F4/N5）补搜 aiwiki、zhaifeiyue、microscale 三源核对，确认 MLA cache=576 的对比对象是 MHA(32768)、压缩比 98.2%，论文 93.3% 为 vs DeepSeek 67B GQA 基线。
- **"主流 LLM 标配（Llama 2 70B、Llama 3、Mistral 等）"**（overview）：获 yobitel 等来源确认，事实正确，但 overview/index 均未标来源（见问题 5 同类）。
- **可运行代码**：页面无可运行代码块，此项不适用。
- **前置知识链接**：`../standard-attention/index.html`、`../mla/index.html` 因禁止读取其他页面无法验证存在性，页面未标注占位提示，假设存在。
- **页面功能**：KaTeX 公式分隔符、折叠 details、目录锚点、主题切换等静态结构正常；未在浏览器实际打开，机械项以 validate.py 为准（本次未运行）。

## 问题

- [阻断·技术] index.html §"为什么 MHA 推理受内存带宽限制" 教学示例 callout（h=128,d_k=128,l=80,n=4096 处）："MHA cache 约 10.7 GB（fp16）"——按公式 2·h·d_k·n·l·2字节 = 2×128×128×4096×80×2 = 2.15×10¹⁰ 字节 ≈ 21.5 GB；10.7 GB 恰为元素数（1.07×10¹⁰）直接当字节数，漏乘 fp16 每元素 2 字节因子；与同页 S4 折叠块（MHA ≈ 21.5 GB）及 overview.html（21.5 GB）直接矛盾，读者会在同页两处得到冲突数字：将"约 10.7 GB（fp16）"改为"约 21.5 GB（fp16）"，并复核 2×128×128×4096×80×2 的乘法链 ｜ 修复：已将 callout 中"MHA cache 约 10.7 GB（fp16）"改为"约 21.5 GB（fp16，$2\times128\times128\times4096\times80\times2\approx 2.15\times10^{10}$ 字节）"，与同页 S4 折叠块及 overview.html 一致 ｜ 复验：
- [阻断·技术] index.html §"边界与后续" 第五段："MLA cache = 576 元素/token/layer，比 MQA 的 2×128=256 还小——但更重要的是 MLA 质量优于 MHA"——576 > 256，比较方向读反，结论反转；DeepSeek-V2 §2.1.4 中 576 的对比对象是 MHA（2·n_h·d_h = 2×128×128 = 32768），非 MQA（256）；页面 N5 自身亦写"对应 MHA = 32768，比值约 1/57"，正文与来源说明自相矛盾；论文 93.3% 压缩比为 vs DeepSeek 67B GQA 基线，非 vs MQA：删去"比 MQA 的 2×128=256 还小"的对比，改为"比 MHA 的 2×128×128=32768 小得多（约 1/57）"；如需提及 MQA，应说明 MLA(576) 反而大于 MQA(256)，MLA 的优势是相比 MHA/GQA 基线在更小 cache 下保质量，而非比 MQA 更小 ｜ 修复：已删去"比 MQA 的 2×128=256 还小"对比，改为"比 MHA 的 2×128×128=32768 小得多（约 1/57）；注意 MLA(576) 反而大于 MQA(256)，MLA 的优势是相比 MHA/GQA 基线在更小 cache 下保质量"；同时把"MLA 质量优于 MHA"改为"在更小 cache 下保质量"以契合原文"保持质量"措辞 ｜ 复验：
- [重要·技术] index.html §"MQA" 投影形状表及 F1：P^V 列写作 R^{d×d_v}（MQA）/ R^{h×d×d_v}（MHA），引入 d_v 维度，但 cache 公式 F1/F2/F3 统一用 d_k（2·h·d_k），全文未说明 d_v 与 d_k 的关系；小白无法判断 V 份 cache 按 d_k 还是 d_v 计，且 Shazeer 原文 §2.2 区分 k 与 v 两维度：在投影形状表下方或 F1 处补一句"实践中 d_v = d_k（Vaswani 2017 设定），cache 公式统一记 d_k" ｜ 修复：已在 F1 公式下方补一句说明"Shazeer 2019 §2.2 区分 key 维度 $d_k$ 与 value 维度 $d_v$，但实践沿用 Vaswani 2017 设定 $d_v = d_k$，故本页 cache 公式统一记 $d_k$（K、V 各一份，每份 $d_k$ 元素）" ｜ 复验：
- [轻微·盲读] index.html §"为什么 MHA 推理受内存带宽限制" 章末检查项"说出性能比值 Θ(n/d+1/b) 中 n/d 项的系数是 h"：正文仅引 Shazeer §3.1 "reduced the offensive n/d by a factor of h"（MQA 把 n/d 减为 1/h），未显式推导 MHA 下 n/d 项带 h 倍系数，读者需自行反推"减为 1/h ⇒ 原系数是 h"：在 F5 附近补一句"MHA 下每头各一份 K/V，n/d 项的搬运量带 h 倍系数，故 MQA 共享后减为 1/h" ｜ 修复： ｜ 复验：
- [轻微·技术] index.html §"为什么 MHA 推理受内存带宽限制" V100/A100/H100 算力（15.7/19.5/67 TFLOPS）与带宽（900/2039/3352 GB/s）数字正确但未标来源；overview.html 同样引用并衍生 4.3×/3.7× 倍数亦未标来源：在"来源与教学说明 > 外部数字与实验条件"增加一条 GPU 规格来源（如 NVIDIA V100/A100/H100 官方数据表） ｜ 修复： ｜ 复验：
- [轻微·技术] index.html §"边界与后续" "MLA 质量优于 MHA（DeepSeek-V2 §2.1.4 报告 MLA 在保持质量的同时减少 93.3% KV cache）"——DeepSeek-V2 §2.1.4 为 cache 对照 Table 1，质量消融见 §2.1.5 及 Appendix，来源节次标注存疑；且"质量优于 MHA"与"保持质量"措辞有别，需核对原文是"严格优于"还是"保持"：核对 DeepSeek-V2 质量消融的实际节次与原文措辞，修正节次标注；若原文为"保持质量"则改为"在保持质量的同时减少 cache"，避免扩大来源结论 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html §"手算对比" S4 折叠块末句"MHA 单条请求的 KV cache 约 21.5 GB，超过单张 A100 80GB 除去模型权重后的可用空间"——21.5 GB 本身不一定超过 80 GB 卡除去权重后的可用空间（取决于模型权重大小，如权重 40 GB 则剩余 40 GB > 21.5 GB），说法不严谨且未给权重规模假设：改为"在长序列/多并发下会迅速占满"或给出具体权重规模假设（如"权重 60 GB 时剩余空间已被 21.5 GB cache 压缩大半"） ｜ 修复： ｜ 复验：

## overview.html 审查结论

overview.html 未发现独立错误：其引用的 21.5 GB、Table 1 数据（1.51s/47.2、0.24s/46.6、0.28s/47.1）、4.3×/3.7× 倍数、uptraining 5% + 均值池化、G=h/G=1 等均与来源及 index 正确部分一致；且 overview 未出现 index 第五章"MLA 比 MQA 小"的错误表述（仅讲机制不同）。overview 的"已成主流 LLM 标配（Llama 2 70B、Llama 3、Mistral 等）"获来源确认，唯一不足是来源未标（与问题 5 同类，不单列）。index 修复问题 1/2 后，overview 与 index 将完全一致。

## 学习目标闭环核对

以页面"读完你能回答"5 条目标为准（scope.md 因隔离不可读）：

1. MHA 受内存带宽限制、KV cache 随 h/n 增长 —— 第一章回答 ✓，但含问题 1（10.7 GB 阻断错误），修复后闭环
2. MQA 共享 K/V 减到 1/h 及代价 —— 第二章回答 ✓
3. GQA 插值、uptraining、G=8 折中点 —— 第三章回答 ✓
4. 手算 4 头 MHA / 2 组 GQA / 1 组 MQA —— 第四章回答 ✓
5. MQA/GQA 与 MLA 根本区别 —— 第五章回答 ✓，但含问题 2（MLA vs MQA 比较方向阻断错误），修复后闭环

## 结论

- 统计：阻断 2 / 重要 1 / 轻微 4
- 处置：进入修复（2 个阻断 + 1 个重要未关闭，不得发布；修复后阻断与重要全部关闭方可复验发布）
- 关键发现：两处阻断均为数值/比较方向错误，非表述瑕疵——第一章漏乘 fp16 字节因子致 cache 缩半且与同页及 overview 自相矛盾；第五章把 MLA cache 的对比对象从 MHA(32768) 偷换为 MQA(256) 且方向读反，直接导致"MLA 比 MQA 更省"的错误结论，与 DeepSeek-V2 原文及页面自身 N5 来源说明冲突
- 已对照确认无误：C1/C3/C5/C6/C7/C8、F1/F2/F3、F5、N2/N3/N4 与 Shazeer 2019 / Ainslie 2023 来源一致；overview.html 无独立错误
- 隔离受限项：scope.md 学习目标、前置概念页 standard-attention/mla 链接有效性、validate.py 机械项本次未验证
