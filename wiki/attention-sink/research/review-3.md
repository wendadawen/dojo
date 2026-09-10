# 注意力汇聚点审查记录（第 3 轮）

- 页面版本：4f0f7ea745fffb4196065c6fea1693157118d0e8（index.html）；63d28e1ee2247f59ba69759892f6bf40929c2ea7（overview.html）
- 审查时间：2026-09-10 16:41
- 审查者：独立审查者（编排者派发，未参与写作与前序轮次）
- 已完整阅读章节：核心问题（4 题及解答）、常见误解（4 条）、1. 现象（含本章问题 2 题）、2. 成因（含本章问题 2 题）、3. 两种做法（3.1 形态一、3.2 形态二、含本章问题 3 题）、4. 对缓存与推理意味着什么（4.1/4.2/4.3、含本章问题 2 题）、来源与范围说明（论断 C、公式 F、外部数字、构造示例、辅助解释、简化条件）；overview.html 全文。

## 机械验证结果

- **validate.py**：`/usr/bin/python3 .dojo/scripts/validate.py wiki/attention-sink/index.html` 返回 `validation ok`，退出码 0。
- **§4.2 代码块**：复制实跑，输出与页面"预期输出"逐字符一致（`sink= 0.0: 份额 0.0078 = 0.78%` … `sink= 8.0: 份额 0.9588 = 95.88%`）。
- **KaTeX 渲染**：headless Chrome `--dump-dom` 成功（155 KB DOM）。统计：`.katex` 节点 69 个、`katex-display` 4 块、`katex-error` 0、`KaTeX parse error` 0、红色错误 `#cc0000` 0；全文制表符 0，无 `\text` 被写坏为制表符的情况。
- **verify_sink.py**：本机有 torch 2.8.0，脚本实跑成功，输出与 verify_sink.out 一致——[1] 汇聚点只进分母，与解析解逐元素最大差 `0.00e+00`；[2] 整行 -1 输出全零 `[0.0,0.0,0.0,0.0]`；[3] 份额 0.78%/5.46%/53.69%/95.88%；[4] 43 张量 = 主干 40 层 + MTP 3 层，形状 `{(64,)}`、dtype F32，主干层号 0..39 连续。
- **checkpoint 头**：headers.json 解析确认 43 个 `attn_sink` 张量（layers.0..39 + mtp.0..2），形状 `(64,)`、F32；分片字段为 `model-00001-of-00048.safetensors`，与 [N4]"48 个分片头"一致。
- **官方材料**：config.json `n_layers=40, n_mtp_layers=3, n_heads=64, window_size=128`；model.py L639 `self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))`；kernel.py 见下。
- **论文引文核对**（arXiv:2309.17453 HTML v3，逐条已定位原文）：
  - "beyond the bottom two layers, the model consistently focuses on the initial tokens across all layers and heads."（§3.1）
  - "The nature of the SoftMax function (Equation 1) prevents all attended tokens from having zero values." … "dump unnecessary attention values to specific tokens."（§3.1）
  - "removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function"（§3.1）
  - "initial tokens are visible to all subsequent tokens" … "more easily trained to serve as attention sinks"（§3.1）
  - "keeps the attention sink tokens' KV (with just 4 initial tokens sufficing)"（§1）；"Introducing four initial tokens generally suffices; further additions have diminishing returns."（Table 2 图注）；"focuses on positions within the cache rather than those in the original text."（§3.2）
  - Table 1（Llama-2-13B，PG19 首本书 65K）：0+1024=5158.07、4+1020=5.40、4"\n"+1020=5.60。
  - Table 2（Falcon-7B/MPT-7B/Pythia-12B/Llama-2-7B，PG19 拼接 400K）结论"4 个初始位置一般足够"。
  - Table 3（160M，PG19 首个样本）：Learnable Sink 1+1023=18.01，Vanilla 2+1022=18.05；§4.2 "employed the Pythia-160M codebase and followed its training recipe"。
  - 附录 A："does not extend the models' context window or enhance their long-term memory capabilities."

## 问题

- [重要·技术] §3.2「形态二」公式（index.html L896）及 §4.2 份额公式：`m` 的处理前后不一致。公式把可见槽位写成 `e^{q·k_t/\sqrt{d}}`（未减行最大值），却把 sink 项写成 `e^{\text{sink}-m}`（减了 `m`）；这与 [C4] 内核 kernel.py 不符——内核对可见槽位与 sink 都做最大减法（L369 `T.reduce_max(acc_s, scores_max, dim=1)`、L373 `acc_s[i,j] = T.exp(acc_s[i,j] - scores_max[i])`、L383 `sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`），也与 §3「本章问题」解答中"`m` 同时出现在分子与分母的公共因子里"的自述矛盾。由此 [F2] 声称"由 [F1] 在「W 个等权槽位」条件下直接推论"出 `e^{\text{sink}}/(W+e^{\text{sink}})` 不成立——等权（可见分数同为 `s`）时按 [F1] 的 sink 项只能得到 `e^{\text{sink}-s}/(W+e^{\text{sink}-s})`，仅当可见分数为 0（`m=0`）时才等于页面公式。｜引文依据：kernel.py L373/L383 对可见槽位与 sink 均减 `scores_max`；verify_sink.out 份额行 `exp(sink)/(128+exp(sink))` 对应 `q=0`（`m=0`）构造。｜修复要求：统一 `m` 写法——把 [F1] 可见槽位改写为 `e^{q·k_t/\sqrt{d}-m}`（与内核一致），并把 §4.2/[F2] 与「简化条件」中的假设由"W 个等权槽位/分数相同"改为"W 个可见槽位分数均为 0（即 `m=0`）"，与 §3.2 构造示例（`q=0`）对齐；改后重新核对 [F1]/[F2]、§3 本章问题解答、构造示例、代码输出的一致。｜修复：[F1] 公式改写为可见槽位也减 m：分子 e^{q·k_t/√d − m} v_t、分母 Σe^{q·k_t/√d − m} + e^{sink−m}；符号表新增「可见槽位项与 sink 项共享同一个 m」；[F2] 与 4.2、简化条件、[N5]、核心问题、构造示例的条件统一改为「W 个可见槽位分数均为 0（m=0）」；简化条件补「若分数同为 s 则份额为 e^{sink−s}/(W+e^{sink−s})」｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·格式] 来源与范围说明「外部数字与实验条件」h3（index.html L1036）：标题缺 style-guide §1 固定命名中的"（N）"后缀。｜引文依据：style-guide.md L21 固定命名为"外部数字与实验条件（N）"。｜修复要求：h3 改为「外部数字与实验条件（N）」。｜修复：h3 改为「外部数字与实验条件（N）」｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·格式] 来源与范围说明「论断与来源（C）」（index.html L1024-1030）：论断编号从 [C5] 直接跳到 [C7]，缺 [C6]，编号不连续。｜引文依据：不适用（编号机械检查）。｜修复要求：将 [C7] 重编号为 [C6] 并同步正文引用（§4.3 的 `<sup>[C7]</sup>`），或补齐 [C6]。｜修复：[C7] 重编号为 [C6]（正文 §4.3 与来源章节同步），C5→C6 编号连续｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·格式] 来源与范围说明「辅助解释与类比边界」（index.html L1047）：该节描述的「注意力预算的零钱罐」类比未在正文出现（"零钱罐"全文仅此一处），属残留死引用；按 style-guide §6"没有内容的小节不保留"应删除或改写。｜引文依据：不适用。｜修复要求：删除该辅助解释小节，或将"零钱罐"类比补回正文相应位置并保持两处一致。｜修复：删除「零钱罐」死引用，改为与正文一致的表述「注意力必须分配完、需要一个去处」这一归一化约束本身｜复验：已复跑 validate.py 通过并核对修改位置｜

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（存在 1 条重要问题未关闭，尚不满足 check.md §5 发布条件；修复后需重新核对 [F1]/[F2] 来源一致性并重跑 validate.py）
