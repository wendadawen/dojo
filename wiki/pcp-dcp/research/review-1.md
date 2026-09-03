# PCP/DCP 第 1 轮审查

## 元信息
- 审查日期：2026-09-03
- 审查范围：index.html + overview.html
- 引用核对覆盖：C1–C22 ✓ / F1–F4 ✓（F4 公式复算通过，但正文缺 [F4] 上标锚点，见 PR2-001）/ N1–N4 ✓
- 核对方式：S1 用本地 fork 副本全文核对；S2 博客、S3 `parallel.py`（main）、S4 RFC #26133 逐条在线核对原文；F1–F4 与 N1–N4 用 Python 复算；硬规则用 grep + `.dojo/scripts/validate.py`（validate.py 返回 validation ok）
- 本轮只读不改，修复结果/复验结果栏留待修复轮填写

## 问题清单

### P0（阻断发布，必须修复）

无。所有 C1–C22 论断均能在 S1–S4 找到支持原文；未发现来源不支持或内容不符的论断。

### P1（影响重要结论，建议修复）

#### PR1-001 [链接/页面功能] 引言（L753）与第 1 章开头（L816）
- 问题：3 处 Markdown 链接语法残留，HTML 中会原样显示为文本 `[KV cache](../kv-cache/index.html)`，链接完全失效。L753 一处（KV cache）、L816 两处（标准注意力、KV cache）。其余同类链接（如 L836 chunked prefill）均正确使用 `<a href>`。
- 证据原文片段：不适用（页面源码 L753：`却要背着全部历史的 [KV cache](../kv-cache/index.html)（键值缓存）往下走`）
- 建议处理：将 3 处改为 `<a href="../kv-cache/index.html">KV cache</a>` 与 `<a href="../standard-attention/index.html">标准注意力</a>`，与其余链接写法一致。

#### PR1-002 [来源论断] 第 5 章（L1431）、核心问题 5 解答（L799）、第 5 章本章问题解答（L1475）
- 问题：三处把 C10 的约束（MLA：$t \ge d$ 且 $t \bmod d = 0$；GQA：$t/H \ge d$ 且 $(t/H) \bmod d = 0$）定性为"**源码层面的硬校验**"。但 C10 的来源定位是 S2 博客 §5.3/§5.4 的 "Constraints"（vLLM Usage 节），并非源码；S3 `parallel.py` 的 `_validate_parallel_config` 实际只校验 `tp % dcp == 0`（无 MLA/GQA 分支）——页面自己在 C9 折叠块（L1453–1460）贴出的代码恰好自证了这一点，前后矛盾。"源码硬校验"的定性超出来源支持范围，且会误导读者以为 GQA 约束由 parallel.py 强制。
- 证据原文片段：S2 §5.4 "the sequence-split degree is capped by the duplication factor `tp // num_key_value_heads`. Constraints: (tensor_parallel_size // num_key_value_heads) >= decode_context_parallel_size"（博客用法约束，未称源码校验）；S3 `_validate_parallel_config`：`if pcp == 1: if tp % dcp != 0: raise ValueError(f"tp_size={tp} must be divisible by dcp_size={dcp}.")`（仅整除校验）
- 建议处理：改为"官方博客给出的使用约束（MLA/GQA 分模型）"，或"vLLM 使用约束"；保留 C9（真正的源码校验）为"源码硬校验"。若坚持"源码"表述，需补 S3 之外的源码定位（如 runner 层校验代码），否则按 check.md 2.2 降级处理。

#### PR1-003 [硬规则] 核心问题 4 解答（L792）
- 问题："PCP 扩展 world size（设备数变为 TP × PCP）"中乘号用了裸 Unicode 字符 `×`，未包在 `$...$` 中，违反 check.md 2.2-9 / style-guide §11"数学符号全部由 KaTeX 渲染，正文无 Unicode 数学字符直接出现"；且与第 4.3 节正文（L1386）的 $\mathrm{PP} \times \mathrm{TP} \times \mathrm{PCP}$ 同页写法不一致（style-guide"同一变量全页写法一致"）。注：validate.py 未拦截该字符（`8×B200` 型号写法导致 × 被豁免），但规则条文明确。
- 证据原文片段：不适用（页面 L792 原文："设备数变为 TP × PCP"）
- 建议处理：改为 `$\mathrm{TP} \times \mathrm{PCP}$`（若同时采纳 PR2-003，则为 `$\mathrm{PP} \times \mathrm{TP} \times \mathrm{PCP}$`）。`8×B200` 型号写法可保留（来源同款写法，非数学运算）。

### P2（小问题，可后续清理）

#### PR2-001 [结构/引用体系] 第 4.2 节公式（L1316）与来源表
- 问题：F4 在文末来源表有条目（L1550），但正文从头到尾没有出现 `<sup>[F4]</sup>` 上标——第 4.2 节的块负载公式 $(i+1)b^2 + (2N-i)b^2 = (2N+1)b^2$ 及配对规则只标了 [C18]。style-guide §6 要求正文上标与来源章节"双向对应"，F4 的正文侧锚点缺失，来源表 F4 行成孤立条目。
- 证据原文片段：不适用（grep 全文：`[F1]`×1、`[F2]`×1、`[F3]`×2、`[F4]`×0）
- 建议处理：在 L1316 公式或 L1314 配对规则处补 `<sup>[F4]</sup>`（可与 [C18] 组合为 `<sup>[C18, F4]</sup>`）。

#### PR2-002 [数值/表述] 常见误解第 1 条（L808）
- 问题："实测的'吞吐约 3 倍'来自并发能力"——S2 原文只给出 1,863 与 6,091 两个数，没有"3 倍"表述；6,091/1,863≈3.27 是页面换算，且"约 3 倍"偏保守（约 3.3 倍）。按 check.md 2.2-4"构造示例和解释没有写成来源结论"，换算数字应标注或写明出处算法。
- 证据原文片段：S2 §2.2 "throughput plateaus near 1,863 tok/s/GPU"；"DCP reaches 6,091 tok/s/GPU at c512"（无倍数表述）
- 建议处理：改为"实测约 3.3 倍（6,091/1,863，页面换算）"或直接引用两个原始数字。

#### PR2-003 [表述不一致] 误解区（L810）、4.3 对照表（L1394）、核心问题 4 解答（L792）
- 问题：PCP 扩展后的设备数，4.3 节正文与 C19/S3 一致地写 $\mathrm{PP} \times \mathrm{TP} \times \mathrm{PCP}$，但误解区写"PCP 才扩展设备数（$\mathrm{TP}\times\mathrm{PCP}$）"、4.3 对照表"设备数"行写"扩展：$\mathrm{TP} \times \mathrm{PCP}$"、核心问题 4 解答写"TP × PCP"，三处均缺 PP，同页两种口径。PP=1 部署下两者相等到，但作为一般表述不完整。
- 证据原文片段：S3 `__post_init__`：`self.world_size = (self.pipeline_parallel_size * self.tensor_parallel_size * self.prefill_context_parallel_size)`
- 建议处理：统一为 $\mathrm{PP} \times \mathrm{TP} \times \mathrm{PCP}$，或在不写 PP 处注明"单 PP 级（PP=1）时即 TP × PCP"。

#### PR2-004 [来源论断/引申] 第 1 章（L818）
- 问题："vLLM 对长上下文 prefill 的目标，就是把这部分计算**摊到多个 rank 上并行**，把 TTFT 压下来[C2]"——S1 原文为"amortizing the computation time of the prefill **across query tokens**"（跨 query token 摊销计算时间），未提 rank；"摊到多 rank 并行"是对 CP 语境的引申解读（依据 S1 PCP 节的 N 卡切分才成立）。
- 证据原文片段：S1 "we need to control the TTFT (time to first token) by amortizing the computation time of the prefill across query tokens"
- 建议处理：改为贴近原文的"把 prefill 的计算时间摊到各段 query token 上（由多卡并行承担，见第 4 章）"，或标注为引申。

#### PR2-005 [来源状态] 第 4.2 节（L1314）
- 问题："vLLM 的 RFC 给出的解法是首尾配对"——S4（RFC #26133）当前状态为 **Closed as not planned**（2026 年初 stale 自动关闭），页面未注明该状态，读者可能以为这是 vLLM 已采纳或在研的官方方案。配对切分思想本身有 S4 原文支持（且评论区 DCP 维护者确认 DCP 侧采用同样的 head-tail 方式），论断内容无误，缺的是状态语境。
- 证据原文片段：S4 "The sequence is divided into 2 × cp_world_size chunks. Each CP rank i is assigned both the i-th chunk and the (2 × cp_world_size - i - 1)-th chunk."；issue 状态 "Closed as not planned"
- 建议处理：在正文或来源表 C18 行注明"该 RFC 已关闭（not planned），配对设计作为 RFC 描述的设计呈现；DCP 相关 PR 采用了同样的首尾配对思想"。

#### PR2-006 [格式] 来源与范围说明章节 h3（L1496、L1541）
- 问题：来源章节 h3 使用了"核心论断与来源""核心公式与来源"，与 style-guide §1 固定命名"论断与来源（C）""公式与来源（F）"不符（其余四个小节命名均符合）。
- 证据原文片段：不适用（规范原文：来源章节 h3 "使用固定命名……：`论断与来源（C）`、`公式与来源（F）`……"）
- 建议处理：两个 h3 改为固定命名。

#### PR2-007 [术语] 核心问题 1 解答（L771）
- 问题：SLO 首次出现在解答折叠块中（"两种服务等级目标"语境下的"两种服务目标（SLO）"），未展开全称；正文（L755）只说"要优化的指标不同"。C1 原文带全称。
- 证据原文片段：S1 开篇 "have quite different SLO (service level objectives)"
- 建议处理：首次出现处补全称，如"服务等级目标（SLO, service level objectives）"。

#### PR2-008 [格式] 各公式（L864、L1133、L1316）
- 问题：style-guide §11 要求"公式后紧跟 `<ul>` 逐项定义每个符号"，页面对 $\max(1, t/H)$、LSE 公式 $l_r/o_r$、配对负载公式均用行内文字或括注定义符号，无定义列表。符号在上下文中均有定义，不影响理解，属轻微格式偏差。
- 证据原文片段：不适用
- 建议处理：可为三个核心公式补符号定义 `<ul>`；若与既有概念页惯例一致，记录接受理由后保留现状。

## 通过项（确认正确的点）

- **C1–C4**（S1 开篇/Decode 节）：两阶段特性与 SLO、TTFT/KV 空间目标、decode 少量 query 对大量 KV、TP 沿头切与 $t/H$ 重复，逐字一致。
- **C5、C6**（S2 §1）：GQA 头少/MLA 有效单头完整复制、DCP 序列维分片每卡唯一切片，逐字一致。
- **C7、C8、C20**（S1）：dcp 不增加 GPU 数、范围 $[1, t/H]$ 与通信代价、先加 tp 再加 dcp 的建议，逐字一致。
- **C9**（S3 `_validate_parallel_config`）：PCP 不支持 DP、无 PCP 时 `tp % dcp == 0`、有 PCP 时 `dcp ∈ {1, pcp, tp*pcp}`，页面折叠块代码与上游 main 逐字一致。
- **C11**（S1 + S3 docstring + S4）：交错分片、token $i$ 落 rank $i \bmod d$（interleave_size=1 默认值确认）、Chao Hong/ arXiv:2507.07120 出处，一致；块级对齐在"辅助解释与类比边界"中正确补充。
- **C12**（S2 §4.1）："AllGather Q → Compute → AllGather + ReduceScatter" 节奏、query 单 token 故便宜、LSE online-softmax 合并，逐字一致；页面三段流程图与之一致。
- **C13**（S2 §4）：200K 请求 GPU 0–3 各存 50K 区间的示意，一致；与 C11 交错实现的"概念 vs 实现"关系已按 evidence 要求呈现（L1082）。
- **C14**（S3 docstring + S2 §4.1）：ag_rs/a2a 两后端、a2a 把 MLA 每层 NCCL 3→2、`VLLM_DCP_Q_REPLICATE=1`（来自 S2，定位正确）、"博客 future work vs main 已有 a2a"的时间线说明，均一致。
- **C15**（S2 §5.3/5.4）：MLA k_up 上投影、GQA 副本填切片 + tensor broadcast，逐字一致。
- **C16**（S2 §7）："DCP puts every GPU to work: sharding the sequence during attention, then immediately reconfiguring those same GPUs to amortize FFN weight loading across the full pool."，一致。
- **C17**（S1 Prefill 节）：两种策略的适用条件与表述、"Both approaches are under active development"，一致。
- **C18**（S4）：causal 负载不均与 $2 \times$ cp 配对切分，引文逐字一致（状态问题见 PR2-005）。
- **C19**（S3）：PCP 扩展/DCP 不扩展 world size、world_size = PP×TP×PCP，一致（页面 4.3 正文正确）。
- **C21、C22**（S1 + S2 §6）：DCP 支持近一年（"vLLM has supported DCP for almost a year"确认）、PCP roadmap、P/D disaggregation 加固表述，一致。
- **F1**（S1）："a request with T tokens in the context needs to store `H * T` key/value tensors"，逐字一致。
- **F2**：$t/H$ 重复与 $(t/H)/d$ 降副本为算术推导，构造模型 4 卡 32 格中 8 个不同的换算正确。
- **F3**：LSE 手算例子 Python 复算全部通过——$e^1=2.718282$、$e^2=7.389056$、$e^3=20.085537$、$e^4=54.598150$；$D_0=22.803819$、$D_1=61.987206$；$o_0=0.119203$、$o_1=0.880797$；$l_0=3.126928$、$l_1=4.126928$；merged = global = **0.675973**，`equal: True`；页面代码块预期输出与实际运行输出一致。
- **F4**：causal 例子 Python 复算通过——总负载 36，顺序切分 $[3,7,11,15]$（最慢 15，加速上界 $36/15=2.4$），首尾配对 $[9,9,9,9]$（加速上界 $36/9=4$）；公式 $(i+1)b^2 + (2N-i)b^2 = (2N+1)b^2$ 代数正确（锚点问题见 PR2-001）。
- **N1**（S2 §2.2/2.3）：基线并发 64 时 KV 100% 撞墙、平台期约 1,863 tok/s/GPU；DCP 并发 512 时 6,091 tok/s/GPU、KV 82%；200k+ 桶高且稳定前沿；实验条件（单节点 8×B200、Kimi K2.6 NVFP4、并发 16→512）逐项一致。
- **N2**（S2 §2.1）：中位 ~67k、输出 ~400、≈53% ≥64k（重尾 ~1M）、≈47% <64k、~18% <8k、~8% >128k、~3–4% >256k，一致。
- **N3**（S1 Case study）：R1（1 头，tp 8 重复 8 份，dcp 8）、K2（tp 16 重复 16 份，dcp 16 全消 / dcp 8 降 2 份且通信在节点内）、Qwen3-235B-A22B（4 头，tp 8 重复 2 份，dcp 2），逐字一致；R1 行"重复全消"为 $8/8=1$ 的算术换算，与第 3.1 节推导自洽。
- **N4**（S3 docstring）："Reduces NCCL calls from 3 to 2 per layer for MLA models."，逐字一致。
- **硬规则**：无 ASCII 框线图（index/overview 均 0 处）；两处结构图为内联 SVG + `foreignObject` 承载 KaTeX 标签，`<text>` 内仅纯文字数字，无 ASCII 近似公式；`.dojo/scripts/validate.py` 返回 validation ok。
- **问题块**：两级命名正确（"核心问题"/"本章问题"）；核心问题 5 题 + 各章本章问题（2/3/4/3/3 题）每题均配 `解答：` 折叠块，答案独立成段（含结论、推理、成立条件），核心问题答案末尾均指明完整论证章节；答案与正文结论核对一致。
- **章节结构**：h2 编号 1–5 连续、h3 按 3.1–3.4 / 4.1–4.3 编号；前置 section 顺序符合 style-guide（meta → 引言 → 核心问题 → 常见误解 → 正文）；章节衔接有承接句；折叠块 summary 前缀（补充/展开/代码）符合规范；正文无 S1/S2 代号残留（均以"官方文档/官方博客/源码"指称）。
- **C1–C22 上标完整性**：22 条论断上标全部在正文出现（含 [C1, C2] 组合形式），与文末来源表双向对应（F4 除外，见 PR2-001）。
- **链接**：9 个前置概念页（kv-cache、standard-attention、chunked-prefill、model-parallelism、mqa-gqa、mla、gpu-communication、causal-mask、ppd-disaggregation）全部存在；本地 libs 资源（katex/prism）齐全；overview.html 与 index.html 相互链接。
- **三处表面不一致的说明**（L1539）：S1 推荐范围 vs S3 硬校验、连续示意 vs 交错实现、A2A 时间线，处理方式与来源事实相符，未写成矛盾。
- **overview.html**：覆盖核心机制（TP 头维天花板、DCP 切存储与交错、三段通信与 LSE 合并恒等性、PCP 切计算与首尾配对、选型建议、PD 分离辨析、实测数字），无公式推导，符号均用 `$...$$`；数字与 N1 一致；互链正常。

## 统计与处置

- 统计：阻断 0 / 重要 3（PR1-001～003）/ 轻微 8（PR2-001～008）
- 处置：修复后进入第 2 轮独立审查
