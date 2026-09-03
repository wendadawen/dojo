# PCP/DCP 第 2 轮审查

## 元信息
- 审查日期：2026-09-03
- 审查范围：index.html + overview.html
- 侧重：章节衔接 / 公式严谨 / 表述与支撑一致 / 折叠块职责 / 表格一致 / 双向对应
- 审查方法：按 `guides/concept/check.md` 执行。通读 index.html 全文（含全部折叠块、图示、来源章节）与 overview.html；对 S1（本地 fork 全文）、S2（博客全文摘录）、S3（main 分支 parallel.py 摘录）、S4（issue #26133）逐条核对；F2/F3/F4 公式与全部手算数值独立复算。
- 工具调用统计：正文提取与编号统计（grep 全部 C/F/N 上标）、来源四份逐条核对、图示坐标逐弧线核对。

## 问题清单

### P0

（无。核心结论——两阶段瓶颈分化、TP 头维天花板与 $t/H$ 重复、DCP 交错分片与 LSE 恒等合并、PCP 首尾配对、选型两层约束——均正确且来源支撑充分。）

### P1

#### PR2-101 [格式/链接] index.html L753（导语第 1 段）
- 问题：导语首段存在 markdown 链接残留 `[KV cache](../kv-cache/index.html)`，浏览器会按纯文本渲染原始语法，链接失效，违反 check.md 2.2-6"前置概念链接有效"。第 1 章正文（L816、L820）的同名链接均为正确的 `<a>` 标签，唯此处残留。
- 证据原文片段（页面 L753）：`decode 阶段每步只生成一个新 token，却要背着全部历史的 [KV cache](../kv-cache/index.html)（键值缓存）往下走`
- 建议处理：改为 `<a href="../kv-cache/index.html">KV cache</a>`，与第 1 章写法一致。

#### PR2-102 [技术/来源一致性] index.html L1475（第 5 章"本章问题"问题 1 解答）
- 问题：解答写"源码硬校验 MLA 为 $t \ge d$ 且 $t \bmod d = 0$、GQA 为 $t/H \ge d$ 且 $(t/H) \bmod d = 0$"。这把博客（S2 §5.3/§5.4）的模型侧约束误标为"源码硬校验"：S3 `_validate_parallel_config` 在 `pcp == 1` 时只校验 `tp % dcp != 0` 报错，无 MLA/GQA 之分、无 `>=` 约束。且与同章正文 L1431（"main 分支源码目前的实际校验只看 $t \bmod d = 0$"）和折叠块 L1462（源码引文）直接矛盾——同一事实页内两种说法，读者无法判断源码行为。
- 证据原文片段（S3 `_validate_parallel_config`）：`if pcp == 1: # DCP reuses the TP ranks when PCP is disabled. if tp % dcp != 0: raise ValueError(f"tp_size={tp} must be divisible by dcp_size={dcp}.")`；页面 L1431："main 分支源码目前的实际校验只看 $t \bmod d = 0$<sup>[C9]</sup>，比博客的 GQA 约束更松"
- 建议处理：将解答中"源码硬校验"一句改为与正文一致的两层表述：源码（C9）只校验 $t \bmod d = 0$；$t \ge d$（MLA）与 $(t/H) \ge d$ 且整除（GQA）是博客（C10）给出的模型侧使用约束。

#### PR2-103 [双向对应/编号] index.html L1312–L1316（第 4.2 节）与来源表 L1550
- 问题：F4 双向对应不成立。来源表声明"正文上标编号 C1–C22、F1–F4、N1–N4"并列有 F4 行（"块负载 $(j+1)b^2$；配对和 $(2N+1)b^2$ 恒定"），但正文中 F4 上标出现次数为 0（grep 全文确认）：第 4.2 节 causal 负载与配对和公式（L1316 `$(i+1)\,b^2 + (2N-i)\,b^2 = (2N+1)\,b^2$`）只有 `[C18]` 上标，无 `[F4]`。来源表条目失去正文锚点，读者无法从公式回查来源行。
- 证据原文片段（页面 L1314–L1316）：`rank $i$ 同时拿第 $i$ 块和第 $2N-1-i$ 块<sup>[C18]</sup>。每卡负载（$b$ 为块长，单位 $b^2$）：$$(i+1)\,b^2 + (2N-i)\,b^2 = (2N+1)\,b^2$$`；来源表 L1550：`<tr><td>F4</td><td>块负载 $(j+1)b^2$；配对和 $(2N+1)b^2$ 恒定</td>…`
- 建议处理：在 L1314 或 L1316 处补 `<sup>[C18, F4]</sup>`（负载公式首次出现处），使 F4 在正文有引用锚点。

#### PR2-104 [公式书写/一致性] index.html L792（核心问题 4 解答）
- 问题：解答中"PCP 扩展 world size（设备数变为 TP × PCP）"存在两个问题：(a) 使用 Unicode `×` 直接书写乘法，违反 check.md 2.2-9"数学符号全部由 KaTeX 渲染……无 Unicode 数学字符直接出现"；(b) 表述与第 4 章 4.3 节正文及对照表（L1386、L1394）的 $\mathrm{PP}\times\mathrm{TP}\times\mathrm{PCP}$ 不一致——遗漏 PP 维度，未按统一口径呈现。
- 证据原文片段（页面 L792）：`资源区别：PCP 扩展 world size（设备数变为 TP × PCP），花硬件买延迟`；页面 L1386：`设备数变为 $\mathrm{PP} \times \mathrm{TP} \times \mathrm{PCP}$`（S3 `__post_init__`：`self.world_size = (self.pipeline_parallel_size * self.tensor_parallel_size * self.prefill_context_parallel_size)`）
- 建议处理：改为 `设备数变为 $\mathrm{PP}\times\mathrm{TP}\times\mathrm{PCP}$（单 PP 级即 $\mathrm{TP}\times\mathrm{PCP}$）`，与第 4 章及误解区（L810）口径统一。

#### PR2-105 [来源状态] index.html L1314（第 4.2 节）与来源表 L1504（S4 行）
- 问题：C18/F4 的来源 S4（RFC #26133）当前状态为 **closed as not planned**，页面全文无任何状态标注（grep "closed"/"not planned" 无结果）。第 4.2 节写"vLLM 的 RFC 给出的解法是首尾配对"、来源表只写"（2025-10）"。读者会误以为该设计是 vLLM 进行中的实现计划；而 S1 同时说明 PCP 两种策略"under active development"，两者的现状必须区分：首尾配对出自一份已被关闭、未计划实施的 RFC，其与现网 PCP 实现的关系页面未交代。
- 证据原文片段（S4 issue #26133 页面顶部状态）：`Closed as not planned`；页面 L1504：`S4：vLLM RFC #26133《[RFC]: Support Context Parallelism with Fully Sharded KV Cache and Ring Attention》（2025-10）`
- 建议处理：在来源表 S4 行（及第 4.2 节首次提及 RFC 处）注明"该 RFC 已关闭（closed as not planned），配对切分作为设计思想呈现，不代表已落地实现"，与 evidence.md C18 的"作为 RFC 设计呈现"定位一致。

### P2

#### PR2-201 [数值一致性] index.html L808（常见误解第 1 条）
- 问题：误解区写"实测的'吞吐约 3 倍'"，而核心问题 5 解答（L799）写"约 3.3 倍（6,091/1,863，页面换算）"。同一页面对同一组数字给出两个口径（3 倍 / 3.3 倍）。来源 S2 只给出 1,863 与 6,091 两个原始值，比值系页面换算，应全页统一。
- 证据原文片段（页面 L808）：`实测的"吞吐约 3 倍"来自并发能力，不是单请求速度（第 3 章）`；页面 L799：`升到 6,091（约 3.3 倍，页面换算）`
- 建议处理：统一为"约 3.3 倍"（或两处均写"约 3 倍"），保证单一口径。

#### PR2-202 [表述精度] index.html L818（第 1 章第 2 段）
- 问题：第 1 章写"vLLM 对长上下文 prefill 的目标，就是把这部分计算摊到多个 rank 上并行"，而 C2 对应的 S1 原文分摊对象是 query tokens（"amortizing the computation time of the prefill across query tokens"）——即把 prefill 计算摊到各段 query token 上，多卡各算一段；"rank"是实现载体而非分摊对象。表述与来源原文的对象存在偏移，易与第 4 章"每卡算一段 query"的机制脱节。
- 证据原文片段（S1 开篇）：`For long context prefill, we need to control the TTFT (time to first token) by amortizing the computation time of the prefill across query tokens.`
- 建议处理：改为"把这部分计算摊到各段 query token 上，由多个 rank 并行承担"一类表述，使分摊对象与来源及第 4 章机制一致。

#### PR2-203 [来源定位] index.html L1533（来源表 C21 行）
- 问题：C21 行来源定位写"S1；S2 第 6 节"，但"DCP 已支持近一年"（"vLLM has supported DCP for almost a year"）实际位于 S2 第 1 节引言；S2 第 6 节只有 PCP roadmap（"there is a longer roadmap for Prefill Context Parallelism (PCP)"）。定位到错误章节，按 check.md 2.2"定位到页面标注的位置，确认该位置的实际内容支持页面的表述"无法通过。
- 证据原文片段（S2 第 1 节）：`vLLM has supported DCP for almost a year, but we are writing this blog now to highlight the feature…`；S2 第 6 节：`…and there is a longer roadmap for Prefill Context Parallelism (PCP).`
- 建议处理：C21 行来源改为"S1；S2 第 1 节（支持近一年）、第 6 节（PCP roadmap）"。

#### PR2-204 [推断边界] index.html L1465（第 5 章 PD 分离段）
- 问题："分离式部署的 prefill 池内可以用 PCP 压 TTFT，decode 池内可以用 DCP 提吞吐"——后半句有 S2 §6"hardening … disaggregation support to make DCP robust in disaggregated serving deployments"支撑，前半句（PCP 用于分离部署 prefill 池）无任何来源直接支撑：PCP 两种策略本身"under active development"（C21），其与 PD 分离的组合支持状态未知。来源表 C22 虽标注"辨析性结论"，正文此句以肯定语气陈述了一个来源未涉及的组合。属于轻微过度引申。
- 证据原文片段（S2 第 6 节）：`as well as hardening prefill/decode (P/D) disaggregation support to make DCP robust in disaggregated serving deployments.`（仅涉及 DCP 与分离部署）
- 建议处理：将前半句降级为推断标记（如"概念上，分离式部署的 prefill 池内也可以用 PCP 压 TTFT"或删去），或注明"PCP 与分离部署的组合尚未有官方支持说明"。

## 通过项（确认正确的点）

1. **章节衔接与论证完整度**：第 1→2→3→4→5 章承接自然（第 1 章结尾设问"沿 token 维切 KV cache 为什么有必要"引出第 2 章；第 2 章结尾"沿 token 维把 KV cache 继续切下去，就是下一章的 DCP"；第 4 章开头"decode 侧的答案是 DCP，prefill 侧的问题与解法都不一样"；第 5 章"机制看完，回到部署视角"）。第 1 章"两阶段瓶颈不同"的结论被第 2 章（存储维度的天花板）、第 3 章（decode 侧解法）、第 4 章（prefill 侧解法）分别拆解，无逻辑跳跃或未支撑论断。
2. **构造模型跨章复用一致**：$H=1$、$T=8$、$\mathrm{tp}=4$ 在第 2 章（重复表）、第 3.1 节（交错分配表）正确复用；"消除全部重复的临界点正是 $d = t/H = 4$"在第 3.1 节正文出现（L1094），与第 2 章"重复因子 $t/H = 4$"及本章问题解答一致。
3. **第 4 章"PCP 扩展 / DCP 不扩展"核心结论在第 5 章保持一致**：第 5 章选型讨论不引入设备数变化；误解区（L810）、4.3 节对照表、C19 均为 $\mathrm{PP}\times\mathrm{TP}\times\mathrm{PCP}$ / 复用 TP rank 口径（唯核心问题 4 解答例外，见 PR2-104）。
4. **F2 推导链闭合**：从 $H\times T$ 条目（L860）→ $\max(1, t/H)$ 重复因子（L862–866，含 $t\le H$/$t>H$ 两情形与整除条件声明）→ 加 dcp 后 $(t/H)/d$（L1096），每步均有构造模型数值例证（4 卡 32 条目 8 不同；d=4 后每卡 2 token、4 份降 1 份）；"简化条件"章节（L1579）显式声明整除假设与非整除时按 C10 整数部分约束。
5. **F3 恒等式完整且数值全部复算无误**：折叠块（L1141–1148）给出分子分母的完整代数（$D_r$ 代换、双重求和并单重、划分不重不漏的说明），非跳步；手算例 $e^1+e^3=22.803819$、$e^2+e^4=61.987206$、$l_0=3.126928$、$l_1=4.126928$、$o_0=0.119203$、$o_1=0.880797$、合并 $57.316432/84.791025=0.675973$ 与整卡基线逐项复算一致；代码折叠块逻辑与预期输出正确（标量简化、float64、减最大值不改恒等式的声明均在）。
6. **F4 代数与几何说明正确**：$(i+1)b^2+(2N-i)b^2=(2N+1)b^2$ 消去 $i$；手算表（顺序等分 3/7/11/15 最慢 15、配对 9/9/9/9、总 36、加速上界 36/15=2.4 与 36/9=4）复算无误；配对图四条弧线（块 0–7、1–6、2–5、3–4）与 GPU 0–3 标签一一对应，图说与表一致。
7. **C 论断与来源核对一致**（抽查全部关键条目）：C1/C2/C3/C4/C7/C8/C17/C20/C21(内容)/F1/N3 与 S1 原文一致；C5/C6/C10/C12/C13/C15/C16/C22(内容)/N1/N2 与 S2 原文一致；C9/C11/C14/C19/N4 与 S3 docstring 及 `_validate_parallel_config` 一致；C18 配对规则与 S4 正文一致（"the sequence is divided into 2 × cp_world_size chunks. Each CP rank i is assigned both the i-th chunk and the (2 × cp_world_size - i - 1)-th chunk"）。
8. **表格数据一致性**：case study 表（R1 tp8/8 份/dcp8；K2 tp16/16 份/dcp16 或 dcp8 降 2 份节点内；Qwen3 H=4/tp8/2 份/dcp2）与 S1 及 N3 一致；PCP vs DCP 对照表、实测表（1,863/6,091/82%/64→512）、第 2 章构造模型表、LSE 手算表均与正文及来源一致。
9. **双向对应（除 F4 外）**：正文上标集合恰为 {C1–C22} ∪ {F1, F2, F3} ∪ {N1–N4}（grep 逐编号统计确认），来源表 22+3+4 行全部有正文锚点、无多余条目；S5（Helix）、S6（Ring Attention）在正文出现且列入来源列表；evidence.md 中 S7 未纳入的取舍与页面一致。
10. **折叠块职责划分合理**：源码校验、a2a 后端、MLA/GQA 路径、F3 完整代数均在正文有自足概括（折叠块收起后正文结论完整），折叠块只承载细节；"本章问题"与"核心问题"每题均有解答折叠块，答案独立可读（唯第 5 章问题 1 解答与正文矛盾，见 PR2-102）。
11. **第 5 章 PR1-002 修复段连贯**：L1431 段"有效范围 $[1, t/H]$（C8）→ 博客按架构的更精细约束（C10）→ 源码实际校验只看 $t \bmod d = 0$（C9）→ 理论上可超 $t/H$ 但上界取 $t/H$（C8）→ 三个 case（N3）"行文顺畅，与前后段落及折叠块自洽。
12. **overview.html 与 index.html 一致**：Qwen3-235B 4 个 KV 头、$t/H$ 重复、交错规则 $i \bmod d$、三段通信节奏、LSE 恒等式、$2N$ 块首尾配对、$[1, t/H]$ 取值、实测数字（1,863/6,091/82%/64→512）、负收益三场景、PD 分离辨析、支持状态——全部能在 index.html 找到一致对应；两页相互链接（index 导航 L727 有 overview.html，overview 导航有 index.html）。
13. **页面元信息完整**：`description`、`dojo:summary`、`dojo:type=concept`、`dojo:topics`、`dojo:tag` 均存在且为有效内容。

## 统计与处置建议

- 统计：P0 0 / P1 5 / P2 4
- 处置：修复。P1 中 PR2-101/103/104 为机械性修复；PR2-102/105 需按建议改写表述并复核来源定位。修复后建议对第 5 章问题解答与来源表做一次针对性复验。
