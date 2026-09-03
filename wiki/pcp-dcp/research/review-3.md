# PCP/DCP 第 3 轮审查

## 元信息

- 审查日期：2026-09-03
- 审查范围：index.html + overview.html
- 侧重：内部一致性 / 教学清晰度 / 可读性 / 公式定义 / 新增位置 / 新引入风险
- 审查方法：按 `guides/concept/check.md` §2.1 从头到尾完整通读（含全部折叠块），按 §2.2 对 S1（本地 v0.28.0 文档）、S2（2026-08-07 博客）、S3（main 分支 parallel.py）、S4（RFC #26133）逐条定位核对；对照 evidence.md、outline.md；独立复算全部手算数值；运行 `.dojo/scripts/validate.py`（返回 validation ok）。审查未参考前两轮报告的结论与修复记录。
- 页面版本：wiki/pcp-dcp/index.html（工作树当前版本）

## 问题清单

### P0

无。

### P1

**PR3-001｜技术·答案与正文及来源矛盾**
- 位置：index.html:1475（第 5 章本章问题 Q1 答案）
- 问题：答案称"源码硬校验 MLA 为 $t \ge d$ 且 $t \bmod d = 0$、GQA 为 $t/H \ge d$ 且 $(t/H) \bmod d = 0$"，把架构相关约束归为源码硬校验。这与三方面矛盾：① 第 5 章正文（index.html:1431）明确"main 分支源码目前的实际校验只看 $t \bmod d = 0$<sup>[C9]</sup>，比博客的 GQA 约束更松"；② S3 `_validate_parallel_config` 实际只有 `tp % dcp != 0` 与 `dcp not in (1, pcp, tp*pcp)` 两类校验，无任何基于 KV 头数的 MLA/GQA 上界检查；③ outline.md 第 5 章 Q1 答案要点为"源码校验 $t \bmod d = 0$"。读者按此答案会误以为源码强制 GQA 上界 $t/H$。
- 证据：S3 原文：`if pcp == 1: # DCP reuses the TP ranks when PCP is disabled. if tp % dcp != 0: raise ValueError(...)`；S2 §5.3/§5.4 的架构约束（`tensor_parallel_size >= decode_context_parallel_size` 等）出自博客而非 parallel.py。
- 建议处理：答案改为与正文一致的表述——"源码硬校验为 $t \bmod d = 0$（main 分支当前逻辑）；官方博客另按架构列出更精细约束：MLA $t \ge d$ 且 $t \bmod d = 0$、GQA $(t/H) \ge d$ 且 $(t/H) \bmod d = 0$"。

**PR3-002｜格式·markdown 链接残留**
- 位置：index.html:753（引言首段）
- 问题：首段残留 markdown 写法链接 `[KV cache](../kv-cache/index.html)（键值缓存）`，在 HTML 页面中不会渲染为链接，读者在全文第一段即看到裸露的方括号与圆括号原文，且该处的前置概念链接功能失效。全页检索确认 `\](` 仅此一处（第 1 章正文 816 行的同位链接是正常的 `<a>` 标签）。
- 证据：grep `\]\(` 全文唯一命中 index.html:753。
- 建议处理：改为 `<a href="../kv-cache/index.html">KV cache</a>（键值缓存）`。

**PR3-003｜技术·答案块与正文限定不一致 + 无源支持的可部署性表述**
- 位置：index.html:1482（第 5 章本章问题 Q2 答案）、index.html:799（核心问题 Q5 答案）
- 问题：第 5 章正文（index.html:1465）新增了限定"PCP 与 PD 分离的组合尚无官方支持说明（PCP 本身仍在开发中）"，但两个答案块未同步：Q2 答案仍称"分离式部署的 prefill 池内用 PCP、decode 池内用 DCP 是合法组合"，核心 Q5 答案仍称"两者正交可组合"。"合法组合"强于任何来源支持——S1、S2、S4 均无 PCP 与 PD 分离组合的支持说明（S2 §6 仅提 DCP 的 P/D 加固与 PCP 的 longer roadmap；S1 全文不涉及 PD 分离）。读者按答案会以为该组合当前可部署。
- 证据：S2 §6 原文："hardening prefill/decode (P/D) disaggregation support to make DCP robust in disaggregated serving deployments"（仅 DCP）；"there is a longer roadmap for Prefill Context Parallelism (PCP)"。evidence.md C22 自注"概念辨析性结论"。
- 建议处理：两个答案块补上与正文一致的限定，例如"正交性是概念层面的辨析；PCP 本身仍在开发中，其与 PD 分离的组合尚无官方支持说明"。check.md §2.2 第 11 条要求答案与正文结论一致。

### P2

**PR3-004｜格式·来源章节 h3 命名偏离固定命名**
- 位置：index.html:1496、index.html:1541
- 问题：style-guide §1 与 outline.md 规定来源章节 h3 固定命名为 `论断与来源（C）`、`公式与来源（F）`，页面实际使用 `核心论断与来源`、`核心公式与来源`（其余四个小节命名符合）。
- 证据：style-guide §1："来源章节（`来源与范围说明`）下的 h3 使用固定命名……`论断与来源（C）`、`公式与来源（F）`……"。
- 建议处理：两个 h3 改为固定命名，或经规范维护者确认后将 style-guide 的命名更新为页面用法（二选一，消除规范与页面的分歧）。

**PR3-005｜来源·C4 定位错误**
- 位置：index.html:1516（C 表 C4 行）
- 问题：C4"TP 沿头切 KV，$t > H$ 后重复 $t/H$ 倍"的来源定位写"S1 开篇"，实际引文位于 S1《Decode Context Parallel》节第 3 点；S1 开篇只有 prefill/decode SLO 差异的两句。evidence.md C4 的定位（Decode 节引文）是正确的。
- 证据：S1 原文位于 "3. Since `H` is limited (determined by the model architecture), when we continue to increase the tensor parallel size, the KV cache for each GPU will be duplicated for `tp_size / H` times."
- 建议处理：C4 定位改为"S1 Decode Context Parallel 节"。

**PR3-006｜来源·C21 定位部分错误**
- 位置：index.html:1533（C 表 C21 行）
- 问题：C21 定位写"S1；S2 第 6 节"，但"已支持约一年"出自 S2 第 1 节（"vLLM has supported DCP for almost a year, but we are writing this blog now..."），第 6 节只有 PCP 的 longer roadmap。
- 证据：S2 §1 原文如上；S2 §6："there is a longer roadmap for Prefill Context Parallelism (PCP)."
- 建议处理：C21 定位改为"S1；S2 第 1 节（约一年）、第 6 节（PCP roadmap）"。

**PR3-007｜来源·C9 行论断不完整，双向对应缺口**
- 位置：index.html:1521（C 表 C9 行）；正文引用处 index.html:1431、799
- 问题：正文两处以 C9 支撑"无 PCP 时源码校验 $t \bmod d = 0$"，但 C9 行论断只写"PCP 不支持 DP；PCP 开启后 $d \in \{1, \mathrm{pcp}, \mathrm{tp}\times\mathrm{pcp}\}$"，缺整除约束，正文引用在来源表中无对应行项。evidence.md C9 包含整除条款。
- 证据：S3 `_validate_parallel_config`：`if pcp == 1: if tp % dcp != 0: raise ValueError(...)`。
- 建议处理：C9 行论断补上"无 PCP 时要求 $t \bmod d = 0$"。

**PR3-008｜来源·无标注的缺位性论断**
- 位置：index.html:1465（第 5 章 PD 分离段末句）
- 问题："PCP 与 PD 分离的组合尚无官方支持说明（PCP 本身仍在开发中）"是无来源编号、未标注推断的缺位性论断。内容经核对属实（S1/S2/S4 均未谈及该组合），但按 check.md §2.2 第 4、5 条，无来源支持的论断应删除或降级为明确标注的推断。
- 证据：不适用（三来源检索均无 PCP+PD 分离组合表述）。
- 建议处理：改为明确标注的推断，如"据本文核对的三处来源（写作时），PCP 与 PD 分离的组合尚无官方支持说明"。

**PR3-009｜一致性·Kimi K2.6 与 Kimi-K2 两个模型未作区分提示**
- 位置：Kimi K2.6——index.html:799、1247、1290、1561；Kimi-K2——index.html:1096、1438–1439、1445、1475；N3 行又用"R1/K2"缩写（index.html:1563）
- 问题：Kimi K2.6（S2 实测模型）与 Kimi-K2（S1 case study 模型）是两个不同模型，页面未在任何地方提示二者不同，读者很容易当成同一模型的两种写法（一个 tp 16 的 case 与一个 8×B200 实测），造成张冠李戴。N3 行的"R1/K2"缩写与正文全名混用，同类问题。
- 证据：S2 §2.2 "serving Kimi K2.6 in NVFP4"；S1 Case study 节 "For Kimi-K2, the architecture is similar to DeepSeek-R1, but with more parameters."
- 建议处理：在 Kimi-K2 首次出现处（index.html:1096 或第 5 章表格）加一句说明（如同为 MLA 架构的不同模型，case study 取自官方文档、实测取自博客）；N3 行改用全名。

**PR3-010｜一致性·引言对构造示例的描述与实际不符**
- 位置：index.html:762（引言末段）
- 问题："全文的手算例子围绕一个构造的极小模型（有效 KV 头数为 1、8 个 token、4 张卡）推进"——实际只有第 2 章与 3.1 使用该模型；3.3 的 LSE 例子（2 卡、4 token）与第 4 章的 causal 例子（8 块、4 卡）是另外两个独立构造。来源章节"构造示例"自己就分列了三者。
- 证据：index.html:1571（构造示例节分列三个例子）。
- 建议处理：引言改为"主要例子围绕一个构造的极小模型推进，另有两个局部小例子"，或删去"全文的"限定。

**PR3-011｜可读性·前向指引落空**
- 位置：index.html:836（第 1 章 chunked prefill 段）
- 问题："两者的区别将在第 4 章展开 PCP 机制时再次对照"——第 4 章全文（4.1–4.3 及本章问题）未再出现 chunked prefill（grep 确认 chunked 仅见于第 1 章）。承诺的回访不存在，属于对读者的失约指引。
- 证据：grep `chunked` 命中仅 index.html:836、850、853（均在第 1 章）。
- 建议处理：删去该句的承诺部分，或在第 4 章合适位置（如 4.1 策略选择处）补一段与 chunked prefill 的简短对照。

**PR3-012｜可读性·折叠块位置指引错误**
- 位置：index.html:1127（3.2 节末段）
- 问题："MLA 的 a2a 后端正是把每层 NCCL 调用从 3 次压到 2 次来省这笔账（本节末的补充折叠块）"——该折叠块（"补充：a2a 通信后端与 query 投影复制"，index.html:1237）实际位于 3.3 节末（3.3 的 MLA/GQA 折叠块之后、3.4 标题之前），不在 3.2 节内。
- 证据：index.html 结构：h3 dcp-communication（1098）→ h3 dcp-lse-merge（1129）→ a2a 折叠块（1237）→ h3 dcp-benefits（1243）。
- 建议处理：改为"（见 3.3 节末的补充折叠块）"或将折叠块移至 3.2 节末。

**PR3-013｜格式·公式符号定义未按 style-guide §11 执行；$v_i$ 无文字定义**
- 位置：全部 4 个行间公式——index.html:864（$\max(1,t/H)$）、1133（$l_r,o_r$ 定义式）、1137（合并式）、1316（配对负载式）
- 问题：重新评估结论：缺符号定义 `<ul>` 不是只出现在三个核心公式，而是全部行间公式的系统性偏离；但除 $v_i$ 外，其余符号（$t$、$H$、$s_i$、$o_r$、$l_r$、$b$、$N$、$i$）均在相邻正文有文字定义，实际阅读障碍集中在 $v_i$——它在 1133 行公式中首次出现，全文无一处文字定义（第 1 章只出现过"query/key/value"字样，未建立 $v_i$ 与 value 的对应）。定级 P2：规范层面的系统性偏离 + 一个真实的符号缺口。
- 证据：style-guide §11"公式后紧跟 `<ul>` 逐项定义每个符号"；1133 行前文只定义了"$s_i$ 为 query 对它们的分数……部分输出 $o_r$ 与本地 LSE $l_r$"。
- 建议处理：至少在 1133 行公式前补"$v_i$ 为第 $i$ 个 token 的 value 向量"；同时评估为全部行间公式统一补符号 `<ul>`，或在 style-guide 中放宽该条并记录接受理由。

**PR3-014｜可读性·术语在核心问题答案块首现未解释**
- 位置：index.html:778（GQA、MLA 首现于核心 Q2 答案）、785（LSE 首现于核心 Q3 答案，"rank" 同）、overview.html:49、54
- 问题：GQA/MLA 的展开在第 2 章、LSE 的展开（log-sum-exp）在 3.3，但三者首次出现都在其前的核心问题答案块中，无解释；"rank"全文未与"第 r 张卡"作对应说明。overview.html 同样裸用 GQA/MLA/LSE。对无领域背景读者，核心问题区是第一阅读区。
- 证据：不适用（可读性）。
- 建议处理：在核心问题答案的首现处加最简括注（如"GQA（分组查询注意力）、MLA（多头潜注意力），见第 2 章""LSE（log-sum-exp，对数和指数）"）；overview 酌情加一句缩写提示。

**PR3-015｜一致性·overview 细节未完全对齐**
- 位置：overview.html:49、53
- 问题：① "Qwen3-235B 为 4 个"缺 A22B 后缀（index 与 S1 均为 Qwen3-235B-A22B，模型名不完整）；② "命令行加一个 `-dcp` 即可"漏掉 size 参数（index 与 S1 均为 `-dcp <size>`）。其余数字与机制表述经逐项核对一致（见通过项）。
- 证据：S1："This is as simple as adding `-dcp <size>` to the command line."
- 建议处理：overview 两处补齐为 Qwen3-235B-A22B 与 `-dcp <size>`。

**PR3-016｜来源·多处定位只到文档级或弱定位**
- 位置：index.html:1519（C7）、1520（C8）、1532（C20）、1534（C22）
- 问题：C7/C8/C20 定位仅写"S1"，未到节（三者均在 Decode Context Parallel 节，C20 为该节末"In short"句）；C22 的"S1 整体结构"是弱定位——S1 全文并无 PD 分离内容，"正交可组合"实际只靠 S2 §6 一句 future work 加页面自身的结构辨析。
- 证据：S1 结构：开篇 / Prefill Context Parallel / Decode Context Parallel / Technical Discussions，无 disaggregation 字样。
- 建议处理：C7/C8/C20 补到"S1 Decode Context Parallel 节"；C22 定位改为"S2 第 6 节（DCP 的 P/D 加固）+ 页面辨析"，与 evidence.md C22 的自注一致。

**PR3-017｜可读性·第 2 章答案使用未引入的 dcp/d 概念**
- 位置：index.html:1071（第 2 章本章问题 Q3 答案）
- 问题：答案使用"dcp 的有效上界——$d = t/H$ 时总副本数恰好降为 1 份"，此时正文尚未引入 dcp 与 $d$（首次定义在 3.1"d 为 dcp size"，命令行参数在 ch3 开篇）。答案块要求独立可读，但该答案未自行定义 $d$。
- 证据：不适用（可读性）。
- 建议处理：答案内加一句"（dcp：decode 阶段沿 token 维继续切 KV 的参数，$d$ 为其取值，见第 3 章）"。

**PR3-018｜格式·"8×B200"中的 Unicode ×**
- 位置：index.html:799、1247、1290、1561（另 overview 无此写法）
- 问题："8×B200"使用 Unicode 乘号 ×，按 style-guide §11"任何位置出现的……数学运算符……必须包在 $...$ 中"的字面要求应为 LaTeX；validate.py 当前通过（未将该上下文判为数学字符），说明是规范字面与工具判定之间的灰区。
- 证据：grep `[×≈≥≤∈⊙√]` 命中即上述 4 行的 "8×B200"。
- 建议处理：二选一——统一改为 `$8\times$B200` 之类的写法，或在 style-guide 中明确硬件命名（如 8×B200、4×A100）豁免于该条；记录接受理由。

## 通过项

以下经本轮独立核对通过：

1. **5 个核心问题与 5 章一一对应**，"完整论证见第 X 章"指引全部正确（Q1→1、Q2→2、Q3→3、Q4→4、Q5→5）；误解区 3 条的章节指引（M1→第 3 章、M2→第 5 章、M3→第 4 章）均正确。
2. **M3（"DCP 需要增加 GPU"）表述精确**：dcp 不扩展 world size、DCP 组复用 TP rank（S1"size does not increase the number of GPUs we need to launch"与 S3 docstring"DCP does not expand the process world size. Without PCP, DCP reuses TP ranks"双确认）；PCP 扩展设备数（S3 `__post_init__`：`world_size = PP * TP * PCP`）。误解归因于资源差异而非通信差异，与 4.3 节一致。
3. **关键数字全页口径一致**：1,863 / 6,091 / 约 3.3 倍（均标"页面换算"，6,091/1,863=3.27 复算成立）/ 并发 64→512 / KV 82% / 67k / 53% / 重尾约 1M，在核心问题、误解区、3.4、第 5 章、N1/N2 与 overview.html 之间逐处一致；S2 原文逐条核对相符（含"从 16 扫到 512"）。
4. **LSE 手算与代码全部复算正确**：$e^1..e^4$、$D_0=22.803819$、$D_1=61.987206$、$o_0=0.119203$、$o_1=0.880797$、$l_0=3.126928$、$l_1=4.126928$、合并=基线=0.675973；代码逻辑与"预期输出"块一致（静态审查 + 数值复核）。
5. **causal 配对手算正确**：$(i+1)b^2+(2N-i)b^2=(2N+1)b^2$；顺序等分 3/7/11/15、配对全 9、总负载 36、加速上界 36/15=2.4 与 36/9=4；与 S4 配对规则原文（"divided into 2 × cp_world_size chunks. Each CP rank i is assigned both the i-th chunk and the (2 × cp_world_size - i - 1)-th chunk"）一致；页面正确呈现为"RFC 描述的设计"且注明 closed as not planned（S4 状态核实相符）。
6. **"沿 token 维"与"沿序列维"** 在第 1 章末（index.html:838）明确定为同义（"沿 token 维（序列维）"），后文两种说法并存不构成歧义。
7. **"三处表面不一致的说明"仍覆盖当前实际**：①S1 推荐 $[1,t/H]$ vs S3 仅硬校验整除（S3 全文确认无头数上界校验）✓；②S2 §4 连续区间示意 vs 交错实现（S3 interleave docstring）✓；③S2 §6 future work A2A vs S3 已有 a2a 选项 ✓。未发现需要新增的第四处来源间不一致。
8. **机制表述逐条对源成立**：交错规则与 Chao Hong/Helix 出处（S1+S3 docstring+S4 round-robin）、三段通信节奏（S2 §4.1）、a2a NCCL 3→2（S3 docstring 原文"Reduces NCCL calls from 3 to 2 per layer for MLA models"）、VLLM_DCP_Q_REPLICATE（S2 §4.1）、k_up/tensor_broadcast（S2 §5.3/5.4）、attention 分片 + FFN 重组（S2 §7）、三个官方 case（S1 Case study 节逐项相符）。
9. **第 5 章正文对"推荐范围 vs 硬校验"的两层呈现正确**（与 S1/S2/S3 三方相符）；"理论上可超 $t/H$、非 attention 层无明确分工、为简单起见取上界"与 S1 原文一致。
10. **第 2 轮新增句的实质内容属实、未过度限定**："PCP 与 PD 分离的组合尚无官方支持说明"与三来源现状一致（问题仅在标注方式与答案块同步，见 PR3-003/008）；其前一句"vLLM 正在加固 DCP 在分离式部署中的支持"与 S2 §6 原文一致。
11. **页面机械项**：validate.py 返回 ok；KaTeX 定界符使用正常；details summary 前缀（补充/展开/代码/解答）规范；前置 section 顺序符合 style-guide §2；h1 格式、章节编号连续；overview 与 index 相互链接；正文用词符合 §12（自称"本文"）；前置概念链接在首次依赖处给出且密度合理（除 PR3-002 一处失效）。
12. **overview.html 与 index 的机制与数字对齐**（除 PR3-015 两处细节外逐项一致）：DCP 切存储/PCP 切计算、交错规则、三段通信、LSE 恒等式表述、选型建议、$[1,t/H]$、实测数字、"PCP 活跃开发 / DCP 已支持约一年"、与 PD 分离正交可组合，均与 index 及来源一致。

## 统计与处置

- 统计：P0 0 / P1 3 / P2 15
- 处置：建议修复（3 条 P1 均为局部修改：1 处答案改写、1 处链接语法、2 处答案块补限定；P2 可逐条酌情处理或记录接受理由）。修复后需复验 PR3-001/003 的答案块与正文一致性。
