# Chunked Prefill 审查记录（第 2 轮）

- 页面版本：index.html `c80b72c60d860960d1ab43412aded6541d1429d8`；overview.html `aa4ac12dc71f4b28c5c022234bf7078aa0f19cef`
- 审查时间：2026-08-19
- 审查者：独立子代理（第 2 轮，未参与写作与第 1 轮）
- 已完整阅读章节：引言与学习目标、核心问题、1 长 prefill 为什么卡住所有人、2 把长 prefill 切成块（2.1/2.2）、3 decode 搭车与不停止的调度（3.1/3.2）、4 块要多大、5 与流水线并行合流 CPP、6 边界与相邻工作、来源与范围说明、overview 全文

## 第 1 轮 9 项修复核查

1. [C16,N4]→[C16]：到位。Sarathi 数字仍标 [C14,N4]，Beyond the Buzz 结论仅标 [C16]。
2. Unicode × 换 $\times$：到位（EP $\times$ PP 两处；构造示例乘号均为 LaTeX）。
3. 缩略语首现补全称：TBT、MLA 到位；FTL 仅正文（第 5 章）补了全称，更早的核心问题 5 未补；ISL、EP 全页无全称——未完全落实（见轻微问题）。
4. N_0 删除：未到位（见重要问题 2）。
5. 20ms 示意值：到位（"几十毫秒的间隔（示意值）"）。
6. overview 5 项 head meta 与 [1][2] 实验条件：到位，[1][2] 原文与 sarathi.txt 一致。
7. 2.2 节列表结构：引入新错误（见阻断问题 1）。
8. "翻平方"表述：到位（N² 量级表述两处）。

## C/F/N 逐条核对（引文依据）

- C1 ✓ sarathi-serve §2："The prefill phase is compute-bound ... whereas the decode phase is memory-bound"
- C2/N1 ✓ sarathi §1："decode cost per token can be as high as ∼ 200 times the prefill cost per token"
- C3/N2 ✓ sarathi §1："on an A6000 GPU, for the LLaMA13B model, a prefill with a sequence length of 512 tokens saturates GPU compute even at a batch size of just one"
- C4 ✓ sarathi 摘要："splits a prefill request into equal sized chunks"
- C5 ✓ sarathi-serve §5.4.1："the first chunk's KV-cache is loaded N − 1 times, the second ... N − 2 times"；"even at small chunk sizes attention prefill operation is compute-bound"；"computational cost is unchanged"
- C6：推断，正文 2.1 已文字标注；但编号未在正文出现（见轻微问题 3）
- C7 ✓ sarathi 摘要："decode requests 'piggyback' and cost up to an order of magnitude less compared to a decode-only batch"
- C8 ✓ sarathi 摘要："decode-maximal batching, which constructs a batch using a single prefill chunk and populates the remaining slots with decodes"
- C9 ✓ sarathi-serve §2/Fig.1："a generation stall in vLLM can last over several seconds"
- C10 ✓ sarathi-serve §4："token_budget, τ"；"adds new requests in a batch without pausing ongoing decodes"
- C11 ✓ sarathi-serve §4.3："one-time profiling ... maximum number of tokens that can be packed in a batch without violating TBT SLO"
- C12/N3 ✓ sarathi-serve §4.3："chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256"
- C13 ✓ sarathi-serve §5.4.1："smaller chunks introduce higher overhead"；"fixed overheads of kernel"
- C14/N4 ✓ sarathi 摘要："reduces bubbles by 6.29×, resulting in an end-to-end throughput improvement of 1.91×"
- C15/N5 ✓ sarathi-serve 摘要："2.6× (Mistral-7B single A100) ... 3.7× (Yi-34B two A100) ... 5.6× (Falcon-180B PP)"
- C16 ✓ paper §4："we found Chunked Pipeline Parallelism (CPP) to be especially effective"；"optimal strategy to maximize throughput while complying with strict FTL SLA"；"DeepSeek-R1 with ISL of 256K on 64 GPUs (EP × PP = 64)"
- C17 ✓ paper §4："redundant computation of down and up projections in multi-latent attention for each prefill chunk. This can be mitigated by temporarily caching the up-projected KV values"
- F1 ✓ $\frac{N(N-1)}{2}$ 由 C5 计数求和；N=8→28、N=4→6 复算正确
- N6 ✓ sarathi 摘要："decode throughput by up to 10× ... 1.33×（LLaMA-13B/A6000）；4.25× ... 1.25×（LLaMA-33B/A100）"
- 链接：五个前置概念页均存在（mechanical check 通过）；overview↔index 互链正常

## 问题

- [阻断·格式] index.html 2.2 节（行 889–907）：第 1 轮列表结构修复引入内容重复与结构损坏——display 公式 $$\sum_{i=1}^{N}(N-i)=\frac{N(N-1)}{2}$$ 及符号说明列表（"$N$：一个 prompt 被切成的块数…"）连续出现两次（行 890–893 与 896–900）；"三个代价"被拆成两个 `<ul>`（KV cache 重复读单独一个，算术强度与固定开销在另一个），中间夹裸 `<p>` 公式、裸说明 `<ul>` 与"计算量本身不变"段落。渲染后正文连续出现两个相同公式与重复文字。｜引文依据：不适用｜修复要求：删除重复的公式段与符号说明 `<ul>`，使三个代价位于同一个 `<ul>` 的三个 `<li>` 中，公式与符号说明并入第一个 `<li>`｜修复：｜复验：
- [重要·技术] index.html 第 4 章本章问题 1 解答（行 1000）："KV 重复读按 $\frac{N(N-1)}{2}$ 从 $\frac{L}{1024}\cdot\frac{N_0-1}{2}$ 量级涨到 16 倍"：(a) $N_0$ 未删除且未定义；(b) "16 倍"与同句实例矛盾——$N=4$ 得 6 块次、$N=16$ 得 120 块次，精确比值为 20 倍，16 倍仅为 $N^2$ 近似，同句并列精确数字与近似倍数自相矛盾；(c) 此处 $L$ 指 prompt 长度，与第 2 章图示中 $L$ 指模型层数符号冲突。｜引文依据：sarathi-serve §5.4.1 "the first chunk's KV-cache is loaded N − 1 times, the second ... N − 2 times"｜修复要求：删去 $N_0$ 表达式；改为"从 6 块次涨到 120 块次（约 20 倍，按 $N^2$ 量级增长）"或等价表述；prompt 长度改用文字或非 $L$ 符号｜修复：｜复验：
- [轻微·可读性] index.html 核心问题 5 解答（行 791）："严格 FTL 约束"为全页 FTL 首次出现，无全称；全称在第 5 章（行 1011）才补。第 1 轮缩略语修复未覆盖核心问题块。｜引文依据：不适用｜修复要求：行 791 首现处补全称或改文字表述｜修复：｜复验：
- [轻微·可读性] index.html 行 1011、1026："ISL 256K"、"EP $\times$ PP = 64" 中 ISL、EP 全页均无全称。｜引文依据：不适用｜修复要求：行 1011 首现处补 ISL（Input Sequence Length，输入序列长度）、EP（Expert Parallelism，专家并行）｜修复：｜复验：
- [轻微·格式] index.html 来源说明（行 1056）声明 C6 编号，但正文无任何 `[C6]` 标记（2.1 节推断仅文字标注"基于证据的推断"）。论断列表与正文标记不一致。｜引文依据：不适用｜修复要求：行 861 推断处补 `<sup>[C6]</sup>`，或从论断列表删去 C6 编号｜修复：｜复验：
- [轻微·格式] index.html 2.1 节 SVG（行 866–870）`<text>` 内"过全部 L 层"中 L 为数学变量的 ASCII 写法，规范要求图内公式用 `<foreignObject>` + KaTeX，`<text>` 内无 ASCII 近似。｜引文依据：不适用｜修复要求：将该 text 改为"过全部层"或用 foreignObject 渲染 $L$｜修复：｜复验：
- [轻微·可读性] overview.html：[1][2] 的来源原文注（行 56）放在"核心机制"标题下首段，而引用出现在上一节"问题背景"（行 53）。｜引文依据：不适用｜修复要求：来源注移至"问题背景"引用段之后｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 5
- 处置：修复。阻断项（2.2 节结构重复）与重要项（$N_0$/16 倍/$L$ 符号冲突）修复并复验后，方可进入第 3 轮审查。本轮为审查轮次，未修改 index.html，未运行 validate.py（修复后由修复者运行）。


## 修复记录

第 2 轮所有问题已修复。重要/阻断问题逐条对应：

### chunked-prefill
- 阻断：2.2 节列表结构重构——将原来拆成两个 `<ul>` 的"切块的三个代价"合并为单 `<ul>` 的三个 `<li>`，公式与符号说明并入第一条 `<li>` 的子 `<ul>`，删除重复的公式与符号说明段。复验：grep 重复的 $\sum\frac{N(N-1)}{2}$ 0 重复。
- 重要：第 4 章本章问题 1 解答重写——删 $\frac{L}{1024}\cdot\frac{N_0-1}{2}$（$N_0$ 未定义且 L 与第 2 章层数符号冲突），改为 "KV 重复读按 $\frac{N(N-1)}{2}$ 从 6 块次涨到 120 块次（精确比值 20 倍，按 $N^2$ 量级增长，对应 prompt 长度 4096 的实例）"。复算：$N=4 \to 6$ 块次、$N=16 \to 120$ 块次、$120/6=20$。
- 轻微：核心问题 5 解答补 FTL 全称；第 5 章 ISL/EP 全称；§2.1 推断加 [C6] 标记；2.1 节 SVG `<text>` 内"过全部 L 层" → "过全部层"（L 改为纯文字避免 ASCII 数学近似）；overview 脚注位置 [1][2] 从"核心机制"标题下移到"问题背景"段后。
