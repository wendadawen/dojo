<!-- review-meta
round: 5
page: wiki/moonep/index.html
reviewed_content_sha256: 8bea572df53343f1
-->
# MoonEP 完美均衡专家并行审查记录（第 5 轮）

- 页面版本：index.html 工作树哈希 6fca38db6748248a8800e9901ffa274625a243cd（overview.html 2e6c0e42a7244034e9af8f2ab63f930793d102d0）
- 审查时间：2026-09-13 20:16
- 审查者：编排者派发的独立审查者（子代理，未参与写作与前序轮次）
- 已完整阅读章节：核心问题（4 条解答折叠块）→ 引言/构造示例 → 1. 传统 EP 的不均衡——根源与 MoonEP 的核心思路（含本章问题 3 条）→ 2. 冗余专家的界——E/R 上界与基本紧性（含补充/展开折叠块、本章问题 3 条）→ 3. 完美均衡的工程收益——buffer、host 同步与 forward/backward 流程（含代码折叠块、本章问题 3 条）→ 4. MoonEP 的边界（含本章问题 3 条）→ 来源与范围说明 → 全文总结
- 来源核对：Kimi K3 Technical Report（arXiv:2607.24653，HTML 全文）§5.2.1 与附录 E 逐句定位；MoonEP 仓库页核对；本地链接 moe-serving / gpu-execution-model 存在；`python3 .dojo/scripts/validate.py wiki/moonep/index.html` 返回 "validation ok"。
- 来源原文关键片段（核对依据）：§5.2.1 "MoonEP requires every rank to receive exactly $S\times K$ tokens, where $S$ is the sequence length and $K$ is the number of experts selected per token"；"preserves the overall computation flow of conventional schemes such as DeepEP [148] and additionally introduces online planning and migration of redundant experts"；"Under worst-case imbalance, supporting the same copy-free data path in DeepEP requires a communication buffer of size $S\times K\times R$, whereas MoonEP requires only a fixed $S\times K$ buffer"；"the per-expert token counts vary across steps and layers, and the host must synchronize with the device at every layer...This eliminates the per-layer MoE host synchronization and alleviates the host-side kernel-launch overhead"；"prior work such as ECHO [138] and UltraEP [133] presets the number of redundant experts or imposes a per-rank token cap...the cap itself requires manual tuning while still leaving residual imbalance"；"Even with the aggregate load perfectly balanced across ranks, the per-expert token counts within each rank remain skewed"。§E："$M(I)=\min_{P}\max_{r}\{m_{r}(P)\}$"、"each fill makes one underloaded rank balanced and it never changes afterwards, so the process terminates after at most $R-1$ fills"、"these tokens belong to at most $E/R$ local experts on rank $s$"、"all $S\times K\times R$ tokens are evenly divided among $E(R-1)/R$ experts, so each expert receives $\frac{SKR^{2}}{E(R-1)}$ tokens"、"$\left\lceil\frac{E(R-1)}{R^{2}}\right\rceil\approx\frac{E}{R}$ when $R$ is large"。页面构造示例复算：$E/R=4/2=2$；$\lceil E(R-1)/R^2\rceil=\lceil 4\cdot1/4\rceil=1$；$SKR^{2}/(E(R-1))=4\cdot1\cdot4/(4\cdot1)=4$；$4/4=1$；$S\times K\times R=8$，$8+0=8$——算式与结论一致，无算术错误。

## 问题

- [重要·技术] overview.html「3. 关键结论与边界」第 5 条（边界）vs index.html「4.2 MoonEP 不解决的问题」：两页对"显存碎片是否被 MoonEP 解决"给出相反结论——overview 断言碎片"随完美均衡的静态形状消除"，index 断言报告未声明碎片被消除｜引文依据：overview 原文"routed-expert 激活形状动态变化造成的显存碎片则随完美均衡的静态形状消除（§5.2.2 memory-efficient training 处理的是激活/梯度/优化器状态的显存预算，与此不同）"；index 原文"报告 §5.2.1 把 routed-expert 激活形状动态变化列为 rank 间负载不均衡的直接后果[C1]，但未声明 MoonEP 消除碎片——MoonEP 固定的是每 rank 总量 $S\times K$，报告同节指出 rank 内 per-expert token 数仍随层偏斜"；报告 §5.2.1 原文 "Even with the aggregate load perfectly balanced across ranks, the per-expert token counts within each rank remain skewed"，报告仅在 §5.2.1 开头把碎片列为不均衡的后果（"the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation"），未声明 MoonEP 将其消除｜修复要求：统一两页口径，以 index 的"报告未声明被消除"为准，删去 overview 该句中的"随完美均衡的静态形状消除"断言，改写为"该碎片在报告中列为 rank 间不均衡的后果，报告未声明被完美均衡消除（每 rank 总量固定为 $S\times K$，rank 内 per-expert 形状仍随层变化）"｜修复：｜复验：
- [重要·技术] index.html「来源与范围说明 · 外部数字与实验条件（N）」：把报告术语 3T-class 记成"训练规模为 3T 级"，与报告的实际指代不符｜引文依据：报告全文 "3T-class" 共 6 处，均指模型/参数规模——§5.2 标题 "Infra for 3T-class Pre-Training"、首段 "scaling the pre-trained foundation to unprecedented 3T-class parameters"、§5.2 "Natively multimodal pre-training at the 3T-class"；报告全文无任何训练数据/训练 token 规模的表述（检索 "trillion token"、"T token"、"15T"、"21T" 命中 0）。页面原文"训练规模为 3T 级（报告 §5.2 提及 "3T-class"）"｜修复要求：删去"训练规模为 3T 级"分句；如需保留该术语，改写为"报告以 3T-class 指代模型规模（原文 "3T-class parameters"），K3 为 2.8 万亿参数级"，不得把 3T 记为训练数据规模｜修复：｜复验：
- [轻微·表述] index.html 元话语与页面自我指路句：§2 开头"现在把"要多少冗余专家"形式化。"；§2 段"注意它不要求每 rank 冗余数相等，只要求最坏 rank 的冗余数不超过给定值。"；§3.1 段"注意这里的 $S\times K\times R$ 是**最坏不均衡**下的 buffer 需求"；§3 本章问题解答"——注意这是最坏不均衡下的需求"；§4 末尾"上述论断、公式与教学简化的完整来源定位集中在「来源与范围说明」一节。"｜引文依据：不适用｜修复要求：逐句改为直接陈述——"形式化要多少冗余专家：给定一个 router 输出 $I$……"；"这个目标不要求每 rank 冗余数相等，只要求最坏 rank 的冗余数不超过给定值。"；"这里的 $S\times K\times R$ 是最坏不均衡下的 buffer 需求"；"——这是最坏不均衡下的需求"；末句删去或改为陈述式（如"公式与教学简化的来源定位见下文"。全页不再出现"注意……"式的读者指令句与"上述……集中在……一节"式指路句）｜修复：｜复验：
- [轻微·表述] index.html 引言第 1 段："本文依据 K3 技术报告 §5.2.1 与 §E 附录，不引用未公开的源码。"与同页 blockquote 自称的"MoonEP 开源仓库"矛盾——源码已公开｜引文依据：报告 §5.2.1 脚注给出 "https://github.com/MoonshotAI/MoonEP"，该仓库公开（仓库标题 "MoonEP: A Perfectly Balanced Expert Parallelism Library via Dynamic Redundant Experts"，HTTP 200）；页面 blockquote 原文"MoonEP 开源仓库 github.com/MoonshotAI/MoonEP；本页不引用其源码"｜修复要求：把"不引用未公开的源码"改为"不引用其源码实现"，与 blockquote 口径一致｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 2
- 处置：修复（无阻断项；两条重要项为跨页结论冲突与来源术语误记，关闭后需重新核对；两条轻微项为表述层面，可直接改写）

## 已核对无误、不列入问题的条目（抽样）

- §5.2.1 各段与页面 C1–C10 引用逐条对应，§E Theorem 1/Theorem 2 的陈述、构造与页面复述一致；F1–F4 定位正确（F4 已明确标注为"由 C3 与总量守恒推出，无独立报告来源"）。
- 构造示例 $E=4,R=2,S=4,K=1$ 全部算式可复算且一致（见上方来源核对段落）。
- $E=896$、$K=16$ 与报告 §2.3 "896 routed experts with 16 active experts per token" 一致；"2.8-trillion-parameter scale" 确在 §2.3。
- "[C7] 最坏不均衡 $S\times K\times R$" 保留 "Under worst-case imbalance" 限定，并声明报告未说明非最坏情况的分配策略，与报告一致。
- 核心问题 4 条、各章本章问题均配有 `解答：` 折叠块且结论与正文一致；正文链接（moe-serving、gpu-execution-model）与 overview.html 互链有效；无"（待生成）"占位、无 research/ 文件路径引用。
