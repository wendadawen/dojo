<!-- review-meta
round: 1
page: wiki/moonep/index.html
reviewed_content_sha256: 0c5a7f35e1719fe6
-->
# MoonEP 完美均衡专家并行审查记录（第 1 轮）

- 页面版本：53377fb50b5981c2433219e70086caa2ec99d32f
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取本页 research/ 规划/修复/前序审查记录）
- 已完整阅读章节：核心问题、1. 传统 EP 的不均衡——根源与 MoonEP 的核心思路、2. 冗余专家的界——$E/R$ 上界与基本紧性、3. 完美均衡的工程收益——buffer、host 同步与 forward/backward 流程、4. MoonEP 的边界——解决与不解决，以及与 ECHO/UltraEP/DeepEP 的区别、来源与范围说明（含全部折叠块、两个图示与图注）

## 来源与核对方式

- 主来源：Kimi K3 Technical Report（arXiv:2607.24653，https://arxiv.org/html/2607.24653），逐条定位到 §5.2.1 与附录 §E 的原文片段核对。
- 辅来源：MoonEP 仓库 https://github.com/MoonshotAI/MoonEP（页首声明不引用其源码，仅用于交叉确认 §5.2.1 术语）。
- 已核对通过（无问题）的核心论断：每 rank 恰好收 $S\times K$（§5.2.1 "requires every rank to receive exactly S×K tokens"）；Theorem 1 $M(I)\le E/R$ 与构造性填充证明（终止性、至多 $R-1$ 次 fill、远端 token 同源）（§E）；Theorem 2 $M(I^*)=\lceil E(R-1)/R^2\rceil\approx E/R$ 及最坏构造（rank 0 空载、其余 $R-1$ rank 均分，每专家收 $SKR^2/(E(R-1))$）（§E "each expert receives SKR²/(E(R-1)) tokens"）；buffer $S\times K\times R\to S\times K$ 且带 "Under worst-case imbalance" 限定（§5.2.1）；每层 host 同步与静态形状（§5.2.1）；planning kernel 近最优 + ILP 离线参考（§5.2.1）；backward reduce 回 home rank（§5.2.1）；ECHO/UltraEP 预设 cap 可能中断（§5.2.1）；Expert-GEMM workload-aware scheduler 与 shared expert 独立 stream（§5.2.1 末节）；$E=4,R=2,S=4,K=1$ 两情形示例的全部数字可复算（情形 B：$\lceil 4\times 1/4\rceil=1<E/R=2$，正确）。
- 未能执行项：页面代码块标注为"伪代码"（`language-text`），按 check.md §2.2 第 3 条改为静态审查，已核对与 §5.2.1 的三步/两步流程一致，未发现与来源冲突。

## 问题

- [重要·技术] 来源与范围说明 / 外部数字与实验条件（N）（index.html:496）：声称"报告未公开 K3 训练 MoE 的具体 $E, R, K, S$ 取值"，与来源不符——报告 §2.3 明确公开了 $E=896$ 与 $K=16$。｜引文依据：报告 §2.3 Stable LatentMoE："This enables Kimi K3 to scale channel mixing to 896 routed experts with 16 active experts per token, corresponding to a sparsity of 56."；§2.3 另处 "activates 16 of 896 routed experts per token"；§5.2.1 仅 "Let E be the number of experts and R the EP size."（只定义 $E,R$，确未给 $R,S$ 数值）。｜修复要求：把该句改为"报告公开了 $E=896$、$K=16$（§2.3），未公开 EP size $R$ 与每 rank 序列长度 $S$ 的具体取值；本页构造示例（$E=4,R=2,S=4,K=1$）为教学构造"，不得保留"未公开 $E,R,K,S$"的整体表述。｜修复：｜复验：
- [重要·技术] 4.2 MoonEP 不解决的三个问题（index.html:421；同一断言亦见 index.html:100 核心问题解答、index.html:516 全文总结）：把"内存碎片"列为 MoonEP 不解决的问题并断言"与完美均衡无关"，与来源矛盾——报告 §5.2.1 开篇把"routed-expert 激活形状动态变化导致显存碎片"直接列为 MoonEP 要解决的问题（该句紧接 "We therefore propose MoonEP"）。｜引文依据：报告 §5.2.1 开头："the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation. We therefore propose MoonEP ..."；§5.2.2 中 "fragmentation" 的原文为 "avoiding multi-stream fragmentation"（内存池层面的碎片，与激活形状碎片不是同一问题）。｜修复要求：删去"与完美均衡无关"的断言；或改写为"报告 §5.2.1 把显存碎片归因于激活形状动态（MoonEP 的静态形状消除该动态性），§5.2.2 另从内存预算角度处理（unified activation manager 以单一内存池避免 multi-stream fragmentation）"，并同步修正 index.html:100、index.html:516 的对应表述。｜修复：｜复验：
- [轻微·表述] 引言后（index.html:106）：元话语 + 固定引入句"下面先用一个最小例子把"不均衡"具体化。"，违反 style-guide §4"示例按用途标记……不使用固定引入句"与 check.md §2.2 第 12 条的元话语排除项。｜引文依据：不适用｜修复要求：删除"下面先用一个最小例子把"不均衡"具体化。"一句，直接以"构造示例。设 $E=4$ 个专家……"起句。｜修复：｜复验：
- [轻微·表述] 1 章开头（index.html:110）、3 章开头（index.html:261）、全文总结（index.html:516）：元话语。"先回顾 EP 的基础数据流……"、"接下来看完美均衡带来的三个工程收益怎样落地……"、"回到开篇的问题：……——以上四章已逐一回答。"均属对读者/文章结构的自指，check.md §2.2 第 12 条要求排除。｜引文依据：不适用｜修复要求：改为直接陈述内容（§1 直接给出 EP 数据流；§3 直接给出本章要讲的三项收益与 forward/backward 差异；总结直接给结论），删去"先回顾""接下来看""回到开篇的问题""以上四章已逐一回答"等引导语。｜修复：｜复验：
- [轻微·表述] 核心问题解答（index.html:79）、1 章正文（index.html:112）：会话指代"完全由"我持有的 $E/R$ 个专家被全网 token 选了多少次"决定"，两处使用第一人称"我"，属 check.md §2.2 第 12 条排除的会话指代。｜引文依据：不适用｜修复要求：两处均改为"该 rank 持有的 $E/R$ 个本地专家"，全页不得再出现第一人称。｜修复：｜复验：
- [轻微·表述] 章间过渡（index.html:161、232、384、443）：四章过渡共用同一句式"本章说明了 X。但 Y——下一章讲 Z。"，与 style-guide §8"不使用固定句式，也不为形式完整而添加过渡"不符（四章同构，读感机械）。｜引文依据：不适用｜修复要求：至少打散其中两处句式，避免四章使用同一模板化过渡。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复。核心论断（$S\times K$ 不变量、$E/R$ 上界与基本紧性、buffer 与 host 同步收益、forward/backward 流程、ECHO/UltraEP/DeepEP 对比）均已逐条对照 §5.2.1 与 §E 原文核对一致，无阻断问题；两条重要问题（$E,K$ 已公开却称未公开、内存碎片归属写反）须按来源改写后再复验。页面机械项（LaTeX 渲染、伪代码标注、图示为 HTML 结构、核心/本章问题两级均有解答折叠块、前置概念链接 moe-serving 与 gpu-execution-model 均真实存在、无"（待生成）"占位）均正常，`.dojo/scripts/validate.py` 返回 "validation ok"。