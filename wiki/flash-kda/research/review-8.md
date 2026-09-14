<!-- review-meta
round: 8
page: wiki/flash-kda/index.html
reviewed_content_sha256: c06120abb3b93ff0
-->
# FlashKDA 与 KDA Context Parallelism 审查记录（第 8 轮）

- 页面版本：index.html 工作树哈希 5d15087e90731e7886b9b2e89c4e40daf81aead4（sha256 3649483522ffeca8…）
- 审查时间：2026-09-14 16:54
- 审查者：编排者派发的独立审查者（未参与写作，也未参与前序轮次）
- 已完整阅读章节：核心问题 → 最容易误解 → 1. 串行状态 vs GPU 并行——为什么 KDA 在四个 regime 瓶颈不同 → 2. FlashKDA——把 chunk 内计算与 chunk 间状态传播重叠 → 3. 设备内 context parallelism——单 rank 的 SM 级切序列 → 4. KCP——为什么不能直接求和，以及 M+S̃ 分解（4.1–4.5）→ 5. KDA 解码——投影输入缓存与状态重建（5.1–5.3）→ 来源与范围说明（含全部折叠块与图注）

来源核对方式：下载 K3 官方仓库内技术报告 PDF（github.com/MoonshotAI/Kimi-K3 的 k3_tech_report.pdf），逐段比对 §2.1.1 / §5.1.1 / §5.1.2 / §5.4.2；config.json 取自 huggingface.co/moonshotai/Kimi-K3。以下关键点本轮均重新核对通过：

- Eq.1：报告原文「St = (I − βt kt k⊤t) Diag(αt)St−1 + βt kt v⊤t ，õt = S⊤t qt」——页面 Eq.1 与 M_t 定义、以及「Diag(α) 位于连乘最右、先按通道衰减旧状态再擦除」的描述一致。
- Eq.17：报告「Mt←1[i+1] := ∏_{r←1}^{t} Mr ∈ R^{dk×dk}，St[i+1] = S̃t[i+1] + Mt←1[i+1] S_{Ti}[i]」——页面 boxed 式一致；页面展开式 S_T^{[i]} = S̃_T^{[i]} + Σ_{j=1}^{i−1}(∏_{l=j+1}^{i} M_{T←1}^{[l]}) S̃_T^{[j]} 与报告 Eq.17 第二行（报告求和到 j=i、空积=I）等价。
- 手算（4.5）：2 rank × 2 token 逐矩阵复算无误——ground truth S_4=[[5.5,7],[8.5,10]]；M_{T←1}^{[1]}=M_{T←1}^{[2]}=0.5I；prefix scan 重组得 [[5.5,7],[8.5,10]]；误用直接求和得 [[6,8],[10,12]]。
- 引文：C1「The serial dependence of the KDA state is at odds with the GPU's preference for wide, uniform parallelism, and it manifests as a different bottleneck in each execution regime.」、C2「We therefore develop FlashKDA … is auto-dispatched as a backend of flash-linear-attention.」、C3「Tensor parallelism partitions heads across devices but never shortens the recurrence … incurs no cross-device communication.」、C4「This direct summation, however, is insufficient for KDA. …」/「we introduce KDA Context Parallelism (KCP) …」/「KCP requires only a fixed-size all-gather for recurrent-state synchronization and achieves linear compute scaling.」、C5「the primary bottleneck shifts …」/「Maintaining a state snapshot for each draft position …」/「The state after any accepted draft prefix … ReplaySSM.」/「Because the projection caches never leave the decode stage …」——逐字与报告一致。
- 数字：config.json head_dim=128、v_head_dim=128（故 d_k=d_v=128）、linear_attn_config.num_heads=96、顶层 dtype=bfloat16；报告表 #Layers=93、Attention-Layer Composition=69 KDA + 24 MLA；α=exp(g)∈(e^{−5},1)；128×128×2 B=32 KiB、两片段合计 64 KiB；H100 SXM5=132 SM。全部相符。
- 机械项：validate.py 返回 `validation ok: wiki/flash-kda/index.html`；正文/图注无 Unicode 数学字符（→、× 依 validate.py 规则属普通排版字符，不判错）；无 <img>，无 $…$ 落入 alt；无「（待生成）」占位；引用页 wiki/kda、wiki/gpu-execution-model、wiki/linear-attention 均存在；overview.html 与 index.html 互链；核心问题（5 条）与各章「本章问题」（每章 3 条）均配有「解答：」折叠块且与正文结论一致。

## 问题

- [轻微·格式] 图注（第 5 章 t1–t4 状态推进图，index.html:544）：该图注用中文弯引号「“回到 $S_a$”」，而全页其余引用（含其他图注、正文、误解题）一律用 ASCII 直引号；站内其他页（kda、linear-attention、kimi-k3、gpu-execution-model、delta-rule）弯引号计数均为 0。同页与站内引号体例不一致。｜引文依据：不适用｜修复要求：将 index.html:544 图注中的「“回到 $S_a$”」改为「"回到 $S_a$"」（ASCII 直引号），保留 $S_a$ 的 KaTeX 写法。｜修复：｜复验：
- [轻微·技术] 第 5.2 节（index.html:553）：正文写「$q$ 用于输出计算 $o_t = S_t^\top q_t$」，报告 Eq.1 中该量为带波浪号的 $\tilde o_t$（输出门之前的注意力输出，最终输出是 Eq.6 的 $y_t$）。页面去掉了波浪号，符号与所引来源不一致。｜引文依据：报告 §2.1.1 Eq.1「St = (I − βt kt k⊤t) Diag(αt)St−1 + βt kt v⊤t ，õt = S⊤t qt .」；Eq.6「yt = Wo [Sigmoid(Wg xt) ⊙ RMSNorm(õt)] .」｜修复要求：将 index.html:553 的 $o_t = S_t^\top q_t$ 改为 $\tilde o_t = S_t^\top q_t$（与 Eq.1 一致），可补一句说明它是输出门前的注意力输出。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：修复（仅两项轻微；修复后即可发布）