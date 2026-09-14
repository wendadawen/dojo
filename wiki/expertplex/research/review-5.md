<!-- review-meta
round: 5
page: wiki/expertplex/index.html
reviewed_content_sha256: cbb292782b822daf
-->
# ExpertPlex 审查记录（第 5 轮）

- 页面版本：2d92b52afca6bd9cc99b152132f43fff28ac0d0a（index.html 工作树哈希）；overview.html = 305c085388099d454c64ff513ce680b0cf3fbcdc
- 论文版本：arXiv:2607.18002v2（v1 2026-07-20、v2 2026-07-21）
- 审查时间：2026-09-13 21:13
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题；1. 两条现有路线，各自的死结（1.1、1.2）；2. ExpertPlex 的架构：共享专家，分离注意力；3. APK：在 tile 边界上调度 GPU（3.1–3.4）；4. 通信：让 attention 侧发起一切（4.1–4.3）；5. 跨栈优化器：从 tile 建模到集群（5.1–5.3）；6. 实验：提升多少，在什么条件下成立（6.1–6.3）；7. 独立评价：三机制互相使能，但验证边界要看清（7.1–7.3）；来源与范围说明。含全部折叠块与图注。
- 来源获取：arXiv abs 页与 arXiv HTML 全文（arxiv.org/abs/2607.18002、arxiv.org/html/2607.18002v2）；本页 assets/ 下 6 张原图逐一打开比对。
- 机械验证：`python3 .dojo/scripts/validate.py wiki/expertplex/index.html` → validation ok；前置概念链接 ../moe-serving/index.html、../gpu-execution-model/index.html 均真实存在；alt 属性无 `$...$`；无「（待生成）」占位。

## 核对摘要（逐条回源，均为原文片段）

- §2.1 权重占比：「MoE weights contribute over 95% of model parameters」；DeepSeek-V4-Pro 95%、GLM-5.1-FP8 96%、MiniMax-M2.7 98%；「Attention holds under 5% of parameters」——与正文 C1/C6 及核心问题解答一致。
- §1/§2.4 PDD 部署单元：「32 GPUs for prefill and 320 GPUs for decode」「176 GPUs as a single unit」「128 H200 GPUs」（Kimi-K2）——与 §1.1 一致。
- §4.1 反差数字：「decode 17.7–34.7 μs」「prefill GEMMs with 16K tokens take 1.8–2.9 ms, or 84–101× longer」——与正文 [C5] 一致；1.8/17.7=101.7、2.9/34.7=83.6，倍数区间自洽。
- §2.5：「a prefill kernel may run for tens to hundreds of milliseconds while a decode kernel finishes in hundreds of microseconds」——与 [C4] 一致。
- §4.1 Table 1 与 GPU 共享设计空间：原文「MPS and Green Context support graph-compatible spatial multiplexing, but fix the allocation during a kernel」，MPS 缺「in-kernel time multiplexing and bounded preemption and reallocation」；API 拦截「offers only launch-boundary time multiplexing」且「lacks CUDA Graph compatibility, spatial multiplexing, and bounded preemption and reallocation」；MIG「H100 exposes only 1g, 2g, 3g, 4g, and 7g profiles, so the only two-way split … is 3g–4g」——与 §3.1 表格与对照段完全一致。
- §4.2/§4.3：tile 边界「2.2–25.3 μs」、GEMM「below 10.7 μs」；抢占上界「one tile execution time plus one local cluster check epoch … independent of the interrupted operation's total length」；切换边界「No accumulator, TMA transaction, shared-memory buffer … remains live … no checkpoint, restore, or recomputation」——与 §3.2/3.3 一致。
- §6.4：q′ 公式、单阶段就绪用全部 cluster、Q_max 保 prefill、先保 decode 的理由——与 §3.4 一致。
- §5.1/5.2：「Reserving receiver-side polling SMs avoids this liveness failure only by wasting compute capacity」；`WaitDone` kernel、NVLink peer stores / one-sided RDMA writes、NVLink loads / one-sided RDMA reads；「MoE servers expose only buffers and readiness words, with no matching communication kernel or reserved polling SMs」——与 §4 一致。
- §5.1 带宽：「DeepSeek-V3 reports 160 GB/s intra-node NVLink … only 50 GB/s cross-node InfiniBand … a 3.2× bandwidth gap」——与 [C20] 一致；160/50=3.2 复算通过。
- §6.1–6.4 四式（Eq.(1) G=min(B_p/T_p, B_d/(T_d·O̅))、Eq.(2) t̂_c=α+βx+γxs+δxs²、Eq.(3) x_moe=Σ⌈m_e/M_t⌉、Eq.(4) q′=min(Q_max,⌈q·x_moe/x_moe*⌉_c)）与 F1–F4 逐符号一致。
- §5.1 手算检查（构造示例，页面已标注）：B_p/T_p=8/0.5=16、B_d/(T_d·O̅)=64/(0.05×20)=64、G=min(16,64)=16；扩至 B_p=32 时两项均 64——分项与合计复算无误。
- §7.2 四组数字：11.3 req/s/node、5.65×/2.72×/2.01×/1.41×（MiniMax+ShareGPT）；MiniMax+LooGLE 对 Colocated 4.12×、对 PDMux 1.28×、ChunkedPrefill「cannot sustain the SLO」；GLM+ShareGPT 3.3×/1.5×、与 PDMux「about 1.5 requests per second per node」；GLM+LooGLE 5.0×/2.5×/1.66×；GLM PDD「runs out of memory … on 24 GPUs」，部分基线「on the largest compatible 16-GPU layout」——与 6.2 表逐格一致；正文与核心问题/本章问题解答三处数字一致。
- §7.3 微基准：CUDA stream「13.79×」、MPS「3.33×」、Green Context「4.07×」、ExpertPlex「8% … 1.12×」；设置「decode … 128 … prefill … 8192 … eight experts」「launched 10 μs after」——与 6.3 一致。
- §7.4/7.5/7.6：「less than 12%」「less than 20 μs」「below 10%」；「within about 5%」「within about 45 μs」；「below 25.3 μs」「below 10.7 μs」「REEF's best reported delay is 35 μs and requires recomputing the preempted kernel」——与 6.3 全部一致。
- §7.1 设置：SLO 1s/50ms、10s/100ms、2s/100ms、20s/100ms；8×H800、最多 3 节点、8×200Gbps IB；P90 goodput 定义；长度按 PDD KV-cache 截断；Poisson 到达；req/s/node；MiniMax「1P1D」；PDMux「based on the open-source MuxWise … modify it to support MoE … TP for attention and EP for MoE」——与 6.1 表一致。
- §8 归因：「attention-expert disaggregation systems… built on instance-level PDD, so they inherit the same limitations」——支持 §7.3 的一般性论断。
- 原图比对：img-06=Figure 2（「Larger DoP! (1.5x for prefill, 2x for decode)」「Prefill: 2/3」「Decode: 1/3」「Blocking!」「Bubble!」「Network Interference!」全在图中）；img-05=Figure 3（Prefill/Decode/MoE Server、Node 0/1）；img-04=Figure 4（CTA 0、CTA Cluster w/ DSMEM、System scope 的 P、Device scope 的 p_i）；img-03=Figure 5（WaitDone/Combine/Dispatch/Data Transfer/Preempt!、Decode 绿/Prefill 蓝图例）；img-02=Figure 6（激活专家越多延迟越高）；img-01=Figure 11（x=decode grouped GEMM 延迟、y=prefill grouped GEMM 延迟，CUDA Stream 点在最右）。编号与图注一一对应。
- 表述维度：全文逐段通读（含折叠块与图注），未发现「本页将…」「下面来看…」「需要注意的是」类元话语，未出现「我/我们/你」会话指代；`下图…`为 write.md 4.8 要求的图引导句，非元话语；无 Unicode 数学字符直排，h1/h2/h3、summary、正文、列表、表格内数学全部走 `$...$`；全文无「（待生成）」占位。

## 问题

- [轻微·一致性] overview.html 头部元信息：该行「arXiv 2026 v2 · 更新于 2026-08-07」中的日期与论文实际版本不符——arXiv 提交历史为 v1 2026-07-20、v2 2026-07-21，index.html 亦写「v2（2026-07-21 修订）」，同一「论文版本日期」在两页不一致。｜引文依据：arXiv abs 页「v2 revision: Tuesday, July 21, 2026」；index.html「arXiv 预印本，2026 年，v2（2026-07-21 修订）」。｜修复要求：把 overview.html 该行改为与 v2 实际日期一致（2026-07-21），或改写为明确指「页面更新日期」的措辞，使两页对同一事实的表述一致。｜修复：｜复验：
- [轻微·来源] §7.3 适用场景与位置：把相邻的 attention-expert 分离系统具名为「MegaScale-Infer、Step3-AFD、Janus」，但论文全文未出现这三个系统名，Related Work 用作者键引用（[Zhu et al., 2025b]、[StepFun, 2025]、[Zhang et al., 2026]）泛指该类系统，本页未为这三个具体系统名补来源、也未标注为解读者补充识别。｜引文依据：§8「these designs are built on instance-level PDD, so they inherit the same limitations」；全文检索 MegaScale-Infer / Janus 均无命中，Step3 仅出现 StepFun 的模型名「Step-3」。｜修复要求：为这三个系统名补外部来源，或改为论文的泛指表述（「相邻的 attention-expert 分离系统」）并注明具体系统名是解读者补充，避免让读者误以为出自论文。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（两条轻微问题建议随本轮修复一并处理；均不影响核心问题回答与原文一致性）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
