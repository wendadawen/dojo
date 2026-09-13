<!-- review-meta
round: 2
page: wiki/expertplex/index.html
reviewed_content_sha256: 331a07dbed423a76
-->
# ExpertPlex 审查记录（第 2 轮）

- 页面版本：index.html 工作树哈希 63f0c1f12e5a3f425c786cea78f90fc1808fd572
- 论文版本：arXiv:2607.18002v2（v1 2026-07-20，v2 修订 2026-07-21）
- 审查时间：2026-09-13
- 审查者：编排者派发的独立子代理
- 已完整阅读章节：核心问题（页级）、贯穿引言、1. 两条现有路线各自的死结（含本章问题）、2. ExpertPlex 的架构（含本章问题）、3. APK：在 tile 边界上调度 GPU（含本章问题）、4. 通信：让 attention 侧发起一切（含本章问题）、5. 跨栈优化器：从 tile 建模到集群（含本章问题）、6. 实验：提升多少，在什么条件下成立（含本章问题）、7. 独立评价：三机制互相使能，但验证边界要看清（含本章问题）、来源与范围说明（全部小节），含全部折叠块、图注与隐藏章节。另核对了 overview.html。

## 核对依据（来源侧，逐条已定位）

- §2.1 第 120 行：`95% in DeepSeek-V4-Pro ..., 96% in GLM-5.1-FP8 ..., and 98% in MiniMax-M2.7`；§3 第 90 行 `Attention holds under 5% of parameters`。页面 95%/96%/98%、attention 不足 5% 一致。
- §1/§2.4 第 164–166 行：`32 prefill and 320 decode GPUs`、`176 GPUs as a single unit`、`Kimi-K2 ... 128 H200 GPUs`。页面一致。
- §2.5 第 190 行：`a prefill kernel may run for tens to hundreds of milliseconds while a decode kernel finishes in hundreds of microseconds`。页面一致。
- §4.1 第 246 行：`with EP4 on MiniMax-M2.7, ... 17.7–34.7 μs, while matched prefill GEMMs with 16K tokens take 1.8–2.9 ms, or 84–101× longer`。页面 17.7–34.7 微秒 / 1.8–2.9 毫秒 / 84–101× 一致（1.8ms÷17.7μs≈101.7，2.9ms÷34.7μs≈83.6）。
- §4.2 第 274 行：`tile boundaries occur every 2.2–25.3 μs, independent of the total operation length`；§7.6 第 558 行：`below 25.3 μs, ... GEMM intervals stay below 10.7 μs`。页面 2.2–25.3 微秒、<10.7 微秒一致。
- §4.1 Table 1（HTML 解析）：ExpertPlex 是唯一同时具 CUDA Graph / Temporal / Spatial / Bounded Fast Preemption / Bounded Fast Reallocation 的列；API interception 仅 Temporal。页面「APK 是表中唯一五条全占的」一致。
- §4.1 第 266 行：`H100 exposes only 1g, 2g, 3g, 4g, and 7g profiles, so the only two-way split ... is 3g–4g`。页面一致。
- §4.3 第 272 行：`system-scope word P`、`one device-scope word p_i for each cluster i`、`broadcasts the decision through DSMEM`、`mbarrier handoff`、`bounded by one tile execution time plus one local cluster check epoch`。页面一致。
- §5.1 第 322–326 行：两侧 ring buffer + credit、跨阶段死锁、`160 GB/s intra-node NVLink ... only 50 GB/s cross-node InfiniBand ..., a 3.2× bandwidth gap`。页面一致。
- §5.2 第 328–335 行：`NVLink peer stores or one-sided RDMA writes`、`a small WaitDone kernel`、`NVLink loads or one-sided RDMA reads`、`no matching communication kernel or reserved polling SMs`。页面一致。
- §5.3 第 347–353 行：`same local rank on that node`、`multicasts it out over NVLink`、`lower-priority InfiniBand virtual lane than decode`。页面一致。
- §6.1 Eq.(1) 第 366 行：`G(\ell,q)=\min(B_p/T_p, B_d/(T_d \bar O))`。页面 F1 一致。
- §6.2 Eq.(2) 第 380 行：`\hat t_c(x,s)=\alpha_c+\beta_c x+\gamma_c x s+\delta_c x s^2`；Eq.(3) 第 384 行：`x_moe=\sum_{e|m_e>0}\lceil m_e/M_t\rceil`，且 `with x=x_moe and s=1`。页面 F2/F3 一致。
- §6.4 Eq.(4) 第 425 行：`q'=\min(Q_max, \lceil q x_moe / x_moe^\star \rceil_c)`。页面 F4 一致；「只有一个阶段就绪时用全部 CTA cluster」对应第 424 行 `When only one phase is ready, it uses all CTA clusters`。
- §7.1 第 456–465 行：`256 routed experts ... activate eight routed experts per token`、`230 GB FP8`、`756 GB FP8 and 724.8B routed expert parameters ... about 22.6B ... full attention ... DSA`、`up to three machines`、`eight 200 Gbps InfiniBand NICs per node`。页面设置表一致。
- §7.1 第 488–491 行 SLO：1s/50ms、10s/100ms、2s/100ms、20s/100ms；页面一致。第 487 行 P90 goodput 定义一致。
- §7.1 第 475–479 行基线：`MiniMax-M2.7 uses a 1P1D deployment`、`GLM-5.1-FP8 runs out of memory under this PDD layout on 24 GPUs`、`based on the open-source MuxWise implementation`、`compatible only with tensor-parallel attention`、`those baselines run on the largest compatible 16-GPU layout`。页面一致。
- §7.2 第 496–516 行：11.3 req/s/node、5.65×/2.72×/2.01×/1.41×；LooGLE 4.12×/1.28×、ChunkedPrefill `cannot sustain the SLO`；GLM 3.3×/5.0×、1.5×/2.5×、ShareGPT `about 1.5 requests per second per node` 持平、LooGLE `1.66×`。页面 6.2 表逐格一致。
- §7.3 第 532–534 行：`CUDA streams increase decode latency by 13.79× ... MPS and Green Context ... slow prefill by 3.33× and 4.07× ... ExpertPlex adds only 8% overhead to the decode phase and slows prefill by only 1.12×`；第 483 行 `128 ... 8192 ... eight experts`、`launched 10 μs after`。页面微基准表一致。
- §7.4 第 544–548 行：`less than 12% overhead`、`less than 20 μs`、`falls below 10%`。页面一致。
- §7.5 第 551–553 行：`On 16 GPUs ... within about 5% ... within about 45 μs`。页面一致。
- §7.6 第 560–562 行：`REEF's best reported delay is 35 μs and requires recomputing the preempted kernel`、`lack ... TMA multicast, CTA clusters, warp specialization, and CUDA Graph`。页面一致。
- 作者与单位（arXiv 作者块）：`Bingyang Wu / Chao Jin / Zili Zhang / Xinming Wei / Yinmin Zhong / Ruidong Zhu / Xin Jin — Peking University；Chengxu Yang / Yuliang Liu — Independent Researcher`。页面 meta 一致。
- 原图核对（逐张读取 assets/）：img-06=Figure 2、img-05=Figure 3、img-04=Figure 4、img-03=Figure 5、img-02=Figure 6 均与页面图注对应；img-01=Figure 11 内容对应，但坐标轴方向与图注不符（见问题 1）。
- 机制性论断：§8 第 576–577 行 `Attention-side techniques ... are orthogonal`、`MoE load-balancing ... also orthogonal`；参考文献 `Zhu et al. (2025b) = MegaScale-Infer`、`StepFun (2025) = Step-3`、`Zhang et al. (2026) = Janus`、`Zhao et al. (2025) = DeepGEMM`、`DeepEP v1`、`SGLang`。页面第 7 章解读与之一致。
- 机械项：`validate.py wiki/expertplex/index.html` → `validation ok`；`dojo:type=paper`、`dojo:topics=推理系统`（在 ALLOWED_TOPICS 内）、`dojo:tag=MoE`（在 ALLOWED_TAGS 内）；`index.html ↔ overview.html` 互链；`../moe-serving/index.html`、`../gpu-execution-model/index.html` 均真实存在；无 `（待生成）` 占位、无 research/ 路径引用、无 Unicode 数学字符、无 ASCII 框线图。手算示例 $B_p/T_p=16$、$B_d/(T_d\bar O)=64$、$G=16$，扩 $B_p$ 至 32 后 $G=64$，与页面表述一致。

## 问题

- [重要·技术] 「6. 实验」Figure 11 图注（index.html 第 497 行）：横纵轴的表述与原文图颠倒。页面写「横轴 prefill 性能、纵轴 decode 延迟」，而原文 Figure 11 的横轴是 decode 延迟、纵轴是 prefill 延迟，读者照此读图会把两个方向整体对调。｜引文依据：本页 assets/img-01.webp（即原文 Figure 11）标签为横轴 `Latency of decode's grouped GEMM (us)`、纵轴 `Latency of prefill's grouped GEMM (us)`；图例为 Exclusive / Green Context / ExpertPlex / CUDA Stream / MPS。｜修复要求：把图注改为「横轴 decode 延迟、纵轴 prefill 延迟」，并复核同句「左下区域」的方位描述（低 decode 延迟=左、低 prefill 延迟=下，故「decode 低延迟 + prefill 高性能」落在左下，含义不变，保留即可）；该图无坐标轴以外的倍数数字，不需改数值。｜修复：｜复验：

- [轻微·表述] 全文多处元话语与临场修辞（index.html 第 134、163、241、475 行）。第 134 行「先立一个贯穿全文的具体场景：」「这就是全部故事的起点。」；第 163 行「把贯穿场景代进去：」；第 475 行「注意一个反例：」；第 241 行「这些机制在 GPU 执行模型页都对照过」。另「贯穿场景」在第 134、163、186、241、264 行被反复当术语使用。｜引文依据：不适用（表述维度）。｜修复要求：改为客观陈述——「先立一个贯穿全文的具体场景」→「下文用一个具体场景贯穿全文」（首次出现处定名）；删去「这就是全部故事的起点」这类修辞句；「把贯穿场景代进去：」→「代入以上数字：」；「注意一个反例：」→「一个反例是：」；「这些机制在 …页都对照过」删除或改为「这些机制见 GPU 执行模型页」。修复后「场景」一词不再作为无定义术语跨段复用。｜修复：｜复验：

- [轻微·格式] 页内 `<style>` 存在死规则（index.html 第 37–46 行 `.diagram {...}`）。页面无任何元素使用 `class="diagram"`，仓库自带检查器已将其判为死规则。｜引文依据：不适用（机械项）——`python3 .dojo/scripts/check_unused_css.py wiki/expertplex/index.html` 输出 `wiki/expertplex/index.html: 1 rules  - .diagram`、`found 1 dead rules in 1 files`。｜修复要求：删除该页 `<style>` 中未使用的 `.diagram` 规则，保留其余仍在用的 `.chapter-questions` 规则。｜修复：｜复验：

## 未发现问题（本轮已逐条核对通过）

核心论断与全部实验数字回原文核对一致，未发现「定位不到」「来源不支持」或「把实验条件写成无条件结论」的条目；四处公式（Eq.1–Eq.4）与页面 F1–F4 逐字一致且可复算；符号全文统一（$x_{\mathrm{moe}}$、$x_{\mathrm{moe}}^\star$、$\lceil\cdot\rceil_c$、$q'$、$Q_{\max}$、$B_p/B_d/T_p/T_d/\bar O$）；页面无声称可运行的代码，该项不适用；页级「核心问题」5 题与 7 个章节的「本章问题」均配有解答折叠块，答案独立可读且指向完整论证所在章节；正文与解读的判断在「来源与范围说明」中已分开标注（第 7 章声明为解读）；简化条件与限制已列出。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（问题 1 为唯一重要项，须修复后进入第 3 轮；问题 2、3 为轻微项，一并处理）
