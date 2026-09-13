<!-- review-meta
round: 5
page: wiki/ppd-disaggregation/index.html
reviewed_content_sha256: 14e1565b00ea1c8e
-->
# PPD 分离审查记录（第 5 轮）

- 页面版本：index.html（工作树，指纹 14e1565b00ea1c8e；git HEAD 5d5ab40）
- 论文版本：Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving，arXiv:2603.13358v2（2026-05-05 修订，Comments: ICML 2026）
- 来源获取：arXiv abs 页（作者/版本/ICML）、arXiv HTML 全 v2 正文与附录、论文原始 figure PNG（fig1_pareto / interference_tpot / weight_tradeoff_curve / scaling_simulation / real_validation_e2e / interference_tpot_4prefills）
- 审查时间：2026-09-13 20:23
- 审查者：独立子代理（未参与写作与前序轮次；未读取本页 research/ 下任何文件）
- 已完整阅读章节：head 元信息 → 论文信息块 → 构造示例导语 → 核心问题（5 题含解答）→ 术语表 → §1 多轮对话暴露 PD 分离的两个代价（含本章问题）→ §2 full prefill 与 append-prefill 差一个数量级（2.1 复杂度对比、2.2 干扰微基准，含本章问题）→ §3 没有静态最优（3.1 配置空间、3.2 Pareto、3.3 winner 分布、3.4 x=1 的 TTFT 改善，含本章问题）→ §4 PPD（4.1 打分函数、4.2 两阶段、4.3 解耦、4.4 vLLM 原型，含本章问题）→ §5 真实负载、慢网络与权重旋钮（5.1–5.5，含本章问题）→ §6 方法评价（6.1–6.3，含本章问题）→ §7 附：PPD 架构概念图 → 来源与范围说明 → 页面脚本与资源引用

## 核对结论（数字与论断全部回源）

逐条回源核对结果：核心论文数字全部与原文一致，未发现数字不符或推断被包装成来源结论的情况。已核对：48%/2% @batch200（Fig.2）、+57%/+21% 4 并发（C.1 Fig.7）、32K 3–4×/64K <25%（C.1 Fig.8）、3060=17×18×10、Table 1 全表（-57.8/-65.2/-73.3、-47.7/-51.6/-56.2、-44.3/-38.1/-24.9，汇总 48–73%）、92.2%、Table 2 全表（63.3/0.6/0/21.3 等）、15–25%、3.1 轮/会话、~75% KV 削减、~3× 网络负载差、Table 3 全表（10/0/4、5/13/27、12/14/27）、143.7→170.6 ms(+18.7%)/PPD ~51 ms/64%→70%、3028→3169 ms(+4.7%)/+0.9%、w_tpot=1/3/6→95/50/20%、94–96% TTFT 降与 7–12% TPOT 退化、Fig.6 子图 -96%/+7.0% 与 -94%/+12.1%（读原图核对）、~68%、128 KiB/token、2K→~256 MB、5115 token→~670 MB / 4.5/27/67 ms、Table 5 失败率全表（11/44/61/89、6/22/44/67、0/6/11/22、0/0/6/11、4R 全 0）、~70% @2–16 轮 8B/14B/30B、10 档 QPS、18 负载 4/2/3 构成、17 配置含 7 混合、<1 ms、30s→10s 阈值对比、vLLM 实现细节（kv_role/ZeroMQ/Quart/MD5/60 min/10 s·30 s 心跳）、作者与单位、ICML 2026 与 v2 日期。页面自定的构造示例算式可复算且自洽：1250²=1,562,500、50×1250=62,500、比值 25 与 n/m=24；Δ 代入 0.6−0.08=0.52、0.6−6×0.08=0.12、0.6−10×0.08=−0.20、0.2−0.05=0.15、0.2−0.5=−0.30 全部正确；128 KiB = 2×8×128×2×32 = 131072 B。页面 6 张原图（img-01…img-06）与论文 Fig.3/Fig.6/Fig.5/Fig.4/Fig.1/Fig.2 逐一比对一致，图注与图内容对应。`overview.html` 与 `index.html` 互链，4 个前置概念页（standard-attention / mqa-gqa / gpu-communication / prefix-caching）均存在，无「（待生成）」占位，正文与来源说明未指向不存在的文件路径。`.dojo/scripts/validate.py` 返回 `validation ok`。页面无声称可运行的代码（§4.2 为不可执行伪代码，与 Alg.1 一致），无执行核对项。

## 问题

- [重要·技术] §2.2 图注（index.html:218）与来源说明 N2/N3 行（index.html:654–655、668–669 对应的 N2/N3）：正文两处把辅助实验的来源位置写成「见原文 Fig.A1」「见原文 Fig.A2」，N 表定位列同样写「§4.1 Fig.A1」「§4.1 Fig.A2」，但论文没有 Fig.A1 / Fig.A2 这两个编号——论文图编号连续为 Figure 1–11，这两个图是 Figure 7 与 Figure 8，位于附录 C.1；引用编号在论文中定位不到。｜引文依据：论文 C.1 「Figure 7: Prefill-decode interference with 4 concurrent prefills. Same setup as Figure 2 but with 4 concurrent prefill operations. Full prefill causes ∼57% slowdown at batch size 200; append-prefill remains within ∼21% of baseline.」；「Figure 8: Interference scaling with context length. Full prefill interference grows to 3–4× at 32K tokens; append-prefill stays below 25% even at 64K.」｜修复要求：正文两处改为「见原文 Fig.7（附录 C.1）」「见原文 Fig.8（附录 C.1）」，N2/N3 定位列同步改为「附录 C.1 Fig.7」「附录 C.1 Fig.8」；不得保留 A1/A2 字样。｜修复：｜复验：

- [重要·技术] §1 第 2 段导语（index.html:72）与 §1 第 1 段（index.html:135）：4 处 [C6] 被用于它不支持的三条论断——「PD 分离的设计动机是消解两类工作负载的相互干扰」「允许 P 与 D 独立扩缩容与硬件异构」「已被 vLLM、SGLang、TensorRT-LLM、DeepSeek、Gemini 等几乎所有主流引擎采用」；而 C6 在来源说明中定义为「Turn 1 恒 x=0（无缓存 KV）｜§3、Alg.1」（index.html:621），§3/Alg.1 只讲 Turn 1 返回 x=0，三条论断均无法在该位置找到支持。｜引文依据：论文 §2.2「Prefill-decode (PD) disaggregation mitigates interference by physically separating prefill and decode onto distinct GPU pools … This architecture eliminates interference, enables independent scaling of P and D resources, and permits hardware heterogeneity (Patel et al., 2024).」「It is supported by all major serving frameworks (vLLM (Kwon et al., 2023), SGLang (Zheng et al., 2024), TensorRT-LLM, LMDeploy (Contributors, 2023), and NVIDIA Dynamo (NVIDIA, 2025)) and is deployed at production scale by providers such as DeepSeek (DeepSeek-AI et al., 2025) and Gemini (Team et al., 2025).」｜修复要求：这三处（含 index.html:72 的 [C1, C6]）删去 [C6]，改引 §2.2 的编号（C1 已覆盖 §2.2）；或在 C 表补一条新编号（如「PD 已成为主流范式、被主流服务框架采用｜§1、§2.2」）后再引用。｜修复：｜复验：

- [轻微·技术] §5.1（index.html:446）：数据集来源被标为 [C13]，而 C13 定义为「原型基于 vLLM disaggregated serving：kv_role producer/consumer、ZeroMQ、标准协议无需定制修改；Quart 代理、MD5 会话哈希、60 min TTL、10s/30s 心跳｜§5 末段、App.B.2–B.5」，不支持 ShareGPT/WildChat 数据集出处。｜引文依据：论文 §6.1「We use two publicly available multi-turn conversation datasets: ShareGPT (Chiang et al., 2023) with user-shared ChatGPT conversations … and WildChat (Zhao et al., 2024) with in-the-wild user conversations with varied prefill-to-decode ratios.」｜修复要求：把该处 [C13] 改为 §6.1 对应的来源编号。｜修复：｜复验：

- [轻微·技术] §4.1（index.html:337）、核心问题 Q4 答案（index.html:102）、§4 本章问题（index.html:421）：[C18] 被挂在打分函数与「算子指定的 SLO 权重」上，而 C18 定义（index.html:631）为「PPD 是 routing actuator，不强制端到端 SLO 界｜§6.5 末句」，与该表述无关（[C18] 在 index.html:570 的用法才是正确的）。｜引文依据：论文 §6.5 末句「PPD is a routing actuator: it does not enforce end-to-end SLO bounds (which require closed-loop admission control and batch scheduling) …」；权重定义见 §3「Given π and operator-specified weights w=(w_ttft,w_tpot)」。｜修复要求：三处去掉 [C18]，保留 [F1]（或补 §3 的对应编号）。｜修复：｜复验：

- [轻微·技术] 术语表 full prefill 行（index.html:121）与 append-prefill 行（index.html:122）、§2 核心问题答案（index.html:88）：符号 $n$ 全页不单义——术语表 full prefill 行写 $O(n^2)$（此 $n$ 是整段 prompt 长度），紧邻的 append-prefill 行写 $O(m(n+m))$（此 $n$ 是已缓存前缀长度），而 §2.1（index.html:198）明确「设 $n$ 为已缓存的前缀 token 数」；§2 核心问题答案更在同一句内先写「full prefill 处理 $n$ 个新 token，attention 复杂度 $O(n^2)$」、紧接着写 append-prefill「每个关注 $n+m$ 个 key」。｜引文依据：论文 §4.1「For Turn 2+ with m new tokens appended to the context of n cached tokens, append-prefill computes attention only for the m new tokens (each attending over n+m keys), yielding O(m(n+m)) complexity.」；§2.1「Prefill … is compute-bound with O(n²) attention complexity for n tokens」。｜修复要求：全页统一 $n$=已缓存前缀、$m$=本轮新增，full prefill 一律写 $O((n+m)^2)$（术语表 full prefill 行同步改），核心问题答案的「full prefill 处理 $n$ 个新 token」改为「处理全部 $n+m$ 个 token」。｜修复：｜复验：

- [轻微·技术] §4.1（index.html:353）：「两者在下列代入中均取正值表示本地更优」与紧随其后的「$\Delta_{\text{tpot}}=+0.08$（本地 TPOT 比走 P 慢 8%）」自相矛盾——对 $\Delta_{\text{tpot}}$ 取正值表示本地更差，不是本地更优。｜引文依据：论文 Eq.1「where Δ_ttft is the relative TTFT improvement and Δ_tpot the relative TPOT degradation when ψ is processed locally rather than through P」。｜修复要求：改为「$\Delta_{\text{ttft}}$ 取正值表示本地更优，$\Delta_{\text{tpot}}$ 取正值表示本地退化」。｜修复：｜复验：

- [轻微·表述] §1（index.html:165）：「下文会看到，『让 D 本地处理 append-prefill』与『让 P 重新 prefill』是同一个路由参数 $x$ 的两种取值」——「下文会看到」为元话语（前瞻性叙事）。｜引文依据：不适用。｜修复要求：删去「下文会看到，」，直接从「『让 D 本地处理 append-prefill』与……」起句。｜修复：｜复验：

- [轻微·表述] §3.4（index.html:304）：「两个值得注意的模式<sup>[C14]</sup>：……」——「值得注意」为元话语/临场评价。｜引文依据：不适用。｜修复要求：改为直接陈述，如「两条规律：一是……二是……」。｜修复：｜复验：

- [轻微·可读性] §3.3 第 1 条（index.html:282）：「零网络传输、本地前缀缓存，可 Turn 2 延迟最低」句子不成句（「可」后缺谓语，读不出「因此使 Turn 2 延迟最低」之意）。｜引文依据：不适用。｜修复要求：改为「零网络传输、本地前缀缓存使 Turn 2 延迟最低」。｜修复：｜复验：

- [轻微·技术] §3 标题（index.html:247）与 §1 导语（index.html:72）：把 3060 个数据点写成「3060 个配置」（「没有静态最优：3060 个配置的扫描证据」「3060 个配置的系统扫描」），与本页 §3.1（index.html:255）「实验在 17 个配置上展开……共 17×18×10 = 3060 个数据点」及 §3.3「3060 个数据点」不一致，易被读成共有 3060 个配置。｜引文依据：论文 §4.2「Using these three machine types with 4 GPUs, we explore 17 configurations, including 7 hybrid configurations」；「Each configuration is evaluated at 10 QPS … yielding 17 × 18 × 10 = 3,060 data points.」｜修复要求：§3 标题与 §1 导语改为「3060 个数据点（17 配置 × 18 负载 × 10 QPS）」或「3060 个数据点的扫描」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 8
- 处置：修复（关闭上述 2 个重要问题与 8 个轻微问题后复验；本轮数字与实验结论均与 arXiv:2603.13358v2 一致，无阻断项）
