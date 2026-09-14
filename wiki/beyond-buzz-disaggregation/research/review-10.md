<!-- review-meta
round: 10
page: wiki/beyond-buzz-disaggregation/index.html
reviewed_content_sha256: da786762f38261ad
-->
# Beyond the Buzz 审查记录（第 10 轮）

- 页面版本：0cc99ce30d065b93961e1c03cdb254b882c7c95e（index.html 工作树哈希；overview.html = 77881d671a2d07522ed3474fe44041f083ad77e2）
- 论文版本：arXiv:2506.05508v1（2025-06-05 提交；TeX 源码文件时间戳 2025-06-06；preamble 载入 `\usepackage[preprint]{neurips_2025}`；PDF 页脚 "Preprint. Under review."）
- 审查时间：2026-09-14 16:52
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题（含 5 个解答折叠块）→ 术语速查 → 1. 方法：模拟器与设计空间（含本章问题 2 问答）→ 2. 什么条件下分离收益最大（2.1/2.2/2.3，含本章问题 3 问答）→ 3. 配套机制：切分策略与动态 rate matching（3.1/3.2/3.3，含本章问题 2 问答）→ 4. 分离的代价：KV cache 传输带宽（4.1/4.2/4.3，含本章问题 3 问答）→ 5. 方法评价：可操作结论与边界（5.1/5.2，含本章问题 2 问答）→ 来源与范围说明（论断与来源 C / 核心公式与来源 F / 外部数字与实验条件 N / 原图 / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）。

## 核对方式

- 论文原文：`curl -sL https://arxiv.org/pdf/2506.05508v1` → `pdftotext -layout`；TeX 源码：`curl -sL https://arxiv.org/e-print/2506.05508v1` → tar 解包（得 design_principles.tex、disaggregation_in_practice.tex、system_considerations.tex、related_work.tex、appendixA/B/D.tex、main.tex、preamble.tex、neurips_2025.sty）。
- 图内数值：将 assets/*.webp 转 PNG 后做像素测量（定位坐标轴/网格线做标定，再取曲线端点与逐列极值）。

## 核对结果（无问题项，逐条留证）

- 图 9（assets/img-03.webp）像素测量：蓝 DeepSeek-R1 3.64→0.056、橙 LLaMa-70B 0.94→0.45、绿 LLaMa-405B 2.07→0.20、红 LLaMa-8B 平于 0.33–0.43。与正文「约 3.6→约 0.05 / 约 0.95→约 0.45 / 约 2.1→约 0.2 / 几乎水平≈0.4」及 alt 一致。
- 图 10（assets/img-02.webp）像素测量：蓝 Optimal Rate Matching 在 x=0.15–0.99 各采样点均为最高；绿（ratio 0.5）平于 0.42；橙（ratio 3.5）x=0.2 时 0.96（贴近蓝 0.975）、x=0.99 降至 0.05——支持「0.5 卡在约 0.4」「3.5 宽松端接近 Optimal、紧延迟端退化」「蓝线全区间最高」。
- 图 5（assets/img-04.webp）像素测量：右轴校准得 2^5 在 y=285、每 log2 单位≈121.5px；红 FTL 在 PP=1 为 2^6.48≈89 s、PP=32 为 2^1.55≈2.9 s，蓝归一化吞吐 0.99–1.03。与「约 90 秒（log2 2^6.5）→约 3 秒（log2 2^1.5）」「吞吐几乎保持 1.0」一致。x 轴刻度 1,2,4,8,16,32，题注 "DeepSeek-R1 Prefill ISL 256K" 且正文 EP×PP=64 与论文 Fig.5 caption 一致。
- 图 12（assets/img-01.webp）：蓝 (ISL 16384 OSL 2048) 0.4→1.23、红 (ISL 1048576 OSL 2048) 峰值≈1.82、谷值≈0.98。与正文/alt「整体 0.4–1.8、蓝 0.4–1.2、红 1.0–1.8 GBps/GPU」及论文 Fig.12 原图题注 "Maximum of egress and ingress bandwidth across various TTLs" 一致；(ISL 1M, OSL 2k) 即 1048576/2048 换算无误。
- 图 1/2/6/7/8/14 的 ISL/OSL、模型、曲线含义逐图比对：Fig.1 (16384/2048 与 1024/32768)、Fig.2（IFB / IFB+Chunked Piggybacking / IFB+Disaggregation，与论文 Fig.2 caption "(left) co-located and (right) disaggregated ... prefill (dark boxes) / decode iterations (light boxes)" 对应）、Fig.6（DeepSeek-R1 与 LLaMa-3.1-70B，16K/2K）、Fig.7（8B/70B/405B，ISL 4096 OSL 256）、Fig.8（16384/2048、16384/16384、2048/2048、2048/16384）、Fig.14（"LLaMa-3.1-70B Dynamic vs Static Traffic"，蓝 Dynamic、红 ISL 4096 OSL 512，两线皆为实线）均与页面正文及 alt 一致。
- 带宽公式 Eq.(1)/(2) 与 system_considerations.tex 第 11/17 行逐符号一致；KV 复制边界（TP 域超过 KV 头数时按只计真正切分 KV 的 GPU）与原文 "two GPUs ... the duplication factor ... is equal to the ratio of tensor parallel ranks to KV heads" 段一致。
- 构造示例复算：61×32×16384×128×1×1/(2×8)=255,852,544 B/s≈0.256 GB/s/卡，与页面 0.256 一致；MLA KV 维度（潜向量 512 + rope 64 = 576）、DeepSeek-R1 61 层、d_head=128 与公开架构一致，且页面已标注「构造示例」「仅为量级示意」。
- rate matching 伪代码逐步对照 appendixB.tex：Algorithm 1 吞吐 = B/(FTL×G)、`if FTL < FTL_cutoff`；Algorithm 2 第 46 行 `decode_request_throughput ← decode_throughput/(OSL−1)`、α=round(best_prefill_throughput/decode_request_throughput, tolerance=0.03)、num_prefill_gpus=numerator(α)×G_dec、num_decode_gpus=denominator(α)×G_prefill。页面标注的「App.B, line 46」在源文件中确为第 46 行，逐字命中。
- 源文件名与章节映射全部命中：design_principles.tex=§3 Design space exploration、disaggregation_in_practice.tex=§4（含 §4.1 Model sensitivity、§4.2 Traffic sensitivity、§4.3 Dynamic rate matching、§4.4 NVLink sensitivity）、system_considerations.tex=§5/§5.1 Bandwidth、related_work.tex=§6、appendixD.tex=附录 C（appendix.tex 按 A/B/D 顺序输入，D 排第三即 C）——页面「附录 C（源文件名 appendixD.tex）」正确。
- 引文抽查逐条命中：abstract「hundreds of thousands of design points」「dynamic rate matching and elastic scaling」；§1「>10B parameters」「ISL >> OSL」；§3.2「All design points with an FTL > 10 seconds, a relaxed yet practical constraint, are excluded」；§4「FTL, ranging from hundreds of milliseconds to several minutes」「TTL, typically spanning a few milliseconds」；§4.1 MLA「redundant computation of down and up projections ... mitigated by temporarily caching the up-projected KV values」；§4.2「mappings, if prioritized to balance decoding speed, can significantly compromise prefill processing throughput」「piggybacking is most promising on decode-heavy traffic」；§4.3「A similar effect is expected in small-scale GPU deployments」；§5.1「existing provisioned datacenter bandwidth is sufficient to support KV cache transfer without becoming a bottleneck」；§6「fall short of providing concrete guidance on when and how disaggregation is beneficial」「largely focused on small-scale testbeds and peak throughput scenarios, without examining the full throughput–interactivity Pareto frontier」；§7 future work 四项；§8「We also highlight scenarios where disaggregation offers limited benefit—such as serving small-scale models or generation-heavy traffic」。
- 版本元信息核实：arXiv 提交 2025-06-05 ✓；源码文件时间戳为 2025-06-06 ✓；neurips_2025.sty + [preprint] ✓；arXiv 页面无会议录用信息 ✓；「代码未公开」与论文 "proprietary ... simulator" 一致 ✓。
- 机械检查：`python3 .dojo/scripts/validate.py wiki/beyond-buzz-disaggregation/index.html` → `validation ok`（exit 0），含数学字符、结构图、元数据词表、本地引用检查。
- 结构检查：全部 5 个章节均有 h3「本章问题」；页面级「核心问题」5 题、章节级共 12 题，全部有 `<details>` 且 summary 一律以「解答：」开头（grep 反查无例外）；页面级答案均以「完整论证见第 X 章」收束；h2 编号 1–5 连续、h3 章内连续；引用概念页 moe-serving / model-parallelism / chunked-prefill / mla / mqa-gqa / gpu-communication 均存在；「Chunked Prefill 页第 5 章」与「MoE Serving 页 PD 分设机制」两个跨页指向均准确（前者 h2「5. 与流水线并行合流：CPP」）。overview.html 与 index.html 双向链接有效，数值（3.6→0.05、0.4–1.8 GBps/GPU、>10B、P50 近似）与正文一致。

## 问题

- [轻微·表述] 开篇段（「没人说得清」「都在推」「要不要上 PD 分离」）、§5.1 清单⑦标题与正文（「带宽规划不必焦虑」「带宽未爆的前提」）、§2.3 末段（「被重复计算的开销吃掉」）、§3.3（「被卡在约 0.4 的吞吐」）：多处以口语化措辞替代中性技术表述。｜引文依据：不适用（论文无对应表述，属页面行文）｜修复要求：逐处改为中性表述，例如「没人说得清」→「尚无明确结论/缺少可操作指导」（对应 §6 "fall short of providing concrete guidance"）、「带宽规划不必焦虑」→「带宽规划的要求可满足/不构成约束」、「未爆的前提」→「不成为瓶颈的前提」、「吃掉」→「抵消」、「被卡在」→「被限制在」，且不改变原意的数值与结论。｜修复：｜复验：
- [轻微·表述] §4.3 末段括号句（"这一组带宽数字按图读出：横轴为归一化 tokens/s/user，纵轴为 GBps/GPU 绝对值；模型参数为构造示例所用 DeepSeek-R1 公开架构，非论文直接给出的数字。"）：把两类不同来源的信息并入同一括号——0.4–1.8 是从论文原图 Fig.12 读出的数字，而「构造示例所用架构参数」属同页 §4.1 的手算示例，两者无推导关系；「模型参数为构造示例所用 DeepSeek-R1 公开架构」在此处无对应对象，语义含混。｜引文依据：Fig.12 为论文原图（题注 "Bandwidth requirements for KV cache transfer: Maximum of egress and ingress bandwidth across various TTLs"）；构造示例见本页 §4.1 与来源说明「构造示例」小节｜修复要求：拆开表述——保留「按图读出（横轴归一化 tokens/s/user、纵轴 GBps/GPU 绝对值），论文正文未直接给出该组数字」，删去或改指「构造示例所用 DeepSeek-R1 公开架构」这一分句。｜修复：｜复验：
- [轻微·技术] §4.1 符号说明（"$bytes_{element}$：每个 KV 元素的字节数（FP4 下为 0.5 等）"）：页面把 `bytes_element` 定义为「每个 KV 元素的字节数」；论文原文同一句写的是别的量（见引文依据）。页面读法与公式量纲及变量名一致，是唯一自洽的读法，但与原文措辞字面不符且未加注，读者对照原文时可能困惑。｜引文依据：system_considerations.tex 第 13 行 "…$bytes_{{element}}$ indicates the number of KV cache bytes per token, $FTL$ represents…"（原文作 "per token"）｜修复要求：在符号说明中保留页面的正确读法并加一句括注原文措辞（如「论文原文表述为 'number of KV cache bytes per token'；按公式量纲此处指每个 KV 元素的字节数」），不改动公式与构造示例。｜修复：｜复验：

## 未发现问题的高风险项（已逐条排除，供复核参考）

- 「摘要/正文/图注/overview 同一数字不一致」：3.6→0.05、0.4–1.8 GBps/GPU、>10B、FTL 90s→3s、CPP 条件（DeepSeek-R1、ISL 256K、64 GPU、EP×PP=64）四处一致。
- 「算式与结论不符」：egress 构造示例可复算且结论量级自洽；两公式符号全文单义（`bytes_element`、`NumGPU_*` 在 §4.1/§4.2 与来源说明 F 小节写法一致）。
- 「实验条件被写成无条件论断」：compute-bound/memory-bound 分类、短 ISL 上 CPP 的泛化、比例跨度差异的机制、显存碎片/调度异常未建模、小规模部署退化——均已就地标注「推断」「页面推断，论文未给出该机制」「论文未在短 ISL 上验证」。
- 「构造示例写成来源事实」：egress 手算与本页对齐的标注齐全，参数来源（DeepSeek-R1 公开架构）与「非论文实测」声明均在正文与来源说明两处给出。
- 「指向不存在文件/链接」：页面正文引用的概念页与本地资源全部存在；validate.py 本地引用检查通过；无「（待生成）」占位。
- 「无来源支持的判断写成结论」：无。全页 [C]/[F]/[N]/[G] 标注均可回源，C/F/N/G 编号在正文与来源说明之间闭合（C 正文用到 C1–C3、C5–C10、C12–C25，全部在来源说明有定义）。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（仅需处理 3 条轻微表述项；无阻断、无重要问题，核对范围内的方法、公式、实验数字与图内读数均与 arXiv:2506.05508v1 一致，可通过轻量修正后发布）
