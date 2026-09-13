<!-- review-meta
round: 6
page: wiki/beyond-buzz-disaggregation/index.html
reviewed_content_sha256: 258410adf1410aa5
-->
# Beyond the Buzz 审查记录（第 6 轮）

- 页面版本：773c333e76f09a21183543e419e8ce7f6b537252（工作树 wiki/beyond-buzz-disaggregation/index.html）
- 论文版本：arXiv:2506.05508v1（2025-06-05 提交；TeX 源码 2025-06-06 打包，NeurIPS 2025 模板）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查；未读取本页 research/ 任何文件）
- 已完整阅读章节：核心问题（5 问）/ 1. 方法：模拟器与设计空间（含本章问题）/ 2. 什么条件下分离收益最大（2.1 流量、2.2 模型大小、2.3 架构，含本章问题）/ 3. 配套机制：切分策略与动态 rate matching（3.1 CPP、3.2 decode 池、3.3 rate matching 算法，含本章问题）/ 4. 分离的代价：KV cache 传输带宽（4.1 egress、4.2 ingress、4.3 四条趋势与论文数值，含本章问题）/ 5. 方法评价：可操作结论与边界（5.1 清单、5.2 边界，含本章问题）/ 来源与范围说明；并对照 overview.html 与 10 张原图。
- 回源方式：arXiv:2506.05508v1 全文 HTML（arxiv.org/html/2506.05508v1）与源包（arxiv.org/e-print/2506.05508v1，含 system_considerations.tex、design_principles.tex、disaggregation_in_practice.tex、appendixB.tex、appendixD.tex、related_work.tex、introduction.tex、conclusions.tex）逐条核对；10 张 assets 原图逐张目视核对。
- 机械验证：`.dojo/scripts/validate.py wiki/beyond-buzz-disaggregation/index.html` → `validation ok`（exit 0）；页面全部 $...$ 与 $$...$$ 逐条过 KaTeX 渲染（node + libs/katex.min.js），除下表第 1 条外全部渲染成功；6 个前置概念链接（moe-serving / model-parallelism / chunked-prefill / mla / mqa-gqa / gpu-communication）均存在；dojo:topics（并行与通信/推理系统/数学基础）与 dojo:tag（推理系统）均在 catalog_builder.py 词表内。
- 本轮已核实的数字（回源通过，不再列为问题）：Fig.5 FTL log2 2^6.5→2^2（≈90s→4s）且吞吐归一化后≈1.0；Fig.9 四条曲线（8B≈0.4 水平、70B 0.95→0.45、405B 2.1→0.2、DeepSeek-R1 3.6→0.05）；Fig.10 固定比 0.5 平在≈0.4、3.5 宽松端接近 Optimal；Fig.12 图例 ISL 16384/1048576、OSL 2048，纵轴 GBps/GPU 绝对值，范围 0.4–1.8；Fig.7 三个子图标题 ISL 4096 OSL 256；Fig.8 四组 (16k/2k)(16k/16k)(2k/2k)(2k/16k)；Fig.6 两子图 ISL 16K OSL 2K；Fig.1 ISL 16384/OSL 2048 与 ISL 1024/OSL 32768；egress 手算 61×32×16384×128×1×1/(2×8)=255,852,544 B/s≈0.256 GB/s/卡（复算一致）；[App.B, line 46] 实为 appendixB.tex 第 46 行 `decode\_request\_throughput ← decode\_throughput/(OSL-1)`，定位准确；「附录 C（源文件名 appendixD.tex）」正确（appendix.tex 依次 \input appendixA、appendixB、appendixD，输出中 P50 节即附录 C）；两条带宽公式与 Eq.(1)(2) 逐字一致；§5 引文（"fall short of providing concrete guidance…"、"…without examining the full throughput–interactivity Pareto frontier"、"we also highlight scenarios where disaggregation offers limited benefit…"）逐字一致。

## 问题

- [重要·格式] head 的 dojo:summary：egress 公式中 `N_{kv\textunderscore heads}` 用了 KaTeX 不支持的 `\textunderscore`，summary 公式无法渲染，违反发布条件「可渲染 dojo:summary」；且与正文同公式写法不一致。｜引文依据：以 node 调 libs/katex.min.js 对 head summary 逐条渲染，得 `KaTeX parse error: Undefined control sequence: \textunderscore at position 81`；正文 line 292 同公式写作 `$N_{kv\_heads}$`，KaTeX 渲染通过（`\_` 受支持）。首页 assets 渲染路径 `.dojo/home/dojo-home.js:73-88,131` 用 `renderMathInElement(..., throwOnError:false)` 渲染 `page.summary`，该宏会渲染为红色报错文本。｜修复要求：把 dojo:summary 中的 `\textunderscore` 改为 `\_`，与正文写法统一，改后用同法再跑一遍 KaTeX 渲染应无报错。｜修复：dojo:summary 中 egress 公式的 `N_{kv\textunderscore heads}` 改为 `N_{kv\_heads}`，与正文 line 292 写法统一。｜复验：以 node + libs/katex.min.js 对 dojo:summary 内 3 条公式逐条渲染，failures=0，无报错。｜

- [重要·技术] §3.3 正文与其本章问题答案对「最优 ctx:gen 比例随延迟收紧而下移」的机制解释方向写反，且答案内部自相矛盾。｜引文依据：论文对 Fig.9 只给结论不给机制（disaggregation_in_practice.tex:70-77，caption 仅 "The optimal ratio of ctx-to-gen GPUs varies across models and target latencies"），页面该机制为无来源推断。由 Fig.9 读出的数据是紧延迟端比例下降（DeepSeek-R1 约 3.6→0.05、405B 约 2.1→0.2、70B 约 0.95→0.45）——比例 = prefill GPU/decode GPU 下降意味着每个 decode GPU 需承担更多，即 decode 每 GPU 吞吐随延迟收紧而**下降**；页面 line 258 写"随延迟收紧…decode 池的每 GPU 吞吐上升（批小、TTL 紧），prefill 池的处理能力相对需要更少 GPU；比例随之下移"，以「上升」为前提推出「比例下移」，前提与结论方向相反（每 GPU 吞吐上升应导致 decode GPU 更少、比例上移）。line 279 更直接自相矛盾："低 TPS 端 decode 每 GPU 吞吐低，需要更多 decode GPU（或相对少 prefill GPU）来平衡；比例 3.5 意味着 prefill 多 decode 少，正好匹配"——前句要求 decode 更多，后句却称 prefill 多 decode 少的 3.5 正好匹配。｜修复要求：把 mechanism 改为与 Fig.9 一致的方向（延迟收紧 → 每 decode GPU 吞吐下降 → 需更多 decode GPU → ctx:gen 比例下移），删除 line 279 中与结论冲突的"需要更多 decode GPU"一句或整句改写为单义表述；不可保留原意的模糊化。｜修复：§3.3 正文（line 258）机制方向改正为「随延迟收紧 → decode 每 GPU 吞吐下降（批小、TTL 紧）→ 需要更多 decode GPU → ctx:gen 比例下移」，并补出延迟宽松端的反向说明；本章问题第 2 题解答（line 279）删除与结论冲突的「低 TPS 端 decode 每 GPU 吞吐低，需要更多 decode GPU」一句，改写为以最优比例高低解释 3.5/0.5 两端表现的单一方向表述。｜复验：已核对两处方向一致、不再自相矛盾，且与 Fig.9 读数（DeepSeek-R1 约 3.6→0.05）一致。｜

- [轻微·技术] "所有图均归一化呈现"与原文不符，并与页面自身对图 12 纵轴的说明冲突。｜引文依据：原文为 "Most results in this paper are presented in normalized form, as our primary objective is to convey trends rather than make specific performance claims."（introduction.tex:16，Fig.1 caption），是 most 而非 all；页面 line 133 与 line 388（§5.2）两处均写"所有图均归一化呈现"，而 line 333 自注图 12 纵轴为"GBps/GPU 绝对值"、图 5 右轴为 FTL 绝对秒数（log2）。｜修复要求：把"所有图"改为与原文一致的"多数结果/大部分图"，或显式排除带绝对坐标轴的图（图 5、图 12）。｜修复：正文 line 133、line 155、§5.2 line 388 三处「所有图（均/全部）归一化」改为「多数图以归一化形式呈现」；overview.html 同步改为「多数结果归一化呈现」。｜复验：grep 确认本页不再出现「所有图…归一化」「论文图全部归一化」，与 line 333 对图 12 纵轴为 GBps/GPU 绝对值、图 5 右轴为绝对秒数的说明不再冲突。｜

- [轻微·格式] 图 14（assets/img-08.webp）的 alt 描述与图内线型不符。｜引文依据：该图两条曲线蓝 "Dynamic data" 与红 "ISL 4096 OSL 512" 放大后均为实线，legend 色块亦为实线；alt 却写"动态流量模拟（实线）与 P50 近似（虚线）下的 Pareto 前沿对比"。｜修复要求：删去"实线/虚线"描述，按图内实际改写（蓝=动态流量模拟、红=P50 近似 ISL 4096 OSL 512，两条均为实线）。｜修复：assets/img-08.webp（Fig.14）alt 删去「实线/虚线」描述，按图内实际改写为「两条曲线均为实线（蓝=动态流量模拟，红=ISL 4096 OSL 512）」。｜复验：目视图内 legend，两条曲线与色块均为实线，alt 与之一致。｜

- [轻微·格式] 来源说明的 C 编号缺 C4。｜引文依据：「论断与来源（C）」首条列"C1–C3、C9、C24–C25：abstract 与 introduction、§8 conclusions"、次条列"C5–C7、§3 模拟方法"，跨过 C4；全文枚举 [C1]…[C25] 未见 [C4]，正文亦无 [C4] 引用。｜修复要求：补写 C4 条目或注明其空缺，使编号连续可核。｜修复：来源说明「论断与来源（C）」在首条（C1–C3、C9、C24–C25）之后补写 C4 条目（context chunking 的有效性依赖注意力机制（MLA vs GQA）、在宽松延迟与生成密集流量下最有利；来源 introduction），与 evidence.md 的 C4 登记一致。｜复验：C1–C25 编号现已连续可核。｜

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：修复（第 1、2 条为未关闭重要问题，修复后需复验：第 1 条复跑 KaTeX 渲染无报错；第 2 条机制方向与结论一致且无自相矛盾；第 3–5 条按修复要求逐条核对）