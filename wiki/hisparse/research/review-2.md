# HiSparse 审查记录（第 2 轮）

- 页面版本：index.html 0123d1fb746a7ca590f21180e6724428eac7d4fa；overview.html 7dee9775f87e59f726dba7624d0561bb52cf53df
- 论文版本：arXiv:2608.07009v1（NeurIPS 2026 投稿预印本，TeX 源码 /tmp/hisparse-research/src/）
- 审查时间：2026-09-03 13:48
- 审查者：独立子代理（第二轮，未参与写作与前序轮次）
- 已完整阅读章节：index.html 顺序通读——术语表、核心问题（5 问 5 答）、第 1 章（1.1–1.4）、第 2 章（2.1–2.4）、第 3 章（3.1–3.4 含折叠演算与可运行代码块）、第 4 章（4.1–4.3 含伪代码折叠块）、第 5 章（5.1–5.5）、第 6 章（6.1–6.5）、第 7 章（7.1–7.3）、来源与范围说明；overview.html 全文。论文原文全部 9 个 TeX 文件逐字通读。
- 机械验证：`.dojo/scripts/validate.py` 对两页均返回 validation ok；页面第 3 章可运行代码实际执行，输出与页面预期一致（65.5% / 34.1% / 3.9%）；概念链接 ../kv-cache/、../dsa/、../strata/、../vllm-cudagraph/ 的 index.html 均存在；assets/ 6 张 PNG 均有效；dojo:topics（推理系统,内存与缓存,注意力机制）在 AGENTS.md 固定大类内。

## 逐项审查结果

### 2.1 可读性（按顺序完整阅读，含全部折叠块）

- 术语首次出现均解释：术语表覆盖 KV cache、HBM、top-$k$/indexer、$\mathcal{S}_t^{(\ell)}$、TTFT/TPOT、PD 两种模式、pinned host memory、GPU cache（含"hot device buffer"出处）、LRU/Bélady、anchor/shared 层。通过。
- 前置知识先于依赖给出：KV cache 与 DSA 在第 1 章开头给出概念页链接；CUDA graph 捕获在 4.1 引用 vllm-cudagraph 概念页；Strata 在 4.2 阶段 4 引用概念页。通过。
- 公式说明用途、符号与范围：footprint 公式（2.1）符号与原文 Table 2 一一对应并有代入检查（13.09 GB→0.4 GB 约 30 倍）；F3 反设估算给出完整算式。通过。
- 推导无跳步：贯穿示例 5 步 LRU 演算、Resolve 五阶段手算、$B{=}2$ 对照均可逐步复算（本审查复算全部一致，miss 5/10 与 7/10 正确）。通过。
- 示例说明输入、变化项与结果：8 位置示例处处标注"构造示例"，来源与范围说明单列构造条件与 hit 排位约定。通过。
- 折叠收起后正文仍完整：核心结论（30%→13.4%、五阶段语义、7.7→3.0 ms 等）均在正文；折叠块为完整演算与代码。通过。
- 类比无误导致：TLB 类比标注"只对应接口形态，不延伸到 TLB 的其他语义"；"账单/驻留费"措辞在来源说明中声明为行文类比。通过。
- 章节衔接明确：每章开头承接上一章遗留问题（第 3 章"第 2 章留下的悬念"、第 5 章"回忆第 4 章末尾的等待点"）。通过。
- 核心问题均由正文完整回答，5 个核心问题、12 个本章问题全部有"解答："折叠块，答案独立可读，核心问题答案末尾均指明"完整论证见第 X 章"。通过。

### 2.2 原文与技术（14 项）

1. **论断与数字**：全部核心数字逐一对照原文——13.09 GB/0.4 GB/1M>100 GB/141 GB（§1）、4 GB/60 并发/240 GB/777 tokens/s/约 15 请求（§2.2）、200 MB/7 GB/s/64 GB/s（§2.3）、2430→2668、511→1824、111→520、2288→2280、624→1919、232→680、600→1257、1511→4308、26/829/171 s、15.9/16.0 ms（§4.1–4.2）、100,384 token/1,799 步/前 1,000 步/78 层/30%/13.4%/17.2%/16.1%/8.2%/6.7%（§4.3）、112→29 μs/±5%/1–4 μs/60 μs（§4.4–4.5）、24.1/24.8/7.7/22.0/3.0/11.2/618/1515/1727/2034/85%/74%/13–15%/14–17%/2.8×/21+57 层/27%/三分之二到五分之四（§4.6）、480 GB LPDRAM/2 TB/1 TB（§5）——全部一致。发现 2 处轻微偏差（"一个 CUDA kernel"、§4.3 定位误挂），见问题 2、3。
2. **实验覆盖**：三组端到端（DeepSeek-V4-Flash、GLM-5.1、Qwen3+Quest）+ GLM-5.2 prefetch 实验全部纳入；未采用的 Figure 4 与 Figure 7 的关键数字（N3、N9）以表格和正文完整呈现，无选择性省略；推测式预取负结果如实报告。通过（1 处七配置 vs 六行表格的口径小出入，见问题 4）。
3. **公式与推导**：footprint、admission 预留、naive offload 估算与原文一致；贯穿示例与代码复算全部通过；符号（$B$、$k$、$L_{\text{ctx}}$、$N_{\text{batch}}$、$N_{\ell}$、$W_{\text{KV}}$、$s$）全页一致。1 处约数未标"约"（问题 5）。
4. **代码**：页面唯一可运行代码实际执行，输出逐字符一致（1048/1600=65.5%、546/1600=34.1%、63/1600=3.9%）；简化条件在代码块后声明（构造 trace、非论文数据）。通过。
5. **事实与推断**：第 7 章开头声明"分析性判断"；"与 KV 压缩可叠加"标注推断；6.3 混布 TTFT 因果链标注"含推断成分"。1 处推断误挂原文定位（问题 3）。
6. **不确定信息**：PD 分离 2.9× 明确标注 decode-only proxy 而非物理部署实测（§4.2 原文 "We do not run a physically disaggregated deployment"）；$B$ 静态配置与动态调整留作未来工作（§4.5）均已说明。通过。
7. **原图**：6 张图与原文 Figure 编号对应正确（motivation=Fig 1、hisparse_overview2=Fig 2、swap_kernel=Fig 3、peak_throughput_comparison_paper=Fig 5、topk_miss_rate_trace=Fig 6、prefetch_sweep=Fig 8）；图注内容与原文 caption 一致；图前有引导句、图后有解释段。通过。
8. **页面链接**：4 个概念页链接目标均存在；overview.html 与 index.html 互链；外部链接仅 arXiv 页与 SGLang 仓库（页面明确引用的来源）。通过。
9. **简化条件**：贯穿示例与可运行代码的简化条件在来源与范围说明集中声明，且不改变核心结论。通过。
10. **页面功能**：validate.py 通过；KaTeX/Prism 本地库存在；目录锚点由显式 id 支撑；图片 lightbox、代码复制、折叠交互结构完整。通过。
11. **公式书写**：发现 Unicode 数学字符直接出现且倍率写法不一致（×/→/±/↔/←），见问题 1。不通过（轻微）。
12. **图示**：第 2 章 dg-stack 为 HTML 结构；第 5 章时间线为内联 SVG（viewBox、dg-box/dg-line/dg-accent 类、颜色用 CSS 变量）；SVG `<text>` 仅纯文字无 ASCII 数学近似；节点与箭头含义在图注定义；窄屏下 dg-flow 折叠为纵向、SVG 按宽度缩放。通过。
13. **问题块**：页面级"核心问题"与章节级"本章问题"命名正确；两级全部问题均有"解答："折叠块；答案独立可读且与正文一致；核心问题答案指明论证章节。通过。
14. **格式一致性**（write.md 第 4 节）：标题编号连续（h2 1–7、h3 各章内 1.x 连续，核心问题/本章问题/来源与范围说明不编号）；问题块用 `<ol class="chapter-questions">`；伪代码标 language-text 并写明输入/状态/输出；构造示例标注；来源与范围说明齐备。发现半角冒号笔误 1 处（问题 6）与问题 1 的字符规范违反。

## 问题

- [轻微·格式] index.html 正文、dojo:summary 与表格单元格中 Unicode 数学字符直接出现，且倍率写法全页不一致：×（如 1.2 节 "8×H200"、5.3 节 "4.7×"、6.2 表格 "2.1$\times$" 为 LaTeX 而同行 "2.9×" 为 Unicode、head 的 dojo:summary "3.1×、Qwen3+Quest 200K 达 4.7×"；overview.html "PCIe Gen5 ×16" 同类）、→（核心问题 5 解答 "111→520"、3.4 节 "30%→13.4%"、2.4 节 "位置 2→槽 0"；overview.html "2288→2280"）、±（6.4 节 "±5% 以内"）、↔（4.1 节 "地址翻译↔KV 记录定位"）、←（3.4 折叠块 "槽 0←2"）｜引文依据：不适用（check.md 2.2 第 11 条、write.md 4.2 "不使用 Unicode 数学字符替代……同一变量全页保持同一种写法"；代码块内字符按代码原样保留）｜修复要求：正文、summary、列表、表格中的上述字符改为 KaTeX 写法（$\times$、$\to$、$\pm 5\%$、$\leftrightarrow$、$\leftarrow$），倍率统一为 $\times$；第 4 章伪代码 code 块内的 →/∈ 保留原样｜修复：用 Python 正则把 ×（U+00D7）→ $\times$、→（U+2192）→ $\to$、←（U+2190）→ $\leftarrow$、↔（U+2194）→ $\leftrightarrow$，并把 ±NUMBER% → $\pm$NUMBER\%，全部在非 `<pre>/<code>/<script>/<style>/<svg>` 区间内做替换；代码块内容原样保留。复验：grep 五种 Unicode 字符在非 skip 区间的剩余数均为 0；validate.py ok；Chrome headless `.katex` 数从 278 → 357（增加 79 个新公式节点），无错误；SVG `<text>` 重叠 0 对。overview.html `.katex` 数从 17 → 21，0 错误。
- [轻微·技术] index.html 2.3 节与 overview.html 方法概述："约 2200 行 Python 加一个 CUDA kernel"与原文不符：原文为"约 2200 行 Python（六个模块）加一个 CUDA kernel 头文件"，该头文件内含 Resolve 的 token 级与压缩布局两种实现及 copy-only kernel，另有 HIP 变体——"一个 CUDA kernel"会被读成整个系统只有一个 kernel｜引文依据：附录 A "roughly $2{,}200$ lines of new Python across six modules plus a CUDA kernel header"、"implement \textsc{Resolve} for token-level and compressed KV layouts---with optional recording of the miss plan---plus the copy-only kernel"、"a HIP kernel variant supports AMD GPUs"｜修复要求：两页改为"约 2200 行 Python 加一个 CUDA kernel 头文件（Resolve 的两种布局变体与 copy-only kernel）"或等义准确表述｜修复：｜复验：
- [轻微·技术] index.html 3.2 节表格下方："不同负载的选择局部性不同，miss 率会随之变化（§4.3）"——§4.3 没有跨负载泛化的表述，该半句是页面的合理推断，不应挂原文定位｜引文依据：§4.3 原文仅见 "The trace comes from a GLM-5.1 request serving a $100{,}384$-token LongBenchV2 prompt … differences reflect replacement decisions alone"，全节无 workload 泛化句｜修复要求：删去"（§4.3）"定位并将该半句标注为推断（如"（推断）不同负载的选择局部性不同……"），或直接删除该半句｜修复：｜复验：
- [轻微·可读性] index.html 3.2 节正文"七条 miss 率曲线对应七种缓存配置"与随附表格（6 行）口径不一致：缺 Bélady $B{=}8192$ 这一行，读者无法从表格数出第七种配置｜引文依据：Figure 6 caption "under seven cache configurations"；§4.3 "B\'elady reaches $8.2\%$ already at $B{=}4096$ (and lower still at $B{=}8192$, also shown)"｜修复要求：表格补一行"Bélady 离线最优，$B{=}8192$（原文未给出具体数值，低于 8.2%）"，或将正文改为与表格一致并说明第七种配置为 Bélady $B{=}8192$｜修复：在 §3.2 表格 Bélady $B{=}4096$ 行后追加新行 `Bélady 离线最优，$B{=}8192$` + 数值列 `<8.2%` + 说明列 "Figure 6 中"也展示"，§4.3 未给出具体数值"——表格行数 6 → 7，与正文"七种缓存配置"对齐。复验：grep Bélady 表格行从 1 → 2；表格行总数与"七种缓存配置"叙述一致；validate.py ok。
- [轻微·技术] index.html 核心问题 1 解答："上下文 32K、$k{=}2048$ 时驻留的是读量的 16 倍"——32768/2048=15.625，且若按 32K 输入 + 8K 输出的完整上下文计为 40K/2048≈19.5；既未加"约"也未说明口径｜引文依据：§2.2 "must fit $N_{\text{batch}} \times L_{\text{ctx}}$ tokens of KV state … while each decode step's attention reads only $N_{\text{batch}} \times k$"｜修复要求：改为"驻留的是读量的约 16 倍（按 32K 上下文计）"或补充口径说明｜修复：｜复验：
- [轻微·格式] index.html 4.1 节："这也顺便解释了 indexer 无关性:DSA、NSA、Quest"——半角冒号后无空格，与全页全角冒号惯例不一致｜引文依据：不适用｜修复要求：改为全角冒号"："｜修复：半角冒号 `:` → 全角 `：`。复验：grep "无关性\uff1aDSA"（全角）命中；grep "无关性:DSA"（半角）0 命中。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：修复。全部核心论断、机制描述、公式与实验数字经逐条对照原文核对一致，实验覆盖无选择性省略，可运行代码输出与页面一致，两级问题块完整；6 条轻微问题（Unicode 数学字符与写法一致性、两处与原文的表述偏差、一处推断误挂定位、一处表格口径、一处约数口径、一处标点）逐条修复后即可进入第三轮。
