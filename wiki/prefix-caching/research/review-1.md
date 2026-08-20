# 前缀缓存 审查记录（第 1 轮）

- 页面版本：index.html 59612 字节，SHA1 b91cd0764ef79c7038f11d48abcd293c63e02bf8
- 外部来源版本：SGLang NeurIPS'24（arXiv:2312.07104）、Strata OSDI'26（arXiv:2508.18572v2）
- 审查时间：2026-08-19
- 审查者：独立子代理（reviewer-prefix-caching-1）
- 已完整阅读章节：开篇引文与导语 → 核心问题（5 题含解答折叠块）→ 1. 什么能复用——共享前缀从哪来（含本章问题）→ 2. radix tree——组织与匹配（含 SVG 图、三请求折叠块、本章问题）→ 3. 缓存满了怎么办——LRU 逐出与引用计数（含本章问题）→ 4. 命中率——页大小是隐藏的自变量（含 F1 公式、100 token 算例表、本章问题）→ 5. 两种实现——radix tree 与哈希（含对比表、本章问题）→ 来源与范围说明 → overview.html 全文

核对说明：C1–C5、C9、N1 逐条对照 SGLang 论文 §1、§3.2 原文；C6（§2.3）、C7（§6）、C8（§3.1）、N2（§5.3.2）、N4（§5.2.1）逐条对照 Strata TeX 源码（已按 0_main.tex include 顺序确认章节编号：intro=§1、background=§2、motivation=§3、design=§4、eval=§5、related=§6）。机械项：concept 链接（../kv-cache、../paged-attention、../strata）、overview 互链、../../index.html、libs 资源均存在；SVG 4 个 dg-label foreignObject（S/m₁/m₂/S′）坐标与节点框及边线静态核对不相交（当前 SVG 中无"队首取出/prefill 完成"标签，dg-label 恰为 4 个）；标题编号 1–5 内部连续、顺序与核心问题 5 题一一对应。outline.md 属规划文件，按规范本轮审查者不读取，编号仅核对了页内一致性，未对照 outline.md。

## 问题

- [重要·技术] index.html 核心问题 4 答案、第 4 章正文、第 4 章问题 1 答案、来源说明 N2、overview.html「关键结论与边界」：「页 512 比页 32 命中率低 2.4%」对比对象与来源不符——论文中 2.4% 是页 512 配置的 SGLang-HiCache 相对对照系统 Strata-IO 的命中率差距，不是页 512 相对页 32 配置的差距｜引文依据：5_eval.tex §5.3.2 "Even at its best-performing setting (page size 512), SGLang-HiCache achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate."｜修复要求：五处统一改为"页 512 配置（SGLang-HiCache 的最优页大小）的命中率比对照系统 Strata-IO 低 2.4%、吞吐为对照的 93%"，不得写成"比页 32 低 2.4%"｜修复：｜复验：
- [重要·技术] index.html 第 4 章示例表格「32（vLLM/TensorRT）」行：把页 32 归为 vLLM 的典型页与来源矛盾——Strata §2.2 明确 vLLM 典型页为 16、TensorRT-LLM 为 32、SGLang 为 1；32 仅是 vLLM 在 CUDA GPU 上的最大支持值与实验设定值｜引文依据：1.5_background.tex §2.2 "Typical page sizes are small—e.g., 32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang"；2_motivation.tex §3.1 "Page size is set to 32, a value recommended in prior works … and a maximum supported size in vLLM for CUDA GPUs"｜修复要求：该行标注改为「32（TensorRT-LLM 默认；vLLM 实验设定/最大支持值，默认 16）」或删去 vLLM 改标 TensorRT-LLM，并在正文或来源说明注明 vLLM 默认页为 16｜修复：｜复验：
- [重要·技术] index.html 来源说明「核心论断与来源」C2 与第 4 章问题 2 答案：来源说明把「cache-aware 调度 + 兼容性」列入 C2（引 SGLang §1、§3.2），但正文从未呈现 SGLang 自己实现了 cache-aware 调度与兼容性（continuous batching/paged attention/tensor parallelism）这两项；正文唯一一处「cache-aware 调度」（第 4 章问题 2 答案「其调度缓解是 Strata 一文的 cache-aware 调度部分」）将该概念归给 Strata 一文，与 C2 的归属矛盾，会使读者误以为 cache-aware 调度源自 Strata｜引文依据：sglang-paper.txt §3.2 "We implement an LRU eviction policy and a cache-aware scheduling policy to enhance the cache hit rate. RadixAttention is compatible with techniques like continuous batching [60], paged attention [23], and tensor parallelism [44]."｜修复要求：二选一并保持一致：在正文（第 2 或第 3 章）补一句 SGLang 实现 cache-aware 调度与兼容性的原论断；或收窄来源说明中 C2 的表述为正文实际覆盖的范围，并把第 4 章答案措辞改为「Strata 一文针对 delay hit 的调度缓解」以避免与 SGLang 的 cache-aware scheduling 混同｜修复：｜复验：
- [重要·可读性] index.html 第 2 章「三个请求走树的完整过程」折叠块：R3 的 prefill 计算量 230 = |S′| + |m₃|，但 S′ 与 m₃ 的长度全文（含来源说明「构造示例」段）从未给出（S=200、m₁=30、m₂=40 均已给），500 vs 690 的算例无法复算，推导缺输入定义｜引文依据：不适用｜修复要求：在折叠块 R3 处补「设 $S'$ 长 200 token、$m_3$ 长 30 token」（或在来源说明构造示例段一并给出），使 230、500、690 可复算｜修复：｜复验：
- [轻微·格式] index.html 行 781（summary「负载共享结构 × 缓存容量 × 页粒度」）、行 875（3 处「根 → 节点 A/B/D」）、行 1008/1011（「页大小 = 1 token」「100 token × 页 1/32/256」），overview.html 行 62（「命中率 = … × … × …」）：Unicode 数学字符 ×、→、= 在 summary、正文、列表中直接出现，违反「数学符号全部由 KaTeX 渲染」条款（nav 与 footer 中的 ·、→ 属界面文案，不计）｜引文依据："不适用"｜修复要求：× 改 $\times$、→ 改 $\to$ 或改写为文字、= 改「为」或 $=$，两文件同步修改｜修复：｜复验：
- [轻微·格式] index.html 第 4 章问题 1 答案：「$\lfloor$共享前缀长度$/ $页大小$\rfloor$」把一个表达式拆成多段 math 与文字混排，且与 F1 展示式（\frac 写法）写法不一致｜引文依据："不适用"｜修复要求：统一为 $\left\lfloor \text{共享前缀长度}/\text{页大小} \right\rfloor$ 一类单一 math 段的写法｜修复：｜复验：
- [轻微·可读性] index.html 第 5 章末句「这个谱系在第 5 章结尾交代」：该句本身位于第 5 章结尾，形成自指，读者无从获得增量信息｜引文依据："不适用"｜修复要求：删去该短句，或改为「这一谱系的分层扩展在 Strata 一文展开」｜修复：｜复验：
- [轻微·可读性] index.html 第 2 章正文段首「匹配的走法。构造示例。设系统提示词…」与第 4 章「构造示例。两个请求共享…」：连续无谓语碎句，阅读磕绊｜引文依据："不适用"｜修复要求：合并为完整句，如「下面用一个构造示例讲匹配的走法。设…」「构造示例：两个请求共享…」｜修复：｜复验：
- [轻微·可读性] index.html 来源说明「简化条件」第二条：缩写「LPM」首次（且唯一）出现未展开，正文亦无定义｜引文依据："不适用"｜修复要求：写为「最长前缀优先（LPM）」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 5
- 处置：修复


## Round 1 修复记录

| 编号 | 问题简述 | 修复 | 复验 |
|---|---|---|---|
| 重要 #1 | 五处 2.4% 对照对象错位 | 全文与 overview 统一改为 "页 512 配置的 SGLang-HiCache 命中率比 Strata-IO 低 2.4%"；Strata-IO（页 1）作对照 | 验证 |
| 重要 #2 | 表格 32（vLLM/TensorRT）错配 | 改 "32（TensorRT-LLM 默认；vLLM 默认 16，32 是最大支持值）" | 验证 |
| 重要 #3 | C2 含 cache-aware 调度 + 兼容性但正文未呈现 SGLang 实现 | C2 收窄为正文实际覆盖：LRU 逐出 + 引用计数 + 页 1 token + radix tree 组织 + 与运行请求共享内存池 | 验证 |
| 重要 #4 | R3 长度未定义导致 500/690 算例不可复算 | 折叠块开头补 "$S$ 长 200、$m_1$ 长 30、$m_2$ 长 40、$S'$ 长 200、$m_3$ 长 30 token" | 验证 |
| 轻微 #1 | Unicode ×、→、= | 批量 replace body 内 → 至 $\to$，× 至 $\times$，=（部分）至 $=$；summary 中 "页 512 比页 32 低 2.4%" 同步改为对照对象修正版 | 可见 ×=0、→=0 |
| 轻微 #2 | floor 多段写法 | 改 $\left\lfloor\text{共享前缀长度}/\text{页大小}\right\rfloor$ | 验证 |
| 轻微 #3 | "这个谱系在第 5 章结尾交代" 自指 | 改 "这一谱系的分层扩展在 Strata 一文展开"（链接） | 验证 |
| 轻微 #4 | "匹配的走法。构造示例。" 碎句 | "下面用一个构造示例走一遍匹配过程。设 ..."；§4 "构造示例：两个请求共享 100 token 前缀（与 PagedAttention 的页大小权衡同源）" | 验证 |
| 轻微 #5 | LPM 未展开 | 来源说明首次出现改为 "最长前缀优先（LPM）缩写" | 验证 |

机械验证：validate.py ok；body 内 unicode 数学字符清零；overview 同步。
