<!-- review-meta
round: 3
page: wiki/deepseek-moe/index.html
reviewed_content_sha256: 30686c800d13718b
-->
# DeepSeek MoE 审查记录（第 3 轮）

- 页面版本：`git hash-object wiki/deepseek-moe/index.html` = `2e129bd4281d4b69bb7eaa1ddf398fdfbb9fe4c2`（overview.html = `4017ad13e59ce1a9eff81c61dec3854525da4634`）
- 审查时间：2026-09-10 21:50（CST）
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查；未读取 `wiki/deepseek-moe/research/` 下任何文件）
- 已完整阅读章节：引言（含 scope 声明）→ 核心问题（4 题及答案折叠块）→ 1. 传统 MoE 的两个毛病（含 flow-diagram、本章问题）→ 2. 把专家切小（含两个构造示例、手算表格、`代码：` 折叠块、本章问题）→ 3. 把通用知识拎出来（含 flow-diagram、`补充：` 折叠块、本章问题）→ 4. 相同算力下真的更好（含对比表格、callout-yellow、本章问题）→ 5. 影响与继承（含本章问题）→ 结语段 → 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）→ overview.html 全文

## 机械验证结果

### 1. 规范脚本

```
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/deepseek-moe/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/deepseek-moe/index.html   (exit 0)
```

### 2. 逐项核对

| 检查项 | 结果 |
|---|---|
| 引用双向闭合 | 正文 `<sup>` 引用：C1、C2、C3、C4、C6、C7、C8、F3、F4、F5、F6、F7、N1–N8；来源章节定义 C1–C8、F1–F7、N1–N8。**正文未引用 C5、F1、F2**（见问题 7）；无「引用但未定义」项 |
| 相邻双上标 | 正则 `</sup>\s*<sup>` 命中 0 处 |
| Unicode 数学字符（正文，排除 `<pre>`） | 仅出现 `×`(U+00D7)、`→`(U+2192)、`↓`(U+2193)、`↑`(U+2191)。validate.py 源码注释明确「× – → 等在中文技术散文中作为普通排版字符使用，不列入」；希腊字母、上下标数字、`∈ ≤ ≥ ≈ √ Σ ∂ ⊙` 等 0 处 |
| TAB | 0 行 |
| 占位符 | `【】` / `TODO` / `TBD` / `待生成` 均 0 处 |
| `<head>` 五项元数据 | index.html：`description`、`dojo:summary`、`dojo:type=concept`、`dojo:topics=模型结构`、`dojo:tag=MoE 架构` 五项齐全且非空。overview.html 无 `dojo:*`，与同站 `moe-serving/overview.html`、`stable-latent-moe/overview.html` 写法一致，validate.py 亦只校验 index.html |
| overview ↔ index 互链 | overview.html → `index.html`（「深度教学 →」）；index.html 顶部 nav → `overview.html`（「快速阅读」）。双向存在 |
| 前置概念链接有效 | `../moe-serving/index.html`、`../stable-latent-moe/index.html` 两个目录与 index.html 均存在 |
| 可运行代码块实跑 | 提取页面唯一 `<pre><code class="language-python">`，`python3` 执行：输出 11 行与页面「预期输出」逐行一致（4.0/1.0/4→28；120→4426165368；7/1/2/4.0/1.0，守恒均 True），无 stderr，rc=0 |
| 人称与用词 | 无「你/您/我们」；无未作答问题块（核心问题 4 题、5 章本章问题共 14 题均带 `解答：` 折叠块） |

### 3. 来源逐条核对（对照原文片段，按页面标注的章节/公式/表号定位）

- **C1/C2/C3**（知识混合、知识冗余、专门化定义）：Dai et al. 2024 §1 逐字含 "**Knowledge Hybridity**: existing MoE practices often employ a limited number of experts (e.g., 8 or 16), and thus tokens assigned to a specific expert will be likely to cover diverse knowledge... the designated expert will intend to assemble vastly different types of knowledge in its parameters, which are hard to utilize simultaneously"、"**Knowledge Redundancy**: tokens assigned to different experts may require common knowledge. As a result, multiple experts may converge in acquiring shared knowledge in their respective parameters"、"expert specialization, i.e., each expert acquires non-overlapping and focused knowledge"。摘要亦有同句。✔
- **C4/F3/F4**（细粒度分割、Eq.(6)(7)(8)）：§3.1 逐字 "we segment each expert FFN into *m* smaller experts by reducing the FFN intermediate hidden dimension to 1/m times its original size... we also increase the number of activated experts to *m* times"；Eq.(6) `h=Σ_{i=1}^{mN}(g·FFN)+u`、Eq.(7) Topk 范围 `1≤j≤mN` 取 `mK`、Eq.(8) `s=Softmax_i(u^T e_i)`，与页面 F3/F4 逐项一致。✔
- **C5/N8**（组合数）：§3.1 逐字 "A typical top-2 routing strategy can yield `C(16,2)=120` possible combinations. By contrast, if each expert is split into 4 smaller experts... `C(64,8)=4,426,165,368` potential combinations"。✔
- **C6/F5/F6**（共享专家、Eq.(9)(10)(11)）：§3.2 逐字 "we further isolate K_s experts to serve as shared experts. Regardless of the router module, each token will be deterministically assigned to these shared experts"；Eq.(9)(10)(11) 与页面 F5/F6 逐项一致；同节 "the total number of routed experts is mN−K_s, and the number of nonzero gates is mK−K_s"。✔
- **C7**（三架构参数与计算恒定）：Figure 2 caption 逐字 "across these three architectures, the number of expert parameters and computational costs remain constant"；§3.1 逐字 "while maintaining a consistent number of expert parameters and computational cost"；§3.2 逐字 "In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by K_s"。三处标注与原文完全对应。✔
- **F1/F2**（通用 top-K 层 Eq.(3)(4)(5)）：§2 逐字 Eq.(3) `h=Σ_{i=1}^{N}(g·FFN)+u`、Eq.(4) Topk 范围 `1≤j≤N` 取 `K`、Eq.(5) `s=Softmax_i(u^T e_i)`。✔
- **F7**（守恒推导）：由 C7 声明与 F3/F5 直接推出，复算 `mK·(1/m)=K`、`K_s·(1/m)+(mK−K_s)·(1/m)=K` 成立。✔
- **N1/N2**（2B 实验）：Table 2 逐字 "# Total Expert Params 2.83B / # Activated Expert Params 0.35B"（GShard×1.5）、"Pile (Loss) 1.808"（GShard×1.5）与 "1.808"（DeepSeekMoE）；Table 2 caption 逐字 "DeepSeekMoE nearly approaches the performance of a dense model with 16 times FFN parameters, which sets the upper bound for MoE models in terms of the model capacity"；摘要逐字 "GShard 2.9B, which has 1.5 times the expert parameters and computation"。✔（表格标签精度见问题 6）
- **N3**（2B 配置）：§4.1.3 逐字 "the number of Transformer layers to 9 and the hidden dimension to 1280... approximately 2B total parameters, with the number of activated parameters around 0.3B"；§4.2 "1 shared expert and 63 routed experts, where each expert is 0.25 times the size of a standard FFN"；§4.5 "1 shared expert and 7 out of 63 routed experts being activated"。数值全对，**标注章节不完整**（见问题 4）。✔/⚠
- **N4**（16B vs LLaMA2 7B）：Table 4 逐字 "FLOPs per 4K Tokens 187.9T（LLaMA2 7B）/ 74.4T（DeepSeekMoE 16B）"，caption "With only 39.6% of computations..."。✔
- **N5**（16B 配置）：§5.1.2 逐字 "28 ... hidden dimension to 2048 ... 2 shared experts and 64 routed experts, where each expert is 0.25 times the size of a standard FFN. Each token will be routed to these 2 shared experts and 6 out of 64 routed experts ... 16.4B total parameters ... around 2.8B ... 2T training tokens"。✔
- **N6**（单卡部署与速度）：§5.2.1 逐字 "it enables single-device deployment on a GPU with 40GB of memory. With appropriate operator optimizations, it can achieve nearly 2.5 times the inference speed of a 7B dense model"。✔
- **N7**（145B）：摘要逐字 "performance comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations"；§7/Table 6 逐字 "with 28.5% of computations, DeepSeekMoE 145B achieves comparable performance with DeepSeek 67B"、"DeepSeekMoE 142B (Half Activated) ... with only 18.2% of computations"。✔
- **C8**（V3 继承）：DeepSeek-V3 arXiv:2412.19437 §2.1.2 逐字 "DeepSeek-V3 uses the sigmoid function to compute the affinity scores"、Eq.(16) 含 bias `b_i` 参与 top-K、"we pioneer an auxiliary-loss-free load balancing strategy"；1 共享 + 256 路由、激活 8 个在 §4.2 逐字 "Each MoE layer consists of 1 shared expert and 256 routed experts ... 8 experts will be activated for each token"。✔（"256 + 8" 的标注位置见问题 5）
- **§3.3 负载均衡**（补充折叠块）：§3.3 逐字 "**Expert-Level Balance Loss.** ... L_ExpBal"、"**Device-Level Balance Loss.** ... L_DevBal"。✔
- **引用的前置页章节**：`moe-serving/index.html` 第 1 章含 "FFN 的两个矩阵分别是 $d\times d_{ff}$ 和 $d_{ff}\times d$"（即 $2dd_{ff}$）；第 2 章含 `y(x)=\sum_{i\in S_k(x)} g_i(x)\cdot E_i(x)` 与「均衡机制（辅助损失或偏置调整）」。页面 §1/§2/§5 的三处「前置页」指引均落地。✔

## 问题

- [轻微·格式] index.html:704（引言段）：变量 `m` 未用 `$...$` 包裹——原文"（每个专家切 m 份、激活数也 ×m，计算量守恒但组合更灵活）"中的 `m` 为裸 ASCII，同页其余位置（如 :717、:787、:795、:808）一律写 `$m$`，违反 style-guide §11「同一变量在页面中保持同一种写法」与 check.md §5「数学符号全部使用 LaTeX 书写」。｜引文依据：不适用（写法一致性，非来源论断）｜修复要求：把该处改为"每个专家切 $m$ 份、激活数也增至 $m$ 倍"（或等价的 `$...$` 写法），使全页不再出现裸 `m`｜修复：index.html:704 改为"每个专家切 $m$ 份、激活数也增至 $m$ 倍"，裸 `m` 与 Unicode `×` 一并消除。另按格式要求清除页面全部裸 Unicode 数学字符：index.html 的 10 处 `×`、9 处 `→`、3 处 `↓`，overview.html 的 1 处 `×`、2 处 `→`（含义为"从…到"的改写成"增到/倍"，`Dense×16` 改写为 `Dense$\times$16`，fd-arrow 箭头改写为 `$\downarrow$`）。｜复验：`grep -c '×\|→\|↓'` 在两文件均返回 0；仅余返回顶部按钮的 `↑` 与 overview 导航的 `←` 两个 UI 控件字形（validate.py `UI_CONTEXT_TAGS` 明确排除界面控件），已在此说明。validate.py 通过。
- [轻微·格式] index.html:893（§2「教学简化」段）：`mK` 未用 `$...$` 包裹——原文"组合数假设 router 无约束地选任意 mK 个，实际还受负载均衡影响"。同一论断在来源章节「简化条件（2）」写作"选任意 $mK$ 个"，两处写法不一致。｜引文依据：不适用｜修复要求：改为"选任意 $mK$ 个"｜修复：index.html:893 改为"组合数假设 router 无约束地选任意 $mK$ 个"。｜复验：正文该处与来源章节"简化条件（2）"写法一致，均为 `$mK$`；全页不再出现裸 `mK`。validate.py 通过。
- [轻微·技术] index.html:1016（§4 第一段）："DeepSeekMoE 2B 不仅打赢了计算更贵的 GShard 2.9B" 与本节表格（:1009"性能相当（Pile Loss 均 1.808）"）及来源措辞冲突，属对来源结论的拔高。｜引文依据：摘要逐字"DeepSeekMoE 2B achieves comparable performance with GShard 2.9B"；Table 2 caption 逐字"DeepSeekMoE achieves comparable performance with a GShard model containing 1.5 times expert parameters and computation"；Table 2 Pile (Loss) 两列同为 1.808（论文 §4.3 逐字"achieves comparable performance with GShard×1.5"）｜修复要求：改为与来源一致的表述，如"与计算更贵的 GShard 2.9B 性能相当（在多数基准上小幅领先）"，删除"打赢了"｜修复：index.html:1016 删去"不仅打赢了"，改为"DeepSeekMoE 2B 与计算更贵的 GShard 2.9B 性能相当（Pile Loss 均 1.808），还接近了同总参数的稠密模型…"；未采用"多数基准小幅领先"（该论断本页无 benchmark 逐项数据支撑）。同时把同段"几乎榨干了 MoE 架构的潜力"降级为来源措辞"已接近 MoE 架构在模型容量上的上界"。｜复验：与摘要"comparable performance with GShard 2.9B"、Table 2（Pile Loss 均 1.808）、Table 2 caption"achieves comparable performance with a GShard model containing 1.5 times expert parameters and computation"、§4.3"achieves comparable performance with GShard×1.5"逐条一致，且与本节对比表"性能相当（Pile Loss 均 1.808）"不再冲突。
- [轻微·技术] index.html:1086（来源章节 N3）：标注位置「§4.1.3」只含层数（9）、隐藏维度（1280）与总参/激活参数量（2.0B/0.3B）；N3 中的"1 shared + 63 routed"、"每专家 0.25× 标准 FFN 即 m=4"、"激活 1+7=8"分属 §4.2、§4.5 与 Table 2，按 check.md §2.2 第 1–2 步在标注位置定位不到。｜引文依据：§4.2 逐字"DeepSeekMoE has 1 shared expert and 63 routed experts, where each expert is 0.25 times the size of a standard FFN"；§4.5 逐字"1 shared expert and 7 out of 63 routed experts being activated"｜修复要求：N3 定位改为"§4.1.3、§4.2、§4.5；Table 1/Table 2"｜修复：N3 定位已改为"§4.1.3、§4.2、§4.5；Table 1、Table 2"。｜复验：逐项在标注位置定位成功——§4.1.3 逐字"the number of Transformer layers to 9 and the hidden dimension to 1280 … approximately 2B total parameters, with the number of activated parameters around 0.3B"；§4.2 逐字"1 shared expert and 63 routed experts, where each expert is 0.25 times the size of a standard FFN"；§4.5 逐字"comprising 2.0B total parameters, with 1 shared expert and 7 out of 63 routed experts being activated"（该句即指向 Table 1）。
- [轻微·技术] index.html:1080（来源章节 C8）：「1 个共享 + 256 个路由、top-8」并入 DeepSeek-V3 §2.1.2 标注，但 §2.1.2 仅以 $N_s$、$N_r$、$K_r$ 符号给出，具体数值在 §4.2。｜引文依据：§2.1.2 逐字"where $N_s$ and $N_r$ denote the numbers of shared experts and routed experts, respectively ... $K_r$ denotes the number of activated routed experts"；§4.2 逐字"Each MoE layer consists of 1 shared expert and 256 routed experts ... 8 experts will be activated for each token"｜修复要求：C8 定位补注 §4.2（sigmoid/偏置/aux-loss-free 保留 §2.1.2）｜修复：C8 尾部改为"…arXiv:2412.19437 §2.1.2（sigmoid 亲和度、偏置项、aux-loss-free 均衡）、§4.2（1 共享 + 256 路由、激活 8）"。｜复验：V3 §4.2 逐字"Each MoE layer consists of 1 shared expert and 256 routed experts … 8 experts will be activated for each token"；§2.1.2 逐字含"DeepSeek-V3 uses the sigmoid function to compute the affinity scores"与"we pioneer an auxiliary-loss-free load balancing strategy"、"add it to the corresponding affinity scores $s_{i,t}$ to determine the top-K routing"。数值与机制分列 §4.2 与 §2.1.2，定位完整。
- [轻微·技术] index.html:1009（§4 对比表 2B 行）："GShard 2.9B（2.9B 总参/0.35B 激活，…）"与来源记法不符——Table 2 中 2.83B 是 "# Total Expert Params"、0.35B 是 "# Activated Expert Params"，均为专家参数，不是模型总参。｜引文依据：Table 2 逐字"# Total Expert Params 2.83B / # Activated Expert Params 0.35B"（GShard×1.5 列）｜修复要求：改为"GShard 2.9B（专家参数 2.83B/激活 0.35B，1.5× 专家参数与计算）"或明确标注为专家参数｜修复：对比表 2B 行的 GShard 单元格改为"GShard 2.9B（专家参数 2.83B/激活 0.35B，1.5 倍专家参数与计算）"，明确标注为专家参数（"1.5×"按本页格式规范写作"1.5 倍"）。｜复验：Table 2 行标签逐字为"# Total Expert Params 2.83B""# Activated Expert Params 0.35B"，与页面新写法一致；不再把 2.83B 误作模型总参。
- [轻微·格式] index.html:1083–1086（来源章节）：C5、F1、F2 在来源章节有定义，但正文无对应 `<sup>` 引用（同一事实正文改用 N8 与前置页链接承载），不满足 style-guide §6 的「与来源章节双向对应」。｜引文依据：不适用｜修复要求：在 §2 组合数句补 `<sup>[C5]</sup>`、在 §1 通用 MoE 公式句补 `<sup>[F1, F2]</sup>`；若判断为冗余，则从来源章节删除 C5/F1/F2 三条目。二者择一，使定义集合与引用集合一致｜修复：采用"补引用"方案——§2 组合数句改 `<sup>[N8]</sup>` 为 `<sup>[C5, N8]</sup>`；§1 通用 top-K MoE 公式句在句末补 `<sup>[F1, F2]</sup>`（组合引用写在同一上标内，未产生相邻双上标）。｜复验：脚本比对正文 `<sup>` 引用集合与来源章节定义集合，两者均为 {C1–C8, F1–F7, N1–N8}，`ref-not-def` 与 `def-not-ref` 均为空；`</sup>\s*<sup>` 正则命中 0；来源章节 h3 命名仍为固定的"论断与来源（C）"/"公式与来源（F）"/"外部数字与实验条件（N）"等，与 `moe-serving`、`latent-moe` 一致。
- [轻微·技术] index.html:6–7（`<head>` 的 description / `dojo:summary`）：两处引入"K3 的 Stable LatentMoE"这一归因，但正文第五章只写"Stable LatentMoE"，从未出现或解释"K3"，且 `blockquote.meta` 声明的主要依据只有 Dai 2024，读者无法从页面定位该归因。｜引文依据：不适用（可读性/一致性；「K3」的归属见前置页 `stable-latent-moe/index.html` 的 `description`，不在本页来源范围内）｜修复要求：二选一——①在 §5 首次提及时写成"Stable LatentMoE（K3 使用）"并补来源；②把 description 与 `dojo:summary` 中的"K3 的"删去，只写"Stable LatentMoE"｜修复：采用方案②——删去 `description` 与 `dojo:summary` 中的"K3 的"（`description` 内的裸变量 `m` 一并改写为"更小的多个小专家"，使元数据不含 ASCII 数学变量）。｜复验：`K3` 全文命中 0；`description` 不含 `$`，`dojo:summary` 的 `$` 成对（validate.py 两条相关规则通过）；§5 正文继续只写"Stable LatentMoE"，与元数据一致。另：overview.html 导航"深度教学 →"的 `→` 一并去除，使该页亦无裸 Unicode `→`。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 8
- 处置：修复

### check.md §5 发布条件逐条核对

| 条件 | 状态 |
|---|---|
| 三轮审查均由未参与写作的独立审查者执行 | 第 3 轮由本轮独立审查者执行 ✔；前两轮记录未读取（不在允许输入内），就本轮而言不构成障碍 |
| 每条来源论断都有引文依据记录；无法核对的已删除或降级 | ✔ 本轮逐条定位并记录原文片段（见「来源逐条核对」），未发现无法核对仍需保留的论断 |
| 所有阻断和重要问题均已关闭 | ✔ 本轮无阻断、无重要问题 |
| 遗留轻微问题具有明确的接受理由 | ✘ 未满足：本轮 8 条轻微问题未给接受理由，其中问题 1、2 触及 §5「数学符号全部使用 LaTeX 书写」硬条件 |
| 全部学习目标由正文章节完整回答 | ✔ 4 条核心问题分别由第 1–4 章完整回答，第 5 章另有承接 |
| 核心问题与各章本章问题均有解答折叠块 | ✔ 核心问题 4/4、本章问题 14/14，答案独立可读且与正文一致 |
| 数学符号全部使用 LaTeX，结构图为 HTML 或内联 SVG | ✘ 未满足：index.html:704 的裸 `m`、:893 的裸 `mK`（其余公式均由 KaTeX 承载；两处 flow-diagram 为 HTML div 结构，无等宽字符框线图，公式在 `fd-caption`/`fd-box` 中由 KaTeX 渲染） |
| `.dojo/scripts/validate.py` 返回成功 | ✔ validation ok，exit 0 |
| 可运行代码的结果与页面描述一致 | ✔ 实跑输出与「预期输出」逐行一致 |
| 关键论断和数字已重新核对来源 | ✔ 见「来源逐条核对」 |
| `<head>` 五项元数据有效 | ✔ index.html 五项齐全；`dojo:topics=模型结构` 在词表内 |
| overview.html 与 index.html 相互链接 | ✔ |
| 页面引用的概念链接有效或有明确占位 | ✔ `../moe-serving/index.html`、`../stable-latent-moe/index.html` 均存在 |
| 递归生成的前置概念页已完成各自质检 | ⚠ **无法确认**：本页引用的 `moe-serving`、`stable-latent-moe` 两页均存在且链接有效，但各前置页的质检记录不在本轮允许输入范围内（审定规范 §1 限定审查者只读待审页面、外部来源与规范），本轮未读取其 `research/` 记录，无法确认其已完成三轮质检 |

综上：除「轻微问题接受理由/数学符号 LaTeX 化」与「前置页质检状态无法确认」两条外，其余发布条件均满足。修复上述 8 条轻微问题（尤其问题 1、2）并重跑 validate.py 后，可满足 check.md §5 全部可自证条件；前置页质检状态需由编排者补充确认。
