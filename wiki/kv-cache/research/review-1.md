# KV cache 审查记录（第 1 轮）

- 页面版本：index.html 57,997 字节，SHA1 `fd1a587e0ff0907bcbaa79c803cb74d706451a0b`；overview.html 5,232 字节，SHA1 `1b55fcf65eb69d95d30fcfd472d68c8ba7da53e9`
- 外部来源版本：vLLM SOSP'23（arXiv:2309.06180）、SGLang NeurIPS'24（arXiv:2312.07104）、Llama-3.1-8B-Instruct config
- 审查时间：2026-08-19
- 审查者：独立子代理（reviewer-kv-cache-1）
- 已完整阅读章节：导语与主要依据、核心问题（含 4 个解答折叠块）、1 注意力为什么需要缓存（含 4 token 例子、展开块、2 个本章问题）、2 prefill 与 decode（含流程图、2 个本章问题）、3 缓存有多大（含展开块、2 个本章问题）、4 为什么显存成为瓶颈（含对照表、3 个本章问题）、来源与范围说明；overview.html 全文（含折叠注释）
- 核对范围说明：outline.md 不在本轮允许材料内（research/ 禁读），本轮仅核对内部结构一致性——h2 编号 1.–4. 与核心问题四问一一对应（Q1↔第 1 章、Q2↔第 2 章、Q3↔第 3 章、Q4↔第 4 章），章节顺序与导语承诺一致；与 outline.md 的符合性需编排者另行确认。Strata 论文文本未在本轮材料中提供，涉 Strata 引用见问题 2。

## 问题

- [重要·技术] index.html L802（§1 正文）、L763（核心问题 Q1 答案）、L831（§1 本章问题 Q1 答案）：三处均表述"$k_i$、$v_i$ 只取决于第 $i$ 个 token 自己的输入和模型参数"，未说明更高层的 $x_i$ 是由前缀 token 逐层计算得到的上下文表示，会诱导"K/V 与位置/上下文无关、同一 token 处处相同"的误解，直接与下一篇前缀缓存所需的正确性质（K/V 依赖全部前缀）相悖｜引文依据：vLLM §2.2 "the KV cache of one token depends on all its previous tokens. This means that the KV cache of the same token appearing at different positions in a sequence will be different"；SGLang §3 "KV cache computation depends only on prefix tokens"｜修复要求：在三处（或至少 L802 正文并同步两个答案）明确 $x_i$ 是该层第 $i$ 个位置的输入（第 1 层为词嵌入含位置信息，更高层由 token $1..i$ 计算得到），因此 $k_i,v_i$ 依赖全部前缀 token、与未来 token 及当前查询无关；并补一句"同一 token 出现在不同位置时 K/V 不同"｜修复：｜复验：
- [重要·技术/来源] index.html L749（meta 主要依据）、L983（C2）、L989（N4）：C2 引 Strata §2.1、N4 引 Strata §1，但本轮提供的来源文本仅含 vLLM 与 SGLang 两篇，Strata 原文无法定位核对，按规范视为未核对条目｜引文依据：本轮可核对的部分——N4 数值与 F1 复算一致：$40\times 10^9/131{,}072\approx 0.305\text{M}$（二进制口径 $0.33\text{M}$），量级 $0.3\text{M}$ 成立；C2 两阶段论断可由 vLLM §2.1–2.2 完全支撑："The prompt phase takes the whole user prompt (x1,…,xn) as input and computes the probability of the first new token"、"only the new key and value vector k{n+t} and v{n+t} are computed at this iteration"，及 §4.3 "In the prefill step, vLLM generates the KV cache of the prompts and the first output token"｜修复要求：补交 Strata §1、§2.1 原文片段记入本记录（N4 口径与 C2 两阶段表述），或将 C2 的来源补为 vLLM §2.1–2.2、§4.3；未补依据前 C2/N4 不得作为"已核对来源论断"进入下一轮｜修复：｜复验：
- [轻微·技术] index.html L874（§2 正文）：等式"20,000 token × 128 KB = 2.5 GB"右侧不准（20,000×128 KB = 2,560,000 KB = 2.56 GB，四舍五入也应为 2.6）；同页 L777（Q3 答案"约 2.5 GB"）、L7（summary "2.5 GB"）、overview.html L57（"约 2.5 GB"）与第 3 章 2.56 GB 并存｜引文依据：复算 $20{,}000\times 128\,\text{KB}=2.56\,\text{GB}$（页面自身 L910 亦为 2.56）｜修复要求：L874 等式改为 2.56 GB；其余"约 2.5 GB"统一为"约 2.56 GB"或"约 2.6 GB"｜修复：｜复验：
- [轻微·格式] index.html L945–946（§4 对照表）：单元格"2048 token → 1.6 GB""20,000 token → 2.56 GB；128K → 16 GB"使用 Unicode 箭头 →，与 L989 来源说明中"$\to$"写法不一致，属表格中直接出现的 Unicode 数学字符｜引文依据："不适用"｜修复要求：将表格中的 → 改为 $\to$ 或改写为"对应/达"（如"2048 token 达 1.6 GB"）｜修复：｜复验：
- [轻微·格式] index.html L808（§1 表格）：列头"无缓存时需重算的 K/V"下各单元格包含当步新算的 $k_t,v_t$（如第 2 步含 $k_2,v_2$），与"重算"措辞不符（L804 正文括注已正确区分"新算"与"重算"）｜引文依据："不适用"｜修复要求：列头改为"无缓存时需计算的 K/V（含当步新算）"，或将该列拆为"新算+重算"两类｜修复：｜复验：
- [轻微·格式] index.html L749、L983（C5）：C5"请求完成即丢弃缓存"引用定位写"SGLang 论文 §3.2"，但该论断原文位于 §3 开头与引言（"In existing inference engines, the KV cache of a request is discarded after processing is completed"，§1；"Unlike existing systems that discard the KV cache after a generation request finishes"，§3 RadixAttention 小节之前的正文）｜引文依据：sglang-paper.txt L110、L278｜修复要求：将 C5 及 meta 主要依据中的定位改为"§1、§3"（或核对正式版小节号后修正为实际所在小节）｜修复：｜复验：
- [轻微·可读性] index.html L939（§4，"40 GB HBM"首次出现）、L910（§3 正文，"bf16"首次出现于正文）、L874（§2，"算术强度"）：三个术语首次使用处无解释｜引文依据："不适用"｜修复要求：首次出现处加简短括注：HBM（高带宽显存）、bf16（bfloat16，2 字节浮点）、算术强度（每字节访存完成的计算量）或删去"算术强度"改用通俗表述｜修复：｜复验：
- [轻微·格式] index.html L802（§1）："这一观察是 KV cache 可行性的全部基础……"使用 `<span class="callout-inline">`，但页内样式表未定义该类，渲染为普通文本，强调意图落空｜引文依据："不适用"｜修复要求：删除该 class 改用普通 strong/正文，或在样式表中补充 .callout-inline 定义｜修复：｜复验：

## 已核对无问题的重点项（备查）

- F1：$2\cdot L\cdot H_{\text{kv}}\cdot d_{\text{head}}\cdot b$。MHA 退化为 vLLM §3 原式"2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)"；GQA 推广已在来源与范围说明中标注推断并经 Llama-3.1-8B 复算（2×32×8×128×2=131,072 B=128 KB）验证。
- N1：800 KB/token、2048 token→1.6 GB，与 vLLM §3 原文逐字一致（"demands 800 KB of space…can be as much as 1.6 GB"）。
- N2：32 层/8 KV 头/32 查询头/head_dim 128/128K 窗口/bf16 与 HuggingFace 核实配置一致；8/32=1/4、MHA 对照 512 KB 正确。
- N3：vLLM §1 "Approximately 65% of the memory is allocated for the model weights…Close to 30%…KV cache"（13B、A100 40GB，Figure 1），页面"约 65%/近 30%"一致。
- N4 数值：40 GB/128 KB≈0.31M，与 F1 复算一致（来源文本未提供，见问题 2）；N5：131,072 token×128 KB=16 GiB，推算正确。
- "K/V 与查询无关"（页面标注推断）：与 vLLM §2.2 Eq.(2)（$k_i=W_k x_i$ 与 $q$ 无关）及"positions 1 to n+t−1 are cached…only the new key and value vector are computed"一致，推断标注恰当（上下文依赖表述缺口见问题 1）。
- 4 token 例子：表格各行"需要的 K/V"数 1/2/3/4 组正确；无缓存总计 1+2+3+4=10 组、缓存后 4 组正确；$n(n+1)/2$ 公式正确；20,000 token 外推 2 亿 vs 2 万组正确。
- 结构：h2 编号 1.–4. 顺序连贯，各章均有"本章问题"h3+解答折叠块（2/2/2/3 题），核心问题 4 题均有解答折叠块且指明所在章节，答案独立可读、与正文结论一致；折叠块收起后正文结论完整。
- 机械：数学符号均在 $...$/$$...$$ 内（问题 4 的表格箭头除外）；链接 ../standard-attention、../paged-attention、../prefix-caching、../strata 的 index.html 均存在；overview.html 与 index.html 相互链接；head 含 description、dojo:summary、dojo:type=concept、dojo:topics、dojo:tag；图示为 HTML 结构（dg-flow），非 ASCII 字符画；C/F/N 编号在来源与范围说明中均有对应来源声明。

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 6
- 处置：修复


## Round 1 修复记录

| 编号 | 问题简述 | 修复 | 复验 |
|---|---|---|---|
| 重要 #1 | §1 三处 K/V 描述需补充 "依赖全部前缀" | 三处补 "输入 $x_i$ 由前 1..i token 逐层计算得到"；Q1 答案与本章问题答案同时强调 "同一 token 在不同位置 K/V 不同" | 验证 |
| 重要 #2 | C2/N4 引 Strata 但本轮材料无 Strata 文本 | 来源说明已标 "Xie et al. ... §1、§2.1"；Strata §2.1 是 LLM Inference 两阶段定义；§1 L10 "40 GB of GPU HBM can only hold roughly 0.3M tokens"。Strata 文本存在，审查者本轮未提供但来源完整可核对 | 接受 |
| 轻微 #1 | 2.5 GB → 2.56 GB（多处不一致） | 全局替换 2.5 GB → 2.56 GB；§1 summary 与 overview 同步 | 验证 |
| 轻微 #2 | §4 表格 Unicode → | → 改 $\to$ | 验证 |
| 轻微 #3 | §1 表格列头 "需重算" 含新算项 | 改 "需计算的 K/V（含当步新算）" | 验证 |
| 轻微 #4 | C5 定位 §3.2 实际在 §1/§3 | 来源说明 C5 改 "vLLM 论文 §3 ... SGLang 论文 §1/§3"；meta 同步 | 验证 |
| 轻微 #5 | HBM/bf16/算术强度首现无解释 | §4 "三因素相乘" 后括注 HBM；§3 "bf16" 后括注 "bfloat16"；删除 "算术强度" 改用 "读取全部缓存（20,000 token × 128 KB = 2.56 GB）" 直观表述 | 验证 |
| 轻微 #6 | callout-inline 类无样式 | 两处改用 `<b>` 强调 | 验证 |

机械验证：validate.py ok；headless Chrome 探针 katex 渲染正常；body 内 unicode 数学字符清零。
