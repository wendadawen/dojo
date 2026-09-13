<!-- review-meta
round: 2
page: wiki/attention-sink/index.html
reviewed_content_sha256: 884e8e8684c178d3
-->
# 注意力汇聚点审查记录（第 2 轮）

- 页面版本：`b522f2a5c951c0116a70cb75ad505dd35ddffe71`（`git hash-object wiki/attention-sink/index.html`）
- 审查时间：2026-09-10
- 审查者：独立审查者（未参与写作，未阅读 `research/` 下规划、修复记录与前序审查）
- 已完整阅读章节：标题与 meta → 元信息 blockquote → 开篇 callout → 核心问题（4 条）→ 常见误解 → 1. 现象（含本章问题）→ 2. 成因（含本章问题）→ 3. 两种做法（3.1、3.2、展开折叠块、本章问题）→ 4. 对缓存与推理意味着什么（4.1、4.2、代码块、4.3、本章问题）→ 来源与范围说明（论断与来源 C / 公式与来源 F / 外部数字与实验条件 N / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）；另完整阅读 `overview.html`。

## 机械验证结果

| 项目 | 命令 / 方式 | 结果 |
|---|---|---|
| 结构校验 | `/usr/bin/python3 .dojo/scripts/validate.py wiki/attention-sink/index.html` | `validation ok: wiki/attention-sink/index.html`（退出码 0） |
| 页面代码块（4.2） | 复制第 983–988 行脚本写入 `/tmp/sink_share.py`，用 `/usr/bin/python3` 实跑 | 输出 4 行，与页面「预期输出」块（第 992–995 行）逐字符一致：`sink= 0.0: 份额 0.0078 = 0.78%` / `sink= 2.0: 份额 0.0546 = 5.46%` / `sink= 5.0: 份额 0.5369 = 53.69%` / `sink= 8.0: 份额 0.9588 = 95.88%` |
| `verify_sink.py` | `/usr/bin/python3` 与 `~/.workbuddy/binaries/python/versions/3.13.12/bin/python3` 均报 `ModuleNotFoundError: No module named 'torch'`；`pip3 list` 无 torch | 无法执行，改为静态审查 `verify_sink.out`（见下） |
| `verify_sink.out` 静态核对 | 逐行比对页面论断 | [1] `sink=-inf → [2.0,3.0,4.0,5.0]`（即 $(v_1{+}v_2)/2$）、`sink=0.0 → [1.333…,2.0,2.666…,3.333…]`（即 $(v_1{+}v_2)/3$）、`sink=2.0 → [0.42602…,0.63904…,0.85205…,1.06506…]`（$(1{+}3,2{+}4,3{+}5,4{+}6)/(2{+}e^2)=(4,6,8,10)/9.389$ 逐项吻合），三档「最大差 0.00e+00」；[2] 整行 `-1` 输出 `[0.0,0.0,0.0,0.0]`；[3] 份额四值与页面一致；[4] `张量总数 43 (主干 40 层 + MTP 3 层), 形状集合 {(64,)}, dtype 集合 {'F32'}`、`主干层号 0..39 连续: True` |
| 独立复算 checkpoint 头 | 直接读 `wiki/deepseek-v4-1/research/ckpt/headers.json` | `count 43`，`shapes {(64,)}`，`dtypes {'F32'}`，键名 `layers.0.attn.attn_sink` … `mtp.2.attn.attn_sink` |
| 官方源码定位 | `wiki/deepseek-v4-1/research/official/inference/` | `model.py:639 self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))`；`model.py:780/1067` 随稀疏注意力传入；`kernel.py:332 attn_sink: T.Tensor[(h,), FP32]` |
| 官方配置 | `official/inference/config.json` | `n_layers = 40`、`n_heads = 64`、`head_dim = 512`、`window_size = 128` |
| 公式外 Unicode 数学字符 | 对 `index.html` 去掉 `style`/`script` 后遮罩 `$...$` 与 `$$...$$`，扫描希腊字母、上标、数学运算符等区间 | 命中 0 |
| 引用编号闭环 | 正则提取正文 `<sup>[Cx/Fx/Nx]</sup>` 与来源章节 `[Cx/Fx/Nx]` | 引用 14 个（C1–C7、F1–F2、N1–N5），定义 14 个，无未定义、无定义未引用 |
| 前置概念链接 | 检查 `../standard-attention/`、`../kv-cache/`、`../sliding-window-attention/` | 三处 `index.html` 均存在 |
| 结构统计 | 正则计数 | h2 7 个（含「核心问题」「常见误解」「来源与范围说明」）；`details` 16 个，其中 `解答：` 13 个；页面级核心问题 4 条、章节级本章问题 4 组各 2 条 = 12 条，与 13 个「解答」之差 1 为 3.2 节的 `展开：` 折叠块（合理）；`callout` 仅 1 个（blue）；`table-scroll` 3 个；页面无内联 SVG 与结构图 |
| 本地资源 | `libs/katex.min.js`、`libs/auto-render.min.js`、`libs/prism*.js/css` | 均存在 |

## 问题

- [重要·可读性] 第 4.2 节正文（第 978 行）：在「$W=128$、槽位等权」这一构造假设下，份额 $e^{\text{sink}}/(W+e^{\text{sink}})$ 只能随 $\text{sink}$ 从 $0$ 单调升向 $1$，永远取不到 $1$，但正文写「份额随标量取值可覆盖 $\mathbf{0.78\%}$–$\mathbf{95.88\%}$」，读起来是在断言覆盖率上限为 95.88%；同时该句没有交代 95.88% 对应的标量取值（第 4.2 节正文只提「标量从 0 增到 5」，$8$ 只出现在折叠代码块与核心问题答案里），而上限与取值一一对应是这个论断成立的唯一条件。另有第 3.2 节表格「数量」行用「43 层 $\times$ 64 头」标注 $[64]$ 参数规模，与来源章节 `[C4]`「覆盖主干层 0–39 与 3 个 MTP 层」的表述口径不一，容易被读成 43 个主干层。｜引文依据：不适用（属表述完备性；来源侧口径见 `[C4]`：「43 个 `attn_sink` 张量，形状集合 $\{(64,)\}$，dtype F32，覆盖主干层 0–39 与 3 个 MTP 层」，以及 `verify_sink.out` [4]：「张量总数 43 (主干 40 层 + MTP 3 层)」）｜修复要求：把第 978 行改为明确「份额随标量增大单调上升，$sink=0$ 时为 0.78%，$sink=8$ 时为 95.88%；真实取值由训练决定」并写出 8 这一取样点，或改成不含上限暗示的表述（如「随标量取值上升，在 0–8 区间内为 0.78%–95.88%」），二者择一；同时把第 3.2 节表格「数量」行的「该模型 43 层 $\times$ 64 头」改为「40 主干层 + 3 MTP 层 $\times$ 64 头」，与 `[C4]` 口径一致。｜修复：第 978 行改为「份额随 sink 单调上升：sink=0 为 0.78%、sink=2 为 5.46%、sink=5 为 53.69%、sink=8 为 95.88%；取值越大越接近 1 但永远取不到 1」，写出 8 这一取样点并去掉上限暗示；3.2 表格「数量」行改为「主干 40 层 + 3 个 MTP 层，各 64 头」｜复验：已复跑 validate.py 通过并核对修改位置｜

- [重要·可读性] 第 3.2 节「本章问题」第 2 题解答（第 946 行）：解答结论是「行最大值只用于数值稳定，实现在累积循环之后单独把 sink 项加入分母」。但该段在给出「对 $m$ 做公共平移只会让分子分母同时缩放，比值不变」之后，没有指出实现里分子同样带着 $e^{-m}$ 因子——即分子被 $scores\_scale$ 逐步缩放（`kernel.py:379` `acc_o[i, j] *= scores_scale[i]`）、分母由 `sum_exp` 累积后与 $e^{\text{sink}-m}$ 相加（`kernel.py:383`），两者共享同一个最终 $m$，因此「sink 不进 $m$」才不改变比值。读者按现状只能推出「$m$ 的取值无关紧要」，无从检验结论，形成推理跳步。同一行还有一个格式问题见下一条。｜引文依据：不适用（源文件侧对应 `kernel.py:371` `scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])`、`kernel.py:379` `acc_o[i, j] *= scores_scale[i]` 与 `kernel.py:382-383` `for i in T.Parallel(h): sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`）｜修复要求：在该段补一句说明分子与分母共用同一个最终行最大值：分子在累积过程中被 $e^{m_{\text{prev}}-m_{\text{cur}}}$ 逐块缩放，分母累积完成后加 $e^{\text{sink}-m}$，因此把 sink 计入 $m$ 只会让分子分母同乘一个因子、比值不变；并保留该段已有的「这个选择来自实现结构，不是数学上的必要条件」结论。｜修复：该段补一句「分子与分母共用同一个最终行最大值：内核在累积循环里用 e^{m_prev−m_cur} 逐块缩放分子与指数和，循环结束后加 e^{sink−m}，公共因子约掉、比值不变」，保留「选择来自实现结构而非数学必要性」的结论｜复验：已复跑 validate.py 通过并核对修改位置｜

- [重要·格式] 第 946 行（第 3.2 节「本章问题」第 2 题解答内）：`$e^{` + TAB(U+0009) + `ext{sink}-m}$` 中的制表符是 `\text` 被写坏的结果，KaTeX 渲染 `ext` 会在 `throwOnError: false` 下产出红色错误片段；该公式位于默认收起的折叠块内，`.dojo/scripts/validate.py` 只扫公式定界符之外的数学字符、不检查 LaTeX 可渲染性，因此能通过校验而不被肉眼发现。｜引文依据：不适用（`od -c` 确认第 946 行字节序列为 `e ^ { \t e x t { s i n k } - m }`；全页其余 `\text{...}` 均正常，同页第 7、896、901 行的 `e^{\text{sink}-m}` 写法正确）｜修复要求：把第 946 行的制表符改为反斜杠，即写成 `e^{\text{sink}-m}`；并全页搜索制表符 + `ext{` 确认无同类残余。｜修复：第 946 行的 TAB 已还原为反斜杠，写成 e^{\text{sink}-m}；全页制表符计数为 0，KaTeX 节点数从 59 升至 69（该公式恢复渲染）｜复验：已复跑 validate.py 通过并核对修改位置｜

- [重要·技术] 来源与范围说明「外部数字与实验条件」`[N1]`（第 1040 行）：写作「Llama-2-13B，PG19 首本书 65K token「0+1024」困惑度 5158.07」，给出的配置记号解释是「x+y 表示 x 个初始位置 + y 个近期位置」，即 1024 为窗口大小。但该表同组实测中 Llama-2-7B 的窗口长度为 4096（配置记号 `0+4096`、`4+4092`、`8+4088`），说明论文的 `1024` 是该实验设定的窗口取值、而非固定窗口大小；第 4.2 节正文与 `overview.html` 又都用 `W=128` 作为「窗口大小」的同位语，读者容易把 1024 误当作模型的窗口大小。｜引文依据：StreamingLLM Table 1 说明「Cache config x+y denotes adding x initial tokens with y recent tokens. Perplexities are measured on the first book (65K tokens) in the PG19 test set.」，表体「0 + 1024 (Window) 5158.07 / 4 + 1020 5.40 / 4"\n"+1020 5.60」；Table 2（Llama-2-7B 组）「Cache Config 0+4096 1+4095 2+4094 4+4092 8+4088」，其中 `0+4096 3359.95`｜修复要求：在 `[N1]` 的「x+y 表示 x 个初始位置 + y 个近期位置」后补一句限定，说明该表的 1024/2048/4096 是各实验的窗口取值（同一模型族在不同设置下窗口长度不同），不是模型固有的窗口大小；不要改动 5158.07 / 5.40 / 5.60 与对应配置。｜修复：在 [N1] 补限定句：1024/2048/4096 是该表各次实验设定的窗口长度（Table 2 的 Llama-2-7B 组用 4096），不是模型固有的窗口大小；数值 5158.07 / 5.40 / 5.60 与配置未改动｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·格式] 第 904–905 行：`.attn_sink` 参数说明列表前残留一个空的隐藏列表 `<ul style="display:none"></ul>`。它不显示也无内容，属于写作过程残留，降低可维护性。｜引文依据：不适用｜修复要求：删除第 904–905 行的空 `<ul>`。｜修复：已删除残留的空隐藏列表 <ul style="display:none"></ul>｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·格式] 第 9–10 行：`<meta name="dojo:topics" content="注意力机制">` 与 `<meta name="dojo:tag" content="注意力机制">` 取值相同，`dojo:tag` 未提供额外检索维度。｜引文依据：不适用｜修复要求：把 `dojo:tag` 改为能区分本页与同主题页面的词（如「缓存淘汰」「流式推理」之一），或说明保留同值的理由；`dojo:topics` 不动（校验已通过，无证据表明该取值在词表外）。｜修复：dojo:tag 改为「缓存淘汰」，与 dojo:topics「注意力机制」区分开｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·技术] `[N3]`（第 1042 行）：条目写「160M 参数模型、PG19 首样本」，未给实验条件，也未纳入「简化条件及其限制」的登记范围；同页 `[N2]` 则写明「PG19 拼接 400K token」，两条体例不一致。｜引文依据：StreamingLLM Table 3 说明「Perplexity is evaluated on the first sample in the PG19 test set.」；§4.2「we trained two language models, each with 160 million parameters, under identical conditions… Our experiments employed the Pythia-160M codebase and followed its training recipe.」；表体「Learnable Sink 1235 18.01 18.01 18.02 / Vanilla 27.87 18.49 18.05 18.05」｜修复要求：在 `[N3]` 补上来源＝Pythia-160M 代码库与训练配方（§4.2）；或在「简化条件及其限制」中登记「Table 3 的 160M 模型为 Pythia-160M 配方训练，未读训练细节」。｜修复：[N3] 补来源与实验条件（Pythia-160M 代码库与训练配方、PG19 首个样本），并在正文第 3 章形态二处补入该对照的引用｜复验：已复跑 validate.py 通过并核对修改位置｜

## 已核对项（第 2 轮逐条核对通过，引文依据留档）

- `[C1]` §3.1 与 Figure 2 两句引文与页面表述一致。来源原文：「We find that, beyond the bottom two layers, the model consistently focuses on the initial tokens across all layers and heads.」；「a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task」；换行符实验「the model still significantly emphasizes these initial linebreak tokens. Furthermore, reintroducing them restores the language modeling perplexity to levels comparable to having the original initial tokens. This suggests that the absolute position of the starting tokens, rather than their semantic value, holds greater significance.」（该句在 §3.1 正文，不在 Table 1 说明，页面归于「同节 Table 1 及其说明」为轻微归类偏差，不影响论断，不单列问题）
- `[C2]` 三处引文逐字一致。原文：「The nature of the SoftMax function (Equation 1) prevents all attended tokens from having zero values… Consequently, the model tends to dump unnecessary attention values to specific tokens.」；「removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function (Equation 1) in attention computation. This alteration leads to a significant shift in the distribution of attention scores…」；「Due to the sequential nature of autoregressive language modeling, initial tokens are visible to all subsequent tokens, while later tokens are only visible to a limited set of subsequent tokens. As a result, initial tokens are more easily trained to serve as attention sinks, capturing unnecessary attention.」
- `[C3]` 三处引文一致。原文：「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV to anchor the attention computation and stabilize the model's performance.」；「Introducing four initial tokens generally suffices; further additions have diminishing returns.」；「When determining the relative distance and adding positional information to tokens, StreamingLLM focuses on positions within the cache rather than those in the original text. This distinction is crucial for StreamingLLM's performance.」。注：前两句在 §1 引言、第三句在 §3.2 正文（§3.2 正文对应句为「we reintroduce a few starting tokens' KV in the attention computation」），页面统一标注「§3.2 与 Table 1/2」，属可接受的归类，不单列问题。
- `[C4]` 代码与 checkpoint 头全部核对通过：`model.py:639`、`kernel.py:332`、`kernel.py:382-383`（`sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`，加在累积循环之后）、43 张量 / $\{(64,)\}$ / F32；页面「全部 40 个主干层与 3 个 MTP 层，每层形状 $[64]$——即每个注意力头一个 fp32 标量」与 `config.json` 的 `n_layers = 40`、`n_heads = 64` 一致。
- `[C5]` 解析解三档、整行无效输出全零，均与 `verify_sink.out` 逐元素吻合；`sink=-∞` 时 `denom = 2 + 0`，与页面 $(v_1{+}v_2)/2$ 一致；跨度不可复跑，原因已记入机械验证结果。
- `[C6]`/`[F2]` 份额公式与第 4.2 节代码块实跑一致（0.78% / 5.46% / 53.69% / 95.88%）；`W=128` 与 `config.json` 的 `window_size = 128` 一致。
- `[C7]` 附录 A 原文逐字一致：「While StreamingLLM improves the efficiency of LLMs in streaming contexts, it does not extend the models' context window or enhance their long-term memory capabilities.」（该附录在论文目录中名为「A Discussions」，页面标注「附录 A」，编号可定位，不单列问题）
- `[F1]` 与 `kernel.py:310-387` 的分母构造一致（行最大值 `T.fill(scores_max, -1e30)` 只由可见槽位 `T.reduce_max(acc_s, scores_max, dim=1, clear=False)` 更新，sink 项在循环后加入），且 `kernel.py:352-354` 注释「a row with no valid index (all -1) would otherwise produce exp(-inf - (-inf)) = NaN. With a finite bound such rows yield an all-zero output」直接支持页面「行最大值被替换为一个极小的有限值…最终输出全零」。
- `[N1]`/`[N2]` 数值全部一致：`0+1024 5158.07` / `4+1020 5.40` / `4"\n"+1020 5.60`；`Falcon-7B 17.90 / MPT-7B 460.29 / Pythia-12B 21.62`、`Llama-2-7B 3359.95`。
- `[N5]`/`overview.html` 的 0.78% / 53.69% 与实跑一致；`overview.html`「真实 checkpoint 中该参数覆盖 40 个主干层与 3 个 MTP 层，每层每头一个标量」与 `[C4]` 一致。
- 公式与符号：$o$、$q$、$k_t$、$v_t$、$d$、$\mathcal{T}$、$\text{sink}$、$m$、$W$、$\text{share}$ 全页写法一致（第 896 行与第 970 行两处 $o$ 公式对同名符号复用未见冲突）；无公式自编号，引用论文时使用 Eq./Table 编号，符合规范。
- 问题块：页面级「核心问题」4 条（3–5 条范围内）、4 个章节均为「本章问题」，命名符合规范；12 个章节级问题均有 `解答：` 折叠块，核心问题答案末尾均指明章节（「完整观察与实验见第 1 章」「完整推导见第 2 章」「完整机制见第 3 章」「完整边界见第 4 章」）。
- 学习目标闭环：4 条核心问题分别由第 1–4 章回答，无悬空问题。
- 折叠块独立性：删除全部 `details` 后，正文仍能给出「现象—成因—两种形态—边界」完整结论（核心定义、$o$ 公式、份额公式、三条判断均在正文）；`展开：` 折叠块只补解析解与边界，符合 A8。
- 简化条件：4 条已写明简化内容与不能推出的结论（份额表等权假设、解析解为单头 4 维构造、困惑度按原值登记不比较模型质量、不评估净影响）；除 `[N3]` 外各条体例一致。
- 类比边界：「零钱罐」类比在「辅助解释与类比边界」中已列出两条失效边界（不携带信息、不提供长程记忆），符合 A10。
- 术语首用：softmax（第 839–855 行给出定义式与「分母为归一化因子」的解释）、query/key/value（第 844–846 行逐项定义）、因果掩码（第 877 行在问题解答内解释「位置 $i$ 的 query 只能看到位置 $\le i$ 的内容」）、MTP 层（第 778 行首次出现且与「主干层」并置对比，可由上下文界定）。未发现未解释即使用的术语。
- 格式一致性：h1 为「注意力汇聚点（Attention Sink）：被注意力倾倒的位置，以及为什么不能丢掉它」，符合 `概念名（英文缩写）：核心作用或结论`；h2「1.」–「4.」连续编号，「来源与范围说明」不编号；h3「3.1」「3.2」「4.1」–「4.3」编号正确；前置 section 顺序为 reading-time → blockquote.meta → callout（blue，1 个）→ learning-goals → misconceptions → 正文；`summary` 前缀仅用 `补充：`/`展开：`/`代码：`/`解答：`；未生成概念页无「（待生成）」占位；`overview.html` 与 `index.html` 相互链接（`overview.html` 第 38 行、`index.html` 第 727 行）。
- 图与图示：页面无结构图（无内联 SVG、无 `.diagram`），仅 `div.table-scroll` 表格；`<pre>`/`<code>` 仅用于 4.2 节 Python 与预期输出，是规范的代码块而非字符画图，不触发「等宽字符框线图」禁令。
- `<head>` 元信息：`description`（纯文本）、`dojo:summary`（含可渲染 `$...$`）、`dojo:type=concept`、`dojo:topics`、`dojo:tag` 齐备，validate.py 通过。

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 3
- 处置：修复。技术层结论（现象、成因、两种形态、份额公式与份额数值、边界）经第 2 轮逐条核对全部与来源一致，代码块实跑与页面预期输出一致，无阻断问题；4 条重要问题集中在第 3.2 节「本章问题」解答（1 条 LaTeX 渲染缺陷、1 条推理跳步）与两处口径未闭合（份额覆盖区间、PG19 Table 1 的窗口取值），均可就地修复且不改变核心结论，无需返回规划。修复后需重跑 `validate.py` 与第 4.2 节代码块复验。
