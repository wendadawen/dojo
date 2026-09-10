# MLA 审查记录（第 3 轮）

- 页面版本：ed789198e398965df701b6daf1eda8b4812b5ec6（`git hash-object wiki/mla/index.html`；overview.html = 098e2b4e189efe63d35650c101cdb4d4e02f89b4）
- 审查时间：2026-09-10 21:46 CST
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查；未读取 `wiki/mla/research/` 下任何文件）
- 已完整阅读章节：引言（含 reading-time、blockquote.meta）→ 核心问题（5 条问题与解答折叠块）→ 1. MLA 压缩了什么——KV 联合压缩的核心机制（含「补充：为什么 $\mathbf{k}_t^C$ 的形状是 $\mathbb{R}^{d_h n_h}$」折叠块、本章问题）→ 2. 推理时不重建 K/V——矩阵吸收（含「补充：$W^{UV}$ 吸进 $W^O$ 的完整推导」折叠块、推理路径对比图、本章问题）→ 3. 为什么 RoPE 要解耦——位置编码与矩阵吸收的冲突（含「补充：$R_t W^{UK}$ 不可吸收的反例代数」「补充：为什么 $\mathbf{k}_t^R$ 所有头共享而 $\mathbf{q}_t^R$ 每头一份」、本章问题）→ 4. KV cache 到底减少了多少——与 MHA / GQA / MQA 对照（含黄色 callout、紫色 callout、本章问题）→ 5. K3 的 Gated MLA——NoPE 与 full-rank output gate（5.1、5.2、「补充：为什么 $W^g$ 满秩重要」、本章问题）→ 回顾全文学习目标 → 来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件、构造示例、辅助解释与类比边界、简化条件及其限制）；另完整阅读 overview.html。所有 `<details>` 折叠块均已展开阅读。

## 机械验证结果

命令与结果：

```
$ /usr/bin/python3 .dojo/scripts/validate.py wiki/mla/index.html
validation ok: wiki/mla/index.html            # exit 0
$ /usr/bin/python3 .dojo/scripts/validate.py wiki/mla/overview.html
validation ok: wiki/mla/overview.html         # exit 0
```

- 引用编号双向闭合：正文引用集合 = {C1–C8, F1–F6, N1, N2, N4–N7}；来源章节定义集合 = {C1–C8, F1–F6, N1–N7}。**正文 − 来源 = ∅；来源 − 正文 = {N3}**（N3 未在正文引用，见问题 9）。
- 相邻双上标 `<sup>[X][Y]</sup>`：0 处（未出现需合并的相邻上标）。但存在把位置说明混入引用编号的写法 `<sup>[F1, Eq.7]</sup>`、`<sup>[F4, Eq.18]</sup>`（见问题 8）。
- Unicode 数学字符：U+2212（减号）0 处；**U+00D7（乘号）index.html 3 处、overview.html 3 处**（见问题 7）；U+2192（→）仅出现在 `div.diagram` 流程图中，属 content-examples A5 认可的流程箭头用法。
- TAB：index.html 0 个，overview.html 0 个。
- 占位符（待生成/TODO/TBD/XXX/占位/FIXME）：0 处。
- `<head>` 五项元数据（index.html）：`description`、`dojo:summary`、`dojo:type=concept`、`dojo:topics=注意力机制`、`dojo:tag` 均存在；`dojo:topics` 取值「注意力机制」在 AGENTS.md 固定大类词表内。overview.html 无这五项，但抽查 wiki/rope、wiki/kv-cache、wiki/kimi-k3、wiki/mqa-gqa 的 overview.html 同样无，属仓库既有约定，不作为问题。
- overview 与 index 互链：index.html 顶部「快速阅读」→ `overview.html`；overview.html 导航「深度教学 →」→ `index.html`。双向有效。
- 前置概念链接：`../../wiki/mqa-gqa/index.html`、`../../wiki/low-rank-projection/index.html`、`../../wiki/rope/index.html` 三处目标文件均存在，无 404。
- 可运行代码块：index.html **不含任何 `<pre>`／代码块**，无「预期输出」需逐字符比对，该项不适用；页面引用的官方源码片段改为静态核对（见下），手算数值全部复算通过：$2\times128\times128\times60=1{,}966{,}080$；$\times2$ 字节 $=3.75$ MiB/token；$\times131{,}072=480$ GiB；$576/32{,}768=1/56.9$（减少 98.24%）；$512/16{,}384=1/32$；$576/24{,}576=1/42.7$；$\mathrm{sigmoid}(1)=0.7311$、$\mathrm{sigmoid}(-1)=0.2689$、$0.2689\times3=0.8067$；$1/\sqrt6=0.4082$。

## 问题

- [重要·技术] 第 4 章黄色 callout（行 961）、第 4 章本章问题第 3 题解答（行 999）、来源 N3（行 1111）：页面三处把 DeepSeek 67B 写成「65 层」，与实际配置不符（实为 95 层）｜引文依据：DeepSeek LLM 官方 config.json `"num_hidden_layers": 95, "hidden_size": 8192, "num_attention_heads": 64, "num_key_value_heads": 8`（huggingface.co/deepseek-ai/deepseek-llm-67b-chat/config.json）；DeepSeek LLM 论文（arXiv:2401.02954）Table：`67B & 95 & 8192 & 64 & 8`。DeepSeek-V2 论文全文未出现 DeepSeek 67B 的层数｜修复要求：将三处「65 层」改为「95 层」；若需保留架构描述，同时把依据补为 DeepSeek LLM 报告（arXiv:2401.02954）或官方 config.json，不得仅挂在 DeepSeek-V2 论文下（该论文不含此数字）｜修复：将第 4 章黄色 callout（行 961）、第 4 章本章问题第 3 题解答（行 999）与来源 N3（行 1111）三处「65 层」全部改为「95 层」；并把依据从 DeepSeek-V2 论文改挂到 DeepSeek LLM 报告（arXiv:2401.02954）与官方 deepseek-llm-67b-chat config.json（num_hidden_layers=95, hidden_size=8192, num_attention_heads=64, num_key_value_heads=8），N3 条目重写为该架构来源并在黄色 callout 以 <sup>[C8, N3]</sup> 就近引用。｜复验：index.html 中「65 层」计数 0，三处均为「95 层」；官方 config 实测返回 num_hidden_layers: 95；N3 定义与正文引用双向闭合。validate.py exit 0。
- [重要·技术] 第 4 章紫色 callout（行 976）与来源 N7（行 1115）：「Moonshot 团队确认 K3 加回 RoPE 后效果无明显变化（tahou.com 报道）」无法核对，且 K3 报告 §2.1.2 无此陈述｜引文依据：可核对的同源文本为 besthub.dev《Inside K3》原文 "RoPE can be re-added without noticeable effect, so it is omitted for simplicity."（社区分析，非 Moonshot 官方声明）；检索未找到 tahou.com 的对应报道；K3 报告 §2.1.2 原文仅有 "applies No Position Encoding (NoPE) to all MLA layers" 与 "This separation also avoids modifying positional-encoding parameters when extending the context length"｜修复要求：删除「Moonshot 团队确认」这一主体归因，或删除整句；如需保留该设计取舍，改为「社区分析（besthub.dev《Inside K3》）认为」并同步改写 N7，删除或替换无法访问的 tahou.com 出处｜修复：删除紫色 callout（行 976）中「Moonshot 团队确认 K3 加回 RoPE 后效果无明显变化」的主体归因及其 tahou.com 出处，改写为「社区分析（besthub.dev《Inside K3》）」并引其可核对原文 "RoPE can be re-added without noticeable effect, so it is omitted for simplicity."；来源 N7（行 1115）同步改写并注明「以上为社区分析，非 Moonshot 官方声明」，删除 tahou.com。｜复验：index.html 中「Moonshot 团队确认」「tahou」计数均为 0；K3 报告 §2.1.2 无该陈述，正文现只归于社区分析并保留 B 端可核对引文。validate.py exit 0。
- [轻微·技术] 第 4 章黄色 callout（行 961）与来源 C8（行 1094）：C8 标注为「DeepSeek-V2 摘要与 §3.2.3 Table 5」，但 §3.2.3 内无 Table 5；Table 5 是第 4 章 AlignBench 排行榜，与 93.3% baseline 无关｜引文依据：Table 5 标题 "Table 5: AlignBench leaderboard rated by GPT-4-0613. Models are ranked in descending order based on the overall score."；§3.2.3（Training and Inference Efficiency）只给出 42.5% 训练成本与 5.76× 吞吐，对 KV cache 仅有 "requires significantly less KV cache than DeepSeek 67B" 的定性表述，无 93.3% 数值。93.3% 数值仅见于摘要与引言："Compared with DeepSeek 67B … reduces the KV cache by 93.3%"｜修复要求：把 C8 的位置改为「摘要与引言」，删除 `Table 5`；93.3% 的数值依据固定为摘要/引言原文｜修复：C8（行 1094）位置由「DeepSeek-V2 摘要与 §3.2.3 Table 5」改为「DeepSeek-V2 摘要与引言」，删去 §3.2.3 内不存在的 Table 5，并附原文 "Compared with DeepSeek 67B … reduces the KV cache by 93.3%"。｜复验：index.html 中「Table 5」计数 0；93.3% 的依据现固定为摘要/引言原文。validate.py exit 0。
- [轻微·技术] 来源 N1（行 1109）与 N2（行 1110）：标注位置「§2.1.2 末段 + §2.1.4」不含页面列出的具体超参数（$d=5120$、$n_h=128$、$d_h=128$、$l=60$、$d_c=512$、$d_c'=1536$、$d_h^R=64$）｜引文依据：上述数值出自 §3.1.2 Model Hyper-Parameters："We set the number of Transformer layers to 60 and the hidden dimension to 5120 … we set the number of attention heads $n_h$ to 128 and the per-head dimension $d_h$ to 128. The KV compression dimension $d_c$ is set to 512, and the query compression dimension $d_c'$ is set to 1536 … we set the per-head dimension $d_h^R$ to 64."；§2.1.4 Table 1 只给出 $d_c=4d_h$、$d_h^R=d_h/2$ 的比例关系，无数值｜修复要求：N1、N2 的位置标注改为「§3.1.2 + §2.1.4 Table 1」｜修复：N1（行 1109）位置由「§2.1.2 末段 + §2.1.4」改为「§3.1.2 + §2.1.4 Table 1」；N2（行 1110）由「§2.1.4 Table 1」改为「§3.1.2 + §2.1.4 Table 1」。所列超参数数值实出自 DeepSeek-V2 §3.1.2。｜复验：与 wiki/low-rank-projection 页所引 §3.1.2 原文（"we set the number of attention heads n_h to 128 … The KV compression dimension d_c is set to 512"）一致；§2.1.4 Table 1 仅给比例关系，标注已作区分。validate.py exit 0。
- [轻微·技术] 5.2 节正文（行 1032）与「补充：为什么 $W^g$ 满秩重要」（行 1053）：门向量维度写成 $d$，与同段落中 $d$ 表示模型隐藏维度的用法冲突｜引文依据：同页行 1031 定义 $W^g\in\mathbb{R}^{d_h n_h\times d}$、$\mathbf{x}_t\in\mathbb{R}^d$；K3 报告 §2.1.2 Eq.(7) 的 gate 作用在 $\tilde{\bm{o}}_t$ 上，其通道数即 attention 输出维度：$\bm{y}_t=\mathbf{W}_o[\operatorname{Sigmoid}(\mathbf{W}_g\bm{x}_t)\odot\tilde{\bm{o}}_t]$；官方实现 `self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim)` 表明门输出维度为 $d_h n_h$ 而非 $d$｜修复要求：行 1032 的 $\mathrm{Sigmoid}(W^g\mathbf{x}_t)\in(0,1)^d$ 改为 $\in(0,1)^{d_h n_h}$；行 1053 的「秩 $r<d$」「剩余 $d-r$ 个通道」改为「秩 $r<d_h n_h$」「剩余 $d_h n_h-r$ 个通道」｜修复：5.2 节正文（行 1032）$\mathrm{Sigmoid}(W^g \mathbf{x}_t) \in (0,1)^d$ 改为 $\in (0,1)^{d_h n_h}$；补充折叠块（行 1053）「秩 $r<d$」改为「秩 $r<d_h n_h$」、「剩余 $d-r$ 个通道」改为「剩余 $d_h n_h-r$ 个通道」。｜复验：与 K3 §2.1.2 Eq.(7) gate 作用于 $	ilde{m{o}}_t$（通道数即 $d_h n_h$）及官方 g_proj = nn.Linear(hidden_size, num_heads*v_head_dim) 一致，消除与同段 $d$（隐藏维度）的冲突。validate.py exit 0。
- [轻微·可读性] 核心问题第 5 条解答（行 708）、第 4 章（行 972）、5.1 节（行 1016）：缩略语 KDA 首次出现于核心问题解答且未解释，全页未出现全称；NoPE 同样在核心问题（行 705）首次出现而无解释｜引文依据：不适用（K3 报告 §2.1.1 标题为 "Kimi Delta Attention (KDA)"，可作全称来源）｜修复要求：在 KDA 首次出现处（行 708）改为「KDA（Kimi Delta Attention，线性注意力）」或注明「见 5.1 节」；在核心问题出现 NoPE 处补一句「NoPE：不施加显式位置编码」｜修复：核心问题第 5 条（行 705）NoPE 首现处补「（不施加显式位置编码）」；其解答（行 708）KDA 首现处补全称「（Kimi Delta Attention，一种线性注意力）」。｜复验：KDA 全称取自 K3 §2.1.1 标题 "Kimi Delta Attention (KDA)"；全页 KDA/NoPE 首现均有解释，后续出现不再需要额外说明。validate.py exit 0。
- [轻微·格式] 引言（行 669，2 处）与来源 N3（行 1111，1 处）；overview.html 第 2 节（行 52，2 处）与第 2 节第 4 条（行 55，1 处）：U+00D7（×）直接出现在正文，未经 KaTeX 渲染｜引文依据：不适用｜修复要求：把「128 头 × 128 维 × 60 层」改写为 $128\times128\times60$（或中文「128 头、128 维、60 层」），「头数 × 头维」改写为「头数与头维之积」，「5.76× 吞吐」改写为「5.76 倍吞吐」；改后重跑 validate.py 确认仍返回成功｜修复：引言（行 669）「128 头 × 128 维 × 60 层」改为 $128 \times 128 \times 60$（128 头、128 维、60 层）；来源 N3（行 1111）重写后不再含 U+00D7；overview.html 第 2 节（行 52）同改写为 $128\times128\times60$（128 头、128 维、60 层），第 2 节第 4 条（行 55）「头数 × 头维」改为「头数与头维之积」。｜复验：U+00D7 在 index.html 与 overview.html 计数均为 0；两文件 validate.py 均 exit 0。
- [轻微·格式] 第 1 章（行 722）`<sup>[F1, Eq.7]</sup>`、第 3 章（行 883）`<sup>[F4, Eq.18]</sup>`：把位置说明混入引用编号，破坏 style-guide §6「上标引用与来源章节双向对应」的编号格式｜引文依据：不适用｜修复要求：改为 `<sup>[F1]</sup>`（位置说明写在正文句中，如「即 F1 所指 Eq.(7)」）与 `<sup>[F4]</sup>`，或拆成 `<sup>[F1]</sup>（Eq.7）` 的写法；改后引用编号集合应保持双向闭合｜修复：第 1 章（行 722）<sup>[F1, Eq.7]</sup> 改为「（即 F1 所指的 Eq.(7)）<sup>[F1]</sup>」；第 3 章（行 883）<sup>[F4, Eq.18]</sup> 改为「（因为拼接后向量维度变了；即 F4 所指的 Eq.(18)）<sup>[F4]</sup>」，位置说明移入正文。｜复验：上标内仅剩纯编号，全文无 <sup>[X, Eq.*]</sup> 形式；引用编号集合仍双向闭合（无相邻双上标）。validate.py exit 0。
- [轻微·格式] 来源章节 N3（行 1111）：该编号定义后未在正文任何位置引用，来源与正文未双向对应｜引文依据：不适用｜修复要求：或在正文（如第 4 章黄色 callout）就近引用 `<sup>[N3]</sup>`，或删除 N3 条目（其只含未被正文使用的 -42.5%、5.76× 数字）｜修复：采用「重写并就近引用」方案——N3（行 1111）由未被正文使用的 -42.5%/5.76 数字改为 DeepSeek 67B 架构来源（同时服务问题 1 的层数依据），并在第 4 章黄色 callout（行 961）以 <sup>[C8, N3]</sup> 引用。｜复验：正文引用集合 = 来源定义集合 = {C1–C8, F1–F6, N1–N7}，「来源 − 正文」差集为空。validate.py exit 0。
- [轻微·格式] 来源章节 h3 标题（行 1107）：写作「外部数字与实验条件」，与 style-guide §1 的固定命名「外部数字与实验条件（N）」不一致｜引文依据：不适用｜修复要求：标题改为「外部数字与实验条件（N）」｜修复：来源章节 h3（行 1107）由「外部数字与实验条件」改为固定命名「外部数字与实验条件（N）」。｜复验：与 style-guide §1 的固定命名一致（其余 h3 亦为 x（C）/x（F）式）。validate.py exit 0。
- [轻微·格式] blockquote.meta（行 666）：只给出 DeepSeek-V2 的 arXiv 编号，未给出多处依赖的 K3 技术报告编号与官方源码出处｜引文依据：K3 官方发布信息 "Tech report: https://arxiv.org/abs/2607.24653"；N6/N7 依据的 `modeling_kimi_linear.py` 与 `config-kimi-k3.json` 均未在 blockquote 中列为来源｜修复要求：blockquote.meta 补「Kimi K3 技术报告（arXiv:2607.24653，§2.1.1–2.1.2）；Kimi-K3 官方 config（`mla_use_nope`、`mla_use_output_gate`）与 `modeling_kimi_linear.py`」｜修复：blockquote.meta（行 666）补「Kimi K3 技术报告（arXiv:2607.24653，§2.1.1–2.1.2）」、「Kimi-K3 官方 config.json（mla_use_nope、mla_use_output_gate）与官方 modeling_kimi_linear.py」，并补 DeepSeek 67B 架构依据「DeepSeek LLM 报告（arXiv:2401.02954）与官方 deepseek-llm-67b-chat config」。｜复验：arXiv:2607.24653 与 K3 官方发布信息一致（wiki/kimi-k3/research/scope.md 记录同号）；N6/N7 依赖的 config 与源码已在 meta 列为来源。validate.py exit 0。

已核对通过的来源论断（引文依据留档）：

- C1/F2（行 732、734）：DeepSeek-V2 §2.1.2 Eq.(9)(10)(11) 原文逐字一致 —— "$\mathbf{c}_t^{KV}=W^{DKV}\mathbf{h}_t$ (9) / $\mathbf{k}_t^C=W^{UK}\mathbf{c}_t^{KV}$ (10) / $\mathbf{v}_t^C=W^{UV}\mathbf{c}_t^{KV}$ (11)"，且 "where $\mathbf{c}_t^{KV}\in\mathbb{R}^{d_c}$ … $W^{UK},W^{UV}\in\mathbb{R}^{d_h n_h\times d_c}$"。
- F3（行 746）：Eq.(12)(13) 与 "$W^{DQ}\in\mathbb{R}^{d_c'\times d},W^{UQ}\in\mathbb{R}^{d_h n_h\times d_c'}$" 一致。
- F5/C4（行 798、800）：§2.1.2 末段原文 "since $W^{UK}$ can be absorbed into $W^Q$, and $W^{UV}$ can be absorbed into $W^O$, we even do not need to compute keys and values out for attention."。
- C3/F4（行 861–888）：§2.1.3 首段原文 "RoPE is incompatible with low-rank KV compression … a RoPE matrix related to the currently generating token will lie between $W^Q$ and $W^{UK}$"；Eq.(14)–(19) 逐条一致，含 Eq.(15) "$\mathbf{k}_t^R=\mathrm{RoPE}(W^{KR}\mathbf{h}_t)$"、Eq.(18) 分母 $\sqrt{d_h+d_h^R}$、$W^{QR}\in\mathbb{R}^{d_h^Rn_h\times d_c'}$、$W^{KR}\in\mathbb{R}^{d_h^R\times d}$。
- C2/F1（行 718–724）：§2.1.1 Eq.(1)–(8) 一致，Eq.(7) 逐字为 "$\mathbf{o}_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_j(\mathbf{q}_{t,i}^T\mathbf{k}_{j,i}/\sqrt{d_h})\mathbf{v}_{j,i}$"，Eq.(8) 为 $W^O$ 输出投影。
- C7/F2/N2（行 935–958）：§2.1.4 Table 1 逐行一致（MHA $2n_hd_hl$ / GQA $2n_gd_hl$ / MQA $2d_hl$ / MLA $(d_c+d_h^R)l\approx(9/2)d_hl$，能力列 Strong/Moderate/Weak/Stronger），且 "For DeepSeek-V2, $d_c$ is set to $4d_h$ and $d_h^R$ is set to $d_h/2$"。
- C5（行 1012）：K3 报告 §2.1.2 逐字 "Kimi K3 follows the hybrid design of Kimi Linear and applies No Position Encoding (NoPE) to all MLA layers. Consequently, no explicit positional encoding is applied to their queries or keys."；"The intervening KDA layers provide position-sensitive and recency-aware sequence mixing, while the MLA layers provide unrestricted global content interaction."；"avoids … retuning a RoPE frequency base or applying YaRN"。
- C6/F6（行 1024–1049）：K3 §2.1.2 Eq.(7) 逐字 "$\bm{y}_t=\mathbf{W}_o[\operatorname{Sigmoid}(\mathbf{W}_g\bm{x}_t)\odot\tilde{\bm{o}}_t]$"；"The gate projection $\mathbf{W}_g$ is full rank, matching the new parameterization used by KDA in Kimi K3."；"This gate allows each token to modulate the channels read from global attention"。§2.1.1「Full-rank gate」小节原文 "Kimi K3 changes KDA's output gate from the low-rank parameterization used by Kimi Linear to an input-dependent full-rank projection."。
- N4/N5（行 964–976）：`config-kimi-k3.json` 逐项一致 —— `q_lora_rank=1536`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`、`mla_use_nope=true`、`mla_use_output_gate=true`、`num_attention_heads=96`、`hidden_size=7168`、`num_hidden_layers=93`；`linear_attn_config.full_attn_layers` 共 24 项（4,8,…,92,93），`kda_layers` 69 项。
- N6（行 975、1114）：`modeling_kimi_linear.py` 静态核对一致 —— 行 403 `self.rotary_emb = None`；行 396 `assert self.use_nope`；行 426–428 `compressed_kv = self.kv_a_proj_with_mqa(hidden_states)` 后 `torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)`；行 435–444 `k_rot` 被 expand、拼进 `key_states` 并 `past_key_values.update(...)`，`forward` 中无 `if self.use_nope` 跳过分支；行 470–473 `g = self.g_proj(hidden_states).sigmoid(); attn_output = attn_output * g`。
- N7（行 976、1115）前半：besthub.dev《Inside K3》原文 "The extra 64-dimensional concatenation in MLA is kept to preserve compatibility with existing MLA infrastructure and avoid extra projection, without sacrificing performance."；vLLM 侧原文（vllm.models.kimi_k3.nvidia.mla）`self.rotary_emb: RotaryEmbedding | None = None`、`self.head_size = kv_lora_rank + qk_rope_head_dim`，fused kernel 文档 "The optional positions / cos_sin_cache pair enables GPT-J-style RoPE inside the epilogue. Omitting both keeps the K3 NoPE fast path."。
- 构造示例（行 752–760、819–825、890–899、1039–1049）：四组手算全部复算一致（见机械验证结果末尾）。
- 简化条件（行 1129–1133）：4 项均写明简化内容、可推出与不可推出结论，与正文一致。

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 9
- 处置：修复

check.md §5 发布条件逐条核对：

1. 三轮审查均已完成且每轮由独立审查者执行 —— 本轮（第 3 轮）由独立审查者完成；前两轮的独立性无法由本轮材料确认（未读取 `research/`），如实记录为无法确认。
2. 每条来源论断都有引文依据记录，无法核对的已删除或降级 —— **未满足**：问题 1（65 层与来源不符）、问题 2（Moonshot 团队确认的归因无法核对）尚未关闭。
3. 所有阻断和重要问题均已关闭 —— **未满足**：2 条重要问题待修复。
4. 遗留轻微问题具有明确的接受理由 —— 未满足：9 条轻微问题中，问题 3–5、6 涉及来源定位与符号一致性，建议一并修复；问题 7–11 为格式项，修复成本低。
5. 全部学习目标由正文章节完整回答 —— 满足：核心问题 5 条分别由第 1、3、4、2、5 章完整回答，各章末尾「本章问题」的 15 个问题均有解答折叠块且答案独立可读。
6. 页面级「核心问题」与每个章节「本章问题」均有解答折叠块 —— 满足：核心问题 5 条、5 个正文章节各 1 组，共 20 个 `<details>` 解答块，无只列问题未作答。
7. 数学符号全部使用 LaTeX，结构图为 HTML 或内联 SVG —— **未满足**：问题 7 的 6 处 U+00D7 尚在正文中；结构图为 HTML `div.diagram`（流程箭头用 →，符合 content-examples A5）。
8. `.dojo/scripts/validate.py` 返回成功 —— 满足（index.html 与 overview.html 均 exit 0）。
9. 可运行代码的结果与页面描述一致 —— 不适用（页面无可运行代码块）；页面手算示例与源码片段已按静态核对与复算确认一致。
10. 关键论断和数字已重新核对来源 —— 满足（本轮逐条核对，问题 1 的 65 层错误即由此发现）。
11. `<head>` 含有效纯文本 `description`、可渲染 `dojo:summary`、`dojo:type=concept`、`dojo:topics`、`dojo:tag` —— 满足（index.html；topics 取值在词表内）。
12. `overview.html` 与 `index.html` 相互链接 —— 满足。
13. 页面引用的概念链接有效，或具有明确占位 —— 满足（mqa-gqa、low-rank-projection、rope 三页均存在）。
14. 递归生成的前置概念页已完成各自质检 —— **无法确认／疑似未满足**：`wiki/mqa-gqa/research/` 与 `wiki/low-rank-projection/research/` 下只有 `review.md`、`review-2.md`，无第 3 轮记录；`wiki/rope/research/` 有 `review-3.md`。按 `wiki/mla/research/` 的命名约定（本轮为 review-3.md），推断前两者仅完成两轮。需编排者确认这三个前置页的质检轮次后再发布。

综上：本轮发现 2 条重要问题未关闭，尚不满足 §5 全部发布条件，处置为**修复**（修复问题 1、2，并建议一并处理问题 3–11 后重跑 validate.py 并复验）。
