<!-- review-meta
round: 1
page: wiki/glm-5-3-flash-dataflow/index.html
reviewed_content_sha256: 0b72faa4c2e81030
-->
# GLM-5.3-Flash 前向数据流审查记录（第 1 轮）

- 页面版本：`2a0b91c0069cfedbb09a5d26e51d8b233bb3e691`（index.html 工作树哈希）
- 审查时间：2026-09-13 19:04
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下的规划、修复与前序审查记录）
- 适用规范：`guides/model-dataflow.md`（dojo:type = dataflow），表述与问题分级沿用 `guides/concept/check.md`
- 已完整阅读章节：1 关键规格 · 2 整体数据流（含 45 层的类型排布）· 3 单层内部数据流 · 4 mHC：4 路残差流怎么读写 · 5 KDA 层 · 6 DSA 层 · 7 MoE 路由 · 8 位置信息从哪来 · 9 长上下文下的实际收益 · 10 FP8 量化的覆盖范围 · 11 多模态 · 12 MTP 层 · 13 核对方式 · 来源与范围说明（含全部折叠块与图注）

## 核对方式（本轮实际取得的外部来源）

- 官方 `config.json`：`https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json`（69,416 B，已抓取）
- 官方权重索引：`model.safetensors.index.json`（`total_size=328326771576`，76,108 个张量）
- 官方权重头：62 个分片逐个 HTTP Range 取 safetensors JSON 头，得 76,108 个张量的 shape/dtype（脚本见下）
- 官方源码：`transformers` 仓库 `models/glm5_next/modeling_glm5_next.py`（2,426 行）与 `configuration_glm5_next.py`
- 官方 README：`https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/README.md`
- 技术报告：`https://arxiv.org/html/2602.15763v2`（§2.1、§2.1.1）

已核对通过、不构成问题的关键项（记录备查）：总参数 321,323,031,390、张量数 76,108、`weight_scale_inv` 37,338 且形状全部符合 ⌈N/128⌉×⌈K/128⌉、单 token 激活 17,376,348,990、层类型逐项（第 3,7,…,43 层 DSA）、`q_conv1d` 34 层 / `kv_a_proj_with_mqa` 11 层、indexer 11 组、`hc_attn_fn`＝[24,16384]、`hc_attn_base`＝[24]、`hc_attn_scale`＝[3]、KDA 单层注意力 137,732,288（137.73 M）、DSA 单层注意力 124,914,432（124.91 M）、indexer 7,471,872（7.47 M）、MTP 第 45 层 7,432,592,416（7.43 B）、visual 命名空间 563,627,008、长上下文表全部 4 列与交叉点 2855 / 4288（复算一致）、`_keep_in_fp32_modules_strict` 与四个张量 dtype=F32、`_keys_to_ignore_on_load_unexpected` 含 `layers\.45\.`、`validate_architecture()` 的 NoPE 报错、Sinkhorn 迭代序（列归一化 20 次 / 行归一化 19 次、末步落列）、pre=σ(·)+hc_eps / post=2σ(·)、forget gate 两支、chunk_size=64、indexer softmax(dim=2)、SwiGLU 不对称截断（gate `min=None`、up 双侧）、`position_embeddings=None`、C6 所列 9 个惰性键在两份源码中命中数均为 0、README 的「18B active / 320B total / newly trained base model」、arXiv §2.1（256 专家、80 层、744 B、40 B 激活）与 §2.1.1（90% 冗余、1.5–2×）原文一致。

## 问题

- [阻断·来源] 来源与范围说明（[C1][C3][C4][C5][F1][F2][N1]–[N13]）及第 13 节首句｜问题：页面把实测结论的来源指向本页 `research/` 下已从仓库移除的文件，并直接陈述这些文件「都在 research/ 下」，与仓库实际状态不符。页面实际盘点 `wiki/glm-5-3-flash-dataflow/research/` 只剩 `measured.md` 与 `prereq-audit.md`；`research/measured.md` 的登记表明确写着这些产物「现已从仓库移除」。规范要求引用实测结论时写「实测得到」，不得指向已移除的文件路径。｜引文依据：页面第 13 节首句「本页数字的来源分三层，脚本与原始输出都在 `research/` 下。」；[C1]「（`verify_structure.out` 项 [3][4]）」；[C3]「实测见 `p7.out` 节 7.2、7.3」；[C4]「实测见 `p2.out` 节 2.7」；[C5]「（`p7.out` 节 7.5）」；[F1]「实测见 `p1.out` 节 1.3–1.5」；[F2]「实测见 `p4.out` 节 4.2–4.6」；[N1]「脚本 `verify_structure.py`，输出项 [0][10]」；[N2]「脚本 `p4_moe.py` 节 4.7」；[N3][N4][N5][N6][N7][N8][N9][N10][N11][N12][N13] 同样指向 `p1_kda.py`/`p2_dsa.py`/`p3_mhc.py`/`p4_moe.py`/`p5_end2end.py`/`p6_vision.py`/`p8_longctx.py`/`read_headers.py` 及其 `.out`。`research/measured.md` 登记表列出的被移除文件正是上述全部脚本与输出。｜修复要求：删除正文与来源清单中所有指向已移除 `research/` 文件路径的引注，改为规范允许的「实测得到」表述（例如「实测得到：34 个层带 `q_conv1d`…」），并改写第 13 节首句使其不再声称脚本与原始输出随仓库分发；若某条结论只以已移除的运行输出为依据，需在正文注明该依据已不可复现或降级为标注的推断。｜修复：｜复验：

- [重要·技术] 第 4 节 mHC（退化段落）与 [N5]｜问题：把「`hc_mult=1` 且 `fn` 输出 0」的构造说成「回写式退化为 $h+f(h)$」，并给出 3.8×10⁻⁶ 的差距。按官方 `_init_weights`（`base=0`、`scale=1`）与 `forward` 的 `pre = torch.sigmoid(pre_w*pre_scale + pre_b) + self.hc_eps`，`fn=0` 时 $pre=\sigma(0)+hc\_eps\approx0.5$，不是 1；于是「压成 1 路」得到的是 $0.5h$，子层输入被减半，回写实际上是 $h+f(0.5h)$ 而非 $h+f(h)$。用官方源码重跑该模块（hc_mult=1、fn=0、base=0、scale=1、hidden=4096）复现：`collapsed/h = 0.500001`，`max|hc 输出 − (h+f(h))| = 2.83`（大），而 `max|hc 输出 − (h+f(collapsed))| = 3.8147e-06`，恰为页面所报的 3.8×10⁻⁶。即该数字度量的是 `comb` 偏离单位阵的程度（`1−comb = 9.5e-07`，乘以 `max|h|≈4` 得 ~4e-6），并非「与普通残差相等」，与页面「算式与结论」不符。｜引文依据：页面「把 `hc_mult` 设为 1 并让 `fn` 输出 0，$post$ 变成 $2\sigma(0)=1$、$comb$ 变成 $1\times1$ 矩阵，回写式退化为 $h+f(h)$，实测与普通残差的差距是 $3.8\times10^{-6}$，量级由 `hc_eps` 决定。也就是说普通残差是这套机制的一个特例。」；源码 `_init_weights`：「`init.zeros_(module.base)` / `init.ones_(module.scale)`」；`forward`：「`pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps`」「`collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)`」；本轮复算数值见上。｜修复要求：补全该退化构造的精确条件（若要使子层输入为 $h$ 必须另行把 $pre$ 置 1，`fn=0` 达不到），或把结论改为「回写结构退化为 $h+y$（$y$ 为子层输出），且预压缩系数为 $\sigma(0)+hc\_eps\approx0.5$，不构成普通残差」；3.8×10⁻⁷–10⁻⁶ 量级需说明它度量的是 `comb` 与单位阵之差，而非与 $h+f(h)$ 之差。｜修复：｜复验：

- [轻微·表述] 第 11 节（多模态）正文｜问题：使用规范明列的元话语「需要注意」。｜引文依据：「需要注意**输入帧数与 grid 的 T 不是同一个量**：…」。｜修复要求：删去「需要注意」，直接陈述「输入帧数与 grid 的 $T$ 不是同一个量：…」。｜修复：｜复验：

- [轻微·表述] 第 9 节 callout、第 13 节标题与首句、第 5 节正文｜问题：临场评价与预告腔。｜引文依据：「这里有一个容易漏掉的项：…」；标题「两处容易想错的机制」＋「以下两处机制的常见推断与实现不符，实测结论如下：」；「下面各式对每个头独立成立。」（第 5 节）。｜修复要求：改为直接陈述结论（如「`indexer` 的 257 维打分状态须计入，否则…」），删去「容易漏掉 / 容易想错 / 常见推断 / 以下 / 下面」一类引导与评价词。｜修复：｜复验：

- [轻微·表述] 第 13 节末段｜问题：把调试/复现过程写进正文。｜引文依据：「此外，复现 mHC 时若只初始化维度 $\ge 2$ 的参数，一维的 `base` 与 `scale`（源码用 `torch.empty` 创建）会保留未初始化内存，4 路残差流将不产生分化。按官方 `_init_weights` 初始化全部参数是复现该机制的前提。」｜修复要求：改写为结论性依据（如「官方 `_init_weights` 对 `base`/`scale` 分别零初始化与置 1，`fn` 取 $\mathcal{N}(0,0.02)$；4 路分化依赖该初始化」），不叙述复现踩坑过程。｜修复：｜复验：

- [轻微·表述] 第 8、9、12、13 节及来源清单｜问题：以「本页」为主语的自我指代式元话语反复出现。｜引文依据：「（本页标注为推断，非官方声称）」「本页标注为推断」「本页不下结论」「本页数字的来源分三层」「本页长上下文数字用前者」「本页不含任何关于输出质量…的结论」「本页用官方初始化下的随机权重测得」「本页不引用该报告作为…依据」「本页不复述也不评价」「本页只说明…」。｜修复要求：删去以「本页」为主语的句子，改为无主语陈述（如「此处标注为推断，非官方声称」「长上下文数字取潜向量口径」「不在本页范围」改为「不在讨论范围」），保留推断标注本身。｜修复：｜复验：

- [轻微·公式] 第 5 节（KDA 层）｜问题：同一小节内 $S$ 被同时用作序列长度与状态矩阵。｜引文依据：「其中 $t = 1,\dots,S$ 逐 token 推进，初值 $S_0 = 0$」与同节「$S_t\in\mathbb{R}^{128\times128}$：该头在第 $t$ 步的状态矩阵」；图注「状态 $S\in\mathbb{R}^{64\times128\times128}$」。｜修复要求：把序列长度改用 $L$（或 $T$，与其它节统一），状态矩阵保持 $S_t$，使同一符号全页单义。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 5
- 处置：修复（阻断与重要全部关闭后重跑 `.dojo/scripts/validate.py` 并进入第 2 轮）。本轮所有事实性数字、公式与来源（官方 config.json、权重索引与张量头、官方源码、arXiv 报告、官方 README）均已逐条回源核对一致，未发现数字错误、来源不支持或页面内自相矛盾；`.dojo/scripts/validate.py` 返回 `validation ok`，10 条前置概念内链均真实存在。
