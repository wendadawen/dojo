<!-- review-meta
round: 9
page: wiki/standard-attention/index.html
reviewed_content_sha256: ec3d3c7a6d3d7f79
-->
# 标准 Transformer 注意力审查记录（第 9 轮）

- 页面版本：4d0f2c04b771a8dc36dc1ec7a95d044c5518b7d4
- 审查时间：2026-09-14 17:15
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 规范：`guides/concept/check.md`（`dojo:type=concept`）、`guides/concept/style-guide.md`
- 已完整阅读章节（按顺序）：核心问题 / 最容易误解 / 1. 注意力要解决什么问题——从 RNN 的"逐步传递"到"直接查询" / 2. 缩放点积公式——每个符号与每一步 / 3. 为什么除以 $\sqrt{d_k}$——缩放因子的方差推导与不缩放的后果 / 4. 多头注意力——拆分子空间、拼接、参数量等价 / 5. 复杂度、瓶颈与边界——标准注意力不能做什么 / 来源与范围说明（含全部 details 折叠块、图注、表格与 callout）；同页 `overview.html` 一并通读

## 核对依据（回源与复算）

- 论文正文与公式：arXiv:1706.03762（ar5iv 全文）§3.2.1 Eq.(1) 原文 "Attention(Q,K,V) = softmax(QK^T/√d_k)V"；脚注 "the components of q and k are independent random variables with mean 0 and variance 1 … has mean 0 and variance $d_k$"；§3.2.1 "additive attention outperforms dot product attention without scaling for larger values of $d_k$"、"extremely small gradients"；§3.2.2 "$W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$"、$h=8$、$d_k=d_v=d_{\text{model}}/h=64$；§3.2.3 "masking out (setting to −∞) all values in the input of the softmax"；§3.5 "Since our model contains no recurrence and no convolution … inject some information about the relative or absolute position"；Table 1 逐行（Self-Attention $O(n^2\cdot d)$/$O(1)$/$O(1)$；Recurrent $O(n\cdot d^2)$/$O(n)$/$O(n)$；Convolutional $O(k\cdot n\cdot d^2)$/$O(1)$/$O(\log_k(n))$）；Table 3 base 行 $d_{\text{model}}=512$、$h=8$、$d_k=64$、$d_v=64$——全部与页面一致。
- FlashAttention（arXiv:2205.14135）：§1 "Benchmarking Attention" bullet "FlashAttention is up to 3× faster than the standard attention implementation across common sequence lengths from 128 to 2K"；摘要 "15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record"。页面两个数字均属实。
- Linear Attention（arXiv:2006.16236）：Eq.6 "$(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V)$"；复杂度表述 $O(NCM)$（C=M=D 即 $O(ND^2)$），"scale linearly with respect to the sequence length"——页面 "$\phi(Q)(\phi(K)^\top V)$、$O(n\cdot d^2)$" 一致。
- Michel et al. 2019（arXiv:1905.10650）：摘要 "a large percentage of attention heads can be removed at test time without significantly impacting performance"——页面表述一致。
- 图内/正文数值像素外复算：2×2 例 $QK^\top=I$、$/√2=0.7071$、softmax 得 $[0.670,0.330]$、$AV=[[1.66,2.66],[2.34,3.34]]$ 逐项复算一致；3×3 遮罩例三行 softmax 得 $[1,0,0]$、$[0.401,0.599,0]$、$[0.258,0.316,0.426]$ 复算一致；$d_k=64$ 饱和表按"$q,k$ 各分量独立 $N(0,1)$、$n=256$、2000 次试验"复算得 $d_k{=}4/64/1024$ 为 0.171/3.97、0.754/0.74、0.943/0.14（页面 0.170/3.97、0.749/0.75、0.936/0.16，在蒙特卡洛误差内）；"最大 logit 与典型值差值"复算 $n=8$ 得 11.4、$n=256$ 得 22.5（页面"约 11""约 23"，吻合），渐近主项 $8\sqrt{2\ln n}$ 在 $n=8/256$ 为 16.3/26.6 确实高估，与页面"对有限 $n$ 会高估"一致；$n=2048$ 时 $n^2=4.19\times10^6$、$n=32768$ 时 $n^2=1.07\times10^9$ 与页面 $4.2\times10^6$、$1.1\times10^9$ 相符。
- 结构/格式：`validate.py` 对 `index.html` 与 `overview.html` 均返回 ok；C1–C10/F1–F6/N1–N2 与正文上标双向对应无缺号；`../rope/index.html`、`../mla/index.html` 均真实存在；无"（待生成）"占位；无含 `$...$` 的 alt；结构图为 HTML 结构（`dg-flow` div）；`dojo:topics=注意力机制` 在 AGENTS.md 词表内；两页互链。

## 问题

- [轻微·来源表述] 第 5 章 Flash vs Linear callout（`<li><strong>Flash Attention</strong>…可在常见序列长度（128–2K）上最多约 3 倍快于标准注意力实现（BERT-large、序列长度 512，相对 MLPerf 1.1 训练记录，端到端训练提速 15%）`）：括号把两种彼此独立的测量并置——"最多约 3 倍快于标准注意力"是 §1 "Benchmarking Attention" 的 kernel 级基准，而 "BERT-large、512、MLPerf 1.1 记录、15%" 是摘要的端到端训练提速，结构上易被读成"3× 是在 BERT-large@512 上测得"｜引文依据：FlashAttention §1 "FlashAttention is up to 3× faster than the standard attention implementation across common sequence lengths from 128 to 2K"；摘要 "15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record"｜修复要求：拆成两句，或改写为"（前者为注意力实现的基准测量；论文另报 BERT-large、seq 512 相对 MLPerf 1.1 记录的端到端训练提速 15%）"，使两组条件不再共用一个括号｜修复：｜复验：
- [轻微·格式] 第 3 章"不缩放 vs 缩放"对照表及其标注（折叠 summary 写"$d_k=64$、$n=256$ 个 key 随机初始化时未缩放 vs 缩放的最大权重对照"，表内却有 $d_k=4/64/1024$ 三行；表头"未缩放熵（nats）"带单位而"缩放后熵"不带；来源章节"构造示例"同写作"$d_k=64$ 饱和数字"）：summary 与来源条的 $d_k$ 取值覆盖范围小于实际表格，熵列单位标注不齐｜引文依据：表内三行为 `4 / 0.170 / 3.97 / 0.043 / 5.05`、`64 / 0.749 / 0.75 / 0.043 / 5.05`、`1024 / 0.936 / 0.16 / 0.043 / 5.05`（复算同量级，数值可用）｜修复要求：summary 与来源条改为"$n=256$ 个 key、$d_k=4/64/1024$ …的对照"，并把第五列表头补为"缩放后熵（nats）"｜修复：｜复验：
- [轻微·格式] 第 1 章复杂度对比表及其表注（表体用 $O(n\cdot d^2)$、$O(k\cdot n\cdot d^2)$、$O(n^2\cdot d)$，表注写"$d$ 是隐藏维度"）与全页其余 25 处使用 $d_{model}$（如"$X\in\mathbb{R}^{n\times d_{model}}$"、"$W^O\in\mathbb{R}^{hd_v\times d_{model}}$"）表示同一量：同一变量两种写法，且 $d_{model}$ 全页未给出定义｜引文依据：Table 1 原文用 d、§3.2.2 用 $d_{\text{model}}$；复算 $n^2d$ vs $nd^2$ 得"$n<d$ 时自注意力更省"，结论不受影响｜修复要求：在表注写明"$d=d_{model}$（模型/隐藏维度）"，或把表体统一改为 $d_{model}$｜修复：｜复验：
- [轻微·格式] 第 5 章 Flash vs Linear callout 首句"机制见对应变体章节"：本页章节为 1–5 加"来源与范围说明"，无任何变体机制章节，页面内也未给出指向 Flash / Linear Attention 的链接（现有链接只有 RoPE 与 MLA 两页），读者按此指示无法定位｜引文依据：不适用｜修复要求：改为具体可定位的表述（如"两者机制不在本页范围，Linear Attention 见对应概念页"并给出链接，若无对应页则删去指向"章节"的措辞）｜修复：｜复验：
- [轻微·格式] 三处外部来源论断未纳入来源章节、也未加编号上标：正文"（Dao et al. 2022, FlashAttention）"、"（Katharopoulos et al. 2020）"、"Michel et al. 2019（"Are Sixteen Heads Really Better than One?"）"，而"来源与范围说明"的 C1–C10/F1–F6/N1–N2 全部只对应 Vaswani 2017，与 style-guide 第 6 节"正文使用 `<sup>[Cx]</sup>` 上标引用…与来源章节双向对应"不符｜引文依据：FlashAttention §1/摘要、arXiv:2006.16236 Eq.6 与复杂度、arXiv:1905.10650 摘要（三处论断本身经回源核对均成立，问题只在编号与来源条缺失）｜修复要求：为这三条后续工作论断在"外部数字与实验条件（N）"下补条目（作者、年份、论文名、可定位的章节/公式/摘要句），并在正文对应句加 `<sup>[Nx]</sup>`｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：可发布（核心结论、公式、构造示例与全部外部数字均已回源核对或复算通过；5 项轻微项不阻塞发布，建议在下一轮前逐条修复）
- 说明：本轮未发现来源不支持、定位不到、把推断包装成来源结论的论断；2×2 例、3×3 遮罩例、$d_k$ 饱和对照表三处构造数字均可复算，且均已标注"构造示例/非论文一手数据"；符号、summary 公式、折叠块答案与正文结论一致。

统计：阻断 0 / 重要 0 / 轻微 5