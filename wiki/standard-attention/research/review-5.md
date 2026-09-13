<!-- review-meta
round: 5
page: wiki/standard-attention/index.html
reviewed_content_sha256: adc3d50a78ba730c
-->
# 标准 Transformer 注意力审查记录（第 5 轮）

- 页面版本：a6e62f965f1ee44e1552e2d51c4a2340da61c3e2（git blob 哈希；工作树未修改）
- 审查时间：2026-09-13 20:25
- 审查者：独立子代理（编排者派发的独立审查者，未参与写作，未参与前序轮次；未读取本页 research/ 下任何文件）
- 已完整阅读章节（按顺序）：head（description / dojo:summary / dojo:type / dojo:topics / dojo:tag）→ blockquote.meta 主要依据 → 引言 → 核心问题（4 题含解答）→ 最容易误解 → 1. 注意力要解决什么问题 → 2. 缩放点积公式 → 3. 为什么除以 √d_k → 4. 多头注意力 → 5. 复杂度、瓶颈与边界 → 结语 → 来源与范围说明（论断与来源 C / 公式与来源 F / 外部数字与实验条件 N / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）
- 已核对的来源：Vaswani et al. 2017 arXiv:1706.03762（PDF，§3.2.1 正文、Eq.(1)、脚注 4、§3.2.2、§3.2.3、§3.3、§4 Table 1 与正文）；Dao et al. 2022 FlashAttention arXiv:2205.14135（摘要与 §1）；Katharopoulos et al. 2020 Linear Attention arXiv:2006.16236（摘要）
- 机械项：`python3 .dojo/scripts/validate.py wiki/standard-attention/index.html` 返回 `validation ok`（退出码 0）；本地链接（../rope/index.html、../mla/index.html、overview.html、../../index.html、5 个 libs 资源）均存在；无 `（待生成）` 占位；页面与 overview.html 双向互链

## 问题

- [阻断·技术] §5「Flash vs Linear 的区分」callout 第 1 条：页面写「Flash Attention（Dao et al. 2022, FlashAttention）……结果与标准注意力数值等价，但快 2-4 倍」，把「2-4×」安在「Flash Attention 相对标准注意力」这一对照上。核对该论文：2-4× 是 block-sparse FlashAttention 相对（稠密）FlashAttention 的数字，论文对 Flash Attention 相对标准注意力的表述是「最多 3×」。｜引文依据：arXiv:2205.14135 摘要「3× speedup on GPT-2 (seq. length 1K), and 2.4× speedup on long-range arena (seq. length 1K-4K)」「15% end-to-end wall-clock speedup on BERT-large」；§1「As a proof of concept, we implement block-sparse FlashAttention, a sparse attention algorithm that is 2-4× faster than even FlashAttention」；§1「Benchmarking Attention. FlashAttention is up to 3× faster than the standard attention implementation across common sequence lengths from 128 to 2K」；§4「is up to 3× faster than standard attention for common seq. lengths (up to 2K)」｜修复要求：把倍数改为论文支持的值——Flash Attention 相对标准注意力「最多约 3×」（常见序列长度 128–2K；训练端到端 15% / 2.4× / 3×），或直接删去该倍数；不得再用 block-sparse 的 2-4× 充当稠密 Flash Attention 相对标准注意力的数字。｜修复：｜复验：

- [轻微·技术] §3「不缩放的后果」正文段：写「$n$ 个 key 中最大 logit 与典型值的差值约 $E[\max]\approx 8\sqrt{2\ln n}$（$n=8$ 时约 16，$n=256$ 时约 27）」。$8\sqrt{2\ln n}$ 只是高斯极值期望的渐近主项（省略了 $-\dfrac{\ln\ln n+\ln 4\pi}{2\sqrt{2\ln n}}$ 修正项），按该段自称的设定（$q,k$ 各分量独立取自 $N(0,1)$、$d_k=64$、$n$ 个 key）复算，真实 $E[\max]$ 明显小于页面给出的值。｜引文依据：按同一设定复算（standard normal 极值 200000 次、$\sigma=8$）：$n=8$ 时 $E[\max]=11.37$（页面写约 16），$n=256$ 时 $E[\max]=22.61$（页面写约 27）；页面所用主项 $8\sqrt{2\ln 8}=16.3$、$8\sqrt{2\ln 256}=26.6$ 与之相符，说明偏差来自公式而非笔误。｜修复要求：把 16/27 改为按上述设定复算得到的约 11/约 23（同时核对紧随的 $e^{16}$ 结论句改用对应 $n$ 的数值），或明确注明 $8\sqrt{2\ln n}$ 只是渐近主项、对小 $n$ 系统性偏高。｜修复：｜复验：

- [轻微·可读性] §1 第 1 段：示例句「The cat sat on the mat because it was tired.」中「cat」是第 2 个 token，页面写「传统 RNN 把"cat"的信息编码进第 1 步的隐状态 $h_1$，再传递给 $h_2$、$h_3$……直到"it"所在的第 8 步才读到」，起点错位一位（「it」在第 8 步的记述本身正确）。｜引文依据：不适用｜修复要求：改为「编码进第 2 步的隐状态 $h_2$，再经 $h_3,\dots,h_8$ 传递到 it 所在的第 8 步」，或改述为对第 1 个 token 的传递（与同章「第 1 个 token 的信息要经过 $n-1$ 个 RNN 单元」一致）。｜修复：｜复验：

- [轻微·可读性] 元话语（告知读者阅读动作的句子）：① 引言末句「下面依次讨论四个问题：标准注意力解决什么问题、公式每步为何如此设计、$O(n^2)$ 瓶颈在哪、哪些问题它本身不解决。」——这四个问题已由紧随其后的「核心问题」块逐条列出，此处重复且以「下面……讨论」预告结构；② §5 首句「知道机制后，接着看它在哪里会失效、为什么需要变体。」；③ §5 callout 首行「（下面只讨论瓶颈，机制由对应变体章节承担；……）」。｜引文依据：不适用｜修复要求：删去「下面依次讨论」「接着看」「下面只讨论」这类叙述自身阅读节奏的措辞，直接给结论或范围，例如 ② 改为「标准注意力的失效点决定了哪些问题必须由后续变体解决」，③ 改为「这里只比较两者对瓶颈的处理，机制见对应变体章节」。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 3
- 处置：修复（阻断 1 条必须关闭后方可发布；轻微 3 条建议同轮一并处理）
- 复算与核对已通过、无需修改的项：§1 复杂度表与 N2（论文 Table 1：Self-Attention $O(n^2\cdot d)$/O(1)/O(1)、Recurrent $O(n\cdot d^2)$/O(n)/O(n)、Convolutional $O(k\cdot n\cdot d^2)$/O(1)/$O(\log_k n)$，逐格相符）；论文超参数 N1（$d_{model}=512$、$h=8$、$d_k=d_v=64$、$W^O\in\mathbb{R}^{8\cdot64\times512}$，§3.2.2 与 §3.3 相符）；参数量等价 $h\cdot d_{model}\cdot(d_{model}/h)=d_{model}^2$；2×2 构造示例四步（$QK^\top$、$\div\sqrt2$、softmax $[0.670,0.330]$、$AV=[[1.66,2.66],[2.34,3.34]]$ 全部手算/复算相符）；3×3 因果遮罩示例（每行 softmax 结果 $[1,0,0]$、$[0.401,0.599,0]$、$[0.258,0.316,0.426]$ 全部复算相符）；softmax 雅可比 $p_i(\delta_{ij}-p_j)$；不缩放 vs 缩放对照表（按 $N(0,1)$、$n=256$、2000 次复算得 0.171/3.97、0.754/0.74、0.943/0.14，与页面 0.170/3.97、0.749/0.75、0.936/0.16 在「约」范围内相符，缩放列 0.043–0.044/5.05 相符；$\ln 256=5.545$、$1/256\approx0.004$ 相符）；$e^{16}\approx8.89\times10^6$；$n=2048\to4.2\times10^6$、$n=32768\to1.1\times10^9$；C6/C7/C10 原文引用（§3.2.2「Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.」、§3.2.3「masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections」、§3.2.1「While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$.」逐字相符）；§3.2.1 脚注 4（「components of q and k are independent random variables with mean 0 and variance 1 … has mean 0 and variance $d_k$」）支撑 F3 与折叠推导；Linear Attention 的核分解与 $O(n^2)\to O(n)$ 表述与 arXiv:2006.16236 摘要相符；两级问题块（核心问题 4 题、五章本章问题各 3 题）均有解答折叠块且核心问题答案指明完整论证所在章节；全文无「我们」「你」、无调试叙事、无指向 research/ 的路径、无 Unicode 数学字符越界（validate.py 通过）。
