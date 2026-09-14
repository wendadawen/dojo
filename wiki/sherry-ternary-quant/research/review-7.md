<!-- review-meta
round: 7
page: wiki/sherry-ternary-quant/index.html
reviewed_content_sha256: c2aab96cdcf65cca
-->
# Sherry 稀疏三值量化（STQ1_0）审查记录（第 7 轮）

- 页面版本：index.html 工作树哈希 fb7c6ad294180b907064348472a2b65e709538ae
- 审查时间：2026-09-14 17:44（CST）
- 审查者：编排者派发的独立审查者（独立子代理，未参与写作与前序审查）
- 已完整阅读章节：核心问题、常见误解、1. 三值量化的打包困境、2. 3:4 稀疏三值——4 个权重恰好 5 bit、3. STQ1_0 字节布局——1.3125 bpw 的账、4. 量化决策——缩放系数与零位怎么选、5. 为什么快——SIMD 解码与实测、6. 训练侧的 weight trapping 与 Arenas——为什么 Hy4 轻量版用不到、来源与范围说明（含全部折叠块、图注、图内标签、代码与预期输出）

## 核对版本

- 论文：arXiv:2601.07892**v1**（2026-01-12 提交，arXiv 页仅此一版，无 v2，故不涉及 v1/v2 编号差异）。逐条核对 §2.1、§2.3、§3.1、§4.2、附录 A/B/C/D/F、Table 1、Table 4、Fig. 6。
- llama.cpp PR #22836「ggml-cpu: add STQ1_0 ternary quantization with ARM NEON vec_dot kernel」：引言、Stride-16 sparsity layout、Performance 三节。
- HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡：文件表、「Our encoder」节、侧栏参数规模。
- 腾讯混元官方「Hy4 preview 轻量版」文章：见「未核尽项」。

## 已核对且一致（引文依据）

- M4 Pro 表与 PR Performance 表逐行一致：STQ1_0 1.31 bpw / 358.00 MiB / pp512 732.69±20.00 / tg128 147.47±1.36；TQ1_0 1.69 / 401.50 / 728.69±19.88 / 138.87±0.96；TQ2_0 2.06 / 445.00 / 689.25±16.61 / 175.06±1.30；Q1_0「—」/ 336.25 / 768.47±14.75 / 109.62±16.93。派生比例复算：358/401.5=0.892（小约 11%）、358/445=0.805（小约 20%）、358/336.25=1.065（大约 6%）、147.47/109.62=1.345（高约 35%）——均成立。
- 论文 Table 4 逐行一致（0.7B：BF16 34.01/1360.0、I2_S 132.13/256.56、TL2 116.83/233.44、Sherry 148.27/205.50；3B：7.55/6190.0、41.87/873.65、38.80/846.01、45.55/712.40）。「0.7B 快约 27%」复算 148.27/116.83=1.269 成立；「3B 快 18%」为论文 §4.2 原文「for the 3B model, Sherry achieves a 18% speedup over the 1.67-bit baseline」。
- 论文 Table 1：1B Tequila 1.67 平均 0.519 vs Sherry 1.25 平均 0.519；3B Tequila 0.576 vs Sherry 0.567；ARC-c 1B 0.305→0.309、3B 0.346→0.364（两规模反超）——与页面一致。实验条件「10B tokens sampled from the UltraFineWeb dataset」「averaged over three independent runs」「PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande」与 N4 一致。
- 论文 §3.1：「(4C3)×2³=32 unique permutations」「This mathematically saturates a 5-bit index (2⁵=32)」「a 2:4 scheme only utilizes C(4,2)·2^(2−1)=12 states, resulting in bit-waste」「This constrains the index to 4 bits (2⁴=16 entries), implying B−1≤4」——与 §1/§2.1/§2.2 一致。
- 论文附录 C：「A 3:4 ternary block with one shared sign bit has a total of C(4,3)·2^(3−1)=16 unique patterns」「This perfectly saturates the 2⁴ entries in the LUT」——与 §2.2 一致。
- 论文附录 D：α*_j=(4/(3·d_in))Σ_{i∈S_j}|W_ij|（留存项平均绝对值），保最大三项、剪最小一项最优——与 §4.1 Sparse-AbsMean 及 F4 一致。
- 论文附录 F / §6 侧：「ER<750」「the gradient matrix having a total dimensionality of 4096」「comparable to that of binary quantization」、Arenas「Y = XTα + λ_t XW」「Zero-Overhead Inference」——与 §6 一致。
- 模型卡「Our encoder」：d = sum(w*sel*x)/sum(w*sel²)；零位 argmin w[j]*(x[j]^2-(|x[j]|-d)^2)；两步交替 3 轮；1200 行真实专家权重上最小二乘缩放 −89.7%、imatrix 再 −4.1%；上游「d = amax」并置零最小幅值——与 §4.1/§4.2/§4.3 及正文第 4 章一致。
- 模型卡文件表与配方：Hy4-preview-STQ1_0.gguf 213.66 GiB、2.38 bpw；MIX_STQ1_0 作用于 routed-expert gate/up 投影（29 层 STQ1_0 1.3125 bpw + 48 层 IQ2_XXS）；侧栏「770B params」——与引言与 N6 一致。
- PR 布局描述：「5 bits per 4-weight group, i.e. a 4-bit codebook index + 1-bit sign, plus a single fp16 scale per 256-weight block: 42 B / 256 = 1.3125 bpw」「qs[32] + sign[8] + fp16 scale」「w0, w16, w32, w48 ... no deinterleave or repack required」——与 §3.1/§3.2 一致。
- 代码实跑（python3，固定种子 20260901）输出与页面「预期输出」逐字一致：amax 1.1743/117.5101、WLS+argmin 0.5152/19.6010、WLS+imatrix(3 轮) 0.5381/17.7533、全 1 重要性两规则一致=True、贯穿示例 SSD=0.0300、amax SSD=0.0600、42B、bpw=1.3125。
- 公式复算：C(4,3)×2³=32；log₂81≈6.34；(2−1.585)/1.585≈26%；Σwᵢ(xᵢ−dqᵢ)² 对 d 求导得 d=Σwᵢxᵢqᵢ/Σwᵢqᵢ²；x²−(|x|−d)²=2|x|d−d² 且随 |x| 递增——均成立。
- 机械项：`.dojo/scripts/validate.py` 返回 validation ok；`dojo:topics=推理系统` 属固定大类；全文无 Unicode 数学字符（仅 UI 图标 ⌂ ◐ ☀ ↑）；`alt` 无 `$...$`；C1–C14 / F1–F6 / N1–N6 全部被正文引用且全部在来源章节有定义；站内链接 `../hy4-preview-lite/index.html`、`../mixed-precision-quant/index.html`、`overview.html` 与本地资源文件均存在；图均为 HTML 结构（dg-flow / dg-stack），无等宽字符框线图；全文无第一人称复数、无第二人称称呼读者。

## 问题

- [重要·技术] 来源与范围说明（C6 条目）与 §1 正文：C6 把「稀疏加速单元的限定」这一子论断的位置标注为「§3.1 与附录 C.1/C.2」，但该论断实际出自论文附录 B.3。｜引文依据：附录 B.3（N:M sparsity）原文「most efforts are not coordinated with ultra-low bit quantization, as they are largely designed for Sparse Tensor Cores on GPUs, which currently prioritize 16-bit or 32-bit floating-point arithmetic」及「the intersection of N:M sparsity and ternary quantization remains largely unexplored」；§3.1 与附录 C.1/C.2 无 Sparse Tensor Core / GPU 稀疏加速单元的任何句子（§3.1 仅笼统提到「GPU-vendor kernels」）。｜修复要求：把 C6 中「稀疏加速单元限定」子项的位置改写为附录 B.3（§3.1 与附录 C.1/C.2 可保留为「LUT 约束」与「2:4 的 12 状态」两项的依据），使三个子论断各对应正确位置。｜修复：｜复验：
- [重要·技术] §2.3 手算示例、§4.1 手算对照 与 §3.1、§4.3 代码：缩放系数 $d$ 的粒度页内不一致。手算示例把 $d$ 当作单个 4 权重组私有（§2.3「取一组 4 个权重 … 设缩放系数 d=0.6」；§4.1「amax 取 d=0.7」并据此算平方误差和），而 §3.1 明确「一个全块共享的 fp16 缩放系数」且把该系数记作 $d$（每 256 权重块一个），§4.1 对 amax 自身的描述也是「$d$ 取块内绝对值最大的权重」「$d$ 被 256 个权重中最大的离群值钉死」，§4.3 代码同样按整块 256 个权重取 amax / 加权最小二乘。页面未说明示例与格式在 $d$ 粒度上的差异。｜引文依据：PR #22836「plus a single fp16 scale per 256-weight block: 42 B / 256 = 1.3125 bpw」；本页 §3.1「再加一个全块共享的 fp16 缩放系数，2 字节」与「$2$：一个 fp16 缩放系数 $d$ 占用的字节数」；§4.1「$d$ 被 256 个权重中最大的离群值钉死」；§4.3 代码「d = max(abs(x) for g in groups for x in g)」（groups 覆盖全部 256 个权重，实测 d=1.1743）。｜修复要求：在 §2.3 与 §4.1 手算示例处注明示例为便于复算按 4 权重组取值、与格式中每 256 权重块共享一个 fp16 缩放系数不同；或把两处示例统一到「块」这一单位并同步修正 §4.1 对 amax 的对照说明。｜修复：｜复验：
- [轻微·技术] §5.2 第 5 章末段：把「1-bit 二值配置相对 1.25-bit 三值有约 3 个百分点的精度差距」表述为「按论文结果」，但引文编号 [C13] 指向 llama.cpp PR #22836；论文正文对该对比只有 Fig. 6（「Ablation study of Arenas」）的定性描述，未在正文给出该百分点数字。｜引文依据：PR #22836「Citing the Sherry paper (Fig. 6), 1.25-bit ternary reportedly matches 1.67-bit accuracy with 25% fewer bits, while 1-bit binary has a ~3 pp accuracy gap」；论文 Fig. 6 图注为「Figure 6: Ablation study of Arenas」，正文仅称 Arenas「yields consistent gains across all configurations」。｜修复要求：将该处表述改为「按 PR #22836 转述的论文 Fig. 6」，或在 [C13] 条目中写明该数字为 PR 转述论文 Fig. 6 而非论文正文数字。｜修复：｜复验：
- [轻微·技术] 引言与 N6：BF16 权重「约 1.5 TB」的原始出处（混元官方文章）本轮未能取回原文，页面 N6 未附可逐条核对的原文片段。｜引文依据：模型卡侧栏给「770B params」，但全页未给出 BF16 体积；「1.5 TB」与 770B×2 B≈1.54 TB 自洽，但非模型卡明载。｜修复要求：在 N6 补上混元官方文章的原文片段或可定位链接，使该数字可逐条核对（若确实无法取得片段，则将其降级为明确标注的概数转述）。｜修复：｜复验：

## 未核尽项

- N6 引用的腾讯混元官方技术文章「Hy4 preview 轻量版」本轮无法取回原文（本次会话的网络检索额度已用尽，直接抓取该站返回 TLS 错误）。上一条已把由此产生的核对缺口记为轻微问题。其余来源（arXiv v1、PR #22836、模型卡）均已取回并逐条核对。

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 2
- 处置：修复（无阻断；2 条重要问题与 2 条轻微问题需在下一轮前关闭）
