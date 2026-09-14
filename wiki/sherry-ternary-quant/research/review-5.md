<!-- review-meta
round: 5
page: wiki/sherry-ternary-quant/index.html
reviewed_content_sha256: d97e52ca88007589
-->
# Sherry 稀疏三值量化（STQ1_0）审查记录（第 5 轮）

- 页面版本：17859fb958e0b770c39665b20d93e9ad97ea2559
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：引言（主要依据／核心问题／常见误解）→ 1. 三值量化的打包困境 → 2. 3:4 稀疏三值——4 个权重恰好 5 bit（2.1／2.2／2.3）→ 3. STQ1_0 字节布局——1.3125 bpw 的账（3.1／3.2）→ 4. 量化决策——缩放系数与零位怎么选（4.1／4.2／4.3）→ 5. 为什么快——SIMD 解码与实测（5.1／5.2）→ 6. 训练侧的 weight trapping 与 Arenas → 来源与范围说明

## 问题

- [轻微·技术] 4.3 代码块后「验证的机制」：把 `scale_mode` 说成对应 4.1 的「三种缩放选法」，但代码只实现两种（amax、加权最小二乘），4.1 的第一种 Sparse-AbsMean 未进入代码对照，读者会误以为三种缩放都在代码里跑过｜引文依据：代码原文 `for name, scale_mode, zero_mode, rounds in [("amax + argmin 零位…","amax",…), ("WLS + argmin…","wls",…), ("WLS + imatrix…","wls",…)]` 与函数内 `if scale_mode == "amax": … else:  # 加权最小二乘`，取值只有 `amax`/`wls` 两种；页面 4.1 节列出 Sparse-AbsMean、amax、加权最小二乘三种｜修复要求：把该句改为「分别对应 4.1 的 amax 与加权最小二乘两种缩放选法、4.2 的两种零位选法」，或补一句说明 Sparse-AbsMean 属论文训练侧、未纳入本页代码对照｜修复：｜复验：
- [轻微·格式] 3.2 节与 4.1 节的符号 $w$ 一符两义：3.2 用 $w$ 表示权重本身（下标为权重序号），4.1 及 4.3 代码用 $w_i$ 表示「第 $i$ 个权重的重要性」（来自 imatrix），而全页权重统一记作 $x$（2.3 节 $x=(0.6,-0.1,0.5,-0.7)$、4.1 节 $x_i$）｜引文依据：3.2「第一组是 $(w_0, w_{16}, w_{32}, w_{48})$，第二组是 $(w_1, w_{17}, w_{33}, w_{49})$」；4.1 符号表「$w_i$：第 $i$ 个权重的重要性，来自 imatrix（校准数据统计）」｜修复要求：3.2 及本章问题解答中的 stride-16 示例改用全页统一的权重记法（写「第 $k$ 个权重」或 $x$ 加下标），把 $w$ 留给重要性一义｜修复：｜复验：
- [轻微·技术] 第 4 章本章问题第 2 题解答把交替轮数写成收敛：「两者交替 3 轮收敛」，「收敛」无来源支持，模型卡只给固定轮数｜引文依据：模型卡原文「…alternating for 3 rounds. Measured on 1200 real expert rows: the LS scale alone gives -89.7% weighted SSD, and the imatrix terms a further -4.1% of the remainder.」未出现收敛性表述｜修复要求：删去「收敛」，改为「两者交替 3 轮」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复

### 本轮核对记录（供复验参照）

- 规范判定：`dojo:type=concept`，按 `guides/concept/check.md` 审查；`.dojo/scripts/validate.py wiki/sherry-ternary-quant/index.html` 返回 `validation ok`。
- 代码：抽出页面内嵌 Python 原样执行，输出与「预期输出」块逐行一致（amax d=1.1743/加权SSD=117.5101；WLS+argmin d=0.5152/19.6010；WLS+imatrix 3 轮 d=0.5381/17.7533；全 1 重要性下两零位规则一致 True；贯穿示例 q=[1,0,1,-1]、SSD=0.0300、amax SSD=0.0600；42B、bpw=1.3125）。
- 论文（arXiv:2601.07892v1）：摘要、§2.3「Challenge」、§3.1「3:4 Sparse Ternary Quantization」、附录 A/C/D 与 Table 1/Table 4 逐条核对——$\binom{4}{3}\times2^{3-1}=16$、2:4 的 $\binom{4}{2}\times2^{2-1}=12$、vpshufb 16 项上限（B−1≤4）、Sparse-Absmean（附录 D「Optimality of the Sparse-Absmean」）、有效秩 ER<750/4096、$Y=XT\alpha+\lambda_tXW$、Table 4（0.7B：34.01/132.13/116.83/148.27；3B：7.55/41.87/38.80/45.55）、Table 1（1B 均值 0.519 对 0.519；3B 0.567 对 0.576；ARC-c 1B 0.309 对 0.305、3B 0.364 对 0.346）、10B tokens UltraFineWeb、3 次运行，均一致；Tequila 在 Table 1 标注为 1.67 bit，与页面写法相符。
- PR #22836（正文原文）：bench 表 8 行（STQ1_0 358.00 MiB/732.69±20.00/147.47±1.36；TQ1_0 1.69 bpw/401.50/728.69±19.88/138.87±0.96；TQ2_0 2.06 bpw/445.00/689.25±16.61/175.06±1.30；Q1_0 336.25/768.47±14.75/109.62±16.93，Q1_0 无 bpw）、「~11% smaller than TQ1_0, ~20% smaller than TQ2_0」「~6% size increase but ~35% higher tg128」「1.6875-bit 3-way packing」、stride-16（w0/w16/w32/w48，vld1q_s8 无需重排）、vqtbl2q+vdotq_s32、12 核 24GB/‑ngl 0/8 线程，均一致。
- 模型卡 AngelSlim/Hy4-preview-GGUF：「2 + 32 + 8 = 42 bytes…1.3125 bpw」、`d = sum(w*sel*x)/sum(w*sel^2)`、零位 `w[j]*(x[j]^2-(|x[j]|-d)^2)`、`alternating for 3 rounds`、`-89.7%`/`-4.1%`、`1200 real expert rows`、PTQ、文件表 213.66 GiB/2.38 bpw、路由专家 gate/up 在 29 层用 1.3125 bpw（另 48 层 2.0625 bpw），均一致。
- 链接与资源：`../hy4-preview-lite/index.html`、`../mixed-precision-quant/index.html`、`overview.html`、`../../index.html` 与全部 `../../libs/` 资源均存在；正文无 `research/` 路径、无「待生成」占位。
- 表述维度：通读全文含折叠块与图注，未发现会话指代（我/我们/你）、调试或复现踩坑叙事、以「场景」当术语、公文连接词堆叠；「本页…」句式与「主角」「大头」等口语化措辞在站内同批页面中属通行写法（如 mixed-precision-quant 首段同构句式），本轮不另记问题。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
