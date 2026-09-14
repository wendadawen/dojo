<!-- review-meta
round: 10
page: wiki/sherry-ternary-quant/index.html
reviewed_content_sha256: d191f2715d608f5e
-->
# Sherry 稀疏三值量化（STQ1_0）审查记录（第 10 轮）

- 页面版本：wiki/sherry-ternary-quant/index.html 工作树哈希 3d27a2d1af6bd7a3ed9703dbfc7b110e7df25f5e
- 审查时间：2026-09-14 18:09
- 审查者：独立子代理（未参与写作，未参与前序审查；未读取本页 research/ 下任何文件）
- 适用规范：dojo:type=concept → guides/concept/check.md（并对照 guides/concept/style-guide.md）
- 已完整阅读章节（按顺序）：核心问题 / 常见误解 / 1. 三值量化的打包困境 / 2. 3:4 稀疏三值——4 个权重恰好 5 bit（2.1–2.3）/ 3. STQ1_0 字节布局——1.3125 bpw 的账（3.1–3.2）/ 4. 量化决策——缩放系数与零位怎么选（4.1–4.3，含代码折叠块全文）/ 5. 为什么快——SIMD 解码与实测（5.1–5.2）/ 6. 训练侧的 weight trapping 与 Arenas / 来源与范围说明；overview.html 全文。

## 核对依据（所核对的版本与原文片段）

1. **Sherry 论文 arXiv:2601.07892v1（2026-01-12）**，用 arXiv HTML 版逐节核对（摘要页另核，确认 arXiv 页未标注任何会议）。原文片段：
   - §2.1「constraining weights to the set {-1,0,+1}」；§2.3「a theoretical lower bound of 1.58 bits (log2 3)」；§2.3「1.67-bit Strategy: This scheme packs three weights into 5-bit blocks ... often results in slower inference speeds than the 2-bit strategy」。
   - §3.1「exactly three are quantized to non-zero values (±1), and one is fixed to zero」；「(4 choose 3) × 2^3 = 32 unique permutations. This mathematically saturates a 5-bit index (2^5 = 32)」；「3:4 structured scheme maintains a 25% sparsity level」；「the maximum capacity for a single-instruction lookup table is 16 bytes ... splits the 5-bit representation into 1 sign bit and 4 index bits」。
   - 附录 A「every four-element block is packed into a 5-bit metadata structure: a 4-bit index ... and a 1-bit value represents the shared or dominant sign」；附录 C.2「a total of C_4^3 · 2^{3-1} = 16 unique patterns ... a 2:4 scheme only utilizes C_4^2 · 2^{2-1} = 12 states, resulting in bit-waste」。
   - 附录 B.3「largely designed for Sparse Tensor Cores on GPUs, which currently prioritize 16-bit or 32-bit floating-point arithmetic」「the intersection of N:M sparsity and ternary quantization remains largely unexplored」。
   - 方法节 Eq.4/5（argmin|W| 置零、α*=(1/|S|)Σ|W|）与附录 D「Appendix D Optimality of the Sparse-Absmean」；Eq.7「Y = XTα + λ_t XW」；§3.2「Effective Rank (ER) for 3:4 sparse training ... low ER (ER<750) ... total dimensionality of 4096」。
   - Table 1/2：TequilaLLM 1B 0.519 / SherryLLM 1B 0.519；TequilaLLM 3B 0.576 / SherryLLM 3B 0.567；ARC-c 0.305→0.309（1B）、0.346→0.364（3B）；§4.2「for the 3B model, Sherry achieves a 18% speedup over the 1.67-bit baseline」。
   - Table 4（Intel i7-14700HX）：0.7B BF16 34.01/1360.0、I2_S 132.13/256.56、TL2 116.83/233.44、Sherry 148.27/205.50；3B BF16 7.55/6190.0、I2_S 41.87/873.65、TL2 38.80/846.01、Sherry 45.55/712.40。
   - §4.1「All results are averaged over three independent runs with random seeds」（对应 N4 的「3 次运行」）；五任务 PIQA/ARC-e/ARC-c/HellaSwag/WinoGrande、UltraFineWeb 10B。
2. **llama.cpp PR #22836**（抓取 PR 页原始 HTML 正文核对）：「yielding 1.3125 bits per weight (5 bits per 4-weight group, i.e. a 4-bit codebook index + 1-bit sign, plus a single fp16 scale per 256-weight block: 42 B / 256 = 1.3125 bpw) while admitting fast SIMD decode through a 32-entry codebook lookup」；「decode directly through vqtbl2q + vdotq_s32 with no bit-shuffling」；「the on-disk layout (qs[32] 4-bit codebook indices + sign[8] 1-bit per group + an fp16 scale) ... codebook lookup → sign flip → scale」；Stride-16「groups weights that are stride-16 within each 64-weight chunk (e.g. w0, w16, w32, w48)」「a plain vld1q_s8 at offsets 0/16/32/48 is all that is needed, with no deinterleave or repack required」；性能表（Apple M4 Pro 12 核 24 GB、-ngl 0、8 threads、1.24 B）：STQ1_0 358.00 MiB / 732.69±20.00 / 147.47±1.36，Q1_0 336.25 / 768.47±14.75 / 109.62±16.93，TQ1_0 401.50 / 728.69±19.88 / 138.87±0.96，TQ2_0 445.00 / 689.25±16.61 / 175.06±1.30；「~11% smaller than TQ1_0, ~20% smaller than TQ2_0」「~6% size increase but still has ~35% higher tg128 throughput」「per the Sherry paper (Fig. 6), 1.25-bit ternary matches 1.67-bit ternary accuracy at 25% fewer bits, while the 1-bit binary configuration shows a ~3 pp accuracy gap」；「TQ1_0, whose 1.6875-bit 3-way packing is SIMD-unfriendly」；「recently accepted to ACL 2026」。该性能表确实没有 Q1_0 的 bpw 单元格（已逐行核对表格标记）。
3. **PR 分支源码**（raw.githubusercontent.com/sjl623/llama.cpp/STQ_0/ggml/src/ggml-common.h）：
   - `typedef struct { uint8_t qs[QK_K/8]; uint8_t sign[QK_K/32]; ggml_half d; } block_stq1_0;` 与 `static_assert(sizeof(block_stq1_0) == sizeof(ggml_half) + QK_K/8 + QK_K/32)` → QK_K=256 时 2+32+8=42 B，且字段顺序 qs→sign→d 与页面第 3.1 节叠层图「自上而下为存储顺序」一致。
   - `// STQ1_0 codebook: index = (sign << 4) | slot -> packed 4-lane ternary pattern.`；`GGML_TABLE_BEGIN(uint8_t, stq1_0_codebook, 32)`；`// The sign=1 half is precomputed so decode is a single load`。据此复核页面 2.3 手算示例：零位在第 2 位、相对符号 (+,+,−) → slot=(1<<2)|0b10=6、sign=0 → 码本项 0x26 → lane3..lane0 = (00,10,01,10) = (−1,+1,0,+1) 取反序后即 (+1,0,+1,−1)，与页面 q=(+1,0,+1,−1) 一致。
4. **HF AngelSlim/Hy4-preview-GGUF 模型卡**：「exactly one of every four lanes forced to zero」「2 + 32 + 8 = 42 bytes per 256 weights ... 1.3125 bpw」；「Our encoder」：`d = sum(w*sel*x) / sum(w*sel^2)`、置零 `w[j]*(x[j]^2 - (|x[j]|-d)^2)` 最小者、两步交替 3 轮；「on 1200 real expert rows, the LS scale alone yields −89.7% weighted SSD, and the imatrix terms add a further −4.1% of the remainder」；「amax pins d to the single largest outlier among 256 weights」；文件表 Hy4-preview-STQ1_0.gguf 213.66 GiB / 2.38 bpw；侧栏 770B params；「The routed-expert gate/up projections run at 1.3125 bpw (STQ1_0) on 29 layers」。
5. **代码实跑**：把 4.3 折叠块内 Python 原样写入 /tmp/stq_check.py 执行（random.seed(20260901)），逐行 diff 与页面「预期输出」完全一致（唯一差异是文件末尾换行符）：amax+argmin d=1.1743/SSD=117.5101；WLS+argmin d=0.5152/19.6010；WLS+imatrix×3 d=0.5381/17.7533；一致性检查 True；贯穿示例 SSD=0.0300，amax SSD=0.0600；42B / 1.3125。
6. **机械项**：`.dojo/scripts/validate.py wiki/sherry-ternary-quant/index.html` → `validation ok`；全页 `$...$` 之外无 Unicode 数学字符（唯一命中 α/Σ/· 均位于 <pre><code> 代码块内，规范豁免）；17 个 `<details>` 与 17 个 `<summary>` 一一对应；无 `<svg>`/`<text>`、无位图、无「（待生成）」占位；所有引用资源存在（libs/katex.min.css、katex.min.js、auto-render.min.js、prism*.css/js、dojo-concept.css、../../index.html、overview.html、../hy4-preview-lite/index.html、../mixed-precision-quant/index.html）；图用的 dg-flow/dg-node/dg-stack/dg-layer/dg-arrow/dg-label 类均在 libs/dojo-concept.css 中定义（HTML 结构图，非等宽字符框线图）。

## 问题

- [轻微·技术] 来源与范围说明·论断与来源（C）第 574 行：模型卡的读取日期写作「2026-08 读取」，与本页引言「2026 年 9 月，腾讯混元发布 Hy4 preview 轻量版」自相矛盾，也与同一模型卡在姊妹页的标注不一致｜引文依据：本页第 574 行「HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡（2026-08 读取）：C10、C11、F5、F6、N5 见「Our encoder」节」；wiki/hy4-preview-lite/index.html 第 144 行「HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡（2026-09 读取）」；本页第 65 行「2026 年 9 月，腾讯混元发布 Hy4 preview 轻量版」｜修复要求：将该处「2026-08 读取」改为「2026-09 读取」，与发布日及姊妹页标注一致｜修复：｜复验：

（未发现阻断与重要问题。以下为本轮明确核对为合规、不应作为问题上报的项，供编排者参考：① 以「本页」为主语的自我指代属 style-guide.md §12「自称使用"本页"或"本文"」明确许可，页面未出现第一/第二人称、调试叙事或临场评价，章节衔接句均为一次性、无固定套语；② 正文引用用「4.1/4.2/4.3 节」而非全称标题，该写法在全仓 13 个页面共出现 65 次，属既有约定，非违规；③ 2-bit「多花约四分之一」的 26% 复算为 (2−1.585)/1.585=26.2%，\log_2 81≈6.34 与「编码空间砍掉约六成」（81→32，余 39.5%）同页并存且不矛盾；④ 0.7B「快约 27%」为 Table 4 复算（148.27/116.83=1.269），页面在 N3 与 overview 均标明为复算；⑤ TQ2_0 免查表、M4 Pro 与论文 CPU 不可跨硬件比较等均已显式标注为推断或简化条件；⑥ 论文「被 ACL 2026 接收」明确标注为 PR #22836 的述及、arXiv 页未标注，处理规范。）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：修复（仅一条轻微引用元数据，改毕即可发布；不影响任何结论与数字）
