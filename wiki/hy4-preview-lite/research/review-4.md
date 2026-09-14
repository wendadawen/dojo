<!-- review-meta
round: 4
page: wiki/hy4-preview-lite/index.html
reviewed_content_sha256: 9e826724ccace418
-->
# Hy4 preview 轻量版压缩链路审查记录（第 4 轮）

- 页面版本：e6d6cbecc052e77513e02d73c4bfb17406157d32
- 审查时间：2026-09-14 16:59
- 审查者：独立子代理
- 页面类型：note（依 `guides/note.md` 审查）
- 已完整阅读章节（含图注，按顺序）：压缩对象与结果 → 图表「Hy4 轻量版压缩流水线」及其图注 → 低档格式：Sherry 稀疏三值量化 → 层间分档：MIX-STQ1_0 逐层混合精度 → 效果核对 → 异构设备联合推理 → 部署要点 → 来源

## 核对与复算

来源：HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡（原始 README 全文）、HuggingFace tencent/Hy4-preview 模型卡、arXiv:2601.07892（Sherry 论文）、腾讯混元官方文章及快科技/凤凰科技/智东西/IT之家等转载、llama.cpp PR #22836。以下均给出核对时看到的关键数值。

- 体积与 bpw：模型卡文件表逐字为「Hy4-preview-Q4_K_M.gguf / 435.20 GiB / 4.86；Hy4-preview-UD-IQ1_M.gguf / 219.83 GiB / 2.44；Hy4-preview-STQ1_0.gguf / 213.66 GiB / 2.38」，与页面表格三项逐一相符。
- Sherry 编码：模型卡「That is `2 + 32 + 8 = 42` bytes per 256 weights = **1.3125 bpw**」「a 4-bit code plus a 1-bit table-select, indexing a 32-entry codebook」；1.25 bit/权重与 $\binom{4}{3}\times 2^3=32=2^5$ 复算成立（$336/256=1.3125$）；「8B 符号」的命名与前置概念页 `sherry-ternary-quant` 的「64 个符号位共 8 字节，1 bit 记录整体符号是否翻转」一致，非改写。
- 编码器误差：模型卡「LS scale alone −89.7% weighted SSD; imatrix terms a further −4.1% of the remainder」「1200 real expert rows」，与页面一致；WLS 公式 $d=\sum w_i x_i q_i/\sum w_i q_i^2$ 由 $\partial/\partial d\,\sum w_i(x_i-dq_i)^2=0$ 复算成立，符号 $x_i,q_i,w_i,d$ 全页单义且首次使用处即定义。
- 层间分档：模型卡「STQ1_0 (29 layers) / IQ2_XXS (48 layers)」「1.3125 bpw (STQ1_0) on 29 layers and 2.0625 bpw (IQ2_XXS) on the other 48」；「The three routed-expert families are 97.7% of all parameters」；down=IQ3_XXS（末 3 层 IQ4_XS）、注意力 Q5_K、MLA Q8_0、DSA Q8_0/F32、router/norms/output F32 均能在模型卡对应小节定位。页面「平均 1.78 bpw（复算）」复算：$(29\times1.3125+48\times2.0625)/77=1.7800$，成立且已标「复算」。
- 体积差：$219.83-213.66=6.17$，页面标「复算」，成立；官方文章「还比 UD-IQ1_M 少占了 5 个多 GiB」支持「少占超 5 GiB」。
- 精度：官方转载「MCP-Atlas 从 83.7 微降至 83.2」「SWE-Bench multi 82.9 → 81.3」明确可定位；保留率复算 $81.3/82.9=98.1\%$、$81.1/81.3=99.8\%$，与页面「98.1% 到 99.8%」一致。MRCR 81.3→81.1、IFBench 73.5→72.5 未能在可抓取的转载正文中逐字定位（评测对比以图片形式给出），但与页面同源的 `mixed-precision-quant` 页同数并注明出处为第三方转述，故不作来源不符处理。
- 吞吐：模型卡「pp512 204.56 ± 1.42 t/s」「tg128 20.47 ± 0.02 t/s（英文节）」「19.52 ± 0.01 t/s（中文节）」，页面记录的两处不一致属实。
- 异构推理：官方转载「一台 4090 笔记本 + 一台四卡 A4000 服务器，显存合计 80GB、内存合计 64GB，分处两个局域网，1.02 token/s，约 6 倍」；$16+4\times16=80$ 自洽。
- 部署：模型卡原始 README「0001-hyv4-architecture.patch — 18 files」「0002-stq1_0-quant-and-cuda.patch — 25 files」「`--jinja` is required for chat」「~435 GiB (Q4_K_M) or ~214 GiB (STQ1_0)」「over NFS random page faults run at ~12 MB/s, turning a 1-minute load into hours」「An imatrix is mandatory for STQ1_0」，逐项相符。
- 链接：`../sherry-ternary-quant/index.html`、`../mixed-precision-quant/index.html` 均存在；tencent/Hy4-preview、AngelSlim/Hy4-preview-GGUF 页面存在。
- 机械项：`.dojo/scripts/validate.py` 返回 `validation ok`；无 `<pre><code>` 可运行代码；alt 属性中无 `$...$`（SVG 用 `aria-label`，图为静态内联 SVG）；含公式的 summary 由 KaTeX 渲染（`$\binom{4}{3}\times 2^3=32$`）。

## 问题

- [轻微·技术] 压缩对象与结果段（及来源段）/ 部署要点段：同一「约 214」量在两处以不同单位出现——「把权重压缩到 213.66 GiB（官方概称约 214 GB）」「1.5 TB 与 214 GB 概数」与「214 GiB（STQ1_0）」。213.66 GiB ≈ 229 GB，与「约 214 GB」相差约 7%，GB 与 GiB 混用易被读作同一量。｜引文依据：模型卡「VRAM for full residency: ~435 GiB (Q4_K_M) or ~214 GiB (STQ1_0)」（GiB）；官方文章「1.5TB压到214GB」（GB）。｜修复要求：统一该数量的单位，或将括注改为明确指向官方口径（十进制 GB 概称，对应 213.66 GiB），使「214 GB」与「214 GiB」不再同形。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（遗留 1 项轻微，建议修复）。本轮未发现事实性论断回源不符、同页矛盾、算式与结论不符、图文读数不符或编号不符；事实、公式（WLS 与位宽账）、复算数字（1.78 bpw、6.17 GiB、98.1%–99.8%）与部署数字均可在官方模型卡与官方文章定位，表述维度未发现元话语/会话指代/调试叙事/AI 拼接腔。
