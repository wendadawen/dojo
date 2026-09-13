<!-- review-meta
round: 4
page: wiki/quantization-basics/index.html
reviewed_content_sha256: a6ee2aa2403cfdc1
-->
# 量化基础审查记录（第 4 轮）

- 页面版本：384ea99ebb100a211675274045b86bad95c52f63
- 审查时间：2026-09-13 19:48
- 审查者：独立子代理（未参与写作，未参与前序审查与修复）
- 已完整阅读章节：核心问题；1. 量化要解决什么问题；2. 均匀量化怎么把浮点变成整数（含 2.1 对称量化特例、2.2 手算与舍入约定折叠块）；3. 量化误差从哪来（含完整过程折叠块）；4. 对称/非对称 与 量化粒度（4.1、4.2，含有效位宽折叠块）；5. PTQ vs QAT 与 浮点 block-wise 量化（5.1、5.2 含两个代码折叠块、5.3）；来源与范围说明

## 已核对的来源（引文依据）

- Gholami et al. 2021（arXiv:2103.13630）§III-B Eq.2/3 原文："Q(r) = Int(r/S) − Z"；"r̃ = S(Q(r) + Z)"；"Note that the recovered real values r̃ will not exactly match r due to the rounding operation." → 支持 C1/C5 及 2 章仿射公式。
- 同文 §III-C 原文："in 'restricted range' S is chosen as max(|r|)/(2^{n−1}−1), which only uses the range of [-127,127]"；"S = (β−α)/(2^b − 1)"；"the activation after ReLU that always has non-negative values"；"Symmetric quantization is widely adopted in practice for quantizing weights" → 支持 C2/C3。
- 同文 §III-C 原文："this approach is susceptible to outlier data in the activations. These could unnecessarily increase the range and, as a result, reduce the resolution of quantization." → 支持 C6。
- 同文 §III-G/§IV 与 Figure 4 原文："In PTQ, a pre-trained model is calibrated using calibration data (e.g., a small subset…)"；Summary (PTQ)："determined without any re-training of the NN model" → 支持 C7。
- 同文 §III-G 原文："the gradient of this operator is zero almost everywhere, since the rounding operation in Eq. 2 is a piece-wise flat operator"；STE "approximates it with an identity function" → 支持 C8/F4。
- NVIDIA Developer Blog 原文："x_q = clip(round(x/s), α_q, β_q)"、"x ≈ x' = s · x_q"（AbsMax）；"If the Llama2 7B model is stored in FP16/BF16, each parameter occupies 2 bytes, resulting in a total memory usage of approximately ~14 GB" → 支持 C2/N1。
- emergentmind.com/topics/mxfp4 与 zeroentropy.dev/concepts/mxfp4 原文："each group of 32 values shares a single 8-bit exponent scale (E8M0)"；"4-bit E2M1 floating-point layout"；"Storage: 8 bits scale + 32 × 4 bits = 136 bits per block"、"4.25 bits per element"；"The E2M1 codebook yields the set {0,±0.5,±1,±1.5,±2,±3,±4,±6}"；"scale is just an exponent shift, no FP multiply on the scale path" → 支持 C9/N2/N4 与 5.3 节全部论断。
- karam-nus.github.io/language-modelling/20_quantization_fundamentals 原文："Uses calibration dataset (~128–512 samples)" → 支持 N3。
- 一致性：自算对称范围、INT4=15 等级、有效位宽 136/32=4.25 与 128 块 4.0625、E2M1 16 码字→15 值，均与页面一致。

## 机械核对结果

- 代码：执行 5.2 折叠块内 Python，输出与页面「预期输出」逐行一致（E1 scale=0.8、x_q=[2,4,7]、error=[0.4,-0.2,0.0]；E2 scale=7.1429、x_q=[0,0,1,7]、error=[-1.2,-3.4,1.5429,0.0]；E3 块 A error=[0.4,-0.2,0.0]、块 B scale=7.1429）。
- 手算：2.1 表格、2.2 手算、3 章离群值手算、本章问题 [2.0,4.0,6.0] 全部复算通过（误差均 ≤ s/2）。
- `.dojo/scripts/validate.py wiki/quantization-basics/index.html` → validation ok。`dojo:topics=训练与优化` 在 AGENTS.md 固定大类内。overview.html ↔ index.html 互链；`../mxfp4-qat/index.html` 目标页存在；页面无 research/ 死链；无外部超链接。数学符号全部 KaTeX 渲染（正文仅有的 × → · 属 validate.py 明确排除的排版字符）。

## 问题

- [重要·技术] 5.1 节「补充：非对称 zero-point 的推导」折叠块（约 310 行）：同一折叠块内两处推导/结论与算式不符。其一，推导理由写「由于 round(α/s) 通常是整数附近的值，再 round 一次保证 z 是整数」——按定义 round(α/s) 本身即整数，z = q_min − round(α/s) 已是整数，不存在「再 round 一次」的必要；正确根据是 q_min ∈ ℤ 时 round(q_min − α/s) = q_min − round(α/s)。其二，验证句写「最小值精确重建」，但反量化得 s·round(α/s) ≈ α，只在 α/s 为整数时取等，应为「近似重建」。｜引文依据：Gholami §III-B Eq.2/3「Q(r)=Int(r/S)−Z」「r̃=S(Q(r)+Z)」；本页式 $s=(\beta-\alpha)/(q_{\max}-q_{\min})$、$z=\mathrm{round}(q_{\min}-\alpha/s)$ 正确，仅推导理由与「精确重建」表述有误。｜修复要求：把推导改为「解出 z = q_min − round(α/s)；因 q_min 为整数，round(q_min − α/s) = q_min − round(α/s)，故也写作 z = round(q_min − α/s)」，并把「最小值精确重建」改为「最小值近似重建（α/s 为整数时取等）」。｜修复：｜复验：
- [轻微·表述] 来源与范围说明（582、586 行）、5.2 代码折叠块（508、630 行）、4.2 有效位宽折叠块（344 行）：(a) 以「本页」为主语的自我指代四处——「本页用伪代码展示其机制」（两处）、「本页 $z=-Z$」「本页采用的现代统一写法」、「本页正文把误差归纳为…」；(b)「…成为主流的关键经济学」中「关键经济学」为生硬抽象名词（economics 直译）。｜引文依据：不适用。｜修复要求：(a) 统一改用「本文/本节」，如「正文用伪代码展示其机制」；(b) 改为「这是 per-block 在 INT4 量化中成为主流的关键原因」。｜修复：｜复验：
- [轻微·技术] 5.2 代码折叠块「观察重点」（507 行）与「构造示例」（612 行）：「误差从 $[-0.4,0]$ 量级跳到 $[-3.4,0]$ 量级」用区间写法描述误差，两端符号与实测不符——E1 误差为 $[+0.4,-0.2,0]$（最大幅度在 $+0.4$ 端），E2 正常值误差为 $[-1.2,-3.4,+1.543]$（含 $+1.543$）。｜引文依据：本页代码实测 error=E1 $[0.4,-0.2,0.0]$、E2 $[-1.2,-3.4,1.5429,0.0]$。｜修复要求：改为按幅度表述，如「正常值误差幅度从 $0.4$ 以内放大到 $3.4$」，或写出完整区间 $[-0.2,0.4]$ 与 $[-3.4,1.543]$。｜修复：｜复验：
- [轻微·技术] 4.2 节 callout（349 行）及其复述 overview.html（49 行）：「per-tensor(4-bit) 与 per-block(4-bit, 块 32) 的精度可能差几个数量级」无来源标注，且与本页自身示例（同数据下误差 0.4 对 1.2/3.4，约一个量级）不一致。｜引文依据：不适用（页面未给来源与条件）。｜修复要求：删除「几个数量级」，改为可支撑的表述（如「精度差距显著」），或补充可核对的来源与条件。｜修复：｜复验：
- [轻微·技术] 5.3 节（550 行）：「OpenAI gpt-oss 采用 MXFP4 存储混合专家权重」标注为 [C9, N2]，但 C9/N2 所列来源均为 MX 格式规范与格式介绍页（OCP 规范、emergentmind、zeroentropy、AMD ROCm blog），不覆盖 gpt-oss 这一部署事实。｜引文依据：C9/N2 条目内容（均只讨论 MX 格式本身）；该事实可另经 TensorRT-LLM gpt_oss README「MoE weights quantized with mxfp4. All the other weights are in bf16」核对。｜修复要求：为该句补一条覆盖 gpt-oss 的来源（OpenAI gpt-oss 模型卡或 TensorRT-LLM 文档），或删去该例。｜修复：｜复验：
- [轻微·技术] 来源与范围说明 C1/C3/C4（582、584、585 行）引用的 aiwiki.ai/wiki/quantization 本轮无法定位原文（该站返回 Vercel 反爬 429，无法取得可核对的片段）；C1/C3/C4 的实质内容已由 Gholami et al. 2021 §III-B/§III-C/§III-E 独立核实，该引用为冗余。｜引文依据：不可得（页面被反爬拦截）。｜修复要求：删除 aiwiki.ai 引用，或人工打开后补录可定位的原文片段，再行保留。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复
