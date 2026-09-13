<!-- review-meta
round: 3
page: wiki/swiglu/index.html
reviewed_content_sha256: d910db9a40fddd44
-->
# SwiGLU 审查记录（第 3 轮）

- 页面版本：aba7efc7aae95cc3a44f97b78d9a8ef7e399b10a
- 审查时间：2026-09-13 19:11
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：引言与主要依据 → 核心问题（5 条含解答折叠块）→ 最容易误解 → 1. 从 GLU 到 SwiGLU（含结构图 SVG、本章问题）→ 2. SwiGLU 的公式、Swish 定义与手算（含两个折叠块、本章问题）→ 3. 把 SwiGLU 塞进 Transformer FFN（含折叠验算、本章问题）→ 4. 经验结论与边界 4.1–4.5（含本章问题）→ 全文总结 → 来源与范围说明（论断与来源 C / 公式与来源 F / 外部数字与实验条件 / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）
- 外部来源获取方式：Shazeer 2020 arXiv:2002.05202 取 ar5iv HTML 全文（下载后转纯文本逐句核对）；LLaMA arXiv:2302.13971 取 ar5iv HTML；Gemma arXiv:2403.08295 取 ar5iv HTML；PaLM arXiv:2204.02311 取 ar5iv HTML。引用 `research/` 不得读，未读。
- 机械验证：`python3 .dojo/scripts/validate.py wiki/swiglu/index.html` → `validation ok`。全文（去除 `$$…$$` 与 `$…$` 后）无非 ASCII 数学字符残留（仅剩 `§`、`·`、`–` 与中文标点）。Table 1 八行数值、手算例、参数量等式、`8/3 d` 与 `11008` 取整均已重算一致。

## 问题

- [阻断·来源] 来源与范围说明·论断与来源（C）C3（第 535 行）；同一论断在正文第 157、261、140 行重复出现：C3 把 `"we use β=1 in our experiments"` 当作 Shazeer 2020 §1 末段的英文原句加引号引用，该句在来源中根本不存在。｜引文依据：ar5iv 版 §1 全文只有 `Swishβ(x)=xσ(βx) [Ramachandran et al. 2017]` 与 Eq.(3) `FFN_Swish(x,W1,W2)=Swish_1(xW1)W2`；全文不出现 "experiment" 一词（唯一涉及实验的是 §3.2 `Identically to [Raffel et al. 2019], we pre-train for 524,288 steps on the span-filling objective on the C4 dataset.`），也不存在任何形如 `we use β=1` 的句子。｜修复要求：删除该伪造英文引文。`β=1` 只能由 §1 Eq.(3)、§2 Eq.(6) 写成 `Swish_1` 直接推出，须改写为「由 Eq.(3)/Eq.(6) 写作 $\mathrm{Swish}_1$ 可推得实验取 $\beta=1$」这类明确标注的推断并给出公式定位；不得保留引号内的英文原句。｜修复：｜复验：

- [重要·来源] 正文第 157、261、140 行（C3 被引用于此处）：`现代 LLM 部署（LLaMA、PaLM 等）也固定 β=1`——该部署论断被挂在 C3（Shazeer 2020）上，但 Shazeer 全文不提任何 LLM 部署。另，正文第 157、261、140 行写 `Shazeer §1 与 §3.1 实验固定 β=1`，与 C3 自述 `§3.1 实验设置未提 β 调参` 自相矛盾（§3.1 确实不含 β 的任何字样）。｜引文依据：LLaMA 论文 §2.2 只写 `We replace the ReLU non-linearity by the SwiGLU activation function, introduced by Shazeer 2020 to improve the performance. We use a dimension of 2/3·4d instead of 4d as in PaLM.`，未出现 β；Shazeer §3.1 全文 `The FFN layers have hidden size dff=3072. ... we reduce the hidden layer to dff=2048`，无 β。｜修复要求：为「部署固定 β=1」补可定位来源（如 LLaMA/PyTorch 源码用 `F.silu` 即 $\mathrm{Swish}_1$ 的路径与行号），无法给出则删除该部署论断；同时把「§1 与 §3.1 实验固定 β=1」改为与来源一致的说法（§1 Eq.(3)/§2 Eq.(6) 用 $\mathrm{Swish}_1$；§3.1 未提 β）。｜修复：｜复验：

- [重要·来源] 正文第 467 行（另见第 515、471、128、138 行与 overview.html 第 4 节）：`工具链锁定：HuggingFace Transformers、llama.cpp、vLLM 等主流训练/推理栈都为 SwiGLU 形状优化过；新模型改回 ReLU 或 GEGLU 不再"免费"`——这是一条被当作事实写进结论的机制论断，全页无任何来源支持；§4.3 的结论（第 471 行）把它与 PaLM/LLaMA 采用并列为本页四支柱之一。｜引文依据：Shazeer 2020 全文无此内容；页面来源章节（C1–C12、F1–F5、N1–N2）无一条指向 HF/llama.cpp/vLLM 的实现或 kernel。｜修复要求：补可定位来源（如 vLLM/SGLang 的 SiLU-and-mul 融合算子源码路径与行号、llama.cpp 的 SILU 分支），无法定位则删去该表述或明确标注为本文的工程推断。｜修复：｜复验：

- [轻微·来源] 正文第 448、468、508 行与第 128 行：`两者差距 0.003，在实验噪声内`被写成由 [C8, N1] 支持的来源结论。｜引文依据：Table 1 仅在 65,536 步一组用括号给出 inter-run std（`FFN_SwiGLU 1.944 (0.010)`、`FFN_GEGLU 1.942 (0.004)`），524,288 步一行无 std；「0.003 落在噪声内」是页面自己的推断（0.003 < 0.010 成立）。｜修复要求：标注为本文推断，或引用 Table 1 中 65,536 步那组的 inter-run std 数值作为依据。｜修复：｜复验：

- [轻微·格式] 来源与范围说明第 3 个小节标题（第 556 行）为 `外部数字与实验条件`，缺固定后缀。｜引文依据：不适用（guides/concept/style-guide.md §1 固定命名为 `外部数字与实验条件（N）`）。｜修复要求：标题改为 `外部数字与实验条件（N）`。｜修复：｜复验：

- [轻微·格式] 正文第 277、299 行：`构造示例。 取` / `构造示例。 同` 句号后多一个半角空格。｜引文依据：不适用｜修复要求：删除句号后的多余空格。｜修复：｜复验：

- [轻微·格式] `<head>` 第 7 行 `dojo:summary` 用 `$xW_{gate}$`、`$xW_{up}$` 记两套投影，正文（第 253、259、262、263 行等）一律用 `$W$`/`$V$`（Shazeer 记法）。｜引文依据：不适用（style-guide.md §11 要求同一变量全页同一种写法）。｜修复要求：把 summary 的 `W_{gate}`/`W_{up}` 统一为正文的 `W`/`V`，或反之并同步全文。｜修复：｜复验：

- [轻微·表述] 正文第 426 行：`会算了、会部署了，最后一章看 Shazeer 的实验说了什么、没说什么，以及为什么社区选了 SwiGLU 而不是其它变体`——「会算了、会部署了」为口语化临场叙事。｜引文依据：不适用｜修复要求：改为中性陈述句，例如「公式与 FFN 部署已确立，本章看 Shazeer 实验的结论与边界」。｜修复：｜复验：

- [轻微·来源] 正文第 379 行：`除 Google 的 Gemma 系列外，几乎所有领先开权重基础模型都用 SwiGLU 作 FFN 激活`——「几乎所有…基础模型」是无来源的全局论断，C10 只列了 LLaMA/PaLM/Mistral/Qwen/DeepSeek 与 Gemma（Gemma 用 GeGLU 已核对属实：ar5iv 版 §2 `GeGLU Activations (Shazeer 2020). The standard ReLU non-linearity is replaced by the approximated version of the GeGLU activation function.`）。｜引文依据：Shazeer 2020 无此分布性结论；C10 自身也只写「多源交叉」，未给各模型论文的章节/页码定位。｜修复要求：把「几乎所有…」限定为可枚举的模型清单，或补来源；C10 应给出各模型论文采用 SwiGLU 的可定位位置。｜修复：｜复验：

- [轻微·格式] 来源与范围说明（F）中 `F4 参数量等式 $d_{ff}'=\tfrac23 d_{ff}$`（第 552 行）在正文无任何 `<sup>[F4]</sup>` 引用，与 style-guide.md §6「与来源章节双向对应」不符（正文相应位置引用的是 C6/F5）。｜引文依据：不适用｜修复要求：删除 F4，或在正文第 372 行处的参数量等式后补 `[F4]`。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 7
- 处置：修复
- 说明：阻塞项为 C3 伪造的英文直接引文，须知来源不支持即删除或降级；两项重要问题为「部署固定 β=1」的错误归因与「工具链锁定」的无来源机制论断。其余核对通过：§2 Eq.(5) `SwiGLU(x,W,V,b,c,β)=Swishβ(xW+b)⊗(xV+c)`、Eq.(6) `FFN_SwiGLU(x,W,V,W2)=(Swish1(xW)⊗xV)W2`、§2 末段 2/3 引文、§4 `divine benevolence` 引文与所在节号（§4 Conclusions）、§3.1 设置（12 层、d_model=768、3072→2048）均与来源逐字一致；Table 1 八行 524,288 步数值（1.677/1.679/1.683/1.663/1.648/1.645/1.636/1.633）全部核对无误；手算例、Swish 边界值与极小点（z≈−1.278、−0.278）、参数量等式（8d²=134,217,728，12d² 多 50%）、`8/3·4096≈10923` 与 LLaMA-7B `11008=43×256` 均可复算一致；LLaMA §2 采用 SwiGLU 并引用 PaLM 属实。
