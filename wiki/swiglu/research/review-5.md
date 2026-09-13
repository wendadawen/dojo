<!-- review-meta
round: 5
page: wiki/swiglu/index.html
reviewed_content_sha256: c6057c5837a7ec44
-->
# SwiGLU 审查记录（第 5 轮）

- 页面版本：33175a644c41401bc7b5fcbb3c64801241a53883（git hash-object）
- 审查时间：2026-09-13 20:26
- 审查者：独立子代理（未参与写作与前序轮次审查）
- 已完整阅读章节：核心问题；最容易误解；1. 从 GLU 到 SwiGLU——为什么要换门（含本章问题）；2. SwiGLU 的公式、Swish 定义与手算——逐符号拆解与边界值验证（含全部折叠块与本章问题）；3. 把 SwiGLU 塞进 Transformer FFN——三矩阵与 $2/3$ 缩放（含折叠块与本章问题）；4. 经验结论与边界——SwiGLU 为什么成为 LLM 标配（4.1–4.5，含本章问题）；来源与范围说明（含辅助解释、简化条件）

## 已核对来源（依据，供复验）

- Shazeer 2020（arXiv:2002.05202, ar5iv HTML）原式编号逐条核对一致：Eq.(1) FFN 通用式；Eq.(2) FFN_ReLU；Eq.(3) = FFN_GELU + FFN_Swish（`FFN_Swish(x,W1,W2)=Swish_1(xW1)W2`）；Eq.(4) = GLU + Bilinear（`GLU(x,W,V,b,c)=σ(xW+b)⊗(xV+c)`）；Eq.(5) = ReGLU + GEGLU + SwiGLU（`SwiGLU(x,W,V,b,c,β)=Swish_β(xW+b)⊗(xV+c)`）；Eq.(6) = `FFN_SwiGLU=(Swish_1(xW)⊗xV)W2`。页面 C1/C5/C3 对 Eq.(5)/(6)/(3) 的编号引用全部正确。
- §2 末段原文："All of these layers have three weight matrices...we reduce the number of hidden units $d_{ff}$...by a factor of $\frac{2}{3}$..."；§3.1："12 layers...$d_{model}=768$...$d_{ff}=3072$...reduce the hidden layer to $d_{ff}=2048$"；§3.2："The GEGLU and SwiGLU variants produce the best perplexities."；§4："We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence." — 均逐字一致。
- Table 1 全部 8 行 524,288 步值核对一致：ReLU 1.677 / GELU 1.679 / Swish 1.683 / GLU 1.663 / Bilinear 1.648 / ReGLU 1.645 / SwiGLU 1.636 / GEGLU 1.633；65,536 步标准差 SwiGLU 1.944(0.010)、GEGLU 1.942(0.004) 与页面一致。
- 自算复核无误：Swish(1)=0.7311、Swish(−1)=−0.2689、Swish 极小点 z=−σ/σ′ 时 z≈−1.278、值≈−0.278；手算例 xW=[1,−1]、Swish 门=[0.7311,−0.2689]、输出=[0.7311,−0.1345]、GLU 对照=[0.7311,+0.1345]；参数量 2d·4d=8d²=134,217,728 与 3d·(8/3)d=8d² 相等、未缩时为 12d² 多 50%；3072×2/3=2048、8/3×4096≈10922.67、LLaMA-7B/Qwen1.5-7B 取 11008（Qwen1.5-7B config.json hidden_size=4096 / intermediate_size=11008 已核）。
- 外部来源核对一致：LLaMA 2023 §2.2 "We replace the ReLU non-linearity by the SwiGLU activation function, introduced by Shazeer 2020"、"dimension of (2/3)4d instead of 4d as in PaLM"；Gemma §2 "The standard ReLU non-linearity is replaced by the approximated version of the GeGLU activation function."；PaLM（2204.02311）"We use SwiGLU activations (Swish(xW)·xV)...(Shazeer 2020)"；Dauphin 2017 §2 Eq.(1) `h_l(X)=(X*W+b)⊗σ(X*V+c)`（σ 在 V 分支，印证页面 Dauphin 主记法）、§5.3 讨论 bilinear 并引 Mnih & Hinton 2007；K3（2607.24653）§2.3.2 "both multiplicative factors in SwiGLU are unbounded, so coincident large coordinates can produce activation outliers and increase overflow risk in low-precision arithmetic" 与 `softcap(x,β)=β·tanh(x/β)` 公式一致。
- 机械项：页面无 research/ 路径引用；本地链接（glu、situ-glu、overview、libs、index）全部解析；无"（待生成）"占位；SVG 图内公式在 foreignObject 由 KaTeX 渲染、`<text>` 无 ASCII 数学；`.dojo/scripts/validate.py wiki/swiglu` 返回成功。

## 问题

- [重要·技术] §1「1. 从 GLU 到 SwiGLU」第 2 段与第 3 段（L149 / L151）相邻两句：同一对象 GLU 的 $W/V$ 标签互相交换且未在切换处标注，违反符号全文单义。第 2 段按 Dauphin 记法写 GLU「门 $\sigma(xV+c)$ ...值分支 $xW+b$」（$W$=值支、$V$=门支）；紧接第 3 段写家族「统一形式为 $\mathrm{激活}(xW+b)\otimes(xV+c)$」（$W$=门支、$V$=值支）。读到此处的读者尚未看到图注（其后）或 §2 记法说明，$W$ 在两段之间改变所指。｜引文依据：Dauphin 2017 Eq.(1) `h_l(X)=(X*W+b)⊗σ(X*V+c)`（σ 在 V 分支）与 Shazeer 2020 Eq.(4)/(5) `σ(xW+b)⊗(xV+c)`（激活在 W 分支）确为 $W\leftrightarrow V$ 互换；页面 L171「门分支线性部分 $xW=-1$」使用 Shazeer 记法。｜修复要求：在第 3 段「统一形式为…」处补一句明确注明此处起改用 Shazeer 记法（激活在 $W$ 分支），与上一段 GLU 的 Dauphin 记法（$\sigma$ 在 $V$ 分支）标签互换，使 $W/V$ 在切换点有显式标记。｜修复：｜复验：

- [重要·技术] §2 折叠块「展开：同输入下 GLU 与 SwiGLU 的逐维对照」末段（L306）：「Swish 保留它的负号并放大到 $-0.27$」与本页对"放大"的定义及数值互相矛盾。L171 定义「放大（输出大于 $1$ 的正值让值绝对值变大）」；而此处门支线性部分 $-1.0\to$ Swish 门 $-0.27$，绝对值由 $1.0$ 降到 $0.27$，属"压低"而非"放大"。同页 L342 对同一步骤写作「Swish 保留负号得 $-0.2689$」（无"放大"）。｜引文依据：Swish $(-1)=-1\times\sigma(-1)=-1\times0.2689=-0.2689$（页面自身 C2/F1 定义可得）；L171 放大定义、L342「保留负号得 $-0.2689$」。｜修复要求：删去"放大"改为"保留负号"，使该句与 L342 表述及页面"放大"定义一致，不得保留"放大"这一与数值相反的说法。｜修复：｜复验：

- [轻微·来源] 来源与范围说明 C10：对 Mistral、Qwen、DeepSeek 仅写「在各自论文与公开配置中采用 SwiGLU（多源交叉）」，未给出可定位的章节号 / 文件路径；同条中 LLaMA（§2.2 引文）与 Gemma（§2 引文）已给出原文片段。｜引文依据：C10 无 Mistral/DeepSeek 片段；对照可核项——LLaMA 2023 §2.2、Gemma §2、PaLM「We use SwiGLU activations」、Qwen1.5-7B config.json（hidden_size=4096 / intermediate_size=11008）。｜修复要求：为 Mistral、DeepSeek 补可定位锚点（论文章节号或 HF config.json 路径与字段值），无法补则删除相应模型名。｜修复：｜复验：

- [轻微·表述] §3 开头「这一章看它怎么塞、以及为什么 LLaMA 内部维度取 $\tfrac83 d$。」与 §4 开头「本章看 Shazeer 的实验说了什么、没说什么…」：以"这一章看/本章看"预告章节内容，属元话语式的路标句（与 check.md 所举"下面来看…"同类）。｜引文依据：不适用。｜修复要求：改为直接陈述（如"本节推导 $\tfrac83 d$ 的来源"）或删去预告语，保留原有的章节衔接句（如"能部署不等于经验上更优"）以维持逻辑过渡。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 2
- 处置：修复（无阻断项；两条重要问题均为页面内部一致性/符号单义问题，修复范围限于所涉句子及直接受影响的引用处，不涉及范围或大纲变更）
