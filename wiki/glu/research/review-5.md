<!-- review-meta
round: 5
page: wiki/glu/index.html
reviewed_content_sha256: 16628584bb943906
-->
# GLU 审查记录（第 5 轮）

- 页面版本：b5e311449b5dc9e565b76534bc508798f356cee3（工作树干净，无未提交改动）
- 审查时间：2026-09-14 16:55
- 审查者：独立子代理（未参与写作，未读取 research/ 下任何文件）
- 适用规范判定：`<meta name="dojo:type" content="concept">` → 依 `guides/concept/check.md`（并对照 `guides/concept/style-guide.md`）
- 来源获取方式：arXiv 原文（ar5iv HTML 版 1612.08083 / 2002.05202，逐节定位原文片段）+ 页内数值全部复算
- 已完整阅读章节（含折叠块与图注）：引言 / 核心问题（5 条）/ 最容易误解 / 1. 为什么需要"门"+本章问题 / 2. GLU 的公式与手算（含「展开：门 → 1 与门 → 0 的退化」折叠块）+本章问题 / 3. 为什么 GLU 给梯度留了一条线性通路（含「补充：链式法则步骤」折叠块）+本章问题 / 4. GLU 家族（4.1 派生规则、4.2 记号差异、4.3 塞进 Transformer FFN，含参数量验算折叠块）+本章问题 / 5. 经验结论与边界（5.1）+本章问题 / 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）/ 图注与内联 SVG 结构图

## 回源核对（每条来源论断均定位到原文片段）

- C1/F1 定义（Dauphin §2 Eq.(1)）：原文 §2 "Approach" Eq.(1) 即 `hl(X)=(X∗W+b)⊗σ(X∗V+c)`。页面写法一致。
- C3/F4 GLU 梯度（Dauphin §3 Eq.(3)）：原文 Eq.(3) `∇[X⊗σ(X)]=∇X⊗σ(X)+X⊗σ′(X)∇X`，与页面逐字一致；原文紧随其后 "a multiplicative skip connection which helps gradients flow through the layers"，页面"乘性跳连"用词与归因一致。
- C4/F3 GTU 梯度（Dauphin §3 Eq.(2)）：原文 Eq.(2) `∇[tanh(X)⊗σ(X)]=tanh′(X)∇X⊗σ(X)+σ′(X)∇X⊗tanh(X)`，与页面一致；原文 "Notice that it gradually vanishes as we stack layers because of the downscaling factors tanh′(X) and σ′(X)" 支持"GTU 第一项带 tanh′、随 |X| 增大趋零"。原文限定语 "without downscaling for the activated gating units in σ(X)" 与页面 §3 末段"对激活的门单元不额外缩放"一致（页面省略 "in σ(X)"，不改含义）。
- GTU 命名：原文 §3 "the gating of which we dub gated tanh unit (GTU)"，页面 "GTU（Gated Tanh Unit）" 一致。
- C5/F5 Bilinear（Dauphin §5.3）：原文 §5.3 "Another variation of GLUs are bilinear layers … (Mnih & Hinton 2007)"，形式 `hl(X)=(X∗W+b)⊗(X∗V+c)`，与页面一致。
- C6/F6 变体（Shazeer §2 Eq.(5)）：原文 Eq.(5) `ReGLU=max(0,xW+b)⊗(xV+c)`、`GEGLU=GELU(xW+b)⊗(xV+c)`、`SwiGLU=Swish_β(xW+b)⊗(xV+c)`，与页面逐字一致；Shazeer 记法"激活在 W 分支"对应原文 Eq.(4)，与页面 4.2 表一致。
- C7/F7/F8 FFN_GLU（Shazeer Eq.(6)）与 2/3 缩放：原文 Eq.(6) `FFN_GLU(x,W,V,W₂)=(σ(xW)⊗xV)W₂`，与页面一致；原文 "All of these layers have three weight matrices, as opposed to two for the original FFN" 与 "we reduce the number of hidden units d_ff … by a factor of 2/3" 支持 §4.3 推导；基线 FFN 公式见于 §1 的无偏置版（Eq.(2)），页面标注"§1，无偏置版本"一致。
- C8/N1 数字（Shazeer Table 1，取 524,288 步一列）：原文 ReLU 1.677 / GELU 1.679 / Swish 1.683 / GLU 1.663 / Bilinear 1.648 / ReGLU 1.645 / SwiGLU 1.636 / GEGLU 1.633 —— 页面正文表与 N1 逐值一致，最优 GEGLU 1.633。原文 §3.2 "The GEGLU and SwiGLU variants produce the best perplexities" 支持页面"GEGLU 与 SwiGLU 最优"。Table 1 caption "Heldout-set log-perplexity … segment-filling task. All models are matched for parameters and computation." 支持页面"heldout log-perplexity / segment-filling 任务 / 参数与计算量匹配"；§3.1 支持 d_model=768、h=12、d_ff=3072→2048。§4 结语与页面引用逐字一致："We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."
- N2（Shazeer §3.1）：原文 d_model=768、h=12、FFN 隐藏维 3072，GLU 变体 2048，支持页面 3072→2048。
- 复算全部通过：σ(1.0)=0.73106、σ(−0.5)=0.37754、h=[0.7311, 0.18877]；σ(10)=0.999955、σ(−5)=0.006693、h≈[1.0, 0]；2×768×3072=4,718,592、3×768×3072=7,077,888（恰为基础 1.5 倍）、3×768×2048=4,718,592；3072×2/3=2048、4d×2/3=(8/3)d。无一处算式与结论不符，无同页两处矛盾，正文/summary/overview/图注数字与引文编号一致（C1–C9、F1–F8、N1–N2 全部在来源章节有定义且被引用）。
- 结构复核：KJ 公式/符号全文单义（`⊗` 逐元素乘、`σ` 门支激活、`W↔V` 标签差已显式说明并全文一致）；SVG 图为内联 SVG，数学符号全部置于 `<foreignObject>`、`<text>` 内仅纯文字；无 ASCII 近似写法；description 为纯文本无 `$`，`alt` 属性无 `$...$`；`../situ-glu/index.html`、`../swiglu/index.html` 均真实存在；无"（待生成）"占位；两级问题块每题均有 `解答：` 折叠块；`validate.py` 返回 `validation ok`。

## 问题

- [轻微·格式] 全文（`description` 第 6 行、`dojo:summary` 第 7 行、正文第 118 / 406 / 446 行）：同一页内变体名拼写不统一，同时出现 "GEGLU" 与 "GeGLU"——第 118 行写 "SwiGLU/GeGLU/ReGLU"，紧接其解答的第 120/121 行即写 "GEGLU"，同段翻转。页面其余 11 处及 sibling 页 `wiki/swiglu/index.html`、`wiki/glu/overview.html` 均用 "GEGLU"。｜引文依据：Shazeer §2 Eq.(5) 原文写作 "GEGLU"（`GEGLU(x,W,V,b,c)=GELU(xW+b)⊗(xV+c)`）。｜修复要求：将本页 5 处 "GeGLU"（第 6、7、118、406、446 行）统一改为 "GEGLU"。｜修复：｜复验：
- [轻微·表述] §4.1 第 382 行："虽无线性门控激活，逐元素乘本身仍是……"——"线性门控激活"搭配自相矛盾：Bilinear 去掉的是门控激活本身，不存在"线性/非线性"之分，读者可能误读为"Bilinear 连线性也去掉了"。｜引文依据：不适用。｜修复要求：删去多余的"线性"，改为"虽无门控激活"（或"虽去掉了门控激活"）。｜修复：｜复验：

## 结论

- 处置：可发布。2 条均为轻微，不影响核心正确性与主线理解，建议随手修复；核心结论、公式、数字与全部引文编号经原文逐条核对无误。
- 统计：阻断 0 / 重要 0 / 轻微 2