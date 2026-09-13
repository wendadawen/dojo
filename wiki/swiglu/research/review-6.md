<!-- review-meta
round: 6
page: wiki/swiglu/index.html
reviewed_content_sha256: 89648ea03e84fc4e
-->
# SwiGLU 审查记录（第 6 轮）

- 页面版本：08934be8a06c5b635a012da610a39d89afac6af7（git hash-object；工作树 clean）
- 审查时间：2026-09-13 21:19
- 审查者：独立子代理（未参与写作与前序轮次）
- dojo:type：concept → 适用 guides/concept/check.md
- 已完整阅读章节：核心问题（5 题）／最容易误解（5 条）／1. 从 GLU 到 SwiGLU／2. SwiGLU 的公式、Swish 定义与手算／3. 把 SwiGLU 塞进 Transformer FFN／4. 经验结论与边界（4.1–4.5）／全文总结／来源与范围说明（C1–C12、F1–F5、N1–N2、构造示例、类比边界、简化条件）／全部折叠块与图注

## 问题

本轮未发现阻断、重要或轻微问题。

## 核对记录（引证依据）

技术/来源维度，逐条回源：

- SwiGLU 定义 §2 Eq.(5)：ar5iv 2002.05202 原文 `SwiGLUβ(x,W,V,b,c)=Swishβ(xW+b)⊗(xV+c)`，与页面公式逐符号一致。FFN_SwiGLU §2 Eq.(6) `(Swish1(xW)⊗xV)W2` 一致。
- Swish 定义在 §1：原文 `Swishβ(x)=x σ(β x)`；§1 Eq.(3) 与 §2 Eq.(6) 均写作 `Swish1`（β=1），页面 C3 将其明确标为「本页推断」，处理正确。
- §2 末段 2/3 规则：原文 `To keep the number of parameters and the amount of computation constant, we reduce the hidden units dff by a factor of 2/3 ...`，与页面引文框逐字一致。
- §3.1：`T5 base、d_model=768、12 encoder/decoder layers、dff 3072（基线）/2048（GLU 变体）`，与页面 C7 一致。
- Table 1（524,288 步）：ReLU 1.677 / GELU 1.679 / Swish 1.683 / GLU 1.663 / Bilinear 1.648 / ReGLU 1.645 / SwiGLU 1.636 / GEGLU 1.633 —— 页面表格八个数字全对，无一处偏差；「GEGLU 1.633 最优、SwiGLU 1.636 紧随、差 0.003」成立。
- Table 1（65,536 步 run 间标准差）：SwiGLU 1.944 (0.010)、GEGLU 1.942 (0.004)；524,288 步一组无括号 —— 页面「65,536 步一组给标准差、524,288 步一组未给」与原文一致；据此推断 0.003 落在噪声内已标注「本页推断」。
- §3.2 引文：`The GEGLU and SwiGLU variants produce the best perplexities.`（章节号 §3.2 核对无误）。§4 结语：`We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.`（§4 核对无误）。
- LLaMA（2302.13971）§2.2 原文：`We replace the ReLU non-linearity by the SwiGLU activation function, introduced by Shazeer 2020 to improve the performance.` 及 `We use a dimension of 2/3 4d instead of 4d as in PaLM.` —— 与 C10/C11 及 8/3 d 推导一致，且可验证其引用 PaLM。
- Gemma（2403.08295）§2 原文：`The standard ReLU non-linearity is replaced by the approximated version of the GeGLU activation function.` —— 与 C10 引文逐字一致。
- config.json（HuggingFace raw）：Mistral-7B 4096/14336/silu；Qwen1.5-7B 4096/11008/silu；DeepSeek-V2 5120/12288/silu —— 与 C10 所列三组数字完全一致。
- Dauphin 2017（1612.08083）：GLU 定义在 §2 Eq.(1)；Bilinear 在 §5.3「Non-linear Modeling」以 `(X∗W+b)⊗(X∗V+c)` 形式出现，参考文献含 Mnih & Hinton 2007 —— 家族表 Bilinear 行来源标注正确。
- K3 softcap：与 ../../wiki/situ-glu/index.html 的 §2.3.2 / `softcap(x,β)=β tanh(x/β)` / Stable LatentMoE 路由分支表述一致，C12 归属无误。

公式与数字复算（全部通过）：

- Swish 边界值：σ(1)=1/(1+e⁻¹)≈0.7311、σ(-1)≈0.2689，Swish(0)=0、Swish(1)≈0.7311、Swish(-1)≈-0.2689。Swish 极小值驻点式 z=-σ(z)/σ'(z) 正确，z≈-1.278 处 ≈-0.278 与数值一致。
- 手算例：xW=[1.0×1+0.5×0, 1.0×0+0.5×(-2)]=[1.0,-1.0]；Swish→[0.7311,-0.2689]；xV=[1.0,0.5]；逐元素积=[0.7311,-0.1345]。对照 GLU 输出 [0.7311,+0.1345]。全部可复算，与页面数字一致。
- 参数量：8d²=8×4096²=134,217,728=2d·4d；12d²=201,326,592=3d·4d（比基线多 50%）；3d·(8/3)d=8d²，与基线相等；3×(8/3)=8=2×4。8/3×4096≈10922.67，向上取 256 倍数得 11008（256×43）——与 LLaMA-7B 实际值一致，验算无误。
- 经验增益 1.677-1.636=0.041，与页面一致。

一致性与功能维度：

- 同一数字在 description/summary/正文/折叠块/overview.html 间全部一致（1.633/1.636/0.003、0.731/0.7311、-0.269/-0.2689、8/3 d、3072→2048、11008、134,217,728）。无正文与 summary/overview 冲突。
- 标题级「核心问题」5 题与四章「本章问题」11 题均有解答折叠块，答案独立可读且与正文结论一致；每题答案均指明完整论证所在章节。
- 数学符号全部由 KaTeX 渲染；全文（标题、summary、正文、列表、表格）grep 未见裸露 Unicode 数学字符（仅 nav 分隔符「·」与 HTML 注释中的 σ，均非渲染文本）；符号全文单义（W/V 门值记法在改用处显式声明，两处 SVG 图按对应记法绘制且与正文一致）。
- 结构图为内联 SVG，公式写在 <foreignObject> 内、<text> 仅含文字标签，无等宽框线图、无 ASCII 近似；图注正文解释了记法差异与节点含义。
- 前置概念链接 ../../wiki/glu/index.html 与下游 ../../wiki/situ-glu/index.html 均真实存在；无「（待生成）」占位；overview.html 与 index.html 相互链接。
- alt 属性无 `$...$`；description 为纯文本；dojo:topics=模型结构、dojo:tag=网络结构 均在 ALLOWED_TOPICS / ALLOWED_TAGS 词表内。
- .dojo/scripts/validate.py wiki/swiglu/index.html 返回 `validation ok`。

表述维度（逐段通读含折叠块与图注）：未发现会话指代（我/我们/你）、调试或复现踩坑叙事、临场评价、AI 拼接腔（抽象名词堆叠、公文连接词、把「场景」当术语）。三处以「本章…」开头的章节收束句为对本章内容的回顾性小结，承担 check.md 2.1 要求的章间逻辑衔接，非「本页将…」式前瞻元话语；「下文除公式外一律取 β=1」为必要的记号约定声明；均判定为合格表述，不计为问题。所有无直接来源的支持性判断（工具链惯性、0.003 在噪声内）均以「本页推断」显式标注，构造示例均标注「数字为教学构造」。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布
