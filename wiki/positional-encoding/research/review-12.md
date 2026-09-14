<!-- review-meta
round: 12
page: wiki/positional-encoding/index.html
reviewed_content_sha256: 6db9d83483acc38a
-->
# 位置编码基础审查记录（第 12 轮）

- 页面版本：88fd66e406f33e0c6580366c952d552b775226d9（`git hash-object wiki/positional-encoding/index.html`）
- 审查时间：2026-09-14 17:57
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 已完整阅读章节：核心问题；常见误解；引言；1. 为什么 Transformer 需要位置编码；2. 绝对正弦位置编码（含「展开：$d_{model}=4$ 手算」与「补充：和角公式推导」）；3. 可学习绝对位置编码（含「展开：Table 3 row (E)」）；4. 相对位置编码（含「补充：T5 分桶」）；5. 四类方案对比与 NoPE 选择（含 5.1）；来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）
- 核对来源版本：Vaswani et al. 2017, arXiv:1706.03762（ar5iv 正文版）；Raffel et al. 2020, T5, arXiv:1910.10683（ar5iv 正文版，JMLR 21 卷）；Su et al. 2021, arXiv:2104.09864（RoFormer）；Press et al. 2021, arXiv:2108.12409（ALiBi）；DeepSeek-V2, arXiv:2405.04434；Kazemnejad et al. 2023, arXiv:2305.19466；Chen et al. 2023, arXiv:2306.15595（PI）；Peng et al. 2023, arXiv:2309.00071（YaRN）；Gehring et al. 2017, arXiv:1705.03122；Kimi K3 技术报告 arXiv:2607.24653v2 与官方 `config.json`。逐一打开核对，页面所标编号与来源标题、内容一一对应。
- 机械项：`.dojo/scripts/validate.py wiki/positional-encoding/index.html` 返回 `validation ok`；`dojo:type=concept`、`dojo:topics=注意力机制`、`dojo:tag=位置编码` 均在 `catalog_builder.py` 词表内；页面引用的 `wiki/rope`、`wiki/nope`、`wiki/standard-attention`、`wiki/kimi-k3`、`wiki/mla` 目录均真实存在；`index.html` 与 `overview.html` 相互链接；五个「本章问题」h3 均有唯一 id（`questions-why-pos/sinusoidal/learned/relative/comparison`），无重复锚点。
- 图内数值：本页唯一 `<figure class="diagram">` 是 HTML 数据流堆叠图（`dg-stack`/`dg-layer`），非坐标图/刻度图，无可做像素测量的数值读数；已改为按文字逐句核对数据流方向。`libs/dojo-concept.css` 中存在 `.dg-stack`/`.dg-layer` 等样式。

## 问题

- [轻微·来源] 来源章节「论断与来源（C）」C12：括注把 NoPE 的提出归给 Kazemnejad et al. 2023｜引文依据：该论文正文写 "We also consider removing the positional encoding (NoPE) to better understand its role."，并把「无位置编码仍可工作」这一观察追溯到既有工作 "This was observed early on by Shen et al. 2018 and later explained by Tsai et al. 2019."；论文自述贡献是系统比较、理论表征与注意力模式分析（"Transformers without positional encoding (NoPE) outperform all explicit positional encoding schemes."、"We show that NoPE is theoretically capable of representing both absolute and relative PEs."），并未主张首次提出 NoPE。页面 C12 主论断（NoPE 依赖因果掩码提供隐式位置）成立，仅括注「提出」一词偏强｜修复要求：把「（对 decoder-only Transformer 提出 NoPE）」改为不主张首创的表述，例如「（系统评估并推荐 decoder-only Transformer 使用 NoPE）」｜修复：｜复验：

## 核对通过项（引文依据）

- 正弦公式与线性性质（F1/C4）：ar5iv 原文 "PE_{(pos,2i)} = sin(pos/10000^{2i/d_model})"、"PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_model})"；"since for any fixed offset k, PE_pos+k can be represented as a linear function of PE_pos"；选 sin/cos 的理由原文含 "we hypothesized it would allow the model to easily learn to attend by relative positions"。页面公式、旋转角 $-\Delta\omega_i$ 与和角公式推导均可复算、符号全页单义。
- 波长范围（C3/N2）：原文 "The wavelengths form a geometric progression from 2π to 10000·2π."。页面「$i=0$ 波长 $2\pi$（约 6 个位置转一圈）」「$d_{model}=512$ 时 $i=255$ 波长 $10000\cdot2\pi$（约 62832）」与该原句一致（论文本身即取整表述；精确值 $\omega_{255}=1/9647$、波长约 60611，页面写 $\omega_{255}\approx1/10000$ 未越出论文舍入）。
- $d_{model}=4$ 手算（构造示例，非来源事实）：$\omega_0=1$、$\omega_1=0.01$；$PE_1=(0.8415,0.5403,0.0100,1.0000)$、$PE_2=(0.9093,-0.4161,0.0200,0.9998)$ 与表格、核心问题解答、公式表格三处数值逐一复算一致；多尺度差值 0.0678/0.9564/0.0100/-0.0002 复算一致。
- Table 3 row (E)（C5/N1）：ar5iv Table 3 中 base 行 BLEU(dev, newstest2013)=25.8，行 "(E) positional embedding instead of sinusoids" BLEU=25.7；big 行 BLEU=26.4(dev)，Table 2 big EN-DE=28.4(test newstest2014)。页面「正弦 25.8 对可学习 25.7」「big 28.4 出自 Table 2」与之一致；原文 "found that the two versions produced nearly identical results (see Table 3 row (E))" 与 "may allow the model to extrapolate" 均在页面对应处引用。
- T5 相对偏置（C7/F4）：§2.1 原文 "each 'embedding' is simply a scalar that is added to the corresponding logit used for computing the attention weights"；"we use 32 embeddings ... with ranges that increase in size logarithmically up to an offset of 128" 且超出 "assign all relative positions to the same embedding"；"within a given layer each attention head uses a different learned position embedding"、"share the position embedding parameters across all layers"。页面「32 桶 / max_distance 128 / clamp 到最大桶 / 每头独立各层共享 / 邻近距离 $|i-j|<8$ 每距离一桶」全部吻合。
- ALiBi（C11）：原文 "biases query-key attention scores with a penalty that is proportional to their distance"、"static, non-learned bias"、"a head-specific slope fixed before training"、"inductive bias towards recency"。页面「$-m_h\cdot|i-j|$、$m_h$ 固定正斜率不学习、远距离分数单调下降、外推性好」吻合。
- RoPE×MLA（C13）：DeepSeek-V2 §2.1.3 "RoPE is incompatible with low-rank KV compression"，旋转矩阵阻碍矩阵吸收、需 decoupled RoPE。页面「位置敏感的旋转破坏矩阵吸收，因而需要解耦 RoPE」吻合。
- K3（C10）：K3 报告 §2.1.2 "applies No Position Encoding (NoPE) to all MLA layers"、"The intervening KDA layers provide position-sensitive and recency-aware sequence mixing"、"while the MLA layers provide unrestricted global content interaction"、"retuning a RoPE frequency base or applying YaRN"；§3.4 "the model extrapolates directly to 1M-token contexts without any positional-encoding modification"，位置信号 "through the recurrent gating and decay mechanism of KDA"。官方 `config.json`（`text_config` 内）确有 `"mla_use_nope": true`、`"mla_use_output_gate": true`、`"qk_rope_head_dim": 64`、`"qk_nope_head_dim": 128`，支持页面「保留 RoPE 接口（拆出 rot 分量）、mla_use_nope=true、所有 MLA 层不施加位置编码」。
- 其余编号（C6/C14）：BERT(Devlin 2018)、GPT-2(Radford 2019) 采用可学习位置编码属实；PI = arXiv:2306.15595（"linearly down-scales the input position indices to match the original context window size"）、YaRN = arXiv:2309.00071（按频率分组缩放，高频维度基本保持、低频插值）标题与内容吻合；[9] = Gehring et al. 2017, arXiv:1705.03122（Convolutional Sequence to Sequence Learning）编号与标题吻合。
- 表述维度：全文（含折叠块、图注）无第一人称复数与第二人称称呼读者；「本页」仅出现在「简化条件及其限制」小节，属 style-guide §12 明确许可的自称用法；无调试叙事、临场评价、AI 拼接腔固定套语；章节衔接按依赖用一至两句说明、未使用固定句式；details 前缀全部为「解答：/展开：/补充：」，符合 style-guide §5。
- 无「（待生成）」占位；无指向仓库中不存在文件的路径；无 `<img>` 与 alt，故不存在 alt 内 `$...$` 问题；页面无交互视图，无脚本时正文与折叠块（原生 `<details>`）仍可读；KaTeX 定界符规范，表格/summary/标题内数学符号均以 `$...$` 书写。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（修复上述「轻微·来源」C12 括注用词后）
