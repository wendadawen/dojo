<!-- review-meta
round: 10
page: wiki/rope/index.html
reviewed_content_sha256: ab71802df6c3cd1a
-->
# RoPE 旋转位置编码审查记录（第 10 轮）

- 页面版本：index.html 工作树哈希 ea7ac32c6257389e4c0a0024a8cb2bdce0771ee3
- 审查时间：2026-09-14 17:13
- 审查者：独立子代理
- 适用规范：guides/concept/check.md（dojo:type=concept）
- 已完整阅读章节（按正文顺序）：引言与核心问题（5 题）→ 1. 为什么 Transformer 需要位置编码——RoPE 与四类方案的差异（含本章问题 3 题）→ 2. 2 维 RoPE 是怎么旋转的——最小可手算机制（含本章问题 3 题、折叠块「展开：2 维手算例子的完整代入过程」）→ 3. 内积为何只依赖 $m-n$——旋转矩阵群的性质（含本章问题 3 题、折叠块「补充：$R_m^T R_n = R_{n-m}$ 的三角恒等式推导」）→ 4. d 维推广——分块对角旋转矩阵与 $\theta_i$ 几何级数（含本章问题 3 题、折叠块「展开：$d=4$ 多频率手算例子」）→ 5. 远程衰减——相位抵消与 QK-only 机制（含本章问题 3 题、折叠块「补充：从内积公式到归一化和的推导」）→ 6. 适用边界——长度外推、K3 MLA 的 NoPE 选择（6.1、6.2、本章问题 3 题）→ 结语 → 来源与范围说明。overview.html 全文亦已通读并做跨页一致性比对。

## 核对方法与来源取用

- 官方材料：arXiv:2104.09864v5（RoFormer）摘要、§3.1 Eq.(11)、§3.2.1 Eq.(12)–(13)、§3.2.2 Eq.(14)–(16) 及 $\Theta$ 定义、§3.3「Properties / Long-term decay」、§3.4.1 Eq.(20)–(33)、§3.4.2 Eq.(34)、§3.4.3 Eq.(35)–(37) 与 Figure 2；arXiv:2607.24653（Kimi K3 技术报告）§2.1.2、§3.4；arXiv:2306.15595（PI）、arXiv:2309.00071（YaRN，§3.2 与 Definition 2/3）、arXiv:2402.13753（LongRoPE）摘要；arXiv:2305.19466（NoPE）摘要。
- 数值复核：以 Python 独立实现 $\theta_i=\mathrm{base}^{-2i/d}$ 与归一化内积 $\frac{1}{64}\sum_{i=0}^{63}\cos((n-m)\theta_i)$ 复算全表，并逐位复算 2 维、$d=4$ 两处手算例。
- 图内数值：对 SVG 坐标做像素测量（原点 (150,250)，向量端点、弧半径、标签框）核对。

## 本轮核对结果（不构成问题的部分，记录引文依据备查）

- 公式 F2 与论文一致：Eq.(12) 的 $\mathrm{Re}[(W_qx_m)(W_kx_n)^*e^{i(m-n)\theta}]$ 展开为 $(q\!\cdot\!k)\cos((m-n)\theta)+(q_0k_1-q_1k_0)\sin((m-n)\theta)$，与本页写作的 $(q\!\cdot\!k)\cos((n-m)\theta)-(q_0k_1-q_1k_0)\sin((n-m)\theta)$ 恒等（余弦为偶、正弦为奇）。§3.4.1 复核：Eq.(33) $f_q=(W_qx_m)e^{im\theta}$、$f_k=(W_kx_n)e^{in\theta}$，同源。
- 引文编号逐条落地：Eq.(11) 在 §3.1「Formulation」，内容为 $\langle f_q(x_m,m),f_k(x_n,n)\rangle=g(x_m,x_n,m-n)$（相对位置目标）；Eq.(12) 2 维复数解；Eq.(13) 2 维矩阵/旋转形式；Eq.(14) $f_{q,k}(x_m,m)=R^d_{\Theta,m}W_{q,k}x_m$；Eq.(15) 分块对角 $R^d_{\Theta,m}$；Eq.(16) $R^d_{\Theta,n-m}=(R^d_{\Theta,m})^T R^d_{\Theta,n}$；Eq.(20)–(33) §3.4.1；Eq.(34) §3.4.2 逐元素实现；Eq.(35)–(37) §3.4.3。与行 62 metadata、C1–C5、F1–F8 描述全部吻合。
- $\Theta$ 定义核实：§3.2.2 记 $\Theta=\{\theta_i=10000^{-2(i-1)/d},\,i\in[1,\dots,d/2]\}$；§3.3「Long-term decay」小节原文 "Following Vaswani et al. (2017), we set $\theta_i=10000^{-2i/d}$."。本页行 614/632 把两条分别归到 §3.2.2 与 §3.3，并把 F6 索引平移说明为「原文索引从 $i=1$ 起写为 $10000^{-2(i-1)/d}$，与本页从 $i=0$ 的写法数学等价」——归属与平移说明均正确。
- 摘要引文核实：arXiv:2104.09864v5 摘要含 "encodes the absolute position with a rotation matrix" 与 "incorporates the explicit relative position dependency"；"flexibility of sequence length" 确在论文列举的 valuable properties 中。行 154/552 的引用与解读（结构上可任意扩展 ≠ 简单外推不变）成立。
- C5/F8 引文核实：§3.4.3 原文 "the value of 1/(d/2)∑|S_i| decay with the relative distance m−n increases … as shown in Figure 2"，Figure 2 caption 为 "Long-term decay of RoPE."，且全文与图注均未给出该图的维度取值——与本页「论文未标注图中维度取值」一致。
- 数值全部复算通过：$d=128,\mathrm{base}=10000$ 下 $\theta_0=1$、$\theta_1=0.86596$、$\theta_{16}=0.1$、$\theta_{63}=1.1548\times10^{-4}$；$2\pi/\theta$ 依次 6.28 / 7.26 / 62.83 / 54410.1，表内 6/7/63/54410 四舍五入正确；衰减表 10 行 1.0000 / 0.9702 / 0.7373 / 0.6691 / 0.5462 / 0.4772 / 0.2985 / 0.1590 / −0.0070 / −0.0279 与独立复算逐位相同；$d=4$ 手算 $\cos1+\cos0.5=1.417885\approx1.4179$ 正确；2 维例 $q=k=(1,0),\theta=\pi/4$ 得 $\sqrt2/2\approx0.7071$，本章问题例 $q=(1,1),k=(0,1),\theta=\pi/6$ 得 $(\sqrt3-1)/2\approx0.3660$，直接旋转法与 F2 法一致。
- 图注读数与图上刻度一致（像素测量）：原点 (150,250)；x 轴向量 (150,250)→(298,250) 长 148 px，$R_1q$ 端点 (254.7,145.3) 长 148.07 px、方位 45°，$R_2k$ 端点 (150,102) 长 148 px、方位 90°；两条 θ 弧半径 54（0°→45°）与 90（45°→90°），端点距原点分别为 54.02/89.94 px，与图注「夹角 $(n-m)\theta=\pi/4$、内积 $0.71$」吻合。标签框均未压线/重叠。
- 扩展方法逐条核实：PI 摘要 "linearly down-scales the input position indices to match the original context window size"（与 $m'=m\cdot L_{\text{train}}/L_{\text{target}}$ 同义）；YaRN §3.2 "if the wavelength λ is much smaller than the context size L, we do not interpolate" / "if the wavelength λ is equal to or bigger than the context size L, we want to only interpolate"，Definition 3 定义 YaRN = NTK-by-parts + attention scaling（与「高频基本保持、低频缩放」一致）；LongRoPE 摘要 "identify and exploit two forms of non-uniformities in positional interpolation through an efficient search" 且 "extends the context window … to an impressive 2048k tokens"（与「按维度非均匀搜索、支持百万级上下文」一致）。
- K3 核实：技术报告 §2.1.2 原文对全部 MLA 层 "applies No Position Encoding (NoPE) to all MLA layers"，并称 "The intervening KDA layers provide position-sensitive and recency-aware sequence mixing"；§3.4 标题即 "Long-Context Extension"，述及 K3 "encodes positional information implicitly through the recurrent gating and decay mechanism of KDA"、可直接外推 1M token、四阶段渐进扩展。与行 121/576/595 及 C8 一致（正文"分阶段扩展训练达到 1M token"有据）。
- N3 核实：arXiv:2305.19466 摘要称 ALiBi/Rotary/APE "not well suited for length generalization in downstream tasks"，NoPE "outperforms other explicit positional encoding methods"；本页对 ALiBi 一行与 NoPE 主张的转述与之一致。
- N4 核实：T5 相对位置偏置按对数分桶、每头一份可学习偏置，bucket 数 32，与行 78/163 一致。
- 机械项：`.dojo/scripts/validate.py wiki/rope/index.html` 退出 0；`dojo:topics=注意力机制`、`dojo:tag=位置编码` 均在 AGENTS.md/catalog_builder.py 词表内；`dojo:summary` 公式（`$q_m^\top k_n$`、`$\theta_i=\mathrm{base}^{-2i/d}$`）为合法 KaTeX；`description` 为纯文本；无「（待生成）」占位；引用的 6 个概念页（standard-attention / positional-encoding / causal-mask / nope / mla / kimi-k3）均真实存在；overview.html 与 index.html 双向互链；全文无 Unicode 数学字符替代、无 alt 含 `$...$`、无交互视图（无脚本依赖）；全页无「本页／我们／你」等不合规自称（改用 style-guide 许可的「本文」）。

## 问题

- [轻微·来源] 行 65（引言首段）："RoPE 是当前自回归 LLM 中采用最广的位置编码方案"｜问题：这是对领域现状的判断，无来源支持即写成结论。紧邻的模型清单已用括注标明"此句为公开模型资料的概括，未逐模型核对到可定位出处"[C6]，而这个更强的最高级判断既未被该括注覆盖，C6 条也只声明覆盖"模型清单"（"LLaMA / Mistral / Falcon / Qwen / PaLM / Gemma / GPT-NeoX 等模型家族在公开描述中常被列为 RoPE 的使用者"），未声明覆盖"采用最广"。页面对同类概括（C6、N3）一律标注为推断，此处口径不一致。｜引文依据：C6 条全文仅涉及模型清单；无法给出"采用最广"的可定位出处。｜修复要求：把该判断改写为明确标注的推断，或补可核对出处（如各模型官方 config.json 的 `rope_theta`/`rope_scaling` 字段统计）；不得以无标注的断言形式保留。｜修复：｜复验：
- [轻微·格式] 行 129（第 1 章末过渡段）："这个问题在 2 维 RoPE 一章手算回答，并在内积证明一章验证它只依赖 $m-n$。"｜问题：正文引用其他章节应使用章节标题（guides/concept/style-guide.md §1），本页行 82、行 93 引用同一章时写作「内积为何只依赖 $m-n$」一章，行 129 却写作"内积证明一章"，"2 维 RoPE 一章"亦非标题（标题为"2. 2 维 RoPE 是怎么旋转的——最小可手算机制"）。同一章在全页出现两种称法，读者比对目录时需要额外推断。｜引文依据：不适用。｜修复要求：行 129 的两处章节指代改为与行 82/93 一致的形式（「2 维 RoPE 是怎么旋转的」与「内积为何只依赖 $m-n$」两章），或统一采用同一简称形式并全页一致。｜修复：｜复验：

（未发现阻断、重要级问题：核心结论、公式、全部数字、引文编号、图注读数、跨页数字均与来源一致；未发现同页两处矛盾、算式与结论不符、来源不支持而被写成结论的机制描述、构造示例被写成来源事实、无来源判断被包装成实验结论等情形。）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：修复（两条均为轻微，不阻碍发布；建议在下一轮修复后复验，或按第 5 节作为「有明确接受理由的遗留轻微问题」接受。本轮无阻断、无重要问题，页面主要结论与来源一致性、数值与图注、结构与功能均已通过独立核对）

统计：阻断 0 / 重要 0 / 轻微 2