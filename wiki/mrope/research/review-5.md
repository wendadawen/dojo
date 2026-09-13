<!-- review-meta
round: 5
page: wiki/mrope/index.html
reviewed_content_sha256: 57f34498ab2e2430
-->
# MRoPE 审查记录（第 5 轮）

- 页面版本：98c150d6ac56de8f40f52fa7b1f597c0f669e0a5
- 审查时间：2026-09-13 20:23
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查；本轮只读 index.html、overview.html、规范与外部来源，未读本页 research/）
- 已完整阅读章节：核心问题；常见误解；1. 文本有序、图像有格——一维位置轴装不下多模态（含图注、本章问题）；2. 位置 id 变三元组——三种模态的分配规则（含构造示例表、代码折叠块、本章问题）；3. 一个头维装三条轴——分段与交错两种槽位排布（含补充折叠块、本章问题）；4. 位置轴上的省账——推进量与长序列外推（含本章问题）；来源与范围说明；overview.html。

## 核对依据（外部来源，逐条定位）

- Qwen2-VL 论文（arXiv:2409.12191v2）§2.1 原文：「For text inputs, these components utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE.」「When processing images, the temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components based on the token's position in the image.」「For videos, ... the temporal ID increments for each frame ...」「position numbering for each modality is initialized by incrementing the maximum position ID of the preceding modality by one.」「M-RoPE ... reduces the value of position IDs for images and videos, enabling the model to extrapolate to longer sequences during inference.」→ 页面对应 C1/C2/C3/C4/C7/C6 全部吻合。
- §1 原文：「The use of one-dimensional position embeddings in current models significantly limits their ability to model three-dimensional space and temporal dynamics effectively.」+「Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions.」→ C5 吻合。
- §3.3.2 Table 8 标题：「Ablation studies of M-RoPE. Compared to 1D-RoPE, using M-RoPE achieves better performance in downstream tasks, particularly in video benchmarks.」；表内数值 NextQA 43.9→46.0、STAR 55.5→57.9、RWQ 54.5→53.7、InfoVQA 50.8→50.3；表注「RWQ means RealworldQA」；骨干「We employ Qwen2-1.5B and ViT-L as the backbone」→ N1 全部吻合（含页面用的缩写 RWQ 与论文表注一致）。Figure 5「Length extrapolation capability evaluation」+ 16K 训练 / 80K 推理语境「Qwen2-VL-72B on Video-MME medium-length videos」→ N2 吻合。
- Figure 3 标题「A demonstration of M-RoPE. By decomposing rotary embedding into temporal, height, and width components ...」→ F1 关于「论文正文无显式公式、以 Figure 3 与文字描述呈现」的说明成立。
- Qwen/Qwen2-VL-7B-Instruct config.json：hidden_size 3584、num_attention_heads 28、head_dim 128、max_position_embeddings 32768、rope_scaling.type "mrope"、mrope_section [16, 24, 24] → 页面 §3 分段排布举例（[16,24,24]、64 槽位、T[0,15]/H[16,39]/W[40,63]）逐项吻合。
- transformers@36deb0b5（拉取源码核对行号）：modeling_qwen2_vl.py `apply_multimodal_rotary_pos_emb` def 在 L180，`mrope_section = mrope_section * 2` 在 L212，`m[i % 3] ... cos.split(...)` 在 L213-216 → C8 标注「L180-216」精确；docstring「we split the channel dimension to 3 chunks」吻合。modeling_qwen4_exp.py `apply_interleaved_mrope` def 在 L140，`length = mrope_section[dim] * 3` L152、`idx = slice(offset, length, 3)` L153，docstring「Reorganizes frequency layout from chunked [TTT...HHH...WWW] to interleaved [THWTHWTHW...TT], preserving frequency continuity.」→ C9「L140-155 及 docstring」、F2「L150-154」吻合；`get_vision_position_ids` 在 L1980-2030（T=arange(n_t)*time_interval+s、H=arange(g_h/merge)+s、W=arange(g_w/merge)+s）→ F1 公式逐项吻合；`get_rope_index` 中 `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size` 在 L2115 → C10 吻合（qwen2_vl 同式在 L1008）。freqs 末维 = head_dim×partial_rotary_factor/2 = 32，故 slice(1,33,3)={1,4,…,31}、slice(2,30,3)={2,5,…,29}，与页面槽位表 T=11/H=11/W=10 逐槽吻合。
- 代码块（§2 折叠块）：本机执行，输出与页面「预期输出」逐行一致（196/14、(8,8,8)/(8,21,21)、(22,22,22)/(204,204,204)、1980 与 60 等）。
- `.dojo/scripts/validate.py wiki/mrope/index.html` → validation ok。

## 问题

- [重要·技术] 页首引言、§1 正文与图注、本章问题第 1 题答案、学习目标第 1 题答案：把拉平后「同行相邻」与「同列相邻」在一维轴上的距离都写成 1（「右与下不可区分」），并据此说「往右 3 个」与「往下 3 个」都变成「往后 3 个」。行优先拉平下索引 $\text{idx}=rW+c$，同行左右相邻差值 $\Delta=1$，同列上下相邻差值 $\Delta=W$（$W$ 为网格列数），二者数值不同，右与下可由位置差区分；「往右 3 个」$\Delta=3$ 与「往下 3 个」$\Delta=3W$ 也不等价。这是算式与结论不符的机制陈述，且未在任何位置标注为简化或类比。｜引文依据：页面原文「同一个 patch 的右邻与下邻，在一维轴上距离都变成 1」「拉平后：同行相邻与同列相邻的距离都为 1，右与下不可区分」「拉平后两者在位置轴上的距离同为 1」「拉平进一维轴后「同行相邻」与「同列相邻」的距离变得相同」；对照 $3\times3$ 网格行优先索引：(0,0)=0、(0,1)=1、(1,0)=3，右邻 $\Delta=1$、下邻 $\Delta=3$。｜修复要求：把该处机制改写为「一维位置差是单个标量：横向相邻对应 $\Delta=1$、纵向相邻对应 $\Delta=W$；同一 $\Delta$ 在不同宽度的网格下对应不同空间位移（$\Delta=W$ 既可指向下一行，也可指同一行向右 $W$ 个），模型无法从位置差判断位移沿哪个轴」，并同步修正图注「同行相邻与同列相邻的距离都为 1，右与下不可区分」、本章问题第 1 题答案中「距离同为 1」「隔 3 行的同行 token 与隔 3 列的同行 token 距离相同」两句、以及学习目标第 1 题答案中「距离变得相同」。｜修复：｜复验：

- [轻微·可读性] §3「分段排布」段：句中「$T$ 独占槽位 $[0,15]$（cos/sin 维度 $[0,31]$）、$H$ 独占 $[16,39]$、$W$ 独占 $[40,63]$」在列表中途切换单位，仅对 $T$ 标出「cos/sin 维度」换算，$H$、$W$ 未标单位；若读者把 $H[16,39]$ 读成维度区间，会与同句「频率槽位共 64 个」及 $T$ 的维度区间 $[0,31]$ 冲突（重叠）。｜引文依据：不适用｜修复要求：统一标注，如「三者在槽位轴上分别为 $T[0,15]$、$H[16,39]$、$W[40,63]$（对应 cos/sin 维度 $T[0,31]$、$H[32,79]$、$W[80,127]$）」。｜修复：｜复验：

- [轻微·技术] §3 正文与本章问题第 2 题答案：一处写「$W$ 的槽位从 2 到 29 也近似覆盖整个范围（不含两端点）」，同一章另一处写「每个分量都覆盖完整频率范围」。按同段槽位表，$W$ 仅占 $\{2,5,\ldots,29\}$，缺槽位 0、1、30、31 共四档；$T$ 缺 31、$H$ 缺 0，均非「完整频率范围」。｜引文依据：页面槽位表 $W=\{2,5,8,11,14,17,20,23,26,29\}$（10 档）、总槽位 32；源码 `slice(2, 30, 3)`。｜修复要求：把「每个分量都覆盖完整频率范围」改为「每个分量都近似覆盖从高频到低频的整个范围（各自缺少数个端点附近槽位）」，与同章「近似覆盖整个范围（不含两端点）」及本章问题答案「任何分量在任何尺度都有一定分辨率」保持一致。｜修复：｜复验：

- [轻微·可读性] §4 末段：「token 数变 1777，位置轴仍是 $8+42+5-1=54$——序列长了一处，位置轴几乎没动」中「序列长了一处」语义不清（token 数由 209 增至 1777，是大幅增长，「一处」不能表达该含义）。｜引文依据：不适用｜修复要求：改为可直接判定的表述，如「序列长度大幅增加，位置轴几乎没动」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（修正 §1 机制陈述后复验；三条轻微问题随该段修改一并处理）
- 说明：核心事实（C1-C10）、公式（F1、F2）、外部数字（N1-N4）与代码输出均已按来源逐条定位核对，引文依据栏所列原文片段与数值全部吻合；两处槽位表经源码 `slice(1,33,3)` / `slice(2,30,3)` 逐槽复算一致；页面链接、公式渲染、折叠块、问题块结构、`dojo:*` 头字段与 validate.py 均通过。