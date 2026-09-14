<!-- review-meta
round: 8
page: wiki/mrope/index.html
reviewed_content_sha256: e0d189e6bb01d071
-->
# MRoPE 审查记录（第 8 轮）

- 页面版本：wiki/mrope/index.html（工作树哈希 b7561d075633ee01802ca5322e60255d6f808d2c）
- 审查时间：2026-09-14 17:07
- 审查者：独立子代理（第 8 轮独立审查，未参与写作，未读取本页 research/）
- 输入范围：index.html、overview.html；外部来源 arXiv:2409.12191v2 全文、huggingface.co/Qwen/Qwen2-VL-7B-Instruct 与 Qwen/Qwen3.8-Flash-Next、Qwen/Qwen3.5-397B-A17B 的 config.json、huggingface/transformers@36deb0b5 的 modeling_qwen2_vl.py 与 modeling_qwen4_exp.py；guides/concept/check.md、guides/concept/style-guide.md
- 已完整阅读章节：核心问题 / 常见误解 / 1. 文本有序、图像有格——一维位置轴装不下多模态 / 2. 位置 id 变三元组——三种模态的分配规则 / 3. 一个头维装三条轴——分段与交错两种槽位排布 / 4. 位置轴上的省账——推进量与长序列外推 / 来源与范围说明（含全部 h3 与折叠块）

## 核对结果（回源通过项）

- 论文 §1 动机引文属实：「Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions」「The use of one-dimensional position embeddings ... significantly limits their ability to model three-dimensional space and temporal dynamics effectively」——对应正文 [C5] 与核心问题 1 的解答。
- 论文 §2.1（Multimodal Rotary Position Embedding 小节）支撑 C1–C4、C7：「deconstructing the original rotary embedding into three components: temporal, height, and width」；文本「utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE」；图像「the temporal IDs of each visual token remain constant」；视频「the temporal ID increments for each frame」；跨模态「initialized by incrementing the maximum position ID of the preceding modality by one」。C6 引文：「reduces the value of position IDs for images and videos ... enabling the model to extrapolate to longer sequences during inference」。
- §3.3.2 Table 8（骨干「Qwen2-1.5B and ViT-L」，预训练模型）逐值属实：NextQA 43.9→46.0、STAR 55.5→57.9、RealWorldQA 54.5→53.7、InfoVQA 50.8→50.3。正文「NextQA 46.0 对 43.9、STAR 57.9 对 55.5」（M-RoPE 在前）次序正确，RealWorldQA/InfoVQA「小幅下降」属实。N2：Figure 5 语境为 Qwen2-VL-72B 在 Video-MME Medium Video，「limiting the maximum tokens per video to 16K during training」「robust performance when the inference length exceeded the maximum training length of 16384 tokens」「maximum inference length of 80K tokens」，与页面一致。
- 配置数值逐项复核：Qwen2-VL-7B config.json `rope_scaling = {type: mrope, mrope_section: [16,24,24]}`、head_dim = 3584/28 = 128 → 64 频率槽位，与 §3 分段例一致；Qwen3.8-Flash-Next `head_dim 256`、`partial_rotary_factor 0.25` → rotary_dim 64 → 32 槽位、`mrope_section [11,11,10]`、`rope_theta 1e7`、`max_position_embeddings 262144`、`mrope_interleaved True`；Qwen3.5-397B-A17B 同项逐项相同。页面「Qwen3.5-397B-A17B 的 MRoPE 配置与此逐项相同」属实。
- 源码定位属实：qwen4_exp `apply_interleaved_mrope` 在 L140-156（docstring 原文「Reorganizes frequency layout from chunked [TTT...HHH...WWW] to interleaved [THWTHWTHW...TT], preserving frequency continuity.」，与页面转述一致），L151-154 为 `length = mrope_section[dim] * 3; idx = slice(offset, length, 3)`，配 [11,11,10] 即 slice(1,33,3) / slice(2,30,3)，与页面及第二张槽位表逐槽吻合（T {0,…,30} 11 个、H {1,…,31} 11 个、W {2,…,29} 10 个）；`get_vision_position_ids` 在 L1980-2030，`position_temporal = arange(llm_grid_t) * time_interval`、`position_height/width = arange(grid//merge) + start_position`、`vision_position_ids[0] += start_position`，与 F1 三式逐项一致；`get_rope_index` 的推进语句 `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size` 在 L2115（页面标「L2115」正确）；qwen2_vl `mrope_section = mrope_section * 2` 与 `m[i % 3]` 在 L212-216（页面标 L180-216 覆盖该函数）。
- 代码块实跑：脚本逐字执行，输出与「预期输出」四行完全一致（196/14、784/28、1764/42、1980/60；视觉段首末 (8,8,8)(8,21,21)；图后首文本 (22,22,22)）。构造示例表 6 行 id 与公式、实测一致；§4 的 209=8+196+5、26=22+5-1、1777=8+1764+5、54=8+42+5-1 均可复算通过。推进量比值 = min(g_h,g_w)/merge，表中 14.0/28.0/42.0/33.0 与逐行 token 数相除一致。
- 链接与结构：../rope/、../positional-encoding/、../vit/、../qwen3-8-flash-next-dataflow/、../qwen3-5-dataflow/index.html 均真实存在，无「（待生成）」占位；两级问题块命名与折叠块齐备且与正文结论一致；正文 sup 引用 [C1]-[C10]、[F1][F2]、[N1]-[N4] 与「来源与范围说明」双向对应、无缺漏；dojo:type=concept、topics 在词表内、summary 的 KaTeX 语法可渲染；全文（含 head/summary/标题/表格）无 Unicode 数学字符，无 alt 内 `$...$`；`.dojo/scripts/validate.py wiki/mrope/index.html` 返回 validation ok；overview.html 与 index.html 相互链接。

## 问题

- [重要·技术] 2 章跨模态衔接段（line 174）与 4 章本章问题解答（line 335）：把视觉块的最大位置 id 无条件写成 $s+\max(g_h,g_w)/\text{merge}-1$（并称「时间分量恒为 $s$」「论文用前一种表述，实现源码算后一种，数值相同」）。这与同页 F1（$T=\mathrm{arange}(n_t)\cdot\text{interval}+s$）及同页表格「视频 $t$ 逐帧递增」直接冲突：$n_t>1$ 时时间分量取到 $s+(n_t-1)\cdot\text{interval}$，可超过空间最大值，此时块内最大位置 id 不是 $s+\max(g_h,g_w)/\text{merge}-1$，实现的推进量（$s+\max(g_h,g_w)/\text{merge}$）也不再等于「最大 id 加一」。｜引文依据：qwen4_exp L2026-2029 `position_temporal = torch.arange(llm_grid_t, device=device) * time_interval` / `vision_position_ids[0] += start_position`；qwen2_vl get_rope_index L1005-1008 对视频传入完整 (T,H,W) 网格（不像 qwen4_exp 那样拆成单帧），故视频 $n_t>1$；论文 §2.1「the temporal ID increments for each frame」。｜修复要求：把该论断限定为图像块/单帧网格（$n_t=1$），或改写为 $\max\big((n_t-1)\cdot\text{interval},\; g_h/\text{merge}-1,\; g_w/\text{merge}-1\big)$；4 章本章问题解答里「时间分量恒为 $s$」同步加限定。｜修复：｜复验：

- [轻微·技术] 4 章推进量表前的概括「规模越大省得越多」（line 287 附近）：与同页表格第 4 行自相矛盾——$(1,84,84)$ 比值 42.0，$(1,66,120)$ token 数更多（1980 > 1764）比值反而只有 33.0。｜引文依据：同页表内数值 196/14=14.0、784/28=28.0、1764/42=42.0、1980/60=33.0；比值恒等于 $\min(g_h,g_w)/\text{merge}$（短边决定），与「规模」并非单调。｜修复要求：改为「短边越长省得越多」或删去该概括。｜修复：｜复验：

- [轻微·技术] 3 章槽位段「$T$ 与 $H$ 从最高频槽位铺到最低频」（line 278）：与同页槽位表相抵——$T$ 止于槽位 30（缺最低频 31）、$H$ 起于槽位 1（缺最高频 0），两者单独都不覆盖两端点；紧随的「$W$ …（不含两端点）」对比句更强化了「T、H 覆盖两端点」的误读。｜引文依据：同页表 $T$ = 0,3,…,30；$H$ = 1,4,…,31；源码 `slice(1,33,3)` 末位 31、`slice(2,30,3)` 末位 29，`freqs_t = freqs[0]` 使 31 被 $H$ 覆盖、0 被 $H$ 之外即由 $T$ 保留。｜修复要求：改为「$T$ 与 $H$ 合起来从最高频槽位铺到最低频」，或分别写明各自缺失的端点槽位。｜修复：｜复验：

- [轻微·表述] 4 章收尾段「位置轴仍是 $8+42+5-1=54$——序列长度大幅增加，位置轴几乎没动」（line 327）：与同句给出的数字不符——位置轴由 26 增至 54（约 2.08 倍），并非「几乎没动」，「仍是」也不成立。｜引文依据：同段前句「位置轴只用到 $22+5-1=26$」，本句 54；token 209→1777。｜修复要求：改为与数字相符的表述，如「位置轴只增到约两倍，远小于序列长度的增幅」。｜修复：｜复验：

- [轻微·格式] 2 章「构造示例。 序列 = 8 个文本 token + …」（line 176）：示例标记用了「构造示例」，不在 style-guide §4 允许的三种标记（「计算示例」「代码示例」「构造数据」）之内；且句号后多一个半角空格。｜引文依据：guides/concept/style-guide.md §4「示例按用途标记为"计算示例""代码示例"或"构造数据"」；文件实际字节为 `构造示例。 序列`（全角句号 + U+0020）。｜修复要求：改为允许的标记（如「构造数据」），并去掉多余空格；「来源与范围说明」下同为固定 h3 名的「构造示例」不受影响。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（按上表逐条处理后重跑 validate.py，并由下一轮独立审查复验；overview.html 无需改动）