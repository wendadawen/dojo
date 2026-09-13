<!-- review-meta
round: 4
page: wiki/mrope/index.html
reviewed_content_sha256: aa6097bf34a36ad6
-->
# MRoPE 审查记录（第 4 轮）

- 页面版本：c02dbba331bc2952978307a24c474919251ab02e（git hash-object wiki/mrope/index.html）
- 审查时间：2026-09-13 19:45
- 审查者：编排者派发的独立审查者（独立子代理，未参与写作与前三轮审查）
- 已完整阅读章节：核心问题、常见误解、1. 文本有序、图像有格——一维位置轴装不下多模态、2. 位置 id 变三元组——三种模态的分配规则、3. 一个头维装三条轴——分段与交错两种槽位排布、4. 位置轴上的省账——推进量与长序列外推、来源与范围说明（含全部 details 折叠块、代码块与图注）

## 核对说明（来源与复算）

- 论文 Qwen2-VL（arXiv:2409.12191v2）逐句核对通过：三分量拆分 "deconstructing the original rotary embedding into three components: temporal, height, and width"（C1）；文本退化 "For text inputs, these components utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE"（C2）；图像 "the temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components"（C3）；视频 "the temporal ID increments for each frame"（C4）；三维动机 §1（C5）；外推 "reduces the value of position IDs for images and videos, enabling the model to extrapolate to longer sequences"（C6）；跨模态衔接 "position numbering for each modality is initialized by incrementing the maximum position ID of the preceding modality by one"（C7）。
- Table 8 消融数值逐格核对：NextQA 1D 43.9 / M 46.0（页「46.0 对 43.9」）；STAR 1D 55.5 / M 57.9（页「57.9 对 55.5」）；RWQ 1D 54.5 / M 53.7 与 InfoVQA 1D 50.8 / M 50.3（页「个别基准如 RWQ、InfoVQA 小幅下降」）——均相符（N1）。16K/80K：原文 "Despite limiting the maximum tokens per video to 16K during training, the model still exhibits exceptional performance at a maximum inference length of 80K tokens"，Figure 5 为 Qwen2-VL-72B、Video-MME 中等时长视频（N2），相符。
- 源码核对（transformers@36deb0b5，经 gh api 实拉文件）：C8 modeling_qwen2_vl.py L180 起 apply_multimodal_rotary_pos_emb、L212 `mrope_section = mrope_section * 2`、L213 `m[i % 3]`——与「[32,48,48] 后按 i%3 取段」逐字相符；C9/F2 modeling_qwen4_exp.py L140 `apply_interleaved_mrope` 及 docstring "Reorganizes frequency layout from chunked [TTT...HHH...WWW] to interleaved [THWTHWTHW...TT], preserving frequency continuity"、L153 `idx = slice(offset, length, 3)`——与页「slice(1,33,3) / slice(2,30,3)」相符；F1 get_vision_position_ids L1980-2030 与页三式一致；C10 推进量 L2115 `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`——与页「max(h,w)/merge」相符。行号全部命中。
- Qwen2-VL-7B config.json：`mrope_section [16,24,24]`、head_dim 128（=3584/28）→ 64 频率槽位、T/H/W 段 [0,31]/[32,79]/[80,127]（槽位 [0,15]/[16,39]/[40,63]）——与页相符。
- 代码复算：页面 Python 代码块本机运行，输出与「预期输出」六行逐字一致（196/14、首末 (8,8,8)(8,21,21)、图后 (22,22,22)、四组 grid 的 token 数与推进量 14/28/42/60）；比值列 14.0/28.0/42.0/33.0 全部复算正确；token 数 209=8+196+5、位置轴 22+5-1=26、(1,84,84) 情形 1777 与 8+42+5-1=54 均正确。
- 机械项：`.dojo/scripts/validate.py wiki/mrope/index.html` 返回 validation ok；`dojo:type=concept`、`dojo:topics=注意力机制,多模态`（均在 AGENTS.md 固定大类内）、`dojo:tag=位置编码`（在 ALLOWED_TAGS 内）；C1–C10、F1–F2、N1–N4 全部有正文上标与来源小节双向对应，无孤儿编号；无「（待生成）」占位；除 JS 注释外无 ASCII 框图，结构图为 HTML div；公式全部 KaTeX 渲染（validate 通过），非数学的「·」仅作标题分隔符。

## 问题

- [重要·技术] 2. 位置 id 变三元组（分配表 / $T$ 公式符号说明 / 两级解答）：视频时间分量的递增粒度在全页自相矛盾。分配表写「视频 $t$ 逐帧递增」，核心问题与本章问题解答写「视频 $t$ 逐帧递增」「第 $k$ 帧的所有 token 共享 $t=s+k$」「视频 $t>1$ 时 $T$ 逐帧递增，与上表一致」，而同一节的 $T$ 公式符号说明写「$t$ 为时间片数（源码中每 2 帧合成一个时间片）」。若 2 帧合成 1 个时间片，则 $t$ 每 2 帧才加一，与「逐帧（每帧加一）」相差一倍，二者不可能同时成立。｜引文依据：页面标注来源 qwen4_exp get_rope_index 实传 `self.get_vision_position_ids(current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device)`（modeling_qwen4_exp.py L2111-2112，第 3 位参数即 temp_merge_size，取 1，时间维不折半）；论文原文 "For videos, which are treated as sequences of frames, the temporal ID increments for each frame"（§2.1）。页面的「每 2 帧合成一个时间片」在所指源码位置并无对应。｜修复要求：全页只保留一种粒度——或按论文口径统一为「逐帧递增」并删去「每 2 帧合成一个时间片」；或按实现口径统一为「逐时间片（每 2 帧一片）递增」并同步改分配表、核心问题解答、本章问题解答与「简化条件及其限制」中所有「逐帧」字样。｜修复：｜复验：
- [重要·来源] 来源与范围说明·外部数字与实验条件（N）：「脚本见本页代码块与数据流页 research/」指向已移除的 research/ 路径。数据流页 research/ 现仅存 md 文件，页面所指的实测脚本已从仓库移除。｜引文依据：目录 qwen3-8-flash-next-dataflow/research/ 只含 measured.md、prereq-audit.md、review-1.md、review-2.md；其 measured.md 表内登记 `assert_dims.py | 0.0 KB | 实测脚本`、`verify_page_numbers.py | 0.0 KB | 实测脚本` 等「现已从仓库移除」；research/ 亦不参与 GitHub Pages 部署。｜修复要求：删除「与数据流页 research/」这一指向，N3/N4 只保留「本页代码块」（该代码块已完整复现推进量与构造序列实测）。｜修复：｜复验：
- [轻微·表述] 引言末段：「本文的学习路线：先看一维位置轴在多模态下缺什么；再看……；然后看……；最后算一笔……」。属元话语路线图（「本页将…/先看…再看…」句式）。｜引文依据：不适用。｜修复要求：改为直陈本文内容范围，或直接删去该句，由小标题承担结构引导。｜修复：｜复验：
- [轻微·表述] 3. 一个头维装三条轴：「频率槽位的方向需要注意：RoPE 的 inv_freq 随槽位下标递减……」。「需要注意」为规范点名的元话语句式。｜引文依据：不适用。｜修复要求：删去「需要注意」，直接陈述「RoPE 的 inv_freq 随槽位下标递减，槽位 0 是最高频……」。｜修复：｜复验：
- [轻微·表述] 4. 位置轴上的省账末段：「这就是「位置轴预算」视角的全部含义。」属收尾式临场评价。｜引文依据：不适用。｜修复要求：改为陈述结论，删去「的全部含义」这类评断语。｜修复：｜复验：
- [轻微·技术] 2. 位置 id 变三元组：符号 $t,h,w$ 兼作两种含义——分配表列名与正文「时间分量 $t$／高分量 $h$／宽分量 $w$」指位置 id 分量取值；而 $T$ 公式与其符号说明中 $t$＝时间片数、$h,w$＝网格 patch 数，正文「最大位置 id 是 $s+\max(h,w)/\text{merge}-1$」的 $h,w$ 亦为网格尺寸。同一变量在同一章内表两义，违反全页单义。｜引文依据：不适用。｜修复要求：把公式中的计数改用不同符号（如 $n_t / g_h / g_w$）或在公式处显式声明「此处 $t,h,w$ 指片数与网格尺寸，与分配表中的分量取值同名但含义不同」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复（两条重要问题均需修复：视频时间分量粒度统一、删除指向已移除 research/ 的路径；四条轻微问题一并处理。本轮无核心结论错误，来源与数值核对全部通过，可运行代码输出与页面描述一致。）