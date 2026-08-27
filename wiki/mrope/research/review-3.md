# MRoPE审查记录（第 3 轮）

- 页面版本：70388d2458b2d3fc1a66935d5bfa6e0ef9caa8a7（wiki/mrope/index.html 工作树哈希）
- 审查时间：2026-08-27 15:19
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节（按顺序，含折叠块）：
  - index.html：引言、核心问题（4 题及解答）、常见误解、1 文本有序、图像有格（含本章问题 2 题）、2 位置 id 变三元组（含表格、公式、构造示例表、代码折叠块及预期输出、本章问题 2 题）、3 一个头维装三条轴（含交错槽位两表、分段说明、补充折叠块、本章问题 2 题）、4 位置轴上的省账（含比值表、本章问题 2 题）、来源与范围说明（C/F/N/构造示例/类比边界/简化条件各小节）
  - overview.html：定义、问题背景、核心机制、关键结论与边界（全文）

## 核对记录（引文依据）

- C1（§2.1）："This is achieved by deconstructing the original rotary embedding into three components: temporal, height, and width." ✔
- C2（§2.1）："For text inputs, these components utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE (Su, 2024)." ✔
- C3（§2.1）："When processing images, the temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components based on the token's position in the image." ✔
- C4（§2.1）："For videos, which are treated as sequences of frames, the temporal ID increments for each frame, while the height and width components follow the same ID assignment pattern as images." ✔
- C5（§1）："Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions. The use of one-dimensional position embeddings in current models significantly limits their ability to model three-dimensional space and temporal dynamics effectively." ✔
- C6（§2.1）："M-RoPE not only enhances the modeling of positional information but also reduces the value of position IDs for images and videos, enabling the model to extrapolate to longer sequences during inference." ✔
- C7（§2.1）："In scenarios where the model's input encompasses multiple modalities, position numbering for each modality is initialized by incrementing the maximum position ID of the preceding modality by one." ✔
- C8（qwen2_vl L212-216）：`mrope_section = mrope_section * 2`；`cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)`。[16,24,24]×2=[32,48,48]，段 0/1/2 依次取 cos[0]/cos[1]/cos[2]，即 T 独占槽位 [0,15]（cos 维度 [0,31]）、H 独占 [16,39]、W 独占 [40,63]。✔
- C9（qwen4_exp L140-155）：docstring "Reorganizes frequency layout from chunked [TTT...HHH...WWW] to interleaved [THWTHWTHW...TT], preserving frequency continuity."；L150-154 实现 `freqs_t = freqs[0]`，H 覆盖 `slice(1, 33, 3)`、W 覆盖 `slice(2, 30, 3)`。✔
- C10（qwen4_exp L2115 / qwen2_vl L1008）：`current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`。✔
- F1（qwen4_exp L2023-2029）：`position_temporal = torch.arange(llm_grid_t) * time_interval`；`position_height = torch.arange(llm_grid_h) + start_position`；`position_width = torch.arange(llm_grid_w) + start_position`；`vision_position_ids[0] += start_position  # must be after time_interval multiply`；`time_interval: int = 1`。与页面公式 T=arange(t)·interval+s、H=arange(h/merge)+s、W=arange(w/merge)+s 一致。论文 §2.1 确无显式公式（文字 + Figure 3），页面已作对应标注。✔
- F2（qwen4_exp L150-154，独立复算）：mrope_section=[11,11,10] 时 H=slice(1,33,3)→{1,4,…,31} 共 11、W=slice(2,30,3)→{2,5,…,29} 共 10、T=其余→{0,3,…,30} 共 11，与页面两张槽位表逐项一致，和为 32。✔
- N1（论文 Table 8，Qwen2-1.5B + ViT-L）：NextQA 43.9→46.0、STAR 55.5→57.9、RWQ 54.5→53.7、InfoVQA 50.8→50.3（11 项中 9 升 1 平 2 降，「多数」成立）；表题 "Compared to 1D-RoPE, using M-RoPE achieves better performance in downstream tasks, particularly in video benchmarks."。✔
- N2（§3.3.2 + Figure 5 图注）："despite limiting the maximum tokens per video to 16K during training, the model still exhibits exceptional performance at a maximum inference length of 80K tokens"；"Evaluate the length extrapolation capability of Qwen2-VL-72B on Video-MME Medium Video."。✔
- 官方 config.json（Qwen2-VL-7B-Instruct）：`"mrope_section": [16, 24, 24]`；hidden_size 3584 / num_attention_heads 28 → head_dim 128 → 64 槽位，16+24+24=64 自洽。✔
- 源码仓库 commit 验证：/tmp/qwen38fn/tf HEAD = 36deb0b，与页面「transformers@36deb0b5」标注一致。✔
- 页面代码块实际运行：输出与「预期输出」逐行一致（196/14、(8,8,8)(8,21,21)、(22,22,22) 而非 (204,…)、784/28、1764/42、1980/60）。✔
- 构造示例复算：8 文本+(1,28,28) 图+5 文本，图后首文本 (22,22,22)、位置轴 22+5-1=26、(1,84,84) 时 token 1777 / 位置轴 54，全部复算一致；overview 的 1344×1344（patch 16、合并 2）→ 1764 token / 推进 42 一致。✔
- 机械项：validate.py 通过；站内链接（rope/、positional-encoding/、vit/、qwen3-8-flash-next-dataflow/、../../index.html）与本地资源（katex、prism 全部文件）存在；overview 与 index 相互链接；正文标注 C1-C10、F1-F2、N1-N2 齐全；无 Unicode 数学字符（「·」仅作标题间隔号）；结构图为 HTML 结构非 ASCII 框线图，图内公式为 HTML 内 LaTeX；公式均由 KaTeX 定界符包裹；问题块两级命名、解答折叠块、「解答：」前缀、「核心问题答案指明章节」均符合 style-guide。

## 问题

- [轻微·技术] overview.html「关键结论与边界」末条：「外推收益随多模态占比上升」为机制延伸推断，论文无此直接表述，且 overview 未像 index 那样设「辅助解释与类比边界」小节标注推断身份｜引文依据：论文仅支持「纯文本等价 1D-RoPE（§2.1）」与「多模态降低位置 id、支持外推（§2.1、§3.3.2）」，「随占比上升」为两者之间的插值推断｜修复要求：改为有依据的表述（如「纯文本上无额外收益，外推收益来自视觉段的位置压缩」）或删除该推断从句｜修复：｜复验：
- [轻微·格式] index.html「来源与范围说明」：N3/N4 在「外部数字与实验条件」中定义，但正文无任何 `<sup>[N3]</sup>`/`<sup>[N4]</sup>` 上标对应，违反 style-guide 第 6 节「双向对应」｜引文依据：正文标注检索结果为 C1-C10、F1-F2、N1-N2，无 N3/N4｜修复要求：在正文代码折叠块「验证的机制」或「观察重点」处补 [N3, N4] 上标，或将 N3/N4 合并入 N1/N2 的叙述并删除孤立编号｜修复：｜复验：
- [轻微·技术] index.html §2 公式符号说明：「$t$ 为帧数」未说明 backbone 先按 temporal_patch_size（Qwen2-VL 为 2）把帧合并为时间片；temporal patch 数才是 $t$ 的取值，temporal_patch_size=2 时两者差一倍。「简化条件及其限制」也未列此项｜引文依据：config.json `"temporal_patch_size": 2`；qwen2_vl get_rope_index docstring 的 interval 例子以 temporal patch 为单位；论文 §2.1 表述为 "temporal ID increments for each frame"（页面与论文口径一致，但与源码取值存在帧/时间片差异）｜修复要求：在简化条件节补充「帧先按 temporal_patch_size 合并为时间片，$t$ 按时间片计数」一句，或将符号说明改为「$t$ 为时间片数（temporal patch 数）」｜修复：｜复验：
- [轻微·技术] index.html §3：「源码文档字符串说明了改动动机：从分段的 $[TTT\ldots HHH\ldots WWW]$ 改为交错的 $[THWTHW\ldots]$，同时保持频率连续性」——docstring 实际描述改动内容与性质（"Reorganizes frequency layout from chunked … to interleaved …, preserving frequency continuity"），未陈述动机（为何要交错）｜引文依据：qwen4_exp L141-143 docstring 原文如上，无动机性表述｜修复要求：把「说明了改动动机」改为「说明了改动」，后半句「同时保持频率连续性」保留｜修复：｜复验：
- [轻微·技术] index.html §3 与 §4：Qwen3.8-Flash-Next 的 rotary_dim 64 与 max_position_embeddings 262144 的可定位依据为站内数据流页（正文标注「见数据流页」），本轮允许输入不含该页，无法独立核对；mrope_section $[11,11,10]$ 已通过 qwen4_exp 源码槽位规则独立复算一致（11+11+10=32=64/2 自洽），262144 支撑的「纯文本最多 26 万多个 token」一句未获独立核对｜引文依据：qwen4_exp 源码与官方 config.json 均不含该模型 config 值；页面自身标注来源为 ../qwen3-8-flash-next-dataflow/｜修复要求：保留现状可接受，但发布前须确认数据流页已完成自身质检（发布条件「前置/被引页面完成质检」）；或在页面标注中明确「数值承自数据流页实测」｜修复：｜复验：
- [轻微·可读性] index.html §3 本章问题 2 解答：「可以把更多低频槽位分给长程分量（如 $t$），按模态特性定制」是分段机制的理论能力举例，但与正文给出的 Qwen2-VL 实际配置 $[16,24,24]$（$T$ 配额最小且独占最高频段、$W$ 独占最低频段）方向相反，读者可能误以为 Qwen2-VL 实际向 $t$ 倾斜｜引文依据：config.json mrope_section=[16,24,24]；解答原文「可以把更多低频槽位分给长程分量（如 $t$）」｜修复要求：在该句后补一句「（实际 Qwen2-VL 配置 $[16,24,24]$ 并未向 $t$ 倾斜，$t$ 反而配额最小）」或将举例改为与实际配置不冲突的表述｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：修复

补充说明：本轮未发现阻断与重要问题。全部 6 条轻微问题修复并复验、且第 5 条确认数据流页已完成质检后，即满足规范第 5 节发布条件。机械验证（validate.py、代码运行、链接与资源、引文标注、KaTeX/图示规范）本轮全部通过。

## 发布记录（编排者）

第 3 轮 6 项轻微全部处理：overview「外推收益随占比上升」改为可由推进量公式直接推出的表述、N3/N4 上标补齐、t 补时间片说明（每 2 帧一片）、docstring 措辞改「改动内容」、数据流页依赖项确认（该页已三轮审查+机械验证完毕）、分段倾斜举例改为中性事实（[16,24,24] 给 H/W 各 24 档）。三轮共 6+0+0 项重要问题全部关闭，无阻断。遗留 0。validate 与渲染实测（123 个 KaTeX 节点）通过。**可发布**。
