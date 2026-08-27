# MRoPE 审查记录（第 2 轮）

- 页面版本：cf6dbdb31f89e2106ccf69dcdae459dcd1573ec5（index.html 工作树哈希；overview.html 为 3e396621fc6f6d70fa60f620b85628540b5a391e）
- 审查时间：2026-08-27 14:59
- 审查者：独立子代理
- 已完整阅读章节（按顺序）：核心问题（含全部解答折叠块）、常见误解、1. 文本有序、图像有格——一维位置轴装不下多模态（含本章问题）、2. 位置 id 变三元组——三种模态的分配规则（含代码折叠块与本章问题）、3. 一个头维装三条轴——分段与交错两种槽位排布（含补充折叠块与本章问题）、4. 位置轴上的省账——推进量与长序列外推（含本章问题）、来源与范围说明（全部小节）、overview.html 全文

## 来源核对记录

- C1 三分量拆分：论文 §2.1 "deconstructing the original rotary embedding into three components: temporal, height, and width"。
- C2 文本退化：论文 §2.1 "For text inputs, these components utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE"。
- C3 图像分配：论文 §2.1 "the temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components"。
- C4 视频递增：论文 §2.1 "the temporal ID increments for each frame, while the height and width components follow the same ID assignment pattern as images"。
- C5 三维动机：论文 §1 "Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions. The use of one-dimensional position embeddings in current models significantly limits their ability to model three-dimensional space and temporal dynamics effectively."
- C6 降低位置 id 与外推：论文 §2.1 "M-RoPE not only enhances the modeling of positional information but also reduces the value of position IDs for images and videos, enabling the model to extrapolate to longer sequences during inference."
- C7 跨模态衔接：论文 §2.1 "position numbering for each modality is initialized by incrementing the maximum position ID of the preceding modality by one"。
- C8 分段排布：modeling_qwen2_vl.py L212-218 "mrope_section = mrope_section * 2; cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)"；[16,24,24] 与 Qwen2-VL-7B-Instruct config.json "mrope_section": [16, 24, 24] 一致；head_dim 128 可由 config hidden_size 3584 / num_attention_heads 28 复算。
- C9 交错排布：modeling_qwen4_exp.py L140-155 docstring "Reorganizes frequency layout from chunked [TTT...HHH...WWW] to interleaved [THWTHWTHW...TT], preserving frequency continuity."
- C10 推进量：modeling_qwen4_exp.py L2115 "current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size"；qwen2_vl.py L1008 同式。
- F1 视觉段三维生成式：modeling_qwen4_exp.py L2017-2030 "position_temporal = torch.arange(llm_grid_t) * time_interval; position_height = torch.arange(llm_grid_h) + start_position; ... vision_position_ids[0] += start_position"，与页面 $T=\mathrm{arange}(t)\cdot\text{interval}+s$ 等三式逐项一致（$s$ 在 interval 乘法之后加，源码注释 "must be after time_interval multiply"）。
- F2 交错槽位分配：modeling_qwen4_exp.py L150-154 "freqs_t = freqs[0]; for dim, offset in enumerate((1, 2), start=1): length = mrope_section[dim] * 3; idx = slice(offset, length, 3)"。按 mrope_section=[11,11,10] 独立复算：H 覆盖 slice(1,33,3)={1,4,…,31} 共 11 个、W 覆盖 slice(2,30,3)={2,5,…,29} 共 10 个，与页面槽位表逐项一致。mrope_section 默认值 [11,11,10] 见 modeling_qwen4_exp.py L92 "config.rope_parameters.get(\"mrope_section\", [11, 11, 10])"。
- N1 消融数字：论文 §3.3.2 Table 8（Qwen2-1.5B + ViT-L 骨干，"We employ Qwen2-1.5B and ViT-L as the backbone"）：NextQA 46.0 vs 43.9、STAR 57.9 vs 55.5、RWQ 53.7 vs 54.5（降）、InfoVQA 50.3 vs 50.8（降）；11 项基准 8 升 2 降 1 平，"多数基准优于、视频提升最明显"成立；原文 "achieves better performance in downstream tasks, particularly in video benchmarks"。
- N2 外推数字：论文 §3.3.2 "despite limiting the maximum tokens per video to 16K during training, the model still exhibits exceptional performance at a maximum inference length of 80K tokens"；Figure 5 图注 "Evaluate the length extrapolation capability of Qwen2-VL-72B on Video-MME Medium Video... exceeded the maximum training length of 16384 tokens"。
- 数值复算：示例表（首 (8,8,8)、末 (8,21,21)、图后文本 (22,22,22)）、推进量表（196/14、784/28、1764/42、1980/60）、位置轴 22+5-1=26 与 8+42+5-1=54、overview 中 1344×1344（patch 16、合并 2）→ 84×84 patch 网格 → 1764 token、推进 42，均复算一致。
- 代码验证：页面 Python 代码块实际运行，输出与页面「预期输出」逐行一致。
- 机械验证：`.dojo/scripts/validate.py wiki/mrope/index.html` 返回 validation ok；正文与 overview 可见文本无 Unicode 数学字符；前置概念链接 ../rope/、../positional-encoding/、../vit/、../qwen3-8-flash-next-dataflow/ 与 ../../index.html 目标均存在；overview.html 与 index.html 互链正常。

## 问题

- [轻微·格式] index.html 核心问题解答及正文多处：<samp>见第 1 章</samp>「论证见第 1 章」「规则与实例见第 2 章」「两种实现都在第 3 章」「实测与证据见第 4 章」「这正是第 4 章的主题」「回到第 2 章结尾的数字」等处以裸编号「第 N 章」引用其他章节，style-guide.md 第 1 节要求「正文引用其他章节时使用章节标题」｜引文依据：不适用｜修复要求：将指向其他章节的裸「第 N 章」引用改为章节标题引用（如「见〈位置轴上的省账〉一章」，或「第 4 章〈位置轴上的省账〉」形式）；修正确认方式为全文不再存在不带章节标题的跨章「第 N 章」引用｜修复：｜复验：
- [轻微·技术] index.html 第 3 章「头维 256、只旋转前 64 维（rotary_dim 为 $256\times0.25$，config 键见数据流页）」与第 4 章「max_position_embeddings 为 262144，见数据流页」：head_dim 256、rotary_dim 64、max_position_embeddings 262144 三个数值的来源为站内数据流页，本轮允许输入不含该来源，未能核对（mrope_section $[11,11,10]$ 已由 modeling_qwen4_exp.py L92 默认值佐证；rotary_dim 64 与 mrope_section 和 32 内部自洽）｜引文依据：modeling_qwen4_exp.py L92 "config.rope_parameters.get(\"mrope_section\", [11, 11, 10])"；configuration_qwen4_exp.py 默认 max_position_embeddings=32768、partial_rotary_factor 默认 1.0，与页面数值不同，说明页面数值依赖具体 checkpoint 配置｜修复要求：由可访问数据流页或 Qwen3.8-Flash-Next config 原件的轮次核对 head_dim=256、rotary_dim=64、max_position_embeddings=262144 并留下引文；若数值不符则按来源修正｜修复：｜复验：
- [轻微·技术] index.html 核心问题 3 解答、常见误解第 2 条、第 3 章正文两处「每个分量都覆盖从最低频到最高频的完整范围」：该表述对 $W$ 不严格成立——$W$ 占用槽位 $\{2,5,\ldots,29\}$，未到达最高频端点 0 与最低频端点 31，仅 $T$（$\{0,\ldots,30\}$）与 $H$（$\{1,\ldots,31\}$）各达一端；「覆盖完整范围」可被读作「包含两端」｜引文依据：modeling_qwen4_exp.py L152-153 "length = mrope_section[dim] * 3; idx = slice(offset, length, 3)"，按 [11,11,10] 复算 W 槽位为 2,5,…,29｜修复要求：将相关表述改为「横跨从高频到低频的整个范围」并注明 $W$ 因少一档不达端点（正文已有「$W$ 少一档」说明，需在首次使用「完整范围」处同步限定）｜修复：｜复验：
- [轻微·可读性] index.html 常见误解第 2 条「$W$ 因 32 不被 3 整除而少一档」：「32」（频率槽位总数）在误解区出现时尚未定义——读者此时还未读到第 3 章的「32 个频率槽位」，数字 32 无落点｜引文依据：不适用｜修复要求：该句改为「$W$ 因频率槽位总数（32）不被 3 整除而少一档」或等价形式，使 32 在首次出现处可理解｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4（格式 1、技术 2、可读性 1）
- 处置：修复（4 项轻微问题逐条修复并复验后进入下一轮；本轮无阻断与重要问题，全部来源论断 C1-C10、F1-F2、N1-N2 均已给出引文依据并核对一致）
