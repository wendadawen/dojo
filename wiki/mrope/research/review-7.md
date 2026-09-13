<!-- review-meta
round: 7
page: wiki/mrope/index.html
reviewed_content_sha256: 4f259ca6e43af7ba
-->
# MRoPE 审查记录（第 7 轮）

- 页面版本：index.html a5330fde4112ae0bfeda244ac1f03445d5425f8e；overview.html 08029372dbb75ededf34cad08cc72714049ff8d2
- 审查时间：2026-09-13 21:54
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：头部（description / dojo:summary / 主要依据 / 引言 / 核心问题 / 常见误解）→ 1. 文本有序、图像有格——一维位置轴装不下多模态（含本章问题与图注）→ 2. 位置 id 变三元组——三种模态的分配规则（含构造示例表、代码折叠块、本章问题）→ 3. 一个头维装三条轴——分段与交错两种槽位排布（含槽位表、补充折叠块、本章问题）→ 4. 位置轴上的省账——推进量与长序列外推（含实测表、本章问题）→ 来源与范围说明

## 来源核对记录（本轮实际打开并核对的来源）

- Qwen2-VL 论文 arXiv:2409.12191v2（HTML 全文）：§2.1 "deconstructing the original rotary embedding into three components: temporal, height, and width"、text "these components utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE"、image "the temporal IDs of each visual token remain constant"、video "the temporal ID increments for each frame"、跨模态 "incrementing the maximum position ID of the preceding modality by one"、C6 "reduces the value of position IDs for images and videos… extrapolate to longer sequences during inference"；§1 "Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions." / "The use of one-dimensional position embeddings in current models significantly limits their ability… to model three-dimensional space and temporal dynamics effectively."；§3.3.2 Table 8（Qwen2-1.5B + ViT-L，预训练）NextQA 43.9→46.0、STAR 55.5→57.9、RWQ 54.5→53.7、InfoVQA 50.8→50.3、caption "better overall performance, particularly in video benchmarks"；§3.3.2 Figure 5 "Evaluate the length extrapolation capability of Qwen2-VL-72B on Video-MME Medium Video"、"limiting the maximum tokens per video to 16K during training"、"still exhibits exceptional performance at a maximum inference length of 80K tokens"。
- transformers@36deb0b5：modeling_qwen2_vl.py（apply_multimodal_rotary_pos_emb 定义在 L180、`mrope_section = mrope_section * 2` 在 L212、`m[i % 3]` 在 L213/L216；rotate_half 在 L173-177）；modeling_qwen4_exp.py（apply_interleaved_mrope 在 L140-155，docstring "Reorganizes frequency layout from chunked [TTT...HHH...WWW] to interleaved [THWTHWTHW...TT]"；`idx = slice(offset, length, 3)`、`length = mrope_section[dim] * 3` 在 L152-153；get_vision_position_ids 在 L1980-2031；get_rope_index 内 `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size` 在 L2115，`Separate video grid thw into multiple grids…` 与 `video_grid_thw[:, 0] = 1` 在 L2067-2069；rotate_half 在 L566-570）。
- Qwen/Qwen2-VL-7B-Instruct config.json：rope_scaling.mrope_section [16,24,24]、hidden_size 3584、num_attention_heads 28（head_dim 128）、rope_theta 1e6。
- Qwen/Qwen3.8-Flash-Next config.json：max_position_embeddings 262144、rope_parameters.mrope_section [11,11,10]、rope_theta 1e7、partial_rotary_factor 0.25、head_dim 256、mrope_interleaved true、model_type qwen4_exp。
- Qwen/Qwen3.5-397B-A17B config.json：max_position_embeddings 262144、mrope_section [11,11,10]、rope_theta 1e7、partial_rotary_factor 0.25、head_dim 256。
- 本页代码块：以 python3 原样执行，输出与「预期输出」逐行一致（196/14、(8,8,8)(8,21,21)、(22,22,22) 而非 (204,204,204)、784/28、1764/42、1980/60）。
- 本机按源码规则复算：qwen4_exp 交错槽位 T={0,3,…,30}（11）、H=slice(1,33,3)={1,4,…,31}（11）、W=slice(2,30,3)={2,5,…,29}（10），与页面第二张槽位表逐项一致；qwen2_vl 分段 [16,24,24]×2=[32,48,48] 对应 cos/sin 维度 T[0,31]/H[32,79]/W[80,127]。
- 页面内链：../rope/、../positional-encoding/、../vit/、../qwen3-8-flash-next-dataflow/、../qwen3-5-dataflow/index.html 均存在；overview.html 与 index.html 双向链接；validate.py wiki/mrope/index.html 返回 validation ok。

## 问题

- [轻微·技术] 第 2 章「本章问题」Q2 解答与「视频」行：把视频时间分量写成无条件「逐帧加一」（$t=s+k$），并紧接在取自 qwen4_exp 的 F1 生成式之后（"视频 $n_t>1$ 时 $T$ 逐帧递增"）。本页主例 Qwen3.8-Flash-Next（model_type=qwen4_exp）的 get_rope_index 实际把视频拆成 t=1 的单帧网格，每帧 T 恒为该帧起始位置、位置轴每帧推进 $\max(g_h,g_w)/\text{merge}$，「逐帧加一」只对论文与 qwen2_vl 口径成立。｜引文依据：qwen4_exp L2067-2069 "Separate video grid thw into multiple grids because timestamps are used to separate videos." + "video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)" + "video_grid_thw[:, 0] = 1"（同模型族的 qwen3-5-dataflow 页亦记「每帧的 T 维恒为该帧起始位置…若不拆分，T 维才会承载帧号 0,1,2」）｜修复要求：在视频行与该问答处标注「逐帧加一」属论文与 qwen2_vl 口径，或在「简化条件及其限制」中补一句「Qwen2.5-VL 起视频按帧拆分为单帧网格，帧内 T 恒为该帧起始位置」，使之与主例实现一致｜修复：｜复验：
- [轻微·技术] 第 3 章「补充：为什么 32 个槽位对应 64 个旋转维」：写成「RoPE 的每个频率槽位使用相邻的两个维度」。所引 transformers 实现用 rotate_half 将前半与后半配对（第 $i$ 维配第 $i+d/2$ 维），两个维度相隔 $d/2$、并不相邻（qwen4_exp 的 64 个旋转维中配对为 $(i,i{+}32)$）。｜引文依据：qwen2_vl L173-177 与 qwen4_exp L566-570 `def rotate_half(x): x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2 :]; return torch.cat((-x2, x1), dim=-1)`｜修复要求：把「相邻的两个维度」改为「相隔 $d/2$ 的两个维度」，或删去配对位置的描述、只保留「每个频率对应两个旋转维、$64/2=32$ 个频率」这一数量结论（数量本身正确）｜修复：｜复验：
- [轻微·可读性] 核心问题 Q3 解答「交错排布…使每个分量都覆盖从低频到高频的完整范围」与第 3 章正文「每个分量都近似覆盖从高频到低频的整个范围（各自缺少数个端点附近槽位）」对同一事实的限定强弱不一致，核心问题答案漏掉了「近似」限定。｜引文依据：不适用｜修复要求：核心问题 Q3 解答补上「近似」限定（可写「近似覆盖从低频到高频的整个范围，各自缺少数个端点附近槽位」），与正文一致｜修复：｜复验：
- [轻微·格式] blockquote.meta「主要依据」只列「Qwen2-VL 论文（arXiv:2409.12191v2）§2.1、§3.3.2」，而「来源与范围说明」写 C1–C7「均出自…§2.1 与 §1」；C5（三维动机）引用的原文句出自 §1（Introduction），按 meta 列出的范围核对不到。｜引文依据：论文 §1 "Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions."｜修复要求：meta 的依据列表补上 §1，与来源章节一致｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布（四条轻微问题均已定位到具体行与来源片段：前两条为限定条件/配对描述的精确性，后两条为同页措辞与依据清单的一致性；均不影响三分量拆分、槽位配额、推进量三条核心结论，可在下一轮或发布后顺带修）
