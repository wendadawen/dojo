<!-- review-meta
round: 5
page: wiki/glm-5-3-flash-dataflow/index.html
reviewed_content_sha256: 9a6515de628c89dd
-->
# GLM-5.3-Flash 前向数据流审查记录（第 5 轮）

- 页面版本：8d209f1d989a0b64395a7354007b9202c6b884a2（工作树，未改动）
- 审查时间：2026-09-14 17:06
- 审查者：编排者派发的独立审查者（未参与写作，未参与第 1–4 轮审查；未读取 research/ 下任何文件）
- 适用规范：guides/model-dataflow.md（页面 dojo:type=dataflow）；记录格式按 guides/concept/check.md §3
- 已完整阅读章节（含折叠块与全部图注）：1. 关键规格｜2. 整体数据流（含「45 层的类型排布」）｜3. 单层内部数据流（含代码折叠块）｜4. mHC：4 路残差流怎么读写（含「末端 4 路如何收敛回 1 路」折叠块）｜5. KDA 层：34 层线性注意力（递推公式 / 遗忘门的实际取值范围 / 两套实现）｜6. DSA 层：11 层稀疏注意力（k-pool / 打分公式 / 跨层共享 / 「展开：把预算缩到 16」折叠块）｜7. MoE 路由（每 token 激活量的构成）｜8. 位置信息从哪来｜9. 长上下文下的实际收益｜10. FP8 量化的覆盖范围｜11. 多模态：图像与视频共用 token｜12. MTP 层｜13. 核对方式｜来源与范围说明（C/F/N 三组 + 简化条件 + 范围之外）
- 来源获取：zai-org/GLM-5.3-Flash 的 config.json / README.md / model.safetensors.index.json 与 62 个分片 safetensors 头（HTTP Range）；huggingface/transformers 的 modeling_glm5_next.py 与 configuration_glm5_next.py（历史修订 eb4d9e2a64=2026-08-26、83d46aa2a2=2026-08-28，当前 main=93ebf6b111）与 cache_utils.py；arXiv:2602.15763v2 正文 §2.1/§2.1.1/§2.1.2。图内数值按 SVG 几何像素复算。

## 问题

- [轻微·来源支持] §5「遗忘门的实际取值范围」末段（正文）：「这个下界的作用是防止状态被单步彻底清零。」｜引文依据：官方材料只定义该键，未陈述其用途——configuration_glm5_next.py 的字段说明为 `linear_lower_bound (float, optional, defaults to -5.0): Whether the forget gate has a lower bound to apply to the decay.`，config.json 中 `linear_attn_config.gate_lower_bound: -5.0` 只有取值；前文已自行给出 g∈(-5,0)、e^g∈(e^-5,1)，该句是把数学后果写成了设计动机。｜修复要求：改为陈述数学后果（如「因为 σ(·)>0，衰减因子 e^g 恒大于 e^-5，单步衰减不会把状态清零」），或在句末标注为推断。｜修复：｜复验：
- [轻微·技术] §3 代码折叠块 summary「代码：单层执行序（对应源码 L1291–1329）」｜引文依据：该段落（`residual = hidden_states` 到 `return hidden_states, topk_indices`）在官方 modeling_glm5_next.py 中的实际位置为——eb4d9e2a64（2026-08-26）L1293–1329、83d46aa2a2（2026-08-28）L1294–1330、当前 main L1296–1332；页面未标注所依据的修订，起点与任一修订都差 2–5 行。｜修复要求：在 §13 注明所依据的版本（commit/transformers 版本），并把行号改为与该版本一致；或删去具体行号，改为「对应 Glm5NextTextDecoderLayer.forward」。｜修复：｜复验：

## 本轮核对依据（抽样，均为来源原文片段或实测数值）

**结构与配置（C 组）**
- [C1] config.json：`text_config.layer_types` 45 项，索引 3,7,11,…,43 为 `deepseek_sparse_attention`（11 项），其余 34 项 `linear_attention`；`linear_attn_config.full_attn_layers = [3,7,11,15,19,23,27,31,35,39,43]`。checkpoint：`self_attn.q_conv1d.weight` 34 个、`self_attn.kv_a_proj_with_mqa.weight` 12 个（层 3,7,…,43 共 11 + 层 45），两组层号与 layer_types 逐项一致。
- [C2] config.json `qk_rope_head_dim: 0`；configuration_glm5_next.py `validate_architecture()`：`raise ValueError(f"Expecting NoPE for the DSA attention layers, but got {self.qk_rope_head_dim} as RoPE dim.")`；modeling 主干 forward 每层传 `position_embeddings=None`。
- [C3] config `quantization_config`: `quant_method=fp8, fmt=e4m3, weight_block_size=[128,128]`；checkpoint 37,338 个 `weight_scale_inv`，逐张量核对形状全部等于 ⌈N/128⌉×⌈K/128⌉（0 例外）；被量化的 self_attn 张量只有 `q_a_proj/q_b_proj/kv_a_proj_with_mqa/o_proj`（各 12 个＝11 层 DSA + MTP），KDA 层 self_attn 下被量化张量数 **0**；`kv_b_proj`、indexer 全部权重、视觉塔、嵌入、输出头均未被 scale 覆盖。
- [C4] config `indexer_types` 45 项全为 `"full"`；modeling：`self.skip_topk = config.indexer_types[layer_idx] == "shared"`、`self.indexer = None if self.skip_topk else Glm5NextTextIndexer(...)`。
- [C5] checkpoint 层 45 有 `eh_proj.weight [4096,8192]`、`enorm.weight [4096]`、`hnorm.weight [4096]`、`shared_head.norm.weight [4096]`，带 `kv_a_proj_with_mqa` 与 indexer，无 `hc_*` 与 `q_conv1d`；`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]`。
- [C6] 在 modeling_glm5_next.py 与 configuration_glm5_next.py 中检索 `mla_use_nope` / `first_k_dense_replace` / `mhc` / `scoring_func` / `topk_method` / `indexer_rope_interleave` / `index_share_for_mtp_iteration` / `num_nextn_predict_layers` / `moe_router_dtype`，命中数均为 0。

**公式（F 组）**
- [F1] `recurrent_kimi_delta_attention` 与页面递推式逐项对应（`last_state *= g.exp()`；`delta=(v-∑S·k)*β`；`S += k⊗δ`；`out=∑S·q`；`q,k` 经 l2norm，`q *= 1/√128`），`chunk_kimi_delta_attention` 默认 `chunk_size=64`。独立抽取两函数实跑：L∈{1,7,64,65,130} 下输出/终态最大误差 7.5e-9…3.2e-6，与页面「10^-7 量级、浮点累加顺序差异」同量级（页面表格为单次运行值，随输入分布浮动）。
- [F2] `Glm5NextTextTopkRouter.forward`：`scores = router_logits.sigmoid()`；`scores_for_choice = scores + e_score_correction_bias` 仅用于 topk；`topk_weights = scores.gather(...)`（不含偏置）→ 归一化 → `*= routed_scaling_factor(2.5)`。`n_group=topk_group=1` ⇒ `group_mask`/`score_mask` 全 1。`_apply_gate`：`gate.clamp(min=None, max=10)`、`up.clamp(min=-10, max=10)`；输入 [-50,-10,0,5,10,50] 复算得 gate=[-50,-10,0,5,10,10]、up=[-10,-10,0,5,10,10]，与页面一致。`Glm5NextTextMoE.forward`：`experts(...) + self.shared_experts(residuals)`（共享专家吃原始输入）。
- [F3] `Glm5NextTextIndexer.forward`：`gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)`；`packed_states = cat([k, gate_scores, valid_channel], dim=-1)`；`get_pooled_states`：`logits = grouped_gate_scores + index_kpool_compress_ape`（ape 形状 [4,128]），沿池内维 `softmax(dim=2)`。打分式 `relu(q·k̄ /√128)`、头权 `·32^-0.5`，实测缩放常数 0.08838835 / 0.17677670 与式一致。

**外部数字与实测条件（N 组）**
- [N1] index.json 共 76,108 个张量（37,338 个 `weight_scale_inv`）；排除 scale 后按 shape 累加 = **321,323,031,390**。
- [N2] 分项累加 = **17,376,348,990**（MoE 9,562,238,784 + KDA 4,682,897,792 + DSA 1,374,058,752 + 嵌入/输出头 1,268,776,960 + 稠密 MLP 452,984,832 + mHC 35,391,870）；逐项独立重算全部相符（KDA 单层 137,732,288、DSA 单层 124,914,432 其中 indexer 7,471,872＝5.98%）。README 原文：「320B total parameters and just 18B active parameters」。
- [N3] checkpoint visual 命名空间 347 个张量按 shape 累加 = **563,627,008**，与页面完全一致；token 换算式 `grid.prod(-1)//spatial_merge_size²`（空间合并 2×2）在源码 `get_image_features`/`get_video_features` 中。
- [N4] 未独立复现（需新版 transformers 构建 8 层缩小模型；本机为 transformers 4.57.6，不含 glm5_next）。页面已写明全部构造条件，未发现与其他数字矛盾。
- [N5] 按源码 comb 管线独立复现（fn~N(0,0.02)、base=0、scale=1、hc_mult=4、hidden=4096、B=1/L=6）：iters=1/3/10/20/40 → 行和最大偏差 5.10e-1 / 1.86e-1 / 4.78e-2 / **1.01e-2** / 4.76e-4，列和 4.05e-6 / 1.19e-6 / 1.07e-6 / **1.07e-6** / 1.01e-6，与页面表格一致；迭代序确为「先 1 次列归一化，再 19 轮（行+列）」，共 20 次列 / 19 次行、末步落列。`Glm5NextTextHyperHead` = `hidden_streams.mean(dim=2)`，docstring「Unlike DeepSeek-V4, this is an unweighted mean.」
- [N6] 单站点 24×16384+24+3 = 393,243，×2×45 = 35,391,870；checkpoint `hc_attn_fn [24,16384] BF16`、`hc_attn_base [24] F32`、`hc_attn_scale [3] F32`。
- [N7] safe 分支 `g = λ·σ(e^A⊙(W_fb W_fa x + b_dt))`、λ=-5（config `gate_lower_bound: -5.0`）；官方 `_init_weights` 对 safe 分支 `zeros_(A_log)`。独立复现（A_log=0、dt_bias=0、B=1 L=16 标准正态）得 g∈[-4.17,-0.82]、衰减因子∈[0.0155,0.4390]，与页面 [-4.02,-0.84] / [0.018,0.431] 同区间同量级。
- [N8] `select_k = min(index_topk // index_kpool, index_scores.shape[-1])`、候选池数 = ⌈L/4⌉；表中 6 行按此式重算（含 25.0%、100%、50%、12.5%、1.56%、0.195%）全部相符。
- [N9] `append_visible_tail`：`tail_count = visible_count.remainder(index_kpool)`；`output_width = index_topk + index_kpool - 1`（2048+3=2051）；折叠块中 4 行尾巴长度（61%4=1、62%4=2、63%4=3、64%4=0）自洽；`index_topk % index_kpool != 0` 在配置校验中直接抛错，故「缩到 16」合法。
- [N10] cache_utils `lazy_initialization`：`torch.zeros((*conv_states.shape[:-1], conv_kernel_size), dtype=conv_states.dtype, ...)`，宽度等于 `conv_kernel_size`＝4（非 3）、dtype 随激活；modeling `update_recurrent_state(last_recurrent_state.to(torch.float32))`；`packed = cat([k, gate_scores, valid_channel])` = 128+128+1 = **257**。按潜向量口径重算 4 行：0.204/0.264、0.655/2.112、2.204/8.448、16.661/67.588 GiB 与 22.89/68.97/73.91/75.35% 全部相符；交叉点 149,291,008 B ÷(34×769×2 B)=2855 token；漏 257 维时 149,291,008÷(34×512×2)=4288，L=4096 行错算为「多占 3.54%」亦复算相符。
- [N11] `expand_kv` 先展开成 64 头再入 cache：K、V 各 [B,64,L,256]，每 token 每层 2×64×256 = 32768 元素，为潜向量口径 64 倍。
- [N12] `get_placeholder_mask`：`in_video_span = (input_ids==video_start_token_id).cumsum(-1) > (input_ids==video_end_token_id).cumsum(-1)`；`special_image_mask = (id==image_token_id) & ~in_video_span`、`special_video_mask = (id==image_token_id) & in_video_span`；`video_token_id=154855` 在判定路径中未被使用。
- [N13] 全量重算：总 321,323,031,390 = 主干 **313,890,438,974**（313.89 B）+ MTP **7,432,592,416**（7.43 B）；routed 专家 311,653,564,416（311.65 B，96.99%）；主干 42 层 routed 304,405,807,104（304.41 B，96.98%）；主干 routed/主干 = 96.98%，17.38 B/313.89 B = 5.54%。
- 附：index.json `total_size` = **328,326,771,576**，按 dtype×shape 逐张量累加完全相符；328,326,771,576/321,323,031,390 = 1.022 字节/参数，高估 2.2%。

**外部文献**
- README 含「GLM-5.3-Flash starts from a newly trained base model」；arXiv:2602.15763v2 §2.1「GLM-5 scales to 256 experts and reduces its layer count to 80」「744B parameter model (40B active parameters)」；§2.1.1「90% of attention entries in long contexts are indeed redundant」「DSA reduces the attention computation by roughly 1.5-2× for long sequences」；§2.1.2 比较 GDN / SimpleGDN / SWA。页面「范围之外」对报告的转述与之逐项一致。

**机械项**
- KaTeX：7 处 `$$` 全部配平，全部 `$...$` 片段（含 dojo:summary 内 6 段）经 `katex.renderToString(..., {throwOnError:true})` 通过；正文/标题/列表/表格无 Unicode 数学字符（仅「·」作分隔、「→」作流程箭头），`<text>` 内无公式，公式均在 `<foreignObject>` 中。
- 结构图：7 张内联 SVG，均无元素越界（含 foreignObject），矩形两两不重叠；类型排布图的 12 个刻度 x 坐标按 45 层总宽 610px 复算，落在 layer 3,7,…,43 的中心，与 11 个 DSA 层 + 首层 0 相符。
- 链接与元数据：10 个站内前置概念链接（residual-connection / kda / kv-cache / delta-rule / low-rank-projection / quanti\nzation-basics / swiglu / aux-loss-free-routing / vit / speculative-decoding）全部存在，无「（待生成）」；`dojo:type=dataflow`、`dojo:topics=模型结构,注意力机制`（均在上位词表内）、`dojo:tag=数据流`（在 ALLOWED_TAGS 内）；`.dojo/scripts/validate.py` 返回 `validation ok`。
- 表述：全文通读（含折叠块与图注）未见元话语（「本页将…」「下面来看…」）、会话指代（我/我们/你）、调试叙事或临场评价；推断处（§8 callout、§12 模块用途）均已显式标注为推断。
- 本机无 headless 浏览器（无 playwright/puppeteer），渲染一项降级为静态审查：KaTeX 语法 + SVG 几何 + 本地资源（katex/auto-render/prism/dojo-dataflow.css）存在性均已核对。

## 结论

- 处置：修复 2 条轻微；本轮 0 阻断 0 重要，核心结论、数字与来源全部可核对，不构成发布障碍
- 统计：阻断 0 / 重要 0 / 轻微 2