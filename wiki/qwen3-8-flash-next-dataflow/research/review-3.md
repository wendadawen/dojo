<!-- review-meta
round: 3
page: wiki/qwen3-8-flash-next-dataflow/index.html
reviewed_content_sha256: 1915e0be39b3d8ec
-->
# Qwen3.8-Flash-Next 前向数据流审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 8af751aedf1aa3b82e32b1f9297b19e512502aea
- 审查时间：2026-09-13 19:50
- 审查者：编排者派发的独立审查者（未参与写作与前序审查）
- 已完整阅读章节：1. 关键规格 / 2. 交互式数据流 / 3. 要点（整体结构、GDN、QSA、Gated Residual、N-gram Embedding、MoE 路由、长上下文开销的来源）/ 4. 视觉编码器与多模态融合（4.1–4.8）/ 5. 核对方式 / 来源与范围说明，含全部 9 个视图的节点、公式与 tooltip 文本
- 来源获取：huggingface.co/Qwen/Qwen3.8-Flash-Next 的 config.json 原文、huggingface/transformers commit 36deb0b5 的 configuration_qwen4_exp.py 与 modeling_qwen4_exp.py、QwenLM/Qwen3.8-Flash-Next 仓库与模型卡；`.dojo/scripts/validate.py wiki/qwen3-8-flash-next-dataflow/index.html` 返回 validation ok；16 个前置概念内链逐一验证存在。

## 已核对通过（非问题，供复验参考）

- config 全部字段与页面取值逐项一致（hidden_size/hc_count=4/hc_lowrank=320/indexer_budget=2048/indexer_compress_ratio=4/indexer_n_heads=4/indexer_kv_heads=1/moe_intermediate_size=640/shared_expert_intermediate_size=640/num_experts=512/num_experts_per_tok=10/ngram_size=3/heads_per_ngram=8/ple_layer_ids=[2]/split_ngram_parts=128/make_ngram_vocab_size_divisible_by=128/partial_rotary_factor=0.25/mrope_section=[11,11,10]/max_position_embeddings=262144/vocab_size=248320/tie_word_embeddings=false/image_token_id=248056/video_token_id=248057/vision_start=248053/vision_end=248054；vision_config depth=27/hidden_size=1152/intermediate_size=4304/num_heads=16/patch=16/temporal_patch=2/spatial_merge=2/num_position_embeddings=2304/out_hidden_size=2560）。
- N-gram 质数表按源码算法 `_find_nth_prime_after(base-1, head_idx+1)` 复算：20000003…20000171 共 16 个，和 320001446，对齐 128 得 320001536（补 90 行），÷128=2500012 行/分片，×160=51200245760，与页面完全一致；三个 checkpoint 乘子 23703573157769/20109073645365/8052911324071 与 `2*(splitmix64(seed+10007*ℓ+GAMMA*(j+1)) % B)+1`（seed=1234，ℓ=0，B=18571544855136）逐元素一致。
- 各分项与合计复算通过：MoE 单层全量 512×4915200+4915200+512×2560+2560=2522810880、激活 10×4915200+4915200+1310720+2560=55380480（2.195%）；48 层路由专家 2516582400×48=120795955200（67.11%）；超连接 2×6604800×48+6563840=640624640（0.36%）；视觉塔 267890544+143451648+33039616+2654208+1770624+124416=448931056；KV/索引器/GDN 缓存表 0.094/0.012/0.108→0.213、6.000/0.750/0.108→6.858、24.000/3.000/0.108→27.108 均复算成立；视觉 token 196/784/1764/2040/1568 与位置推进 14/28/42/60 及占比 0.075%/0.299%/0.673%/0.778%/0.598% 一致。
- 机制与源码一致：GDN 递归 `S=e^g S + k(β(v−S^T k))^T`、输出 `S^T q /√d_k`；索引器池化取均值、块得分 `ReLU(q_h·k̄)` 按头求和再 ÷√128（relu 在 sum 之前）、残块 `tail` 无条件 concat、掩码与因果掩码按位与（浮点相加）；超连接读出/写回公式（`σ(up(silu(down(x̂)/4)))` 取 mean、`2σ(W_inj x̂/4)`）；`q_proj` 两倍宽 12288、gate 走 `attn_output*sigmoid(gate)`、partial RoPE 用 `cos.shape[-1]`；`apply_interleaved_mrope` 文档串「chunked [TT…HH…WW…] → interleaved [THW…]」与 4.7 节及槽位表一致（H=slice(1,33,3)、W=slice(2,30,3)，各占 T11/H11/W10）；视觉 patch_embed Conv3d kernel=stride、pos_embed 双线性重采样、merger `use_postshuffle_norm=False` 使 LayerNorm 在 1152 维上、merger 参数 33039616 与带 bias 的 LayerNorm 账目吻合。

## 问题

- [重要·技术] §3「GDN（Gated DeltaNet，36 层）」第 2 条（行 167）与视图「GDN 层」节点 `alpha`（行 555）：把 g 的区间跨度写反，构成 a×b≠积 的数值断言错误。文中写「本机实测 g_t 落在 [-63.34, -0.15]，最大值约为最小值的 420 倍」；该区间最大值为 −0.15、最小值为 −63.34，−0.15 ≠ 420×(−63.34)，而真正成立的关系是 |最小值|/|最大值| ≈ 63.34/0.15 ≈ 422。｜引文依据：页内自述区间 [-63.34, -0.15]；源码 `torch_recurrent_gated_delta_rule` 中 `g_t = g[:, :, i].exp()`、`last_recurrent_state = last_recurrent_state * g_t`，确认 g 为负对数衰减。｜修复要求：两处均改为「最小值的绝对值约为最大值的 420 倍」（或「|g| 的最大值约为 |g| 最小值的 420 倍」），保留区间数字。｜修复：｜复验：
- [重要·规范] §2「交互式数据流」：整节内容不在 HTML 里，而由页内第二个 `<script>` 的 `VIEWS` 对象经 Cytoscape 画布生成；`#tabs`/`#cy`/`#legend` 都是空容器，脚本失效（禁用 JS、CSP 拦截、CDN 资源缺失）时该节只剩标题与空框，读者看不到任何视图。｜引文依据：guides/model-dataflow.md「视图：脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」；「发布前检查：交互视图在无脚本时仍可读」。｜修复要求：把各视图的节点/连线内容落成 HTML 结构或内联 SVG（脚本仅切换显隐），至少让默认「整体总览」视图在无脚本下可见；可参照同仓库 wiki/glm-5-3-flash-dataflow/index.html 的内联 SVG + foreignObject 写法。｜修复：｜复验：
- [轻微·表述] §1 note（行 139）、来源与范围说明（行 310，3 处）、视图 `overview` 节点 `mtp`（行 533）：以「本页」为主语的自我指代共 5 处（「本页统一称 QSA」「本页以累加值为准」「本页统一按实际行为称 QSA」「本页覆盖语言主干…不在范围内」「本页不展开其调度逻辑」）。｜引文依据：guides/concept/check.md 2.2 第 12 条「以"本页"为主语的自我指代」。｜修复要求：改为直接陈述或换主语，如「这 12 层统称 QSA」「此处以累加值为准」。｜修复：｜复验：
- [轻微·表述] §3 QSA 第 3 条（行 177）与视图节点 `score`(617)、`norm`(636)、`mix`(639)、`ids`(658)：共 5 处以「注意…」开头的提示语（「注意不是 softmax」「注意是 ReLU 后求和」「注意权重形式是 (1+w)」「注意是 mean 而非 sum」「注意注入用的是 id 而非 embedding」），属规范排除的元话语。｜引文依据：guides/concept/check.md 2.2 第 12 条「元话语（"本页将…""下面来看…""需要注意的是"）」。｜修复要求：删去提示词，将内容并入陈述句，如「该处为 ReLU 后求和，不是 softmax」。｜修复：｜复验：
- [轻微·技术] 视图 `hc` 节点 `down`（行 637）「除以 hc_count=4 是为了让输入尺度与流数无关」、§3 N-gram 第 3 条（行 197）与节点 `mod`（行 661）「用质数是为让各头碰撞模式尽量独立」、§3 GDN 第 3 条（行 168）「这使 S_{t-1}^T k_t 的量级可控」：这些是设计动机/作用推断，源码未给出理由，页面却以「是为了/使…」写成事实。｜引文依据：configuration/modeling 源码对 hc_count 除法与质数取模均无动机注释；页面他处对同类推断已加「（对实现的解读）」（如 4.2 位置重采样）。｜修复要求：三处按同页既有做法降级为明确标注的推断，或删去理由部分只留可核对的行为。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：修复（本轮无阻断项；2 项重要需在下一轮前关闭：负对数衰减跨度的倍数方向、无脚本时 §2 视图不可读）