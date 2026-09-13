<!-- review-meta
round: 1
page: wiki/qwen3-8-flash-next-dataflow/index.html
reviewed_content_sha256: 345d0a21cb795e70
-->
# Qwen3.8-Flash-Next 前向数据流 审查记录（第 1 轮）

- 页面：`wiki/qwen3-8-flash-next-dataflow/index.html`（`dojo:type=dataflow`，适用规范 `guides/model-dataflow.md`）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下的规划、修复或前序审查记录）
- 已完整阅读：第 1 节 关键规格；第 2 节 交互式数据流（overview / gdn / qsa / indexer / hc / ple / moe / vision / fusion 九个视图的全部节点 label、io、公式与 tooltip 说明，含图注与连线标签）；第 3 节 要点（整体结构 / GDN / QSA / Gated Residual / N-gram / MoE / 长上下文）；第 4 节 视觉编码器与多模态融合（4.1–4.8）；第 5 节 核对方式；来源与范围说明
- 来源获取方式：WebFetch + curl 抓取官方 `huggingface.co/Qwen/Qwen3.8-Flash-Next` 的 `config.json` 与 `README.md` 原文；curl 抓取 `raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen4_exp/` 下 `configuration_qwen4_exp.py` / `modeling_qwen4_exp.py` / `modular_qwen4_exp.py` 后本地逐段阅读；本页 `research/measured.md`

## 已核对通过（不作为问题）的主要项

config.json 全部结构键逐项对上（`layer_types` 36 linear + 12 full、`full_attention_interval=4`、`hidden_size=2560`、`hc_count=4`、`hc_lowrank=320`、`linear_num_key_heads=16`/`linear_num_value_heads=48`/头维 128、`num_attention_heads=24`/`num_key_value_heads=2`/`head_dim=256`、`indexer_n_heads=4`/`indexer_kv_heads=1`/`indexer_head_dim=128`/`indexer_compress_ratio=4`/`indexer_budget=2048`、`num_experts=512`/`num_experts_per_tok=10`/`moe_intermediate_size=640`/`shared_expert_intermediate_size=640`、`ple_layer_ids=[2]`、`partial_rotary_factor=0.25`、`mrope_section=[11,11,10]`、`rope_theta=1e7`、`mamba_ssm_dtype=float32`、`output_gate_type=sigmoid`、`mtp_num_hidden_layers=1`、`vision_config` 全项、`vocab_size=248320`、`image_token_id=248056`/`video_token_id=248057`/`vision_start_token_id=248053`/`vision_end_token_id=248054`、`tie_word_embeddings=false`、`split_ngram_parts=128`、`make_ngram_vocab_size_divisible_by=128`）。源码实现逐点对上（`hidden_states.repeat(1,1,hc_count)`、GatedResidual 的 `.mean(dim=-2)` 与 `λ=2·sigmoid(·/hc_count)`、`block_inject_weight` 仅 `use_combine=True` 时存在、indexer 均值池化 + `relu(scores).sum(-1)/sqrt(128)` + `topk(min(512,n_blocks))` + 残块无条件拼接 + 选择张量宽 `token_budget+ratio-1=2051`、`conv_dim=2·key_dim+value_dim=10240`、`g=-A_log.exp()*softplus(a+dt_bias)`、`beta=b.sigmoid()`、`repeat_interleave(num_v//num_k=3)`、`output_gate_type or hidden_act` 回退、`q_proj` 两倍宽 + `attn_output*sigmoid(gate)`、部分 RoPE 只旋转 `q[..., :rotary_dim]`、`_build_layer_multipliers` 公式与 `seed` 默认 1234、PLE 质数词表/前缀和偏移/(k-1)·dilation=9 空洞卷积、`|g|.clamp_min(1e-6).sqrt()*sign`、Vision `Conv3d(3,1152,(2,16,16),(2,16,16))`、`pos_embed` 48×48 双线性 `align_corners=True`、`merger use_postshuffle_norm=False`、2D RoPE 36 维 + 自身拼接 72、`cu_seqlens` 变长打包、`inputs_embeds.masked_scatter(image_mask, image_embeds)`、`get_vision_position_ids` 与 `current_pos += max(h,w)//spatial_merge_size`、`recomposition_frequencies` 的 `slice(1,33,3)`/`slice(2,30,3)` 与槽位归属 T=11/H=11/W=10）。数字复算全部一致：180.00B 分解 = 125,743,653,760 + 51,200,245,760 + 448,931,056 + 2,607,150,848 = 179,999,981,424；N-gram 16 质数和 320,001,446 / 对齐 320,001,536（补 90）/×160 = 51,200,245,760；Vision 448,931,056 分组累加一致；超连接 6,604,800×2×48+6,563,840 = 640,624,640（0.36%）；MoE 单层 2,522,810,880 / 激活 55,380,480（2.195%）；路由专家 120,795,955,200（67.11%）；KV/索引器/GDN 缓存 0.094/0.75/6.0/24.0 GiB 与合计 0.213/0.951/6.858/27.108 GiB；B=((2^63-1)//248320)//2 = 18,571,544,855,136。三个真实乘子 23703573157769 / 20109073645365 / 8052911324071 用 `seed=1234`、`ple_layer_index=0`、`V=248320` 复现一致。可视长度 2049/51.21%、2049/25.61%、2048/17.07% 与上界 2051 自洽。页面无可运行代码（唯一 `<pre>` 为槽位示意图），按静态审查。`.dojo/scripts/validate.py` 返回成功；16 个站内概念链接目标页面全部存在。

## 问题

- 阻断｜官方源码（transformers `models/qwen4_exp/modeling_qwen4_exp.py`、`configuration_qwen4_exp.py`、`modular_qwen4_exp.py`）｜第 4.7 节末句（index.html:279），同一表述复见于交互图 fusion 视图 `mrope` 节点说明（index.html:740）｜页面把 MRoPE「交错排布 vs chunked 排布」的出处写成「源码 apply_interleaved_mrope 文档字符串所描述的」，但被引来源中不存在该函数，该对比也不能追溯到任何被引来源｜引文依据：在 qwen4_exp 三个文件中检索 `apply_interleaved_mrope` 零命中；实现该逻辑的函数是 `Qwen4ExpTextRotaryEmbedding.recomposition_frequencies`（modeling_qwen4_exp.py:140、视觉侧 :1754），其文档字符串全文为「Recompose the frequencies into the final spatial layout used per each grid.」，无 chunked 排布对比；三文件中 `chunked` 仅出现在 GDN 的 `torch_chunk_gated_delta_rule`（:262-275），与位置编码无关｜修复要求：删除对 `apply_interleaved_mrope` 文档字符串的引用；若保留「与 chunked 排布不同」的比较，改引实际函数 `recomposition_frequencies`（写明文件与行号，如 modeling_qwen4_exp.py:140）并注明该比较是审查者/页面的推断，或直接删除该句只保留可核对的 `slice(1,33,3)`、`slice(2,30,3)` 事实。

- 阻断｜官方模型卡（README.md）｜第 1 节关键规格表「MTP 草稿层」行（index.html:134）与交互图 overview 视图 `mtp` 节点（index.html:534）｜页面把 MTP 草稿层记为「1 层，2.61B 参数」，与官方模型卡给出的 MTP 参数量不符，且全页未按数据流规范写明这处分歧（第 5 节与「来源与范围说明」只声明已复现官方的 125B/51B/6B 三项，恰好略过 MTP）｜引文依据：官方 README.md 第 46 行「Number of Parameters: 125B with 6B activated, plus 51B n-gram embedding and 4B MTP」、第 70 行「MTP: 1 layer, trained with multi-steps」（官方亦为 1 层）；页面据权重头算得 2,607,150,848（≈2.61B）｜修复要求：在 MTP 行写明官方「4B」与本机按张量头累加所得「≈2.61B」的分歧，说明以哪一方为准及判定依据（规范要求「配置与源码对不上时，以源码为准并在页面里写明分歧」）；同时把「六项官方对外表述」的复现说明补全为四项或注明 MTP 一项存在分歧。

- 重要｜规范 `guides/model-dataflow.md`（「实测」与「发布前检查」节）｜第 5 节末段（index.html:300）｜页面正文把实测脚本与原始输出写成「存于 research/ 目录」，并逐一列出 `read_headers.py`、`verify_structure.py`、`count_params.py`、`assert_dims.py`、`probe1`~`probe8`、`render_check.py`、`measured-output.txt` 等文件名，但这些文件已从仓库移除，页面指向了已移除的文件路径；该陈述也与本页 `research/measured.md` 的登记（「原先存放在本目录下，现已从仓库移除」）不一致｜引文依据：`research/measured.md` 第 3–4 行「本页的实测产物原先存放在本目录下，现已从仓库移除（内容不发布，且体积可观）」；目录内现仅 `measured.md`、`prereq-audit.md`；规范原文「实测脚本与运行输出不随仓库分发，清单记在 research/measured.md；页面正文引用实测结论时写清『实测得到』而不是指向已移除的文件路径」｜修复要求：删除 research/ 下的脚本与输出文件路径清单，改为「实测脚本与原始输出不随仓库分发，清单与判定见 research/measured.md」，正文实测结论统一用「实测得到/本机实测」表述。

- 轻微｜规范 `guides/model-dataflow.md`（表述节：不写引导语）｜第 2 节引导句（index.html:142）｜元话语与对本页的指代：「下图是实际前向路径，也是本页的主体。九个标签页切换粒度……」为引导语并评述本页结构，规范要求「直接进入路径」｜引文依据：不适用｜修复要求：删去「也是本页的主体」一类自述，改为直接说明该视图表达的前向路径与标签页含义。

- 轻微｜规范 `guides/model-dataflow.md`（表述节，与 concept 同一套）｜第 3 节「长上下文开销的来源」末段（index.html:222）｜元话语「需要注意 2048 这个预算是 config 的默认取值」，正是规范点名的「需要注意的是」式引导｜引文依据：不适用｜修复要求：删去「需要注意」，直接陈述「2048 为 config 默认值，块选择仅在完整块数超过 512 时生效」。

- 轻微｜规范 `guides/model-dataflow.md`（表述节：不写引导语）｜第 4 节开头（index.html:225）｜引导语「下面的『视觉编码器』与『多模态融合』两个视图给出完整路径」，属于规范排除的「下面……」式引导｜引文依据：不适用｜修复要求：删去该句。

- 轻微｜规范 `guides/model-dataflow.md`（表述节）｜第 5 节开头（index.html:295）｜元话语「本页的数字分三类来源，各自的取得方式如下」，且后文实际给出四个分组（结构与维度、公式与数值行为、checkpoint 缓冲数值、端到端数据流），「三类」与后文分组数不符｜引文依据：不适用｜修复要求：删去元话语式开头，直接进入各来源的取得方式，并使分组数与表述一致。

- 轻微｜规范 `guides/model-dataflow.md`（表述节：不写临场评价）｜第 4.6 节首句（index.html:264）与第 3 节 GDN 要点（index.html:170）｜临场评价性提示语「这是容易被忽略的一点」「易被忽略的状态」，属于对读者注意力的评述而非路径内容｜引文依据：不适用｜修复要求：删去「容易被忽略」类评价，保留其后的事实陈述。

- 轻微｜本机实测记录（页面 index.html:167 与交互图 gdn 视图 `alpha` 节点 index.html:556）｜「本机实测 g 落在 [-63.34, -0.15]，衰减强度跨三个数量级」｜由 63.34/0.15 ≈ 4.2×10²，跨度约 2.6 个数量级，写作「三个数量级」偏大，表述与给定数值不匹配｜引文依据：页面自述 g ∈ [-63.34, -0.15]（比值 422，log10≈2.63）｜修复要求：改为「跨约 2.6 个数量级」或直接写「最大值约为最小值的 420 倍」。

- 轻微｜页面内部一致性（规范：同一写法全页一致）｜meta description（index.html:8）、`dojo:summary`（index.html:9）与 page-lead（index.html:111）用「第 2 层」，而第 3 节要点（index.html:131、:161）用「第 1 层（0 起）」｜同一层被 meta/lead 以 1 起编号、正文以 0 起编号描述，读者易误认为两层｜引文依据：index.html:131「N-gram 注入层（0 起）仅第 1 层｜ple_layer_ids=[2]（1 起）」对照 index.html:111「第 2 层额外注入一张 51.2B 参数的哈希 N-gram 查表」｜修复要求：统一编号口径（或统一注记「（0 起）」），使 meta、summary、lead 与正文指向同一层。

- 轻微｜外部模型陈述无来源支持（规范：无来源支持的判断不写成结论）｜第 3 节 QSA 要点（index.html:176）与第 4.2 节（index.html:233）｜（a）「这与 DeepSeek-V4 用数据相关权重加权求和的压缩不同」——被链接的 `../dsa/` 概念页描述的是 indexer 打分后做 top-k 位置的完整 MLA 注意力，并无「数据相关权重加权求和」的压缩，该对比在被引来源中得不到支持；（b）「这个设计的意义是位置信息跟随图像真实长宽比，不需要把输入 resize 成固定尺寸」是对设计动机的推断，来源中并无该动机陈述｜引文依据：`wiki/dsa/index.html` 的 `dojo:summary`「lightning indexer 给历史位置打分，只对 top-k=2048 个位置做完整 MLA 注意力」；modeling 中 VisionModel 仅有注释「How the (square) learned position grid is resampled to each image's grid.」｜修复要求：（a）删去对 DeepSeek-V4 压缩方式的对比，或补上可定位的出处；（b）把设计动机改为标注的推断（如「这样位置网格可随图像实际长宽比伸缩」并注明为对实现的解读），不作为来源结论陈述。

## 结论

- 统计：阻断 2 / 重要 1 / 轻微 8
- 处置：有问题，修复（两处阻断须先关闭：虚构的源码函数名引用、MTP 参数量与官方模型卡不符且未标注分歧；重要项须修正 research/ 已移除文件路径的引用）
