<!-- review-meta
round: 2
page: wiki/qwen3-8-flash-next-dataflow/index.html
reviewed_content_sha256: 345d0a21cb795e70
-->
# Qwen3.8-Flash-Next 前向数据流审查记录（第 2 轮）

- 页面版本：e9daa7072844aa7cbd47299d4abba9883f7af819（index.html 工作树 git hash-object）
- 审查时间：2026-09-13 19:08
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：head 的 description / dojo:summary；1. 关键规格；2. 交互式数据流（含 overview / gdn / qsa / indexer / hc / ple / moe / vision / fusion 九个视图的节点 label、公式与说明）；3. 要点（整体结构、GDN、QSA、Gated Residual、N-gram Embedding、MoE 路由、长上下文开销）；4. 视觉编码器与多模态融合（4.1–4.8）；5. 核对方式；来源与范围说明。
- 来源核对方式：官方 config.json（huggingface.co/Qwen/Qwen3.8-Flash-Next）与 HF API 元数据（safetensors.parameters BF16=179999981424、I64=35，total_size 359999963128）；官方实现 transformers@36deb0b5 的 configuration_qwen4_exp.py、modeling_qwen4_exp.py、modular_qwen4_exp.py、models/qwen3_5_moe/modeling_qwen3_5_moe.py、vision_utils.py、models/qwen2_vl/image_processing_qwen2_vl.py；真实 checkpoint 的 model.safetensors.index.json（1658 个张量）与按 HTTP Range 读取的分片 JSON 头（核验 q_proj / o_proj / index_qk_proj / in_proj_qkv / in_proj_z / out_proj / experts.gate_up_proj / experts.down_proj / hyper_connection.* / ple.* / visual.* / mtp.* 等形状）。另核对站内前置概念链接与 `.dojo/scripts/validate.py`。

已复算并全部对上的关键数字（列出以说明本轮覆盖面）：总参数 179,999,981,424（= (359999963128−280)/2，与 HF 元数据 BF16 值一致）；语言主干 125,743,653,760；N-gram 表 320001536×160=51,200,245,760；单 token 激活 6,035,598,720；MTP 2,607,150,848；视觉塔 448,931,056；MoE 单层全量 2,522,810,880 / 激活 55,380,480；超连接每层 13,209,600、48 层加 mixer 640,624,640；B=18,571,544,855,136（V=248320）；16 个质数之和 320,001,446 → 补 90 行对齐 320,001,536；可见 token 上界 2051、位置 4000/8000/11999 的 2049/2049/2048；KV/索引器/GDN 缓存表 4K–1M 各行；MRoPE 槽位 0..31 的 THW 交错归属；448/896/1344 的网格与 token 数。

## 问题

- 阻断｜技术来源｜§5「核对方式」末段 +「来源与范围说明」表末行｜页面指向已移除的 `research/` 文件路径，并声称实测脚本与原始输出「存于 `research/` 目录」；实际该目录下已无任何脚本或输出。规范 `guides/model-dataflow.md` 明文要求「页面正文引用实测结论时写清『实测得到』而不是指向已移除的文件路径」。｜引文依据：页面原文「实测脚本与一次性跑完全部脚本的原始输出（1058 行）存于 research/ 目录：read_headers.py 读张量头，verify_structure.py 做结构交叉验证，count_params.py 核算参数量…measured-output.txt 为原始输出」；同表末行「本机实测，脚本与原始输出见 research/」；实际 `ls wiki/qwen3-8-flash-next-dataflow/research/` 仅 measured.md（其正文写明「现已从仓库移除」）、prereq-audit.md、review-1.md。｜修复要求：删除末段的脚本清单与「存于 research/ 目录」表述，把「来源与范围说明」表中「脚本与原始输出见 research/」改为「本机实测得到」；确需给清单时指向 `research/measured.md` 的登记，不得指向已移除路径。改完重新核对该段每一条实测结论仍在正文中以「实测得到」表述。

- 阻断｜技术来源｜§4.4 表「1920×1080 图」行；§4.6 表「(1,66,120)」行；「视觉编码器」视图 tok 节点｜1920×1080 的网格、patch 数与视觉 token 数算错。官方 smart_resize 以 `factor = patch_size × merge_size = 16 × 2 = 32` 取四舍五入，1080 → round(1080/32)×32 = 1088，故 grid=(1,68,120)、patch 数 68×120=8,160、视觉 token 8,160/4=2,040、占 262144 为 0.778%；页面写成 (1,66,120)/7,920/1,980/0.755%，对应 h=66（即 1056 像素）是 floor 而非 round 的结果，与官方处理器不符。｜引文依据：`models/qwen2_vl/image_processing_qwen2_vl.py`「h_bar = round(height / factor) * factor」「factor = patch_size * merge_size」；`preprocessor_config.json`「patch_size: 16」「merge_size: 2」；round(1080/32)=34 → 1088//16=68。页面原文「1920×1080 图｜(1,66,120)｜7,920｜1,980｜0.755%」及「(1,66,120)｜1,980｜60｜33.0×」。｜修复要求：§4.4 该行改为 grid (1,68,120)、patch 8,160、视觉 token 2,040、占比 0.778%；§4.6 该行 token 数改 2,040，比值改 34.0×（推进量 max(68,120)/2=60 不变）；「视觉编码器」视图 tok 节点 1920×1080 改 2,040，并把五项占比串 0.075%/0.299%/0.673%/0.755%/0.598% 的第四项改 0.778%。其余四行（448×448、896×896、1344×1344、16 帧 448×448）复算无误，保留。

- 重要｜技术来源｜§3「QSA」第 2 条；「索引器」视图 pool 节点｜「与 DeepSeek-V4 用数据相关权重加权求和的压缩不同」属无来源支持的机制归因。本页「来源与范围说明」表逐行只列 Qwen 官方 config / checkpoint / 源码，未含任何 DeepSeek-V4 材料；所链前置页 `../dsa/` 描述的是 DeepSeek-V3.2 DSA 的按位置 top-k 选择，并不支持「按块压缩 / 数据相关权重加权求和」。｜引文依据：本页来源表「全部 config 数值｜官方 config.json」「数据流路径、各算子公式、缓存布局｜官方实现 transformers commit 36deb0b5」「张量形状与 dtype…｜真实 checkpoint 的 safetensors 文件头」「等价性、选择行为…｜本机实测」；`wiki/dsa/index.html`「lightning indexer 给历史位置打分，只对 top-k=2048 个位置做完整 MLA 注意力」。页面原文「这与 DeepSeek-V4 用数据相关权重加权求和的压缩不同，此处是无参数的固定池化」。｜修复要求：补 DeepSeek-V4 官方来源（报告章节或源码路径与行号）后再保留该对比，或删除该对比句，只保留「此处是无参数的固定池化」（§3 与「索引器」视图 pool 节点两处同改）。

- 重要｜表述｜§2 引言；§3「长上下文开销的来源」末段；§4 引言；§4.6 开头；§5 开头｜元话语、引导语与临场评价，违反规范「不写『本页将展示』『下面我们来看』一类的引导语，直接进入路径」与「不把调试过程写进正文」。｜引文依据：§5 开头「本页的数字分三类来源，各自的取得方式如下。」；§2 引言「下图是实际前向路径，也是本页的主体。九个标签页切换粒度：整体总览之外，还可下钻到…」；§4 引言「下面的「视觉编码器」与「多模态融合」两个视图给出完整路径。」；§4.6「这是容易被忽略的一点。」；§3 长上下文「需要注意 2048 这个预算是 config 的默认取值，块选择只在完整块数超过 512 时才实际生效…」。｜修复要求：逐句改为直接陈述——删除「也是本页的主体」与九个标签页的枚举（或并入图注说明视图用途）；§4 引言删「下面的…两个视图给出完整路径」；§4.6 删「这是容易被忽略的一点。」直接以「语言塔用三维 MRoPE…」开头；§3 长上下文删「需要注意」直接写「2048 是 config 默认的 indexer_budget…」；§5 删「本页的数字分三类来源，各自的取得方式如下。」直接进入「结构与维度：…」。

- 轻微｜技术来源｜§3「GDN」第 2 条；「GDN 层」视图 alpha 节点｜「衰减强度跨三个数量级」与实测区间不符：g ∈ [−63.34, −0.15]，63.34/0.15 ≈ 422，约 2.6 个数量级；α=exp(g) 则是 0.86 到 3×10⁻²⁸，跨度远超三个数量级，两种读法都对不上「三个数量级」。｜引文依据：页面原文「本机实测 g 落在 [-63.34, -0.15]，衰减强度跨三个数量级」；63.34 ÷ 0.15 = 422.3。｜修复要求：改为「跨两个数量级」，或直接给区间与比值（「g 从 −0.15 到 −63.34，跨约 400 倍」），两处同改。

- 轻微｜技术来源｜§3「N-gram Embedding」第 2 条；「N-gram 注入」视图 tbl 节点｜「128 个分片」与 §5／来源表的「131 个 safetensors 分片」用词相同、所指不同。checkpoint 中 N-gram 表是单个 embedding 的 128 个分片张量（`split_ngram_parts=128`，`ngram_embedding.shard_0..127`），与整仓 131 个 safetensors 文件不是同一层级。｜引文依据：权重索引含 `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight … shard_127.weight`（128 个，shard_0 形状 [2500012, 160]），全仓 `model-00001-of-00131.safetensors` 共 131 个；页面原文「真实权重 128 个分片的行数之和恰为 320,001,536」。｜修复要求：该处改为「128 个分片张量（split_ngram_parts=128，各 2,500,012 行）」，与 safetensors 分片区分，两处同改。

- 轻微｜格式一致性｜head 的 `description` 与 `dojo:summary`｜编号基准与正文不一致：head 写「只在第 2 层注入」，正文一律用 0 起编号（「N-gram 注入层（0 起）｜仅第 1 层」「对应 0 起的第 1 层」）。｜引文依据：`description`「51.2B 哈希 N-gram 查表只在第 2 层注入」；`dojo:summary`「51.2B 参数的哈希 N-gram 查表只在第 2 层注入」；§1 表「N-gram 注入层（0 起）｜仅第 1 层」。｜修复要求：head 两处改为「只在第 1 层（0 起）注入」或「只在第 2 层（1 起）注入」，与正文基准统一。

## 结论

- 统计：阻断 2 / 重要 2 / 轻微 3
- 处置：修复

其余已核对项（本轮通过，无需修改）：`layer_types` 被 `__post_init__` 改写为 `qwen_sparse_attention` 且注释一致；语法层面 `linear_attn` 恰 36 层、`self_attn`/`indexer` 恰 12 层（3,7,…,47）、`ple.*` 仅第 1 层、`mtp.*` 恰 31 个张量、视觉 blocks 0..26 恰 27 层；索引器 `mean(dim=1)` 池化 + RMSNorm + 块首 token RoPE、`relu(scores).sum(-1)/sqrt(index_head_dim)`、`block_topk=2048//4`、掩码张量宽 `token_budget+compress_ratio-1=2051`、残块无条件保留；`q_proj` 两倍宽 12288、`o_proj` 246×256=6144、`k_proj/v_proj` 512；`get_vision_position_ids` 的 llm_grid=(t, h/2, w/2) 与 `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`；`apply_interleaved_mrope` 的 slice(1,33,3)/slice(2,30,3)；N-gram 乘子公式与 3 个真实乘子、质数表与偏移前缀和、3 个 I64 缓冲共 35 个 int64；`hidden_states.repeat(1,1,4)`、`use_combine=False` 的 mixer 仅 3 个张量（少 block_inject_weight [4,10240]）；真实 `block_inject_weight` 非零（实测值 −18996…15881），与「四条流因 λ 不同而分化」一致。`.dojo/scripts/validate.py wiki/qwen3-8-flash-next-dataflow/index.html` 返回 success；页面引用的 16 个前置概念页均真实存在。