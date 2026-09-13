<!-- review-meta
round: 4
page: wiki/qwen3-8-flash-next-dataflow/index.html
reviewed_content_sha256: 65474b6a2b7b8bfa
-->
# Qwen3.8-Flash-Next 前向数据流审查记录（第 4 轮）

- 页面版本：`3a81764ef39eab2a0e42a6011ba497ed67d04c56`
- 审查时间：2026-09-13 20:24
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何文件）
- 适用规范：`guides/model-dataflow.md`（`dojo:type=dataflow`），表述部分与 `guides/concept/check.md` 同一套
- 已完整阅读章节：1. 关键规格 / 2. 交互式数据流（含无脚本静态 SVG、图例、图注）/ 3. 要点（整体结构、GDN、QSA、Gated Residual、N-gram Embedding、MoE 路由、长上下文开销的来源）/ 4. 视觉编码器与多模态融合（4.1–4.8）/ 5. 核对方式 / 来源与范围说明；并逐条通读 9 个交互视图（overview、gdn、qsa、indexer、hc、ple、moe、vision、fusion）的全部节点 io/公式/说明与边

## 核对情况

- 官方 config.json（huggingface.co/Qwen/Qwen3.8-Flash-Next，config.json）逐键核对通过：hidden_size=2560、num_hidden_layers=48、full_attention_interval=4、num_attention_heads=24、num_key_value_heads=2、head_dim=256、num_experts=512、num_experts_per_tok=10、moe_intermediate_size=640、linear_num_key_heads=16/linear_num_value_heads=48/linear_key_head_dim=128、indexer_n_heads=4/indexer_kv_heads=1/indexer_head_dim=128/indexer_budget=2048/indexer_compress_ratio=4、ple_layer_ids=[2]、ple_embed_dim=2560、ple_conv_kernel_size=4、hc_count=4、hc_lowrank=320、ngram_size=3、ngram_vocab_size_base=20000000、split_ngram_parts=128、make_ngram_vocab_size_divisible_by=128、mamba_ssm_dtype=float32、output_gate_type=sigmoid、partial_rotary_factor=0.25、mrope_section=[11,11,10]、mrope_interleaved=true、rope_theta=1e7、max_position_embeddings=262144、vocab_size=248320、tie_word_embeddings=false、attention_bias=false、mtp_num_hidden_layers=1、image_token_id=248056、vision_start_token_id=248053、vision_end_token_id=248054；vision_config: depth=27/hidden_size=1152/intermediate_size=4304/num_heads=16/num_position_embeddings=2304/out_hidden_size=2560/patch_size=16/temporal_patch_size=2/spatial_merge_size=2/hidden_act=gelu_pytorch_tanh。上述全部与页面一致。
- 官方模型卡（README）核对：Total Parameters 125B / Activated 6B / N-gram Embedding 51B / MTP 4B；48 层、512 专家（10 routed + 1 shared）、262144 原生上下文、QSA、Gated Residual、N-gram Embedding 均与页面一致。
- 逐项复算通过（与页面数字吻合）：GDN conv_dim 2·16·128+48·128=10240；QSA q_proj 24·256·2=12288、k/v 2·256=512；MoE 单专家 1280·2560+2560·640=4915200，512 个 2516582400，48 层 120795955200（=120.80B，占 179999981424 的 67.11%）；MoE 单层全量 2522810880、激活 55380480（2.195%）；超连接每层两组 13209600、48 层加 mixer 640624640（0.36%）；视觉塔分组累加 267890544+143451648+33039616+2654208+1770624+124416=448931056，且 411466608 与 33039616 各自精确对上；27 层 LayerNorm 124416、patch_embed 带 bias 1770624；N-gram 16 质数（20000003…20000171）之和 320001446，按 128 对齐 320001536（补 90 行），×160=51200245760；乘子公式 c_j=2(splitmix64(seed+10007ℓ+γ(j+1)) mod B)+1、B=18571544855136 与 checkpoint 三个真实乘子逐元素复算一致（seed=1234 成立）；
- 视觉 token/位置复算通过：448×448→784 patch→196 token、896→784、1344→1764、1920×1080→8160→2040、16 帧 448×448→6272→1568；位置推进 max(h,w)/2 分别为 14/28/42/60；`(T,H,W)` 在 [8,21] 遍历、后续文本从 22 开始；MRoPE 交错槽位 T=11/H=11/W=10 与 slice(1,33,3)/slice(2,30,3) 逐槽核对一致。
- 长上下文表真值复算：QSA KV=12×2048 B/token、索引器=12×256 B/token、GDN=36×(3 MiB+60 KiB)=0.10754 GiB；1M 时 24 GiB 与「48 层全 QSA 则 96 GiB、降为 1/4」一致。
- 机械项：`.dojo/scripts/validate.py` 返回 `validation ok`；`check_inline_js.py` 返回 ok；`dojo:type=dataflow`、`dojo:topics="模型结构,多模态"`（均在词表内）、`dojo:tag="数据流"` 合法；正文仅引用 `research/measured.md`（该文件存在于 `wiki/qwen3-8-flash-next-dataflow/research/measureed.md` 同目录，且已按规范写明「实测得到」而非指向脚本文件）；无 Unicode 数学字符越界（`×`、`→` 按 validate.py 注释属「中文技术散文普通排版字符」，不计入；希腊字母/下标仅出现在交互视图的 JS 标签中，属界面元素）。

## 问题

- [轻微·可读性] 第 3 节「整体结构 · 沿深度」：把「读出」归给 attn_hyper_connection、「写回」归给 mlp_hyper_connection，与两个下钻视图不符——每个超连接都各自包裹一个子层并各做一次读出与写回。｜引文依据：页面原文「每层内部 attn_hyper_connection 与 mlp_hyper_connection 各调用一次：前者把 4 条流归约成 2560 宽送进子层，后者把子层输出按每条流一个标量写回。」；对照 `VIEWS.gdn.nodes.hc1/hc2` 标签「attn_hyper_connection 读出」「mlp_hyper_connection 读出」与节点 `wb1`/`wb2`（写回四流）、边 `proj→wb1`、`wb1→hc2`。｜修复要求：改为「两者各包裹一个子层：attn_hyper_connection 在注意力子层前后读出与写回，mlp_hyper_connection 在 MoE 子层前后读出与写回」。｜修复：｜复验：
- [轻微·技术] 第 3 节「长上下文开销的来源」表：4K 与 32K 两行显示值分项之和≠合计。｜引文依据：4K 行「0.094 / 0.012 / 0.108 / 0.213」，0.094+0.012+0.108=0.214≠0.213；32K 行「0.750 / 0.094 / 0.108 / 0.951」，0.750+0.094+0.108=0.952≠0.951。真值 0.09375+0.01171875+0.10754395=0.2130、0.75+0.09375+0.10754395=0.9513 与合计一致，差异全部来自三列各自四舍五入（256K、1M 两行恰好看不出一致）。｜修复要求：三列与合计统一舍入口径，或在表下加注「分项四舍五入，合计按未舍入值」。｜修复：｜复验：
- [轻微·技术] 第 1 节「关键规格」表「语言主干（除 N-gram 表与 MTP）125.74B」行：排除项漏列视觉编码器，与表内总参数量口径不一致。｜引文依据：该 125.74B 实际也不含视觉塔——125,743,653,760+51,200,245,760+448,931,056+2,607,150,848=179,999,981,424（恰为表内「180.0B」）；若按标签字面（仅除 N-gram 表与 MTP）相加则为 125.74+51.20+2.61=179.55B，与 180.0B 差 0.449B（即视觉塔，被重复计）。｜修复要求：排除项补为「除 N-gram 表、视觉编码器与 MTP」。｜修复：｜复验：
- [轻微·来源] 第 1 节表与「来源与范围说明」：「官方称」四项的来源表述与官方模型卡不一致、且未列出模型卡出处。｜引文依据：官方模型卡（huggingface.co/Qwen/Qwen3.8-Flash-Next，README）表内标签为「Total Parameters | 125B」「Activated Parameters | 6B」「N-gram Embedding Parameters | 51B」「MTP Parameters | 4B」，页面转述为「官方称「Transformer 参数 125B」」；页面另称「三项均已独立复现（分别算得 125.74B、51.20B、6.04B）」，但 125.74B 与官方 125B 差 0.74B（0.6%）未说明；「来源与范围说明」表只列 config.json / transformers 源码 / checkpoint 文件头 / I64 缓冲 / 本机实测，125B、6B、51B、4B 四项的依据（模型卡）未成行列出。｜修复要求：125B 一行转述与模型卡原文对齐（注明官方标签为 Total Parameters），来源表补「模型卡」一行，并对 125.74B 与 125B 的 0.74B 差写明来源（官方为约数）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布。核心结论（48 层 3:1 层型、4 条残差流、N-gram 第 1 层注入、512 专家 top-10+共享、QSA 块选择、27 层视觉塔与融合、长上下文缓存构成）均与官方 config.json、模型卡及可复算的算式一致，未发现来源不支持、把推断写成来源结论、同页两处矛盾或分项与合计的算术错误；4 项轻微问题不影响正确性与主线理解，建议修后发布。