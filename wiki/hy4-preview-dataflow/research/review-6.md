<!-- review-meta
round: 6
page: wiki/hy4-preview-dataflow/index.html
reviewed_content_sha256: 00d3ac3211bfb5d8
-->
# Hy4-Preview 前向数据流审查记录（第 6 轮）

- 页面版本：b85ce6122cea0b83e6314d9e15c6fb1e7574948c
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序轮次）
- 页面类型：dataflow（按 guides/model-dataflow.md 审查）
- 已完整阅读章节：1. 关键规格（表 + 精度 note）、2. 交互式数据流（含八视图 noscript 表与全部节点 tooltip 文案）、3. 要点（整体结构 / iHC / DSA 索引器 / MLA / MoE / prefill-decode 等价性 / KV cache 两种口径）、4. 实现对照（RoPE 布局表 + 命名归属）、5. MTP 草稿层、6. 核对方式、来源与范围说明
- 机械项：`.dojo/scripts/validate.py` 通过；`check_inline_js.py` 通过（3 blocks）；八视图边与 drill 目标全部可解析；七个站内概念链接（deepseek-moe/dsa/mla/ihc/eagle-speculative/aux-loss-free-routing/rope）均存在；`alt` 仅 lightbox 空串，无 `$...$`。

## 核对依据（外部来源逐条回源）

- config.json（huggingface.co/tencent/Hy4-preview/raw/main/config.json）：层 78、hidden 6144、头 64、vocab 120832、max_position 1048576、num_nextn_predict_layers 1、index_n_heads 32 / index_head_dim 128 / index_topk 2048、layer_types 全 `deepseek_sparse_attention`、mlp_layer_types 首层 `dense`、n_routed_experts 256 / top-8 / shared 1、routed_scaling_factor 2.827、n_group=topk_group=1、swiglu_limit 10.0、q_lora 2048 / kv_lora 512 / qk_nope 192 / qk_rope 64 / v_head 256、hc_mult 4 / hc_magnitude 2.0 / hc_eps 1e-6、rope_theta 1e7、learnable_sink_init 0.0、gating_type elementwise、enable_lm_head_fp32 true、tie_word_embeddings false —— 与页面逐项一致。
- indexer_types 实测 `['full','full','shared'×3,'full',...]`，full 下标 = [0,1,5,9,13,...,73,77] 共 21 —— 与页面「0, 1, 5, 9, …, 77 共 21 个、其余 57 层 shared」完全一致。
- model.safetensors.index.json：2006 张量 / 131 分片（页面对）、mtp 张量 27 个且无任何 hc 参数（对）、主干 sink 78、主干 e_score_correction_bias 77 + MTP 1、hc_pre.{hc_fn,hc_base,hc_scale} 各 156、indexer.* 110（=22×5，主干 21 + MTP 1）。
- 分片头 dtype 抽样：主干 sink F32[64]、hc_fn F32[8,24576]、hc_base F32[8]、hc_scale F32[2]、hc_head_fn F32[4,24576]、主干路由偏置 F32[256]、lm_head BF16[120832,6144]、embed_tokens BF16[120832,6144]、k_norm BF16[128]、weights_proj BF16[32,6144]、indexer.wk BF16[128,6144]、MTP sink BF16[64]、MTP 路由偏置 BF16[256] —— 印证页面「1380 BF16 + 626 F32」及 F32 清单（78×2×3 + 78 + 77 + 3 = 626）。
- 真实小张量实值（按 data_offsets 精确取字节解码）：第 0 层 attn hc_base = [-0.8859,-0.8793,-0.7178,-0.9103,-1.5183,-1.4180,-1.3979,-0.4856]（页面 [-0.886,-0.879,-0.718,-0.910]/[-1.518,-1.418,-1.398,-0.486]，对）；hc_scale 第 0 层 = [0.2503,0.0058]（页面 [0.250,0.0058]，对）；第 77 层 hc_base 后 4 = [0.0124,0.0030,0.0122,0.2079]（页面 [0.012,0.003,0.012,0.208]，对）；hc_head_base = [-1.1441,-1.1512,-1.2124,-0.9716]（对）、hc_head_scale = 0.1053（对）；第 0 层 sink mean 0.383616 / min -6.5407 / max 3.1750；第 40 层 sink 0.6656、第 77 层 -0.0222；MTP sink mean 0.9501 / [-2.3750,2.9219]；主干第 1 层路由偏置 [-0.0967,0.0323]、median 0.0013、mean≈0；第 40 层 median 0.0074、第 77 层 median 0.0064；MTP 路由偏置 median 0.0074；k_norm gamma mean 1.0614 / beta mean -0.0023 —— 与页面全部一致（唯一例外见轻微 3）。
- 源码：transformers `modeling_hy_v4.py`（main）—— `HYV4Indexer.forward` 收 `q_resid = q_a_layernorm(q_a_proj(x))`（印证索引器复用主注意力 q_a 输出）；`torch.split(q,[head_dim-qk_rope_head_dim, qk_rope_head_dim])` 得 [64|64] 且 rope 在尾；`weights = weights_proj(x)*(n_heads**-0.5)*softmax_scale`（softmax_scale=head_dim^-0.5）—— 印证打分公式；`apply_rotary_pos_emb` 用 `rotate_half`（半分式），主注意力处注释 `# Non-interleave RoPE`；`_init_weights` 中 `base_value=-log(hc_mult-1)`、`base[:hc_mult]=base_value`、`hc_scale=0.01`、`sinks=learnable_sink_init` —— 印证「前半 pre=-log3、后半 post=0」与各初始值；`HYV4ForCausalLM._keep_in_fp32_modules_strict` 含 `lm_head`。vLLM `vllm/models/hy_v4/nvidia/attention.py`：主 MLA 与索引器两处 `get_rope(..., is_neox_style=False)`，注释「Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims」「interleaved (Megatron/PTM) layout」。SGLang `python/sglang/srt/configs/hy_v4.py`：`rope_interleave=True`、`indexer_rope_interleave=True`；`python/sglang/srt/models/hunyuan_v4.py`：`permute_hyv4_indexer_weight` 把每头末尾 rope_dim 块移到组内前部。transformers `deepseek_v2/modeling_deepseek_v2.py`：`apply_rotary_emb` 用 `torch.view_as_complex`（交错）。HF 仓库 `finetune/llama_factory_support/hy_v4_patches.py` 存在，只做键名映射（`mlp.router.gate.`→`mlp.gate.` 等）与专家 3D 融合，无任何 RoPE 置换。README.md：「comprises 770B total parameters, of which 49B are activated per token」「1 native MTP layer (10B total parameters, 0.7B activated)」，部署指引只给 vLLM 与 SGLang。
- 参数量自算复现：78×265,685,568 + 21×9,371,904 + 78×393,236 + 98,309 + 77×341,311,744 + 339,738,624 = 47,571,697,917（47.57B）；再加 embed/lm_head 各 742,391,808 得 49,056,481,533（49.06B）；主干全量（含 78 层两组 norm 958,464 与路由偏置）769,907,408,797（769.91B）；+ MTP 10,053,583,936 = 779,960,992,733（779.96B）—— 与页面数字逐位吻合。KV cache：32768×2×78 = 4.8765 MiB/token、576×2×78 = 87.75 KiB/token、32768/576 = 56.89、21×132 = 2772 B、21×128×2 = 5376 B、1M 下 87.8+2.7 ≈ 90.5 GiB —— 与页面一致。

## 问题

- [重要·技术] 第 1 节「关键规格」DSA 索引器行（"…top-2048，打分全程 fp32"）与第 3 节「DSA 索引器」要点（"打分公式（全程 fp32，含 ReLU）"）：把「索引器打分全程 fp32」写成模型的无条件性质，但它只在 transformers 参考实现下成立；官方部署路径 vLLM 把索引器 q 量化为 FP8、k 以 FP8 写入缓存，官方血统的参考 Indexer 本身也不是 fp32 打分。同页第 2 节索引器视图又写「vLLM 用 FP8 存储约 2772 B/token」，两处口径相互张力。｜引文依据：transformers `modeling_hy_v4.py` HYV4Indexer.forward docstring——"This is the bf16 equivalent of the reference Indexer which uses `rotate_activation` (Hadamard transform) and `fp8_index` (FP8 quantized scoring kernel). Since the Hadamard transform is orthogonal … and FP8 quantization is a precision optimization, we skip both and compute scores directly in bf16/fp32."；vLLM `vllm/models/hy_v4/nvidia/attention.py`——"Only q is quantized here; k quantization is fused with cache insertion."、`per_token_group_quant_fp8(..., quant_block_size=128, scale_fmt="ue8m0")`、"FP8 naive cache: values in fp8 plus one fp32 scale per quant_block_size elements"。｜修复要求：把该论断限定到实现（例如「transformers 参考实现按 fp32 打分」），或在第 4 节实现对照里补一条索引器打分口径差异（参考 Indexer / vLLM：Hadamard + FP8 量化打分），使规格表、要点与第 2 节 FP8 缓存口径不再互相矛盾。｜修复：｜复验：
- [轻微·技术] 第 4 节 RoPE 布局正文：句子以"三方"作数量词，括注却只举出两个实现。｜引文依据：原文"三方一致的部署实现（vLLM、SGLang）都按交错消费这份 checkpoint"；README「For production serving, we recommend using vLLM or SGLang.」（只列两者）。｜修复要求：改为"两方/两个部署实现"，或补出被计为第三方的实例。｜修复：｜复验：
- [轻微·技术] 第 2 节 MLA 视图 sink 说明、第 3 节「MLA 注意力」要点、noscript MLA 表三处：第 0 层 sink 取值范围上限写作 3.18。｜引文依据：checkpoint `model.layers.0.self_attn.learnable_sink_param`（model-00089-of-00131.safetensors，F32[64]）解出 min = -6.5407、mean = 0.383616、max = 3.174999952316284，上限应为 3.17（或写 3.175）；均值 0.384 与下限 -6.54 与实值一致。｜修复要求：三处上限统一改为 3.17（或 3.175）。｜修复：｜复验：
- [轻微·表述] 第 2 节 MLA 视图 kv 节点标签「kv_a_LN → kv_b_proj」：用 LN 指代该归一化，与同格公式 $\mathrm{RMSNorm}$ 命名不一致，页面别处又用 LayerNorm 专指真正带 beta 的 `indexer.k_norm`，易混；另第 4 节 note 以「本节」为主语自我指代。｜引文依据：checkpoint 仅存在 `model.layers.*.self_attn.kv_a_layernorm.weight`（无 bias，即 RMSNorm）；页面同格公式为 $W_{kvb}\,\mathrm{RMSNorm}(c_{lat})$；note 原文"本节的证据性质是源码与张量结构级的交叉核对"。｜修复要求：标签改为与公式一致的写法（如 `kv_a_RMSNorm`）；note 改为直接陈述（如"这些结论的证据性质是源码与张量结构级的交叉核对"）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（重要 1 条需在本轮关闭；三条轻微建议一并处理）
