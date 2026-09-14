<!-- review-meta
round: 8
page: wiki/hy4-preview-dataflow/index.html
reviewed_content_sha256: 0985617a8e8b9eca
-->
# Hy4-Preview 前向数据流 审查记录（第 8 轮）

- 页面：/Users/wendadawen/code/dojo/wiki/hy4-preview-dataflow/index.html
- 页面 head 的 dojo:type = dataflow，适用规范 guides/model-dataflow.md（表述部分引用 guides/concept/check.md）
- 本轮完整阅读：1 关键规格、2 交互式数据流（含 noscript 八张全量兜底表与 JS 视图数据）、3 要点、4 实现对照、5 MTP 草稿层、6 核对方式、来源与范围说明；head 的 description / dojo:summary 一并核对。
- 未读取本页 research/ 下任何文件。

## 一、来源核对（逐条给出核对时看到的关键数值/片段）

### 官方 config（huggingface.co/tencent/Hy4-preview config.json，本轮 curl 原件核对）
- hidden_size 6144 / num_attention_heads 64 / intermediate_size 18432 / num_hidden_layers 78 / vocab_size 120832 / max_position_embeddings 1048576 全部与页面一致。
- q_lora_rank 2048、kv_lora_rank 512、qk_nope_head_dim 192、qk_rope_head_dim 64、v_head_dim 256、index_n_heads 32、index_head_dim 128、index_topk 2048、indexer 走 fp32 打分：与页面表 1 各行一致。
- n_routed_experts 256 / num_experts_per_tok 8 / n_shared_experts 1 / moe_intermediate_size 2048 / routed_scaling_factor 2.827 / norm_topk_prob true / n_group 1 / topk_group 1 / swiglu_limit 10.0 / hc_mult 4 / hc_magnitude 2.0 / hc_eps 1e-06 / learnable_sink true + init 0.0 / gated_mla true + gating_type elementwise / num_nextn_predict_layers 1 / enable_ihc true / enable_lm_head_fp32 true / rope_theta 1e7 / one nextn layer：全部一致。
- layer_types 78 项全为 deepseek_sparse_attention；mlp_layer_types 第 0 项 dense、其余 sparse；indexer_types 为 0、1 full，2–4 shared，5 full……77 full（共 21 full）：与「78 层全 DSA / 首层 dense / full=0,1,5,…,77」一致。

### 真实 checkpoint 张量头（HTTP Range 读全部 131 个分片的 JSON 头，本轮独立重取）
- 张量总数 2006、分片 131：一致。
- dtype 分布：BF16 1380 / F32 626，与页面一致。F32 明细 = hc_fn 156 + hc_scale 156 + hc_base 156 + 主干 sink 78 + 主干路由偏置 77 + hc_head 3 = 626，与「iHC 门控（78×2×3）、主干 sink（78）、主干路由偏置（77）、hc_head（3）」完全吻合。
- 形状实测：q_b_proj [16384,2048]、kv_b_proj [28672,512]、kv_a_proj_with_mqa [576,6144]、q_a_proj [2048,6144]、o_proj [6144,16384]、linear_gate [16384,6144]、indexer.wq_b [4096,2048]、indexer.wk [128,6144]、indexer.weights_proj [32,6144]、hc_pre.hc_fn F32 [8,24576]、mtp_layers.0.eh_proj [6144,12288]、embed_tokens/lm_head 均 BF16 [120832,6144]、experts.gate_up_proj [256,4096,6144]：与页面全部一致。
- 具名核对：indexer.* 恰好出现在 0,1,5,9,…,73,77 共 21 层；mlp.gate_proj 仅 layer 0；mlp.experts.* 覆盖 1–77；hc_attn_layer/hc_mlp_layer 覆盖 78 层；mtp_layers.0.* 恰 27 个且不含任何 hc 张量：与页面一致。

### checkpoint 小张量实值（按 data_offsets 精确下载，本轮独立解码）
- hc_head_base [-1.1441,-1.1512,-1.2124,-0.9716]；hc_head_scale 0.1053。
- layer0 hc_base 前 4 = [-0.8859,-0.8793,-0.7178,-0.9103]，后 4 = [-1.5183,-1.4180,-1.3979,-0.4856]；layer0 hc_scale [0.2503, 0.0058]。
- layer77 hc_base 后 4 = [0.0124,0.0030,0.0122,0.2079]；layer77 hc_scale [0.2577,0.1843]。
- 主干 sink：layer0 mean 0.3836（min -6.5407 / max 3.1750）、layer40 0.6656、layer77 -0.0222。
- MTP sink（BF16）mean 0.9501（min -2.3750 / max 2.9219）；layer1 路由偏置 min -0.0967 / max 0.0323 / mean 4.06e-08 / median 0.00127；layer40 median 0.0074；layer77 median 0.0064；MTP 路由偏置 median 0.0074；k_norm gamma mean 1.0614、beta mean -0.0023。
  以上与页面所载数值逐项吻合（含「均值 4×10^-8」这类细节）。

### 参数量（本轮按结构规则独立复算，与页面完全吻合）
- 单层注意力 265,685,568；full 索引器 9,371,904；单层 iHC 两模块 393,236；hc_head 98,309；单层 MoE 激活 341,311,744；首层 dense 339,738,624；78 层两组 layernorm 958,464。
- 主干合计复算得 769,907,408,797（769.91B），MTP 复算得 10,053,583,936（10.05B），两者相加 = 779,960,992,733，与页面总量逐位相等。
- 单 token 激活复算 47,571,797,917（47.57B）；再加 embedding 与 lm_head 各 742,391,808 得 49,056,581,533（49.06B），与官方 README「49B are activated per token」口径一致。
- 路由专家 744,103,084,032（744.10B），744.10/779.96 = 95.40%，与页面一致。

### 官方源码
- transformers（commit cbc1651a，页面所标快照，本轮按该 commit 取原文核）：`_keep_in_fp32_modules_strict` 在 HYV4ForCausalLM 中含 "lm_head"（基类清单含 e_score_correction_bias/fn/scale/base/hc_fn/hc_scale/hc_base/weights_proj/k_norm/sinks）；`hidden_states = self.norm(self.hc_head(hidden_states))`；`inputs_embeds.unsqueeze(2).expand(-1,-1,hc_mult,-1)`；`apply_rotary_pos_emb` 用 `rotate_half`（注释「Non-interleave RoPE」）；indexer 用 `q.float()/k.float()` + `F.relu`、`weights_proj(...).float()*(n_heads**-0.5)*self.softmax_scale` 且 `softmax_scale = head_dim**-0.5`（=1/√128）；k_norm 为 `nn.LayerNorm`；`mixes = F.linear(flat, fn.float()) * input_norm(flat)`，`pre = sigmoid(pre_logits*pre_scale+pre_b)+hc_eps`、`post = 2.0*sigmoid(...)+hc_eps`；`_init_weights` 中 `scale→0.01`、`base[:hc_mult] = -log(hc_mult-1)`、`base[hc_mult:] = 0`；router 用 sigmoid+偏置选 top-8、权重从原始分数 gather、归一化后 ×routed_scaling_factor；`_apply_gate` 为 `gate.clamp(max=limit)` 与 `up.clamp(±limit)`，共享专家为不带截断的 HYV4MLP。—— 页面第 3/4/5 章与各图注的对应表述逐条成立。
- transformers deepseek_v2：`apply_rotary_emb` 用 `view_as_complex(x.reshape(...,-1,2))`（相邻配对）——支持页面「deepseek_v2 用交错」一句。
- vLLM（commit 385dce3，页面所标 commit，按该 commit 取原文核）：attention.py 第 420 行与第 434 行两处 `is_neox_style=False`；第 222 行注释「Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims.」；第 426–427 行注释「interleaved (Megatron/PTM) layout … (is_neox_style=False)」；indexer k_cache 为 `head_dim + head_dim//128*4`（132 B/token/层，×21 层 = 2772 B/token）——与页面 RoPE 表、索引器缓存 2772 B 一致。
- SGLang（commit 55bf338，页面所标 commit）：configs/hy_v4.py 第 91–92 行 `rope_interleave = True`、`indexer_rope_interleave = True`；models/hunyuan_v4.py 第 46 行 `permute_hyv4_indexer_weight`，实现为 `torch.cat((w[:, -rope_dim:], w[:, :-rope_dim]), dim=1)`（把每头尾部旋转块挪到头部，属块内顺序置换）——与页面表述一致。
- vLLM mtp.py：第 351 行 `mtp_config.enable_ihc = False`；第 500 行 `torch.where((positions == 0)..., 0, inputs_embeds)`；第 739 行 `"lm_head.weight": f"model.layers.{mtp_start}.shared_head.head.weight"`——与页面 MTP 三处表述一致。
- vLLM hc.py：`HYV4HCPreLayer(prefix=f"{prefix}.hc_pre")` 且 `hc_post` 为无参模块——支持「checkpoint 里只有 hc_pre」的结构性解释。
- hub 仓库：finetune/llama_factory_support/hy_v4_patches.py 只做键名映射（`("mlp.router.gate.", "mlp.gate.")` 等）与专家权重融合，全仓库（含 ms_swift/deepspeed 支持）无任何 RoPE 维度置换；仓库内不存在 RoPE 转换脚本——与页面「hub 没有提供该转换脚本」一致。

### 页面内部一致性
- 同一数字在 head/正文/noscript 表/JS 视图数据/图注之间逐项抽查（10.05B、779.96B、769.91B、341,311,744、744.10B/95.40%、56.89×、87.8 KiB/GiB、4.88 MiB/TiB、2772/5376 B、0.2500、0.9999、0.389/0.615/0.500、47.8%、23.4%、0.0235、-69.9968、7.3e-10、sink 0.121/0.502、门控实值、bias 实值）无一处互相矛盾。
- noscript 八张表与 JS 八个视图的节点/边/说明集合一致；KV cache 换算（576×2B×78 = 87.75 KiB→87.75 GiB；32768×2B×78 = 4.875 MiB→4.875 TiB；32768/576 = 56.89；132×21 = 2772；256×21 = 5376）全部复算通过。
- 前 4 行 pre / 后 4 行 post 的切分与源码 `base.split(hc_mult)`、`mixes.split(hc_mult)` 一致。

### 机械项
- `.dojo/scripts/validate.py wiki/hy4-preview-dataflow/index.html` → validation ok。
- 站内链接 ../deepseek-moe/、../dsa/、../mla/、../ihc/、../eagle-speculative/、../rope/、../aux-loss-free-routing/ 均存在；libs 下 KaTeX/cytoscape/dagre/prism/dojo-dataflow.css 全部存在。
- 三段内联 JS `node --check` 通过；alt 属性无 `$...$`；<noscript> 兜底表存在（禁用脚本可读）。

## 二、问题

- [轻微·技术] §4「checkpoint 是 vLLM 命名的 inner 格式」首条：末句「vLLM 的 load_weights 直接按此命名加载」与所引源码不符。｜引文依据：vLLM commit 385dce3 vllm/models/hy_v4/nvidia/model.py 中 load_weights 明确做了 `.indexer.wk.` → `.indexer.wk_weights_proj.`（第 491 行）、`gate.e_score_correction_bias` → `expert_bias`（第 584–586 行）、去掉 `router.`（第 588–590 行）、`hc_fn` → `hc_fn.weight`（第 590–591 行）、专家权重映射到 `.experts.routed_experts.w13_weight/w2_weight`（第 447–449 行）；即 checkpoint 张量名与 vLLM 参数名并非逐名直载，该句把「模块结构同源」写成了「无需映射直接加载」。｜修复要求：把该分句改为不产生「逐名直载」印象的表述，例如「命名与 vLLM 的模块结构同源，vLLM 的 load_weights 只需少量键名改写（wk 与 weights_proj 融合、路由偏置改名、专家权重映射）即可加载」。｜修复：｜复验：
- [轻微·可读性/表述] §3「MLA 注意力」第 4 条：「即 sink 为零时仍平均吃掉 12% 的注意力质量」——「吃掉」为口语化措辞。｜引文依据：不适用。｜修复要求：改为书面表述，如「即 sink 初值为零时仍平均吸收 12% 的注意力质量」。｜修复：｜复验：
- [轻微·可读性] 等比缩小模型的构造条件在页面内两处不一致：§6 写作「（10 层、隐藏维 512、8 头、16 专家 top-8、4 条流、24 token）」，§3「prefill 与 decode 的等价性」写作「等比缩小模型（47 token）」，而该节表格的 1128 个条目 = 47×48/2、「47 列 vs 48 列」「47 个查询」均指向 47 token。｜引文依据：§6 原文「…4 条流、24 token）」；§3 原文「等比缩小模型（47 token）上的五步证据链」。｜修复要求：在 §6 的构造说明中写清序列长度的取值（如「序列长度 24；等价性验证另跑 47 token」），避免读者把同一模型的两个长度当成同一条件。｜修复：｜复验：

（本轮未发现阻断级或重要级问题：所有事实性论断、公式、数字均能回源到官方 config、官方源码、真实 checkpoint 张量头/小张量实值或官方 README；未发现同页两处互相矛盾、算式与结论不符、引文编号与文献表不符、图注读数与刻度不符；无元话语、无「本页」自我指代、无会话指代、无调试叙事、无临场评价。）

## 三、结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（三条轻微问题建议随下轮一并处理，均不影响核心结论与主线理解）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
