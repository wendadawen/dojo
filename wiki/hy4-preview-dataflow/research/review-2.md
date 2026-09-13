<!-- review-meta
round: 2
page: wiki/hy4-preview-dataflow/index.html
reviewed_content_sha256: 2d30b46b5da1295f
-->
# Hy4-Preview 前向数据流审查记录（第 2 轮）

- 页面版本：c291a37f27c48f57aef9e9865617892fdbf6491e（审查期间页面被改动过一次：19:14 版本 7622009d…，本轮以 19:21 之后的 c291a37f… 为准）
- 审查时间：2026-09-13 19:24
- 审查者：独立子代理（未参与写作，未参与第 1 轮审查；未读取本页 research/ 下任何规划、修复或前序审查记录）
- 已完整阅读章节：1. 关键规格 / 2. 交互式数据流（含 8 个视图的全部节点 label、io、f、d 与边标签）/ 3. 要点（整体结构、iHC、DSA 索引器、MLA 注意力、MoE、prefill 与 decode 的等价性、KV cache 的两种口径）/ 4. 实现对照（RoPE 布局、inner 格式）/ 5. MTP 草稿层 / 6. 核对方式 / 来源与范围说明
- 回源材料（本轮实际抓取核对）：huggingface.co/tencent/Hy4-preview 的 config.json、model.safetensors.index.json 与 131 个分片的 safetensors 文件头（HTTP Range）、assets/README.md、finetune/llama_factory_support/hy_v4_patches.py；transformers src/transformers/models/hy_v4/modeling_hy_v4.py、.../deepseek_v2/modeling_deepseek_v2.py；vLLM vllm/models/hy_v4/nvidia/{attention,model,mtp}.py；SGLang python/sglang/srt/configs/hy_v4.py、.../models/hunyuan_v4.py
- 机械项：`.dojo/scripts/validate.py wiki/hy4-preview-dataflow/index.html` → validation ok；页面无 `<pre>` 代码块，第 3 项「可运行代码」不适用；站内链接 ../deepseek-moe/、../dsa/、../mla/、../ihc/、../eagle-speculative/、../aux-loss-free-routing/、../rope/ 全部存在，无「（待生成）」占位

## 已回源通过的关键数字（供后续轮次复用，不计为问题）

- config.json 逐项吻合：model_type=hy_v4、78 层、hidden 6144、64 头、q_lora_rank 2048、kv_lora_rank 512、qk_nope_head_dim 192、qk_rope_head_dim 64、v_head_dim 256、index_n_heads 32、index_head_dim 128、index_topk 2048、n_routed_experts 256、num_experts_per_tok 8、n_shared_experts 1、moe_intermediate_size 2048、intermediate_size 18432、routed_scaling_factor 2.827、norm_topk_prob true、n_group 1、topk_group 1、swiglu_limit 10.0、hc_mult 4、hc_magnitude 2.0、hc_eps 1e-6、vocab 120832、rope_theta 1e7、max_position 1048576、num_nextn_predict_layers 1、enable_lm_head_fp32 true、tie_word_embeddings false、indexer_types 21 full / 57 shared（full 在 0,1,5,…,77）、mlp_layer_types 首层 dense
- 参数量：把 131 个分片文件头的 shape 全部相乘求和得 779,960,992,733，与页面逐位一致；MTP（model.mtp_layers.0.*，27 个张量）合计 10,053,583,936＝10.05B，backbone 769,907,408,797＝769.91B；dtype 张量计数 BF16 1380 + F32 626＝2006，且 626 个 F32 恰为 iHC 门控 468（78×2×3）+ 主干 sink 78 + 主干路由偏置 77 + hc_head 3
- 张量头：q_b_proj [16384,2048]、kv_b_proj [28672,512]、kv_a_proj_with_mqa [576,6144]、indexer.wq_b [4096,2048]、indexer.wk [128,6144]、indexer.weights_proj [32,6144]、linear_gate [16384,6144]、experts.gate_up_proj [256,4096,6144]（3D 融合、无 .weight 后缀）、hc_attn_layer.hc_pre.{hc_fn[8,24576],hc_base[8],hc_scale[2]}、hc_head.{hc_head_fn[4,24576],hc_head_base[4],hc_head_scale[1]}、首层 mlp.gate_proj [18432,6144]、lm_head 与 embed_tokens 均 BF16 [120832,6144]、MTP 的 learnable_sink_param 与 e_score_correction_bias 为 BF16
- 真实权重实值（按 data_offsets 下载解码）：第 0 层 attn hc_base 前 4=[-0.8859,-0.8793,-0.7178,-0.9103]、后 4=[-1.5183,-1.4180,-1.3979,-0.4856]；第 77 层 post 半区=[0.0124,0.0030,0.0122,0.2079]；hc_scale 第 0 层={0.2503,0.0058}、第 77 层={0.2577,0.1843}；hc_head_base 值集含 -1.2124/-0.9716、hc_head_scale 0.10535；sink 第 0/40/77 层均值 0.3836/0.6656/-0.0222（第 0 层范围 [-6.5407,3.1750]）；路由偏置第 1 层范围 [-0.0967,0.0323]、均值 4.06e-8、中位数 0.00127，第 40 层中位数 0.00740，第 77 层 0.00640；MTP sink 均值 0.9501、范围 [-2.375,2.922]、路由偏置中位数 0.00737；k_norm gamma 均值 1.0614、beta 均值 -0.00226
- 公式与源码：indexer 打分 `weights·n_heads**-0.5·head_dim**-0.5` 与 `F.relu(scores)`、`topk=min(index_topk, shape[-1])`；sink `cat([attn_weights, sinks])`→减 max→softmax→去尾列；gate 用 `sigmoid(gate_states)` 逐元素；iHC `pre=sigmoid(pre_logits*pre_scale+pre_b)+eps`、`post=magnitude*sigmoid(...)+eps`、`base[:hc_mult]=-log(3)`（即 base 前 4=-log3、后 4=0）；router 为 sigmoid→+e_score_correction_bias 选 top-8→由原始 sigmoid 分数 gather→norm_topk_prob 归一化→×routed_scaling_factor；`silu(clamp(gate,max=10))*clamp(up,-10,10)`；`self.norm(self.hc_head(hidden_states))`；`enable_lm_head_fp32`+`_keep_in_fp32_modules_strict` 含 lm_head。vLLM `is_neox_style=False` 两处（含 “Checkpoint (PTM) layout” 注释）、`enable_ihc=False`、`torch.where(positions==0, 0, inputs_embeds)`、indexer 缓存 head_dim+head_dim//128*4=132B；SGLang `rope_interleave=True`/`indexer_rope_interleave=True`、`permute_hyv4_indexer_weight` 把末 64 维挪到头部；deepseek_v2 用 `view_as_complex` 相邻配对；hy_v4 patches 只做键名映射与专家融合。以上与页面表述一一吻合。

## 问题

- [阻断·技术] 第 2 节「整体总览」视图 grp 节点：节点标签与说明称 Layers 2..76 含 18 个 full 层，同一节点的公式却写成 (full + 3 shared) × 19，即 19 个 full 层，同一节点内两处互相矛盾｜引文依据：该节点 label='Layers 2..76\n18 full + 57 shared'、f='(\mathrm{full} + 3\,\mathrm{shared}) \times 19'、d='本区间（2..76）内含 5…73 的 18 个 full 层与全部 57 个 shared 层'；官方 config indexer_types 中 full 位于 0,1,5,9,…,77，落在此区间的只有 5,9,…,73 共 18 个，(full+3shared)×19 覆盖的是 1..76｜修复要求：把 f 改为与本区间一致的算式（如 $(3\,\mathrm{shared} + \mathrm{full})\times 18 + 3\,\mathrm{shared}$），或把节点改为覆盖 1..76、同时把 label 改成 19 full + 57 shared｜修复：｜复验：
- [阻断·技术] 第 3 节「DSA 索引器」第 4 条与第 2 节索引器视图 topk 节点：把 top-k 的候选宽度写成「可见长度」并断言「top-k 不会选出未来 token」，与源码不符；同一节点的 label 与 f 也互相不一致｜引文依据：modeling_hy_v4.py 的 `topk = min(self.index_topk, index_scores.shape[-1])`、`return index_scores.topk(topk, dim=-1).indices`——候选宽度取键序列长度 T（不是可见长度），且在因果掩码置 −inf 之后 topk 仍固定返回 min(2048,T) 个索引，当某个查询的可见数小于该值时，分数为 −inf 的未来位置同样会被返回（因果性由注意力掩码保证，不是由 top-k 保证）；同节点 label='top-2048\nmin(2048, 可见长)' 与 f='\mathrm{topk} = \min(2048,\ T)' 不一致｜修复要求：把正文与 label 统一改为 min(2048, T)，并把「因此 top-k 不会选出未来 token」改为「未来位置即使被 top-k 选中，也在因果掩码下不可见」｜修复：｜复验：
- [轻微·技术] 第 3 节 iHC 与 iHC 视图：归一化项与门控项共用符号 $\epsilon$，但两者取值不同（前者 1e-5、后者 1e-6），公式无法按页面标注复算｜引文依据：源码 `HYV4UnweightedRMSNorm(eps=config.rms_norm_eps)`（rms_norm_eps=1e-5）与 `self.hc_eps`（hc_eps=1e-6）是两个不同常量，而页面 line 150/577 的 rsqrt(…+ε) 与 line 152/154/579/582 的门控 ε 同写 ε，第 114 行又把 iHC 的 ε 标为 $10^{-6}$｜修复要求：两处 ε 改用不同符号，或在公式处分别注明取值｜修复：｜复验：
- [轻微·技术] $W_g$ 一名两用：第 3 节 MLA 与 MLA 视图用它表示注意力输出门 linear_gate，MoE 视图「共享专家」节点又用它表示 shared_experts.gate_proj｜引文依据：`linear_gate.weight` [16384, 6144] 与 `mlp.shared_experts.gate_proj.weight` [2048, 6144] 是两个不同模块｜修复要求：共享专家处改用 $W_g^{sh}$（或 $W_{sh}$）区分｜修复：｜复验：
- [轻微·技术] cache 视图 ratio 节点说明「（kv_b_proj 可逆展开）」不成立｜引文依据：kv_b_proj 是 [28672, 512] 的线性映射（512→28672），秩不超过 512，不存在逆映射；正确的说法是展开后的 K/V 由这 576 维潜向量唯一决定｜修复要求：改为「展开后的 K/V 由 576 维潜向量唯一决定，存潜向量不丢信息」｜修复：｜复验：
- [轻微·表述] cache 视图 hf 节点说明「也是潜向量口径存在的理由」把无来源支持的归因写成结论｜引文依据：不适用（页面与官方 README 均未给出该因果陈述）｜修复要求：删去该因果归因，或降级为「这类展开缓存的体积是潜向量口径的动机之一」并注明为推断｜修复：｜复验：
- [轻微·表述] 残留的元话语与自我指代：第 4 节末尾「本节的证据性质是…本机不可行」、来源与范围说明「本页覆盖语言主干与 MTP 草稿层的完整前向数据流」、第 4 节开头「两处容易踩坑的不一致」、「注意这是逐元素门，不是每头一个标量」「注意是求和不是均值」、「MTP 按官方 vLLM 实现的语义拼装缩小版跑通」｜引文依据：不适用｜修复要求：改为不以上文/本节为主语的写法（如「覆盖范围：…」），「容易踩坑」改客观表述，「注意…」删去提示词直接陈述，「跑通」改「实测通过」｜修复：｜复验：
- [轻微·表述] 第 4 节「三方一致的部署实现（vLLM、SGLang）都按交错消费这份 checkpoint」中「三方」与列出的两项不符｜引文依据：同句只列举 vLLM 与 SGLang 两个实现｜修复要求：改为「两个一致的部署实现（vLLM、SGLang）」｜修复：｜复验：
- [轻微·格式] 显示层字符串里的 Unicode 数学字符未用 LaTeX 或文字表达｜引文依据：line 532 io='[1,T,T] -> [1,T,≤2048]'、line 558 label='logits + 稀疏掩码\nqkᵀ/√256'、line 582 label='post 门\n2σ(·) + ε'、line 566 边标签='σ(W_g x)'、line 649 说明='…≈ 90.5 GiB'（说明字段经 renderD 支持 $...$，可直接写 LaTeX）｜修复要求：说明字段改用 $...$；标签内不便渲染处改为「≤」「约」等文字（如 'min(2048, T)'、'qk^T/√256' 改文字表述）｜修复：｜复验：

## 结论

- 统计：阻断 2 / 重要 0 / 轻微 7
- 处置：修复。第 1 轮已修掉的三处（来源表由「脚本与原始输出见 research/」改为「本机实测得到（清单记在 research/measured.md）」、关键规格补 49.06B 与官方 README「49B are activated per token」的换算、F32 清单补 weights_proj）本轮复核通过。两处阻断均为「同一页内两处互相矛盾 / 算式与结论不符」，属工具提示与正文的算式问题，改动范围局限于第 2 节 grp 节点与索引器 top-k 一处表述及其 label，不影响页面其余结论；其余数字与公式已全部逐项回源核对通过（含总参数量逐位相等、626 个 F32 张量构成、真实权重小张量实值、三条实现链的 RoPE 与缓存口径）。