<!-- review-meta
round: 5
page: wiki/hy4-preview-dataflow/index.html
reviewed_content_sha256: 5d78fa4ae6e12ad3
-->
# Hy4-Preview 前向数据流 审查记录（第 5 轮）

- 页面版本：cdff93fda6e6182ab75c1dd81f6ff7d9c11326d8（工作树；正文指纹 5d78fa4ae6e12ad3）
- 审查时间：2026-09-13 22:00
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：1. 关键规格；2. 交互式数据流；3. 要点（整体结构 / iHC / DSA 索引器 / MLA 注意力 / MoE / prefill-decode 等价性 / KV cache 两种口径）；4. 实现对照（RoPE 布局、checkpoint 命名）；5. MTP 草稿层；6. 核对方式；来源与范围说明；以及 8 个交互视图（overview/layer/indexer/mla/ihc/moe/mtp/cache）的全部节点、边、图例、悬停文本与页内脚本
- 只读了页面本身、外部来源与本规范；未读本页 research/ 下任何文件

## 本轮来源核对（逐条给出核对到的原文/数值）

结构与 config：从 `huggingface.co/tencent/Hy4-preview/raw/main/config.json` 逐项核对——`num_hidden_layers` 78、`hidden_size` 6144、`num_attention_heads` 64、`n_routed_experts` 256、`num_experts_per_tok` 8、`n_shared_experts` 1、`index_topk` 2048、`index_n_heads` 32、`index_head_dim` 128、`vocab_size` 120832、`max_position_embeddings` 1048576、`moe_intermediate_size` 2048、`intermediate_size` 18432、`q_lora_rank` 2048、`kv_lora_rank` 512、`qk_nope_head_dim` 192、`qk_rope_head_dim` 64、`v_head_dim` 256、`hc_mult` 4、`hc_magnitude` 2.0、`hc_eps` 1e-06、`routed_scaling_factor` 2.827、`swiglu_limit` 10.0、`n_group`/`topk_group` 均 1、`learning_sink` 系 `learnable_sink=true`/`learnable_sink_init=0.0`、`gated_mla=true`/`gating_type="elementwise"`、`tie_word_embeddings=false`、`enable_lm_head_fp32=true`、`rope_parameters.rope_theta=10000000`、`mlp_layer_types` 首项 `dense` 其余 77 项 `sparse`、`indexer_types` 78 项中 `full` 21 次 / `shared` 57 次——与页面第 1 节逐格一致。
官方 README：原文「770B total parameters, of which 49B are activated per token」「1 native MTP layer (10B total parameters, 0.7B activated)」，部署指引只列 vLLM 与 SGLang——与页面「官方 770B 口径」「49B 激活」「MTP 10.05B」「README 只覆盖这两者」一致。
checkpoint 张量头（HTTP Range 逐片读 131 个 safetensors 的 JSON 头，实测）：张量总数 2006、BF16 1380 + F32 626；F32 分布 = iHC 门控 468（78×2×3）+ 主干 sink 78 + 主干路由偏置 77 + hc_head 3；总参数 779,960,992,733；MTP 前缀参数 10,053,583,936，剔除后 769,907,408,797（769.91B）；第 0 层注意力（不含索引器）265,685,568、索引器 9,371,904、iHC 393,236、hc_head 98,309、首层 dense 339,738,624、embed 与 lm_head 各 742,391,808；按结构规则累加得单 token 激活 47,571,797,917（47.57B），加 embed/lm_head 得 49,056,581,533（49.06B）——与页面第 1 节所给数字逐位相同，分项之和与合计自洽。
形状（同上实测）：`q_a_proj [2048,6144]`、`q_b_proj [16384,2048]`、`kv_a_proj_with_mqa [576,6144]`、`kv_b_proj [28672,512]`、`o_proj [6144,16384]`、`linear_gate.weight [16384,6144]`、`learnable_sink_param [64]`、`indexer.wq_b [4096,2048]`、`indexer.wk [128,6144]`、`indexer.weights_proj [32,6144]`、`indexer.k_norm [128]`、`hc_pre.hc_fn [8,24576]`、`hc_pre.hc_base [8]`、`hc_pre.hc_scale [2]`、`hc_head_fn [4,24576]`/`hc_head_base [4]`/`hc_head_scale [1]`、`experts.gate_up_proj [256,4096,6144]`、`gate.weight [256,6144]`、`shared_experts.gate_proj [2048,6144]`、`mtp_layers.0.eh_proj.weight [6144,12288]`、第 0 层 `mlp.gate_proj [18432,6144]`——页面所有张量形状标注全部对上。
checkpoint 键名（safetensors index 与文件头实测）：`model.layers.N.hc_attn_layer.hc_pre.hc_fn|hc_base|hc_scale`（无任何 `hc_post`）、`self_attn.learnable_sink_param`、`self_attn.linear_gate.weight`、`mlp.gate.weight`、`mlp.experts.gate_up_proj`（3D 融合、无 `.weight` 后缀）、`hc_head.hc_head_{fn,base,scale}`、`mtp_layers.0.*` 恰 27 个且不含任何 hc 参数——与第 4 节「checkpoint 是 vLLM 命名的 inner 格式」「只有 hc_pre、没有 hc_post」「MTP 27 个张量」一致。
小张量实值（按 `data_offsets` 精确定位字节区间下载并按位模式解码，实测）：第 0 层 sink 均值 0.384、范围 [-6.541, 3.175]；第 40 层 0.666；第 77 层 -0.022；MTP sink 均值 0.950、范围 [-2.375, 2.922]；第 0 层 `hc_base` = [-0.8859, -0.8793, -0.7178, -0.9103, -1.5183, -1.4180, -1.3979, -0.4856]，第 77 层后 4 位 = [0.0124, 0.0030, 0.0122, 0.2079]；`hc_scale` 第 0 层 [0.2503, 0.0058]、第 77 层 [0.2577, 0.1843]；hc_head base [-1.1441, -1.1512, -1.2124, -0.9716]、scale 0.1053；路由偏置第 1 层 [-0.0967, 0.0323]、均值 4.06e-08、中位数 0.0013，第 40 层中位数 0.0074、第 77 层 0.0064、MTP 0.0074；第 0 层 `indexer.k_norm.weight` 均值 1.0614、`k_norm.bias` 均值 -0.0023——页面第 3、5 节与视图节点里引用的每个实数都对上（含小数位）。
源码：transformers `modeling_hy_v4.py` 的 `apply_rotary_pos_emb` 用 `rotate_half`（半分式，函数内注释 `# Non-interleave RoPE`）；transformers `modeling_deepseek_v2.py` 用 `view_as_complex` 相邻配对（交错）；vLLM `vllm/models/hy_v4/nvidia/attention.py` 主注意力与索引器两处 `get_rope(..., is_neox_style=False)`，并带 `Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims.` 注释；SGLang `configs/hy_v4.py` 硬编码 `rope_interleave=True`、`indexer_rope_interleave=True`，`models/hunyuan_v4.py` 的 `permute_hyv4_indexer_weight` 把每头后 `qk_rope_head_dim` 维旋转块移到前部——与第 4 节表格逐行一致。hub 的 `finetune/llama_factory_support/hy_v4_patches.py` 只有键名映射与专家融合、无任何 RoPE 置换——与页面「不含任何 RoPE 置换」「hub 没有提供该转换脚本」一致。
算式复算：q_a 12,582,912 + q_b 33,554,432 + kv_a 3,538,944 + kv_b 14,680,064 + o_proj 100,663,296 + linear_gate 100,663,296 + sink 64 + q_a/kv_a RMSNorm 2,560 = 265,685,568；wq_b 8,388,608 + wk 786,432 + weights_proj 196,608 + k_norm 256 = 9,371,904；iHC 2×(196,608+8+2) = 393,236；hc_head 98,304+4+1 = 98,309；256 专家 ×37,748,736 →744.10B、占 779.96B 的 95.40%；1M = 2^20 下 65536 B/token/layer → 4.88 MiB、4.88 TiB，65536/1152 = 56.89×，5376 B×2^20 = 5.25 GiB（页面写 2.7–5.2 GiB，取截断）；87.8+2.7 = 90.5 GiB——全部可复算且与页面标注一致。
表述维度：逐段通读（含 8 个视图的悬停说明与折叠内容）未见元话语、以「本页/本节」为主语的引导语、会话指代、调试与复现踩坑叙事、临场评价或 AI 拼接腔；`注意`/`踩坑`/`本页将` 等前轮清理过的措辞无残留，全页无 Unicode 近似公式（√、ᵀ、≤、σ、ε、² 均无）。已核并接受的非问题项：`±10`（第 1 节表格与 MoE 视图节点标签）是全页唯一的非 CJK 数学符号，与全站通用的 `×`/`→` 同类且为中文行文的区间写法，按站内惯例不计为问题。
链接与机械项：`../deepseek-moe/`、`../dsa/`、`../mla/`、`../ihc/`、`../eagle-speculative/`、`../aux-loss-free-routing/`、`../rope/` 七个内链目标页面均存在；`research/measured.md` 存在；页面引用的源码路径（transformers/vLLM/SGLang/hub finetune）均可取到；`.dojo/scripts/validate.py wiki/hy4-preview-dataflow/index.html` 返回 `validation ok`；无 `alt` 含 `$...$`、无「（待生成）」占位、无不可达文件路径；页面无可运行代码块，故无代码执行核对项。
前轮修复复验：索引器缓存 1M 口径现写 2.7–5.2 GiB（5376 B×2^20 复算 5.25 GiB，取一位小数 5.2，成立）；第 3 节「半区归属」条目已把证据限定为「只支持半区归属的顺序」，与所引实测（第 0 层后 4 位已从 0 移出至约 -1.4、第 77 层后 4 位近 0）不再冲突。

## 问题

- [重要·页面功能/视图] 第 2 节「交互式数据流」（`<div class="viz">`，index.html L127–135）｜八个视图的标签页、面包屑、图例与全部节点/边内容都只在运行时由脚本生成，HTML 里没有任何静态回退，禁用脚本时该区只剩一个 620px 的空框｜引文依据：L132 `<div id="cy"></div>`（空容器）；L713–715 `var TABS=[...]; document.getElementById('tabs').innerHTML=TABS.map(...)`；L706 `document.getElementById('legend').innerHTML=...`；L739 `loadView('overview');`；同类型同仓库页面 deepseek-v4-dataflow/kimi-k3-dataflow/qwen3-5-dataflow 均带 `<noscript>` 静态节点表（deepseek-v4-dataflow 的该段为提交 332fb1e 全量新增 115 行 `<noscript>` 表格），qwen3-8-flash-next-dataflow 用默认可见的 `<svg id="vizStatic">` + 隐藏的 `#cy`；`guides/model-dataflow.md`「视图」节原文「脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」、「发布前检查」节原文「交互视图在无脚本时仍可读」｜修复要求：把 8 个视图（至少 overview 主干与各下钻视图）的节点写成 HTML——`<noscript>` 内的节点表（列：节点 / 维度 / 公式 / 说明）或默认可见、脚本生效后再隐藏的内联 SVG/表格，使无脚本时能读出主干路径与各节点的形状、公式、依据｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 0
- 处置：修复（修完该条后即可发布；本轮未发现事实性、公式或数字错误）
