<!-- review-meta
round: 1
page: wiki/hy4-preview-dataflow/index.html
reviewed_content_sha256: f34382d033b16c06
-->
# Hy4-Preview 前向数据流审查记录（第 1 轮）

- 页面版本：04c2915667c53b52ed84458fcbf924d8902332f907a4dfe7177b2225941900dd（index.html 工作树 sha256）
- 审查时间：2026-09-13 19:03
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：1. 关键规格 / 2. 交互式数据流 / 3. 要点（整体结构、iHC、DSA 索引器、MLA 注意力、MoE、prefill 与 decode 的等价性、KV cache 的两种口径）/ 4. 实现对照 / 5. MTP 草稿层 / 6. 核对方式 / 来源与范围说明；并逐条读完八个视图的 VIEWS 节点定义（含 label、io、公式、d 说明）与全部图注/悬停文案。
- 核对的外部来源：huggingface.co/tencent/Hy4-preview 的 config.json、model.safetensors.index.json、README.md、finetune/llama_factory_support/hy_v4_patches.py，以及 131 个 safetensors 分片的文件头与小张量实值（HTTP Range）；transformers `src/transformers/models/hy_v4/modeling_hy_v4.py` 与 `deepseek_v2/modeling_deepseek_v2.py`；vLLM `vllm/models/hy_v4/nvidia/{attention,model,mtp}.py` 与 `vllm/models/hy_v4/__init__.py`；SGLang `python/sglang/srt/models/hunyuan_v4.py`、`configs/hy_v4.py`。
- 机械验证：`.dojo/scripts/validate.py wiki/hy4-preview-dataflow/index.html` 通过；页面引用的 7 个前置概念页（deepseek-moe、dsa、mla、ihc、eagle-speculative、aux-loss-free-routing、rope）均存在；无「（待生成）」占位。

## 事实核查结果（支持性依据）

config.json 数值全部一致：model_type=hy_v4、vocab_size=120832、hidden_size=6144、num_attention_heads=64、num_hidden_layers=78、layer_types 78 项全为 deepseek_sparse_attention、q_lora_rank=2048、kv_lora_rank=512、qk_nope_head_dim=192、qk_rope_head_dim=64、v_head_dim=256、index_n_heads=32、index_head_dim=128、index_topk=2048、n_routed_experts=256、num_experts_per_tok=8、n_shared_experts=1、routed_scaling_factor=2.827、norm_topk_prob=true、n_group=1、topk_group=1、swiglu_limit=10.0、hc_mult=4、hc_magnitude=2.0、hc_eps=1e-6、learnable_sink=true、learnable_sink_init=0.0、gating_type=elementwise、rope_theta=1e7、max_position_embeddings=1048576、num_nextn_predict_layers=1、intermediate_size=18432、moe_intermediate_size=2048。

indexer_types 的 full 下标实测为 [0,1,5,9,…,73,77]（21 个），shared 57 个，与页面第 110/143 行一致。safetensors 索引 2006 张量 / 131 分片与页面一致；全分片 dtype 统计恰为 BF16 1380 + F32 626，且 626 个 F32 精确定位为 hc 门控 468（78×2×3）+ sink 78 + 路由偏置 77 + hc_head 3，与页面第 119/237 行完全吻合。抽验形状全部一致：q_b_proj[16384,2048]、kv_b_proj[28672,512]、kv_a_proj_with_mqa[576,6144]、learnable_sink_param[64]、linear_gate[16384,6144]、experts.gate_up_proj[256,4096,6144]、mtp eh_proj[6144,12288] 与 MTP 索引器 wq_b[4096,2048]/wk[128,6144]/weights_proj[32,6144]。真实小张量实值复算：第 0 层 hc_base=[-0.8859,-0.8793,-0.7178,-0.9103,-1.5183,-1.4180,-1.3979,-0.4856]、hc_scale=[0.2503,0.0058]、第 77 层 post 半区=[0.0124,0.0030,0.0122,0.2079]、hc_head_base=[-1.1441,-1.1512,-1.2124,-0.9716]、hc_head_scale=0.1053、第 0/40/77 层 sink 均值 0.3836/0.6656/-0.0222、MTP sink 均值 0.9501（范围 [-2.375,2.922]）、路由偏置中位数 0.00127/0.00740/0.00640、MTP 路由偏置中位数 0.00737——页面引用的每一位数字均与 checkpoint 实值一致。算术自洽：单 token 激活按页面分组累加得 47,571,797,917（47.57B）、770B 口径 769.91B、MoE 每层 341,311,744、路由专家 744.10B/95.40%、KV 口径 56.89 倍、索引器 FP8 21×132=2772 B 均复算成立。源码侧：iHC 与 hc_head 的前后半分、-log3 与 0.01 初始化、pre=σ(·)+ε / post=2σ(·)+ε、外积写回、sink 拼接后丢列、elementwise 门、sigmoid+偏置 top-8、silu(min(g,10))·clip(u,-10,10)、索引器 ReLU 打分与 1/√32·1/√128 缩放、topk=min(2048,T)、decoder 先 hc_head 再 norm 等，均在 modeling_hy_v4.py 中逐条对到；vLLM 主 MLA 与索引器两处 is_neox_style=False 且注释「Checkpoint (PTM) layout」、SGLang rope_interleave=True/indexer_rope_interleave=True 与 permute_hyv4_indexer_weight 后 64 维前置、transformers rotate_half 与 deepseek_v2 view_as_complex、MTP 的 eh_proj/enorm/hnorm/enable_ihc=False/position==0 置零/shared_head 复用主干 lm_head 均已对到源码。

## 问题

- [重要·技术] 交互视图 overview 节点 grp 的说明（index.html 第 479 行，悬停可见）：full 索引层位置写成「0,1,5,9,…,73」并称「共 21 个」，末位与计数互相矛盾，也与同页多处相冲突。｜引文依据：官方 config.json 的 indexer_types 中 full 下标为 [0,1,5,9,13,…,73,77]（21 个，末位是 77）；同页第 110 行「full 21 层：0, 1, 5, 9, …, 77」、第 143 行「full 索引层是 0, 1, 5, 9, …, 77」、第 480 行「最后一层重新自选索引（74–76 复用 73 之后）」均为 77。纯枚举「0,1,5,…,73」只有 20 项。｜修复要求：把该节点说明改为「full 索引层在 0,1,5,9,…,77 共 21 个」使末位与计数一致；同节点标签「Layers 2..76\nshared 层复用 top-k」应点明该区间内还含 18 个 full 层（5…73），否则与「其余 57 层直接复用」并读会误以为区间内全是 shared。｜修复：｜复验：

- [重要·来源] 来源与范围说明表（第 265 行）：实测类结论的来源写成「本机实测，脚本与原始输出见 research/」，而 research/ 下的脚本与运行输出已从仓库移除，页面据此指向的是不存在的文件。｜引文依据：本页 research/measured.md 首句「本页的实测产物原先存放在本目录下，现已从仓库移除」；guides/model-dataflow.md「实测脚本与运行输出不随仓库分发，清单记在 research/measured.md；页面正文引用实测结论时写清「实测得到」而不是指向已移除的文件路径」。同目录现存仅 README.md 与 measured.md，无任何脚本或 .out。｜修复要求：删去「见 research/」这一指向，改为按规范写「本机实测得到」，需要登记时只指向 research/measured.md，不得指向已移除的脚本与输出。｜修复：｜复验：

- [重要·规范] 第 2 节「交互式数据流」的八个视图完全由页内 <script> 的 VIEWS 对象驱动、经 cytoscape 绘制到 <canvas>，无脚本时视图区为空，无任何 HTML 承载的节点/形状/公式。｜引文依据：guides/model-dataflow.md「脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」与发布前检查「交互视图在无脚本时仍可读」；页面第 132–135 行 `<div id="cy"></div>` 为空容器，第 467–656 行 VIEWS、第 662–682 行 cytoscape 初始化均在 <script> 内，全文无 <noscript>。｜修复要求：为视图补无脚本可读的承载——把各视图的节点、io 形状、公式随页面以 HTML 列表或 <noscript> 输出，脚本只负责切换到 canvas；若判定该规范条目需调整，则本项连同全部 dataflow 页交回规划统一处理（同仓 deepseek-v4-dataflow、kimi-k3-dataflow 等亦为此模式，属跨页问题）。｜修复：｜复验：

- [轻微·技术] 第 104 行「单 token 激活 47.57B（不含 embedding 与 lm_head）」与官方 README 的激活参数口径未对齐，页面只复现了 770B 这一项。｜引文依据：官方 README「The model comprises 770B total parameters, of which 49B are activated per token」；本页 47.57B + embed_tokens 0.742B + lm_head 0.742B = 49.06B ≈ 官方 49B。｜修复要求：在核对说明或该行补一句等式，写明官方 49B 激活口径 = 本页 47.57B + embedding 与 lm_head（各 0.742B），与已给的 770B 复现并列。｜修复：｜复验：

- [轻微·技术] 第 237 行把 626 个 F32 与 transformers 的 _keep_in_fp32_modules_strict 清单比对时，只举 k_norm、lm_head 为文件内仍存 BF16 的例外，漏掉同在清单内的 weights_proj。｜引文依据：modeling_hy_v4.py 第 764–775 行 `_keep_in_fp32_modules_strict = ["e_score_correction_bias","fn","scale","base","hc_fn","hc_scale","hc_base","weights_proj","k_norm","sinks"]`；实测 model.layers.0.self_attn.indexer.weights_proj.weight 为 BF16 [32,6144]。｜修复要求：把该句改为「清单中的 k_norm、weights_proj 与 lm_head 在文件中仍是 BF16」，使例外枚举完整。｜修复：｜复验：

- [轻微·表述] 第 252 行「本页数字分四类来源，各自的取得方式如下。」与第 218 行「本节逐项给出证据」属元话语；第 170/248/620 行把「场景」当术语（「等比模型 24 token 场景」「decode 场景」「decode 场景下位置 0」）。｜引文依据：不适用（表述类）。｜修复要求：改为直陈，如「数字来源分四类：…」并删去「本节逐项给出证据」；「24 token 场景」改为「24 token 下的复算」，「decode 场景」改为「decode 时」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 3
- 处置：修复。页面的来源一致性与算术正确性经复核全部成立（config、checkpoint 张量头与小张量实值、三个实现源码逐条对上），核心结论无误，无阻断项。逐条修复第 1、2 项与三条轻微项后即可进入下一轮；第 3 项为跨全部 dataflow 页的规范符合性问题，需返回规划决定是补无脚本承载还是调整该规范条目。