<!-- review-meta
round: 3
page: wiki/hy4-preview-dataflow/index.html
reviewed_content_sha256: 2d30b46b5da1295f
-->
# Hy4-Preview 前向数据流审查记录（第 3 轮）

- 页面版本：c291a37f27c48f57aef9e9865617892fdbf6491e（index.html 工作树哈希）
- 审查时间：2026-09-13 19:27
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：1. 关键规格 ｜ 2. 交互式数据流（八个视图的全部节点、连边与悬停提示）｜ 3. 要点（整体结构 / iHC / DSA 索引器 / MLA 注意力 / MoE / prefill 与 decode 的等价性 / KV cache 的两种口径）｜ 4. 实现对照 ｜ 5. MTP 草稿层 ｜ 6. 核对方式 ｜ 来源与范围说明
- 说明：本页 head 的 `dojo:type` 为 `dataflow`，适用规范为 `guides/model-dataflow.md`（其「表述」一节指向 concept 规范），问题分级与记录格式取自 `guides/concept/check.md` 第 3、6 节。

## 本轮实际打开并比对的外部来源

- 官方 `config.json`（huggingface.co/tencent/Hy4-preview/raw/main/config.json）。
- 真实 checkpoint 的 131 个 safetensors 文件头（HTTP Range 读取全部 131 个头，得 2006 个张量的名称/形状/dtype）；按 `data_offsets` 精确下载小张量实值。
- transformers `src/transformers/models/hy_v4/modeling_hy_v4.py`（commit cbc1651a）；同 commit 的 `models/deepseek_v2/modeling_deepseek_v2.py`。
- vLLM `vllm/models/hy_v4/nvidia/{attention,model,mtp,moe,hc,flashmla_sparse,triton_ihc}.py`（commit 385dce3）。
- SGLang `python/sglang/srt/configs/hy_v4.py`、`python/sglang/srt/models/hunyuan_v4.py`（commit 55bf338）。
- hub 仓库 `README.md`、`finetune/llama_factory_support/hy_v4_patches.py`、`finetune/` 文件清单。

## 本轮核对通过的主要项（复核无误，不逐条列为问题）

- 总参数 779,960,992,733 与 2006 个张量逐个累加结果逐位相等；BF16 1380 / F32 626 与真实文件头统计一致；F32 清单恰为 iHC 门控 468（78×2×3）+ 主干 sink 78 + 主干路由偏置 77 + hc_head 3 = 626。
- 单 token 激活 47.57B 的各分项均可按张量形状复算吻合：注意力每组 265,685,568、full 索引器每个 9,371,904、每组 iHC 393,236、hc_head 98,309、每层 MoE 341,311,744、首层 dense 339,738,624；路由专家合计 744.10B / 95.40% 复算一致；官方 770B 口径 769.91B（剔除 MTP 10.05B）复算一致。
- 关键张量形状（文件头实读）：q_b_proj [16384,2048]、kv_b_proj [28672,512]、kv_a_proj_with_mqa [576,6144]、indexer.wq_b [4096,2048]、indexer.wk [128,6144]、indexer.weights_proj [32,6144]、experts.gate_up_proj [256,4096,6144]、down_proj [256,6144,2048]、hc_fn [8,24576] F32、hc_base [8]、hc_scale [2]、hc_head_fn [4,24576]、hc_head_base [4]、hc_head_scale [1]、learnable_sink_param [64] F32、linear_gate.weight [16384,6144]、eh_proj [6144,12288]、lm_head/embed_tokens BF16 [120832,6144]——与页面标注一致；mtp_layers.0.* 恰 27 个且不含任何 hc 参数。
- checkpoint 小张量实值（按 data_offsets 下载解码）：第 0 层 hc_base 前 4 [-0.8859,-0.8793,-0.7178,-0.9103]、后 4 [-1.5183,-1.4180,-1.3979,-0.4856]；第 77 层 post 半区 [0.0124,0.0030,0.0122,0.2079]；hc_scale 第 0 层 [0.2503,0.0058]、第 77 层 [0.2577,0.1843]；hc_head_base [-1.1441,-1.1512,-1.2124,-0.9716]、hc_head_scale 0.10535；sink 第 0/40/77 层均值 0.3836/0.6656/-0.0222（第 0 层范围 [-6.541,3.175]）；e_score_correction_bias 第 1/40/77 层中位数 0.00127/0.00740/0.00640、第 1 层范围 [-0.0967,0.0323]、均值 4.1e-8；k_norm gamma 均值 1.0614、beta 均值 -0.0023；MTP sink BF16 均值 0.9501、范围 [-2.375,2.922]、路由偏置中位数 0.00737——与页面逐项一致。
- 机制与源码：索引器 ReLU 打分与 `min(index_topk, 长度)`、因果 -inf 掩码、sink 拼接后 softmax 去尾列、逐元素 sigmoid 门、softmax 缩放 `qk_head_dim**-0.5`、sigmoid 路由 + e_score_correction_bias 选择 + 归一化 ×2.827、swiglu gate 只截上限 / up 双向 ±10、iHC 前 4 pre 后 4 post、`base = -log(hc_mult-1)` 与 `scale=0.01` 初始化、入口 `unsqueeze(2).expand(hc_mult)`、末端 `norm(hc_head(...))` 顺序、MTP 的 enorm/hnorm/eh_proj/`enable_ihc=False`/位置 0 置零/共享主干 head——均与源码一致。
- RoPE 布局对照：vLLM 两处 `is_neox_style=False` 且注释 "Checkpoint (PTM) layout"；SGLang `rope_interleave=True`/`indexer_rope_interleave=True` 且有 `permute_hyv4_indexer_weight`；transformers hy_v4 用 `rotate_half`、deepseek_v2 用 `view_as_complex`；hub README 部署指引只覆盖 vLLM/SGLang——均与页面表格一致。
- 页内 7 个前置概念链接（deepseek-moe / dsa / mla / ihc / eagle-speculative / rope / aux-loss-free-routing）均存在；`.dojo/scripts/validate.py` 通过。

## 问题

- [阻断·技术] §1 关键规格表「iHC」行：`hc_attn_layer.hc_pre.hc_fn [8, 24576] $=2\times4\times6144$` 的等式不成立。$2\times4\times6144=49152$，而 $[8,24576]$ 共 $8\times24576=196608$ 个元素。正确分解为 $(2\times4)\times(4\times6144)$（out $2\times\mathrm{hc\_mult}=8$，in $\mathrm{hc\_mult}\times\mathrm{hidden}=24576$），页面漏了一个 4。｜引文依据：transformers modeling_hy_v4.py `HYV4HyperConnection.__init__`：`mix = 2 * self.hc_mult` 与 `self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))`（hc_mult=4 → [8,24576]）；SGLang hunyuan_v4.py `HYV4HCPreLayer`：`ReplicatedLinear(config.hidden_size * config.hc_mult, 2 * config.hc_mult)`；真实文件头 `model.layers.0.hc_attn_layer.hc_pre.hc_fn F32 [8, 24576]`。｜修复要求：把该单元格的等式改为 `$=(2\times4)\times(4\times6144)$` 或 `$=8\times24576$`，使两侧元素数一致。｜修复：｜复验：
- [重要·技术] §2 整体总览视图 `grp` 节点说明：「full 索引层在 0,1,5,9,…,73 共 21 个」。按列举到 73 只有 20 个（0、1，加 5–73 每 4 层一个共 18 个），与同句「共 21 个」矛盾，也与 §1「full 21 层：0, 1, 5, 9, …, 77」及 §3 首条「…,77」自相矛盾。｜引文依据：config.json `indexer_types` 中 `full` 的下标为 [0,1,5,9,13,…,73,77]（21 个，含 77）。｜修复要求：端点改为 77，即「0,1,5,9,…,77 共 21 个」，与 §1、§3 及 config 一致。｜修复：｜复验：
- [重要·技术] 来源与范围说明表末行：`本机实测，脚本与原始输出见 research/`。本页的实测脚本与运行输出已从仓库移除（research/ 现只剩清单与说明文件），该指向违反 model-dataflow.md「页面正文引用实测结论时写清「实测得到」而不是指向已移除的文件路径」。｜引文依据：research/measured.md：「本页的实测产物原先存放在本目录下，现已从仓库移除（内容不发布，且体积可观）」——其中 `probe_forward.py`、`verify_structure.py`、`mini_model.py` 等脚本与 `.out` 输出均已不存在。｜修复要求：改为指向登记清单（如 `research/measured.md`）或改为「本机实测」措辞，不再指向已移除的脚本与原始输出。｜修复：｜复验：
- [轻微·技术] §2 整体总览视图 `grp` 节点公式 `(\mathrm{full} + 3\,\mathrm{shared}) \times 19` 与标签「Layers 2..76」不符：$4\times19=76$ 层，而 2..76 共 75 层。该公式实际描述的是「第 1 层起，每个 full 后接 3 个 shared」的 19 组（层 1–76），但第 1 层另有独立节点 `l1`。｜引文依据：config.json `indexer_types`：full 于 1,5,…,73 共 19 个，其后各接 3 个 shared（2–4、6–8、…、74–76）。｜修复要求：标签改为「Layers 1..76」，或公式改为 $18\,\mathrm{full}+3\times19\,\mathrm{shared}$ 并保留标签 2..76。｜修复：｜复验：
- [轻微·表述] §4 引言「之间存在两处容易踩坑的不一致」中的「容易踩坑」属口语化临场评价；§3「注意这是逐元素门，不是每头一个标量」与 iHC 视图 `mix` 节点说明「注意是求和不是均值」属「注意…」式元话语（对应 check.md 第 2.2 条 12 中「需要注意的是」一类）。｜引文依据：不适用。｜修复要求：删除「注意」等引导语，改为直接陈述，如「该门为逐元素门，输出宽度 $64\times256$，不是每头一个标量」「四条流各乘自身门后相加，不是取均值」；「容易踩坑」改为中性描述（如「两处不一致」）。｜修复：｜复验：
- [轻微·格式] 视图节点标签与提示框说明中存在未经 KaTeX 渲染的 Unicode 数学字符：`topk` 节点 `io:'[1,T,T] -> [1,T,≤2048]'`（≤）、`post` 节点 `label:'post 门\n2σ(·) + ε'`（σ、ε）、`logits` 节点 `label:'logits + 稀疏掩码\nqkᵀ/√256'`（√）、MLA 视图边标签 `'σ(W_g x)'`（σ）、KV cache 视图 `sum` 节点说明「≈ 90.5 GiB」（≈）。这些字段（节点 label、io 与部分 d 文本）走纯文本渲染路径，不经 KaTeX。｜引文依据：guides/concept/style-guide.md §11「页面任何位置出现的数学变量、希腊字母、上下标、数学运算符和关系符都必须包在 `$...$` 或 `$$...$$` 中，由 KaTeX 渲染……不因位置而放宽」（该节列出 √、≈、≤ 与希腊字母）；.dojo/scripts/validate.py 的 `BARE_MATH_CHARS` 收录 σ/ε/√(221a)/≤(2264)/≈(2248)（仅 ×、→、± 被明确豁免）。｜修复要求：把标签类字段中带数学含义的部分改走 KaTeX 可渲染写法或移入 `f` 字段，提示框 `d` 中的 ≈ 写成 `$\approx$`。｜修复：｜复验：
- [轻微·技术] §1「单 token 激活 47.57B（不含 embedding 与 lm_head）」的分组累加口径不一致：注意力分组（每组 265,685,568）含注意力内部的 q_a_layernorm [2048] 与 kv_a_layernorm [512]，但每层解码器的 input_layernorm 与 post_attention_layernorm（各 6144，78 层共 958,464）不在任何分组内。读者按列出的分组复算会与「按结构规则累加」的隐含完整性对不上（差值约 0.96M，不影响 47.57B 的量级与其余数字）。｜引文依据：modeling_hy_v4.py `HYV4DecoderLayer.__init__`：`self.input_layernorm = HYV4RMSNorm(config.hidden_size, ...)`、`self.post_attention_layernorm = HYV4RMSNorm(config.hidden_size, ...)`；文件头 `model.layers.0.self_attn.q_a_layernorm.weight [2048]`、`model.layers.0.self_attn.kv_a_layernorm.weight [512]`。｜修复要求：或在分组中补入每层两个 RMSNorm，或在说明中写明「不含各层 input_layernorm / post_attention_layernorm 权重」。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 4
- 处置：修复（阻断项 §1 `hc_fn` 等式须改正后方可发布；其余为重要 2、轻微 4，逐条修复后复验）
