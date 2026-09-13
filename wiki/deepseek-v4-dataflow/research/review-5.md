<!-- review-meta
round: 5
page: wiki/deepseek-v4-dataflow/index.html
reviewed_content_sha256: 8e85dfd39d58a871
-->
# DeepSeek-V4-Pro 前向数据流审查记录（第 5 轮）

- 页面版本：061bf57d09e4087c519ead1c1466db1e853c96f0
- 审查时间：2026-09-13 21:08
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：1. 关键规格；2. 交互式数据流；3. 要点（整体结构：两种注意力逐层交替 + 四份残差流／CSA／HCA／两种层共有的四个细节／mHC／MoE 路由／长上下文效率的来源）；4. 本机实测与核对；来源与范围说明

## 核对记录（来源回源与复算）

**来源获取**：官方 config.json 与 inference/config.json、官方 inference/model.py 与 inference/kernel.py（均取自 huggingface.co/deepseek-ai/DeepSeek-V4-Pro）、技术报告 arXiv:2606.19348v1。

**config 逐值核对，全部一致**：num_hidden_layers=61、num_nextn_predict_layers=1、hidden_size=7168、num_attention_heads=128、num_key_value_heads=1、head_dim=512、qk_rope_head_dim=64、sliding_window=128、index_topk=1024、index_n_heads=64、index_head_dim=128、n_routed_experts=384、num_experts_per_tok=6、n_shared_experts=1、moe_intermediate_size=3072、scoring_func=sqrtsoftplus、routed_scaling_factor=2.5、num_hash_layers=3、hc_mult=4、hc_sinkhorn_iters=20、max_position_embeddings=1048576、rope_scaling.factor=16、original_max_position_embeddings=65536、rope_theta=10000、compress_rope_theta=160000、q_lora_rank=1536、o_groups=16、o_lora_rank=1024、expert_dtype=fp4、quantization_config(fp8/e4m3/ue8m0/block 128)、vocab_size=129280、tie_word_embeddings=false、swiglu_limit=10.0、norm_topk_prob=true。compress_ratios 为 62 项，前 12 项＝[128,128,4,128,4,128,4,128,4,128,4,128]，末项＝0；据此第 0、1 层连续 HCA、第 2 层起奇偶交替，HCA 31 层、CSA 30 层——与页面一致。

**报告核对**：2.2 式(6)「A_l = σ(Ã_l)」、式(7)「C_l = 2σ(C̃_l)」、式(8)「M^(t) = T_r(T_c(M^(t-1)))」（末步为行归一化，页面「与 kernel 顺序相反」的判断成立）；2.3.3「avoids exploding attention logits」；2.3.4「attention computation within the lightning indexer is performed in FP4 precision」、「reduces the KV cache size by nearly half compared with pure BF16 storage」、「approximately 2% times of that baseline」、「27% of the single-token FLOPs ... relative to DeepSeek-V3.2」——均支持页面表述。

**源码核对**：Compressor 的 `self.overlap = compress_ratio == 4`；把窗口索引与压缩索引 cat 后调用一次 sparse_attn（共享同一 softmax 分母）；`apply_rotary_emb(..., inverse=True)` 用共轭旋转；Transformer.forward 用 `h.unsqueeze(2).repeat(1,1,hc_mult,1)` 扩展残差流；compress_ratio==0 时关闭 YaRN 且用 rope_theta=10000。均与页面一致。

**复算，无一处不符**：参数分项 1547.396＋19.465＋4.030＋0.927＋0.927＋0.168＋0.084＋25.840＝1598.837 B，占比取整合计 100.1%（页面已注明为显示取整）；路由专家占比 1547.396/1598.837＝96.8%；每专家 3×3072×7168＝66.06M，61×6×66.06M＝24.18B，非专家 19.47＋4.03＋0.93＋0.17＋0.08＝24.68B，合计 48.86B；KV 每条目 64×2＋448＝576 B＝纯 BF16 的 56.25%，1M 下 (30×262144＋31×8192)×576 B＝4.355 GiB（页面 4.36），相对基线 1048576×61×8×128×2×2 B＝244 GiB 为 1.78%（页面 1.79%）；Indexer 30×262144×128×2＝1.875 GiB（页面 1.88），FP4 折算 0.469 GiB（页面 0.47）；4K/64K/1M 候选集 128＋1024 与 128＋32/512/8192、缩小配置 132＝128＋512/128、144＝128＋min(16,512/4) 均可复算；sigmoid(12)=0.999994、√softplus(12)=3.46 复核无误。未发现分项之和≠合计、算式与结论不符、同一数字正文与 summary 两处矛盾。

**表述**：通读全文（含图注与折叠内容）未见元话语、以「本页」为主语的自我指代、会话指代（我/我们/你）、调试踩坑叙事、临场评价或 AI 拼接腔。

**结构**：内链 ../hyper-connections/、../deepseek-moe/ 均存在；research/measured.md 存在；alt 无 `$...$`；`.dojo/scripts/validate.py` 通过；katex/cytoscape/dagre/prism 本地资源在位，扩展全局名 cytoscapeDagre 正确。

## 问题

- [重要·功能] 第 2 节「交互式数据流」：视图内容只存在于页尾内联脚本的 VIEWS 对象（第 449–650 行）里，由 loadView（第 688–706 行）注入空容器 `#tabs`／`#crumb`／`#cy`／`#legend`（第 137–145 行）并用 cytoscape 画布渲染，页面没有任何 `<noscript>` 静态回退。规范「视图」节要求「脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」，发布前检查亦要求「交互视图在无脚本时仍可读」；脚本失效时本节（页面同名章节，承载逐层前向路径）对读者完全不可见。仓库内同类 dataflow 页（wiki/kimi-k3-dataflow、wiki/qwen3-5-dataflow）的视图均提供 noscript 表格回退，且该回退正是由审查修复提交 36538b0 补入的，本页缺失。｜引文依据：规范原文「脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」「交互视图在无脚本时仍可读」；页面第 137–145 行为空容器，内容全部由脚本注入。｜修复要求：为第 2 节补 `<noscript>` 回退，把 8 个视图的节点（名称／维度 io／公式 f／说明 d）与边整理为静态表格或列表，数字必须与 VIEWS 数据逐项一致，使无脚本时仍能读到与脚本视图相同的前向路径与数字。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 0
- 处置：修复（事实、公式、数字与来源全部核对通过；唯一未关闭项为交互视图缺少无脚本回退，需补 noscript 后发布）