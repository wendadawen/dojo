<!-- review-meta
round: 4
page: wiki/moonvit-v2/index.html
reviewed_content_sha256: adcb48daa43b53e0
-->
# MoonViT-V2审查记录（第 4 轮）

- 页面版本：5ceccfd039bd21045ca6afb0333882f03b5f7f17（git hash-object 工作树 wiki/moonvit-v2/index.html）
- 审查时间：2026-09-13 21:15
- 审查者：独立子代理（未参与写作与前序轮次；未读取本页 research/ 下任何文件）
- 已完整阅读章节：核心问题 → 1. 主流做法遇到什么问题 → 2. 从零训练的方案 → 3. 训练是怎么进行的 → 4. 服务稳定性的架构 → 5. 图像/视频共享与高分辨率 → 6. 结果与结论 → 来源与范围说明（含全部折叠块与图注）；另读 overview.html
- 来源获取：Kimi K3 技术报告 arXiv:2607.24653v2（HTML 全文 + PDF，pdftotext 抽出 Fig.6 图内文字）、官方 config.json（https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json，逐字段解析）

## 来源核对（通过项，本轮逐条回源）

- C1：§2.4 "we train Kimi K3 vision encoder, MoonViT-V2, entirely from scratch with next-token prediction" ✓
- C2 / Fig.6：§2.4 "the SigLIP-initialized MoonViT-3D shows persistently higher gradient norms with frequent spikes, while MoonViT-V2 remains stable throughout training"；Fig.6 图注 "the from-scratch MoonViT-V2 maintains lower gradient norms with fewer spikes"；PDF 图内文字含纵轴 "Vision-tower gradient norm"、刻度 0/0.2/0.4/0.6，横轴 "Training step (×10³)" 刻度 7…30，图例 "MoonViT-3D (SigLIP init.)" / "MoonViT-V2 (from scratch)" ✓（与页面 N1 所述坐标轴、量级一致）
- C3：§2.4 "…rather than by a contrastive loss that favors global semantics over fine-grained textual and structural cues" ✓
- C4：§2.4 "a 27-layer vision transformer with roughly 0.4B parameters that adopts RMSNorm and removes all bias terms from its linear and attention projections, a design that further stabilizes the from-scratch optimization above" ✓；config vision_config：vt_num_hidden_layers=27、vt_hidden_size=1024、qkv_hidden_size=1536、vt_intermediate_size=4096、vt_num_attention_heads=12、patch_size=14、norm_type=rmsnorm、attn_bias/linear_bias/patch_embed_proj_bias 均为 false ✓
- C5：§2.4 "Images and videos are processed with fully shared parameters, as in MoonViT-3D: attention is factorized into intra-frame spatial and inter-frame temporal passes, and temporal pooling further compresses tokens along the time dimension" ✓
- C6：§2.4 "Before projection, a pixel-shuffle operation with 2 × 2 downsampling reduces the number of visual tokens by a factor of four, keeping inputs of up to 3584 × 3584 pixels affordable within the 1M-token context"；config merge_kernel_size=[2,2]、merge_type=sd2_tpool、patch_size=14 ✓
- C7：§2.4 "we find MoonViT-V2 matches the SigLIP-initialized baseline across vision evaluations, indicating that contrastive pre-training is unnecessary as an initialization for multimodal language models at scale" ✓
- C8：§2.4 "visual inputs are first encoded by MoonViT-V2 and then mapped by a lightweight MLP projector into the LLM" ✓
- C9：§3.3 "…native multimodal training strategy in which language and vision are jointly optimized from the start of training, rather than grafting a vision encoder onto a pre-trained language model through a post-hoc alignment stage. Under this paradigm, visual and textual tokens are interleaved within a single next-token prediction objective…"；§2.4 "no post-hoc modality-alignment stage" ✓
- C10：§3.1 "four primary text domains—Web Text, Code, Mathematics, and Knowledge—together with a large-scale vision corpus. The vision data covers captions, interleaved image–text documents, OCR, perception, video, and visual coding data"；"coordinate supervision is provided in both absolute and normalized ([0,1]) formats"；"…including SVG, 3D assets, Webpage, Game, and CAD schematics" ✓
- C11：§5.2.3 "large images and long videos substantially increase the computation time of the vision encoder and cause significant load imbalance across devices"；"A single large image is partitioned along the patch dimension across multiple devices, and attention is computed by gathering key–value pairs (gather-KV) across CP ranks"；"…preventing the communication fraction from growing with scale"；"the Decoupled Encoder Process (DEP), which splits ViT and text training into separate stages"；"The ViT forward passes of the first PP micro-batches are executed synchronously upfront, the remaining forward passes are scheduled into pipeline bubbles…most of the ViT computation is hidden within pipeline bubbles" ✓
- N3：§3.3 "We optimize the model using the Per-Head Muon optimizer (§ 2.5) together with the weight-clipping mechanism introduced in Kimi K2, while adopting QB (§ 2.3.3) for MoE load balancing. We use a cosine learning rate schedule with a 1% linear warmup. Weight decay is set to 0.1 throughout." ✓
- 数字（报告 Table 1）："Total Parameters 2.78T"、"Total Parameters of ViT 401M"、"# ViT Layers 27"、"Patch Size of ViT 14"、"# Attention Heads of ViT 12" ✓（页面所引 2.78T / 401M / 27 / 14 / 12 全部一致）
- F1 复算：4×(1024×1536)=6,291,456；2×(1024×4096)=8,388,608；2×1024=2,048；每层 14,682,112；×27 = 396,417,024 ≈ 0.40B ✓；401M−396.4M=4.58M，与 patch embedding（14×14×3×1024≈0.60M）+ divided_fixed 位置嵌入（init_pos_emb_height=width=64、time=4）之和同量级 ✓
- F2 复算：3584/14=256 → 256²=65,536 → /4=16,384 ✓
- 代码：折叠块 Python 在本机 python3 实跑，输出与页面「预期输出」逐行一致（Q/K/V/O=1,572,864；attn/layer=6,291,456；mlp/layer=8,388,608；norms/layer=2,048；per_layer=14,682,112；total=396,417,024 (~0.40B)；65536→16384）✓
- 链接与资源：../siglip/、../vit/、../standard-attention/ 三页均存在；index.html ↔ overview.html 互链；无「（待生成）」占位；数学符号全部 LaTeX（× 为 validate.py 明确豁免的散文排版字符）；`<text>` 内无公式、图内公式均在 `<foreignObject>`；alt/aria-label 无 `$`；validate.py exit 0 ✓
- 跨页一致：0.4B / 27 层 / 2.78T / 65536→16384 / 六类视觉语料在 description、dojo:summary、正文、overview.html 完全一致，未发现「同一数字两处不一致」「算式与结论不符」✓

## 问题

- [轻微·格式] 来源与范围说明第 3 个 h3（L729）：标题为「外部数字与实验条件」，缺 style-guide §1 规定的固定后缀「（N）」，且与同页「论断与来源（C）」（L707）、「公式与来源（F）」（L722）的写法不一致。｜引文依据：style-guide §1「来源章节（来源与范围说明）下的 h3 使用固定命名…`外部数字与实验条件（N）`」；页面 L729 `<h3>外部数字与实验条件</h3>`。｜修复要求：把 L729 标题改为「外部数字与实验条件（N）」。｜修复：｜复验：
- [轻微·格式] 第 3 章内容子标题（L308「训练用什么数据」、L323「训练配置：整个模型统一一套」、L327「系统层面怎么跑得动」）：使用未编号 h4，且从 h2 直接跳到 h4；style-guide §1 规定内容子标题为编号标题（h3 为 `1.1 标题`、更深层为 `1.1.1`）。同页其余章节无同层子标题，形成结构不一致。｜引文依据：style-guide §1「h3 编号 + 标题，格式固定为 `1.1 标题`…更深层标题按同样方式延长编号（如 `1.1.1`）」。｜修复要求：将三个 h4 改为带编号的 h3（`3.1 训练用什么数据`、`3.2 训练配置：整个模型统一一套`、`3.3 系统层面怎么跑得动`）。｜修复：｜复验：
- [轻微·格式] 第 5 章两个 diagram（L516–582 分解注意力图、L588–620 pixel-shuffle 图）：`<figure class="diagram">` 缺 `<figcaption class="diagram-caption">`，与第 1 章（L150）、第 3 章（L303）两图及 content-examples.md 的 figure+figcaption 范例不一致；读者只能读到 aria-label，无可见图注，折叠/引用时缺少说明。｜引文依据：不适用（格式一致性）。｜修复要求：为两图各补一行 `<figcaption class="diagram-caption">`，说明图的内容与构造性质（参照 L150/L303 的写法）。｜修复：｜复验：
- [轻微·来源] 第 2 章 L181：「方案有两个动机，报告将二者并列。」报告把训练稳定性列为主要动机、目标对齐列为其后的附带动因；页面称二者「并列」，并在核心问题 1 中以「原因有二」平行呈现，弱化了报告给出的主次。｜引文依据：§2.4 "We depart from this practice primarily for training stability." 与紧接的 "Training with next-token prediction also allows the encoder's representations to be shaped directly by the language-modeling objective…"。｜修复要求：把 L181 的「报告将二者并列」改为如实表述（如「报告以训练稳定性为主要动机、目标对齐为附带动因」），或在「动机一/动机二」处标明主次。｜修复：｜复验：
- [轻微·来源] 第 6 章 L681：「SigLIP 作为通用视觉表征模型，在零样本图像分类等任务上仍然有效」——该判断在本页来源列表（C/F/N）中无对应条目，K3 报告与 config.json 均不涉及 SigLIP 的通用视觉能力，属无来源支持的判断写成结论。｜引文依据：不适用（来源列表无对应条目）。｜修复要求：为其补可定位来源（引用 SigLIP 页/原论文并登记编号），或改为不含事实主张的边界表述（如「结论只限定于初始化的适用性，不评价 SigLIP 自身性能」）。｜修复：｜复验：

## 结论

- 处置：可发布（阻断 0 / 重要 0；5 条轻微项已逐条给出可复验的修复要求，可按 check.md 第 4 节修复后收口）

统计：阻断 0 / 重要 0 / 轻微 5
