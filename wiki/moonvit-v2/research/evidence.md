# MoonViT-V2 核心论断与证据

来源优先级：K3 技术报告 §2.4（原始论文）> HuggingFace 官方 config.json（对应版本官方配置）> WebSearch 获取的 SigLIP/ViT 公开资料（仅用于前置概念最小定义）。

## C 论断（核心论断）

### C1
- 论断：MoonViT-V2 完全从零训练（next-token prediction），不使用 SigLIP 等对比预训练模型初始化。
- 来源定位：K3 报告 §2.4 "A key departure from Kimi K2.5 is that we train Kimi K3 vision encoder, MoonViT-V2, entirely from scratch with next-token prediction."
- 适用条件：K3 的训练设置
- 置信状态：已确认

### C2
- 论断：从零训练的动机之一是训练稳定性——SigLIP 初始化的 MoonViT-3D 梯度范数持续更高且频繁尖峰，从零训练的 MoonViT-V2 全程更稳定。
- 来源定位：K3 报告 §2.4 + Fig.6 "Compared with the SigLIP-initialized MoonViT-3D, the from-scratch MoonViT-V2 maintains lower gradient norms with fewer spikes, indicating more stable optimization."
- 适用条件：同一架构（MoonViT-3D 设计）下的对照；Fig.6 横轴为训练步（×10³），纵轴为 vision-tower 梯度范数
- 置信状态：已确认

### C3
- 论断：next-token prediction 让编码器表征直接被语言建模目标塑造，而对比损失偏向全局语义、不利于细粒度文本与结构线索。
- 来源定位：K3 报告 §2.4 "Training with next-token prediction also allows the encoder's representations to be shaped directly by the language-modeling objective, rather than by a contrastive loss that favors global semantics over fine-grained textual and structural cues."
- 适用条件：对比预训练目标（SigLIP 等）与 next-token prediction 目标的对照
- 置信状态：已确认（报告论断）

### C4
- 论断：MoonViT-V2 是 27 层 vision transformer，约 0.4B 参数，采用 RMSNorm，去除所有 linear 与 attention 投影的 bias 项。
- 来源定位：K3 报告 §2.4 "MoonViT-V2 is a 27-layer vision transformer with roughly 0.4B parameters that adopts RMSNorm and removes all bias terms from its linear and attention projections, a design that further stabilizes the from-scratch optimization above."；config.json vision_config：vt_num_hidden_layers=27, norm_type=rmsnorm, attn_bias=false, linear_bias=false, patch_embed_proj_bias=false
- 适用条件：MoonViT-V2 的发布配置
- 置信状态：已确认（报告与 config.json 双源一致）

### C5
- 论断：图像与视频用完全共享的参数处理；注意力分解为帧内空间与帧间时间两趟；时间池化沿时间维压缩 token。
- 来源定位：K3 报告 §2.4 "Images and videos are processed with fully shared parameters, as in MoonViT-3D: attention is factorized into intra-frame spatial and inter-frame temporal passes, and temporal pooling further compresses tokens along the time dimension."
- 适用条件：MoonViT-V2 与 MoonViT-3D 共享的架构设计
- 置信状态：已确认

### C6
- 论断：投影前用 2×2 pixel-shuffle 下采样将视觉 token 数减少 4 倍；支持最高 3584×3584 像素，在 1M token 上下文内可负担。
- 来源定位：K3 报告 §2.4 "Before projection, a pixel-shuffle operation with 2 × 2 downsampling reduces the number of visual tokens by a factor of four, keeping inputs of up to 3584 × 3584 pixels affordable within the 1M-token context."；config.json vision_config：merge_kernel_size=[2,2], merge_type=sd2_tpool, patch_size=14
- 适用条件：MoonViT-V2 的 token 压缩设计
- 置信状态：已确认（报告与 config.json 一致：merge_kernel_size [2,2] 即 2×2）

### C7
- 论断：MoonViT-V2 在视觉评测上匹配 SigLIP 初始化的 baseline，说明对比预训练作为多模态大模型视觉编码器初始化在规模上并非必要。
- 来源定位：K3 报告 §2.4 "we find MoonViT-V2 matches the SigLIP-initialized baseline across vision evaluations, indicating that contrastive pre-training is unnecessary as an initialization for multimodal language models at scale."
- 适用条件：K3 报告所述规模
- 置信状态：已确认

### C8
- 论断：MoonViT-V2 的视觉通路整体设计沿用 Kimi K2.5（视觉输入先由 MoonViT-V2 编码，再由轻量 MLP 投影器映射进 LLM）。
- 来源定位：K3 报告 §2.4 "This training recipe builds on a vision pathway that follows the overall design of Kimi K2.5: visual inputs are first encoded by MoonViT-V2 and then mapped by a lightweight MLP projector into the LLM."
- 适用条件：通路结构
- 置信状态：已确认

### C9（2026-09-03 修订新增）
- 论断：K3 采用原生多模态训练策略——语言和视觉从训练一开始就联合优化，无事后模态对齐阶段；视觉与文本 token 在单一 next-token prediction 目标下交织，共享主干从一开始学习统一多模态表示。MoonViT-V2 因此不是"先单独训好再接入"的独立模型，而是从预训练第一步起作为 LLM 前端参与联合优化。
- 来源定位：K3 报告 §3.3（官方 PDF 第 11 页）"Kimi K3 adopts a native multimodal training strategy in which language and vision are jointly optimized from the start of training, rather than grafting a vision encoder onto a pre-trained language model through a post-hoc alignment stage. Under this paradigm, visual and textual tokens are interleaved within a single next-token prediction objective, enabling the shared backbone to learn unified multimodal representations from the outset."；§2.4（官方 PDF 第 5 页）"Kimi K3 is natively multimodal: text, images, and videos are processed by a single shared backbone within one context, with no post-hoc modality-alignment stage. … Rendered outputs and the code that produced them live in the same token stream"
- 适用条件：K3 的训练策略；与 C1（从零 + NTP）、C8（通路结构）共同构成训练流程的证据链
- 置信状态：已确认（官方 PDF 英文原文核对，arXiv 2607.24653）

### C10（2026-09-03 修订新增）
- 论断：K3 预训练语料由四大文本域（Web Text、Code、Mathematics、Knowledge）加大规模视觉语料构成；视觉语料覆盖六类数据——图注（captions）、图文交织文档、OCR、感知（perception）、视频、视觉编程数据；视觉语料沿用 Kimi K2.5 的分类体系，结合开源集合与内部过滤/合成/去重流水线；坐标监督同时提供绝对与归一化 [0,1] 两种格式（支撑分辨率鲁棒的定位）；程序化多模态数据把代码片段与其渲染视觉配对，格式含 SVG、3D 资产、网页、游戏、CAD 图纸。
- 来源定位：K3 报告 §3.1（官方 PDF 第 10 页）"Kimi K3 is pre-trained on a curated corpus spanning four primary text domains—Web Text, Code, Mathematics, and Knowledge—together with a large-scale vision corpus. The vision data covers captions, interleaved image–text documents, OCR, perception, video, and visual coding data."；"The vision corpus follows the taxonomy of Kimi K2.5, combining open-source collections with in-house pipelines for filtering, synthesis, and deduplication. During training, coordinate supervision is provided in both absolute and normalized ([0,1]) formats, enabling precise and resolution-robust localization. In addition to classical text-captioned images, we substantially scale up programmatic multimodal data, coupling code snippets with their rendered visuals across domain-specific formats including SVG, 3D assets, Webpage, Game, and CAD schematics."
- 适用条件：K3 预训练的数据构成；报告未披露各域采样率数字与视觉 token 占比
- 置信状态：已确认（官方 PDF 英文原文核对）

### C11（2026-09-03 修订新增）
- 论断：长上下文多模态训练中大图与长视频显著增加 vision encoder 计算时间并造成跨设备负载不均；两个对应设计——(a) 动态上下文并行：单张大图沿 patch 维切分到多设备、注意力通过跨 CP rank 聚合 KV（gather-KV）计算、CP 组分成若干子组并把多张大图负载均衡分布其中，控制通信占比不随规模增长；(b) K2.5 的解耦编码器进程（DEP）把 ViT 与文本训练切分为不同阶段，K3 进一步分解 ViT 计算——第一个 PP 微批的 ViT 前向提前同步执行、其余前向与反向调度进流水线气泡，大部分 ViT 计算被藏进气泡、大幅消除视觉编码器的有效开销。
- 来源定位：K3 报告 §5.2.3（官方 PDF 第 21 页）"In long-context multimodal training, large images and long videos substantially increase the computation time of the vision encoder and cause significant load imbalance across devices. To address this, we extend context parallelism to such large samples. A single large image is partitioned along the patch dimension across multiple devices, and attention is computed by gathering key–value pairs (gather-KV) across CP ranks. In addition, we divide each CP group into several sub-CP groups and distribute multiple large images across them in a load-balanced manner, preventing the communication fraction from growing with scale."；"In Kimi K2.5, we introduced the Decoupled Encoder Process (DEP), which splits ViT and text training into separate stages and balances vision forward and backward passes across PP stages. … The ViT forward passes of the first PP micro-batches are executed synchronously upfront, the remaining forward passes are scheduled into pipeline bubbles, and the backward passes are handled analogously. As a result, most of the ViT computation is hidden within pipeline bubbles, largely eliminating the effective overhead of the vision encoder."
- 适用条件：K3 预训练基础设施中针对视觉编码器的部分；不展开 1F1B 调度细节
- 置信状态：已确认（官方 PDF 英文原文核对）

## F 公式 / 可核对手算

### F1（参数量手算）
- 论断：MoonViT-V2 的 27 层 transformer 约含 3.96 亿参数，对应报告"roughly 0.4B"。
- 来源定位：config.json vision_config：vt_hidden_size=1024, qkv_hidden_size=1536, vt_intermediate_size=4096, vt_num_hidden_layers=27, attn_bias=false, linear_bias=false；K3 报告 §2.4 "roughly 0.4B parameters"；K3 报告 Table 1（官方 PDF 第 11 页）"Total Parameters of ViT - 401M / #ViT Layers - 27 layers / Patch Size of ViT - 14 / #Attention Heads of ViT - 12"
- 适用条件：不计 patch embedding、位置嵌入、投影器；仅核心 27 层（Q/K/V/O + 2 层 MLP + 2 个 RMSNorm/层）。Table 1 的 401M 为 ViT 总参数，与核心层手算 396M 相符；401M 的确切构成报告未披露（MLP 投影器单层到 7168 维约 29M，是否计入未说明）
- 推导链：每层 = Q(1024×1536)+K(1024×1536)+V(1024×1536)+O(1536×1024)+MLP(fc1: 1024×4096 + fc2: 4096×1024)+2×RMSNorm(1024 增益) = 4×1,572,864 + 2×4,194,304 + 2×1,024 = 6,291,456 + 8,388,608 + 2,048 = 14,682,112；×27 = 396,417,024 ≈ 0.40B
- 置信状态：已确认（与 config 一致，与报告 0.4B 与 Table 1 的 401M 一致）

### F2（token 数手算）
- 论断：3584×3584 像素输入经 patch=14 切分得 65536 个 patch token，经 2×2 pixel-shuffle 压缩为 16384 个 token。
- 来源定位：config.json vision_config：patch_size=14, merge_kernel_size=[2,2]；K3 报告 §2.4 "up to 3584 × 3584 pixels" 与 "reduces the number of visual tokens by a factor of four"
- 推导链：3584/14 = 256；256×256 = 65536；2×2 下采样 → 65536/4 = 16384
- 置信状态：已确认

### F3（RMSNorm 无 bias）
- 论断：MoonViT-V2 采用 RMSNorm（仅缩放增益、无 bias），与 config 的 norm_type=rmsnorm 及"去除所有 bias"一致。
- 来源定位：config.json vision_config：norm_type=rmsnorm；K3 报告 §2.4 "adopts RMSNorm and removes all bias terms"
- 适用条件：MoonViT-V2 的归一化选择
- 置信状态：已确认

## N 数字（外部数字与实验条件）

### N1
- 数字：vision-tower 梯度范数——MoonViT-3D（SigLIP init.）峰值约 0.6、全程更高且尖峰频繁；MoonViT-V2（from scratch）全程更低、尖峰更少（Fig.6a 全程轨迹；Fig.6b 14k–16k 局部放大，纵轴 0–0.15）
- 来源定位：K3 报告 Fig.6（§2.4）
- 实验条件：K3 预训练消融；同一架构 MoonViT-3D 设计下，SigLIP 初始化 vs 从零训练对照；横轴训练步 ×10³
- 置信状态：已确认（图为定性对照，具体数值从图中读取，正文不引用精确小数，只引用"更高/更低、尖峰更频繁/更少"的定性结论与"峰值约 0.6"的量级）

### N2
- 数字：MoonViT-V2 在视觉评测上匹配 SigLIP 初始化 baseline（报告未给出逐项分数表，只给定性结论"matches across vision evaluations"）
- 来源定位：K3 报告 §2.4
- 实验条件：K3 训练规模下的对照
- 置信状态：已确认（定性结论）；不构造具体分数数字，避免捏造

### N3（2026-09-03 修订新增）
- 数字/配置：K3 整个模型的训练配置——Per-Head Muon 优化器（§2.5）配合 Kimi K2 引入的权重裁剪机制（weight clipping）、QB（§2.3.3）做 MoE 负载均衡、余弦学习率调度（1% 线性 warmup）、权重衰减 0.1（全程）
- 来源定位：K3 报告 §3.3（官方 PDF 第 11 页）"We optimize the model using the Per-Head Muon optimizer (§2.5) together with the weight-clipping mechanism introduced in Kimi K2, while adopting QB (§2.3.3) for MoE load balancing. We use a cosine learning rate schedule with a 1% linear warmup. Weight decay is set to 0.1 throughout."
- 实验条件：K3 整个模型（含 LLM 主干与视觉通路）的统一训练配置；报告未单独披露 vision tower 的学习率或优化器超参
- 置信状态：已确认（官方 PDF 英文原文核对）；页面引用时必须注明这是整模级配置，vision tower 单独超参报告未披露

注：前置概念（SigLIP/ViT/RMSNorm/next-token prediction/自注意力）的最小定义来自 WebSearch 获取的公开资料，仅用于支撑正文阅读，不作为本概念核心论断的依据；核心论断全部以 K3 报告 §2.4 与 config.json 为准。
