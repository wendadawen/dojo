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

## F 公式 / 可核对手算

### F1（参数量手算）
- 论断：MoonViT-V2 的 27 层 transformer 约含 3.96 亿参数，对应报告"roughly 0.4B"。
- 来源定位：config.json vision_config：vt_hidden_size=1024, qkv_hidden_size=1536, vt_intermediate_size=4096, vt_num_hidden_layers=27, attn_bias=false, linear_bias=false；K3 报告 §2.4 "roughly 0.4B parameters"
- 适用条件：不计 patch embedding、位置嵌入、投影器；仅核心 27 层（Q/K/V/O + 2 层 MLP + 2 个 RMSNorm/层）
- 推导链：每层 = Q(1024×1536)+K(1024×1536)+V(1024×1536)+O(1536×1024)+MLP(fc1: 1024×4096 + fc2: 4096×1024)+2×RMSNorm(1024 增益) = 4×1,572,864 + 2×4,194,304 + 2×1,024 = 6,291,456 + 8,388,608 + 2,048 = 14,682,112；×27 = 396,417,024 ≈ 0.40B
- 置信状态：已确认（与 config 一致，与报告 0.4B 一致）

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

注：前置概念（SigLIP/ViT/RMSNorm/next-token prediction/自注意力）的最小定义来自 WebSearch 获取的公开资料，仅用于支撑正文阅读，不作为本概念核心论断的依据；核心论断全部以 K3 报告 §2.4 与 config.json 为准。
