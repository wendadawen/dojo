# evidence.md — K3 核心论断与证据对应

来源优先级：config.json（数值）> 技术报告 §X/Eq./Table（机制与动机）> 官方源码（数据流）。冲突标注。

## 核心论断（C）

### C1：K3 是 2.8T 参数 MoE 模型，104B 激活，1M 上下文
- **来源**：§Abstract "2.8T parameter Mixture-of-Experts model with 104 billion activated parameters, and a 1-million-token context window"；Table 1 "Total Parameters 2.78T, Activated Parameters 104.2B, Training Context Length 1M"
- **config.json**：max_position_embeddings=1048576
- **适用条件**：无
- **置信状态**：已确认

### C2：KDA+AttnRes+Stable LatentMoE 三维度扩展信息流
- **来源**：§2 "scales information flow along three complementary dimensions: sequence length, network depth, and model width"；Fig.2 架构图
- **适用条件**：无
- **置信状态**：已确认

### C3：3:1 KDA:MLA 混合，末尾加 1 MLA
- **来源**：§2.1 "Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3:1 mixing ratio... An additional Gated MLA layer is placed at the end of the backbone"
- **config.json**：full_attn_layers=[4,8,12,...,92,93]（24 个），kda_layers=[1,2,3,5,6,7,...,91]（69 个）；69+24=93
- **适用条件**：无
- **置信状态**：已确认

### C4：8 blocks × 12-layer size，N≈8 恢复大部分收益
- **来源**：§2.2 "we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer"；"N ≈ 8 recovers most of the benefit across model scales [57]"
- **config.json**：attn_res_block_size=12
- **适用条件**：93 层 / 12 = 7.75，即 7 个完整 12 层 block + 1 个 9 层部分 block = 8 个 block
- **置信状态**：已确认

### C5：896 路由专家 top-16，2 共享，稀疏度 56
- **来源**：§2.3 "scale channel mixing to 896 routed experts with 16 active experts per token, corresponding to a sparsity of 56"；"Kimi K3 fixes the number of full-width shared experts to Ns = 2"
- **config.json**：num_experts=896, num_experts_per_token=16, num_shared_experts=2
- **适用条件**：稀疏度 = 896/16 = 56
- **置信状态**：已确认

### C6：三件稳定化（RMSNorm + SiTU-GLU + QB）
- **来源**：§2.3 "Stable LatentMoE addresses these two failure modes with three components: an RMSNorm before the up-projection and Sigmoid Tanh Unit GLU (SiTU-GLU) to suppress activation explosion, and Quantile Balancing (QB) for load balancing"
- **config.json**：latent_moe_use_norm=true, activation_situ_beta=4.0, activation_situ_linear_beta=25.0, moe_router_activation_func="sigmoid"
- **适用条件**：β₁=4（gate branch），β₂=25（up branch），输出界 β₁β₂=100
- **置信状态**：已确认

### C7：约 2.5× scaling efficiency over K2
- **来源**：§1 "approximately 2.5× improvement in overall scaling efficiency over Kimi K2"；§3.2 "the scaling law curves in (Fig. 7) show that these improvements collectively deliver an approximately 2.5× gain in overall scaling efficiency over Kimi K2"
- **Fig.7**：Fitted scaling-law curves，K3 达到 K2 相同 validation loss 所需 FLOPs 减半多
- **适用条件**：OOD 验证 loss 上测量；collectively（架构+数据+训练配方组合效果）
- **置信状态**：已确认（论文未分解各因素贡献，C19 标注为推断）

### C8：cosine decay 优于 WSD
- **来源**：§3.2 "Under their respective optimal hyperparameter settings, cosine decay consistently achieves a lower final loss than WSD"
- **适用条件**：各自独立搜索最优超参后比较；固定最小学习率
- **置信状态**：已确认

### C9：NoPE 直接外推到 1M
- **来源**：§3.4 "Kimi K3 uses no explicit positional embedding (NoPE), and instead encodes positional information implicitly through the recurrent gating and decay mechanism of KDA. As a result, the model extrapolates directly to 1M-token contexts without any positional-encoding modification"
- **config.json**：mla_use_nope=true
- **适用条件**：MLA 层用 NoPE，KDA 隐式编码位置；需配合四阶段渐进训练
- **置信状态**：已确认

### C10：SFT→RL→MOPD 三阶段后训练，3域×3努力=9专家
- **来源**：§4.1 "a three-stage paradigm: initializing baseline agent capabilities via supervised fine-tuning (SFT), developing specialized domain experts at varying reasoning effort via Reinforcement Learning (RL), and consolidating these domain-specific policies into a single model using Multi-Teacher On-Policy Distillation (MOPD)"；"Crossing these three domain experts with three reasoning effort levels in {low, high, max} yields a total of nine expert models"
- **适用条件**：三域 = (i) general, (ii) general agents, (iii) coding agents；三努力 = {low, high, max}
- **置信状态**：已确认

### C11：QAT 从 SFT 阶段开始，MXFP4 权重 + MXFP8 激活
- **来源**：§4.1.4 "we quantize the MoE expert weights... to MXFP4, with activations computed in MXFP8, while all non-expert components... remain in higher precision. We perform quantization-aware training (QAT) throughout the entire post-training stage, covering both SFT and RL"
- **config.json**：quantization_config format="mxfp4-pack-quantized", group_size=32, num_bits=4, ignore=[re:.*self_attn.*, re:.*shared_experts.*, re:.*mlp.*, re:.*lm_head.*, re:.*vision_tower.*, re:.*mm_projector.*]
- **适用条件**：只量化 MoE 专家权重；注意力、共享专家、MLP、lm_head、视觉塔保持高精度
- **置信状态**：已确认

### C12：性能落后 Fable 5 和 GPT-5.6 Sol，优于 Opus 4.8/GPT-5.5/GLM-5.2
- **来源**：§1 "its overall performance still trails the most powerful proprietary models, namely Claude Fable 5 and GPT-5.6 Sol"；§6.1.4 "Kimi K3 closely trails the strongest proprietary models... while consistently outperforming Claude Opus 4.8, GPT-5.5, and GLM-5.2"
- **Table 2**：逐 benchmark 对比
- **适用条件**：Fable 5 含 fallback，GPT-5.6 Sol 含 cyberguard；coding 用不同 harness
- **置信状态**：已确认

### C13：MoonViT-V2 从零训练，匹配 SigLIP 初始化基线
- **来源**：§2.4 "we train Kimi K3 vision encoder, MoonViT-V2, entirely from scratch with next-token prediction... we find MoonViT-V2 matches the SigLIP-initialized baseline across vision evaluations, indicating that contrastive pre-training is unnecessary as an initialization for multimodal language models at scale"
- **Fig.6**：MoonViT-V2 梯度范数低于且更稳定于 SigLIP 初始化的 MoonViT-3D
- **适用条件**：27 层，~0.4B 参数，RMSNorm，无 bias
- **置信状态**：已确认

## 核心公式（F）

### F1：KDA 递归更新（Eq.1）
- **来源**：§2.1.1 Eq.1：S_t = (I − β_t k_t k_t^T Diag(α_t)) S_{t-1} + β_t k_t v_t^T, õ_t = S_t^T q_t
- **含义**：channel-wise forget gate (Diag(α_t)) 在 delta-rule 更新前应用
- **子页面**：kda 已讲清，正文引用

### F2：KDA lower-bounded decay（Eq.5）
- **来源**：§2.1.1 Eq.5：g_t^h = g_min Sigmoid(e^{A_h} z_t^h) ∈ (g_min, 0), α_t^h = exp(g_t^h) ∈ (e^{g_min}, 1)
- **含义**：scaled sigmoid 约束 log-decay 下界，g_min=-5 固定，使 BF16 不溢出
- **config.json**：gate_lower_bound=-5.0
- **子页面**：kda 已讲清

### F3：KDA full-rank output gate（Eq.6）
- **来源**：§2.1.1 Eq.6：y_t = W_o [Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t)]
- **含义**：data-dependent full-rank output gating
- **config.json**：use_full_rank_gate=true

### F4：Gated MLA output gate（Eq.7）
- **来源**：§2.1.2 Eq.7：y_t = W_o [Sigmoid(W_g x_t) ⊙ õ_t]
- **含义**：MLA 也加 input-dependent channel-wise full-rank output gate
- **config.json**：mla_use_output_gate=true

### F5：Block AttnRes 跨块注意力（Eq.8-10）
- **来源**：§2.2 Eq.8（keys/values 定义）、Eq.9（attention weights）、Eq.10（block-level V matrix）
- **含义**：块内求和，跨块全注意力，O(Ld)→O(Nd)
- **子页面**：block-attnres 已讲清

### F6：Stable LatentMoE 前向（Eq.11）
- **来源**：§2.3 Eq.11：u = Σ p_i E_i^{routed}(W↓x), y = Σ E_j^{shared}(x) + W↑ RMSNorm(u)
- **含义**：RMSNorm 插入 W↑ 前
- **config.json**：latent_moe_use_norm=true, routed_expert_hidden_size=3584

### F7：SiTU-GLU（Eq.12）
- **来源**：§2.3.2 Eq.12：SiTU-GLU(x) = β₁ tanh(W_g x/β₁) ⊙ Sigmoid(W_g x) ⊙ β₂ tanh(W_u x/β₂)
- **含义**：softcap tanh 约束两个因子，输出界 |f|≤β₁β₂=100
- **config.json**：activation_situ_beta=4.0, activation_situ_linear_beta=25.0
- **子页面**：situ-glu 已讲清

### F8：Quantile Balancing 更新（Eq.14）
- **来源**：§2.3.3 Eq.14：b̃_j^{(t+1)} ← −quantile_{1−k/n}(s_{:,j}^{(t)} − α^{(t)}), b^{(t+1)} ← b̃^{(t+1)} − mean(b̃^{(t+1)})
- **含义**：从 router score 分位数直接设定 bias，使每专家获目标负载
- **子页面**：quantile-balancing 已讲清

### F9：MOPD per-token OPD reward（Eq.15）
- **来源**：§4.1.3 Eq.15：r_opd^d(y_t|e,x,y<t) = clip(sg(log π_teacher/π_θ), −R_max, R_max)
- **含义**：stop-gradient，clip 约束极端 advantage

### F10：EAGLE LK loss（Eq.16）
- **来源**：§4.1.4 Eq.16：L_LK = −Σ log min(p(x), q(x))
- **含义**：直接优化接受率的负对数

## 关键数字（N）

### N1：2.78T 总参，104.2B 激活
- **来源**：Table 1
- **config.json**：可从 num_experts=896, moe_intermediate_size=3072, hidden_size=7168 等推算
- **适用条件**：无
- **置信状态**：已确认

### N2：93 层（69 KDA + 24 MLA）
- **来源**：Table 1 "Attention-Layer Composition 69 KDA + 24 MLA"
- **config.json**：num_hidden_layers=93, full_attn_layers（24个）, kda_layers（69个）
- **置信状态**：已确认

### N3：hidden_size=7168, 96 heads, head_dim=128（KDA）/ nope 128 + rope 64（MLA）
- **来源**：Table 1 "Hidden Dimension 7,168, Attention Heads 96"
- **config.json**：hidden_size=7168, num_attention_heads=96, num_key_value_heads=96, linear_attn_config.head_dim=128, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=512, q_lora_rank=1536
- **置信状态**：已确认

### N4：K2 vs K3 架构对比
- **来源**：Table 1
- **关键 Δ**：Layers 61→93（+52%），Total 1.04T→2.78T（+167%），Activated 32.6B→104.2B（+220%），Experts 384→896（+133%），Active 8→16（+100%），Shared 1→2（+100%），Heads 64→96（+50%），Context 128K→1M（8×）
- **置信状态**：已确认

### N5：2.5× scaling efficiency
- **来源**：§1, §3.2, Fig.7
- **适用条件**：OOD 验证 loss；collectively
- **置信状态**：已确认

### N6：Terminal-Bench 2.1 — K3 88.3 vs GPT-5.6 Sol 88.8 vs Fable 5 88.0
- **来源**：Fig.1, Table 2
- **适用条件**：harness 差异——K3 报 best across harnesses；AA 报 85%（不同来源）
- **置信状态**：已确认（harness 差异已标注）

### N7：ProgramBench — K3 77.8（第一）
- **来源**：Fig.1, Table 2
- **置信状态**：已确认

### N8：SWE-Marathon — K3 42.0（第一，+7 over Fable 5 35.0）
- **来源**：Fig.1, Table 2
- **适用条件**：H20-calibrated 分支，Docker images/performance gates/recalibrated for H20；Fable 5 hits fallbacks on 35% tasks
- **置信状态**：已确认

### N9：BrowseComp — K3 91.2（第一）at $2.03/任务
- **来源**：Fig.1, Table 2, §6.4
- **适用条件**：300K token 触发 context compaction；1M 全上下文无压缩时 K3 得 90.4%
- **置信状态**：已确认

### N10：第三方评估
- **来源**：§6.3, Table 5
- **关键数字**：AA Intelligence Index 57.1（#4/580）；Vals Index 74.7%（#2/39）；WebDev Arena 1678 Elo（#1/99，首个开源登顶）；Agent Arena 9.1（#4/37）
- **适用条件**：as of July 23, 2026
- **置信状态**：已确认

### N11：MoonViT-V2 — 27 层，~0.4B 参数，patch_size=14
- **来源**：§2.4 "MoonViT-V2 is a 27-layer vision transformer with roughly 0.4B parameters"
- **config.json**：vision_config.vt_num_hidden_layers=27, patch_size=14, vt_num_attention_heads=12, vt_hidden_size=1024
- **置信状态**：已确认

### N12：渐进上下文扩展四阶段
- **来源**：§3.4 "four-stage curriculum. The window grows from 8K to 64K tokens during pre-training, and from 256K to 1M tokens during the cooldown phase"
- **置信状态**：已确认

### N13：MTP/EAGLE draft — 7 步展开
- **来源**：§4.1.4 "the draft is unrolled for seven steps during training"
- **置信状态**：已确认

## 冲突记录

### 冲突 1：Terminal-Bench 分数
- **双方**：论文 Fig.1/Table 2 报 88.3（best across harnesses）；用户提及 AA 报 85%
- **可能原因**：不同 harness 或评估条件
- **处理**：正文标注 harness 差异，两个数字都列出

### 冲突 2：DeepSWE 分数
- **双方**：Fig.1 报 67.5（v1.1 tasks）；论文 §6.1.3 提及 "Kimi K3 attains 67.3 with the mini-SWE-agent harness"
- **可能原因**：不同 harness
- **处理**：正文用 67.5（主表数字），脚注 mini-SWE-agent 的 67.3

## 原图候选

| 候选 | 原文 Figure | 内容一句话 | 教学点 | 获取途径 |
|---|---|---|---|---|
| Fig.2 | Figure 2 | K3 架构图，token/channel/layer mixing + 视觉通路 | 三维度架构总览 | PDF 截图 |
| Fig.7 | Figure 7 | K2 vs K3 scaling-law 曲线，2.5× | scaling efficiency 定量 | PDF 截图 |
| Table 1 | Table 1 | K2 vs K3 架构对比 | 量化改进幅度 | 文本表格（自绘）|
| Fig.1 | Figure 1 | 主结果柱状图 | 性能定位 | PDF 截图 |

注：本页不内联原图（PDF 截图质量受限），用自绘表格和文字描述呈现关键数据。Table 1 和 Table 2 用自绘 HTML 表格。
