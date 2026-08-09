# glossary.md — K3 术语表

登记全文首次出现的术语、缩写和符号。后续阶段写作和审查以此为准。

## 术语与缩写

| 术语/缩写 | 首现位置 | 定义或含义 |
|---|---|---|
| K3 / Kimi K3 | §标题 | 本文主角，2.8T 参数 MoE 多模态模型 |
| K2 / Kimi K2 | §1 | K3 的前代模型，1.04T/32.6B，61 层全 MLA |
| MoE | §Abstract | Mixture-of-Experts，稀疏激活架构，每 token 只激活部分专家 |
| KDA | §1, §2.1.1 | Kimi Delta Attention，channel-wise forget gate + delta rule 的线性注意力变体 |
| MLA | §2.1.2 | Multi-head Latent Attention，DeepSeek-V2 提出的 KV 低秩压缩注意力 |
| Gated MLA | §2.1 | K3 的 MLA 变体，加 input-dependent output gate + NoPE |
| NoPE | §2.1.2 | No Position Encoding，不用显式位置编码 |
| AttnRes | §1, §2.2 | Attention Residuals，跨层选择性检索的残差变体 |
| Block AttnRes | §2.2 | AttnRes 的分块版本，块内求和、跨块全注意力 |
| LatentMoE | §2.3 | 分离全宽共享专家与紧凑潜在空间路由专家的 MoE 变体 |
| Stable LatentMoE | §2.3 | K3 的 LatentMoE 变体，加 RMSNorm + SiTU-GLU + QB 三件稳定化 |
| SiTU-GLU | §2.3.2 | Sigmoid Tanh Unit GLU，用 softcap tanh 约束 SwiGLU 的有界激活 |
| SwiGLU | §2.3.2 | Swish-gated GLU，主流 FFN 激活，两个因子无界 |
| GLU | §2.3.2 | Gated Linear Unit，sigmoid 门控的线性单元 |
| QB | §2.3.3 | Quantile Balancing，从 router score 分位数设定 expert bias 的负载均衡 |
| aux-loss-free routing | §2.3.3 | DeepSeek-V3 的无辅助损失路由，用 bias 调节 dispatch |
| MoonViT-V2 | §2.4 | K3 的视觉编码器，从零训练，27 层，~0.4B |
| MoonViT-3D | §2.4 | 对比基线，SigLIP 初始化的视觉编码器 |
| SigLIP | §2.4 | 对比预训练视觉模型，常用于初始化视觉编码器 |
| Muon | §2.5 | 矩阵参数优化器，用 Newton-Schulz 正交化 |
| Per-Head Muon | §2.5 | K3 的 Muon 变体，按头分别正交化 |
| Newton-Schulz | §2.5 | 矩阵正交化的迭代算法 |
| SFT | §4.1.1 | Supervised Fine-Tuning，监督微调 |
| RL | §4.1.2 | Reinforcement Learning，强化学习 |
| MOPD | §4.1.3 | Multi-Teacher On-Policy Distillation，多教师在线策略蒸馏 |
| GRM | §4.1.2 | Agentic Generative Reward Model，智能体生成式奖励模型 |
| QAT | §4.1.4 | Quantization-Aware Training，量化感知训练 |
| MXFP4 | §4.1.4 | Microscaling FP4，4-bit 浮点量化格式 |
| MXFP8 | §4.1.4 | Microscaling FP8，8-bit 浮点激活格式 |
| MTP | §4.1.4 | Multi-Token Prediction，多 token 预测 |
| EAGLE-3 | §4.1.4 | 投机解码的 draft model 方法 |
| draft model | §4.1.4 | 投机解码中小模型先草拟、大模型验证的加速方法 |
| speculative decoding | §4.1.4 | 投机解码，用小模型加速大模型推理 |
| LK loss | §4.1.4 | 直接优化投机解码接受率的损失函数 |
| FlashKDA | §5.1.1 | KDA 的 CUTLASS chunkwise 融合内核 |
| KCP | §5.1.2 | KDA Context Parallelism，KDA 的跨设备上下文并行 |
| MoonEP | §5.2.1 | Moonshot 的完美均衡专家并行方案 |
| EP | §5.2 | Expert Parallelism，专家并行 |
| TP | §5.1.1 | Tensor Parallelism，张量并行 |
| PP | §5.2 | Pipeline Parallelism，流水线并行 |
| VP | §5.2 | Virtual Pipeline，虚拟流水线阶段 |
| CP | §5.1.2 | Context Parallelism，上下文并行 |
| ZeRO | §5.2 | Zero Redundancy Optimizer，零冗余优化器 |
| AgentENV | §5.3.2 | K3 的 microVM 沙箱运行时 |
| microVM | §5.3.2 | 轻量级虚拟机，提供高保真隔离 |
| XTML | §4.1.1, 附录F | eXtensible Token Markup Language，K3 的 chat template 格式 |
| partial rollout | §4.1.2 | 部分轨迹完成即进行策略优化的 RL 技术 |
| harness | §6.1.3 | 评估时的智能体框架（如 Kimi Code/Claude Code/Codex） |
| OOD validation | §3.2 | Out-of-Distribution 验证数据，用于 scaling law 评估 |
| cosine decay | §3.2 | 余弦退火学习率调度 |
| WSD | §3.2 | Warmup Stable Decay，另一种学习率调度 |
| sparsity | §2.3 | 稀疏度 = 总专家/激活专家 = 896/16 = 56 |
| shared expert | §2.3 | 共享专家，全宽，每层始终激活，K3 有 2 个 |
| routed expert | §2.3 | 路由专家，潜在空间，由 router 选择，K3 有 896 个 top-16 |
| delta rule | §2.1.1 | 快速权重编程器的更新规则，KDA 的递归基础 |
| chunkwise parallel | §2.1.1 | KDA 跨 chunk 递归、chunk 内并行的计算形式 |
| Tensor Core | §2.1.1 | GPU 上的矩阵计算单元 |
| BF16 | §2.1.1 | Bfloat16，16-bit 浮点格式 |
| prefix cache | §5.4.1 | 前缀缓存，复用已计算的 KV/state |
| KV cache | §2.1.2 | Key-Value 缓存，注意力推理时存储的历史 |
| recurrent state | §2.1.1 | KDA 的固定大小递归状态 S ∈ R^{dk×dv} |

## 符号

| 符号 | 首现位置 | 含义 |
|---|---|---|
| d | §2.1.1 | model hidden dimension（7168）|
| d_k | §2.1.1 | key/query 维度（KDA head_dim=128）|
| d_v | §2.1.1 | value 维度（v_head_dim=128）|
| S_t | §2.1.1 Eq.1 | KDA 递归状态 ∈ R^{dk×dv} |
| α_t | §2.1.1 Eq.1 | channel-wise one-step retention factor ∈ (0,1)^dk |
| β_t | §2.1.1 Eq.1 | delta-rule write strength ∈ (0,1) |
| g_t^h | §2.1.1 Eq.5 | per-step log-decay ∈ (g_min, 0) |
| g_min | §2.1.1 Eq.5 | log-decay 下界，固定 -5 |
| A_h | §2.1.1 Eq.5 | learnable per-head log-scale |
| z_t^h | §2.1.1 Eq.2 | decay logit |
| C | §2.1.1 | chunk size |
| Γ | §2.1.1 Eq.3 | channel-wise cumulative decay |
| L | §2.2 | 网络深度（93）|
| N | §2.2 | block 数（8）|
| S (block) | §2.2 | block size = L/N = 12 |
| b_n | §2.2 | block n 的层输出求和表示 |
| q_l | §2.2 Eq.8 | layer l 的 pseudo-query = w_l |
| α_{i→l} | §2.2 Eq.9 | layer l 对 layer i 的 attention weight |
| ℓ | §2.3 | routed-expert latent space width（3584）|
| W↓ | §2.3 Eq.11 | 路由路径下投影 |
| W↑ | §2.3 Eq.11 | 路由路径上投影 |
| E_i^{routed} | §2.3 Eq.11 | 路由专家 FFN |
| E_j^{shared} | §2.3 Eq.11 | 共享专家 FFN |
| p_i | §2.3 Eq.11 | router weight |
| N_s | §2.3 | 共享专家数（2）|
| β₁ | §2.3.2 Eq.12 | SiTU-GLU gate branch softcap（4）|
| β₂ | §2.3.2 Eq.12 | SiTU-GLU up branch softcap（25）|
| softcap(x,β) | §2.3.2 | β tanh(x/β)，平滑截断 |
| s_{i,j} | §2.3.3 Eq.13 | token i 对专家 j 的 router score |
| b_j | §2.3.3 Eq.13 | expert j 的 bias |
| T_i | §2.3.3 Eq.13 | token i 的 top-k 选中专家集 |
| q | §2.3.3 | 目标负载 = mk/n |
| m | §2.3.3 | batch 中 token 数 |
| n | §2.3.3 | 专家数（896）|
| k | §2.3.3 | 每 token 激活专家数（16）|
| α_i | §2.3.3 | token i 的 top-(k+1) cutoff |
| τ | §4.1.2 | reasoning effort budget multiplier |
| R_max | §4.1.3 Eq.15 | OPD reward clip 阈值 |
| π_teacher | §4.1.3 Eq.15 | 教师模型 |
| π_θ | §4.1.3 Eq.15 | 学生模型 |
| E | §5.2.1 | 专家总数（896）|
| R | §5.2.1 | EP size |
| E/R | §5.2.1 | redundant expert 上界 |

## config.json 关键字段速查

| 字段 | 值 | 含义 |
|---|---|---|
| num_hidden_layers | 93 | 总层数 |
| hidden_size | 7168 | 隐藏维度 |
| num_attention_heads | 96 | 注意力头数 |
| num_key_value_heads | 96 | KV 头数 |
| kv_lora_rank | 512 | MLA KV 低秩维度 |
| q_lora_rank | 1536 | MLA Q 低秩维度 |
| qk_nope_head_dim | 128 | MLA NoPE 部分 head dim |
| qk_rope_head_dim | 64 | MLA RoPE 部分 head dim |
| v_head_dim | 128 | V head dim |
| num_experts | 896 | 路由专家数 |
| num_experts_per_token | 16 | 每 token 激活路由专家数 |
| num_shared_experts | 2 | 共享专家数 |
| moe_intermediate_size | 3072 | MoE 每专家中间维度 |
| routed_expert_hidden_size | 3584 | 路由专家潜在空间宽度 ℓ |
| intermediate_size | 33792 | dense FFN 中间维度 |
| first_k_dense_replace | 1 | 前 1 层用 dense FFN |
| vocab_size | 163840 | 词表大小 |
| max_position_embeddings | 1048576 | 最大上下文（1M）|
| attn_res_block_size | 12 | AttnRes block 大小 |
| activation_situ_beta | 4.0 | SiTU-GLU β₁ |
| activation_situ_linear_beta | 25.0 | SiTU-GLU β₂ |
| mla_use_nope | true | MLA 用 NoPE |
| mla_use_output_gate | true | MLA 加 output gate |
| latent_moe_use_norm | true | LatentMoE 加 RMSNorm |
| moe_router_activation_func | "sigmoid" | router 用 sigmoid |
| gate_lower_bound | -5.0 | KDA g_min |
| use_full_rank_gate | true | KDA full-rank gate |
| head_dim (linear_attn) | 128 | KDA head dim |
| short_conv_kernel_size | 4 | KDA short conv kernel |
| full_attn_layers | 24 个 | MLA 层索引（4,8,...,92,93）|
| kda_layers | 69 个 | KDA 层索引（1,2,3,5,...,91）|
| vt_num_hidden_layers | 27 | MoonViT-V2 层数 |
| patch_size | 14 | MoonViT-V2 patch 大小 |
