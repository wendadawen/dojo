# K3 知识体系：递归页面树规划

以 K3 技术报告（arXiv:2607.24653v1）精读为根，递归生成 K3 依赖的全部技术子页面。用户授权解除递归深度限制（原 2 层），要求全面深入、宁可多生成。

## 正本来源（三层并用，冲突标注）

- 官方 config.json：huggingface.co/moonshotai/Kimi-K3（数值）
- 官方实现源码：modeling_kimi_linear.py（数据流与公式，含 fla chunk_kda）
- 技术报告 arXiv:2607.24653v1（动机/方法/训练/性能，47 页）
- 三者冲突时记录双方与可能原因，不强行统一

## 页面树

### 主页面（paper）

| 目录 | 类型 | 主题 | 状态 |
|---|---|---|---|
| wiki/kimi-k3/ | paper | K3 技术报告精读（架构动机/各组件机制/训练/性能/评价） | 待生成 |

### 第一层子页面（K3 直接依赖）

注意力类：
| 目录 | 类型 | 主题 | 依赖前置 | 状态 |
|---|---|---|---|---|
| wiki/kda/ | concept | Kimi Delta Attention（channel-wise forget gate + delta rule + lower-bounded decay + full-rank gate） | linear-attention, delta-rule, gated-deltanet | 待生成 |
| wiki/mla/ | paper | Multi-head Latent Attention（DeepSeek-V2，KV 低秩压缩） | mqa-gqa, low-rank-projection, rope | 待生成 |
| wiki/nope/ | concept | No Position Encoding（无显式位置编码） | positional-encoding, rope | 待生成 |
| wiki/block-attnres/ | concept | Attention Residuals / Block AttnRes（跨层选择性检索） | residual-connection, standard-attention | 待生成 |

MoE 类：
| 目录 | 类型 | 主题 | 依赖前置 | 状态 |
|---|---|---|---|---|
| wiki/stable-latent-moe/ | concept | Stable LatentMoE（LatentMoE + RMSNorm 稳定化） | moe-serving(已有), deepseek-moe, latent-moe | 待生成 |
| wiki/situ-glu/ | concept | SiTU-GLU（有界激活，softcap tanh） | glu, swiglu | 待生成 |
| wiki/quantile-balancing/ | concept | Quantile Balancing（分位数负载均衡） | aux-loss-free-routing | 待生成 |

视觉类：
| 目录 | 类型 | 主题 | 依赖前置 | 状态 |
|---|---|---|---|---|
| wiki/moonvit-v2/ | concept | MoonViT-V2（从零训练的视觉编码器） | vit, siglip | 待生成 |

优化器类：
| 目录 | 类型 | 主题 | 依赖前置 | 状态 |
|---|---|---|---|---|
| wiki/per-head-muon/ | concept | Per-Head Muon（按头正交化的 Muon 变体） | muon-optimizer, newton-schulz | 待生成 |

量化类：
| 目录 | 类型 | 主题 | 依赖前置 | 状态 |
|---|---|---|---|---|
| wiki/mxfp4-qat/ | concept | MXFP4 量化感知训练 | quantization-basics | 待生成 |

后训练类：
| 目录 | 类型 | 主题 | 依赖前置 | 状态 |
|---|---|---|---|---|
| wiki/mopd/ | concept | Multi-Teacher On-Policy Distillation | knowledge-distillation, rlhf | 待生成 |
| wiki/eagle-speculative/ | concept | EAGLE-3 投机解码 draft model | speculative-decoding | 待生成 |

基础设施类：
| 目录 | 类型 | 主题 | 依赖前置 | 状态 |
|---|---|---|---|---|
| wiki/flash-kda/ | concept | FlashKDA 与 KDA Context Parallelism | gpu-execution-model(已有), linear-attention | 待生成 |
| wiki/moonep/ | concept | MoonEP 完美均衡专家并行 | moe-serving(已有) | 待生成 |

### 第二层子页面（前置概念）

| 目录 | 类型 | 主题 | 被谁依赖 | 状态 |
|---|---|---|---|---|
| wiki/linear-attention/ | concept | 线性注意力（固定状态替代 KV cache） | kda, flash-kda | 待生成 |
| wiki/delta-rule/ | concept | Delta 规则与 DeltaNet（写前擦除的递归更新） | kda | 待生成 |
| wiki/gated-deltanet/ | concept | Gated DeltaNet（门控 delta 网络） | kda | 待生成 |
| wiki/mqa-gqa/ | concept | MQA / GQA（多查询/分组查询注意力） | mla | 待生成 |
| wiki/low-rank-projection/ | concept | 低秩分解（矩阵低秩近似） | mla | 待生成 |
| wiki/rope/ | concept | RoPE 旋转位置编码 | mla, nope | 待生成 |
| wiki/positional-encoding/ | concept | 位置编码基础（绝对/相对/旋转） | nope, rope | 待生成 |
| wiki/residual-connection/ | concept | 残差连接（ResNet） | block-attnres | 待生成 |
| wiki/standard-attention/ | concept | 标准 Transformer 注意力（scaled dot-product） | block-attnres, mqa-gqa | 待生成 |
| wiki/deepseek-moe/ | paper | DeepSeekMoE（细粒度专家+共享专家） | stable-latent-moe | 待生成 |
| wiki/latent-moe/ | concept | LatentMoE（隐空间路由） | stable-latent-moe | 待生成 |
| wiki/glu/ | concept | Gated Linear Unit | situ-glu, swiglu | 待生成 |
| wiki/swiglu/ | concept | SwiGLU（Swish 门控 GLU） | situ-glu | 待生成 |
| wiki/aux-loss-free-routing/ | concept | 辅助损失无关路由（DeepSeek-V3 bias 更新） | quantile-balancing, stable-latent-moe | 待生成 |
| wiki/vit/ | concept | Vision Transformer | moonvit-v2 | 待生成 |
| wiki/siglip/ | concept | SigLIP（对比视觉预训练） | moonvit-v2 | 待生成 |
| wiki/muon-optimizer/ | concept | Muon 优化器 | per-head-muon | 待生成 |
| wiki/newton-schulz/ | concept | Newton-Schulz 正交化 | per-head-muon, muon-optimizer | 待生成 |
| wiki/quantization-basics/ | concept | 量化基础（INT8/FP8/FP4） | mxfp4-qat | 待生成 |
| wiki/knowledge-distillation/ | concept | 知识蒸馏 | mopd | 待生成 |
| wiki/speculative-decoding/ | concept | 投机解码 | eagle-speculative | 待生成 |

## 已有可复用页面

| 目录 | 类型 | 主题 |
|---|---|---|
| wiki/moe-serving/ | concept | MoE 推理与服务基础（router/top-k/EP/prefill-decode 等） |
| wiki/gpu-execution-model/ | concept | GPU 执行模型与 kernel 调度 |
| wiki/gpu-communication/ | note | GPU 通信原语与传输路径 |
| wiki/kimi-k3-dataflow/ | note | K3 前向数据流（原 note，保留作速查） |

## 执行顺序与优先级

K3 架构理解的关键路径（先做）：
1. K3 主 paper plan（识别缺口）
2. 第二层基础概念（被多人依赖的先行）：linear-attention, delta-rule, standard-attention, residual-connection, glu, rope, positional-encoding
3. 第一层架构核心：kda, mla, block-attnres, stable-latent-moe, situ-glu, quantile-balancing, nope
4. K3 主 paper write + check

K3 训练/后训练/基础设施（后做）：
5. moonvit-v2, per-head-muon, mxfp4-qat, mopd, eagle-speculative, flash-kda, moonep
6. 对应第二层：vit, siglip, muon-optimizer, newton-schulz, quantization-basics, knowledge-distillation, speculative-decoding, aux-loss-free-routing, deepseek-moe, latent-moe, mqa-gqa, low-rank-projection, gated-deltanet

## 执行方式

- 每个页面完整三阶段：plan（scope/evidence/outline/glossary）→ write（index.html + overview.html + draft-check.md）→ check（独立子代理审查 + review.md + 修复复验 + 发布门控）
- 子页面用子代理（Agent）执行完整三阶段，编排者不转发作者推理
- 独立子页面可并行生成
- 每个页面完成后更新 content.json 与首页
