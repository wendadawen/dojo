# MLA 内容范围

## 1. 概念歧义处理

- 名称：Multi-head Latent Attention，缩写 MLA。
- 同名问题：MLA 在其他领域可指 Machine Learning Accelerator、Master of Landscape Architecture 等；在注意力机制语境下无歧义。
- 同族术语：Linear Attention、Gated Linear Attention、Gated Delta Attention（KDA）都与 MLA 名称相近但机制不同。
- 裁定：本文采用 DeepSeek-V2 §2.1 提出的 Multi-head Latent Attention 含义，依据为该论文首次提出 MLA 并给出完整定义。K3 §2.1.2 的 Gated MLA 是在该基础上的改动，本文一并覆盖 K3 的改动部分，但 DeepSeek-V2 的原始 MLA 是本体。

## 2.1 概念含义

- 概念名称：Multi-head Latent Attention（多头潜注意力，MLA）。
- 一句话定义：MLA 把每个 token 的 K 和 V 压缩成一个低维潜向量 c_t 缓存起来，注意力计算时再用学习到的上投影把 K 和 V 重建出来，从而在大幅减少 KV cache 的同时保留全局 token-to-token 注意力。
- 正式定义：DeepSeek-V2 §2.1.2 Eq.(9)–(11) 给出 KV 联合压缩 $\mathbf{c}_t^{KV} = W^{DKV}\mathbf{h}_t$，$\mathbf{k}_t^C = W^{UK}\mathbf{c}_t^{KV}$，$\mathbf{v}_t^C = W^{UV}\mathbf{c}_t^{KV}$；Eq.(12)–(13) 给出 query 压缩 $\mathbf{c}_t^Q = W^{DQ}\mathbf{h}_t$，$\mathbf{q}_t^C = W^{UQ}\mathbf{c}_t^Q$；§2.1.3 Eq.(14)–(19) 给出解耦 RoPE 与最终注意力公式。推理时每 token 仅缓存 $\mathbf{c}_t^{KV}$（与一个共享的解耦 RoPE key $\mathbf{k}_t^R$），共 $(d_c + d_h^R)l$ 个元素。
- 本文语境：以 DeepSeek-V2 原始 MLA 为本体（覆盖动机、压缩机制、矩阵吸收、解耦 RoPE、KV cache 对比），再覆盖 K3 §2.1.2 的两处改动（NoPE、input-dependent full-rank output gate）。K3 报告中与 KDA 混合使用 MLA 的部分（哪些层是 MLA、哪些是 KDA）只做背景提及，不展开调度细节。

### 包括什么

- KV 联合压缩公式与符号：c_t^{KV}、W^{DKV}、W^{UK}、W^{UV}（DeepSeek-V2 §2.1.2）。
- Query 压缩公式与符号：c_t^Q、W^{DQ}、W^{UQ}（§2.1.2）。
- 解耦 RoPE：q^R、k^R、拼接 [·;·]、注意力分母 $\sqrt{d_h + d_h^R}$（§2.1.3）。
- 推理时的"矩阵吸收"：$W^{UK}$ 吸进 $W^Q$、$W^{UV}$ 吸进 $W^O$（§2.1.2 末段）——属 MLA 推理效率的核心组成，不是工程附加。
- KV cache 对比表（§2.1.4 Table 1：MHA / GQA / MQA / MLA 每 token 元素数）。
- K3 改动（§2.1.2）：MLA 层用 NoPE；output gate $y_t = W^O[\mathrm{Sigmoid}(W^g x_t) \odot \tilde o_t]$。
- K3 实际数值（config.json）：q_lora_rank=1536、kv_lora_rank=512、qk_nope_head_dim=128、qk_rope_head_dim=64、v_head_dim=128、mla_use_output_gate=true、mla_use_nope=true、num_attention_heads=96、hidden_size=7168。

### 不包括什么

- 训练 kernel 实现（FlashAttention / FP32 output tile / KV staging buffer 重叠等工程细节，K3 §2.1.2 末段）：与"MLA 机制是什么"无关，属 GPU 工程页范围。
- K3 的混合调度（哪些层是 MLA、哪些是 KDA、attn_res_block_size=12 的设计动机）：属 K3 整体架构页，本文只说明 MLA 层不施加 RoPE 这一与 MLA 直接相关的改动。
- 线性注意力族（KDA、DeltaNet、 gated linear attention）的机制：与 MLA 机制不同（MLA 仍是 softmax 全局注意力），不纳入；只在一句话对照中说明 MLA 不是线性注意力。
- 量化 / 投机解码等推理系统话题：与 MLA 本身无关。

### 相邻概念

- Grouped-Query Attention（GQA）/ Multi-Query Attention（MQA）：同样为减少 KV cache 而设计，但通过共享 K/V 头而非低秩压缩；MLA 与之对比在 §2.1.4 Table 1。纳入对照（比较表中给出），不展开 GQA/MQA 的内部机制——前置概念 mqa-gqa 负责。
- Linear Attention：把 softmax 换成核分解降阶到 $O(N)$；MLA 不改 softmax，仍做全局 $O(N^2)$ 注意力，只是压缩存储。不纳入。
- Low-rank projection（低秩投影）：MLA 的核心数学工具，是前置概念。
- RoPE（旋转位置编码）：MLA 解耦 RoPE 的前提知识，是前置概念。

## 2.2 学习目标

### Q1：MLA 用低秩联合压缩如何减少 KV cache，同时保留全局 token-to-token 注意力？

- 完成答案：读者应能写出 c_t^{KV} = W^{DKV} h_t、k_t^C = W^{UK} c_t^{KV}、v_t^C = W^{UV} c_t^{KV}；解释"只缓存 c_t^{KV}、不缓存 K/V"如何把每 token cache 从 2 n_h d_h 降到 d_c；并指出注意力公式仍是 softmax(q^T k / sqrt(·)) v，对所有前序 token 求和——全局 token-to-token attention 没有被改成稀疏或滑动窗口。
- 为什么是核心目标：不理解压缩与重建机制就无法理解 MLA 是什么；不指出全局注意力保留就无法与稀疏/线性注意力区分。
- 依赖内容：MHA 的 KV cache 来源、低秩投影概念、压缩与重建公式、注意力公式。

### Q2：为什么 MLA 要把 RoPE 解耦，不能直接对压缩潜向量 c_t 应用 RoPE？

- 完成答案：读者应能说明——若对 k_t^C = W^{UK} c_t^{KV} 应用 RoPE，则 RoPE 矩阵 R 会插在 q^T W^{UK} 之间（位置敏感），导致 W^{UK} 无法在推理时被吸收进 W^Q；解耦方案是额外用一个共享的 k_t^R = RoPE(W^{KR} h_t) 承载位置信息，content 部分 q^C/k^C 保持可吸收，注意力用拼接 [q^C; q^R] 与 [k^C; k^R] 计算，分母变成 $\sqrt{d_h + d_h^R}$。
- 为什么是核心目标：不理解解耦就无法理解 MLA 为什么公式这么"奇怪"（两个 key、拼接、改了分母），也理解不了 K3 干脆对 MLA 层用 NoPE 的动机。
- 依赖内容：RoPE 是位置敏感的旋转矩阵、矩阵吸收（Q4）、解耦公式 Eq.(14)–(18)。

### Q3：MLA 的 KV cache 每 token 占多少元素，相比 MHA / GQA / MQA 减少了多少？

- 完成答案：读者应能写出四种机制的每 token KV cache 元素数：MHA = 2 n_h d_h l、GQA = 2 n_g d_h l、MQA = 2 d_h l、MLA = (d_c + d_h^R) l；用 DeepSeek-V2 数值（n_h=128, d_h=128, d_c=512, d_h^R=64, l=60）算出 MLA = 576×60、MHA = 32768×60，比值约 1/57；用 K3 数值（n_h=96, d_h=128, d_c=512, d_h^R=64）算出每 MLA 层每 token cache = 576，对应 MHA = 24576，比值约 1/43。
- 为什么是核心目标：MLA 的核心价值就是 KV cache 减少；不会算就无法判断 MLA 是否值得这些复杂度。
- 依赖内容：四种注意力机制每 token 缓存什么、l 的含义、DeepSeek-V2 与 K3 config 数值。

### Q4：推理时为什么可以不显式重建 K 和 V——"矩阵吸收"如何工作？

- 完成答案：读者应能说明——注意力的 q^T k^C 项可改写为 q^T W^{UK} c_t^{KV} = (W^{UK} T q)^T c_t^{KV}，因此推理时把 W^{UK} 吸收进 W^Q 得到 q' = W^{UK} T W^Q c_t^Q，直接用 c_t^{KV} 当 key；同理 v_t^C = W^{UV} c_t^{KV} 代入 o = sum softmax(...) v 后，W^{UV} 可外推吸进 W^O。结果：推理时不存 K、V，也不显式算 k_t^C / v_t^C，只存 c_t^{KV} 并直接参与注意力。但 RoPE 部分不能吸收（见 Q2），所以解耦 key k_t^R 仍需单独缓存。
- 为什么是核心目标：矩阵吸收是 MLA 推理效率的另一半关键；不理解它就会以为 MLA 推理时还要算 K/V，从而低估其效率优势。
- 依赖内容：注意力公式的代数等价变形、矩阵乘法结合律、Q2 的 RoPE 不可吸收结论。

### Q5：K3 在 DeepSeek-V2 的 MLA 上做了哪两处改动，各自解决什么问题？

- 完成答案：读者应能指出——（a）K3 对所有 MLA 层使用 NoPE（不施加 RoPE），来自 K3 §2.1.2 第二段，动机是与 KDA 混合架构下让 MLA 层负责"无位置约束的全局内容交互"、KDA 层负责"位置敏感的近邻混合"，且免去扩展上下文时调整 RoPE 频率基（如 YaRN）的麻烦；config.json 中 mla_use_nope=true 对应此项。（b）K3 给 MLA 加 input-dependent、channel-wise full-rank output gate $y_t = W^O[\mathrm{Sigmoid}(W^g x_t) \odot \tilde o_t]$（K3 §2.1.2 Eq.(7)），W^g 满秩；作用是让每个 token 可调制从全局注意力读出的通道；config.json 中 mla_use_output_gate=true 对应此项。
- 为什么是核心目标：K3 是本概念页的直接应用对象，不理解这两处改动就无法把 MLA 与 K3 的 Gated MLA 对应起来。
- 依赖内容：DeepSeek-V2 原始 MLA、K3 §2.1.2 报告文本、K3 config.json。

## 2.3 内容分级

### 核心内容（缺一不可，对应学习目标）

- MHA 下 KV cache 的来源（每头一份 K 和 V、随 n_h 线性增长）——为 Q1 提供动机；结论：每 token 每层 cache 2 n_h d_h。
- KV 联合压缩三公式 Eq.(9)(10)(11) 与符号——Q1 直接依赖；结论：c_t^{KV} 是 d_c 维潜向量，K/V 由它重建。
- Query 压缩 Eq.(12)(13)——Q4 矩阵吸收需要；结论：q 也走了低秩中间表示。
- 注意力公式仍是对全部前序 token 的 softmax 加权和——Q1 区分 MLA 与稀疏/线性注意力的关键；结论：注意力结构未变。
- KV cache 对比 Table 1 四种机制的公式——Q3 直接依赖；结论：MLA = (d_c + d_h^R) l。
- 矩阵吸收的两个代数变形（W^{UK} 吸进 W^Q，W^{UV} 吸进 W^O）——Q4 直接依赖；结论：推理时不显式重建 K/V。
- 解耦 RoPE 的 q^R、k^R、拼接、$\sqrt{d_h+d_h^R}$ 公式 Eq.(14)–(18)——Q2 直接依赖；结论：RoPE 单独拆出，content 部分仍可吸收。
- RoPE 与矩阵吸收冲突的原因——Q2 直接依赖；结论：RoPE 矩阵位置敏感，破坏 W^{UK} 吸收。
- K3 NoPE 改动与动机——Q5 直接依赖；结论：MLA 层不施加 RoPE，免去扩展上下文时的频率调整。
- K3 output gate 公式与满秩 W^g——Q5 直接依赖；结论：token 级通道调制。
- K3 config.json 实际数值（q_lora_rank 等）——Q3、Q5 数字依据。

### 辅助内容（消除关键理解障碍）

- GQA/MQA 的共享头思路——Q3 对照表的背景，避免读者把 MLA 误认为 GQA 极端形式。
- DeepSeek-V2 的 d_c = 4 d_h、d_h^R = d_h/2 选择——让读者理解 MLA cache 量的具体数值不是任意设定的。
- "推理时只算 c_t^{KV}" 的端到端流程图示——Q4 的辅助可视化。
- 解耦 key k_t^R 也需缓存——澄清"为什么 MLA cache 是 (d_c + d_h^R) 而不是 d_c"。

### 扩展内容

- 纳入：DeepSeek-V2 报告 93.3% KV cache 减少的来源说明（vs DeepSeek 67B 而非 vs 同配置 MHA）——避免常见误解。
- 排除：FlashAttention 的 FP32 output tile 工程实现（K3 §2.1.2 末段）——属 GPU 工程页。
- 排除：K3 hybrid 架构中 MLA 与 KDA 的层调度——属 K3 架构页。
- 排除：MLA 在训练时的算力开销分析——本文聚焦推理效率，训练算力只在文末一句话提及"训练时不缓存、每步前向都重建"。

## 2.4 前置知识映射

| 前置概念 | 被哪些目标依赖 | 概念页状态 | 递归层级 |
|---|---|---|---|
| mqa-gqa（Multi-Query / Grouped-Query Attention） | Q3（对照表） | 未生成，占位提示 | 第 1 层登记不生成 |
| low-rank-projection（低秩投影 / 矩阵分解） | Q1（c_t = W h 的低秩含义）、Q4（W^{UK} 吸收） | 未生成，占位提示 | 第 1 层登记不生成 |
| rope（旋转位置编码） | Q2（RoPE 是位置敏感旋转矩阵） | 未生成，占位提示 | 第 1 层登记不生成 |

注：根据本次任务指令，前置概念 mqa-gqa、low-rank-projection、rope 未生成则标占位，不递归生成。正文首次用到时给出占位链接 + 最小事实陈述（一句话级别），不内联展开前置概念本身。

## 2.5 明确不展开的内容

- GQA/MQA 的内部机制（router 共享、复制策略）：与 MLA 的对照只需 cache 公式即可，GQA/MQA 内部机制属另一概念页。
- Linear Attention 家族的核分解推导：MLA 不属于该族，对照中只需一句话。
- 训练阶段的算力与通信：MLA 的训练前向不缓存、每步重建，但训练算力分析超出本文。
- K3 报告中 KDA 与 MLA 的层调度（attn_res_block_size=12、kda_layers 与 full_attn_layers 列表）：与 MLA 机制本身无关，属 K3 架构页。
- FP32 attention output 与 FlashAttention 改造（K3 §2.1.2 末段）：GPU kernel 工程，超出本文。

## 2.6 常见误解和适用边界

### 误解 M1

- 错误理解：MLA 就是 GQA 的极端形式（头数减到 1）。
- 正确结论：GQA 通过让多个 query 头共享同一组 K/V 头来减 cache，每个被缓存的 K/V 仍是完整 head 维度 d_h；MLA 不共享头，而是把所有头的 K/V 压成一个 d_c 维潜向量，再由学习到的上投影为每个头重建 K/V。机制完全不同。GQA 的极端（n_g=1）即 MQA，cache = 2 d_h l；MLA cache = (d_c + d_h^R) l，DeepSeek-V2 配置下为 4.5 d_h l，比 MQA 还小（因为 d_c=4 d_h < 2 d_h × ... 比较时要看 d_h^R）——但更重要的是 MLA 性能优于 MHA，而 MQA 性能弱。
- 形成原因：两者目标相同（减 KV cache），容易被归为一类。
- 影响目标：Q3。

### 误解 M2

- 错误理解：MLA 缓存的是"压缩后的 K 和 V"。
- 正确结论：缓存的是潜向量 c_t^{KV}（d_c 维），K 和 V 在推理时不显式存在——它们由 c_t^{KV} 经 W^{UK}/W^{UV} 重建，且通过矩阵吸收连重建都可省略。把 c_t^{KV} 称为"压缩 K/V"会误以为 K/V 信息无损保留，实际上低秩压缩是有损的，重建出的 K/V 与原始 MHA 的 K/V 不等价。
- 形成原因：从"压缩 KV cache"的字面意思推断。
- 影响目标：Q1、Q4。

### 误解 M3

- 错误理解：MLA 牺牲了全局注意力（只看局部或稀疏 token）。
- 正确结论：MLA 的注意力公式 Eq.(18) 与 MHA 相同——query 对所有前序 token 求 softmax 加权和，是全局 token-to-token attention。压缩的是存储（cache），不是注意力结构。MLA 与稀疏注意力、滑窗注意力、线性注意力都不同。
- 形成原因：把"压缩"误读为"少看一些 token"。
- 影响目标：Q1。

### 误解 M4

- 错误理解：可以直接对压缩潜向量 c_t^{KV} 应用 RoPE，省掉解耦。
- 正确结论：RoPE 是位置敏感的旋转矩阵，若作用在 k_t^C = W^{UK} c_t^{KV} 上，会在 q^T (R · W^{UK} c_t^{KV}) 中插入 R，破坏 W^{UK} 吸进 W^Q 的可能性（吸收要求 W^{UK} 不依赖位置）。这正是 DeepSeek-V2 §2.1.3 引入解耦 RoPE 的原因。K3 则更进一步直接对 MLA 层用 NoPE，省掉解耦分支。
- 形成原因：未注意到 RoPE 矩阵与权重矩阵不可交换。
- 影响目标：Q2、Q5。

### 适用边界

- MLA 解决：自回归推理时 KV cache 随上下文长度线性增长、随头数线性增长的问题；保留全局 softmax 注意力质量。
- MLA 不解决：注意力本身的 $O(N^2)$ 算力——算力仍是 N(N+1)/2 次 q·k 内积，只是存储被压缩。也不解决训练时的激活内存。
- 成立条件：低秩压缩维度 d_c 远小于 d_h n_h 才有意义（DeepSeek-V2：512 vs 128×128=16384，比值 1/32）；且 W^{UK}/W^{UV} 可被吸收才获得推理效率。
- 不满足时：若 d_c 接近 d_h n_h，压缩比失效；若用了 RoPE 作用在 k_t^C 上，矩阵吸收失效，推理时必须显式重建 K/V，效率大幅退化。
