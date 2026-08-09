# MXFP4 量化感知训练独立审查（第二轮）

- 审查者：独立上下文（AI 模拟 / 小白读者视角）
- 页面版本：index.html @ ac5b744（2026-08-09）
- 时间：2026-08-09
- 审查范围：段 A 盲读（index.html + overview.html）+ 段 B 对照来源（K3 报告 §4.1.4 "Deployment-Aware Post-Training" 行 922–955 + HuggingFace moonshotai/Kimi-K3 config.json quantization_config 字段，WebFetch 于 2026-08-09 获取；OCP Microscaling Formats v1.0）

## 段 A 盲读

按页面顺序阅读 index.html，扮演完全小白读者，记录理解主线上的卡点。

**S1（把 896 个 MoE 专家压到 4-bit 能省多少显存）**：MoE 结构（router、专家、稀疏激活）用前置概念链接引用并给最小结论。稀疏激活只省计算不省显存——明确。每专家参数量手算（3×3584×3072≈33.03M），总专家参数量、BF16/MXFP4 显存估算逐步展开。折叠块含完整手算。config.json ignore 列表正则匹配逻辑解释清晰（专家路径含 .experts. 段故不匹配 dense FFN 正则）。小白可跟上。

**S2（MXFP4 是怎么编码一个权重的）**：从"每个值独立 4-bit 不够用"切入。块结构（32 元素共享 E8M0 scale）用 ASCII 图展示。E2M1 编码（1 符号 + 2 指数 + 1 尾数，bias=1）正数可表示值 8 个 {0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0} 列出。反量化公式 x̂_i = s_b·FP4(q_i) 清晰。4 元素教学块（block1 无损、block2 有误差）手算逐步展开。有效位宽 4+8/32=4.25 bit、压缩 16/4.25≈3.76× 交叉验证。小白可跟上。

**S3（QAT 在前向和反向做了什么，与 PTQ 差在哪）**：PTQ vs QAT 对照清晰。fake-quant 前向（用 ŵ 算损失）和 STE 反向（∂ŵ/∂w≡1）逐步讲解。单权重 w=0.80 手算完整。折叠块含可运行 Python 代码，有预期输出。两个误解点破（"QAT 恢复精度"错、"PTQ 和 QAT 差别不大"错）。STE"透明窗户"类比标注边界。小白可跟上。

**S4（K3 怎么把 QAT 贯穿 SFT 和 RL）**：RL 最小框架（rollout 产出训练 target）标注为教学框架。train-inference mismatch 解释清晰。K3 解法（rollout 与训练共享同一 MXFP4+MXFP8 方案）用循环图展示。draft model 一致性提及。误解再点破。小白可跟上。

**S5（K3 量化了哪些组件、不量化哪些）**：选择性量化表完整（8 类组件），每类有动机说明。config.json quantization_config 字段折叠块节选展示。两个误解点破（"MXFP4 就是 4-bit 整数"错、"应该全量化"错）。小白可跟上。

**学习目标核对**：
1. 为什么专家权重是量化首选，MXFP4 能省多少 → S1 完整回答 ✓
2. MXFP4 怎么编码一个权重，为什么共享 scale 比独立 4-bit 强 → S2 完整回答 ✓
3. QAT 前向反向做了什么，与 PTQ 差在哪 → S3 完整回答 ✓
4. K3 为什么贯穿 SFT 和 RL，RL 中 rollout 与训练共享方案 → S4 完整回答 ✓
5. K3 量化了哪些、不量化哪些，为什么 → S5 完整回答 ✓

段 A 未发现阻断或重要卡点。

## 段 B 对照来源

逐条核对页面表述与 K3 报告 §4.1.4（行 922–955）及 config.json quantization_config 字段的一致性。

**定义与机制**：
- C1（MoE 专家权重 MXFP4、激活 MXFP8、QAT 贯穿 SFT+RL）：报告 §4.1.4 "we quantize the MoE expert weights — which dominate the model's parameter memory — to MXFP4, with activations computed in MXFP8" + "we perform quantization-aware training (QAT) throughout the entire post-training stage, covering both SFT and RL" ✓
- C2（rollout 与训练共享同一量化方案、消除 mismatch）：报告 §4.1.4 "During RL, rollout and training share the same quantization scheme — eliminating the train–inference mismatch" ✓
- C3（非专家组件保持高精度）：报告 §4.1.4 "all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision" ✓
- C4（draft model 沿用 QAT 配置）：报告 §4.1.4 "Draft fine-tuning follows the post-training QAT configuration (§ 4.1.4), with MoE expert weights in MXFP4 and their input activations in MXFP8" ✓

**config.json 逐项核对**（WebFetch 获取 text_config.quantization_config）：
- format: "mxfp4-pack-quantized" → 页面 ✓
- group_size: 32 → 页面 ✓
- num_bits: 4 → 页面 ✓
- scale_dtype: "torch.uint8" → 页面 ✓
- strategy: "group" → 页面 ✓
- symmetric: true → 页面 ✓
- type: "float" → 页面 ✓
- ignore 列表 6 条正则 → 页面节选与实际一致 ✓
- num_experts: 896 → 页面 ✓
- num_experts_per_token: 16 → 页面 ✓
- num_shared_experts: 2 → 页面 ✓
- hidden_size: 7168 → 页面 ✓
- routed_expert_hidden_size: 3584 → 页面 ✓
- moe_intermediate_size: 3072 → 页面 ✓
- num_hidden_layers: 93 → 页面 ✓
- first_k_dense_replace: 1 → 页面 ✓

**公式与推导**：
- F1（反量化 x̂_i = s_b·FP4(q_i)，s_b=2^{e_b}）：OCP MX v1.0 spec ✓
- F2（STE ∂ŵ/∂w≡1）：Jacob et al. 2018 ✓
- F3（有效位宽 4+8/32=4.25 bit，压缩 16/4.25≈3.76×）：已用总字节数交叉验证 ✓
- E2M1 正数可表示值 {0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0}：按 E2M1 编码（1 符号 + 2 指数 bias=1 + 1 尾数）逐值复算一致 ✓
- 块 1/块 2 手算：逐项复算（归一化 → 最近邻舍入 → 反量化 → 误差）一致 ✓
- 单权重 QAT 一步（w=0.80, s_b=0.25 → ŵ=0.75, 偏差 0.05, STE 梯度=1.0）：复算一致 ✓

**可运行代码**：从页面提取 Python 代码（折叠块"可运行代码：MXFP4 量化模拟与 QAT 一步"）实际执行，输出与页面预期完全一致：
- block1 误差: [0.0, 0.0, 0.0, 0.0] ✓
- block2 误差: [0.05, -0.025, -0.05, 0.0125] ✓
- w=0.8, w_hat=0.75, 前向偏差=0.0500 ✓
- STE 传回 w 的梯度=1.0 ✓

**专家参数量核对**（独立复算）：
- per_expert = 3 × 3584 × 3072 = 33,030,144 (33.03M) ✓
- total = 896 × 92 × 33,030,144 = 2,722,740,830,208 (2.7227T)
- BF16 = 5.4455 TB
- MXFP4 = 1.4465 TB
- 压缩 = 3.7647×
- 页面声称 total ≈ 2.71T、BF16 ≈ 5.42 TB、MXFP4 元素 ≈ 1.355 TB——见下方轻微问题

**事实与推断**：
- N1–N2（896 专家、每 token 激活 16、2 共享专家、hidden_size 等）：config.json 逐项一致 ✓
- N3（每专家 33.03M、总 2.71T、BF16 5.4 TB、MXFP4 1.44 TB、压缩 3.76×）：数字为教学估算，标注"数量级估算，忽略对齐 padding 与 latent MoE 内部结构" ✓（但总参数量有算术偏差，见下方问题）
- N4（块大小 32）：config.json group_size=32 ✓
- N6（激活 MXFP8 E4M3）：报告 §4.1.4 "activations computed in MXFP8" ✓

**前置知识引用**：MoE 推理与服务基础、quantization-basics 均用占位链接或标注"尚未生成"并给最小结论 ✓

**教学简化**：块大小教学缩写为 4（真实 32）标注 ✓；scale 教学给定而非推导标注 ✓；代码用 Python 列表与 min 模拟标注 ✓；N3 估算忽略 padding 标注 ✓

**页面功能**：validate.py 退出码 0 ✓

## 问题

- [轻微·技术] S1 正文 + S1 折叠块手算 + N3 教学说明：专家总参数量声称"≈ 2.71T"，实际 896×92×33,030,144 = 2,722,740,830,208 ≈ 2.72T（偏差 0.01T，约 0.37%）。该偏差传播至 BF16 估算（页面 5.42 TB，实际 5.44 TB）和 MXFP4 元素估算（页面 1.355 TB，实际 1.36 TB），但最终四舍五入值（5.4 TB、1.44 TB、3.76×）仍近似正确，不影响"专家 dominate 模型参数"和"压缩约 3.76×"等结论：将"≈ 2.71T"改为"≈ 2.72T"，同步修正折叠块中"2.71T × 2 ≈ 5.42 TB"为"2.72T × 2 ≈ 5.44 TB"、"2.71T × 0.5 ≈ 1.355 TB"为"2.72T × 0.5 ≈ 1.36 TB"、"84.7 GB"为"85.1 GB" ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（轻微问题接受：0.37% 的算术偏差不影响任何结论，最终四舍五入值 5.4 TB / 1.44 TB / 3.76× 仍正确；修复仅需替换数字不改结构）

段 A 盲读未发现阻断或重要卡点，学习目标全部由正文章节完整回答。段 B 对照来源逐条核对，核心论断（C1–C4）、config.json 16 项字段、公式（F1–F3、E2M1 可表示值、块手算、QAT 一步）均一致。可运行代码已重跑，输出与页面描述完全一致。validate.py 退出码 0。关键论断和数字已重新对照外部来源。唯一偏差为专家总参数量 2.71T 应为 2.72T（0.37%），属轻微。
