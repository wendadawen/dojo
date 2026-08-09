# Kimi Delta Attention（KDA）独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源核查）
- 页面版本：index.html 629811a / overview.html 66e2d2f
- 时间：2026-08-09

## 审查范围与方法

段 A 盲读：按页面顺序（index.html → overview.html）以小白视角阅读，记录主线卡点，结束时逐题核对学习目标。

段 B 对照来源：逐条核对页面表述与 K3 技术报告 §2.1.1（Eq.1–6, Fig.3）及 HuggingFace 官方 config.json 一致性。复算全部教学示例。

### 来源核查摘要

- Eq.1–6 六个公式逐条对照 K3 报告原文：**全部一致**（含 Eq.1 乘法顺序 Diag(α_t) 在 erase 项右侧、Eq.4 的 V_e = U − WS、Eq.5 的 Sigmoid(e^{A_h} z) 形式）。
- config.json 数值逐项核对：69 KDA 层（kda_layers 数组 69 项）、24 Gated MLA 层（full_attn_layers 数组 24 项）、gate_lower_bound = −5.0、head_dim = 128、num_heads = 96、short_conv_kernel_size = 4、use_full_rank_gate = true、hidden_size = 7168、max_position_embeddings = 1048576、num_hidden_layers = 93 —— **全部一致**。
- 教学示例复算：S1 KV cache（1,048,576 × 96 × 128 × 2 × 2 ≈ 5.15×10¹⁰ bytes ≈ 48 GB ✓）、S2 四维两步（S₁ = all-ones 4×4 → Diag(α₂) 衰减 → erase 第 1 行 → 写入 (3,3,3,3) → S₂ ✓）、S3 z=1 对比（Kimi Linear α ≈ 0.269；K3 α ≈ 0.0259 ✓）、S5 Γ 手算（Γ₁→₃ = (0.35, 0.48, 0.855, 0.12) ✓）、BF16 算账（e⁸⁰ ≈ 5.54×10³⁴ < 3.4×10³⁸ ✓）—— **全部复算通过**。
- scaled sigmoid vs negative-softplus 区别：**讲清**。两种映射的公式、g 范围、α 范围、1/Γ 上界、对角 tile 实现方式（position-pair vs Tensor Core）、代价（牺牲彻底遗忘）均有对照表和手算示例。
- 前置页链接：delta-rule、linear-attention 两页均存在（Glob 确认），层级正确（../../wiki/<name>/index.html）。index.html 与 overview.html 互相链接，均含返回首页链接。
- 学习目标 5 条逐题核对：全部由正文章节完整回答（S1→目标1，S2→目标2，S3→目标3，S4+S5→目标4，S6→目标5）。

## 问题

- [重要·技术] index.html S4「参数化与 full-rank output gate」ShortConv 文字描述（约 L937）：文字写"在投影前做 kernel=4 的短卷积"，但同页 Eq.2 公式 `ShortConv(W_{q/k}^h x_t)` 和同页数据流图（W → ShortConv → Swish → L2Norm → q/k）均表明 ShortConv 在线性投影 W **之后**。K3 报告 Eq.2 原文同样是 `L2Norm(Swish(ShortConv(W x_t)))`（投影在内层）。文字与公式、图、来源三方矛盾，读者按文字理解会得到相反的参数化顺序。修法：将"在投影前做"改为"在投影后做"或"对投影后的特征做 kernel=4 的短卷积"。 ｜ 修复：已将"在投影前做 kernel=4 的短卷积"改为"对投影后的特征做 kernel=4 的短卷积"，与 Eq.2 公式 `ShortConv(W x_t)`、数据流图（W → ShortConv → Swish → L2Norm → q/k）及 K3 报告原文一致。 ｜ 复验：
- [轻微·盲读] index.html S5「累积衰减 Γ」教学示例（约 L980）：正文写"把 S2 的 3 步迷你序列当作一个 C=3 的 chunk"，但 S2（KDA 递归核心章）的示例为 2 步（step 1: k₁,v₁,β₁=1 无 α；step 2: k₂,v₂,α₂），不含第 3 步，α₃=(0.7,0.6,0.95,0.4) 为此处新增。同页教学说明（约 L1141）写"S5 的 C=3 chunk"，与正文"S2 的"内部不一致。小白读者会困惑"第 3 步从哪来"。修法：删去"把 S2 的"，改为"构造一个 C=3 的 3 步迷你序列（α₁, α₂, α₃ 如下）"。 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html S4 ShortConv 归属（约 L937）："这是 K3 相对原始 DeltaNet 的小改动"——K3 报告 §2.1.1 称参数化（含 ShortConv）"Following Kimi Linear [63]"，ShortConv 源自 Kimi Linear 而非 K3 的改动。页面 context box 也只列"两处改动"（scaled sigmoid + full-rank gate），此处将 ShortConv 归为"K3 改动"与之不一致。修法：改为"继承自 Kimi Linear 的 ShortConv（相对原始 DeltaNet 的改动）"或"沿用 Kimi Linear [63] 的 ShortConv"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html S3「lower-bounded decay」A_h 提前引用（约 L810）：公式 `z_t^h = W_α^{↑↓} x_t + b_h^α ∈ R^{d_k}` 后紧跟"其中 A_h 是 per-head 可学习 log-scale、b_h^α 是 per-head bias"，但 A_h 不在此公式中（A_h 出现在 Eq.5 的 `Sigmoid(e^{A_h} z)`）。"其中"引导的符号解释让读者在 z_t^h 公式中找不到 A_h。修法：将"A_h 是 per-head 可学习 log-scale"移至 Eq.5 的符号说明处（约 L831 已有 A_h 说明，可合并），或在此处加注"A_h 在下步映射（Eq.5）中使用"。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：进入修复

### 说明

1. **无阻断问题**。六个核心公式（Eq.1–6）全部与 K3 报告一致，config 数值全部核对通过，所有教学示例复算正确，学习目标 5 条全部由正文回答。scaled sigmoid vs negative-softplus 的区别（有下界 vs 无下界、Tensor Core 可用 vs position-pair、牺牲彻底遗忘的代价）讲清。
2. **1 个重要问题**（ShortConv 顺序文字与公式矛盾）需修复后复验。该问题不影响公式本身的正确性（Eq.2 公式和图均正确），但文字描述与来源和同页图直接矛盾，会造成读者对参数化顺序的误解。
3. **3 个轻微问题**可一并修复：S5 交叉引用错误（S2→应为 S5 新构造）、ShortConv 归属措辞、A_h 提前引用。
4. 页面功能静态检查：KaTeX 公式渲染脚本已加载，details 折叠元素结构正常，heading ID 齐全（TOC 由 JS 自动生成），index.html 与 overview.html 互链正常。未运行 validate.py（属发布门控步骤，非审查步骤）。
