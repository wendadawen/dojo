# Gated DeltaNet 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 段B 对照源核验）
- 页面版本：index.html `c29e494e291e4f17fdb66ff9793a560e707e86ca` / overview.html `ff0de131aae1964fa87cdf26b02ce20658f9cc44`
- 时间：2026-08-09

## 审查来源

- 论文：Yang, Kautz & Hatamizadeh. "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025, arXiv:2412.06464（WebSearch + WebFetch arxiv PDF/cnblogs 逐字提取 Table 1/2/3）
- K3 报告：/tmp/kimi-k3-research/k3-report.txt §2.1.1（KDA 公式与参数化）、§5.1（KDA 系统）、Table（69 KDA + 24 MLA）

## 段A 盲读核查

按页面顺序通读，记录理解主线上的卡点。

§1 互补性论证：DeltaNet 公式 F4 与 Mamba2 公式 F3 并排，"方向擦除 vs 标量全局衰减"对比清晰，前置链接到 delta-rule 页。小白可跟上。三模型对比表 highlight-row 标注 Gated DeltaNet 行，结构清楚。

§2 公式与符号：F1 公式 boxed 展示，符号逐条定义，形状检查（$(d_v,d_k)$ 链路）完整。"先衰减后擦写"等价推导基于 α_t 标量可交换，前提条件在 §2 末尾与 §5 前置说明均点明。盲读无卡点。

§3 手算：d=2 / 3 步例子，逐步矩阵代入。复算第 3 步 $S_3 = [[0,0],[1,0.5]]$ 正确。三模型对比表 + details 折叠的"先衰减后擦写"分步验证一致。代码块可运行（numpy 标量循环），预期输出与手算一致。盲读无卡点。

§4 退化与并行：α→1 退化为 DeltaNet、β→0 退化为 $\alpha_t S_{t-1}$（Mamba2 无写入特殊情形）、α→0 清空。chunkwise 算法两层（chunk 内 WY 并行 / chunk 间递归），α_t 逐元素乘法不破坏 Tensor Core 兼容性。盲读无卡点。

§5 KDA 与实验：α_t 标量 vs channel-wise、sigmoid vs scaled sigmoid、full-rank output gate 三项差异列表。实验数字表 + S-NIAH 检索表。盲读卡点见问题 2、3。

学习目标闭环核查（逐题）：
- Q1（DeltaNet 缺陷 + 为何加 α_t）：§1 答完 ✓
- Q2（手算一步 + α/β 分工）：§2-3 答完 ✓
- Q3（α→1 / β→0 退化）：§4 答完 ✓
- Q4（并行训练问题 + 为何不能串行）：§4 答完 ✓
- Q5（Gated DeltaNet vs KDA α_t 区别）：§5 答完 ✓

## 段B 对照来源核查

### 已逐条核对通过的项

1. **核心公式 F1**：论文 Table 1（Gated DeltaNet 行）逐字为 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T$，与页面 F1 完全一致 ✓
2. **F3 Mamba2 公式**：论文 Table 1 逐字为 $S_t = \alpha_t S_{t-1} + v_t k_t^T$ ✓
3. **F4 DeltaNet 公式**：论文 Table 1 逐字为 $S_t = S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$ ✓
4. **互补性引文 C7**：arxiv 摘要逐字 "gating enables rapid memory erasure while the delta rule facilitates targeted updates" ✓
5. **KDA 公式**：K3 报告 §2.1.1 Eq.1 逐字为 $S_t = (I - \beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t)S_{t-1} + \beta_t k_t v_t^\top$，与页面 §5 一致 ✓
6. **KDA α_t channel-wise**：K3 报告 line 238 "αt ∈ (0, 1)^{dk} is the channel-wise one-step retention factor" ✓
7. **KDA lower-bounded decay**：K3 报告 line 312 "gmin = −5"、line 340 "αt,j > e^{−5} ≈ 6.7 × 10^{−3}"，与页面 §5 数字一致 ✓
8. **KDA scaled sigmoid**：K3 报告 line 329 "Kimi K3 instead uses a scaled sigmoid to bound the log-decay" ✓
9. **KDA full-rank output gate**：K3 报告 line 347-350 "changes KDA's output gate from the low-rank parameterization... to an input-dependent full-rank projection" ✓
10. **K3 层数 69 KDA + 24 MLA = 93**：K3 报告 line 757 "69 KDA + 24 MLA" ✓
11. **实验数字 N1（PPL + 常识推理）**：cnblogs 逐字提取论文 Table 3，Transformer++ 18.53/18.32/52.25、Mamba2 16.56/12.56/54.89、DeltaNet 17.71/16.88/52.14、Gated DeltaNet 16.42/12.17/55.32、H1 16.07/12.12/56.40、H2 15.91/12.55/56.18，全部与页面表一致 ✓
12. **实验数字 N2（S-NIAH 4K）**：arxiv PDF Table 2 逐字为 DeltaNet S-NIAH-2 4K=18.6 / S-NIAH-3 4K=22.4、Mamba2 56.2 / 4.6、Gated DeltaNet 92.2 / 27.6，全部与页面表一致 ✓
13. **手算复算**：第 3 步 $S_3 = [[0,0],[1,0.5]]$、DeltaNet $[[0,0],[1,1]]$、Mamba2 特殊 $[[0.5,0],[0,0.5]]$，逐矩阵乘法验证通过 ✓
14. **代码复算**：gated_delta_step / delta_step / mamba2_step 逐行 trace，退化验证 np.allclose 结果 True，预期输出与页面一致 ✓
15. **WY 表示**：ArxivLens 与论文摘要确认 "WY representation" 用于并行训练 ✓
16. **作者与会议**：Songlin Yang, Jan Kautz, Ali Hatamizadeh；ICLR 2025 camera ready ✓
17. **NVlabs/GatedDeltaNet 开源**：cnblogs 确认 https://github.com/NVlabs/GatedDeltaNet ✓

### 发现的问题

- [重要·技术] index.html §5 line 1091 / line 1104 + overview.html line 56：S-NIAH-2 被描述为"真实文本单针检索"/"真实文本检索"，但论文 Table 2 标题逐字为 "S-NIAH-2(number in haystack)"（数字针检索），S-NIAH-3 为 "uuid in haystack"。三者 needle 类型分别是 pass-key / number / uuid，均非"真实文本"。页面据此推断"DeltaNet 在真实文本 S-NIAH-2/3 上性能显著下降"会误导读者高估结果对真实文本检索的适用性。数字本身（18.6 / 92.2）正确，问题在 benchmark 类型误述。修法：将"真实文本单针检索"改为"数字针单针检索"或"number-in-haystack 检索"；将"真实文本 S-NIAH-2/3"改为"干扰更强的 S-NIAH-2/3（number/uuid 针）"；overview.html 同步修改。 ｜ 修复：已将 index.html line 684 "S-NIAH 真实文本检索"改为"S-NIAH-2 数字针检索"；line 1091 "真实文本单针检索"改为"number-in-haystack 数字针单针检索"；line 1097 "真实文本检索上显著退化"改为"数字针检索上显著退化"；line 1104 "真实文本 S-NIAH-2/3"改为"干扰更强的 S-NIAH-2/3（number/uuid 针，非真实文本）"；overview.html line 56、70 同步改为"number-in-haystack 检索"/"数字针检索"。 ｜ 复验：validate.py 通过

- [轻微·盲读] index.html §5 line 1070：Gated DeltaNet 公式用 $v_t k_t^\top$（$S \in \mathbb{R}^{d_v \times d_k}$，$S_{t-1}$ 在左侧），KDA 公式用 $k_t v_t^\top$（$S \in \mathbb{R}^{d_k \times d_v}$，$S_{t-1}$ 在右侧）。两者状态约定转置，页面说"两者形式同源"但未说明约定差异，小白对比两公式时可能困惑 $v_t k_t^\top$ 与 $k_t v_t^\top$ 为何方向相反。修法：在 §5 公式对比处加一句"KDA 采用转置状态约定（$S \in \mathbb{R}^{d_k \times d_v}$），$k_t v_t^\top$ 对应 Gated DeltaNet 的 $v_t k_t^\top$"。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html §5 line 1060 对比表：decay 参数化行出现 "g ∈ (g_min, 0)"，但 g（log-decay）首次出现未定义，$\alpha = e^g$ 关系要到 line 1068 才隐含给出。小白在表格处不知道 g 是什么。修法：在表格该格首次出现 g 处加括注"g 为 log-decay，$\alpha_t = e^{g}$"，或在表格前一句话定义 g。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html §3 line 852-863 对比表：第三行标签"Mamba2"实际指 $\beta=0$ 无写入的特殊情形（非完整 Mamba2，完整 Mamba2 有 $+v_t k_t^\top$ 写入项）。line 852 文字有限定"Mamba2（β = 0，只衰减不写入）"，教学说明 line 1171 也披露了，但表格标签"Mamba2"单独出现时可能误导略读者。修法：将表格该行标签改为"Mamba2（β=0 特殊）"或"退化：β=0"。 ｜ 修复： ｜ 复验：

- [轻微·技术] index.html §5 line 664 / line 1111："Qwen3-Next 等用 Gated DeltaNet 作线性层"无来源标注，且未从给定来源（K3 报告 + WebSearch 论文相关结果）中验证。K3 报告只涉及 KDA，WebSearch 5 条结果均未提及 Qwen3-Next 采用 GDN。修法：补充来源链接或标注"待核实"；若无法定位来源则改为泛述"多个开源混合架构模型采用 Gated DeltaNet 作线性层"。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：进入修复

重要问题为 S-NIAH-2 benchmark 类型误述（"真实文本"→ 实为 number-in-haystack），影响读者对实验结果适用范围的理解，但不影响核心公式、手算、KDA 对比与互补性论证的正确性。四个轻微问题分别涉及公式约定说明、符号首现定义、表格标签精度、无来源应用声明，均不阻断主线理解。

所有学习目标（Q1-Q5）由正文章节完整回答，无遗漏。核心公式 F1/F3/F4 与论文 Table 1 逐字一致。实验数字 N1/N2 与论文 Table 2/3 逐字一致。KDA 三项差异与 K3 报告 §2.1.1 一致。手算与代码复算通过。
