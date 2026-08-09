# 辅助损失无关路由独立审查（第二次）

- 审查者：独立上下文（AI 模拟小白读者 + 来源对照）
- 页面版本：wiki/aux-loss-free-routing/index.html（1204 行）、overview.html（82 行）
- 时间：2026-08-09
- 来源：[原始论文] arXiv:2408.15664（Wang et al., 2024, Auxiliary-Loss-Free Load Balancing）；[V3] arXiv:2412.19437（DeepSeek-V3 Technical Report §2.2）

## 段 A 盲读

按页面顺序阅读，记录小白读者理解主线上的卡点。

1. 开篇 callout 用"4 个专家选 top-2"的直觉场景引入问题，语言通俗，小白可跟进。

2. S1 介绍路由坍塌与辅助损失方案。$L_{bal}$ 公式中 $f_i$（选中频率）与 $P_i$（平均分数）首次出现时有定义。"干扰梯度（interference gradients）"术语有括注解释"混进来的梯度"。可理解。

3. S1 引入"MaxVio（最大负载偏差）"作为实验对照指标，但未说明 MaxVio 的计算方式——小白不知道它衡量什么、值域多大、越小越好还是越大越好。只能从"均劣于本方法"的上下文间接推断。轻微卡点。

4. S2 公式 F1-F3 清晰。"选择依据是 $s_{i,t}+b_i$""门控权重取原始 $s_{i,t}$"两点用编号列表强调。两个边界检查（整体平移不变、bias 不进 mixture weight）帮助巩固理解。

5. S3 sign 更新规则清晰。因果约束（用历史 batch 更新）有解释。sign vs 幅度对比有 Table 3 数据支撑。

6. S4 手算例子完整：首轮路由 → 负载统计 → bias 更新 → 第二轮路由。逐步推进，每步有表。$\gamma=0.05$ 的对照折叠块进一步说明 sign 的步长特性。代码验证通过（详见段 B）。

7. S5 DeepSeek-V3 配置落地。$\gamma$ 调度、互补损失、推理吸收、K3 边界逐一覆盖。

8. 逐题核对学习目标：
   - "MoE 训练为什么需要负载均衡？传统辅助损失方案为什么有缺陷？" → S1 ✓
   - "bias 在前向传播里做什么、不做什么？为什么不会引入干扰梯度？" → S2 ✓
   - "bias 在训练时怎么被更新？为什么用 sign？" → S3 ✓
   - "给定 4 专家 top-2 的打分与初始 bias，手算一轮路由、统计负载、更新 bias、再跑一轮路由" → S4 ✓
   - "DeepSeek-V3 中本方法的实际配置是什么？解决什么、不解决什么？" → S5 ✓

   全部学习目标由正文章节完整回答。

## 段 B 对照来源

### 1. 定义与机制

- C1（路由坍塌）：[原始论文] §1 确认。"routing collapse (Shazeer et al., 2017), where the model consistently selects only a few experts, hindering sufficient training of the other experts"。一致 ✓
- C2（辅助损失两难）：[原始论文] §1 确认。"a large auxiliary loss will introduce non-negligible interference gradients into training and thus impair the model performance"；"a small α causes routing collapse"。一致 ✓
- C3（bias 只做选择不进权重）：[V3] Eq.16 确认。"the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score $s_{i,t}$"。一致 ✓
- C4（sign 规则式更新）：[原始论文] Algorithm 1 确认。"Update $b_i$ by $b_i = b_i + u \cdot \text{sign}(e_i)$"。一致 ✓
- C5（sign 与变体对比）：[原始论文] Table 3 确认。sign 版 PPL=9.50, MaxVio=0.044；幅度版 PPL=9.51-9.53, MaxVio=0.028-0.040。页面表格数据逐行匹配 ✓
- C6（互补 sequence-wise loss）：[V3] Eq.17-20 确认。一致 ✓

### 2. 公式与推导

- F1（$s_{i,t}=\sigma(u_t^\top e_i)$）：[V3] 确认用 Sigmoid。页面来源标注"DeepSeek-V3 报告 Eq.16"——按论文公式编号，sigmoid 打分公式为 Eq.15，选择规则为 Eq.16（见问题 1）。
- F2（$g'_{i,t}$ 选择规则）：[V3] Eq.16 原文逐字符匹配。$$g'_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}+b_i\in\text{Topk}(\{s_{j,t}+b_j\},K_r)\\0,&\text{otherwise}\end{cases}$$ ✓
- F3（$g_{i,t}$ 归一化）：[V3] §2.1 确认 ✓
- F4（$b_i\leftarrow b_i+\gamma\cdot\text{sign}(\bar c_i-c_i)$）：[原始论文] Algorithm 1 确认。"Calculate the load violation error $e_i = \bar{c}_i - c_i$; Update $b_i$ by $b_i = b_i + u \cdot \text{sign}(e_i)$"。符号 $u$→$\gamma$ 的映射已说明 ✓
- F5（$h_t'$ MoE 层输出）：[V3] Eq.14 确认 ✓
- F6（$L_{Bal}$ sequence-wise loss）：[V3] Eq.17-20 确认。$L_{Bal}=\alpha\sum f_i P_i$（Eq.17），$f_i=\frac{N_r}{K_r T}\sum\mathbb{I}(\cdot)$（Eq.18），$s'_{i,t}=\frac{s_{i,t}}{\sum_j s_{j,t}}$（Eq.19），$P_i=\frac{1}{T}\sum_t s'_{i,t}$（Eq.20）。页面 F6 含全部四式但来源标注"Eq.17-19"，缺 Eq.20（见问题 2）。

### 3. 可运行代码

页面 S3 的伪代码（Algorithm 1 简化形式）标注"伪代码，不是 Python"，不声称可运行。S4 手算例子为静态数值推导，已用 Python 复算验证：

- 首轮路由选择：$t_1,t_2,t_3 \to \{E_3,E_0\}$；$t_4 \to \{E_3,E_1\}$ ✓
- 负载 $c=(3,1,0,4)$，均值 $=2$，偏差 $e=(-1,+1,+2,-2)$ ✓
- 新 bias $=(-0.001,+0.001,+0.001,-0.001)$ ✓
- 第二轮路由不变 ✓
- $\gamma=0.05$ 对照：top-2 仍不变 ✓

全部计算与页面描述一致 ✓

### 4. 事实与推断

- N1（DeepSeek-V3 规模 671B/37B, 256+1, top-8, 61 层 3+58）：671B/37B 由 [V3] 摘要确认 ✓；top-8 由 §3.2.2 确认 ✓；256 routed + 1 shared 与 61 层(3+58) 未在论文 HTML 提取内容中找到，但多个二手来源（spawn08, yudonglee）一致确认。
- N2（$\gamma$ 调度 0.001→0, 14.3T+500B=14.8T）：axiomlogica 来源确认"$\gamma = 0.001$ for the first 14.3T training tokens, then sets $\gamma = 0.0$ for the final 500B tokens" ✓
- N3（$\alpha=0.0001$）：yudonglee 来源确认 ✓。[V3] 论文 HTML 提取仅说"extremely small value"，具体数值未在提取内容中出现。
- N4（原始论文 1B/3B, 100B/200B token, $u=0.001$ 最佳）：[原始论文] §4 确认模型规模与训练量。$u=0.001$ 最佳由 §4.3 实验确认（页面标注 §4.1，见问题 3）。Table 2 对照数据（1B: 9.56 vs 9.50, MaxVio 0.72 vs 0.04；3B: 7.97 vs 7.92, MaxVio 0.52 vs 0.04）逐行匹配 ✓
- N5（变体对照）：Table 3 四行数据逐行匹配 ✓
- "推理时 bias 被吸收进 router 权重"：[V3] 论文 HTML 提取未找到此说法，但 yudonglee 来源确认"bias 在训练完成后被吸收到 router 权重,推理无额外开销"。
- "干扰梯度（interference gradients）"术语：[原始论文] 多次使用此术语（摘要、引言、Table 1、结论），页面正确归因到原始论文而非 V3 报告 ✓

### 5. 前置知识引用

- ../moe-serving/index.html — 目录存在 ✓
- ../quantile-balancing/index.html — 目录存在 ✓

### 6. 教学简化

- 专家数设为 4、top-k 设为 2、batch 设为 4 token——已在"教学简化及其限制"中说明 ✓
- F6 只点出公式不展开推导——已说明 ✓
- S5 中 K3 的 Quantile Balancing 只引用对比——已说明 ✓

### 7. 页面功能

- KaTeX 公式渲染：delimiters 配置正确 ✓
- 折叠块（Table 3 对照、伪代码、$\gamma=0.05$ 对照）：收起后正文主线不依赖其内容 ✓
- 目录锚点：h2/h3 带 id ✓

## 问题

- [轻微·来源] 来源与教学说明 > 核心公式 > F1：$s_{i,t}=\sigma(u_t^\top e_i)$ 标注来源为"DeepSeek-V3 报告 Eq.16"。按论文公式编号，sigmoid 打分公式为 Eq.15，bias+top-k 选择规则才是 Eq.16。F1 应标注 Eq.15。同时页面 meta 标注"Eq.16-19"应改为"Eq.15-20"以覆盖 sigmoid（Eq.15）到 $P_i$（Eq.20）。修法：F1 来源改为"Eq.15"；meta 改为"Eq.15-20" ｜ 修复： ｜ 复验：
- [轻微·来源] 来源与教学说明 > 核心公式 > F6：$L_{Bal}$ 标注来源为"DeepSeek-V3 报告 Eq.17-19"，但 F6 实际包含四个公式：$L_{Bal}$（Eq.17）、$f_i$（Eq.18）、$s'_{i,t}$（Eq.19）、$P_i$（Eq.20）。应标注"Eq.17-20"。修法：F6 来源改为"Eq.17-20" ｜ 修复： ｜ 复验：
- [轻微·来源] 来源与教学说明 > 外部数字 > N4："$u=0.001$ 为最佳"标注来源为"arXiv:2408.15664 §4.1"。原始论文中"u=0.001 is best"的结论出自 §4.3（实验对照不同 $u$ 值），§4.1 仅说明"在 1B 规模上调参，3B 直接继承"。修法：N4 来源改为"§4.3"或"§4.1, §4.3" ｜ 修复： ｜ 复验：
- [轻微·盲读] S1 最后一段："MaxVio（最大负载偏差）均劣于本方法"——MaxVio 作为实验指标首次出现，未说明其计算方式与含义（如：单个专家负载与平均值的最大偏差占比，越小越好）。小白读者只能从上下文间接推断。修法：首次出现时加括注"（单个专家负载偏离平均的最大比例，越小越均衡）" ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：进入修复（4 条轻微均为来源标注精确性与首现术语解释，改动量小，不影响核心结论与主线理解）
