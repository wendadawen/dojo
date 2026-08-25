# OPD（On-Policy Distillation）— 内容范围

## 1. 概念歧义处理

概念名称：On-Policy Distillation，缩写 OPD，中文译名"在线策略蒸馏"。

状态：已裁定。在大语言模型后训练语境下，OPD 指学生模型自己生成轨迹、教师模型对学生轨迹逐 token 给出反馈的蒸馏方法。该含义在以下来源中一致：

- Agarwal et al. (2023)《On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes》（GKD 论文，ICLR 2024）提出 on-policy KD 作为 GKD 框架的 $\lambda=1$ 特例，这是该术语在 LLM 蒸馏文献中的建立点。
- Thinking Machines Lab (2025)《On-Policy Distillation》将其实现为"逐 token 反向 KL 作 advantage 接入策略梯度"。
- Qwen3 技术报告 (2025) §4.5 在强到弱蒸馏流水线中把 on-policy distillation 作为第二阶段。
- Kimi K3 技术报告 §4.1.3 使用 per-token OPD 奖励（Eq.15）构建 MOPD。

需要区分的相邻名称（正文明确区分，不展开为独立主题）：

- **GKD（Generalized Knowledge Distillation）**：Agarwal et al. 的完整框架，包含数据源混合比例 $\lambda$ 和散度选择两个自由度。on-policy KD 是其 $\lambda=1$ 的实例。日常语境中"OPD"常直接指这个特例及其后续变体。
- **标准/监督蒸馏（supervised KD、SeqKD）**：在固定数据集或教师生成的序列上训练学生，$\lambda=0$，属于 off-policy。
- **MOPD**：多教师扩展，见已有页面 `wiki/mopd/`，本文只链接不展开。

## 2.1 概念含义

- **概念名称**：On-Policy Distillation（OPD，在线策略蒸馏）
- **一句话定义**：训练时由学生模型自己采样生成 token，教师模型在学生实际到达的每个状态上对学生采样的 token 逐个打分（对数概率比），把这个逐 token 的分数作为稠密奖励/监督信号更新学生。
- **正式定义**（以 GKD 论文与 Thinking Machines 博客为基准）：训练数据分布来自学生当前策略 $\pi_S$ 的自采样（on-policy 的含义），训练目标是缩小教师与学生分布在学生访问到的状态上的差异——GKD 论文表述为 $\lambda=1$ 时在 $y\sim p_S(\cdot|x)$ 上最小化逐 token 散度；Thinking Machines 表述为把逐 token 的 $\log(\pi_{\text{teacher}}/\pi_S)$ 作为 advantage 接入策略梯度。
- **本文采用的语境**：大语言模型后训练阶段，把一个（或一组）已训练好的强教师模型的能力转移给学生模型；推理任务为主要例子来源。
- **包括什么**：
  - 分布失配问题：为什么在教师序列上训练、在自己序列上推理会导致复合误差——OPD 的动机。
  - GKD 统一框架：$\lambda$ 数据源混合、supervised KD 与 on-policy KD 是两个端点。
  - per-token 反馈机制：逐 token 对数概率比的计算、符号含义、如何作为 advantage 进入策略梯度更新。
  - 与 RL 的关系：为什么逐 token 稠密信号比序列级稀疏奖励样本效率高、计算便宜。
  - 散度选择（forward KL / reverse KL / JSD）与初始化、batch size 等实现选择。
  - 实测证据：Qwen3 Table 21、Thinking Machines 计算量对比、GKD 实验结论。
  - 能力边界：不教新知识、以教师为参照、覆盖条件，以及与 MOPD 的衔接。
- **不包括什么**：
  - MOPD 的多教师路由与条件化学生（独立概念，见 `wiki/mopd/`）。
  - RLHF/PPO/GRPO 算法内部（本文只消费"策略梯度按 $\nabla\log\pi\cdot A$ 更新"这一结论）。
  - MiniLLM、ImitKD、f-distill 等其他蒸馏变体的内部机制（只在 GKD 框架的特例表中出现名字）。
  - 温度 softmax 与 Hinton 蒸馏的机制细节（已有页面 `wiki/knowledge-distillation/`）。
  - 具体训练基础设施（Tinker cookbook、partial rollout 调度实现）。
- **相邻概念**：
  - **标准知识蒸馏**：教师生成或固定数据集，$\lambda=0$。区别在"谁生成训练轨迹"。纳入范围作为对比主线。
  - **RLVR/RLHF**：奖励模型或可验证奖励给序列级（或稀疏）信号。区别在奖励来源与粒度。纳入范围作为效率对比。
  - **自采样 SFT**：用学生自己生成的序列做普通 SFT。形似 OPD 但无教师逐 token 反馈，实测会退化（TML 实验），作为误解澄清材料。
  - **DAGGER / 过程奖励模型**：谱系来源，只在动机处提及。

## 2.2 学习目标

### Q1：为什么在教师生成的序列上训练学生（标准蒸馏/SFT）会失配，on-policy 蒸馏用什么方式解决？

- 完成答案：读者应能说明自回归模型的训练-推理失配：off-policy 蒸馏训练时学生只见过教师（或数据集）的输出前缀，推理时学生要接续自己生成的前缀；一旦学生早期犯一个教师不会犯的错，就进入训练时从未见过的状态，误差复合放大。OPD 的解法是让训练分布 = 学生自己的生成分布（on-policy），教师反馈在学生实际到达的状态上给出。
- 为什么是核心目标：不理解分布失配，会把 OPD 误解为"换一种数据来源的工程技巧"，而不是针对复合误差的机制性解法。
- 依赖内容：off-policy/on-policy 的含义、复合误差、GKD 框架的 $\lambda$ 参数。

### Q2：on-policy 蒸馏的核心机制是什么——学生采样、教师逐 token 打分、分数如何进入更新？

- 完成答案：读者应能描述完整流程：学生采样得到轨迹（采样时顺便得到学生的 logprobs）；教师对学生轨迹做一次前向得到教师的 logprobs；逐 token 计算分数 $r=\log\pi_{\text{teacher}}(y_t)-\log\pi_S(y_t)$（= 负的 per-token reverse KL 采样值）；把 $r$ 作为该 token 的 advantage 交给策略梯度更新，更新只改变学生参数，教师不更新。GKD 视角下等价于在学生分布上最小化逐 token 散度。
- 为什么是核心目标：这是概念的定义性机制，所有效率与边界结论都从它推出。
- 依赖内容：策略梯度的最小定义、KL 散度最小定义、GKD 的 per-token 散度形式、TML 的 advantage 形式、K3 Eq.15（工程变体）。

### Q3：手算 per-token 奖励——给定教师与学生在某前缀下的 next-token 分布和学生实际采样的 token，算出奖励并判断更新方向？

- 完成答案：读者应能用一个 3-token 词表的数值例子（如学生 $(0.3,0.4,0.3)$、教师 $(0.6,0.3,0.1)$）算出采样各 token 时的 $r=\log(\pi_T/\pi_S)$（如采 B 时 $\ln(0.3/0.4)\approx-0.288$），说明正分数抬高该 token 的对数概率、负分数压低它，并能验证期望奖励等于负的 reverse KL。
- 为什么是核心目标：机制的符号行为必须通过具体数字确认，否则"教师更认可则奖励为正"停留在文字。
- 依赖内容：对数运算、Q2 的机制。

### Q4：相比带奖励模型的 RL，on-policy 蒸馏为什么样本效率和计算效率更高？有哪些实测证据？

- 完成答案：读者应能从信号密度（RL 每 episode 传递 $O(1)$ bits，蒸馏传递 $O(N)$ bits）、奖励不可被 hack、无需单独奖励模型、无需完整 rollout、教师只需一次前向等方面说明效率来源；并引用 Qwen3 Table 21（8B 模型上 on-policy 蒸馏以约 1/10 GPU 小时超过 RL 的效果）与 Thinking Machines 的成本缩减数字。
- 为什么是核心目标：OPD 近期受到重视的主要原因是效率证据，不理解证据来源就无法判断适用性。
- 依赖内容：Q2 机制、稀疏 vs 稠密奖励、Qwen3 与 TML 实验条件。

### Q5：on-policy 蒸馏的边界——它不能做什么、什么条件下会失效？

- 完成答案：读者应能说明：OPD 转移教师已有能力、不发明新能力（RL 是搜索，蒸馏只学最终策略）；学生学习以教师分布为参照（方向性上限，GKD 自蒸馏实验中学生测试分数可超教师，属于分布匹配下的特例）；教师策略需落在学生初始化的支持集内，否则需要显著更大的 batch 或先用 SFT/midtrain 扩支持集；散度选择任务相关；GKD 框架里可与 RL 组合（正则化对象从初始策略换成教师）。
- 为什么是核心目标：避免把 OPD 当万金油——教师不够强、初始化覆盖不足、想超越教师时它会失效或不划算。
- 依赖内容：Q2 机制、mode-seeking 与支持集概念、各实验条件。

## 2.3 内容分级

### 核心内容

- C1：自回归蒸馏的分布失配与复合误差 → Q1。必须讲清：训练见过的前缀 vs 推理时自己生成的前缀；早期错误导致偏离训练分布。
- C2：GKD 统一框架（$\lambda$ 混合；supervised KD 与 on-policy KD 是两个端点）→ Q1、Q2。
- C3：per-token 散度形式（散度作用在每个位置的 next-token 分布上）→ Q2。
- C4：TML 的 advantage 形式（逐 token $\log(\pi_T/\pi_S)$ 作 advantage 进策略梯度；= 负 per-token reverse KL）→ Q2、Q3。
- C5：教师只需一次前向 compute_logprobs、学生 logprobs 采样时已得到；对带 KL 正则的 RL 实现是换正则模型的改动 → Q2、Q4。
- C6：信号密度 $O(1)$ vs $O(N)$ bits；reverse KL 不可被 hack → Q4。
- C7：无需完整 rollout（partial rollouts 可用）→ Q4。
- C8：Qwen3 强到弱蒸馏两阶段定义（off-policy 打底，on-policy 对齐教师 logits 最小化 KL）→ Q4、Q5。
- C9：Qwen3 Table 21 数字（1/10 GPU 小时且效果更好）→ Q4。
- C10：TML 计算量结论（150 步达 70%、9–30× 成本缩减、7–10× 梯度步、50–100× 计算效率）→ Q4。
- C11：RL 的本质是搜索，蒸馏只学最终策略；RL 需要非零成功率起步，蒸馏无此要求但需要覆盖 → Q5。
- C12：散度选择（forward/reverse/JSD），任务相关；reverse KL mode-seeking；SFT 扩支持集、reverse KL 在支持集内模式寻求 → Q5。
- C13：初始化与 batch size 条件（强初始化小 batch，弱初始化大 batch）→ Q5。
- C14：自采样 SFT 会退化（形似而非 OPD）→ Q1、Q5（误解澄清）。
- C15：以教师为参照的上限 + GKD 自蒸馏学生测试分数可超教师的特例 → Q5。
- C16：GKD 与 RLHF/RLAIF 组合（正则化对象换成教师策略）→ Q5。
- C17：K3 把 per-token OPD 奖励（Eq.15 clip+sg）作为 MOPD 的基础 → Q5（衔接）。

### 辅助内容

- A1：DAGGER 与过程奖励模型的谱系（TML 引 Ross et al. 2010、Lightman et al. 2023）→ 澄清 C1 的方法论来源。
- A2：forking tokens 现象（教师惩罚把学生引偏的短语开头 token，最终错误答案反而不被惩罚）→ 澄清 C4 的打分粒度直觉。
- A3：GKD 实验结论摘要（XSum 2.1×、WMT 1.7×、GSM8K 1.9×；5% 数据超过全量 supervised KD）→ 支持 C2/C4 的有效性证据。
- A4：LoRA 与 batch size 的交互（rank 32 下 SFT 后落后 13% vs 蒸馏后 6%）→ 澄清 C13。
- A5：Qwen3 流水线先 off-policy 再 on-policy 的工程理由（先有模式切换能力再 on-policy 训练）→ 澄清 C8。

### 扩展内容

- E1：MiniLLM、ImitKD、f-distill 等变体内部机制 → 排除，只在 GKD 特例表列名。
- E2：Tinker cookbook 实现细节 → 排除，工程专题。
- E3：多教师/MOPD 机制 → 排除，见 `wiki/mopd/`。
- E4：与 DPO、逆强化学习的理论联系 → 排除，理论专题，正文一句话提及即可。

## 2.4 前置知识映射

| 前置概念 | 被哪些学习目标依赖 | 概念页状态 |
|---|---|---|
| 知识蒸馏（off-policy 蒸馏的基准形态） | Q1、Q2 | 已有：`wiki/knowledge-distillation/index.html`，正文链接 |
| 策略梯度（$\nabla\log\pi\cdot A$ 更新规则） | Q2、Q3 | 无页面，最小定义内联（见下说明） |
| KL 散度（含 forward/reverse 方向） | Q2、Q3、Q5 | 无页面，最小定义内联（见下说明） |

说明（沿用 knowledge-distillation 页的裁定）：策略梯度与 KL 散度在本文中只作为支撑性基础概念使用，首次出现时给出服务于当前上下文的最小定义并标注"最小定义"，不展开自身推导，不触发递归生成。softmax、交叉熵同此处理（由 knowledge-distillation 页覆盖其完整讲解，本页直接链接）。

## 2.5 明确不展开的内容

- **MOPD 的多教师路由、条件化学生、partial rollout 调度**：属于独立概念，见 `wiki/mopd/`；本文只在边界章节点出 Eq.15 与 MOPD 的关系。
- **RLHF/PPO/GRPO 算法内部**：本文只使用策略梯度的更新形式这一结论；算法细节是独立概念。
- **RL 的完整训练动力学（curriculum、中间策略）**：只在"RL 是搜索"的对比中使用结论性描述，来源为 TML 的表述。
- **蒸馏变体族谱的完整机制对比**：MiniLLM/ImitKD/f-distill 只在 GKD 特例表中列名，不展开。
- **Hinton 蒸馏的温度机制**：已有 `wiki/knowledge-distillation/` 页面，本文不重复。

## 2.6 常见误解和适用边界

### 常见误解

- **误解 1："on-policy 蒸馏就是用学生自己生成的数据做 SFT。"** 错误。自采样 SFT 没有教师逐 token 反馈：TML 实验中，在学生自采样、教师打分为 KL=0 的数据集上跑 SFT，任何大于零的实用学习率都会导致性能退化，因为有限 batch 的经验分布偏离策略自身，训练逐渐变成 off-policy。正确结论：OPD 的关键不是"数据来自学生"，而是"教师反馈持续校准学生访问的每个状态"。
- **误解 2："教师的梯度会回传到学生。"** 错误。教师只输出标量 logprob（一次前向），分数作为常数 advantage 进入学生的策略梯度更新；K3 的 $\text{sg}$ 算子把这一点显式化。教师参数不更新，梯度不从教师流向学生。
- **误解 3："蒸馏能让学生超越教师、学会教师不会的东西。"** 不成立（方向性结论）。OPD 的信号方向始终指向教师分布；要获得教师没有的能力需要带任务奖励的 RL 探索。特例：GKD 自蒸馏实验中同架构学生的测试分数可超过教师——这是分布匹配目标下的评估现象，不是"发明新能力"。
- **误解 4："OPD 必须用 reverse KL。"** 不准确。GKD 框架里散度是自由度（forward KL、reverse KL、JSD 均可），最优选择任务相关；TML 与 K3 的实践用 reverse KL 形式，GKD 的 on-policy KD 定义用 forward KL。

### 适用边界

- 解决的问题：把已训练好的强教师的能力高效转移给学生，尤其在推理任务上比 RL 便宜。
- 不解决的问题：教师没有的能力（无奖励来源）；教师策略不在学生初始化支持集内且未先扩支持集（信号推不动零概率 token）；需要探索超越教师的目标。
- 成立条件：存在可查询 logprobs 的强教师；学生初始化覆盖教师支持集（否则需大 batch 或先 SFT/midtrain）；训练框架能消费 per-token advantage。
- 条件不满足时：教师弱 → 学生学到弱表现；覆盖不足 → 需显著更大的 batch size，成本优势缩小。
