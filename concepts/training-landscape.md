# 训练 / RL / Post-training 全景图：零基础入门地图

> 写给：完全零基础入门训练的读者。读完能回答：ICLR 2026 论文里那些名词（SFT / RLHF / DPO / PPO / GRPO / RLVR）到底在干嘛，它们之间什么关系，我从哪里开始看。

## 一、先搞清楚：机器学习有几种

在讲 LLM 训练之前，先把机器学习的三大分支分清楚。这是后面所有名词的基础。

| 分支 | 全称 | 学什么 | 数据长什么样 | 例子 |
|---|---|---|---|---|
| **SL** | Supervised Learning 监督学习 | 模仿标准答案 | `(输入, 正确答案)` 对 | 图像分类、SFT |
| **RL** | Reinforcement Learning 强化学习 | 试错 + 奖励信号 | `(状态, 动作, 奖励)`，没有标准答案 | AlphaGo、机器人、RLHF |
| **UL** | Unsupervised Learning 无监督学习 | 找数据内在结构 | 只有输入，没标签 | 聚类、预训练 |

### SL vs RL 的本质区别（关键）

这是整篇文章最核心的一段，看懂了这个，后面的 RLHF / DPO / PPO 都好懂。

**监督学习（SL）**：告诉模型"这道题正确答案是 2，模仿它。"
```
数据：(1+1=, 2)
训练：让模型输出 2 的概率变大
```

**强化学习（RL）**：告诉模型"你随便生成，生成 2 我给你 1 分，生成 3 我给你 0 分。自己试，往高分方向调。"
```
数据：(1+1=, 生成 2, reward=1.0), (1+1=, 生成 3, reward=0.0)
训练：让模型学会生成高 reward 的回答
```

RL 不知道"正确答案"是什么，只知道"这个回答好不好"。模型通过试错，慢慢学会生成高 reward 的回答。

### RL 的三个要素

RL 里有三个核心概念，后面会反复出现：

1. **state（状态）**：当前情况。LLM 里就是 prompt（如 "1+1="）
2. **action（动作）**：模型选什么。LLM 里就是生成的 token（如 "2"）
3. **reward（奖励）**：环境给的反馈。LLM 里是 RM（奖励模型）打的分数

还有一个词 **policy（策略）**：就是"给定状态，选什么动作"的策略。在 LLM 里，policy = 你正在训练的那个 LLM 本身。不是模型的某个部分，就是整个模型。

## 二、LLM 的一生：三个阶段

把一个 LLM 的生命周期压成一行：

```
预训练 (Pretraining)  →  后训练 (Post-training)  →  推理 (Inference)
  万亿 token             SFT + 对齐 + ...           你在这里
  next-token             让模型变"有用"             模型已定型
  烧几个月 GPU·月         几~几千 GPU·天            部署、加速
```

- **预训练**：海量文本上做 next-token prediction，产出 base model。能续写文本，但不会"听指令"。
- **后训练**：让模型学会听指令、对齐人类偏好。**ICLR 2026 大部分论文都在这里。**
- **推理**：模型定型后用起来。这是你熟悉的。

预训练和推理不是本文重点。本文聚焦后训练——ICLR 2026 那堆名词全挤在这里。

## 三、后训练的核心问题：让模型对齐人类偏好

预训练后的模型能生成文本，但可能：
- 不听指令（问"1+1=" 答"你好我是 AI"）
- 不安全（教人做坏事）
- 风格不好（啰嗦、不礼貌）

后训练就是让模型"乖"起来。分两步：

### 第 1 步：SFT（Supervised Fine-tuning，监督微调）

用监督学习，告诉模型"这道题正确答案是什么"。

```
数据：(1+1=, 2)  ← prompt + 标准答案
方法：监督学习，让模型输出"2"的概率变大
```

SFT 就是"前向 + 算 loss + 反传"，和图像分类本质一样。核心 5 行代码：
```python
logits = model(seq)                       # 前向
loss = cross_entropy(logits, target)      # 算 loss
loss.backward()                           # 反传
opt.step()                                # 更新
opt.zero_grad()                           # 清梯度
```

SFT 教模型"该说什么"，但问题来了：**很多问题没有唯一标准答案。**

比如 "怎么学 Python？" 可以答"看官方文档"，也可以答"做项目练手"。两个都好，没法说哪个是"标准答案"。SFT 只能教一个，没法教"两个都不错但 A 更好"这种偏好。

需要第 2 步。

### 第 2 步：对齐（Alignment）

让模型学会"在几个都可以的答案里，挑人类更喜欢的那个"。这需要**偏好数据**。

偏好数据格式：`(prompt, chosen 好回答, rejected 差回答)`
```
(1+1=, 2, 3)              ← "2" 比 "3" 好
(怎么学Python, 看官方文档, 不知道)  ← "看官方文档" 比 "不知道" 好
```

有了偏好数据，怎么训？有两条主流路线：

## 四、对齐的两条路线

### 路线 A：RLHF（经典路线，ChatGPT 用的）

**RLHF = Reinforcement Learning from Human Feedback**（基于人类反馈的强化学习）

完整流程 3 步：

```
① 偏好数据 → 训 RM（奖励模型）
② 用 RM 当环境
③ 用 RL（PPO 算法）训 policy（LLM）
```

**第 1 步：训 RM（Reward Model，奖励模型）**

RM 是一个"打分器"：输入 (prompt, 回答)，输出一个标量分数。分数高 = 这个回答好。

为什么要训 RM？因为人工标注太贵（每条回答都要人看一遍），但训完 RM 后它可以批量给任意回答打分，代替人工。

RM 用 **Bradley-Terry 模型**训练：把"chosen 比 rejected 好"建模成概率 `P = σ(r_w - r_l)`，loss 是 `-log P`。让 RM 学会 `r(chosen) > r(rejected)`。

**第 2 步：用 RM 当环境**

policy（LLM）生成一个回答，RM 给它打分。这个分数就是 reward。

```
prompt "1+1=" → policy 生成 "2" → RM 打分 1.0
prompt "1+1=" → policy 生成 "3" → RM 打分 -1.0
```

**第 3 步：用 RL 训 policy**

用 **PPO 算法**（下面详讲）让 policy 学会生成高 reward 的回答。本质就是：
- policy 采样一个回答
- RM 打分
- 用这个分数当 reward，通过梯度上升让高 reward 的回答概率变大

**RLHF 的优点**：RM 训完后可以反复用，policy 可以持续探索新回答。
**RLHF 的缺点**：流程复杂（3 步），RM 可能被骗（reward hacking），训练不稳定。

### 路线 B：DPO（简化路线）

**DPO = Direct Preference Optimization**（直接偏好优化）

DPO 的核心洞察：数学上可以证明，最优的 policy 可以直接从偏好数据 + 参考模型算出来，**不需要显式训 RM，也不需要 PPO**。

```
RLHF：偏好数据 → 训 RM → 用 PPO 优化 policy（3 步）
DPO：  偏好数据 → 直接优化 policy（1 步）
```

DPO 的 loss 直接在偏好对上算：
```
L_DPO = -log σ(β log π_θ(y_w)/π_ref(y_w) - β log π_θ(y_l)/π_ref(y_l))
```

让 chosen 相对参考模型的概率上升，rejected 下降。一个 loss 搞定。

**DPO 的优点**：简单，不用 RM 不用 PPO，训练稳定。
**DPO 的缺点**：不能持续探索（只用静态偏好数据），效果上限可能不如 RLHF。

### 两条路线怎么选

| 维度 | RLHF | DPO |
|---|---|---|
| 复杂度 | 3 步 | 1 步 |
| 需要训 RM | 是 | 否 |
| 需要 RL | 是（PPO） | 否 |
| 能持续探索 | 是 | 否 |
| 训练稳定性 | 较差 | 好 |
| 实际使用 | ChatGPT、Claude | Llama 3、Qwen |

工业界现在流行 **DPO 做 base，RLHF 做精细调整**。

## 五、RLHF 里用到的 RL 算法：PG → PPO → GRPO

RLHF 的第 3 步要用 RL 算法。这个算法演化路径：

### PG（Policy Gradient，策略梯度）—— 最基础

**核心公式**：`∇_θ J = E[G · ∇_θ log π(a|s)]`

**直觉**：
- 采样一个 action（policy 生成一个 token）
- 算这个 action 的 return（reward 累加）
- 用 return 当权重：return 大的 action，让它的概率上升；return 小的，下降

**PG 的问题**：on-policy——采一次用一次就丢。每次更新后策略变了，旧样本就失效。样本效率低。

### PPO（Proximal Policy Optimization）—— PG 的改良

**PPO 解决 PG 的痛点**：让旧样本能复用。

两个新概念：
1. **importance ratio** `r_t = π_new(a) / π_old(a)`：新策略相对旧策略对这个 action 的偏好倍数。用这个修正旧样本的分布偏移。
2. **clip**：限制 `r_t` 在 `[1-ε, 1+ε]` 范围（通常 ε=0.1~0.3）。超出范围梯度消失，相当于"自动刹车"——防止策略偏太远导致训练不稳定。

**PPO 目标**：`L = min(r_t · A, clip(r_t, 1-ε, 1+ε) · A)`

PPO 是 RLHF 的事实标准。OpenAI、Anthropic、Google 训 ChatGPT 类模型都用 PPO。

### GRPO（Group Relative Policy Optimization）—— PPO 的简化

**GRPO 是 DeepSeek-R1 用的算法。**

PPO 需要训一个 value model（额外模型预测"这个状态的预期 return"当 baseline）。GRPO 去掉 value model，改用"组内相对奖励"：对同一个 prompt 采样多个回答，用它们 reward 的相对大小当 baseline。

**GRPO 的优点**：少训一个模型，省显存。
**GRPO 的场景**：数学、代码这种能明确判断对错的任务（配合 RLVR）。

### 三者关系一句话

```
PG（基础） → PPO（加 importance ratio + clip，复用旧样本） → GRPO（去掉 value model，用组内 baseline）
```

每一步都是解决前一步的痛点。**你现在只要懂 PG 就能看懂 RLHF HTML**，PPO 和 GRPO 是优化改良。

## 六、reward 从哪来：RLHF vs RLVR

RLHF 和 RLVR 都是"用 RL 训 policy"，区别只在 reward 来源：

| 路线 | reward 来源 | 适用场景 |
|---|---|---|
| **RLHF** | 训一个 RM，RM 从人工偏好学来 | 开放对话、写作、风格 |
| **RLVR** | 外部验证器（答案对不对、代码能不能跑） | 数学、代码、逻辑推理 |

RLVR = Reinforcement Learning from Verifiable Rewards。

```
RLHF：reward = RM(prompt, response)              ← RM 是另一个模型
RLVR：reward = 1 if 答案对了 else 0              ← 规则验证
```

**RLVR 的优点**：reward 可靠（不会被骗）、便宜（不用训 RM）。
**RLVR 的局限**：只在有明确对错的任务上有效。写诗、聊天这种没有"对错"的用不了。

**DeepSeek-R1 / o1 类推理模型**主要用 RLVR + GRPO。数学题做对就是 1 分，做错就是 0 分，reward 信号清晰。

## 七、几个容易混的术语澄清

**Q：训练（Training）/ 微调（Fine-tuning）/ 后训练（Post-training）什么关系？**
- Training：口语伞词，泛指"让模型变成现在这样"的整个过程
- Post-training：预训练之后所有训练的统称
- Fine-tuning：最早专指"在预训练模型上继续训练"，LLM 时代经常特指 SFT
- 读论文时看上下文判断

**Q：对齐（Alignment）和 Post-training 什么关系？**
- 对齐是 post-training 的一个子阶段（让模型安全、符合偏好）
- 有些论文把对齐和 SFT 并列，有些把 SFT 也算对齐的一部分
- 边界 fuzzy，不要纠结

**Q：论文标题里的 "RL" 指什么？**
- LLM 后训练论文标题里的 RL 通常指"用 RL 工具（PPO/GRPO）训模型"
- 机器人/游戏论文里的 RL 指传统强化学习
- 看到标题先判断：是 LLM 论文 → RL 多半指 RLHF/RLVR；是机器人/游戏 → 传统 RL

**Q：SFT 和 RLHF 必须都做吗？顺序固定吗？**
- 典型顺序：SFT → RLHF/DPO
- 因为 RLHF 需要模型已经会"基本输出"才能比较好坏
- 但也有只用 SFT、或只用 DPO 的模型
- RLVR 通常建在 SFT 之后

**Q：policy / value / reward / advantage 这些 RL 术语什么意思？**
- **policy（策略）**：你正在训练的模型本身。LLM 里 = LLM
- **reward（奖励）**：RM 给这个回答打的分
- **value（价值）**：这个状态的"预期未来 return"。用于算 advantage
- **advantage（优势）**：`A = G - V`，这个 action 相对平均好不好。正 = 比平均好，负 = 比平均差
- **return（回报）**：未来奖励的累计。单步场景 `G = r`

## 八、推荐学习顺序

按难度递增，每一步只新增一两个概念：

1. **backward**（最基础）：理解梯度下降、反传怎么工作。[concepts/backward/](backward/)
2. **SFT**：前向 + 算 loss + 反传。你已经会一半（前向）
3. **DPO**（跳过 RL 也能理解对齐）：偏好数据 + 一个 loss。[concepts/dpo/](dpo/)
4. **RLHF**（进入 RL 领域）：RM + PG + PPO 完整 pipeline。[concepts/rlhf/](rlhf/)
5. **GRPO / RLVR**（2024-2025 推理模型路线）：DeepSeek-R1 / o1 类用的

到第 4 步完成，ICLR 2026 后训练论文你能看懂 70%。第 5 步是当前热点（推理模型训练）。

## 九、ICLR 2026 论文阅读顺序

按页面分类，性价比从高到低：

| 顺序 | 分类 | 篇数 | 看什么 |
|---|---|---|---|
| 1 | 基础/前沿模型 > 指令微调与对齐 | 186 | 后训练主战场，论文最多最全 |
| 2 | 强化学习 > 基于偏好/反馈的 RL | 23 | RLHF 专门方向，量小好啃 |
| 3 | 基础/前沿模型 > LLM 预训练 | 69 | 建立对预训练的直觉 |
| 4 | 对齐/安全 > 安全对齐 | 127 | RLHF/DPO 的应用层延伸 |
| 5 | 优化 > 优化器设计 + 训练动态 | 80 | 训练侧的"为什么训得动" |
| 6 | 基础设施 > 训练系统 | 11 | 推理工程师最容易共情的部分 |

前 2 个分类读完，ICLR 后训练论文你能看懂 70%。
