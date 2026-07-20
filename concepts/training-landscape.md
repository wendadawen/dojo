# 训练 / RL / Post-training 全景图：给推理工程师的入门地图

> 写给：会推理、不懂训练的工程师。读完能回答：ICLR 2026 那堆论文到底在干嘛，那些名词谁包含谁，我从哪里开始看。

## 一、你站在哪，前面是什么

把一个 LLM 的生命周期压成一行：

```
预训练 (Pretraining)  →  后训练 (Post-training)  →  推理 (Inference)
   万亿 token              SFT + 对齐 + ...           你在这里
   next-token              让模型变"有用"            模型已定型
   算几万吨 GPU·月          算几十~几千 GPU·天        部署、加速
```

你熟悉的"推理"是最后一环。ICLR 2026 那个页面上让你眼花缭乱的"训练 / RL / fine-tuning / SFT / post-training / RLHF / DPO / GRPO / RLVR ..."，几乎全部挤在中间那一环——**后训练**。

所以你看到的混乱不是你不懂，而是这个领域术语用得乱：一个词在不同论文里指代范围不一样，"训练"是口语伞词、"fine-tuning"有时指 SFT 有时泛指后训练、"RL"在 LLM 语境里往往特指"用 RL 做对齐"而不是传统强化学习。下文逐个拆开。

## 二、名词地图：谁包含谁

先看一张全景，再逐格解释。

```
训练 (Training)  ← 口语伞词，泛指"让模型变成现在这样"的整个过程
│
├── 预训练 (Pretraining)
│   └── next-token prediction，万亿 token，产出 base model
│       base model 能续写文本，但不会"听指令"
│
└── 后训练 (Post-training)  ← ICLR 2026 大部分论文集中在这里
    │
    ├── SFT (Supervised Fine-tuning, 指令微调)
    │   监督学习，喂 (prompt, 理想回答) 对
    │   教模型"听指令 + 输出格式 + 风格"
    │   本质就是前向 + 算 loss + 反传，你已会一半
    │
    ├── 对齐 (Alignment)  ← 让模型"乖"
    │   │
    │   ├── RLHF (RL from Human Feedback)
    │   │   ① 用人工标注的偏好对训练一个 reward model
    │   │   ② 用 RL (PPO) 优化 policy 来最大化 reward
    │   │   经典 ChatGPT 路线
    │   │
    │   ├── DPO (Direct Preference Optimization)
    │   │   跳过 reward model，直接在偏好对上优化 policy
    │   │   一个 loss 搞定，比 RLHF 简单
    │   │
    │   ├── GRPO (Group Relative Policy Optimization)
    │   │   PPO 的简化版（DeepSeek-R1 用的）
    │   │   不要 value model，用组内相对奖励当 baseline
    │   │   现在训推理模型的主力方法之一
    │   │
    │   └── RLVR (RL from Verifiable Rewards)
    │       奖励是"答案对不对"（数学可验证、代码能跑通）
    │       不用学一个 reward model，奖励来自外部验证器
    │       o1 / DeepSeek-R1 路线的核心
    │
    ├── PEFT (Parameter-Efficient Fine-Tuning, 高效微调)
    │   只动一小部分参数，省显存
    │   代表：LoRA、Adapter、DiaBlo
    │   与 SFT/RLHF 正交——你可以用 LoRA 做 SFT，也能用 LoRA 做 RLHF
    │
    └── 安全对齐 / 红队 / 防越狱 / 去幻觉 ...
        应用层面的对齐工作，ICLR 这部分论文很多

强化学习 (RL)  ← 独立的学科，不是 LLM 专属
│
├── 传统 RL：MDP、agent、environment、机器人、游戏
│   ICLR "强化学习" 大类 306 篇，大部分是这类
│
└── RL 用于 LLM：RLHF / RLVR / PPO / GRPO
    借用了 RL 的 policy gradient / advantage / PPO 等工具
    你不需要先精通传统 RL 才能看懂 LLM 的 RLHF
```

## 三、每个名词一句话


| 名词                    | 一句话                                          | 你需要先懂的依赖                           |
| --------------------- | -------------------------------------------- | ---------------------------------- |
| **预训练**               | 万亿 token 上做 next-token prediction，烧几万吨 GPU·月 | 懂 transformer 前向即可                 |
| **SFT**               | 拿 (prompt, 理想回答) 对做监督学习，和图像分类同构              | 前向 + loss + 反传                     |
| **Post-training**     | 预训练之后、推理之前的所有训练阶段的统称                         | SFT 概念                             |
| **对齐 (Alignment)**    | 让模型"听指令、不害人、符合人类偏好"的训练阶段                     | SFT                                |
| **RLHF**              | 训一个 reward model，再用 PPO 优化 policy            | 需要懂 RL 基础（policy/reward/advantage） |
| **DPO**               | 不要 reward model，直接在偏好对上算 loss                | SFT + 偏好数据的概念                      |
| **GRPO**              | PPO 的简化版，组内相对奖励做 baseline                    | RLHF（PPO）                          |
| **RLVR**              | reward 来自"答案对不对"（数学/代码可验证），不用学 RM            | RLHF 概念                            |
| **PEFT / LoRA**       | 只训练很少的参数（低秩矩阵等），省显存                          | SFT                                |
| **Reward Model (RM)** | 学一个"给回答打分"的模型，RLHF 第一阶段                      | 偏好数据                               |
| **Policy / Value**    | RL 术语，policy=模型本身，value=预期未来奖励               | RL 基础                              |
| **PPO**               | 一种 policy gradient 算法，RLHF 的主力               | RL 基础                              |
| **Preference data**   | (prompt, chosen, rejected) 三元组，对齐的数据燃料       | —                                  |


## 四、从推理视角切入的入门路径

你已有的武器：模型架构、前向、KV cache、batching、显存管理。这套武器正好可以借力——训练只是多了"反传 + 优化器 + 数据 pipeline"，前向那半你已经会。

按下面的顺序，每一步都只新增一两个概念：

**第 1 步：SFT（最容易，先建立信心）**

- 概念：拿 (prompt, 理想回答) 对，前向算 next-token loss，反传更新权重。
- 你只需要新增一个概念：**损失函数从推理时的"不计算"变成"算交叉熵并反传"**。
- 推荐读：任意一篇 SFT 基础博客，或 ICLR 2026「指令微调与对齐」分类下的简单论文。

**第 2 步：偏好数据 + DPO（跳过 RL 也能理解对齐）**

- 概念：人工/模型标"回答 A 比回答 B 好"，得到 (prompt, chosen, rejected) 三元组。
- DPO 的妙处：直接在这上面算一个 loss，**完全不用 RL 数学**，效果接近 RLHF。
- 推荐读：DPO 原论文（Rafailov et al. 2023）。
- 这一步之后你已经能看懂 ICLR 一半的对齐论文。

**第 3 步：RLHF / PPO（进入 RL 领域，难度跳升）**

- 这一步要补 RL 基础术语：policy、reward、advantage、value function、policy gradient、PPO 的 clip 机制。
- 推荐路径：先读 OpenAI InstructGPT 论文（2022）建立全局图，再读任意 PPO 教程补数学。
- 这一步之后你能看懂 RLHF 类论文。

**第 4 步：GRPO / RLVR（2024-2025 推理模型路线）**

- GRPO：DeepSeek-R1 用的，PPO 简化版，去掉了 value model。
- RLVR：奖励从"模型打分"换成"答案对不对"，数学/代码场景特别有效。
- 这一步对应 ICLR 2026 的热点——R1 之后的推理模型训练。
- 推荐读：DeepSeek-R1 论文 + GRPO 原论文。

**第 5 步（并行）：LoRA / PEFT**

- 这是"怎么省显存地做 SFT/RLHF"，与上面正交，可以随时插入。
- 推荐读：LoRA 原论文（Hu et al. 2021）。

**第 6 步（可选）：预训练**

- 概念最简单（next-token prediction），工程最复杂（数据 pipeline、并行策略、稳定性）。
- 如果你的目标只是看懂 ICLR 后训练论文，这一步可以最后看。
- 推荐读：LLaMA 论文系列。

## 五、SFT 代码示例（可直接运行）

**核心就三行**：前向 → 算 loss → 反传。前向你已经会，只新增两行。这个版本用真正的 transformer（self-attention + causal mask），和 GPT/Llama 同构，只是参数少。

```python
"""SFT 示例：真正的 transformer 架构（causal mask + self-attention）"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 数据 ===
# 10 个词的词表
vocab = ["<pad>", "hi", "there", "how", "are", "bye", "now", "see", "ya", "<end>"]
#          0       1     2       3      4      5     6      7     8      9

# 2 条训练数据：(prompt, response)
# transformer 有 attention，能看清上下文，token 可以自然重复，不用刻意避开
TRAIN = [
    ([1, 2], [3, 4, 9]),   # "hi there"  → "how are <end>"
    ([5, 1], [3, 7, 9]),   # "bye hi"    → "how see <end>"   ← "how" 在两条里都有
]

# === 模型：真正的 LLM 架构（缩小版） ===
class TinyLLM(nn.Module):
    def __init__(self, vocab_size=10, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        # TransformerEncoderLayer = self-attention + FFN，是 GPT/Llama 的核心组件
        # d_model=16: 词向量维度
        # nhead=2: attention 头数
        # batch_first=True: 输入形状用 [batch, seq, dim]
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d, vocab_size)   # 输出层：词向量 → 词表得分

    def forward(self, x):
        # x 形状: [batch, seq]
        T = x.size(1)
        # causal mask：上三角为 -inf，让每个位置只能看前面的位置（自回归）
        # 这是 GPT/Llama 等 LLM 的关键——不能"偷看"未来的 token
        causal_mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        h = self.embed(x)                          # [batch, seq, d]
        h = self.transformer(h, mask=causal_mask)  # self-attention
        return self.head(h)                        # [batch, seq, vocab_size]

torch.manual_seed(0)
model = TinyLLM()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

# === 训练循环 ===
PROMPT_LEN = 2
for step in range(200):
    total_loss = 0.0
    for prompt_ids, resp_ids in TRAIN:
        # 拼成 [prompt..., response...]
        # 例：[1,2] + [3,4,9] = [1,2,3,4,9]，形状 [1, 5]（1 是 batch）
        seq = torch.tensor(prompt_ids + resp_ids).unsqueeze(0)  # [1, 5]
        logits = model(seq)[0]                                    # [5, 10]

        # ★ 只对 response 部分算 loss
        # 关键规律：logits[i] 预测 seq[i+1]（next-token prediction）
        #
        #   seq   = [1,    2,     3,    4,    9]     长度 5
        #   位置:    0     1      2     3     4
        #           └prompt┘   └─response─┘
        #
        #   logits[0] → 预测 seq[1] = "there"     ← prompt 内部，不算
        #   logits[1] → 预测 seq[2] = "how"       ★ 算 loss
        #   logits[2] → 预测 seq[3] = "are"       ★ 算 loss
        #   logits[3] → 预测 seq[4] = "<end>"      ★ 算 loss
        #   logits[4] → 预测 seq[5]（不存在）       ← 越界，不算
        #
        # 所以要切：
        #   pred   = logits[1:4]   ← 即 logits[PROMPT_LEN-1 : seq_len-1]
        #   target = seq[2:5]      ← 即 seq[PROMPT_LEN : seq_len]
        seq_len = seq.size(1)                          # 5
        pred   = logits[PROMPT_LEN - 1 : seq_len - 1]  # [3, 10]
        target = seq[0, PROMPT_LEN:]                    # [3]
        loss = F.cross_entropy(pred, target)
        total_loss += loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防训练不稳定
        opt.step()
        opt.zero_grad()

    if step % 40 == 0 or step == 199:
        print(f"step {step:3d}  loss {total_loss.item():.4f}")

# === 测试：自回归生成 ===
print("\n测试：给 prompt，模型自己一个一个生成 token")
model.eval()
with torch.no_grad():
    for prompt_ids, expected in TRAIN:
        ids = list(prompt_ids)
        for _ in range(3):
            x = torch.tensor(ids).unsqueeze(0)        # [1, len]
            logits = model(x)[0, -1]                    # 取最后一个位置的预测
            next_id = logits.argmax().item()
            ids.append(next_id)
            if next_id == 9:  # 遇到 <end> 停止
                break
        resp_pred = [vocab[i] for i in ids[len(prompt_ids):]]
        resp_expected = [vocab[i] for i in expected]
        prompt_str = " ".join(vocab[i] for i in prompt_ids)
        print(f"  prompt='{prompt_str:10}  期望={resp_expected}  实际={resp_pred}")
```

**预期输出**：

```
step   0  loss 5.1148
step  40  loss 0.0094
step  80  loss 0.0032
step 120  loss 0.0036
step 160  loss 0.0007
step 199  loss 0.0046

测试：给 prompt，模型自己一个一个生成 token
  prompt='hi there    期望=['how', 'are', '<end>']  实际=['how', 'are', '<end>']
  prompt='bye hi      期望=['how', 'see', '<end>']  实际=['how', 'see', '<end>']
```

loss 从 5.1 降到 0.005，两条数据都完全学对。注意"bye hi"→"how see `<end>`" 这种 case——"how" 在两条数据里都出现但下一个 token 不同，transformer 靠 attention 看清上下文能学会，position-wise 模型学不会。

### torch 核心概念速查

第一次接触 torch 训练，只要记住这 5 个概念：


| 概念       | 代码                                     | 干什么       | 类比     |
| -------- | -------------------------------------- | --------- | ------ |
| **前向**   | `logits = model(seq)`                  | 模型给输入打分   | 模型"猜"  |
| **loss** | `loss = F.cross_entropy(pred, target)` | 算"猜得多离谱"  | 老师判分   |
| **反传**   | `loss.backward()`                      | 算每个参数该往哪调 | 找出错原因  |
| **更新**   | `opt.step()`                           | 按反传结果调参数  | 学生订正   |
| **清梯度**  | `opt.zero_grad()`                      | 清掉上轮梯度    | 翻到新的一页 |


**训练循环固定套路**（所有训练代码都是这个骨架）：

```python
for step in range(N):
    logits = model(data)          # 前向
    loss = criterion(logits, tgt) # 算 loss
    loss.backward()              # 反传
    opt.step()                   # 更新
    opt.zero_grad()              # 清梯度
```

### 三个关键点

**1. SFT 就是"前向 + 算 loss + 反传"——你已会一半**

推理时你也做前向，只是不算 loss。SFT 多了两行：

```python
loss = F.cross_entropy(pred, target)   # 算 loss
loss.backward(); opt.step()            # 反传 + 更新
```

**2. 只对 response 位置算 loss**

```python
PROMPT_LEN = 2
seq_len    = 5
pred   = logits[PROMPT_LEN - 1 : seq_len - 1]   # logits[1:4]，跳过 prompt 内部
target = seq[0, PROMPT_LEN:]                     # seq[2:5]，只取 response 部分
```

模型不需要学"复述 prompt"，只要学"给 prompt 后该怎么回答"。`logits[i]` 永远在预测 `seq[i+1]`，所以切的时候错开一位。工业界用 `labels = [-1, -1, ..., resp_token1, ...]` 实现，-1 的位置被 cross_entropy 忽略。

**3. 数据格式决定方法类型**

```python
# SFT 的数据：(prompt, 标准答案)
([1, 2], [3, 4, 9])              # "hi there" → "how are <end>"

# 对比 RLHF（下一章会写）：(prompt, chosen, rejected)
# 同一个 prompt，给一个好回答 + 一个坏回答
```

SFT 有"标准答案"→ 监督学习。RLHF 没有"标准答案"，只有"谁更好"→ 强化学习。这就是两者本质不同。

### 关于模型架构

这个 demo 的 `TinyLLM` 和真实 LLM（GPT/Llama）同构，只是规模小：

- **Embedding**：词 ID → 词向量（真实 LLM 用 4096+ 维，这里用 16 维）
- **TransformerEncoderLayer + causal mask**：self-attention 让每个位置看前面所有 token，causal mask 保证不能偷看未来
- **Linear head**：词向量 → 词表得分

工业级 SFT 和这个 demo 的唯一区别是工程包装：批处理、padding、attention mask、分布式训练、checkpoint 保存等。**核心训练逻辑就这 5 行**：

```python
logits = model(seq)
loss  = F.cross_entropy(pred, target)
loss.backward()
opt.step()
opt.zero_grad()
```

### backward 在做什么（线性层例子可视化）

`loss.backward()` 是训练里最神秘的一步。它做的事其实就一件：**算每个参数要往哪调、调多少，才能让 loss 变小**。

完整的可视化动画在：[backward/](backward/)（浏览器打开，9 步动画，每步带数值演算）。

下面是这个例子的最小版本——一个 2→3 的线性层 + softmax + cross_entropy。

**模型与数据**：

$$
x = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}, \quad
W = \begin{bmatrix} 0.5 & -0.5 \\ 0.2 & 0.3 \\ -0.1 & 0.4 \end{bmatrix}, \quad
b = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \quad
\text{target} = 1, \quad \text{lr} = 0.5
$$

**前向**（$\text{logits} = Wx + b \to \text{softmax} \to \text{cross\_entropy}$）：

$$
\text{logits} = Wx + b = \begin{bmatrix} -0.5 \\ 0.8 \\ 0.7 \end{bmatrix}
$$

$$
\text{probs} = \text{softmax}(\text{logits}) = \begin{bmatrix} 0.125 \\ 0.459 \\ 0.416 \end{bmatrix}
$$

$$
\text{loss} = -\log(\text{probs}[\text{target}]) = -\log(0.459) = 0.778
$$

**反传**（梯度从 loss 往左传，每一步用链式法则）：

**第 1 步**：loss 对 logits 的梯度——cross_entropy + softmax 的梯度有一个优美的公式：

$$
\frac{\partial L}{\partial \text{logits}} = \text{probs} - \text{one\_hot}(\text{target}) = \begin{bmatrix} 0.125 \\ 0.459 \\ 0.416 \end{bmatrix} - \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.125 \\ -0.541 \\ 0.416 \end{bmatrix}
$$

> target 位置的梯度是负的 → logits[1] 要变大（让 target 概率上升）

**第 2 步**：loss 对 W 的梯度——链式法则。因为 $\text{logits}[i] = \sum_j W[i,j] \cdot x[j]$，所以：

$$
\frac{\partial L}{\partial W[i,j]} = \frac{\partial L}{\partial \text{logits}[i]} \cdot x[j]
$$

$$
\frac{\partial L}{\partial W} = \begin{bmatrix} 0.125 & 0.250 \\ -0.541 & -1.082 \\ 0.416 & 0.831 \end{bmatrix}
$$

> target 行（第 2 行）的梯度都是负的——因为 W[1,j] 增大会让 logits[1] 增大，loss 减小。梯度大小还乘了 x[j]：x[1]=2 比 x[0]=1 大，所以 W[i,1] 的梯度绝对值更大。

**第 3 步**：loss 对 b 的梯度——b 的系数是 1，所以：

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \text{logits}} = \begin{bmatrix} 0.125 \\ -0.541 \\ 0.416 \end{bmatrix}
$$

**更新**（$W \mathrel{-}= \text{lr} \cdot \frac{\partial L}{\partial W}$）：

$$
W_{1,0}: \, 0.2 - 0.5 \times (-0.541) = 0.470 \quad (\text{增大} \checkmark)
$$

$$
W_{1,1}: \, 0.3 - 0.5 \times (-1.082) = 0.841 \quad (\text{增大} \checkmark)
$$

$$
W_{0,0}: \, 0.5 - 0.5 \times 0.125 = 0.437 \quad (\text{减小} \checkmark)
$$

**验证**（用新 $W$ 再前向一次）：

$$
\text{logits}_{\text{new}} = W_{\text{new}} x + b_{\text{new}} = \begin{bmatrix} -0.876 \\ 2.422 \\ -0.547 \end{bmatrix}
$$

$$
\text{probs}_{\text{new}} = \begin{bmatrix} 0.034 \\ 0.919 \\ 0.047 \end{bmatrix}, \quad \text{loss}_{\text{new}} = -\log(0.919) = 0.085
$$

> loss 从 0.778 降到 0.085，target 位置的概率从 0.459 升到 0.919。一次训练步完成。

**核心直觉**（记住这三点就懂了 backward）：

1. **梯度符号告诉你方向**：负号 = 这个参数该变大，正号 = 该变小。target 位置的梯度永远是负的（因为 probs 永远小于 1，probs - 1 永远是负的），所以 W 的 target 行会被增大，其他行会被减小——模型学会了"让 target 位置得分更高"。

2. **梯度大小告诉你力度**：绝对值越大 = 离正确答案越远 = 该调的幅度越大。$x[j]$ 越大，$\frac{\partial L}{\partial W[i,j]}$ 的绝对值越大——这就是为什么"输入越大，对应的权重更新越快"。

3. **链式法则就是"传梯度"**：loss 对 logits 有梯度，logits 对 W 有梯度（就是 x），所以 loss 对 W 的梯度 = 两者相乘。整个深度学习的 backward 就是这样一层层把梯度从 loss 往回传到每一个参数。

## 六、ICLR 2026 论文的阅读顺序

按页面分类，性价比从高到低：


| 顺序  | 分类                     | 篇数  | 看什么             |
| --- | ---------------------- | --- | --------------- |
| 1   | 基础/前沿模型 > **指令微调与对齐**  | 186 | 后训练主战场，论文最多最全   |
| 2   | 强化学习 > **基于偏好/反馈的 RL** | 23  | RLHF 专门方向，量小好啃  |
| 3   | 基础/前沿模型 > **LLM 预训练**  | 69  | 建立对预训练的直觉       |
| 4   | 对齐/安全 > **安全对齐**       | 127 | RLHF/DPO 的应用层延伸 |
| 5   | 优化 > **优化器设计 + 训练动态**  | 80  | 训练侧的"为什么训得动"    |
| 6   | 基础设施 > **训练系统**        | 11  | 推理工程师最容易共情的部分   |


前 2 个分类读完，ICLR 后训练论文你能看懂 70%。

## 七、常见混淆（FAQ）

**Q：论文标题里的"RL"是不是指强化学习那一整个学科？**
A：不一定。LLM 后训练论文标题里的 RL 通常指"用 RL 工具（PPO/GRPO）训模型"，是传统 RL 在语言模型上的一个特定应用。看到标题先判断：是 LLM 论文 → RL 多半指 RLHF/RLVR；是机器人/游戏论文 → RL 指传统强化学习。

**Q："fine-tuning" 和 "post-training" 什么关系？**
A：口语里经常混用。严格说：post-training 是预训练之后所有训练的统称；fine-tuning 最早专指"在预训练模型上继续训练"，但在 LLM 时代经常被用来特指 SFT。读论文时看上下文判断它指的是 SFT 还是泛指后训练。

**Q："对齐 (alignment)" 和 "post-training" 什么关系？**
A：对齐是 post-training 的一个子阶段（让模型安全、符合偏好）。但有些论文把对齐和 SFT 并列，有些把 SFT 也算作对齐的一部分。边界 fuzzy，不要纠结。

**Q：SFT 和 RLHF 必须都做吗？顺序固定吗？**
A：典型顺序是 SFT → RLHF/DPO，因为 RLHF 需要模型已经会"基本输出"才能比较好坏。但也有只用 SFT、或只用 DPO 的模型。RLVR 通常建在 SFT 之后。

**Q：RLVR 和 RLHF 的区别？**
A：reward 来源不同。RLHF 的 reward 来自一个学出来的 reward model（基于人工偏好）。RLVR 的 reward 来自外部验证器（数学答案对不对、代码能不能跑通）。后者更便宜、更可靠，但只在有明确对错的任务上有效。

**Q：我需要先学 RL 课程才能看懂这些论文吗？**
A：不需要。理解 RLHF 只需要懂 policy gradient + PPO 的 clip 思想，半天能补齐。传统 RL 课程里大量的 MDP 理论、表格方法、探索理论对 LLM 后训练用处不大。

## 八、下一步建议

1. 先读 SFT 一篇基础博客（半天）
2. 读 DPO 原论文（一天）—— 这一步之后你已经能入门 ICLR 一半论文
3. 补 PPO 基础 + 读 InstructGPT 论文（两天）
4. 读 DeepSeek-R1 论文 + GRPO 原论文（两天）—— 这一步之后 ICLR 2026 的推理模型训练论文你能看懂
5. 回到 ICLR 2026 那个页面，按上面第五节顺序读

到第 4 步完成，"训练"对你不再是黑盒，ICLR 那些名词之间的关系也会自然清晰。