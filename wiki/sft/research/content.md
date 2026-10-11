# SFT

- 描述：监督微调（SFT）是在一个已经会做下一 token 预测、还没学过按指令作答的模型上，用人工写好的「指令—回答」标注数据继续训练，让模型从「接着续写」变成「按指令作答」。
- 摘要：用人工写好的示范数据继续训练预训练模型，只对回答部分算损失；模型由此学会按指令作答，并成为后续偏好优化的起点。
- 主题：训练与优化
- 标签：训练

## 资料

- Long Ouyang, Jeff Wu, Xu Jiang 等. Training language models to follow instructions with human feedback. NeurIPS 2022, arXiv:2203.02155. [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- Hugo Touvron, Louis Martin, Kevin Stone 等. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288, 2023. [https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)

## 短页

### 在示范数据上继续训练

监督微调（SFT）是在一个预训练模型（在大批文本上训练过下一 token 预测、还没按指令调过的模型）上，用人工写好的「指令—回答」对继续训练。文本先被切成 token（模型读写文字的基本单位），训练目标仍是下一 token 预测：把一条样本的指令和回答拼成序列，逐位置预测下一个 token，与真实 token 比交叉熵（衡量两个概率分布差多少的量，见[交叉熵](wiki/cross-entropy)），要最小化的就是这个量，叫损失。继续这个目标，就是把「看到指令该给出什么回答」写进模型的参数。
依据：paper/llama2/src/templateArxiv.tex L343-L392；paper/instructgpt/src/neurips_2021.tex L121

### 续写不是回答

预训练模型的训练目标是在网页文本上预测下一个 token，用户要的是遵循指令、有帮助且安全，两者不同。在输入里给几个例子能让模型照着模仿，但这只用当前这段输入，不改模型参数。SFT 换用示范数据继续训练，让模型收到指令时作答。
依据：paper/instructgpt/src/neurips_2021.tex L121；paper/instructgpt/src/neurips_2021.tex L191

### 损失只算在回答上

SFT 的一条样本是指令加回答；多条拼接成固定长度的序列，指令段与回答段之间用一个特殊 token 分隔。训练时把指令段 token 上的损失置零，梯度只从回答 token 回传，模型学的是「给定指令、产出回答」。
依据：paper/llama2/src/templateArxiv.tex L343；paper/llama2/src/templateArxiv.tex L391

## 长页

语言模型做的是下一 token 预测：一段文本先被切成 token（模型读写文字的最小单位），给定前面的 token，模型输出词表（所有 token 的集合）上每个 token 作为下一个 token 的概率，真实 token 的概率越高越好，误差用交叉熵衡量（概率越低误差越大）。在大批文本上反复做这件事，得到的模型叫预训练模型；它学到的是「给定前面这些词，下一个词最可能是什么」，不是「收到指令该怎么回答」。监督微调（SFT）在这个预训练模型上继续用下一 token 预测训练，训练数据换成人工写好的「指令—回答」对，让模型收到指令时给出回答而不是继续续写。
依据：paper/llama2/src/templateArxiv.tex L343-L392；paper/instructgpt/src/neurips_2021.tex L121

### 目标错位

预训练模型的训练目标是预测网页文本的下一个 token；用户对它的要求是遵循指令、有帮助且安全。两个目标并不相同，模型常出现不遵循指令、编造事实这类行为。
依据：paper/instructgpt/src/neurips_2021.tex L121

在输入里给几个例子（提示）能让模型照着模仿，但这种行为只用当前这段输入，不改参数。SFT 改用示范数据训练，把这种反应写进模型参数。
依据：paper/instructgpt/src/neurips_2021.tex L121；paper/instructgpt/src/neurips_2021.tex L191

### 数据与损失

SFT 的一条训练样本是一段指令加一段回答。把「指令＋回答」拼成一条序列一次送入，靠[因果掩码](wiki/causal-mask)让每个位置只看到它前面的 token，于是第 $t$ 个位置的状态只依赖它前面的 token，可以用来预测第 $t$ 个 token；一次送入就得到所有位置的预测，再逐位置与真实 token 算损失，用的是下一 token [交叉熵](wiki/cross-entropy)。
依据：paper/llama2/src/templateArxiv.tex L391

#### 样本形态

标注者写过的一条回答，就是模型该给出的输出。为了让序列填到模型固定的长度，训练时把若干条样本的指令和回答依次拼在一起，指令段与回答段之间用一个特殊 token 分隔。
依据：paper/llama2/src/templateArxiv.tex L370；paper/llama2/src/templateArxiv.tex L385；paper/llama2/src/templateArxiv.tex L391

多轮对话的样本另有一种控制方法（Ghost Attention）：把所有轮次合起来算一条样本时，只保留最后一轮回答的损失，前面各轮的 token（包括此前的回答）损失都置零，这样开头那条固定指令（系统消息）只出现在第一条里，也不会和训练时的上下文不一致。
依据：paper/llama2/src/templateArxiv.tex L674-L680

#### 交叉熵与掩码

记一条样本拼成的 token 序列为 $z_1, \dots, z_T$，位置 $t$ 的模型输出是词表上的概率分布 $p_\theta(\cdot \mid z_{<t})$（$\theta$ 是模型的可训练参数），逐位置的交叉熵是 $-\log p_\theta(z_t \mid z_{<t})$；整条序列的损失是这些位置上的平均，$\mathcal{A}$ 是计入损失的位置集合。
依据：paper/llama2/src/templateArxiv.tex L391

::: 从逐位置交叉熵到序列损失
序列的总损失是各位置交叉熵的平均：
$$ \mathcal{L} = -\frac{1}{|\mathcal{A}|} \sum_{t \in \mathcal{A}} \log p_\theta(z_t \mid z_{<t}) $$
依据：paper/llama2/src/templateArxiv.tex L391

$|\mathcal{A}|$ 是计入损失的位置个数。
依据：paper/llama2/src/templateArxiv.tex L391
:::

这里说的掩码是损失掩码，和上面改变可见范围的因果掩码不是一回事：损失掩码把 $\mathcal{A}$ 限定为回答 token 的位置，指令段的 token 不计入损失。
依据：paper/llama2/src/templateArxiv.tex L391

::: 小词表上算一遍掩码前后
设词表只有若干 token，一条样本的序列是：一个指令 token，一个用来分隔指令与回答的特殊 token，一个回答 token。三个位置各自给出下一个 token 的概率如下表。
依据：自行构造

| 位置 | 真实 token | 模型给的概率 | 交叉熵 |
| --- | --- | --- | --- |
| 1 | 指令 token | 0.5 | 0.693 |
| 2 | 特殊 token | 0.25 | 1.386 |
| 3 | 回答 token | 0.125 | 2.079 |

- 把三个位置都计入，平均损失是 $(0.693 + 1.386 + 2.079) / 3 = 1.386$。
- 只把回答 token（位置 3）计入，损失是 $2.079$。
依据：自行构造

```python
import math

probs = [0.5, 0.25, 0.125]          # 三个位置给真实 token 的概率
losses = [-math.log(p) for p in probs]
print("逐位置损失", [round(x, 3) for x in losses])
print("全部位置平均", round(sum(losses) / len(losses), 3))
print("只算回答位置", round(losses[-1], 3))
```

```text
逐位置损失 [0.693, 1.386, 2.079]
全部位置平均 1.386
只算回答位置 2.079
```
:::

### 在训练流程中的位置

SFT 是后训练（预训练之后的训练阶段，继续用数据把模型调到符合使用要求）的第一步，产出一个会按指令作答的模型。之后用人类对多个回答的比较数据训练一个给回答打分的模型（奖励模型），再按它的打分做偏好优化：在 SFT 的基础上继续调整，让模型更常给出被偏好的那类回答。
依据：paper/instructgpt/src/neurips_2021.tex L191-L195

SFT 的监督损失只对齐单个示范回答，没有「同一指令下两个回答哪个更好」这层信息；这层信息由比较数据提供。
依据：paper/instructgpt/src/neurips_2021.tex L191-L195

### 数据量与质量

收集 SFT 数据时，第三方数据来源很多，但其中许多多样性、质量不足，不容易把模型的行为对齐（调到符合使用要求）成对话式回答；改用少量自己请标注者写的高质量数据后，结果提升。
依据：paper/llama2/src/templateArxiv.tex L378-L384

::: 具体数量与标注平台
收集到 27540 条后停止标注。不同标注平台和供应商会让下游模型（用这批数据训练出来的模型）的表现有差别，所以即使把标注外包，也要做数据检查。
依据：paper/llama2/src/templateArxiv.tex L378-L384
:::

### 训练设置

监督微调的训练轮数（把所有训练样本各用一次算一轮）从 2 轮到 16 轮不等，学习率（每步更新参数走的步长）按余弦形状从初始值衰减到原值的约 10%；具体数值随模型不同。
依据：paper/llama2/src/templateArxiv.tex L389-L392；paper/instructgpt/src/neurips_2021.tex L296；paper/instructgpt/src/neurips_2021.tex L1202

::: 两篇各自的配置
- Llama 2（Meta 的对话模型）：初始学习率 $2\times10^{-5}$，权重衰减 0.1（每步把参数按比例往 0 拉一点，防止它变大），批大小 64（一步用多少条样本），序列长度 4096（一条序列最多多少个 token），训练 2 轮。
- InstructGPT（OpenAI 在 GPT-3 这个更早的预训练模型上做指令微调得到的模型）：训练 16 轮，dropout 0.2（训练时按概率把一部分中间层的输出随机关掉）。
依据：paper/llama2/src/templateArxiv.tex L389-L392；paper/instructgpt/src/neurips_2021.tex L296
:::

### 适用条件与边界

训练时验证损失（在没参与训练的样本上算的损失）在第 1 轮之后就转为上升，这叫过拟合（在训练没见过的样本上损失不降反升）；但这不代表应当停止训练：继续训练更多轮，奖励模型的打分和人类偏好都更高。选哪个检查点（某一轮结束时的参数快照），应按奖励模型打分而不是验证损失。
依据：paper/instructgpt/src/neurips_2021.tex L296；paper/instructgpt/src/neurips_2021.tex L1202

数据不是越多越好：达到高质量所需的量是数万条。
依据：paper/llama2/src/templateArxiv.tex L381
