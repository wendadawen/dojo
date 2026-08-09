# 知识蒸馏（Knowledge Distillation）— 术语表

登记全文所有首次出现的术语、缩写和符号。保证全文含义一致，防止同一对象出现多种记号或术语漂移。

## 术语

| 术语 | 首次出现 | 定义或含义 |
|---|---|---|
| 知识蒸馏（Knowledge Distillation, KD） | 标题、S1 | 训练学生模型匹配教师模型的软概率分布，将教师能力迁移到更小的学生。本文特指 Hinton 2015 的 logit/输出层蒸馏。 |
| 教师（teacher） | S1 | 已训练好的大模型，输出软标签作为监督信号。记 logits 为 $v$。 |
| 学生（student） | S1 | 待训练的小模型，目标是模仿教师。记 logits 为 $z$。 |
| 硬标签（hard label / hard target） | S1 | one-hot 形式的真实标签，非目标类全为零。记为 $y$。 |
| 软标签（soft label / soft target） | S1 | 教师在温度 $T$ 下输出的概率分布，非目标类有非零概率。 |
| 暗知识（dark knowledge） | S1 | 教师软分布中非目标类的相对概率所携带的类间关系信息，硬标签丢弃。 |
| 温度 softmax（temperature softmax） | S2 | 用温度 $T$ 软化 logits 的 softmax：$q_i = \exp(z_i/T)/\sum_j \exp(z_j/T)$。 |
| 软化（softening） | S2 | 通过增大 $T$ 让分布变平、放大非目标类概率的操作。 |
| 软损失（soft loss） | S4 | KD 损失中匹配教师软分布的项，使用 KL 散度。 |
| 硬损失（hard loss） | S4 | KD 损失中匹配硬标签的项，使用标准交叉熵。 |
| KL 散度（Kullback–Leibler divergence） | S4 | 衡量两个分布差异的非对称度量，$\mathrm{KL}(p\|q) = \sum_i p_i \ln(p_i/q_i)$。 |
| 交叉熵（cross-entropy, CE） | S4 | $\mathrm{CE}(y, p) = -\sum_i y_i \ln p_i$，硬标签训练的标准损失。 |
| logit matching | S4 折叠 | 高温零均值极限下 KD 退化为最小化 $\|z-v\|^2/2$ 的等价形式。 |
| off-policy 蒸馏 | S5 | 教师生成完整序列/分布、学生在固定教师输出上拟合的标准 KD。 |
| on-policy 蒸馏 | S5 | 学生自生成轨迹、教师对学生输出逐 token 评分的变体；MOPD 属于此类。 |
| MOPD（Multi-Teacher On-Policy Distillation） | S5 | K3 提出的多教师在线策略蒸馏，on-policy 蒸馏的代表；完整机制见 [MOPD 概念页](../../wiki/mopd/index.html)。 |

## 缩写

| 缩写 | 全称 | 首次出现 |
|---|---|---|
| KD | Knowledge Distillation | 标题 |
| MOPD | Multi-Teacher On-Policy Distillation | S5 |
| KL | Kullback–Leibler | S4 |
| CE | Cross-Entropy | S4 |
| WER | Word Error Rate | S5 |
| HVD15 | Hinton, Vinyals, Dean 2015（论文简称，仅用于来源与教学说明） | 文末 |

## 符号

| 符号 | 含义 | 首次出现 |
|---|---|---|
| $K$ | 类别总数 | S2 |
| $z_i$ | 学生给第 $i$ 类的 logit（任意实数） | S2 |
| $v_i$ | 教师给第 $i$ 类的 logit（任意实数） | S4 |
| $T$ | 温度参数，$T > 0$；$T=1$ 标准 softmax；$T>1$ 软化 | S2 |
| $q_i$ | 温度 softmax 输出的第 $i$ 类概率 | S2 |
| $p_T^{\text{teacher}}$ | 教师在温度 $T$ 下的概率分布 | S4 |
| $p_T^{\text{student}}$ | 学生在温度 $T$ 下的概率分布 | S4 |
| $p_1^{\text{student}}$ | 学生在 $T=1$ 下的标准 softmax 分布，用于硬标签项与推理 | S4 |
| $y$ | 硬标签（one-hot 向量） | S4 |
| $\alpha$ | 软损失项权重，$\alpha \in [0,1]$；Hinton 报告中通常接近 1 | S4 |
| $1-\alpha$ | 硬损失项权重 | S4 |
| $T^2$ | 软损失项前的缩放因子，补偿温度对 logits 的梯度压低效应 | S4 |
| $\mathrm{KL}(p\|q)$ | $p$ 相对 $q$ 的 KL 散度 | S4 |
| $\mathrm{CE}(y, p)$ | $y$ 与 $p$ 的交叉熵 | S4 |
| $N$ | 高温极限推导中的类别数（与 $K$ 同义，仅推导中沿用论文记号） | S4 折叠 |

## 全文一致性约束

- 教师 logits 全文统一用 $v$；学生 logits 全文统一用 $z$。不混用 $w, x, \theta$ 等。
- 温度统一用 $T$，不写作 $\tau$。
- 软损失项权重统一用 $\alpha$（接近 1 时软项主导），不与"硬损失权重"混用——硬损失权重明确写为 $1-\alpha$。
- "训练温度"与"推理温度"在 S4 明确区分：训练用 $T$，推理与硬损失项用 $T=1$。
- KL 散度方向统一为 $\mathrm{KL}(p_T^{\text{teacher}} \| p_T^{\text{student}})$（教师在前，学生在后），不混用反向。
