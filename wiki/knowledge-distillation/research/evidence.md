# 知识蒸馏（Knowledge Distillation）— 核心论断与证据

编号规则：C 论断 / F 公式 / N 数字。来源优先级：原始论文 > 权威综述 > 教程/博客（仅作辅助定位，不作核心论断唯一依据）。

主要来源：

- Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531 [stat.ML], 2015. NIPS 2014 Deep Learning Workshop. 简记为 **HVD15**。
  - URL: https://arxiv.org/abs/1503.02531

## 核心论断

### C1：硬标签丢弃类间关系信息

- **论断内容**：硬标签（one-hot）将所有非目标类概率置零，丢掉了教师 softmax 输出中携带的"非目标类之间的相对概率"——例如一张"7"的图，教师 softmax 可能在 "9" 上有比 "2" 更高的概率，说明 7 与 9 在表示空间中更接近；这部分信息 Hinton 称为 "dark knowledge"（暗知识）。
- **来源定位**：HVD15 摘要、§1 引言、§2 "The simplest form of distillation"。
- **适用条件**：教师输出为概率分布而非 one-hot；多分类任务。
- **置信状态**：已确认。

### C2：温度 softmax 公式

- **论断内容**：教师与学生的 logits 经过温度 $T$ 软化后输出概率 $q_i = \exp(z_i/T) / \sum_j \exp(z_j/T)$，$T>1$ 时分布比标准 softmax 更平。
- **来源定位**：HVD15 §2，公式直接给出。
- **适用条件**：$T > 0$；$T=1$ 退化为标准 softmax。
- **置信状态**：已确认。

### C3：温度 $T$ 的极限行为

- **论断内容**：$T \to 0^+$ 时温度 softmax 趋向 one-hot argmax；$T=1$ 即标准 softmax；$T \to \infty$ 时趋向均匀分布 $1/K$。$T$ 越大非目标类概率被放大得越多。
- **来源定位**：HVD15 §2 描述与公式直接推导。
- **适用条件**：logits 有限且不全相等。
- **置信状态**：已确认。

### C4：温度选择与师生容量的关系

- **论断内容**：Hinton 报告中 MNIST 主实验取 $T=20$（教师为 2×1200 dropout 大模型）；当学生容量很小时（例如每层仅 30 单元）最佳 $T$ 落在 $T \approx 2.5$–$4$ 之间——大 $T$ 暴露过多细微信息学生难以吸收。论文未给出"教师—学生容量—最佳 $T$"的解析公式。
- **来源定位**：HVD15 §2、§3 MNIST 实验。
- **适用条件**：MNIST 设置，2-layer MLP。
- **置信状态**：已确认。

### C5：KD 总损失（标准形式）

- **论断内容**：总损失为软目标项与硬标签项的加权和：

$$\mathcal{L} = \alpha \cdot T^2 \cdot \mathrm{KL}\!\left(p_T^{\text{teacher}}\,\|\,p_T^{\text{student}}\right) + (1-\alpha)\cdot \mathrm{CE}\!\left(y,\,p_1^{\text{student}}\right)$$

  其中 $p_T$ 是温度 $T$ 下的概率分布，$p_1$ 是 $T=1$ 下的标准 softmax 分布，$y$ 是硬标签。HVD15 原始论文以 cross-entropy 形式书写软项（教师分布固定时 CE 与 KL 相差一个常数），现代实现普遍用 KL。
- **来源定位**：HVD15 §2，方程形式（原论文未使用 $\alpha$、$\mathrm{KL}$ 符号，但表达式等价）。
- **适用条件**：师生共享输出空间；$\alpha \in [0, 1]$。
- **置信状态**：已确认（公式等价性由 cross-entropy / KL 的标准关系推出）。

### C6：$T^2$ 缩放因子

- **论断内容**：温度把 logits 缩小为 $z/T$，导致软损失对原始 logits 的梯度被压低 $1/T^2$ 倍；乘以 $T^2$ 把梯度恢复到与硬项可比的量级，使得固定 $\alpha$ 时改变 $T$ 不会偷偷改变软硬两项的相对权重。论文原文明确给出这一解释。
- **来源定位**：HVD15 §2 末段，"Since the magnitudes of the gradients produced by the soft targets scale as $1/T^2$ ..."。
- **适用条件**：使用软硬两项加权时。
- **置信状态**：已确认。

### C7：MNIST 实验结论

- **论断内容**：在 MNIST 上，2×1200 dropout 大教师测试错误 67 个；2×800 学生只用硬标签训练错误 146 个；2×800 蒸馏学生（$T=20$）错误降至 74 个，几乎追平教师。
- **来源定位**：HVD15 §3 表格。
- **适用条件**：MNIST 数据集，2-layer MLP，$T=20$ 蒸馏。
- **置信状态**：已确认。

### C8：MNIST "无数字 3" 实验

- **论断内容**：Hinton 在迁移训练集中移除所有数字 3 的样本后重新蒸馏，经偏差校正后学生仍能正确分类 996/1010 个测试集中的 3，证明软分布确实携带了"3 在类别空间中相对其他数字的位置"这一信息——硬标签训练（无任何 3 样本）根本无法做到。
- **来源定位**：HVD15 §3。
- **适用条件**：MNIST，soft target 训练集无 3。
- **置信状态**：已确认。

### C9：语音识别实验结论

- **论断内容**：在 Android 语音搜索风格的声学模型上（8 层 LSTM、每层 2560 单元、14,000 路 HMM 状态、约 85M 参数、约 2000 小时英语语音、约 700M 帧），10 模型 ensemble 帧准确率 61.1%、WER 10.7%；蒸馏得到的单一模型帧准确率 60.8%、WER 10.7%，几乎完全吸收了 ensemble 的 WER 增益。
- **来源定位**：HVD15 §4。
- **适用条件**：声学建模任务，10 个模型 ensemble。
- **置信状态**：已确认。

### C10：学生质量以教师为上限

- **论断内容**：Hinton 实验中蒸馏学生最多接近教师能力（MNIST 74 vs 教师 67），未报告显著超越教师；学生通过 KD 学到的是教师的能力上限而非更高。要超过教师需要额外的非蒸馏训练信号（任务奖励、RL 探索等）。
- **来源定位**：HVD15 §3–§4 实验结论直接推出；MOPD 概念页对 on-policy 变体亦有相同边界结论。
- **适用条件**：标准 off-policy KD。
- **置信状态**：已确认。

### C11：与 MOPD 的关系

- **论断内容**：标准 KD（Hinton 2015）是 off-policy——教师先生成完整序列/分布，学生在固定教师输出上拟合；K3 的 MOPD 是 on-policy 变体——学生自己生成 token，教师对学生的每个 token 评分作为稠密奖励，在 RL 框架内完成蒸馏。MOPD 的完整机制由 [MOPD 概念页](../../wiki/mopd/index.html) 展开。
- **来源定位**：HVD15 §2（off-policy 设定）；K3 报告 §4.1.3 与 [MOPD 概念页](../../wiki/mopd/index.html) §scope。
- **适用条件**：标准 KD 为基线，MOPD 为变体。
- **置信状态**：已确认。

### C12：高温零均值极限下的 logit matching

- **论断内容**：在高温极限（$T \to \infty$）且 logits 零均值（$\sum_i z_i = 0$）下，温度 softmax 的 KL 软损失对 logits 的梯度近似为 $\frac{1}{N T^2}(z_i - v_i)$（$z_i$ 学生 logit、$v_i$ 教师 logit、$N$ 类别数），此时 KD 等价于最小化 $\frac{1}{2N}\|z - v\|^2$，即"软目标 KD 退化为 logit matching"。
- **来源定位**：HVD15 §2 末段推导。
- **适用条件**：$T \to \infty$ 且 $\sum_i z_i = 0$；实际有限 $T$ 下是近似。
- **置信状态**：已确认（论文给出推导）。

## 核心公式

### F1：温度 softmax

$$q_i = \frac{\exp(z_i/T)}{\sum_{j=1}^{K} \exp(z_j/T)},\qquad i \in \{1, \ldots, K\}$$

来源：HVD15 §2。

### F2：KD 总损失

$$\mathcal{L} = \alpha \cdot T^2 \cdot \mathrm{KL}\!\left(p_T^{\text{teacher}}\,\|\,p_T^{\text{student}}\right) + (1-\alpha)\cdot \mathrm{CE}\!\left(y,\,p_1^{\text{student}}\right)$$

来源：HVD15 §2。$T^2$ 缩放来自论文 §2 末段。

### F3：KL 散度（最小定义）

$$\mathrm{KL}(p\,\|\,q) = \sum_i p_i \ln\frac{p_i}{q_i}$$

来源：信息论标准定义，HVD15 §2 隐含使用。

### F4：高温零均值极限下梯度

$$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_i} \approx \frac{1}{N T^2}(z_i - v_i) \quad (T \to \infty,\, \sum_i z_i = 0)$$

来源：HVD15 §2 末段。乘 $T^2$ 后梯度为 $\frac{1}{N}(z_i - v_i)$，揭示 KD 与 logit matching 的关系。

## 外部数字

### N1：MNIST 错误数

| 模型 | 测试错误数 |
|---|---:|
| 2×1200 dropout 教师 | 67 |
| 2×800 硬标签学生 | 146 |
| 2×800 蒸馏学生（$T=20$） | 74 |

来源：HVD15 §3。

### N2：MNIST "无数字 3" 实验

经偏差校正后 996/1010 个测试集中的 3 被正确分类。来源：HVD15 §3。

### N3：语音识别性能

| 系统 | 帧准确率 | WER |
|---|---:|---:|
| 基线单模型 | 58.9% | 10.9% |
| 10 模型 ensemble | 61.1% | 10.7% |
| 蒸馏单模型 | 60.8% | 10.7% |

声学模型：8 层、每层 2560 单元、14,000 路 HMM 状态、约 85M 参数、约 2000 小时英语语音、约 700M 帧级训练样本。来源：HVD15 §4。

### N4：温度取值

- MNIST 主实验：$T=20$。
- 小容量学生（每层 30 单元）：$T \approx 2.5$–$4$。

来源：HVD15 §2、§3。

### N5：$\alpha$ 取值倾向

Hinton 报告中硬标签项权重 $(1-\alpha)$ 通常需要远小于软项权重 $\alpha$，即 $\alpha$ 接近 1。论文未给出统一最优值，工程上常用 $\alpha \in [0.7, 0.9]$。

来源：HVD15 §2、§3 工程结论。
