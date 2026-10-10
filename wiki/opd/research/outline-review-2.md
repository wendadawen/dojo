# 审大纲 第 2 轮

## 读者

| 类别 | 位置 | 问题 | 依据 | 改法 |
|---|---|---|---|---|
| 顺序错乱 | 短页 写自己的句子，让教师逐字批改 | 「token」没说清 | 短页三节都只用不解释，长页也没有节点讲过 | 首次出现处用一句释义，或作为前置概念链接成页 |
| 顺序错乱 | 短页 采样、逐 token 打分、进策略梯度 | 「logprobs」「前向」没说清 | 此前节点都没解释 | 首次出现处各补一句最小定义 |
| 多余 | 长页 把训练分布换成学生自己的分布 | GKD 的 λ 混合是别的框架的内部机制 | 「GKD 用学生数据比例 λ 混合两端」 | GKD 只保留「提出 on-policy KD 框架」一句 |

## 对照资料

| 类别 | 位置 | 问题 | 依据 | 改法 |
|---|---|---|---|---|
| 资料里没有（依据缺） | 长页 教师在每个 token 上打分 | 「单次前向」「取负作 advantage」在所引 L133-L139 之外 | blog L119「requires just a single forward pass」、L141「per-token advantage to the negative reverse KL」 | 补 blog L119、L141 |
| 资料里没有（位置错） | 长页 适用条件与边界 | 「不发明新能力」在所引四处中无依据 | 出自 blog L285「pruning the space of actions present in the base policy」、L325「only the final strategy learned」 | 依据补 blog L285、L325 |
| 资料里没有（口径错） | 长页 教师在每个 token 上打分 | 「对数概率比取负」与摘要「$r$ 充当 advantage」符号相反 | blog L155-L156「reverse_kl = sampled_logprobs - teacher_logprobs；advantages = -reverse_kl」 | 与摘要统一：$r=\log\pi_{\text{teacher}}-\log\pi_S$，advantage 是「取负的 reverse KL」即 $r$ |

## 篇幅与冗余

无（重复、无关、预算不合理、短页不合格均无）。

## 处理

- 读者「token 未解释」：成立。token 是基线外词。在长页开头与短页第 1 节首次出现处用一句释义（「语言模型处理的文本最小单位」），不展开、不另设页。→ 已改大纲「写什么」，标注这一句。
- 读者「logprobs／前向未解释」：成立。在机制节（长页）与短页第 3 节首次出现处各补一句最小定义（logprobs＝模型对词表各位置给的对数概率；前向＝模型按输入算一遍输出）。→ 已改。
- 读者「删 GKD λ 节」：**已由第 1 轮仲裁判「不必改」**（`paper/gkd-v3.txt` L241-L248：on-policy KD 是 GKD 的 $\lambda=1$ 端点，属 OPD 自身定义）。不再采纳；已在派发时列入「不要再提」。
- 对照资料「机制节依据缺」：成立，依据补 `blog … L119`、`blog … L141`。
- 对照资料「边界节不发明新能力位置错」：成立，依据补 `blog … L285`、`blog … L325`。
- 对照资料「advantage 符号相反」：**不成立，属表述未对齐**。blog L111 的 reverse KL 取负后等于 $\log\pi_{\text{teacher}}-\log\pi_S$，正是摘要里的 $r$；两处说的是同一个量。把机制节的表述统一为「取负的 reverse KL」，不再单说「对数概率比取负」，避免读成相反符号。→ 已改大纲。

改完重新运行 `check_outline.py`，通过。
