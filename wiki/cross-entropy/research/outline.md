# 交叉熵

- 描述：交叉熵是度量「用一个预测分布去描述真实结果」的平均代价，训练分类器和语言模型时最小化的就是它，对真实标签取 one-hot 时化简为真实类预测概率的负对数。
- 摘要：把「模型认为真实结果有多可能」变成可求导的代价：$H(P,Q) = -\mathbb{E}_{x\sim P}\log Q(x)$；one-hot 标签下化简为 $-\log q(y)$，最小化它等价于最大似然。
- 主题：训练与优化, 概率与统计
- 标签：训练

## 长页

- 开头｜用「预测天气晴的概率 0.8、第二天确实晴了」引入：模型输出的是概率，训练要一个数告诉它这一步差多少；这个数按模型给真实结果留了多少概率来算，就是交叉熵｜350｜blog/deeplearningbook-prob.txt L1854-L1872；blog/deeplearningbook-ml.txt L2746-L2752
- ### 信息量为什么取负对数｜三条要求（确定事件为零、越罕见越大、独立事件相加）推算 $-\log P(x)$ 的过程折叠；讲清这个量叫自信息、对数底与 nats｜200｜blog/deeplearningbook-prob.txt L1683-L1721
- ### 熵与 KL 散度｜熵是自信息的期望；离散变量下 KL 是用 $Q$ 的编码传 $P$ 多出的平均位数、非负不对称；各两三句定义，展开与手算折叠｜420｜blog/deeplearningbook-prob.txt L1728-L1746；blog/deeplearningbook-prob.txt L1751-L1815
- ### 交叉熵｜定义 $H(P,Q)=-\mathbb{E}_{x\sim P}\log Q(x)$、等于熵加 KL、最小化它等价于最小化 KL、$0\log 0$ 惯例｜600｜blog/deeplearningbook-prob.txt L1854-L1883
- ### 训练里的交叉熵｜训练集定义经验分布，最大似然化简成最小化交叉熵的推导折叠；讲清 one-hot 后期望只剩真实类一项；二分类、多分类手算折叠｜680｜blog/deeplearningbook-ml.txt L2615-L2752
- ### 下一 token 的交叉熵｜各一句话讲清 token、词表、语言模型、softmax；每位置输出词表分布，取真实下一 token 的负对数，一段按预测单元平均；小词表例子折叠｜550｜blog/deeplearningbook-ml.txt L2775-L2791；paper/gpt2.txt L226-L235
- ### 适用条件与边界｜最小化只作用在模型分布 $Q$ 上；常见误解是只有 softmax 或伯努利输出才有交叉熵，任何负对数似然都是交叉熵｜300｜blog/deeplearningbook-prob.txt L1870-L1876；blog/deeplearningbook-ml.txt L2746-L2752

## 短页

- ### 用预测概率度量代价｜是什么：按模型给真实结果的概率算代价，概率越低代价越大，等于按真实分布对负对数取平均｜290｜blog/deeplearningbook-prob.txt L1683-L1697；blog/deeplearningbook-prob.txt L1854-L1872
- ### 最小化交叉熵就是最大似然｜为什么需要：训练要让已见数据出现得最可能（最大似然），而最大化似然与最小化交叉熵是同一件事｜300｜自行构造
- ### 只算真实类那一项｜怎么做：训练标签是只把真实类标 1 的 one-hot 向量，按它取平均只剩真实类一项，损失就是模型给真实类概率的负对数｜290｜blog/deeplearningbook-ml.txt L2658-L2752
