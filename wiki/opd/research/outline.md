# OPD（On-Policy Distillation）

- 描述：OPD 在训练时让学生模型自己生成序列，教师模型对学生实际到达的状态、学生实际写下的每个 token 给出对数概率比分数，再把这个逐 token 分数当作 advantage 进入策略梯度更新学生。
- 摘要：训练序列由学生自己采样，监督是教师逐 token 的对数概率比 $r=\log\pi_{\text{teacher}}(y_t)-\log\pi_S(y_t)$，它充当策略梯度里的 advantage；逐 token 的稠密反馈把每个 episode 的信息量从与 token 数无关的 $O(1)$ 提高到 $O(N)$。
- 主题：训练与优化, 模型结构
- 标签：训练

## 长页

- 开头｜概览 OPD 的学生自采样与教师逐 token 打分，为什么需要它：训练推理分布失配与 RL 奖励稀疏；顺带用一句给出 token 与学生/教师策略的基线外释义｜260｜blog/tml-on-policy-distillation.txt L63-L65；blog/tml-on-policy-distillation.txt L71；blog/tml-on-policy-distillation.txt L79；blog/tml-on-policy-distillation.txt L91
- ### 在别人写的序列上训练，推理要接自己的序列｜train-inference 分布失配：训练见到的前缀与推理时自己生成的不同，早期错误被复合放大，长序列更严重｜440｜paper/gkd-v3.txt L184；paper/gkd-v3.txt L108；blog/tml-on-policy-distillation.txt L79
- ### 把训练分布换成学生自己的分布｜on-policy 与 off-policy 的分界，GKD 用学生数据比例 $\lambda$ 混合两端，散度逐位置比较再对长度取平均（公式折叠）｜500｜paper/gkd-v3.txt L241-L248；paper/gkd-v3.txt L186；paper/gkd-v3.txt L163
- ### 教师在每个 token 上打分｜一步机制：学生采样、教师一次前向打分、逐 token 取负的 reverse KL 作 advantage 进策略梯度；基线外词一句释义｜620｜blog/tml-on-policy-distillation.txt L91；blog/tml-on-policy-distillation.txt L119；blog/tml-on-policy-distillation.txt L139；blog/tml-on-policy-distillation.txt L141；blog/tml-on-policy-distillation.txt L285
- ### 一个手算的逐 token 打分｜三 token 词表下逐个算出分数、判断抬高压低，并核对期望等于负的 reverse KL；例子全过程折叠｜100｜自行构造
- ### 适用条件与边界｜成立条件（可查 logprobs、覆盖支持集否则需更大 batch，两词一句释义）、只搬教师已有策略、可复用同一 prompt（RL 会退化成记答案）｜420｜blog/tml-on-policy-distillation.txt L277；blog/tml-on-policy-distillation.txt L303；blog/tml-on-policy-distillation.txt L285；blog/tml-on-policy-distillation.txt L325；blog/tml-on-policy-distillation.txt L309

## 短页

- ### 写自己的句子，让教师逐字批改｜OPD 是什么：学生自己生成序列，教师对学生实际写下的每个 token 给出对数概率比分数作训练信号（token 一词释义，只给定义，不讲打分怎么算）｜180｜blog/tml-on-policy-distillation.txt L91；blog/tml-on-policy-distillation.txt L111
- ### 训练时见不到自己写的前缀｜为什么需要：在教师或数据集写的序列上训练，学生在推理时要接自己写的前缀，一旦早期走偏就进了训练没见过的状态，误差越滚越大｜300｜blog/tml-on-policy-distillation.txt L79；paper/gkd-v3.txt L184
- ### 采样、逐 token 打分、进策略梯度｜怎么做：学生采样、教师一次前向给 logprobs，逐 token 对数概率比取负作 advantage，按策略梯度更新学生，基线外词一句释义｜330｜blog/tml-on-policy-distillation.txt L119；blog/tml-on-policy-distillation.txt L139；blog/tml-on-policy-distillation.txt L141
