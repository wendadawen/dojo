# MOPD（Multi-Teacher On-Policy Distillation）

- 描述：MOPD 是一种后训练算法，它按领域各自用强化学习训出多个专项教师模型，再让学生在自采样轨迹上接受对应领域教师逐 token 的监督，把多项能力合并进一个统一的学生模型。
- 摘要：每个领域的强化学习教师冻结后按 prompt 路由给学生打分，学生最小化逐 token 反向 KL $\mathbb{E}_{x,y\sim\pi_\theta}\big[\tfrac{1}{|y|}\sum_t\sum_v \pi_\theta(v)\log\frac{\pi_\theta(v)}{\pi_{\phi_d}(v)}\big]$ 合入多项能力，无需在权重空间平均。
- 主题：训练与优化
- 标签：训练

## 长页

- 开头｜交代后训练、强化学习、策略、轨迹、token、蒸馏的释义，并点明 on-policy 指训练轨迹由学生当前策略生成；MOPD 是什么、为什么需要它｜340｜paper/mopd-v1.txt L16-L27；paper/mopd-v1.txt L38-L40；paper/mopd-v1.txt L67-L77
- ### 后训练里的能力集成问题｜强化学习按领域各自改进能力，跨领域联合训练互相干扰；旧做法分四类（混数据联合 RL、按域顺序训练、在教师轨迹上离线微调、平均模型参数），各有缺口｜420｜paper/mopd-v1.txt L38-L40；paper/mopd-v1.txt L41-L66
- ### 三段流水线：SFT、按领域 RL、蒸馏｜Stage 1 共享 [SFT](wiki/sft)、Stage 2 各领域独立并行训出教师、Stage 3 教师冻结并合成学生；教师同源于 SFT 起点｜420｜paper/mopd-v1.txt L128-L142；paper/mopd-v1.txt L169-L173
- ### 用逐 token 反向 KL 合成学生｜一步机制的顺序：采样、按领域标签路由、学生生成、教师回填、按逐 token 反向 KL 更新；解释 reverse KL 与「学生采样、教师打分」模板｜480｜paper/mopd-v1.txt L176-L182；paper/mopd-v1.txt L207-L218；paper/mopd-v1.txt L114-L116
- #### 折成策略梯度：教师与学生之差当 advantage｜把反向 KL 的梯度写成策略梯度形式，教师与学生之差充当逐 token advantage 并两侧裁剪；解释策略梯度、advantage、裁剪；推导折叠｜220｜paper/mopd-v1.txt L220-L239
- #### 只取教师 top-k 的实现｜在教师的 top-k token 上做蒸馏，损失多一项 $\pi_{\phi_d}(v)-\pi_\theta(v)$ 修正截断偏差；解释 top-k；公式折叠｜200｜paper/mopd-v1.txt L241-L263
- ### 教师回填不增加训练时间｜把教师部署成独立的回填服务放在 RL 训练器之外，与采样异步重叠，教师开销藏在采样背后｜200｜paper/mopd-v1.txt L265-L279
- ### 一个手算的逐步打分｜三 token 词表下逐个教师与学生分布算出 advantage、判断抬高压低，并核对策略梯度权重等于 $\pi_\theta$；例子全过程折叠｜150｜自行构造
- ### MOPD 的结构优势与同源教师｜MOPD 为何有效：密集监督降方差提样本效率、学生自采样消除 exposure bias、策略空间集成更稳；同源教师为何更稳、两种损失相当；数字折叠｜340｜paper/mopd-v1.txt L184-L193；paper/mopd-v1.txt L485-L523；paper/mopd-v1.txt L414-L428
- ### 适用条件与边界｜成立条件（同源教师、初始 KL 低、可查对数概率）；每领域仍留缺口，MiMo 上两处回退；参数合并跨领域不稳；两条被实验否定的想法各点名；数字折叠｜300｜paper/mopd-v1.txt L521-L523；paper/mopd-v1.txt L409-L411；paper/mopd-v1.txt L375-L379；paper/mopd-v1.txt L394-L397；paper/mopd-v1.txt L485-L523；paper/mopd-v1.txt L414-L428

## 短页

- ### 把多个专项教师并成一个学生｜MOPD 是什么：多个领域 RL 教师合入一个学生；释义 RL 教师、策略、轨迹、token｜260｜paper/mopd-v1.txt L16-L27；paper/mopd-v1.txt L176-L182
- ### 一个模型难以同时做好多个领域｜为什么需要：RL 按领域分别变强，而跨领域联合训练会互相干扰、留下缺口，想要的是一个模型在全领域都强｜300｜paper/mopd-v1.txt L38-L40；paper/mopd-v1.txt L41-L66
- ### 采样、路由、回填、更新｜怎么做：按领域并行训教师、学生自采样、按提示路由、教师回填对数概率、逐 token 反向 KL 更新；解释反向 KL；不写策略梯度与 top-k｜340｜paper/mopd-v1.txt L128-L142；paper/mopd-v1.txt L176-L182；paper/mopd-v1.txt L207-L218
