# Muon

- 描述：Muon 是用于神经网络隐藏层二维参数的优化器：先把 SGD 动量得到的更新矩阵用 Newton-Schulz 迭代正交化，再按学习率更新参数。
- 摘要：把二维参数的更新矩阵替换成最接近它的半正交矩阵，再按学习率更新参数；正交化用 $5$ 步 Newton-Schulz 迭代近似算出。
- 主题：训练与优化
- 标签：优化器

## 长页

- 开头｜Muon 是什么、为什么需要它、大概怎么工作；就地讲清神经网络隐藏层、二维参数、动量｜380｜blog/muon-blog.txt L3-L13；blog/muon-blog.txt L63-L69；blog/muon-blog.txt L71-L78
- ### 正交化的更新｜本章讲核心操作：把更新换成最接近的半正交矩阵，以及为何有效；内容在三个小节｜0｜blog/muon-blog.txt L63-L78
- #### 更新矩阵为何近乎低秩｜人工检查发现 transformer 这类语言模型里二维参数的更新矩阵通常条件数很高、几乎低秩，少数方向主导；就地讲清条件数；折叠：作者推测正交化抬高次要方向｜430｜blog/muon-blog.txt L78
- #### 正交化是什么｜半正交矩阵的含义、SVD 取 $UV^\top$ 即最接近的半正交矩阵（就地点清 SVD 或链接）；折叠：2×2 例｜480｜blog/muon-blog.txt L65-L69
- #### 用迭代算出正交矩阵｜不做 SVD；用 5 步 Newton-Schulz 迭代近似算出，能在低精度浮点格式 bfloat16 上跑，细节见其页面｜320｜blog/muon-blog.txt L17；blog/muon-blog.txt L84-L90；blog/muon-blog.txt L129
- ### 完整更新规则｜把动量、正交化、步长合写成一次更新；就地讲清 Nesterov 动量、学习率、形状缩放因子；折叠：更新公式｜450｜blog/muon-blog.txt L63；blog/muon-blog.txt L194；repo/Muon/muon.py L34-L41；repo/Muon/muon.py L91-L93
- ### 哪些参数用它｜只用于二维隐藏层权重；标量向量与输入输出层交给标准方法（如 AdamW）；卷积核展平、QKV 分开｜380｜blog/muon-blog.txt L13；blog/muon-blog.txt L38-L39；blog/muon-blog.txt L192；blog/muon-blog.txt L195-L196
- ### 运行开销｜正交化额外开销很小；折叠：额外 FLOP 上限与占比 $Tm/B$、两个训练场景的数字｜350｜blog/muon-blog.txt L135-L149
- ### 适用条件与边界｜开销低于 $1\%$ 的适用范围、正交化让每步变慢的代价、资料未下结论的三个问题｜400｜blog/muon-blog.txt L143-L149；blog/muon-blog.txt L210；blog/muon-blog.txt L214-L216

## 短页

- ### 沿正交方向走一步｜Muon 是什么：用于隐藏层二维参数的优化器，就地讲清隐藏层、二维参数、半正交矩阵、动量｜300｜blog/muon-blog.txt L13；blog/muon-blog.txt L63-L69
- ### 少数方向主导的更新｜为什么需要：人工检查发现更新矩阵几乎低秩、少数方向主导，作者推测正交化抬高次要方向｜250｜blog/muon-blog.txt L78
- ### 动量、迭代、步长｜怎么做：累积动量、用 Newton-Schulz 迭代正交化、按学习率更新，三步各一句｜350｜blog/muon-blog.txt L63；blog/muon-blog.txt L17；blog/muon-blog.txt L129
