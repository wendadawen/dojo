# Per-Head Muon

- 描述：Per-Head Muon 是把注意力 Q/K/V 投影的动量矩阵沿头维度切块、每块单独做 Newton-Schulz 正交化的 Muon 改法。
- 摘要：把动量矩阵按头切开、逐头正交化，使各头更新尺度均衡；分布式下用 P2P 取回分片替代全参数 all-gather。
- 主题：训练与优化
- 标签：优化器

## 长页

- 开头｜Per-Head Muon 是什么、改哪层参数；一两句引入 Muon 与 Newton-Schulz（前置链接），说明投影权重按头堆叠；点出它要解决的问题｜380｜paper/k3-src/2-model-architecture.tex Per-Head Muon；blog/muon-blog.txt L63-L69
- ### 为什么整块正交化对多头不够好｜资料给出的解释：全矩阵正交化把各头当一个耦合块，大尺度头主导、小尺度头更新未充分正交化；正交化等价于把奇异值拉平为 1；两头例子对照（折叠）｜500｜paper/k3-src/2-model-architecture.tex Per-Head Muon；blog/muon-blog.txt L63-L69
- ### 按头切分并分别正交化｜沿头维度切成 H 个块（每块对应一个头的行块）、每块单独正交化；与全矩阵结果对照（折叠）；逐头正交化的伪代码折叠｜600｜paper/k3-src/2-model-architecture.tex Per-Head Muon
- ### 均衡更新尺度与开销｜均衡的是各头更新幅度，方向仍由各头自己的奇异向量决定；逐头正交化的实践效果；开销略降的原因，本节用一句解释 Gram 矩阵是什么（Gram 规模对照折叠）｜400｜paper/k3-src/2-model-architecture.tex Per-Head Muon
- ### 分布式实现里的 P2P 参数取回｜参数分片与 NS 需要完整矩阵的矛盾、朴素 all-gather 的代价、P2P 取回做法与流水化；分片、all-gather、P2P 在本节一两句解释｜500｜paper/k3-src/5-infrastructure.tex P2P-based Muon orthogonalization
- ### 适用条件与边界｜只对有头维度的投影成立；效果的成立条件；三条常见误解｜250｜paper/k3-src/2-model-architecture.tex Per-Head Muon；paper/k3-src/5-infrastructure.tex P2P-based Muon orthogonalization

## 短页

- ### 按头做的正交化｜是什么：Muon 在注意力投影上把正交化对象改成逐头的改法，改的是投影的动量矩阵；先解释动量矩阵、头、Newton-Schulz（前置链接或一两句）｜250｜paper/k3-src/2-model-architecture.tex Per-Head Muon
- ### 整块正交化的毛病｜为什么需要：资料给出的解释是全矩阵正交化把所有头当一个耦合块，大尺度头主导、小尺度头更新被压低｜280｜paper/k3-src/2-model-architecture.tex Per-Head Muon
- ### 切块、逐头正交｜怎么做：沿头维度把动量矩阵切成每头一块，每块单独做 Newton-Schulz 正交化，再按头拼回；各头更新尺度被拉平｜300｜paper/k3-src/2-model-architecture.tex Per-Head Muon
