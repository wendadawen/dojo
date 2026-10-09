# iHC（identity Hyper-Connections）

- 描述：iHC 是 Hyper-Connections 的一个变体：残差流扩成 n 条，流间的残差混合映射在 Hy4 实现里等价于恒等。
- 摘要：把残差流从一条扩成 $n$ 条，每层的输入由 $n$ 条流汇合而得，层的输出再分发回各条流；iHC 中流间的残差混合映射等价于恒等，省掉它的矩阵乘法。
- 主题：模型结构
- 标签：模型架构

## 长页

- 开头｜残差连接一句话；HC 把残差流扩成 n 条、给每层输入输出配权重；iHC 把流间残差混合取成恒等；腾讯混元的 Hy4 preview 里 n=4｜400｜paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/hc-2409.19606v3.txt L120-L175；paper/src-v3/app.tex Pre-Norm Residual Connection；repo/hy4-transformers-modeling.py L603-L607；repo/hy4-transformers-configuration.py L47-L52；blog/hy4-model-card.md L63-L84
- ### 残差流与它的两条通路｜残差流是什么；Pre-Norm 与 Post-Norm 各自的写法与失效方式，连接强度预先固定｜250｜paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections；paper/src-v3/app.tex Pre-Norm Residual Connection；paper/src-v3/app.tex Post-Norm Residual Connection；paper/hc-2409.19606v3.txt L68-L76；paper/hc-2409.19606v3.txt L113-L115
- ### 把残差扩成 n 条流｜n 条流的记号与形状、三组映射各是什么、更新式；残差连接是 n=1 的特例；纯公式推导折叠｜400｜paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections；paper/src-v3/app.tex Pre-Norm Residual Connection；paper/hc-2409.19606v3.txt L182-L192；paper/hc-2409.19606v3.txt L294-L318；paper/dsv4-2606.19348v1.txt L355-L397
- #### 一层的前向顺序｜入口把输入复制成 n 条流；每层先汇入、再过层、再分发；一个块里两个 HC；HC 原式末端按行求和；手算折叠｜150｜paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/hc-2409.19606v3.txt L182-L192；repo/hy4-transformers-modeling.py L667-L716；paper/dsv4-2606.19348v1.txt L365-L385
- #### 映射的参数从哪来｜三组权重由当前流经归一化与线性层计算得到、加静态偏置；HC 原始形式用 tanh，DeepSeek-V4 与 Hy4 用 Sigmoid；公式折叠｜200｜paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections；paper/src-v3/iclr2025_conference.tex Initialization；paper/dsv4-2606.19348v1.txt L393-L395；paper/dsv4-2606.19348v1.txt L398-L424；repo/hy4-transformers-modeling.py L603-L641
- ### Identity Hyper-Connections｜Hy4 实现里残差映射等价于恒等，省下它的矩阵乘法；与 mHC 的区别一两句，含 mHC 针对 HC 深层堆叠数值不稳定的动机；两组权重的算法折叠｜350｜repo/hy4-transformers-modeling.py L603-L641；repo/hy4-transformers-modeling.py L643-L664；repo/hy4-transformers-modeling.py L667-L716；repo/hy4-transformers-configuration.py L47-L52；paper/dsv4-2606.19348v1.txt L380-L383；paper/dsv4-2606.19348v1.txt L386-L431
- ### 效果与代价｜参数与计算增量可忽略；1B 稠密、7B MoE 上的结果，n=1 反而不如基线、n=4 与 n=8 相差很小；实验设置与数字折叠｜250｜paper/src-v3/iclr2025_conference.tex Ablation Study；paper/src-v3/iclr2025_conference.tex MoE Models；paper/hc-2409.19606v3.txt L409-L422；paper/hc-2409.19606v3.txt L465-L470；paper/hc-2409.19606v3.txt L168-L170；paper/hc-2409.19606v3.txt L431-L433；paper/hc-2409.19606v3.txt L987-L1027
- ### 适用条件与边界｜训练显存占用增加；identity 易被误解为流间完全不交流；本页资料里没有 iHC 与 HC 的对照实验｜150｜paper/hc-2409.19606v3.txt L1029-L1057；paper/src-v3/app.tex Parameters, Computation and Memory Footprint Analysis；repo/hy4-transformers-modeling.py L603-L641

## 短页

- ### 流间取恒等的那一族｜是什么：HC 把残差流扩成 n 条，iHC 是其中把流间残差混合取成恒等的一族｜150｜paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；repo/hy4-transformers-modeling.py L603-L641；repo/hy4-transformers-configuration.py L47-L52；blog/hy4-model-card.md L63-L84
- ### 连接强度预先固定｜为什么需要：残差连接只有一条宽度固定的通路，连接强度预先固定、不由模型学习｜150｜paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/hc-2409.19606v3.txt L113-L115
- ### 汇入、分发、取恒等｜怎么做：复制成 n 条、每层汇成输入再分发输出，残差混合取恒等｜300｜paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections；repo/hy4-transformers-modeling.py L603-L641
