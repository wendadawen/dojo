# Block AttnRes

- 描述：Block AttnRes 是把 Transformer 的残差连接换成按深度做注意力的做法：层分成若干块，每层用一个可学查询对前面各块的块级表示做 softmax 加权，块内仍用普通残差相加。
- 摘要：把残差里各层输出的等权相加，换成每层对前面各块表示做 softmax 注意；层分成 $N$ 块，内存与通信从 $O(Ld)$ 降到 $O(Nd)$。
- 主题：模型结构, 注意力机制
- 标签：模型架构

## 长页

- 开头｜概览 Block AttnRes 是什么、为什么需要、大概怎么工作；残差连接一句带过｜550｜paper/src-v1/0-abstract.tex L1-L8；paper/src-v1/1-intro.tex Introduction；paper/src-v1/3-attnres.tex Block Attention Residuals
- ### 固定权重残差随深度稀释每一层｜普通残差等权相加；PreNorm 下幅度随深度增长、每层贡献被稀释、早期信息被聚合后无法在深层取回；三条局限；梯度展开折叠｜600｜paper/src-v1/2-preliminary.tex Residual Learning；paper/src-v1/2-preliminary.tex Generalizing Residuals；paper/src-v1/2-preliminary.tex Limitations；paper/src-v1/5-exp.tex Training dynamics
- ### 把深度当成一条序列来注意｜按深度做 softmax 注意；每层一个伪查询算权重；键取层输出并做 RMSNorm；公式折叠｜700｜paper/src-v1/3-attnres.tex The Duality of Time and Depth；paper/src-v1/3-attnres.tex Full Attention Residuals；paper/src-v1/5-exp.tex Architecture Details；paper/src-v1/6-discuss.tex Prior Residuals as Depth-Wise Linear Attention
- ### 分块：块内相加，块间注意｜先交代层的定义；$L$ 层分 $N$ 块，块内相加成块表示与部分和、块间只注意完成的块；块数取极端退回全深度注意或普通残差；降阶存储与层间注意；例子与伪代码折叠｜1000｜paper/src-v1/2-preliminary.tex Notation；paper/src-v1/3-attnres.tex Block Attention Residuals；paper/src-v1/3-attnres.tex Intra-Block Accumulation；paper/src-v1/3-attnres.tex Inter-Block Attention；paper/src-v1/3-attnres.tex Efficiency；paper/src-v1/figures/code.tex L12-L49
- ### 分块数 N 怎么选｜损失先给一句释义；约 8 块拿回大部分收益；块大小扫描与多规模损失对比折叠｜450｜paper/src-v1/3-attnres.tex Efficiency；paper/src-v1/5-exp.tex Cross-layer access；paper/src-v1/5-exp.tex Scaling Behavior；paper/src-v1/figures/block.tex L83-L108
- ### 训练时输出与梯度更均匀｜输出幅度在块边界被重置、限于块内，softmax 让梯度跨深度更均匀；设计消融折叠｜450｜paper/src-v1/0-abstract.tex L1-L8；paper/src-v1/5-exp.tex Training dynamics；paper/src-v1/5-exp.tex Component design；paper/src-v1/figures/block.tex L24-L37
- ### 适用条件与边界｜整除去余；即插即用、只多一个向量与一处归一化；块级压缩像隐式正则（一句释义）；代价是块表示要存储与传递；更偏好深而窄的网络；推理先给一句释义｜450｜paper/src-v1/3-attnres.tex Intra-Block Accumulation；paper/src-v1/5-exp.tex Architecture Details；paper/src-v1/5-exp.tex Analyzing Learned AttnRes Patterns；paper/src-v1/5-exp.tex Optimal Architecture；paper/src-v1/4-infra.tex Pipeline communication

## 短页

- ### 每层在深度上取一次信息｜是什么：普通残差等权相加；Block AttnRes 让每层对前面各块表示做 softmax 加权，块内仍等权相加；解释残差连接、块、查询、softmax｜330｜paper/src-v1/0-abstract.tex L1-L8；paper/src-v1/3-attnres.tex Block Attention Residuals；paper/src-v1/3-attnres.tex The Duality of Time and Depth
- ### 等权相加把每一层稀释掉｜为什么需要：残差等权累加随深度稀释每层、幅度增长、早期信息被埋｜330｜paper/src-v1/1-intro.tex Introduction；paper/src-v1/2-preliminary.tex Limitations
- ### 分块相加，块间注意｜块内首层只注意前面块表示，其后各层再对前面块表示与当前部分和做 softmax 注意｜330｜paper/src-v1/3-attnres.tex Block Attention Residuals；paper/src-v1/3-attnres.tex Inter-Block Attention
