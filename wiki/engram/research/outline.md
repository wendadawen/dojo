# Engram

- 描述：Engram 是给 Transformer 加的一张条件记忆表：把以当前位置结尾的几个 token 哈希成嵌入表的行号，取出的向量由当前隐藏状态门控后加回残差流，用一次查表代替早期层对固定搭配的重建。
- 摘要：Engram 把以当前位置结尾的 $n$ 个 token 哈希成嵌入表的行号，取回的向量由当前隐藏状态门控后加回残差流。查表的计算量不随表增大，与 MoE 构成两条互补的稀疏轴；总稀疏参数在两者间分配时，损失随分配比例呈 U 形。行号只由 token 决定，大表可放主机内存并提前预取。
- 主题：模型结构，内存与缓存
- 标签：模型架构

## 长页

- 开头｜说明 Engram 是什么、为什么需要、大概怎么做，并解释 MoE、激活参数这两个基线外的词｜正文｜400｜paper/src-v2/sections/intro.tex L3-L9
- ### 取回：用最近几个 token 决定查哪几行｜讲取回要做的事：以当前位置结尾的 N-gram 作键去嵌入表取向量，并解释 N-gram 与嵌入表｜正文｜150｜paper/src-v2/sections/arch.tex Sparse Retrieval via Hashed
- #### 后缀 N-gram 与分词压缩｜讲后缀 N-gram 的定义，以及为什么先把等价的 token 合并成同一个标准 id｜公式｜220｜paper/src-v2/sections/arch.tex Tokenizer Compression
- #### 多头哈希｜讲为什么必须哈希，每个阶数配多个独立哈希头与素数大小的表，取回的向量怎样拼成记忆向量｜公式｜330｜paper/src-v2/sections/arch.tex Multi-Head Hashing
- ### 融合：由当前上下文决定这份记忆用多少｜讲取回的向量与上下文无关、可能带噪，所以要先过一道门再写回｜正文｜200｜paper/src-v2/sections/arch.tex Context-aware Gating
- #### 门控公式｜给出 key、value 投影与门 α，说明各符号，以及记忆与上下文不符时门趋近 0｜公式｜330｜paper/src-v2/sections/arch.tex Context-aware Gating
- #### 短卷积与写回残差流｜讲门控后的值再过一层短的深度因果卷积，最后加回残差流｜公式｜200｜paper/src-v2/sections/arch.tex Context-aware Gating
- #### 一个能手算的例子｜写出例子说明的道理，设定与数字放进折叠块｜折叠块｜80｜自行构造
- ### 稀疏参数怎么分：U 形规律｜讲同样的总参数与每 token 计算量下，稀疏参数既能给专家也能给记忆表，要回答怎么分｜正文｜80｜paper/src-v2/sections/scaling_law_exp.tex Scaling Laws and Sparsity Allocation
- #### 分配比例 ρ｜定义总参数、激活参数、稀疏参数和分给专家的比例 ρ，说明比较时总参数与激活参数都固定｜公式｜240｜paper/src-v2/sections/scaling_law_exp.tex Compute-matched formulation；paper/src-v2/sections/scaling_law_exp.tex Allocation ratio
- #### 损失随 ρ 呈 U 形｜讲纯 MoE 不是最优、最优点的位置、两端各缺什么｜正文｜260｜paper/src-v2/sections/scaling_law_exp.tex Results and Analysis
- #### 表越大，损失越低｜讲表可以继续放大时，损失随槽位数对数线性下降，且不增加每 token 计算｜正文｜120｜paper/src-v2/sections/scaling_law_exp.tex Engram under Infinite Memory Regime
- #### 分配实验的条件｜列出两档训练算力下的模型规模与只改哪两项｜折叠块｜250｜paper/src-v2/sections/scaling_law_exp.tex Experimental protocol
- ### 地址只由 token 决定：存储与计算分开｜讲行号不依赖隐藏状态带来的系统后果｜正文｜80｜paper/src-v2/sections/arch.tex System Efficiency: Decoupling Compute and Memory
- #### 读哪些行在前向开始前就已确定｜讲行号只取决于 token id，不像 MoE 路由要等隐藏状态，所以能提前知道｜正文｜180｜paper/src-v2/sections/arch.tex System Efficiency: Decoupling Compute and Memory
- #### 训练与推理时表放在哪里｜讲训练时按设备分片、用 all-to-all 收发，推理时放主机内存并提前预取、与前面层的计算重叠｜正文｜280｜paper/src-v2/sections/arch.tex System Efficiency: Decoupling Compute and Memory；paper/src-v2/sections/analysis.tex System Efficiency
- #### 高频模式与多级缓存｜讲 N-gram 的访问频率服从 Zipf 分布，热行放快的层级，长尾放慢而大的介质｜正文｜130｜paper/src-v2/sections/arch.tex System Efficiency: Decoupling Compute and Memory
- #### 卸载大表的实验条件｜列出卸载实验的设置，以及这组实验为何只算保守估计｜折叠块｜250｜paper/src-v2/sections/analysis.tex System Efficiency
- ### 放在哪几层｜讲放得早能替骨干省下前面的层，放得晚上下文更充分且读取延迟更易遮蔽，所以要折中｜正文｜300｜paper/src-v2/sections/analysis.tex Structural Ablation and Layer Sensitivity；paper/src-v2/sections/arch.tex System Efficiency: Decoupling Compute and Memory
- ### 效果的证据：它存了什么、省了什么｜讲后面几节的实验各验证什么性质｜正文｜80｜paper/src-v2/sections/intro.tex L11-L15
- #### 知识靠它，上下文理解靠骨干｜讲屏蔽 Engram 输出后，事实类任务大幅下降，阅读理解基本保持｜正文｜170｜paper/src-v2/sections/analysis.tex Sensitivity Analysis
- #### 门在固定搭配结束处打开｜讲门控热力图在多 token 实体和固定短语结束的位置最亮，中英文都如此｜正文｜120｜paper/src-v2/sections/analysis.tex Case Study
- #### 浅层相当于更深的层｜解释 LogitLens 与 CKA 各测什么，结论：早层更快收敛，浅层表示对应基线更深的层｜正文｜260｜paper/src-v2/sections/analysis.tex Accelerated Prediction Convergence；paper/src-v2/sections/analysis.tex Representational Alignment and Effective Depth
- #### 同样的参数与计算，换来更好的结果｜讲同总参数与同激活参数下优于纯 MoE，收益也在推理、代码、数学与长上下文｜正文｜170｜paper/src-v2/sections/large_scale_exp.tex Experimental Results；paper/src-v2/sections/long_context_exp.tex Experimental Results
- ### 适用条件与边界｜讲对哪类模式有效、代价是什么、门和哈希挡不住什么，并澄清与经典 N-gram 语言模型的区别｜正文｜350｜paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/scaling_law_exp.tex Results and Analysis；paper/src-v2/sections/related_work.tex L3-L9

## 短页

- ### 一张按最近几个词寻址的记忆表｜说明 Engram 是在 Transformer 里加一张按最近几个 token 寻址的嵌入表｜正文｜250｜paper/src-v2/sections/intro.tex L9
- ### 重建固定搭配占用了前面的层｜说明早期层用计算重建固定搭配，查表就能做到，且大表不增加每 token 的计算｜正文｜280｜paper/src-v2/sections/intro.tex L7
- ### 哈希取行、门控、加回残差｜说明哈希取回、门控、卷积写回三步，以及地址只由 token 决定带来的预取｜正文｜380｜paper/src-v2/sections/arch.tex Overview
