# Engram

- 描述：Engram 是给 Transformer 加的一张条件记忆表：把以当前位置结尾的几个 token 哈希成嵌入表的行号，取出的向量由当前隐藏状态门控后加回残差流，用一次查表代替早期层对固定搭配的重建。
- 摘要：Engram 把以当前位置结尾的 $n$ 个 token 哈希成嵌入表的行号，取回的向量由当前隐藏状态门控后加回残差流。查表的计算量不随表增大，与 MoE 构成两条互补的稀疏轴；总稀疏参数在两者间分配时，损失随分配比例呈 U 形。行号只由 token 决定，大表可放主机内存并提前预取。
- 主题：模型结构，内存与缓存
- 标签：模型架构

## 长页

- 开头｜说明 Engram 是什么、为什么需要、大概怎么做，只用「取回向量、按上下文决定用多少、加进模型」这类说法，并用一句话解释 MoE｜正文｜400｜paper/src-v2/sections/intro.tex L3-L9
- ### 背景：token、隐藏状态与残差流｜解释后文反复用到的词：token、隐藏状态、残差流，一层里的注意力与前馈，以及损失｜正文｜230｜paper/src-v2/sections/arch.tex Overview；paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/large_scale_exp.tex Evaluation Protocol；paper/src-v2/sections/intro.tex L7
- ### 取回：用最近几个 token 决定查哪几行｜讲取回要做什么：以结尾的 N-gram 作键查嵌入表取向量，向量是训练出来的而非计数，每个 token 只读固定几行所以表大小不影响计算｜正文｜250｜paper/src-v2/sections/arch.tex Sparse Retrieval via Hashed；paper/src-v2/sections/scaling_law_exp.tex Compute-matched formulation；paper/src-v2/sections/large_scale_exp.tex Training Data and Model Configurations；paper/src-v2/sections/related_work.tex L3-L5
- #### 后缀 N-gram 与分词压缩｜给出后缀 N-gram 的记号，讲为什么先把等价的 token 合并成同一个标准 id｜公式｜170｜paper/src-v2/sections/arch.tex Tokenizer Compression
- #### 多头哈希｜讲为什么必须哈希，每个阶数配多个独立哈希头，向量怎样拼成记忆向量｜公式｜280｜paper/src-v2/sections/arch.tex Multi-Head Hashing
- ### 融合：由当前上下文决定这份记忆用多少｜讲取回的向量与上下文无关、可能带噪，所以要先过一道门再写回｜正文｜100｜paper/src-v2/sections/arch.tex Context-aware Gating
- #### 门控公式｜给出 key、value 投影与门 α，说明各符号，以及记忆与上下文不符时门趋近 0｜公式｜330｜paper/src-v2/sections/arch.tex Context-aware Gating
- #### 短卷积与写回残差流｜讲门控后的值过一层短卷积以扩大感受野，再加回残差流，且插在选定的层、该层注意力之前｜正文｜120｜paper/src-v2/sections/arch.tex Context-aware Gating
- #### 短卷积的公式｜给出短卷积、激活与残差相加的公式，说明各符号｜折叠块｜150｜paper/src-v2/sections/arch.tex Context-aware Gating
- #### 一个能手算的例子｜例子演示两个机制：不同 bigram 在一个头里撞行、在另一个头里分开；门在不符时压低｜折叠块｜450｜自行构造
- ### 稀疏参数怎么分：与专家争同一笔预算｜定义总参数、激活参数、稀疏参数和分给专家的比例 ρ，说明比较时总参数与激活参数都固定｜公式｜220｜paper/src-v2/sections/scaling_law_exp.tex Compute-matched formulation；paper/src-v2/sections/scaling_law_exp.tex Allocation ratio
- #### 损失随 ρ 呈 U 形｜讲纯 MoE 不是最优，最优点偏向专家一侧、记忆只需分得一小部分，两端各缺什么｜正文｜200｜paper/src-v2/sections/scaling_law_exp.tex Results and Analysis
- #### 表越大，损失越低｜讲表可以继续放大时，损失随槽位数对数线性下降｜正文｜90｜paper/src-v2/sections/scaling_law_exp.tex Engram under Infinite Memory Regime
- #### 分配实验的条件｜列出两档训练算力下的模型规模，以及只改专家数与记忆槽位数｜折叠块｜250｜paper/src-v2/sections/scaling_law_exp.tex Experimental protocol
- ### 地址只由 token 决定：大表放主机内存，提前预取｜讲行号只取决于 token id，故可提前预取；大表放不进显存就放主机内存，与前面层计算重叠，开销很小｜正文｜260｜paper/src-v2/sections/arch.tex System Efficiency: Decoupling Compute and Memory；paper/src-v2/sections/analysis.tex System Efficiency
- #### 卸载实验的条件与结果｜写卸载整张大表时的设置与吞吐变化｜折叠块｜250｜paper/src-v2/sections/analysis.tex System Efficiency
- ### 放在哪几层｜讲放得早能替骨干省下前面的层，放得晚门的上下文更足、前面可遮住读取延迟的计算也更多，所以位置是折中；也可分成早晚两处各放一份小表｜正文｜180｜paper/src-v2/sections/analysis.tex Where should memory be injected；paper/src-v2/sections/arch.tex System Efficiency: Decoupling Compute and Memory
- ### 哪些部件起作用｜讲去掉门控或分词压缩损失明显变差，去掉短卷积只略微变差｜正文｜130｜paper/src-v2/sections/analysis.tex Which components matter
- ### 效果的证据：整体更好，也更深更省｜讲同总参数与同激活参数下优于纯 MoE，收益不限于知识类任务；查表接手局部依赖后，注意力可专注全局，长上下文检索更好｜正文｜200｜paper/src-v2/sections/large_scale_exp.tex Experimental Results；paper/src-v2/sections/intro.tex L14-L15；paper/src-v2/sections/long_context_exp.tex Experimental Results
- #### 大规模对比的设置｜列出比较的几个模型、匹配的量和任务类别，不放分数｜折叠块｜250｜paper/src-v2/sections/large_scale_exp.tex Training Data and Model Configurations；paper/src-v2/sections/large_scale_exp.tex Evaluation Protocol
- #### 浅层相当于更深的层｜用一两句说明 LogitLens 与 CKA 各测什么，结论：早层更快收敛，浅层表示对应基线更深的层｜正文｜160｜paper/src-v2/sections/analysis.tex Accelerated Prediction Convergence；paper/src-v2/sections/analysis.tex Representational Alignment and Effective Depth
- ### 适用条件与边界｜讲对局部静态的固定搭配有效，事实类依赖它而阅读理解靠骨干，记忆不能替代计算，不替换输入嵌入层与输出层｜正文｜260｜paper/src-v2/sections/intro.tex L7；paper/src-v2/sections/arch.tex L1-L8；paper/src-v2/sections/scaling_law_exp.tex Results and Analysis；paper/src-v2/sections/analysis.tex Case Study；paper/src-v2/sections/analysis.tex Sensitivity Analysis

## 短页

- ### 一张按最近几个词寻址的记忆表｜说明 Engram 是在 Transformer 里加一张按最近几个 token 寻址的嵌入表｜正文｜250｜paper/src-v2/sections/intro.tex L9；paper/src-v2/sections/arch.tex Overview
- ### 固定搭配不必再靠计算重建｜说明早期层用计算重建固定搭配、查表即可；MoE 是每个 token 只激活少数专家的结构，表与专家共用同一笔不增加计算的参数，混合优于纯 MoE｜正文｜320｜paper/src-v2/sections/intro.tex L3-L9；paper/src-v2/sections/scaling_law_exp.tex Compute-matched formulation；paper/src-v2/sections/scaling_law_exp.tex Results and Analysis
- ### 哈希取行、门控、加回残差｜说明哈希取回、按当前隐藏状态门控、卷积后加回残差流三步，并一句话解释隐藏状态与残差流｜正文｜350｜paper/src-v2/sections/arch.tex Overview；paper/src-v2/sections/arch.tex Context-aware Gating
