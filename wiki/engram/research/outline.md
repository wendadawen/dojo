# Engram

- 描述：Engram 是给大语言模型加的条件记忆模块：用最近几个 token 的哈希去查一张大嵌入表，取回静态向量，经门控后加回隐藏状态。
- 摘要：用后缀 $N$-gram 的哈希在 $O(1)$ 时间里查大嵌入表，把实体、固定搭配这类静态模式交给查表，计算层留给推理。
- 主题：模型结构, 内存与缓存
- 标签：模型架构

## 长页

- 开头｜先交代层、隐藏状态、残差、嵌入表、MoE；Engram 是什么（条件记忆，与条件计算互补）；为什么需要：实体要多层计算才拼出（逐层表折叠）；大概怎么工作｜700｜paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/analysis.tex L9-L49；paper/src-v2/sections/arch.tex Overview；paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/large_scale_exp.tex Training Data and Model Configurations
- ### 检索：从 token 到记忆向量｜检索的整条流水线及其先后：压缩词表、取后缀 N-gram、多头哈希、查表拼接｜100｜paper/src-v2/sections/arch.tex Sparse Retrieval via Hashed
- #### 词表压缩｜为什么先把等价 token 合成同一个 ID，怎么合，压掉多少；合并最多的几例折叠｜200｜paper/src-v2/sections/arch.tex Tokenizer Compression；paper/src-v2/main.tex Case Study of Tokenizer Compression
- #### 后缀 N-gram 与多头哈希｜后缀 N-gram 怎么取，为什么哈希，每阶用 K 个头、每个头一张素数长度的表（演示代码里各头取不同素数），向量怎么拼；小表手算例子和代码折叠｜550｜paper/src-v2/sections/arch.tex Multi-Head Hashing；repo/engram/engram_demo_v1.py L181-L296
- ### 融合：门控、卷积与残差｜本章回答取回的向量怎样接进主干，只有标题，内容在两个小节｜0｜paper/src-v2/sections/arch.tex Context-aware Gating
- #### 上下文门控｜为什么要门控（碰撞、多义），用隐藏状态当查询算标量门 $\alpha_t$ 的公式，门在哪些 token 上打开；数值例子折叠｜450｜paper/src-v2/sections/arch.tex Context-aware Gating；repo/engram/engram_demo_v1.py L358-L378；paper/src-v2/sections/analysis.tex Case Study: Gating Visualization
- #### 短卷积、残差与训练｜门控后的短因果卷积做什么，输出怎样以残差加回；表和投影随模型一起训练；优化器与初始化折叠｜300｜paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/large_scale_exp.tex Training Data and Model Configurations
- ### 插在哪几层，多分支怎么接｜插早插晚的权衡，单层最优与两层更优；一句说明多分支主干；各分支独立算门，是消融最关键三项之一；共享表和投影的细节折叠｜350｜paper/src-v2/sections/analysis.tex Where should memory be injected；paper/src-v2/sections/analysis.tex Which components matter；paper/src-v2/sections/arch.tex Integration with Multi-branch Architecture
- ### 稀疏参数怎么分给专家和记忆｜固定总参数和计算量时，分配比 $\rho$ 的定义、U 形曲线与最优点、只加记忆时的对数线性趋势；实验条件与数字折叠｜450｜paper/src-v2/sections/scaling_law_exp.tex Optimal Allocation Ratio；paper/src-v2/sections/scaling_law_exp.tex Engram under Infinite Memory Regime
- ### 确定性地址带来的预取与卸载｜先讲与 MoE 路由的区别：地址只凭 token ID 提前算出；表放显存外，推理时预取并与计算重叠，插深好掩盖延迟、建模却要早插；分片、缓存各一句；吞吐表折叠｜400｜paper/src-v2/sections/arch.tex System Efficiency: Decoupling；paper/src-v2/sections/analysis.tex L163-L206
- ### 查表在模型里起了什么作用｜等效加深网络（LogitLens、CKA）、关掉 Engram 后事实类与阅读理解类的差别、注意力腾给长上下文；数字折叠｜400｜paper/src-v2/sections/analysis.tex Is Engram functionally equivalent；paper/src-v2/sections/analysis.tex Sensitivity Analysis；paper/src-v2/sections/long_context_exp.tex Long Context Training
- ### 适用条件与边界｜高阶 N-gram 未必有益；两条误解各一句、对上资料原文：记忆能替代计算（只写结论，不再解释曲线）、只对知识类任务有用｜250｜paper/src-v2/sections/scaling_law_exp.tex Results and Analysis；paper/src-v2/sections/analysis.tex Which components matter；paper/src-v2/sections/large_scale_exp.tex Experimental Results

## 短页

- ### 一张按 N-gram 查的大表｜Engram 是什么：条件记忆模块，按最近几个 token 查大嵌入表，向量加回隐藏状态；解释嵌入表和隐藏状态；不写哈希等步骤｜250｜paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/arch.tex Overview
- ### 模型在用计算模拟查表｜为什么需要：一句话说明 MoE（参数多、每 token 计算不变），实体和固定搭配要靠多层计算重建，没有查表原语｜300｜paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/analysis.tex Is Engram functionally equivalent
- ### 压缩、哈希、门控、预取｜怎么做：词表压缩、多头哈希检索、上下文门控与残差融合、表放在显存之外并提前预取｜300｜paper/src-v2/sections/arch.tex Sparse Retrieval via Hashed；paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/arch.tex System Efficiency: Decoupling
