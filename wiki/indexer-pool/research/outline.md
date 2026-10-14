# IndexPool

- 描述：IndexPool 是稀疏注意力（先挑少量相关位置再做注意力的方法）的索引器（负责做这个挑选的模块）把相邻索引 key（索引器为每个 token 算出的向量）组成固定大小的池，先压缩成池 key、按池选池，再把选中池还原为 token 索引（缓存槽位编号）的机制。
- 摘要：长上下文里索引器要给每个查询找相关 token；GLM-5.3-Flash 的 IndexPool 先把 4 个索引 key 加权池化成 1 个，再从中选池，用 $\lfloor N/4\rfloor$ 个完整池候选加不完整尾部替代 $N$ 个 token 候选。
- 主题：注意力机制，推理系统
- 标签：推理加速

## 长页

- 开头｜IndexPool 是稀疏注意力索引器里的一层压缩机制；正文先讲它的候选来源，再依次走池化、选择、还原和尾部｜260｜blog/glm-5.3-flash.txt L15-L19；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L772-L783
- ### 索引器和候选池｜稀疏注意力先把逐 token 候选缩小，索引器用独立轻量投影做挑选；它的相邻索引 key 后续组成池｜300｜blog/glm-5.3-flash.txt L15-L19；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L776-L778；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L859-L900
- #### 索引 key 与池化参数｜索引 key 怎么生成，池大小 $P$、索引候选预算 $K$、门控和位置参数分别控制什么｜160｜repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L776-L807；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L833-L834；blog/glm-5.3-flash-config.json L21-L26
- #### 拼接缓存状态｜key、池化门控向量和来自填充掩码的有效位如何拼成缓存状态，为什么动态缓存按层保存它｜160｜repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L818-L846；repo/transformers/src/transformers/cache_utils.py L318-L358
- ### 池 key 的生成｜相邻 key 怎样对齐成完整池，有效位如何决定池合法性，门控 softmax 与位置参数怎样得到池 key｜520｜repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L942-L1008
- ### 按池选择｜查询怎样对池 key 打分、把各头分数合成一个池分，为什么只选 $K/P$ 个池｜560｜repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L858-L892；repo/transformers/src/transformers/models/glm5_next/configuration_glm5_next.py L209-L218
- #### 得到候选分数｜余弦式池分、各头权重和因果可见性怎样共同筛出候选池｜220｜repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L861-L882
- #### 选池与还原 token｜为什么预算除以池大小，选中池的原始 token 索引怎样还原｜220｜repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L883-L900
- ### 不完整尾部｜尾部为什么不能强行池化，尾部 token 怎样直接加入索引，候选输出宽度为何最多增加 $P-1$；六个 key 的例子折叠｜550｜repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L902-L911；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L1010-L1062
- ### 适用条件与边界｜只适用于固定池长的完整池，要求 $K$ 能被 $P$ 整除；代价是池内 token 绑定选择、池 key 有压缩损失，误解是它压缩主 KV cache｜370｜repo/transformers/src/transformers/models/glm5_next/configuration_glm5_next.py L63-L66；repo/transformers/src/transformers/models/glm5_next/configuration_glm5_next.py L214-L218；blog/glm-5.3-flash.txt L19-L21

## 短页

- ### 候选数量带来的开销｜说清索引器承担挑选，长上下文里这个挑选模块带来延迟和内存压力｜180｜blog/glm-5.3-flash.txt L15-L19
- ### IndexPool 是什么｜说清相邻索引 key 组成固定池，门控加权池化成池 key，选择按池进行再还原 token｜300｜blog/glm-5.3-flash.txt L19-L19；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L776-L1008；repo/transformers/src/transformers/cache_utils.py L336-L338
- ### 从 key 到 token 索引｜按顺序讲分池、算池 key、选池、还原 token，并保留不完整尾部｜280｜repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L861-L1008；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L1010-L1062
