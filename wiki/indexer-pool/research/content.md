# IndexPool

- 描述：IndexPool 是稀疏注意力（先挑少量相关位置再做注意力的方法）的索引器（负责做这个挑选的模块）把相邻索引 key（索引器为每个 token，即文本切出的最小单位，算出的向量）组成固定大小的池，先压缩成池 key（每池索引 key 压成的一个向量）、按池选池，再把选中池还原为 token 索引（缓存槽位编号）的机制。
- 摘要：长上下文里索引器要给每个查询找相关 token；GLM-5.3-Flash 的 IndexPool 先把 4 个索引 key 加权池化成 1 个，再从中选池，用 $\lfloor N/4\rfloor$ 个完整池候选加不完整尾部替代 $N$ 个 token 候选。
- 主题：注意力机制，推理系统
- 标签：推理加速

## 资料

- Z.ai. GLM-5.3-Flash. 官方博客, 2026. [https://z.ai/blog/glm-5.3-flash](https://z.ai/blog/glm-5.3-flash)
- Z.ai. GLM-5.3-Flash 模型卡与配置. 2026. [https://huggingface.co/zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
- Hugging Face Transformers. GLM-5.3-Flash implementation, 2026. [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

## 短页

### 候选数量带来的开销

稀疏注意力（见 [DSA](wiki/dsa)）先用索引器从长上下文里找少量相关位置，再只对这些位置做注意力。上下文越长，这个挑选模块的延迟和内存开销越突出。
依据：blog/glm-5.3-flash.txt L15-L19

### IndexPool 是什么

索引 key 是索引器为每个 token 算出的向量。IndexPool 把相邻的这些 key 组成固定大小的池；GLM-5.3-Flash 的池大小 $P$ 取 4，所以每池把 4 个索引 key 向量压成 1 个池 key。
依据：blog/glm-5.3-flash.txt L19-L19；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L776-L778；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L834-L846；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L859-L900；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L966-L1008；repo/transformers/src/transformers/cache_utils.py L336-L338

每个 token 的隐藏状态（这一层传给下一层的向量）投影出用于池内加权的门控向量；它和池内位置参数经 softmax（把一组分数变成和为 1 的非负权重的方法）得到池化权重。索引器对池 key 打分和选择，选中池后再还原成其中的 token 索引。IndexPool 降低的是索引器在长上下文里的候选开销，不是主 [KV cache](wiki/kv-cache)（主注意力的键值缓存）本身的大小。
依据：blog/glm-5.3-flash.txt L19-L19；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L836-L846；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L859-L900；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L998-L1008；repo/transformers/src/transformers/cache_utils.py L336-L338

### 从 key 到 token 索引

先从第一个有效 token（非填充的真实位置）起每 $P$ 个相邻 key 分成一池，$P$ 是池大小；对每池算一个加权池 key。查询（当前要找相关位置的 token 表示）给每个池打分，选出至多 $K/P$ 个分数最高的池，$K$ 是 token 候选预算；把每池的 token 索引（缓存槽位编号）展开还原。GLM-5.3-Flash 开启尾部开关时，序列末尾不足 $P$ 个 token 的不完整池直接作为尾部索引保留。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L861-L900；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L902-L911；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L966-L1008；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L1010-L1062；blog/glm-5.3-flash-config.json L23-L24

## 长页

IndexPool 是接在稀疏注意力（先挑少量相关位置，再只对这些位置做注意力）索引器（负责这个挑选的模块）里的一层压缩机制：它不改变“先挑选、再做注意力”的流程，而是改变索引器挑选候选的方式。
依据：blog/glm-5.3-flash.txt L15-L19；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L772-L783

### 索引器和候选池

稀疏注意力先缩小逐 token（文本切出的最小单位）候选，再在选中的位置上做注意力；索引器使用与主注意力路径分开的轻量线性投影，主注意力是选中位置后执行的那层注意力。索引 key 就是这些轻量线性投影为每个 token 算出的向量；IndexPool 把相邻索引 key 组织成池，让打分和选择先发生在池这一层。
依据：blog/glm-5.3-flash.txt L15-L19；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L776-L778；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L833-L834；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L859-L900

#### 索引 key 与池化参数

索引 key 是索引器自己的轻量 key，不等于主注意力的 [KV cache](wiki/kv-cache)（主注意力的键值缓存）条目。隐藏状态（这一层传给下一层的向量）先经线性投影（乘一个可学习矩阵）得到索引 key，再经过 LayerNorm（把每个向量的各分量按均值和方差重新缩放的方法）稳定数值；池大小 $P$、索引候选预算 $K$、池化门控矩阵和池内位置参数在索引器初始化时确定。GLM-5.3-Flash（使用这个机制的资料来源模型）的配置取 $P=4$、$K=2048$。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L776-L778；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L794-L807；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L833-L834；blog/glm-5.3-flash-config.json L21-L26；repo/transformers/src/transformers/cache_utils.py L336-L338

#### 拼接缓存状态

索引器把索引 key、池化门控向量（由门控矩阵从隐藏状态投影出、用于池内加权的向量）和一个有效位拼成一个状态向量，再交给缓存更新。有效位来自布尔填充掩码，标记这个位置是不是真实 token，而不是填充位；动态缓存是生成时逐 token 扩展的缓存，它按注意力层保存这份状态，后续才能按层重新分池。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L818-L846；repo/transformers/src/transformers/cache_utils.py L318-L358

### 池 key 的生成

池 key 是一个池内全部索引 key 的加权平均表示。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L998-L1008

池化从第一个有效 token 开始，而不是从缓存起始槽位开始；这样填充位不会把相邻 token 错分进同一池。每个池只接受 $P$ 个位置都在边界内的完整组，末尾不足 $P$ 个 token 的组不进入完整池。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L942-L995

门控向量给池内 token 提供池化权重。每个 token 的门控向量与该池内的位置参数相加；填充位对应的位置先被设为负无穷，再做 softmax（把一组分数变成和为 1 的非负权重的方法），使这些位置不贡献池 key。权重与索引 key 逐项相乘后求和，得到池 key。池 key 与索引 key 同维，池索引仍保留池内每个 token 的原始位置。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L966-L1008

### 按池选择

#### 得到候选分数

索引器分多个头并行算池分；头是索引器里并行的独立计算单元，每个头有自己的查询向量。每个头用自己的查询向量和池 key 做点积，经缩放和 ReLU（把负数变成零的函数）得到该头分数，再由隐藏状态投影出的每头权重汇总成一个标量。池的因果可见性按池内最后一个 token 判断：只有位置不晚于查询的位置才可见；最后一个 token 对当前查询可见且整池合法，它才是可用池；填充或越界的池会被置为不可选。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L790-L801；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L833-L834；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L859-L882

#### 选池与还原 token

选择的预算从 $K$ 个 token 变成至多 $K/P$ 个可用池：索引器取池分最高的可用池，可用池不足时取全部可用池；完整池展开得到至多 $K$ 个候选槽位，可用池不足时更少。选中池里若有无效位置，展开时这些槽位仍会被标记为无效。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L883-L900

### 不完整尾部

GLM-5.3-Flash 打开尾部开关：把当前不完整池里的可见 token 直接追加为 token 索引，候选输出宽度因此最多增加 $P-1$。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L902-L911；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L1010-L1062；blog/glm-5.3-flash-config.json L23-L24

::: 六个 key 的例子
设 $P=4$，6 个真实 token 的索引 key 依次是 $\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{d},\mathbf{e},\mathbf{f}$，没有填充位，所以第一池是 $(\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{d})$，第二池不完整。
依据：自行构造

1. 第一个池的池化权重设为 $(0.1,0.2,0.3,0.4)$，池 key 是 $\mathbf{p}=0.1\mathbf{a}+0.2\mathbf{b}+0.3\mathbf{c}+0.4\mathbf{d}$。
2. 用二维数代替真实 128 维 key：$\mathbf{a}=(1,0),\mathbf{b}=(0,1),\mathbf{c}=(1,1),\mathbf{d}=(1,1)$，则 $\mathbf{p}=(0.8,0.9)$；尾部是 $\mathbf{e}=(1,1),\mathbf{f}=(1,2)$。
3. 设查询 $\mathbf{q}=(1,2)$。这一步用单头未缩放点积代替完整池分，便于比较排序：第一池分数是 $0.8+2\times0.9=2.6$。尾部 token 不参与池选择，所以不需要给它们算池分。
4. 若预算 $K=4$，则完整池预算 $K/P=1$，第一池被选中并还原为位置 0、1、2、3；位置按缓存索引从 0 开始编号。尾部位置 4、5 直接保留，最终索引是 0、1、2、3、4、5。
依据：自行构造
:::

这个例子说明池化把 4 个原始候选合成一个选择对象；被选中的不是压缩向量本身，而是它背后的 4 个 token 位置。
依据：自行构造

### 适用条件与边界

IndexPool 的完整池依赖固定池长：$P$ 必须为正整数，$K$ 必须能被 $P$ 整除，配置检查会拒绝不满足这个条件的设置。
依据：repo/transformers/src/transformers/models/glm5_next/configuration_glm5_next.py L63-L66；repo/transformers/src/transformers/models/glm5_next/configuration_glm5_next.py L214-L218；blog/glm-5.3-flash-config.json L22-L26

它的选择粒度是池，不是单个 token：查询不能越过池边界保留一个索引 key 而丢弃同池其余索引 key。
依据：repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L883-L900；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L998-L1008；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L902-L911；repo/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py L1010-L1062

IndexPool 压缩的是索引器 key，不是主 KV cache：它的目标是在 1M token 上下文里降低索引器的延迟和内存开销；索引器状态和主 KV cache 是分开缓存的不同张量。
依据：blog/glm-5.3-flash.txt L19-L21；repo/transformers/src/transformers/cache_utils.py L318-L358
