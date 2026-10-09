# FusedMoE

- 描述：FusedMoE 是 vLLM 里的 MoE 层实现，把每个 token 与其 top-k 专家的计算合并起来，用两次分组 GEMM 加一次加权求和批量算完。
- 摘要：把 $M$ 个 token 的各路专家计算按专家重排对齐，收进两次分组 GEMM 和一次求和，用少数几个 kernel 批量算完。
- 主题：模型结构, 推理系统
- 标签：MoE

## 资料

- vLLM Team. Fused MoE Modular Kernel. vLLM design documentation. [https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/)
- vLLM Team. Fused MoE Kernel Features. vLLM design documentation. [https://docs.vllm.ai/en/stable/design/moe_kernel_features/](https://docs.vllm.ai/en/stable/design/moe_kernel_features/)
- vLLM Team. vllm-project/vllm source tree. [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

## 短页

### 一次算完所有专家的活

混合专家（MoE）把一层前馈网络换成多个并行的专家子网络。每个 token（输入序列里的一个词元）由门控网络（gate）算一组分数，只取分数最高的 top-$k$ 个专家，各自算了再按分数加权相加。一个专家是两层线性层，中间夹一次激活函数，所以每个被选中的「token 与专家的组合」要做两次矩阵乘（GEMM，通用矩阵乘）加一次激活。FusedMoE 是 vLLM（一个大模型推理引擎）里 MoE 层的实现：它不让每个专家各自调用矩阵乘，而是把整批专家计算交给同一批内核（kernel，在 GPU 上并行执行的函数）算完。
依据：blog/fused_moe_modular_kernel.md L70-L80；repo/vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py L80-L122；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L355-L381

### 逐专家调用 GEMM 行数太少

如果每个专家各自调用一次矩阵乘，每个专家拿到的矩阵太瘦。设这批有 $M$ 个 token、每个 token 选 $k$ 个专家、全层共 $E$ 个专家，每次调用平均只有 $Mk/E$ 行；而列数是 token 向量的维度和专家内部的中间维度，比行数大得多。
依据：自行构造

小批量推理时这个行数很小，整层是访存受限的。融合之后，所有专家的行块排进同一次内核调用的并行网格里，一次调用覆盖全部专家。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1371-L1372；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L822-L837

### 对齐、两次 GEMM、加权求和

做法是先把组合按专家排好，让一个行块（按固定行数切出的一段连续行）里的行都属于同一个专家，每个专家的行数再补齐到行块大小的整数倍，块内才好做矩阵乘。之后用一批内核走三步：一次分组 GEMM 算完所有专家的第一层，一次逐元素激活，再一次分组 GEMM 算完所有专家的第二层；最后按门控分数把每个 token 的 $k$ 路结果相加，得到这个 token 的输出向量。分组 GEMM 就是一次矩阵乘调用里，每个行块各自换一份权重。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py L67-L127；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1787-L1809；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1811-L1813；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1826-L1848；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1850-L1852

## 长页

混合专家（MoE）把一层前馈网络换成多个并行的专家子网络。一层里配一个门控网络（gate）：它按 token（输入序列里的一个词元）当前的隐藏向量给每个专家打一个分数。实际只取分数最高的 top-$k$ 个专家，$k$ 通常远小于专家总数 $E$。每个被选中的专家各自对这个 token 做一次前馈计算，结果再按门控分数加权相加，合成一个输出向量。所以一个 token 走一遍 MoE 层，是它与选中的 $k$ 个专家配成的 $k$ 个「组合」各算一次，再合并。每个组合要做两次矩阵乘（GEMM，通用矩阵乘；原因见下一章）。
依据：blog/fused_moe_modular_kernel.md L70-L80；repo/vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py L80-L122

如果每个专家各自调用一次矩阵乘，代价出在矩阵的形状上。这一批有 $M$ 个 token，共 $Mk$ 个组合，分给 $E$ 个专家，每次调用平均只有 $Mk/E$ 行。$M$ 小、$E$ 大时这个行数很小：取 $M=64$、$k=2$、$E=64$，每个专家平均两行，而列数是 token 向量的维度和专家内部的中间维度，是几千量级。
依据：自行构造

小批量推理时，整层是访存受限的。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1371-L1372

FusedMoE 是 vLLM（一个大模型推理引擎）里 MoE 层的实现。所有组合合起来是「每个专家各算一批 token」，所以它先把组合按专家排序并补齐（让每个专家的行数凑成固定块大小的整数倍），再交给同一批内核（kernel，在 GPU 上并行执行的函数）：一次矩阵乘算完所有专家的第一层，一次逐元素激活，一次矩阵乘算完所有专家的第二层，最后一次把每个 token 的 $k$ 路结果按门控分数相加。所有专家的行块排进同一次内核调用的并行网格里，一次调用覆盖全部专家。
依据：blog/fused_moe_modular_kernel.md L70-L80；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1787-L1809；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1811-L1813；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1826-L1848；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1850-L1852；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L822-L837

### 一个专家在算什么

一个专家是两层线性层，中间夹一次激活函数。设 token 向量的维度叫隐藏维，记 $H$；激活之后的中间维记 $I$。第一层把 $H$ 维扩到 $2I$ 维（$2I$ 是门与值两半，见「两次 GEMM 之间的激活」），激活把 $2I$ 维压成 $I$ 维，第二层再把 $I$ 维压回 $H$ 维。层里所有专家的权重各堆成一个张量，输出维在前：第一层是 $(E, 2I, H)$，第二层是 $(E, H, I)$。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L361-L369；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1702-L1708；repo/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py L860-L877

### moe_align_block_size：按专家分段再补齐

批量做矩阵乘要求一个行块（对齐后按固定行数，即行块大小，切出的一段连续行）里的行都属于同一个专家，否则块内各行要乘不同的权重，矩阵乘就做不成。所以要把组合按专家排好，再把每个专家的行数补齐到行块大小的整数倍。这一步叫对齐，由函数 `moe_align_block_size` 完成。它读入每个 token 的 top-$k$ 专家编号（形状是「token 数乘 $k$」），输出三样东西：
依据：repo/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py L67-L87

1. `sorted_token_ids`：按专家分组排好的组合下标。组合下标是把「token 数乘 $k$」的组合展平后的编号，等于 token 号乘 $k$ 加第几路，除以 $k$ 取整才是 token 号。
2. `expert_ids`：每个行块用哪个专家，一个行块一个编号。
3. `num_tokens_post_padded`：补齐之后的总行数。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py L88-L102；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L466-L477

补齐位填一个越界的下标，取值是组合总数 $Mk$。矩阵乘里每行按「下标是否小于组合总数」判断是否有效：补齐位的输入装成 0，写出结果时被掩码挡住，不写进输出。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py L104-L127；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L419；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L530-L535；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L607-L608

::: 补齐后总行数的上界
设行块大小为 $B$，补齐后的总行数不超过 $Mk + E(B-1)$：最坏情况下每个专家都多补 $B-1$ 行。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py L137-L139
:::

::: 三个 token、三个专家走一遍对齐
设一层有 $3$ 个专家，$3$ 个 token，每个 token 选 $2$ 个专家，行块大小取 $2$。三个 token 选的专家编号依次是 $(0,1)$、$(0,2)$、$(0,1)$。
依据：自行构造

1. 展平：组合下标等于 token 号乘 $2$ 加第几路，下标 $0$ 到 $5$ 的组合所选专家依次是 $0, 1, 0, 2, 0, 1$。
2. 分桶：专家 $0$ 收到下标 $0, 2, 4$，专家 $1$ 收到下标 $1, 5$，专家 $2$ 收到下标 $3$。
3. 补齐：专家 $0$ 有 $3$ 行，补 $1$ 个到 $4$ 行；专家 $1$ 有 $2$ 行，不用补；专家 $2$ 有 $1$ 行，补 $1$ 个到 $2$ 行。补齐位填组合总数 $6$。
4. 切成四个行块：$[0,2]$、$[4,6]$、$[1,5]$、$[3,6]$。`sorted_token_ids` 是 $0, 2, 4, 6, 1, 5, 3, 6$，`expert_ids` 是 $0, 0, 1, 2$，`num_tokens_post_padded` 是 $8$。
5. 取输入：下标除以 $2$ 取整就是 token 号，依次是 $0, 1, 2, -, 0, 2, 1, -$；两个补齐位的下标 $6$ 不小于 $6$，被掩码挡住。
依据：自行构造
:::

### 对齐后的分组 GEMM

排好之后，所有专家的第一层收进一次内核调用，叫分组 GEMM：一次矩阵乘，但每个行块用不同的权重。它的输出是 $(M, k, 2I)$——token、它的每一路、第一层输出宽度。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L361-L381；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1735-L1758

内核把输出切成大小相等的输出块，每个输出块交给一组并行单元算，输出块总数是行方向的块数乘列方向的块数。行方向的块就是行块，列方向是沿输出宽度 $2I$ 切出的段。第 $i$ 个行块用 `expert_ids[i]` 那个专家的权重，所以同一个行块的各个输出块共用同一个专家。排列输出块时，内核不按行顺序逐行扫，而是把相邻的几个行块归为一组依次处理，让它们共用的权重留在二级缓存里被复用。这只在每个专家的行块够多时有用；专家多、每个专家只分到几个 token 时，分组没有效果。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L386-L395；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L833-L836；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1391-L1394

每个输出块先读出所属行块的专家编号，据此把权重张量的指针挪到该专家那一段：专家编号乘上「一个专家占多少元素」，就是这个专家那层权重的起点。普通矩阵乘整批用同一份权重，分组矩阵乘每块换一份。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L421-L433；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L466-L477

::: 三个专家的权重怎么分段取
沿用上一节的例子，四个行块的专家编号依次是 $0, 0, 1, 2$。第一层输出宽度 $2I$ 取 $2$（也就是 $I$ 取 $1$），隐藏维 $H$ 取 $2$。第一层权重堆成一个 $(3, 2, 2)$ 的张量，第 $e$ 个专家占的元素从 $4e$ 开始，共 $4$ 个。
依据：自行构造

1. 第一、二个行块的专家编号是 $0$，指针挪到偏移 $0$，取权重元素 $0$ 到 $3$。
2. 第三个行块的专家编号是 $1$，指针挪到偏移 $4$，取权重元素 $4$ 到 $7$。
3. 第四个行块的专家编号是 $2$，指针挪到偏移 $8$，取权重元素 $8$ 到 $11$。
4. 四块用同一份代码、同一次内核调用，只有指针的起始偏移不同。
依据：自行构造
:::

#### 两种输入激活格式

上面这套对齐按标准格式的输入做：输入是一个 $(M, H)$ 的连续张量，配上每个 token 的 top-$k$ 专家编号与门控分数，还没有按专家分好，所以要自己按专家排序（置换），算完再还原。另一种叫分桶格式，输入形状是「专家数乘每桶最大 token 数乘 $H$」，附一个每个专家有效 token 数的计数：由分发（all2all 通信步骤，用来实现专家并行）直接把选中同一专家的 token 批到同一个桶里，所以省掉置换和还原，其余步骤一样。取哪种格式由所用的 All2All 分发决定。
依据：blog/fused_moe_modular_kernel.md L15-L27；blog/fused_moe_modular_kernel.md L72-L80；blog/moe_kernel_features.md L7；blog/moe_kernel_features.md L11

### 两次 GEMM 之间的激活

两次 GEMM 之间夹着一次门控激活，逐元素作用，不改变行数，也不重排。第一层的输出分成前后两半，各 $I$ 维，前半叫门，后半叫值。以常用的 SiLU 门控为例，结果是 $\mathrm{SiLU}(\text{门}) \times \text{值}$，其中 $\mathrm{SiLU}(x) = x \cdot \sigma(x)$，$\sigma$ 是 Sigmoid。所以 $(M, k, 2I)$ 的第一层输出经激活变成 $(M, k, I)$，再进第二次 GEMM。
依据：blog/fused_moe_modular_kernel.md L74-L80；repo/vllm/vllm/model_executor/layers/activation.py L116-L119；repo/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py L860-L877

### 第二次 GEMM 与加权求和

第二次分组 GEMM 与第一次同构：按行块做，每块按 `expert_ids` 取该专家的第二层权重，去乘激活后的结果，得到 $(M, k, H)$——每个 token 的每一路各得一个向量，还没有合并。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1826-L1852

合并是最后一步，在第二次 GEMM 之后单独做：把 $(M, k, H)$ 沿 $k$ 方向求和，写进 $(M, H)$ 的输出。默认情况下门控分数已在第二次 GEMM 里乘进了每一路的结果，所以这里只做求和。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1850-L1852；blog/fused_moe_modular_kernel.md L80

### 门控分数与数值精度

#### 门控分数乘进哪一层

门控分数乘在哪次矩阵乘上，由内核的一个常量开关决定。默认乘在输出上：第一次 GEMM 不乘，第二次乘。有些模型（例如 top-$k$ 等于 $1$ 的情形）要求先把门控分数乘在输入上，实现上是乘在第一次 GEMM 的结果上、第二次不乘。两次调用传给开关的值正好相反。乘法要在累加器转成输出精度之前做，以保证数值稳定。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L584-L606；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1787-L1798；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1826-L1838；blog/moe_kernel_features.md L17

#### 累积与量化

矩阵乘沿隐藏维 $H$ 把输入切成小段逐段乘加，累加器用 32 位浮点以保证精度，循环结束后才转成输出精度。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L511-L518；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L600-L601

量化的意思是用更少的位数存权重或激活，另配一个缩放因子，用时再乘回去。配了量化时，激活在进 GEMM 前压成低精度：第一次 GEMM 之前压一次输入，第二次 GEMM 之前压一次激活函数的输出。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1766-L1772；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1815-L1821

缩放因子在累加完之后乘回：权重的缩放乘在所有输出上，激活的缩放乘在对应的行上。如果量化按块分组（每个量化块共用一个缩放因子），缩放因子随内层维度的块逐段乘。有些专家带偏置，就是两次矩阵乘各自多加一项与输出维等长的向量，存在另一个按专家分开的张量里；偏置不量化，也不该被缩放因子缩放，所以加在去量化之后。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L505-L509；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L539-L550；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L574-L588

### 适用条件与边界

token 数很少、专家数很多时，有一条跳过对齐的快速路径：当「token 数乘 top-$k$ 再乘 $4$」不超过全局专家数 $E$，且没有专家编号重映射（专家并行下每个设备只持有部分专家，编号要换成设备内的编号）、权重也不是按块分组量化的 int8 或 int4 时，内核不做排序与补齐，直接让每个行块对应一路（一个 token 的一次专家选择），块内只有一行有效，权重按这一路选中的专家取。$4$ 是一个启发式的余量系数，意在让这一批只激活全部专家中的一小部分。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py L81-L84；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1554-L1566；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L374-L376；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L411-L414

行块大小这类参数优先从按设备名命名的调优表里查。表的文件名里带着专家数、中间维、数据类型、量化分块大小和硬件名，查表时取表里最接近当前批大小的一档。找不到对应的表时，退回按批大小规则给出的默认配置，并警告性能可能欠佳。
依据：repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1093-L1104；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1114-L1121；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1161-L1168；repo/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py L1436-L1447
