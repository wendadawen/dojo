# Mixed-Precision Quantization

- 描述：逐层混合精度量化给同一模型的不同层分别指定不同的量化档位，对量化误差敏感的层留高比特、不敏感的层压低比特。
- 摘要：各张量对量化误差的耐受度不同：校准数据统计出的重要性越高、或写回残差流的张量，档位留得越高；占体积主体的路由专家 gate/up 压到最低档。Hy4 的 MIX-STQ1_0 里路由专家 `gate`/`up` 在 $29$ 层用 $1.3125$ bpw、另 $48$ 层用 $2.0625$ bpw，整模平均 $2.38$ bpw。
- 主题：推理系统, 内存与缓存
- 标签：量化

## 资料

- Hy4-preview GGUF 模型卡. Hugging Face, 2026. [https://huggingface.co/AngelSlim/Hy4-preview-GGUF](https://huggingface.co/AngelSlim/Hy4-preview-GGUF)
- llama.cpp tools/imatrix README. GitHub, 2026. [https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
- MIX-STQ1_0 张量类型配方. Hugging Face, 2026. [https://huggingface.co/AngelSlim/Hy4-preview-GGUF/blob/main/hy4-preview-patch/Hy4-preview-STQ1_0.tensortypes](https://huggingface.co/AngelSlim/Hy4-preview-GGUF/blob/main/hy4-preview-patch/Hy4-preview-STQ1_0.tensortypes)

## 短页

### 同一模型里，不同层存成不同比特宽度

量化把模型的权重（模型里存的数值参数）从高精度浮点存成更少的比特。模型的主干是一叠结构相同的层，一层里的权重分成若干张量（多维数组），功能相同、各层同名的那些张量叫一个张量族。每个权重平均占的比特数叫每权重比特（bpw）。把权重压到更低位宽会引入量化误差，也就是存进去再取出来的值与原来的值之差。每个量化档位是一套量化格式，各自对应一个固定的每权重比特，所以档位之间可以按这个数排序。单一档位量化让整个模型用一个档位。逐层混合精度量化把模型按张量族和层拆开，每一组各取一个档位，这些档位按参数量加权，得到整模的平均每权重比特。路由专家是被挑中的少数前馈子网络，前馈指每层里逐个位置各算各的那一部分；每个路由专家的前馈有 gate、up、down 三个投影，投影指一层里一个线性变换，gate 与 up 把输入升到中间维度、down 再降回输出。同一批权重里，一部分层的路由专家 gate、up 两个投影用 $1.3125$ bpw、另一部分层用 $2.0625$ bpw，其余张量族留高精度，整模平均 $2.38$ bpw，就是一种这样的分配。
依据：blog/hy4-gguf-model-card.md L8-L12；blog/hy4-gguf-model-card.md L30-L32；blog/hy4-gguf-model-card.md L105-L111

### 各张量对误差的耐受度不一样

同一个模型里，各张量对量化误差的耐受度不同，把所有层压到同一档位不是最优。有一类张量的输出直接写回残差流（各层输出依次加回去的那条主干表示），量化误差不会被后面的门控衰减，它们要留更高的档位。路由专家的三个投影里，down 的输出直接写回残差流，量化误差不被后续门控衰减，所以比 gate、up 高两档。
依据：blog/hy4-gguf-model-card.md L105-L111

### 先量重要性，再按张量族分档

校准文本是一批有代表性的文本。让模型在它们上跑一遍前向（输入经过各层算出输出的整段过程），逐张量统计它的取值，得到每个张量在真实数据下的重要程度，这份分数叫重要性矩阵（imatrix）。再按这份分数和每个张量在计算里的角色，给每个张量族和层指定档位：占体积主体的压到最低档，写回残差流的留高档，其余较小的张量族留高精度。
依据：blog/llama-cpp-imatrix-README.md L92-L102；blog/llama-cpp-imatrix-README.md L3-L4；blog/hy4-gguf-model-card.md L105-L116

## 长页

把一个大模型放进内存有限的设备里，最直接的手段是量化：用比训练时更少的比特来存权重（模型里存的数值参数）。每个权重平均占用的比特数叫每权重比特（bits per weight，bpw），等于整份权重的总比特数除以权重个数。把权重压到更低位宽会引入量化误差，也就是存进去再取出来的值与原来的值之差。
依据：blog/hy4-gguf-model-card.md L8-L12

模型的主干是一叠结构相同的层，一层里的权重分成若干张量（多维数组）。前馈是每层里逐个位置各算各的那一部分；前向是输入经过这些层算出输出的整段过程，前向中流动和产生的张量，也就是各层的输入和输出，统称激活。校准是一段单独的过程：挑一批有代表性的文本（校准文本），让模型在它们上做前向，同时统计各张量的取值，得到每个张量在真实数据下的重要程度，这份分数叫重要性矩阵（imatrix）。张量族指功能相同、在所有层里同名的那些张量；投影指一层里一个线性变换，它有自己的一组权重。
依据：blog/llama-cpp-imatrix-README.md L92-L102；blog/llama-cpp-imatrix-README.md L3-L4

每个量化档位都是一套具体的格式，各自对应一个固定的每权重比特。最简单的做法是让整个模型用一个档位，即单一档位量化，所有层都被压到同一精度。逐层混合精度量化改的是这一点：把模型按张量族和层拆开，每一组各自指定一个档位，整模的平均每权重比特是这些档位按参数量加权的平均。这样把比特从耐受度高的层挪给耐受度低的层，总比特数不需要增加。要做这种分配，先要有一个信号说明哪些张量更重要，那就是校准得到的重要性分数；分配的结果通常写成一份档位清单（配方），逐条写哪个张量族用哪一档。
依据：blog/hy4-gguf-model-card.md L30-L32；blog/hy4-gguf-model-card.md L105-L116；repo/Hy4-preview-STQ1_0.tensortypes L109-L111

### 各张量族取不同档位

大模型的前馈层常换成混合专家（MoE）：前馈层变成很多并行的专家子网络，模型先把文本切成 token（词或词的片段），每个 token 只走其中少数几个专家，参数总量因此可以远大于每个 token 实际用到的计算量。被选中的那些叫路由专家。每个路由专家的前馈有三个投影：gate 与 up 把输入升到中间维度，down 再降回输出维度，对应的权重张量分别叫 `ffn_gate_exps`、`ffn_up_exps`、`ffn_down_exps`。
依据：blog/tencent-hy4-preview-model-card.md L63；blog/hy4-gguf-model-card.md L105-L111

分配在路由专家上的样子是：gate 与 up 两投影按层分两档，一部分层用最低档 STQ1_0，其余层用高一档 IQ2_XXS，用哪一档由该层的重要性分数推导；down 投影比最低档 STQ1_0 高两档，用 IQ3_XXS，最后三层再提一档到 IQ4_XS。gate、up 占的体积最大，档位跟着重要性分数走；down 的输出直接写回残差流（各层输出依次加回去的那条主干表示），量化误差不会被后面路由输出处的门控（按专家得分给各专家输出加权的环节）衰减，所以刻意留高两档。
依据：blog/hy4-gguf-model-card.md L105-L111

其余较小的张量族，配方让它们取更高的档位：注意力（[标准注意力](wiki/standard-attention)，在位置之间传信息的那一部分）和决定每个位置能看多远的稀疏索引器（[DSA](wiki/dsa)）里，一部分用 $5$ 位档 Q5_K、一部分用 $8$ 位档 Q8_0；混合矩阵（主干各条表示之间的混合权重）、router（给专家打分选路的一层）、归一化层（把激活重新标定的层）、sink（[attention sink](wiki/attention-sink)，序列开头接收多余注意力分的 token 用到的偏置）与输出层（把最后一层表示映成词表分数的张量）保持高精度浮点（F32）。
依据：blog/hy4-gguf-model-card.md L105-L116

::: 各张量族的档位
| 张量族 | 档位 |
|---|---|
| 路由专家 gate、up | STQ1_0（29 层）／IQ2_XXS（48 层） |
| 路由专家 down | IQ3_XXS，最后三层 IQ4_XS |
| 注意力的输出、门控与查询投影 | Q5_K |
| 注意力的压缩投影（MLA） | Q8_0 |
| 稀疏索引器 | Q8_0／F32 |
| 混合矩阵、router、归一化层 | F32 |
| sink 与输出层 | F32 |
依据：blog/hy4-gguf-model-card.md L105-L116

这个模型的主干有 $78$ 层，从 $0$ 编号到 $77$，其中第 $0$ 层是普通的稠密前馈层，其余 $77$ 层才有路由专家。路由专家的 gate、up 两投影在这 $77$ 层里，$29$ 层用 $1.3125$ bpw、$48$ 层用 $2.0625$ bpw，按层数加权得到 $1.78$ bpw；三个路由专家族占模型全部参数的 $97.7\%$，其余张量加起来不到 $3\%$。最后三层指编号 $75$、$76$、$77$ 的三层。
依据：blog/tencent-hy4-preview-model-card.md L63；blog/hy4-gguf-model-card.md L105；blog/hy4-gguf-model-card.md L110；repo/Hy4-preview-STQ1_0.tensortypes L168-L171
:::

### 敏感度怎么量：校准数据与重要性分数

重要性分数把激活的若干统计量汇成一个张量级的重要程度，档位就按这份分数的高低分配。
依据：blog/llama-cpp-imatrix-README.md L92-L102

::: 逐张量统计的口径
校准工具按张量统计这些量。
依据：blog/llama-cpp-imatrix-README.md L92-L94

1. 均值与标准差：刻画该张量激活的分布。
2. 偏度与峰度：峰度高说明激活稀疏、长尾，有离群值，这类张量通常需要更高精度。
3. 熵：激活在各神经元上的分布是否均匀。
4. 激活能量：激活平方之和；能量高的张量通常需要更高精度。
依据：blog/llama-cpp-imatrix-README.md L95-L102
:::

### 一个两层两档的例子

把模型缩到两层、每层一个权重单位，档位只取低、高两档，就能看清分配的作用。设两层参数量相同，两种分法用同一个总比特数，平均每权重比特一样。每层用低档与用高档各有一个误差，两层这两个误差之差不同：把高档分给差值更大的那层，总误差比反过来分小得多。平均比特相同时，比特落在哪一层由两层各自的误差差决定。
依据：自行构造

::: 两种分配的逐步手算
1. 设敏感层与不敏感层各有一个权重单位，每层可选低档 $1.25$ bpw 或高档 $2.0$ bpw。
2. 两层用同一总比特数 $1.25 + 2.0 = 3.25$，平均 $\frac{3.25}{2} = 1.625$ bpw，两种分配的平均一样。
3. 敏感层用低档会损失 $100$ 的精度、用高档只损失 $20$，两者之差 $80$；不敏感层用低档损失 $30$、用高档损失 $25$，两者之差 $5$。
4. 方案一，敏感层高档、不敏感层低档：总误差 $20 + 30 = 50$。
5. 方案二，敏感层低档、不敏感层高档：总误差 $100 + 25 = 125$。
6. 两边之差 $80 > 5$，所以把高档给敏感层总误差最小；比值 $\frac{125}{50} = 2.5$，分配方向选反对总误差的影响很大。
依据：自行构造
:::

### 配方文件的写法与两处易错点

落地方案是一份张量类型配方文件：每一行写张量名模式等于档位，量化工具逐张量按这些行决定档位。模式按出现顺序做子串匹配，第一条命中的规则生效，所以具体的逐层规则必须排在泛化的规则前面，否则会先被泛化规则命中。
依据：repo/Hy4-preview-STQ1_0.tensortypes L121-L124；repo/Hy4-preview-STQ1_0.tensortypes L167-L171

一是压缩后的注意力 [MLA](wiki/mla) 的 q_b 投影名 `attn_q_b.weight` 也会匹配到索引器里的 `indexer.attn_q_b.weight`，所以索引器的规则要排在通用规则之前。二是单独一行 `ffn_down_exps` 会覆盖它下面的逐层提档，所以那三行必须写在泛化行的前面。
依据：blog/hy4-gguf-model-card.md L111-L113；repo/Hy4-preview-STQ1_0.tensortypes L122-L124；repo/Hy4-preview-STQ1_0.tensortypes L137；repo/Hy4-preview-STQ1_0.tensortypes L147；repo/Hy4-preview-STQ1_0.tensortypes L167-L171

::: 配方里的顺序与两处易配错的规则
1. 索引器规则排在注意力规则之前：`indexer.attn_q_b=q8_0`、`indexer.attn_k=q8_0`、`indexer.proj=f32`。
2. 最后三层的 down 规则排在泛化 down 之前：`blk.75.ffn_down_exps=iq4_xs`、`blk.76.ffn_down_exps=iq4_xs`、`blk.77.ffn_down_exps=iq4_xs`、`ffn_down_exps=iq3_xxs`。
3. 泛化的 `attn_q_b.weight=q8_0` 排在索引器规则之后。
依据：repo/Hy4-preview-STQ1_0.tensortypes L121-L124；repo/Hy4-preview-STQ1_0.tensortypes L138-L150；repo/Hy4-preview-STQ1_0.tensortypes L167-L171
:::

### 适用条件与边界

逐层混合精度量化只决定每一层用哪一档，不改单档编码本身。每个档位背后的量化格式怎么把权重压成比特、编码器（把权重按该格式压成比特的程序）怎么求解，是格式和编码器的事，不是这套分配解决的。
依据：blog/hy4-gguf-model-card.md L93-L95

它依赖校准。STQ1_0 的编码器强制需要 imatrix，用它求解一组权重共用的缩放因子与选择把哪个权重强制置零，没有它就不能编码。
依据：blog/hy4-gguf-model-card.md L122-L123；blog/hy4-gguf-model-card.md L90-L91

它依赖运行时。这些档位并不都是通用格式，产物要由能加载它们的运行时才能用，最低档 STQ1_0 还需要运行时侧的额外支持。
依据：blog/hy4-gguf-model-card.md L14-L15；blog/hy4-gguf-model-card.md L44-L45
