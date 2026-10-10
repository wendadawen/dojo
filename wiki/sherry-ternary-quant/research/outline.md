# 稀疏三值量化

- 描述：把神经网络权重量化到三个取值 $\{-d,0,+d\}$，并强制每 4 个权重恰好一个为零，使 4 个权重恰好用 5 bit 编码，平均 1.25 bit/权重。
- 摘要：稀疏三值量化用一条 3:4 稀疏约束把 4 个权重装进 5 bit：每 4 个权重恰好一个零，其余三个取 $\pm d$。合法组合从 $3^4=81$ 压到 $32=2^5$，5 bit 编码零浪费，分组大小 4 又是 2 的幂，解码天然对齐 SIMD——同时解决三值量化「小的不快、快的不小」的两难。
- 主题：训练与优化，模型结构
- 标签：量化

## 长页

- 开头｜三值是什么；打包两难，Sherry 用 3:4 兼顾；共用一个缩放 d；给权重量化、bpw、SIMD、QAT/PTQ 最小含义，公式折叠｜460｜paper/src/Contents/2preliminary.tex L5-L14；paper/src/Contents/2preliminary.tex L20-L21；paper/src/Contents/2preliminary.tex L30-L32；paper/src/Contents/0abstract.tex L1-L5
- ### 三值权重的打包困境｜三值下限 1.585 bit 不是 2 的幂；2-bit 浪费、1.67-bit 三路分组更慢，两头不划算｜460｜paper/src/Contents/2preliminary.tex L30-L32
- ### 3:4 稀疏：4 个权重 5 bit｜本章合讲 3:4 约束与 5 bit 编码，内容全在下面三个小节里｜0｜paper/src/Contents/3method.tex L12-L24
- #### 32 种组合与 1.25 bit｜每 4 个权重恰好一个零，组合从 81 压到 32=2^5，4 权重用 5 bit，平均 1.25 bit｜460｜paper/src/Contents/3method.tex L67
- #### 4 bit 索引 + 1 bit 符号｜镜像对称把 32 种组合配成 16 对，拆成 4 bit 索引与 1 bit 符号，恰好填满 SIMD 查表｜460｜paper/src/Contents/3method.tex L68；paper/src/Contents/8appendix.tex L17-L23
- #### 一组 4 权重走一遍｜取一组 4 权重从定零位、查码本、定符号到乘缩放走一遍，看误差，全过程折叠｜100｜自行构造
- ### 为什么是 4 与 3:4｜在目标约 1.25 bit 下，由分组大小 M 取 2 的幂、查表索引上限、稀疏度不超 50% 三条约束，推出 3:4 是唯一优解，推导折叠｜200｜paper/src/Contents/8appendix.tex L62-L85
- ### STQ1_0 的文件布局与 1.3125 bpw｜256 权重 42 字节的构成、1.25 bit 与 1.3125 bpw 之差、stride-16 分组，布局与推导折叠｜420｜paper/src/Contents/8appendix.tex L13-L25；blog/huggingface-Hy4-preview-GGUF.md L86-L105；blog/llamacpp-PR22836-body.md L13-L16
- ### 编码器决定误差｜amax 把缩放钉在最大离群值上；改用 imatrix 加权缩放与零位代价，交替 3 轮降九成误差，公式折叠｜460｜blog/huggingface-Hy4-preview-GGUF.md L93-L105；repo/AngelSlim-sherry/Sherry/models/utils_quant.py L87-L115
- ### 训练侧的权重陷阱与 Arenas｜硬约束下梯度同质化使权重坍缩成类二值；Arenas 退火残差恢复多样性，推理零开销｜460｜paper/src/Contents/3method.tex L73-L86；paper/src/Contents/3method.tex L90；paper/src/Contents/8appendix.tex L186-L205；repo/AngelSlim-sherry/Sherry/models/utils_quant.py L583-L638
- ### 实测：速度与精度｜实测本格式：精度与更高比特的三值方法持平，及自身解码速度与体积，数字与条件折叠｜420｜blog/llamacpp-PR22836-body.md L120；paper/src/Contents/4experiments.tex L99-L116
- ### 适用条件与边界｜1.25 bit 是纯权重编码、只解决单层比特、不解决激活；精度依赖 QAT 或带校准 PTQ｜420｜paper/src/Contents/7limitation.tex L1-L10；paper/src/Contents/2preliminary.tex L20-L21；blog/huggingface-Hy4-preview-GGUF.md L93-L105

## 短页

- ### 三值权重与 3:4 稀疏｜权重取 $\{-d,0,+d\}$ 三个值，$d$ 是一组共用的缩放；每 4 个恰好一个零，其余三个取 $\pm1$｜240｜paper/src/Contents/2preliminary.tex L5-L14；paper/src/Contents/3method.tex L12-L24
- ### 打包两难逼出的 5 bit｜三值装不满 2 的幂：2-bit 对齐却浪费、1.67-bit 紧凑却更慢；3:4 组合数正好 2^5，4 个权重装进 5 bit｜330｜paper/src/Contents/2preliminary.tex L30-L32；paper/src/Contents/3method.tex L67
- ### 编码与解码｜32 种合法组合收成一张 32 项码本，用 4 bit 索引加 1 bit 符号查表；解码查表、定符号、乘缩放｜370｜paper/src/Contents/8appendix.tex L17-L23；blog/huggingface-Hy4-preview-GGUF.md L86-L105
