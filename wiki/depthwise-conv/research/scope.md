# scope.md：深度可分离卷积

## 1. 概念含义

- 名称：深度可分离卷积（Depthwise Separable Convolution）
- 简要定义：把标准卷积「空间滤波 + 跨通道组合」两件事拆成两步：先逐通道滤波（depthwise），再 1×1 跨通道线性组合（pointwise），乘加量降为约 $1/N+1/D_K^2$ 倍
- 正式定义：成本 $D_K\cdot D_K\cdot M\cdot D_F\cdot D_F+M\cdot N\cdot D_F\cdot D_F$（MobileNets Eq. 5），由 depthwise（Eq. 3/4）与 pointwise 组成
- 语境：以 MobileNets 为原始出处讲解拆分与省算原理；落点是 KDA 等线性注意力层里的短深度卷积（只用 depthwise 半边）
- 包括：标准卷积的拆解、两步定义、$1/N+1/D_K^2$ 推导、KDA 用法（GLM conv1d groups 验证）
- 不包括：MobileNet 的宽度/分辨率乘数、网络架构全文、卷积的反向传播
- 相邻概念：标准卷积（无专页，页内以 Eq.1 最小自包含）；ViT patch embedding（有页 vit，提及 Conv 用法不展开）

## 2. 学习目标

### Q1：标准卷积一步同时做哪两件事？成本由什么决定？

- 答案：滤波（空间邻域加权）与组合（跨通道线性混合）一步完成（Eq.1）；成本 $D_K^2 M N D_F^2$（Eq.2），对 M、N、$D_K$、$D_F$ 都是乘性依赖
- 核心理由：拆分的前提是看清「一步做了两件事」
- 依赖：加权求和（自包含）

### Q2：拆成两步后各自做什么？总成本多少？

- 答案：depthwise 每通道一个 $D_K\times D_K\times1$ 核只滤波（Eq.3/4）；pointwise 用 $1\times1\times M\times N$ 核只组合；总成本 Eq.5
- 核心理由：本页定义性内容
- 依赖：Q1

### Q3：省了多少？$1/N+1/D_K^2$ 怎么来？

- 答案：Eq.5/Eq.2 逐项相除得 $1/N+1/D_K^2$；$3\times3$、$M=N$ 时约 8~9 倍（论文结论）；构造数字例 $D_K=3,M=N=64$ 实算比值 0.126736、7.9 倍
- 核心理由：这是该结构被采用的核心动机
- 依赖：Q2

### Q4：KDA 里的短卷积是完整的深度可分离卷积吗？

- 答案：不是。GLM KDA 的 conv1d 是 depthwise（groups=24576=通道数、kernel=4、因果），没有配对的 pointwise；跨通道混合由其后的 q/k/v 线性投影承担。作用是给线性注意力提供局部时序上下文（无位置编码模型里也提供局部相对位置）
- 核心理由：纠正「见到 groups=C 就叫深度可分离」的常见误读；也是本概念在大模型里的真实用例
- 依赖：Q2；线性注意力语境链接 linear-attention/kda 页

## 3. 内容分级

核心：Q1-Q4。
辅助：1D 因果版（KDA 用的是 1D）；手算例。
排除：MobileNet 全架构、反向传播、深度卷积的历史（Xception 提及一句出处即可）。

## 4. 前置知识映射

- 标准卷积：无页面 → 页内第 1 章自包含（Eq.1+符号）
- 线性注意力 / KDA：有页面 linear-attention / kda → 第 4 章链接
- 维度符号（$D_F,M,N,D_K$）：页内定义

## 5. 不展开

- depthwise 卷积 GPU 实现效率问题（im2col 稀疏性等）：工程细节
- Xception 的「极端 Inception」叙事：一句出处，不展开
- depthwise 反向传播的不对称性

## 6. 误解与边界

误解：
1. 「groups=C 的卷积就是深度可分离卷积」——错：那只是 depthwise 半边；深度可分离 = depthwise + pointwise 两步
2. 「拆开一定省 8~9 倍」——比值是 $1/N+1/D_K^2$，N 小或 $D_K$ 大时收益缩水；8~9 倍对应 $3\times3$ 且 N 较大的情形
3. 「精度无损」——论文说「at only a small reduction in accuracy」，有损但小；引用时不夸大

边界：
- 成本公式假设 stride=1、正方形核与特征图；其他配置需另行推导
- KDA 段的结论只对应 GLM-5.3-Flash 的实现（源码验证），不代表所有线性注意力实现
