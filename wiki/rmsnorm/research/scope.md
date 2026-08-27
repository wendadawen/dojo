# scope.md：RMSNorm

## 1. 概念含义

- 名称：RMSNorm（Root Mean Square Layer Normalization，均方根层归一化）
- 简要定义：LayerNorm 去掉均值中心化（re-centering）、只保留缩放（re-scaling）的归一化层：除以均方根 RMS 而不是标准差
- 正式定义：$\bar{a}_i=\frac{a_i}{\mathrm{RMS}(\mathbf{a})}g_i$，$\mathrm{RMS}(\mathbf{a})=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a_i^2}$（论文 Eq. 4）
- 语境：作为层归一化的一种，讲解它与 LayerNorm 的关系、为什么省掉均值仍然有效、省了多少计算
- 包括：LayerNorm 公式与两件事的拆解、RMSNorm 公式、零均值等价、有效性论证（假设+实验证据）、计算量与实现细节
- 不包括：BatchNorm/GroupNorm 对比（非本页主线）；pRMSNorm 展开计算（一句提及）；训练动力学分析
- 相邻概念：LayerNorm（无专页，页内以对照形式给最小定义）；残差连接（有页，提及 Pre-Norm 语境即可，不展开）

## 2. 学习目标

### Q1：LayerNorm 对神经元输入做哪两件事？

- 完成答案：对加权和 $\mathbf{a}$（Eq.1）做中心化（减均值 $\mu$，Eq.3）与缩放（除标准差 $\sigma$），再乘增益 $g_i$ 加偏置 $b_i$（Eq.2）。中心化=re-centering，缩放=re-scaling
- 核心理由：RMSNorm 的定义就建立在对这两件事的取舍上
- 依赖：均值/方差/标准差（基础统计，自包含）

### Q2：RMSNorm 去掉了什么、留下什么？公式差别在哪？

- 完成答案：去掉减均值，把 $\sigma$（对均值的偏差）换成 RMS（对 0 的偏差）：Eq.4。输出不再零均值（例：(1,2,3,4) 的 RMSNorm 输出全正）；输入零均值时两者完全相等（实测差 0）
- 核心理由：这是本页的定义性内容
- 依赖：Q1

### Q3：为什么去掉中心化仍然有效？

- 完成答案：论文假设 LayerNorm 成功的原因是 re-scaling 不变性而非 re-centering（§4）；论据链：①零均值时两者等价 ②RMSNorm 完整保留 re-scaling 不变性（Table 1）③实验 Table 5 显示 RMSNorm 的激活均值其实同样稳定 ④Figure 4 鲁棒性
- 核心理由：这是理解「为什么敢去掉」的关键，也是论文的主要贡献
- 依赖：Q2；不变性的含义页内自包含

### Q4：省了多少计算？实现上要注意什么？

- 完成答案：对 $n$ 维输入省去约 $3n$ 次标量运算（求均值 n 加、方差 n 减、归一化 n 减）；论文实测 7%~64% 运行时间下降（依框架/硬件/架构，Transformer 上 6.9%）；实现普遍加 ε 防除零（论文 v1 公式无 ε）、在 fp32 里算再转回（GLM 实现核对）
- 核心理由：RMSNorm 被广泛采用的直接动机是效率，数字要给准
- 依赖：Q2

## 3. 内容分级

核心：Q1-Q4 全部。
辅助：pRMSNorm 一句提及；「GLM-5.3-Flash 用它」的实现证据。
排除：BN 对比、各框架 benchmark 细节（只引论文汇总区间与 Transformer 端点）。

## 4. 前置知识映射

- 均值/标准差：无页面，自包含（一句话）
- LayerNorm：无页面，本页第 1 章以最小定义自包含给出（它是 RMSNorm 的对照基线，属本页核心内容而非外部前置）
- 残差连接（residual-connection 页）：Pre-Norm 语境一句提及，不依赖

## 5. 不展开

- pRMSNorm 的部分估计推导与实测（论文 §5-6）
- 归一化与优化 landscape 的理论（只引用论文转述 Santurkar et al. 的一句结论，标注转述）

## 6. 常见误解与边界

误解：
1. 「RMSNorm 是 LayerNorm 的一种近似」——它是精确的不同算子；零均值输入下两者相等，一般情况下输出不同且都不零均值（LayerNorm 输出零均值、RMSNorm 不）
2. 「去掉中心化会掉精度」——论文各任务质量与 LayerNorm 相当（如 Transformer BLEU 26.8/27.7 vs 26.6/27.7）；这是论文实验结论，标注来源
3. 「论文公式里有 ε」——v1 原文 Eq.(3)(4) 均无 ε；ε 是工程实现加的（GLM/PyTorch 都加）

边界：
- 7%~64% 的加速区间是 2019 年硬件/框架下的实测（TITAN X/V100/2080Ti 等），不能直接换算到当代硬件
- 有效性论证是「假设+证据」而非定理；页面如实分级
