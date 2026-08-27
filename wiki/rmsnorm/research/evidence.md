# evidence.md：RMSNorm

来源缩写：
- [RN] Zhang & Sennrich, Root Mean Square Layer Normalization, arXiv:1910.07467v1, NeurIPS 2019
- [GLM] transformers main, modeling_glm5_next.py Glm5NextTextRMSNorm L66-83
- [实测] research/concept_probes.out 探针 B

## C 论断

- C1 LayerNorm 同时做 re-centering 与 re-scaling，其计算开销拖慢网络（尤其 RNN）：[RN] 摘要。已确认
- C2 论文假设 re-centering 不变性非必需、re-scaling 才是成功原因：[RN] §1「We argue…does not reduce the variance…」与 §4「We hypothesize…」。已确认（假设，页面标注性质）
- C3 零均值时 RMSNorm 与 LayerNorm 完全相等：[RN] §4 原句。已确认；实测差 0（探针 B2）
- C4 RMSNorm 完整保留 re-scaling 不变性（权重/数据/单样本缩放），放弃 re-centering 类性质：[RN] §4.1 Table 1。已确认
- C5 梯度对 W 的项与权重缩放负相关=隐式学习率自适应：[RN] §4.2。已确认
- C6 实验中 RMSNorm 的激活均值与 Baseline 相比同样稳定（Table 5：波动 -0.40~-0.74 vs baseline -2.60~-1.19）：[RN] §6.1。已确认
- C7 Figure 4：异常初始化下 RMSNorm 比 LayerNorm 更鲁棒或相当：[RN] §6.1。已确认
- C8 各任务质量与 LayerNorm 相当（Transformer BLEU 26.8/27.7 vs 26.6/27.7）；无归一化的 Transformer 训练失败：[RN] §6。已确认
- C9 GLM 实现：fp32 内部计算、eps（variance_epsilon=1e-5）、权重乘在转换回原 dtype 之后：[GLM] L75-80。已确认

## F 公式

- F1 加权和 $a_i=\sum_j w_{ij}x_j$：[RN] Eq.(1)。已确认
- F2 LayerNorm：$\bar{a}_i=\frac{a_i-\mu}{\sigma}g_i$ 与 $\mu,\sigma$ 定义：[RN] Eq.(2)(3)。已确认
- F3 RMSNorm：$\bar{a}_i=\frac{a_i}{\mathrm{RMS}(\mathbf{a})}g_i$，一般式 $\mathbf{y}=f(\frac{\mathbf{Wx}}{\mathrm{RMS}(\mathbf{a})}\odot\mathbf{g}+\mathbf{b})$：[RN] Eq.(4)(5)。已确认
- F4 运算量差 $3n$（求均值 n 加法、方差 n 减法、归一化 n 减法）：由 F2/F3 公式直接清点；非论文原文（论文无逐项计数）。已确认（推导），页面标注为推导
- F5 实测：(1,2,3,4) 的 LN 输出 [-1.3416,-0.4472,0.4472,1.3416] vs RMS [0.3651,0.7303,1.0954,1.4606]；零均值向量差 0；torch.nn.RMSNorm 与手写差 1.19e-7；GLM 式（fp32+ε=1e-5）与论文式差 9.5e-7：[实测] B1-B4。已确认（构造示例）

## N 数字

- N1 加速 7%~64%（Transformer+RMSNorm 6.9%、RNNSearch 24.7%/34.0%/11.0%、Order-Embedding 40.8%、CNN 20.5%）：[RN] Table 2/3/4/6/8/10。已确认；标注 2019 年硬件与框架
- N2 Table 5 均值波动区间：[RN] §6.1。已确认
- N3 GLM 的 rms_norm_eps=1e-5：[GLM] config（GLM 数据流页核对）。已确认

## 冲突与缺口

- 论文 v1 公式无 ε，工程实现（PyTorch/GLM）普遍加：如实分开陈述（C9 与 F3 并列，不混写）
- 「3n 次运算」是本页按公式清点的推导，论文原文无此计数：标注为推导而非论文数字
