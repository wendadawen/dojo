# evidence.md：超连接与 mHC

来源缩写：
- [HC] Zhu et al., Hyper-Connections, arXiv:2409.19606v3 (2025-03-18)
- [mHC] Xie et al., mHC: Manifold-Constrained Hyper-Connections, arXiv:2512.24880v2 (2026-01-05)
- [GLM] transformers main, src/transformers/models/glm5_next/modeling_glm5_next.py（Glm5NextTextHyperConnection L219-295）
- [实测] wiki/hyper-connections/research/concept_probes.out（本机实跑输出）

## C 论断

- C1 恒等映射是残差稳定性的来源：[HC] §1 引用 he2016identity；[mHC] §1 Eq.(1)(2) 及「identity mapping refers to the component $\mathbf{x}_l$ itself」。已确认
- C2 Pre-Norm 缓解梯度消失但深层表示坍塌、Post-Norm 相反（seesaw effect）；残差连接（两种变体）预定义了层输入输出连接强度，是 HC 要松开的东西：[HC] §1 摘要与引言原文。已确认
- C3 Pre-Norm 与 Post-Norm 都是 n=1 的非可训练超连接特例：[HC] §3.1 Eq.(15)(16)。已确认
- C4 n=1 时 seesaw 仍在、性能不升；n>1 才能同时调强度与重排层：[HC] §1。已确认
- C5 HC 的 $\mathcal{H}^{\mathrm{res}}$ 无约束导致复合映射偏离恒等，前向/反向信号爆炸或消失：[mHC] §1 Eq.(4) 与 §3.1 论述。已确认
- C6 HC 复合映射增益峰值≈3000（理想 1），27B 模型约 12k 步处 loss 突增：[mHC] §3.1（图 5）。已确认（实验数字，引用时标注实验条件）
- C7 mHC 复合映射最大增益≈1.6，比 HC 低三个数量级：[mHC] §5.4。已确认（实验数字）
- C8 双随机矩阵三性质：谱范数≤1；乘法封闭；Birkhoff polytope 是置换矩阵的凸包：[mHC] §4.1 三条性质原文。已确认
- C9 n=1 时双随机条件退化为标量 1，恢复恒等映射：[mHC] §4.1 原文「when n=1, the doubly stochastic condition degenerates to the scalar 1」。已确认
- C10 mHC 对 pre/post 施加非负约束（防正负系数复合抵消）：[mHC] §4.1 末段。已确认
- C11 GLM-5.3-Flash：hc_mult=4、每层两站点（attn_hc/ffn_hc）、fn=[24,16384]、pre=σ+ε、post=2σ、comb=softmax(dim=-1)+ε 后列归一一次再循环 19 次（行、列）、末端 hc_head 无权重均值：[GLM] L219-302 与 L1477/1493。已确认
- C12 GLM 实现与论文三处差异（初始正矩阵 softmax vs exp；归一化分母加 ε；迭代序末步为列方向）：[GLM] L286-290 对照 [mHC] Eq.(9)。已确认（差异本身是源码事实）
- C13 GLM mHC 全模型参数 35,391,870、占 321.32B 的 0.011%、激活显存代价 4×：[GLM] 张量头统计与 L1477；参数量已由 checkpoint 交叉验证。已确认

## F 公式

- F1 HC 单层传播 $\hat{\mathbf{H}}=\mathbf{B}^{\intercal}\mathcal{T}(\mathbf{H}^{\intercal}\mathbf{A_{m}})^{\intercal}+\mathbf{A_{r}}^{\intercal}\mathbf{H}$：[HC] §2.1 Eq.(2)，配 Eq.(3)(4)(5) 的三映射定义。已确认
- F2 标准残差递归 $\mathbf{x}_L=\mathbf{x}_l+\sum\mathcal{F}$：[mHC] §1 Eq.(2)（[HC] 引言同形式）。已确认
- F3 HC 复合映射（不稳定根源）：[mHC] §1 Eq.(4)。已确认
- F4 mHC 流形约束定义：[mHC] §4.1 Eq.(6)。已确认
- F5 mHC 参数化（展平+线性投影）与最终映射（σ/2σ/Sinkhorn）：[mHC] §4.2 Eq.(7)(8)。已确认
- F6 Sinkhorn-Knopp 迭代 $\mathbf{M}^{(t)}=\mathcal{T}_r(\mathcal{T}_c(\mathbf{M}^{(t-1)}))$，$\mathbf{M}^{(0)}=\exp(\tilde{\mathcal{H}}^{\mathrm{res}})$，$t_{\max}=20$：[mHC] §4.2 Eq.(9) 及正文。已确认
- F7 Sinkhorn 数值行为（3×3 构造矩阵 20 次行列和均到 0 偏差；双随机乘积行列和恰为 1；谱范数=1.000000；无约束随机链 24 层行和 1.1e6 vs 双随机链恒 1）：[实测] 探针 A1-A5。已确认（构造示例）
- F8 GLM 实现公式 pre=σ(·)+ε、post=2σ(·)、comb=softmax(·)+ε→Sinkhorn、h'=post⊗y+comb^T@h：[GLM] L283-294 与 L1316-1318；实测等价性已在 GLM 数据流页验证（该页 p3 探针）。已确认

## N 数字

- N1 27B、12k 步、增益 3000 vs 1.6：[mHC] §3.1/§5.4，实验条件为 mHC 论文的训练设置。引用时标注
- N2 GLM：hc_sinkhorn_iters=20、hc_eps=1e-6、单站点参数 393,243、每层两站点 786,486、45 层合计 35,391,870：[GLM] config 与张量头统计（GLM 数据流页 verify_structure 交叉验证）。已确认
- N3 GLM 实测：20 次迭代后列和偏差 1.1e-6、行和偏差 1.0e-2；hc_mult=1 且 fn=0 时退化为普通残差（误差 3.8e-6，量级由 ε 决定）：GLM 数据流页 p3 探针（随机初始化构造条件）。已确认，标注构造条件
- N4 探针 A4 无约束链行和 1.095e6（24 层 4×4 随机矩阵，seed=0）：[实测]。构造示例

## 冲突与缺口

- [mHC] Eq.(9) 的迭代序（列后行、末步行）与 GLM 实现（末步列）方向相反：如实并列写出，不裁决孰优——两者都只保证「有限次后近似双随机」
- HC 论文 Eq.(14) 的初始化（A_r=I, B=1, A_m=one-hot）与 GLM 的 _init_weights（fn~N(0,0.02), base=0, scale=1）不同：页面不展开 HC 初始化细节，只讲 GLM 的
- mHC 论文未给出 ε；GLM 的 hc_eps=1e-6 是实现细节，来源标 [GLM]
