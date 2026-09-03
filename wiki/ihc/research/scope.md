# scope.md：iHC（identity Hyper-Connections）

## 1. 概念含义

- 概念名称：iHC，identity Hyper-Connections（恒等超连接）。vLLM 官方实现的 docstring 中写作 independent Hyper-Connections，指同一机制（无流间混合矩阵），正文中注一笔，不做裁决。
- 简要定义：把 Transformer 的单一残差流扩成 $n$ 条并行残差流；每个子层入口把 $n$ 条流加权合并成一份输入，子层照常计算一次，出口把这份输出按各自权重加回 $n$ 条流；流与流之间不做任何线性混合——混合矩阵固定为单位阵 $I$。
- 正式定义：HC 论文的单层传播 $\hat{\mathbf{H}}=\mathbf{B}^{\intercal}\mathcal{T}(\mathbf{H}^{\intercal}\mathbf{A_{m}})^{\intercal}+\mathbf{A_{r}}^{\intercal}\mathbf{H}$ 中，$\mathbf{A_r}$ 是流间混合矩阵。iHC 是 $\mathbf{A_r}=\mathbf{I}$ 的特化：$\hat{\mathbf{H}}=\mathbf{B}^{\intercal}\mathcal{T}(\mathbf{H}^{\intercal}\mathbf{A_{m}})^{\intercal}+\mathbf{H}$；读写权重 $\mathbf{A_m}$、$\mathbf{B}$ 保留为逐 token 动态计算的 sigmoid 门（Hy4 实现的证据见 evidence.md F1–F3）。
- 本文采用的语境：Hy4-preview（腾讯混元 2026-08-28 发布的 770B MoE 模型）残差通路的实际落地，$n=4$。
- 包括什么：
  - iHC 的三个计算块：pre（读门与合并）、post（写门与残差回加）、head（末端合并）——这是官方实现的全部机制
  - 读写门的计算路径（展平、RMS 归一化、线性投影、sigmoid 门）与初始化及其初始行为
  - 「为什么把混合矩阵钉死在 $I$」的论证：mHC 学出的混合矩阵接近单位阵、双随机矩阵连乘坍缩、流语义一致性、Sinkhorn 开销消失
  - Hy4 的配置、checkpoint 张量结构与参数量、与 GLM-5.3-Flash mHC 实现的对照
- 不包括什么：
  - HC 的一般机制、三映射推导、Pre-Norm/Post-Norm 特例（超连接页已完整讲解，本页引用结论）——避免重复
  - mHC 的双随机约束动机、三条性质、Sinkhorn-Knopp 算法（同上，本页只用其结论）
  - Gated DSA、IndexCache、MoE、MTP 的机制（Hy4 其他部件，链接已有页面，各一句话）
  - HPC 融合 kernel 的实现机制（工程扩展，折叠块一句话带过）
  - Hy4 的训练方法与评测结果（与 iHC 机制无关）
- 相邻概念：
  - 超连接 HC 与 mHC（有页面 hyper-connections）——本页的直接前置与对照对象，正文反复引用
  - 残差连接（有页面 residual-connection）——更底层前置
  - RMSNorm（有页面 rmsnorm）——门控计算前的归一化
  - 稀疏注意力 DSA（有页面 dsa）——Hy4 架构位置的邻接部件，一句话链接

## 2. 学习目标（核心问题）

### Q1：iHC 相对 mHC 拿掉了什么、留下了什么？

- 完成答案：mHC 在 HC 的三映射（读 $\mathbf{A_m}$、混合 $\mathbf{A_r}$、写 $\mathbf{B}$）基础上把 $\mathbf{A_r}$ 约束为双随机矩阵（Sinkhorn 投影实现）。iHC 把 $\mathbf{A_r}$ 直接固定为单位阵：流间零混合，旧状态原样传递；保留的是逐 token 动态的读门与写门（sigmoid 参数化）。所以 iHC = HC 减去流间混合、保留动态读写。
- 为什么是核心目标：这是本页的定义；不理解「拿掉的是混合、留下的是读写」就无法把它与 mHC、普通残差区分开。
- 依赖内容：HC 三映射与 mHC 约束的结论（链接 hyper-connections 页）。

### Q2：一个子块在 iHC 下怎么完整走一遍？

- 完成答案：模型入口把 embedding 复制 $n$ 份成 $n$ 条流；每个子块（每层两个：注意力、MLP/MoE）依次执行：pre 把 $n$ 条流按读门加权合并成一份输入并顺手算出写门 → RMSNorm → 子层计算一次 → post 把子层输出按写门加回每条流（每条流的旧值原样保留）。78 层走完后，head 把 $n$ 条流按门控权重合并回单一隐状态，再过 final RMSNorm 进 lm_head。残差不再有独立张量，内嵌在 $n$ 条流里。
- 为什么是核心目标：这是 iHC 的全部数据流，是后续公式与论证的载体。
- 依赖内容：RMSNorm 的作用（链接 rmsnorm 页）。

### Q3：读写门从哪来？初始时是什么行为？

- 完成答案：pre 块把 $n$ 条流展平成一个 $n\cdot d$ 维向量，先算整体 RMS，再经一个 $[2n, nd]$ 的线性投影得到 $2n$ 个 logits（读 $n$ 个、写 $n$ 个），乘上 RMS 归一化因子后过 sigmoid 加 $\varepsilon$ 得到读门；写门再乘幅度 $m$。初始化让读门 $\approx 1/n$（等权平均）、写门 $\approx m/2$（Hy4 取 $m=2$，即系数 1）：初始状态下 iHC 近似「子层输入 = 四流平均，输出 = 满幅写回」，行为贴着标准 Pre-Norm 残差，训练中门再分化。
- 为什么是核心目标：门是 iHC 唯一的可学习通路（混合已固定），它的计算与初始化决定了「iHC 从哪里出发、学什么」。
- 依赖内容：RMSNorm、sigmoid。

### Q4：为什么把混合矩阵钉死在 $I$ 效果反而更好？

- 完成答案：四条依据。① mHC 实测学出的单层混合矩阵本来就接近单位阵（对角约 0.96、非对角约 0.01），混合没学到多少；② 双随机正矩阵连乘会坍缩：Dobrushin 遍历系数几何衰减，深层把 $n$ 条流混成均匀矩阵，多流失去分化（本页构造实验复现）；③ 流语义一致性：identity 等于所有层用同一恒等置换，读门写门无需追踪「流被重排到哪」，且 $I^L=I$ 不坍缩；④ Sinkhorn 迭代的近似误差（实测行和标准差 0.12 量级）与专用 kernel 的工程负担一并消失。标注：这些是 Hy4 README 引用的社区研究结论（Qwen3 1.7B/8B、150B tokens 的 from-scratch 实验，Identity HC > mHC > mHC lite > mHC orthogonal），iHC 本身没有正式论文。
- 为什么是核心目标：这是 iHC 的动机核心——「少学一点反而更好」的反直觉结论，不论证它页面就只是机制描述。
- 依赖内容：mHC 双随机约束结论（链接 hyper-connections 页）、矩阵乘法。

### Q5：Hy4 里 iHC 的具体配置和代价是什么？

- 完成答案：4 条流、hidden 6144、78 层每层两个 HC 边界（注意力、MLP 各一，参数独立）、写门幅度 2.0、$\varepsilon=10^{-6}$。checkpoint 实测：每站点 hc_fn 是 $[8, 24576]$ 的 float32 矩阵加 scale/base，全局一个 head；iHC 总参数 30,770,717（约 30.77M，占 770B 的 0.004%）。代价主要是激活显存：残差状态从 1 份变 4 份。门控权重保持 float32、不参与 FP8 量化；门控计算在 float32 下进行。MTP 草稿层不用 iHC（checkpoint 无对应张量）。
- 为什么是核心目标：把概念落到真实模型的规格与代价，并给出可复算的参数量。
- 依赖内容：Q2、Q3 的机制。

## 3. 内容分级

核心内容：
- iHC 定义与家族位置（Q1）
- 入口复制、pre/RMSNorm/子层/post、head 的完整数据流（Q2）
- 门控计算公式与初始化的初始行为（Q3）
- identity 的四条依据与坍缩构造实验（Q4）
- Hy4 配置、张量结构、参数量、显存代价（Q5）

辅助内容：
- 初始行为 ≈ 标准 Pre-Norm 残差的推导（帮读者建立「小改动」直觉）
- 与 GLM-5.3-Flash mHC 实现的逐项对照表（澄清两家的异同）
- identity/independent 命名分歧说明（读源码时不困惑）
- 每子块只算一次 vs 激活 4 份的辨析（堵「计算量×4」误解）

扩展内容：
- HPC 融合 kernel（eager 20/5/15 个 kernel 融合为 1；sm100/103 约束）——折叠块，标注工程细节
- 知乎文章的实验设置细节（模型规模、token 数、排序结论）——折叠块，标注社区实验

## 4. 前置知识映射

| 前置概念 | 状态 | 用在哪 |
|---|---|---|
| 残差连接 | 已有页 residual-connection | 第 1 章开头，取「恒等映射保梯度」结论 |
| 超连接 HC 与 mHC | 已有页 hyper-connections | 第 1 章主体前置：三映射、双随机约束、Sinkhorn 结论；第 4 章论证复用 |
| RMSNorm | 已有页 rmsnorm | 第 3 章门控计算的归一化 |
| DSA 稀疏注意力 | 已有页 dsa | 第 5 章 Hy4 架构一句话链接 |
| 投机解码/MTP | 已有页 speculative-decoding | 第 5 章 MTP 一句话链接 |
| sigmoid | 无页面 | 页内一句话自包含（压缩到 (0,1) 的光滑函数） |

## 5. 明确不展开的内容

- Sinkhorn-Knopp 的收敛理论与 20 步迭代的误差分析细节：属于 hyper-connections 页范围，本页只引用「行和残差 1e-2 量级」结论
- HC 论文 Eq.(15)(16) 的 Pre-Norm/Post-Norm 特例推导：已有页覆盖
- HPC kernel 的 CUDA 实现机制（tile 划分、post+pre 跨层融合的 TODO）：只影响工程性能，不影响机制理解
- Hy4 的后训练、评测分数、产品协同：与 iHC 无关
- 为什么选 4 条流而不是 2 或 8：官方材料未给出消融，如实标注「未见公开依据」

## 6. 常见误解和适用边界

误解：
1. 「四条流 = 计算量翻 4 倍」——错：每个子层只对合并后的单份输入计算一次；变 4 倍的是残差状态的激活显存，不是子层 FLOPs。
2. 「四条流各过各的注意力/MLP」——错：复制只发生在模型入口；每个子块把 4 条流合并成一份输入、过同一个子层、再写回。流是状态通道，不是并行计算。
3. 「流间不混合 = 四条流互不相干」——错：读门把 4 条流加权合并（信息在子层计算里汇合），写门让每条流都吸收同一份子层输出。不混合的只是「旧状态如何传递」。
4. 「identity 指的就是残差连接那个 identity」——不精确：$I$ 指流间混合矩阵（mHC 记号 $\mathcal{H}^{res}$）取单位阵；残差通路的恒等性另由 post 里「+旧流」这一项保证。
5. 「iHC 被论文证明优于 mHC」——错：iHC 没有正式论文；「Identity > mHC」是 Hy4 README 引用的社区实验结论（Qwen3 1.7B/8B、150B tokens），引用时必须带实验条件。

边界：
- 机制事实以 Hy4 官方实现（vLLM PR #54160，腾讯工程师提交）与 checkpoint 张量为准；「为什么 identity」是社区研究论证 + 实现注释，不是官方论文结论
- 门控的 sigmoid/magnitude/初始化是 Hy4 实现的取值；换一个模型可以参数化不同（GLM 的 post 就没有 $\varepsilon$）
- 参数量 30,770,717 由 checkpoint 张量头逐项算出，对应 BF16 主检查点；FP8 变体中 hc_* 仍是 float32
- 坍缩论证对「Sinkhorn 输出的严格正矩阵」成立；纯置换矩阵序列不满足一致正性条件、不坍缩（边界条件如实写）

## 7. 概念歧义处理

- iHC 在 LLM 残差结构语境下指 identity Hyper-Connections（Hy4 官方 README 明确），无其他主流含义。医学免疫组化 IHC 等同名缩写与本页无关，不并列。
- identity（README/model card）vs independent（vLLM 实现 docstring）两种展开：已裁定，正文采用官方 README 的 identity（与「$\mathbf{A_r}=\mathbf{I}$」的数学含义直接对应），在定义处注一笔 vLLM 注释的写法，不裁决哪个更「正确」。
