# glossary.md：iHC

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| iHC | 页面开头 | identity Hyper-Connections，恒等超连接：多残差流 + 流间混合矩阵固定为单位阵的超连接变体。vLLM 实现注释亦写作 independent Hyper-Connections |
| 超连接（HC, Hyper-Connections） | 第 1 章 | 把单一残差流扩成 $n$ 条、每子块读入-混合-写回的残差结构（ByteDance 论文 arXiv:2409.19606），本页只取结论，详见 [超连接与 mHC](../../wiki/hyper-connections/index.html) |
| mHC | 第 1 章 | Manifold-Constrained Hyper-Connections：把 HC 的混合矩阵约束为双随机矩阵的变体（arXiv:2512.24880），GLM-5.3-Flash 采用 |
| 残差流（residual stream） | 第 1 章 | 贯穿所有层、逐层被读写的隐状态通道；标准 Transformer 每个_token 一条，iHC 为 4 条 |
| $\mathbf{A_m}$ / 读门 $H_{pre}$ | 第 1 章 | 子块入口把 $n$ 条流加权合并的权重。论文记号 $\mathbf{A_m}$；Hy4 实现里是逐 token 的 sigmoid 门，记 $H_{pre}$ |
| $\mathbf{A_r}$ / 混合矩阵 | 第 1 章 | 流间线性混合的 $n\times n$ 矩阵（论文记号 $\mathbf{A_r}$，mHC 论文记 $\mathcal{H}^{res}$，GLM 实现称 comb）。iHC 中固定为 $\mathbf{I}$ |
| $\mathbf{B}$ / 写门 $H_{post}$ | 第 1 章 | 子块出口把输出分发回流上的权重。论文记号 $\mathbf{B}$；Hy4 实现里是逐 token 的门，记 $H_{post}$，带幅度 $m$ |
| 站点 | 第 2 章 | 一个子块（注意力或 MLP/MoE）自带的 pre+post 边界，Hy4 每层两个，参数独立 |
| pre 块 | 第 2 章 | HYV4HCPreLayer：展平 → RMS → 线性投影 → sigmoid 读门合并，同时产出写门 |
| post 块 | 第 2 章 | HYV4HCPostLayer：$\hat{x}_i = H_{post,i}\cdot z + x_i$，无参数 |
| head（合并层） | 第 2 章 | HYV4HCHeadLayer：主干末端把 $n$ 条流门控合并回单一隐状态，位于 final RMSNorm 之前 |
| hc_mult | 第 5 章 | 流数 $n$，Hy4 取 4 |
| 幅度 $m$ / hc_magnitude | 第 3 章 | 写门的乘子，Hy4 取 2.0；写门值域约 $(\varepsilon, m+\varepsilon)$ |
| $\varepsilon_{hc}$ / hc_eps | 第 3 章 | 加在两个门上的小量，Hy4 取 $10^{-6}$，保证门严格为正 |
| hc_fn | 第 3 章 | 门控线性投影权重 $W\in\mathbb{R}^{2n\times nd}$，float32；head 对应 hc_head_fn $[n, nd]$ |
| hc_scale / hc_base | 第 3 章 | 门的缩放向量与偏置向量，可学习，初始化 $0.01$ / $-\ln(n-1)$ 与 $0$ |
| RMSNorm | 第 2 章 | 均方根归一化，见 [RMSNorm](../../wiki/rmsnorm/index.html) |
| 双随机矩阵 | 第 4 章 | 非负、行和列和均为 1 的方阵；mHC 的约束目标，详见超连接页 |
| Sinkhorn-Knopp | 第 4 章 | 交替行列归一化把正矩阵投影到双随机流形的迭代法，详见超连接页 |
| Dobrushin 遍历系数 | 第 4 章（折叠） | 度量随机矩阵行分布距离的系数，具次乘性；用于论证双随机连乘坍缩 |
| 均匀矩阵 $\tfrac{1}{n}\mathbf{11}^{\intercal}$ | 第 4 章 | 全部元素为 $1/n$ 的矩阵，双随机连乘的坍缩终点 |
| 流语义一致性 | 第 4 章 | identity 下流 $i$ 在所有层都在位置 $i$，读写门无需追踪流的重排 |
| Gated DSA | 第 5 章 | Hy4 的稀疏注意力（带门控的 DeepSeek Sparse Attention + IndexCache），见 [DSA](../../wiki/dsa/index.html)，本页只作架构定位 |
| MTP | 第 5 章 | 多 token 预测层，用于投机解码；Hy4 的 MTP 草稿层不用 iHC |
| HPC kernel | 第 5 章（折叠） | vLLM 的单 kernel 融合实现，需 VLLM_ENABLE_HPC_OPS=1 与 sm100/103 |

写作约束：读写门全页统一用 $H_{pre}$/$H_{post}$（不混用 $\alpha$/$\beta$ 或 A/B 记号）；论文记号 $\mathbf{A_m}$/$\mathbf{A_r}$/$\mathbf{B}$ 只在第 1 章做映射时出现一次；「混合矩阵」「流间混合」统一叫混合矩阵；「站点」一词全页同义使用。
