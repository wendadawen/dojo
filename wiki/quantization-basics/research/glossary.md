# 量化基础 glossary：术语表

登记全文所有首次出现的术语、缩写和符号。保证全文含义一致，防止同一对象出现多种记号或术语漂移。

## 术语

| 名称 | 首次出现 | 定义或含义 |
|---|---|---|
| 量化（quantization） | S1 | 把高精度浮点权重或激活映射到低精度整数或浮点网格上以减少显存与计算的方法 |
| 位宽（bit-width） | S1 | 每个数值占用的比特数；FP32=32、FP16/BF16=16、INT8=8、INT4=4 |
| 压缩比 | S1 | 量化前后每参数字节数之比；FP16→INT8 为 2×、FP16→INT4 为 4× |
| 均匀量化（uniform quantization） | S2 | 量化网格等间距的量化；本文主线 |
| 非均匀量化（non-uniform quantization） | S2 | 量化网格不等间距的量化（如对数量化）；本文不展开 |
| 仿射量化（affine quantization） | S2 | 用 scale 与 zero-point 做仿射变换的均匀量化，即非对称量化的通用形式 |
| 对称量化（symmetric quantization） | S2 | zero-point 取 0、整数范围关于 0 对称的均匀量化 |
| 非对称量化（asymmetric quantization） | S2 | zero-point 非 0、整数范围可关于 0 不对称的均匀量化 |
| PTQ（Post-Training Quantization，训练后量化） | S1 | 训练结束后用校准数据确定 $(s,z)$ 后部署的量化路径 |
| QAT（Quantization-Aware Training，量化感知训练） | S5 | 训练前向插入伪量化、反向用 STE 的量化路径 |
| 校准（calibration） | S5 | PTQ 中用少量数据统计激活范围以确定 $(s,z)$ 的过程 |
| 伪量化（fake quantization） | S5 | QAT 前向把 $w$ 量化再反量化得 $\hat{w}$、用 $\hat{w}$ 算损失的过程 |
| STE（Straight-Through Estimator，直通估计器） | S5 | 令 $\partial\hat{w}/\partial w \equiv 1$、让梯度绕过不可导的 round/clip 的反向处理 |
| 离群值（outlier） | S3 | 张量中显著大于其它值的元素；会拉大 per-tensor 的 scale |
| 量化粒度（granularity） | S4 | 一组 $(s,z)$ 覆盖多少元素；由粗到细为 per-tensor → per-channel → per-block |
| per-tensor | S4 | 整张张量一组 $(s,z)$ |
| per-channel / per-axis | S4 | 每输出通道一组 $(s,z)$ |
| per-block / per-group | S4 | 每固定大小块（如 32、128）一组 $(s,z)$ |
| 有效位宽（effective bits per element） | S4 | 含 scale 元数据后每元素的实际位宽；per-block(32) + INT4 + 8-bit scale ≈ 4.25 bit |
| 浮点量化（floating-point quantization） | S5 | 元素为低精度浮点（如 FP8、FP4）而非整数的量化 |
| 整数量化（integer quantization） | S5 | 元素为定点整数（如 INT8、INT4）的量化 |
| MXFP4 | S5 | OCP Microscaling v1.0 的 4-bit 微缩浮点格式；块大小 32、E8M0 共享 scale、E2M1 元素 |
| OCP Microscaling Formats | S5 | Open Compute Project 微缩浮点格式规范 v1.0（Rouhani et al. 2023） |
| E2M1 | S5 | 1 符号 + 2 指数 + 1 尾数的 4-bit 浮点元素格式；MXFP4 的元素类型 |
| E8M0 | S5 | 8-bit 无符号指数的 scale 格式；表示 $s_b = 2^{e_b}$；MXFP4 的块 scale 类型 |
| ReLU | S4 | 把负值置零的激活函数；其输出取值 $[0,\max]$，是激活常取非对称量化的典型场景 |

## 符号

| 符号 | 首次出现 | 含义 |
|---|---|---|
| $x$ | S2 | 待量化的实数（浮点权重或激活元素） |
| $x_q$ | S2 | 量化后的整数 |
| $\hat{x}$ | S2 | 反量化重建的实数 |
| $s$ | S2 | scale，正实数，量化网格的步长 |
| $z$ | S2 | zero-point，整数，使实数 0 精确对应某个整数 |
| $q_{\min}, q_{\max}$ | S2 | 整数网格的端点；INT8 对称取 $\{-127,127\}$，INT4 对称取 $\{-7,7\}$ |
| $b$ | S2 | 位宽；整数总等级数为 $2^b$，对称有符号范围 $[-2^{b-1}+1,\, 2^{b-1}-1]$ |
| $\alpha, \beta$ | S4 | 观测范围的端点；$\alpha = \min(x)$、$\beta = \max(x)$ |
| $w$ | S5 | 原始高精度权重 |
| $\hat{w}$ | S5 | 伪量化后的近似权重 |
| $s_b$ | S5 | MXFP4 的块 scale，$s_b = 2^{e_b}$、$e_b$ 为 E8M0 编码 |
| $e_b$ | S5 | E8M0 scale 的指数值（无符号 8-bit） |
| $\mathrm{FP4}(\cdot)$ | S5 | 把 4-bit 码字映射为 E2M1 浮点值的查表函数 |
| $\mathrm{round}(\cdot)$ | S2 | 四舍五入到最近整数（本文按"四舍五入"约定，0.5 取较大的整数） |
| $\mathrm{clip}(\cdot, a, b)$ | S2 | 把值截断到 $[a,b]$ |
