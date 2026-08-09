# 量化基础 evidence：核心论断与证据

来源优先级：原始论文与标准 > 权威教材与同行评审综述 > 对应版本官方文档 > 固定版本官方源码。

## C 论断（核心机制）

### C1
- 论断：均匀仿射量化把实数 $x$ 映射到整数 $x_q$ 的公式为 $x_q = \mathrm{clip}(\mathrm{round}(x/s) + z,\, q_{\min},\, q_{\max})$，反量化 $\hat{x} = s\,(x_q - z)$。
- 来源定位：Gholami, Kim, Dong, Yao, Mahoney, Keutzer. "A Survey of Quantization Methods for Efficient Neural Network Inference." arXiv:2103.13630, 2021, §2.1；aiwiki.ai/wiki/quantization（同一公式的现代统一写法）
- 适用条件：$s>0$、$z$ 为整数、$q_{\max} > q_{\min}$
- 置信状态：已确认

### C2
- 论断：对称量化取 $z=0$、整数范围 $[-2^{b-1}+1,\, 2^{b-1}-1]$（如 INT8 取 $[-127,127]$、INT4 取 $[-7,7]$），反量化简化为 $\hat{x} = s\,x_q$；scale 取 $s = \max(|x|)/(2^{b-1}-1)$。
- 来源定位：Gholami et al. 2021 §2.2；NVIDIA Developer Blog "Model Quantization: Concepts, Methods, and Why It Matters"（AbsMax 方法）
- 适用条件：分布近似零中心；对称牺牲一端的一个等级（如 INT8 不用 -128）换算术简化
- 置信状态：已确认

### C3
- 论断：非对称量化用非零 zero-point $z$ 处理偏斜分布，scale 取 $s = (\beta-\alpha)/(q_{\max}-q_{\min})$（$[\alpha,\beta]$ 为观测范围），$z$ 选为使 $\alpha$ 映射到 $q_{\min}$ 的整数。
- 来源定位：Gholami et al. 2021 §2.2；aiwiki.ai/wiki/quantization
- 适用条件：$q_{\max} > q_{\min}$、分布非零中心
- 置信状态：已确认

### C4
- 论断：量化粒度由粗到细为 per-tensor（一组 $(s,z)$）→ per-channel / per-axis（每输出通道一组）→ per-group / per-block（每固定大小块一组）。粒度越细精度越好、元数据越多。
- 来源定位：Gholami et al. 2021 §2.3；aiwiki.ai/wiki/quantization（粒度对照表）
- 适用条件：低精度量化（INT4 及以下）通常需要更细粒度
- 置信状态：已确认

### C5
- 论断：量化误差三来源——离散级粗（grid coarseness，位宽低则等级少）、舍入（rounding，单值最大误差 $\le s/2$）、裁剪（clipping，超出 $[\alpha,\beta]$ 的值误差等于超出量）。
- 来源定位：Gholami et al. 2021 §2.4（量化噪声分析）；NVIDIA Developer Blog
- 适用条件：均匀量化
- 置信状态：已确认

### C6
- 论断：per-tensor 下单个离群值会让 $\max(|x|)$ 被该离群值主导、$s$ 被放大，正常值被舍到 0 或 1 附近；这是 per-channel / per-block 出现的直接动机。
- 来源定位：Gholami et al. 2021 §2.4（outlier discussion）；emergentmind.com/topics/mxfp-formats（"outliers no longer poison their channel"）
- 适用条件：per-tensor 粒度 + 张量内存在显著离群值
- 置信状态：已确认

### C7
- 论断：PTQ 在训练结束后用少量校准数据统计激活范围、确定 $(s,z)$ 后部署；无需重训、分钟到小时级；低位宽精度下降显著。
- 来源定位：Gholami et al. 2021 §2.5；NVIDIA Developer Blog；karam-nus.github.io/language-modelling/20_quantization_fundamentals
- 适用条件：训练与推理分离的常规部署
- 置信状态：已确认

### C8
- 论断：QAT 在训练前向插入伪量化（先把 $w$ 量化再反量化得 $\hat{w}$，用 $\hat{w}$ 算损失），反向用直通估计器（STE）令 $\partial\hat{w}/\partial w \equiv 1$ 让梯度绕过不可导的 round/clip 直传到 $w$。
- 来源定位：Jacob, Kligys, Chen, Zhu, Corrado, Le. "Quantization and training of neural networks for efficient integer-arithmetic-only inference." CVPR 2018；Gholami et al. 2021 §2.5（QAT 与 STE）
- 适用条件：可以重训的场景；STE 是最简且最常用的反向处理
- 置信状态：已确认

### C9
- 论断：MXFP4（OCP Microscaling v1.0）把每 32 个连续值组成一块，块共享一个 8-bit E8M0 power-of-two scale（$s_b = 2^{e_b}$），每个元素存 4-bit E2M1 浮点；反量化 $\hat{x}_i = s_b \cdot \mathrm{FP4}(q_i)$；元素有独立指数位（与定点整数不同），per-block scale 抑制离群值污染。
- 来源定位：Rouhani et al. "Microscaling Formats (MX) v1.0." OCP Specification, 2023；emergentmind.com/topics/mxfp4；zeroentropy.dev/concepts/mxfp4；AMD ROCm blog "High-Accuracy MXFP4, MXFP6, and Mixed-Precision Models"
- 适用条件：OCP MX v1.0 规范；块大小固定 32
- 置信状态：已确认

## F 公式（核心公式）

### F1（仿射量化与反量化）
- 公式：$x_q = \mathrm{clip}(\mathrm{round}(x/s) + z,\, q_{\min},\, q_{\max})$，$\hat{x} = s\,(x_q - z)$
- 来源：C1
- 推导链：直接由来源给出

### F2（对称量化特例）
- 公式：$z = 0$，$s = \max(|x|)/(2^{b-1}-1)$，$x_q = \mathrm{clip}(\mathrm{round}(x/s),\, -2^{b-1}+1,\, 2^{b-1}-1)$，$\hat{x} = s\,x_q$
- 来源：C2
- 推导链：F1 在 $z=0$ 下的特例

### F3（非对称 scale/zero-point）
- 公式：$s = (\beta-\alpha)/(q_{\max}-q_{\min})$，$z = \mathrm{round}(q_{\min} - \alpha/s)$
- 来源：C3
- 推导链：要求 $\alpha \mapsto q_{\min}$，即 $\mathrm{round}(\alpha/s) + z = q_{\min}$，解出 $z$

### F4（STE 约束）
- 公式：$\partial\hat{w}/\partial w \equiv 1$（伪量化 $\hat{w} = \mathrm{dequant}(\mathrm{quant}(w))$ 的反向处理）
- 来源：C8
- 推导链：量化中的 round 不可导，STE 直接令导数为 1 让梯度穿过；不保证最优，让训练能进行

## N 数字（外部数字）

### N1
- 数字：FP32 每参数 4 字节；FP16/BF16 每参数 2 字节；INT8 每参数 1 字节；INT4 每参数 0.5 字节。7B 模型在 FP16 下约 14 GB，INT8 下约 7 GB，INT4 下约 3.5 GB。
- 来源：Gholami et al. 2021 §1；NVIDIA Developer Blog（Llama2 7B 在 FP16 下约 14 GB 的例子）
- 实验条件：参数量 × 字节数 / 参数；INT4 以 4 bit/参数计算
- 置信状态：已确认

### N2
- 数字：MXFP4 块大小 32；元素 4 bit（E2M1：1 符号 + 2 指数 + 1 尾数）；scale 8 bit（E8M0）；每块总存储 $32 \times 4 + 8 = 136$ bit，有效位宽 $136/32 = 4.25$ bit。
- 来源：OCP Microscaling Formats v1.0；emergentmind.com/topics/mxfp4；zeroentropy.dev/concepts/mxfp4
- 实验条件：OCP 规范定义
- 置信状态：已确认

### N3
- 数字：典型 PTQ 校准数据量 128–512 样本。
- 来源：Gholami et al. 2021 §2.5；karam-nus.github.io/language-modelling/20_quantization_fundamentals
- 实验条件：PTQ 流程的常见工程值，非硬性规则
- 置信状态：已确认

### N4
- 数字：E2M1 浮点元素可表示的 16 个有限值为 $\{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\}$；最大绝对值 6。
- 来源：OCP Microscaling Formats v1.0；emergentmind.com/topics/mxfp4；AMD ROCm blog
- 实验条件：E2M1 编码（含 subnormal）
- 置信状态：已确认
