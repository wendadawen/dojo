# 注意力汇聚点 核心论断与证据

来源代号：
- **S** = StreamingLLM（Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, arXiv:2309.17453）§3 与 Table 1/2/3
- **KRN** = `wiki/deepseek-v4-1/research/official/inference/kernel.py` 行号（sparse_attn_kernel）
- **SRC** = `wiki/deepseek-v4-1/research/official/inference/model.py` 行号
- **HDR** = `wiki/deepseek-v4-1/research/ckpt/headers.json`（HTTP Range 读到的分片头）
- **MEAS** = 本页 `research/verify_sink.py` 与 `verify_sink.out`

## 核心论断（C）

### C1：现象——初始位置持续获得大量注意力
- 论断：除最底部两层外，模型在各层各头持续把大量注意力分给初始 token；把最初 4 个 token 替换成换行符后模型仍显著依赖它们，说明起作用的是绝对位置而非语义。
- 来源定位：S §3.1 与 Figure 2、Table 1
- 引文依据：「We find that, beyond the bottom two layers, the model consistently focuses on the initial tokens across all layers and heads.」；「the first four tokens are substituted with the linebreak token '\n'. The observations indicate that the model still significantly emphasizes these initial linebreak tokens.」
- 置信状态：**已确认**

### C2：成因——softmax 归一化需要一个"倾倒去处"
- 论断：softmax 不允许被关注位置全为零权重，模型必须把注意力分配完；初始位置对所有后续位置可见，最容易被训练成汇聚点。移除它们的 KV 会移走分母的一大块，导致注意力分布显著偏移、模型崩溃。
- 来源定位：S §3.1
- 引文依据：「The nature of the SoftMax function prevents all attended tokens from having zero values. This requires aggregating some information from other tokens across all heads in all layers, even if the current embedding has sufficient self-contained information for its prediction. Consequently, the model tends to dump unnecessary attention values to specific tokens.」；「removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function (Equation 1) in attention computation.」
- 置信状态：**已确认**

### C3：形态一——保留若干初始位置的 KV
- 论断：StreamingLLM 的做法是在滚动窗口之外额外保留 4 个初始位置的 KV，两部分拼成缓存；位置按"缓存内位置"编号。论文实测保留 4 个初始位置即可恢复稳定困惑度。
- 来源定位：S §3.2 与 Table 1/2
- 引文依据：「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV to anchor the attention computation and stabilize the model's performance.」；「Introducing four initial tokens generally suffices; further additions have diminishing returns.」
- 置信状态：**已确认**

### C4：形态二——可学习 sink 参数（DeepSeek-V4.1-Flash 的实现）
- 论断：每个注意力头有一个可学习的 fp32 标量 sink，只加在 softmax 分母上、不参与行最大值；真实 checkpoint 中该参数出现在全部 40 个主干层与 3 个 MTP 层，形状 $[64]$（每头一个）。
- 来源定位：SRC L639（`self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))`）；KRN L310-403（分母构造）；HDR（43 个 `attn_sink` 张量，形状集合 $\{(64,)\}$，dtype F32）
- 引文依据：KRN 的稀疏注意力在算出各行最大值 $m$ 后，分母为选中槽位的 $\sum e^{s-m}$ 再加上汇聚点项；SRC 注释把 `attn_sink` 与稀疏注意力一起传入。
- 置信状态：**已确认**（源码 + checkpoint 头 + MEAS 解析解）

### C5：参数化 sink 的实测语义
- 论断：给定 $q=0$、两个等权可见槽位，输出为 $(v_1+v_2)/(2+e^{\text{sink}})$——汇聚点只改变分母、不改变分子；$q$ 的行最大值 $m$ 不含汇聚点项；整行槽位都无效时输出全零（汇聚点项被行最大值推成无穷大，分子为 0）。
- 来源定位：KRN L310-403 的语义；MEAS `verify_sink.out` 第 1、2 节（与解析解逐元素差 0.00e+00）
- 置信状态：**已确认**

### C6：sink 份额可占主导
- 论断：汇聚点在分母中的份额为 $e^{\text{sink}}/(W+e^{\text{sink}})$（$W$ 个等权槽位）；$W=128$ 时 logit 为 0、2、5、8 分别对应 0.78%、5.46%、53.69%、95.88%。
- 来源定位：MEAS `verify_sink.out` 第 3 节（按分母构成直接计算）
- 适用条件：所有可见槽位分数相同的理想情形，用于说明量级；实际分数分布下份额随分数变化。
- 置信状态：**已确认**（构造示例）

### C7：sink 不扩展上下文能力
- 论断：保留汇聚点只解决流式场景下的稳定性，不扩展模型上下文窗口、不增强长程记忆。
- 来源定位：S 附录 A
- 引文依据：「While StreamingLLM improves the efficiency of LLMs in streaming contexts, it does not extend the models' context window or enhance their long-term memory capabilities.」
- 置信状态：**已确认**

## 核心公式（F）

### F1：含汇聚点的注意力分母
$$o = \frac{\sum_{t\in\mathcal{T}} e^{\,q\cdot k_t/\sqrt{d}}\,v_t}{\sum_{t\in\mathcal{T}} e^{\,q\cdot k_t/\sqrt{d}} + e^{\,\text{sink}-m}},\qquad m=\max_{t\in\mathcal{T}}\frac{q\cdot k_t}{\sqrt{d}}$$
- 符号：$\mathcal{T}$ 为可见槽位集合；$m$ 只由可见槽位的分数取得，不含 sink；sink 为每头一个可学习标量。
- 来源定位：KRN L310-403；MEAS 解析解对照。
- 置信状态：**已确认**

### F2：sink 在分母中的份额
$$\text{share} = \frac{e^{\text{sink}}}{W + e^{\text{sink}}}$$
- 适用条件：$W$ 个可见槽位分数相同（构造示例）。
- 来源定位：由 F1 直接推出；MEAS 复算。

## 外部数字与实验条件（N）

| 编号 | 数值 | 来源 | 条件 |
|---|---|---|---|
| N1 | 纯窗口（0+1024）困惑度 5158.07；保留 4 个初始位置（4+1020）后 5.40 | S Table 1 | Llama-2-13B、PG19 首本书 65K token；x+y = x 个初始位置 + y 个近期位置 |
| N2 | 4 个初始位置一般足够，继续增加收益递减 | S Table 2 | Falcon-7B / MPT-7B / Pythia-12B / Llama-2-7B，PG19 拼接 400K |
| N3 | 可学习 sink 训练的模型在 1+1023 下困惑度 18.01；vanilla 模型在 2+1022 下 18.05 | S Table 3 | 160M 参数模型、PG19 首样本 |
| N4 | 43 个 `attn_sink` 张量，形状 $(64,)$，dtype F32（40 主干层 + 3 MTP 层） | HDR（MEAS 第 4 节） | 官方 checkpoint 分片头 |
| N5 | 128 个等权槽位下 sink 份额：0.78% / 5.46% / 53.69% / 95.88%（对应 logit 0/2/5/8） | MEAS | 构造示例 |

## 无法独立核对、需降级处理的条目

- sink 参数的具体训练取值：checkpoint 头只给出形状与类型，未读权重数值；本页不声称任何具体取值。
- massive activations 与 sink 的因果关系：未找到可定位的一手来源，不写。

## 构造示例与辅助解释

- **构造示例**：$q=0$ 两槽位的解析解；$W=128$ 的份额表；换行符实验属论文实验，按原值登记。
- **辅助解释**：把 sink 比作"注意力预算的零钱罐"——预算必须花完，零钱需要一个去处；边界：这个类比不说明 sink 是否携带信息，实际上它不提供长程记忆（C7）。

## 简化条件

- 份额表假设可见槽位分数相同，实际分布下 sink 份额随分数变化。
- 解析解用单头、4 维的构造张量，只验证分母构成，不反映真实头维与量化误差。
- 本页不评估可学习 sink 对模型质量的净影响（需训练实验）。
