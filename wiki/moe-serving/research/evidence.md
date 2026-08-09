# MoE 大模型推理与服务基础 evidence：核心论断与证据

固定来源版本：

- [Vaswani2017] Vaswani et al., Attention Is All You Need, NeurIPS 2017, arXiv:1706.03762
- [Shazeer2017] Shazeer et al., Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, ICLR 2017, arXiv:1701.06538
- [GShard] Lepikhin et al., GShard, arXiv:2006.16668
- [Switch] Fedus et al., Switch Transformers, arXiv:2101.03961
- [DSV3] DeepSeek-AI, DeepSeek-V3 Technical Report, arXiv:2412.19437（v1 核对）
- [DSV3-Infra] DeepSeek 官方《DeepSeek-V3/R1 推理系统概览》，2025-03-01（知乎/github.com/deepseek-ai 发布；中文转载 baoyu.io 核对一致）
- [DeepEP] github.com/deepseek-ai/DeepEP（MoE dispatch/combine 通信库）
- [DistServe] Zhong et al., DistServe, OSDI 2024, arXiv:2401.09670（v3 核对）
- [Splitwise] Patel et al., Splitwise, ISCA 2024, arXiv:2311.18677
- [ExpertPlex] Wu et al., ExpertPlex, arXiv:2607.18002 v2（TeX 源码已核对 §2）

## C 论断

- C1：Transformer 层由一个 attention 模块和一个 FFN 模块组成；attention 跨 token 混合信息，FFN 逐 token 独立变换。
  来源：[Vaswani2017] §3.1（"each of which has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a ... fully connected feed-forward network"）；[ExpertPlex] §2.1（"Attention mixes information across tokens ... The FFN instead transforms each token independently"）。
  条件：标准 decoder-only/encoder Transformer 层结构。置信：已确认。

- C2：在 Vaswani 配置（d_model=512，d_ff=2048）下，每层 FFN 参数 2·d·d_ff = 8d²，attention（Q/K/V/O 四个投影）参数 4d²，FFN 约为 attention 的 2 倍。
  来源：[Vaswani2017] §3.1 与 Table 1 配置，FFN 为两个线性变换（§3.1 "two linear transformations"）；参数计数为直接推导。
  条件：忽略 bias 与 LayerNorm 参数；多头拆分不改变参数总量。置信：已确认（推导自来源配置）。

- C3：MoE 通过稀疏激活把模型总参数量与每 token 计算量解耦：容量可随专家数增长而每 token 计算量近似不变。
  来源：[Shazeer2017] 摘要与 §1（conditional computation；137B 参数、计算量与稠密基线相当）；[ExpertPlex] §2.1（"sparse activation increases model capacity without proportionally increasing computation per token"）。
  条件：top-k 中 k 固定。置信：已确认。

- C4：现代 MoE 层 = router + 多个 routed expert（可选加 shared expert）；router 对每个 token 打分并选 top-k，输出为选中专家输出的加权和；shared expert 处理所有 token。
  来源：[Shazeer2017] §2（y = Σ G(x)_i E_i(x)）；[DSV3] §2.1.2（256 routed + 1 shared，sigmoid 亲和度 + top-8，Eq.(14)–(16)）；[ExpertPlex] §2.1（"Shared experts process all tokens, while a router dynamically selects a small top-k subset of routed experts"）。
  条件：token-choice 路由（主流）。置信：已确认。

- C5：top-k 路由不保证专家负载均衡；训练需辅助损失（[Shazeer2017]）或 bias 调整（[DSV3] Eq.(16) 及正文"auxiliary-loss-free"）缓解；服务侧负载不均表现为各 GPU 收到的 token 数不同。
  来源：[Shazeer2017] §2 末与附录；[DSV3] §2.1.2；[ExpertPlex] §2.5（"the number of activated experts and the number of tokens on a GPU change across layers"）。
  条件：—。置信：已确认。

- C6：MoE 权重常超出单卡显存，服务系统用专家并行（EP）把专家分片到多张 GPU。
  来源：[ExpertPlex] §2.3（"MoE weights often exceed one GPU's memory, so serving systems use expert parallelism (EP) to shard experts across GPUs"）。
  条件：—。置信：已确认。

- C7：dispatch 把 token 激活发送到选中专家所在 rank，combine 把专家输出送回；二者通常用 all-to-all 通信实现。
  来源：[ExpertPlex] §2.3（"dispatch sends activations to the ranks hosting the selected experts and combine returns their outputs. Both operations are commonly implemented as all-to-all communication [DeepEP]"）；[GShard] §3（all-to-all 用于 MoE 分片）。
  条件：EP 部署。置信：已确认。

- C8：TBO（two-batch overlap）用一个微批的通信与另一个微批的计算重叠；SBO（single-batch overlap）在同一微批内用 shared expert 计算与 routed expert 的通信及计算重叠。
  来源：[ExpertPlex] §2.3（定义句）与 §4.x 延迟模型段（"Under TBO ... overlaps one microbatch's attention with another's communication and MoE computation ... Under SBO, overlaps shared-expert computation with routed-expert computation and communication within one microbatch"）；[DSV3-Infra]"双批次重叠"（prefill 两微批交替，一个微批通信与另一个计算重叠）；[DSV3] §deployment prefill 段（"processes two micro-batches with similar computational workloads ... overlapping the attention and MoE of one micro-batch with the dispatch and combine of another"）。
  条件：批内/批间存在可并行的计算。置信：已确认。

- C9：推理分两阶段：prefill 并行处理全部输入 token、构建 KV cache、产出首 token，吞吐导向；decode 每步基于前缀 KV cache 生成一个 token，延迟敏感。
  来源：[DistServe] §2.1；[ExpertPlex] §2.1（"Prefill processes all input tokens in parallel, builds their KV cache, and produces the first output token ... Decode then generates one token per iteration"）。
  条件：自回归 LLM。置信：已确认。

- C10：attention 在各层产出 key-value 张量并缓存为 KV cache，后续迭代复用，避免重算前缀。
  来源：[ExpertPlex] §2.1（"materializes key-value tensors as the KV cache, which later iterations reuse"）。
  条件：—。置信：已确认。

- C11：TTFT（time to first token）= prefill 阶段时长；TPOT（time per output token）= 每请求除首 token 外平均每 token 生成时间；总延迟 = TTFT + TPOT × decode 阶段生成 token 数。
  来源：[DistServe] §1 及脚注 1（"The overall request latency equals TTFT plus TPOT times the number of generated tokens in the decoding phase"）。
  条件：—。置信：已确认。

- C12：（每 GPU）goodput = 满足 SLO 达成率目标（如 90%）前提下可服务的最大请求率；SLO 对 TTFT 与 TPOT 分别设定；不同应用侧重不同（实时聊天重 TTFT；TPOT 快过人阅读速度约 250 词/分钟即可；文档摘要重 TPOT）。
  来源：[DistServe] §1（goodput 定义句与 250 words/min 例），§2.1 重申 goodput。
  条件：—。置信：已确认。

- C13：PDD 把 prefill 与 decode 放到不同 GPU 实例：消除跨阶段干扰、允许各自独立配置与扩缩容；代价是每个实例须持有完整模型副本、阶段间要传 KV cache、且须按 P:D 配比以部署单元为单位供给。
  来源：[DistServe] §1/§2.1（干扰与各自并行策略；通信开销可管理）；[Splitwise]（分池部署）；[ExpertPlex] §2.4（"Each prefill or decode instance must hold a complete model replica ... PDD must then provision these indivisible instances in a prefill-to-decode resource ratio"）。
  条件：—。置信：已确认。

- C14：PD 合设（colocation）让两阶段共享同一实例与一份权重，省显存、无跨实例 KV 传输，但 prefill 步与 decode 步互相拖延（干扰）。
  来源：[DistServe] §2.1（"colocation leads to strong prefill-decoding interference ... A prefill step often takes much longer than a decoding step"）；[ExpertPlex] §2.5（"Colocation avoids separate model replicas by sharing one instance across phases"）。
  条件：—。置信：已确认。

## F 公式

- F1：MoE 层输出 y(x) = Σ_{i∈S_k(x)} g_i(x)·E_i(x)，其中 S_k(x) 为 router 打分最高的 k 个 routed expert，g_i 为对应（归一化）门控权重；shared expert 的输出另行恒等加入（或并入求和，依实现）。
  来源：[Shazeer2017] §2（y = Σ_i G(x)_i E_i(x)，G 稀疏即 top-k）；[DSV3] Eq.(14)–(16)（u_t = Σ g_{i,t} FFN_i(u_t)，top-k 选择式）。
  条件：token-choice top-k；不同实现归一化细节不同（正文以"加权和"表述，不绑定某一归一化）。置信：已确认。

## N 数字

- N1：DeepSeek-V3 总参数 671B、每 token 激活约 37B。37/671 ≈ 5.5%。
  来源：[DSV3] 摘要（"a Mixture-of-Experts ... with 671B total parameters, of which 37B are activated for each token"）。置信：已确认。

- N2：DeepSeek-V3 每个 MoE 层含 1 个 shared expert 与 256 个 routed expert，每 token 激活 8 个 routed expert（top-8）。
  来源：[DSV3] §2.1.2；[DSV3-Infra]（"每层仅激活 256 个专家中的 8 个"）。置信：已确认。

- N3：DeepSeek-V3 共 61 层，前 3 层保留稠密 FFN，其余 58 层为 MoE 层。
  来源：[DSV3] §2.1.2（"except for the first three layers"）与官方 config（num_hidden_layers=61，first_k_dense_replace=3）。置信：已确认。

- N4：DeepSeek 在线推理系统采用 PDD：prefill 部署单元 4 节点 32 GPU（路由专家 EP32，每卡 9 路由 + 1 共享），decode 部署单元 18 节点 144 GPU（EP144，每卡 2 路由 + 1 共享），两阶段并行度不同；prefill 用双批次重叠隐藏通信。
  来源：[DSV3-Infra] 正文。置信：已确认。

- N5：DeepSeek-V3 技术报告版部署：prefill 最小部署单元 4 节点 32 GPU（EP32），decode 最小部署单元 40 节点 320 GPU（EP320），一个配比单元合计 32P + 320D GPU。
  来源：[DSV3] §deployment（Inference and Deployment：prefill "minimum deployment unit consists of 4 nodes with 32 GPUs ... EP32"；decode "minimum deployment unit consists of 40 nodes with 320 GPUs ... EP320"）；[ExpertPlex] §2.4（"One reported DeepSeek-V3 unit combines 32 prefill and 320 decode GPUs"）。置信：已确认。

- N6：DistServe 相比当时最优系统可服务最多 7.4× 请求或收紧 12.6× SLO（>90% 请求达标）。
  来源：[DistServe] 摘要。置信：已确认。

- N7：Vaswani 配置下 d=512：attention 4d² = 1,048,576 ≈ 105 万参数，FFN 8d² = 2,097,152 ≈ 210 万参数，FFN 约占两者之和的 2/3。
  来源：由 [Vaswani2017] Table 1 配置直接计算。置信：已确认。

## 教学构造（非来源数字）

- E1：8 个 routed expert、top-2 的路由手算（打分表人为构造）
- E2：2 GPU × 2 专家、4 个 token、top-2 的 dispatch/combine 计数（路由表人为构造）
- E3：prompt 4 token、生成 3 token 的 K/V 计算次数对比（忽略 attention 打分本身，只数 K/V 投影计算次数）
- E4：prefill 0.3 s + 5 步 × 0.05 s 的时间线（人为构造）；goodput 场景数字（人为构造）
- 可运行代码：模拟 8 专家 top-2 路由 + 2 卡 EP 的负载统计（纯 Python，无依赖）
