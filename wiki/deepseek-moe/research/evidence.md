# DeepSeekMoE 核心论断与证据

来源：Dai et al. 2024, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", ACL 2024, arXiv:2401.06066。所有定位以 arXiv HTML 版本 https://arxiv.org/html/2401.06066v1/ 为准。

## C 论断（机制与定义）

- C1（传统 top-K MoE 的两个问题）：现有 MoE（如 GShard）用有限数量专家（如 8 或 16），存在知识混合（knowledge hybridity）与知识冗余（knowledge redundancy），二者共同妨碍专家专门化。来源：§1 引言；摘要。"each expert acquires non-overlapping and focused knowledge" 是专家专门化的定义。适用条件：top-K MoE 框架内。置信状态：已确认。

- C2（知识混合的成因）：专家数量有限 → 分配给某专家的 token 涵盖多种知识 → 专家被迫在参数中组装截然不同且难以同时利用的知识。来源：§1。适用条件：专家数较少时。置信状态：已确认。

- C3（知识冗余的成因）：不同专家收到的 token 可能需要共有知识 → 多个专家各自独立学习相同共有知识 → 参数冗余。来源：§1。适用条件：路由专家间无共享机制时。置信状态：已确认。

- C4（细粒度专家分割的机制）：在保持专家参数总量与计算成本不变的前提下，将每个专家 FFN 的中间隐藏维度缩减为 1/m，从而把 N 个专家分割为 mN 个更小的专家，同时把每 token 激活专家数从 K 增加到 mK。来源：§3.1；Figure 2(b)。适用条件：m 为正整数，原 FFN 中间维度能被 m 整除。置信状态：已确认。

- C5（细粒度分割的组合灵活性）：分割后每 token 可选的专家组合数从 C(N,K) 增加到 C(mN,mK)。论文示例：N=16, K=2, m=4 → C(16,2)=120 种变为 C(64,8)=4,426,165,368 种。来源：§3.1 正文。适用条件：路由无约束时。置信状态：已确认。

- C6（共享专家隔离的机制）：隔离 K_s 个专家作为共享专家，对所有 token 恒激活、不参与 router 打分与 top-k 选择，承载通用知识；为维持计算量守恒，路由专家总数从 mN 降为 mN-K_s、每 token 激活的路由专家数从 mK 降为 mK-K_s。来源：§3.2；Figure 2(c)。适用条件：K_s < mK。置信状态：已确认。

- C7（三架构计算量恒定）：标准 top-K、细粒度分割、完整 DeepSeekMoE（细粒度 + 共享隔离）三种架构的专家参数总量与每 token 计算成本保持恒定。来源：Figure 2 caption（"across these three architectures, the number of expert parameters and computational costs remain constant"）；§3.1（"while maintaining a consistent number of expert parameters and computational cost"）；§3.2（"In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by K_s"）。适用条件：按论文方式同时调整中间维度、专家数、激活数、共享专家数。置信状态：已确认。

- C8（DeepSeekMoE 与 DeepSeek-V3 的关系）：DeepSeek-V3 继承了 DeepSeekMoE 的 shared+routed 专家组织，但路由打分改为 sigmoid 亲和度 + 偏置、负载均衡改为 aux-loss-free；本页讲 DeepSeekMoE 原始论文（softmax 亲和度）的两策略。来源：DeepSeekMoE §2–§3（softmax 亲和度 Eq.5/8/11）；DeepSeek-V3 Technical Report arXiv:2412.19437 §2.1.2（sigmoid + 偏置）。适用条件：区分原始论文与后续工程化。置信状态：已确认。

## F 公式

- F1（通用 top-K MoE 层，Eq.3）：h_t^l = Σ_{i=1}^{N} (g_{i,t} FFN_i(u_t^l)) + u_t^l。来源：§2, Eq.(3)。g_{i,t} 稀疏，仅 K 个非零。置信状态：已确认。

- F2（通用门控，Eq.4–5）：g_{i,t} = s_{i,t} 若 s_{i,t} ∈ TopK({s_{j,t} | 1≤j≤N}, K) 否则 0；s_{i,t} = Softmax_i((u_t^l)^T e_i^l)。来源：§2, Eq.(4)(5)。置信状态：已确认。

- F3（细粒度 MoE 层，Eq.6）：h_t^l = Σ_{i=1}^{mN} (g_{i,t} FFN_i(u_t^l)) + u_t^l，其中每个 FFN_i 中间维度为原来的 1/m。来源：§3.1, Eq.(6)。置信状态：已确认。

- F4（细粒度门控，Eq.7–8）：g_{i,t} = s_{i,t} 若 s_{i,t} ∈ TopK({s_{j,t} | 1≤j≤mN}, mK) 否则 0；s_{i,t} = Softmax_i((u_t^l)^T e_i^l)。来源：§3.1, Eq.(7)(8)。置信状态：已确认。

- F5（完整 DeepSeekMoE 层，Eq.9）：h_t^l = Σ_{i=1}^{K_s} FFN_i(u_t^l) + Σ_{i=K_s+1}^{mN} (g_{i,t} FFN_i(u_t^l)) + u_t^l。第一项共享专家恒激活（无 g），第二项路由专家稀疏激活。来源：§3.2, Eq.(9)。置信状态：已确认。

- F6（完整 DeepSeekMoE 门控，Eq.10–11）：g_{i,t} = s_{i,t} 若 s_{i,t} ∈ TopK({s_{j,t} | K_s+1≤j≤mN}, mK-K_s) 否则 0；s_{i,t} = Softmax_i((u_t^l)^T e_i^l)。来源：§3.2, Eq.(10)(11)。适用条件：路由只在路由专家范围内选，共选 mK-K_s 个；加上 K_s 个恒激活共享专家，总激活数为 (mK-K_s)+K_s = mK，与细粒度分割一致。置信状态：已确认。

- F7（计算量守恒推导，由 F3/F5 直接推出）：标准 top-K 每 token 计算量 = K·(单位 FFN)；细粒度 = mK·(1/m·单位 FFN) = K·(单位 FFN)；完整 DeepSeekMoE = K_s·(1/m·单位 FFN) + (mK-K_s)·(1/m·单位 FFN) = mK·(1/m·单位 FFN) = K·(单位 FFN)。三者相等。置信状态：已确认（由论文 C7 与 F3/F5 直接推出）。

## N 数字（外部实验数字）

- N1（2B vs GShard 2.9B）：DeepSeekMoE 2B 与 GShard 2.9B（GShard×1.5，专家参数 1.5×、计算 1.5×）性能相当。来源：摘要；§4.3。具体：DeepSeekMoE 2B 总参数 2.0B、激活 0.3B；GShard 2.9B 总参数 2.9B、激活 0.35B；Pile Loss 均为 1.808。置信状态：已确认。

- N2（2B 接近 dense 上界）：DeepSeekMoE 2B 接近 Dense×16（同总参数的稠密模型，是 MoE 模型的严格上界）的性能。来源：§4.3。置信状态：已确认。

- N3（2B 实验配置）：9 层 Transformer，隐藏维度 1280；1 共享专家 + 63 路由专家，每个专家 = 0.25× 标准 FFN（即 m=4），激活 1+7=8（即 K_s=1, mK=8, mK-K_s=7）；总参数 ≈ 2B，激活参数 ≈ 0.3B。来源：§4.1.3。置信状态：已确认。

- N4（16B vs LLaMA2 7B）：DeepSeekMoE 16B 与 LLaMA2 7B 性能相当，仅需约 39.6% 的计算（74.4T vs 187.9T FLOPs/4K tokens）。来源：摘要；§5.2.2；Table 4。置信状态：已确认。

- N5（16B 配置）：28 层 Transformer，隐藏维度 2048；2 共享专家 + 64 路由专家，每个专家 = 0.25× 标准 FFN（m=4），激活 2+6=8；总参数 ≈ 16.4B，激活参数 ≈ 2.8B；训练 2T tokens。来源：§5.1.2。置信状态：已确认。

- N6（16B 部署优势）：16B 模型可在 40GB 显存的单 GPU 上部署，推理速度约为 7B 稠密模型的 2.5 倍。来源：§5.2.1。置信状态：已确认。

- N7（145B vs DeepSeek 67B）：DeepSeekMoE 145B 与 DeepSeek 67B 性能相当，仅需 28.5%（甚至可能 18.2%）的计算。来源：摘要；§7。置信状态：已确认（论文为 preliminary effort）。

- N8（组合数对比）：N=16, K=2 → C(16,2)=120；m=4 → mN=64, mK=8 → C(64,8)=4,426,165,368。来源：§3.1 正文。置信状态：已确认。
