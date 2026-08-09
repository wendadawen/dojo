# LatentMoE 核心论断与证据

来源优先级：原始论文 > 权威综述 > 官方文档 > 固定版本源码。
- 主源 1：Elango et al., 2026, "LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts", arXiv:2601.18089（NVIDIA）。引用为 [L]。
- 主源 2：Kimi K3 技术报告 §2.3（Stable LatentMoE），含 Eq. 11 与官方 config.json。引用为 [K3]。
- 辅源：Sebastian Raschka, "Latent MoE"（LLM Architecture Gallery，对照 Nemotron-3 与 K3 配置）。引用为 [R]。
- 前置页：`wiki/moe-serving/`（标准 MoE 与服务部署）、`wiki/deepseek-moe/`（shared+routed 组织）。

## C 论断（核心论断）

### C1（LatentMoE 正式定义）
- 论断：LatentMoE 在路由分支两端各加一次共享线性投影——下投影 $W_\downarrow \in \mathbb{R}^{\ell \times d}$ 把 token 从模型隐层 $d$ 压到隐空间 $\ell$（$\ell < d$），路由专家在 $\ell$ 维空间内计算，上投影 $W_\uparrow \in \mathbb{R}^{d \times \ell}$ 把聚合结果投回 $d$。共享专家与 router 仍在原始 $d$ 维工作。
- 来源定位：[L] Figure 1(b) 与 §2（"wrap the routed path with two shared linear layers: Project tokens down to a smaller latent representation (d → ℓ) before dispatch; Perform expert dispatch/combine and expert compute in that latent space; Project expert outputs back up to the model's hidden representation (ℓ → d)"）；"the router still computes gating decisions from the model's hidden representation—only the routed payload and routed expert computation move into the latent space. Shared experts, if present, operate in the original hidden dimension."
- 适用条件：$\ell < d$；$\ell$ 不低于任务特征秩下限（见 C5）。
- 置信状态：已确认。

### C2（标准 MoE 的全宽路由开销）
- 论断：标准 MoE 中每个被选中的路由专家都接收完整的 $d$ 维 token 表示，专家 FFN 权重也在 $d$ 维空间。因此扩大路由倍数（更多 top-k、更多专家）时，dispatch 通信量与专家权重读取量都随路由倍数线性增长。
- 来源定位：[L] §1 与 §2（"In real deployments, MoE cost is frequently dominated by memory movement & communication: Interactive serving: streaming expert weights from HBM dominates latency when few tokens hit each expert. Throughput serving: all-to-all routing dominates when token vectors must move across GPUs."）；[L] 设计原则 1 与 2（"Low-latency inference is memory-bound"、"High-throughput inference is communication-bound. Routing volume scales with tokens × top-k × routed width"）；前置页 `wiki/moe-serving/` §3（dispatch/combine 与 all-to-all）。
- 适用条件：分布式部署、专家分片到多卡。
- 置信状态：已确认。

### C3（LatentMoE 缓解开销的比例）
- 论断：把路由宽度从 $d$ 压到 $\ell$ 后，路由 dispatch 通信量与路由专家权重大小都大致按 $d/\ell$ 比例缩小。
- 来源定位：[L] Figure 1 caption（"reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ"）；[L] §2（"By shrinking what must be moved across GPUs and what expert weights must be read per token, LatentMoE reduces both memory traffic and all-to-all routing volume"）。
- 适用条件：只对路由部分有效；router 与共享专家仍在 $d$ 维，不受此比例影响。
- 置信状态：已确认。

### C4（Reinvestment 策略）
- 论断：LatentMoE 把节省的预算通常同时投入到更大的专家数 $N$ 与更高的 top-k（按同一因子 $d/\ell$ 放大），以提升专家组合多样性 $\binom{N}{k}$，从而在保持服务成本大致不变的前提下提升准确率。论文同时强调：保持 top-k × 专家中间维度（非线性容量）不变是关键。
- 来源定位：[L] Figure 1 caption（"We use this efficiency to increase the total number of experts and the top-k active experts per token by the same factor d/ℓ, which improves the accuracy of the model while keeping overall inference cost approximately constant"）；[L] 设计原则 3（"Preserve nonlinear capacity. Model quality tracks the effective nonlinear budget per token: top-k × expert intermediate dimension. Consequently, ... we should keep both top-k and expert intermediate dimension unchanged"）与原则 5（"Exploit combinatorial sparsity. Increasing both the number of experts and top-k expands the space of expert combinations dramatically"）。
- 适用条件：$\ell$ 不低于特征秩下限；同时放大 $N$ 与 $k$ 而非只放大其一。
- 置信状态：已确认。

### C5（压缩下限与投影成本）
- 论断：存在任务相关的"特征秩"下限，把 $\ell$ 压到低于此下限会损害模型质量；同时 down/up-projection 本身有计算成本，因此压缩比 $d/\ell$ 不等于整体加速比。论文实验显示压缩比到 4 时质量仍可保持（再压缩需配合 reinvestment）。
- 来源定位：[L] 设计原则 4（"Don't over-compress features. There is a task-dependent 'feature rank' that imposes a lower limit on the reduction of hidden dimension d to latent dimension ℓ. Reducing below this limit degrades model quality"）；[R]（"The down- and up-projections add work, so the 4x bottleneck does not imply a 4x speedup for an MoE layer or for the complete model"）；[L] §3 实验在 16B 参数规模上发现压缩比到 4 时质量保持。
- 适用条件：任务相关；需实验确定下限。
- 置信状态：已确认。

### C6（LatentMoE 与 Stable LatentMoE 的关系）
- 论断：Stable LatentMoE（K3）= LatentMoE 主结构 + 三件稳定化（Normalized LatentMoE 在 $W_\uparrow$ 前加 RMSNorm、SiTU-GLU 有界激活、Quantile Balancing 负载均衡），用于在 N=896, k=16 的极端稀疏与 2.8T 规模下保持训练稳定。
- 来源定位：[K3] §2.3（"Stable LatentMoE"小节，含 Eq. 11 与三件稳定化的引入动机）；`wiki/stable-latent-moe/index.html` §"三件稳定化"。
- 适用条件：K3 的具体配置（N=896, k=16, d=7168, ℓ=3584）；其他配置不一定需要全部三件稳定化。
- 置信状态：已确认。

### C7（router 仍在全宽）
- 论断：LatentMoE 的 router 仍从模型隐层 $d$ 计算门控决策，只有路由 payload 与路由专家计算进入隐空间 $\ell$。
- 来源定位：[L] §2（"the router still computes gating decisions from the model's hidden representation—only the routed payload and routed expert computation move into the latent space"）。
- 适用条件：通用 LatentMoE 设计。
- 置信状态：已确认。

## F 公式（核心公式）

### F1（标准 MoE 层输出，引用前置页）
- 公式：$y(x) = \sum_{i \in S_k(x)} g_i(x) \cdot E_i(x)$
- 来源：`wiki/moe-serving/` 第 2 章 F1；[L] §1 标准 MoE 定义。
- 状态：引用前置页已确认公式。

### F2（LatentMoE 层输出，本页核心公式）
- 公式：
  $$u = \sum_{i \in T_k(x)} p_i \cdot E_i^{\text{routed}}(W_\downarrow x), \qquad y = \sum_{j=1}^{N_s} E_j^{\text{shared}}(x) + W_\uparrow u$$
- 来源：[K3] Eq. 11（去掉 K3 特有的 RMSNorm，回到通用 LatentMoE 形式）；[L] Figure 1(b) 的结构描述。K3 的 Stable LatentMoE 在 $W_\uparrow$ 前加 $\mathrm{RMSNorm}(u)$，本页讲通用 LatentMoE 时不加；该差异在 Q5 点明。
- 符号：
  - $x \in \mathbb{R}^d$：层输入，$d$ 为模型隐层宽度。
  - $W_\downarrow \in \mathbb{R}^{\ell \times d}$：下投影，把 $x$ 压到隐空间 $\mathbb{R}^\ell$。
  - $E_i^{\text{routed}}: \mathbb{R}^\ell \to \mathbb{R}^\ell$：第 $i$ 个路由专家，输入输出都在隐空间。
  - $T_k(x)$：对 $x$ 选中的 top-k 路由专家下标集合。
  - $p_i$：路由权重，由 router 在 $d$ 维空间给出（C7）。
  - $u \in \mathbb{R}^\ell$：路由专家的加权聚合。
  - $W_\uparrow \in \mathbb{R}^{d \times \ell}$：上投影，把 $u$ 投回 $\mathbb{R}^d$。
  - $E_j^{\text{shared}}: \mathbb{R}^d \to \mathbb{R}^d$：第 $j$ 个共享专家，输入输出都在 $d$。
  - $N_s$：共享专家数。
- 状态：已确认。

### F3（压缩比与开销缩小比例）
- 公式：路由部分开销缩小比例 $\approx d/\ell$
- 来源：[L] Figure 1 caption。
- 状态：已确认（"roughly"表示近似，因为 down/up-projection 本身有成本）。

### F4（Reinvestment：同时放大 N 与 k）
- 公式：把 $N \to \alpha N$、$k \to \alpha k$，其中 $\alpha = d/\ell$；组合数从 $\binom{N}{k} \to \binom{\alpha N}{\alpha k}$。
- 来源：[L] Figure 1 caption（"increase the total number of experts and the top-k active experts per token by the same factor d/ℓ"）。
- 状态：已确认。

## N 数字（外部数字与实验条件）

### N1（K3 Stable LatentMoE 配置）
- 数字：$d = 7168$、$\ell = 3584$（压缩 2x）、$N = 896$ 路由专家、$k = 16$ 每 token 激活、$N_s = 2$ 共享专家、稀疏度 $N/k = 56$。
- 来源：[K3] §2.3 与官方 `config.json`（`hidden_size=7168`, `routed_expert_hidden_size=3584`, `num_experts=896`, `num_experts_per_token=16`, `num_shared_experts=2`）。
- 实验条件：K3 模型，2.8T 参数规模。
- 状态：已确认。

### N2（Nemotron-3 Super 配置）
- 数字：路由路径 $4096 \to 1024 \to 4096$（压缩 4x）；120B 总参/12B 激活。
- 来源：[R] "Latent MoE"（"For Super, this was 4096 -> 1024 -> 4096"）；[L] 摘要与 §4（95B 参数规模实验）。
- 实验条件：Nemotron-3 Super，95B-8BA Transformer MoE。
- 状态：已确认。

### N3（Nemotron-3 Ultra 配置）
- 数字：路由路径 $8192 \to 2048 \to 8192$（压缩 4x）；550B 总参/55B 激活；512 路由专家、top-22、1 共享专家；路由专家中间维度 5120、共享专家中间维度 10240。
- 来源：[R] "Nemotron 3 Ultra and Latent MoE Scaling"（"Ultra doubles each width and uses 8192 -> 2048 -> 8192. Both models therefore use the same 4x compression ratio. Each Ultra MoE layer contains 512 routed experts and selects 22 for each token"）。
- 实验条件：Nemotron-3 Ultra。
- 状态：已确认。

### N4（NVIDIA 实验压缩比上限）
- 数字：16B 参数规模消融实验中，压缩比到 4 时模型质量保持；再压缩需配合 reinvestment 才不损失。
- 来源：[L] §3 消融实验（"At the 16 billion parameter scale used for ablations, model quality held for compression ratios up to four"）；辅助来源 [R] 与 kriraai.com 转述。
- 实验条件：16B 参数，特定训练数据量与超参。
- 状态：已确认（任务相关，不可外推到任意规模）。

### N5（LatentMoE 投影计算成本）
- 数字：down/up-projection 在 K2-1T 规模上约增加 9% 计算（相对原生 K2-1T）。
- 来源：[L] "Projected Serving Impact at Trillion-Parameter Scale"（"this added compute is modest (~9% relative to native Kimi-K2-1T)"）。
- 实验条件：K2-1T 规模投影分析。
- 状态：已确认。

### N6（LatentMoE 在 iso-accuracy 下的加速比投影）
- 数字：在万亿参数规模、iso-accuracy 下，LatentMoE 相对标准 MoE 投影可达约 3.5x 加速；标准 MoE 要达到相同准确率需多约 350B 参数。
- 来源：[L] Figure 2 与 §4（"if a standard MoE is scaled to match LatentMoE's accuracy gain, it requires 350B additional parameters in our analysis. At iso-accuracy, LatentMoE is projected to achieve up to 3.5× speedup over standard MoE"）。
- 实验条件：万亿参数规模投影分析，非实测。
- 状态：已确认（投影值，非实测）。

## 置信状态汇总

所有 C1–C7、F1–F4、N1–N6 均为已确认状态。无存在冲突或证据不足的核心论断，可进入生产阶段。
