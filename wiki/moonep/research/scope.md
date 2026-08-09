# MoonEP 完美均衡专家并行 · 内容范围

## 1. 概念歧义处理

- 名称：MoonEP（Moonshot Expert Parallelism）。K3 报告 §5.2.1 中提出，开源仓库 https://github.com/MoonshotAI/MoonEP。
- 歧义裁定：状态为「已裁定」。MoonEP 是 Moonshot AI 在 K3 训练系统中使用的专家并行方案名称，与通用术语"expert parallelism（EP）"是不同概念。本文中 EP 指通用专家并行机制（在 [MoE 推理与服务基础](../../wiki/moe-serving/index.html)中已讲），MoonEP 特指 K3 报告里这套带动态冗余专家的完美均衡方案。报告未公开 MoonEP 的另一种含义。
- 同名风险：报告脚注 3 把 MoonEP 链接到上述 GitHub 仓库；本页不引用仓库源码，仅依据报告 §5.2.1 与 §E 的文字描述。

## 2. 概念含义

### 2.1 一句话定义

MoonEP 是一种 MoE 训练的专家并行方案：在保留 DeepEP 等传统 EP 总体计算流的前提下，通过在线规划和迁移"冗余专家"让每个 EP rank 恰好收到 $S\times K$ 个 token，从而把负载、通信 buffer 形状和每层计算形状都变成静态已知。

### 2.2 正式定义

K3 报告 §5.2.1：「MoonEP, an EP scheme that achieves perfect load balance with dynamic redundant experts. MoonEP preserves the overall computation flow of conventional schemes such as DeepEP and additionally introduces online planning and migration of redundant experts.」核心机制由四部分组成（均来自 §5.2.1）：

1. **动态冗余专家**：forward 阶段从当前 micro-batch、当前层的 router 输出在线规划冗余专家，并在 routed-expert 计算前预取；backward 阶段把冗余专家的梯度暂存到本地 reduce buffer，计算完成后再 reduce 回它们 home rank 的梯度 buffer。
2. **有界冗余**：每 rank 至多 $E/R$ 个冗余专家即可保证存在可行解（§E Theorem 1），且该界基本紧（§E Theorem 2 给出需要 $\lceil E(R-1)/R^2\rceil\approx E/R$ 的最坏构造）。预留 $E/R$ 个槽位即保证训练不中断。
3. **零拷贝通信**：fused permute/unpermute 算子让 planning kernel 预计算每个 token 的目的地，直接发到远端 expert-grouped 位置，返回 buffer view 免中间拷贝。完美均衡下只需固定 $S\times K$ buffer（DeepEP 在最坏不均衡下要 $S\times K\times R$）。
4. **静态形状免同步**：每 rank 收 $S\times K$ token，所有层计算形状静态已知，消除每层 MoE host 同步。
5. **Expert-GEMM 调度**：workload-aware scheduler 适配 per-expert token 偏斜，shared expert 分到独立 stream 重叠。

### 2.3 本文采用的语境

只讲训练场景下的 EP（推理不在本页范围）。$S$ 指每 rank 本地序列长度（micro-batch 切到 DP rank 后每 rank 持有的 token 数），$K$ 是每 token 选的专家数，$E$ 是专家总数，$R$ 是 EP size（EP rank 数）。

### 2.4 包括什么

- 冗余专家是什么、为什么需要它（消除不均衡的载体）。
- $E/R$ 上界的含义与"基本紧"的含义（不展开完整 ILP 推导，但展示构造性证明的关键引理与最坏构造）。
- 完美均衡为何让通信 buffer 形状与每层计算形状同时静态化。
- forward/backward 在 MoonEP 下的具体步骤（与 DeepEP 的差异）。
- MoonEP 与 ECHO、UltraEP、DeepEP 的关键区别（来自 §5.2.1 的对比段）。

### 2.5 不包括什么

- DeepEP 的具体 kernel 实现（属于 DeepEP 论文/源码，本页只引用其"传统 EP 方案"的总体计算流和最坏 buffer 大小）。
- ECHO/UltraEP 的内部机制（属于各自论文，本页只引用报告 §5.2.1 对它们的对比描述）。
- ILP 离线求解器与 GPU planning kernel 的具体算法（报告只说"near-optimal, negligible overhead, respects $E/R$"，未公开算法细节；本页只引用其存在性结论）。
- Expert-GEMM workload-aware scheduler 的具体参数与 autotuning 流程（报告只给思路，本页只讲它解决什么问题）。
- Memory-efficient training（§5.2.2）、multimodal encoder（§5.2.3）：与 MoonEP 同属 §5.2 但属独立子问题，本页不展开。
- 推理场景下的 EP：由 [MoE 推理与服务基础](../../wiki/moe-serving/index.html)负责；MoonEP 是训练方案。

### 2.6 相邻概念

- **专家并行（EP）+ all-to-all dispatch/combine**：MoonEP 的前提。讲清"token-expert pair 在 rank 间搬运"这一抽象即可，不重复推导。链接到 [MoE 推理与服务基础](../../wiki/moe-serving/index.html)的 EP 章节。
- **DeepEP**：传统 EP 方案的代表性实现，MoonEP 保留其总体计算流但增加冗余专家规划。本页把 DeepEP 作为"不均衡 baseline"引用，不展开其内部机制。
- **负载均衡损失（auxiliary loss）**：训练 MoE 时常见的辅助损失（让 router 把 token 均匀分到专家）。MoonEP 不依赖它实现完美均衡——MoonEP 在 router 输出给定后做规划，不论 router 是否训练得均衡都能保证每 rank 收 $S\times K$。两者关系在边界一节说明。

## 3. 学习目标

### Q1：传统 EP 在 MoE 训练中为什么会让 rank 间负载不均衡？MoonEP 用什么核心思路让每 rank 恰好收到 $S\times K$ 个 token？

- 完成答案：读者应能说明（a）传统 EP 下每 rank 收到的 token 数由 router 偏好决定、随 micro-batch 和层变化，因此负载动态不均；（b）MoonEP 的思路是在 router 输出给定后，**在线规划"冗余专家"**——把某些专家的副本临时迁移到需要更多 token 的 rank 上计算——使得每 rank 的总 token 数恰好等于 $S\times K$。
- 为什么是核心目标：不理解传统 EP 的不均衡来源，就无法理解 MoonEP 的设计动机；不理解"冗余专家作为负载载体"这一核心思路，后续的界、通信、同步都无从落地。
- 依赖内容：传统 EP 的 dispatch/combine 数据流、token-expert pair 概念、冗余专家的定义。

### Q2：「冗余专家」是什么？为什么预留 $E/R$ 个槽位足以保证可行解存在，这个界为什么"基本紧"？

- 完成答案：读者应能说明（a）冗余专家 = 把某个 home rank 上的专家临时复制到另一个 rank 上、让该 rank 在本地处理原本要发出去的 token；（b）$E/R$ 上界来自一个构造性证明的关键引理——存在一种规划让每 rank 的"远端 token 全部来自同一个 home rank"，而一个 home rank 至多有 $E/R$ 个本地专家，故冗余数 $\le E/R$；（c）"基本紧"指存在最坏 router 输出使得至少需要 $\lceil E(R-1)/R^2\rceil\approx E/R$ 个冗余专家，所以不可能有显著小于 $E/R$ 的通用上界。
- 为什么是核心目标：这是 MoonEP 区别于 ECHO/UltraEP 的核心——后者预设冗余数或 token cap、训练可能因无解而中断；MoonEP 用一个**有证明的、基本紧的界**保证总有可行解。
- 依赖内容：冗余专家定义、构造性证明的关键引理、最坏构造（§E Theorem 2）。

### Q3：完美均衡怎样让通信 buffer 从 $S\times K\times R$ 降到 $S\times K$、消除每层 host 同步？forward/backward 怎样在线规划与归还冗余专家？

- 完成答案：读者应能说明（a）传统 EP 在最坏不均衡下要支持 zero-copy 数据路径需要 $S\times K\times R$ 的通信 buffer（任何 rank 都可能收到至多 $S\times K\times R$ token）；MoonEP 保证每 rank 恰好收 $S\times K$，所以固定 $S\times K$ buffer 即可；（b）传统 EP 每层 token 数变化，host 必须在每层与 device 同步拿真实计算形状再 launch，MoonEP 让所有层形状静态已知，消除每层 MoE host 同步；（c）forward：planning kernel 从当前 micro-batch、当前层 router 输出规划冗余专家并预取，然后 fused permute 直接把 token 发到远端 expert-grouped 位置；（d）backward：冗余专家的梯度暂存本地 reduce buffer，计算完成后再 reduce 回 home rank 的梯度 buffer。
- 为什么是核心目标：这是 MoonEP 三个工程收益的统一来源——完美均衡同时让通信 buffer、计算形状、host 同步都静态化。不理解这三者的统一来源，就会把 MoonEP 误读成"只是另一种负载均衡"。
- 依赖内容：DeepEP 的 buffer 量级、host-device 同步动机、forward/backward 在 EP 中的位置。

### Q4：MoonEP 解决什么、不解决什么？与 ECHO、UltraEP、DeepEP 的关键区别是什么？

- 完成答案：读者应能说明（a）MoonEP 解决训练时 EP 的负载不均衡、buffer 形状动态、每层 host 同步三个问题；（b）不解决 per-expert token 偏斜（由 Expert-GEMM scheduler 单独处理）、不解决 router 本身的训练（router 均衡靠辅助损失仍可能需要，MoonEP 只在 router 输出给定后规划）、不解决内存碎片（由 §5.2.2 memory-efficient training 独立处理）；（c）与 ECHO/UltraEP 区别：后者预设冗余数或 token cap、训练可能因无解而中断且需手动调参，MoonEP 用 $E/R$ 上界保证总有解；（d）与 DeepEP 区别：MoonEP 保留 DeepEP 总体计算流但增加冗余专家规划，最坏 buffer 从 $S\times K\times R$ 降到 $S\times K$。
- 为什么是核心目标：没有这一节，读者会把 MoonEP 误读为"通用负载均衡方案"或"DeepEP 的替代品"，忽略它的训练场景定位和与 router 训练的边界。
- 依赖内容：前三问的结论、ECHO/UltraEP/DeepEP 的对比描述（§5.2.1）。

## 4. 内容分级

### 4.1 核心内容（缺一不可，直接服务学习目标）

| 内容 | 对应学习目标 | 必须讲清的结论 |
|---|---|---|
| 传统 EP 不均衡的来源（router 偏好动态变化） | Q1 | 每 rank 收到 token 数 = 该 rank 上专家被选中的总次数，随 micro-batch 和层变化 |
| 冗余专家的定义与作用 | Q1, Q2 | 冗余专家 = 临时复制到非 home rank 的专家副本，让该 rank 在本地处理本应发出的 token |
| $E/R$ 上界的关键引理与构造性证明 | Q2 | 存在规划使每 rank 远端 token 全来自同一 home rank；home rank 至多 $E/R$ 本地专家；故冗余数 $\le E/R$ |
| 最坏构造与"基本紧"含义 | Q2 | 存在 router 输出使 rank 0 需要 $\lceil E(R-1)/R^2\rceil$ 冗余专家；大 $R$ 下近似 $E/R$ |
| 完美均衡 → buffer 从 $S\times K\times R$ 降到 $S\times K$ | Q3 | DeepEP 最坏需 $S\times K\times R$；MoonEP 每 rank 恰好 $S\times K$ |
| 完美均衡 → 消除每层 host 同步 | Q3 | 传统 EP 每层 token 数变化，host 要与 device 同步拿形状；MoonEP 静态形状免同步 |
| forward/backward 的规划与归还流程 | Q3 | forward：planning kernel 规划冗余专家并预取；backward：梯度暂存本地 reduce buffer 再 reduce 回 home rank |
| MoonEP 解决/不解决的问题与边界 | Q4 | 解决负载、buffer、host 同步；不解决 per-expert 偏斜、router 训练、内存碎片 |
| 与 ECHO/UltraEP/DeepEP 的区别 | Q4 | ECHO/UltraEP 预设 cap 可能无解；DeepEP 最坏 buffer 大；MoonEP 有界保证 |

### 4.2 辅助内容（不直接构成核心答案，消除理解障碍或澄清误解）

| 内容 | 服务的核心内容 / 误解 |
|---|---|
| DeepEP 作为"传统 EP 方案"的代表性实现 | 澄清 MoonEP 不是 DeepEP 的替代，而是保留其计算流并加冗余专家规划 |
| 负载均衡辅助损失与 MoonEP 的关系 | 澄清"router 训练均衡 ≠ EP 后负载均衡"这一误解 |
| Expert-GEMM workload-aware scheduler | 说明完美均衡只保证 rank 间总量均衡，per-expert 偏斜由独立 scheduler 处理 |
| ILP 离线求解器与 GPU planning kernel | 说明 MoonEP 不每步求精确最优，而用近最优 GPU kernel；具体算法未公开 |
| 报告原文对 "ECHO/UltraEP presets / imposes cap" 的描述 | 提供对比来源定位 |

### 4.3 扩展内容（与本概念相关但不影响学习目标回答）

| 内容 | 处理 |
|---|---|
| MoonEP 开源仓库的实际代码结构 | 排除（报告未公开实现细节，本页不引用仓库源码） |
| 与 SonicMoE 的关系（§5.2.2 提到） | 排除（属于 memory-efficient training，与 MoonEP 独立） |
| K3 训练 2.8T MoE 的具体配置（E、R、K、S 取值） | 排除（属于 K3 架构 note 页，不影响机制理解） |
| TBO/SBO 通信掩盖 | 排除（属于推理场景，由 moe-serving 页负责；训练场景下 MoonEP 的通信开销由静态形状本身减小，不是 TBO/SBO 思路） |

## 5. 前置知识映射

| 前置概念 | 被哪些学习目标依赖 | 概念页链接 / 生成状态 | 递归层级 |
|---|---|---|---|
| MoE 基础（router、top-k、shared/routed expert、token-expert pair） | Q1, Q3 | [MoE 推理与服务基础](../../wiki/moe-serving/index.html) 已有 | 0 |
| 专家并行（EP）+ all-to-all dispatch/combine | Q1, Q3 | [MoE 推理与服务基础](../../wiki/moe-serving/index.html) 已有 | 0 |
| GPU host-device 同步动机（kernel launch 形状确定） | Q3 | [GPU 执行模型与 kernel 调度](../../wiki/gpu-execution-model/index.html) 已有 | 0 |
| reduce（梯度归约）语义 | Q3 | moe-serving 页 S3 提到 all-reduce，但 EP backward 的 reduce-back 概念在正文用最小一句话解释；不强制递归生成 reduce 概念页（递归层级 1 起，登记不生成） | 1（登记） |

无递归生成需求（moe-serving 已覆盖最大前置；reduce 语义在正文就地最小解释，不展开）。

## 6. 明确不展开的内容

| 不展开的内容 | 与概念的关系 | 不展开的原因 |
|---|---|---|
| DeepEP 的 fused kernel 内部实现 | DeepEP 是 MoonEP 保留的"总体计算流"基础 | 属于 DeepEP 论文/源码；报告只引用其 buffer 量级，不依赖实现细节 |
| ECHO、UltraEP 的内部机制 | 与 MoonEP 对比的相邻方案 | 报告只给对比描述（preset cap / impose cap），未给机制细节；本页只引用对比结论 |
| GPU planning kernel 的具体算法 | MoonEP 的在线规划实现 | 报告未公开算法细节，只说 near-optimal、negligible overhead、respect $E/R$；本页只引用其存在性 |
| Expert-GEMM scheduler 的 autotuning 流程 | MoonEP 的工程优化 | 报告只给思路（analytical cost model + offline autotuning），未给系数；本页只讲它解决什么问题 |
| ILP 离线求解器在代表性 case 上的具体解法 | 验证 GPU kernel 近最优性的参考 | 报告只说"offline with ILP for representative cases as references"，未公开 case 与解；本页只引用其方法定位 |
| Memory-efficient training（§5.2.2） | 与 MoonEP 同属 §5.2 的相邻子问题 | 独立子问题，与完美均衡无关 |
| Multimodal encoder optimization（§5.2.3） | 与 MoonEP 同属 §5.2 的相邻子问题 | 独立子问题 |
| 推理场景下的 EP | MoonEP 是训练方案 | 由 moe-serving 页负责 |

## 7. 常见误解与适用边界

### 7.1 常见误解

| 误解 | 正确结论 | 形成原因 | 影响的目标 |
|---|---|---|---|
| "MoonEP 是一种 router 负载均衡损失" | MoonEP 不是 router 训练手段，而是在 router 输出给定后做 token-专家规划的工程方案。router 是否均衡训练由辅助损失负责，与本页独立。 | "load balance"一词在 MoE 训练里既可指 router 均衡也可指 EP 后 rank 均衡，容易混。 | Q1, Q4 |
| "冗余专家 = 多一份专家权重常驻" | 冗余专家是**当前 micro-batch、当前层**临时迁移的副本，forward 预取、backward 归还；不是永久副本。 | "冗余"一词容易让人以为是常驻冗余。 | Q1, Q2 |
| "$E/R$ 是工程经验值，调一调也行" | $E/R$ 是有证明的上界（Theorem 1），且基本紧（Theorem 2）。ECHO/UltraEP 才是预设经验值。 | 容易把"工程参数"和"有证明的界"混为一谈。 | Q2, Q4 |
| "完美均衡消除了 per-expert 偏斜" | 完美均衡只保证 rank 间总 token 数相等，per-expert token 数仍可能偏斜（§5.2.1 最后一段明确说）。per-expert 偏斜由 Expert-GEMM scheduler 独立处理。 | "完美均衡"听起来像所有维度都均衡。 | Q3, Q4 |
| "MoonEP 适用于推理" | MoonEP 是训练方案。推理 EP 的负载问题由 moe-serving 页讲的 TBO/SBO、PD 分离等思路处理，机制不同。 | EP 一词在训练和推理都出现。 | Q4 |
| "DeepEP 的 buffer 总是 $S\times K\times R$" | $S\times K\times R$ 是 DeepEP **最坏不均衡**下要支持 zero-copy 数据路径所需的 buffer；非最坏情况下可以更小。MoonEP 因保证完美均衡才把 buffer 恒定为 $S\times K$。 | 报告原文用了 "Under worst-case imbalance" 限定。 | Q3 |

### 7.2 适用边界

- **MoonEP 解决的问题**：训练时 EP 的 rank 间负载不均衡、通信 buffer 形状动态、每层 host 同步开销。
- **MoonEP 不解决的问题**：
  - per-expert token 偏斜（由 Expert-GEMM scheduler 独立处理）；
  - router 本身的训练（router 均衡靠辅助损失仍可能需要）；
  - 内存碎片（由 §5.2.2 memory-efficient training 独立处理）；
  - 推理场景的 EP 负载（机制不同，见 moe-serving 页）。
- **结论成立需要的条件**：
  - 每个专家被至多一份副本迁移（构造性证明假设）；
  - 每 rank 预留 $E/R$ 个冗余槽位（保证可行解存在）；
  - router 输出给定后规划才发生（MoonEP 不修改 router）。
- **条件不满足时**：若预留槽位少于 $E/R$，则可能存在 router 输出使无解（Theorem 2 的构造需要 $\lceil E(R-1)/R^2\rceil$ 个槽位）；若 router 训练极度不均衡，MoonEP 仍能保证每 rank 收 $S\times K$，但冗余专家迁移的通信开销会接近上界。
