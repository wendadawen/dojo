# EPLB 术语表

## 术语

| 名称 | 首次出现 | 定义/含义 |
|---|---|---|
| EPLB（Expert Parallelism Load Balancer，专家并行负载均衡器） | 页面开头 | DeepSeek 开源的专家并行负载均衡算法（eplb.py），也是 vLLM 内置同名功能；输入各逻辑专家的估计负载，输出「复制 + 排布」方案 |
| 专家并行（EP，expert parallelism） | 第 1 章 | 把 MoE 的不同专家分片到不同 GPU 的并行方式；每卡只持有部分专家 |
| dispatch / combine | 第 1 章 | EP 中把 token 激活送到专家所在卡（dispatch）与把专家输出送回原卡加权合并（combine）的两次 all-to-all 通信 |
| all-to-all | 第 1 章 | 每张卡同时与所有其他卡交换数据的通信模式 |
| 木桶效应 | 第 1 章 | 整层计算在同步点等所有卡完成，层时间由负载最大的卡决定（本文用语，描述该同步现象） |
| 层利用率 | 第 1 章 | 构造示例定义：平均负载 ÷ 最大负载，衡量忙闲不均的代价 |
| 负载 $w_e$ | 第 1 章 | 一段时间内发往逻辑专家 $e$ 的 token 数（估计值，来自统计） |
| bias（偏置项） | 第 1 章 | V3 辅助损失无关均衡中加在亲和分数上、只影响 top-K 选择的可学习偏置；训练步末按过载/欠载 ±γ 更新 |
| 逻辑专家（logical expert） | 第 2 章 | 模型定义中的专家，路由的候选单位；数量记为 num_log |
| 物理专家（physical expert）/ 副本（replica） | 第 2 章 | 显存中实际存放的一份专家参数；一个逻辑专家可对应多个物理副本；物理专家总数记为 num_phy / num_replicas |
| 副本数 $c_e$ | 第 2 章 | 逻辑专家 $e$ 拥有的物理副本个数 |
| 冗余专家（redundant expert） | 第 2 章 | 为均衡负载而额外复制的专家副本；总数 = num_phy − num_log |
| 均摊假设 | 第 2 章 | 发往同一逻辑专家的 token 被各副本均分，单副本负载 $w_e/c_e$；EPLB 打包计算的显式前提 |
| 贪心复制 | 第 2 章 | 每次给 $w_e/c_e$ 最大的逻辑专家增加一个副本的算法（replicate_experts） |
| 均衡打包（balanced packing） | 第 3 章 | 按负载降序遍历、每个对象放入仍有容量的当前最轻包、每包恰好 n/m 个对象的贪心（balanced_packing） |
| 专家组（expert group）/ 组 | 第 3 章 | 逻辑专家按连续编号划分的固定分组；EPLB 的 num_groups 参数；层次策略以组为单位装节点 |
| 节点（node） | 第 3 章 | 一台服务器（多 GPU），节点内 NVLink 快于跨节点 IB |
| 节点受限路由（node-limited routing）/ 组受限路由（group-limited expert routing） | 第 3 章 | V3 的路由约束：每 token 至多发给 M 个节点，节点按其上专家的最高 K_r/M 亲和分数之和选出；EPLB README 称 group-limited expert routing |
| NVLink / IB（InfiniBand） | 第 3 章 | 节点内 / 跨节点互连；V3 报告：160 GB/s vs 50 GB/s |
| phy2log（物理→逻辑映射） | 第 3 章 | 算法输出：每个物理槽位上放的是哪个逻辑专家 |
| 层次策略（hierarchical） | 第 3 章 | 节点数整除组数时启用：组→节点、节点内复制、物理→GPU 三步 |
| 全局策略（global） | 第 3 章 | 其余情况：忽略分组全局复制、全局打包 |
| max/mean 比 | 第 3 章 | 构造示例定义：最大 GPU 负载 ÷ 平均 GPU 负载，衡量均衡程度（越小越均衡，1 为完全均衡） |
| 重平衡（rebalance） | 第 4 章 | 周期性重算排布方案并应用（搬权重、更新映射）的过程 |
| 窗口（window）/ window_size | 第 4 章 | vLLM 中用于重平衡决策的最近引擎步数（默认 1000） |
| step_interval | 第 4 章 | vLLM 中每 N 个引擎步执行一次重平衡（默认 3000） |
| prefill / decode | 第 4 章 | 请求的预填充阶段与逐 token 解码阶段 |
| 共享专家（shared expert）/ 路由专家（routed expert） | 第 4 章 | 每 token 必经的专家 / 由 router 从中选 top-k 的专家（V3：1 共享 + 256 路由，每 token 激活 8 个路由专家） |
| top-k / $K_r$ | 第 4 章 | 每个 token 激活的路由专家数 |
| EP rank | 第 4 章 | 专家并行中的一个参与方（一张 GPU 上的执行实例）；vLLM 文档用语 |
| preserve_intragpu_slots | 第 4 章 | vLLB 机制：留在同一 GPU 的专家保持原槽位、新专家填空槽，减少权重搬运 |
| log_balancedness | 第 4 章 | vLLM 指标：avg tokens per expert ÷ max tokens per expert |

## 符号约定

- $w_e$：逻辑专家 $e$ 的估计负载（token 数）。
- $c_e$：逻辑专家 $e$ 的副本数。
- $w_e/c_e$：单副本承载的打包负载。
- $M$：节点受限路由允许的最大节点数（V3 取 4）。
- $K_r$：每 token 激活的路由专家数（V3 取 8）。
- $e_j$：第 $j$ 个逻辑专家（e0–e11，构造示例编号）。
- G0–G3：构造示例的 4 个专家组。
- 全文「负载」统一指 token 数统计；「卡」与 GPU 同义；「rank」仅出现在 vLLB 引用语境。
