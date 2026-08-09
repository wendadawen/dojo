# MLA 教学大纲

## 1. 页面开头

### 钩子（最小场景，30 秒内定位问题）

读者设想一个 128K 上下文的模型在自回归生成。每生成一个新 token，它必须保留前序所有 token 的 K 和 V，否则下一步注意力无 key/value 可比。标准 MHA 下，每 token 每层要缓存 2 × n_h × d_h 个数——DeepSeek-V2 是 128 头 × 128 维 × 60 层，即每 token 约 2 MB（BF16）KV cache，128K 上下文下整个 cache 约 256 GB。如何把这个数减到 1/57 同时不掉点？这就是 MLA 要解决的问题。

### 一句话解释

MLA 把每 token 的 K 和 V 压成一个低维潜向量 c_t^{KV} 缓存，注意力时再用学习到的上投影把 K 和 V 重建出来——压缩存储、保留全局注意力。

### 学习承诺（与 scope.md Q1–Q5 完全对应）

读完这一页，你应该能够：
1. 说明 MLA 用低秩联合压缩如何减少 KV cache，同时保留全局 token-to-token 注意力（Q1）；
2. 解释为什么 MLA 要把 RoPE 解耦，不能直接对压缩潜向量应用 RoPE（Q2）；
3. 算出 MLA 的 KV cache 每 token 占多少元素，并与 MHA / GQA / MQA 比较（Q3）；
4. 说明推理时如何通过"矩阵吸收"避免显式重建 K 和 V（Q4）；
5. 指出 K3 在 DeepSeek-V2 MLA 上做的两处改动——NoPE 与 input-dependent full-rank output gate——各自解决什么问题（Q5）。

### 首个具体场景

引入贯穿例子的小维度配置：d_h = 4（每头维度）、n_h = 2（头数）、d_c = 3（KV 压缩维度）、d_h^R = 2（解耦 RoPE 维度）、l = 1（单层，先看单层 cache）。说明这是教学构造的小数字，便于手算，对应真实数值在 Q3 处给出（DeepSeek-V2、K3）。

### 与第一章的过渡

"下面先看 MHA 下 KV cache 是怎么来的——这能让我们看清楚 MLA 到底压缩了什么。"

## 2. 章节设计

章节采用扁平结构（h2 直接跟内容），每章只承担一个主要教学问题。每章末尾有"完成检查"，每章后给过渡到下一章的逻辑。

### S1：MLA 压缩了什么——KV 联合压缩的核心机制

- 主要教学问题：MLA 用什么方法减少 KV cache，同时保留全局注意力？
- 对应范围：Q1；核心内容 C1、C2；公式 F2、F3；数字 N1。
- 正文要点：
  - 先回顾 MHA 下 q/k/v 怎么来（一句话给公式 F1，不展开推导，前置 mqa-gqa 概念页占位）。
  - 引出"每头一份 K/V"导致 cache = 2 n_h d_h l。
  - 引入 c_t^{KV} = W^{DKV} h_t（F2 Eq.(9)）——把 h_t 压成 d_c 维潜向量，d_c ≪ d_h n_h。
  - 给出 K/V 重建：k_t^C = W^{UK} c_t^{KV}、v_t^C = W^{UV} c_t^{KV}（F2 Eq.(10)(11)）。
  - Query 也走压缩：c_t^Q = W^{DQ} h_t、q_t^C = W^{UQ} c_t^Q（F3 Eq.(12)(13)）。
  - 注意力公式仍是 softmax 全局求和（F4 Eq.(18)，先暂用未解耦的简化形式，分母 sqrt(d_h)）——明确"全局 token-to-token attention 没变"。
  - 贯穿例子手算：用 d_h=4, n_h=2, d_c=3 算一个 token 的 h_t（4 维）→ c_t^{KV}（3 维）→ k_t^C（8 维 = 2×4）；显示 cache 从 MHA 的 16 元素降到 MLA 的 3 元素。
- 讲解材料及职责：
  - 公式 F1（一句话回顾，给前置链接占位）。
  - 公式 F2（核心，逐符号定义）。
  - 公式 F3（核心，解释为什么 query 也压缩——为 S3 矩阵吸收铺路）。
  - 数字例子（贯穿例子第一次推进：手算 c_t^{KV} 和 cache 大小）。
  - ASCII 图示：MHA 缓存全 K/V 矩阵 vs MLA 只缓存 c_t^{KV} 向量的对比。
- 前置知识安排：
  - mqa-gqa 概念页占位链接 + 一句话事实（"标准 MHA 为每个头独立产生 K 和 V"）。
  - low-rank-projection 概念页占位链接 + 一句话事实（"用矩阵 W^{DKV} 把高维向量投影到低维子空间"）。
- 完成检查：
  - 写出 c_t^{KV} 的形状、k_t^C 的形状（用贯穿例子的维度）。
  - 说明 MLA 的注意力是否仍是全局 softmax（是/否，给一句话理由）。
- 过渡：cache 已经从 2 n_h d_h 降到 d_c。但推理时还要不要显式算出 K 和 V？下一章讲怎么把这一步也省掉。

### S2：推理时不重建 K/V——矩阵吸收

- 主要教学问题：推理时为什么可以不显式重建 K 和 V？
- 对应范围：Q4；核心内容 C4；公式 F5；数字 N1。
- 正文要点：
  - 给出注意力核心项 q^T k^C，代入 k_t^C = W^{UK} c_t^{KV}：q^T W^{UK} c_t^{KV}。
  - 用矩阵乘法结合律改写：(W^{UK T} q)^T c_t^{KV}——把 W^{UK} 从 key 侧搬到 query 侧，得到"等价 query" q' = W^{UK T} q，直接和 c_t^{KV} 做内积。
  - 这意味着推理时把 W^{UK} 吸进 W^Q（合并成一个新矩阵），就再也不用显式算 k_t^C。
  - 同理：v_t^C = W^{UV} c_t^{KV} 代入 o = sum softmax(...) v 后，W^{UV} 可外推吸进 W^O，连 v_t^C 也不用算。
  - 结论：推理时只存 c_t^{KV}，直接做 attention；K、V 在整个推理路径上不出现。
  - 关键限制：吸收要求 W^{UK} 不依赖位置；如果对 k_t^C 用了 RoPE，吸收失效——这是下一章要解决的问题。
- 讲解材料及职责：
  - 推导折叠块：从 q^T k^C 到 (W^{UK T} q)^T c_t^{KV} 的两步代数，标注"结合律"和"转置规则"。
  - ASCII 图示：MHA 推理路径（h → K, V → attention → o）vs MLA 推理路径（h → c → attention → o，K/V 不出现）。
- 前置知识安排：low-rank-projection 占位（"AB 的转置等于 B^T A^T"），不内联展开。
- 完成检查：
  - 写出 q' = W^{UK T} q 的形状（用贯穿例子：q 是 4 维，W^{UK} 是 4×3，故 q' 是 3 维）。
  - 说明为什么 W^{UV} 能吸进 W^O 而不能吸进 W^Q（提示：W^{UV} 在 v 侧、与 q 无关；W^{UK} 在 k 侧、与 q 内积）。
- 过渡：矩阵吸收看上去很美，但有个隐藏前提——W^{UK} 必须与位置无关。可是 MHA 系模型几乎都用 RoPE 给 K 加位置信息，怎么办？下一章讲这个冲突和 MLA 的解法。

### S3：为什么 RoPE 要解耦——位置编码与矩阵吸收的冲突

- 主要教学问题：为什么 MLA 要把 RoPE 解耦，不能直接对 c_t 应用 RoPE？
- 对应范围：Q2；核心内容 C3；公式 F4；数字 N1。
- 正文要点：
  - 给出 RoPE 的事实（rope 概念页占位 + 一句话）：RoPE 是位置敏感的旋转矩阵 R_t，作用在 k 上是 R_t k；不同 token 的 R_t 不同。
  - 假设直接对 k_t^C = W^{UK} c_t^{KV} 应用 RoPE：注意力项变成 q^T (R_t W^{UK} c_t^{KV})。
  - 关键观察：R_t 介于 q 和 W^{UK} 之间，且每个 token 的 R_t 不同，无法把 R_t W^{UK} 合并成单个矩阵吸收进 W^Q——位置敏感的旋转破坏了吸收的前提。
  - DeepSeek-V2 的解法（F4 Eq.(14)–(18)）：
    - 额外用一组多头 query q_t^R = RoPE(W^{QR} c_t^Q) 承载 RoPE（每头一份）。
    - 额外用一个共享 key k_t^R = RoPE(W^{KR} h_t) 承载 RoPE（所有头共享同一份）。
    - content 部分与 RoPE 部分拼接：q_{t,i} = [q_{t,i}^C; q_{t,i}^R]、k_{t,i} = [k_{t,i}^C; k_t^R]。
    - 注意力分母改为 sqrt(d_h + d_h^R)（因为拼接后向量维度是 d_h + d_h^R）。
    - content 部分 W^{UK} 仍可吸进 W^Q；RoPE 部分 k_t^R 不能吸收，必须单独缓存。
  - 贯穿例子手算：d_h=4、d_h^R=2，content+rope 拼接后 query 是 6 维，分母 sqrt(6)；k_t^R 是 2 维向量，每个 token 也要缓存。
  - cache 公式更新为 (d_c + d_h^R) l：d_c 来自 c_t^{KV}，d_h^R 来自 k_t^R。
- 讲解材料及职责：
  - 公式 F4（核心，逐符号定义，特别说明 [;] 是拼接、k_t^R 无下标 i 表示所有头共享）。
  - 推导折叠块：直接展示"若 R_t 作用在 k_t^C 上则 R_t W^{UK} 不可吸收"的代数（一处矩阵乘积中含位置敏感矩阵的反例）。
  - 数字例子折叠块：用贯穿例子算拼接后的 query 维度、cache 大小变化。
- 前置知识安排：rope 概念页占位 + 一句话事实（"RoPE 把位置 t 编码为一个旋转矩阵 R_t，作用在 q 和 k 上"）。
- 完成检查：
  - 说明为什么 RoPE 不能作用在 k_t^C 上而要单独拆出 k_t^R（一句话）。
  - 写出 MLA cache 公式 (d_c + d_h^R) l 中两项分别来自哪里。
  - 说明注意力分母为什么从 sqrt(d_h) 变成 sqrt(d_h + d_h^R)。
- 过渡：到这里 MLA 的核心机制已经讲完。下面用真实数值看 cache 到底减少了多少，以及与 GQA/MQA 的对比。

### S4：KV cache 到底减少了多少——与 MHA/GQA/MQA 对照

- 主要教学问题：MLA 的 KV cache 每 token 占多少，相比 MHA/GQA/MQA 减少多少？
- 对应范围：Q3；核心内容 C7、C8；公式 F2 的 cache 推论；数字 N1、N2、N3、N4。
- 正文要点：
  - 给出四种机制的每 token 每 layer KV cache 公式表（F2 + Table 1）：
    - MHA：2 n_h d_h l（每头一份 K、一份 V）
    - GQA：2 n_g d_h l（n_g 组共享 K/V）
    - MQA：2 d_h l（所有头共享一组 K/V）
    - MLA：(d_c + d_h^R) l（一份潜向量 + 一份共享 RoPE key）
  - 用 DeepSeek-V2 真实数值代入（N1）：n_h=128, d_h=128, d_c=512, d_h^R=64, l=60。
    - MHA：2 × 128 × 128 × 60 = 1,966,080
    - MQA：2 × 128 × 60 = 15,360
    - MLA：(512 + 64) × 60 = 34,560
    - MLA/MHA = 576 / 32768 ≈ 1/57（reduce ~98.2%）
  - 用 K3 真实数值代入（N4）：n_h=96, d_h=128, d_c=512, d_h^R=64（K3 中 MLA 层 num_attention_heads=96，single MLA layer cache = 576）。
    - MHA 单层：2 × 96 × 128 = 24,576
    - MLA 单层：576
    - 比值 ≈ 1/43
  - 关于 93.3% 的澄清（C8）：摘要的 93.3% 是 vs DeepSeek 67B（GQA baseline、不同配置），不是 vs 同配置 MHA。同配置下 MLA 的减少约 98.2%。
  - K3 NoPE 对 cache 的影响：K3 §2.1.2 第二段说 MLA 层不施加 RoPE，但 config.json 仍保留 qk_rope_head_dim=64；报告未明确 NoPE 是否让 k_t^R 免缓存。本页按 config 数值计算 (d_c + d_h^R) l，并标注此不确定性。
- 讲解材料及职责：
  - 对照表格（组件 14）：四种机制 × 公式 / 每 token 元素数 / 能力。
  - 数字例子（正文）：DeepSeek-V2 与 K3 两组真实数值代入。
  - callout：93.3% 的来源澄清，标黄（注意事项）。
- 前置知识安排：mqa-gqa 占位（"GQA 让 n_g 组 query 头共享一组 K/V；MQA 是 n_g=1 的特例"）。
- 完成检查：
  - 写出四种机制的每 token 每 layer cache 公式。
  - 用 DeepSeek-V2 数值算 MLA vs MHA 的比值。
  - 说明摘要的 93.3% 是相对什么 baseline。
- 过渡：MLA 本体讲完了。最后一章看 K3 在此基础上做了哪两处改动，以及为什么。

### S5：K3 的 Gated MLA——NoPE 与 full-rank output gate

- 主要教学问题：K3 在 DeepSeek-V2 MLA 上做了哪两处改动，各自解决什么？
- 对应范围：Q5；核心内容 C5、C6；公式 F6；数字 N4、N5。
- 正文要点：
  - 改动一（C5）：K3 对所有 MLA 层用 NoPE（不施加 RoPE），来自 §2.1.2 第二段。
    - 动机一：与 KDA 混合架构的分工——MLA 层负责"无位置约束的全局内容交互"，KDA 层负责"位置敏感的近邻混合"。
    - 动机二：免去扩展上下文时调整 RoPE 频率基（如 YaRN）的工程麻烦——MLA 层根本没用 RoPE，自然不需要改。
    - config.json 字段：mla_use_nope=true。
    - 与 S3 的关系：NoPE 意味着 K3 的 MLA 层不存在"RoPE 与矩阵吸收冲突"的问题，S3 的解耦分支在 K3 下事实上不施加 RoPE（但参数结构 qk_rope_head_dim=64 保留）。
  - 改动二（C6）：K3 给 MLA 加 input-dependent full-rank output gate（Eq.(7)）。
    - 公式：y_t = W^O [Sigmoid(W^g x_t) ⊙ õ_t]，其中 õ_t 是未 gate 的 MLA 输出，W^g 是满秩矩阵（K3 §2.1.2 末段说明与 KDA 的 gate 一致）。
    - 作用：让每个 token 根据当前输入 x_t 调制从全局注意力读出的通道（channel-wise 乘性 gate）。
    - config.json 字段：mla_use_output_gate=true。
    - 与 S1 的关系：S1 的最终输出 u_t = W^O [...] 在 K3 下变成 y_t = W^O [Sigmoid(W^g x_t) ⊙ ...]。
  - 贯穿例子最后一次推进：在 S1 的 õ_t 基础上手算 gate——构造 4 维 x_t、W^g（4×4 单位阵便于手算）、Sigmoid 输出 ⊙ õ_t，看 gate 如何按通道缩放输出。
  - K3 MLA 层的位（N5）：93 层中 24 层是 MLA（full_attn_layers），其余 69 层是 KDA；本文不展开调度。
- 讲解材料及职责：
  - 公式 F6（核心，逐符号定义）。
  - 数字例子（贯穿例子最后一次推进）。
  - callout（标蓝，提示）：K3 的 NoPE + output gate 是改动，不是 MLA 本体；DeepSeek-V2 原始 MLA 没有 gate，且用解耦 RoPE。
- 前置知识安排：本节无新前置概念。
- 完成检查：
  - 指出 K3 的两处改动各是什么（一句话）。
  - 说明 NoPE 解决了 DeepSeek-V2 MLA 中的什么冲突（提示：S3）。
  - 写出 output gate 公式中 W^g 的满秩性质如何让 gate 不退化。
- 过渡：（文末"来源与教学说明"前的小结）四问已答完；MLA 用低秩压缩 + 矩阵吸收把 KV cache 减到 MHA 的 ~1/57、保留全局注意力；解耦 RoPE 解决了位置编码与吸收的冲突；K3 用 NoPE 直接绕开这个冲突，并加了 output gate 提升表达力。

## 3. 讲解顺序检查

- 先讲为什么需要它（S1 开头回顾 MHA cache 来源），再讲是什么（S1 给压缩公式）。
- 一次只引入一个新变量：S1 引入 c_t^{KV}；S2 引入"等价 query" q'（不引入新对象，只是 W^{UK} 与 W^Q 合并）；S3 引入 k_t^R；S4 引入数值对照；S5 引入 W^g。
- 前置概念在首次用到前给占位 + 一句话事实，不展开。
- S2 矩阵吸收依赖 S1 的 W^{UK}、W^{UV}（已给）；S3 解耦 RoPE 依赖 S2 的吸收结论（已给）；S4 数值依赖 S1 公式（已给）；S5 K3 改动依赖 S1、S3（已给）。无前置倒置。

## 4. 贯穿例子

### 配置（贯穿全篇）

教学示例。 小维度配置：d_h = 4、n_h = 2、d_c = 3、d_h^R = 2、l = 1（先看单层）。所有数字便于手算；真实数值在 S4 给出。

### 第一次推进（S1）

- 构造 h_t = (1, 0, 1, 0)（4 维）。
- 构造 W^{DKV}（3×4，便于手算的 0/1/-1 矩阵）。
- 手算 c_t^{KV} = W^{DKV} h_t（得到 3 维向量）。
- 显示 cache：MHA = 2 × 2 × 4 = 16 元素，MLA = 3 元素。
- 注：不手算 K/V 重建（只说明形状，避免 S2 矩阵吸收还没讲就先算 K/V）。

### 第二次推进（S2）

- 用 S1 的 c_t^{KV}（3 维）。
- 构造 W^{UK}（4×3）、q（4 维）。
- 手算 q' = W^{UK T} q（3 维）。
- 手算 q'^T c_t^{KV}（标量，即注意力分数）。
- 显示：等价于先算 k^C = W^{UK} c_t^{KV}（4 维）再算 q^T k^C，但 c_t^{KV} 直接参与、k^C 不出现。

### 第三次推进（S3）

- 用 S1、S2 的量。
- 构造 q_t^R（2 维）、k_t^R（2 维）。
- 拼接 query = [q^C; q^R]（4+2=6 维）、key = [k^C; k^R]（4+2=6 维）。
- 注意力分母 sqrt(6)。
- cache 更新：c_t^{KV}（3）+ k_t^R（2）= 5 元素（MHA 仍是 16）。

### 第四次推进（S4）

- 不引入新构造，直接用真实数值。
- DeepSeek-V2：n_h=128, d_h=128, d_c=512, d_h^R=64, l=60。
- K3：n_h=96, d_h=128, d_c=512, d_h^R=64, l=24 MLA 层。
- 算 cache 比值。

### 第五次推进（S5）

- 构造 x_t（4 维）、W^g（4×4 单位阵便于手算）。
- Sigmoid(W^g x_t) = Sigmoid(x_t)。
- 取 S1 算出的 õ_t（4 维）。
- 手算 Sigmoid(x_t) ⊙ õ_t（4 维）。
- 显示 gate 如何按通道缩放。

## 5. 讲解材料职责汇总

| 材料 | 服务章节 | 教学问题 |
|---|---|---|
| 公式 F1 回顾 | S1 开头 | MHA 的 q/k/v/cache 怎么来 |
| 公式 F2 KV 联合压缩 | S1 | 压缩与重建机制 |
| 公式 F3 Query 压缩 | S1 | 为 S2 吸收铺路 |
| 公式 F4 解耦 RoPE | S3 | RoPE 与吸收的冲突与解法 |
| 公式 F5 矩阵吸收代数 | S2 | 推理时不重建 K/V |
| 公式 F6 K3 output gate | S5 | K3 改动二 |
| 对照表格 Table 1 | S4 | 四机制 cache 对比 |
| 数字例子（小维度） | S1/S2/S3/S5 | 贯穿例子逐步推进 |
| 数字例子（真实数值） | S4 | DeepSeek-V2 与 K3 cache 量级 |
| ASCII 图示 MHA vs MLA 路径 | S2 | 推理路径对比 |
| 推导折叠块 | S2、S3 | 完整代数展开 |
| 数字例子折叠块 | S3 | 拼接与 cache 更新的完整计算 |

## 6. 正文与折叠块分工

### 必须放正文

- KV cache 来源（MHA 下每头一份 K/V）。
- c_t^{KV}、W^{DKV}、W^{UK}、W^{UV}、c_t^Q、W^{DQ}、W^{UQ} 的定义与符号。
- 矩阵吸收的结论（W^{UK} 吸进 W^Q、W^{UV} 吸进 W^O）。
- 解耦 RoPE 的 q^R、k^R、拼接、sqrt(d_h + d_h^R) 公式。
- KV cache 对照表与四种机制公式。
- K3 两处改动（NoPE、output gate）的结论与公式。
- 贯穿例子的关键推进（每章的最终数字结果）。
- 误解 M1–M4 的正确结论。

### 可放折叠块

- 矩阵吸收的完整代数（从 q^T k^C 到 (W^{UK T} q)^T c_t^{KV} 的两步）。
- 解耦 RoPE 冲突的反例代数（R_t W^{UK} 不可吸收）。
- 贯穿例子的完整手算（每章的具体数字代入过程）。
- W^{UV} 吸进 W^O 的完整推导（与 W^{UK} 吸进 W^Q 对称，正文给一个，折叠给另一个）。

折叠块全部收起时正文仍须回答 Q1–Q5：正文保留每个公式、每个结论与贯穿例子的最终数字，折叠块只放推导过程与中间步骤。

## 7. 范围与证据约束检查

- 大纲只使用 scope.md 中已纳入范围的内容。✓
- 章节任务单一，每章对应一个学习目标。✓
- 贯穿例子固定输入（d_h=4, n_h=2, d_c=3, d_h^R=2, h_t = (1,0,1,0)）逐步推进。✓
- 前置概念占位提示在首次用到前给出。✓
- 误解 M1–M4 在正文相应章节处理（M1、M3 在 S1；M2 在 S1/S2；M4 在 S3/S5）。✓
- 不展开的内容（GPU kernel、K3 调度、训练算力）明确排除。✓
