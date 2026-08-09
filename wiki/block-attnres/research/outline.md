# Block AttnRes：教学大纲

## 1. 页面开头

### 钩子问题

"93 层的 K3 主干，每加一层都把前 92 层的信息等权累加进同一个残差流——第 1 层 embedding 在第 93 层还能被找回多少？标准残差连接把深度方向变成了一个 RNN：所有历史被压成单一流，越深越稀释。K3 用 Block AttnRes 替代这个等权累加：每层不再被动接收'累加好的过去'，而是用 attention 主动检索此前块级表征。"

### 一句话解释

Block AttnRes 是 Kimi K3 的跨层信息流机制：把主干按 12 层分块，每层用可学习 pseudo-query 对"embedding + 此前块级快照 + 当前块 partial sum"做 softmax 加权检索，让深网络每层都能按内容回看历史块。

### 学习承诺（读完你能回答）

见组件 `learning-goals`，5 个核心问题对应 scope.md §1.2 的 Q1-Q5。

### 首个具体场景

引入贯穿例子：一个 $N=3$ 块、每块 $S=2$ 层、$d=2$ 的小网络，用来在后续章节手算 Block AttnRes 的加权检索。

### 与第一章的过渡

"先看标准残差在深度上的瓶颈——这决定了 AttnRes 必须解决什么。"

## 2. 章节设计

### S1：标准残差在深度上的瓶颈——为什么需要 AttnRes

- **主要教学问题**：标准残差把所有前层信息等权压进单一流，深网络中会发生什么？AttnRes 用什么思路替代？
- **对应范围**：Q1（C1）
- **正文要点**：
  1. 复用前置概念页 `wiki/residual-connection/` 的结论：标准残差 $h_{l+1}=h_l+F_l(h_l)$ 把每层输出等权加进流，所有历史被压成单一流 $h_l$。
  2. 用 RNN 类比说明瓶颈：RNN 在时间维度把所有历史压进 hidden state，深网络的标准残差在深度维度做同样的事——越深，早期信息越被稀释或覆盖。
  3. 引入 AttnRes 的核心思路：把"沿深度做累加"替换为"沿深度做 attention"——每层用可学习 pseudo-query 对此前所有层输出做 softmax 加权检索，按内容选择要回看哪些层。
  4. 与序列方向 attention 的类比（一句话）：标准 attention 把"沿 token 维度做累加"替换为"沿 token 维度做 attention"；AttnRes 把同样的思路用到深度维度。
- **讲解材料及职责**：
  - 对照表（标准残差 vs AttnRes）：服务"等权累加 → softmax 加权"的对比
  - ASCII 图示（深度方向的 RNN 瓶颈）：服务"所有历史压成单一流"的可视化
- **前置知识安排**：引用 `wiki/residual-connection/` 的"等权累加"性质与"退化问题"动机，不内联重复讲解。
- **完成检查**：
  1. 用一句话说明标准残差在深度上的瓶颈与 RNN 瓶颈的类比关系。
  2. 说出 AttnRes 用什么思路替代等权累加。
- **过渡**：本章说明了 AttnRes 的思路，但还没给出公式。下一章看 Full AttnRes 的具体定义——pseudo-query、keys/values、softmax kernel 各是什么。

### S2：Full AttnRes 的公式——pseudo-query 如何检索前序层

- **主要教学问题**：Full AttnRes 的公式是什么？pseudo-query、keys/values、softmax kernel 各代表什么？
- **对应范围**：Q2（C2、F1、F2、F5）
- **正文要点**：
  1. 给出 Full AttnRes 的定义（Eq.8、Eq.9）：层 $l$ 自带可学习 pseudo-query $q_l=w_l\in\mathbb{R}^d$；keys 与 values 取 $h_1$（embedding，$i=0$）与 $f_i(h_i)$（前序各层输出，$1\le i\le l-1$）。
  2. 解释 softmax kernel $\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k))$：内积衡量方向相似度，exp 把内积转成正权重，RMSNorm 防止幅值大的层主导（详细机制留给 S5）。
  3. 给出权重 $\alpha_{i\to l}$ 与输出 $h_l=\sum_i\alpha_{i\to l}v_i$。
  4. RMSNorm 最小说明（公式 F5 + 一段话）：把每个 key 按其均方根归一化，等价于只用方向不用幅值；与 softmax 配合让权重按方向分配。
  5. 边界检查：若所有 key 方向相同，softmax 接近均匀；若 pseudo-query 与某 key 方向高度一致，权重集中在该 key。
  6. 引入贯穿例子（$N=3$、$S=2$、$d=2$）的 Full AttnRes 版本：先假设不分块，对第 6 层（最后一层）用 pseudo-query $q_6$ 对 6 个候选（embedding + 5 个层输出）做加权检索。
- **讲解材料及职责**：
  - 公式 F1、F2、F5：表达变量关系
  - 数字例子折叠块（Full AttnRes 版本，$d=2$、6 个候选）：展示 softmax 加权的计算步骤，先不用 RMSNorm 简化（用原始内积），让读者先理解 attention 检索的本质；S5 再加入 RMSNorm
  - 补充推导折叠块（softmax 对大值的敏感性）：服务 S5 的伏笔
- **前置知识安排**：softmax 与 RMSNorm 在页面内给最小说明（不递归生成概念页）。
- **完成检查**：
  1. 写出 Full AttnRes 的权重公式，说出 $q_l$、$k_i$、$v_i$、$\alpha_{i\to l}$ 各代表什么。
  2. 说出 pseudo-query $q_l=w_l$ 与标准 attention 中 $Q=W_Q x$ 的关键区别。
  3. 在 $d=2$、3 个候选的小例子里手算 softmax 权重（先不算 RMSNorm）。
- **过渡**：Full AttnRes 公式成立，但它的开销是 $O(Ld)$ 内存——93 层的 K3 用不起。下一章看 Block AttnRes 如何分块、把内存降到 $O(Nd)$。

### S3：Block AttnRes 的分块与块间 attention——把内存从 O(Ld) 降到 O(Nd)

- **主要教学问题**：Block AttnRes 如何分块、块内求和、块间 attention？内存从 $O(Ld)$ 降到 $O(Nd)$ 的来源是什么？
- **对应范围**：Q3（C4、C5、F3、F4）
- **正文要点**：
  1. 说明 Full AttnRes 的开销：$O(L^2 d)$ 算力（$L<100$ 可承受）+ $O(Ld)$ 内存（含跨 pipeline stage 通信），后者是实际瓶颈。
  2. 给出 Block AttnRes 的分块：$L$ 层分 $N$ 个 block、每个 $S=L/N$ 层。
  3. 块内求和：block $n$ 内的层输出求和成单表征 $b_n=\sum_{j\in B_n}f_j(h_j)$；$b_0=h_1$ 为 embedding；定义 partial sum $b_n^i=\sum_{j\in B_n,\,j\le i}f_j(h_j)$。
  4. 候选集合（Eq.10）：对 block $n$ 的第 $i$ 层，候选 = $[b_0,\dots,b_{n-1}]$（$i=1$）或 $[b_0,\dots,b_{n-1},b_n^{i-1}]$（$i\ge 2$）。强调候选数随 block index 增长，最大为 $N+1$。
  5. attention 仍按 Eq.(8)(9) 计算，但作用在 block 级表征上。
  6. 内存对比：Full AttnRes 保留 $L$ 个层输出（$O(Ld)$），Block AttnRes 只保留 $N$ 个 block 级表征（$O(Nd)$）；跨 stage 通信量同比例下降。
  7. 贯穿例子升级：把 S2 的 $N=3$、$S=2$、$d=2$ 例子从 Full AttnRes 改为 Block AttnRes——对 block 3 的第 2 层，候选 = $[b_0, b_1, b_2, b_3^1]$ = 4 个，手算加权检索。
- **讲解材料及职责**：
  - 公式 F3、F4：表达块内求和与候选集合
  - ASCII 图示（Full vs Block AttnRes 的候选集合对比）：服务"从 L 个候选降到 N+1 个候选"的可视化
  - 数字例子折叠块（Block AttnRes 版本，$N=3$、$S=2$、$d=2$、对 block 3 第 2 层）：展示块内求和与块间 attention 的完整计算
  - 伪代码折叠块（Block AttnRes 的前向计算）：服务"输入、状态、核心步骤、输出"的完整呈现
- **前置知识安排**：S2 的 Full AttnRes 公式。
- **完成检查**：
  1. 说出 Block AttnRes 的分块方式与块内求和 $b_n$ 的定义。
  2. 对 block $n$ 的第 $i$ 层，写出候选集合的两种情况（$i=1$ 与 $i\ge 2$）。
  3. 解释内存从 $O(Ld)$ 降到 $O(Nd)$ 的来源。
  4. 在 $N=3$、$S=2$、$d=2$ 的小例子里手算 block 3 第 2 层的候选集合（4 个候选）。
- **过渡**：Block AttnRes 的机制清楚后，下一章看 K3 具体怎么用——8 块、12 层、9 个候选来源。

### S4：K3 的具体配置——8 块×12 层、9 个候选、加权三次

- **主要教学问题**：K3 的具体配置是什么——8 块×12 层、9 个候选来源、加权三次的位置在哪里？
- **对应范围**：Q4（C6、C7、C8、C9、N1、N3）
- **正文要点**：
  1. K3 主干 93 层（69 KDA + 24 MLA），按 `attn_res_block_size=12` 分为 8 个 block：前 7 个各 12 层、第 8 个 9 层（partial block）；加上 embedding 共 9 个 block 级表征。
  2. 9 个候选来源的具体构成：$b_0$（embedding）+ $b_1\dots b_7$（7 个完整 block 的快照）+ $b_8^{i-1}$（当前 partial block 的 partial sum）= 9 个。强调"9"是最后一个 block 内 $i\ge 2$ 层的最大候选数；其他 block 的层候选数更少。
  3. K3 加权三次的位置：每个 attention 子层前一次（用 `self_attention_res_norm/proj` 参数）、每个 MLP（LatentMoE）子层前一次（用 `mlp_res_norm/proj` 参数）、模型末尾 final norm 前一次（output AttnRes，用 `output_attn_res_norm/proj` 参数）。明确标注：K3 报告原文确认 "each module"（即 attention 模块与 MLP 模块各一次）+ 末尾聚合（C7）；具体三次位置来自 `wiki/kimi-k3-dataflow/` 对官方源码的核对（间接证据）。
  4. 对照表（Full AttnRes vs Block AttnRes vs K3 实例化）：让读者看清三个层次的关系。
  5. 引用 K3 报告 §2.2 经验结论：$N\approx 8$ 在多数模型尺度下恢复大部分收益（标注为间接证据，原 preprint 未获取）。
- **讲解材料及职责**：
  - 对照表（Full vs Block vs K3 实例化）：服务三个层次的对比
  - ASCII 图示（K3 的 8 个 block + embedding = 9 个候选）：服务"9 个候选来源"的可视化
  - 配置数值表（来自 `config.json`）：服务事实核对
- **前置知识安排**：S3 的 Block AttnRes 机制。
- **完成检查**：
  1. 说出 K3 主干层数、block 数、block size、最后一个 block 的层数。
  2. 说出 K3 加权三次的具体位置，并标注哪些来自 K3 报告原文、哪些来自源码核对。
  3. 解释"9 个候选来源"的构成，说出它对应哪个 block 的哪一层。
- **过渡**：公式与配置都清楚后，最后一章看一个容易被忽略的细节——softmax kernel 里 RMSNorm 的作用。

### S5：softmax kernel 中的 RMSNorm——为什么不能直接用内积

- **主要教学问题**：softmax kernel 为什么用 $\exp(q^\top\mathrm{RMSNorm}(k))$？RMSNorm 防大值主导的具体含义和边界？
- **对应范围**：Q5（C3、F2、F5）
- **正文要点**：
  1. 复现 S2 中暂时跳过的 RMSNorm：现在解释为什么需要它。
  2. 不加 RMSNorm 的问题：若直接用 $\exp(q^\top k)$，幅值大的层（如某个 $f_i(h_i)$ 模长很大）会让 $q^\top k_i$ 远大于其他项，softmax 后权重几乎集中在该层，等于"屏蔽"了其他来源。
  3. RMSNorm 的作用：把每个 key 归一化到单位 RMS 后再内积，让 pseudo-query 按方向而非幅值选择来源，权重更平滑。
  4. 边界：RMSNorm 不改变 key 的方向，只去掉幅值；若所有 key 方向接近，softmax 仍会接近均匀；RMSNorm 也不保证 exp 不溢出，工程实现通常配合减最大值。
  5. 用 S2 的小例子加 RMSNorm 重算一次：对比"不加 RMSNorm（权重集中）"与"加 RMSNorm（权重平滑）"两种结果。
- **讲解材料及职责**：
  - 公式 F2、F5：表达 kernel 与 RMSNorm
  - 数字例子折叠块（同一例子在"不加 RMSNorm"与"加 RMSNorm"两种情况下的对比）：服务"RMSNorm 改变权重分布"的可观察证据
  - 补充推导折叠块（softmax 对大值的敏感性）：服务"为什么大 key 会主导"的数学解释
- **前置知识安排**：S2 的 softmax kernel 公式与 RMSNorm 最小说明。
- **完成检查**：
  1. 说出不加 RMSNorm 时大 key 为什么会主导 softmax。
  2. 说出 RMSNorm 改变了 key 的什么、不改变什么。
  3. 说出 RMSNorm 的两个失效边界（所有 key 方向接近时 softmax 仍接近均匀；RMSNorm 不防 exp 溢出）。
- **过渡**：本章是最后一个学习目标。文末给出完整来源与教学说明。

## 3. 讲解顺序

S1 → S2 → S3 → S4 → S5。理由：
- S1 给动机，必须最先。
- S2 给 Full AttnRes 公式，是 S3 的 Block AttnRes 的基础。
- S3 在 S2 基础上分块，是 S4 的 K3 实例化的基础。
- S4 把抽象公式落到 K3 具体配置。
- S5 回到公式细节（RMSNorm），需要 S2 的公式与 S3 的小例子作为对照。

S5 放最后而非紧跟 S2 的原因：S5 的边界讨论需要读者先理解 Block AttnRes 的整体机制（否则"大 key 主导"的例子不直观）；且 S5 是细节性目标，不应打断 S2→S3→S4 的主线。

## 4. 贯穿例子

### 主例子：$N=3$、$S=2$、$d=2$ 的小块网络

- **首次出现**：S1 末尾引入，作为"用一个小例子贯穿后续章节"的预告。
- **输入与变量**（S1 末尾完整说明）：
  - 6 层网络（$L=6$），分 3 个 block（$N=3$），每个 block 2 层（$S=2$）。
  - hidden 维度 $d=2$（便于手算）。
  - embedding $h_1\in\mathbb{R}^2$。
  - 每层输出 $f_l(h_l)\in\mathbb{R}^2$，$l\in\{1,\dots,6\}$。
  - 教学示例数字（在 S2 首次使用时给出）：$h_1=[1,0]$、$f_1(h_1)=[0,1]$、$f_2(h_2)=[1,1]$、$f_3(h_3)=[0.5,0.5]$、$f_4(h_4)=[1,0.5]$、$f_5(h_5)=[0.5,1]$、$f_6(h_6)=[1,1]$（人为构造，便于手算）。
  - 最后一层 pseudo-query $q_6=[0.5,0.5]$（教学示例）。
- **S2 推进**（Full AttnRes 版本）：对第 6 层，候选 = $[h_1, f_1, f_2, f_3, f_4, f_5]$ = 6 个；先不加 RMSNorm，用 $q_6^\top k_i$ 算内积，softmax 得权重，加权求和得 $h_6$。
- **S3 推进**（Block AttnRes 版本）：把 6 层分 3 块——block 1 = layer 1-2、block 2 = layer 3-4、block 3 = layer 5-6；$b_0=h_1=[1,0]$、$b_1=f_1+f_2=[1,2]$、$b_2=f_3+f_4=[1.5,1]$；对 block 3 第 2 层（即 layer 6），候选 = $[b_0, b_1, b_2, b_3^1]$ = $[[1,0],[1,2],[1.5,1],[1.5,1]]$（$b_3^1=f_5=[0.5,1]$）= 4 个候选；手算加权检索。
- **S4 推进**：把 $N=3$、$S=2$ 类比到 K3 的 $N=8$、$S=12$：K3 最后一个 block 的第 2 层有 9 个候选（$b_0$ + $b_1\dots b_7$ + $b_8^1$），与本例的 4 个候选结构一致。
- **S5 推进**：把 S2 的小例子加 RMSNorm 重算：对 6 个候选分别 RMSNorm 后再与 $q_6$ 内积，对比权重分布。

### 局部例子

- **S1 的 RNN 瓶颈类比**：服务"所有历史压成单一流"的直观理解，不复用主例子的数字。
- **S4 的 K3 配置表**：服务"9 个候选来源"的工程实例化，复用主例子的结构（block 级表征 + partial sum）但用 K3 真实数字。

## 5. 讲解材料职责

| 材料 | 类型 | 服务教学问题 | 出现位置 |
|---|---|---|---|
| 标准残差 vs AttnRes 对照表 | 对照表 | S1：等权累加 → softmax 加权 | S1 |
| 深度方向 RNN 瓶颈 ASCII 图示 | ASCII 图示 | S1：所有历史压成单一流 | S1 |
| Full AttnRes 公式 Eq.(8)(9) | 公式 | S2：核心定义 | S2 |
| RMSNorm 公式 F5 | 公式 | S2、S5：最小说明 | S2 |
| Full AttnRes 小例子（$d=2$、6 候选，不加 RMSNorm） | 数字例子折叠块 | S2：softmax 加权的手算 | S2 |
| softmax 大值敏感性推导 | 补充推导折叠块 | S5 伏笔 | S2 末尾 |
| Block AttnRes 公式 Eq.(10) | 公式 | S3：块内求和与候选集合 | S3 |
| Full vs Block 候选集合 ASCII 图示 | ASCII 图示 | S3：从 L 个候选到 N+1 个候选 | S3 |
| Block AttnRes 小例子（$N=3$、$S=2$、$d=2$、block 3 第 2 层） | 数字例子折叠块 | S3：块内求和与块间 attention 的手算 | S3 |
| Block AttnRes 前向伪代码 | 伪代码折叠块 | S3：完整算法步骤 | S3 |
| K3 配置表（来自 config.json） | 对照表 | S4：事实核对 | S4 |
| Full vs Block vs K3 实例化对照表 | 对照表 | S4：三个层次的关系 | S4 |
| K3 的 9 个候选来源 ASCII 图示 | ASCII 图示 | S4：9 个候选的可视化 | S4 |
| 同一例子"不加 RMSNorm"vs"加 RMSNorm"对比 | 数字例子折叠块 | S5：RMSNorm 改变权重分布 | S5 |
| softmax 大值敏感性补充推导 | 补充推导折叠块 | S5：为什么大 key 主导 | S5 |

无图片组件（本页所有结构都用 ASCII 图示表达，无需外部图片）。无可运行代码组件（Block AttnRes 的实现涉及大量张量操作，教学代码会隐藏核心机制或过于冗长，伪代码已足够；不强行加可运行代码）。

## 6. 正文与折叠块分工

### 必须放正文

- S1：标准残差的等权累加性质（引用前置页）、RNN 瓶颈类比、AttnRes 的核心思路
- S2：Full AttnRes 公式 Eq.(8)(9)、pseudo-query/keys/values/softmax kernel 的符号定义、RMSNorm 最小说明、边界检查（key 方向相同/不同时的权重分布）
- S3：Block AttnRes 的分块、块内求和 $b_n$、候选集合 Eq.(10)、内存从 $O(Ld)$ 降到 $O(Nd)$ 的来源
- S4：K3 的 $N=8$、$S=12$、9 个候选来源、加权三次的位置（含间接证据标注）
- S5：RMSNorm 防大值主导的机制、两个失效边界
- 贯穿例子的关键推进（S2 的 6 候选不加 RMSNorm 手算结果、S3 的 4 候选手算结果、S5 的加 RMSNorm 对比结果）

### 可放折叠块

- S2：Full AttnRes 小例子的完整计算过程（数字例子折叠块）
- S2 末尾：softmax 大值敏感性的补充推导（推导折叠块，作为 S5 伏笔）
- S3：Block AttnRes 小例子的完整计算过程（数字例子折叠块）
- S3：Block AttnRes 前向伪代码（伪代码折叠块）
- S5：同一例子加/不加 RMSNorm 的对比完整计算（数字例子折叠块）
- S5：softmax 大值敏感性的补充推导（推导折叠块）

折叠块全部收起时，正文仍须回答全部 5 个学习目标：
- Q1：S1 正文回答。
- Q2：S2 正文给出公式与符号定义，正文给出小例子的计算结果（不放折叠块）。
- Q3：S3 正文给出分块、块内求和、候选集合、内存对比，正文给出小例子的计算结果。
- Q4：S4 正文给出 K3 配置与加权三次位置。
- Q5：S5 正文给出 RMSNorm 机制与两个失效边界，正文给出加/不加 RMSNorm 的对比结论。

## 7. 范围与证据约束

本大纲只使用 scope.md §1.1-1.6 中已纳入范围的内容。发现以下情况返回重新规划：
- 需要新增 K3 之外的其他模型实例（如 nano-kpu）作为正文例子 → 不纳入，已在 scope.md §1.1 排除。
- 需要展开 Block AttnRes kernel 的两阶段 schedule → 不纳入，已在 scope.md §1.1 排除。
- 需要补充 AttnRes 原 preprint [57] 的完整消融实验 → 证据不足，已在 evidence.md 标注。

无新增事实需要扩大概念边界。大纲成立。

## 8. 术语表预登记

见 glossary.md。本大纲使用的所有术语与符号在 glossary.md 中首次登记，保证全文一致。
