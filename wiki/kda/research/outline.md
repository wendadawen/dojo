# outline.md：KDA 教学大纲

## 4.1 页面开头

**钩子**：1M token 的上下文里，softmax 注意力要把每个 token 的 K、V 都存进 cache——1M × 96 头 × 192 维 × 2 (K+V) × 2 字节（BF16）大致是 70+ GB 量级，单卡装不下。Kimi K3 在 93 层里挑了 69 层换成 KDA，把"不断增长的 KV cache"换成"固定大小的矩阵状态 $S \in \mathbb{R}^{128 \times 128}$"，单层状态只占几十 KB。但固定状态会"记不清"——KDA 用 delta rule 的"先擦后写"加每通道独立衰减来补救，并专门改了一处 decay 参数化让它在 BF16 上不溢出、能用 Tensor Core。

**一句话解释**：KDA 是 Kimi K3 的线性注意力变体，把 delta rule 的擦写递归加上每通道遗忘门，并把衰减率限定在有下界的区间，让长序列训练在 BF16 上数值稳定且硬件友好。

**要解决的具体问题**：(1) 长序列下 KV cache 爆炸；(2) vanilla 线性注意力的 key 碰撞与"记不清"；(3) Kimi Linear 的 negative-softplus decay 让 chunkwise 形式里 $1/\Gamma$ 无界，对角 tile 不得不用 position-pair 计算。

**学习承诺**（与 scope.md Q1-Q5 一致）：读完你能回答 KDA 在 K3 里做什么、channel-wise forget gate 怎么改写 delta rule、K3 的 lower-bounded decay 与 Kimi Linear 的区别及原因、full-rank gate 与 chunkwise 形式怎么让递归可训练、KDA 在 K3 中的配置与边界。

**首个具体场景**：一个 4 维 key/value 的 3 步迷你序列，用 $\alpha=(0.5, 0.8, 0.9, 0.3)$ 的教学构造衰减率手算"先衰减、再擦写"。

**与第一章的过渡**：先看 KDA 要替代什么——softmax 的 KV 增长与 vanilla 线性注意力的碰撞。

## 4.2 章节设计

### S1：为什么 K3 需要 KDA——KV cache 爆炸与线性注意力的"记不清"

- **主要教学问题**：K3 把上下文推到 1M，softmax 的 KV cache 装不下；vanilla 线性注意力状态固定但会 key 碰撞。KDA 要同时解决这两个问题。
- **对应范围**：Q1；C1（KDA 角色）、C8（69/93 层）。
- **正文要点**：
  1. softmax 注意力在 1M token 下的 KV cache 量级估算（用 N6 的 1M 与 N5 的 head_dim/num_heads 算一次"教学估算"，明确标为教学估算不是工程基准）。
  2. 引用 [线性注意力](../../wiki/linear-attention/index.html) 的结论：因果掩码下线性注意力变成固定大小递归状态 $s_i = s_{i-1} + \phi(k_i)v_i^\top$，与 $N$ 无关。
  3. 引用 [delta rule](../../wiki/delta-rule/index.html) 的结论：纯加性累加在 $L > d$ 时 key 碰撞不可避免，需要"先擦后写"。
  4. K3 的选择：69 层 KDA（固定状态 + delta rule + 遗忘门）+ 24 层 Gated MLA（全局内容交互），3:1 交替，最后一层是 Gated MLA。
- **讲解材料及职责**：
  - 数字估算（N6 + N5）：展示 1M KV cache 的量级，引出固定状态需求。
  - ASCII 图示：3:1 混合的层布局（69 KDA + 24 MLA，末尾 2 层 MLA）。
  - 前置概念链接：linear-attention 页（递归状态）、delta-rule 页（碰撞与擦写）。
- **前置知识安排**：首次依赖 linear-attention 的"因果递归状态"结论时给链接；首次依赖 delta-rule 的"先擦后写"结论时给链接。
- **完成检查**：(a) 估算 1M token 下单层 softmax KV cache 的字节数量级；(b) 说明 K3 为何不全部用 KDA而要混 MLA；(c) 说出 69/93 这个比例。
- **过渡**：KDA 要替代 softmax 与 vanilla 线性注意力，但它的递归具体长什么样？下一章拆开 Eq.1。

### S2：KDA 的递归核心——delta rule 加 channel-wise forget gate

- **主要教学问题**：KDA 的递归 $S_t = (I - \beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t)S_{t-1} + \beta_t k_t v_t^\top$ 怎么把"先衰减、再擦写"组合起来？
- **对应范围**：Q2；C2（forget gate 在 delta 前）、C10（状态约定转置）。
- **正文要点**：
  1. 状态约定：$S_t \in \mathbb{R}^{d_k \times d_v}$，$\tilde o_t = S_t^\top q_t$。说明这与 delta-rule 概念页的 $S \in \mathbb{R}^{d_v \times d_k}$ 是转置约定，机制等价。
  2. 公式分三步读：(i) $\mathrm{Diag}(\alpha_t) S_{t-1}$ 把旧状态按通道衰减；(ii) $(I - \beta_t k_t k_t^\top)$ 在 $k_t$ 方向擦除；(iii) $+ \beta_t k_t v_t^\top$ 写入新 outer product。
  3. forget gate 在 delta 之前：Eq.1 的乘法顺序决定 $\mathrm{Diag}(\alpha_t)$ 先作用，擦写作用于"已衰减的旧状态"。
  4. $\alpha_t$ 是 channel-wise：每通道独立衰减率，与 $\beta_t$（标量写入强度）职责不同。
  5. 贯穿例子第 1 步：4 维 key/value，$S_0 = 0$，第 1 步写入 $\beta_1 k_1 v_1^\top$（无旧状态可衰减）；第 2 步先按 $\alpha_2 = (0.5, 0.8, 0.9, 0.3)$ 衰减 $S_1$ 再 delta 擦写，手算每个通道的衰减效果。
- **讲解材料及职责**：
  - 公式 F1（Eq.1）：表达三步组合关系。
  - 数字例子（教学构造 $\alpha = (0.5, 0.8, 0.9, 0.3)$）：展示通道级衰减的差异化效果。
  - ASCII 图示：单步三阶段流程（衰减 → 擦除 → 写入）。
- **前置知识安排**：首次依赖 delta-rule 的 $I - \beta_t k_t k_t^\top$ 擦除项时给链接，不重讲几何。
- **完成检查**：(a) 说出 $\mathrm{Diag}(\alpha_t)$ 与 $(I - \beta_t k_t k_t^\top)$ 的作用顺序；(b) 说明 $\alpha_t$ 与 $\beta_t$ 的职责差异；(c) 手算第 2 步的 $S_2$ 第 3 通道（$\alpha=0.9$）。
- **过渡**：$\alpha_t$ 是 channel-wise，但它的值从哪来？K3 怎么保证它不溢出？下一章讲 decay 的参数化。

### S3：K3 的关键改动——lower-bounded decay

- **主要教学问题**：K3 为什么把 Kimi Linear 的 negative-softplus 换成 scaled sigmoid？这个改动怎么让对角 tile 也能用 Tensor Core？
- **对应范围**：Q3；C3（scaled sigmoid）、C4（negative-softplus 对照）、C5（BF16 + Tensor Core 收益）。
- **正文要点**：
  1. decay 的两段映射：先从 $x_t$ 算 decay logit $z_t^h = W_\alpha^{\uparrow\downarrow} x_t + b_h^\alpha$（Eq.2），再从 $z_t^h$ 算 log-decay $g_t^h$。
  2. Kimi Linear 的映射：$g = -e^{A_h}\mathrm{Softplus}(z) \in (-\infty, 0)$（C4），$\alpha = e^g \in (0, 1)$ 但可任意接近 0。
  3. K3 的映射：$g = g_{\min}\mathrm{Sigmoid}(e^{A_h} z) \in (g_{\min}, 0) = (-5, 0)$（C3、F5），$\alpha = e^g \in (e^{-5}, 1) \approx (0.0067, 1)$（N1、N2）。
  4. 为什么这个下界重要：chunkwise 形式里要把 keys 除以 $\Gamma_{1\to C}$（见 S5），$\Gamma$ 是 $\alpha$ 的乘积。$\alpha$ 无下界时 $1/\Gamma$ 无界，BF16 会溢出；$\alpha > e^{-5}$ 时 16-token tile 的 $1/\Gamma < e^{80}$（N3），BF16 能表示。
  5. 工程收益：对角 tile 不再需要 position-pair 计算，全部 dense matmul，用 Tensor Core（C5、Fig.3b）。
  6. 贯穿例子第 2 步：给定 $z_t^h = 1$、$A_h = 0$，对比 Kimi Linear（$g = -\mathrm{Softplus}(1) \approx -1.313$，$\alpha \approx 0.269$）与 K3（$g = -5 \cdot \mathrm{Sigmoid}(1) \approx -5 \times 0.731 = -3.655$，$\alpha \approx 0.0259$）。
- **讲解材料及职责**：
  - 公式 F5（Eq.5）：表达两种映射的对比。
  - 数字例子（$z=1, A=0$ 手算）：展示 sigmoid 有下界、softplus 无下界的差异。
  - 对照表格：Kimi Linear vs K3 在 $g$ 范围、$\alpha$ 范围、$1/\Gamma$ 上界、对角 tile 实现四列对比。
- **前置知识安排**：sigmoid/softplus 用一句话点名，不展开。
- **完成检查**：(a) 写出两种映射的 $g$ 取值范围；(b) 算出 $z=1, A=0$ 时两者的 $\alpha$；(c) 说明 $g_{\min}=-5$ 为何让对角 tile 能用 Tensor Core。
- **过渡**：decay 解决了数值稳定性，但 KDA 还要能训练——下一章讲参数化与 output gate。

### S4：参数化与 full-rank output gate——把递归包成可训练的一层

- **主要教学问题**：q/k/v/β/z 怎么从 $x_t$ 算出来？full-rank gate 相对 low-rank 改了什么？
- **对应范围**：Q4；F2（参数化）、F6（full-rank gate）、C6（K3 改）。
- **正文要点**：
  1. 参数化链（Eq.2）：$x_t \to$ ShortConv $\to$ Swish $\to$ L2Norm（仅 q/k）$\to$ 投影得到 $q_t, k_t, v_t, \beta_t, z_t$。
  2. ShortConv 的职责：在投影前做短卷积（kernel=4，N5），让 $k_t$ 带一点局部时序信息。
  3. L2Norm 的职责：让 $q_t, k_t$ 近单位范数，使 $I - \beta_t k_t k_t^\top$ 接近正交投影（与 delta-rule 页的 $\|k\|=1$ 结论呼应）。
  4. $\beta_t = \mathrm{Sigmoid}(W_\beta x_t)$：数据相关的写入强度。
  5. full-rank output gate（Eq.6）：$y = W_o[\mathrm{Sigmoid}(W_g x) \odot \mathrm{RMSNorm}(\tilde o)]$。$W_g$ 满秩（C6），与 Kimi Linear 的 low-rank 对照。
  6. RMSNorm 在门控前：稳定 $\tilde o$ 的幅度，让 sigmoid 门控只调制通道而非放大数值。
- **讲解材料及职责**：
  - 公式 F2（Eq.2）：表达参数化链。
  - 公式 F6（Eq.6）：表达 full-rank gate。
  - ASCII 图示：参数化链的数据流。
- **前置知识安排**：RMSNorm/Swish 用一句话点名。
- **完成检查**：(a) 说出 q/k 比 v 多了哪两步；(b) 说明 $W_g$ 满秩的含义；(c) 解释 RMSNorm 为何在门控前。
- **过渡**：单步递归可训练了，但串行跑 1M 步太慢——下一章讲 chunkwise 并行。

### S5：chunkwise 并行形式——chunk 内并行 + chunk 间递归

- **主要教学问题**：怎么把串行递归变成可在 GPU 上并行的 matmul？lower-bound 在这里具体产生收益？
- **对应范围**：Q4；C7（chunkwise）、F3（Γ）、F4（Eq.4）、C5（对角 tile 收益的具体位置）。
- **正文要点**：
  1. 切 chunk：序列分成长度 $C$ 的 chunk（Kimi Linear 用 $C=16$ 的 secondary tile，N3）。
  2. 累积衰减 $\Gamma_{1\to r} = \prod_{r'=1}^r \alpha_{r'}$（Eq.3，F3）。
  3. UT 变换得 $V_e[t] = U[t] - W[t]S[t]$（来自 [63]，不展开推导）。
  4. chunk 内并行（Eq.4，F4）：$A[t] = \mathrm{Tril}((Q[t]\odot\Gamma)(K[t]/\Gamma)^\top)$，$O[t] = (\Gamma\odot Q[t])S[t] + A[t]V_e[t]$。
  5. 两项分工：第一项 $(\Gamma\odot Q)S[t]$ 是 inter-chunk（前序 chunk 状态），第二项 $A[t]V_e[t]$ 是 intra-chunk。
  6. Tril 保留对角：每个 output 读 current-token update 后的状态（Eq.4 正文）。
  7. lower-bound 在这里收益：$K[t]/\Gamma$ 要除以 $\Gamma$，$1/\Gamma$ 无界时对角 tile 溢出。lower-bound 后 $1/\Gamma < e^{80}$，对角 tile 也 dense matmul（C5）。
- **讲解材料及职责**：
  - 公式 F3（Eq.3）+ F4（Eq.4）：表达累积衰减与 chunk 内并行。
  - ASCII 图示：chunk 切分 + inter/intra 两项的数据流。
  - 对照表格：lower-bound 前后对角 tile 的实现差异。
- **前置知识安排**：引用 linear-attention 页的"核函数结合律"结论，说明 chunkwise 是其并行化形式。
- **完成检查**：(a) 说出 inter-chunk 与 intra-chunk 两项分别是什么；(b) 说明 Tril 保留对角的原因；(c) 指出 $K/\Gamma$ 这一步为何是 lower-bound 的收益点。
- **过渡**：机制讲完，最后落到 K3 的具体数值。

### S6：KDA 在 K3 中的配置与边界

- **主要教学问题**：KDA 在 K3 里具体用了多少层、什么参数？它不解决什么？
- **对应范围**：Q5；C8（69+24）、C9（数值）、误解 4（KDA ≠ K3 全部注意力）。
- **正文要点**：
  1. 层布局：69 KDA + 24 Gated MLA = 93 层，3:1 混合，末尾 2 层均为 Gated MLA（C8、N4）。
  2. 数值表：$g_{\min}=-5$、head_dim=128、num_heads=96、short_conv=4、hidden_size=7168、max_position=1M（C9、N5、N6）。
  3. KDA 不解决：全局内容交互（MLA 负责）、跨设备状态同步（KCP 负责）、kernel 调度（FlashKDA 负责）。
  4. 适用边界：$g_{\min}=-5$、$C=16$、BF16 是 lower-bound 收益的前提；FP32 训练时 negative-softplus 也能用但成本更高。
- **讲解材料及职责**：
  - 对照表格：K3 配置汇总。
  - ASCII 图示：3:1 层布局（与 S1 的图呼应，加上具体编号 1-93）。
- **完成检查**：(a) 说出 69/24/93 三个数的关系；(b) 列出 $g_{\min}$、head_dim、num_heads 三个数值；(c) 说出 KDA 不负责的一项任务。
- **过渡**：文末"来源与教学说明"。

## 4.3 讲解顺序

S1（动机）→ S2（递归核心，引入 $\alpha_t$）→ S3（decay 参数化，引入 $g$）→ S4（参数化与 gate，引入 $W_g$ 满秩）→ S5（并行形式，引入 $\Gamma$）→ S6（配置与边界）。

一次只引入一个新变量：S2 引入 $\alpha_t$，S3 引入 $g_t$ 与 $A_h$，S4 引入 $W_g$ 与 RMSNorm，S5 引入 $\Gamma$ 与 $V_e$。每个变量首次出现时已具备解释它的全部前置。

## 4.4 贯穿例子

**主题**：4 维 key/value 的 3 步迷你序列。

**第 1 次出场（S2）**：
- 输入：$S_0 = 0 \in \mathbb{R}^{4 \times 4}$；$k_1 = (1, 0, 0, 0)^\top$，$v_1 = (2, 2, 2, 2)^\top$，$\beta_1 = 1$。
- 第 1 步：$S_1 = \beta_1 k_1 v_1^\top = \begin{pmatrix} 2&2&2&2\\0&0&0&0\\0&0&0&0\\0&0&0&0 \end{pmatrix}$（无旧状态可衰减）。
- 第 2 步：$k_2 = (0, 1, 0, 0)^\top$，$v_2 = (3, 3, 3, 3)^\top$，$\beta_2 = 1$，$\alpha_2 = (0.5, 0.8, 0.9, 0.3)$（教学构造）。
- 衰减：$\mathrm{Diag}(\alpha_2) S_1 = \begin{pmatrix} 1&0&0&0\\0&0.8&0&0\\0&0&0.9&0\\0&0&0&0.3 \end{pmatrix} \begin{pmatrix} 2&2&2&2\\0&0&0&0\\0&0&0&0\\0&0&0&0 \end{pmatrix} = \begin{pmatrix} 2&2&2&2\\0&0&0&0\\0&0&0&0\\0&0&0&0 \end{pmatrix}$（$S_1$ 只有第一行非零，$\alpha_{2,1}=0.5$ 没作用到非零行——这里需要修正构造）。

> **构造修正**：为了让 $\alpha$ 的通道差异可见，把 $k_1$ 改为 $(1,1,1,1)^\top / 2$（单位范数），$v_1 = (2,2,2,2)^\top$，则 $S_1 = k_1 v_1^\top = \frac{1}{2}\begin{pmatrix}2&2&2&2\\2&2&2&2\\2&2&2&2\\2&2&2&2\end{pmatrix} = \begin{pmatrix}1&1&1&1\\1&1&1&1\\1&1&1&1\\1&1&1&1\end{pmatrix}$。第 2 步衰减后每行按 $\alpha_{2,j}$ 缩放：$\mathrm{Diag}(\alpha_2) S_1 = \begin{pmatrix}0.5&0.5&0.5&0.5\\0.8&0.8&0.8&0.8\\0.9&0.9&0.9&0.9\\0.3&0.3&0.3&0.3\end{pmatrix}$。通道 3 衰减最慢（0.9 留存），通道 4 衰减最快（0.3 留存）。

**第 2 次出场（S3）**：同一个 $\alpha_2$ 的第 3 通道值 0.9，反推 $g$ 与 $z$：$\alpha = 0.9 \Rightarrow g = \ln 0.9 \approx -0.105$。若 $A_h=0$、$g_{\min}=-5$，则 $g = -5\mathrm{Sigmoid}(z) \Rightarrow \mathrm{Sigmoid}(z) = 0.021 \Rightarrow z \approx -3.84$。对照 Kimi Linear：$g = -\mathrm{Softplus}(z) = -0.105 \Rightarrow \mathrm{Softplus}(z) = 0.105 \Rightarrow z \approx -2.25$（因为 $\mathrm{Softplus}(z) = \ln(1+e^z)$）。展示两种映射都能产出 $\alpha=0.9$，但 $z$ 的取值不同，且 K3 的 $\alpha$ 有下界 $e^{-5} \approx 0.0067$，Kimi Linear 无下界。

**第 3 次出场（S3 末）**：取 $z=1, A=0$ 的极端值对比（已在要点 6 写明）。

**第 4 次出场（S5）**：把第 1-3 步当作一个 $C=3$ 的 chunk，手算 $\Gamma_{1\to 1}, \Gamma_{1\to 2}, \Gamma_{1\to 3}$，展示累积衰减的乘法性质。

贯穿例子全部用教学构造数字，明确标注。

## 4.5 讲解材料职责

| 材料 | 服务的教学问题 | 出现位置 |
|---|---|---|
| 公式 F1（Eq.1） | 表达三步组合的递归关系 | S2 |
| 公式 F2（Eq.2） | 表达参数化链 | S4 |
| 公式 F3（Eq.3）+ F4（Eq.4） | 表达累积衰减与 chunk 内并行 | S5 |
| 公式 F5（Eq.5） | 表达两种 decay 映射的对比 | S3 |
| 公式 F6（Eq.6） | 表达 full-rank gate | S4 |
| 数字例子（$\alpha=(0.5,0.8,0.9,0.3)$） | 展示通道级衰减差异化 | S2 |
| 数字例子（$z=1, A=0$ 对比） | 展示 sigmoid 有下界、softplus 无下界 | S3 |
| 数字例子（$C=3$ chunk 手算 $\Gamma$） | 展示累积衰减的乘法性质 | S5 |
| ASCII 图示（3:1 层布局） | 展示 69/24 的层编号 | S1, S6 |
| ASCII 图示（单步三阶段流程） | 展示衰减→擦除→写入顺序 | S2 |
| ASCII 图示（参数化链数据流） | 展示 $x_t$ 到 $q,k,v,\beta,z$ 的路径 | S4 |
| ASCII 图示（chunk 切分 + inter/intra） | 展示 chunkwise 形式的两项分工 | S5 |
| 对照表格（Kimi Linear vs K3） | 对比 $g$ 范围、$\alpha$ 范围、$1/\Gamma$ 上界、对角 tile 实现 | S3, S6 |

无伪代码与可运行代码——KDA 是模型层机制，不是独立算法，伪代码会与 Eq.1/2/4 重复；可运行代码需要完整训练框架，超出教学职责。S5 的 chunkwise 形式本身已是并行算法描述，不再额外伪代码化。

## 4.6 正文与折叠块分工

### 必须放正文

- KDA 在 K3 的角色与 69/93 比例（S1）
- Eq.1 三步分解与 forget gate 在 delta 前的顺序（S2）
- 贯穿例子的第 1、2 步关键结果（S2）
- 两种 decay 映射的公式与范围对比（S3）
- $g_{\min}=-5$、$\alpha > e^{-5}$、$1/\Gamma < e^{80}$ 三个数值（S3）
- 参数化链与 full-rank gate 公式（S4）
- chunkwise 形式的两项分工与 Tril 保留对角（S5）
- 69/24/93 配置与 KDA 不解决的三件事（S6）
- 所有前置概念页链接（S1、S2）

### 可放折叠块

- S2 第 2 步的完整手算过程（衰减 → 擦除 → 写入的逐元素矩阵运算）
- S3 的 $\mathrm{Softplus}(z) = \ln(1+e^z)$ 反推 $z$ 的完整推导
- S3 $z=1, A=0$ 时两种映射的完整数值对比
- S5 的 $C=3$ chunk 手算 $\Gamma_{1\to 1}, \Gamma_{1\to 2}, \Gamma_{1\to 3}$ 完整过程
- S5 lower-bound 前后对角 tile 实现差异的详细描述
- S1 的 1M KV cache 字节数估算过程

折叠块全部收起时，正文仍能回答 Q1-Q5 全部学习目标。

## 4.7 范围与证据约束

大纲全部内容来自 scope.md 已纳入范围。无新增学习目标、无新增核心内容、无排除项被纳入。所有事实附 C/F/N 编号，与 evidence.md 一致。
