<!-- review-meta
round: 4
page: wiki/stable-latent-moe/index.html
reviewed_content_sha256: 3e950807eebcb005
-->
# Stable LatentMoE 审查记录（第 4 轮）

- 页面版本：c8c737f3fcc345a5de4a92c0d5134547158b6d09
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题（5 条含解答）→ 1. 扩大专家池的代价 → 2. Stable LatentMoE 的层结构（含表格、SVG 数据流图、折叠块）→ 3. 极端稀疏下的两个失败模式（3.1、3.2、SVG 链图）→ 4. 三件稳定化（4.1-4.4）→ 5. 为什么 RMSNorm 必须插在路由聚合之后（5.1-5.4、构造示例折叠块）→ 6. 适用边界与不能推出的结论 → 来源与范围说明。另读 overview.html。

## 来源核对依据（本轮已逐条打开来源，记录关键片段）

- K3 技术报告（arXiv:2607.24653v2）§2.3 原文：`the routed path composes W↓, a gated multi-branch expert feed-forward network, and W↑ into a chain of nearly four consecutive matrix multiplications`；`ill-conditioned structure, combined with the 2.8-trillion-parameter scale, produces exploding internal activations`；`balancing the load of nearly 10^3 experts exceeds the regime in which existing auxiliary-loss-free bias updates remain well behaved`；`896 routed experts with 16 active experts per token, corresponding to a sparsity of 56`。Eq. 11 原文 `u=∑_{i∈T_k(x)} p_i E_i^routed(W↓x)`、`y=∑_{j=1}^{N_s} E_j^shared(x)+W↑ RMSNorm(u)`，与页面逐符号一致。`Kimi K3 fixes the number of full-width shared experts to N_s=2 in every layer`。
- §2.3.1：`The original LatentMoE directly applies W↑ to the aggregated routed representation u`；`Kimi K3 instead inserts RMSNorm between expert aggregation and the up-projection`；`Beyond stabilizing training, the additional RMSNorm consistently improves validation loss and downstream benchmarks`。
- §2.3.2 Eq. 12 原文 `[β1 tanh(W_g x/β1) ⊙ Sigmoid(W_g x)] ⊙ [β2 tanh(W_u x/β2)]`、`β1=4 for the gate branch and β2=25 for the up branch`、`|f(x)|≤β1β2=100`，与页面一致。
- §2.3.3 Eq. 14 原文 `b̂_j^(t+1) ← −quantile_{1−k/n}(s_{:,j}−α^(t))`、`b^(t+1) ← b̂^(t+1) − mean(b̂^(t+1))·1`；`the target load is q:=mk/n tokens per expert`；`the second line removes a common offset that leaves Top-k selection unchanged`；固定步长规则 `b_j^(t+1)=b_j^(t)+γ sign(ℓ̄−ℓ_j^(t))` [27]。参考文献 [27]=DeepSeek-v3 technical report（2412.19437），[31]=LatentMoE，[22]=DeepSeekMoE（2401.06066）。
- Hugging Face `moonshotai/Kimi-K3` config.json（raw 拉取）：`hidden_size=7168`、`routed_expert_hidden_size=3584`、`moe_intermediate_size=3072`、`num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`、`latent_moe_use_norm=true`、`activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`、`dtype=bfloat16`。
- 官方 `modeling_kimi_linear.py`：`intermediate_size = config.moe_intermediate_size * config.num_shared_experts`（共享专家宽度 6144=3072×2，支持页面表格行）；`if self.latent_moe_use_norm: y = self.routed_expert_norm(y)` 后接 `routed_expert_up_proj`（核实 RMSNorm 插在聚合后、上投影前）。
- DeepSeekMoE（arXiv:2401.06066）§3.2「Shared Expert Isolation」原文：`isolate certain experts to serve as shared experts that are always activated`；`Regardless of the router module, each token will be deterministically assigned to these shared experts`。
- 机械项：`python3 .dojo/scripts/validate.py wiki/stable-latent-moe/index.html` → `validation ok`；公式定界符外无 Unicode 数学字符；`dojo:topics=模型结构`、`dojo:tag=MoE` 均在词表内；`../moe-serving/`、`../deepseek-moe/`、`../situ-glu/`、`../quantile-balancing/`、`../latent-moe/` 目标页均存在；index.html 与 overview.html 互相链接；无 `research/` 残留引用。构造示例复算通过：u_A=(1,1)、u_B=(10,10)，RMS 分别 1 与 10（√((1+1)/2)=1、√((100+100)/2)=10），归一化后同为 (1,1)；SiTU-GLU 上界 4×25=100；稀疏度 896/16=56。

## 问题

- [重要·技术] 「3.2 失败模式二」：同一符号 ℓ 在本页承担两个含义——§1 起 ℓ 表示隐空间宽度（ℓ=3584），§3.2 又用 $\bar{\ell}$、$\ell_j$ 表示专家负载，且 $\ell_j$ 在全页未定义。违反「符号全文单义」。｜引文依据：页面 §1「$d$ 与路由专家宽度 $\ell$」、§2 符号表「$\ell$ 路由隐空间宽度 3584」；§3.2 公式 `$b_j^{(t+1)} = b_j^{(t)} + \gamma \cdot \mathrm{sign}(\bar{\ell} - \ell_j)$` 全文无一处说明此处 ℓ 是负载（K3 原文该符号即负载 `sign(ℓ̄−ℓ_j^(t))`）。｜修复要求：将负载变量改为不与宽度冲突的符号（如 $L_j$、$\bar{L}$）并在首次出现处定义，或明确写出「此处 $\ell$ 借自来源、表示专家负载，区别于前述隐空间宽度 ℓ」；全文保持单一含义。｜修复：｜复验：

- [轻微·技术] 「3.2 失败模式二」与「5.3」：γ 同时表示 bias 更新步长（§3.2）与 RMSNorm 可学习缩放参数（§5.3 构造示例简化条件及来源章）。同一变量两名。｜引文依据：§3.2「步长 $\gamma$ 太小则适应慢」；§5.3「省略 RMSNorm 的可学习缩放参数 $\gamma$」。｜修复要求：将两处之一改用其他符号（如 RMSNorm 缩放参数写 $\gamma_{\text{norm}}$ 或直接写 $\gamma$ 仅在 RMSNorm 语境定义），或在首次使用处显式区分。｜修复：｜复验：

- [轻微·技术] 「3.1 失败模式一」：分解项数与括号说明不自洽。正文列举「$W_\downarrow$ 一次、路由 FFN 内部门支与值支各一次、$W_\uparrow$ 一次」共 4 项并称「近四个连续矩阵乘法」，但括号又写「路由 FFN 内部另有 down projection 未单列」——若再计入一次矩阵乘法应为 5 项，与「近四个」冲突。｜引文依据：K3 §2.3 `a chain of nearly four consecutive matrix multiplications`，源仅列 `W↓、a gated multi-branch expert feed-forward network、W↑` 三项。｜修复要求：删除该括号，或改写为与「nearly four」一致的表述（如说明来源的 nearly four 为近似计数，不额外声称存在未计入的 down projection）。｜修复：｜复验：

- [轻微·技术] 「5.4 换位置会怎样」第一条：`$u$ 的尺度由 $T_k(x)$ 与 $p_i$ 决定，与 $x$ 的初始尺度无关` 表述不精确。由 Eq. 11，$u=\sum_{i\in T_k(x)} p_i E_i^{\text{routed}}(W_\downarrow x)$ 仍依赖 $x$ 的尺度（$x$ 缩放会经 $W_\downarrow x$ 传递到 $u$）。｜引文依据：页面 Eq. 11；K3 §2.3 Eq. 11。｜修复要求：改为只陈述本段真正需要的理由——归一化输入尺度无法消除聚合阶段（$T_k(x)$ 与 $p_i$ 变化）带来的逐 token 漂移；不要断言 $u$ 与 $x$ 初始尺度无关。｜修复：｜复验：

- [轻微·表述] 「1. 扩大专家池的代价」末段：`这一点在极端稀疏下的两个失败模式一章会看到它是激活爆炸的根源` 属读者导向的前向叙述（元话语）。｜引文依据：不适用。｜修复要求：改为平实的跨章指代，如「这一点是后文失败模式一的根源（见「3.1 失败模式一」）」，去掉「会看到」。｜修复：｜复验：

- [轻微·表述] 「6. 适用边界与不能推出的结论」结论段：`综上，Stable LatentMoE ...` 的「综上」为公文连接词。｜引文依据：不适用。｜修复要求：删除「综上」直接起句，或改为实质性收束表述。｜修复：｜复验：

- [轻微·技术] 「6. 适用边界」与来源章：`bfloat16 训练` 被列为成立条件。所引来源（K3 §2.3 与 config.json）仅给出模型权重 `dtype: "bfloat16"`，报告 §2.3 未直接声明训练精度。｜引文依据：config.json `"dtype": "bfloat16"`；K3 §2.3 无 bfloat16 训练声明（报告仅在 §2.3.2 提及 `low-precision arithmetic`）。｜修复要求：改为「权重配置为 bfloat16（config.json `dtype`）」，不写成来源直接给出的训练精度条件；或补一条来源支持。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 6
- 处置：修复（关闭 1 项重要问题后发布；6 项轻微随本轮一并处理或记录接受理由）
- 说明：页面所有事实性论断、公式（Eq. 11、12、14）与关键数字（7168/3584/3072/6144/896/16/2/56/2.8T/100）均已回源核对一致；构造示例可复算且与来源结论相符；来源标注（C1-C3、C7-C13、F1/F3/F4、N1-N8）无悬空、无未支持论断、无把推断写成来源结论（§5.4 三条「位置不可换」论据已在来源章明确标注为本文推导）。判为可发布前须消除 ℓ 的双重含义与未定义问题。