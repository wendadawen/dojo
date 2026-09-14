<!-- review-meta
round: 7
page: wiki/stable-latent-moe/index.html
reviewed_content_sha256: 25ea42f024eabbd7
-->
# Stable LatentMoE 审查记录（第 7 轮）

- 页面版本：a3a6b864c9eb3ec65307f5f9a334259c23a4e718（wiki/stable-latent-moe/index.html 工作树）
- 审查时间：2026-09-14 17:43
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 页面类型：concept（依据 head 的 `dojo:type=concept`，适用 `guides/concept/check.md`）
- 已完整阅读章节（按顺序）：核心问题（页面级，5 条）；1. 扩大专家池的代价——为什么需要 LatentMoE；2. Stable LatentMoE 的层结构（含内联 SVG 数据流图与两张对照表）；3. 极端稀疏下的两个失败模式（3.1 失败模式一 + 矩阵乘法链 SVG、3.2 失败模式二 + 总览表）；4. 三件稳定化（4.1–4.4）；5. 为什么 RMSNorm 必须插在路由聚合之后（5.1–5.4，含「展开：构造两个不同 $T_k$ 与 $p_i$ 的 $u$」折叠块）；6. 适用边界与不能推出的结论；来源与范围说明（论断 C / 公式 F / 外部数字 N / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）；各章「本章问题」折叠块；overview.html。

## 来源核对（本轮实际打开的版本与引文依据）

- **K3 技术报告**：arXiv:2607.24653v2（HTML 全文），逐节比对 §2.3、§2.3.1、§2.3.2、§2.3.3 与 Eq. 11–14。
  - Eq. 11：原文 "$y = \sum_{j=1}^{N_s} E_j^{shared}(x) + W_\uparrow \cdot RMSNorm(u)$"，"$u = \sum_{i\in T_k(x)} p_i E_i^{routed}(W_\downarrow x)$" —— 与正文 L196、核心问题 2 一致。
  - SiTU-GLU：原文 "SiTU-GLU(x) = [β₁·tanh(W_g x / β₁) ⊙ Sigmoid(W_g x)] ⊙ [β₂·tanh(W_u x / β₂)]"，"β₁ = 4 ... β₂ = 25 ... |f(x)| ≤ β₁β₂ = 100" —— 与 L428、L430 一致。
  - QB：原文 "$b̂_j^{(t+1)} ← −quantile_{1−k/n}(s_{:,j} − α^{(t)})$；$b^{(t+1)} ← b̂^{(t+1)} − mean(b̂^{(t+1)})·1$"，"target is q := mk/n"，"Routing runs Top-(k+1) on the biased score" —— 与 L436、L438 一致。
  - "nearly four consecutive matrix multiplications"；"That ill-conditioned structure, combined with the 2.8-trillion-parameter scale, produces exploding internal activations in the routed branch"；"896 routed experts with 16 active experts per token, corresponding to a sparsity of 56"；"the additional RMSNorm consistently improves validation loss and downstream benchmarks"；"the count of full-width shared experts is fixed at two (N_s = 2) in every layer" —— 分别与 L336、L149、L285/L332、L489、L192 一致。
- **官方配置**：https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json（2026-09-14 抓取原文件核对）。`text_config` 内 hidden_size=7168、routed_expert_hidden_size=3584、moe_intermediate_size=3072、num_experts=896、num_experts_per_token=16、num_shared_experts=2、latent_moe_use_norm=true、dtype=bfloat16、activation_situ_beta=4.0、activation_situ_linear_beta=25.0；`text_config.quantization_config` 为 mxfp4-pack-quantized、num_bits=4、group_size=32 —— 与 §2 配置表、N1–N8、§6 全部一致，`config 字段` 一列写的路径名也正确。原文件确无「共享专家中间维度」字段，页面对 6144 标注「推断」并写明「不应作为官方配置引用」，处理正确。
- **DeepSeek-V3 技术报告**：arXiv:2412.19437v2（HTML）§2.1.2。原文 "decrease the bias term by γ if its corresponding expert is overloaded, and increase it by γ if its corresponding expert is underloaded"、"the bias term is only used for routing"、"the gating value ... is still derived from the original affinity score" —— 支持 L382「bias 不进入 mixture 权重」与无辅助损失路由归属；摘要 "pioneers an auxiliary-loss-free strategy for load balancing" 支持 L382/L95 的「由 DeepSeek-V3 引入」。
- **DeepSeekMoE 论文**：arXiv:2401.06066，§3.2 为 Shared Expert Isolation；原文 "further isolate K_s experts to serve as shared experts" —— 支持 L192 的 shared+routed 归属。该节未给出「共享专家中间维度 = 路由专家中间维度 × 共享专家数」，页面把该推导规则写成「按 DeepSeek 共享专家约定」略欠精确，但 6144 已明确标为推断并禁止当作官方配置引用，未升级为问题。
- **构造示例算术复算**（L516–L523）：$u_A=(1,1)$、$u_B=(10,10)$；$\mathrm{RMS}$ 分别为 1、10；归一化后均为 $(1,1)$；$W_\uparrow u_B$ 尺度为 $W_\uparrow u_A$ 的 10 倍 —— 全部可复算，且已声明为构造示例、不代表 K3 数值范围。
- **图核对**（两张内联 SVG）：数学标签全部在 `<foreignObject>` 内由 KaTeX 渲染，`<text>` 内只有标题与步骤号 (1)(2)(3)(4)，无 ASCII 近似写法；图内 $d=7168$、$\ell=3584$、Top-$k=16$、$N_s=2$ 与正文及 config 一致；两张图在明暗主题下用 CSS 变量着色，无硬编码颜色。
- **机械项**：`python3 .dojo/scripts/validate.py wiki/stable-latent-moe/index.html` → `validation ok`；全部前置概念链接（moe-serving、deepseek-moe、situ-glu、quantile-balancing、latent-moe）对应目录下 index.html 均存在，无「（待生成）」占位；alt/aria-label 内无 `$...$`；无声称可运行的代码块（无须执行）。

## 问题

- [轻微·表述] L163 / L303 / L396 / L455 / L527（五处章节末过渡句）：五段套用同一固定句式「⟨本章结论⟩已经⟨动词⟩。⟨下章问题⟩——⟨指示下一章⟩」，其中 L163 与 L527 结尾完全相同（"这是下一章的内容"）。｜引文依据：style-guide.md §8「章节顺序存在依赖时，用一至两句说明前一节结论与下一节问题的关系。不使用固定句式，也不为形式完整而添加过渡」｜修复要求：至少改写其中三处，去掉「已经+动词」+「是下一章的内容／留给下一章展开」的模板化结构，改为直接点出下一章要解决的具体量（例如下一章要给的量名或公式项）。｜修复：｜复验：
- [轻微·技术] L390（3.2 两个失败模式总览表「原因」列）：写成「连续 4 次矩阵乘法 + 2.8T 规模」，与正文 L336/L404 及 3.1 图的「近四个连续矩阵乘法（报告原文 nearly four）」不一致，同一事实在本页出现两种说法。｜引文依据：K3 报告 §2.3 "nearly four consecutive matrix multiplications"｜修复要求：该单元格改为「近四次连续矩阵乘法 + 2.8T 规模」，与正文用词一致。｜修复：｜复验：
- [轻微·格式] L601–L610、L616（来源与范围说明）：论断编号缺 C4–C6，公式编号缺 F2，列表从 C3 直接跳到 C7、从 F1 跳到 F3。全文未出现 C4/C5/C6/F2 的引用，故无悬空引用，但编号不连续会让逐条核对者以为漏列条目。｜引文依据：不适用｜修复要求：或补齐为连续编号，或在「论断与来源（C）」「公式与来源（F）」处注明跳号原因（例如已并入相邻条目）。｜修复：｜复验：
- [轻微·格式] L147（$z = W_\downarrow x$ 公式）、L428（SiTU-GLU 公式）、L436（QB 更新公式）：公式后均直接接段落，未按规范以 `<ul>` 逐项定义符号；同页 Eq. 11（L196）后则有完整符号 `<ul>`，页内不一致。｜引文依据：style-guide.md §11「公式后紧跟 `<ul>` 逐项定义每个符号」｜修复要求：三处公式后各补一个 `<ul>`，逐项列出该公式新出现的符号（分别为 $W_\downarrow$、$z$；$\beta_1$、$\beta_2$、$W_g$、$W_u$；$s_{:,j}$、$\alpha$、$m$、$n$、$q$）。｜修复：｜复验：
- [轻微·表述] L496（5.4 第二条「放在 $W_\uparrow$ 之后」）与 L128（核心问题 4 答案）：用「$p_i$ 大的 token 与 $p_i$ 小的 token」说明被抹掉的信息，并称放在 $W_\uparrow$ 之后会「抹掉路由权重 $p_i$ 传递的相对大小信息」。但 $p_i$ 是逐 token 归一化的路由权重（$\sum_{i \in T_k(x)} p_i = 1$），token 之间不存在全局「$p_i$ 大／小」；且按本页 5.3 的说明与构造示例（L521–L523 两个 token 归一化后完全一致），放在聚合之后的 RMSNorm 同样抹掉 $u$ 的整体尺度，该条因此未能说明两处归一化的差别究竟在哪里。｜引文依据：K3 报告 Eq. 13 "p_{i,j} = s_{i,j} / Σ_{r∈T_i} s_{i,r}"（逐 token 归一）；本页 L521–L523 构造示例｜修复要求：把该条改为陈述两处归一化的实际差别——放在 $W_\uparrow$ 之后时归一化常数取 $\mathrm{RMS}(W_\uparrow u)$，路由支输出被强制为固定尺度；K3 的位置取 $\mathrm{RMS}(u)$，路由支输出尺度仍随 $u$ 的方向变化。删去「$p_i$ 大的 token 与 $p_i$ 小的 token」的表述，并同步修改 L128 核心问题 4 答案的对应措辞。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：修复（仅轻微项）。本轮逐条回源核对后，Eq. 11/12/14、SiTU-GLU 上界 100、QB 分位数更新与去均值、稀疏度 56、896/top-16/2 共享、`latent_moe_use_norm=true`、dtype 与 MXFP4 量化、2.8T 规模、"nearly four" 与 ill-conditioned 归因，均与 arXiv:2607.24653v2 与官方 config.json 一致；构造示例可复算；图内数值、页面链接、公式渲染均正常。无阻断、无重要问题，轻微项修复后可发布。
