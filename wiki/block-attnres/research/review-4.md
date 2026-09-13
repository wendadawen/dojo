<!-- review-meta
round: 4
page: wiki/block-attnres/index.html
reviewed_content_sha256: b3c72634415af871
-->
# Block AttnRes 审查记录（第 4 轮）

- 页面版本：bd83ec7a614d15fdecb0acbb308f2cea5bf0e1cd
- 审查时间：2026-09-13 19:35
- 审查者：独立子代理（未参与写作，未读取本页 research/）
- 已完整阅读章节：核心问题 / 1. 标准残差在深度上的瓶颈——为什么需要 AttnRes / 2. Full AttnRes 的公式——pseudo-query 如何检索前序层 / 3. Block AttnRes 的分块与块间 attention——把内存从 $O(Ld)$ 降到 $O(Nd)$ / 4. K3 的具体配置——8 块×12 层、9 个候选、加权三次 / 5. softmax kernel 中的 RMSNorm——为什么不能直接用内积 / 来源与范围说明
- 外部来源核对方式：arXiv:2607.24653v2《Kimi K3: Open Frontier Intelligence》§2、§2.2、Table 1、§5.2.2、§5.4.2、§7（curl 抓 HTML 全文逐段核对）；huggingface.co/moonshotai/Kimi-K3 的 config.json（逐字段核对）。下列 §2.2 原文均已定位确认：
  - Eq.(8)「$\bm{k}_{i}=\bm{v}_{i}=\begin{cases}\bm{h}_{1}&i=0\\f_{i}(\bm{h}_{i})&1\leq i\leq l-1\end{cases}$」
  - Eq.(9)「$\phi(\bm{q},\bm{k})=\exp(\bm{q}^{\top}\operatorname{RMSNorm}(\bm{k}))$ … the \(\operatorname{RMSNorm}\) prevents layers with large-magnitude outputs from dominating the weights」「$\alpha_{i\to l}=\frac{\phi(\bm{q}_{l},\bm{k}_{i})}{\sum_{j=0}^{l-1}\phi(\bm{q}_{l},\bm{k}_{j})},\bm{h}_{l}=\sum_{i=0}^{l-1}\alpha_{i\to l}\bm{v}_{i}$」
  - 「Since network depth is modest (\(L<100\)), the \(O(L^{2}d)\) arithmetic of this full form is affordable; the practical overhead is the \(O(Ld)\) memory (and cross-stage communication under pipeline parallelism) for keeping all layer outputs alive.」
  - 「$\bm{b}_{n}=\sum_{j\in\mathcal{B}_{n}}f_{j}(\bm{h}_{j})$ … we set \(\bm{b}_{0}=\bm{h}_{1}\) so the token embedding is always included as a source」「The final output layer then aggregates all \(N\) block representations. Under Block AttnRes, memory and communication overhead drop from \(O(Ld)\) to \(O(Nd)\)」
  - 「Empirically, \(N\approx 8\) recovers most of the benefit across model scales [60]; for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer.」
  - Table 1：「# Layers 93」「Attention-Layer Composition 69 KDA + 24 MLA」「Hidden Dimension 7,168」「Attention Heads 96」
  - config.json：attn_res_block_size=12、num_hidden_layers=93、hidden_size=7168、num_attention_heads=96、full_attn_layers 24 个、kda_layers 69 个
  - §5.2.2「the block representation is generated once at the boundary layer」「wrapped with checkpointing」「cache-based pipeline communication」；§5.4.2「two-phase schedule: a batched inter-block pass … online-softmax merge」「sequence parallelism (SP) for activations」「launch the inter-block kernel on a side stream」；§7 Chip design「Block AttnRes with a block size of two」+ 仓库 github.com/MoonshotAI/nano-kpu
- 复算：§2 手算（6 候选，无 RMSNorm）$h_6=[0.703,0.703]$；§3 手算（4 候选）$h_6=[1.059,1.241]$；§5 加 RMSNorm 重算权重 $[0.2058,0.2620,0.2704,0.2620]$、$h_6=[1.004,1.056]$；2 候选/3 候选小例；exp 值、归一化分母、分项之和（权重列和=1.0000）逐项复算，全部与页面标注一致。$\exp(0.5)=1.6487$、$\exp(1.5)=4.4817$、$\exp(1.25)=3.4903$、$\exp(0.75)=2.1170$、$\exp(0.7071)=2.0281$、$\exp(0.9487)=2.5822$、$\exp(0.9806)=2.6658$、$\exp(4)=54.598$、$\exp(2)=7.389$、$\exp(5)=148.41$ 均核对无误。93=7×12+9；69+24=93；O(L²d)/O(Ld)/O(Nd) 原文一致。
- 机械项：validate.py 通过；residual-connection、kimi-k3-dataflow 两个前置链接页均存在；index↔overview 互链；无 research/ 路径引用、无「（待生成）」占位；dojo:type=concept、dojo:topics=注意力机制（在 AGENTS.md 词表内）、dojo:tag=注意力（在 ALLOWED_TAGS 内）、description/dojo:summary 齐备且 summary 含可渲染公式。

## 问题

- [阻断·技术] §2「Full AttnRes 的公式」符号定义（第 283 行，涉第 269 行与第 163 行）：同一变量 $h$ 在同一页被定义为两种互相矛盾的角色。第 283 行符号表写「<b>$h_l$</b>：层 $l$ 的输出（这里指 AttnRes 加权后的结果，会送入后续归一化与子层）」；而同一节第 269 行写「$h_i$ 是第 $i$ 层的输入（进入该层的残差流），$f_i(h_i)$ 是第 $i$ 层的输出」，§1 第 163 行写「把第 $l$ 层的输入 $h_l$ 直接加到该层变换 $f_l$ 的输出上」。$h_l$ 在来源中是层 $l$ 的输入（Eq.(9) 定义的 $h_l$ 正是 Eq.(8) 中 $f_l(h_l)$ 的实参），第 283 行的「输出」是错的，且与其自身括号内「送入后续归一化与子层」自相矛盾。这是本页核心符号的定义冲突，会让读者误以为 $h_l$ 是 $f_l$ 的输出。｜引文依据：报告 Eq.(8)「$\bm{k}_{i}=\bm{v}_{i}=\begin{cases}\bm{h}_{1}&i=0\\f_{i}(\bm{h}_{i})&1\leq i\leq l-1\end{cases}$」——$h_l$ 是 $f_l$ 的自变量即层输入；页面第 163、269 行亦均作「输入」｜修复要求：把第 283 行改为「$h_l$：层 $l$ 的输入（AttnRes 检索结果，替代原残差流后送入该层归一化与子层）」，使 $h$「层输入」的定义在第 163、269、283 行三处一致；若确要表达「AttnRes 模块的输出」，须另用不同符号并全页统一。｜修复：｜复验：

- [重要·技术] §5「两个失效边界」（第 727 行）：减最大值的 $m$ 定义错误。原文「工程实现通常在 softmax 前减去最大值（即 $\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k)-m)$，$m$ 为全部候选 $\phi(q_l,k_j)$ 的最大值）」。$\phi$ 已被本页定义为 $\exp(\cdot)$ 的结果，按此写法得到 $\exp(\text{logit}-\max_j\exp(\text{logit}_j))$，不是标准的 max-subtraction；标准做法是 $m=\max_j\bigl(q_l^\top\mathrm{RMSNorm}(k_j)\bigr)$（对 logit 取最大值）。页面此处把「取最大值的对象」写成了 exp 之后的值。｜引文依据：本页 Eq.(9) 自身定义「$\phi(q_l,k_i)=\exp(q_l^\top\mathrm{RMSNorm}(k_i))$」，故 $\max\phi=\exp(\max\ \text{logit})\neq\max\ \text{logit}$｜修复要求：把 $m$ 改为「$m=\max_j\,q_l^\top\mathrm{RMSNorm}(k_j)$（全部候选 logit 的最大值）」，或删去该括号内的形式化定义、只保留「减最大值」的文字描述。｜修复：｜复验：

- [轻微·表述] 通读全文（含折叠块）残留元话语与临场评价：第 225 行「这里的关键转换是：把…」；第 346 行「下面的补充推导说明这个现象。」；第 686 行「但还有一个容易被忽略的设计选择——下一章讲…」；第 239 行「这个区分是本文最常见的误解之一」（对读者群体作无来源判断）；另有 5 处以祈使句「注意…」直接指导读者（第 269、343、596、777、780 行附近：「注意 $k_i=v_i$」「注意本例没有用 RMSNorm」等）。均非错误，但属规范要排除的元话语／临场评价。｜引文依据：不适用｜修复要求：第 225 行改为直陈「关键转换在于把…」；第 346 行删去「下面的…说明这个现象」改为直接给出推导；第 686 行去掉「容易被忽略的」；第 239 行删去「最常见的误解之一」或改为「这一区分容易被误解」；5 处「注意 X」改为直陈句（如「此处 $k_i=v_i$：key 与 value 同源」）。｜修复：｜复验：

- [轻微·技术] §2 末（第 239 行）：来源角标误挂。句「Full AttnRes 公式成立<sup>[C3]</sup>，但它的开销是 $O(Ld)$ 内存…」把 [C3] 挂在一句关于内存开销的论断上，而 C3 在来源章节定义为「RMSNorm 防大值主导（§2.2 第 390-392 行）」，与该句无关。｜引文依据：本页来源章节「C3（RMSNorm 防大值主导）：K3 报告 §2.2 第 390-392 行」｜修复要求：将此处角标改为与内存/公式对应的来源（[C2] 或 [C5]），或删除该角标。｜修复：｜复验：

- [轻微·技术] §2 首段（第 265 行）：命名动机无来源。句「它叫"pseudo"是因为它不依赖该层输入（与标准 attention 中 $Q=W_Q x$ 不同）」把「pseudo 命名的理由」写成事实，但 K3 报告只给「layer-specific learnable pseudo-query $\bm{q}_l=\bm{w}_l\in\mathbb{R}^d$」，未解释命名缘由。可核对的事实只有「$q_l$ 是层参数、不依赖输入」。｜引文依据：报告 §2.2「we define a layer-specific learnable pseudo-query $\bm{q}_{l}=\bm{w}_{l}\in\mathbb{R}^{d}$」（无命名解释）｜修复要求：删去「它叫"pseudo"是因为」的因果表述，改为「$q_l=w_l$ 是层自带的参数向量，不依赖该层输入（与标准 attention 的 $Q=W_Q x$ 不同）」。｜修复：｜复验：

- [轻微·技术] 来源章节（第 847、849 行）：实验条件标签「K3 Instruct 模型版本」无来源。N1「实验条件：K3 Instruct 模型版本」、N3「实验条件：K3 Instruct 模型版本」。K3 报告与 HuggingFace 模型卡均未使用「K3 Instruct」这一名称（报告通篇称 Kimi K3；模型卡称「open-weight, native multimodal agentic model」，未标注 instruct 变体）。｜引文依据：报告 §2.2「for Kimi K3, we partition its layers into 8 blocks…」；config.json 来自 huggingface.co/moonshotai/Kimi-K3（无 instruct 字段）；模型卡无「instruct」表述｜修复要求：把「K3 Instruct 模型版本」改为有依据的表述（如「官方发布的 Kimi K3 权重（huggingface.co/moonshotai/Kimi-K3）config.json」），或删除该实验条件标签。｜修复：｜复验：

- [轻微·格式] 来源章节（第 877 行「未核对与遗留问题」）：新增了 style-guide 未列出的 h3。`guides/concept/style-guide.md` 第 1 节规定「来源与范围说明」下的 h3 使用固定命名——`论断与来源（C）`、`公式与来源（F）`、`外部数字与实验条件（N）`、`构造示例`、`辅助解释与类比边界`、`简化条件及其限制`；本页多出「未核对与遗留问题」一节（全 wiki 仅此页有此 h3）。其内容（原 preprint [60] 未核对、源码核对为间接证据、未展开的工程优化）本身符合 check.md 对「不确定信息」的要求。｜引文依据：不适用｜修复要求：把该节内容并入本节内既有的固定小节（原 preprint/源码核对归「外部数字与实验条件（N）」，工程优化归「简化条件及其限制」），或按「需要新大类时先改词表」的方式先更新 style-guide 再保留。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 5
- 处置：修复（第 283 行的 $h_l$ 输入/输出定义冲突为阻断项，须与第 163、269 行统一；第 727 行 $m$ 定义须改为对 logit 取最大值。全部数字、公式复算与来源核对一致，五个核心问题均有解答折叠块，章节与来源结构完整，validate.py 通过。）