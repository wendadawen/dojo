<!-- review-meta
round: 5
page: wiki/stable-latent-moe/index.html
reviewed_content_sha256: 232651a01d6441b1
-->
# Stable LatentMoE 审查记录（第 5 轮）

- 页面版本：edc259d3e555352786be89cc35ed932480f4ef33
- 审查时间：2026-09-13 20:24
- 审查者：独立子代理
- 已完整阅读章节：1. 扩大专家池的代价——为什么需要 LatentMoE；2. Stable LatentMoE 的层结构（含数据流 SVG 与两表）；3. 极端稀疏下的两个失败模式（3.1 失败模式一、3.2 失败模式二，含矩阵链 SVG）；4. 三件稳定化（4.1 Normalized LatentMoE、4.2 SiTU-GLU、4.3 Quantile Balancing、4.4 对应关系表）；5. 为什么 RMSNorm 必须插在路由聚合之后（5.1-5.4，含构造示例折叠块）；6. 适用边界与不能推出的结论；来源与范围说明；全部「核心问题」与「本章问题」折叠块。

## 来源核对（本轮逐条回源，均给出原文片段）

- Eq. 11：报告 §2.3 原文 "u= Σ_{i∈Tk(x)} pi Eirouted(W↓ x)；y= Σ_{j=1}^{Ns} Ejshared(x) + W↑ RMSNorm(u)"，页面 Eq. 11 逐字一致。
- Eq. 12（SiTU-GLU）：报告 §2.3.2 原文 "SiTU-GLU(x) = β1 tanh(Wg x/β1) ⊙ Sigmoid(Wg x) ⊙ β2 tanh(Wu x/β2)"，页面一致；上界报告 Eq. 19 "∥SiTU-GLU(x)∥∞ ≤ β1 β2 = 100"，页面取 4×25=100 复算正确。
- Eq. 14（QB）：报告 §2.3.3 原文 "bb(t+1)_j ← − quantile_{1−k/n}(s_{:,j}(t) − α(t))；b(t+1) ← bb(t+1) − mean(bb(t+1))1"，页面一致；"第二行去掉不影响 Top-k 的公共偏移" 对应原文 "removes a common offset that leaves Top-k selection unchanged"；q=mk/n 对应原文 "the target load is q := mk/n tokens per expert"。
- 数字：稀疏度 56 = 报告原文 "896 routed experts with 16 active experts per token, corresponding to a sparsity of 56"；近四个矩阵乘法 = "a chain of nearly four consecutive matrix multiplications"；近 10^3 专家 = "balancing the load of nearly 10^3 experts exceeds the regime"；2.8 万亿 = 报告 Table 1 "Total Parameters … 2.78T" 与 README "2.8T-parameter model"。
- config.json（HF moonshotai/Kimi-K3 原文拉取）：hidden_size=7168、routed_expert_hidden_size=3584、moe_intermediate_size=3072、num_experts=896、num_experts_per_token=16、num_shared_experts=2、latent_moe_use_norm=true、activation_situ_beta=4.0、activation_situ_linear_beta=25.0、dtype=bfloat16——页面符号表逐项一致。共享专家中间维度 6144：config 无独立字段，页面以「moe_intermediate_size 与 num_shared_experts 之积（3072×2）」显式标注为派生值，与 DeepSeek 实现约定一致，复算无误。
- 归因：无辅助损失路由 [30] = DeepSeek-V3 报告（arXiv 2412.19437），报告 [30] 条目核对一致；shared+routed 组织 [23] = DeepSeekMoE（arXiv 2401.06066），该文 §3.1 为 Fine-Grained Expert Segmentation、§3.2 为 Shared Expert Isolation，页面标注 §3.2 正确。
- 构造示例：u_A=(1,1) RMS=1、u_B=(10,10) RMS=10，相距 10 倍；归一化后均为 (1,1)。逐式复算正确，示例已标注为人为构造、不代表真实训练幅度。
- 机械项：validate.py 返回 "validation ok"；本地链接 moe-serving / deepseek-moe / situ-glu / quantile-balancing / latent-moe 均指向真实页面；overview.html 与 index.html 互链；正文与来源说明无 research/ 或不存在的文件路径；无未定义的 C/N/F 引用；数学符号全部由 KaTeX 渲染，正文无 Unicode 数学字符。

## 问题

- [轻微·表述] 引言段与「来源与范围说明」全节：以「本页」「本文」为主语的自我指代。引言 "本页只讲 Stable LatentMoE 的层结构与三件稳定化的插入位置…"；来源说明 10 处 "本文…"（如 "本文只引用上界结论""本文不补""本文基于 Eq. 11 与 RMSNorm 定义作出的推导"）。｜引文依据：不适用｜修复要求：删除页面自我指代主语，改为不带主语的直接陈述。例如 "本页只讲 Stable LatentMoE 的层结构与三件稳定化的插入位置" 改为 "讨论范围限于 Stable LatentMoE 的层结构与三件稳定化的插入位置"；来源说明的 "本文只引用上界结论" 改为 "只引用上界结论"、"本文不补" 改为 "此处不补"、"本文基于 Eq. 11 与 RMSNorm 定义作出的推导" 改为 "基于 Eq. 11 与 RMSNorm 定义作出的推导"。改后全页不再出现以 "本页/本文/本节" 为主语的句子。｜修复：｜复验：
- [轻微·表述] 各章末衔接句与临场提示为元话语。1-5 节末共 5 处 "下一章讲…"（如 "下一章讲 Stable LatentMoE 层的完整结构"）、5 处 "本章…"（如 "本章说明了 LatentMoE 分离两个宽度的动机与代价"）；另 "容易看错的地方有三处"、"这里只需要知道一个结论"、"…等推导不在此展开"。｜引文依据：不适用｜修复要求：保留章间衔接的同时去掉对页面自身结构的指称——把 "本章说明了 X。但 Y 尚未展开，下一章讲 Z" 改写为直接给出下一节结论或问题的内容式过渡（如 "宽度分离解决了流量问题，但多出的两次投影使路由分支变成一条更长的矩阵乘法链"），并将 "容易看错的地方有三处" 改为 "三处常见误读："、"这里只需要知道一个结论" 改为 "两个线性因子…"。改后全页不再出现 "本章/下一章/本页" 作主语的过渡句。｜修复：｜复验：
- [轻微·技术] 5.4 第一条标题与其正文自相矛盾：标题写 "放在 $W_\downarrow$ 之前（归一化 $x$ 或 $z$）"，但 $z = W_\downarrow x$ 位于 $W_\downarrow$ 之后，且正文只论证归一化 $x$ 一种情形、未处理归一化 $z$。｜引文依据：报告 §2.3 定义 "the routed path projects it to z = W↓ x ∈ Rℓ"（即 z 在 W↓ 之后）。｜修复要求：将该条标题改为准确涵盖正文两类情形的位置描述，例如 "放在分支点之前（归一化 $x$）或路由支内、专家之前"，并删除标题中未被论证的 "$z$"，或在正文补上对归一化 $z$（专家输入）情形的同样论证。修改后标题列举的位置与正文论证逐一对应。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复
- 说明：本轮未发现事实性、公式或数字问题；Eq. 11/12/14、四组 config 数字、稀疏度 56、上界 100、2.8T 参数、DeepSeekMoE §3.2 与 DeepSeek-V3 归因均已逐条回源并给出原文片段。三条遗留问题均为表述/措辞层面，不影响正确性与主线理解。修复后即可发布。
