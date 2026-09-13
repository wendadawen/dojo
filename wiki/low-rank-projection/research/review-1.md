<!-- review-meta
round: 1
page: wiki/low-rank-projection/index.html
reviewed_content_sha256: d2e7e83d1f1581bd
-->
# 低秩分解审查记录（第 1 轮）

- 页面版本：8bc60e6a3b9fac922bcfe9f0aaaf5d0dcf395f0b
- 审查时间：2026-09-13 18:48
- 审查者：独立子代理（未参与写作、未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题 → 1. 大矩阵为什么贵——参数与存储的成本 → 2. 矩阵的秩——能被压缩多少由什么决定 → 3. SVD 与最优低秩近似——误差由奇异值决定 → 4. LoRA——把权重更新参数化为低秩 → 5. MLA——把 KV 压进低维潜向量 → 6. 适用边界——低秩不是万能的 → 来源与范围说明（含全部 details 折叠块、flow-diagram 图注、表格）
- 核对来源：arXiv:2106.09685v2（LoRA，全文 §1/§2/§4/Figure 1）、arXiv:2405.04434v2（DeepSeek-V2，摘要、§2.1.2、§2.1.4 Table 1、附录表目录）、Wikipedia "Singular value decomposition" 与 "Low-rank approximation" 原文、Kimi Linear 技术报告 arXiv:2510.26692v2、以及本仓 wiki/kda 页已核对记录
- 机械验证：`python3 .dojo/scripts/validate.py wiki/low-rank-projection/index.html` → `validation ok`；被引前置概念页 `wiki/kda`、`wiki/mla`、`wiki/standard-attention` 均真实存在；`overview.html` 与 `index.html` 双向链接；页面无「（待生成）」占位；无第一人称复数与第二人称（grep 无「我们/你」）

## 问题

- [阻断·技术] §5 正文（443 行）＋来源 [N4]（557 行）＋[F6]（551 行）：KV cache 对照数据 110.6K→15.6K、860.2K→34.6K 被标注为「§2.1.4 Table 7」。§2.1.4 中的表实际是 Table 1（KV cache 每 token 公式对照表，不含这些数值）；论文的 Table 7 位于附录 B.2（DeepSeek-V2-Lite Chat 评测），与该数据无关。按页面标注的位置无法定位该论断。｜引文依据：Table 1 标题 "Comparison of the KV cache per token among different attention mechanisms"；arXiv HTML 表目录中 Table 7 标题为 "Performance of DeepSeek-V2-Lite Chat, DeepSeekMoE 16B Chat, and DeepSeek 7B Chat"；这些数值实际出现在附录 D.2 的 MLA/MHA 对照（第三方精读："论文附录 D.2 还比较了 MLA 和 MHA，在两个 MoE 规模上训练并评估"，数值 110.6K/15.6K、860.2K/34.6K）。｜修复要求：把引注改为数据实际所在位置（附录 D.2 的 MLA/MHA 对照表，核实准确表号并在 [F6]/[N4] 中统一），或删除该段对照与其数字。｜修复：｜复验：

- [重要·技术] §5 第 443 行：把 110.6K→15.6K、860.2K→34.6K 描述为「DeepSeek-V2 MoE 与不同 MHA 基线（含 DeepSeek 67B 等）的整体对照，基线的 $n_h, d_h, d_c$、层数与模型规模均不同」。该对照是同一 MoE 规模下 MHA 与 MLA 的消融，规模基本一致，也不含 DeepSeek 67B。｜引文依据：附录 D.2 对照表：Small MoE MHA 110.6K / MLA 15.6K，总参 15.8B / 15.7B；Large MoE MHA 860.2K / MLA 34.6K，总参 250.8B / 247.4B。｜修复要求：改为「同一 MoE 规模下 MHA 与 MLA 的对照」；删去「含 DeepSeek 67B 等」与「模型规模均不同」；93.3% 仍单独归属与 DeepSeek 67B 的对照。｜修复：｜复验：

- [重要·技术] §6（500、502、504 行）：把推断写成来源结论。「DeepSeek-V2 选 $d_c=512$ 是在压缩率和表达力之间权衡的结果」与「K3 把它改为 full-rank，正是因为低秩门控的表达力不足——每个输出通道没有独立的门控信号」均以断言给出，但所引来源未陈述动机：[C3] 只说明 MLA 做 KV 低秩联合压缩，[C4] 只给出配置标志 `use_full_rank_gate = true`。且低秩 $W_g$ 仍为每个输出通道输出门控值（取值被限制在 rank 维子空间），「每个输出通道没有独立的门控信号」不准确。｜引文依据：[C3] 引文 "The core of MLA is the low-rank joint compression for keys and values to reduce KV cache"；[C4] 仅有 `use_full_rank_gate = true`；Kimi Linear 报告 §4 关于输出门为 "the output gate adopts a low-rank parameterization similar to the forget gate"（该报告未出现 `use_full_rank_gate` 或 K3 字样）。｜修复要求：把 $d_c=512$ 与 full-rank 改动的动机降级为明确标注的推断，或删除动机断言；把「每个输出通道没有独立的门控信号」改为「门控取值被限制在低维子空间，表达力受限」。｜修复：｜复验：

- [轻微·技术] §3 展开块（298 行）：例中 $\sqrt{2.9^2+2.8^2}\approx4.04$ 有舍入错误。｜引文依据：$\sqrt{2.9^2+2.8^2}=\sqrt{8.41+7.84}=\sqrt{16.25}=4.0311$。｜修复要求：改为 $\approx 4.03$。｜修复：｜复验：

- [轻微·表述] 元话语：374 行「这里要区分一个容易混淆的点」、457 行「这里要强调一点」、443 行「注意上面的 1/64 是…」、535 行「回顾全文：」。｜引文依据：不适用。｜修复要求：去掉宣告写作动作的框架，改为直接陈述（如「LoRA 近似的是更新量 $\Delta W$，不是 $W_0$」）。｜修复：｜复验：

- [轻微·表述] 临场评价：227 行「这就是低秩分解的威力。」、504 行「一个来自 K3 的真实教训」。｜引文依据：不适用。｜修复要求：改为中性陈述句。｜修复：｜复验：

- [轻微·技术] 符号一致性：§2（203–205 行）以 $W=AB$ 定义分解（$A$ 为左因子 $m\times r$），§4（334 行）LoRA 用 $\Delta W=BA$（$B$ 为左因子 $d\times r$），同一符号 $A,B$ 的左右角色在两章互换，页面未说明对应关系。｜引文依据：203 行「$A \in \mathbb{R}^{m \times r}$、$B \in \mathbb{R}^{r \times n}$」；334 行「$B \in \mathbb{R}^{d \times r}, \ A \in \mathbb{R}^{r \times d}$」。｜修复要求：在 §4 首次出现处加一句说明两种写法的对应（或全文统一记号）。｜修复：｜复验：

- [轻微·技术] 第 160 行「每个元素都是一个要存储和训练的参数」表述不准（元素被存储，参数被训练），且引注 [F5]（参数计数 $mn\to r(m+n)$）与该句不贴合。｜引文依据：[F5]「参数计数 $mn \to r(m+n)$」。｜修复要求：改写为「矩阵的 $mn$ 个元素都是参数、都要存储」，并按需调整引注。｜修复：｜复验：

- [轻微·表述] 表格表头「典型场景」（492 行）把「场景」当术语。｜引文依据：不适用。｜修复要求：改为「典型例子」或具体对象名。｜修复：｜复验：

- [轻微·格式] 正文/summary/图示中直接出现 Unicode 数学符号，未由 KaTeX 渲染：443 行「110.6K→15.6K」、290 行 summary「3×3」、215/348 行 flow-box「×」、421/428 行 flow-box「→」。style-guide §11 要求正文、summary、表格等位置的数学符号一律写成 LaTeX。validate.py 通过，属边缘项。｜引文依据：不适用。｜修复要求：正文改为文字表述（「110.6K 降至 15.6K」）；图示连接符保留或用文字标签。｜修复：｜复验：

- [轻微·技术] 条件缺失：223 行与 500 行首次陈述「rank 可低至 1 或 2 就足够」未带实验条件。[N2] 引文明确限定 GPT-3 175B。｜引文依据：[N2] 引文 "a very low rank (i.e., r in Figure 1 can be one or two) suffices even when the full rank (i.e., d) is as high as 12,288"（GPT-3 175B 语境）。｜修复要求：首次出现处补「（GPT-3 175B 上）」。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 8
- 处置：修复

## 核对留痕（关键来源原文）

- LoRA §1："We take inspiration from Li et al. 2018a; Aghajanyan et al. 2020 which show that the learned over-parametrized models in fact reside on a low intrinsic dimension. We hypothesize that the change in weights during model adaptation also has a low 'intrinsic rank'"（支持 [C1]）
- LoRA §2："the number of trainable parameters |Θ| can be as small as 0.01% of |Φ₀|"；Abstract："reduce the number of trainable parameters by 10,000 times"（支持 [N1]，正文归为 §2 与 §1 均可定位）
- LoRA §4.1 Eq.(3)："h=W₀x+ΔWx=W₀x+BAx"；"We use a random Gaussian initialization for A and zero for B, so ΔW=BA is zero at the beginning of training."；Figure 1 注 "We only train A and B."（支持 [F1]）
- DeepSeek-V2 §2.1.2："The core of MLA is the low-rank joint compression for keys and values to reduce KV cache"（支持 [C3]）；Eq.(9)(10)(11)：$c_t^{KV}=W^{DKV}h_t$、$k_t^C=W^{UK}c_t^{KV}$、$v_t^C=W^{UV}c_t^{KV}$（支持 [F4]）
- DeepSeek-V2 §2.1.4 Table 1：MHA $2n_h d_h l$、MLA $(d_c+d_h^R)l$（支持 [F6]）；§3.1.2："we set the number of attention heads $n_h$ to 128 and the per-head dimension $d_h$ to 128. The KV compression dimension $d_c$ is set to 512"（支持 [N3]）；摘要 "reduces the KV cache by 93.3%"（相比 DeepSeek 67B，支持 [N4] 前半）
- Wikipedia "Singular value decomposition"：$M=U\Sigma V^*$ 定义；"$\tilde M$ is the best approximation of $M$ by any matrix of rank less than or equal to $t$, under the Frobenius norm"；"This is known as the Eckart–Young theorem, as it was proved by those two authors in 1936."（支持 [C2]/[F2]）
- Wikipedia "Low-rank approximation"：$\|D-\hat D^*\|_F=\sqrt{\sigma_{k+1}^2+\cdots+\sigma_r^2}$、$\|A-A_k\|_2=\sigma_{k+1}$（支持 [F3]）；"originally solved by Erhard Schmidt … later rediscovered by C. Eckart and G. Young. L. Mirsky generalized the result to arbitrary unitarily invariant norms"（支持 [C2] 历史沿革，惟「Mirsky 1960」年份不在该文，属常识性补充）
- 数值复算全部通过：$12288^2=150{,}994{,}944$；$2\times12288\times2=49152$；$49152/150994944=0.0326\%$；$2\times128\times128=32768$；$512/32768=1/64$；$\sqrt{1.25}=1.118$；$\sqrt{4.01}=2.002$；（唯一例外见「轻微·技术」第 1 条 $\sqrt{16.25}=4.03$）
