<!-- review-meta
round: 6
page: wiki/eagle-speculative/index.html
reviewed_content_sha256: 414b5dac66a98a0d
-->
# EAGLE-3 投机解码 draft 模型审查记录（第 6 轮）

- 页面版本：7731ae133a874716377ff7e674899188f5d99751
- 审查时间：2026-09-13 21:07
- 审查者：编排者派发的独立审查者（未参与写作与前序审查）
- 已完整阅读章节：核心问题；1. 为什么独立小模型 draft 有两难——EAGLE 的思路转向；2. EAGLE-1 的核心机制——在 feature 空间做自回归；3. EAGLE-3 的两项架构改变——直接 token 预测 + 多层特征融合；4. 推理时的自回归 draft——单层 decoder 如何生成 γ 个 draft token；5. 训练 draft 模型——training-time test 与接受率损失；6. 工程实例——K3 的 EAGLE-3 部署与边界；来源与范围说明（含全部折叠块与图注）

## 来源核对（本轮逐条回源）

- [C1] EAGLE-1 §1 原文已核（arXiv:2401.15077v3）：「Applying speculative sampling hinges on finding a draft model that mirrors the original LLM's functionality but with reduced latency」「it is not feasible for instruct-tuned models due to the inconsistency in instruction templates between LLaMA2-Chat and TinyLLaMA-Chat」「Despite the 7B model's potential as a draft model, its high overhead diminishes acceleration gains」「TinyLLaMA is trained on 3,000B tokens, whereas EAGLE is trained on 2-4B tokens」——与页面一致。
- [C2] 已核：摘要「Firstly, autoregression at the feature (second-to-top-layer) level is more straightforward than at the token level. Secondly, the inherent uncertainty in feature ... level autoregression constrains its performance」「By incorporating a token sequence advanced by one time step, EAGLE effectively resolves the uncertainty」——一致。
- [C3] EAGLE-3 摘要已核（arXiv:2503.01840v2）：「abandons feature prediction in favor of direct token prediction and replaces reliance on top-layer features with multi-layer feature fusion via a technique named training-time test」——一致。
- [C4] §3.1 已核：「In Step 2, the prefix becomes 'How can I'. ... However, this is not possible because the token 'I' has not yet been checked by the target model, and we cannot obtain g_I. Instead, we use the output a_I from the draft model in the previous step to replace g_I」——引文逐字一致；论文另一处「we input a_I into the LM head and sample to obtain the draft token 'do'」亦与 [F2] 引文一致。
- [C5] §3.2 与 Figure 6 图注已核：「During training, we perform test steps, where we generate a and feed it back into the draft model for further training.」「It sequentially shows a native training step (the first step) and two simulated training steps (the second and third steps).」——一致。
- [C6][F4] K3 报告 §4.1.4 Eq.(16) 已核（arXiv:2607.24653v1）：「Since minimizing the conventional KL-divergence surrogate does not guarantee maximizing this rate for a capacity-limited draft model, we directly optimize the likelihood-based LK loss [103], the negative logarithm of the acceptance rate itself, L_LK = -log Σ_{x∈V} min(p(x), q(x)), (16) with p and q evaluated at temperature 1 and no auxiliary ground-truth cross-entropy term.」——一致。
- [C7][C8][C9][N4] K3 §4.1.4 已核：「mirrors the structure of a backbone block」「comprises a single decoder layer whose structure matches the MTP layer, we fine-tune the pre-trained MTP layer into an EAGLE-3-style draft model, with the target model frozen and only the draft layer and its feature-fusion projection updated」「the draft is unrolled for seven steps」「taken from the outputs of the 1st, 4th, and final AttnRes blocks, respectively (§ 2.2)」「a bias-free matrix W_E3, initialized as [0 0 I] so that the fused representation coincides at initialization with the high-level feature h_h — the input on which the MTP layer was pre-trained」「MoE expert weights in MXFP4 and their input activations in MXFP8, while non-expert modules remain in higher precision」——逐字一致。
- [C10] EAGLE-3 §4 已核：「EAGLE-3 does not modify the target model's weights and uses strict speculative sampling acceptance conditions, ensuring no loss in performance.」；EAGLE-1 摘要「maintaining the distribution of the generated text」；HF 模型卡 README（yuhuili/EAGLE-Qwen2-72B-Instruct）「maintaining lossless performance」——一致。
- [F1][F2] §3.1 已核：「We concatenate the k-dimensional vectors l, m, and h to form a 3k-dimensional vector, then pass it through a fully connected (FC) layer to reduce it to k-dimensions, obtaining a feature g」；「The concatenated vector is then passed through an FC layer to reduce its dimensionality to k, and subsequently inputted into a single layer decoder, producing the output a.」——一致。
- [F5][F6] Leviathan et al. 2023 §3、§3.3 已核（arXiv:2211.17192v2）：Theorem 3.5「β = ... = Σ_x min(p(x), q(x))」；Theorem 3.8「The expected improvement factor in total walltime by Algorithm 1 is (1-α^{γ+1})/((1-α)(γc+1))」；Appendix A.1 标题「Correctness of Speculative Sampling」——一致。
- [N1] EAGLE-1 摘要已核：「For LLaMA2-Chat 70B, EAGLE achieved a latency speedup ratio of 2.7x-3.5x, doubled throughput」；实验设置为 MT-bench、贪心（temperature=0），与 Figure 1 说明一致。
- [N2] EAGLE-3 摘要与 §4 已核：「a speedup ratio up to 6.5x, with about 1.4x improvement over EAGLE-2」「1.38x throughput improvement at a batch size of 64」「evaluated on five tasks」「Chat model's evaluation dataset is MT-bench, and the reasoning model's evaluation dataset is GSM8K」——一致。
- [N3] HF 模型卡 README 已核：「5.6 faster than vanilla decoding (13B). 1.8x faster than EAGLE-1 (13B). Inference is conducted on 2x RTX 3090 GPUs at fp16 precision using the Vicuna 13B model.」「3x faster than vanilla decoding (13B)」；EAGLE-3 Table 1 已核（temperature=0）：V 13B | EAGLE 3.07x、EAGLE-2 4.26x、EAGLE-3 5.58x——与 [N3] 一致。
- [N5] 见 [C1]，一致。
- 构造示例复算（逐行核）：Step 1 W_a·[0.4,0.2,0.1,0.6,1,0,0,0] = [0.08,0.45,0.14,0.13]，tanh ≈ [0.080,0.422,0.139,0.129]，logits [0.160,0.844,0.278,0.259,0.385]，Σe^logits = 7.585，q_1 ≈ [0.155,0.307,0.174,0.171,0.194]，argmax "do"；Step 2 W_a·[0.080,0.422,0.139,0.129,0,1,0,0] = [0.092,0.084,0.329,0.040]，a_do ≈ [0.092,0.084,0.318,0.040]，logits [0.184,0.167,0.635,0.080,0.267]，Σ = 6.660，q_2 ≈ [0.180,0.178,0.283,0.163,0.196]，argmax "it"；Step 3 W_a·[0.092,0.084,0.318,0.040,0,0,1,0] = [0.026,0.1038,0.022,0.540]，a_it ≈ [0.026,0.103,0.022,0.493]，logits [0.052,0.207,0.045,0.985,0.322]，Σ = 7.387，q_3 ≈ [0.143,0.166,0.142,0.363,0.187]，argmax "now"；偏差 [0.370,-0.022,-0.039,0.071]，‖g_I−a_I‖₂ ≈ 0.380——全部与页面一致，无 a×b 与标注积不符、分项之和不等于合计的问题。
- 正文与 summary/overview 数字一致性：5.6x（正文/表格/overview）、6.5x、1.38x、2.7x-3.5x、3000B/2-4B tokens、7 步 unroll、MXFP4/MXFP8——正文与 head 描述、dojo:summary、overview.html 三处一致。
- 公式与符号：α、γ、c、k、K、g/l/m/h、a、q、p 全文单义；$L_{\text{LK}}$、$L_{E3}$ 写法一致；无 Unicode 数学字符出现在标题、summary、正文段落、列表、表格（`<pre>` 伪代码内的 γ、∈、← 按 style-guide §11 代码块豁免）。
- 结构图：均为 HTML 结构（dg-flow/dg-node/dg-stack，TTT 为 table），无等宽字符框线图；图内公式写在 HTML `<div>` 中由 KaTeX 渲染。
- 链接与页面功能：../../wiki/speculative-decoding/index.html、../../wiki/mxfp4-qat/index.html 均真实存在，无「（待生成）」；validate.py 返回 `validation ok: wiki/eagle-speculative/index.html`；两级「核心问题」「本章问题」均有解答折叠块，核心问题答案均指向完整论证章节；overview.html 与 index.html 相互链接。
- 代码块为明示伪代码（「以下是伪代码，不是 Python」），非「声称可运行的代码」，静态审查：g_seq 逐步追加 a、target 前向 1 次、draft 调用 γ 次，与 §4.2–4.3 一致。

## 问题

- [轻微·表述] §3.3 末尾过渡段与 §6 末尾收束段：正文出现对本页自身编排的预告/回指——「后面的构造示例会把这三步逐步手算一遍」「回到开篇的五个问题：……上述论断、公式与数字的来源与范围说明见后文。」，属元话语（预告/回指写作安排），不陈述主题内容。｜引文依据：不适用｜修复要求：改写为对机制本身的陈述——§3.3 过渡段直接给出后续要回答的机制问题、不承诺「构造示例会手算一遍」；§6 收束段直接给出五条结论、删去「回到开篇的五个问题」与「见后文」等对页面结构的指代。｜修复：｜复验：
- [轻微·技术] §4.2 公式与说明：写作 $a_{t+1} = \text{DraftLayer}([g_{1:t};\, e_{t+1}])$ 并注明「$\text{DraftLayer}$ 即 §3 的单层 transformer decoder」，把 2k 维拼接向量直接送入 decoder，省去了 §3.3 图与来源中的 FC 降维层（k×2k），与 §3.3「拼接 $[g_{1:t};\, e_{t+1}]$（$2k$ 维）$\to$ FC（$k \times 2k$）$\to$ 单层 decoder $\to$ $a_{t+1}$」及 [F2] 引文不一致；DraftLayer 若确为 decoder，其输入维应为 k。｜引文依据：Li et al. 2025 §3.1「The concatenated vector is then passed through an FC layer to reduce its dimensionality to k, and subsequently inputted into a single layer decoder, producing the output a.」；本页 §3.3 图注「拼接 $[g_{1:t};\, e_{t+1}]$（$2k$ 维）$\to$ FC（$k \times 2k$）$\to$ 单层 decoder $\to$ $a_{t+1}$（$k$ 维）」。｜修复要求：在 §4.2 的公式/说明中补出 FC 降维层（如 $a_{t+1}=\text{DraftLayer}(\mathrm{FC}([g_{1:t};\, e_{t+1}]))$）或将 FC 明确并入 DraftLayer 的定义，使 §4.2 与 §3.3 及来源一致。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：修复（关闭上述 2 项轻微后可发布；本轮未发现需要回源删除或降级的论断，未发现与来源不符的数字、算式或与 summary/overview 冲突之处）
- 统计：阻断 0 / 重要 0 / 轻微 2