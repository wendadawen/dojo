<!-- review-meta
round: 7
page: wiki/eagle-speculative/index.html
reviewed_content_sha256: e07da113aad0c4ef
-->
# EAGLE-3 投机解码 draft 模型审查记录（第 7 轮）

- 页面版本：index.html 工作树哈希 59e51136a0efd78d4a0ae4786ff96b33fd92e132（sha256 405af6f045161a09…）
- 审查时间：2026-09-14 16:56
- 审查者：独立子代理（未参与写作，未读取 research/ 下任何文件）
- 页面类型：dojo:type = concept（适用 guides/concept/check.md）
- 来源获取：EAGLE-1 arXiv:2401.15077v3、EAGLE-3 arXiv:2503.01840v1 HTML 全文（curl + 文本化后逐条比对）；Leviathan et al. 2023 arXiv:2211.17192（ar5iv 全文）；K3 报告 arXiv:2607.24653 v1/v2 HTML 全文；Hugging Face 模型卡 yuhuili/EAGLE-Qwen2-72B-Instruct。
- 已完整阅读章节（含折叠块、图注、来源与范围说明）：1. 为什么独立小模型 draft 有两难——EAGLE 的思路转向；2. EAGLE-1 的核心机制——在 feature 空间做自回归；3. EAGLE-3 的两项架构改变——直接 token 预测 + 多层特征融合；4. 推理时的自回归 draft——单层 decoder 如何生成 γ 个 draft token（含 4.4 构造示例与伪代码折叠块）；5. 训练 draft 模型——training-time test 与接受率损失；6. 工程实例——K3 的 EAGLE-3 部署与边界；来源与范围说明（C1–C10、F1–F6、N1–N5、构造示例、类比边界、简化条件）。

## 问题

- [重要·技术] index.html §2「EAGLE-1 的核心机制」前向数据流图（`<figure class="diagram">` 第四个节点）：把 draft 采样得到的 token 标为 `$x_{t+1}$`。该图自身在第一个节点已把 `$e_{t+1}$` 定义为「上一步采样的 token」，即 `$x_{t+1}$` 的 embedding——同一张图里 draft 的输入 token 与输出 token 是同一个 `$x_{t+1}$`，自相矛盾；且与同页 §4.2/§4.3 及 §4.4 手算例不符（§4.2：输入 `concat(g_1,…,g_t, e_{t+1})` → `a_{t+1}` → lm_head → 「采样得到第一个 draft token `$x_{t+2}$`」；§4.3 图注：「本轮产出 draft token `$x_{t+2}, x_{t+3}, x_{t+4}$`」）｜引文依据：EAGLE-3 §3.1「Finally, we input a_I into the LM head and sample to obtain the draft token "do".」（前缀 "How can"、target 采样 "I"，draft 首个 token "do" 位于第 4 位，即 t+2）；EAGLE-1 §3.2 分类损失 `p_{i+2}=\text{Softmax}(\text{LM\_Head}(f_{i+1}))`（feature 下标 i+1 对应 token 下标 i+2）；EAGLE-1 §1「The method adds only a lightweight plug-in (a single transformer decoder layer)…The Autoregression Head consisting of an FC layer and a decoder layer…we utilize the decoder layer to predict the next feature」｜修复要求：把该节点改为「采样得到 draft token `$x_{t+2}$`」，与 §4 及来源统一（§2 正文「还包含第 t 步采样得到的 token 的 embedding」指代 `$e_{t+1}$`，可保留，但需保证与图一致）｜修复：｜复验：

- [轻微·技术] index.html §3.2：「$W_{\text{fuse}} \in \mathbb{R}^{k \times 3k}$ 是融合矩阵（K3 实现为无偏矩阵 $W_{E3}$）」。「无偏」在中文 ML 语境指 unbiased（如无偏估计），与源文 bias-free（无偏置项）含义不同，属误译｜引文依据：K3 报告 §4.1.4「These features are concatenated and projected to the hidden size by a bias-free matrix W_E3, initialized as [0 0 I]…」；同页 [C8] 引文亦为「bias-free matrix WE3」｜修复要求：改为「无偏置矩阵」或「bias-free 矩阵」，与 [C8] 英文原文对应｜修复：｜复验：

- [轻微·表述] index.html §2 末段：「EAGLE-1 在 feature 空间做自回归有效，但 EAGLE-3 发现扩大训练数据对 EAGLE-1 提升有限——扩大数据看到的曲线在 EAGLE 上饱和。」破折号后「扩大数据看到的曲线在 EAGLE 上饱和」成分残缺、与前句语义重复，读不通｜引文依据：不适用（表述问题）｜修复要求：改写为通顺单句，例如「即扩大数据后接受率曲线在 EAGLE 上趋于饱和」｜修复：｜复验：

- [轻微·来源] overview.html §1 一句话定位：「它把 draft 减到单层 decoder」。「减到」（把层数减到单层）暗示层数变更是 EAGLE-3 的改进，与 index.html §3.3「EAGLE-1 与 EAGLE-3 均为单层 decoder，层数不是 EAGLE-3 的两项改变之一」及 EAGLE-1 原文不符；overview.html 自身 §4 也写「EAGLE-1 是单层 decoder + …」，前后不一致｜引文依据：EAGLE-1 §1「The method adds only a lightweight plug-in (a single transformer decoder layer) to the LLM」；EAGLE-1 §3.1「The Autoregression Head consisting of an FC layer and a decoder layer」｜修复要求：把「把 draft 减到单层 decoder」改为不暗示层数变更的表述（如「draft 只有单层 decoder」），与 index.html §3.3 及 overview.html §4 一致｜修复：｜复验：

## 已核对且未发现问题的主要来源论断（本轮通过项）

- [C1][N5] EAGLE-1 §1 原文逐字命中：「An alternative could be to use TinyLLaMA, but it is not feasible for instruct-tuned models due to the inconsistency in instruction templates」「TinyLLaMA is trained on 3,000B tokens, whereas EAGLE is trained on 2-4B tokens」「Despite the 7B model's potential as a draft model, its high overhead diminishes acceleration gains」。
- [C2] EAGLE-1 摘要逐字命中：「autoregression at the feature (second-to-top-layer) level is more straightforward than at the token level」「the inherent uncertainty in feature (second-to-top-layer) level autoregression constrains its performance」「By incorporating a token sequence advanced by one time step, EAGLE effectively resolves the uncertainty」。
- [C3] EAGLE-3 摘要逐字命中：「abandons feature prediction in favor of direct token prediction and replaces reliance on top-layer features with multi-layer feature fusion via a technique named training-time test」。
- [C4][F1][F2] EAGLE-3 §3.1 逐字命中：「the token "I" has not yet been checked by the target model, and we cannot obtain g_I」「we use the output a_I from the draft model in the previous step to replace g_I」「concatenate the k-dimensional vectors l, m, and h to form a 3k-dimensional vector, then pass it through a fully connected (FC) layer to reduce it to k-dimensions, obtaining a feature g」「The concatenated vector is then passed through an FC layer to reduce its dimensionality to k, and subsequently inputted into a single layer decoder, producing the output a」。
- [C5] EAGLE-3 §3.2 逐字命中：「During training, we perform test steps, where we generate a and feed it back into the draft model for further training.」；Figure 6 图注逐字命中（native 第一步 + two simulated steps；gray = training data；blue/yellow = 第 1/2 轮预测）。
- [C6][F4] K3 报告 §4.1.4 Eq.(16)：引文与 v1 全文逐字一致（含「likelihood-based LK loss [103]」「evaluated at temperature 1 and no auxiliary ground-truth cross-entropy term」；v2 该处引用编号漂移为 [104]，Eq.(16) 号不变，页面引文与 v1 相符）。
- [C7][C8][C9][N4] K3 报告 §4.1.4 v2 逐字命中：MTP 层镜像 backbone block、target 冻结只更新 draft 层与 feature-fusion 投影；1st/4th/final AttnRes blocks；W_E3「initialized as [0 0 I] so that the fused representation coincides at initialization with the high-level feature h_h」；「unrolled for seven steps during training」；「MoE expert weights in MXFP4 and their input activations in MXFP8, while non-expert modules remain in higher precision」。
- [C10] EAGLE-3 §4 逐字命中：「EAGLE-3 does not modify the target model's weights and uses strict speculative sampling acceptance conditions, ensuring no loss in performance.」。
- [F5] Leviathan 2023 Corollary 3.6：α = E(min(p,q))；`α = Σ min(p,q) = 1 − TV(p,q)` 由 min 恒等式可复算，成立。
- [F6] Leviathan 2023 Theorem 3.8 逐字命中：「The expected improvement factor in total walltime by Algorithm 1 is (1−α^{γ+1})/((1−α)(γc+1))」，与页面 `S = (1 − α^{γ+1}) / [(1−α)(1 + γc)]` 等价；c 定义「the ratio between the time for a single run of Mq and the time for a single run of Mp」与页面「draft 与 target 的单步成本比」一致。
- [N1] EAGLE-1 摘要逐字命中「For LLaMA2-Chat 70B, EAGLE achieved a latency speedup ratio of 2.7x-3.5x, doubled throughput, while maintaining the distribution of the generated text.」。
- [N2] EAGLE-3 摘要逐字命中「EAGLE-3 achieves a speedup ratio up to 6.5x, with about 1.4x improvement over EAGLE-2」「In the SGLang framework, EAGLE-3 achieves a 1.38x throughput improvement at a batch size of 64.」。
- [N3] 模型卡逐字命中「5.6 faster than vanilla decoding (13B)」「1.8x faster than EAGLE-1 (13B)」「3x faster than vanilla decoding (13B)」；EAGLE-3 Table 1（Temperature=0，Vicuna 13B / MT-bench）实测 EAGLE = 3.07x、EAGLE-3 = 5.58x，与页面「约 3x / 5.6x」一致。
- §4.4 构造示例全部可复算：tanh([0.08,0.45,0.14,0.13])≈[0.080,0.422,0.139,0.129]；softmax([0.160,0.844,0.278,0.259,0.385])≈[0.155,0.307,0.174,0.171,0.194]（Σ≈7.585）；Step 2/3 的 W_a 行乘、tanh、logits 与 q 亦逐项复算通过（个别末位舍入差 ≤0.001，不构成缺陷）；偏差 [0.370,−0.022,−0.039,0.071]、L2≈0.380 复算为 0.379，正确。
- 结论性检查：页面级「核心问题」5 题与各章「本章问题」每题均有解答折叠块，答案与正文结论一致，核心问题答案均给出所在章节；两面互链（index↔overview）存在；`../../wiki/speculative-decoding/index.html`、`../../wiki/mxfp4-qat/index.html` 均存在且被引章节名（Draft-then-Verify、为什么这条规则能保分布、工程实例与边界）在目标页真实存在；无「（待生成）」占位；`alt` 中无 `$…$`；结构图为 HTML/内联表格，非等宽字符框线；MathJax/KaTeX 公式在标题与正文中均以 LaTeX 书写（Unicode 数学字符仅出现在 `<pre><code>` 伪代码块内）；`python3 .dojo/scripts/validate.py wiki/eagle-speculative/index.html` 返回 `validation ok`；`dojo:topics=训练与优化`、`dojo:tag=推理加速` 均在词表内。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（第 7 轮无阻断；1 条重要 + 3 条轻微，按 check.md §4 逐条修复后进入复验，本轮问题关闭后方可发布）

统计：阻断 0 / 重要 1 / 轻微 4