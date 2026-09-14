<!-- review-meta
round: 10
page: wiki/eagle-speculative/index.html
reviewed_content_sha256: 1c2eccb926e45af1
-->
# EAGLE-3 投机解码 draft 模型审查记录（第 10 轮）

- 页面版本：f9493f10c25b269799a7bba436dcf625ffe88c44（工作树 wiki/eagle-speculative/index.html）
- 审查时间：2026-09-14 17:56
- 审查者：独立子代理（编排者派发，未参与写作，也未参与前序轮次）
- 已完整阅读章节：核心问题 / 1. 为什么独立小模型 draft 有两难——EAGLE 的思路转向 / 2. EAGLE-1 的核心机制——在 feature 空间做自回归 / 3. EAGLE-3 的两项架构改变——直接 token 预测 + 多层特征融合（含 3.1–3.3）/ 4. 推理时的自回归 draft——单层 decoder 如何生成 γ 个 draft token（含 4.1–4.4 与两个折叠块）/ 5. 训练 draft 模型——training-time test 与接受率损失（含 5.1–5.3 与补充折叠块）/ 6. 工程实例——K3 的 EAGLE-3 部署与边界（含 6.1–6.4）/ 来源与范围说明（C/F/N 与构造示例、辅助解释与类比边界、简化条件及其限制）

## 来源核对记录（标明本轮核对所依据的版本）

- EAGLE-1：Li et al., arXiv:2401.15077v3。摘要核对「a latency speedup ratio of 2.7x-3.5x, doubled throughput, while maintaining the distribution of the generated text」；§1 核对「but it is not feasible for instruct-tuned models due to the inconsistency in instruction templates between LLaMA2-Chat and TinyLLaMA-Chat」「Despite the 7B model's potential as a draft model, its high overhead diminishes acceleration gains」「TinyLLaMA is trained on 3,000B tokens, whereas EAGLE is trained on 2-4B tokens」；Figure 1 图注核对贪心 temperature=0 设置。对应 [C1][C2][N1][N5]，均一致。
- EAGLE-3：Li et al., arXiv:2503.01840v3。摘要核对「speedup ratio up to 6.5x」「about 1.4x improvement over EAGLE-2」「1.38x throughput improvement at a batch size of 64」「abandons feature prediction in favor of direct token prediction」「multi-layer feature fusion via a technique named training-time test」「scaling up data provides limited improvements for EAGLE」；§3.1 核对「we use the output a_I from the draft model in the previous step to replace g_I」与 FC「reduce its dimensionality to k」句；§3.2 核对「we generate a and feed it back into the draft model for further training」；Figure 6 图注核对 native（第 1 步）/ simulated（第 2、3 步）；Table 1（Vicuna 13B / MT-bench）核对 EAGLE=3.07x、EAGLE-3=5.58x；§4 核对「EAGLE-3 does not modify the target model's weights and uses strict speculative sampling acceptance conditions, ensuring no loss in performance」。对应 [C3][C4][C5][F1][F2][N2][N3]，均一致。
- K3 报告：arXiv:2607.24653 §4.1.4。核对「fine-tune the pre-trained MTP layer into an EAGLE-3-style draft model, with the target model frozen and only the draft layer and its feature-fusion projection updated」「taken from the outputs of the 1st, 4th, and final AttnRes blocks」「initialized as [0 0 I] so that the fused representation coincides at initialization with the high-level feature h_h」「the draft is unrolled for seven steps during training」「MoE expert weights in MXFP4 and their input activations in MXFP8, while non-expert modules remain in higher precision」；Eq.(16) 核对「we directly optimize the likelihood-based LK loss, the negative logarithm of the acceptance rate itself」「with p and q evaluated at temperature 1 and no auxiliary ground-truth cross-entropy term」。对应 [C6][C7][C8][C9][F4][N4]，均一致。
- Leviathan et al. 2023：arXiv:2211.17192。Theorem 3.8 核对改进因子 (1−α^(γ+1))/((1−α)(γc+1))、Definition 3.7 中 c 为成本系数（一次近似模型 run 对一次 target run 之比）、Corollary 3.6 α=E(min(p,q))、附录 A.1 分布等价证明。对应 [F5][F6][C10]，均一致。
- Hugging Face 模型卡 yuhuili/EAGLE-Qwen2-72B-Instruct。核对「5.6 faster than vanilla decoding (13B)」「1.8x faster than EAGLE-1 (13B)」「Inference is conducted on 2x RTX 3090 GPUs at fp16 precision using the Vicuna 13B model」「3x faster than vanilla decoding (13B)」。对应 [N3]，均一致。
- 4.4 构造示例手算：逐行复算 W_a·input、tanh、W_lm·a、softmax 归一化常数与 L2 偏差，四舍五入后全部相符（Step 1 求和 7.585、Step 2 6.661、Step 3 7.387；a_I 与 g_I 的 L2≈0.380）。构造性已在「构造示例」「简化条件及其限制」登记。
- 机械项：validate.py 返回 validation ok；无「（待生成）」占位；前置概念页 wiki/speculative-decoding/index.html、wiki/mxfp4-qat/index.html 均存在；无 alt 含 $...$；图为 HTML 结构、无脚本时仍可读；dojo:type=concept、dojo:topics=训练与优化（词表内）、dojo:tag=推理加速（词表内）；代码块内的 ∈/←/·/γ 等符号依 style-guide §11 属代码块例外，不计入 Unicode 数学符号违规。26 个 details、6 章「本章问题」+ 页面级「核心问题」均配解答折叠块。

## 问题

- [轻微·表述] 4 章伪代码折叠块（`<code>` 内 `a ← DraftLayer(FC(concat(g_seq, e)))`）：该行把整个 g 序列与单个 embedding 扁平拼接，与 4.2 正文「$[\cdot;\cdot]$ 表示逐位置拼接……而非把整个 g 序列与单个 embedding 拼成一个长向量」的定义相冲突；按伪代码字面实现会得到与正文不同的输入形状｜引文依据：不适用（本页内部一致性）｜修复要求：把该行改为与 4.2 一致的逐位置拼接写法（写明 concat 的对象是「每个位置的 g 与该位置移位的 token embedding」配成的 2k 维对，或改为 `concat([g_seq ; e_next])` 并在注释注明 e_next 为与各位置配对的移位 embedding），使伪代码与正文定义不再冲突｜修复：｜复验：
- [轻微·表述] 5.1 末段（`<p>教学解释。 把 TTT 理解为……`）：段首裸标签「教学解释。」是对文本自身性质的注释（元话语），既非本规范许可的示例标签（计算示例/代码示例/构造数据），也非 details 的固定前缀｜引文依据：不适用｜修复要求：删除该裸标签，把类比句并入正文（直接以「把 TTT 理解为……」起句），或改用折叠块「补充：」承载该类比｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：修复（仅轻微项）后可发布

统计：阻断 0 / 重要 0 / 轻微 2