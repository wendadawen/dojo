<!-- review-meta
round: 5
page: wiki/knowledge-distillation/index.html
reviewed_content_sha256: de1709514fdd8b41
-->
# 知识蒸馏（Knowledge Distillation）审查记录（第 5 轮）

- 页面版本：1bf72dc086f87a89322ad4236f4127ee0297ff58
- 审查时间：2026-09-13 20:15
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题 / 1. 硬标签丢掉了什么——非目标类概率携带的暗知识 / 2. 温度 softmax——把分布软化 / 3. 手算温度 softmax——3 类 logits 在 $T=1$ 与 $T=5$ 下的分布 / 4. KD 总损失——软项 + 硬项 + $T^2$ 缩放（含 4.1）/ 5. 边界——KD 解决什么、不解决什么、与 MOPD 的关系（含 5.1、5.2）/ 来源与范围说明；含全部折叠块（各章「本章问题」解答、4.1 推导折叠块、代码折叠块）；另读 overview.html。

## 问题

- [阻断·技术] `<head>` 的 `dojo:summary` 与 §4 正文：同一符号 $\alpha$ 在两处含义相反，违反「符号全文单义」。`dojo:summary` 把 $\alpha$ 乘在硬项上（α 为硬项权重），正文 §4 把 $\alpha$ 乘在软项上（α 为软项权重）。｜引文依据：`dojo:summary`：「典型损失为 $\alpha\,\mathrm{CE}_{hard}+(1-\alpha)\,\mathrm{KL}_{soft}$。」；§4 正文：「$\mathcal{L} = \alpha \cdot T^2 \cdot \mathrm{KL}(p_T^{\text{teacher}}\,\|\,p_T^{\text{student}}) + (1-\alpha)\cdot \mathrm{CE}(y,\,p_1^{\text{student}})$」、符号表「$\alpha \in [0,1]$：软损失项权重」、以及「本文采用 $\alpha$ 为软损失权重的约定；部分文献用 $\beta = 1 - \alpha$ 作硬损失权重」。｜修复要求：把 `dojo:summary` 的公式改为与正文同一约定 $\alpha\,\mathrm{KL}_{soft}+(1-\alpha)\,\mathrm{CE}_{hard}$（或把摘要里的硬项权重改用 $\beta=1-\alpha$），使全页 $\alpha$ 只表示软损失权重。｜修复：｜复验：

- [轻微·表述] §1 第 4、5 段与 §2 第 4 段：临场评价与口语化措辞。｜引文依据：「一旦学生学到这种类间关系，相当于免费拿到了大量辅助监督信号。」「这正是暗知识的力量。」「温度 $T$ 越大，logits 之间的差距被"压缩"得越厉害」。｜修复要求：改为中性陈述，去掉「免费拿到」「力量」「越厉害」等评价性/口语化表达（如「这正是暗知识的力量」改为「这属于教师软分布携带、可被学生利用的监督信息」；「越厉害」改为「越明显」）。｜修复：｜复验：

- [轻微·一致性] overview.html §2 与 index.html §1：两页对「暗知识」一词出处的表述互相冲突。｜引文依据：overview.html「丢掉了"错得离哪个类近"的类间关系信息（Hinton 称为"暗知识"）」；index.html §1 callout「这是业界流传的称谓，HVD15 论文正文并未使用该词」、C1「HVD15 全文无该词」。｜修复要求：把 overview.html 的「（Hinton 称为"暗知识"）」改为「（业界习称"暗知识"，非 HVD15 论文用词）」，与 index.html 一致。｜修复：｜复验：

## 逐项核对结论（无问题部分）

- 回源核对通过（来源 HVD15 = arXiv:1503.02531，经 ar5iv 全文核对）：
  - 温度 softmax 公式 $q_i=\exp(z_i/T)/\sum_j\exp(z_j/T)$：HVD15 §2 直接给出，一致。
  - $T^2$ 缩放：原文「Since the magnitudes of the gradients produced by the soft targets scale as $1/T^2$ it is important to multiply them by $T^2$ when using both hard and soft targets.」——与页面 §4.1 及 C6 一致。
  - 高温零均值极限梯度「∂C/∂z_i ≈ 1/(N T²)(z_i−v_i)」：原文 Eq.4「If we now assume that the logits have been zero-meaned separately for each transfer case so that ∑z_j=∑v_j=0 ... ∂C/∂z_i≈1/(N T²)(z_i−v_i)」——一致。页面按「乘 T² 后为 (1/N)(z_i−v_i)，等价于最小化 (1/2N)‖z−v‖²」自洽（论文只写到「minimizing 1/2(z_i−v_i)², provided the logits are zero-meaned」，其 1/N 被吸收在「up to the scale factor 1/(N T²)」中；页面把 T² 吸收进软损失后给出 (1/2N)，算式与自身结论一致，非错误）。
  - 硬项权重：「We found that the best results were generally obtained by using a considerably lower weight on the second objective function.」（§2）——页面引文准确、归属章节正确。
  - §4.1 权重 0.5：「used a relative weight of 0.5 on the cross-entropy for the hard targets」确在 §4.1（Results），页面「§4.1 的具体实验以 0.5 作为硬标签交叉熵的相对权重（按本文 α 为软损失权重的约定即 α = 0.5）」换算正确。
  - MNIST（HVD15 §3）：教师 2×1200 dropout 67 错误；2×800 硬标签学生 146；2×800 蒸馏学生（T=20）74——页面 §5 表格与 N1 一致；小容量（30 单元/层）最佳 T≈2.5–4 一致。
  - 「无数字 3」实验（HVD15 §3）：原文「the distilled model only makes 206 test errors of which 133 are on the 1010 threes in the test set」；偏差 +3.5 后 3 的错误降到 14（即 1010−14=996 正确）。页面「1010 个 3 里 996 个正确；未校正错误 206、其中 133 个是 3（3 中正确 877/86.8%）」——1010−133=877、877/1010=86.8%，数值与推算一致。
  - 语音（HVD15 §4）：基线 58.9%/10.9%、10 模型 ensemble 61.1%/10.7%、蒸馏单模型 60.8%/10.7%，8 层×2560 单元、约 85M 参数、约 2000 小时英语语音——页面 §5 表格与 N3 逐项一致。
  - K3 MOPD（arXiv:2607.24653 §4.1.3）：9 个专家（3 领域×3 努力程度）、per-token log 概率比稠密奖励、学生自生成 token（on-policy）、接入 RL 框架并复用 partial rollout——页面 §5.2 表述一致。
- 代码：§3 折叠块内 Python 脚本实际运行，输出与「预期输出」逐行逐位一致（T=1: [0.665241,0.244728,0.090031]；T=5: [0.40176,0.328933,0.269307]；T=0.5/2/10/100 及 1/3=0.333333 亦一致）。
- 手算可复算：§3 例 exp 值、分母、归一化、对比表「变化」列（−0.263/+0.084/+0.179，合计 0）均正确；§1 软分布示例 7 类外概率和 0.080=1−0.920 正确。
- 问题块：页面级「核心问题」5 条、各正文章节「本章问题」齐全，均带 `解答：` 折叠块，核心问题答案均指明完整论证所在章节。
- 页面链接：`../../wiki/mopd/index.html`、`../opd/index.html` 均真实存在；`../../index.html`、`overview.html` 有效；正文与来源说明中无 `research/` 路径引用，无「（待生成）」占位。
- 格式：h2/h3 编号连续（1–5、4.1、5.1、5.2）；`来源与范围说明` 下 h3 命名符合规范；`dojo:topics=训练与优化` 在 AGENTS.md 固定大类内；`<head>` 的 description/summary/type/tag 齐全。
- `.dojo/scripts/validate.py wiki/knowledge-distillation/index.html` 返回 `validation ok`。

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 2
- 处置：修复（修订 `dojo:summary` 的 α 约定后方可发布）