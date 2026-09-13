<!-- review-meta
round: 5
page: wiki/newton-schulz/index.html
reviewed_content_sha256: 21591f96c51f3037
-->
# Newton-Schulz 迭代审查记录（第 5 轮）

- 页面版本：64c88c888c4dc35003e571b46ac0e17a593b198d（`git hash-object wiki/newton-schulz/index.html`，工作树）
- 审查时间：2026-09-13 20:19
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题；1. 不用 SVD 的正交化——为什么需要矩阵乘法方案；2. 迭代公式与机制——奇多项式如何把奇异值推向 sign；3. 收敛条件与 Frobenius 预处理——如何保证落入收敛域；4. 零奇异值保持为 0——而不被拉平为 1；5. 在 Muon 优化器中的应用——正交化动量矩阵；来源与范围说明（含全部折叠块、代码块、表格与图注）

## 来源核对（本轮实际打开并定位）

- [S1] docs.modula.systems/algorithms/newton-schulz：核对到 "M = U Σ V^⊤ ↦ U V^⊤"；"the iterations we consider will actually fix zero singular values at zero"；"if we iterate f an infinite number of times, we will obtain precisely the sign function on the interval [−√3,√3]"；"We can achieve this via a simple pre-processing step, mapping X ↦ X / ‖X‖_F"；"We used a particular [cursed quintic iteration] in the Muon optimizer"；cursed quintic 系数 3.4445, −4.7750, 2.0315，且"oscillates and in fact does not converge"。以上均支持页面表述。
- [S2] arXiv:2506.10935（ar5iv 全文）：摘要 "it relies solely on matrix multiplications"；§1 "the orthogonal Procrustes problem … the solution being Q=W the polar factor of X"；"Polar decomposition … X=WH, where W … has orthonormal columns and H … positive semidefinite"；Eq.(1) "X_(k+1)=3/2 X_k−1/2 X_k X_k^T X_k, X_1=X"；"This iteration converges to the orthogonal factor of the polar decomposition if σ_1(X)<√3 and σ_n(X)>0."。支持 C2/F1/F5。
- [S3] nhigham.com/2020/12/15/what-is-the-matrix-sign-function：逐句核对，含 "A matrix multiplication-based iteration is the Newton–Schulz iteration"、"X_{k+1} = (1/2) X_k (3I - X_k^2) … This iteration is quadratically convergent if ||I-A^2|| < 1 for some subordinate matrix norm"，该条件确指 Newton–Schulz（非 sign 的 Newton 迭代）。支持 C6。
- [S5] kellerjordan.github.io/posts/muon："Improved the speed record for training to 3.28 val loss on FineWeb … by a factor of 1.35x"；"(a, b, c) = (3.4445, -4.7750, 2.0315)" 5 步；"Muon is an optimizer for 2D parameters of neural network hidden layers."；"Scalar and vector parameters … as well as the input and output layers, should be optimized by a standard method such as AdamW."；"AdamW should be used for the embedding and final classifier head layers"。支持 N1/C7/第 5 章。
- 代码核验：页面 Python 代码在 python3 下实际运行，输出与「预期输出」逐行一致（diag(0.5,0)：0.5→0.6875→0.8687744140625→0.9752996308188813→0.9990923725928302→0.9999987646925808，终态 0.999999999997711；G：‖G‖_F=1.7321，误差 0.881917/0.737435/0.507949/0.226273/0.041296/0.001297/0.000001/0.000000，final Q=[[0.8944271909999159,0.4472135954999578],[-0.44721359549995776,0.8944271909999159]]）。numpy 旁证：G/√3 奇异值 ≈ (0.93417236,0.35682209)，极因子 UV^⊤ 与页面终态 Q 一致。数字全部可复算。
- `.dojo/scripts/validate.py wiki/newton-schulz/index.html` → "validation ok"。

## 问题

- [重要·技术] 第 2 章末段（约第 189 行）与「核心问题」第 2 题解答（约第 97 行）：写"当所有非零奇异值都到 $1$ 时，$Q_k^{\!\top}Q_k\to I$，即 $Q_k$ 收敛到正交矩阵——而且正是极分解的正交因子 $W=UV^{\!\top}$"，未加"无零奇异值（$\sigma_{\min}>0$）"限定；该无条件形式与本页第 4 章自己的构造示例（diag(0.5,0) 迭代到 diag(1,0)，此时 $Q_k^{\!\top}Q_k=\mathrm{diag}(1,0)\ne I$，且 $W=UV^{\!\top}=I\ne\mathrm{diag}(1,0)$）直接矛盾。｜引文依据：[S2] arXiv:2506.10935 §1 原文 "This iteration converges to the orthogonal factor of the polar decomposition if σ_1(X)<√3 and σ_n(X)>0."（收敛到极因子以 σ_n>0 为前提）；本页第 4 章第 408 行"终态 $\mathrm{diag}(1,0)$ 同时展示了'非零被拉平'和'零被保持'"。｜修复要求：在第 2 章该句与核心问题第 2 题解答中补加限定——"当 $\sigma_{\min}>0$（无零奇异值）时"才收敛到正交矩阵/极因子 $W=UV^{\!\top}$；存在零奇异值时极限为 $\mathrm{diag}(1,0)$ 这类半正交（秩保持）矩阵，而非正交矩阵。`<head>` description 中"收敛到正交矩阵"一处同步修正。｜修复：｜复验：
- [轻微·表述] 第 5 章（约第 449 行）：以"需要强调的边界："作引导，属元话语式过渡（与 check.md 第 2.2 节"需要注意的是"同类）。｜引文依据：不适用｜修复要求：改为直接陈述该边界（如"Muon 只对二维矩阵参数用 NS；……"），删去"需要强调的边界："引导语。｜修复：｜复验：
- [轻微·技术] 第 5 章（约第 449 行）："前者因正交化对一维参数无意义（一维向量只有一个奇异值，正交化退化为标量归一化）"——此理由无来源支持，属把推断写成设计事实。｜引文依据：[S5] 仅称 "Muon is an optimizer for 2D parameters of neural network hidden layers." 与 "Scalar and vector parameters …, as well as the input and output layers, should be optimized by a standard method such as AdamW."，未给出"一维向量只有一个奇异值"这一理由。｜修复要求：删去该括号理由，或改写为明确标注的推断（如"（此处可理解为：NS 以二维矩阵为对象…）"），不写成来源结论。｜修复：｜复验：
- [轻微·格式] `<head>` description："基于数值线性代数教材核对"与来源清单不符——[S1] 为文档站、[S3] 为 Higham 博客文章、[S2] 为 arXiv 论文、[S5] 为博客，无"教材"类来源。｜引文依据：来源清单 S1–S5 条目本身。｜修复要求：将"基于数值线性代数教材核对"改为与实际来源一致的表述（如"依据 Modula 文档与 Higham、Grishina 等来源核对"）。｜修复：｜复验：
- [轻微·格式] `<style>` 中 `.diagram` 规则（第 34–43 行）在页面中无任何 `class="diagram"` 使用，为死 CSS。｜引文依据：不适用｜修复要求：删除未使用的 `.diagram` 规则块。｜修复：｜复验：
- [轻微·格式] 正文 `<sup>` 引用与来源章节非双向对应：正文仅出现 [C3]、[F1,C3]、[F4,C5]、[N1] 四处，而"论断与来源（C）"定义 C1–C7、"公式与来源（F）"定义 F1–F6、"外部数字与实验条件（N）"定义 N1–N3，其中 C1、C2、C4、C6、C7、F2、F3、F5、F6、N2、N3 在正文无对应上标。｜引文依据：guides/concept/style-guide.md §6"正文使用 `<sup>[Cx]</sup>` 上标引用…与来源章节双向对应"。｜修复要求：在正文相应论断处补上缺失的上标引用（至少 C1/C2/C4/C6/C7/F2/F3/F5/F6），使来源条目与正文一一对应。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复

（说明：核心论断、公式、数字与代码均回源核对通过——迭代式与 [S2] Eq.(1) 代数等价，$\sqrt3$ 收敛边界、Frobenius 预处理、零奇异值固定、二次收敛、Muon 的 1.35× 与 cursed quintic 系数均与来源一致；本轮唯一重要问题为第 2 章收敛表述漏掉 $\sigma_{\min}>0$ 限定，与第 4 章构造示例冲突。）