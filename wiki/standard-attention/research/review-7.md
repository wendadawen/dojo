<!-- review-meta
round: 7
page: wiki/standard-attention/index.html
reviewed_content_sha256: 6e365876dba10c02
-->
# 标准 Transformer 注意力审查记录（第 7 轮）

- 页面版本：7e33a176bb4a4d98efaafd88e623ed5b2153e098
- 审查时间：2026-09-13 21:52
- 审查者：独立子代理（未参与写作与前序轮次）
- 页面 dojo:type：concept（适用 guides/concept/check.md）
- 已完整阅读章节：head（description / dojo:summary / type / topics / tag）、核心问题（4 题及解答折叠块）、最容易误解、1 注意力要解决什么问题、2 缩放点积公式（含 2×2 构造示例与 softmax 数值稳定折叠块）、3 为什么除以 √d_k（含方差推导折叠块、雅可比折叠块、不缩放对照表折叠块）、4 多头注意力（含 3×3 因果遮罩折叠块）、5 复杂度、瓶颈与边界、来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）；含全部图注与折叠块。
- 来源核对方式：arXiv:1706.03762（ar5iv 全文，§3.2.1/§3.2.2/§3.2.3/§3.5/§4 Table 1）、arXiv:2205.14135（FlashAttention）、arXiv:1905.10650（Michel et al. 2019）；数字复算用 NumPy 独立重跑（2×2、3×3、softmax、不缩放/缩放统计表、高斯极值）。

## 问题

- [重要·技术] §5「Flash vs Linear 的区分」callout（index.html:520）：把 FlashAttention 的实验条件下的数字写成无条件论断。原文为"训练端到端另有 15% 提速"，未带被测模型与基线，读者会当成 FlashAttention 的通用端到端收益。｜引文依据：arXiv:2205.14135 摘要 "15% end-to-end wall-clock speedup on BERT-large (seq. length 512)"，基线为 "the MLPerf 1.1 training speed record"（Table 1：20.0 min → 17.4 min）；同条目中 3 倍一项页面已正确标注序列长度区间，§4.3 原文 "FlashAttention runs significantly faster than exact attention baselines, up to 3× faster than the PyTorch implementation"，benchmark 覆盖 "common sequence lengths from 128 to 2K"。｜修复要求：补回条件，例如改为"（BERT-large、序列长度 512，相对 MLPerf 1.1 训练记录，端到端训练提速 15%）"，不得保留无条件写法。｜修复：｜复验：

- [轻微·技术] §4「多头"子空间分化"是设计目标」（index.html:447）与 §5「注意力不保证多头可解释」（index.html:530）：对 Michel et al. 2019 的转述强度高于原文。｜引文依据：arXiv:1905.10650 摘要 "a large percentage of attention heads can be removed at test time without significantly impacting performance"（"不显著影响"），页面写作"删除部分头不降低性能"（"不降低"）。｜修复要求：改为"删除部分头后性能无显著下降"，与原文措辞一致。｜修复：｜复验：

- [轻微·可读性/页面功能] 五处 `<h3>本章问题</h3>`（index.html:149、273、353、471、548）均无 id 属性：页面脚本 `if (!h.id) { h.id = h.textContent... }` 会按文本为它们生成同一个 id「本章问题」，产生 5 个重复 id；侧栏目录中第 2–5 章的「本章问题」链接 `data-target="本章问题"` 全部命中第 1 章（`getElementById` 取首个），滚动高亮也会同时点亮 5 项。同类概念页 wiki/rope/index.html 已为每处「本章问题」给出唯一 id（如 `why-position-encoding-questions`）。｜引文依据：不适用｜修复要求：为每个 `<h3>本章问题</h3>` 指定唯一 id（如 `why-attention-questions`、`sqrt-dk-questions` 等），使目录锚点各自指向所属章节。｜修复：｜复验：

- [轻微·可读性] §3 开头（index.html:302）："Vaswani et al. 2017 在正文只用一句话引出……完整推导藏在脚注里。该推导在脚注中给出，下面展开。"两点不合：一是与来源不符——脚注只给出假设与结论，没有逐步推导；二是"下面展开"属元话语（同类见 index.html:496"标准注意力机制到这里完整了"）。｜引文依据：arXiv:1706.03762 §3.2.1 脚注原文 "assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q·k = Σ q_i k_i, has mean 0 and variance d_k."（仅假设与结论，无推导步骤）；同段正文为两句："We suspect that for large values of d_k, the dot products grow large in magnitude... To counteract this effect, we scale the dot products by 1/√d_k."｜修复要求：改为"论文脚注给出该结论的假设与结果（均值 0、方差 d_k），逐步推导如下"；删除"下面展开""到这里完整了"一类元话语，并去掉"藏在脚注里。该推导在脚注中给出"的重复表述。｜修复：｜复验：

- [轻微·格式] 教学/讨论块句号后多一个半角空格，共 4 处：index.html:137"教学解释。 可以把"、225"构造示例。 设"、444"常见误解。 多头"、457"构造示例。 设"；同类概念页 wiki/rope/index.html 无此写法。｜引文依据：不适用｜修复要求：删除句号后的多余空格，与站内其余页面一致。｜修复：｜复验：

## 已核对且无问题（本轮抽查依据）

- 2×2 构造示例逐位复算通过：QK^T=I（`1×1+0×0`、`1×0+0×1`）、1/√2≈0.7071、softmax([0.7071,0])=[0.670,0.330]（e^0.7071≈2.0282）、AV=[[1.66,2.66],[2.34,3.34]]，与正文一致。
- 3×3 因果遮罩复算通过：[1,0,0]、[0.401,0.599,0]、[0.258,0.316,0.426]（e^0.3≈1.350、e^0.7≈2.014、e^0.4≈1.492、e^0.6≈1.822、e^0.9≈2.460）。
- 不缩放/缩放统计表独立复算通过（q,k 分量 iid N(0,1)、n=256、大样本）：d_k=4 → 0.178/3.92；d_k=64 → 0.741/0.76；d_k=1024 → 0.939/0.15；缩放列 0.043/5.05（20 次种子重跑 0.0429–0.0433、熵 5.05），与页面数值在抽样误差内一致。1/256≈0.004、ln 256=5.545 正确。
- 高斯极值数字复算通过：E[max of n]≈√(2 ln n) 的确高估有限 n（n=8：16.3 vs 模拟 1.42×8≈11.4；n=256：26.6 vs 2.83×8≈22.6），页面标注的"n=8 约 11、n=256 约 23"及"对有限 n 会高估"成立。
- 论文数字一致：C/F/N 与原文吻合——`Attention(Q,K,V)=softmax(QK^T/√d_k)V`（Eq.1）、`MultiHead=Concat(head_1..head_h)W^O`（Eq.2）、`head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)`、W_i^Q,W_i^K∈R^{d_model×d_k}、W_i^V∈R^{d_model×d_v}、W^O∈R^{h·d_v×d_model}、d_model=512/h=8/d_k=d_v=64、§3.2.3 masking 置 −∞、§3.5 位置编码、Table 1 三行复杂度（RNN O(n·d²)/O(n)/O(n)；CNN O(k·n·d²)/O(1)/O(log_k n)；自注意力 O(n²·d)/O(1)/O(1)）；C10 引文与原文 "While for small values of d_k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_k." 逐字一致；C6 引文 "jointly attend to information from different representation subspaces at different positions" 逐字一致。
- 参数量手算 h·d_model·(d_model/h)=d_model²、h·d_k=8×64=512=d_model 正确；复杂度表 n²d_k / O(n²) / n²d_v 正确；n=2048→4.19×10⁶、n=32768→1.07×10⁹（标 1.1×10⁹）正确。
- 页内链接有效：../rope/index.html、../mla/index.html 均存在；index.html 与 overview.html 双向链接(overview.html:16、56)；overview 的数字（d_model=512/h=8/d_k=64、n=32768 数 GB、O(n²) 在 QK^T）与正文一致。
- head 标注齐全：description 为纯文本、dojo:summary 为可渲染 KaTeX、dojo:type=concept、dojo:topics=注意力机制（词表内）、dojo:tag=注意力；无 alt 含 `$...$`；公式全部 KaTeX，无 Unicode 数学字符；结构图为 HTML 结构块，无等宽字符框线图；无交互视图依赖脚本方可读的情形。
- 表述面：无"本页"为主语的自我指代、无"我们/你"会话指代、无调试与复现踩坑叙事、无临场评价；`python3 .dojo/scripts/validate.py wiki/standard-attention/index.html` 返回 `validation ok`。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（关闭 1 条重要与 4 条轻微后复验；本轮未发现阻断级问题，页面事实、公式、数字与来源一致，表述基本干净）
