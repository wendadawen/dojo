<!-- review-meta
round: 4
page: wiki/delta-rule/index.html
reviewed_content_sha256: 0bb2b464bd38420d
-->
# Delta 规则与 DeltaNet 审查记录（第 4 轮）

- 页面版本：126f31ebdddf3896b344fe8337f09b0423e9b08c（wiki/delta-rule/index.html 工作树）
- 审查时间：2026-09-13 19:37
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下的规划、修复与前序审查记录）
- 已完整阅读章节：核心问题；最容易误解；1. 为什么线性注意力会"记不清"——key 碰撞与 retrieval error（1.1、1.2、本章问题、折叠块）；2. Delta 规则的紧凑公式与手算（2.1、2.2、本章问题）；3. 等价改写——"先擦除后写入"与几何直觉（3.1、3.2、本章问题、折叠块、SVG 图）；4. 边界情况与 β_t 的退化（4.1–4.4、本章问题）；5. DeltaNet 与相邻模型对比（5.1–5.5、含代码折叠块、本章问题）；来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件、构造示例、辅助解释与类比边界、简化条件及其限制）

## 已核对无误项（作为本轮核对依据）

- 手算与公式：1.2 节 $d=2$、3 个非正交 key 的 $S=\begin{pmatrix}1&1\\1&2\end{pmatrix}$、$Sk_2=(2,3)^\top$、污染项 $=v_1+v_3=(2,1)^\top$，与 2/3/4 章主例子（$S_1,S_2$、等价形式、$\beta_2=0.5$ 的 $(0.5,0.5)^\top$）全部逐项复算一致；$P^2=P$、$P^\top=P$ 推导正确；C4→C3 的代数代入（系数 $-1+(1-\beta_t)=-\beta_t$）正确。
- 代码：折叠块 numpy 代码实际执行，输出与页面"预期输出"逐字符相同（Linear $S_2=[[1,0],[1,0]]$、$S_2k_2=[1,1]$；Delta $S_2=[[0,0],[1,0]]$、$S_2k_2=[0,1]$；Matches compact form: True）。
- 数字（N1/N2）：arXiv:2406.06484 v3 Table 1（MAD）与 Table 2（LM）逐格核对一致——Mapper/Transformer 94.1/29.8/86.8/99.6/85.2/74.5；Mamba 90.4/6.7/90.1/86.3/89.5/69.3；GLA 80.8/6.9/81.6/88.6/63.3/60.0；DeltaNet 100/35.7/100/100/52.8/71.8，Compress DeltaNet 42.2、Mamba 52.7；Average 确为 6 项均值（DeltaNet $430.7/6=71.8$）。PPL：DeltaNet (w. conv) 16.87/12.21、Mamba (w. conv) 17.06/13.89、GLA (w. conv) 17.25/14.92、Transformer++ 16.85/13.44 一致。原文 "The 340M/1.3B models are trained for 15B/100B tokens respectively." 支持页面标注的实验设置。
- 引文逐字核对通过："a purely additive update rule makes it difficult to deallocate past key-value associations, eventually leading to key 'collisions' when $L>d$"（Yang 2024 §2.2）；"$\bm{v}_t^{\text{old}}=\mathbf{S}_{t-1}\bm{k}_t$"、"$\bm{v}_t^{\text{new}}=\beta_t\bm{v}_t+(1-\beta_t)\bm{v}_t^{\text{old}}$"（Yang 2024 §2.2）；"when $\beta_t=1$, $\mathbf{I}-\bm{k}_t\bm{k}_t^{\mathsf{T}}$ becomes a projection matrix, erasing information in one subspace while preserving the other $d-1$ subspaces"（Yang 2024 §3.3）；"akin to the famous error-correcting delta-rule (Widrow & Hoff, 1960)"（Schlag 2021 引言）；"essentially implements the famous error-correcting delta rule (Widrow & Hoff, 1960)"（Schlag 2021 §4.2）；remove/write 形式 Eq. 23（Schlag 2021 §4.2）；"We refer to the Linear Transformer with our delta update rule as a Delta Network."（Schlag 2021，支持 2.1 图中"原文称 Delta Network"）；GDN 摘要 "gating enables rapid memory erasure while the delta rule facilitates targeted updates"；GDN §1 "$\mathbf{S}_t=\alpha_t\mathbf{S}_{t-1}+\bm{v}_t\bm{k}_t^{\intercal}$ … $\alpha_t\in(0,1)$"，"since this process only modifies a single key-value pair at a time, the model lacks the ability to rapidly clear outdated or irrelevant information"（支持"逐 key 串行/$\alpha_t$ 一步全局衰减"）；gated delta rule 公式 $\mathbf{S}_t=\mathbf{S}_{t-1}(\alpha_t(\mathbf{I}-\beta_t\bm{k}_t\bm{k}_t^{\intercal}))+\beta_t\bm{v}_t\bm{k}_t^{\intercal}$；"although it somehow struggles on the 'Memorize' task"（arXiv v6 原文，与页面引文一致）；"DeltaNet: Linear Transformers with the Delta Update Rule"（Yang 2024 §2.2 标题）。
- C9：Kimi K3 报告 arXiv:2607.24653 §2.1.1 "KDA extends the delta-rule recurrence … with a channel-wise forget gate"，支持"基于 delta 规则递归 + channel-wise 遗忘门"。C10：github.com/fla-org/flash-linear-attention 可达（HTTP 200）。`$\beta_t=\sigma(\mathbf{W}_\beta\bm{x}_t)\in(0,1)$`（Yang 2024 §2.2）支持符号表。
- 机械项：`.dojo/scripts/validate.py` 返回 validation ok；`dojo:type=concept`、`dojo:topics=注意力机制`（词表内）、`dojo:tag=注意力`；正文 $\sup$[C1–C10,F1–F3,N1,N2] 与来源章节双向对应无遗漏；overview.html 与 index.html 互链；前置概念页 wiki/linear-attention/index.html 存在，无"（待生成）"；无 Unicode 数学字符外露；结构图为内联 SVG，图内公式用 `<foreignObject>` 承载。

## 问题

- [重要·技术] 5. 章末"全文总结"（第 719 行）及核心问题 1 解答（第 80 行）、dojo:summary：把 delta 规则的作用写成"解决了线性注意力在 $L>d$ 时的 key 碰撞问题"。｜引文依据：Yang 2024 §2.2 原文只到 "A model should ideally learn to remove less important key-value associations to make room for new ones"；GDN §1 只到 "selectively updates memory by (softly) replacing an old key-value pair with the incoming one"，且同文指出 DeltaNet "performs moderately on real-world tasks"；来源均未出现"消除碰撞"。页面 1.1 节又自行论证 $L>d$ 时非正交不可避免（"$\mathbb{R}^d$ 中两两正交向量最多 $d$ 个"），与该断言方向相反：按 $\beta=1,\|k\|=1$ 递推，两非正交 key 下 $S_2k_1=v_1(1-(k_1^\top k_2)^2)+v_2(k_2^\top k_1)\neq v_1$，较早写入的 key 仍受后续写入干扰。｜修复要求：把"解决 key 碰撞"改为来源支持的表述（"写入前先擦除旧关联，覆盖同方向旧值、降低干扰"），或在该处补一句限定：碰撞来自 key 非正交，delta 规则移除的是陈旧关联而非维度上限。｜修复：｜复验：
- [轻微·技术] 来源章节 C1/C7/C8/N1/N2 的表号与式号未标注引用版本，跨版本不可定位。｜引文依据：arXiv:2406.06484 v3/v4/v5 的 caption 为 "Table 1: Results on the synthetic MAD benchmark"、"Table 2: Main language modeling results"（v6 正文写 "The results are shown in Table 4."，LM 表为 "Table 1"）；页面写的 "§4.1 Table 2（MAD）/ §4.2 Table 3（LM）" 只在 v1/v2 成立，而同一页的 "Table 4 第一行" 只在 v3–v5 成立。arXiv:2412.06464 v2/v3 中 gated delta rule 是 Eq. (10)、Mamba2 在 §2.1，只有 v1 是 Eq. 8 与 §2.2，与 C7 一致。｜修复要求：在"主要依据"或来源章节固定一个引用版本（如 NeurIPS 2024 版或某一 arXiv 版本号），并统一按该版本的表号/式号/小节号引用。｜修复：｜复验：
- [轻微·技术] C1 引用的 "Linear Attention [40] = Katharopoulos et al. 2020" 文献编号不匹配。｜引文依据：arXiv:2406.06484 v3 Table 4 首行为 "Linear Attention [47, 46]"，v6 为 "[48, 47]"；v3 中 [47] 才是 Katharopoulos et al.，[40] 是 Irie et al. 2022c。｜修复要求：按所选版本改为 [47]（v3–v5）或 [48]（v6）。｜修复：｜复验：
- [轻微·技术] F1 把术语 "retrieval error" 的出处记为 "Yang 2024 NeurIPS §2.1"。｜引文依据：Yang 2024 全文（v3 与 v6）检索 "retrieval error" 均为 0 处；该术语与 $Sk_j$ 展开式出自 Schlag 2021 §4.1（"storing more than $d_{dot}$ associations will result in a retrieval error"），页面 C2 已引到该节。｜修复要求：把 F1 中 "retrieval error" 的出处改为 Schlag 2021 §4.1（§2.2 只讨论 key "collisions"）。｜修复：｜复验：
- [轻微·技术] 第 2 章符号表把 $\beta_t$ 注为"论文中也称'学习率'"。｜引文依据：Yang 2024 §2.2 写 "$\beta_t=\sigma(\mathbf{W}_\beta\bm{x}_t)\in(0,1)$ is a soft 'writing strength'"；"learning rate" 的用法在 Yang, Kautz & Hatamizadeh §3.1（"where $\beta_t$ represents the (adaptive) learning rate"）。｜修复要求：注明该称呼出自 GDN 论文，或直接沿用"writing strength（写入强度）"。｜修复：｜复验：
- [轻微·技术] 5.4 节正文写 "GLA = 17.25 / 14.92"，而 N1 写 "GLA (w. conv) = 17.25 / 14.92"。｜引文依据：Yang 2024 Table 2 含 GLA (w/o. conv) 17.22/14.47 与 GLA (w. conv) 17.25/14.92 两行，页面取的是 w. conv 值。｜修复要求：正文补 "(w. conv)"，与 DeltaNet、Mamba 标注一致。｜修复：｜复验：
- [轻微·表述] 1.2 节最后一段用"注意 $k_2$ 未归一化时 $v_2$ 项系数是 …"直接向读者下指令。｜引文依据：不适用｜修复要求：改为陈述句，如"$k_2$ 未归一化时 $v_2$ 项系数是 $k_2^\top k_2=2$ 而不是 1"。｜修复：｜复验：
- [轻微·可读性] "GLA" 与 "LMB PPL" 首次出现（5.4 节表格与正文）均未展开。｜引文依据：不适用（Yang 2024 表列名即 "GLA"、"LMB PPL"；LMB 指 LAMBADA）｜修复要求：首次出现时给出全称（Gated Linear Attention；LMB = LAMBADA）。｜修复：｜复验：
- [轻微·格式] 来源章节 h3 "外部数字与实验条件" 缺固定后缀 "（N）"。｜引文依据：不适用（guides/concept/style-guide.md 第 1 节规定来源章节固定命名为 `论断与来源（C）`、`公式与来源（F）`、`外部数字与实验条件（N）`；同页 C、F 两节均带括号字母）｜修复要求：改为 "外部数字与实验条件（N）"。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 8
- 处置：修复（核心结论、公式、数字与引文均已回源核对通过，无阻断项；重要项为 5 章总结处对 delta 规则作用的表述强于来源，需按来源改写并补限定；8 项轻微中，表号/式号与文献编号问题建议随版本固定一并修正）
