<!-- review-meta
round: 5
page: wiki/delta-rule/index.html
reviewed_content_sha256: dd8c26837d32f4b6
-->
# Delta 规则与 DeltaNet 审查记录（第 5 轮）

- 页面版本：92f645b860485b10b0a33e8bbf40f232754f6052
- 审查时间：2026-09-13 20:10 CST
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题、最容易误解、1. 为什么线性注意力会"记不清"、2. Delta 规则的紧凑公式与手算、3. 等价改写、4. 边界情况与 $\beta_t$ 的退化、5. DeltaNet 与相邻模型对比、来源与范围说明（含全部 details 折叠块、SVG 图注、代码块）

## 问题

- [阻断·技术] 第 2 章「本章问题」第 1 题解答（index.html:315）：题设 $k_t=(0,1)^\top$、$v_t=(2,0)^\top$、$\beta_t=1$，解答把 $v_tk_t^\top$ 写成 $\begin{pmatrix}0&0\\2&0\end{pmatrix}$，与全页 $vk^\top$ 的约定（第 $i$ 行第 $j$ 列 $=v_i k_j$）不符——这实际是 $k_tv_t^\top$ 转置后的结果。按约定 $\begin{pmatrix}2\\0\end{pmatrix}\begin{pmatrix}0&1\end{pmatrix}=\begin{pmatrix}0&2\\0&0\end{pmatrix}$，故 $S_t$ 应为 $\begin{pmatrix}0&2\\0&0\end{pmatrix}$，页面写成 $\begin{pmatrix}0&0\\2&0\end{pmatrix}$ 错误（summary 结论同错）。｜引文依据：同页 §2.2 用同一约定算得 $v_2k_2^\top=(0,1)^\top(1,0)^\top=\begin{pmatrix}0&0\\1&0\end{pmatrix}$，与 numpy `np.outer` 一致；据此 $(2,0)^\top(0,1)=\begin{pmatrix}0&2\\0&0\end{pmatrix}$。｜修复要求：把该题解答与 summary 中的 $v_tk_t^\top$、$S_t$ 改为 $\begin{pmatrix}0&2\\0&0\end{pmatrix}$。｜修复：｜复验：

- [阻断·技术] §5.4 正文引文（index.html:606）：把来源句写成 "DeltaNet is better at recalling tasks, especially on Fuzzy Recall as expected, although it **somehow** struggles on the 'Memorize' task" 并加引号。原文该句不含 "somehow"，是页面自行加词后包装成来源引文（来源在此并未表达"莫名/费解"的语气）。｜引文依据：arXiv:2406.06484 v3 §4.1 原文 "Compared with other architectures, including MHA, DeltaNet is better at recalling tasks, especially on Fuzzy Recall as expected, although it struggles on the "Memorize" task."（arxiv.org/html/2406.06484v3 与 ar5iv 多次抓取均无 "somehow"；逐字核对确认为 No）。｜修复要求：删除引文中的 "somehow"，使引号内文字与 v3 原文逐字一致；若要保留评价语气，须移出引号改为明确标注的页面判断。｜修复：｜复验：

- [重要·技术] 来源 C6（index.html:735）：称 WY 表示论文 "Yang 引用为 [9]"。[9] 实为 xLSTM，WY 表示是 [11]，定位编号错误。｜引文依据：v3 参考文献 [9] "M. Beck, K. Pöppel, M. Spanring, ... xlstm: Extended long short-term memory. arXiv:2405.04517, 2024."；[11] "C. H. Bischof and C. V. Loan. The WY representation for products of householder matrices. In SIAM Conference on Parallel Processing for Scientific Computing, 1985."；§3.1 正文 "the compact WY representation [11]"。年份 1985 与标题无误，仅编号错。｜修复要求：把 C6 中 "Yang 引用为 [9]" 改为 "[11]"。｜修复：｜复验：

- [轻微·表述] 来源 C6（index.html:735）：「该句措辞在 arXiv v2 与 v3 起有所不同」一句语法不通（"起"误用），且属跨版本比对的编辑过程说明，读者无法据此定位或核对。｜引文依据：不适用｜修复要求：改为可核对表述（如「引自 arXiv:2406.06484 v3 §3.3」），或删除该分句。｜修复：｜复验：

- [轻微·技术] 来源 C9（index.html:741）：把「官方 config.json / 源码」列为来源，但未给出仓库与文件路径、行号，不满足来源可定位要求（规范 §2.2.1 要求源码路径与行号）。该条最小事实"基于 delta 规则递归"可由技术报告 §1 直接核对，config.json 属无定位的多余来源。｜引文依据：不适用｜修复要求：补充可定位的仓库路径，或删除「官方 config.json / 源码」只保留技术报告这一可核对来源。｜修复：｜复验：

## 已核对（本轮通过，附依据）

- 手算全部复算一致：§1.2 碰撞例（$S=\begin{pmatrix}1&1\\1&2\end{pmatrix}$、$Sk_2=(2,3)^\top$、差值 $(2,1)^\top=v_1+v_3$）、§2.2（$S_2=\begin{pmatrix}0&0\\1&0\end{pmatrix}$ 且 $S_2k_2=v_2$）、§4.3（$\beta_2=0.5$ 得 $S_2=\begin{pmatrix}0.5&0\\0.5&0\end{pmatrix}$）。第 2 章本章问题第 1 题除外（见上）。
- 代码：页面 Python 代码实际运行，逐行输出与页面「预期输出」完全一致（含 Equivalent form 与 `Matches compact form: True`）。
- C1/F1：v3 §5.1 Table 4 第一行 Linear Attention $S_t=S_{t-1}+v_tk_t^\top$、readout $o_t=S_tq_t$、引用 [47]=Katharopoulos et al. 2020 "Transformers are RNNs" 均核对；Schlag 2021 §4.1 "storing more than d_dot associations will result in a retrieval error" 原文命中。
- C2：v3 §2.2 "a purely additive update rule makes it difficult to deallocate past key-value associations, eventually leading to key 'collisions' when L>d, as pointed out by Schlag et al." 与页面一致。
- C3/C4/C5：v3 §2.2 $S_t=S_{t-1}-\boldsymbol{v}_t^{old}k_t^\top+\boldsymbol{v}_t^{new}k_t^\top$、$v_t^{old}=S_{t-1}k_t$、$v_t^{new}=\beta_tv_t+(1-\beta_t)v_t^{old}$、soft "writing strength"、§2.2 标题 "DeltaNet: Linear Transformers with the Delta Update Rule" 均命中；Schlag 2021 §4.2 Eq.23 [write]−[remove] 形式与 §1 "akin to the famous error-correcting delta-rule"、Widrow & Hoff 1960 均命中。
- C6（正文部分）：v3 §3.1 "generalized Householder transformation"、§3.3 "𝐈−𝒌t𝒌t𝖳 becomes a projection matrix, erasing information in one subspace while preserving the other d−1 subspaces" 命中。
- C7/F3：arXiv:2412.06464 v3 §3.1 Eq.10 $S_t=S_{t-1}(\alpha_t(I-\beta_tk_tk_t^\top))+\beta_tv_tk_t^\top$、§2.1 $S_t=\alpha_tS_{t-1}+v_tk_t^\top$（$\alpha_t\in(0,1)$）、"gating enables rapid memory erasure while the delta rule facilitates targeted updates"、$\alpha_t\to0$ 清空 / $\alpha_t\to1$ 退化为 pure delta rule 均命中；§2.2 亦称 $\beta_t$ 为 "writing strength"，§3.1 称其为 "(adaptive) learning rate"。
- C8/N2：MAD 表 1 全部数字命中（DeltaNet 100/35.7/100/100/52.8/42.2/Avg 71.8；Mamba 90.4/6.7/90.1/86.3/89.5/52.7/69.3；GLA 80.8/6.9/81.6/88.6/63.3/38.8/60.0；Transformer 94.1/29.8/86.8/99.6/85.2/51.6/74.5），且 Average=6 项均值可复算，页面"含未展示 Compress 任务"的说明属实。
- N1：v3 §4.2 Table 2 命中——DeltaNet (w. conv) 16.87/12.21；Mamba (w. conv) 17.06/13.89；GLA (w. conv) 17.25/14.92；Transformer++ 16.85/13.44（该表同时有 GLA (w/o. conv) 17.22/14.47，页面取 w. conv 正确）。
- C9：arXiv:2607.24653《Kimi K3: Open Frontier Intelligence》存在；正文确认 KDA（Kimi Delta Attention）"extends the delta-rule recurrence with a channel-wise forget gate"，与页面引用一致。
- 页头：validate.py 通过；dojo:type=concept、dojo:topics=注意力机制（在允许大类内）、dojo:tag=注意力（在词表内）；前置概念页 wiki/linear-attention/ 真实存在；overview.html 与 index.html 双向链接；正文无指向 research/ 的路径；SVG 图内公式由 foreignObject + KaTeX 承载，无 Unicode 数学字符替代。

## 结论

- 统计：阻断 2 / 重要 1 / 轻微 2
- 处置：修复（问题均可在原位修复，无需返回规划；两处阻断须关闭后方可发布）