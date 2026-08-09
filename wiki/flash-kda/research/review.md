# FlashKDA 与 KDA Context Parallelism 独立审查

- 审查者：独立上下文（AI 模拟 / 真实目标读者）
- 页面版本：index.html blob 541f09c640f0c041e197eda1e6b1e6997e36c8a8（工作树未提交，git hash-object 取值）
- 时间：2026-08-09
- 审查依据：guides/concept/check.md
- 审查对象：wiki/flash-kda/index.html、wiki/flash-kda/overview.html
- 对照来源：K3 报告 /tmp/kimi-k3-research/k3-report.txt §5.1.1（行 1168–1192）、§5.1.2（行 1194–1254）、§5.4.2（行 1605–1622）、§2.1.1 Eq.1（行 229–238）、§3.2 配置表（行 740–762）

## 段 A 盲读小结

按页面顺序通读，主线为"KDA 串行状态 vs GPU 并行 → 四 regime 各解一个"。整体结构清晰：S1 建立冲突框架与四 regime 定位表，S2–S5 各解一个 regime，每章末有自测题，来源与教学说明完整。

盲读过程中的卡点：
- S3 的"关键观察"（段转移可独立于入状态计算）是 S4 分解的前置，S3 只给结论并明确说"具体怎么分解是 S4 的核心，这里先用结论"。这是合理的叙事选择，但小白在 S3 需暂时信任该结论。
- S4 展开求和公式（行 870）在尝试用手算例子验证时，代入后得到的是页面自身明确警告的"错误直接求和"结果（详见问题 1）。这是主线上的硬卡点。
- S4 中 $S_T^{[i]}$（大写 $T$）与 $S_t^{[i+1]}$（小写 $t$）的区分未显式说明 $T$ 代表"该 rank 的完整本地 token 数"，需从上下文推断。

学习目标核对（以页面"读完你能回答"5 条为准，未读 research/scope.md）：
1. 串行状态与 GPU 冲突 + 四 regime 瓶颈不同 → S1 Eq.1 串行依赖 + 四 regime 表回答。✓
2. FlashKDA 重叠 intra-chunk 与 cross-chunk → S2 token-parallel + head-parallel 回答。✓
3. 设备内 CP 单 rank 切序列且无跨设备通信 → S3 SM 级 CP planner 回答。✓
4. KCP 为何不能直接求和 + $M+\tilde S$ 分解 + 通信量与序列长度无关 → S4 四小节回答（但展开求和公式有误，见问题 1）。✓ 目标覆盖，公式需修。
5. 解码回滚 + 状态流量不爆炸 → S5 投影输入缓存 + 片上重建回答。✓

## 段 B 对照来源小结

逐条核查结果：
1. 定义与机制：Eq.1 因式形式 $S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$、$M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$ 与 §5.1.2（行 1200–1201）逐字一致；§2.1.1（行 234）给出未因式形式，代数等价。C1–C5 五条核心论断的英文原文逐条核对，表述未扩大来源结论。✓
2. 公式与推导：Eq.17 第一形式 $S_t^{[i+1]} = \tilde S_t^{[i+1]} + M_{t\leftarrow 1}^{[i+1]} S_T^{[i]}$ 与 §5.1.2（行 1211–1216）结构一致。但**展开求和形式（行 870）连乘上限错一位**（详见问题 1）。手算例子 4 步 ground truth + KCP 3 步分解 + 误用直接求和的对比数值全部复算通过。✓（例子正确）✗（展开公式错误）
3. 可运行代码：页面无声称可运行的代码块，教学说明已显式声明"无伪代码与可运行代码"并给出理由。手算例子作为静态验证已足够。✓
4. 事实与推断：α 范围 $(e^{-5},1)$ 与 §2.1.1（行 335–340）一致；96 head、69 KDA + 24 Gated MLA 与 §3.2 配置表（行 751、757）+ §2.1.2 标题（行 352）一致；"substantially outperforms Triton""sub-linearly"为报告原文定性结论，页面只复述不数值化。但 **$d_k=d_v=128$ 在 K3 报告中无出处**（详见问题 2）。教学构造已标记。
5. 前置知识引用：KDA / GPU 执行模型 / 线性注意力三个概念页文件均存在，相对路径 `../../wiki/<name>/index.html` 层级正确。✓
6. 教学简化：手算例子参数简化（$d_k{=}d_v{=}2$、$\alpha{=}0.5$ 标量、$\beta{=}1$、单位 $k$）已标记为教学构造并说明限制；四条教学解释的失效边界已逐条列出。✓
7. 页面功能：`python3 .dojo/scripts/validate.py wiki/flash-kda/index.html` 退出码 0；overview.html 同样退出码 0。KaTeX 定界符（`$$`/`$`）、details 折叠、自动 TOC 结构正确。

## 问题

- [阻断·技术] index.html S4「KCP 的分解」展开求和公式（行 870）：公式 $\displaystyle S_T^{[i]} = \tilde S_T^{[i]} + \sum_{j=1}^{i-1}\bigl(\prod_{l=j+1}^{i-1} M_{T\leftarrow 1}^{[l]}\bigr)\tilde S_T^{[j]}$ 中连乘上限 $i-1$ 错一位，应为 $i$。用页面自身的 2-rank 手算例子验证：取 $i=2$，$j=1$ 时该式给出 $\prod_{l=2}^{1}=\text{空积}=I$，于是 $S_T^{[2]}\stackrel{?}{=}\tilde S_2 + I\cdot\tilde S_1 = \tilde S_2+\tilde S_1 = \begin{pmatrix}6&8\\10&12\end{pmatrix}$——这恰是页面在 details 折叠块（行 951）明确标注为"错误做法"的直接求和结果，而非正确值 $\begin{pmatrix}5.5&7\\8.5&10\end{pmatrix}$。正确展开应为 $\prod_{l=j+1}^{i}$（含 rank $i$ 自身的 $M_{T\leftarrow 1}^{[i]}$），此时 $i{=}2,j{=}1$ 给出 $M_{T\leftarrow 1}^{[2]}\tilde S_1 = 0.5I\cdot\tilde S_1$，与正文前一行（行 866）"$M_{t\leftarrow 1}^{[i+1]}$…不需要入状态"及手算例子的 prefix scan 步骤（行 935，$S_T^{[2]}=\tilde S_2+M累积_2\cdot S_T^{[1]}$）一致。页面自己的例子用的是正确形式，写出来的展开公式却与此矛盾。｜ 修复：已将连乘上限从 $i-1$ 改为 $i$（index.html 行 870），与 evidence.md 公式及手算例子 prefix scan 步骤一致 ｜ 复验：
- [重要·来源] index.html S1（行 690）、S4（行 898）、overview.html（行 70）：页面多次出现"K3 配置 $d_k=d_v=128$，约 32KB"作为事实陈述，但 K3 报告 §3.2 配置表（行 740–762）只列 Hidden Dimension 7168、Attention Heads 96、Attention-Layer Composition 69 KDA+24 MLA 等，并未列出 KDA 的 head dimension $d_k$/$d_v$；§2.1.1（行 232）只写 $S_t\in\mathbb{R}^{d_k\times d_v}$ 不给数值；§5.1（行 1163）同样只写 $S\in\mathbb{R}^{d_k\times d_v}$。报告全文检索"128"仅命中 Training Context Length 128K、Math-Vision 引用 [128]、量化 group size 128，均与 KDA head dim 无关。"$d_k=d_v=128$"无可在 K3 报告中定位的依据，由此派生的"约 32KB"通信量数字同源受影响。该数字用于支撑 S4"通信量固定"的定量说明（行 898）及 overview 的核心结论段（行 70），若数值有误会误导。注：核心论断"通信量与序列长度无关"是定性结论，不依赖 $d_k$ 具体值，故不阻断；但具体数字需有出处或去掉。｜ 修复：已将 S1（行 690）、S4（行 898）的「K3 配置」改为「config.json」/「config.json 中」，overview（行 70）加「config.json」标注。$d_k=d_v=128$ 来源为 KDA 概念页 evidence.md N5（HuggingFace config.json `linear_attn_config.head_dim=128`），经 KDA 页 review.md 逐项核对确认 ｜ 复验：
- [轻微·盲读] index.html S4 展开求和公式引文（行 868）："把这个分解递推到所有 rank，$t=T_{i+1}$ 时离开 rank $i$ 的状态可以展开成…"——$T_{i+1}$ 是 rank $i{+}1$ 的本地 token 数，而"离开 rank $i$ 的状态"$S_T^{[i]}$ 是 rank $i$ 跑完其 $T_i$ 个 token 后的状态，应在 $t=T_i$ 时取值。下标 $T_{i+1}$ 与"rank $i$"不匹配，小白读到此会困惑"$T_{i+1}$ 从何而来"。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 1
- 处置：进入修复。阻断项为公式连乘上限修正（$i{-}1\to i$），属确定范围的定点修复，不涉及研究范围或教学大纲变更。重要项为补充 $d_k{=}d_v{=}128$ 的出处或去掉无来源的具体数值。轻微项为下标订正。修复后需重新跑 validate.py 并复验展开公式与手算例子的一致性。
