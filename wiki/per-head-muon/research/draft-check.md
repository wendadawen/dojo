# Per-Head Muon 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成，规划阶段完成条件全部满足（4 个学习目标、6 项核心内容、5 条核心论断、前置知识映射含 2 个占位概念页、误解与边界齐备）。

## 大纲落实

- 章节顺序：S1（为什么整块正交化对多头不够好）→ S2（按头切分动量矩阵并分别正交化）→ S3（均衡更新尺度与开销变化）→ S4（分布式实现中的 P2P 参数取回）→ 文末来源与教学说明。与 outline.md 一致。
- 学习目标：4 个（Q1–Q4），写入 learning-goals 组件，与 scope.md §1.2 一致。
- 前置知识：muon-optimizer、newton-schulz 在 S1 首次依赖时给出占位链接（`../muon-optimizer/index.html`、`../newton-schulz/index.html`），对应占位页面已创建使链接可用。多头注意力结构事实一句话说明不链接。
- 贯穿例子：S1 引入（M1=[3,4]、M2=[0.3,0.4]，全矩阵正交化后小头行块范数≈0.0995）；S2 复用（按头正交化后两头行块都为 [0.6,0.8] 范数 1）；S3 对照表引用。与 outline.md §4 一致。
- 误解和边界：scope.md §1.6 的 4 条误解在正文处理——"Per-Head Muon 不是新优化器"（S2 机制澄清）、"NS 算法本身没变"（S2）、"压低的是幅度不是方向"（S1 正文）、"P2P 不是消除通信"（S4 边界提醒）。适用边界在 S3（C3 定性结论、未给消融数字）与 S4（P2P 收益取决于布局）处理。
- 过渡：S1→S2（"下一章看按头切分再正交化如何修复"）、S2→S3（"带来什么效果、开销怎样变化，下一章讲"）、S3→S4（"分布式训练里 NS 还面临一个工程问题，下一章讲"）、S4→文末（全文总结）。与 outline.md 一致。

## 学习目标闭环

- Q1（全矩阵正交化为何让小尺度头更新不足）：S1 正文完整回答——全矩阵 SVD 混合所有头、大尺度头主导奇异向量、小尺度头行块未获单位范数。手算例子在折叠块补充。正文不依赖折叠块即可回答。
- Q2（按头正交化的机制改变）：S2 正文完整回答——沿头维度切分成 H 块、每块单独 NS、每头块内部奇异值拉平为 1、各头互不耦合。伪代码在折叠块补充。正文不依赖折叠块即可回答。
- Q3（如何均衡各头更新尺度与效果）：S3 正文完整回答——每头独立正交化使各头行块范数趋于 1、C3 三个定性结论、开销略降因 Gram 矩阵更小。Gram 展开在折叠块补充。正文不依赖折叠块即可回答。
- Q4（分布式如何避免全参数 all-gather）：S4 正文完整回答——NS 需要完整矩阵、朴素 all-gather 两个问题、P2P 取回本地所需分片、流水化隐藏通信。无折叠块，正文完整。
- 全部目标由正文章节完整回答，无折叠块独占。

## 代码运行

- 无可运行代码块。核心机制是矩阵正交化的尺度效应，由手算例子（F3）验证。大纲明确不安排可运行代码（NS 本身是前置概念页职责，加入完整 NS 实现会偏离 per-head 主题）。
- 手算例子已用 Python 重新计算验证：σ1=√25.25≈5.025，全矩阵正交化后头 1 行块范数≈0.995（接近 1），头 2 行块范数≈0.0995（远小于 1），比值 10:1；按头正交化后两头行块都为 [0.6,0.8] 范数 1。与页面描述一致。
- 完成检查中的对照例子 M1=[1,0]、M2=[0,2]（两行正交，全矩阵正交化后两头行块范数都接近 1）已验证正确。

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/per-head-muon/index.html`
- 结果：`validation ok: wiki/per-head-muon/index.html`（退出码 0）
- 命令：`python3 .dojo/scripts/validate.py wiki/per-head-muon/overview.html`
- 结果：`validation ok: wiki/per-head-muon/overview.html`（退出码 0）
- 占位符/模板标记检查：`grep -nE '【.*】|@content|@component|TODO|TBD'` 无匹配。

## 公式渲染与交互

- KaTeX 公式（$$...$$ 与 $...$）由外壳脚本自动渲染，符号与 glossary.md 一致：$M$、$M_h$、$U$、$S$、$V^\top$、$\mathrm{Ortho}(\cdot)$、$\sigma_1$、$H$、$d_h$、$d$。
- 折叠块（details）用于手算展开、伪代码、Gram 展开三处，收起时正文仍完整回答 Q1–Q4。
- 侧边目录、章节折叠按钮、j/k 快捷键由外壳脚本处理；h2 id 稳定唯一（why-full-matrix-fails、per-head-mechanism、balanced-scale-and-overhead、p2p-distributed、sources-and-teaching-notes）。
- 占位概念页 muon-optimizer/index.html、newton-schulz/index.html 已创建，链接可用。

## 写作偏差

- 无偏差。未改变大纲、未增删核心章节、未更换贯穿例子、未把正文必要内容移入折叠块、未使用证据不足论断。
- 规划阶段发现一处需在写作中补充：muon-optimizer 与 newton-schulz 的占位链接需要目标文件存在才能通过 validate.py 的本地引用检查；已在 wiki/muon-optimizer/ 和 wiki/newton-schulz/ 创建最小占位 HTML 页面（标注"待生成"），使链接可用。这不改变大纲，属于本地修正。
