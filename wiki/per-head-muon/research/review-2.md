# Per-Head Muon 独立审查（第二轮）

- 审查者：独立上下文（AI 模拟 / 小白读者视角）
- 页面版本：index.html @ ac5b744（2026-08-09）
- 时间：2026-08-09
- 审查范围：段 A 盲读（index.html + overview.html）+ 段 B 对照来源（K3 报告 §2.5 "Per-Head Muon" 行 667–678 + §5.2.2 "P2P-based Muon orthogonalization" 行 1393–1400；Muon 正交化定义沿用 Keller Jordan 原始博客）

## 段 A 盲读

按页面顺序阅读 index.html，扮演完全小白读者，记录理解主线上的卡点。

**S1（为什么整块正交化对多头不够好）**：前置概念 Muon 和 Newton-Schulz 用占位链接引用，给出最小结论（正交化 = SVD 后取 UV^T、奇异值拉平为 1）。权重矩阵沿头维度堆叠的结构 W=[W1;...;WH] 用 ASCII 图展示。问题解释（U 混合所有头、大尺度头主导、小尺度头未充分正交化）清晰。2×2 极小例子 M=[[3,4],[0.3,0.4]] 手算展示"全矩阵正交化后小头行块范数仅约 0.1"。折叠块含完整 SVD 推导。小白可跟上。

**S2（按头切分动量矩阵并分别正交化）**：核心改动一句话讲完。形式化表达清晰。切分维度澄清（切头维度，不切 d_h 内部或模型维度 d）。同一例子续算：按头正交化后两头行块范数均为 1。折叠块含伪代码（非 Python，标注为伪代码）。NS 算法本身未改、只改作用对象——明确说明。小白可跟上。

**S3（均衡更新尺度与开销变化）**："均衡"均衡的是幅度不是方向——明确。三个定性结论（更均衡学习动力学、提升大尺度稳定性、略降开销）用表格列出，附 K3 报告原文与成立条件。开销分析：Gram 矩阵规模对比 H×(d_h²d) vs (Hd_h)²d，比值 ≈1/H。折叠块展开。边界提醒：C3 是定性判断，无消融数字。"乐器混音轨"类比标注只解释"尺度耦合"局部关系。小白可跟上。

**S4（分布式实现中的 P2P 参数取回）**：前置（ZeRO 式分片、NS 需完整矩阵）清晰。朴素 all-gather 两个问题（显存大、通信瓶颈）用 ASCII 图对照。P2P 方案（每 rank 只取本地所需分片）+ 流水化隐藏通信。边界提醒：P2P 降低而非消除通信。小白可跟上。

**学习目标核对**：
1. 为什么整块正交化让大尺度头主导、小尺度头更新不足 → S1 完整回答 ✓
2. 按头正交化在机制上做了什么改变 → S2 完整回答 ✓
3. 如何均衡各头更新尺度，带来什么实践效果 → S3 完整回答 ✓
4. 分布式训练中如何避免全参数 all-gather → S4 完整回答 ✓

段 A 未发现阻断或重要卡点。

## 段 B 对照来源

逐条核对页面表述与 K3 报告 §2.5（行 667–678）及 §5.2.2（行 1393–1400）的一致性。

**定义与机制**：
- C1（全矩阵正交化耦合各头）：报告 §2.5 "full-matrix orthogonalization treats all heads as a single coupled block, so heads with larger gradient or momentum scales dominate the shared update direction, while smaller-scale heads receive insufficiently normalized updates" ✓
- C2（按头切分、每头块单独正交化、均衡各头更新尺度）：报告 §2.5 "we partition their momentum matrices along the head dimension and orthogonalize each head's block separately ... per-head orthogonalization equalizes the update scale across heads" ✓
- C3（更均衡学习动力学、提升大尺度稳定性、略降开销）：报告 §2.5 "this design yields more balanced learning dynamics across heads and improves training stability at larger scales. It also slightly reduces optimizer overhead, as Newton–Schulz iterations on tall per-head blocks are cheaper than on the full projection matrix" ✓
- C4（朴素 all-gather 的内存与通信问题）：报告 §5.2.2 "The naive approach performs an all-gather over the entire parameter buffer on every rank, which incurs a substantial memory footprint on top of making communication the primary bottleneck at scale" ✓
- C5（P2P 取回、消除全参数缓冲区、流水化）：报告 §5.2.2 "each rank retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication ... eliminating the full-parameter buffer and reducing both memory usage and communication volume. Communication and computation are further pipelined at the granularity of model-chunk buffers" ✓

**公式与推导**：
- F1（动量矩阵按头堆叠与逐头正交化）：切分方式由 C2 直接给出；记号 M=[M1;...;MH]、每块 M_h∈R^{d_h×d} 为页面教学标注 ✓
- F2（正交化 = SVD 后取 UV^T）：Muon 原始博客（Keller Jordan, 2024-12）定义 ✓
- 极小例子手算验证：M=[[3,4],[0.3,0.4]]，MM^T=[[25,2.5],[2.5,0.25]]，迹 25.25、行列式 0，σ1=√25.25≈5.025，u1≈[0.995,0.0995]，v1=[0.6,0.8]，Ortho(M) 行 1 范数≈0.995、行 2 范数≈0.0995。逐项复算一致 ✓

**可运行代码**：页面含伪代码（非 Python），标注为伪代码，不需执行 ✓

**事实与推断**：
- 页面明确标注 C3 的三个结论是定性判断，报告未给 per-head 单独贡献的消融数字 ✓
- 页面明确标注 Moonlight 技术报告的 ~2× 效率是整体 Muon 结果，不归因于 per-head，本页不引用以避免误导 ✓

**前置知识引用**：Muon 优化器、Newton-Schulz 正交化均用占位链接引用并给最小结论 ✓

**教学简化**：极小矩阵 H=2/d_h=1/d=2 标注为教学构造；只展示平行同向极端情形；NS 系数 (3.4445, -4.7750, 2.0315) 和 5 步迭代标注为 Newton-Schulz 概念页内容；S4 P2P 图示简化标注 ✓

**页面功能**：validate.py 退出码 0 ✓

## 问题

- [轻微·盲读] S3「均衡更新尺度与开销变化」第一段："原版只保证整块矩阵的奇异值拉平，各头之间仍保留原始动量尺度的比例（大尺度头行块范数大、小尺度头行块范数小，当各行块行空间有重叠时；正交时不保留该比例、各头行块范数均为 1）"：一个括号内并列两种情形（行空间重叠时保留比例 vs 行正交时不保留且各头范数均为 1），读者需在单句内切换两种条件分支，首次阅读时容易漏掉"正交时各头范数均为 1"这一边界情形，影响对 S1 自检题"两行正交时两头行块范数是否都接近 1"的理解铺垫：将该括号拆为两句独立陈述（先说"行空间有重叠时保留比例"，再说"行正交时不保留比例且各头范数均为 1"），或在括号前用"具体而言"引导 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（轻微问题接受：不影响核心正确性和主线理解，仅降低单句可读性；S1 自检题已覆盖"正交时各头范数为 1"的边界情形，读者可通过自检题验证理解）

段 A 盲读未发现阻断或重要卡点，学习目标全部由正文章节完整回答。段 B 对照来源逐条核对，核心论断（C1–C5）、公式与手算（F1–F2、极小例子 SVD 推导逐项复算一致）、来源事实均一致。伪代码标注正确无需执行。validate.py 退出码 0。关键论断已重新对照外部来源。
