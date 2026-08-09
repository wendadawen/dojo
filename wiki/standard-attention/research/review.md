# 标准 Transformer 注意力 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源复算）
- 页面版本：index.html blob `5491c34` / overview.html blob `2f5cf1f`（工作树未提交文件）
- 时间：2026-08-09 15:04 +0800
- 来源核对：Vaswani et al. 2017, "Attention Is All You Need", arXiv:1706.03762（arxiv HTML v7 逐字核对 §3.2.1 / §3.2.2 / §3.2.3 / §4 Table 1 / 脚注）

## 段 A 盲读小结

按页面顺序通读 index.html 五个正文章节（why-attention → scaled-dot-product-formula → why-sqrt-dk → multi-head-attention → complexity-and-boundaries）与 overview.html。主线整体清晰：从 RNN 局限引入注意力动机 → 拆解缩放点积公式四步 → 方差推导解释 √d_k → 多头拼接与参数等价 → 复杂度瓶颈与边界。四条学习目标均由正文章节完整回答（目标 1↔§why-attention，目标 2↔§scaled-dot-product-formula+§why-sqrt-dk，目标 3↔§multi-head-attention，目标 4↔§complexity-and-boundaries）。折叠块（softmax 数值稳定、方差推导完整步骤、softmax 雅可比、d_k=64 饱和数字、3×3 遮罩例子）均为补充，收起后正文主线完整。

## 段 B 对照来源小结

核心公式 F1（§3.2.1 Eq.(1)）、F2（§3.2.2 Eq.(2)）、F5（§3.2.3 masking = −∞）、方差推导（脚注：q·k 方差 = d_k）、Table 1 复杂度（RNN O(nd²)/O(n)/O(n)，CNN O(knd²)/O(1)/O(log_k n)，Self-Attn O(n²d)/O(1)/O(1)）、超参数（d_model=512, h=8, d_k=d_v=64）均与 arxiv HTML v7 逐字一致。2×2 教学例子（A=[[0.670,0.330],[0.330,0.670]], AV=[[1.66,2.66],[2.34,3.34]]）与 3×3 遮罩例子（A3=[[1,0,0],[0.401,0.599,0],[0.258,0.316,0.426]]）经 Python 复算全部正确。softmax 雅可比 ∂p_i/∂z_j = p_i(δ_ij − p_j) 推导正确。下方问题清单列出不一致项。

## 问题

- [重要·盲读] index.html §why-attention 第 1 段（line 689）："The cat sat on the mat because it was tired." 中 "it" 是第 8 个词（The=1, cat=2, sat=3, on=4, the=5, mat=6, because=7, it=8），页面写"直到'it'所在的第 7 步才读到"，且同句使用 h_1（1-indexed），按 1-indexed "it" = h_8：将"第 7 步"改为"第 8 步" ｜ 修复：已将"第 7 步"改为"第 8 步"（it 是第 8 个词）。 ｜ 复验：
- [重要·技术] index.html §why-sqrt-dk "不缩放的后果"段（line 846）："e^16 ≈ 9.9×10^6" 数值错误，e^16 = 8886110.52 ≈ 8.89×10^6（Python 复算确认）；同句"相邻 logit 之间差值常达 16+"为具体数字但无推导或来源（论文脚注只给 Var=d_k，未给 logit 差值；jethroodeyemi 来源只讨论 max weight 与 entropy，未给"相邻差值 16+"）：修正 e^16 ≈ 8.89×10^6；为"16+"补充推导（如 n=256、σ=8 时 E[max logit] ≈ σ·√(2·ln n) ≈ 26.6，故 max 与典型 logit 差值可达 16+）或改为不涉及具体数值的表述（如"logit 差值常达十以上"） ｜ 修复：e^16 修正为 8.89×10^6；"相邻 logit 之间差值常达 16+"改为"n 个 key 中最大 logit 与典型值差值常达 16+（如 n=256 时 E[max]≈8√(2 ln 256)≈27）"。 ｜ 复验：
- [重要·技术] index.html §complexity-and-boundaries Flash vs Linear 紫色 callout（line 998-1005）：声称"Flash Attention……结果与标准注意力数值等价，但快 2-4 倍""Linear Attention……复杂度从 O(n²d) 降到 O(n·d²)"，这些是 Vaswani 2017 之后的具体事实论断，Vaswani 论文不涉及 Flash/Linear，且"来源与教学说明"章节未列任何 Flash/Linear 出处：为"2-4 倍""数值等价""O(n·d²)"补充来源（如 Dao et al. 2022 FlashAttention、Katharopoulos et al. 2020 Linear Attention），或在 callout 内标注"以下为后续工作的概括，非 Vaswani 论文论断" ｜ 修复：callout 标题加注"以下为 Vaswani 2017 之后的后续工作概括，非原论文论断"；Flash 条目加"（Dao et al. 2022, FlashAttention）"，Linear 条目加"（Katharopoulos et al. 2020）"。 ｜ 复验：
- [重要·技术] overview.html §"关键结论与边界"（line 74）与 index.html line 1002 不一致：overview 写"Linear Attention……复杂度 O(n)"，index 写"复杂度从 O(n²d) 降到 O(n·d²)"。Linear Attention 的严格复杂度是 O(n·d²)（核分解后 Q(φ(K)^T φ(V))），overview 丢掉 d² 因子可能误导读者认为对所有维度都是线性：两处统一为 O(n·d²)，或在 overview 加注"d 视为常数时为 O(n)" ｜ 修复：overview.html L74 的"复杂度 O(n)"改为"复杂度 O(n·d²)，d 视为常数时为 O(n)"，与 index.html 一致。 ｜ 复验：
- [轻微·盲读] index.html §why-attention（line 696 vs 710）：line 696 已使用"查询向量 q""键向量 k""值向量 v"，line 710 又称"这里出现三个新词——查询（query, q）、键（key, k）、值（value, v）"，术语先使用后引入：将 line 710 改为"上面出现的三个词——查询、键、值——它们的角色可以借用数据库类比来理解"，或在 line 696 首次使用时即给出完整中英文名 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html §scaled-dot-product-formula 章末检查题（line 813-815）："算出 2×2 例子中 QK^T（[[1,0],[0,1]]）与 softmax 后的 A（[[0.670,0.330],[0.330,0.670]]）"，题目未提及 ÷√d_k 步骤，读者若直接对 [1,0] 做 softmax 会得 [0.731,0.269]（教学说明 line 1071 已给出未缩放值），与答案 [0.670,0.330] 不符：在检查题中补"（经 ÷√d_k 缩放后）"或将题目拆为两步 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html §complexity-and-boundaries（line 1007、1009）："RoPE（旋转位置编码，见概念页 RoPE）""MLA（Multi-head Latent Attention，见概念页 MLA）"未标注"（待生成）"，而同页 line 721/755/759 的矩阵乘法、向量点积、softmax 均标"（待生成）"，标注不一致：为 RoPE、MLA 补"（待生成）"或确认页面已存在后改为有效链接 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 学习目标第 4 条（line 663）："标准注意力本身不解决的三个问题（位置、KV 压缩、复杂度）"，但 §complexity-and-boundaries 适用边界表（line 1019-1023）列出 4 项"不能解决"（复杂度、位置、KV 压缩、多头冗余），目标与正文计数不一致：将"三个"改为"四个"并在括号内补"多头冗余"，或说明前三项为核心边界、多头冗余为次要 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 4
- 处置：进入修复（阻断为 0，重要问题均可通过定点修改关闭，无需改变研究范围或教学大纲）
- 备注：核心公式（F1/F2/F5）、方差推导、softmax 雅可比、2×2 与 3×3 教学例子数字、Table 1 复杂度、论文超参数均经复算与来源核对一致；未发现核心机制或定义错误。页面功能（KaTeX 渲染、details 折叠、TOC 锚点）静态检查无异常，validate.py 机械项需在修复后由修复者运行确认。
