# NoPE 独立审查（第二次）

- 审查者：独立上下文（AI 模拟小白读者 / 第二次审查，未参与生成与第一次审查）
- 页面版本：index.html ac5b7442 / overview.html ac5b7442（HEAD bcb38d6，工作树无未提交修改）
- 时间：2026-08-09 16:59 +0800
- 输入：wiki/nope/index.html、wiki/nope/overview.html、NoPE 论文摘要（WebSearch arXiv:2305.19466）、Kimi K3 技术报告 §2.1.2 与 §3.4（/tmp/kimi-k3-research/k3-report.txt）
- validate.py：`python3 .dojo/scripts/validate.py wiki/nope/index.html` → exit 0（`validation ok`）

## 段 A 盲读

按页面顺序通读 overview.html 与 index.html。学习目标 5 条全部由正文章节回答（第1章对照表+第2章定义回答目标1；第3章回答目标2；第4章回答目标3；第5章回答目标4；第6章回答目标5）。

主线卡点集中在第3章：贯穿全文的问题（index.html L683）声明"三个内容完全相同的 token"，但第3章第二段（L735）断言"即使三个 token 的内容完全相同……注意力输出也会不同"，而紧接的教学示例（L745）却设 $v_1=2, v_2=4, v_3=6$ 不同。小白按"内容相同"自己套公式 $o_t=\sum_{i=1}^t\alpha_{t,i}v_i$ 会得到 $o_1=o_2=o_3=v$（相同），与页面论断冲突；又看到示例 value 不同，会产生"内容相同为何 value 不同"的困惑。其余章节主线顺畅。

## 段 B 对照来源

NoPE 论文摘要（WebSearch 多源一致）逐句核对页面 C1–C4、F1：定义、长度泛化结论、SGD 下主要类似 T5 相对 PE、无额外计算——均与摘要一致；K3 报告 §2.1.2（L352–375）与 §3.4（L782–804）逐句核对页面 C5、C6、F2、N1、N2：MLA 用 NoPE、KDA 提供位置、外推 1M、8K→64K→256K→1M、$g_{\min}=-5$、衰减因子公式——均与来源一致。前置概念页 KDA、linear-attention 链接目标文件存在。详见下方问题。

## 问题

- [阻断·盲读] index.html L683（贯穿全文的问题）与 L735（第3章第二段）：L683 声明"三个内容完全相同的 token 喂给一个不加任何位置编码的注意力，三个位置的输出会一样吗？这个问题会在第三章用手算回答"；L735 断言"即使三个 token 的内容完全相同、没有任何位置编码，不同位置因为可见的 token 数量不同，注意力输出也会不同"。但按页面自己的因果注意力公式 $o_t=\sum_{i=1}^{t}\alpha_{t,i}v_i$（L739）与 NoPE 设定（$q,k,v$ 只来自内容），当三个 token 内容完全相同时 $v_1=v_2=v_3=v$，于是 $o_1=v,\ o_2=(v+v)/2=v,\ o_3=(v+v+v)/3=v$，三个位置输出相同——与 L735 的论断相反。而 L745 教学示例又设 $v_1=2,v_2=4,v_3=6$ 不同（即内容不同），等于用"内容不同"的设定回答"内容相同时输出是否相同"的贯穿问题，设定与问题不一致。读者按"内容相同"设定自算会得到"输出相同"，与页面论断矛盾，可能误以为"因果掩码能让内容相同时输出不同"，对 NoPE 核心机制形成错误结论：把 L683 改为"三个内容不同的 token 喂给一个不加任何位置编码的注意力，三个位置的输出会一样吗？"；把 L735"即使三个 token 的内容完全相同、没有任何位置编码"改为"即使没有任何位置编码"（删除"内容相同"设定），使教学示例（value 不同）与贯穿问题一致；并在 L753"全程没有加任何位置编码"后补一句"这里三个 token 的 value 不同，所以不同位置的加权平均不同"以显式连接设定与结论 ｜ 修复：已按建议三处修改——L683 贯穿问题改为"三个内容不同的 token"；L735 删除"内容完全相同"设定改为"即使没有任何位置编码"；L753 补"这里三个 token 的 value 不同（$v_1=2,v_2=4,v_3=6$），所以不同位置的加权平均不同"。贯穿问题、第3章论断与教学示例现已自洽。 ｜ 复验：
- [轻微·技术] index.html L793（第4章对照表"T5 相对 PE"行的"额外计算"列）：表格写"有（相对偏置，训练/推理更慢）"。论文摘要只直接支持"NoPE outperforms other explicit positional encoding methods while requiring no additional computation"——即 NoPE 无额外计算、其他显式方法有额外计算；"训练/推理更慢"是从"有额外计算"推出的性能判断，摘要未直接支持该具体表述，页面"来源与教学说明"也未为其单独标注来源：把 L793 该格改为"有（相对偏置）"，删除"训练/推理更慢" ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html L816（第5章首句"Gated MLA"首次出现）：MLA 全称"Multi-head Latent Attention"在正文未给出，K3 报告 §2.1.2 首句（k3-report.txt L353）明确写出"Multi-head Latent Attention (MLA), introduced in DeepSeek-V2"。页面 C5 引用也只引"NoPE to all MLA layers"部分，未带全称。小白不知道 MLA 是什么的缩写：在 L816"Gated MLA"后补"（Multi-head Latent Attention，多头潜在注意力）"或在该句后补一句最小定义 ｜ 修复： ｜ 复验：
- [轻微·盲读] overview.html L67 与 index.html L654/L685 等多处："decoder-only Transformer"首次出现处未给最小定义。因果掩码虽已标注"概念页待生成"，但 decoder-only 本身是 NoPE 成立条件的关键术语，首次出现处应有一句最小说明（如"只看左侧 token 的自回归 Transformer"）：在 index.html L685 第1章首次出现"decoder-only"处补一句最小定义，或在 L681"前置概念说明"段把 decoder-only 一并列入待生成概念占位 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html L805（第4章"教学解释"段）："如 YaRN 之类的方法"中 YaRN 首次出现未给最小定义。YaRN 是 RoPE 长度外推方法，对小白是陌生术语：在 L805"YaRN"后补一句最小说明（如"YaRN 是一种 RoPE 长度外推插值方法"）或在"教学解释与类比边界"段标注 YaRN 的最小定义 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 4
- 处置：进入修复。阻断项（第3章"内容相同 vs value 不同"设定矛盾）修复后需复算第3章手算链条与贯穿问题回答是否自洽；其余轻微项最小化修复即可。无需改变研究范围或教学大纲。
