<!-- review-meta
round: 6
page: wiki/moonvit-v2/index.html
reviewed_content_sha256: d3451d37993ba7f8
-->
# MoonViT-V2 审查记录（第 6 轮）

- 页面版本：f0a541e6584bc0a7c61f81d70bd6880ed1e4e032（工作树 hash，git hash-object）
- 审查时间：2026-09-14 17:38
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 → 1. 主流做法遇到什么问题——SigLIP 初始化为何不稳 → 2. 从零训练的方案——next-token prediction 如何替代对比预训练 → 3. 训练是怎么进行的——接入 LLM 的联合预训练 → 3.1 训练用什么数据 → 3.2 训练配置 → 3.3 系统层面怎么跑得动 → 4. 服务稳定性的架构——27 层 ViT、RMSNorm、去 bias → 5. 图像/视频共享与高分辨率——分解注意力、时间池化、pixel-shuffle → 6. 结果与结论——对比预训练初始化是否必要 → 来源与范围说明（含全部折叠块与图注）→ 全文总结
- 核对所用来源版本：Kimi K3 技术报告 arXiv:2607.24653v2《Kimi K3: Open Frontier Intelligence》（HTML 全文 + §2.4/§3.1/§3.2（Table 1）/§3.3/§5.2.3 原文 + Fig.6 原始矢量图 2607.24653v2/vt-grad-norm.svg，逐点解析坐标复算）；HuggingFace moonshotai/Kimi-K3 config.json 的 vision_config 字段。页面正文/概述/meta 中所有数字与引号内原文均按该版本逐条比对。

## 问题

- [阻断·技术] 第 1 章 Fig.6 简化对照图（index.html 118–151 行）：图上画的是带数值刻度的 y 轴（0 / 0.2 / 0.4 / 0.6，128 行折线与 231 行红色虚线），读者据刻度读出的量级与来源图实际取值差约 20 倍，且图注声称"不保留精确刻度"与图上刻度自相矛盾；N1 条目另给出"峰值约 0.6"这一来源不支持的读数｜引文依据：来源 Fig.6（arXiv:2607.24653v2，vt-grad-norm.svg）横轴 "Training step (×10^3)" 刻度 7/10/15/20/25/30、纵轴 "Vision-tower gradient norm" 刻度 0/0.2/0.4/0.6，图注原文 "Figure 6: Vision-tower gradient norms in our pre-training ablations."；逐点解析 panel (a) 两条曲线得：MoonViT-3D（SigLIP init，teal）中位数 ≈0.0208、p90 ≈0.0482、p99 ≈0.1171、最大 ≈0.7400，仅 0.37% 采样点 >0.2；MoonViT-V2（from scratch，red）中位数 ≈0.0079、p90 ≈0.0188、最大 ≈0.4041，仅 0.03% 采样点 >0.2——两条曲线绝大多数时间贴近 0，只偶发单点尖峰。页面图把 MoonViT-3D 画成全程在 0.371–0.6 之间连续起伏、把 MoonViT-V2 画成 0.08 的一条平线；页面图注为"只看两条曲线的相对高低与尖峰密度，不保留精确刻度"，但图上明确画出 0/0.2/0.4/0.6 数字刻度（同页自相矛盾），且正文 116 行称"具体小数不引用"，N1 条目却写"MoonViT-3D 峰值约 0.6"（来源实测峰值 ≈0.74，且该值取自轴顶刻度而非曲线）｜修复要求：二选一——(1) 去掉图上 y 轴数字刻度与 x 轴数字刻度，改为无刻度的相对高低示意，并把图注、正文 116 行、N1 条目统一为"只表达相对高低与尖峰密度、不给出任何数值"；(2) 按来源实测形态重绘：两条曲线基线分别约 0.02 / 0.008，仅在个别训练步出现单点尖峰（MoonViT-3D 约 0.74、MoonViT-V2 约 0.40），并保留 0/0.2/0.4/0.6 刻度，同时删除 N1 的"峰值约 0.6"或改为与曲线一致的数值。无论选哪种，图注与图上刻度必须一致｜修复：｜复验：

- [重要·技术] 第 1 章 153 行与第 2 章 183 行：把"预训练视觉权重与 LM 目标耦合带来的更新冲突"当作梯度不稳定的机制写成事实，报告只观察到不稳定现象、未给出该机制｜引文依据：报告 §2.4 全文只有 "We depart from this practice primarily for training stability. When a pre-trained encoder is attached to the LLM, joint optimization becomes unstable: the SigLIP-initialized MoonViT-3D shows persistently higher gradient norms with frequent spikes..."——未出现 conflict / interference / coupling / mismatch 一类机制表述，也未说明不稳定的成因；页面 183 行原文"从零初始化消除了『预训练视觉权重与 LM 目标耦合』带来的更新冲突——这正是上一章观察到的梯度不稳定的来源"，153 行原文"预训练权重与 LM 目标在联合优化时『打架』，导致更新剧烈波动"｜修复要求：删除"更新冲突/打架"这一机制表述，或改写为明确标注的推断（在「辅助解释与类比边界」补一条，说明它是本页对"joint optimization becomes unstable"的解读，报告未给出成因），正文只保留报告支持的"接入预训练编码器后联合优化不稳定、从零训练全程稳定"｜修复：｜复验：

- [轻微·格式] 全部 6 个正文章节的末尾过渡（155、202、481、634、687 行）＋336 行："本章……。但……——下一章讲……"同一句式逐章复用｜引文依据：不适用（guides/concept/style-guide.md 第 8 节"章节顺序存在依赖时，用一至两句说明前一节结论与下一节问题的关系。不使用固定句式"）｜修复要求：保留"先总结本章结论、再指出尚不能解决的下一步问题"的衔接功能，但至少半数章节改换句式与切入角度，去掉逐章复用的固定模板｜修复：｜复验：

- [轻微·格式] 624 行"别把它误读成『丢信息的有损压缩』"：以祈使句直接对读者说话，属第二人称表达｜引文依据：不适用（style-guide.md 第 12 节"不直接使用第二人称称呼读者"）｜修复要求：改为第三人称陈述，如"它不是丢信息的有损压缩——压缩的是序列长度，信息量由后续投影降维那一步决定"｜修复：｜复验：

- [轻微·可读性] 训练通路图内公式 $\mathcal{L}=-\log p(x_{t+1}\mid x_{\le t})$（296 行 foreignObject）：$\mathcal{L}$、$x_{t+1}$、$x_{\le t}$ 三个符号全文未定义，图注（303 行）也未说明，且该公式未列入「公式与来源（F）」小节（F1–F3 只覆盖参数量与 token 数两个手算）｜引文依据：不适用（style-guide.md 第 11 节"公式后紧跟 <ul> 逐项定义每个符号"；第 6 节 C/F/N 与来源章节双向对应）｜修复要求：在正文首次出现 next-token prediction 的段落（177–179 行）就地说明符号含义（$x_{\le t}$ 为已观察 token 序列、$x_{t+1}$ 为被预测的下一个 token、$\mathcal L$ 为该位置的语言建模损失），或删去该公式只保留文字；并在 F 小节补一条说明其依据为报告 §3.3 的 "single next-token prediction objective"｜修复：｜复验：

- [轻微·格式] 口语化措辞：110 行"让 LLM 看懂图"、632 行"会把上下文吃掉一大块"、153/183 行"打架"（若按上条删除则可一并消除）｜引文依据：不适用（check.md 2.1 表述维度"口语化措辞"）｜修复要求：改为技术表述，如"让 LLM 读取图像内容""会占用很大一部分上下文预算"｜修复：｜复验：

## 已核对且无问题的项（本轮复核结论）

- config.json vision_config 与页面 4 章表格逐项一致：vt_num_hidden_layers=27、vt_hidden_size=1024、qkv_hidden_size=1536、vt_intermediate_size=4096、vt_num_attention_heads=12、patch_size=14、norm_type=rmsnorm、attn_bias/linear_bias/patch_embed_proj_bias=false、merge_kernel_size=[2,2]、merge_type=sd2_tpool、text_hidden_size=7168。
- 两处手算复算通过：per_layer=14,682,112、27 层 = 396,417,024 ≈0.40B（与报告 "roughly 0.4B" 及 Table 1 "Total Parameters of ViT 401M" 一致）；3584/14=256→65,536→/4=16,384。
- 折叠块内 Python 代码已实际执行，输出与页面「预期输出」逐行一致（0.40B、65536、16384 全部吻合）。
- C1–C11、F1–F3、N1–N4 的引号内原文逐条与 arXiv:2607.24653v2 对应段落比对，除上述两项外全部吻合（含 C2 的 §2.4 正文与 Fig.6 题注、C4 的 RMSNorm/去 bias 句、C5 的共享参数句、C6 的 pixel-shuffle 句、C7 的 matches 句、C9 的 §3.3/§2.4 两句、C10 的四域/六类/两种坐标格式/程序化数据句、C11 的 §5.2.3 三段、N3 的 §3.3 训练配置句、N4 的 2.8T 与 Table 1 的 2.78T）。
- 页面未引用报告参考文献编号，不存在 v1/v2 文献表编号错配。
- 结构与机械项：外部链接/前置概念页 ../siglip/、../vit/、../standard-attention/ 均真实存在且无"（待生成）"；overview.html 与 index.html 双向互链；两处数据一致（27 层/0.4B/65536→16384/2.78T/Per-Head Muon + 权重衰减 0.1）；dojo:type=concept、dojo:topics=多模态（词表内）、dojo:tag=视觉与多模态（词表内）、dojo:summary 无公式；aria-label 与 alt 内无 `$...$`；无等宽字符框线图；图内公式均在 foreignObject 内由 KaTeX 渲染；全部数学符号为 LaTeX（除页面通篇使用的半角乘号 ×，与报告 "3584 × 3584" 写法一致且 validate.py 不报错）；`.dojo/scripts/validate.py wiki/moonvit-v2/index.html` 返回 `validation ok`。

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：修复（第 6 轮为追加轮次；阻断项未关闭前不得发布）
