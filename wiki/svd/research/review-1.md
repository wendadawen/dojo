<!-- review-meta
round: 1
page: wiki/svd/index.html
reviewed_content_sha256: 33ccdcae965f7d60
-->
# 奇异值分解（SVD）审查记录（第 1 轮）

- 页面：wiki/svd/index.html
- 页面类型：note（`<meta name="dojo:type" content="note">`，index.html:8）→ 适用规范 `guides/note.md`
- 页面版本（git hash-object 工作树）：44d39790a4e1317a0b2afdf0e1a2f4640e0a51a8
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作）
- 已完整阅读：全页正文（<h1>、导语 index.html:58、正文仅 1 段 index.html:62）。页面无折叠块、无图、无表格、无代码块。
- 机械验证：`.dojo/scripts/validate.py wiki/svd/index.html` → `validation ok`（通过）；`<h2>` 计数 0、`<h3>` 计数 0；去 `<script>` 后可见正文全文仅两句。

## 问题

- [阻断｜规范 guides/note.md｜全页正文 index.html:57-62] 正文缺失：页面除 `<h1>` 标题、`page-lead` 导语和 1 段正文外没有任何内容，无一个 `<h2>`/`<h3>` 章节，模板要求的 `@content` 正文区未被填充；导语与 `description` 列举的用途（低秩近似、极分解正交因子、Procrustes 问题的解、矩阵正交化）在正文中全部没有展开，先例 note 页均含 4–8 个章节与「来源与范围说明」节。页面不满足「可独立阅读」。｜引文依据：提取页面可见文本得正文仅「一个典型应用是 Newton-Schulz 迭代：只用矩阵乘法反复迭代逼近极分解正交因子 $W=UV^{\!\top}$，而不显式算出 $U,\Sigma,V$。」一句；`grep -c '<h2'`=0、`'<h3'`=0；模板 `.dojo/templates/note/index.html:609` 注释「正文按章节组织，一章一件事」；note.md「每篇记录只处理一个中心结论」「基础定义位于首次使用之前，内容按依赖关系排列」。｜修复要求：按 note.md 补齐分章正文，至少覆盖导语已承诺的主题（SVD 的定义与几何含义、极分解正交因子 W=UV⊤、正交 Procrustes 的解法、Newton-Schulz 迭代的机制），一章一事；每条论断接入可定位来源。修复后正文须能独立支撑标题与导语。

- [阻断｜规范 guides/note.md + 来源核对｜全页（导语 index.html:58、正文 index.html:62）] 无来源可核对：页面所有事实性与归因论断——$X=U\Sigma V^{\!\top}$、$\Sigma$ 对角非负即奇异值、SVD 用于定义极分解正交因子 $W=UV^{\!\top}$、Procrustes 问题的解、Newton-Schulz 仅用矩阵乘法迭代逼近正交因子——均无任何来源标注；页内无外链来源（`grep href="http"` 计数 0），`wiki/svd/` 下不存在 `research/` 目录（`find wiki/svd -type f` 仅 index.html），页面也没有「来源与范围说明」章节。按规范，关键论断须注明来源或标注为推断；现既无来源也无推断标注，无法核对。｜引文依据：note.md「关键论断注明来源」「无法核实的内容标记为『未核实』或『推断』」「只记录已经验证的事实和明确确认的判断」；`wiki/svd/` 下无 research 材料、页面 0 条外部链接；同页 `href` 仅 3 个站内/资源链接（`../../index.html`、`../newton-schulz/index.html`、libs 资源）。｜修复要求：为每条论断接入可定位来源（教材/arXiv/官方文档，注明章节号、公式号或行号），或在页面标注其性质（来源事实 / 推断）；补「来源与范围说明」章节。核对时须在审查记录中写出原文片段或公式编号。

- [重要｜规范 guides/note.md（条件缺失）+ 数值复算｜导语 index.html:58、正文 index.html:62] 极分解正交因子公式 $W=UV^{\!\top}$ 的适用条件缺失：导语把 $X$ 说成一般「矩阵」，同时称 $U,V$ 为「正交矩阵」并给出 $W=UV^{\!\top}$。对非方阵 $m\times n$，完整 SVD 的 $U$ 为 $m\times m$、$V$ 为 $n\times n$，二者均正交时 $UV^{\!\top}$ 维度不成立；$W=UV^{\!\top}$ 仅在方阵（或改用 thin/economy SVD、$U$ 取列正交）时成立。页面未说明该前提，属重要条件缺失。｜引文依据：数值复算（numpy）——随机 5×3 矩阵做完整 SVD 得 U(5×5)、V⊤(3×3)，`U @ Vt` 抛维度不匹配错误；economy SVD 下 W=Ue@Ve⊤ 形状 5×3、列正交误差 1.3e-15；方阵 4×4 时 W=U V⊤ 正交误差 8.9e-16 且 W·P=X 误差 7.8e-16。｜修复要求：限定 $X$ 为方阵，或明确说明使用 thin SVD 且 $U$ 为列正交，并同步说明 $W$ 的正交/列正交性质。

- [轻微｜规范 guides/note.md（表述精确性）｜导语 index.html:58] 用词与范围不精确：「它用于定义极分解正交因子 $W=UV^{\!\top}$」中「定义」不准确——正交因子由极分解本身定义，SVD 是给出其表达式与算法；「Procrustes 问题的解」未限定为「正交 Procrustes 问题」，范围偏宽。｜引文依据：note.md「使用正式书面语」「只记录已经验证的事实和明确确认的判断」；数值复算确认 SVD 给出的是正交 Procrustes 的解（由 $A^{\!\top}B=U\Sigma V^{\!\top}$ 得 $\Omega=UV^{\!\top}$，正交误差 1.1e-15）。｜修复要求：改为「给出/用于计算…」，并把「Procrustes 问题」限定为「正交 Procrustes 问题」。

- [轻微｜页面一致性｜description index.html:6、导语 index.html:58、正文 index.html:62] 用途列举三处不一致：`description` 列「低秩近似、极分解和矩阵正交化」，导语列「极分解正交因子、Procrustes 问题的解」，正文只提 Newton-Schulz；三个用途集合互不重合，其中「低秩近似」「矩阵正交化」在导语与正文中均未出现，页面未展开其中任何一个。｜引文依据：note.md「导语概括关键机制与结论」；页面可见文本三处原文如上。｜修复要求：统一 `description`、导语、正文的用途集合，或让正文覆盖所列用途；未在正文展开的用途不应保留在 `description` 中。

- [轻微｜模板一致性｜index.html:59] 页面元信息未填充：`<p class="page-meta"></p>` 为空，模板约定此处填「更新于 <日期> · <内容标签>」，同站 8 个 note 页（如 wiki/hetero-pd/index.html、wiki/gpu-communication/index.html）均已填写，本页缺失。｜引文依据：`.dojo/templates/note/index.html:606`「<p class="page-meta">更新于 【日期】 · 【内容标签】</p>」；同站 note 页对照均非空。｜修复要求：补填「更新于 2026-09-13 · <内容标签>」。

## 结论

- 统计：阻断 2 / 重要 1 / 轻微 3
- 处置：返回修复。页面为近似占位稿——无章节正文、无任何来源，核心内容尚未建立；须先补齐正文与来源，再进入下一轮独立审查。
- 表述维度（本次必查）：通读全页（含导语与正文）未发现元话语、会话指代、调试叙事、临场评价或 AI 拼接腔；`description`/`summary` 无会话指代。表述维度本页通过。
- 符号一致性：全文符号 $X,U,\Sigma,V,W$ 仅出现在导语与正文，含义前后一致；公式均由 KaTeX 定界符包裹，无 Unicode 数学字符。
- 代码核查：页面无可运行代码，无代码核查项。