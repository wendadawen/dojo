# NoPE 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已写齐于 `wiki/nope/research/`，规划完成条件已满足（概念歧义已裁定、5 个学习目标、核心论断 C1–C6 与公式 F1–F2、数字 N1–N2 均完成来源定位且置信状态为已确认、教学大纲齐备）。

## 大纲落实

- 章节结构：S1 why-position-encoding / S2 what-is-nope / S3 causal-mask-implicit-position / S4 length-generalization-and-theory / S5 nope-in-kimi-k3 / S6 boundaries + 文末 sources-and-teaching-notes，共 7 个 h2，id 全页唯一。
- 学习目标：Q1–Q5 在页面开头用 learning-goals 组件列出，与 scope.md 一致。
- 前置知识：位置编码 / RoPE / 因果掩码三个未生成概念页按任务约定登记不生成，正文首次依赖处给"概念页待生成"标注 + 最小衔接（因果掩码在 S3 给最小定义"遮挡未来 token 的规则"以支撑机制讲解）；KDA / 线性注意力已有概念页，在 S5 给真实链接。
- 贯穿例子：三个内容设定（手算用 v1=2,v2=4,v3=6、分数相等→均匀 softmax）在 S1 引入排列等变问题、S3 手算因果（输出 2,3,4）与双向对照（输出全 4）、S6 复用双向对照给出边界。
- 误解和边界：S2 澄清"NoPE 不是又一种位置编码"；S5 澄清"K3 用 NoPE 不等于没有位置信息"；S6 给因果掩码必要前提、双向失效、隐式信号局限、K3 依赖 KDA 四条边界。
- 过渡：每章末完成检查后指出下一章要解决的问题（S1→去掉什么；S2→顺序从哪来；S3→效果如何；S4→用在前沿模型了吗；S5→边界）。

## 学习目标闭环

- Q1（NoPE 是什么 + 与 APE/相对/RoPE 区别）：S2 定义 NoPE 去掉三类操作；S1 对照表列出 APE/相对/RoPE 各做什么、NoPE 三者都不做。正文完整回答。
- Q2（为什么因果注意力仍能区分词序）：S3 因果掩码使可见集合不同 + 手算例子（输出 2,3,4）+ 双向对照（输出全 4）。正文完整回答。
- Q3（长度泛化表现 + 理论）：S4 论文摘要结论（NoPE 优于显式方法、无额外计算）+ 理论（可表示绝对/相对 PE，SGD 下主要类似 T5 相对 PE）+ 外推无需调参的解释。正文完整回答。
- Q4（K3 应用）：S5 NoPE 用在 MLA 层、KDA 提供位置、扩展上下文无需 RoPE 重缩放/YaRN、8K→1M。正文完整回答。
- Q5（边界）：S6 因果掩码必要前提、双向失效、隐式信号局限、K3 依赖 KDA。正文完整回答。
- 全部目标由正文章节完整回答，无目标被折叠块独占。折叠块（S3 双向手算）收起时正文仍回答 Q2。

## 代码运行

无可运行代码。NoPE 的核心是"不做什么"，机制用手算例子（S3 三 token 因果注意力）即可验证，比代码更清晰；按 outline.md §5 不安排可运行代码（符合 write.md"代码只在能帮助理解且实际跑通时才加入"）。手算已逐位置代入验证：因果下 o1=2、o2=3、o3=4；双向下三位置均为 4。

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/nope/index.html` → 结果：`validation ok: wiki/nope/index.html`（退出码 0）。
- 命令：`python3 .dojo/scripts/validate.py wiki/nope/overview.html` → 结果：`validation ok: wiki/nope/overview.html`（退出码 0）。
- 两页均通过：无模板占位符【…】、无 @content/@component/TODO/TBD 标记、无重复 id、无指向缺失 id 的同页锚点、无断链本地引用。
- 占位链接处理：三个未生成概念页（positional-encoding / rope / causal-mask）的引用改为内联文字"概念页待生成"标注（不使用 href 指向不存在文件），故不产生断链；KDA / 线性注意力已有概念页保留真实链接并通过检查。

## 公式渲染与交互

- KaTeX 语法检查：正文 1 个显示公式（F1 因果注意力输出）与若干行内公式（$o_t$、$q_t,k_i,v_i$、$\alpha$、$g_{\min}=-5$ 等）使用标准 LaTeX 语法，`$$`/`$` 定界符与外壳 auto-render 配置一致；`$$` 配对正常（模板 JS 配置 2 处 + 显示公式开闭 2 处 = 4，对应 1 个显示公式）。
- 渲染机制：依赖外壳 `auto-render.min.js` 自动渲染 `$...$` 与 `$$...$$`，与其它概念页同一机制。
- 交互：外壳脚本提供目录、章节折叠、j/k 跳转、主题切换、返回顶部、代码复制按钮，均由外壳统一处理。
- 限制：未开启图形浏览器逐式截图核对；公式语法经文本检查 well-formed，渲染依赖外壳脚本。

## 写作偏差

无写作偏差。未自行增删核心章节、未新增学习目标、未更换贯穿例子、未把正文必要内容移入折叠块、未使用证据不足论断。写作中发现占位链接会触发 validate.py 断链检查，按 block-attnres 既有惯例（未生成概念用内联文字、不递归生成）将三处占位 href 改为内联标注，属机械兼容处理，不改变大纲与内容范围。
