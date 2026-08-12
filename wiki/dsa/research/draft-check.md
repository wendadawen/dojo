# DSA 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 四份均已完成，规划阶段完成条件已满足（概念歧义已裁定、5 个学习目标、11 项核心内容全部映射、7 项前置知识全部已有概念页无需递归、F1–F4 与 N1–N11 与 C1–C14 全部为已确认状态）。

## 大纲落实

- 章节：8 章全部落实，标题与顺序与 outline.md 一致（cost-structure / two-routes / lightning-indexer / token-selection / training / engineering / accounting / positioning）。
- 学习目标：learning-goals 组件列出 5 条，与 scope.md 的 Q1–Q5 一一对应，语序做了口语化调整但含义未变。
- 前置知识：standard-attention（第 1 章首次依赖）、linear-attention + kda（第 2 章）、quantization-basics（第 3 章 FP8）、mla + mqa-gqa（第 4 章）、rope（第 6 章）均在首次依赖处给出链接，未内联展开。
- 贯穿例子：8 位置 / k=3 / 2 头 2 维，按大纲在 7 处推进——第 1 章配对计数（36 与 136）、第 2 章三方案对照表、第 3 章手算 8 个分数、第 4 章取 top-3 得 {2,7,8}、第 5 章目标分布构造、第 6 章可运行代码复现、第 7 章 k=8 展示失效区间。
- 误解和边界：误解 1（不省 KV cache）落在第 7 章 no-memory-saving 小节；误解 2（与 FlashAttention 同类）落在第 1 章 callout；误解 3（可直接加到稠密模型）落在第 5 章开头与两阶段表格后；误解 4（降到线性）落在第 7 章 not-linear 小节；误解 5（每头独立选）落在第 4 章 shared-candidates 小节。适用边界小结落在第 7 章 boundary 小节。
- 过渡：第 1→2 章「既然要少读，就得决定读哪些」；2→3 章「打分器本身若不够便宜，省下的计算就被它自己吃掉」；3→4 章由 callout 收束到复杂度问题再进入选择；4→5 章「前面一直假设 indexer 能打出有意义的分数」；5→6 章「前面讲的是机制，这一章看部件」；6→7 章「把账算清楚」；7→8 章由边界小结进入定位对照。
- 完成检查：8 章每章末尾均有检查项，只给问题不印答案。

## 学习目标闭环

- Q1（为什么需要 DSA，窗口与线性注意力为何不够）：第 1 章正文（代价结构、decode 重读 KV cache）+ 第 2 章正文（两条路线与 DSA 的第三条）完整回答。折叠块无参与。
- Q2（indexer 如何打分、为何便宜）：第 3 章正文给出 F1 与逐符号说明、手算分数表、四来源对比表、复杂度未降的限定。折叠块只承载剩余六个位置的算术过程，正文已含 s=2 的完整代入与分数表全部结果。
- Q3（top-k 如何接回、为何跨头共享）：第 4 章正文给出 F2、选中集合、掩码实现说明、C4 的 kernel 原因、ASCII 图示、MQA 模式关系。折叠块无参与。
- Q4（indexer 参数从哪来、为何两阶段）：第 5 章正文给出起点问题、F3 与目标分布两步构造的文字说明、F4 与 KL 作用域收窄、C10 信号隔离、两阶段差异表（含 N4–N7 全部数字）。折叠块只承载目标分布的算术例子。
- Q5（省什么、没省什么、何时失效）：第 7 章正文给出省/不省对照表、not-linear、no-memory-saving、short-sequence、parity、boundary 五个小节完整回答。折叠块无参与。

折叠块全部收起时逐题复核：5 个目标均由正文章节完整回答，无目标被折叠块独占。

## 代码运行

- 第 6 章可运行代码块（numpy 复现 index score 与 top-k）：运行命令 `/Users/wendadawen/.workbuddy/binaries/python/versions/3.13.12/bin/python3 /tmp/dsa-research/page_code.py`，退出码 0。实际输出 11 行（8 行逐位置结果 + 空行 + 2 行选择结果），与页面「预期输出」块逐字符一致。页面中 s=3、s=4 出现负点积被 ReLU 截成 0，与「观察重点」描述一致；k=8 时选中全部 8 个位置，与失效区间描述一致。
- 页面内另有一个伪代码折叠块（index kernel 四步），标记为 language-text，不声称可运行，未执行。

## 机械检查

- `python3 .dojo/scripts/validate.py wiki/dsa/index.html` → `validation ok: wiki/dsa/index.html`，退出码 0。
- `python3 .dojo/scripts/validate.py wiki/dsa/overview.html` → `validation ok: wiki/dsa/overview.html`，退出码 0。
- 占位符与组件标记残留检查：两份文档中 `【` 计数为 0，`@content` / `@copy-start` / `@component` 计数为 0。
- 内部链接检查：index.html 的 12 个内部引用（含 libs 与 7 个概念页）全部存在；overview.html 的 7 个内部引用全部存在。两份文档互相链接（index → overview.html，overview → index.html）。

## 公式渲染与交互

- KaTeX：index.html 正文共 5 个 display 公式——F1（index score）、贯穿例子的 q/w 取值、F2（稀疏注意力输出）、F3（warm-up 损失）、F4（sparse training 损失）——与多处行内公式，均使用 `$$` / `$` 分隔符，与外壳 auto-render 配置匹配。overview.html 行内公式 3 处（$k$、$O(L^2)$、$O(Lk)$）。libs 路径为 `../../libs/`，与 wiki 两层深度一致，已验证文件存在。
- 折叠交互：5 个 `<details>` 块——第 3 章「剩余六个位置的完整代入过程」1 个、第 5 章「目标分布的构造过程」1 个、第 6 章 3 个（index kernel 四步伪代码、numpy 可运行代码、DCP 论证），其中 `code-details` 2 个。
- 目录锚点：正文 8 个 h2 加文末「来源与教学说明」共 9 个 h2，13 个 h3，全部显式指定 id；脚本核对无重复 id。
- 说明：本条为静态检查结果。浏览器实际渲染与交互检查在质检阶段随页面打开一并进行。

## 写作偏差

- 无返回规划阶段的偏差。
- 局部修正 1：贯穿例子的具体数值在规划阶段未指定，生产阶段实际构造并跑脚本验证，为满足「分数两两不等、至少两处 ReLU 截断、top-k 与最近 k 个不同」三项约束调整过一轮初始取值（首轮出现三路 2.5 并列且当前 token 未进 top-3，已弃用）。属于大纲允许的局部计算补充，未改变大纲结构。
- 局部修正 2：大纲第 4 章原计划在正文说明掩码实现，生产阶段补充了一句「掩码版本仍把全部位置算了一遍，真实收益要靠专门 kernel」，属于防止读者形成错误结论的必要限定，未引入范围外内容。
- 局部修正 3：glossary.md 要求区分数量 $k$ 与 key 向量记号，生产阶段在第 4 章符号列表中显式加了一行说明，属于符号一致性要求的落实。
