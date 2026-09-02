# UltraEP 质检状态

## 当前状态：三轮独立审查全部完成，全部阻断与重要问题已关闭，满足发布条件

## 各轮结果

- 第 1 轮（`review-1.md`）：阻断 2 / 重要 5 / 轻微 7，全部 14 条已修复并逐条复验。最实质的两条：4.5 节 $\tau=9$ 探测的空闲向量写错（脚本复算更正）；Table 3 的 $\sum_e|\mathcal{H}(e)|$ 误读为「实例总数」（实为消耗的冗余槽数，7 处更正）。另一条重要问题：页面代码 `locality=False` 对照分支违反每源需求守恒，修正比例分摊逻辑后该值由 62.5% 更正为 54.2%。
- 第 2 轮（`review-2.md`）：阻断 0 / 重要 0 / 轻微 3，全部修复（EP64 跨 8–16 台服务器补推算标注、overview 日期主语消歧、overview 补齐五项 dojo 元信息）。
- 第 3 轮（`review-3.md`）：阻断 0 / 重要 1 / 轻微 2，全部修复。重要的一条：4.9 节把 45 对 107 的槽数差距归因到 $u_{\min}$ 是页面推断却写成论文事实，已随文标注并把「来源与范围说明」的推断清单由三处改为四处；两条轻微为中继阈值操作数的 ±1 歧义消歧、「§8.2 末段」更正为「§8.2 训练段末」（三处）。

三轮审查者均为独立派发的子代理，未参与写作与前序轮次，未读取 research/ 规划产物。

## 已完成的机械验证结果

- `python3 .dojo/scripts/validate.py wiki/ultraep/index.html` → validation ok（三轮修复后均重过）
- `python3 .dojo/scripts/validate.py wiki/ultraep/overview.html` → validation ok
- 两页无残留【…】占位符与组件标记
- 章节编号：h2 为 1–7 连续；各章 h3 章内连续无跳号
- 问题块：页面级「核心问题」4 题、6 个章节「本章问题」共 20 题，全部 24 题均有以「解答：」开头的折叠块（headless Chrome 实测 noAnswer=0、badSummary=0）
- 表格：9 张表的每行单元格数（计入 colspan）与表头列数全部一致
- 公式渲染：headless Chrome 实测 400 个 `.katex` 节点，正文可见文本中未渲染的 `$...$` 片段为 0
- 原图：11 张 webp 全部加载成功（naturalWidth 非零），无坏图
- 内部链接：8 个前置概念页与首页、libs 资源经文件存在性检查全部有效
- 代码：`python3 /tmp/ultraep/solver.py` 退出码 0，双向守恒断言全部通过，输出与页面「预期输出」块逐字符一致
- 数字自洽：`python3 /tmp/ultraep/verify_numbers.py` 17/17 项通过（由 Figure 11/13/16/17 与 Table 3 的值反算正文声称的百分比与倍数）

## 遗留

- 尚未 commit（仓库约定未获授权不提交）。
