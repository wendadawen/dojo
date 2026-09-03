# EPLB 初稿检查

## 输入版本

- scope.md：已确认（4 个适用范围、Q1–Q5、4 条误解与适用边界、前置知识映射无缺口）
- evidence.md：已确认（C1–C19 + F1–F2 + N1–N6；C12 中「推理期 bias 不再更新」明确标注为推断）
- outline.md：已确认（页面开头 + 5 章 + 折叠块分工 + 贯穿示例与表达材料）
- glossary.md：已确认（~30 个术语与符号约定）

## 大纲落实

- 页面开头：场景示例 + 负载表 + EPLB 简介 + 主要依据 + 核心问题 5 题 + 常见误解 3 条 + 过渡 — 全部落实
- 第 1 章「专家并行的木桶效应」：两张子节 + 章节问题 2 题 + 构造利用率例（800/200） + 训练期/推理期对照表 — 全部落实
- 第 2 章「冗余专家」：三张子节 + 章节问题 3 题 + 贪心手算表 + 冷专家对照 + 贪心逐步验证折叠块 — 全部落实
- 第 3 章「三步排布」：三张子节 + 章节问题 3 题 + 层次策略完整手算 + SVG 排布图 + Step 3 节点 0 展开 + 三步伪代码 + 可运行 Python 复刻 + 全局策略对照表 — 全部落实
- 第 4 章「周期性重平衡」：三张子节 + 章节问题 3 题 + dg-flow 闭环图 + V3 部署对照表 + 探索性动态冗余折叠块 + vLLM 参数表 + 公式 — 全部落实
- 第 5 章「边界」：四条边界 + 边界对照表 + UltraEP/MoonEP 链接 + 章节问题 2 题 — 全部落实
- 文末来源与范围说明：论断与来源（C 列表） + 公式与来源（F） + 外部数字与实验条件（N） + 构造示例 + 辅助解释与类比边界 + 简化条件 — 全部落实
- 贯穿示例：开篇负载表 → 第 2 章贪心复制手算 → 第 3 章完整排布 → 与官方输出逐槽核对 — 全部落实
- 前置知识：deepseek-moe / moe-serving / aux-loss-free-routing / model-parallelism 链接在首次使用前给出 — 全部落实
- 误解：4 条误解中 3 条最常见放在页面开头；第 4 条「绝对均衡」在第 3 章取舍与第 5 章边界处理 — 全部落实

## 目标覆盖检查

- Q1 专家并行负载不均与量化 — 第 1 章（1.1 木桶效应与利用率构造；1.2 训练期 vs 推理期对照 + V3 不丢 token）+ 第 3 章构造例验证 max/mean 1.21
- Q2 冗余专家机制 — 第 2 章（2.1 逻辑/物理 + F1 均摊公式；2.2 槽位整除；2.3 贪心规则与手算）
- Q3 三步算法手算 — 第 3 章（3.1 组与节点受限路由；3.2 三步完整手算 + SVG 图 + Python 复刻；3.3 全局策略取舍）
- Q4 周期性重平衡与 vLLM 集成 — 第 4 章（4.1 闭环；4.2 V3 两阶段部署；4.3 vLLM 参数与公式）
- Q5 边界 — 第 5 章四条边界 + 对照表 + UltraEP/MoonEP 链接

## 代码运行

- 三步算法 Python 复刻（页面 第 3 章可运行代码块）：命令 `python3 eplb_page_demo.py`；退出码 0；输出与页面描述一致（组负载 [262,330,116,325]、节点内复制 e5,e4 与 e10,e1、层次 GPU 负载 121.5/86.5/125/113 与 147.5/131.5/156/152、phy2log 与官方 tensor 逐槽一致）
- SVG 坐标预计算（/tmp/eplb-verify/gen_svg.py）：退出码 0；overlap 检查 no partial overlaps；与官方 phy2log 逐槽一致（已用 Python regex 抽取文本比对验证 MATCH: True；其中 GPU5 槽位顺序在初版生成后修正为 [e10·副本, e2] 以匹配 slots [10,2]）
- 层次 vs 全局完整复刻（/tmp/eplb-verify/eplb_verify.py）：退出码 0；全局策略 GPU 负载 [130.5,95.5,130,138,138.5,134.5,134,132]、max/mean ≈ 1.07、global 复制 e1,e4,e5,e10（已与 outline 第 3.3 节核对一致）

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/eplb/index.html`
- 结果：首次运行报 4 处裸数学字符（max/mean ≈ 1.21/1.07 在 td、±γ 在 C12 行、≈ 在 N 段）；修复后再次运行 `validation ok: wiki/eplb/index.html`
- 修复内容：将三处 ≈ 与 ±γ 用 `$...$` 包裹为 KaTeX 公式（`$\approx$` 与 `$\pm\gamma$`）

## 公式渲染与交互

- 浏览器实测：headless Chrome 探针（/Users/wendadawen/code/dojo/.dojo/tmp-probe/probe_page.html）；探针注入 .katex 节点数、.katex-error 数、残留 `$` 数、SVG 文本重叠、目录项数、details 块数
- 结果：`katexCount: 52, katexErrors: 0, dollarLeft: 0, svgLabelCount: 50, svgOverlaps: [], tocItems: 28, detailsCount: 23, h2Count: 8`
- 结论：所有公式渲染成功、SVG 50 个标签无重叠、目录与折叠块数与大纲一致
- 视觉确认（顶部 1280x800 截图）：h1 单行不超长、负载表正确、KaTeX 公式行内渲染、引文上标显示、核心问题章节样式正常、概览页标题/导航/正文与列表均渲染正确

## 写作偏差

- 无。范围严格按 outline.md 落实；未引入大纲外内容；前置知识链接全部使用 ../<name>/index.html 相对路径；元数据 dojo:topics 已从封闭词表（并行与通信、推理系统）中选