# PCP/DCP 概念页 draft-check

## 大纲落实

- 第 1 章（两阶段瓶颈）：✓ 引言场景、核心问题 5 条（含 解答：折叠）、常见误解 3 条、章节正文、dg-stack 阶段图、本章问题 2 条。
- 第 2 章（TP 头维天花板）：✓ TP 切法回顾与链接 model-parallelism、F1 条目数、重复因子公式与两段讨论、GQA/MLA 与链接 mqa-gqa/mla、贯穿构造模型手算表（H=1, T=8, tp=4）、SVG 对比图（H=1 重复 vs H=4 无重复）、过渡段、本章问题 3 条。
- 第 3 章（DCP 沿 token 维切 KV cache）：✓ 3.1 切法与交错分配（构造模型 T=8/d=4 归属表 + 总副本公式）、3.2 三段通信节奏（dg-flow 图）、3.3 LSE 合并（恒等式、补充推导折叠块、手算表、展开数值折叠块、可运行代码折叠块含真实输出、MLA/GQA 路径补充折叠块、a2a/q_replicate 补充折叠块）、3.4 收益代价与实测数字 N1/N2、表格、本章问题 4 条。
- 第 4 章（PCP 切 prefill 计算）：✓ 4.1 两种策略、4.2 causal mask 负载不均与 2N 首尾配对（手算表 8 块 4 卡 + 加速上界、SVG 配对关系图）、4.3 PCP vs DCP 对照表、本章问题 3 条。
- 第 5 章（选型）：✓ 官方建议、dcp 范围与硬校验、三个 case study 表、Kimi-K2 两两两取舍、PCP 组合约束折叠块（含源码校验代码）、PD 分离辨析与链接 ppd-disaggregation、负收益边界、本章问题 3 条。
- 文末来源与范围说明：✓ 论断/公式/数字表、构造示例、辅助解释与类比边界、简化条件及其限制；三处表面不一致的说明。

## 学习目标覆盖

- Q1 两阶段为何需两种 CP → 第 1 章正文 + Q1 解答折叠
- Q2 GQA/MLA TP 头维天花板与重复因子 → 第 2 章正文 + Q2/Q3 解答折叠
- Q3 DCP 切分、通信、LSE 恒等式 → 第 3 章三段 + 折叠块 + 代码 + Q3 解答折叠
- Q4 PCP 切分、causal 配对、资源本质 → 第 4 章三段 + Q4 解答折叠
- Q5 选型、dcp 取值、与 PD 分离关系 → 第 5 章 + Q5 解答折叠

所有目标均独立成段作答，不依赖回看正文。

## 代码运行记录

- LSE 合并验证脚本（/tmp/pcp_dcp_lse.py）：已实际运行，输出 rank 0 D=22.803819 o_r=0.119203 l_r=3.126928；rank 1 D=61.987206 o_r=0.880797 l_r=4.126928；global 0.675973；merged 0.675973；equal True。预期输出与代码块内一致。
- causal 配对验证脚本（/tmp/pcp_dcp_causal.py）：已实际运行，输出 sequential [3,7,11,15] max 15 speedup 2.4×；paired [9,9,9,9] max 9 speedup 4×。与表格一致。

## 机械检查

- .dojo/scripts/validate.py wiki/pcp-dcp/index.html → validation ok
- .dojo/scripts/validate.py wiki/pcp-dcp/overview.html → validation ok

## 浏览器实测（headless Chrome，浅色主题）

- KaTeX 渲染：294 处 katex 节点，0 处 katex-error，0 处未渲染 `$...$` 残留。
- TOC：6 项（5 章节 + 文末来源，符合 body 直接子 h2 数）。
- 阅读时间：8,396 字，约 28 分钟。
- SVG 标签重叠：0 处（dg-flow HTML 图未参与；两张 SVG 内 text/foreignObject 两两矩形无重叠）。
- SVG 标签越界：0 处（无 text/foreignObject 超出 svg 视口）。
- SVG 视觉抽查（独立页面强制浅色截图）：
  - ch2 左右面板：H=1 左侧 4 张卡各持 8 个 "0" 格子（共 32 份相同数据 → 4 副本），H=4 右侧 4 张卡各持 8 个对应头编号格子（32 份各不相同 → 无重复）；对比直观。
  - ch4 配对关系：上方顺序等分括号覆盖相邻块对，下方 4 条嵌套弧线配对 (块 0, 块 7)、(块 1, 块 6)、(块 2, 块 5)、(块 3, 块 4)，GPU 标签盒嵌在弧线顶点处；层级与配对规则清晰，无文本压线。

## 已知遗留

- 全部三条既有的来源表面不一致（S1 推荐范围 vs S3 硬校验；S2 连续区间示意 vs S1/S3/S4 实际交错；a2a 时间线）均在正文或来源章节显式说明，未改写成模糊表述。
- dcp_q_replicate 折叠块已注明环境变量名。