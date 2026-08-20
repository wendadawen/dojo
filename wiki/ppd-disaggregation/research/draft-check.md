# PPD 分离开页检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 全部完成；evidence.md C1–C22、F1–F5、N1–N21 编号与 outline 章节一一对应。
- 大纲落实：
  - 章节：5 章（1 暴露两个代价；2 full/append-prefill 干扰差一个数量级；3 没有静态最优；4 PPD 动态路由；5 真实负载、慢网络、权重）+ 方法评价 + 架构概念图附章 + 来源与范围说明
  - 核心问题：5 题，每题配 `<details>` 解答折叠，summary 前缀 `解答：`；末尾指明完整论证所在章节
  - 前置知识：moe-serving、standard-attention、gpu-communication、mqa-gqa、prefix-caching 全部以已生成概念页链接；chunked prefill 等相邻工作未生成页故以最小含义表述不链
  - 贯穿示例：5 轮客服对话（输入 1000、回复 200、追加 50）；第 2 章 1250/1150/50 计算量比例；第 4 章 Eq.1 构造示例代入；第 5 章 156 MB KV 量与三档带宽传输时间
  - 误解和边界：6 条误解在对应章节与来源与范围说明处理
  - 评价章节：方法评价含优点、局限、适用场景与位置，分析性判断已集中标注
  - 过渡：每章首段说明与前一章关系，章末「本章问题」链接全文核心问题
- 原图：6 张原图全部内联（图 1 Pareto、图 2 干扰、图 3 概念图、图 4 真实负载、图 5 带宽模拟、图 6 权重），均来自 TeX 源码包（最高优先级途径），其中图 3（ppd.pdf）经 sips 转 PNG 后内联。
- 机械检查：python3 .dojo/scripts/validate.py wiki/ppd-disaggregation/index.html → validation ok；overview.html → validation ok。
- 公式与符号：KaTeX 渲染；description 纯文本（不含公式）；dojo:summary 含公式；正文所有数学符号写在 `$...$` 内；正文运行多个 $S$、$\Delta$、$\psi$、$\pi$、$\mathbf{w}$ 等符号；全文未出现裸 ≤/≥/∈/≈ 等需包 $...$ 的字符。
- 可运行代码：本文写作范围内无可运行代码（实证数据均为论文实验结果，作者已通过 vLLM 原型在真实集群上验证；本文按 cite-then-claim 方式引用，不复现实验）。
- 写作偏差：无偏差，全文按 outline 落实。
