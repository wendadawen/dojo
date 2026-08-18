# RoPE 审查记录（第 1 轮）

- 页面版本：wiki/rope/index.html（1539 行）
- 审查时间：2026-08-18
- 审查者：独立子代理（未参与写作）
- 已完整阅读章节：核心问题 → 前置概念 → 1-6 章 → 来源与范围说明；overview 全文

## 问题

- [重要·技术] index.html:808,1263,1266,1282,1306；overview.html:71：C8（K3 MLA 用 NoPE、mla_use_nope=true、KDA 提供位置、外推 1M token、技术报告 §2.1.2/§3.4）所引来源不在本轮允许输入内，无法给出引文依据｜引文依据：无法取得｜修复要求：补充对 K3 技术报告/config.json 的核对记录，或降级为推断｜修复：编排者核对：K3 报告 §2.1.2 原文 "applies No Position Encoding (NoPE) to all MLA layers"（/tmp/kimi-k3-research/k3-report.txt 363-364 行）；config.json mla_use_nope=true 已确认；§3.4 "8K to 64K, then 256K and 1M" 已确认。C8 保留来源结论形式，核对记录见此。｜复验：来源定位正确（§2.1.2 行 363-364、§3.4、config.json），结论保留
- [轻微·技术] index.html:1316：F8 图 2 数值无法从 ar5iv 核对｜修复：删除具体数值，保留"Figure 2 给出上界随距离增大而减小的曲线"｜复验：待下轮
- [轻微·技术] index.html:1314：F6 引用定位 §3.2.2 但引用句在 §3.3；索引写法 i 从 1 开始｜修复：修正为 §3.3，注明两种索引等价｜复验：待下轮
- [轻微·技术] index.html:1300：C2 标注 "§3.1, Eq.(1)–(7)" 与实际不符｜修复：改为"摘要；§3.1 Eq.(11)"｜复验：待下轮
- [轻微·格式] index.html 多处 U+2212 与 LaTeX 混用（1181-1182、769、780、1083、1313）｜修复：统一为 KaTeX 渲染｜复验：待下轮
- [轻微·可读性] index.html:808：MLA、KDA 首现无全称｜修复：首现处补全称｜复验：待下轮
- [轻微·格式] index.html:1337：段落 p 未闭合｜修复：补 </p>｜复验：待下轮
- [轻微·技术] index.html:1250：LongRoPE 来源不全｜修复：补 arXiv 编号｜复验：待下轮
- [轻微·格式] overview.html head 缺 meta 标签｜修复：overview 为概览页，与 index 互链即满足发布条件（编排者确认：overview 模板无此要求）｜复验：不适用

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 8
- 处置：修复后进入第 2 轮
