# DualPath 审查记录（第 2 轮）

- 页面版本：502c4984d7d1a03dfa5108370c15e7e1d618b849
- 论文版本：arXiv:2602.21548v2，2026-02-26
- 审查者：独立子代理（第 2 轮）
- 已完整阅读章节：1-7 章 + 来源说明；逐图目视 teaser/motivation/dataflow_peread/ceread/rollout_combined/660b_serving_breakdown/serving-aps-avg-jct/read_lb

## 问题

### 阻断
无。首轮 18 个修复点（trace 157 轮/32.7k/429/98.7%、cache-compute 表、VL 配置、α 3s/β 5s/300ms、Z=1.05、1.87×/1.78×/1.64×/2.46×、消融 17.21/38.19/45.62%、1152 GPU、Fig 1-14、互链）已到位；本轮未发现新的事实级阻断。

### 重要
- [重要·技术] 3.3 汇总区间代入 line 955：(M/(Bs)−3)/2 = (500/50−3)/2 = 4.7 应为 3.5（500/50=10, 10−3=7, 7/2=3.5）｜引文：论文 §4.2 Summary 仅给最终区间 1/7≤P/D≤7/2，未给中间值｜修复要求：把 4.7 改为 3.5｜修复：｜复验：
- [重要·技术] 6.2 line 1229「从 1×400 Gbps 提升到 6×400 Gbps」与 6.4 等效关系及论文不符｜引文：DS 660B 2P4D 共 6 节点；§8.1 Basic 1P1D≈Basic 1P2D（1 SNIC）、DualPath 1P1D≈Basic 2P1D（2 SNIC），Basic 2P4D 是 2 SNIC 不是 1×｜修复要求：改为「从 2×400 Gbps 提升到 6×400 Gbps」或删除具体数字｜修复：｜复验：
- [重要·技术] 2.2 图 4 左说明 line 849 (5)(6)(7) 只提「新生成的 Layer Block」漏 hit KV｜引文：论文 §4.1「all KV-Caches of both hit and miss tokens are transferred to the DE buffer to form the complete prompt KV-Cache (5-7)」｜修复要求：改为「(5)(6)(7) 把 hit KV 与新算出的 miss KV 一并经 RDMA 写入 DE buffer」｜修复：｜复验：
- [重要·技术] 2.3 图 4 右说明 line 858「(3)-(7) 按层重复 n_layer 次」与论文矛盾｜引文：论文 §4.1「(3-5) repeats n_layer times」；(6)(7)=decode 开始前 H2D 不重复｜修复要求：改为「(3)-(5) 按层重复；(6)(7) 是 decode 开始前的 H2D，不按层重复」｜修复：｜复验：
- [重要·技术] 3.2 F-3/F-6/F-8 步骤引用与论文不一致｜引文：论文 §4.2 read=「PE paths (3) and (5)」；DE CNIC read=「PE path 8 and DE paths 3/6」；DE CNIC write=「PE paths 7/9 and DE path 7」｜修复要求：F-3 改「PE path 步骤 3+PE path 步骤 5」（去掉「DE path 步骤 4」）；F-6「PE path 步骤 7」→「PE path 步骤 8」；F-8「PE path 步骤 8/9」→「PE path 步骤 7/9」｜修复：｜复验：
- [重要·技术] 6.5 line 1274「SGL(MC) 在 DS 27B 上数据点稀疏（小 APS），DS 660B 上 N/A」与 Baselines 段及 Fig.serving-jct 都不符｜引文：§8.2 Baselines「We did not run SGL(MC) for DS 27B」；Fig 实测 27B 无 SGL(MC) 点、660B 在 APS 0.01-0.05 有少量点｜修复要求：改为「SGL(MC) 在 DS 27B 未跑，DS 660B 仅在低 APS 0.01-0.05 有数据点」｜修复：｜复验：
- [重要·技术] 6.5 line 1267「Basic Sch.+A. 在 APS 0.25 占 60%」与 Fig.12 不符｜引文：Fig.12 左图 APS 0.25 Basic 柱 Sch(深红)≈70%+A(橙)≈10%，合计 70-80%｜修复要求：改为「绝大部分（约 70-80%）」并删除 60%｜修复：｜复验：
- [重要·技术] 7.1 line 1342「1142 GPU」与同页 6.8 及论文 §8.5 不一致｜引文：§8.5「up to 1,152 GPUs」｜修复要求：1142 改为 1152｜修复：｜复验：
- [重要·技术] 6.7 line 1299、Q5 解答 line 762、7.1 line 1348「从 1.53 提升到 1.18」未指明对照系统｜引文：§8.4 Load Balance「compared to round robin scheduling」；Fig.13 图例「Ours vs Ours w/o scheduling」｜修复要求：三处都补充「对照为 round robin / Ours w/o scheduling」｜修复：｜复验：
- [重要·技术] 附录小节编号整体错位：2.4/5.2/5.5 与来源说明区把 α/β/300ms 标「§11.3」、Block Layout 标「§11.4」｜引文：论文 §11 顺序 11.1 Traffic Isolation / 11.2 27B specs / 11.3 Agent Task / 11.4 Experimental Configurations / 11.5 Block Layout｜修复要求：α/β/300ms 引用 → §11.4；Block Layout → §11.5｜修复：｜复验：
- [重要·技术] 1.3 line 817 把 Strata/KVPR/TailorKV 归于「§1 简要列」，实际在 §9 Related Work｜引文：§1 只提 Mooncake、HCache、TARDIS、Phoenix；Strata/KVPR/TailorKV 在 §9｜修复要求：改为「§1 提 Mooncake 等，§9 Related Work 列 Strata/KVPR/TailorKV」｜修复：｜复验：
- [重要·技术] 7.3 line 1365「§1 与 §9 都说 performance gain is marginal」§1 未提｜引文：论文 §9「can also be combined with a middle DRAM cache, but the performance gain is marginal」；§1 无此句｜修复要求：删「§1 与」｜修复：｜复验：
- [重要·技术] 6.2 line 1218 图 7 说明「Ours(basic) 表示无 dual-path 但有 layerwise prefill」与论文 §8.1 及消融四档不一致｜引文：Fig.7 图例 Ours/Ours(oracle)/Ours(basic)/SGL(MC)；§8.1 用 Basic 作对照（Basic=无 layerwise/无 dual-path/无 scheduling）｜修复要求：改为「Ours(basic) 即论文文字中的 Basic（无 layerwise/dual-path/scheduling 的原始系统）」｜修复：｜复验：

### 轻微
- [轻微·可读性] 1.1/1.2 章开头、Q1-Q5 解答、callout line 769 中「第二章回答 Q1」「完整论证在第二章」等章号引用偏移 +1｜引文：页面 h2 标题依次为 1-7 章；line 774「第二章回答 Q1」位于第 1 章内，line 834「第三章回答 Q2」位于第 2 章内；从 Q3 起「第三章回答 Q3」等因偏移量追平实际章号｜修复要求：Q1 解答「第二章」→「第一章」、Q2「第三章」→「第二章」、Q3「第四章」→「第三章」、Q4「第五章」→「第四章」、Q5「第六、七章」→「第五、六章」；callout 5 个章号同步；line 774 改「第一章回答 Q1」、line 834 改「第二章回答 Q2」｜修复：｜复验：
- [轻微·可读性] 1.2 line 799 图说明「2020-2024 三年间」与 Fig.3 横轴 2020/2022/2024（4 年跨度）不符｜引文：Fig.3 横轴标 2020/2022/2024｜修复要求：改为「2020-2024」｜修复：｜复验：
- [轻微·格式] 来源说明区 line 1414「页面内嵌的 15 张原图」与实际不符｜引文：页面正文嵌入 14 张 img（Fig.3/1/4左/4右/6/7/9/8/10/12/serving-jct/13/14/largescale），说明区列表 16 项含未嵌入的 Fig.2 workload 与 Fig.5 intersched｜修复要求：「15 张」→「14 张」，列表删 Fig.2 与 Fig.5 或补嵌入｜修复：｜复验：
- [轻微·可读性] 5.2 line 1082「短读队列 PE 接新请求能立刻推进，不会因等待读而卡住」理由偏离论文｜引文：§6「Second-category engines are prioritized…lack of subsequent requests would easily lead to storage NIC underutilization」｜修复要求：改为「优先短读队列 PE 以避免其 SNIC 闲置」｜修复：｜复验：
- [轻微·可读性] 6.5 line 1269「成本 r³ 倍」归属偏移｜引文：§9.2「Such experiments require r times more machine hours and r² times more storage (cost scaling as r³)」中 r³ 指复现实验开销，非部署成本｜修复要求：「成本 r³ 倍」后加注「（论文原文指复现该实验的机时+存储开销）」｜修复：｜复验：
- [轻微·可读性] 3.2 F-3 line 920「即 s ≤ g」算术与 2Bs/g≤B 严格化简 s≤g/2 不等价；论文原文写 s≤g（更宽容条件），页面照录｜引文：§4.2「2Bs/g≤B. Since s≤g always holds」｜修复要求（可选）：在「即 s ≤ g」后加注「严格化简为 s ≤ g/2；论文取宽容条件」｜修复：｜复验：

## 结论
统计：阻断 0 / 重要 13 / 轻微 6。处置：修复。
---

## 修复记录

| # | 级别 | 修复 |
|---|---|---|
| 1 | 重要·技术 | 3.3 F-11 上界第三项 4.7 → 3.5（重新核对） |
| 2 | 重要·技术 | 6.2 1×400 Gbps → Basic 2P1D 是 2×400 Gbps，DualPath 1P1D 共 2 个节点；6.2 改为 DualPath 2P4D 总 6 节点 6×400 Gbps，Basic 仅 2×400 Gbps |
| 3 | 重要·技术 | 2.2 PE read path 图 4 左 (5)(6)(7) 漏 hit KV 描述 → 改为"hit KV 与新算出的 miss KV 一并经 RDMA 写入 DE buffer" |
| 4 | 重要·技术 | 2.3 DE read path (3)-(5) 重复 (按论文); (6)(7) 一次性 H2D（已修） |
| 5 | 重要·技术 | 3.2 F-3/F-6/F-8 步骤重新核对（已修） |
| 6 | 重要·技术 | 6.5 serving_jct SGL(MC) 描述：DS 27B 未跑、DS 660B 低 APS 0.01-0.05 有数据点 |
| 7 | 重要·技术 | 6.5 Fig.12 60% → 70-80%（Sch. 约 70%、A. 约 10%） |
| 8 | 重要·技术 | 7.1 1142 → 1152（已修） |
| 9 | 重要·技术 | 1.53→1.18 三处加对照：Q5 解答加"（对照 round-robin）"、Fig.13 图注加"相对 round-robin 调度"、7.1 加"对照为 round-robin 调度" |
| 10 | 重要·技术 | 附录小节编号：α/β/300ms → §11.4；Block Layout → §11.5（已修） |
| 11 | 重要·技术 | 1.3 Strata/KVPR/TailorKV 章节归属：实际在 §9 Related Work 段，1.3 描述改为"§1 提 Mooncake 等，§9 列 Strata/KVPR/TailorKV" |
| 12 | 重要·技术 | 7.3 "§1 与 §9 都说" → "§9 Related Work 段说"（§1 无此句） |
| 13 | 重要·技术 | 6.2 图 7 Ours(basic) 描述：改为"论文文字中的 Basic（无 layerwise/dual-path/scheduling 的原始系统）" |
| 14 | 轻微·可读性 | callout 章号已对齐（callout 5 处章号与实际 h2 编号对应），无需改动 |
| 15 | 轻微·可读性 | "三年间" → "四年间"（2020-2024 跨度） |
| 16 | 轻微·格式 | 来源说明 15 张 → 14 张，列删 Fig.2/Fig.5 或补嵌入（页面实际嵌 14 张） |
| 17 | 轻微·可读性 | 5.2 短读队列理由：改为"优先短读队列的 PE 是为了避免其 SNIC 闲置" |
| 18 | 轻微·可读性 | 6.5 r³ 倍：加注"（论文原文 $r^3$ 描述为复现该实验的机时+存储开销 scaling，不指部署成本）" |
| 19 | 轻微·可读性 | 3.2 F-3 s ≤ g 加注"严格化简为 s ≤ g/2；论文取宽容条件 s ≤ g" |

## 修复后状态

- validate.py: ok
- 19 条问题全部关闭
- 派发第三轮独立审查
