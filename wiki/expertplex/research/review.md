# ExpertPlex 独立审查

- 审查者：独立上下文（AI 模拟 / 真实目标读者）
- 页面版本：index.html 工作树哈希 `cb1677265427c65d01d14e83bf184f67cdd4bc90`；overview.html 工作树哈希 `575458c595af29014e7424841b312761c1fb22ab`
- 论文版本：arXiv:2607.18002v2（TeX 源码在 `research/src/`）
- 时间：2026-08-07

审查依据：`guides/paper/check.md` 任务包。输入仅限 `index.html`、`overview.html` 和论文原文（`research/src/sections/*.tex`、`research/src/figures/*.pdf`、`research/src/paper.tex`）；未读 `research/` 下的 `scope.md`、`evidence.md`、`outline.md`、`glossary.md`、`draft-check.md` 等过程产物，未修改两份被审文档。

## 段 A 盲读小结

按完全小白视角顺序阅读两份页面。主线（PDD 死结 → colocation 死结 → 共享专家+分离 attention → APK → 一侧通信 → 跨栈优化器 → 实验 → 独立评价）可以走通：每个机制都先讲"解决什么死结"、再讲"怎么做"、再讲"为什么这样做能成立"，章末自检题与误解澄清对应到位。核心卡点集中在术语首现：APK 作为全文核心机制，缩写在正文首次出现时未给中文展开。贯穿场景的具体数字（EP4、17.7–34.7μs、1.8–2.9ms、84–101×、50ms/token）给得早、量级反差清晰，但 EP4 与 50ms/token 这两个具体取值的含义在首现处未点明。

## 段 B 对照原文小结

逐条核对任务包要求的所有核查项：

1. **数字与论断**：核心数字 11.3 req/s/node（§7.2）、2.01×/1.41×（MiniMax+ShareGPT，§7.2）、1.66×（GLM+LooGLE，§7.2）、4.12×/1.28×（MiniMax+LooGLE，§7.2）、3.3×/5.0×/1.5×/2.5×（GLM，§7.2）、84–101×/17.7–34.7μs/1.8–2.9ms（§4.1）、2.2–25.3μs（§4.2）、<10.7μs（§7.6）、+8%/1.12×/13.79×/3.33×/4.07×（§7.3）、<12%/<20μs/<10%（§7.4）、~5%/~45μs（§7.5）、REEF 35μs（§7.6）、160/50 GB/s 与 3.2×（§5.1）、95%/96%/98%（§2.1）逐条与 `sections/evaluation.tex`、`sections/design.tex`、`sections/background.tex`、`sections/introduction.tex` 比对一致；实验条件（SGLang 系基线、H800、FP8、ShareGPT/LooGLE、Poisson、四组 SLO、3 节点、GLM PDD OOM、部分基线 16GPU）与 §7.1 一致；2.01× 与 1.66× 的成立条件在正文与 §7 评价中均给出（MiniMax+ShareGPT 对 PDD、GLM+LooGLE 对 PDMux），未扩大为"全面碾压"。
2. **实验覆盖**：四组（model × workload）端到端对比表完整呈现，包括 MiniMax+LooGLE 上 ChunkedPrefill 无法满足 SLO、GLM+ShareGPT 上 ExpertPlex 与 PDMux 持平的反例均如实列出，未被省略。
3. **公式与推导**：F1–F4 与论文 §6.1 Eq.(1)、§6.2 Eq.(2)/(3)、§6.4 Eq.(4) 逐符号一致；goodput 取 min 的手算示例（$B_p=8,B_d=64,T_p=0.5,T_d=0.05,\bar O=20$）算术正确（$B_p/T_p=16$，$B_d/(T_d\bar O)=64$，$G=16$），且明确标注为教学构造、非论文实验数据。
4. **可运行代码**：两份页面均未声称包含可运行代码块，无此项需复跑。
5. **事实与推断**：正文 C/F/N 编号引用全部为论文事实，第 7 章独立评价整章用 callout-gray 显式标注"内容属于解读者推断，不是论文的结论"，事实与推断边界清晰。
6. **原图**：嵌入的 Figure 2/3/4/5/6/11 描述与 `sections/*.tex` 中各 `\caption` 及正文段落一致（Figure 2 双失败模式、Figure 3 三类服务器、Figure 4 存储层级传播、Figure 5 跨阶段重叠、Figure 6 tile 足迹、Figure 11 Pareto 前沿），编号未错位。
7. **前置知识引用**：`../moe-serving/index.html` 与 `../gpu-execution-model/index.html` 经 `ls` 确认两目录均存在且各自含 `index.html`/`overview.html`，链接层级正确（`../<name>/index.html`）。
8. **教学简化**：goodput 手算示例、贯穿场景的"2 毫秒/30 微秒"反差、Figure 7–10/12–15 用表格与文字汇总代替原图，均在"来源与教学说明"中标注为教学构造或简化，未导致核心结论失真。
9. **页面功能**：`python3 .dojo/scripts/validate.py` 对两份页面退出码均为 0；用仓库本地 `libs/katex.min.js` 在 node 中渲染 F1–F4 及 15 个内联片段全部成功；前置概念页链接有效。

## 问题

- [重要·技术] index.html 第 628 行（作者元信息）：作者列表列出 8 位（"Bingyang Wu、Chao Jin、Zili Zhang、Xinming Wei、Yinmin Zhong、Ruidong Zhu（北京大学）；Chengxu Yang、Yuliang Liu（Independent Researcher）"），但论文 `paper.tex` 第 38–40 行实际列 9 位作者，遗漏了排在第 8 位的 "Xin Jin（北京大学）"（位于 Chengxu Yang 与 Yuliang Liu 之间）。：在 index.html 第 628 行作者列表中"Ruidong Zhu"后补一位作者，并在北京大学一组中加入"Xin Jin"，使页面作者数与论文 paper.tex 一致（9 位）。 ｜ 修复：已改。作者行补为"Bingyang Wu、Chao Jin、Zili Zhang、Xinming Wei、Yinmin Zhong、Ruidong Zhu、Xin Jin（北京大学）；Chengxu Yang、Yuliang Liu（Independent Researcher）"，共 9 位 ｜ 复验：通过——第 628 行已补回 Xin Jin 并归入北京大学组，共 9 位，与论文 paper.tex 第 38–40 行作者数一致、归属正确
- [重要·盲读] index.html 第 641 行（学习目标）与第 706 行（正文首次出现 APK 处）："APK" 缩写在正文首次使用前未展开为中文全称。论文英文标题（第 627 行）虽含 "Adaptive Persistent Kernels"，但页面是中文页面、目标读者是完全小白，且 overview.html 第 54 行已写过"自适应常驻 kernel（APK）"的展开；index.html 正文从未给出这一对应，读者需自行从英文标题反推。：在 index.html 正文首次出现 APK 的位置（建议第 634 行末段或第 641 行学习目标首次提到 APK 时）首次使用处写为"自适应常驻 kernel（APK）"，与 overview.html 第 54 行保持一致；后续沿用 APK。 ｜ 修复：已改。开头段首次出现处改为"自适应常驻 kernel（Adaptive Persistent Kernel，APK）" ｜ 复验：通过——第 634 行开头段首次出现处已展开为"自适应常驻 kernel（Adaptive Persistent Kernel，APK）"，早于第 641 行学习目标，后续沿用 APK，与 overview.html 第 54 行一致
- [轻微·技术] index.html 第 942 行（来源与教学说明·核心论断与原文定位）：来源列表登记了 "C19（H100 MIG 1g/2g/3g/4g/7g，3g-4g 唯一用满）§4.1"，但正文从未出现 [C19] 引用——第 737 行讨论 MIG 时引用的是 [C7]，C19 是登记了但未在正文使用的来源。：二选一——(a) 在第 737 行 MIG 一句后补一句"H100 只暴露 1g/2g/3g/4g/7g 五档，唯一能让两阶段合用满卡的两路划分是 3g–4g [C19]"并保留来源条目；或 (b) 从第 942 行来源列表删除 C19 条目。 ｜ 修复：已改。采用 (a)，MIG 一句补"H100 只暴露 1g/2g/3g/4g/7g 五档，唯一能让两阶段合用满卡算力的两路划分是 3g–4g"，引用改为 [C7][C19] ｜ 复验：通过——第 737 行已补 H100 五档与 3g–4g 说明，引用改为 [C7][C19]，C19 在正文被引用，与 design.tex 第 32 行一致；来源列表第 942 行 C19 条目不再悬空
- [轻微·盲读] index.html 第 649 行（贯穿场景首段）："EP4" 首次出现未在本页解释其含义。EP 链接到 `../moe-serving/index.html`，但 "EP4" 这个具体取值（4 路专家并行）未点明，小白读者需跳转概念页才能确认。：在第 649 行 "EP4" 后括号补"4 路专家并行"，例如 "在 EP4（4 路专家并行）下这种模型的…"。 ｜ 修复：已改。补为"在 EP4（4 路专家并行）下" ｜ 复验：通过——第 649 行 EP4 后已补"（4 路专家并行）"，小白无需跳转概念页即可理解
- [轻微·盲读] index.html 第 649 行（贯穿场景首段）："几十个已开始的请求正以每 50 毫秒一个 token 的速度 decode" 未说明这 50ms 是 MiniMax+ShareGPT 的 TPOT SLO（第 869 行才在 SLO 表中出现），读者会以为是实测平均速度。：在第 649 行该句尾注明 SLO 来源，例如 "…正以每 50 毫秒一个 token 的 TPOT SLO 速度 decode（见第 6 章 SLO 表）"。 ｜ 修复：已改。补为"以每 50 毫秒一个 token 的速度 decode（这个 50ms 是 MiniMax-M2.7 在 ShareGPT 上的 TPOT SLO，见第 6 章设置表）" ｜ 复验：通过——第 649 行已补 TPOT SLO 来源说明，50ms 与第 869 行 SLO 表"MiniMax+ShareGPT 1s/50ms"一致

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：进入修复 → 复验通过。阻断为 0，2 条重要问题（作者列表遗漏 Xin Jin、APK 缩写首现未展开）与 3 条轻微问题（C19 悬空、EP4 未释义、50ms 未标 SLO 来源）均已修复并复验通过，未触发第二轮。修复未触及研究范围或教学大纲。`validate.py` 两份页面退出码均为 0（已确认）；KaTeX 渲染与前置概念页链接在初轮已验证、本次修复未改动公式与链接，无需复跑。
