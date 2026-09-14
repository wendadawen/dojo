# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-5.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**4** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**19** 条
- 有理由不改：**0** 条

## 修复与判定说明

通读 research/ 下全部 5 份 review（review-1..5）。去重后共 23 条轻微问题：其中 19 条为历轮已修复（当前页面中已不存在，仅作核对不重改），4 条为 review-5 报告但未处理的积压，本轮全部修掉并复验。

本轮实修 4 条（均在 index.html）：
1. [轻微·图示] 第 237 行时间线图底部 <text> 右边界溢出：删除句中"（示意图，非实测 trace）"，并把该说明移入同 figure 的 figcaption；headless Chrome 精测右边界 770.8→647.2，已回到 viewBox 宽 680 内。
2. [轻微·来源] 来源表 C9 dispatch 区间与源码不符：由"329-599（dispatch）"改为"329-668（dispatch，末段 600-668 为 workspace 清理）"，六段区间现连续覆盖 329-1453。
3. [轻微·来源] 来源表 C28 括注的英文名"Chenggang Zhao"在 PR #304 Contributors 中定位不到：删去括注，恢复为"八位贡献者"。
4. [轻微·表述] 4.1 节第一人称"rank0 也把"我贡献了哪些 token-专家对"的源索引写了过来"：改为"rank0 也把本 rank 有哪些 token-专家对的源索引写了过来"。

判为「已不存在」（19 条，历轮修复的副作用，逐条 grep 复核确认当前页面已是修复后状态）：
- 通信量单位混算 21.5 MB（review-1/2 复报同一问题，现为十进制 43 KB/86 KB/22 MB）；4.3"从 TMEM 读出乘积"（现为 gate 与 up 两半）；"DeepSeek 基础设施团队"（已删）；C4 指向未存档 sm100_bf16_mega_moe.hpp（现为官方仓库定位，无"（存档）"）；N1–N4 未单独定义（N 表已拆并连号）；Unicode ×（全页 0 处）；3.1/图2 token 槽 ASCII"t0"（review-1/3 复报同一问题，现为 foreignObject+KaTeX）；引言 SM/rank/warp/TMA 术语先于定义（引言已补定义句）；overview"FP8×FP4"不一致（已统一 FP8xFP4）；SVG<text>"top-k 分槽"/"输出 y"（已改）；PDL 无全称（已补）；4.1"几十字节的元数据"（现为 8 字节计数/4 字节索引）；UMMA 无解释（已补全称）；2.1"两组计算线程"（现为三组）；4.3 clamp 表述不完整（现为 gate 上限、up 上下限）；commit 559d79f 无存档（已补 sources/commit-info.md）；AMD Primus 无来源（已补 sources/primus-extracts.md）；N 表断号 N9→N11（现 N1–N11 连号）；元话语"值得注意/容易忽视"等（全页 0 处）。

另按任务要求专项核查：正文与来源说明无指向仓库不存在文件的路径（research/ 仅引 sources/ 下现存 .md 与官方仓库定位）；alt 中无 $...$；index/summary/overview/图注间数字与编号一致；未改动的 summary/overview 无需同步。无因与 guides 规范冲突而拒修的条目。
