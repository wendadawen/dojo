# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-8.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**3** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**35** 条
- 有理由不改：**0** 条

## 修复与判定说明

通读 wiki/strata/research/ 全部 8 份 review-*.md（review-1…8），汇总全部轻微级报告共 38 条（R1=10、R2=5、R3=8、R4=6、R5=2、R6=1、R7=3、R8=3；其中 R3 的“图前重复描述”与 R4 的“图 4/5/6/12 重复引入句”为同类、不同位置，分开计）。逐条回查当前页面：35 条在当前页面已不存在（历轮修复的副作用，R1/R2/R5 有修复台账，R3/R4/R6/R7 的修复未落台账但页面已改），另 3 条为第 8 轮报告且确实未处理，本轮修掉。实际改动（文件 /Users/wendadawen/code/dojo/wiki/strata/index.html、/Users/wendadawen/code/dojo/wiki/strata/overview.html）：(1) index.html §核心问题 5 解答与 overview.html「适用边界」删除无原文依据且非论文术语的边界项「模型结构（稀疏注意力）」，与页面“分析性判断只在第 6 章”的自我声明一致；(2) index.html §6.3 把 CacheGen/CacheBlend 一句的「前者/后者」改为明确所指（Strata 走精确前缀缓存路线，CacheGen/CacheBlend 走近似缓存），与 §6 “approximate caching schemes” 原文口径一致；(3) index.html §4.4 删除「（高请求率下必然出现）」这一论文未给的条件化断言。已 grep 复核：稀疏/前者/后者/必然出现 均为 0；并全站复核无 research/ 路径、无 alt 内 $...$、无 Unicode 数学字符、无 “Nx” 写法、无“本文/本页/我们/本章(元话语)”残留、无 2.5GB/dedecode/20 KB token/惨不忍睹/不是口号 等旧问题文字。overview 与 index 数字、编号一致；两页 validate 通过。
