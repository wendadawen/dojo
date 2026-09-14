# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-7.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**2** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**20** 条
- 有理由不改：**0** 条

## 修复与判定说明

读遍 research/ 下 review-1..7.md，汇总历轮全部 22 条轻微问题并逐条对照当前页面。20 条已不存在（历轮修复或其副作用），2 条仍存活并已修掉、复验通过。

【已修 2 条】
1) review-7 [来源]：§4.3「窗口之外的另一路可见性（例如压缩后的全局 KV）」原挂 [C4]，但 [C4] 只登记 attn_sink 参数、稀疏注意力分母与 checkpoint 张量头，读不出该例依据。已把官方架构说明（wiki/deepseek-v4-1/research/official/README.md 的 CED/CSA2 段：解码器全局 KV cache 由编码器末层隐状态投影、压缩到约每 token 890 字节）补进 [C4] 条目，使该论断可按引文核对。此修复同时闭合 review-1 第 8 条（同一断言的引用）。
2) review-1 第 6 条 [可读性]：正文前置概念链接此前只补了 KV cache 与滑动窗口注意力，缺「标准注意力」。已在第 2 章首次依赖 softmax 归一化处补 ../standard-attention/index.html 链接，与 overview.html 前置概念映射一致。

【已不存在 20 条】（按轮归类，均逐条在当前页面核对）
· review-1：3.2 表内 Unicode ×、来源 h3 命名、组合引用 [C3][N2]→[C3, N2]、符号表补 $o$ 与复用符号注明、解析解折叠块补 <sup>[C5]</sup> 及 [C6]/[N3] 正文引用、3.1 crucial 设计表述、4.2「训练可调成两种极端」断言、description「份额实测」。
· review-2：空隐藏 <ul style="display:none">、dojo:tag 与 topics 同值、[N3] 实验条件。
· review-3：h3 缺（N）、[C5]→[C7] 跳号（已重编为 C6）、零钱罐死引用。
· review-5：三处元话语引导句（先看观察到的事实／常被忽略的后果／需要明确的是）。
· review-6：W=128 构造取值未交代、「本页」自我指代、[C3] 引文出处归于 §1、summary「43 层 × 64 头」。

【不改 0 条】未出现审查要求与 guides 规定相抵触的情形；unfixed 为空。

一致性：改动不涉及任何数字、份额值或引文编号，summary/overview 无需同步；正文所引路径 research/measured.md、wiki/deepseek-v4-1/research/measured.md、wiki/deepseek-v4-1/research/official/README.md 及三条前置概念页链接均真实存在；无 alt 中 $...$；无相邻 <sup>。
