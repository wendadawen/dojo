# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-4.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**3** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**15** 条
- 有理由不改：**0** 条

## 修复与判定说明

清掉 /Users/wendadawen/code/dojo/wiki/rmsnorm/index.html 积压的轻微问题。读完 research/ 下全部 4 轮 review（review-1..4），汇总轻微级问题共 18 条（R1 4 条、R2 6 条、R3 5 条、R4 3 条；其中「C 编号断号」在 R2 与 R3 复报为同一问题）。

判定：3 条仍在、已修并复验；15 条经历轮修复后已不存在，未重改。

【本轮修掉并复验的 3 条】
1｜[轻微·技术]｜review-4｜index.html:96 核心问题 Q4 解答 summary｜「实现普遍加 $\varepsilon$ 并在 fp32 里算」把 fp32 写成普遍做法，超出来源（[C8] 仅证 GLM 上浮 fp32）｜改为「实现普遍加 $\varepsilon$；GLM 另把统计量放在 fp32 里计算」，与正文 L290/L97 口径一致。
2｜[轻微·技术]｜review-4｜index.html:245「证据二」｜「论文系统对比了两种方法的不变性集合（其 Table 1）」把来源表格范围缩窄为两种方法（Table 1 并列 BatchNorm/WeightNorm/LayerNorm/RMSNorm/ppRMSNorm）｜改为「论文在 Table 1 中并列各归一化方法的不变性；其中 RMSNorm 相对 LayerNorm 只少了权重矩阵 re-centering，其余缩放不变性（权重矩阵缩放、数据集缩放、单样本缩放）保留」，与 [C3] 来源条目一致。
3｜[轻微·技术]｜review-4｜index.html:387「简化条件及其限制」｜「大语言模型上的『质量相当』由后续广泛采用间接支持」为无来源事实判断｜删去该分句，保留「出处只限论文实验与可定位的 GLM 实现源码，不自行外推训练结论」。

【判定为「已不存在」的 15 条（历轮已修或副作用消失，未重改）】
R1-2 相邻双上标 [C6][N2]→现为 [C5, N2]，无 </sup><sup>；R1-3 均值「与 LayerNorm 相当」超出来源→现显式声明论文未作该判断；R1-4 PyTorch ε 位置描述与事实不符→现「都加在根号内，仅默认值不同」并补 [C9]；R1-5 引言「主流大模型…用的都是它」→现收窄为「已被 transformers 的 GLM-4-MoE 实现采用」；R2-1 源码路径未注明版本→meta/[C8]/[C9] 现均标 transformers 4.57.6、torch 2.8.0；R2-2 引言「GLM 官方实现采用」超出来源→同 R1-5 已收窄；R2-3 pRMSNorm/BLEU 未解释→L288/L107 均已补定义；R2-4 式(1) 后 ul 未定义 m/j→L122 已补两条；R2-5 与 R3-4 C 编号断号（缺 C1/C4）→已重排为连续 C1–C9；R2-6 LayerNorm 前置概念链接缺失→L65 现有 residual-connection 内链，且 scope.md §1/§4 已记录「LayerNorm 无专页、页内以最小定义自包含」的判断；R3-3 核心问题第 3 题「各类 re-scaling 不变性」过宽→L90 已改为精确列举；R3-5「证据二」段无上标/来源→已补 <sup>[C3]</sup>；R3-6 引言末句元话语→已删，仅留 L67 结构说明；R3-7「大语言模型场景」把「场景」当术语→已删「场景」；R3-3 核心问题第 3 题（见上）。

【一致性/机械复验】正文 <sup> 上标集合与来源条目 C1–C9/F1–F5/N1–N2 双向闭合；无 research/ 死路径、无 glm5_next/GLM-5.3-Flash/MISSING 链接（residual-connection 与全部 libs 均存在）；无 Unicode 数学字符；img 仅一个 lightbox 空 alt，无 $...$；overview.html 与 summary 的对应表述（GLM 才在 fp32 算、不变性）已与本轮改动一致，无需同步。仅改动 index.html 3 行（+3/-3），未做无关重构。
