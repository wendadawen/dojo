# 辅助损失无关路由 独立审查

- 审查者：独立上下文（AI 模拟小白读者）
- 页面版本：index.html = 5282787bd4bec5a3e73333aef4259c4e3b52514c；overview.html = 33c8c31280549cc4bfc370b970b3cb33c13dd1c1
- 时间：2026-08-09
- 来源：WebSearch "DeepSeek-V3 auxiliary-loss-free load balancing arxiv 2412.19437"；WebSearch "arxiv 2408.15664 Auxiliary-Loss-Free Load Balancing"；WebSearch "DeepSeek-V3 sequence-wise balance loss complementary layer 1 warmup"
- 审查范围：wiki/aux-loss-free-routing/index.html + overview.html（禁止读 research/、其他页面、修改文档）

## 段 A 盲读小结

按 S1→S5 顺序通读，主线可读，五条学习目标均由正文章节完整回答（S1 答"为何均衡/辅助损失缺陷"、S2 答"bias 前向角色/不引入干扰梯度"、S3 答"sign 更新规则/为何不用幅度"、S4 端到端手算、S5 答"V3 配置/边界"）。手算例子数字自洽，三轮路由统计与 F4 套用均复算无误。误解清单与正文论述一致。

## 段 B 对照来源小结

F1-F5 公式、bias"只进选择不进权重"的核心论断、sign 更新规则、γ=0.001、α=0.0001、671B/37B、256+1、top-8、14.8T token 均与 arXiv:2408.15664 摘要与多个 V3 报告解读一致。sequence-wise loss 的作用（处理序列内均衡、与批级 aux-loss-free 互补）经 real-zhangzhe、aipapernotes、kuazhi 等多源交叉确认。Table 3 的 sign(9.50/0.044) vs 幅度(9.51/0.036) 数据与原文一致。链接目标 ../moe-serving/index.html、../quantile-balancing/index.html 均存在；index.html 与 overview.html 互相链接。

## 问题

- [重要·盲读] index.html S5 F6 公式：F6 使用 $s'_{i,t}$（$P_i=\frac{1}{T}\sum_t s'_{i,t}$）但未给出 $s'_{i,t}$ 的定义；V3 报告 Eq.19 定义 $s'_{i,t}=s_{i,t}/\sum_j s_{j,t}$。小白看到 $s'$ 不知道它和 $s$ 的关系，且 F6 是学习目标 5 的边界依据：在 F6 后补一行 $s'_{i,t}=\frac{s_{i,t}}{\sum_j s_{j,t}}$ 并标注"归一化路由分数"，或在 F6 内联展开 ｜ 修复：已在 F6 后的符号说明中补"$s'_{i,t}=\frac{s_{i,t}}{\sum_j s_{j,t}}$ 为归一化路由分数（$\sum_j$ 对该序列内全部 routed 专家求和）" ｜ 复验：
- [重要·盲读] index.html S5 context-box "推理"行 + overview.html "关键结论与边界"：当前写"直接用 $s_{i,t}$ 路由（已训好的 bias 嵌入模型权重配置）"。yudonglee 与 V3 报告原意是"训练完成后 bias 被吸收进 router 权重 $e_i$，推理无额外开销"。当前表述可能让小白误以为推理时完全不加 bias 也不调整 router 权重：改为"推理：bias 不再单独存储，已被吸收进 router 权重 $e_i$；前向按 $s_{i,t}=\sigma(u_t^\top e_i)$ 路由，无需显式加 $b_i$" ｜ 修复：已将 index.html S5 context-box"推理"行、S5 边界强调点、overview.html"关键结论与边界"三处统一改为"bias 被吸收进 router 权重 $e_i$，前向按 $s_{i,t}=\sigma(u_t^\top e_i)$ 路由，无需显式加 $b_i$" ｜ 复验：
- [轻微·技术] index.html "核心公式与来源" F6 条：引用范围写"Eq.17-19"，但 F6 含 L_Bal(Eq.17)、f_i(Eq.18)、P_i(Eq.20) 且使用了 Eq.19 的 $s'$。aipapernotes 转录确认 Eq.17-20：改为"Eq.17-20" ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 开头 callout："专家 0 被选中 90% 的次数"未标注为教学构造数字，S3/S4 教学示例均明确标注，开头独漏：在"90%"后加"（教学示意）"或改为"绝大多数" ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html S3 第一强调点："用历史 batch 的负载更新"与主文"统计本步每个专家收到的 token 数"表述不一致；小白可能误以为用很久以前的 batch 统计：改为"用本步刚结束的 batch 负载统计更新（不依赖未来 batch 信息）" ｜ 修复： ｜ 复验：
- [轻微·技术] index.html S3 第一强调点："因果约束，arXiv:2408.15664 §3.2"——arxiv-vanity 显示 §3.2 附近 future token leakage 讨论出现在与 Expert Choice 对比部分（Figure 9）；是否同时作为 bias 更新本身的设计依据需作者复核原文：核对 §3.2 原文归因；如该因果约束仅在 Expert Choice 对比中讨论，删除括注或将引用改为对比小节 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html S5 + overview.html：K3"896 专家"与 Quantile Balancing 替代 sign 超出本页给定来源（arXiv:2408.15664 + arXiv:2412.19437）范围，仅由 ../quantile-balancing 概念页承担；llms3.com 提到 Kimi K2 系列采用 aux-loss-free 但未含 896/Quantile 细节：在 K3 句后加来源定位（K3 技术报告或 quantile-balancing 页内引用），或显式标注"见 quantile-balancing 概念页来源" ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 5
- 处置：进入修复。核心机制（bias 只进选择、sign 规则式更新、批级 vs 序列内分工）与来源一致，无阻断性问题；两条重要问题均为表述层面的盲读卡点（未定义符号 $s'$、推理阶段 bias 处置歧义），修法明确且改动局部。
- 备注：本审查未读取 research/scope.md（任务约束禁止读 research/），学习目标闭环核对基于 index.html 内"读完你能回答"清单与正文 S1-S5 的对应；该清单五条均由正文完整回答。overview.html 是 index.html 的忠实摘要，无独立事实性偏差。
- 未执行项：可运行代码块审查——页面仅含伪代码（S3 Algorithm 1 伪代码、S4 手算表），无声称可运行的 Python/Shell 代码，故无可重跑项。validate.py 未执行（任务禁止修改文档且审查阶段不要求运行发布门控）。
