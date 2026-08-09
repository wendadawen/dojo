# MoE 大模型推理与服务基础 独立审查

-审查者：独立上下文（AI 模拟目标读者，强推理模型）
- 页面版本：a2af556653379b4ca4c3ed3c4b4640f93a6737ad
- 时间：2026-08-07 14:05

## 问题

- [轻微·技术] 第6 章 / 来源节C11、C12：TTFT、TPOT、goodput、SLO 达成率、250 词/分钟等定义在正文与来源节均标注"来自 DistServe §2.1 及脚注 1"，但核对 DistServe（arXiv:2401.09670v3）发现这些定义首现于 §1 Introduction（TTFT=prefill 时长、TPOT=除首 token 外平均每token 时间、总延迟=TTFT+TPOT×token 数见脚注 1、goodput=满足 SLO 达成率的每GPU 最大请求率、250 words/min例），§2 仅重申 goodput；实质表述与来源完全一致，仅所引小节号偏差：将来源节 C11/C12 与文中括注的"§2.1 及脚注 1"改为"§1（含脚注 1）、§2 重申goodput"或"§1–§2.1" ｜ 修复：已改。来源节 C11 改为"DistServe §1 及脚注 1"、C12 改为"DistServe §1，§2.1 重申 goodput"；同时把第 6 章正文总延迟公式口径明确为"TPOT × decode 阶段生成的 token 数（即不含首 token）"，与脚注 1 原文一致 ｜ 复验：通过（对照DistServe §1 与脚注 1：TTFT/TPOT/goodput/250 词分钟均首现于§1、§2 仅重申 goodput，总延迟公式与脚注 1"TPOT times the number of generated tokens in the decoding phase"逐字一致）
- [轻微·技术] 第 6 章正文（TTFT 定义句）：页面写"TTFT……约等于 prefill 阶段的时长（含排队）"，DistServe 原文为"the duration of the prefill phase"，未含"排队"；"（含排队）"是超出所引来源字面的工程补充（真实 TTFT 确含排队，但来源未这样界定）。修法：将"（含排队）"标注为教学补充（如"约等于 prefill 时长；线上还含排队等待"），或去掉该括注使之严格贴合来源 ｜ 修复：已改。括注改为"（教学补充：线上服务里还包含排队等待）"，TTFT 定义本身严格贴合"the duration of the prefill phase"，排队部分显式标注为教学补充 ｜ 复验：通过（TTFT 定义句正文严格贴合来源"the duration of the prefill phase"，排队以"（教学补充：线上服务里还包含排队等待）"显式标注，不再冒充来源结论）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（两条轻微问题为来源小节号/字面精度，不影响核心正确性与主线理解；建议修复后发布，若接受则逐项写明接受理由）

## 复核记录（段 A 盲读 + 段 B 对照，供发布门控参考，非问题项）

- 段 A 盲读：按页顺序通读，主线连贯，每章"先给缺口问题→给机制→章末可检查小结→过渡指出下一步缺口"符合 A9；token/参数/前向/自回归、专家/router/top-k、EP/dispatch/combine/all-to-all、微批、prefill/decode/KV cache、TTFT/TPOT/SLO/goodput、PD 合设/分离等术语均在首现处就地解释（A1）；无核心定义被藏进折叠块（A8，四个折叠块均为历史/KV-cache 显存/可运行代码等补充，收起后正文完整）；《GPU 执行模型》一律以"规划中/负责"占位文字出现、非断链（符合 C3）；5 个学习目标逐题核对：G1→第1+2章、G2→第3+4章、G3→第5章、G4→第6章、G5→第7章，全部由正文章节完整回答。
- 段 B 数字/公式复算（python3 实核）：4d²=1,048,576≈105万、8d²=2,097,152≈210 万、FFN 占 2/3、37÷671=0.0551(≈5.5%)、归一化 g₃=0.6/g₇=0.4、KV cache 计数 15 vs 7、E4 指标 TTFT/TPOT/总延迟 0.3+5×0.05=0.55——全部与页面一致。
- 段 B 可运行代码：提取第 3 章 Python 代码块实际执行，逐行输出（含 dispatch 计数 卡0=6/卡1=2、均匀随机 1000 token 卡0=1003/卡1=997）与页面"预期输出"完全一致。
- 段 B 来源对照：N1（671B/37B）、N6（7.4× 请求 / 12.6× 更紧 SLO，>90% 达标）核对 DistServe 摘要一致；DeepSeek-V3 门控（sigmoid 亲和度+偏置选 top-k、门控值取自原始亲和度）核对技术报告 §2.1.2 一致；N4（prefill EP32/4节点32卡/每卡9路由+1共享；decode EP144/18节点144卡/每卡2路由+1共享；prefill 双批次重叠）核对《DeepSeek-V3/R1 推理系统概览》一致；N5（prefill 32卡/EP32、decode 40节点320卡/EP320）核对技术报告 §3.4.1/3.4.2 一致；C1/C4/C7/C8/C9/C13/C14 逐条核对 ExpertPlex background.tex（§2.1/2.3/2.4）与GShard/Shazeer 一致；历史脉络折叠块（Shazeer 2017 最多 1370 亿参数带噪声 top-k、GShard top-2、Switch top-1、DeepSeek-V3 top-8）核对各来源摘要一致。
- 段 B 页面功能：validate.py 对 index.html 与 overview.html 均退出码 0；本地 libs/katex.min.js 渲染 y(x)=Σ、4d²、g₃ 等 7 条公式全部成功；index↔overview 互链存在。

---

复验（2026-08-07 15:20，页面哈希 e17dba32e5345fc4e6e2d78607beeaf17078ddb0）：两项修复点全部通过。
