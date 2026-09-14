<!-- review-meta
round: 4
page: wiki/deepseek-moe/index.html
reviewed_content_sha256: 9b5316af1aaae29c
-->
# DeepSeekMoE 审查记录（第 4 轮）

- 页面版本：b01c1c272aea60794b2f4510b29d6caebec2b90f
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何文件）
- 已完整阅读章节（按顺序）：开篇引言 +「核心问题」；1. 传统 MoE 的两个毛病——知识混合与知识冗余；2. 把专家切小——细粒度专家分割（含代码折叠块与预期输出）；3. 把通用知识拎出来——共享专家隔离（含「补充：路由专家仍需训练侧的负载均衡」折叠块）；4. 相同算力下真的更好——2B、16B 与 145B 的证据；5. 影响与继承——Stable LatentMoE 与 DeepSeek-V3；结尾「回到本页开头的问题」段；「来源与范围说明」全部小节（论断与来源 C / 公式与来源 F / 外部数字与实验条件 N / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）；另通读 overview.html 全文。

## 来源核对摘要（以下条目核对后与来源一致，不作为问题记录）

来源获取：arXiv:2401.06066 全文（ar5iv HTML 与 ACL 2024 官方 PDF 2024.acl-long.70 交叉核对）、arXiv:2412.19437（DeepSeek-V3）。

- C1/C2/C3：论文引言原文可定位——"Knowledge Hybridity: existing MoE practices often employ a limited number of experts"…"the designated expert will intend to assemble vastly different types of knowledge in its parameters, which are hard to be simultaneously utilized."；"tokens assigned to different experts may require common knowledge"…"multiple experts may converge in acquiring shared knowledge in their respective parameters"；"each expert acquires non-overlapping and focused knowledge"。
- C5/N8：§3.1 原文 "A typical top-2 routing strategy can yield (16 2)=120 possible combinations" 与 "the fine-grained routing strategy can yield (64 8)=4,426,165,368 potential combinations"。页面 120 → 4,426,165,368（约 44 亿）一致。
- C7：§3.1 "while maintaining a consistent number of expert parameters and computational cost"；§3.2 "In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by K_s"。两处原文均定位到。
- F1–F6：Eq.(3)–(11) 与页面公式逐条对照一致；Eq.(9) = K_s 项 + 路由项 + 残差，Eq.(10) 的 TopK 范围为 K_s+1..mN、选 mK−K_s，Eq.(11) s=Softmax(u^T e)，全部吻合。
- F7：mK·(1/m)=K、K_s·(1/m)+(mK−K_s)·(1/m)=K 可复算，且与 C7 的守恒声明一致。
- N1：Table 2 的 GShard×1.5 行给出 2.83B 专家参数 / 0.35B 激活专家参数、Relative Expert Size 1.5，两模型 Pile loss 均 1.808；§4.3 "1.5 times the expert size, which results in 1.5 times both expert parameters and expert computation"。页面一致。
- N2：§4.3 "we configure 16 shared experts where each expert has the same number of parameters as a standard FFN. This architecture mimics a dense model with 16 times standard FFN parameters, which sets the strict upper bound of MoE models"。页面 "FFN 中间维度放大 16 倍、对应 N=16 的稠密模型" 说法成立。
- N3：§4.2 "DeepSeekMoE has 1 shared expert and 63 routed experts, where each expert is 0.25 times the size of a standard FFN" + "2.0B total parameters and 0.3B activated parameters"；§4.1 9 层 / 隐藏维度 1280。页面一致。
- N4：Table 4 给出 LLaMA2 7B 187.9T、DeepSeekMoE 16B 74.4T FLOPs/4K tokens；caption "With only 39.6% of computations"。页面一致。
- N5：§5.1.2 "2 shared experts and 64 routed experts, where each expert is 0.25 times the size of a standard FFN. Each token will be routed to these 2 shared experts and 6 out of 64 routed experts"，28 层 / 隐藏维度 2048 / 16.4B 总参 / 2.8B 激活 / 2T tokens。页面一致（含官方 PDF 逐字核对）。
- N6：§5.2.1 "it enables single-device deployment on a GPU with 40GB of memory" 与 "nearly 2.5 times the inference speed of a 7B dense model"。页面一致。
- N7：摘要 "only 28.5% (maybe even 18.2%) of computations"。页面 "约 28.5%（可能低至 18.2%）" 一致。
- C8：DeepSeek-V3 §2.1.2 "Each MoE layer consists of 1 shared expert and 256 routed experts"、"8 experts will be activated for each token"、"DeepSeek-V3 uses the sigmoid function to compute the affinity scores"、"we introduce a bias term b_i for each expert and add it to the corresponding affinity scores"、"we pioneer an auxiliary-loss-free load balancing strategy"。页面一致。
- 出处：页面标注 ACL 2024 正确（aclanthology 2024.acl-long.70，pp.1280–1297）。
- 代码（§2 折叠块）：已实际执行，输出与页面「预期输出」逐行完全一致（组合数 4→28、120→4426165368，参数量/计算量守恒均 True；共享专家示例 路由 7 / 激活路由 1 / 总激活 2 / 参数 4.0 / 计算 1.0）。`validate.py` 返回 "validation ok"。页面无 Unicode 数学字符（公式定界符外），无 `$...$` 进入 alt（页面唯一 alt 为空串），无"（待生成）"占位，两处概念页链接（moe-serving、stable-latent-moe）真实存在。
- 表述维度：全文（含折叠块与图注）未发现元话语固定句式、第一人称复数/第二人称会话指代、调试叙事或 AI 拼接腔；`本页` 仅作范围自称（第 96、361、447 行），符合 style-guide §12「自称使用本页」，不计问题。

## 问题

- [轻微·表述] 第 209 行（第 4 章前的正文段）："论文的真实数字更夸张" 是对来源数字的临场评价，把作者的主观反应写进了正文｜引文依据：论文 §3.1 原文只有客观陈述 "A typical top-2 routing strategy can yield (16 2)=120 possible combinations … the fine-grained routing strategy can yield (64 8)=4,426,165,368 potential combinations"，无论述性措辞｜修复要求：改为中性的事实陈述（如"论文用的是更大的规模：$N=16,\ K=2,\ m=4$ 时，组合数从 $\binom{16}{2}=120$ 涨到 $\binom{64}{8}=4{,}426{,}165{,}368$"），删去"更夸张"｜修复：｜复验：
- [轻微·表述] 第 138、168、173、467 行：口语化措辞——"另一类干等着占地方"、"一个专家肚子里装了不止一种难以同时消化的知识"（第 138 行正文与第 168 行本章问题答案重复）、"但诊断只是开胃——下一章讲第一把刀如何砍根"（第 173 行）、"一个专家不再塞杂"（第 467 行），与页面其余部分的技术语体不一致｜引文依据：不适用（可读性/语体问题）｜修复要求：替换为技术表述（如"另一类在该 token 上不被使用却占用参数"、"单个专家被迫在同一份参数里容纳多种难以同时利用的知识"、"本章给出成因诊断，下一章说明第一个策略如何消除它"），保留原有信息量与结论｜修复：｜复验：
- [轻微·格式] 第 401 行第 4 章表格的 2B 行（同口径出现在第 126 行核心问题答案、第 420 行本章问题答案）：同一对比行内两列参数量口径不一致——"DeepSeekMoE 2B（2.0B 总参/0.3B 激活）"用模型级口径，"GShard 2.9B（专家参数 2.83B/激活 0.35B，1.5 倍专家参数与计算）"用专家级口径，读者易误读为同类量直接对比｜引文依据：论文 Table 2 两行均为专家级——GShard×1.5 "total expert params 2.83B / activated expert params 0.35B"，DeepSeekMoE "1.89B / 0.24B"；而 Table 1/§4.2 的 "2.0B total parameters and 0.3B activated parameters" 为模型级（含注意力参数）｜修复要求：把两列统一到同一口径（例如 DeepSeekMoE 2B 写"专家参数 1.89B/激活 0.24B"、GShard 2.9B 写"专家参数 2.83B/激活 0.35B"，或两列都写模型级 2.0B/0.3B vs 2.9B/…），三处（第 126、401、420 行）同步改｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（阻断与重要问题为 0，仅剩 3 条轻微表述/格式问题；修完即可发布）