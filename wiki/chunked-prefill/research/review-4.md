<!-- review-meta
round: 4
page: wiki/chunked-prefill/index.html
reviewed_content_sha256: d66624b2331fe025
-->
# Chunked Prefill 审查记录（第 4 轮）

- 页面版本：690be9b7eba8bfc5d5aa98be2c991e59b9dc1de3
- 审查时间：2026-09-13 19:40
- 审查者：独立子代理（未参与写作，未读取 research/ 下规划、修复与前序审查记录）
- 规范：guides/concept/check.md（页面 head 声明 `dojo:type=concept`）
- 来源获取方式：页面 blockquote.meta 给出三篇论文。按 arXiv 编号抓取原文并抽取全文文本核对：SARATHI arXiv:2308.16369、Sarathi-Serve arXiv:2403.02310、Beyond the Buzz arXiv:2506.05508。核对到的原文位置：Sarathi 摘要、§1、§4.2；Sarathi-Serve 摘要、§2、§3.2、§4.2、§4.3、§5.4.1；Beyond the Buzz §3–§5 与 Figure 5/6 图注。本轮不读取 research/ 任何材料。
- 已完整阅读章节：核心问题（5 条问题及解答）、引言、1. 长 prefill 为什么卡住所有人（含本章问题）、2. 把长 prefill 切成块：机制与代价（2.1、2.2、本章问题）、3. decode 搭车与不停止的调度（3.1、3.2、本章问题）、4. 块要多大：token budget 的权衡（本章问题）、5. 与流水线并行合流：CPP（本章问题）、6. 边界与相邻工作（本章问题）、来源与范围说明；含全部 details 折叠块、两张图与图注。
- 机械校验：`python3 .dojo/scripts/validate.py wiki/chunked-prefill/index.html` 与 `.../overview.html` 均返回 validation ok；`index.html`↔`overview.html` 互链有效；前置链接 `../moe-serving/`、`../causal-mask/`、`../gpu-execution-model/`、`../model-parallelism/`、`../pp-load-balancing/`、`../beyond-buzz-disaggregation/` 均真实存在，无「（待生成）」占位，无 research/ 路径引用。已复算：$\frac{N(N-1)}{2}$ 与逐块计数 $N-i$ 一致（$N=16\to120$、$8\to28$、$4\to6$、$120/6=20$）；$\tau=512$ 时 $64+448=512$；$8000/448\approx18$ 次迭代。已回源核对通过的数字：512 token 饱和／~200×／chunk 257 vs 256 高 32%／LLaMA-13B/A6000 decode 至多 10×、端到端 1.33×／LLaMA-33B/A100 端到端 1.25×、decode 至多 4.25×／GPT-3 气泡 6.29×、吞吐 1.91×／Mistral-7B 2.6×、Yi-34B 3.7×、Falcon-180B 5.6×／Sarathi-Serve 首块 KV 载入 $N-1$ 次、Sarathi 原文为 $N$ 次／Beyond the Buzz 的 CPP、ISL 256K、64 GPU（EP×PP=64）、MLA down/up 投影重复计算。

## 问题

- [重要·技术] 核心问题第 1 条解答（`prefill 并行处理全部输入 token、是 compute-bound，一个 512 token 的请求即可饱和单卡算力`）｜引文依据：Sarathi §1「For example, on an A6000 GPU, for the LLaMA-13B model, a prefill with a sequence length of 512 tokens saturates GPU compute even at a batch size of just one.」；同页 [N2] 亦写「512 token 单请求饱和 A6000（LLaMA-13B）」｜问题：512 token 饱和是 LLaMA-13B 在 A6000 上的实测条件，解答中被写成「一个 512 token 的请求即可饱和单卡算力」，丢掉模型与 GPU 条件，成为无条件论断；正文（§1、§3.1、§4）与 [N2] 处均带条件，只有此处缺，属重要条件缺失｜修复要求：补回条件，改写为「LLaMA-13B 在 A6000 上，512 token 的请求即可饱和单卡算力」，与 [N2] 及正文一致｜修复：｜复验：
- [轻微·表述] §1 开篇两句（`先建立两类迭代的负载画像。`、`现在看 stall 怎么发生。`）｜引文依据：不适用｜问题：两句是以本页视角引导读者的元话语（「先建立…」「现在看…」），不是内容本身，与 check.md 第 2.2 节第 12 条列举的元话语同类；本章其余段落都可直接陈述｜修复要求：删去这两句引导语，直接给出 prefill/decode 的负载对比与 stall 的产生机制｜修复：｜复验：
- [轻微·技术] §2.1 图注（`左侧两个块各自独立过全部 $L$ 层`）｜引文依据：不适用（符号定义缺失，非来源问题）｜问题：$L$ 在本页首次且唯一一次出现，全文未定义其含义（层数），违反符号单义／公式符号需说明的要求｜修复要求：在图注或 §2.1 正文给出 $L$ 的定义（如「$L$ 为层数」），或改为不含符号的「全部层」｜修复：｜复验：
- [轻微·技术] §6 首句（`总计算量不变（FFN 不变）甚至微升（KV 重复读、固定开销）`）｜引文依据：Sarathi-Serve §4.2「This results in increased memory reads from the GPU HBM even though the computational cost is unchanged.」｜问题：与 §2.2「计算量本身不变（FFN 对每块独立计算、总量与不切时相同），变的是访存量」两处抵触——多出来的是访存量与 kernel 启动等固定开销，不是计算量，「计算量…甚至微升」的写法与 §2.2 冲突｜修复要求：改为「总计算量不变（FFN 不变），访存量与固定开销微升」｜修复：｜复验：
- [轻微·技术] §2 正文与「构造示例」注（`8000 token 的 prompt 按 512 一块切成 16 块`；`8000 token prompt / 512 每块 / 16 块`）｜引文依据：不适用（构造示例自洽性）｜问题：$512\times16=8192\neq8000$，$8000/512=15.625$，只有末块取 320 token 才能凑成 16 块，但正文与示例注均未说明末块不足 512；同页 §3.2 对同类除法写的是「约 18 次迭代」，此处缺「约」，三处数字不能同时成立｜修复要求：注明「末块 320 token」，或把示例改为「约 16 块」，使 512×块数与 8000 自洽｜修复：｜复验：
- [轻微·格式] 「论断与来源（C）」首条（`C5（含 N-1/N-2 计数与"compute-bound 可容忍"）、C9–C13、F1 依据、N3、N5：Sarathi-Serve（…）§2、§3、§4.2–4.3`）｜引文依据：Sarathi-Serve 摘要「For Mistral-7B on single A100 GPUs, we achieve 2.6x higher serving capacity and up to 3.7x higher serving capacity for the Yi-34B model on two A100 GPUs as compared to vLLM. … up to 5.6x gain … for Falcon-180B.」；本页 [N5] 条目自述为「Sarathi-Serve 摘要」｜问题：N5 被归入「§2、§3、§4.2–4.3」，与同章节 [N5] 自述的「摘要」来源不一致，按该位置无法在正文定位对照｜修复要求：把 N5 从该条移出，与 [C15] 一并标注为「Sarathi-Serve 摘要」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复
- 逐条来源核对（Sarathi 摘要/§1/§4.2、Sarathi-Serve 摘要/§2/§3.2/§4.2/§4.3/§5.4.1、Beyond the Buzz §3–§5 与 Figure 5/6 图注）未发现与来源冲突或来源不支持的论断：等计算量切块、块间只靠 KV cache 衔接（首块载入 $N-1$ 次）、FFN 计算量不变、decode 搭车最多低一个数量级、token budget 由一次性 profiling 定、CPP 在严格 FTL 下最优、MLA down/up 投影重复计算与缓存上投影 KV 缓解，均能定位到原文片段；构造示例、辅助解释与简化条件均已按规范标注。上述 1 条重要与 5 条轻微问题逐条修复并复验后即可发布。
