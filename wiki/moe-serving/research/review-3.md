<!-- review-meta
round: 3
page: wiki/moe-serving/index.html
reviewed_content_sha256: d82f9f3ebad1211e
-->
# MoE 大模型推理与服务基础审查记录（第 3 轮）

- 页面版本：fd63bb0703bbe6a0de96a4ce92244c4a30ccd566
- 审查时间：2026-09-13 19:44
- 审查者：独立子代理（编排者派发，未参与写作与前序轮次）
- 已完整阅读章节：引言（含「开始之前」「核心问题」）→ 1. 一个字是怎么蹦出来的：token、参数与 Transformer 层 → 2. 把一个 FFN 换成一排专家：MoE 与 top-k 稀疏激活 → 3. 权重放不下了：专家并行与 all-to-all 搬运 → 4. 通信不能干等：TBO 与 SBO 的重叠 → 5. 一次请求的两个阶段：prefill、decode 与 KV cache → 6. 服务好不好怎么量：TTFT、TPOT、SLO 与 goodput → 7. 放一起还是分开：PD 合设与 PD 分离 → 来源与范围说明（含全部折叠块与流程图注）

## 问题

- [轻微·表述] 引言第 3 段（"一句话说清本页内容：MoE…"）：以元话语宣告本页内容，且"本页"作了句子主语。全站其余 8 个概念页（glu、nope、expertplex、cross-entropy、newton-schulz、swiglu、situ-glu、opd）同一句式均写作"一句话说清："，仅本页多出"本页内容"，脱离既有约定并引入自我指代。｜引文依据：不适用（表述类）｜修复要求：删去"本页内容"，改为"一句话说清："或直接陈述主题。｜修复：｜复验：
- [轻微·表述] 第 6 章 goodput 段："goodput 问的是'在不让达成率破防的前提下，还能再加多少请求'。"中"破防"为网络口语，与页面技术语域不符。｜引文依据：不适用（表述类）；旁证：grep 全 wiki，"破防"仅本页 1 处命中。｜修复要求：改为"在不让 SLO 达成率跌破目标的前提下"一类正式表述。｜修复：｜复验：
- [轻微·表述] 第 2 章开头："计算量直接就是服务成本：参数翻倍，每个回答都贵一倍。"为未加限定的绝对论断。服务成本并非只由计算量决定——本页第 5 章自述 decode"大量时间花在读 KV cache 上"（访存受限），第 3 章亦指出显存是瓶颈，两处与该句直接等同的表述存在张力。｜引文依据：不适用（表述类）｜修复要求：加限定，如"在稠密模型这个设定下，每个 token 的计算量是服务成本的主要驱动"。｜修复：｜复验：
- [轻微·格式] `<head>` 内联 `<style>` 第 30–39 行：`.diagram` 类在整页 0 次使用（`class="diagram"` 计数 0），结构图全部由 `.flow` 实现（6 处），属死 CSS。｜引文依据：不适用（格式类）｜修复要求：删除未使用的 `.diagram` 规则。｜修复：｜复验：
- [轻微·格式] 第 3 章"真实规模示例"段：同一句内"256"出现两种写法——`$256$ 除以卡数`与"256 个路由专家"，全页其余 12 处"256"均为正文纯文本。｜引文依据：不适用（格式类）｜修复要求：统一为纯文本"256"，保持同一数字写法一致。｜修复：｜复验：

## 来源核对（片段）

均定位到标注位置，原文片段如下，支持页面表述：

- 数字核对（N1–N3）：DeepSeek-V3 Technical Report 摘要 "671B total parameters with 37B activated for each token"；§2.1.2 "the number of Transformer layers to 61"、"all FFNs except for the first three layers with MoE layers"、"1 shared expert and 256 routed experts"、"8 experts will be activated for each token"（sigmoid 亲和度 + 偏置）。页面"256 选 8 加 1 个共享专家 / 61 层中前 3 层稠密，其余 58 层为 MoE / 671B 总参数、37B 激活"一致；37÷671≈0.055 复算无误。
- 部署数字（N4）：《DeepSeek-V3/R1 推理系统概览》（2025-03-01）"Each deployment unit spans 4 nodes with 32 redundant routed experts, where each GPU handles 9 routed experts and 1 shared expert"（EP32）；"Each deployment unit spans 18 nodes with 32 redundant routed experts, where each GPU manages 2 routed experts and 1 shared expert"（EP144）。页面 4 节点 32 卡 / 9+1、18 节点 144 卡 / 2+1、288=256+32、288/32=9、288/144=2 一致。
- 部署数字（N5）：技术报告 "minimum deployment unit of the prefilling stage consists of 4 nodes with 32 GPUs"、"32-way Expert Parallelism (EP32)"、"minimum deployment unit of the decoding stage consists of 40 nodes with 320 GPUs"、EP320；ExpertPlex §2.4 "32 GPUs for prefill and 320 GPUs for decode"。页面"32 张 prefill GPU 加 320 张 decode GPU（decode 侧 40 节点 320 卡，EP320）"一致；N4/N5 口径区分成立。
- TBO 归因（C8/N4）：概览 "splitting a batch of requests into two microbatches. During the prefilling phase, these two…"（prefill 用双批次重叠；decode 改用 5 段流水线）。页面"DeepSeek 在线系统在 prefill 阶段用的正是这种'双批次重叠'"成立；页面未把 SBO 归给 DeepSeek，无误。
- TBO/SBO 定义（C8）：ExpertPlex arXiv:2607.18002 v2 §2.3 "Two-batch overlap (TBO), which overlaps one microbatch's communication with another's computation, or single-batch overlap (SBO), which overlaps communication with shared-expert computation"。页面定义一致。
- 指标定义与收益（C11/C12/N6）：DistServe arXiv:2401.09670 §1 摘要 "can serve 7.4x more requests or 12.6x tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for > 90% of requests"；§1 "TPOT only remains important until it is faster than human reading speed (i.e., 250 words/min)"；"The overall request latency equals TTFT plus TPOT times the number of generated tokens in the decoding phase"。页面 7.4×/12.6×/>90%、约 250 词/分钟、总延迟公式一致。
- 论文存在性：ExpertPlex arXiv:2607.18002 "ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels"，§2 结构为 2.1 MoE LLM Inference / 2.2 GPU Execution Model / 2.3 Large-scale MoE LLM Serving / 2.4 Prefill-Decode Disaggregation / 2.5 Prefill-Decode Colocation，与页面 [C] 标注的 §2.1/§2.3/§2.4/§2.5 对得上。
- Mooncake（第 7 章）：arXiv:2407.00079 摘要 "leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster to implement a disaggregated cache of KVCache"；正文 "Conductor is responsible for dispatching requests based on the current distribution of the KVCache and workloads"、"the selection of prefill instances considers… not just load but also the prefix cache hit length"。页面"把全集群闲置的 CPU/DRAM/SSD 组织成全局 KVCache 池，再由 Conductor 按缓存复用与负载同时最优来路由请求"成立（且页面已标注为转述）。
- 公式复算（F1/N7）：4×512²=1,048,576；2×512×2048=2,097,152；210/(105+210)=2/3；第 2 章归一化 0.9/1.5=0.6、0.6/1.5=0.4；第 3 章 dispatch 卡0=6/卡1=2；第 5 章 4+5+6=15 与 4+1+1=6；第 6 章 0.3+5×0.05=0.55——均与页面标注一致。
- 代码执行（E2）：以页面 `<pre>` 原文逐字运行 `python3`，输出与"预期输出"完全一致，含末行"均匀随机路由 1000 token：卡0=1003 对，卡1=997 对（期望各 1000）"。
- 机械项：`dojo:type=concept`；`dojo:topics=模型结构`（在 AGENTS.md 固定大类内）；`dojo:tag=MoE`（在 `ALLOWED_TAGS` 内）；description 为纯文本、dojo:summary 可渲染；C1–C14/F1/N1–N7 全部双向对应，无孤立标注；无 `research/` 残留路径、无"（待生成）"占位；`../gpu-execution-model/`、`../megamoe/`、`../expertplex/`、`../mooncake/` 均真实存在；overview.html 与 index.html 双向互链；`python3 .dojo/scripts/validate.py wiki/moe-serving/index.html` 通过。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：修复（5 项均属轻微，不含技术论断错误与来源不符；逐条修正后即可发布。已核查的事实、公式与数字全部回源无误，可作为发布依据）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
