<!-- review-meta
round: 4
page: wiki/moe-serving/index.html
reviewed_content_sha256: b2141db567fcffb3
-->
# MoE 大模型推理与服务基础审查记录（第 4 轮）

- 页面版本：f2a581e2dd8998c0d4b11a030d5b6677d6159475
- 审查时间：2026-09-14 17:04
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 适用规范：guides/concept/check.md（页面 head `dojo:type=concept`）
- 已完整阅读章节（含全部折叠块、图注、代码块）：核心问题｜1. 一个字是怎么蹦出来的：token、参数与 Transformer 层｜2. 把一个 FFN 换成一排专家：MoE 与 top-k 稀疏激活｜3. 权重放不下了：专家并行与 all-to-all 搬运｜4. 通信不能干等：TBO 与 SBO 的重叠｜5. 一次请求的两个阶段：prefill、decode 与 KV cache｜6. 服务好不好怎么量：TTFT、TPOT、SLO 与 goodput｜7. 放一起还是分开：PD 合设与 PD 分离｜来源与范围说明

## 机械验证

- `.dojo/scripts/validate.py wiki/moe-serving/index.html` → `validation ok`。
- 第 3 章可运行代码实际执行（单文件、无第三方依赖、`random.seed(0)`）：固定路由表输出 `dispatch 计数：卡0=6 对，卡1=2 对`，`combine` 行同值；均匀随机 1000 token 输出 `卡0=1003 对，卡1=997 对`。逐行与页面「预期输出」一致。
- 手算复算，全部与页面相符：$4\times512^2=1{,}048{,}576\approx105$ 万；$2\times512\times2048=8\times512^2=2{,}097{,}152\approx210$ 万；$210/(105+210)=2/3$；$37/671\approx0.055$（5.5%）；$288/32=9$、$288/144=2$；$0.9/1.5=0.6$、$0.6/1.5=0.4$；$4+5+6=15$、$4+1+1=6$；$0.3+5\times0.05=0.55$；$0.5+4\times0.1=0.9$；6 对为 2 对的 3 倍；32:320=1:10。
- 引用编号：正文出现的 [C1]–[C14]、[F1]、[N1]–[N7] 与「来源与范围说明」双向对应，无孤立或未定义编号；同一编号在正文／答案折叠块／表格之间一致。
- 链接：`../gpu-execution-model/`、`../megamoe/`、`../mooncake/`、`../expertplex/` 四页均真实存在（`wiki/<name>/index.html`），页面无「（待生成）」占位；`overview.html` 与 `index.html` 相互链接。
- 无 `<img>` 正文图（仅 lightbox 占位 `alt=""`），不存在 alt 属性含 `$...$` 的问题；图注与流程图为 HTML 结构（`.flow`），非等宽字符框线图。

## 来源核对依据（原文片段／关键数值）

- N1（671B／37B）：DeepSeek-V3 arXiv:2412.19437 摘要「Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token」。
- N2（每 MoE 层 1 shared + 256 routed、top-8）：同报告「Each MoE layer consists of 1 shared expert and 256 routed experts」「8 experts will be activated for each token」；推理系统概览「only 8 out of 256 experts per layer are activated」。
- N3（61 层、前 3 层稠密）：报告「set the number of Transformer layers to 61」「We substitute all FFNs except for the first three layers with MoE layers」；官方 `config.json`：`num_hidden_layers=61`、`first_k_dense_replace=3`、`n_routed_experts=256`、`n_shared_experts=1`、`num_experts_per_tok=8`、`scoring_func=sigmoid`。
- C4/F1（sigmoid 亲和度 + 偏置选 top-k；Eq.14–16）：报告 §2.1.2「Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores」，Eq.15 $s_{i,t}=\mathrm{Sigmoid}(u_t^{\top}e_i)$，Eq.16 用 $s_{i,t}+b_i$ 选 Topk，「The bias term is only used for routing」。
- N4（在线部署）：《DeepSeek-V3/R1 推理系统概览》（2025-03-01）「Prefilling Phase [Routed Expert EP32 …]: Each deployment unit spans 4 nodes with 32 redundant routed experts, where each GPU handles 9 routed experts and 1 shared expert」「Decoding Phase [Routed Expert EP144 …]: Each deployment unit spans 18 nodes … each GPU manages 2 routed experts and 1 shared expert」「we employ a dual-batch overlap strategy … During the prefilling phase, these two microbatches executed alternately」。
- N5（技术报告部署单元）：报告 §3.4「minimum deployment unit of the prefilling stage consists of 4 nodes with 32 GPUs」「The minimum deployment unit of the decoding stage consists of 40 nodes with 320 GPUs」「we set 32 redundant experts for the prefilling stage」；ExpertPlex §2.4「combines 32 prefill and 320 decode GPUs to realize its target ratio」。
- N6（DistServe 收益）：arXiv:2401.09670 摘要「7.4x more requests」「12.6x tighter SLO」「while staying within latency constraints for > 90% of requests」。
- C8（TBO/SBO 定义）：ExpertPlex arXiv:2607.18002 v2 §2.3「TBO overlaps one microbatch's communication with another's computation」「SBO overlaps communication with shared-expert computation in the same microbatch」。
- C11/C12（TTFT/TPOT 总延迟、250 词/分、goodput）：DistServe §1「TTFT is the duration of the prefill phase」「TPOT represents the average time taken to generate a token for each request (except for the first token)」，脚注 1「The overall request latency equals TTFT plus TPOT times the number of generated tokens in the decoding phase」；「until it is faster than human reading speed (i.e., 250 words/min)」；goodput 定义为「maximum request rate that can be served adhering to the SLO attainment goal」。
- C9/C10（prefill/decode 定义、KV cache 复用）：同论文 §2.1「the prefill step deals with a new sequence … and processes these tokens concurrently」「each decoding step only processes one new token」；ExpertPlex §2.1「materializes key-value tensors as the KV cache, which later iterations reuse」。
- C13（ExpertPlex 混合架构转述）：ExpertPlex 摘要／正文「hybrid disaggregation-colocation architecture that shares experts across phases but disaggregates their attention modules」「giving each phase whole attention GPUs rather than intra-GPU partitions」——页面 §7 末段「跨阶段共享同一份 MoE 专家，attention 按阶段各自独占整卡」与之一致。
- Mooncake 转述（arXiv:2407.00079）：「Mooncake implements a global scheduler named Conductor」「Conductor is responsible for dispatching requests based on the current distribution of the KVCache and workloads」「reserved for the global KVCache pool」「groups the CPU, DRAM, SSD, and RDMA resources of the GPU cluster to implement a disaggregated KVCache」。
- 历史补充块：Shazeer 2017（arXiv:1701.06538）§2.1 小节名「Noisy Top-K Gating」、摘要「up to 137 billion parameters」；GShard（arXiv:2006.16668）§2.1「a variant of top-2 gating in both the encoder and the decoder」；Switch Transformer（arXiv:2101.03961）「we route to only a single expert」「This k=1 routing strategy is later referred to as a Switch layer」。

## 问题

- [轻微·技术] §2「本章问题」第 3 题解答末句：「两者并存于同一个 MoE 层，输出共同参与这一层的加权求和」措辞不准确。页内 F1 公式 $y(x)=\sum_{i\in S_k(x)} g_i(x)E_i(x)$ 只对 routed 专家加权，shared expert 的输出在来源中是**直接相加、不受门控**，该句使其看起来也参与加权。｜引文依据：DeepSeek-V3 arXiv:2412.19437 §2.1.2 Eq.12「$h_t' = u_t + \sum_{i=1}^{N_s}\mathrm{FFN}_i^{(s)}(u_t) + \sum_{i=1}^{N_r} g_{i,t}\mathrm{FFN}_i^{(r)}(u_t)$」——共享专家求和项无 $g$。｜修复要求：改写该句，明确区分「routed 专家输出按门控权重求和」与「shared expert 输出直接并入本层输出（DeepSeek-V3 中为直接相加）」，使其与 F1 公式口径一致。｜修复：｜复验：
- [轻微·技术] §1「FFN（前馈网络）……它由两个矩阵……构成」与「来源与范围说明·简化条件及其限制」：该定义只对原始 Transformer 成立；现代 LLM（含 DeepSeek-V3）使用门控 FFN、由三个矩阵构成。简化条件已逐条列出 attention 的省略项，却未提这一项，属未声明的理想化。｜引文依据：Vaswani 2017 标准 FFN 为 $\max(0,xW_1+b_1)W_2+b_2$（两矩阵）；DeepSeek-V3 官方 config.json `"hidden_act": "silu"`（门控 FFN，三矩阵）。｜修复要求：在 §1 该句加限定语（如「原始 Transformer 的 FFN 由两个矩阵构成」），或在「简化条件及其限制」中补一条说明门控 FFN 的三矩阵结构，并说明它不改变本节「FFN 是每层参数大户」的结论。｜修复：｜复验：
- [轻微·表述] §5 章末过渡与 §6 章首过渡：「两个阶段脾气完全不同」「两个指标由两个脾气不同的阶段贡献」把计算阶段拟人化，属口语化措辞；且该比喻未按 guides/concept/content-examples.md A10 标记为辅助解释并给出失效边界（它只描述两阶段的资源特征差异，不描述任何硬件行为）。｜引文依据：不适用｜修复要求：改为直述两阶段资源特征（prefill 计算密集、decode 访存密集），或保留比喻但在「辅助解释与类比边界」中标明其只指资源特征差异。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（三条轻微问题逐一处理后复验；无阻断、无重要问题，核心结论、来源一致性与引用编号体系均通过核对）