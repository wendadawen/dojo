<!-- review-meta
round: 4
page: wiki/aux-loss-free-routing/index.html
reviewed_content_sha256: 1b0c7e1eb658ed0c
-->
# 辅助损失无关路由审查记录（第 4 轮）

- 页面版本：576e11da29f7d9cd（wiki/aux-loss-free-routing/index.html，工作树）
- 审查时间：2026-09-13 21:08
- 审查者：编排者派发的独立审查者（未参与写作，未读取本页 research/）
- 已完整阅读：核心问题（5 问 5 答）、最容易误解、1. MoE 训练为什么会负载坍塌——辅助损失方案又卡在哪、2. bias 加在路由分数上——只管选谁，不管用多少、3. bias 的固定步长 sign 更新——规则式，不进梯度、4. 手算一个 4 专家 top-2 的训练步——把路由与更新串起来、5. DeepSeek-V3 中的配置——调度与边界、来源与范围说明（含全部 details 折叠块、表格与图注）

## 关键论断回源核对（本轮，含原文片段）

- Table 2（C2/N4）：原文 1B Loss-Controlled 9.56 / MaxVio 0.72、Loss-Free 9.50 / 0.04；3B 7.97 / 0.52、7.92 / 0.04 → 页面「辅助损失方案 PPL 与 MaxVio 均劣于本方法」成立。
- Table 3（C5/N5）：原文 `b_i=b_i+u*sign(e_i), u=0.001` 9.50 / 0.044；`u*e_i, u=0.01` 9.53 / 0.028；`u*e_i, u=0.001` 9.51 / 0.036；`u*e_i, u=0.0001` 9.51 / 0.040。与页面表格逐行一致。
- Algorithm 1（F4）：原文「Count the number of assigned tokens c_i for each expert, and the average number c̄_i; Calculate the load violation error e_i = c̄_i − c_i; Update b_i by b_i = b_i + u*sign(e_i)」→ 与 F4 `b_i←b_i+γ·sign(c̄_i−c_i)` 完全一致。
- 因果约束（C4）：原文「we update the biases based on the historical balance condition, since utilizing the load information of the current sequence will break the causal constraint」。
- 干扰梯度（C2）：原文「a large auxiliary loss will introduce non-negligible interference gradients into training」。
- routing collapse / 热点卡（C1）：原文 §1「may result in routing collapse (Shazeer et al. 2017)」；§2.2「when experts are distributed across multiple devices, load imbalance can exacerbate computation bottlenecks」。
- Table 4：原文「Addative Bias, u=0.001 9.50 0.044；Multiplicative Bias … 9.52 / 9.54」→ 页面「乘法 bias 性能略差」成立。
- DeepSeek-V3（F1–F6/N1–N3）：Eq.12 `h'_t`、Eq.13 `g_{i,t}=g'_{i,t}/Σg'_{j,t}`、Eq.15 `s_{i,t}=Sigmoid(u_t^T e_i)`、Eq.16 含 `b_i` 的 `g'_{i,t}`、Eq.17 `L_Bal`；§4.2「we set the bias update speed γ to 0.001 for the first 14.3T tokens, and to 0.0 for the remaining 500B tokens」；§2.1.2「The gating value … is still derived from the original affinity score s_{i,t}」；256 routed + 1 shared、top-8、前 3 层稠密。
- §4.1：原文「we train the 1B model on 100B tokens and the 3B model on 200B tokens」；「we tune the bias update rate under only the 1B scale. Experiments under the 3B scale directly inherit …」→ N4 与「3B 直接继承该值」成立。
- Kimi K3（C7）：HuggingFace `moonshotai/Kimi-K3/config.json` 的 `num_experts: 896`、`num_experts_per_token: 16`（`topk_method: noaux_tc`）；技术报告 arXiv:2607.24653 §2.3.3「Maintaining balanced loads becomes more challenging as LatentMoE increases the routed expert pool to 896 per layer」「may leave some experts poorly trained」，QB 分位数更新见 Eq.14。
- 手算例（§4 及折叠块）：复算 c=(3,1,0,4)、c̄=2、e=(−1,+1,+2,−2)、b=(−0.001,+0.001,+0.001,−0.001)；第二轮加 bias 后四个 token 的 top-2 全部不变；γ=0.05 折叠块四个 token 的 top-2 亦全不变。均与页面数字一致。
- 页面自校验：`.dojo/scripts/validate.py wiki/aux-loss-free-routing/index.html` 返回 `validation ok`；正文/表格无 `$...$` 外的 Unicode 数学字符（伪代码块内 →/Σ/∈ 按 style-guide §11 豁免）。概要页与页内互链、概念链接（moe-serving、quantile-balancing）均指向真实存在的页面。

## 问题

- [轻微·表述] 引言第 1 段「MoE 训练里路由器看到 4 个专家、每次让 token 自己选 top-2。训几万步后发现专家 0 被选中 90% 的次数、专家 3 一次都没被选中」：以指示语气叙述一个 4 专家 top-2 情形并给出无来源的具体数字（「90%」「几万步」），未标为设想/构造示例，且「路由器看到 4 个专家」与真实 MoE 规模（如 DeepSeek-V3 的 256 routed expert）不符｜引文依据：不适用（无对应来源）｜修复要求：把该句改写为明确的设想或构造示例（如「设想一个 4 专家 top-2 的迷你 MoE」），或删去「90%」「几万步」等无来源数字｜修复：｜复验：
- [轻微·表述] 引言第 1 段「但项的权重 $\alpha$ 是个玄学」：以口语化措辞（「玄学」）替代技术表述，全站其他页面未见该词｜引文依据：不适用｜修复要求：改为平实技术表述，如「项的权重 $\alpha$ 难以设定」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（本轮无阻断与重要问题；2 条轻微为开篇标记与用词，可在本轮一并修掉后复验）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
