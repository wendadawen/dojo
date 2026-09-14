<!-- review-meta
round: 5
page: wiki/aux-loss-free-routing/index.html
reviewed_content_sha256: 261af177fb511109
-->
# 辅助损失无关路由审查记录（第 5 轮）

- 页面版本：5dc1df11489be8711cb73854047289073c1f2bf6
- 审查时间：2026-09-14 16:45
- 审查者：独立子代理（第 5 轮审查者，未参与写作与前序审查）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. MoE 训练为什么会负载坍塌——辅助损失方案又卡在哪 / 2. bias 加在路由分数上——只管选谁，不管用多少 / 3. bias 的固定步长 sign 更新——规则式，不进梯度 / 4. 手算一个 4 专家 top-2 的训练步——把路由与更新串起来 / 5. DeepSeek-V3 中的配置——调度与边界 / 来源与范围说明（含全部折叠块，以及 overview.html 全文）

## 来源核对（本轮已逐条回源，记录关键依据）

- arXiv:2408.15664 v1（HTML 全文，逐节核对）：§1 引言含 "routing collapse" 并引 "(Shazeer et al., 2017)"；§2.2 Eq.(2) `L_Balance = α Σ f_i P_i`，`f_i = (N/KT) Σ 1(token t selects Expert i)`，`P_i = (1/T) Σ s_{i,t}`，并引 "(Fedus et al., 2021)"（Switch）、"(Lepikhin et al., 2020)"（GShard）；§2.2 "non-negligible interference gradients … impair the model performance"；Eq.(1) `s_{i,t}=G(u_t^T e_i)`，G 为 "a nonlinear gating function"；§3 Eq.(3) 为带 bias 的 top-K 选择式，且明文 "is not added to the g_{i,t} that weights the output of the selected experts"，因果约束句亦在 §3（"utilizing the load information of the current sequence will break the causal constraint of language modeling"）；Algorithm 1 在 §3，输入 "bias update rate u"，更新 `b_i=b_i+u∗sign(e_i)`，`e_i=c̄_i−c_i`；§4.1 基线 "set the auxiliary loss coefficient α to 0.001"，"train the 1B model on 100B tokens and the 3B model on 200B tokens"，"we tune the bias update rate under only the 1B scale"，3B "directly inherit the best configuration for the 1B scale"；§4.2 Table 2：1B Loss-Controlled 9.56/0.72、Loss-Free 9.50/0.04；3B 7.97/0.52、7.92/0.04；§4.1 Eq.(4) MaxVio_global `(max_i Load_i − mean Load_i)/mean Load_i`；§4.3 "Update rate." 含 Figure 4（u=0.0001 收敛慢、u=0.01 振荡，选 u=0.001）；§4.3 Table 3：sign u=0.001 → 9.50/0.044；幅度 u=0.01 → 9.53/0.028、u=0.001 → 9.51/0.036、u=0.0001 → 9.51/0.040，理由 "it does not lead to better performance"；§4.3 Table 4 乘法 bias "slightly worse model performance"。
- DeepSeek-V3 Technical Report arXiv:2412.19437 v2（HTML 全文）：§2.1.2 Eq.12 输出式、Eq.13 归一化、Eq.15 `Sigmoid(u_t^T e_i)`、Eq.16 `s_{i,t}+b_i ∈ Topk(...)`；"Note that the bias term is only used for routing."；Eq.17–20 `ℒ_Bal = α Σ f_i P_i`、`f_i=(N_r/(K_r T))Σ 1(...)`、`s'_{i,t}=s_{i,t}/Σ_j s_{j,t}`、`P_i=(1/T)Σ s'_{i,t}`；§4.2 "decrease the bias term by γ if its corresponding expert is overloaded"、"increase it by γ if … underloaded"、"γ … called bias update speed"、"γ to 0.001 for the first 14.3T tokens, and to 0.0 for the remaining 500B tokens"、"we set α to 0.0001"；§4.2 模型超参 61 层、前三层稠密、1 shared + 256 routed、每 token 激活 8、671B/37B。（报告正文未出现 `Load_i` 记号，见问题 6。）
- Kimi K3 Technical Report arXiv:2607.24653 v2：§2.3.3 "Quantile Balancing" 载 Eq.13（top-k 与 mixture weight）、Eq.14（bias 取下一步 margin 的 `1−k/n` 分位数后减均值）；"adopts auxiliary-loss-free routing"、"it regulates dispatch without altering the mixture weights"；对固定步长："the fixed-step rule"、"γ trades off slow adaptation against load oscillation"、"Maintaining balanced loads becomes more challenging as LatentMoE increases the routed expert pool to 896 per layer"、"may leave some experts poorly trained"；896 routed / 16 active；HF `moonshotai/Kimi-K3/config.json`：`num_experts: 896`、`num_experts_per_token: 16`。
- 图内读数：本页无内联 SVG/图片，无数值图需像素测量。
- 可运行代码：本页唯一的"代码"是 Algorithm 1 伪代码（`language-text`，非可执行代码），无运行输出需核对；页内 JS 为站点模板脚本（进度条/目录/主题/复制/折叠），非页面主张。
- 机械项：`.dojo/scripts/validate.py wiki/aux-loss-free-routing/index.html` → `validation ok`；页面引用的 `../moe-serving/index.html`、`../quantile-balancing/index.html` 均真实存在，无"（待生成）"占位；index.html 与 overview.html 相互链接；无 `alt` 含 `$...$` 的图片；无交互视图。
- 手算例复算（第 4 章）：打分表 → 首轮 top-2 得 c=(3,1,0,4)、c̄=2、e=(−1,+1,+2,−2)、b=(−0.001,+0.001,+0.001,−0.001)；第二表逐格加 bias 后各 token top-2 不变；γ=0.05 对照表四行数值与 top-2 逐项复算无误；"相邻候选分数间距最小 0.05"（t4 的 E1 0.30 对 E0 0.25）成立。摘要/composition 数字（9.50/9.51、9.53、0.044/0.028/0.036/0.040、9.56/7.97/0.72/0.52/0.04、256/top-8/61/3/58、14.3T/500B/14.8T、α=0.0001、896/top-16）在正文、summary、overview 之间均一致。

## 问题

- [轻微·格式] 第 4 章折叠块 summary「展开：把 gamma 调大到 0.05，看一步能否改写选择」：summary 内以 ASCII `gamma` 代替 `$\gamma$`，违反 style-guide §5（summary/h2/h3 内数学符号须用 `$...$`）与 §11；同页其余位置一律写作 `$\gamma$`，写法不一致。｜引文依据：不适用｜修复要求：把该 summary 中的 `gamma` 改为 `$\gamma$`。｜修复：｜复验：
- [轻微·表述] 第 4 章首段「把前面两章串起来，端到端跑一遍。构造示例。 设 $N_r=4$ 个专家…」与「辅助解释与类比边界」首句「辅助解释。 可以把 bias 想成…」：两处"构造示例。""辅助解释。"独立成句、无谓语，读作残留标签。｜引文依据：不适用｜修复要求：删去这两个悬空短语，或改为句内成分（如"下面用一个人为构造的例子"／"一个可用的类比是"）。｜修复：｜复验：
- [轻微·表述] 章节过渡固定句式：第 1/2/3/4/5 章末各用同一句式"……——下一章看……"（共 5 处），另有"这是它的边界，下一章会展开""这正是下一章要讲的边界""下一章给出完整来源与范围说明"，合计 7 处同型句式；style-guide §8 要求过渡"不使用固定句式，也不为形式完整而添加过渡"。｜引文依据：不适用｜修复要求：将重复句式改为互不相同的表述，并删去纯为形式完整的过渡（尤指指向末章的过渡）。｜修复：｜复验：
- [轻微·表述] 口语化措辞与元话语：引言"DeepSeek 团队 2024 年 8 月在 arXiv:2408.15664 上提了个绕开损失函数的方案"；第 1 章"先回顾一下 MoE 是什么"；第 2 章"先列出本页要用的符号""这里要强调三件事"；第 4 章"端到端跑一遍"。｜引文依据：不适用｜修复要求：改为中性书面表述（如"DeepSeek 团队提出……方案""MoE 的基本结构如下""本节所用符号"／"三点需要说明"）。｜修复：｜复验：
- [轻微·来源] 第 1 章「当负载完全均衡时该损失取最小值」：arXiv:2408.15664 §2.2 只给出 `L_Balance = α Σ f_i P_i` 及 `f_i`、`P_i` 的定义与"α 控制辅助损失强度"，未出现任何"均衡即最小"的论断；按本页对 `P_i` 的定义（router 给它的平均分数，为 sigmoid 值、不归一化），该断言并不一般成立（均衡只约束 f_i，不约束 Σ P_i）。｜引文依据：报告 §2.2 原文无对应句；只有 "a large α can impair training, resulting in suboptimal performance" 与 Figure 2 的 MaxVio 扫描。｜修复要求：删去该句，或降级为有依据的表述（该损失把均衡目标写进梯度、向均衡方向施压）。｜修复：｜复验：
- [轻微·来源] 来源说明 F4 末句「DeepSeek-V3 报告里把负载写为 `$\mathrm{Load}_i$`」：本轮在 arXiv:2412.19437 v1、v2 正文与 ar5iv 全文中均未定位到 `Load_i` 或 `Load(Expert_i)` 记号，报告只用文字 "the expert load on the whole batch of each training step"，未定义该符号。｜引文依据：报告 §2.1.2 原文 "During training, we keep monitoring the expert load on the whole batch of each training step."（无 `Load_i`）。｜修复要求：删除该记号声明，或改为报告确有的表述（"expert load"）。｜修复：｜复验：
- [轻微·来源] 第 5 章「Kimi K3 把 routed expert 池扩大到 896（top-16）后，sign 丢失幅度信息的缺陷被放大：差 1 个 token 和差 100 个 token 走相同步长」：Kimi K3 报告 §2.3.3 把 896 规模下的困难归因于固定步长本身（"the fixed-step rule"、"γ trades off slow adaptation against load oscillation"、"Maintaining balanced loads becomes more challenging as LatentMoE increases the routed expert pool to 896 per layer"），未表述为"丢失幅度信息"；此因果系跨来源（2408.15664 的 sign/幅度对比 + K3 的 896 规模）综合，却以 `[C7]` 标注在 K3 名下。｜引文依据：K3 报告 §2.3.3 原文见上；K3 未出现 magnitude/幅度相关表述。｜修复要求：把该因果改为 K3 报告给出的理由（固定步长在近千专家下收敛过慢／振荡两难），或将"丢失幅度信息被放大"明确标注为跨来源推断。｜修复：｜复验：
- [轻微·来源] 引言 blockquote.meta「本页所用 $\gamma$ 取值与调度、$\alpha$ 取值、专家数与 top-k 均来自这两份原文」：其中 Kimi K3 的 896 routed expert / top-16 来自 Kimi K3 Technical Report 与 HF `config.json`（本页 `[C7]` 自述），手算例的 4 专家 top-2 为构造值，均非"这两份原文"。｜引文依据：本页 C7 条目 "Kimi K3 Technical Report §2.3.3（Eq.13–14）；HuggingFace `moonshotai/Kimi-K3/config.json`（`num_experts: 896`、`num_experts_per_token: 16`）"。｜修复要求：把主要依据补入 Kimi K3 Technical Report，或将该句限定为"DeepSeek-V3 的 γ 取值与调度、α 取值、专家数与 top-k"。｜修复：｜复验：
- [轻微·一致性] 同一示意数字在正文两处与另一处及 overview 不一致：第 3 章两处写"差 1 个 token 和差 50 个 token 走的步长完全一样"，第 5 章两处与 overview.html 写"差 1 个 token 和差 100 个 token 走相同步长"。｜引文依据：不适用｜修复要求：统一为同一个示意数字（如统一用 100 个 token）。｜修复：｜复验：

## 结论

- 处置：修复（均为轻微项，不改变结论与大纲）
- 统计：阻断 0 / 重要 0 / 轻微 9
