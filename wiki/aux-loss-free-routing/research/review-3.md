<!-- review-meta
round: 3
page: wiki/aux-loss-free-routing/index.html
reviewed_content_sha256: 1b0c7e1eb658ed0c
-->
# 无辅助损失负载均衡审查记录（第 3 轮）

- 页面版本：`wiki/aux-loss-free-routing/index.html` 工作树 blob `e982f379d0941939eddfbbf1b1bb2f5f30d16a17`（sha256 `76d0c12ba83ef808399e1f6bf5c7090effabe11ac6651080f1d50e43cb47dd70`）；`overview.html` 工作树 blob `58a538bb14548f90c52639e3453ab453898e5a98`（sha256 `c812a4fd54efee5141dcad8b2baf1c2921c1b1c4f999a73600d9eb10907c07af`）
- 审查时间：2026-09-10 22:10 CST
- 审查者：独立审查者（第 3 轮，未参与写作，未读取 `wiki/aux-loss-free-routing/research/` 下任何文件）
- 已完整阅读章节：核心问题（5 条）／最容易误解；1. MoE 训练为什么会负载坍塌——辅助损失方案又卡在哪（含本章问题）；2. bias 加在路由分数上——只管选谁，不管用多少（含本章问题）；3. bias 的固定步长 sign 更新——规则式，不进梯度（含「补充：Table 3」折叠块、「代码：Algorithm 1」嵌套折叠块、本章问题）；4. 手算一个 4 专家 top-2 的训练步——把路由与更新串起来（含「展开：从负载统计到 bias 更新」「展开：把 gamma 调大到 0.05」两个折叠块、本章问题）；5. DeepSeek-V3 中的配置——调度与边界（含本章问题）；来源与范围说明（论断与来源 C／公式与来源 F／外部数字与实验条件 N／构造示例／辅助解释与类比边界／简化条件及其限制）；以及 `overview.html` 全文。折叠块内容全部展开阅读。

## 机械验证结果

命令与输出：

```text
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py \
      /Users/wendadawen/code/dojo/wiki/aux-loss-free-routing/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/aux-loss-free-routing/index.html
EXIT=0
```

逐项机械核对（`index.html` / `overview.html`）：

| 项目 | 结果 |
|---|---|
| validate.py 返回成功 | 通过（exit 0）。**注意**：该脚本只做占位符、重复 id、同页锚点、本地引用、元数据与 Unicode 数学字符检查，**不检查标签开合平衡**，故未发现下方「问题 1」 |
| `<head>` 五项元数据 | 全部存在且有效：`description` 为纯文本（不含 `$`）；`dojo:summary` 含成对 `$…$`；`dojo:type=concept`；`dojo:topics=模型结构`；`dojo:tag=MoE 负载均衡`；词表检查通过 |
| overview ↔ index 互链 | 双向成立。`index.html:658` `<a class="overview-link" href="overview.html">`；`overview.html:38` `<a href="index.html">深度教学 →</a>` |
| 前置概念链接有效 | `../moe-serving/index.html`（81 009 B）、`../quantile-balancing/index.html`（77 146 B）均存在；页面内 3 处引用与 `overview.html` 1 处引用全部指向这两页，无占位 |
| 引用双向闭合 | 上标引用集合 `{C1,C2,C3,C5,C6,F6,N1,N2,N3}` ⊆ 来源定义集合 `{C1–C6,F1–F6,N1–N5}`，无「用了未定义」；反向有缺口（C4、N4、N5 无正文引用），另 F1–F5 以公式内 `(Fx)` 标签而非 `<sup>` 引用 → 见问题 9、11 |
| 相邻双上标 | 无（`</sup>\s*<sup>` 零命中） |
| Unicode 数学字符 | 公式定界符外仅 `→`(20)、`↓`(2)、`↑`(1)；`↓`/`↑` 为表头与回到顶部按钮，`→` 属中文技术散文排版字符，均不在 `validate.py` 的 `BARE_MATH_CHARS` 词表内，`guides/concept/style-guide.md` §11 未列入 → 通过。唯一例外是表格中的 ASCII 下标 `MaxVio_global`，见问题 10 |
| TAB / CRLF / NBSP | 0 / 0 / 0 |
| 占位符与模板残留 | 无（`【】`、`TODO`、`TBD`、`@content`、`@component`、`（待生成）` 均零命中） |
| 可运行代码块 | 页面**未声明**任何可运行代码。`index.html:896-912` 的代码块是 `language-text` 伪代码，已在 892/894 行 summary 与正文中标注「伪代码」，按静态审查核对（输入／状态／核心步骤／输出四段齐全，变量 `c[i]`、`c_bar`、`e_i`、`gamma`、`b[i]` 与 F2/F4 同名对应，循环范围 `1..N_r` 明确） |
| 手算数值复算 | §4 两张表（首轮 4×4 打分表、$\gamma=0.001$ 后的 $s_{i,t}+b_i$ 表）与 $\gamma=0.05$ 对照表逐格复算全部一致：$c=(3,1,0,4)$、$\bar c_i=2$、$e=(-1,+1,+2,-2)$、新 bias $(-0.001,+0.001,+0.001,-0.001)$；两轮 top-2 选择均不变 |
| 公式渲染依赖 | 本地 KaTeX（`../../libs/katex.min.css`、`katex.min.js`、`auto-render.min.js` + `renderMathInElement`）齐备 |
| `overview.html` 元数据 | 无 `description`/`dojo:*`；与同级概念页 `moe-serving/overview.html`、`quantile-balancing/overview.html` 一致，`validate.py` 亦只对 `index.html` 要求，按现状接受 |

**审查前提校正（影响本轮核对范围）**：本轮派发说明称页面 `blockquote.meta` 标注的主要依据是「arXiv:2412.19437 §5.2.3 与 arXiv:2405.04434（DeepSeek-V2）」且「页内还引用了本机官方实现」。实际页面不是这样：

- `index.html:681` 的 `blockquote.meta` 写的是「Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts（Wang et al., 2024, arXiv:2408.15664）；DeepSeek-V3 Technical Report（arXiv:2412.19437 §2.1.2, Eq.12-17）」。
- 页面全文**未出现** arXiv:2405.04434（DeepSeek-V2），也**未引用** `wiki/deepseek-v4-1/research/official/inference/model.py` 或任何本机源码路径。
- 派发说明中的两个定位在原文均不存在：DeepSeek-V3 顶层章节为 1 Introduction / 2 Architecture / 3 Infrastructures / 4 Pre-Training / 5 Post-Training / 6 Conclusion，**无 §5.2.3**（§5.2 只有 5.2.1 Reward Model、5.2.2 GRPO）；arXiv:2405.04434 是 DeepSeek-V2（MLA + DeepSeekMoE），摘要不含 aux-loss-free 负载均衡。

因此本轮按**页面实际标注**的 arXiv:2408.15664 与 arXiv:2412.19437 逐条核对；本机 `model.py` 只作为 `overview.html` 一条与之冲突的论断的对照材料使用（见问题 4）。

## 问题

- [阻断·格式] `index.html:891-915`（第 3 章「代码：每步训练结束后的 bias 更新」折叠块）：`<details>` 开标签出现 24 次、闭标签 23 次，891 行的外层 `<details>` 与 893 行的 `<details class="code-details">` 只对应 915 行一个 `</details>`，必然有一个 `<details>` 未闭合。浏览器会把其后所有兄弟节点收进这个仍打开的 `<details>`（无 `open`，默认收起），于是 914 行说明段、第 3 章「本章问题」、整个第 4 章、第 5 章与「来源与范围说明」默认全部不可见；同时页脚脚本的 `document.querySelectorAll('body > h2, body > h3')`（目录生成 1198 行、滚动高亮 1227 行、折叠按钮 1308 行、j/k 跳章 1326 行）只匹配 `body` 直接子节点，第 4/5 章与来源章节的 h2 变成 `body > details > h2` 后全部从目录和章节导航中丢失。｜引文依据：不适用（结构计数：`details` open=24 / close=23；其余 `section`、`div`、`pre`、`table`、`ol`、`ul`、`blockquote`、`li`、`summary`、`p` 均开合相等）｜修复要求：使 `<details>` 与 `</details>` 数量相等——删除 891-892 行冗余的外层 `<details>`/`<summary>代码：每步训练结束后的 bias 更新</summary>`（保留 893 行的 `details.code-details` 与 914 行说明段在正文），或补上第二个 `</details>`；修复后须满足：开合计数相等、第 3 章「本章问题」及第 4/5 章标题能被子节点选择器命中。｜修复：删除 `index.html:891-892` 冗余外层 `<details>`/`<summary>代码：每步训练结束后的 bias 更新</summary>`，保留 893 行 `details.code-details`（其 915 行 `</details>` 改闭合该内层块）与 914 行说明段在正文。｜复验：配对脚本确认 `<details>`=23 == `</details>`=23、嵌套深度全程回到 0、无未闭合 OPEN。headless Chrome（`--headless=new --disable-gpu --virtual-time-budget=20000 --dump-dom`）解析后 DOM 内 `<details>`/`</details>` 各 23；`h2#s4-hand-example`、`h2#s5-deepseek-v3-config`、`h2#sources-and-teaching-notes` 的 parent 均为 `body`、`open_details_depth=0`（不在任何折叠块内），可被 `body > h2` 选择器命中，目录与 j/k 章节导航恢复；第 4/5 章与来源章节正文可见。

- [重要·技术] `index.html:868`、`937`、`1113`、`1123`（四处）：引用 arXiv:2408.15664 §3.2 / §3.1，该论文**没有编号子节**（第 3 节下只有非编号段落 "Comparison with Other Load Balancing Methods."）。｜引文依据：论文标题层级为 `## 1 Introduction / ## 2 Background（### 2.1 Mixture-of-Experts、### 2.2 Auxiliary Loss for Load Balance）/ ## 3 Auxiliary-Loss-Free Load Balancing Strategy / ## 4 Experiments（### 4.1 / ### 4.2 / ### 4.3）/ ## 5 Discussion（### 5.1 Loss-Free Balancing Is Compatible with Expert Parallelism、### 5.2 Load Balancing and Future Token Leakage）/ ## 6 Conclusion`；被引内容确实在 §3 正文："It is worth noting that we update the biases based on the historical balance condition, since utilizing the load information of the current sequence will break the causal constraint of language modeling, leading to leakage of the information of future tokens." ｜修复要求：868 行与 937 行的「§3.2」、1113 行 C4 的「§3.2」、1123 行 F4 的「§3.2」全部改为「§3」；1112 行 C3 的「§3.1」改为「§3」。｜修复：`index.html:868`、`937` 的「arXiv:2408.15664 §3.2」与 `1113` C4、`1123` F4 的「§3.2」、`1112` C3 的「§3.1」全部改为「§3」。｜复验：全文 `§3.2`、`§3.1` 计数均为 0；WebFetch ar5iv `2408.15664` 确认第 3 节下只有非编号子标题 "Comparison with Other Load Balancing Methods."，被引因果约束句位于 §3 正文。

- [重要·技术] `index.html:1112`（C3 来源条）、`1121`（F2 来源条）：把含 bias 的选择规则 $g'_{i,t}$ 的来源记为「原始论文 Eq.1-2」，但该式在原文是 **Eq.(3)**；Eq.(1) 是不含 $b_i$ 的 gating，Eq.(2) 是辅助损失。｜引文依据：原文 §3 "(3) $g_{i,t}=\{s_{i,t},\ s_{i,t}+b_i\in\mathrm{Topk}(\{s_{j,t}+b_j\}),\ 0\ \text{otherwise}\}$"；§2.1 Eq.(1) 为 $h_t=u_t+\sum_i g_{i,t}\mathrm{FFN}_i(u_t)$ 且 $s_{i,t}=G(u_t^T e_i)$（无 $b_i$）；§2.2 Eq.(2) 为 $\mathcal{L}_{Balance}=\alpha\sum_{i=1}^N f_iP_i$。｜修复要求：F2 与 C3 的「原始论文 Eq.1-2」改为「原始论文 §3 Eq.3」；C3 中「bias 只做选择不进权重」的原文支撑改为 V3 §2.1.2 原句 "Note that the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score $s_{i,t}$."｜修复：F2（`1121`）与 C3（`1112`）的「原始论文 Eq.1-2」改为「原始论文 §3 Eq.3」；C3 的支撑句改为 DeepSeek-V3 §2.1.2 原句「Note that the bias term is only used for routing. The gating value, … is still derived from the original affinity score $s_{i,t}$.」。｜复验：全文 `Eq.1-2` 计数 0；WebFetch 确认原文含 $b_i$ 的 $g_{i,t}$ 选择式为 Eq.(3)，Eq.(1) 无 bias、Eq.(2) 为辅助损失。

- [重要·技术] `index.html:1076`（第 5 章边界条目）、`1152`（简化条件条目）：Kimi K3「routed expert 池扩大到 896（top-16）」「改用 Quantile Balancing 替代 sign」「仍属 aux-loss-free 家族」是带具体数字的外部事实与机制论断，但正文与「来源与范围说明」均无对应 C/N 条目。｜引文依据：本页来源章节 C1–C6、F1–F6、N1–N5 中无 K3 条目；相邻概念页 `wiki/quantile-balancing/index.html:1337` 把同一事实记为「HuggingFace `moonshotai/Kimi-K3/config.json`，`num_experts: 896`, `num_experts_per_token: 16`」，`:1343` 记为「K3 报告 §2.3.3 Eq.13」。｜修复要求：在 1076 行该条目补 `<sup>[C7]</sup>` 或 `<sup>[N6]</sup>`，并在「来源与范围说明」新增对应条目，写明 K3 报告章节或 `config.json` 字段；若本轮无法定位来源，则删去 896／top-16／Quantile Balancing 的具体数字与结论，改为仅指向 `../quantile-balancing/index.html` 的一般表述。｜修复：`1076` 行 K3 条目补 `<sup>[C7]</sup>`（`1099`、`1152`、`1155` 一并补），来源章节新增 C7 条目，写明 Kimi K3 Technical Report §2.3.3（Eq.13–14）与 HuggingFace `moonshotai/Kimi-K3/config.json`（`num_experts: 896`、`num_experts_per_token: 16`）。｜复验：引用集合与定义集合双向闭合（C7 既用也定义）；C7 与相邻页 `wiki/quantile-balancing/index.html:1337`、`:1353` 的 K3 来源条目一致。

- [重要·技术] `overview.html:69`（「关键结论与边界」第 1 条）：「推理阶段 bias 被吸收进 router 权重，前向无需显式加 $b_i$」既无来源，也与本机官方实现相反。｜引文依据：DeepSeek-V3 报告只在 §2.1.2 说明 "the bias term is only used for routing"，全文未涉及推理期对 bias 的处理；`wiki/deepseek-v4-1/research/official/inference/model.py:806` `self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))`，`:818-822` `bias = self.bias … indices = (scores + bias).topk(self.topk, dim=-1)[1]`，`:823` `weights = scores.gather(1, indices)`——前向显式做 `scores + bias`，bias 单独保存、未被折进 `weight`；官方 inference 目录内无 fold/absorb 逻辑。｜修复要求：删除「推理阶段 bias 被吸收进 router 权重，前向无需显式加 $b_i$」这一分句；如要保留推理期描述，改写为可核对的形式并在本页给出出处（例如「推理时 bias 仍作为独立张量保存，前向按 `(scores + bias)` 参与 top-k，门控权重仍取原始分数」）。｜修复：`overview.html:69` 删除「推理阶段 bias 被吸收进 router 权重，前向无需显式加 $b_i$」，改为与源码一致的「bias 只用于路由选择：它不作为权重被折进 router，推理时仍作为独立张量保存，前向按 $s_{i,t}+b_i$ 参与 top-k，门控权重仍取原始分数 $s_{i,t}$（DeepSeek-V3 报告 §2.1.2）」。｜复验：本机 `wiki/deepseek-v4-1/research/official/inference/model.py:806` bias 为独立 `nn.Parameter`、`:822` `indices = (scores + bias).topk(...)`、`:823` `weights = scores.gather(1, indices)` 印证；全文「被吸收进 router」计数 0。

- [重要·技术] `index.html:1062`、`1085`（第 5 章正文与本章问题解答）：「这一调度反映一个事实——训练后期路由已基本稳定，继续更新 bias 反而会干扰最后收敛」写成来源事实，且该句带 `<sup>[C5, N2]</sup>` 标记。｜引文依据：DeepSeek-V3 §4.2 只有取值陈述 "For auxiliary-loss-free load balancing, we set the bias update speed γ to 0.001 for the first 14.3T tokens, and to 0.0 for the remaining 500B tokens."，同节及 §2.1.2 均未给理由；arXiv:2408.15664 全文没有「训练后期冻结 bias」这一做法，其 Algorithm 1 每一步都更新 bias。｜修复要求：把 1062、1085 行的理由句降级为明确标注的推断（如「报告只给出取值，未说明理由；一种可能是训练后期路由已趋稳、继续更新会扰动最后收敛」），或删去理由句只保留 $\gamma$ 取值的 $14.3\mathrm{T}/500\mathrm{B}$ 事实；同时去掉该理由句上的 `[C5, N2]` 指向。｜修复：`1062` 行改为只陈述 DeepSeek-V3 §4.2 的取值，并把「训练后期路由已基本稳定……干扰最后收敛」标注为「本页推断，非报告结论」；去掉理由句上的 `[C5, N2]`、保留 `[N2]`；`1085` 行解答同步降级为推断，`1084` 行 summary 改为「报告未给理由；冻结利于末段收敛为本页推断」。｜复验：DeepSeek-V3 §4.2 只有取值句、无理由，arXiv:2408.15664 全文无冻结做法；正文不再有以来源事实口吻写出的理由句。

- [轻微·技术] `index.html:1062`（第 5 章）：称 $\gamma=0.001$「在原始 1B / 3B 实验上就被选为最佳」，条件不准确。｜引文依据：arXiv:2408.15664 §4.1 "Our experiments are based on two model sizes of 1B and 3B total parameters, and we tune the bias update rate under only the 1B scale. Experiments under the 3B scale directly inherit the best configuration for the 1B scale."；§4.3 "An appropriate choice is u=0.001, which shows good training balance and validation perplexity."｜修复要求：改为「在原始 1B 实验上调参得到最优 $u=0.001$（§4.3 Figure 4），3B 实验直接继承该值（§4.1）」。｜修复：`1062` 行「在原始 1B / 3B 实验上就被选为最佳」改为「在原始 1B 实验上调参得到最佳 $u=0.001$（arXiv:2408.15664 §4.3 Figure 4），3B 实验直接继承该值（§4.1）」。｜复验：WebFetch 原文 §4.1「we tune the bias update rate under only the 1B scale… 3B scale directly inherit」、§4.3「An appropriate choice is $u=0.001$」。

- [轻微·技术] `index.html:886`（Table 3 折叠块末段）：「作者认为幅度版本需要再调一个 $\gamma$、收益不明显，故采用更简单的 sign」中「需要再调一个 γ」是页面推断。｜引文依据：arXiv:2408.15664 §4.3 原文 "Although this variant slightly improves load balance, it does not lead to better performance, as shown in Table 3. Therefore, we maintain the sign version."（只以「未带来更好性能」为由）｜修复要求：改为「论文以『未带来更好性能』为由维持 sign 版本；幅度版本另需扫描 $\gamma$（Table 3 试了 0.01／0.001／0.0001 三个值）为本页观察，非论文结论」。｜修复：`886` 行改为「论文给出的理由是「未带来更好性能」（原文：does not lead to better performance）；幅度版本另需扫描 $\gamma$（Table 3 试了 0.01／0.001／0.0001 三个值）为本页观察，非论文结论」；第 3 章解答（`930`）同类推断一并改为「论文以「未带来更好性能」为由维持 sign」。｜复验：WebFetch 原文 §4.3「Although this variant slightly improves load balance, it does not lead to better performance… we maintain the sign version.」

- [轻微·格式] `index.html:1113`（C4）、`1133`（N4）、`1134`（N5）：三条来源条目在正文中没有任何上标引用，双向对应不完整。｜引文依据：不适用（页面 `<sup>` 引用集合为 `{C1,C2,C3,C5,C6,F6,N1,N2,N3}`，来源章节定义集合为 `{C1–C6,F1–F6,N1–N5}`）｜修复要求：在 868 行（规则式更新／因果约束）补 `<sup>[C4]</sup>`，在 946 行（1B/3B、100B/200B token）补 `<sup>[N4]</sup>`，在 870 行或 879-882 行表格（Table 3 数值）补 `<sup>[N5]</sup>`；或删除来源章节中未被引用的条目。｜修复：`868` 行补 `<sup>[C4]</sup>`（规则式更新／因果约束）；`870` 行补 `<sup>[N5]</sup>`（Table 3 数值，合并为 `[C5, N5]`）；`1062` 行补 `<sup>[N4]</sup>`（1B/3B 条件）。报告建议的「946 行」实为第 4 章构造示例引入段、不含 1B/3B，故 N4 改挂到唯一出现该条件处。｜复验：正文引用集合 `{C1–C7, F1–F6, N1–N5}` 与来源定义集合完全相等，`used-not-defined`／`defined-not-used` 均为空。

- [轻微·格式] `index.html:877`（Table 3 表头）：`MaxVio_global` 以 ASCII 下划线写下标，未包在 `$…$` 中；同页正文 754、870、886 行与 N5 条目都写作纯文字 `MaxVio`，同页写法不一致。｜引文依据：不适用｜修复要求：表头改为 `$\text{MaxVio}_{\text{global}}$`（或与正文一致统一为 `MaxVio`）。｜修复：`877` 行表头 `MaxVio_global` 改为 `$\text{MaxVio}_{\text{global}}$`。｜复验：全文 ASCII `MaxVio_global` 计数 0，与正文 `MaxVio` 写法统一，validate.py 通过。

- [轻微·格式] `index.html:780`、`784`、`788`、`799`、`856`、`1066`：六处公式在 `$$…$$` 内用 `\qquad\text{(F1)}…\text{(F6)}` 自行编号，与 `guides/concept/style-guide.md` §11「公式不自行编号，引用论文 Eq. 编号」不一致；且 F6 在 1064 行又是以上标 `[…, F6]` 引用的，同一类来源键出现两种引用形式。｜引文依据：不适用｜修复要求：删除公式内的 `\qquad\text{(Fn)}`，改在公式后按 §6 用 `<sup>[F1]</sup>`…`<sup>[F6]</sup>` 引用，保持与来源章节双向对应。｜修复：删除 `780`、`784`、`788`、`799`、`856`、`1066` 六处公式内 `\qquad\text{(Fn)}` 自行编号；F1–F5 在紧邻正文改用 `<sup>[F1]</sup>`…`<sup>[F5]</sup>` 引用（F2、F3 分别挂 `782`／`786` 行），F6 沿用 `1064` 行既有 `<sup>[C6, N3, F6]</sup>`。｜复验：`\qquad\text{(F\d)}` 残留 0；引用与定义双向闭合含 F1–F6，公式不再自行编号。

- [轻微·技术] `index.html:681`（`blockquote.meta`）与 `1111`（C2）：定位范围不完整。meta 把主要依据限定为「arXiv:2412.19437 §2.1.2, Eq.12-17」，但正文使用的 $\gamma$ 调度（前 14.3T／后 500B）、$\alpha=0.0001$ 与规模数字均出自 V3 §4.2；C2 把 baseline $\alpha=0.001$ 记为「§1、Table 2」，而原文 Table 2 只列 "Loss-Controlled 9.56/0.72、Loss-Free 9.50/0.04（1B）" 与 "7.97/0.52、7.92/0.04（3B）"，不含 $\alpha$，$\alpha=0.001$ 出自 §4.1 Baseline。｜引文依据：arXiv:2408.15664 §4.1 "For the baseline, we set the auxiliary loss coefficient α to 0.001 to achieve a reasonable trade-off between model performance and load balance"；DeepSeek-V3 §4.2 Training Hyper-Parameters｜修复要求：meta 改为「arXiv:2412.19437 §2.1.2 与 §4.2」；C2 条目改为「§1、§4.1 Baseline、§4.2 Table 2」。｜修复：`681` 行 meta「arXiv:2412.19437 §2.1.2, Eq.12-17」改为「arXiv:2412.19437 §2.1.2、§4.2」；`1111` 行 C2 改为「arXiv:2408.15664 §1、§4.1 Baseline（$\alpha=0.001$）、§4.2 Table 2（对照数值）」。｜复验：WebFetch 确认 DeepSeek-V3 §4.2 含 $\gamma$ 调度与 $\alpha=0.0001$、原文 §4.1 含 baseline $\alpha=0.001$、Table 2 只列对照数值。

## 结论

- 统计：阻断 1 / 重要 5 / 轻微 6
- 处置：修复

**本轮已核对通过的来源论断**（按 check.md §2.2 四步核对，引文依据如下，供复验复核）：

- C1（路由坍塌，§1 引 Shazeer et al. 2017）——§1 原文 "load imbalance, which may result in routing collapse (Shazeer et al., 2017)"。**通过**。
- C2（$\alpha$ 两难）——§1 "a large auxiliary loss will introduce non-negligible interference gradients into training and thus impair the model performance"、§2.2 "The Dilemma Between Load Balance and Model Performance"、Table 2（1B：9.56/0.72 → 9.50/0.04；3B：7.97/0.52 → 7.92/0.04）。**数值通过**，定位需补 §4.1（问题 12）。
- C3（bias 只做选择不进权重）——V3 §2.1.2 Eq.16 与 "the bias term is only used for routing"；原文 §3 Eq.3。**内容通过**，原文定位需改（问题 3）。
- C5（sign 与大幅版本对比、采用 sign 的理由）——§4.3 Table 3 四行（sign $u=0.001$ 9.50/0.044；$e_i$ $u=0.01$ 9.53/0.028；$u=0.001$ 9.51/0.036；$u=0.0001$ 9.51/0.040），与页面 879-882 行逐格一致；Table 4 乘法 bias（9.52/0.041、9.52/0.036、9.54/0.048）与页面「性能略差」一致。**通过**（表述细节见问题 8）。
- C6（互补 sequence-wise loss）——V3 §2.1.2 Eq.17 $\mathcal{L}_{\mathrm{Bal}}=\alpha\sum_{i=1}^{N_r}f_iP_i$ 及 "to prevent extreme imbalance within any single sequence"。**通过**。
- F1 $s_{i,t}=\sigma(u_t^\top e_i)$——V3 §2.1.2 Eq.15 $\operatorname{Sigmoid}(u_t^{T}e_i)$；原文 Eq.1 $s_{i,t}=G(u_t^Te_i)$、§4.1 "we choose sigmoid instead of softmax as the gating function $G$"。**通过**。
- F2 $g'_{i,t}$ 选择规则——V3 §2.1.2 Eq.16 与页面逐字一致（含 $b_i$ 只在 Topk 条件内）。**通过**（定位见问题 3）。
- F3 $g_{i,t}=g'_{i,t}/\sum_j g'_{j,t}$——V3 §2.1 Eq.13 一致。**通过**。
- F4 $b_i\leftarrow b_i+\gamma\,\mathrm{sign}(\bar c_i-c_i)$——原文 Algorithm 1 第 3–4 步 "Count the number of assigned tokens $c_i$ … and the average $\overline{c_i}$; Calculate the load violation error $e_i=\overline{c_i}-c_i$; Update $\mathbf{b}_i$ by $b_i=b_i+u*\mathrm{sign}(e_i)$"，页面符号映射（$\gamma\leftrightarrow u$）已在 F4 条目说明。**通过**（定位见问题 2）。
- F5 $h_t'$ MoE 层输出——V3 §2.1 Eq.12 一致（$\mathbf{u}_t+\sum_i^{N_s}\mathrm{FFN}_i^{(s)}+\sum_i^{N_r}g_{i,t}\mathrm{FFN}_i^{(r)}$）。**通过**。
- F6 $L_{Bal}$——V3 §2.1.2 Eq.17；页内 $f_i$、$P_i$、$s'_{i,t}$ 的定义分别对应原文 Eq.18/20/19。**通过**（$f_i$、$P_i$ 严格属 Eq.18–20，F6 条目只标 Eq.17，可接受）。
- N1（671B/37B、256+1、top-8、61 层、前 3 层稠密）——V3 §4.2 "Each MoE layer consists of 1 shared expert and 256 routed experts … 8 experts will be activated for each token … We set the number of Transformer layers to 61 … We substitute all FFNs except for the first three layers with MoE layers … 671B total parameters, of which 37B are activated for each token"。**通过**。
- N2（$\gamma$ 调度：前 14.3T 用 0.001，后 500B 用 0，总 14.8T）——V3 §4.2 同一句；14.8T 见摘要。**通过**。
- N3（$\alpha=0.0001$）——V3 §4.2 "For the balance loss, we set α to 0.0001, just to avoid extreme imbalance within any single sequence."。**通过**。
- N4（1B/3B、100B/200B token、$u=0.001$ 最佳）——§4.1 "we train the 1B model on 100B tokens"、"the 3B model on 200B tokens"；§4.3 "An appropriate choice is u=0.001"（Figure 4）。**通过**（1B/3B 条件表述见问题 7）。
- N5（sign 9.50/0.044；幅度 9.51/0.036）——§4.3 Table 3 一致。**通过**。
- §4 手算例（构造示例）——4×4 打分、$c=(3,1,0,4)$、$\bar c_i=2$、$e=(-1,+1,+2,-2)$、$b=(-0.001,+0.001,+0.001,-0.001)$、两轮 top-2 均不变，以及 $\gamma=0.05$ 对照表，全部逐格复算一致；已标注为构造示例。**通过**。

**check.md §5 发布条件逐条判定**：

| §5 条件 | 判定 |
|---|---|
| 三轮审查均已完成且每轮由独立审查者执行 | 本轮为第 3 轮、由独立审查者执行；前两轮独立性需编排者确认（本轮不读取 `research/`，无法核实） |
| 每条来源论断都有引文依据记录；无法核对者已删除或降级 | **不满足**：4 处引文定位到不存在的子节（问题 2）、2 处公式定位错误（问题 3）、K3 论断无来源（问题 4）、`overview.html` 推理期论断无来源且与官方实现矛盾（问题 5）、$\gamma=0$ 理由无来源（问题 6） |
| 所有阻断和重要问题均已关闭 | **不满足**：1 阻断 + 5 重要全部未关闭 |
| 遗留轻微问题具有明确的接受理由 | **不满足**：本轮新增 6 条轻微，尚无接受理由 |
| 全部学习目标由正文章节完整回答 | 满足。核心问题 5 条分别由第 1、2、3、4、5 章完整回答，各条解答末尾均指明完整论证所在章节 |
| 核心问题与每个章节的本章问题均有解答折叠块 | 满足。页面级 5 个问题 + 各章 14 个问题（1 章 2 个、2/3/4/5 章各 3 个）共 19 个，全部紧跟 `解答：` 折叠块，无只列问题未作答 |
| 数学符号全部 LaTeX，结构图为 HTML 或内联 SVG | **不满足**：`MaxVio_global` 一处 ASCII 下标（问题 10）；页面无结构图，无 ASCII 框线图 |
| `validate.py` 返回成功 | 满足（exit 0）。但该脚本不检查标签闭合，未覆盖问题 1 |
| 可运行代码的结果与页面描述一致 | 不适用。页面未声明可运行代码，`index.html:896-912` 为已标注的伪代码，改按静态审查并记录原因 |
| 关键论断和数字已重新核对来源 | **不满足**：Table 2/3/4 与 V3 全部数字已核对通过，但问题 2、3、4、5、6 的论断未通过核对 |
| `<head>` 含 description／dojo:summary／dojo:type=concept／dojo:topics／dojo:tag | 满足（五项齐备，topics 在固定词表内，validate.py 通过） |
| `overview.html` 与 `index.html` 相互链接 | 满足（双向） |
| 页面引用的概念链接有效或具有明确占位 | 满足（`../moe-serving/index.html`、`../quantile-balancing/index.html` 均存在） |
| 递归生成的前置概念页已完成各自质检 | 本轮不可核实，需编排者确认 |

**是否满足 check.md §5 全部发布条件：不满足。** 具体缺口为「来源定位／引文依据」两条、「阻断与重要问题关闭」一条、「轻微问题接受理由」一条、「数学符号全部 LaTeX」一条，另有两条需编排者确认（前两轮独立性、递归前置概念页质检）。核心结论（bias 只调选谁不进权重、sign 固定步长规则式更新、只解决批级均衡、V3 配置）本身与来源一致，未被本轮推翻；但页面当前存在一处会使第 3–5 章与来源章节默认不可见、并从目录与章节导航中消失的标签闭合缺陷，须先修复该阻断问题，再逐条关闭 5 条重要问题后重跑 `validate.py` 与复验。
