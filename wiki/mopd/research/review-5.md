<!-- review-meta
round: 5
page: wiki/mopd/index.html
reviewed_content_sha256: 915c01ce03f36390
-->
# MOPD 审查记录（第 5 轮）

- 页面版本：57a54d965cfd8939a496d1e8f6f2c2f4a146604a（wiki/mopd/index.html；`git status --porcelain wiki/mopd/` 为空，工作树与 HEAD 一致）
- 审查时间：2026-09-14 17:04
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复；本轮未读取 wiki/mopd/research/ 下任何文件）
- 适用规范：`guides/concept/check.md`（head `dojo:type=concept` 确认）+ `guides/concept/style-guide.md`
- 已完整阅读章节：主要依据／核心问题（5 个解答折叠块逐个读完）→ 1. 合并动机（含本章问题）→ 2. Per-token 奖励（含 1 个展开折叠块、本章问题）→ 3. 稳定性（含 1 个展开折叠块、本章问题）→ 4. 多教师路由（含 mini-batch 展开折叠块、本章问题）→ 5. 能力边界（含本章问题）→ 结语段 → 来源与范围说明（论断与来源 C／公式与来源 F／外部数字与实验条件 N／构造示例／辅助解释与类比边界／简化条件及其限制）；并完整阅读 overview.html。

## 来源核对（引文依据）

- K3 报告 §4.1.3 Eq.(15) 原文：`r^{d}_{opd}(y_{t}\mid e,x,y_{<t})=\mathrm{clip}(\mathrm{sg}(\log\frac{\pi_{\text{teacher}}^{(d,e)}(y_{t}\mid x,y_{<t})}{\pi_{\theta}(y_{t}\mid e,x,y_{<t})}),-R_{\max},R_{\max})`——页面公式逐符号一致，含教师侧条件顺序 `x,y_{<t}` 与学生侧 `e,x,y_{<t}` 及奖励上标 $d$。
- §4.1.3：`sg(·) denotes the stop-gradient operator, and R_max>0 is a clipping threshold to constrain extreme advantage signals, thereby stabilizing RL training.`——支持 C5 与稳定性章的归因，页面未超出该范围。
- §4.1.3：`This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks.`——支持 C6。
- §4.1.3：`While we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting.`——支持 C7（页面将其限定在 K3 设置内，与原文 `in our setting` 一致）。
- §4.1.2：`Crossing these three domain experts with three reasoning effort levels in {low, high, max} yields a total of nine expert models.`；三领域原文为 `general tasks`／`general agents`／`coding agents`——支持 C1、N1、N2。
- §4.1.2：`Trajectories produced by the resulting experts at all reasoning levels are jointly collected for supervised fine-tuning and multi-teacher on-policy distillation.`——支持 C3，且报告确未说明是否分区（页面注明的"未说明是否分区"成立）。
- §4.1.2：`Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through a per-token regularization. By constraining policy updates within a localized neighborhood … robustly handle highly stale data`，同节 `…policy optimization, which follows the algorithm in Kimi K2.5`——支持"K2.5 策略优化算法里的 per-token 正则化"表述。
- 全文检索：报告中 `merg` 命中 4 处全为 kernel/partial-sum 融合，无 weight averaging、无 model merging；`frozen` 仅出现在 §4.1.4（EAGLE-3 draft 的 target model）与 MoE 推理 bias，未用于蒸馏教师——支持页面把"未讨论权重平均"与"教师不更新"明确标注为本页推断。
- 全文检索 `MOPD` 仅 2 次命中（§4.1 概述与 §4.1.3），报告未给出 $R_{\max}$ 取值、top-k 对比数字或 MOPD 性能数字——支持第 5 章与 N 节的"不引用具体数字"。
- 复算（逐条重算，全部与页面一致）：$\log(0.7/0.5)=0.336472$；$\log(0.2/0.3)=-0.405465$；$\log(0.1/0.2)=-0.693147$；$\log(0.7)=-0.356675$、$\log(0.5)=-0.693147$；$\log(0.9/0.01)=\log 90\approx 4.50$；$\log 900\approx 6.802$；$\log(0.9/0.5)=0.588$；$\log(0.9/0.1)=2.197$；$6.802-5=1.802$；$6.802-3=3.802$；$4.50-3=1.50$；裁剪表 4 行 3 列与正文、折叠块、本章问题第三题三处数值一致。
- 机械项：`.dojo/scripts/validate.py wiki/mopd/index.html` 返回 `validation ok`；页面 12 个本地引用（libs 7 个、`../../index.html`、`overview.html`、`../knowledge-distillation/index.html`、`../opd/index.html`）全部存在；唯一 `alt` 为空串（lightbox 占位），无 `$...$` 混入 alt；无"（待生成）"占位；`dojo:topics=训练与优化`、`dojo:tag=训练` 均在 AGENTS.md 与 `ALLOWED_TOPICS`/`ALLOWED_TAGS` 内；无脚本时 HTML 结构流程图与全部折叠答案仍可读（无依赖 JS 生成的正文内容）。
- 表述维度（逐段含折叠块与图注通读）：全文无"我们/你们/咱们"、无"下面来看/需要注意的是/综上所述"等元话语；`本文/本页` 仅用作自称（style-guide §12 明确允许"本页"或"本文"），出现在推断来源标注与范围说明中；无调试叙事、无临场评价。开篇段"本文讲清楚它的奖励公式、稳定化算子、多教师路由机制和能力边界"与结语段"本文讲清了…"属 style-guide §7 允许的"开篇/收束说明文章结构"，不按元话语计。黄框 callout 用于"易误解澄清"，在 style-guide §3 语义内（全站 42 页有同类用法）。
- 符号一致性：$\pi_{\text{teacher}}^{(d,e)}$、$\pi_\theta$、$R_{\max}$、$\text{sg}$、$r_{\text{opd}}^{d}$ 全页写法统一，Unicode 数学字符只出现在纯文本 `description`（允许）与流程图的 ↓/→ 连接符（结构图元素，非公式）。

## 问题

- [轻微·技术] 来源与范围说明「论断与来源（C）」及正文第 1 章 `<sup>[C2]</sup>`：C2 的来源节归属错节，且 [C2] 在正文的挂载点与 C2 自身的定义不吻合。来源节写"…C2（MOPD 把九个专家合并进统一学生，按 $(d,e)$ 选教师）…：均来自 Kimi K3 技术报告（arXiv:2607.24653）§4.1.2 'Reinforcement Learning'"；但 C2 的两项内容实际都在 §4.1.3 而不在 §4.1.2。同时正文"训练一个统一学生模型 $\pi_\theta$，把努力程度 $e$ 作为模型的条件输入<sup>[C2]</sup>"把 [C2] 挂在了"学生条件化于 $e$"上，而该结论的依据是 Eq.15 的学生侧条件 $\pi_\theta(y_t\mid e,x,y_{<t})$（即 [F1]），并不在 C2 的定义范围内。｜引文依据：§4.1.3 `We adopt Multi-Teacher On-Policy Distillation (MOPD) to consolidate these domain-specialized capabilities across varying reasoning efforts into a unified model` 与 `optimization is guided by the corresponding teacher model π_teacher^{(d,e)} among the nine experts`；§4.1.2 仅含九个专家＝3 领域 × 3 努力程度及轨迹联合收集，无"合并进统一学生"与"按 $(d,e)$ 选教师"。｜修复要求：把 C2 从"均来自 §4.1.2"的归类中拆出并改标为 §4.1.3（C1、C3 保留 §4.1.2），或按定义重写 C2 使其内容与 §4.1.2 相符；并把正文第 1 章该处 [C2] 改挂到与学生条件化无关的句式上（例如改挂到"按 $(d,e)$ 选出对应教师"处，或把 $e$ 作为条件输入的标注改为 [F1]），使上标编号与其定义一一对应。｜修复：｜复验：

## 结论

- 处置：修复（无阻断、无重要问题；仅 1 条轻微来源定位问题，修复并复验后可发布）
- 统计：阻断 0 / 重要 0 / 轻微 1