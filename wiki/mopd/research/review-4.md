<!-- review-meta
round: 4
page: wiki/mopd/index.html
reviewed_content_sha256: 5fcb615f6c047c41
-->
# MOPD 审查记录（第 4 轮）

- 页面版本：index.html 工作树哈希 `24abb6c2e40f228bc86d022820d40b434c8ec09b`；overview.html 工作树哈希 `996f9bc02d53d7c927deb88554902bbb4bd11684`
- 审查时间：2026-09-13 19:43
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题 / 1. 合并动机 / 2. Per-token 奖励（含三 token 手算与折叠块）/ 3. 稳定性（含裁剪对照表与折叠块）/ 4. 多教师路由（含 mini-batch 折叠块）/ 5. 能力边界 / 来源与范围说明；含全部图注与折叠块，overview.html 全文

## 已核对来源（通过）

外部来源用 WebFetch 抓原文核对（K3 报告 arXiv:2607.24653 摘要页、HTML 全文 §4.1/§4.1.2/§4.1.3）：

- **Eq.15 公式**：报告原文 `r^d_opd(y_t|e,x,y_<t) = clip(sg(log π_teacher^(d,e)(y_t|x,y_<t)/π_θ(y_t|e,x,y_<t)), -R_max, R_max)`，与页面 L213 逐符号一致；`sg` 为停梯度算子、`Rmax>0` 为裁剪阈值，与 L226–228 一致。
- **9 个专家 = 3 领域 × 3 努力程度**：报告 "three domain experts ... at every reasoning effort level: (i) general tasks, (ii) general agents, and (iii) coding agents"、"reasoning levels {low,high,max}"、"a total of nine expert models"，支持 L147。
- **三阶段**：报告 "initializing baseline agent capabilities via supervised fine-tuning (SFT), developing specialized domain experts at varying reasoning effort via RL, and consolidating ... using Multi-Teacher On-Policy Distillation"，支持 L167 的 SFT 冷启 → RL 专家 → MOPD。
- **轨迹联合收集**：报告原文 "Trajectories produced by the resulting experts at all reasoning levels are jointly collected for supervised fine-tuning and multi-teacher on-policy distillation."，页面 L167 的直引与"未说明是否分区"的处理一致。
- **稠密奖励接入 RL / partial rollout**：报告 "This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks."，支持 L424 与 C6；partial rollout 定义 "the generation phase pauses as soon as a fraction λ∈(0,1) of trajectories completes" 支持 L424 的转述。
- **裁剪切断极端 advantage**：报告 "Rmax>0 is a clipping threshold to constrain extreme advantage signals, thereby stabilizing RL training."，支持 L317 的引述（译文忠实）。
- **K2.5 per-token 正则化**：报告 "policy optimization, which follows the algorithm in Kimi K2.5" 与 "inherently tolerates such an extreme off-policy regime through a per-token regularization"，支持 L424、L550。
- **top-k 无优势**：报告 "While we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting."，支持 L476、C7。
- **τ、b₀**：报告 "We associate each problem x with an initial token budget b₀(x) ... exceeds a scaled threshold τ·b₀(x)"，符号存在，支持 L550。
- **R_max 取值未披露**：报告仅称 clipping threshold，未给数值，与 L332、L534、L547 的"报告未给出取值"一致。
- **公式可复算**：三 token 例（L259–277）log(0.7/0.5)=+0.336472、log(0.2/0.3)=-0.405465、log(0.1/0.2)=-0.693147 全部复算通过；拆分式 log(π_teacher/π_θ)=log π_teacher−log π_θ 逐步数值一致。裁剪对照（L325–348）log(1.8)=0.588、log(9)=2.197、log(90)=4.500、log(900)=6.802 复算通过；截掉量 6.802−5=1.802、6.802−3=3.802 正确；本章问题手算 log(0.3/0.6)=−0.693 正确。全文符号单义。
- **页面功能**：`python3 .dojo/scripts/validate.py wiki/mopd/index.html` 返回 `validation ok`；内部链接 `../knowledge-distillation/index.html`、`../opd/index.html` 均真实存在；无 `research/` 死链、无"（待生成）"占位；结构图为 HTML + KaTeX；无 Unicode 数学字符出现在可见正文/标题/表格。

## 问题

- [轻微·表述] L102、L523 等：以"本文/本页"为主语的元话语与自我指代｜引文依据：不适用｜修复要求：删除或以无主语表述改写。L102"本文讲清楚它的奖励公式、稳定化算子、多教师路由机制和能力边界。"与 L523"回到开头的核心问题：本文讲清了为什么…"属"本页将…"式路线图元话语；L111/L151/L160/L183/L541 的"本页对…的推理""本页推理"、L167/L223/L476/L534/L540/L550 的"本文…"为自我指代。全文 本文 8 处 + 本页 8 处，显著多于同类页（opd 11 处、knowledge-distillation 7 处、kimi-k3 2 处），属系统性风格问题。推断标注可改为无主语表述（如"这是对平均不可行的推理，非 K3 报告结论"）｜修复：｜复验：
- [轻微·表述] L259、L271、L325、L336、L428：正文中"构造示例。"作为独立句后接空格再起正文（原文 `<p>构造示例。 设词表只有三个 token…`）｜引文依据：不适用｜修复要求：将"构造示例。"改为"构造示例："（冒号）并入后句，或删除该独立短句；全章 5 处统一处理｜修复：｜复验：
- [轻微·技术] L244、L254 图内以事实陈述写出"教师 π_teacher^(d,e) 冻结""更新学生参数 θ（教师不变）"，而正文 L167、L223、L311 明确将该冻结标注为"报告未明文、本文推断（报告未使用'冻结'一词）"，图文口径不一致｜引文依据：报告该节仅有 "sg(·) denotes the stop-gradient operator, and Rmax>0 is a clipping threshold…"，未出现"冻结/frozen"表述｜修复要求：图内加推断标注，如"教师 π_teacher^(d,e)（推断：MOPD 阶段不更新）"，与正文口径一致｜修复：｜复验：
- [轻微·技术] L149 首句"九个专家各自很强。"为无来源支持的判断，直接写成事实｜引文依据：K3 §4.1.2/§4.1.3 只把九个专家描述为按领域与努力程度训练的专项专家（"three domain experts at every reasoning effort level"、"a total of nine expert models"），未给出"强/很强"的评价｜修复要求：删除该句，或改为报告支持的表述（如"九个专家按领域与努力程度各自专精"）；该句不参与后续推理，删除不影响论证｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布。核心公式、全部来源论断与手算/表格数字均与 K3 报告 §4.1.2/§4.1.3 核对一致，无定位不到或扩大适用范围的论断，无来源结论与推断混淆（推断处均已明确标注为推测）。上述 4 条为表述/口径类轻微问题，不改变学习目标与主线结论：其中 L102/L523 元话语与 L149 无来源评价建议随本轮修复一并删除，L259 等"构造示例。"领起句与 L244/254 图内口径建议统一。遗留轻微问题接受理由：均不影响正确性、来源一致性与阅读连续性。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
