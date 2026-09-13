<!-- review-meta
round: 1
page: wiki/mopd/index.html
reviewed_content_sha256: dcf13ee722a7827a
-->
# MOPD 审查记录（第 1 轮）

- 页面版本：index.html c064be95e99f8005c99ce859a3dec06d958c4548（overview.html 21bd937f7174e5ba9c5d9f7ef5cd787a7944e4f2）
- 审查时间：2026-09-13 18:46
- 审查者：独立子代理
- 已完整阅读章节：引言/meta、核心问题、1. 合并动机、本章问题、2. Per-token 奖励、本章问题、3. 稳定性、本章问题、4. 多教师路由、本章问题、5. 能力边界、本章问题、来源与范围说明（全部 h3 与折叠块）、overview.html 全文
- 已核对来源：Kimi K3 技术报告 §4.1.1–§4.1.3 与 Eq.15（arXiv:2607.24653v2，经 arxiv.org/html 与 ar5iv 两路抓取交叉核对）

## 问题

- [阻断·技术] Per-token 奖励章，第 162 行（亦见核心问题 1 答案第 122 行、对照表第 171 行、overview.html 第 28 行）：把"九组权重平均后互相抵消"写成无条件来源事实。该机制论断在来源中定位不到，K3 报告通篇未讨论权重平均／模型合并，页面自身的来源章节（C/F/N/辅助解释与类比边界）也未登记此论断。｜引文依据：K3 报告 §4.1.3 仅称 "We adopt Multi-Teacher On-Policy Distillation (MOPD) to consolidate these domain-specialized capabilities across varying reasoning efforts into a unified model"，全文无 weight averaging / model soup / 为何平均失败 的任何表述（经两路抓取核对）；页面第 162 行"它们的参数空间没有对应关系……平均后会互相抵消"无任何引注。｜修复要求：按 check.md §2.2"无法定位的机制描述删除或降级为明确标注的推断"，二选一执行：(a) 删除"参数空间没有对应关系／互相抵消"的机制陈述，仅保留"直接平均不是 K3 采用的合并路线"；或 (b) 改写为明确标注的推断或辅助解释，说明"这是对独立 RL 微调模型的通用判断，非 K3 报告结论"，并在"辅助解释与类比边界"小节登记其失效边界（同架构同初始化模型之间参数下标其实存在对应关系，真正问题是取值发散导致的相互干涉）。三处正文与 overview 须同步。

- [重要·表述] 全文（元话语与固定句式）：第 414 行"这里有一个细节值得注意："、第 247 行"这个循环可以用下面的图示表示……具体数值由后面的手算例子给出"、第 267 行"下面用一个最小例子手算"、第 333 行"下面用同一组极端概率比对照"；四章末尾过渡统一为"X 清楚了——下一章看 Y"模板（第 206、315、386、478 行）；另有口语化临场评价"它的含义很直接"（第 245 行）、"插进已有的 RL 流水线就行"（第 434 行）。均为 check.md 第 12 项明令排除的元话语／固定句式／临场评价。｜引文依据：不适用。｜修复要求：删除"下面…""值得注意"类引导语，改为直接陈述；第 206／315／386／478 行四章过渡改写为互不相同的句式，各自点明前一节结论与下一节待解决问题之间的逻辑缺口（style-guide §8"不使用固定句式"）；替换"很直接""就行"等口语措辞。

- [重要·事实一致性] overview.html 第 42 行：把教师冻结写成来源事实——"9 个专家必须先由 RL 训练好并冻结"。index.html 第 178、233、321、514 行多处明确标注"教师是否冻结"非 K3 报告明文、系由 sg 算子推出的推断，两页对同一事项一为推断一为事实，互相矛盾。｜引文依据：经抓取核对，K3 报告未说明九个专家教师在 MOPD 阶段是否冻结（§4.1.3 仅称 "optimization is guided by the corresponding teacher model"）；index.html 第 178 行自称"K3 报告未明文使用'冻结'一词，本文依此推断"。｜修复要求：overview.html 第 42 行改为与 index.html 一致的标注推断措辞，例如"9 个专家先由上游 RL 训练好；教师在 MOPD 阶段是否冻结系由 Eq.15 的 sg 推出，K3 报告未明文"。

- [轻微·公式] Per-token 奖励章，第 224、262、324、367 行：奖励符号全页写法不一致——第 224 行写 $r_{\text{opd}}^{d}$，第 262、324、367 行写 $r_{\text{opd}}$；且公式的逐项符号说明（第 228–239 行）未定义 $r_{\text{opd}}$ 本身。｜引文依据：K3 报告 Eq.15 写作 $r^{d}_{opd}$；页面第 224 行 $r_{\text{opd}}^{d}$ 与第 262/324/367 行 $r_{\text{opd}}$ 并存。｜修复要求：统一写作 $r_{\text{opd}}^{d}$（或统一去上标），并在符号表补一行定义 $r_{\text{opd}}^{d}$："教师与学生对数概率比经 clip(sg(·)) 后的 per-token 奖励"。

- [轻微·内容] 第 178、482、513 行："两阶段流程"框架。报告 §4.1 实为三段：§4.1.1 SFT 冷启、§4.1.2 RL 训练专家、§4.1.3 MOPD 合并；页面把 SFT 折出、称后训练为"两阶段流程"。｜引文依据：经抓取核对，报告小节顺序为 §4.1.1（Supervised Fine-Tuning）→ §4.1.2（Reinforcement Learning）→ §4.1.3（MOPD）。｜修复要求：明确"两阶段"仅指 {训练九专家 → MOPD 合并} 这一对，或改为"合并阶段（在 SFT 冷启与专家 RL 之后）"，避免读者以为后训练只有 RL 与 MOPD 两段。

- [轻微·技术] 能力边界章，第 486 行："在 token 级信号上对教师分布的 top-k 概率施加额外约束"是页面自行补出的机制解读，报告原文仅称曾实验更精细的 top-k 目标，未述形式；页面后一句虽标"具体形式报告未详述"，前一句仍以具体机制陈述。｜引文依据：报告原文 "we also experimented with more fine-grained top-k distillation objectives" 与 "observed no clear advantage in either convergence speed or final performance"，未展开该目标形式。｜修复要求：删除对 top-k 目标形式的具体描述，改为"更精细的 top-k 蒸馏目标（具体形式报告未详述）"，不自行补出"对教师分布 top-k 概率施加约束"的机制。

## 核对结论（通过项）

- 公式复算：第 272–275、281–286、310、327、338–339、349–358 行的全部数值（log(0.7/0.5)=0.336、log(0.2/0.3)=−0.405、log(0.1/0.2)=−0.693、log 90≈4.50、log 900≈6.80、log 1.8≈0.588、log 9≈2.197、截断量 1.802/3.802）均复算无误。
- Eq.15 与报告逐字一致：$r^{d}_{opd}(y_t\mid e,x,y_{<t})=\text{clip}(\text{sg}(\log\frac{\pi^{(d,e)}_{teacher}(y_t\mid x,y_{<t})}{\pi_\theta(y_t\mid e,x,y_{<t})}),-R_{\max},R_{\max})$；九个专家（3 领域 general tasks／general agents／coding agents × {low,high,max}）、C3 轨迹联合收集（§4.1.2 Reasoning Effort RL）、C6"integrates seamlessly … such as partial rollout training"（§4.1.3 Eq.15 之后末段）、C7 top-k 无优势、K2.5 per-token 正则化与 partial rollout 机制（§4.1.2）均与报告原文相符；"冻结""部署无需领域路由器"等推断已按 check.md §2.2 标注。
- 页面链接：../knowledge-distillation/index.html、../opd/index.html 均真实存在，无"（待生成）"占位。
- 问题块：页面级"核心问题"5 条、各章"本章问题"各有解答折叠块，核心问题答案均指明完整论证所在章节；命名符合 style-guide §9。
- 机械项：`.dojo/scripts/validate.py` 对 index.html 与 overview.html 均返回 "validation ok"；无 Unicode 数学字符直出（description/dojo:summary 字段正常）；结构图为 HTML（flow-diagram），非等宽框线图；KaTeX/折叠/目录锚点/本地 libs 资源齐全。
- 表述其余项：全文无第一人称复数、无第二人称指代读者、无"场景"当术语、无无结论占位表述；"构造示例。 "前缀与 content-examples A4/A8 的既定写法一致，不算问题。

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 3
- 处置：修复（第 1 条阻断须删除或降级为标注推断并同步 overview；第 2 条元话语/固定句式按第 12 项清理；第 3 条统一 overview 推断标注；轻微项按各自要求处理）。修复后重新运行 validate.py，下一轮从修复后的完整页面重审。
