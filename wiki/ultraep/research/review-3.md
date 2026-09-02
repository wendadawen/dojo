# UltraEP 审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 47f5f0ad8bf533c52a4bcc22bf20588946905fc0；overview.html 工作树哈希 47eee0c340e71dce52f5006e5cbc9d29c0e6ec72
- 论文版本：arXiv:2606.04101v3（2026-06-18 提交；v1 2026-06-02、v2 2026-06-05 均已撤回，经 arXiv 摘要页核对）
- 审查时间：2026-09-02 00:39
- 审查者：独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：index.html 全文（meta blockquote、引言、核心问题 4 题及解答、术语速查、贯穿示例、第 1–7 章全部正文与折叠块、来源与范围说明、页面 JS）；overview.html 全文；论文 TeX 源码 main.tex 与 sections/ 全部 8 个文件；figs/ 中 9 张关键图经 pdftoppm 渲染后与页面核对

## 核对方式与已核对项（无问题，简述）

- **C1–C38 来源论断**：逐条在 TeX 源码定位，全部相符。例如 C12 对应 §4.1 原文 "reduces a single redundant slot from 3.3 GB weights and 6.6 GB gradients to 36 MB and 72 MB per rank"（94 MoE 层、128 专家）；C24 对应 §6.2 "The exposed bound is sender-side: each rank has at most $N_{\text{slot}}$ inbound replicas" 与 "the relay frontier is chosen near $\sqrt{|\mathcal{H}(e)|-1}$"；C34 对应 §2.2 末段 "complementary rather than interchangeable"。
- **公式 Eq.(1)–(6)**：与 §4.3、§5.1 逐式核对一致；Algorithm 1 的 $\tau_{\mathrm{lo}}=\beta\lceil\frac{1}{R}\sum\ell_r\rceil$、$\delta=\min(\mathrm{exc}_r,\mathrm{slk}_{t^\star},\mathrm{cap}_e)$、比例分摊式与双向守恒（Table 1 Q 行）均一致。
- **实验数字**：Figure 11 标注值（757/524/613、ideal 785.4/574.7/637.6、不均衡 1.02/1.03/1.01 等 20 个数）与页面表格全对；派生值 96.4/91.2/96.1% 与均值 94.6%、+20/+12/+29/+42% 复算吻合（41.9%≈42% 等）。Figure 12 四组不均衡（3.68/4.01/3.09/2.06 → 1.04/1.03/1.01/1.01）全对。Figure 13 六项数值全对；派生 0.33 ms（3.16−2.83）、1.8%（0.33/18.31）、+33%（2.24/1.68）、+10%（1.79/1.62）复算吻合。Table 3 五项（1.19/1.03、0.153/0.111、107/45、8.5/6.8、99.9%/96.0%/98.4%）全对；派生 27.4%（0.042/0.153）、57.9%（62/107）、3.9 与 2.4 个百分点吻合。Figure 16 页面读数（0.92/0.73/0.22/0.24 等 20 个数）与我渲染读图一致（误差 ≤0.02 ms），页面已声明为读图值；加速区间 3.1–5.5× 按论文正文引用，未用读数反算，处理正确。Figure 17 标注 ideal 504、no-balancing 425.0 与图一致；派生 425×1.096≈466、466/504≈92.4% 与正文 ">92%" 自洽。
- **代码块**：提取第 4.5 节折叠块内 Python 代码实际运行，输出与页面预期逐行一致（τ=6、不均衡 1.0、耗 2/4 槽、round-robin 1.1667、跨 rank 41.7%/54.2%，双向守恒断言通过）。二分三次探测（τ=9→7→6）与 u_min=3 反事实（最终停在 7）均手算复核成立。
- **图片对应**：img-01..11 与论文 Figure 1/2/6/7/8/10/15/16/11/13/17 对应关系抽查（img-03=Figure 6、img-09=Figure 11、img-11=Figure 17）逐像素级一致；Figure 1 的 "~0.3 ms" 标注、Figure 6 的 Layer 68/57 标题均在图中核实。
- **机械项**：`.dojo/scripts/validate.py` 两页均通过；无 Unicode 数学字符（仅表格行标签用间隔号「·」）；8 个概念页链接全部存在；标题编号 1–7 章连续；「核心问题」「本章问题」命名正确且每题有「解答：」折叠块，核心问题均指明论证章节；overview↔index 互链双向存在；head 元信息齐全；v3 版本与撤回声明经 arXiv 页面核对属实；「2560 卡」传闻的降级处理（不引用并说明原因）符合规范第 2.2 条。

## 问题

- [重要·技术] 第 4.9 节正文（"冗余槽消耗 45 对 107 这一项差距，主要就来自 $u_{\min}$ 剪掉的那些无效副本"一句）：把机制级归因紧随（§8.5）引文写成论文事实，无推断标记；而页面「来源与范围说明」声明正文仅有三处随文标记的推断（不含此句），且同章「本章问题」第二题解答末尾自己承认"把它进一步落到 $u_{\min}$ 这一具体机制上，是本页的推断"——同页两种口径矛盾，构成事实与推断混淆｜引文依据：§8.5 原文 "Unlike EPLB$+$, which blindly replicates experts based on pre-reroute hotness, \sys only materializes a replica when it brings sufficient balancing gain. This accounts for \sys's resource efficiency"——论文把槽数差距归因于「只在有足够收益时才实体化」的整体设计，未点名 $u_{\min}$｜修复要求：在该句随文标注「本页推断」，或改写为论文原意的归因表述；同时在「来源与范围说明」的推断清单中补上这一处｜修复：4.9 节该句末尾补「这一步把归因落到 $u_{\min}$ 这一具体机制上，是本页的推断，论文原文只把差距归因于『只在有足够均衡收益时才实体化副本』的整体设计」；「来源与范围说明」推断清单由「三处」改为「四处」并补入该条｜复验：已重新读取两处确认标注与清单一致，validate.py 重过

- [轻微·技术] 第 5.6 节（"需要发出的远端份数是 2，低于中继阈值 4，因此不启用中继"）与 overview.html 方法概述（"中继集合的宽度近似取副本数的平方根"）：论文 §6.2 阈值操作数是 "replica count"（"For experts whose replica count exceeds the relay threshold (set to 4)"），中继前沿是 $\sqrt{|\mathcal{H}(e)|-1}$；5.6 用 $|\mathcal{H}(e)|-1$（远端份数）与阈值比较，若 "replica count" 含主实例则操作数差 1。两种读法下贯穿示例结论（不启用中继）相同，不影响正确性；index.html 5.5 的公式写法本身是准确的｜引文依据：§6.2 "For experts whose replica count exceeds the relay threshold (set to 4), \sys builds a lightweight two-stage relay … The relay frontier is chosen near $\sqrt{|\mathcal{H}(e)|-1}$"｜修复要求：5.6 改为与论文一致的「副本数」表述或注明与 $|\mathcal{H}(e)|$ 的换算；overview 明确「副本数」不含主实例｜修复：5.6 改为「无论按远端份数还是按含主实例的实例数 3 与阈值比较，都不超过中继阈值 4」，两种读法并列、消歧；overview 改为「中继集合的宽度近似取 $\sqrt{|\mathcal{H}(e)|-1}$（$|\mathcal{H}(e)|$ 为含主实例的实例数）」｜复验：已重新读取两处确认，validate.py 两页重过

- [轻微·格式] 第 6.4 节常见误解框、第 6 章本章问题第二题解答、「来源与范围说明」C30 三处标注「§8.2 末段」：该句实际位于 §8.2 内部 Training 段的末尾，而 §8.2 的最后一段是 Serving Prefill 段，按「末段」定位找不到｜引文依据：5-evaluation.tex 中 "The remaining gap to force-balancing mainly comes from uneven routing in realistic MoE training, instead of residual imbalance or hot-path balancing overhead" 位于 §8.2 Training 段末句，其后还有 Serving Prefill 段｜修复要求：三处改为「§8.2 训练段末」｜修复：6.4 节常见误解框、第 6 章本章问题第二题解答、「来源与范围说明」C30 三处均已改为「§8.2 训练段末」｜复验：grep 确认全站（index.html）已无「§8.2 末段」，validate.py 重过

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（1 条重要问题关闭后满足发布条件；2 条轻微问题建议一并修复）
