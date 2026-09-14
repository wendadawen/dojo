<!-- review-meta
round: 5
page: wiki/ultraep/index.html
reviewed_content_sha256: dcd328db055befd5
-->
# UltraEP 审查记录（第 5 轮）

- 页面版本：b7e70803da07ecd9f4ed5e0773e03c2ef2a893c7（工作树索引，sha256 前 16 位 ad1922347374a3fd）
- 论文版本：arXiv:2606.04101v3（2026-06-18 提交；v1/v2 已撤回，与页面 head 的发表信息一致）
- 审查时间：2026-09-14 17:19
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：术语速查 → 贯穿全文的最小例子 → 1. 专家热度漂移（1.1–1.5 及本章问题）→ 2. 精确负载与 RSN（2.1–2.5 及本章问题）→ 3. 只复制不重排（3.1–3.6 及本章问题，含折叠块）→ 4. 配额规划（4.1–4.9 及本章问题，含代码折叠块）→ 5. tile 流水线与 chunk 中继（5.1–5.7 及本章问题，含实现折叠块）→ 6. 实验结果（6.1–6.7 及本章问题）→ 7. 方法评价（7.1–7.3）→ 来源与范围说明 → 页脚脚本

## 问题

- [轻微·格式] 第 3.6 节（`index.html:392`）：输出描述句出现重复字「输出是配额式复制方案 $U$ 与与之一致的重路由拆分 $q_{r,e,t}$」，「与与之」应为「与之」。｜引文依据：不适用｜修复要求：删除重复的「与」，改为「与之一致」；全文复查同类重复字（已核 `的的`/`了了`/`是是`/`在在`/`和和` 均无）。｜修复：｜复验：
- [轻微·技术] 第 1.3 节（`index.html:192`）与第 1 章本章问题（`index.html:227`）：算例「一个专家变热两倍，它在本 rank 总负载中只占十六分之一」把「增量占比」写成了「占比」。16 个专家等载 L 时，一个专家变 2L 后其份额为 2L/17L≈1/8.5，不是 1/16；十六分之一是这次变热带来的**增量**（(2L−L)/16L）占原负载的比例。机制结论（摊薄能力随每 rank 专家数减少而消失）不受影响，但给出的数量关系与算式不符。｜引文依据：不适用（本页构造算例，非论文数据；论文 §4.1 仅给「large-EP reduces the number of local main experts per rank (often two or four)」）｜修复要求：把两处改为不含歧义的表述，例如「某个专家变热两倍，带来的增量只占本 rank 原负载的十六分之一（EP=64 时这一增量约占本 rank 负载的一半）」，或直接删去「十六分之一」这一数字只保留 EP=8/EP=64 的专家数对比。｜修复：｜复验：

## 核对留痕（本轮逐条回源）

论文侧材料：arXiv:2606.04101v3 HTML 全文（arxiv.org/html/2606.04101v3）与 PDF（15 页）逐节比对；图内数值用像素测量（坐标轴刻度换算，误差 ±0.005）。

- 摘要与 §1：94.3%、1.49×、1.30–4.01→1.01–1.04、训练 94.6%/prefill 93.9%、训练相对 Megatron-LM 1.42×、prefill 相对 SGLang 1.56×、专家复制 3.1×–5.5×、生产 >92%——原文「UltraEP achieves 94.3 % of the force-balanced ideal throughput, delivering 1.49× improvement over no-balancing, while reducing the final inter-rank imbalance from 1.30–4.01 to 1.01–1.04」「94.6 % … in training and 93.9 % in serving prefill」「1.42× over Megatron-LM」「1.56× over SGLang」「accelerates expert replication by 3.1×–5.5×」「over 92 % of ideal throughput」。逐条相符。
- §1 三条后果与复合放大、EPLB 冗余专家策略与周期性部署、EP 度数放大、精确负载进关键路径、4/8 卡单机 scale-up、跨机不实用——原文 §1 第 2–5 段与 §2.1 逐句对应，页面无扩写。
- §3：prefill 三向漂移（step/数据域/层）、Qwen3-235B（top-8 of 128，EP=64）、训练早期不稳/后期辅助损失负反馈/microbatch 抖动且 DeepSeek-V3 更明显、GLM4.5-106B-A12B 与 DeepSeek-V3-671B-A37B 的 EP64 组——原文 §3「Serving Prefill」「Training」两段逐句相符。
- §3 末段：「每个 EP rank 上的专家更少时，大 EP 直接把路由动态转化成明显的 rank 间倾斜」「EPLB 可能加剧不均衡、制造尖峰与新的 straggler」——原文「With fewer experts per EP rank, large-EP directly translates routing dynamics across experts into pronounced inter-rank skew」「EPLB can even worsen imbalance, creating spikes and new stragglers」。相符。
- §4.1：逻辑/物理专家、主槽/冗余槽、只复制不重排及理由、优化器状态只作用于主专家、权重与梯度 buffer 跨层共享、Qwen3-235B-A22B（94 MoE 层、128 专家）单槽 3.3 GB/6.6 GB → 36 MB/72 MB——原文「in Qwen3-235B-A22B (94 MoE layers, 128 experts), this reduces a single redundant slot from 3.3 GB weights and 6.6 GB gradients to 36 MB and 72 MB per rank」。页面「原值除以层数」的换算 3.3×1024/94≈35.9 MB、6.6×1024/94≈71.9 MB 可复算。
- §4.2：前向（复用 notify-dispatch、各 rank 确定性算出同一方案、reroute 与权重分发可重叠但 token dispatch 必须等权重分发结束）、反向（权重重实体化可与 Wgrad 重叠至 Dgrad 前、梯度归约保持无副本形式等价、须在下一层前完成、复用前向缓存元数据）——原文 §4.2「Forward/Backward」两段逐句相符。
- §4.3 Table 1 与 Eq.(1)–(5)：$\mathcal{R},\mathcal{E},h,\mathcal{H}(e),N_{\mathrm{slot}},\Lambda,U,Q,u_{\min}=1024,\beta=1.01$ 全部与原文 Table 1 一致；Eq.(1) 结构（solve_rep + max(reroute, w_distr) + tok_a2a + moe）、Eq.(2)、Eq.(3) $\max_r\sum_e u_{e,r}$、Eq.(4) $\max_r\max(\sum_e\lambda_{r,e},\sum_e u_{e,r})$、Eq.(5) $\max_r\sum_{e\in\mathcal{E}_r}(|\mathcal{H}(e)|-1)$ 与原文公式逐项一致；四条约束一致。
- §4.3 折叠块代入检查：初始 rank 负载 12/6/4/2、行和 8/6/6/4、均衡后 6/6/6/6；Eq.(3) 12→6、Eq.(4) 12→8、Eq.(5) 0→2 全部手算复核通过，方向与原文一致。
- §4.4/4.5 贯穿示例：二元搜索 $\tau_{\mathrm{lo}}=1.01\times\lceil24/4\rceil=6.06$、$\tau_{\mathrm{hi}}=12$；三次探测 τ=9→7→6 的 exc/slk/δ 与最终配额（e0 在 r0/r2/r3 为 4/2/4）、消耗 2/4 冗余槽，逐步手算与页面一致。
- §4.5 折叠块代码：抽取页面 `<pre><code class="language-python">` 原样执行（Python 3.9，exit 0），输出与页面「预期输出」逐行一致（含 `{0: 6, 1: 6, 2: 7, 3: 5}` 不均衡 1.1667、本地优先 41.7% 对 54.2%、两个分支断言均通过）；把 `U_MIN` 改为 3 复跑，得 τ=6 不可行、最终 τ=7、消耗 1 槽，验证了页面 callout 中「$u_{\min}$ 调到 3 则阈值停在 7」的反事实说明。
- §4.9 Table 3：1.19/1.03、0.153/0.111 ms、107/45、8.5/6.8、99.9%/96.0%（98.4% w/o locality）与原文 Table 3 逐格一致；派生百分比 27.4%（1−0.111/0.153）、57.9%（1−45/107）、3.9 个百分点、局部性贡献 2.4 个百分点（98.4−96.0）可复算。
- §4.9 Figure 15 读图：下排 (128,64,1) 面板像素测量 EPLB+ 在初始不均衡 6.0/8.0 处为 1.40/1.395，UltraEP 为 1.075/1.095，与页面「约 1.40」与「约 1.07 与 1.09」相符；论文正文对应表述「EPLB++ shows much higher imbalance up to 1.4, while UltraEP still keeps it below 1.1」一致。
- §5.7 Figure 16 读图：像素测量（换算 val=(549−y)/270，网格线 0.4/0.8/1.2/1.6 间距 108 px）得 torch.distributed 0.926/1.174/1.019/1.126/1.533，DeepEP 0.737/0.874/1.026/1.200/1.281，无中继 0.226/0.274/0.370/0.478/0.522，UltraEP 0.244/0.285/0.293/0.285/0.289；与页面表格五档数值逐格相符（≤0.005），中继启用数 0/0/1/2/3 与图上 x 轴标注一致。页面明确标注该表为读图值且「按论文表述引用 3.1×–5.5×、1.3×–1.8×，未使用反算值」，与原文 §8.5 一致。
- §6.1 Table 2 与基线：模型/专家数/top-k/并行配置/N_slot（128(8) EP64-DP2|— 2；128(8) EP64-DP4|EP64 2；160(8) —|EP40 4；256(8) EP64-PP4|— 2）、每机架 64 卡/16 服务器、scale-up 为 scale-out 的 8–10 倍、prefill 1 机架/训练 2 或 4 机架、200B token/约 4500 global batch/batch 1024→5120、GShard 权重 $10^{-2}$、DeepSeek 配方 $10^{-3}$/$10^{-4}$、Codeforces/SWE-bench/DAPO-Math-17K/GPQA/OpenScience/LongBench/泊松到达、Megatron-LM e93814b、SGLang v0.5.9+bbe9c7e、EPLB 50 步/3 global batch、LPLB 每专家至多一副本——原文 §8.1 逐项一致。
- §6.2 Figure 11 读图与派生：像素/标注值 545/646/618/695/757、785.4；315/449/385/474/524、574.7；509/505/516/553/613、637.6；不均衡 2.23/1.34/1.29/1.10/1.02、2.86/1.34/1.58/1.17/1.03、1.30/1.28/1.09/1.18/1.01——与页面表格逐格一致。复算：757/785.4=96.4%、524/574.7=91.2%、613/637.6=96.1%、三者均值 94.6%；相对 Megatron-LM 三模型平均 EPLB 1.201(+20%)、LPLB 1.123(+12%)、EPLB+ 1.289(+29%)、UltraEP 1.419(+42%)，与原文 §8.2「20 %, 12 %, 29 %, and 42 %」一致。DeepSeek-V3 列 EPLB 505 < Megatron-LM 509 支持页面「相当甚至更差」的表述与原文「similar or even worse performance」。
- §6.3 Figure 12 读图（原图未入页）：SGLang/EPLB/EPLB+/UltraEP = 3.68/2.59/1.11/1.04（Qwen3-235B STEM）、4.01/2.56/1.11/1.03（Qwen3-235B Mixed）、3.09/2.05/1.08/1.01（GLM4.7-358B STEM）、2.06/1.80/1.06/1.01（GLM4.7-358B Mixed），与页面表格逐格一致；摘要 4.01 上界出自 Qwen3-235B·Mixed 一行成立。
- §6.4 Figure 13 读图：前向 Ideal/Megatron-LM/Ours = 1.89/5.95/2.11（MoE 计算）、1.68/6.94/2.24（all-to-all）、2.83/2.88/3.16（其余）；反向 4.32/10.41/4.43、1.62/4.75/1.79、4.49/4.52/4.58——与页面表格逐格一致；派生 5.95/1.89=3.15、6.94/1.68=4.13、2.24/1.68=+33%、1.79/1.62=+10%、3.16−2.83=0.33、4.58−4.49=0.09 均可复算，且页面已标注 3.15/4.13 为本页算出、论文只作定性表述。
- §6.5/§6.6/§6.7：不做均衡时 MoE 活跃内存峰值相对 ideal 训练高 2×、serving 高 11×，训练侧关闭激活重计算并取初期高倾斜峰值、serving 侧只保留当前层瞬时值；RefMoE-288B-A16B/EP32/>92%/9.6%/loss 曲线原因；Figure 17 的 ideal 504 与 w/o balancing 均值 425.0（页面 425.0×1.096≈466、466/504=92.4% 反推链自洽，且图上 Ours 曲线确在 460–467 区间）——原文 §8.4/§8.6 一致。
- §7 全部为分析性判断，章首已用提示框标明「内容属于分析性判断，不是论文的结论」；对 Figure 15 退化、生产证据单薄的描述与 §8.4/§8.6 事实一致。
- 结构图：`index.html` 内两处自绘图为纯 HTML（`.dg-flow`/`.dg-stack`）无等宽字符框线、无内联 SVG、无 `$...$` 出现在 `alt`；KaTeX 渲染全部 415 处数学片段（含 `<meta name="dojo:summary">`）零失败；`validate.py` 返回 `validation ok`；8 个前置概念链接（moe-serving、aux-loss-free-routing、gpu-communication、model-parallelism、chunked-prefill、gpu-execution-model、deepseek-moe、moonep）目标文件均真实存在，无「（待生成）」；overview.html 与 index.html 互链；11 张图片资源齐全且与 Figure 1/2/6/7/8/10/15/16/11/13/17 的编号对应经内容核对成立。
- 表述维度：折叠块与图注逐段通读，未发现「下面来看」「需要注意的是」式元话语、调试叙事或临场评价；「本页」的使用集中在「来源与范围说明」与图片读值/推断标注处，属该站全站既有约定（其余 14 个页面同样使用），且承担 check.md 要求的「事实与推断区分」功能，不构成缺陷；章节问题与核心问题（4+20 个 `<details>` 折叠块共 27 个）逐一有解答，核心问题答案均指明完整论证章节（第 1/2/4/3+5 章）。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布。两条轻微均为局部改字/改句，不涉及论文数字与结论：`index.html:392` 的重复字可直接删除；`index.html:192` 与 `index.html:227` 的「十六分之一」改为「增量占本 rank 原负载的十六分之一」或删去该数字。除此之外，本轮表格数字、图内读数、公式、代码输出与原文逐条对得上，未发现与论文不符或自相矛盾之处。
