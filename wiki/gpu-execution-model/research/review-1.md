<!-- review-meta
round: 1
page: wiki/gpu-execution-model/index.html
reviewed_content_sha256: 3a0c798387f6ec14
-->
# GPU 执行模型与 kernel 调度审查记录（第 1 轮）

- 页面版本：695ee49ee139b5c98d26129bfd9780fa2b0d915e
- 审查时间：2026-09-13 18:48
- 审查者：独立子代理（未参与写作，未读取 research/ 下规划、修复与前序审查记录）
- 规范：guides/concept/check.md（页面 head 声明 `dojo:type=concept`）
- 来源获取方式：本页未给出任何外部链接（`[C1]`–`[C16]`、`[N1]`–`[N8]`、`[F1]` 只给文档名/章节号）。审查时按名称定位到官方原文核对：CUDA Driver API（cuStreamCreateWithPriority、Green Contexts）、NVIDIA MPS 文档、NVIDIA MIG User Guide（supported-mig-profiles）、Hopper Tuning Guide、NVIDIA Hopper Architecture In-Depth 博客。本轮不读取 research/ 任何材料。
- 已完整阅读章节：核心问题、最容易误解、1. 先认识硬件、2. kernel 是怎么跑起来的、3. 为什么矩阵乘要切成 tile（3.1–3.3）、4. Hopper 的协作工具箱（4.1–4.3）、5. 反复启动太贵（5.1–5.2）、6. 分一张卡的四种机制（6.1–6.5）、来源与范围说明；含全部 details 折叠块与 flow 图注。
- 机械校验：`python3 .dojo/scripts/validate.py wiki/gpu-execution-model/index.html` 返回 validation ok；`index.html`↔`overview.html` 互链有效；`../moe-serving/index.html`、`../vllm-cudagraph/index.html` 均真实存在，无「（待生成）」占位。已复算：4×4 例子 64/32/128、表格 $N/T$ 行、$128\times128\times2=32768$B=32KB、双缓冲≈128KB、$512\times512\times2=512$KB、$8192/128=64$→$16384/128=128$，均正确。已核对通过的数字：132 SM、每 SM 4 个第四代 Tensor Core、50MB L2、80GB HBM3@3.35 TB/s、228KB/227KB shared memory、warp=32、每 SM 最多 64 warp、CTA≤1024、TMA 支持 1–5 维、cluster 可移植上限 8 / H100 opt-in 16、8 GPC、MPS 限额语义、MIG 六档 profile 的实例数（7/4/3/2/1/1）、Green Context CUDA 12.4 与 CC 9.0+ 的 8 倍数。

## 问题

- [阻断·技术] §6.3「MIG」段（`H100 有 7 个可用的 GPU 计算切片（对应 7 个 GPC，另留 1 个 GPC 给管理用）`）｜引文依据：NVIDIA MIG User Guide「Supported MIG Profiles」对 H100 80GB 的官方表示是「Fraction of SMs：1/7、2/7、3/7、4/7、7/7」，计算能力以「占 SM 的比例」表述，全文未把计算切片等同于 GPC，也没有「为管理预留 1 个 GPC」的说明；该页同章 [C15] 又写「H100 有 8 个 GPC」，与「7 切片=7 GPC 且 7g.80gb=整卡」相互矛盾（7 个切片不可能既等于 7 个 GPC 又等于含 8 个 GPC 的整卡）｜问题：括号内的硬件映射说明没有任何来源支持，且与官方表述不符，属于把推断写成官方事实｜修复要求：删除「（对应 7 个 GPC，另留 1 个 GPC 给管理用）」，改为与官方一致的「7 个计算切片（每片约占 1/7 的 SM）」，并同步检查 §6.3 表格与 [C13]/[C15] 中涉及 GPC 的表述｜修复：｜复验：
- [重要·技术] §6.3 表格与 [C13]（`H100（80GB）的全部 profile 如下`、`[C13] …（六档 profile 与实例数 …）`）｜引文依据：官方 Supported MIG Profiles 表 H100 80GB 共 7 行——MIG 1g.10gb(7)、1g.10gb+me(1)、1g.20gb(4)、2g.20gb(3)、3g.40gb(2)、4g.40gb(1)、7g.80gb(1)；表内逐行给出「Fraction of Memory / Fraction of SMs / L2 Cache Size / Number of Instances Available」｜问题：页面自称列「全部 profile」却漏掉官方 `1g.10gb+me`，且 [C13] 写作「六档」，与官方 7 档不符，属对官方材料完整性的事实性错误｜修复要求：补上 `1g.10gb+me`（10GB，每卡 1 实例）一行，或把「全部」改为「主要」并同步 [C13] 的「六档」表述｜修复：｜复验：
- [重要·技术] 来源章节 [C2] 与 [C6][C7][C8][C9]（`Hopper Tuning Guide §4.1.1`、`§4.1.1–§4.1.3`）｜引文依据：Hopper Tuning Guide 实际编号为「1.4.1.1. Occupancy」「1.4.1.2. Tensor Memory Accelerator」「1.4.1.3. Thread Block Clusters」（无 §4.x 层级）；同名内容（228KB/227KB shared memory、TMA、cluster 上限 8/opt-in 16、warp specialization）均落在 §1.4.1.x｜问题：正文标注的章节号在来源中不存在，按该位置无法定位核对｜修复要求：把 [C2] 改为「Hopper Tuning Guide §1.4.1.1 Occupancy」，[C6][C7][C8][C9] 改为「§1.4.1.1–§1.4.1.3」｜修复：｜复验：
- [轻微·表述] §3.2（`跟你要算的矩阵总共有多大毫无关系`）｜引文依据：不适用｜问题：直接使用第二人称「你」称呼读者，违反 style-guide §12「不直接使用第二人称称呼读者」与 check.md §2.1(12) 会话指代约束｜修复要求：改为「与所算矩阵的总规模无关」等无人称表述｜修复：｜复验：
- [轻微·表述] §2 正文（`现在说出本章最重要的事实：`）｜引文依据：不适用｜问题：元话语＋旁白式临场评价，向读者发号施令而非陈述事实｜修复要求：删除该引导句，直接给出「一个 CTA 被放到 SM 上之后会运行到完成才释放」的陈述｜修复：｜复验：
- [轻微·表述] §2/§3/§4/§5 章末过渡（`贯穿例子推进：`，共 4 处，行 256/326/391/453）｜引文依据：不适用｜问题：以同一固定句式反复引导过渡，属 style-guide §8「不使用固定句式」明令避免的写法｜修复要求：改写为各自独立的过渡句，不重复同一前缀｜修复：｜复验：
- [轻微·表述] §1/§3/§5/§6（`真正需要记住的`、`全页最关键的结构性结论`、`逐次启动的占比就刺眼了`、`这正是它既强大又危险的原因`／`因此它既强大又危险`、`本页的任务到此为止`、`千军万马`、`是日常`、`值得审视`）｜引文依据：不适用｜问题：调试叙事与临场评价、修辞性修饰，属 check.md §2.1(12) 排除项｜修复要求：删去「刺眼」「既强大又危险」「千军万马」「是日常」「值得审视」「到此为止」等评价性措辞与「全页最关键」这类临场判断，改为中性陈述｜修复：｜复验：
- [轻微·表述] §4 章末（`小 kernel 天然吃不满硬件，这是它「单位效率低」的另一层原因`）｜引文依据：不适用｜问题：无来源支持的一般性判断被写成结论（未标注为推断）｜修复要求：降级为标注推断（如「由此可推」），或补上支持该判断的来源；否则删除｜修复：｜复验：
- [轻微·技术] §5 正文（`CUDA Graphs 章节定性指出「当 kernel 又多又小时，逐次启动的开销占比会变得显著」[C10][N8]`）｜引文依据：以「」引号形式给出的整句在 CUDA C++ Programming Guide 的 CUDA Graphs 章节中未能定位到；目前能找到的最接近表述来自 NVIDIA 开发者博客「Employing CUDA Graphs in a Dynamic Environment」——"When these kernels are many and short-lived, launch overhead can sometimes become a problem. CUDA Graphs provide a way to reduce this overhead."（属博客而非编程指南章节）｜问题：引号句以权威原文口吻呈现，但来源位置不可复核（且与所指「章节」不一致）｜修复要求：给出可定位的原文句子与出处（章节或博客链接）；若只能转述，去掉引号并写明「据 NVIDIA 开发者博客，大意是…」｜修复：｜复验：
- [轻微·格式] 全文正文引用（如 `[C5][N1]`、`[C11]`）｜引文依据：不适用｜问题：未按 style-guide §6「正文使用 `<sup>[Cx]</sup>` 上标引用」书写，正文内 0 处 `<sup>`，全部为纯文本方括号｜修复要求：把正文中的来源引用改为 `<sup>[Cx]</sup>` 形式｜修复：｜复验：
- [轻微·格式] 页首 `blockquote.meta`（行 162–164，「主要依据」）｜引文依据：不适用｜问题：位置不符合 style-guide §2 固定顺序（应为 1 reading-time → 2 blockquote.meta → 3 引言 → 4 learning-goals → 5 misconceptions → 6 正文）；本页置于 misconceptions 之后、正文之前｜修复要求：将「主要依据」blockquote 移到 reading-time 之后、引言之前｜修复：｜复验：
- [轻微·格式] 来源章节「外部数字与实验条件（N）」与「主要依据」（`[N1]–[N8] 编号在文末… 逐条对应`）｜引文依据：不适用｜问题：全文无 `[N7]` 的定义也无 `[N7]` 的引用，编号出现断档，与「[N1]–[N8] 逐条对应」的说法不符｜修复要求：补上 [N7] 对应条目并补正文引用，或重排为连续的 [N1]–[N7] 并同步「主要依据」中的区间｜修复：｜复验：
- [轻微·格式] §3 正文 GEMM 公式（`$C[i][j] = \sum_{k=1}^{K} A[i][k] \cdot B[k][j]$`）｜引文依据：不适用｜问题：未按 style-guide §11「公式后紧跟 `<ul>` 逐项定义每个符号」给出符号表（$M,K,N,i,j,k$ 仅在行文中带过）｜修复要求：在公式后补一个 `<ul>`，逐项说明 $A,B,C$ 的形状与 $i,j,k$ 的取值范围｜修复：｜复验：
- [轻微·格式] §3 各构造示例标签（`构造示例。`、`教学构造数字`、`教学示意`、`教学解释`）｜引文依据：不适用｜问题：style-guide §4 规定示例固定标记为「计算示例」「代码示例」或「构造数据」，本页使用了规范外的多种标签写法｜修复要求：统一为规范给定的三种标签之一（本页数据宜标为「构造数据」）｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 11
- 处置：有问题，需修复。规范所限定的核心结论（硬件层级、CTA 跑完才释放、tile 尺寸由容量决定而非输入长度、四种共享机制无运行期细粒度重分配）经来源核对均成立，故不判停止发布；但 §6.3 关于计算切片与 GPC 的硬件映射为无来源支持且与官方表述矛盾的机制陈述，属必须关闭的阻断项；两条重要项（MIG profile 完备性、Hopper Tuning Guide 章节号）修复后需重核来源。表述维度问题集中在第二人称、元话语、固定句式与临场评价，需一并改写。
