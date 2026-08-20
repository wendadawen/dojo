# PagedAttention 审查记录（第 1 轮）

- 页面版本：index.html 工作树哈希 `7b9bb81874691c840b9e2128bcb3defe1ebb1990`
- 审查时间：2026-08-20 11:32
- 审查者：独立子代理（reviewer-paged-attention-r1）
- 已完整阅读章节：index.html：导语、核心问题（4 题及全部解答折叠块）、1 连续预留的三类浪费（含本章问题）、2 分页机制——块、页表与按需分配（含流程图、A/B 账目折叠块、本章问题）、3 页间不连续——为显存设计，被传输惩罚（含页大小上限折叠块、本章问题）、4 页大小怎么选——一个没有免费午餐的旋钮（含权衡表、本章问题）、来源与范围说明（全部小节）；overview.html：全文（这是什么 / 解决什么问题 / 核心机制 / 关键结论与边界）。

## 问题

- [重要·技术] index.html 第 4 章权衡表「内部碎片与预留 × 页小」列：「浪费小（每请求至多半页）」——「至多半页」是错误上限：内部碎片上限为一整页（当实际 token 数 ≡ 1 mod 页大小时，最后一页仅用 1 槽，浪费页大小−1 槽），「半页」只是均匀假设下的期望值。该表述与正文第 2 章「每请求至多浪费一页<sup>[C6]</sup>」直接矛盾。｜引文依据：vLLM 论文 §4.2 "As all the blocks are filled from left to right and a new physical block is only allocated when all previous blocks are full, vLLM limits all the memory wastes for a request within one block"（上限为一 block，非半 block）｜修复要求：将该格改为「浪费小（平均约半页，至多一页）」或「浪费小（每请求至多一页）」，与 C6 及第 2 章口径一致｜修复：｜复验：
- [重要·来源] index.html 第 4 章正文「（该文的页大小扫描实验显示：基线系统把页调到最优值 512 仍只有其 93% 的吞吐）<sup>[C8]</sup>」及第 4 章本章问题 2 解答「其页大小扫描实验显示：基线系统把页调到自身最优值 512，吞吐仍只有 Strata-IO 的 93%（命中率低 2.4%）」「高效传输粒度低至 128 字节」「Strata 用 GPU kernel（而非 DMA 拷贝 API）搬数据」——数值 512、93%、2.4%、128 字节及「GPU kernel 而非 DMA 拷贝 API」的机制表述在页面标注来源（Strata §1、§2.2、§3.1）及本轮可读的 Strata TeX 原文（1_intro.tex、1.5_background.tex、2_motivation.tex）中均定位不到；C8 的来源说明未给这些数字单独定位。｜引文依据：不适用（定位不到；2_motivation.tex 相关内容仅有 "varying the KV cache page size from 1 to 1024 on the SGLang framework" 与 "rise by up to $2\times$ and $2.9\times$"，无 512/93%/2.4%；三个 TeX 文件中无 128 字节粒度表述，1_intro.tex 仅有 "employs GPU-assisted data transfer to combat KV cache fragmentation"）｜修复要求：为 512、93%、2.4%、128 字节及 kernel 机制表述补充 Strata 论文内的具体定位（章节/图表号）并写入来源说明的 N/C 条目；定位不到则删除这些数字或降级为明确标注的推断｜修复：｜复验：
- [重要·来源] index.html 第 4 章正文「SGLang 为了前缀匹配粒度取到 1（牺牲一点管理开销），TensorRT-LLM 取 32 兼顾传输」——对两个引擎设计动机的归因在来源中无记载，Strata §2.2 仅列举数值无动机陈述，属把推断写成来源结论。｜引文依据：Strata 1.5_background.tex §2.2 "Typical page sizes are small—e.g., 32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang—where each token may span from tens of kilobytes to several megabytes."（仅数值，无各引擎动机）｜修复要求：删除动机归因，或改为明确标注的推断句式（如「从取值看，SGLang 取 1 可能是为前缀匹配粒度……」，并在来源说明标注为推断）｜修复：｜复验：
- [轻微·来源] index.html 来源与范围说明「核心论断与来源」C7 条目：「每 token 十几 KB 到几 MB」——Strata §2.2 原文为 "tens of kilobytes"（几十 KB），「十几 KB」（11–19 KB）与原文不符，且与正文第 3 章「从几十 KB 到几 MB」自相矛盾。｜引文依据：Strata 1.5_background.tex §2.2 "where each token may span from tens of kilobytes to several megabytes"｜修复要求：来源说明 C7 中「十几 KB」改为「几十 KB」｜修复：｜复验：
- [轻微·来源] index.html 第 1 章「其余全部是上述三类浪费<sup>[C2, N1]</sup>」——论文 Figure 2 的浪费类别为 Reservation / Internal frag. / External frag. **& Others**，「其余全部是三类」与图的口径有偏差（存在未归入三类的其他浪费）。｜引文依据：vLLM 论文 Fig. 2 图例 "External frag. & Others"（N1 所引即此图）｜修复要求：改为「其余是预留、内部碎片、外部碎片等浪费」或在句中注明含其他项｜修复：｜复验：
- [轻微·来源] index.html 来源与范围说明「核心论断与来源」：C1–C6 标注章节为 vLLM 论文「§2、§3、§4」，但 C3 的类比三元组与 C4 的原文位于论文 §1（Introduction），§2（Background：生成流程与调度）不含这些论断，定位不准。｜引文依据：vLLM 论文 §1 "one can think of blocks as pages, tokens as bytes, and requests as processes"；§1 "This design alleviates internal fragmentation by using relatively small blocks and allocating them on demand. Moreover, it eliminates external fragmentation as all blocks have the same size. Finally, it enables memory sharing at the granularity of a block"｜修复要求：C1–C6 的章节标注范围改为「§1、§3、§4」（或按各条目分别标注）｜修复：｜复验：
- [轻微·可读性] index.html 第 3 章「40 GB 显存的利用率从 30% 提到 96% 相当于可用缓存接近翻三倍」——96/30≈3.2 倍，「翻三倍」语义歧义（可理解为 ×3 或 ×4），且 3.2 已超过 ×3，「接近」不准确。｜引文依据：不适用｜修复要求：改为「约为原来的 3.2 倍」或「提高到三倍以上」｜修复：｜复验：
- [轻微·格式] index.html 来源与范围说明「核心公式与来源」F1 条目：「页数$\times$页大小−token 数」中的减号「−」（U+2212）为 Unicode 数学字符，直接出现在 $...$ 之外，违反「数学符号全部由 KaTeX 渲染」。｜引文依据：不适用｜修复要求：将整式放入 math 环境，如 $\text{页数}\times\text{页大小}-\text{token 数}$｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 5
- 处置：修复

已核对通过的关键项（备考）：C1 三类浪费、C2 20.4%–38.2%、C3 类比三元组、C4 按需分配/同大小页/页级共享、C6 至多一页均与 vLLM 论文原文一致；C7 页大小 32/16/1（Strata §2.2）、C8 分页致数据碎片化（Strata §1 "paging causes \textit{data} fragmentation"）、N4 8192 token 约 22% PCIe 5.0 / GH200 约 5%（Strata §3.1 "approximately 22% of the theoretical PCIe 5.0 bandwidth"、"falling to as low as ~5% on systems like NVIDIA's Grace-Hopper platform"）一致；A/B 例子全部数字可复算（1948/12、2560/600≈23%、⌈100/16⌉=7、112−100=12、600/624≈96%）；核心问题 4 题与每章本章问题均含解答折叠块且答案独立可读；链接 ../kv-cache、../prefix-caching、../strata 及 KaTeX 本地资源均存在；overview 与 index 相互链接。


## Round 1 修复记录

| 编号 | 问题 | 修复 | 复验 |
|---|---|---|---|
| 重要 #1 | 权衡表"每请求至多半页"与 C6 矛盾 | 改"至多一页；平均约半页为推断" | 验证 |
| 重要 #2 | 512/93%/2.4%/128 字节缺来源定位 | 正文补 §5.3.2 与 §4.2 定位；来源说明新增 N5/N6 条目 | 验证 |
| 重要 #3 | 引擎动机归因无据 | 改写为"从取值推断……此为推断，非来源结论" | 验证 |
| 轻微 #4 | "十几 KB"与原文 tens of KB 不符 | 改"几十 KB" | 验证 |
| 轻微 #5 | "其余全部是三类浪费"与图例含 Others 不符 | 改"等浪费（论文图例含 Others 项）" | 验证 |
| 轻微 #6 | C1–C6 章节标注缺 §1 | 改"§1、§3、§4" | 验证 |
| 轻微 #7 | "接近翻三倍"歧义 | 改"约为原来的 3.2 倍" | 验证 |
| 轻微 #8 | F1 裸 − | 整式入 math 环境 | 验证 |
