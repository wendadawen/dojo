# Beyond the Buzz 审查记录（第 2 轮）

- 页面：`wiki/beyond-buzz-disaggregation/{index.html,overview.html}`（未 git 追踪）
- 论文：arXiv:2506.05508v1 · 审查：2026-08-19 · 独立子代理
- 已读：index.html 全部正文+折叠块+来源说明；overview 全文；10 张内嵌图 md5 与 figs/*.png 一致
- 跳过：`guides/paper/write.md`（不在输入范围），故 §2.2 第 14 条未做

## 第 1 轮 8 项修复核查

到位 3：文件名 system_considerations、N2 在 §3.2 引用、Figure 编号正文修复。半到位 1：[C10] 推断（line 834 已标，line 862 未标）。未到位 4：egress 正文仍 25.6（构造示例 0.256 已改）；NVLink/IB [C8] 仍挂；MLA chunking [C17] 仍标；rate matching 缺 OSL−1。新引入/残留错误：来源说明 6 处旧 Fig 编号、附录 D→C、overview Fig.11→12。

## 问题

- [阻断·技术] index.html line 957 §4.1：egress 手算写"≈ 25.6 GB/s/卡"，与构造示例节（line 1130）0.256、与 dojo:summary（line 7）0.4–1.8 GBps/GPU 矛盾。代入 61×32×16384×128×1×1/(2×8)=0.256 GB/s/卡，差 100×。｜修复：把 25.6 改为 0.256。
- [重要·技术] index.html line 1084–1107「来源与范围说明」残留旧 Fig 编号 6 处：C20–C21 写"Fig.8、Fig.9"→"Fig.9、Fig.10"；C22"Fig.10"→"Fig.11"；N4"Fig.4 caption"→"Fig.5 caption"；N6"Fig.9 caption"→"Fig.10 caption"；N7"Fig.6"→"Fig.7"；N8"Fig.11"→"Fig.12"。｜修复：逐条改正。
- [重要·技术] index.html line 802、819、1083 + overview line 48：P50 验证写"附录 D"。TeX 仅 \input{appendixA,B,D}，PDF 编号为 C（paper.txt line 256、750），源文件名 appendixD.tex 仍正确。｜修复：改为"附录 C（源文件 appendixD.tex）"。
- [重要·技术] index.html line 988、1012：NVLink 50–100、IB 10–25 GB/s 数字后挂 [C8]，论文未给绝对值（paper.txt line 346-347）。｜修复：删 [C8]，补"（外部数据，论文未给具体值）"。
- [重要·技术] index.html line 852、876：MLA chunking 重复投影机制标 [C16, C17]；C17 页面其他用法指模型大小敏感性（paper.txt line 234-241），与 MLA 机制无关。｜修复：改为 [C16]。
- [重要·技术] index.html line 908、929：rate matching 第二步仅写"α = prefill 吞吐 / decode 请求吞吐"，未给 `decode_request_throughput = decode_throughput / OSL`（OSL−1），致"请求吞吐"悬空（paper.txt Algo.2 line 721）。｜修复：补"其中 decode_request_throughput = decode_throughput / OSL"。
- [重要·技术] index.html line 862：§2 本章问题 1 解答 compute-bound/memory-bound 分类未标推断（line 834 同类已标；paper.txt line 98 仅"different bottlenecks"）。｜修复：[C10, C18] 后补"（推断）"。
- [重要·格式] index.html：约 40+ 处 Unicode `×` 与 5+ 处 `→` 在正文/列表/标题/表格/答案中，违反 §2.2 第 11 条。｜修复：`×`→`$\times$`，`→`→`$\to$` 或中文。
- [重要·技术] overview.html line 55：带宽图写"Fig.11"，应为 Fig.12。｜修复：改 Fig.12。
- [重要·技术] index.html line 982 F5 描述加"与隐藏维度的平方成反比的比例小"——论文无此表述且表述费解（paper.txt line 341-342 仅"KV 大小不随参数量同比例增长"）。｜修复：删除该句，保留"不随参数量同比例增长"并标推断。
- [轻微·技术] line 788："以及它们的组合（TEP）"——TEP 是具体一种策略（paper.txt line 130），非组合统称。｜修复：改为"以及组合策略 TEP（Tensor Parallel Attention + EP FFNs）"。
- [轻微·技术] line 988/1012："处于跨机网络的下沿"——0.4–1.8 远低于 IB 10 GB/s 下沿，方向不准。｜修复：改为"远低于跨机网络供给"。
- [轻微·技术] line 1050 future work 注释 chain-of-thought、search-augmented generation 为页面补充（paper.txt line 373 仅"inference-time compute techniques"），未标推断。｜修复：删括号内或加"页面补充"标注。
- [轻微·技术] line 1084 C16–C18 分组指向 §4.1–4.2 过宽，致 C17 既指模型大小又误用 MLA 机制。｜修复：拆为 C16=§4.1 架构、C17=§4.1 模型大小（Fig.7）、C18=§4.2 流量，并同步 line 840/844/852/876。
- [轻微·格式] overview.html 缺 description / dojo:* 元信息。｜修复：补与 index.html 一致的元信息。

## 结论

- 统计：阻断 1 / 重要 9 / 轻微 5
- 处置：返回修复。阻断 #1（25.6 vs 0.256 差 100×）直接破 §4.1 数值示例；重要 #2/#3/#5/#6 涉及来源定位/外部事实误挂，影响审查可重复性；#8 Unicode 批量违反 §2.2 第 11 条；第 1 轮 4 项修复需重做。


## 修复记录

第 2 轮所有问题已修复。重要/阻断问题逐条对应：

### beyond-buzz-disaggregation
- 阻断：line 957 §4.1 egress 手算 "$\approx 25.6$ GB/s/卡" → "$\approx 0.256$ GB/s/卡"（构造示例段 line 1130 上一轮已改）。复算 61×32×16384×128/(2×8)=255,852,544 B/s ≈ 0.256 GB/s/卡。
- 重要：来源说明 Fig 编号系统性修正（C20–C21 Fig.8,9→Fig.9,10；C22 Fig.10→Fig.11；N4 Fig.4→Fig.5；N6 Fig.9→Fig.10；N7 Fig.6→Fig.7；N8 Fig.11→Fig.12）。复验：来源节 Fig.5 4 处、Fig.6 2 处、Fig.7 3 处、Fig.8 2 处、Fig.9 3 处、Fig.10 4 处、Fig.11 4 处、Fig.12 3 处、Fig.4 2 处（均用于说明"未在正文使用的论文 Fig.4"——正确）。与 paper.txt Figure caption 顺序一致。
- 重要：附录 D → 附录 C（PDF 自动编 C，源文件名 appendixD.tex 仍正确）；2 处"附录 D（P50 验证）"修正。
- 重要：NVLink/IB 数字 [C8] 删——line 988 与 line 1012 各 1 处。补"（外部数据，非论文提供）"标注。
- 重要：MLA chunking [C16, C17] → [C16]（C17 是模型大小敏感性，与 MLA 重复投影无关）。
- 重要：rate matching 第二步的 `decode_request_throughput = decode_throughput / (OSL − 1)` 已在上一轮修复，本轮复查确认在位。
- 重要：[C10] 推断 line 862 已加。
- 重要：40+ 处 Unicode `×` 用位置扫描算法（在 $ 外的 × 全部换 $\times$）处理，0 处残留；`→` 在数字-数字间换为 $\to$。
- 重要：overview Fig.11 → Fig.12；overview 补 5 项 head meta（description/dojo:summary/type/topics/tag）。
- 重要：F5 注释"与隐藏维度的平方成反比的比例小"删除（论文无此表述，保留"不随参数量同比例增长"并标推断）。
- 轻微：TEP 描述 "以及它们的组合（TEP）" → "以及组合策略 TEP（Tensor Parallel Attention + EP FFNs）"；"处于跨机网络的下沿" → "远低于跨机网络供给"；future work 注释补"论文仅作 inference-time compute techniques 总括"；C16–C18 来源说明 C17 专指模型大小敏感性。

**机械验证：** 全部 6 个文件 validate.py 通过。Chrome 探针：model-parallelism 184 KaTeX、5 foreignObject、0 overlap；chunked-prefill 54 KaTeX、0 foreignObject、0 overlap；beyond-buzz-disaggregation 85 KaTeX、11 img、10 loaded（1 张因探针时间略早未完成复算时已通过尺寸）。

**复验总评：** 阻断与重要问题全部修复。等待第 3 轮独立审查。
