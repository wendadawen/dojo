<!-- review-meta
round: 10
page: wiki/dualpath/index.html
reviewed_content_sha256: eaade61c893e6a55
-->
# DualPath 审查记录（第 10 轮）

- 页面版本：index.html 34016f065a8bdbaefc3761237bca722d0ce29b01；overview.html eadae37261a492477407fa0ec2fd7edabcaf177f
- 论文版本：arXiv:2602.21548v2（2026-02-26，cs.DC）
- 审查时间：2026-09-14 16:58
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 已完整阅读章节：核心问题；1. agentic 推理的存储 I/O 瓶颈由三因素叠加；2. DualPath 的双路径数据流与块布局；3. 双路径不引入新瓶颈的 P/D 区间；4. CNIC-centric 流量管理；5. Adaptive Request Scheduler：inter-engine 与 intra-engine 调度；6. 实验：offline、online、消融、负载均衡、大规模；7. 方法评价；来源与范围说明（含全部折叠块、图注与页内链接）
- 机械验证：`python3 .dojo/scripts/validate.py wiki/dualpath/index.html` → validation ok；dojo:topics 三项均在 AGENTS.md 固定大类内，dojo:tag「推理系统」在 ALLOWED_TAGS 内；7 个前置概念页（moe-serving、standard-attention、deepseek-moe、mla、dsa、mqa-gqa、gpu-communication）与其锚点均存在；overview↔index 双向链接存在；alt 属性无 `$...$`。
- 回源核对（有原文片段）：摘要 1.87×/1.96×、作者 13 人与单位、21 项机制与实验数字（§4.1 步骤 (1)-(9) 与 (3-7)/(3-5) 重复 n_layer 次、(Label 8 and 9 in 4(a))、§4.2 记号与 eq.(1)-(9)、§5.2 的 5-7 μs 与约 1 μs、§A.1 的 qos_max_vls 4 / qos_high_limit 240 / vlarb 权重、§A.4 的 α=3s、β=5s、compute quota 300 ms、§7.2 testbed、Table 2 的 6 列三行、Table 3 的 3167s/3201s 与 1.739/1.228/0.039s、§8.2 的 69 GB→681 GB 与 r³）、Fig.3 的黄框「GPU Compute: 28.8x / PCIe Bandwidth: 2.0x / GPU Memory: 2.4x」、Fig.13 的 1.528/1.184 标注、Fig.1 的 40%/80%/100% 标注、Fig.4(a)(b) 各步骤编号、Fig.7 的 Top: DS 27B / Middle: DS 660B / Bottom: Qwen 32B、参考文献 [39]（PDF 第 998 行 "Andre Richter … 2016. Resolving Performance Interference in SR-IOV Setups with PCIe Quality-of-Service Extensions"）。

## 问题

- [阻断·技术] §1 本章问题解答（第 170 行）：把 DSA 说成会压缩 KV-Cache——「MLA 与 DSA 都会显著压缩 KV-Cache（DS V3.2 因 DSA + MLA 双重压缩，ratio 仅 13-36 GB/PFLOP）」。论文的归因方向相反，且本页 §1.1 表注（第 138 行）与相邻概念页都写的是相反结论。｜引文依据：论文 §3 "The ratio of DeepSeek-V3.2 is higher than DeepSeek-V3, benefiting from its sparse attention design, lowering computation demands."；Table 1（V3.2 660B 13–36 vs V3 660B 4.8–5.8，V3.2 反而更高）；本页第 138 行「V3.2 因 sparse attention 进一步降低计算量，比值反而比 V3 高」；本仓 wiki/dsa/index.html「DSA 不省 KV cache…每个查询选中的 k 个位置各不相同，全部历史 KV 条目必须保留…V3.2 省显存的功劳属于 MLA 的低秩压缩，不属于 DSA」。｜修复要求：删除「DSA 压缩 KV-Cache」「DSA + MLA 双重压缩」及由此得出的「ratio 仅 13-36」因果，改为「MLA 压缩 KV-Cache；DSA 降低计算量，故 V3.2 的 cache-compute ratio 高于 V3」，并与 §1.1 表注保持一致。｜修复：｜复验：

- [重要·技术] §3.2 四个不等式段（第 265、269 行）：把不相干的图步骤编号写成了承担该向流量的步骤，与原文 Fig.4 以及本页 §2.2/§2.3 自定的步骤编号均冲突。F-4 写「DE 节点把 Layer Block 从 DE buffer 经 CNIC 经 RDMA 写入 PE 节点 CNIC（DE path 步骤 (5)）」；F-6 写「DE 节点从 PE 节点 CNIC 经 RDMA 收 Layer Block（PE path 步骤 8）」。｜引文依据：Fig.4(b) 中 (4) 才是 DE CNIC→PE CNIC 的 RDMA 箭头、(5) 是 PE CNIC→PE GPU；Fig.4(a) 中 (6) 才是 PE CNIC→DE CNIC 的 RDMA，(8) 是 DE DRAM→DE CNIC（论文正文 "The DE first allocates HBM and performs host-to-device (H2D) transfers (Label 8 and 9 in 4(a))" 把 8/9 定位为 DE 侧 H2D）；本页第 192 行已把 (6) 定义为 RDMA、把 (8)(9) 定义为 DE 侧一次性 H2D。｜修复要求：F-4 的 DE path 标号改为 (4)；F-6 改述为「DE CNIC 从 DE DRAM 读回 Layer Block（DE 侧一次性 H2D 的前半段，PE path 步骤 8）」；同时核对 F-8 的「DE 节点从 DE buffer 经 CNIC 把 Layer Block 喂给 DE GPU（PE path 步骤 7/9 与 DE path 步骤 7）」——步骤 7 实为 CNIC→DRAM 的写入方向。｜修复：｜复验：

- [轻微·技术] overview.html 关键结论第 1 条：写「离线 48P96D vs 2P4D 的 JCT 几乎不变（3167s vs 3201s，agent 数 24×）」，按「A vs B（x vs y）」的位置对应会把 48P96D 配到 3167s、2P4D 配到 3201s，与 index.html §6.8「2P4D…JCT 3167s → 48P96D…JCT 3201s」及论文方向相反。｜引文依据：论文 §7.6 "scaling from 2P4D (2K agents) to 48P96D (48K agents) achieves near-linear speedup with comparable JCT (3,167s vs. 3,201s)"；index.html 第 654 行。｜修复要求：改为「离线 2P4D vs 48P96D 的 JCT 几乎不变（3167s vs 3201s）」，使标签顺序与数字顺序一致。｜修复：｜复验：

- [轻微·表述] §1.1 表 1 注（第 138 行）出现审查流程术语「引用依据取 V3.2 660B 的 22 GB/PFLOP」。「引用依据」是审查记录字段名，不是面向读者的表述。｜引文依据：不适用。｜修复要求：改为「正文代表数字取 V3.2 660B 的 22 GB/PFLOP（论文 §3 正文；Table 1 单独列 13–36 GB/PFLOP，16K–64K 区间）」。｜修复：｜复验：

- [轻微·技术] §6.3（第 579、581 行）：「Basic 在 append 3× 时相对 DualPath 的加速比收窄（读 Fig.9 估算约 1.85×）」「DualPath 相对 Basic 的优势收窄」。按 img-08.png 左子图逐柱像素测量（JCT=0 在 y=328、2000 线 y=227.5、4000 线 y=128）得 basic/ours：append 缩放 x1≈1.83×、x1.5≈1.94×、x2≈2.00×、x3≈1.91×——比值并未随 append 变长单调收窄（x3 高于 x1），收窄的是绝对 JCT 差距。｜引文依据：论文 §7.3 "with append length increases, Basic performance gradually approaches DualPath and Oracle"，同段 "DualPath achieves 1.82-1.99× speedup at different append scales"。｜修复要求：把「加速比收窄」改为「绝对 JCT 差距收窄」，并把「约 1.85×」按读图改为「约 1.9×」。｜修复：｜复验：

- [轻微·技术] §6.2（第 572 行）：「DS 660B 离线实验最高（含 64K MAL 配置）1.87× over Basic 的加速意味着，把 1024 条…trajectory…从 T 缩短到 T/1.87」。1.87× 出自 64K MAL 的 2048/4096 agents 配置，1024 agents 同一面板读图约 1.84×，正文未说明 1024 的来源。｜引文依据：论文 §7.3 "DualPath benefits more from larger batch sizes and longer MALs"；Fig.7 中排 64K 面板逐柱测量（1024 agents：basic≈5365s / ours≈2917s ≈1.84×；2048 agents：basic≈10052s / ours≈5365s ≈1.87×）。｜修复要求：把加速比口径与配置对齐——把「1024 条」改为「该 64K MAL 配置（2048 agents）的 trajectory」，或改用 1024 agents 对应的 1.84×。｜修复：｜复验：

- [轻微·技术] §2.2（第 194 行）「关键时序」三段式：「第 k 层的 KV-Cache 加载与第 k 层 attention 重叠；第 k 层的 attention 与第 k 层 FFN 重叠；第 k 层 FFN 与第 k+1 层的 KV-Cache 加载重叠」。论文只给泛述，未给该三段式；其中「第 k 层的 attention 与第 k 层 FFN 重叠」是同层内前后依赖的两个算子，按字面不能重叠，易生误解。｜引文依据：论文 §4.1 "During the prefill forward pass, transfers overlap with computation."（同段仅有 "(3-7) repeats n_layer times"）。｜修复要求：改为论文口径的表述（如「第 k 层的 KV-Cache 加载/传输与相邻层的计算重叠」），或把该三段式明确标注为推断并给出依据。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 5
- 处置：修复（阻断与重要问题须关闭后重跑 validate.py 并进入下一轮）
