# Strata 审查记录（第 2 轮）

- 页面版本：index.html 1078083 字节，SHA1 `32082f3627a001221b9ebe8b33597ca1571debae`；overview.html 6194 字节，SHA1 `df924a6e26b594b933a124429b6077c40edf6f5f`
- 论文版本：arXiv:2508.18572v2 TeX 源码（`/tmp/strata-research/src/`，2025-08-27 提交）
- 审查时间：2026-08-19
- 审查者：独立子代理（reviewer-strata-2）
- 已完整阅读章节：
  - 质检规范 `guides/paper/check.md`
  - 论文 TeX 全文：`0_main.tex`、`1_intro.tex`、`1.5_background.tex`、`2_motivation.tex`、`3_design.tex`、`5_eval.tex`、`6_related.tex`、`7_discussion.tex`
  - 页面：`wiki/strata/overview.html` 全文
  - 页面：`wiki/strata/index.html` 全文（CSS/JS 块除外），包括所有 14 张 base64 嵌入图（Figure 1–14）抽取后逐张核对
  - 外部核对：USENIX OSDI'26 technical sessions 页面（页面明确引用的来源）——确认 Strata 在 OSDI 2026 Track 1 "KV Cache and Long Context" 场次，Session Chair Juncheng Yang
- 工具校验：`.dojo/scripts/validate.py wiki/strata/index.html` → `validation ok`；6 个被引前置概念页（`kv-cache` / `paged-attention` / `prefix-caching` / `standard-attention` / `gpu-execution-model` / `gpu-communication`）均存在

## 问题

- [重要·技术] index.html 行 1148、1150（Figure 8 引导句 + 图注）：图 8 实际轴向是 **纵轴 Average TTFT (s)、横轴 Throughput (token/s)**（已从抽取的 Figure 8 原图核对：12 个子图均以 `Average TTFT (s)` 为 y 轴、`Throughput (token/s)` 为 x 轴）。但页面写"纵轴吞吐（output tokens/s）、横轴平均 TTFT"——两轴标签直接互换。引导句"曲线越靠右上越好——同 TTFT 下吞吐更高，或同吞吐下 TTFT 更低"的方向也错：图 8 是 TTFT 对 throughput 的标准 latency-vs-throughput 图，好区为右下（高吞吐、低 TTFT），不是右上。**引文依据**（Figure 8 子图 y 轴标签实测）："Average TTFT (s)"；x 轴标签实测："Throughput (token/s)"。**修复要求**：把行 1150 图注改为"纵轴 Average TTFT (s)、横轴 Throughput (token/s)"；行 1148 引导句"曲线越靠右上越好"改为"曲线越靠右下越好（同 TTFT 下吞吐更高，或同吞吐下 TTFT 更低）"。｜修复：｜复验：
- [重要·技术] index.html 行 1229（§5 章问题解答 2 的答案）：该答案写"cache 命中率比页 32 低 2.4%"。论文 §5.3.2 L154 原文为"achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate"——2.4% 的对照对象是 Strata-IO，不是 SGLang-HiCache 的页 32。同一页面的 §5.4 caption（行 1192）已经正确标注"论文未明说 2.4% 相对哪个具体对照配置，数字本身见 §5.3.2"；本句却把它锁定为"比页 32 低"，与本页面的另一处自我标注矛盾。**引文依据**（论文 §5.3.2 L154）："achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate"——比较对象是 Strata-IO。**修复要求**：行 1229 改为"cache 命中率比 Strata-IO 低 2.4%"，或与行 1192 的"未明说对照配置"措辞保持一致。｜修复：｜复验：
- [轻微·格式] index.html 多处 + overview.html 行 52、38：Unicode 数学字符直接出现在正文 / 列表 / 表格中，未走 KaTeX。`×` 出现位置：index 行 7（meta description "2 block × 1024 thread"）、行 807、870、970、1136（"8×H200"）、1222、1254、1273；overview 行 52。`→` 出现位置：index 行 738（"页 1→1024"）、759（"40→150 GB/s"）、856、878、918（"CPU→GPU 加载"/"GPU→CPU 备份"）、940、961、991、1004、1102（figcaption 中 in-queue → in-flight → standard）、1192、1273；overview 行 38（"完整解析 →"，属导航可豁免）。`↔` 出现位置：index 行 766（"GPU↔CPU"）。违反 check.md 第 2.2.11 条"数学符号全部由 KaTeX 渲染……无 Unicode 数学字符直接出现"。**引文依据**：check.md 行 63 "公式书写：数学符号全部由 KaTeX 渲染，标题、summary、正文、列表和表格中无 Unicode 数学字符直接出现"。**修复要求**：将正文 / 列表 / 表格中的 `×`（非公式 $…$ 内的 `\times`）、`→`、`↔` 全部改为 KaTeX（如 `$\times$` / `$\to$` / `$\leftrightarrow$`）。导航文字（overview 行 38 "完整解析 →"）属 UI 元素，可保留；其它一律改写。｜修复：｜复验：
- [轻微] index.html 行 1282（来源与范围说明「构造示例」段）：写"第 1 章 A/B 例子（请求 A 上限 2048/实际 100、请求 B 上限 512/实际 500）为构造算例，便于手算三类浪费与按页匹配"——但 grep 全文（"2048"、"上限 512"、"实际 500"）在 index.html 正文中均无匹配，本章实际并不存在该 A/B 例子。同时段还提到"第 5 章'理想化配置'"，但全文检索"理想化"也无匹配。这两处是来源说明与正文不同步的残留引用，会让审计来源的读者无法在正文中找到对应构造。**引文依据**：不适用（属页面内部一致性）。**修复要求**：要么在第 1 章补入"A/B 例子"和"三类浪费"的构造算例，要么把行 1282 的两处不存在的引用删除/改写为已存在的构造示例（如"第 1 章 20,000 token 手册容量账"、"第 3 章 2 层 × 4 token 布局算例"）。｜修复：｜复验：
- [轻微] index.html 行 1020（Figure 7 图注）：写"三策略分别由三处蓝色箭头标注（论文 §4.3 图注）"。**蓝色箭头确实在抽取的 Figure 7 原图中存在**（三处箭头分别从 Delay Hit / Balance Batch / Stall Hiding 标签指向 Strata 行的对应位置），但论文 TeX 中 fig:scheduling 的 caption（3_design.tex L92-93）只列了颜色块的含义（orange=miss、green=device hit、purple=host hit、blue=transfer、gray=decoding），并未提及"三处蓝色箭头"或对三策略做标签。**引文依据**（3_design.tex L92-93 caption 原文）："orange blocks indicate prefill batches experiencing cache miss, green indicates cache hit on device, purple indicates cache hit on host memory, blue indicates data transfer, and the one decoding batch is colored in gray"——无箭头描述。**修复要求**：把"（论文 §4.3 图注）"改为"（图中标注）"或"（图内三处蓝色箭头）"，使归属指向图本身而非论文 caption。｜修复：｜复验：
- [轻微] index.html 行 1256：写"论文 §7 Conclusion 的展望段提及未来可能集成专用 on-chip 内存 I/O 加速器<sup>[C27, §7]</sup>"。但本页"来源与范围说明"的 C27 定义（行 1267）为"C27（相关工作定位）见 §6 L2–21"，对应的是 related work 段的相关工作定位论断。§7 的展望论断与 C27 的定义不对应——C 编号映射表与正文用法不一致。**引文依据**（行 1267 来源表）："C27（相关工作定位）见 §6 L2–21"——§6 是 Related Work，§7 是 Conclusion。**修复要求**：把行 1256 的 `[C27, §7]` 改为新的 C 编号（如 `[C28, §7]`，并相应在行 1267 来源表中追加"C28（Conclusion 展望）见 §7 L2"），或将 C27 的定义改为覆盖 §7 段并把 §6 那条改用其他编号。**注意**：行 1262 处的 `[C27]` 引用内容"Strata 在精确缓存前提下保证请求精度不受影响"对应 §6 L9（"Strata does not impact the accuracy of requests"），与 C27 现有定义一致，请勿改动该处。｜修复：｜复验：
- [轻微] index.html 行 1254：写"transient node 匹配阈值 100、loading_bound 比例阈值 100，论文说'硬件/模型相关、可分别 profiling'"。论文 §3.3.2 L162 仅对 loading_bound 阈值说"hardware- and model-dependent and thus can be profiled separately"；对 transient node 阈值（§3.3.1 L115）论文只说"configurable threshold... default threshold of 100 active token matches proved effective"，未说硬件相关。**引文依据**（3_design.tex L115）："we use a configurable threshold... default threshold of 100 active token matches proved effective"——无硬件相关性描述。**修复要求**：把"transient node 匹配阈值 100、loading_bound 比例阈值 100，论文说'硬件/模型相关、可分别 profiling'"改为分别表述，例如"loading_bound 比例阈值 100（论文说硬件/模型相关，可分别 profiling）；transient node 匹配阈值 100（论文给出默认值，未指定硬件相关性）"。｜修复：｜复验：

## 第 1 轮修复抽样验证

> 说明：本轮输入未提供 `research/review-1.md`（按审查规则"禁止读取 wiki/strata/research/"）。本节按任务说明的"§1.2 重写后的连贯性、Figure 7 嵌入后上下文衔接"两个重点，外加独立识别的高风险改动区，抽查 5 项。每项给出当前页面实际状态与验证结论。修复编号占位（#1–#5）便于与第 1 轮编号对位，但本审查者未读到原编号。

- **#1 §1.2 重写后的连贯性（行 810–832）**：§1.2 标题"瓶颈一：分页带来的碎片化传输"，正文从 PagedAttention 页大小 → Little's Law 三杠杆 → 22%/5% 实测 → 引用 Figure 1 与 Figure 3。Figure 1（Qwen-14B on LooGLE stall CDF）实属"瓶颈一→瓶颈二"的过渡证据，论文 §1 L14/L37 同样用法；Figure 3（不同平台带宽利用率）实属"瓶颈一"证据。三杠杆论证链条完整，未出现"页大小可调"以外的隐含变量泄漏。**结论：连贯性 OK。**（唯一瑕疵：行 812 短语"是几 KB 至几十 MB 量级的小传输"对 4 MB 的归属偏宽，但行 806 折叠块已澄清单层片段 128 KB 的来源，整体可读。）

- **#2 Figure 7 嵌入后上下文衔接（行 1018–1022 嵌入、§4.3 行 1030、§4.4 行 1071 反向引用）**：抽取的 Figure 7 原图显示 FIFO 行 = `A0+A1, B0+B1, C+D0, D1+F, G, Decoding`，Strata 行 = `A0+B0, A1+B1, C+F, D0+D1, Decoding, G`——与页面行 1020 caption 完全一致。三策略标签（Delay Hit / Balance Batch / Stall Hiding）以蓝色箭头在图中分别指向 Strata 行的不同位置。§4.3（行 1030 "C+D0 → C+F"）与 §4.4（行 1071 "G PCIe 加载 + Decoding 并行"）的反向引用在图中能找到对应块。**结论：衔接 OK。**（注：caption 中"（论文 §4.3 图注）"的归属已在上面"问题"段列为轻微条目；图本身在文中位置与复用均无误。）

- **#3 Figure 8（端到端 e2e_bench_all）嵌入与图注（行 1146–1150）**：抽取的 Figure 8 原图确为 4 行 × 3 列（4 数据集 × 3 模型）的 Average TTFT vs Throughput 子图。**但发现：页面图注把两轴写反了**（见"问题"段第 1 条）。**结论：图嵌入位置与前后文字衔接 OK，图注存在错误。**

- **#4 核心问题与各章本章问题的解答折叠块完整性与指向**：页面级核心问题 5 条（行 734–768），每条均有解答折叠块且答案末尾明确指出完整论证所在章节（"完整论证见第 1 章" / "第 2、3 章" / "第 4 章" / "第 5 章" / "第 6 章"）。§1–§5 共 5 组本章问题，每组 2–4 条，全部有解答，答案独立可读，与正文结论一致。**结论：完整性与指向性 OK。**

- **#5 来源与范围说明（行 1264–1288）**：大部分引文定位准确（C1–C26、F1–F2、N1–N20 与论文行号能对位），Figure 编号与原图对应表（行 1276）也准确。**但发现两处来源说明与正文不同步**——行 1282 提到的"第 1 章 A/B 例子（请求 A 上限 2048/实际 100、请求 B 上限 512/实际 500）"与"第 5 章'理想化配置'"在正文中均不存在（见"问题"段第 4 条）。**结论：发现 1 项残留引用未清理。**

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 5
- 处置：**修复**（阻断 0、重要 2 条均为局部表述/图注错误，不影响核心机制与数字正确性，修复后可直接进入第 3 轮或发布前自检；本轮新引入或残留的轻微条目可一并清理）

### 本轮修复优先级建议

1. **重要 #1（Figure 8 轴标签互换）**——修即可，改两行字。涉及读者理解图的核心方向。
2. **重要 #2（2.4% 对照对象）**——修即可，改一处字，让行 1229 与行 1192 的自洽措辞一致。
3. **轻微 #3（Unicode 数学字符）**——按 check.md 2.2.11 全面替换，主要工作量在编辑；可不阻塞发布但应清理。
4. **轻微 #4–#7**——来源说明残留、图注归属、C 编号映射、阈值描述范围——皆为小修改，可与 #3 一并处理。

### 阻断/重要 0/2 的边界

第 2 轮未发现阻断问题：核心机制（GPU-assisted I/O、cache-aware 调度、布局解耦）描述与论文 §3 一致；核心数字（5× / 3.75× / 22% / 5% / 74% / 24% / 4× / 50 GB/s / 40→150 GB / 95% / 1TB / 1–2 MB / 128 字节 / 100 阈值 / 2 block × 1024 thread / 50 GB/s 等）已与论文 TeX 逐项核对一致；6 个前置概念页链接均存在；OSDI 2026 归属与 Track 1 "KV Cache and Long Context" 场次信息经 USENIX 官方页面核对属实；`.dojo/scripts/validate.py` 通过。两条重要问题均为图注/对照对象表述错误，不改变任何机制或数字结论。


## Round 2 修复记录

| 编号 | 问题简述 | 修复 | 复验 |
|---|---|---|---|
| 重要 #1 | Figure 8 轴标签写反 | index.html 引导句与图注改 "纵轴 Average TTFT (s)、横轴 Throughput (output tokens/s)"；方向改 "曲线越靠右下越好" | validate.py 通过 |
| 重要 #2 | 1229 行 2.4% 对照对象错位 | 改为 "SGLang-HiCache@512 的 cache 命中率比 Strata-IO（页 1）低 2.4%" | 验证 |
| 轻微 #3 | 多处 Unicode ×、→、↔ 未走 KaTeX | 批量 replace ×→$\times$、→→$\to$、↔→$\leftrightarrow$（body 内，避开 script/style/nav） | 可见 ×=0、→=0 |
| 轻微 #4 | 1282 行 "A/B 例子" 残留引用 | 改为已存在的构造示例 | 验证 |
| 轻微 #5 | 1020 行 Figure 7 图注归属 | "（论文 §4.3 图注）" → "（图中标注）" | 验证 |
| 轻微 #6 | [C27, §7] 与 C27 定义不对应 | 新增 C28（§7 Conclusion），用 [C28]；来源表加 C28 项 | 验证 |
| 轻微 #7 | transient 阈值错引"硬件/模型相关" | 分别描述：loading_bound 阈值 100（论文 §4.3.2 L162 说硬件/模型相关），transient 阈值 100（§4.3.1 L115 给默认值未指定硬件相关性） | 验证 |
| 附 | description meta 含 $5\times$ 等公式（plain text 违规） | description 改纯文本 "5 倍"、"3.75 倍" | validate.py 通过 |

机械验证：validate.py ok；headless Chrome 探针 katex=79、dollar=0、img=15；body 内 unicode 数学字符清零。
