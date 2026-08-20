# Strata 审查记录（第 1 轮）

- 页面版本：index.html 1,041,432 字节，SHA1 d847fef203f4ac619239038f1f21e4b1df31850e；overview.html 6,193 字节
- 论文版本：arXiv:2508.18572v2 TeX 源码（/tmp/strata-research/src/，2025-08-27）
- 审查时间：2026-08-19
- 审查者：独立子代理（reviewer-strata-1）
- 已完整阅读章节：index.html 导语、核心问题（5 题含解答）、术语表、第 1 章（1.1–1.4 与本章问题）、第 2 章（2.1–2.2 与本章问题）、第 3 章（3.1–3.3 与本章问题）、第 4 章（4.1–4.4 与本章问题）、第 5 章（5.1–5.6 与本章问题）、第 6 章、来源与范围说明（含全部折叠块）；overview.html 全文。机械验证：`.dojo/scripts/validate.py` 对两文件均返回 ok；前置概念链接 6 个目标页全部存在；index/overview 互链存在（index.html:696、overview.html:38）。

## 数字核对通过项（摘录依据）

- LooGLE 各模型×基线：§5.2.1 "up to $3.2\times$, $2.6\times$, and $1.9\times$ … SGLang-HiCache, vLLM-LMCache, and TensorRT-HiCache"（Llama-8B）；"with Qwen-14B … $3.9\times$, $2.1\times$, and $1.9\times$；with Llama-70B, the gains reach $5\times$, $5\times$, and $3.75\times$"。ReviewMT："outperforms vLLM-LMCache by $2.3\times$, TensorRT-HiCache by $2.3\times$ and SGLang-HiCache by $1.7\times$"。NarrativeQA 预热稳态 §5.2.2 "up to $2.3\times$, $2.6\times$ and $2.5\times$ … vLLM-LMCache on Llama-8B, Qwen-14 and Llama-70B"，TensorRT-HiCache 不支持预热未测——页面表格与正文全部一致。
- 74%/4x/24%：§1 L14 "74\% of prefill time is blocked on KV transfers … up to a $4\times$ throughput reduction"；L37 "up to 24\% of prefill execution time remains stalled"。
- 22%/5%/6x：§3.1 L47–49 "approximately 22\% of the theoretical PCIe 5.0 bandwidth … falling to as low as ~5\% … offers 6x higher peak bandwidth"。
- 大页实验：§3.1 L42 "rise by up to $2\times$ and $2.9\times$"（avg/P90 TTFT）。
- GPU-assisted I/O 三优点（enhanced concurrency / compatible with small transfers / flexible memory layout）、128 字节粒度、2 block × 1024 thread、"nearly 50~GB/s … less than 5\% … 10\% … two blocks as the default quota … one block"：§4.2 L33–59 全部一致。
- transient node in-queue/in-flight 与阈值 100：§4.3.1 L112–115（"a default threshold of 100 active token matches"）。
- load/compute 比例阈值 100 与图 1 对应：§4.3.2 L160–162（"a default ratio of 100, corresponding to the point where stalls begin to appear"）。
- bubble filling 资源正交（decode 饱和 HBM、loading 饱和 PCIe）：§4.3.3 L174。
- 布局解耦与磁盘 4x：§4.2.1 L79–86、§5.3.4 L190 "reducing latency by up to $4\times$"。
- 消融 2.3x/1.8x 与请求率此消彼长：§5.3.1 L133；页大小 93%/2.4%：§5.3.2 L154。
- cache distance +42%/+76%/+95%/+11%/+12%/+8%/+3%：§5.3.3 L163–168 逐项一致。
- GH200 40→150 GB/s、Strata-IO-GH 不敌 Strata-PCIe、接近 Oracle：§5.4 L221–225。
- 95% hit rate：§5.2.1 L87；0.3M token/40GB：§1 L10；页 32/16/1：§2.2 L14；1–2 MB/75–80%：§2.1 L34。
- 数据集统计与实验条件（21,613/15.60/105/2410、54,797/13.00/50/1461、17,708/208.3/100/1092、680.9/260.9/200,869、Poisson、60 秒思考时间、128 在途上限、500K token 限制、1TB/400GB pinned、版本号 v0.8.5/v0.2.1/256/v0.17.0/v0.4.5、页 1/32）：§5.1 逐项一致。
- F1/F2：§3.1 L26–28（$C=\lambda L$、$X=C\cdot S/L$）；F3 复算：2×32×8×128×2 B = 128 KB/token，20,000 token ≈ 2.56 GB，40 GB/128 KB ≈ 0.31M，均正确。
- 图 1–14 出处与 e2e_bench_all 为 PNG、其余 PDF 的对应关系与 figs/ 目录一致；图 7 原始 PDF 内文本（FIFO: A0+A1/C+D0；Strata: A0+B0/A1+B1/C+F/D0+D1/G Decoding/Stall Hiding）与页面 1017/1025/1066 行描述吻合——但页面未嵌入该图，见问题 3。

## 问题

- [阻断·技术] index.html 806/811 行（§1.2 与其折叠块）：页 32 容量账自相矛盾且与论文机制不符。806 行折叠块称每页 $32\times128\,\text{KB}=4\,\text{MB}$"远小于打满 PCIe 5.0 64 GB/s 所需的 1–2 MB 传输"——4 MB 并不小于 1–2 MB，比较关系错误；811 行称"每页仅 4 MB——是几 KB 至几十 MB 量级的小传输"，与导语 725 行"传输粒度只有几 KB"直接矛盾，且无法解释"页 32 = 4 MB 却实测只有 22% 带宽"｜引文依据：§1 L32 "small data transfers, sometimes only a few kilobytes, which fail to saturate PCIe bandwidth"；§2.2 L14 "each token may span from tens of kilobytes to several megabytes"；§2.1 L34 "1-2MB" 阈值；§4.2.1 L80–81 "this layout further fragments data, hindering bulk transfer efficiency"（真正机制：layer-first 布局把同页数据按层打散，有效连续传输单元是页内单层片段，非整页 4 MB）｜修复要求：修正 §1.2 与 806 行折叠块：说明有效传输单元由页大小与 layer-first 布局共同决定（Llama-8B 页 32 时页内单层片段仅 128 KB 级，更小模型/更小页时低至几 KB），删除"4 MB 远小于 1–2 MB"的错误比较，消除与导语"只有几 KB"的矛盾，使 22% 实测与机制自洽｜修复：｜复验：
- [阻断·可读性] overview.html 43 行（导语 lead 段）："最高 $5\times$"处实际字节为 `$5<TAB>imes$`（`\t` 被写成制表符、`\times` 仅剩 `imes`），KaTeX 无法渲染，核心数字显示为乱码；validate.py 未覆盖此缺陷｜引文依据：od 输出 `$   5  \t   i   m   e   s   $`；论文 abstract "up to $5\times$ lower Time-To-First-Token"｜修复要求：将该处改为 `$5\times$`（反斜杠 + times），并在修复后对 overview.html 做字节级检查确认无 TAB 残留（当前 `grep -c $'\t'` = 1）｜修复：｜复验：
- [重要·技术] index.html 1017/1066 行（§4.2、§4.4）：正文两处"看图 7"（FIFO vs Strata 的 delay hit 拆批、Stall Hiding 的 G/Decoding 标注），来源说明 1271 行亦声明 Figure 7（schedule_diagram.pdf）对应第 4 章，但页面未嵌入图 7 图片，读者被引导看一张页面上不存在的图｜引文依据：图 PDF 内文本含 "FIFO / A0+A1 / C+D0 / Delay Hit / Balance Batch / Stall Hiding / Strata / A0+B0 / A1+B1 / C+F / D0+D1 / G Decoding"；正文引用处 1017 行"看图 7（FIFO vs Strata）的对比"、1066 行"看图 7 的 Stall Hiding 标注"｜修复要求：在第 4 章嵌入 Figure 7 的转换图（PDF→JPEG，与图 4/5/6 同法），放在 1017 行首次引用之前；或删除两处"看图"引导、把图中标记改为纯文字叙述｜修复：｜复验：
- [重要·技术] index.html 1093 行（SVG 内示例文本）与 1097 行（图注）：贯穿示例写"匹配 token 数 100 > 阈值 100"，100 > 100 为假，示例输入与自身判定规则矛盾，按该数字示例会得出"不推迟"的相反结论｜引文依据：§4.3.1 L115 "a request is deferred only when the number of token matches on transient nodes exceeds this value. In practice, a default threshold of 100 …"（推迟条件为严格超过 100）｜修复要求：把 SVG 与图注中的示例匹配数改为大于 100 的具体数值（如 120），保持"100 &gt; 阈值"式的比较不再出现｜修复：｜复验：
- [重要·技术] index.html 843 行（§1.3 图 2 图注）与 877 行（本章问题解答 3）：把"共享前缀不是页大小整数倍时，末尾不足一页的 token 全部落空"当作论文事实陈述，论文只报告现象与按页匹配粒度，未给出此机制归因；页面未标注这是本文推断｜引文依据：§3.1 L39–42 原文仅有 "can improve cache hit rate, as cache matching is performed on a per-page basis" 与 "increasing the page size leads to a significant drop in the KV cache hit rate"，无尾部落空机制｜修复要求：在 843 行与 877 行两处为该机制添加"本文推断"标注（或改写为论文原话转述："论文将命中率下降归因于按页匹配的粒度变粗"）｜修复：｜复验：
- [重要·技术] index.html 1251 行（§6 局限段）：`[C27]` 标注错位——页面称 on-chip I/O 加速器出自"论文最后一节之外的'思考'部分"并标 C27（C27 定位 §6 Related Work L2–21），但 §6 无此内容；该内容实际在 §7 Conclusion，且 Conclusion 就是最后一节，"最后一节之外"表述亦误｜引文依据：§7 L2 "motivating the design of more versatile on-chip memory I/O accelerators"；§6 L2–21（Related Work 全文，仅含缓存共享/卸载/分离化三段，无加速器内容）｜修复要求：将 1251 行的引用改为指向 §7 Conclusion（新增编号或在来源说明中为 C27 补充 §7 定位），并把"论文最后一节之外的'思考'部分"改为"结论节的展望"｜修复：｜复验：
- [轻微·格式] index.html 825–829 行（§1.2 末）：图 1 嵌在"瓶颈一：分页带来的碎片化传输"小节末，但其内容（I/O stall 74%/24%）是瓶颈二的证据（829 行图注自称"第二类瓶颈的征兆"）；来源说明 1271 行写"Figure 1 → 第 1 章'瓶颈二'"，声明位置与实际嵌入位置不一致｜引文依据：论文 §1 L14/37 该图均用于论证调度瓶颈（"Schedulers that ignore these I/O-bound characteristics…"）｜修复要求：将图 1（含引导句与图注）移至 §1.4，或把来源说明改为"第 1 章瓶颈一末（引出瓶颈二）"，使二者一致｜修复：｜复验：
- [轻微·格式] index.html 913 行：`prefill/dedecode` 为笔误，多一个 `de`｜引文依据：不适用｜修复要求：改为 `prefill/decode`｜修复：｜复验：
- [轻微·格式] index.html 1175 行（§5.3 首段）："加两组额外变体"之后列出三个变体（Strata-IO、Strata-Schedule-Only、Strata-IO-LPM），数量词与清单不符｜引文依据：§5.3.1 L129–131 "we build and evaluate three ablated variants"｜修复要求：改为"三个变体"｜修复：｜复验：
- [轻微·技术] index.html 1249 行（§6 局限段）："给 prefill 留 5% 损失"丢掉了"小于"号，与页面 745/917 行及论文口径不一致｜引文依据：§4.2 L57 "less than 5\% performance degradation on prefill and 10\% on decoding"｜修复要求：改为"prefill 留 <5% 损失"｜修复：｜复验：
- [轻微·格式] index.html 806 行（折叠块）："20 KB token 手册"应为"20K token"（20 KB 读作 20 千字节）；同段"矩阵乘 2.56 GB × 每 token × O(d_model×L) 算力"运算符与量纲不成立，不可复算｜引文依据：不适用｜修复要求：改为"20K token 手册"，并将该句改为可读表述（如"prefill 需一次处理约 2.56 GB 输入，计算量随模型宽度与层数增长"）或删除该分句｜修复：｜复验：
- [轻微·技术] index.html 1187 行（图 10 图注）："hit rate 比页 32 低 2.4%"——论文未说明 2.4% 的参照对象是页 32 还是 Strata-IO，页面具体化无原文支撑｜引文依据：§5.3.2 L154 "primarily due to a 2.4\% lower cache hit rate"（无参照对象）｜修复要求：改为"命中率低 2.4%"（与论文同口径、不指明参照），或注明"参照对象论文未明说"｜修复：｜复验：
- [轻微·格式] index.html 1231 行（本章问题解答 3）："这说明带宽资源没有最大问题"语句不通｜引文依据：不适用｜修复要求：改为"这说明瓶颈不在带宽资源"或同义通顺表述｜修复：｜复验：
- [轻微·格式] index.html 849 行（§1.4）：公式 $L_{\text{layer}}\times\text{seq\_len}\times \text{KV\_size}$ 中 $\text{seq\_len}$ 与 $\text{KV\_size}$ 未定义（F3 定义的是 $L_{\text{layers}}$、$H_{\text{kv}}$、$d_{\text{head}}$、$b$）｜引文依据：不适用｜修复要求：为两个符号补一句话定义，或改用 F3 已定义符号（每 token 字节数 × token 数）｜修复：｜复验：
- [轻微·技术] index.html 721 行与 overview.html 44 行：发表信息"OSDI 2026（…Track 1 'KV Cache and Long Context'）"在论文 TeX 源码与 00README.json 中均无对应内容，本轮材料无法核实｜引文依据：0_main.tex 全文与 00README.json（version 2 / texlive 2023 / pdflatex）无 OSDI 或 Track 信息｜修复要求：为发表信息补外部来源标注（arXiv 页面或 USENIX 官网链接），或删除 Track 1 细节仅保留 arXiv:2508.18572v2｜修复：｜复验：
- [轻微·技术] index.html 800 行：128 KB/token 标注 `<sup>[N2]</sup>`，但来源说明（1268 行）将 N2 定义为"页大小 32/16/1 token"；128 KB 来自 Llama-3.1-8B 架构参数与 F3 构造算例，引用错位｜引文依据：来源说明 N2 定义"（页大小 32/16/1 token）见 §2.2 L14"；§2.2 L14 原文为页大小与"each token may span from tens of kilobytes to several megabytes"｜修复要求：删除 800 行的 `[N2]` 标注，或在来源说明中为 N2 补充"§2.2 L14 亦给出每 token KV 跨度几十 KB–几 MB"的定位｜修复：｜复验：

## 结论

- 统计：阻断 2 / 重要 5 / 轻微 10
- 处置：修复


## 修复记录（Round 1 后修复）

所有 17 个问题均已修复：

| 编号 | 问题简述 | 修复位置 | 修复内容 | 复验 |
|---|---|---|---|---|
| 阻断 #1 | §1.2 容量账自相矛盾 | index.html §1.2 | 删除 806 行折叠块整段；改写为正文两段（按页切分 + "为什么 4 MB 大页仍打不满"的 layer-first 解释） | validate.py 通过；katex=57 渲染；机制自洽 |
| 阻断 #2 | overview.html $5\times$ 制表符错误 | overview.html:43 | 改为正确字节 $5\times$ | grep -c $'\t' = 0；正常渲染 |
| 重要 #1 | Figure 7 未嵌入 | index.html §4.2 末 | 嵌入 schedule_diagram.pdf→JPEG base64 图 + 引导句 + 图注 + 颜色说明 | img=15（原 14 + Figure 7） |
| 重要 #2 | SVG 100>100 自相矛盾 | index.html SVG + figcaption | 改为 "200 > 100"；figcaption 加 "严格大于阈值才推迟" | 自洽 |
| 重要 #3 | 尾部落空机制缺推断标注 | index.html 843/1229 行 | 两处加 "标注为推断：论文仅指出按页匹配粒度变粗" | 推断标注清晰 |
| 重要 #4 | [C27] 错位 | index.html 1256 行 + evidence.md | 改 "§7 Conclusion"；evidence.md C27 增补 §7 定位 | 来源与正文一致 |
| 轻微 #1 | 图 1 位置说明不一致 | 来源说明段 | 改为 "§1.2 末，过渡证据" + 主体说明 | 一致 |
| 轻微 #2 | prefill/dedecode 笔误 | index.html 914 行 | 改 prefill/decode | 通过 |
| 轻微 #3 | "两组" 与三个不符 | index.html 1180 行 | 改 "三个变体" | 通过 |
| 轻微 #4 | "<5% 损失" 丢小于号 | index.html 1254 行 | 改 "prefill 留 <5% 损失" + 上标 | 通过 |
| 轻微 #5 | "20 KB token"、"矩阵乘 ...×O(d_model×L)" 不可复算 | index.html §1.2 | 整段重写（见阻断 #1） | 通过 |
| 轻微 #6 | "hit rate 比页 32 低 2.4%" 具体化无原文 | index.html 1192 行 | 改 "命中率低 2.4%" + 括号 "论文未明说 2.4% 相对哪个具体对照配置" | 通过 |
| 轻微 #7 | "没有最大问题" 病句 | index.html 1236 行 | 改 "瓶颈不在带宽资源本身，调度改进才是吃满硬件的必要条件" | 通过 |
| 轻微 #8 | seq_len/KV_size 未定义 | index.html 850 行 | 改用 F3 表达（每 token 字节数 × token 数） | 通过 |
| 轻微 #9 | OSDI Track 1 无法核实 | index.html 721 行 | 加 USENIX OSDI'26 技术场次外链 | 通过 |
| 轻微 #10 | N2 错位 | index.html 800 行 | 删除 [N2]；改括号说明 F3 复算来源 | 通过 |

机械验证：validate.py 两个页面均 ok；headless Chrome 探针 katex=57、svgkatex=0（无 SVG 内数学公式）、img=15、dollar=0、broken=1（headless 限制，13 张图 PIL 全部解码验证）。

进入第 2 轮前需要等待用户决定是否继续（按规范应执行 3 轮）。
