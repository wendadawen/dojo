<!-- review-meta
round: 8
page: wiki/ppd-disaggregation/index.html
reviewed_content_sha256: ea81aaab73d0fea9
-->
# Not All Prefills Are Equal（PPD）审查记录（第 8 轮）

- 页面版本：index.html 工作树哈希 d081693456ec01383f148153a29ca2107f9d5b20（overview.html 20066c09571a0dffdf1e691546a1029c1e2caddd）
- 论文版本：arXiv:2603.13358v2（v2 修订 2026-05-05，cs.NI，19 pages，Accepted at ICML 2026）。核对底本为 arXiv HTML 全文 https://arxiv.org/html/2603.13358v2 与摘要页 https://arxiv.org/abs/2603.13358；本页元信息标注的版本（v2）与该底本一致。页面用自建 C/F/N/G 编号体系，不使用论文参考文献序号，故无 v1/v2 编号错位风险。
- 审查时间：2026-09-14 17:44
- 审查者：独立子代理（未参与写作与前序轮次审查）
- 已完整阅读章节：核心问题 / 术语表 / 1. 多轮对话暴露 PD 分离的两个代价 / 2. full prefill 与 append-prefill 差一个数量级 / 3. 没有静态最优：3060 个数据点的扫描证据 / 4. PPD：把路由变成带权重的逐请求决策 / 5. 真实负载、慢网络与权重旋钮下的表现 / 6. 方法评价 / 7. 附：PPD 架构概念图 / 来源与范围说明（含全部折叠块与图注），以及 overview.html。
- 机械核验：`python3 .dojo/scripts/validate.py wiki/ppd-disaggregation/index.html` → `validation ok`。概念链接 standard-attention / mqa-gqa / gpu-communication / prefix-caching 均真实存在；无「（待生成）」占位；overview 与 index 互链；dojo:topics 四项与 dojo:tag「推理系统」均在词表内。
- 逐项回源结果（无问题部分）：abstract/§8 的 ~68%、§4.1 的 48%/2%、+57%/+21%、32K 3–4×/64K <25%、§4.2 的 17×18×10=3060、§4.3 Tab.1 三档六数（-57.8/-65.2/-73.3、-47.7/-51.6/-56.2、-44.3/-38.1/-24.9）与汇总 48–73%、§4.4 的 92.2% 与 Tab.2 全表（63.3/0.6/0/21.3、0/38.3/4.4/14.2、3.3/33.3/27.8/21.5、27.2/15.6/38.3/27.0）、§6.2 的 15–25%、3.1 轮/会话、~75%、~3×、§6.3 Tab.3（10/0/4、5/13/27、12/14/27）、§6.4 Fig.5 的 143.7→170.6 ms(+18.7%)、PPD ~51 ms、E2E +4.7%(3028→3169)、+0.9%、64%→70%、§6.5 Fig.6 的 QPS=8「TTFT −96%, TPOT +7.0%」/ QPS=16「TTFT −94%, TPOT +12.1%」与 1/3/6→95/50/20%、App.B.6 的 s_kv=128 KiB、P90 5115 token→~670 MB、4.5/27/67 ms、App.C.3 的 2–16 轮与 8B/14B/30B ~70%、App.C.4 Tab.5 全表、App.A 的 6.1/12.2/16.1%、App.C.5 的 30s→10s 结论，均与 v2 原文逐字/逐数一致。页面构造计算复算无误：1250²=1,562,500、50×1250=62,500、比值 25 与 n/m=24；128 KiB 构成 2×8×128×2×32=131072 B；2K×128 KiB=256 MiB；2048/1250 token 换算 256/156 MiB；Eq.1 三组代入（0.52 / 0.12 / −0.20 与 0.15 / −0.30）全部正确。图内读数用像素测量核对：img-05 三个子图标题为「QPS = 0.5 / 1.0 / 1.5」、legend 为 D-local capable(baseline) 与 baselines，figcaption 对 best 点的描述成立；img-03 横轴标签确为「NVLink (~200 GB/s)」，页面「取图内横轴刻度、论文图注作 ~150 GB/s effective」的处理与原图及图注双双吻合；img-02 四个操作点图例为 Static PD(0%)/20%/50%/95%，与页面一致；img-01 两张「Last Turn's KV」红线确为指向 P 的箭头，页面据此指出其与单向协议的张力，判断正确。

## 问题

- [重要·技术] wiki/ppd-disaggregation/index.html:173（§1 本章问题第 1 题解答）：把「第 1 轮的 200 token 回复」说成在 P 上处理过｜引文依据：v2 §2.2 "P nodes act as KV _producers_ and D nodes as _consumers_, with no reverse channel from D back to P"；v2 §1 "the response token KV from the previous turn already resides on D, it is inaccessible to P"；v2 §4.2 "when x=0, D receives KV transfer from P every turn"。即回复 token 的 KV 由 D 在 decode 阶段生成，P 从未处理过该段文本，其 KV 也从未由 P 传输到 D。页面同一章第 2 题解答（第 180 行）自述「P 接收第 2 轮的请求时是计算意义上的『空』」，与本句「在 P 上早已处理过」直接冲突｜修复要求：把该句改写为「第 1 轮的 1000 token 输入由 P prefill、其 KV 已传到 D；200 token 回复的 KV 由 D 在 decode 阶段就地生成、也留在 D 上」，删除「200 token 回复在 P 上早已处理过、KV 已传到 D」这一把两段历史一并归给 P 的表述；改写后与第 2 题解答及第 1 章正文（第 137 行）保持一致｜修复：｜复验：

- [轻微·技术] wiki/ppd-disaggregation/index.html:577（6.3 与相邻工作的相对位置）：把论文中两条独立引用合并为一个别名「MemServe（LMCache）」｜引文依据：v2 §7 "Distributed KV cache layers (e.g., Mooncake (Qin et al., 2025), MemServe (Hu et al., 2024a)) operate at a different layer than PPD"；v2 §2.3 "Mooncake (Qin et al., 2025) provides cluster-wide distributed KV stores" 与 "LMCache (Liu et al., 2025b) offers a modular KV cache layer" —— 该版文献表把 MemServe（Hu et al., 2024a）与 LMCache（Liu et al., 2025b）列为两条不同条目，页面括注读作「MemServe 即 LMCache」，本页 C14 行（第 630 行）也只写「Mooncake/MemServe」，未含 LMCache｜修复要求：改写为「与 Mooncake、MemServe、LMCache 等分布式 KV 缓存/存储层处于不同层」，或去掉括注中把两者等同的写法｜修复：｜复验：

- [轻微·技术] wiki/ppd-disaggregation/index.html:214（§2.2 图引段）：「原图在 batch 200 处标注 +2%」｜引文依据：对原图 assets/img-06.webp 做像素测量（轴标定：x 轴 0→px170、250→px1247，即 4.308 px/batch；y 轴 0→px863、60→px49）。「+48%」与「+2%」两处标注文字的像素范围分别为 x 1080–1164、x 1080–1152，换算为 batch ≈211–231 与 ≈211–230（中心约 batch 220），并不落在 batch 200（px≈1032）。另：v2 §4.1 正文确写 "Full prefill causes ∼48% slowdown at batch size 200"，故「batch 200 处 48%」的数值表述本身与原文一致，问题只在「原图在该处标注」这一关于图内位置的断言｜修复要求：改为「原图在曲线右端（batch≈200–250）标注 +2%/+48%」，或删去关于标注落点的断言，只保留与论文正文一致的「batch 200 处约 48% / 2%」｜修复：｜复验：

- [轻微·格式] wiki/ppd-disaggregation/index.html:492、568、570、700：同一概念中英两种写法并存、英文原词残留｜引文依据：不适用（表述类）。第 570 行以粗体引入术语「PPD 是 routing actuator」，第 700 行沿用 "routing actuator"，但第 593 行同一概念写作「PPD 只是路由执行器」；第 492 行写「weight 扫描」，而第 358 行同一实验写作「第 5 章权重扫描」；第 568 行「成本 prohibitive」为生硬英文残留（v2 App.C.3 原文 "would incur prohibitive initialization overhead"）｜修复要求：统一为「路由执行器」（或统一保留 routing actuator 并在首次出现时给中文释义），「weight 扫描」改为「权重扫描」，「成本 prohibitive」改为「初始化开销过大」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（唯一重要问题为第 1 章本章问题解答中的机制误述，改动局限在该句；三条轻微问题不影响核心结论与主线理解）

统计：阻断 0 / 重要 1 / 轻微 3