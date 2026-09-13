<!-- review-meta
round: 5
page: wiki/strata/index.html
reviewed_content_sha256: 860c4519b81b083f
-->
# Strata审查记录（第 5 轮）

- 页面版本：c734811c17b2df19bf78e05d3259a11a366fc167（index.html 工作树 blob）
- 论文版本：arXiv:2508.18572v1（2025-08-26；页面标注的固定版本）。核对来源：arXiv abs 页与 HTML 全文（https://arxiv.org/abs/2508.18572 、https://arxiv.org/html/2508.18572v1），页面 14 张原图 assets/img-01…img-14。
- 审查时间：2026-09-13 20:28
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：head 元信息 + 核心问题折叠块 + 术语表；第 1 章（1.1–1.4，含全部折叠块）；第 2 章（2.1–2.2）；第 3 章（3.1–3.3）；第 4 章（4.1–4.4，含 Algorithm 1 折叠块与自绘 SVG）；第 5 章（5.1–5.6）；第 6 章（6.1–6.3）；来源与范围说明。同时逐张核对 14 张原图与正文图注。

## 逐条回源核对（要点，均通过）

- 8192 token 加载约 22% PCIe 5.0 带宽、GH200 约 5%：论文原文一致；img-13（Fig 3）实测曲线 64 GB/s 点≈0.22、384 GB/s 点≈0.05，对应无误。
- 74% I/O stall / 4× 吞吐降、Fig 1 红≈0.72、绿（Strata-IO-Only）≈0.22、论文正文 24%：一致；img-14 图例四条曲线与正文描述相符。
- Little's Law $C=\lambda L$、$X=\lambda S=C\cdot S/L$（F1/F2）：复算正确；75–80% 带宽需 1–2 MB 传输与原文 "1-2MB" 一致。
- 页大小 32/16/1（TensorRT-LLM/vLLM/SGLang）与 §5.1 实验 vLLM 页 32 的差异，页面已在 1.2 节显式标注处理，非矛盾。
- 容量账复算：$2\cdot32\cdot8\cdot128\cdot2=131072$ B=128 KB/token；20000×128 KB≈2.56 GB=40 GB 的 6.4%；40 GB/128 KB≈0.31M；625 页×4 MB、单层片段 128 KB——与标注算式全部吻合。
- 2 block $\times$ 1024 thread、≈50 GB/s、prefill <5%、decode 10%、"端到端整体影响 <5%"：论文 §4.2 原文含 "end-to-end evaluation confirms ... keeping overall performance impact under 5%"，非推断；img-11（Fig 5）曲线相符。1024 线程/SM 上限 2048 线程·64 warp → 理论正确。
- 消融 2.3×（Strata-IO）/1.8×（Strata-Schedule-Only）、页 512→93% 且命中率低 2.4%、cache distance +42/+76/+95/+11/+12/+8/+3、GH200 40→150 GB/s、磁盘 page-first 4×：均与论文原文一致；img-04、img-03、img-01、img-09 图内数值相符（img-09：8B 1.687→0.420≈4×）。
- 实验设置（H200 8×H200/5920… 1.6TB DRAM/64 GB/s；GH200 H100+Grace 64 核/464GB LPDDR5X/384 GB/s；三模型、四数据集统计、Poisson 到达、70B 4 卡张量并行、1TB pinned、在途 128、ShareGPT 500K token 与 60s 思考时间、SGLang-HiCache 自建基线）：逐条与原文一致。
- Fig 7 时间线（FIFO 行 A0+A1/B0+B1/C+D0/D1+F/G/Decoding；Strata 行 A0+B0/A1+B1/C+F/D0+D1/Decoding/G；Delay Hit/Balance Batch/Stall Hiding 三标注）与 img-07 完全一致；配色映射一致。
- 引用/链接：kv-cache、paged-attention、prefix-caching、standard-attention、gpu-execution-model、gpu-communication、kv-cache-layout、dualpath 八个前置页均存在；第 1.4 节 DualPath 陈述与 dualpath 页一致；validate.py 通过（exit 0）；dojo:topics 在允许词表内。
- 表述维度：全文无"本页/下面来看/需要注意的是"式元话语，无"我/我们/你"会话指代，无调试与复现叙事，无临场评价；"x 倍"写法见下。

## 问题

- [重要·技术] 第 70 行（导言"三个机制"段末）："短上下文不退化"标注 `<sup>[C26]</sup>`，但本页来源说明（第 621 行）把 C26 定义为"ROCm 兼容 见 §4.2 L55"，与短上下文无关；同一论断在第 543 行用的是 [C24]（§5.2.3 L106–111），自相矛盾。｜引文依据：论文 §5.2.3 "underlying SGLang engine exhibits a slight performance disadvantage compared to the base engines of vLLM and TensorRT-LLM on the Llama-8B and -70B models due to kernel differences"（短上下文证据）；§4.2 L55 为 ROCm 后端兼容声明。｜修复要求：将第 70 行的 [C26] 改为 [C24]。｜修复：已把导言"三个机制"段末短上下文论断的来源标注由 [C26] 改为 [C24]（C24 对应 §5.2.3 短上下文证据，定义见第 621 行），与第 543 行一致。｜复验：已复跑 `.dojo/scripts/validate.py wiki/strata/index.html`（exit 0）并核对第 70 行。

- [轻微·格式] 全文倍数写法不统一：表格与部分正文用 ASCII "x"（如第 102、479、487–491、496、500、513、548、551 行，共 43 处 "Nx"），同一句内又与 KaTeX `$\times$` 混用——第 102 行"分别 3.2$\times$/2.6$\times$/1.9$\times$；…调度单独 1.8x、I/O 单独 2.3x"；站内其他论文页（如 kv-cache 全用 `\times`）无此写法，本页为孤例，与 check.md #11"同一写法全页一致"不符。｜引文依据：不适用。｜修复要求：把正文与表格中的 "Nx" 统一改写为 `$N\times$`。｜修复：已把正文与表格中全部 43 处 "Nx" 改写为 `$N\times$`（含第 81、101、102、463、479、487–491、496、500、513、548、551、627 行），全文已无 "Nx" 残留，与站内其它论文页写法一致。｜复验：已复跑 `.dojo/scripts/validate.py wiki/strata/index.html`（exit 0）。

- [轻微·可读性] 第 139 行引号错位："分层缓存解决了"存得下"，但搬回来"成为"了新的瓶颈"——引号落在"成为"上，破坏了与"存得下"的并列（同页第 145 行正确写法为"搬回来"正是瓶颈所在）。｜引文依据：不适用。｜修复要求：改为"…但"搬回来"成为了新的瓶颈"。｜修复：已把第 139 行"但搬回来"成为"了新的瓶颈"改为"但"搬回来"成为了新的瓶颈"。｜复验：已复跑 `.dojo/scripts/validate.py wiki/strata/index.html`（exit 0）并核对第 139 行。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（重要问题关闭后可发布；第 621 行/第 543 行的 C24/C26 定义无需改动）

## 备注（未计为问题）

head 声明"发表：OSDI 2026"，可核对的 arXiv v1 其 Comments 为 "under peer review"（v1 早于会议决定），USENIX 技术场次页对抓取返回 403，无法在线复核；站内 kv-cache 等页亦一致记为 OSDI 2026，故不据此判为来源不符。