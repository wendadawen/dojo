<!-- review-meta
round: 5
page: wiki/dflash2/index.html
reviewed_content_sha256: a9f8ad633cbfcfbd
-->
# DFlash 2 审查记录（第 5 轮）

- 页面版本：ba56235f8e435a22c0641e76f3749d6e0a3c3e42
- 审查时间：2026-09-13 21:44
- 审查者：独立子代理（未参与写作与前序轮次审查）
- 已完整阅读章节：核心问题（页面级） → 1. 两个剩余问题——候选池里有答案、块末端在漏气（含本章问题） → 2. 路径选择器——在候选之间打分，而不是重新预测（2.1 打分公式 / 2.2 全并行、串行只在最后 / 2.3 与 DSpark 修正头的对比 / 2.4 最小例子 / 本章问题） → 3. 两抽头卷积——给块末端补上「看见前一位」的通道（3.1 后缀衰减的证据链 / 3.2 两抽头动态深度卷积含 SVG / 3.3 效果含折叠块 / 本章问题） → 4. 组合效果——每次验证多一个 token 的账（4.1 / 4.2 / 4.3 / 本章问题） → 5. 端到端与边界——哪里收益趋近 1（5.1 / 5.2 / 5.3 / 本章问题） → 来源与范围说明

## 核对说明（外部来源回源结果）

- 官方博客 https://inco.ai/blog/dflash2/ ：Table 1（Recall@1 85.4/80.3/79.4/78.3/77.5/75.9/72.9、Recall@16 99.5/97.3/94.8/92.6/90.8/89.4/87.8、oracle 4.27→6.79）、Table 2（DFlash 4.27/3.78、+DSpark 修正 +77.8M/+9.6%/4.49/4.08、+路径选择器 +2.0M/+0.6%/4.61/4.25）、Figure 2（3L 85.21…64.97、5L 85.39…72.86、15L 86.42…78.73、5L+conv 85.83…77.61；15L 3× 参数/+15.2%、conv +3%/+0.7%、16.5M）、块内注意力 30%(L1)→8%(L5)、第 4–5 层 9.4%→0.5%、Table 3/4/5、Figure 5（MATH-500 逐位置，DFlash 2 首位 88.3、末位 86.48，MTP 77.85、DFlash 77.48、DSpark 79.86）、F1 打分公式、F2 卷积公式、"It beats the DSpark correction in both settings with roughly 40× fewer parameters and 16× lower latency overhead"、"Choosing is cheaper than predicting"、"the selector and the convolution together add only 1.3%" —— 全部与页面一致。
- HF 模型卡 https://huggingface.co/incoai/Qwen3.8-27B-DFlash2 ：Throughput 表 15 格倍数（并发 1 2.59/2.69/3.43…、并发 8 2.19/2.23/2.84…、并发 32 1.04/1.13/1.45… 及 MTP MT-Bench 0.77、MATH-500 0.94）、自回归基线 68.9/69.0/69.0/69.0/68.9、Acceptance Length 表（4.28/3.62/4.80）、环境（单卡 H200 + SGLang + FA3 + 块 8 + 推荐采样 + xhigh + 4096）、"It is not a standalone language model"、"Decoding is lossless…" —— 与页面逐格一致；各表均值复算无误（5.97 / 4.92 / 5.49 / 4.54 / 4.80 / 4.44 / 4.48 / 3.62 / 4.28）。
- HF 模型卡 https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2 ：license apache-2.0、"It is not a standalone language model"、块 16（15 草稿 token） —— 与页面一致。
- arXiv:2602.06036 ：v2 存在（v1 2026-02-05、v2 2026-05-28），作者 Jian Chen / Yesheng Liang / Zhijian Liu，Accepted at ICML 2026 —— 与页面 head「Chen, Liang, Liu, ICML 2026, arXiv:2602.06036v2」一致。
- 机械项：`python3 .dojo/scripts/validate.py wiki/dflash2/index.html` → validation ok（exit 0）；KaTeX 实测渲染 dojo:summary 全部 12 个公式与正文两条展示公式，均 OK；本地资源（katex/prism/dojo-concept.css/index.html）与前置概念页（speculative-decoding、dflash、block-diffusion）及锚点 dflash/index.html#inference-pipeline 均存在；无「（待生成）」占位；`<text>` 内无 ASCII 数学近似；`×`/`–`/`→` 经 validate.py 明确列为不参与公式检查的普通排版字符，不算未渲染公式。

## 问题

- [阻断·技术] 第 421 行（4.2 节首段末句）与第 453 行（第 4 章本章问题第 2 题解答）：把「接受长度领先幅度」写成「完整 token 以上」，与同页第 4.1 节列出的数字直接矛盾，也超出来源的表述范围。｜引文依据：博客原文 "The margins are wide: on both models, DFlash 2 averages more than a full token ahead of DSpark."（只声明相对 DSpark）；同页第 413 行「相对 DSpark 5.49 → 5.97，提升 +0.48 token」；页表 4.1/4.2 中 Qwen3.8-27B DFlash 2 4.80 vs MTP 4.28（差 0.52）、Muse Glimmer 5.70 vs 官方 DFlash 4.44（差 1.26）。逐对复算：Qwen3.5-4B 对 DSpark 仅 +0.48、Qwen3.8-27B 对 MTP 仅 +0.52，均不足一个整 token，故「相对各基线的完整 token 以上领先」为假。｜修复要求：删去第 421 行「两个目标模型上 DFlash 2 都拿到完整 token 以上的领先」与第 453 行「相对各基线的完整 token 以上领先」，改为按来源限定比较对象并给出实际差值，例如「相对社区 DSpark 草稿器平均领先一个整 token 以上（Qwen3.8-27B +1.18、Muse Glimmer +1.22）；相对各基线的领先为 +0.48 至 +1.43 token」。｜修复：｜复验：
- [重要·技术] 第 505 行（5.3 部署注意事项「引擎支持状态」条）：「SGLang 稳定支持」在本页引用的来源中无依据，且与同句对 vLLM/llama.cpp「未进稳定版本」的对比构成来源未支持的判断。｜引文依据：博客 Run It Now 中 SGLang 同样是源码安装 —— `pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python"`，并非已发布稳定版；博客对该段的表述仅为 "DFlash 2 already runs in the mainstream inference engines:"，未出现「稳定」「已发布」一类措辞（vLLM/llama.cpp 分别走 `refs/pull/52816/head` 与 `pull/27342/head`，oMLX 为签名预编译包）。｜修复要求：把「SGLang 稳定支持」改为有来源支持的表述，如「SGLang 可直接运行（博客以 git main 源码方式安装）」，并删去「未进稳定版本」这一与 SGLang 形成的来源未支持对比；或把该条整体降级为标注的推断。｜修复：｜复验：
- [轻微·可读性] 第 69 行「本文围绕这两个组件的机制、证据和边界展开」、第 509 行「本页只作背景带过」：以「本文／本页」为主语的自我指代（元话语）。｜引文依据：不适用。｜修复要求：改写为不带自我指代的陈述，例如「以下按两个组件的机制、证据与边界展开」「这些数字属厂商自测，仅作背景」。｜修复：｜复验：
- [轻微·可读性] 第 144 行「演示两个问题在最小例子里长什么样」、第 496 行「Inco AI 自己的模型卡在并发 32 一行把这一面说得很清」：口语化措辞与临场评价。｜引文依据：不适用。｜修复要求：改为中性表述，例如「演示两个问题在最小例子中的形态」「该表在并发 32 一行逐任务给出倍数」。｜修复：｜复验：
- [轻微·格式] 第 322、326、330 行（3.2 节 SVG 的 foreignObject）：图内公式写作 $\mathrm{Conv}(x)_1$、$\mathrm{Conv}(x)_2$、$\mathrm{Conv}(x)_3$，漏掉正文定义式 $\operatorname{Conv}_k(x)_t$（第 295 行）中的下标 $k$，同一算子全页写法不一致。｜引文依据：不适用。｜修复要求：图内统一为 $\mathrm{Conv}_k(x)_1$ 等，或保留简写但在图注中说明「图中省略下标 $k$」。｜修复：｜复验：
- [轻微·技术] 第 205 行（2.3 节）与第 264 行（第 2 章本章问题第 1 题解答）：「路径长度等于块大小（线性而非按词表大小）」是页面自行补出的比较句，来源没有该表述，却与 [N1] 一并标注。｜引文依据：博客相关原文只有 "The only sequential work is the final walk over precomputed scores"，未见「线性而非按词表大小」的论断。｜修复要求：删去该比较句，或明确标注为由机制直接得出的推断（如「走路径每步前进一个位置，长度随块大小线性增长」），不再与 [N1] 并列作为来源结论。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：修复（按上表逐条修复后重新核对来源与数字；阻断项所涉数字修改后需与博客 Table 3/4/5 及模型卡 Acceptance Length 表重新对账）
