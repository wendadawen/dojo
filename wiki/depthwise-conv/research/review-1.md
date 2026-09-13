<!-- review-meta
round: 1
page: wiki/depthwise-conv/index.html
reviewed_content_sha256: 76a4980e3a3f59b4
-->
# 深度可分离卷积审查记录（第 1 轮）

- 页面版本：index.html c3bcfc5a44071565bad2868a019a0050be3c0b30；overview.html b4e72ccc63d6f5b31c104dc53ce77decda3c1908
- 审查时间：2026-09-13 18:48
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节：核心问题、常见误解、1. 标准卷积：滤波与组合一步完成、2. 拆成两步：depthwise 与 pointwise、3. 省了多少：比值推导、4. 大模型里的用法：KDA 的短深度卷积、来源与范围说明（含全部 details 折叠块、SVG 图注）、overview.html
- 核对的主要来源：
  - MobileNets，arXiv:1704.04861 §3.1（经 ar5iv.labs.arxiv.org/html/1704.04861 抓取原文；式 (1)–(6)、定义句、Table 4）
  - GLM-5.3-Flash 官方源码 `transformers/models/glm5_next/modeling_glm5_next.py`（commit 8f542025，`raw.githubusercontent.com` 抓取，2375 行）；`zai-org/GLM-5.3-Flash` config.json
  - 两段代码块均在 python3 实跑（成本计数块无依赖；torch 块在 /tmp/dwrun 目录下运行，避免 /tmp/inspect.py 遮蔽标准库）
- 机械校验：`python3 .dojo/scripts/validate.py wiki/depthwise-conv/index.html` 与 `.../overview.html` 均返回 `validation ok`

## 问题

- [阻断·技术] 来源与范围说明 / 简化条件及其限制（index.html L400）：「本页不引用具体数字（论文正文未给）」一句与来源不符——MobileNets 论文确实给出了 depthwise separable 与标准卷积的精度对照。｜引文依据：MobileNets §3.1 原文“……at only a small reduction in accuracy **as seen in Section 4**”；§4 Table 4：**Conv MobileNet 71.7% / 4866 M Mult-Adds / 29.3 M 参数**，**MobileNet（depthwise separable）70.6% / 569 M / 4.2 M**（即约 1.1 个百分点的精度差、约 8.5 倍的乘加缩减）。｜修复要求：删除「（论文正文未给）」括注；若要保留该条简化说明，改为准确表述（如“论文 §4 Table 4 给出该对照：精度 71.7%→70.6%、乘加量 4866M→569M”），或整体删除该括注。｜修复：｜复验：

- [重要·技术/可读性] 第 4 章 本章问题第 2 题解答折叠块的 summary（index.html L351）：「解答：右侧 padding 后截尾」与同题正文及来源注解矛盾。｜引文依据：同章正文 L352「做法是在序列左侧补 $D_K-1$ 个零再卷积，取前 $S$ 个输出」；[C5] L364「以左 padding + 头部切片实现因果」；源码 `causal_conv1d_fn`（L401–410）`padding = weight.shape[-1] - 1` 后 `F.conv1d(...)[:, :, :seq_len]`——决定因果性的是左侧 $k-1$ 个零，右侧 padding 被头部切片丢弃。｜修复要求：把 summary 改为「左 padding 后取头部」等与正文/L352/[C5] 一致的表述。｜修复：｜复验：

- [重要·来源/表述] 引言（index.html L65）：「……为无位置编码的模型提供局部时序结构」被写成无标注的事实陈述；同一结论在 L292 明确标注「结构解读，非源码直陈」，引言省略标注，等于把推断当来源结论。｜引文依据：[C6] L365「结构解读（源码只有投影与卷积的顺序事实，无此陈述）」；L292 同句重复出现「（结构解读，非源码直陈）」。此外该句以「在大模型里」作一般化陈述，实际仅 GLM-5.3-Flash 一例经核对。｜修复要求：引言处补「（结构解读）」标注，或改为限定表述（如“本页将其解读为……”），与正文标注一致；一般化范围收紧到已核对的实例。｜修复：｜复验：

- [轻微·表述] 第 4 章正文（index.html L292）：括号标注连续出现两次「（结构解读，非源码直陈）（结构解读，非源码直陈）」。｜引文依据：不适用。｜修复要求：删除重复的一处标注。｜修复：｜复验：

- [轻微·来源/表述] 引言（index.html L65）：「深度可分离卷积由此成为移动端视觉模型的标配」为无来源支持的概括性评价（“标配”）。｜引文依据：不适用（页面与 [C1]–[C6] 均无支撑该普遍性的材料）。｜修复要求：删除或降级为可定位的表述（如“MobileNets 之后成为移动端视觉模型的常用结构”），或另加来源。｜修复：｜复验：

- [轻微·表述] 引言（index.html L67）：「结构：先拆解标准卷积做的两件事，再看两步拆分，然后推导缩减比，最后核对……」为路线图式元话语（“再看……然后……最后”）。｜引文依据：不适用。｜修复要求：改为直陈式范围说明，或删除该导航句。｜修复：｜复验：

- [轻微·表述] 第 3 章代码块观察重点（index.html L271）：「$D_K=3$、$M=N=64$ 时 pointwise（约 105 万）比 depthwise（约 15 万）贵得多——**3×3 场景下**主要开销移到了组合这一步」。同一句把“场景”当术语，且“3×3”用 Unicode `×` 直写，与全页其余处（如 L107、L243 的 `$3\times3$`）的 LaTeX 写法不一致（style-guide §11「同一变量全页写法一致」）。｜引文依据：不适用。｜修复要求：改为“$3\times3$ 核时主要开销移到组合这一步”，去掉“场景”。｜修复：｜复验：

- [轻微·链接] 第 4 章正文（index.html L292）：「（模型层面的分析见 GLM-5.3-Flash 数据流页）」未加超链接；同段 L288 的「见 <a href="../kda/index.html">KDA</a>」有链接，且 `wiki/glm-5-3-flash-dataflow/index.html` 确实存在。｜引文依据：不适用。｜修复要求：补上 `<a href="../glm-5-3-flash-dataflow/index.html">GLM-5.3-Flash 数据流页</a>`。｜修复：｜复验：

- [轻微·格式] overview.html L21（lead）：「乘加量降到约 1/N + 1/D_K^2 倍」中 `1/N + 1/D_K^2`、`D_K` 为 ASCII 近似写法，而同页 L35 用 `$1/N+1/D_K^2$`，同一变量在同一页面出现两种写法，违反 style-guide §11。｜引文依据：不适用。｜修复要求：overview lead 改用 `$1/N + 1/D_K^2$` 等 LaTeX 写法。｜修复：｜复验：

- [轻微·格式] 全部四个正文 h2 的副标题使用「主题：副标题」（L112「1. 标准卷积：滤波与组合一步完成」、L142「2. 拆成两步：depthwise 与 pointwise」、L231「3. 省了多少：比值推导」、L286「4. 大模型里的用法：KDA 的短深度卷积」），style-guide §1 规定 h2 副标题用「主题——副标题」。｜引文依据：不适用（同类页 rmsnorm 亦用「：」，属全站未统一项）。｜修复要求：统一改为「——」，或先与 style-guide 对齐后再改。｜修复：｜复验：

## 已核对且通过的来源论断（供复验参考）

- [C1][C2] 定义：MobileNets §3.1「We use depthwise convolutions to apply a single filter per each input channel (input depth). Pointwise convolution, a simple 1×1 convolution, is then used to create a linear combination of the output of the depthwise layer.」——与页面 L123、L146–152 一致。
- [C3][N1] 8~9 倍：§3.1「MobileNet uses 3×3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy」——与 L243、L107、L108、L363 一致。
- [F1]–[F5] 公式：§3.1 式 (1) 标准卷积输出、式 (2) $D_KD_KMND_FD_F$、式 (3) depthwise 输出、式 (4) $D_KD_KMD_FD_F$、式 (5) 合计、式 (6) $1/N+1/D_K^2$，与页面 L116、L127、L148、L150、L156、L235 及 [F1]–[F5] 的式号一一对应；L270 引用的「公式 (2)(4)(5)」式号正确。
- [N2][F5][F6] 代码：L248–259 实跑输出与页面「预期输出」（L263–268）逐行完全一致（9,437,184 / 147,456 / 1,048,576 / 1,196,032 / 0.126736 / 7.9x）；L297–324 torch 块实跑得 `[-1.0, -2.5, -3.5, -3.5, -3.5]`、`[1.0, 3.0, 5.0, 7.0]`、`[10.0, 20.0, 30.0, 40.0]`，与 L330–332 预期输出一致；手算注释逐行复算无误。
- [C5][N3] GLM 配置：config.json `linear_attn_config = {num_heads: 64, head_dim: 128, short_conv_kernel_size: 4, gate_lower_bound: -5.0}`，$\text{qkv\_dim}=64\times128=8192$，源码 `self.conv_dim = self.qkv_dim * 3 = 24576`，`nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, bias=False, kernel_size=conv_kernel_size, groups=conv_dim, padding=kernel_size-1)`（源码 L607–615）——「通道 24576（=3×8192）、kernel 4、groups=通道数、无偏置」全部成立；`causal_conv1d_fn` 的 `F.conv1d` 调用正位于 L404–410，与 [C5] 标注一致；checkpoint 拆 `q/k/v_conv1d` 三个 `[8192,1,4]` 与数据流页交叉验证一致。
- 页面内链 `../kda/index.html`、`overview.html` 均存在且互链正常；`dojo:type=concept`、`dojo:topics=模型结构`（在 AGENTS.md 允许大类内）、`dojo:tag=网络结构`（validate.py 通过）；`description` 为纯文本、`dojo:summary` 为可渲染 LaTeX。

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 7
- 处置：修复（阻断与重要项须关闭后方可发布；轻微项按上述修复要求处理或记录接受理由）
