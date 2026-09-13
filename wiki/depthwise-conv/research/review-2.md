<!-- review-meta
round: 2
page: wiki/depthwise-conv/index.html
reviewed_content_sha256: 76a4980e3a3f59b4
-->
# 深度可分离卷积审查记录（第 2 轮）

- 页面版本：c3bcfc5a44071565bad2868a019a0050be3c0b30
- 审查时间：2026-09-13 19:01
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题、常见误解、1. 标准卷积：滤波与组合一步完成、2. 拆成两步：depthwise 与 pointwise、3. 省了多少：比值推导、4. 大模型里的用法：KDA 的短深度卷积、来源与范围说明（含全部折叠代码块、SVG 图注与两级问题块的折叠答案）

## 来源核对说明

- MobileNets §3.1（arXiv:1704.04861，经 ar5iv 全文核对）原句逐条对上：式 1 `G_{k,l,n}=Σ K_{i,j,m,n}·F_{k+i-1,l+j-1,m}`；式 2 成本 `D_K·D_K·M·N·D_F·D_F`；式 3 depthwise；式 5 总成本 `D_K·D_K·M·D_F·D_F+M·N·D_F·D_F`；pointwise 原句 "Pointwise convolution, a simple 1×1 convolution, is then used to create a linear combination of the output of the depthwise layer."；效率原句 "MobileNet uses 3×3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy."。页面的 [C1][C2][C3][N1][F1]–[F5] 均由此支持。
- 第 3 章代码块与第 4 章 torch 代码块**实际执行**（torch 2.x / Python 3.9）：第 3 章输出逐行一致（9,437,184；147,456；1,048,576；1,196,032；0.126736；7.9x）；第 4 章输出一致（`[-1.0,-2.5,-3.5,-3.5,-3.5]`；groups=2 通道 0 `[1,3,5,7]`、通道 1 `[10,20,30,40]`），手算注释亦正确。
- GLM 侧 [C5][N3] 的配置（通道 24576=3×8192、kernel 4、groups=通道数、无偏置、checkpoint 拆 q/k/v_conv1d 三个 [8192,1,4] 张量）与仓库内 `wiki/glm-5-3-flash-dataflow/index.html` 记载逐项一致，未见矛盾。
- `.dojo/scripts/validate.py wiki/depthwise-conv/index.html` 通过（validation ok）。

## 问题

- [重要·技术] 核心问题第 3 题解答（index.html:90）｜来源：MobileNets §3.1 / 本页第 3 章正文｜位置：核心问题第 3 题「$1/N+1/D_K^2$ 怎么除出来的？」的解答折叠块｜问题：约分描述与算式不符——第一项分子 $D_K^2MD_F^2$ 中不含 $N$，无法「约掉 $D_K^2MN$」；正文第 3 章同一推导写的是「约去 $D_K^2MD_F^2$」。同一页两处推导描述互相矛盾。｜引文依据：解答写「第一项约掉 $D_K^2MN$ 剩 $1/N$，第二项约掉 $MN$ 剩 $1/D_K^2$」；正文（index.html:239）写「第一项（depthwise）：分子分母约去 $D_K^2MD_F^2$，剩 $1/N$」；复算 $D_K^2MD_F^2/(D_K^2MND_F^2)=1/N$，被约的是 $D_K^2$、$M$、$D_F^2$。｜修复要求：把解答改为与正文一致的「分子分母约去 $D_K^2MD_F^2$，剩 $1/N$；第二项约去 $MND_F^2$，剩 $1/D_K^2$」。｜修复：｜复验：

- [重要·技术] 第 1 章正文（index.html:129）｜来源：本页式 [F2] / MobileNets §3.1｜位置：第 1 章「成本也由这两件事相乘决定」段末｜问题：成本 $D_K^2MND_F^2$ 对 $D_K$、$D_F$ 是平方依赖，二者翻倍成本变为 4 倍；「四个量里任何一个翻倍，成本翻倍」只在 $M$、$N$ 上成立。且本章问题把「四个量」界定为 $D_K/D_F/M/N$（答案把 $D_K^2$、$D_F^2$ 映射到空间侧），读者的「四个量」即这四个变量，结论对其中两个不成立。｜引文依据：页面「每个输出位置做 $D_K\cdot D_K\cdot M$ 次乘加、共 $N\cdot D_F\cdot D_F$ 个输出——四个量里任何一个翻倍，成本翻倍」；实算 $(2D_K)^2/D_K^2=4$、$(2D_F)^2/D_F^2=4$。｜修复要求：改写为与平方依赖一致的表述，例如「$M$、$N$ 翻倍成本翻倍，$D_K$、$D_F$ 翻倍成本变为四倍」或「四个因子 $D_K^2$、$M$、$N$、$D_F^2$ 中任一翻倍，成本翻倍」。｜修复：｜复验：

- [重要·技术] 第 4 章「本章问题」第 2 题解答（index.html:351）｜来源：本页代码注释 / [C5] / 本页正文｜位置：第 4 章本章问题「因果方向的卷积怎么实现『只看过去』？」解答折叠块的 summary｜问题：summary 写「右侧 padding 后截尾」，与正文及页内四处「左 padding」矛盾；右侧 padding 不产生因果性，摘要单独成立时会给出错误机制。｜引文依据：summary「解答：右侧 padding 后截尾」；同一解答正文「在序列左侧补 $D_K-1$ 个零再卷积，取前 $S$ 个输出」；index.html:304 代码注释「左 padding + 头部切片 = 因果」；index.html:334「左 padding + 头部切片实现因果」；[C5]（index.html:364）「以左 padding + 头部切片实现因果」。｜修复要求：summary 改为「解答：左侧 padding 后截尾」。｜修复：｜复验：

- [重要·技术] 来源与范围说明 [F5]/[F6]/[N2]（index.html:374、375、381）｜来源：页面自身来源段｜位置：来源与范围说明「公式与来源（F）」[F5]、[F6] 与「外部数字与实验条件（N）」[N2]｜问题：三处把实测数字的来源标为「research/ 实跑输出」，但 research/ 下的实测产物已从仓库移除，现存目录仅 draft-check.md / evidence.md / glossary.md / measured.md / outline.md / review-1.md / scope.md，无任何输出存档；按此路径无法定位核对，属指向已移除路径的来源标注。｜引文依据：[F5]「构造数字例实算一致（research/ 实跑输出）」；[F6]「research/ 实跑输出（构造示例）」；[N2]「…比值 0.126736、7.9 倍…research/ 实跑输出。」；research/measured.md 登记 `dw_page_code.out 464 B 运行输出存档`。｜修复要求：改为引用本页 research/measured.md 中登记的文件名，或改为「本页代码块实跑输出（可复现）」并保留代码块指纹一致，确保读者能定位或复现。｜修复：｜复验：

- [轻微·技术] 第 4 章「本章问题」第 2 题解答（index.html:352）｜来源：PyTorch Conv1d 语义｜位置：第 4 章本章问题第 2 题解答正文首句｜问题：「一维卷积默认两边对称 padding」不成立——`torch.nn.Conv1d` 的 `padding` 默认值为 0（不补零）；页内代码也必须显式写 `padding=3` 才产生两侧补零。｜引文依据：解答「一维卷积默认两边对称 padding」；页内代码 `torch.nn.Conv1d(1, 1, kernel_size=4, groups=1, bias=False, padding=3)`。｜修复要求：改为「指定 `padding` 时 PyTorch 在两侧对称补零」或删除该句。｜修复：｜复验：

- [轻微·表述] 第 4 章正文（index.html:292）｜来源：不适用｜位置：第 4 章「职能上，这个短卷积…」段末｜问题：同一括号批注重复出现两次，为拼接残留。｜引文依据：「…它也是局部相对位置信号的来源之一（结构解读，非源码直陈）（结构解读，非源码直陈）（模型层面的分析见 GLM-5.3-Flash 数据流页）。」｜修复要求：删除重复的一份「（结构解读，非源码直陈）」。｜修复：｜复验：

- [轻微·格式] 来源与范围说明 [F6]（index.html:375）｜来源：guides/concept/style-guide.md §6（来源编号双向对应）｜位置：来源与范围说明「公式与来源（F）」[F6]｜问题：[F6] 已定义但正文从未以 `<sup>[F6]</sup>` 引用（全文 `<sup>` 仅出现 [F1][C1][F2][N2][C1][F3][F3][C2][F4][F5][C3][N1][C5][N3][C6]），第 4 章实现该机制的代码块处无引用，双向对应缺失。｜引文依据：index.html:375 定义 [F6]；正文 sup 引用清单无 [F6]。｜修复要求：在第 4 章代码块（因果 1D depthwise 与 groups 语义）处补 `<sup>[F6]</sup>`，或将 [F6] 并入 [F5]。｜修复：｜复验：

- [轻微·格式] 来源与范围说明（index.html:361-365）｜来源：guides/concept/style-guide.md §6｜位置：「论断与来源（C）」小节｜问题：C 系列编号跳号，缺 [C4]，只列 C1、C2、C3、C5、C6。｜引文依据：[C1][C2][C3][C5][C6] 五条，无 [C4]。｜修复要求：重排为连续的 C1–C5，或补上缺失的 [C4] 条目。｜修复：｜复验：

- [轻微·格式] 第 3 章折叠代码块「观察重点」（index.html:271）｜来源：guides/concept/style-guide.md §11（数学符号一律 LaTeX、同一写法一致）｜位置：第 3 章代码折叠块「观察重点」句｜问题：「3×3 场景下」用 Unicode `×` 写数学表达式，与页面其余处 `$3\times3$`（如 index.html:65、243）写法不一致；同类未渲染运算符还有 index.html:137「空间侧两个因子 × 通道侧两个因子」（含 `=`）、index.html:183 图注标签「每个输出 = 邻域 × 全部输入通道的加权和」。｜引文依据：index.html:271「3×3 场景下主要开销移到了组合这一步」对比 index.html:243「$3\times3$ 核时 $1/9$ 主导」。｜修复要求：统一改为 `$3\times3$` 等 LaTeX 写法；图注标签内的 `=`、`×` 用 `$...$` 包裹或改写成不含数学含义的文字。｜修复：｜复验：

- [轻微·表述] 引言（index.html:65）｜来源：不适用（无来源支持的判断）｜位置：引言第二句｜问题：「深度可分离卷积由此成为移动端视觉模型的标配」把使用范围的判断写成结论，未给来源。｜引文依据：index.html:65「深度可分离卷积由此成为移动端视觉模型的标配」。｜修复要求：降级为明确标注的推断，或补 MobileNets 系列采用情况的来源；亦可改为限定表述（如「MobileNets 等移动端模型采用该结构」）。｜修复：｜复验：

- [轻微·格式] 第 4 章正文 / 来源说明（index.html:292、364）｜来源：guides/concept/check.md §2.2.6（页面链接）｜位置：index.html:292「（模型层面的分析见 GLM-5.3-Flash 数据流页）」、index.html:364「（GLM 数据流页交叉验证）」｜问题：跨页交叉验证依赖 `wiki/glm-5-3-flash-dataflow/index.html`（该页真实存在），但两处均以纯文本提及、未加链接，读者无法跳转核对。｜引文依据：index.html:292、364 的「GLM 数据流页」为纯文本；`wiki/glm-5-3-flash-dataflow/index.html` 存在。｜修复要求：把「GLM-5.3-Flash 数据流页」改为指向 `../glm-5-3-flash-dataflow/index.html` 的链接。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 7
- 处置：修复

核对通过的部分：MobileNets §3.1 的式 1/2/3/5、「8 to 9 times less computation」「small reduction in accuracy」、pointwise 定义原句与页面表述逐条一致；第 3、4 章两段代码块实跑输出与页面「预期输出」完全一致，成本比值与 $1/N+1/D_K^2$ 复算相符（0.126736、7.9x）；核心问题（4 条）与四个正文章节的「本章问题」均有解答折叠块，页面级答案均指明所在章节；`validate.py` 通过；`overview.html` 与 `index.html` 互链、`../kda/index.html` 前置页真实存在；无 Unicode 数学字符出现在 head/summary/标题中，结构图为内联 SVG 且公式由 `<foreignObject>` + KaTeX 承载。本轮无阻断问题，但上述 4 项重要问题需修复后复验。
