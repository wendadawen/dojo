# DFlash 2 审查记录（第 2 轮）

- 页面版本：index.html 工作树哈希 `7331c772f33d7e76e20ee045d95a99f27900beeb`
- 审查时间：2026-08-19 18:44
- 审查者：独立子代理（第 2 轮，未参与写作与第 1 轮审查/修复）
- 已完整阅读章节（按顺序）：index.html 全文（meta 区、引言、核心问题、1 两个剩余问题、2 路径选择器、3 两抽头卷积、4 组合效果、5 端到端与边界、来源与范围说明全部六小节）；overview.html 全文
- 外部来源核对：博客快照 `/tmp/dflash-research/dflash2-blog.txt`（Table 1–5、Figure 2/3/5 数据、正文引文逐条定位）；HF 模型卡 `/tmp/dflash-research/qwen38-modelcard.md`（YAML 字段、Acceptance Length 表、吞吐并发 1/8/32 全 45 格逐格核对）
- 机械验证：`.dojo/scripts/validate.py` 对 index.html 与 overview.html 均返回 `validation ok`；前置概念链接 `../dflash/`、`../block-diffusion/`、`../speculative-decoding/` 与 `../../libs/` 资源均存在；C1–C16、F1–F2、N1–N10 正文 sup 引用与来源章节双向对应齐全

## 问题

### 重要

- [重要·来源] overview.html「是什么」段（`<p>DFlash 2 是 Inco AI…Apache 2.0）`）：Apache 2.0 被笼统加在两份草稿器权重上（"Qwen3.8-27B-DFlash2、Muse-Glimmer-30B-DFlash2，Apache 2.0"），但允许来源只支持 Qwen 草稿器的许可；index.html 核心问题 1 解答已正确写"Muse 草稿器许可未在允许来源中核对"，概览页与之矛盾｜引文依据：模型卡 YAML `license: apache-2.0` 仅属 `incoai/Qwen3.8-27B-DFlash2`；博客仅 "one for Meta's Muse Glimmer"，无 Muse 草稿器许可信息｜修复要求：改为「Qwen3.8-27B-DFlash2（Apache 2.0）、Muse-Glimmer-30B-DFlash2（许可未核对）」或同等限定表述｜修复：｜复验：
- [重要·来源] index.html meta blockquote（`:749`）："DFlash 论文（Chen, Liang, Liu, ICML 2026, arXiv:2602.06036v2）"——arXiv 编号 2602.06036v2 在允许来源中定位不到｜引文依据：模型卡 Citation 仅有 `@inproceedings{chen2026dflash, booktitle = {ICML}, year = {2026}}`，无 arXiv 字段；博客全文无 arXiv 编号｜修复要求：删除 "arXiv:2602.06036v2"，保留 "Chen, Liang, Liu, ICML 2026"；或核对到可定位来源后再恢复｜修复：｜复验：
- [重要·来源] index.html 4.2 正文（`:1108`）与第 4 章本章问题 2 解答（`:1141`）：「两个目标模型上 DFlash 2 都拿到完整 token 以上的领先」「都拿到相对各基线的完整 token 以上领先」——把「>1 token 领先」扩大到所有基线。实测 Qwen3.8-27B 上对 MTP 仅 +0.52（4.80−4.28）；博客只对 DSpark 作出该论断｜引文依据：博客 "on both models, DFlash 2 averages more than a full token ahead of DSpark. It also beats each model's official drafter"（>1 token 仅限 DSpark；对官方草稿器只说 beats）；N6/N7 表 4.80 vs 4.28 = +0.52｜修复要求：改为「对 DSpark 领先完整 token 以上（+1.18 / +1.22），对 MTP +0.52、对官方 DFlash +1.26」之类的具体差距表述，删除「相对各基线的完整 token 以上领先」｜修复：｜复验：
- [重要·技术] index.html 第 1 章末段（`:844`）：「真实数据里位置 2、3 的 Recall@1 已从首位下降（80.3%/79.4% vs 85.4%，见 N2）」——位置标签与本页 Recall 表（0 起始，`:809`）冲突：按该表 80.3% 在位置 1、79.4% 在位置 2，位置 2、3 应为 79.4%/78.3%｜引文依据：博客 Table 1 表头 "0 1 2 3 4 5 6"，Recall@1 行 85.4%(0) 80.3%(1) 79.4%(2) 78.3%(3)｜修复要求：统一位置编号基准后改写（如「位置 1、2（0 起始）的 Recall@1 已从首位 85.4% 降到 80.3%/79.4%」），并保证与第 1 章 Recall 表、2.4 构造示例（1 起始）的编号约定不冲突；建议在 Recall 表或首次混用处说明 0 起始、构造示例 1 起始｜修复：｜复验：
- [重要·来源] index.html 1 章正文（`:805`）：「块内逐位置的『首位命中率』Recall@1 与『首位 top-16 命中率』Recall@16」——两个术语均为残留错误：Recall@1/Recall@16 是逐位置指标而非「首位」指标；同页表行标签（`:811`）已改为「Recall@1（top-1 正确率）」，正文与表标签不一致｜引文依据：博客 Table 1 caption "Recall@1 (how often the top pick is right) and Recall@16 … at each draft position"；页面 `:811` 已用「top-1 正确率」｜修复要求：正文改为「逐位置 top-1 正确率 Recall@1 与 top-16 命中率 Recall@16」，删除两处「首位」限定词｜修复：｜复验：
- [重要·来源] index.html 第 5 章本章问题 2 解答（`:1211`）：「数字全部来自厂商自测环境（H200/B200、SGLang、推荐采样）」——B200 残留。允许来源中所有 Qwen3.8-27B 吞吐数字仅在 H200 上测得；B200 仅在博客背景段以 "Blackwell" 出现且属 DFlash 一代生态句｜引文依据：模型卡 "Runtime: SGLang on one NVIDIA H200"；`：1193` 正文已只写「单卡 H200」｜修复要求：删除 "/B200"，与正文 `:1193`、简化条件②（`:1254`「单卡 H200」）一致｜修复：｜复验：
- [重要·来源] index.html 简化条件⑤（`:1254`）：「社区初步复测（llama.cpp PR 构建、RTX 5090 等）的数据点有限且与厂商环境不同」——「社区初步复测」「RTX 5090」在允许来源中均定位不到，属无来源论断残留｜引文依据：不适用（博客与模型卡均无 RTX 5090 或社区复测数据）｜修复要求：整句改为「HF 模型卡 N8 的并发表是厂商自测，未给出第三方独立复现」，删除社区复测举例｜修复：｜复验：
- [重要·来源] index.html 第 5 章本章问题 1 解答（`:1204`）：「Muse Glimmer 批 1 区间 3.1–4.6×」——「批 1」限定残留。博客给出 3.1–4.6× 时未注明任何并发条件；5.1 正文（`:1179`）已正确改为「博客未注明并发条件」，解答未同步｜引文依据：博客 "That translates into 2.7–3.4× the throughput … on Qwen3.8-27B, and 3.1–4.6× on Muse Glimmer"（无并发标注）；`：1179`「（博客未注明并发条件，模型卡按任务与并发细分）」｜修复要求：删除「批 1」，改为「Muse Glimmer 区间 3.1–4.6×（博客未注明并发条件）」，与 `:1179` 一致｜修复：｜复验：
- [重要·来源] index.html 引言（`:756`）、核心问题 1 解答（`:767`）、overview.html「是什么」段：草稿器仓库名「Muse-Glimmer-30B-DFlash2」（含 30B 规格）在允许来源中定位不到——博客链接文字仅为 "one for Meta's Muse Glimmer"，模型卡只收录 Qwen3.8-27B-DFlash2｜引文依据：博客 "We are releasing two DFlash 2 drafters today: one for Qwen3.8-27B and one for Meta's Muse Glimmer"；模型卡仓库名 `incoai/Qwen3.8-27B-DFlash2`｜修复要求：三处统一改为「Muse Glimmer 草稿器」（或「Muse Glimmer 的 DFlash 2 草稿器」），删除未核对的完整仓库名与 30B 规格；核对到真实仓库后可恢复｜修复：｜复验：
- [重要·可读性] index.html 核心问题 1 解答（`:767`）：答案末尾未指明完整论证所在章节，与其余 4 条核心问题解答（均有「详见『N. …』」）不一致，违反 style-guide 第 9 节对页面级答案的要求｜引文依据：不适用（`：774`「详见『2. 路径选择器…』」、`:781`、`:788`、`:795` 均有）｜修复要求：在解答末尾补「详见『1. 两个剩余问题——候选池里有答案、块末端在漏气』」｜修复：｜复验：
- [重要·格式] index.html「外部数字与实验条件」小节（`:1245`）：出现两个 N10 定义并列——"N10（Qwen3.5-4B 逐基准全表 N4 已含；本号原为 DFlash 论文 N10，与本页不同主题，不引用）。N10（DFlash 接受长度提升 16–25%）…"。第一个 N10 是跳号修复的历史残留说明，与有效定义并存破坏编号唯一性｜引文依据：不适用｜修复要求：删除第一个 N10 说明句，只保留 "N10（DFlash 接受长度提升 16–25%）：博客 \"Across benchmarks the gain runs 16–25%\""｜修复：｜复验：

### 轻微

- [轻微·格式] index.html 第 1 章总结表（`:825`-`:826`）与本章问题 2 解答（`:859`）：残留「路径选择器（第 2 章）」「两抽头卷积（第 3 章）」「（见第 3 章）」章号引用，其余位置已按第 1 轮统一为章名（如 `:905`、`:958`）｜引文依据：不适用｜修复要求：三处改为章名引用（『2. 路径选择器——在候选之间打分，而不是重新预测』『3. 两抽头卷积——给块末端补上「看见前一位」的通道』）｜修复：｜复验：
- [轻微·图示] index.html 3.2 SVG（`:1031`）：图内文字「首位前驱 = 上一周期目标 token」与正文（`:986`）及 figcaption（`:1033`）的「已验证 token」术语不一致（目标模型输出须经验证才是已验证 token）；另外首位前驱的虚线终点是「位置 1」框（`:1028`），而其余前一位虚线均画到 Conv 输出框（`:1025`、`:1026`），画法不统一｜引文依据：博客 "The first position reads the last verified token's representation"｜修复要求：图内文字改为「首位前驱 = 上一周期已验证 token」；将 `:1028` 虚线终点改到 Conv(x)_1 输出框，或在 figcaption 明确说明两种虚线的差异已消除｜修复：｜复验：
- [轻微·公式] index.html 2.3 表头（`:896`「接受长度（T=0 / T=1）」）与本章问题 2 解答（`:957`-`:958`「T=0 接受长度」）：T 为数学变量未包 `$...$`，且 T 指采样温度未在首次出现处说明｜引文依据：博客 Table 2 列头使用 LaTeX $T=0$/$T=1$；Table 3 caption "temperature 1.0"｜修复要求：改为 `$T=0$`/`$T=1$` 并在 2.3 首次出现处补「$T$ 为采样温度」｜修复：｜复验：
- [轻微·格式] index.html 5.1 表并发 32 行（`:1167`）：DFlash 2 仅 GSM8K 格（1.45×）加粗，同行的 1.30/1.16/1.25/1.01 四格同为该列所有方法最优却未加粗；并发 1、8 两行 DFlash 2 全行加粗——同表加粗规则不一致｜引文依据：模型卡并发 32 表中 DFlash 2 五列均为加粗最优（**1,922.5 (1.45×)** 等）｜修复要求：明确本表加粗规则（建议「每列最优加粗」）并统一执行，或全表不加粗｜修复：｜复验：
- [轻微·格式] index.html 来源章节 h3（`:1244`）：「外部数字与实验条件」缺 style-guide 固定命名的「（N）」后缀（同节其两个 h3 为「论断与来源（C）」「公式与来源（F）」）｜引文依据：不适用｜修复要求：改为「外部数字与实验条件（N）」｜修复：｜复验：
- [轻微·可读性] index.html 2.4 走路径段（`:936`）：括号「bonus 即验证顺带产出的下一 token；本节不区分验证 bonus 与拒绝修正 token 的来源」悬空——本节走路径内容未涉及 bonus 或验证来源话题；且 4.27/6.79 首次出现处（`:817`）未说明接受长度含验证器下一 token 的口径｜引文依据：博客 Table 1 caption "Acceptance length includes the verifier's next token"｜修复要求：删除 2.4 的悬空括号，把口径说明移到 `:817`「接受长度」首次出现处（如「接受长度（按博客口径含验证器顺带产出的下一 token）」）｜修复：｜复验：
- [轻微·可读性] index.html 第 1 章构造示例表（`:837`-`:839`）：「独立 top-1」列括号数字为 $U$ 值、「路径选择器输出」列括号数字为 $S$ 值，含义未在表内或表注定义，需到 2.4 才有定义｜引文依据：不适用｜修复要求：表注补一句「括号内数字：独立 top-1 列为该候选的 $U_t$，路径选择器列为选中路径的 $S_t$（2.4 定义）」｜修复：｜复验：
- [轻微·来源] overview.html「关键结论与边界」（`:63`）：「所有数字来自厂商自测（HF 模型卡）环境」——概览页数字多数来自博客（接受长度 5.97/4.80/5.70、16–25% 等），括号限定不准确｜引文依据：博客 Table 3/4/5（接受长度）；模型卡仅覆盖 N6/N8｜修复要求：改为「所有数字来自厂商自测（官方博客与 HF 模型卡）」｜修复：｜复验：
- [轻微·可读性] index.html 核心问题 4 解答（`:788`）：MTP 首次出现未作任何解释（多 token 预测层/模型内置投机路径）；DSpark 首次出现（`:774`）也早于 2.3 节的解释｜引文依据：模型卡 "Qwen3.8's built-in seven-token MTP"；博客 "MTP ships with the model"｜修复要求：核心问题 4 解答首次出现处补「MTP（Qwen3.8 内置的多 token 预测投机路径）」；`:774` 首次出现补「DSpark（用串行修正头改写词表分布的对比方法）」｜修复：｜复验：
- [轻微·可读性] index.html 引言 callout（`:753`）：核心开销概念「循环延迟」首次出现未说明所指（draft–verify 周期的延迟）｜引文依据：博客 "added draft–verify cycle latency"；Table 2 caption "added draft–verify cycle latency"｜修复要求：首次出现处补括号说明「循环延迟（一次 draft–verify 周期增加的延迟）」｜修复：｜复验：
- [轻微·来源] index.html 2.3 段（`:892`）：「路径长度等于块大小（线性而非按词表大小），所以『选比预测便宜』<sup>[N1]</sup>」——该句为机制推理而非 Table 2 的开销数字，引用编号错位（对应内容在 C5/C6）｜引文依据：博客 "The only sequential work is the final walk over precomputed scores"（C6）；"sequential heads that rewrite each position's full-vocabulary distribution"｜修复要求：把 sup 引用改为 [C5, C6]，N1 仅保留在随后数字对比句｜修复：｜复验：
- [轻微·格式] index.html 2.3 段末（`:905`）：段尾「）。</p></p>」存在重复的 `</p>` 闭合标签｜引文依据：不适用｜修复要求：删除多余的 `</p>`，保留一个闭合标签｜修复：｜复验：
- [轻微·公式] index.html 5.1 表全部加速比单元格（`:1158`-`:1167`，45 格）及 3.3 表「1×」「3×」等：乘号使用 Unicode 字符「×」未包 LaTeX，与同页 meta/overview 的 `$\times$` 写法不一致｜引文依据：不适用（overview `:62` 用 `$2.27\!\text{–}\!2.85\times$`）｜修复要求：统一写法——表格单元格改为 `$2.59\times$` 形式，或将「×」统一视为文本并在 overview 同步；二选一全站一致｜修复：｜复验：
- [轻微·可读性] index.html 第 1 章（`:817`）「纯选择空间就有约 1.6× token 的余量」：「1.6× token」措辞含糊——6.79/4.27≈1.6 是倍数，token 差为 2.52，易被读成「1.6 个 token」；`:852` 同句式｜引文依据：博客 "lift the acceptance length from 4.27 to 6.79"｜修复要求：改为「约 1.6 倍（6.79/4.27）的选择余量，折合 2.52 个 token」｜修复：｜复验：

## 已核验无误的关键项（第 1 轮修复抽查）

- 构造示例可手算：U 表（U_2(decoding)=0.55 > U_2(is)=0.40 → 独立 top-1 选 decoding）与 S 表自洽（S_2(decoding,decoding)=0.55−0.30=0.25 < S_2(decoding,is)=0.40+0.20=0.60 → 选 is；S_3(is,parallel)=0.80+0.05=0.85 > S_3(is,now)=0.18+0.02=0.20 → 选 parallel），第 1 章表与 2.4 数值一致
- Table 2 行标签「DFlash + 路径选择器（不含卷积）」与博客 "path selection alone (no convolution)" 一致
- 三目标均值 5.97/4.80/5.70 及全部 30 格逐基准数字与博客 Table 3/4/5 逐格一致，均值列可复算
- N8 全 45 格吞吐倍数与模型卡并发 1/8/32 三表逐格一致，基线 tok/s、采样与 reasoning 条件一致
- SVG：Conv(x)_1/2/3 在 foreignObject 中由 KaTeX 渲染；实线（dg-accent）=自身抽头、虚线（dasharray）=前一位抽头，figcaption 有定义；6 条线段无重复
- 84.88、S_2/S_3 下标、Inco AI、Apache 2.0（index 限定 Qwen）、[C15] 替换厂商宣称、C13 引用、3.3 表参照系说明、5.1 表「略」改写、2.4 引言措辞、N10 正文引用（16–25%）均已落实
- C1–C16、F1–F2 每条在正文有 sup 引用；每个正文 N 引用在来源章节有定义（N10 重复定义除外，见重要第 11 条）
- 来源六小节齐全；两级问题块命名与「解答：」前缀齐全；validate.py 两页通过

## 结论

- 统计：阻断 0 / 重要 11 / 轻微 14
- 处置：修复。无阻断问题；核心结论（两组件机制、三目标接受长度、并发边界、无损性）与来源一致。重要问题集中在两类：①第 1 轮修复在折叠块解答、概览页、meta 区的残留（B200、RTX 5090、批 1、首位命中率、章号、核心问题 1 缺章节指引、双 N10、`</p></p>`），需按上述逐条清理并同步所有镜像位置；②两处来源论断扩大（Apache 2.0 覆盖 Muse、「完整 token 以上领先」扩大到 MTP）与两处不可定位信息（arXiv 编号、Muse 仓库名 30B 规格），需删除或改为具体差距/限定表述。修复后重跑 `.dojo/scripts/validate.py` 并进入第 3 轮独立审查。
