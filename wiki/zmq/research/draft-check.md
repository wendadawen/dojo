# ZeroMQ 初稿检查

## 输入版本

| 产物 | 状态 |
|---|---|
| scope.md | 完成。歧义已裁定（采纳"ZeroMQ 消息库"）；5 个学习目标；核心内容 K1–K10、辅助 B1–B5、扩展 E1–E4；前置知识映射结论为"不递归生成任何前置概念页" |
| evidence.md | 完成。18 条核心论断（C1–C18，跳过 C11 未进正文）、2 项线格式（F1、F2）、9 组实测（N1–N9）；3 项存疑项已列明处理方式 |
| outline.md | 完成。6 个正文章节 + 来源章节；贯穿示例为任务分发系统；讲解顺序依据与两处顺序取舍已记录 |
| glossary.md | 完成。术语按概念名称 / socket 与类型 / 路由与队列 / 消息与线格式 / 连接与运行时 / 符号 / 数值分七组登记 |

## 大纲落实

| 项 | 结果 |
|---|---|
| 章节数量与标题 | 落实。6 个正文 h2 与大纲一一对应，标题文字按大纲章节问题改写为陈述句，未增删章节 |
| 章节顺序 | 落实。裸 TCP 三障碍 → socket 类型路由 → 线格式 → 扇出与路由标识 → 队列满 → 边界 |
| 各章职责 | 落实。每章 h2 只回答一个学习目标；h3 小节与大纲的正文要点条目对应 |
| 学习目标 | 5 条全部落实，逐条见"目标覆盖检查" |
| 前置知识 | 落实。按 scope.md 第 2.4 节结论，未生成也未引用任何前置概念页；TCP 字节流特性、阻塞调用与 EAGAIN、十六进制与位序均在首次使用处就地说明 |
| 贯穿示例 | 落实。任务分发系统（1 分发端 + 3 工作进程 + 编号 1–6 任务 + 监控端）贯穿全部 6 章，每章增加新机制：轮询分发 → 抓包看帧 → 广播与回话 → 卡住时丢或等 → 库不负责的部分 |
| 误解处理位置 | 落实。M1、M2、M3、M5 放页面开头 misconceptions；M1 在第 1 章"名字里的 MQ 和 Zero"关闭；M4 在第 2 章"bind 与 connect"关闭；M2、M5 分别在第 5 章"mute 状态"与第 4 章"扇出与过滤"关闭；M6 在第 6 章"线程安全"关闭 |
| 适用边界 | 落实。集中在第 6 章，并在来源章节"简化条件及其限制"逐项展开 |
| 章间过渡 | 落实。第 1→2、2→3、3→4、4→5、5→6 各有一段过渡，均先总结本章结论再指出未解决的问题；第 6 章末不加过渡（下一节为来源） |
| 组件选择 | 落实。使用了 blockquote.meta、learning-goals、misconceptions、正文 h2/h3、验证问题、补充折叠块、可运行代码折叠块、ASCII 图示、对照表格、callout（yellow×2、purple×1）、来源与范围说明。未使用 context-box（大纲已说明不用）、数值示例折叠块（帧头手算与 HWM 对照表均直接进正文/表格，篇幅短无需折叠）、图片组件 |

## 目标覆盖检查

| 目标 | 负责章节 | 核对结果 |
|---|---|---|
| Q1：解决了裸 TCP 之上哪三个具体问题 | 第 1 章"裸 TCP 到消息库之间缺了什么" | 通过。三个 h3 分别给出消息边界、多对端分发、断线恢复，ASCII 图示给出"问题→机制"映射。与 scope.md 的完成答案一致 |
| Q2：一条消息去往哪个对端由谁决定 | 第 2 章"send 不指定收件人"+ 第 4 章"广播与定向回话" | 通过。第 2 章给出三种出站策略定义、公平队列定义、8 类型对照表与轮询实测；第 4 章补齐扇出与按标识定向。与完成答案一致 |
| Q3：队列满时丢还是阻塞取决于什么 | 第 5 章"队列满了：谁丢消息，谁阻塞" | 通过。队列粒度、HWM 定义与默认值 1000、mute 状态两类行为、丢弃发生在发布端四项均在正文。与完成答案一致 |
| Q4：一条消息在 TCP 线上长什么样 | 第 3 章"一条消息在 TCP 上的真实字节" | 通过。greeting 64 字节切分、READY 元数据、flags 三位定义、短/长帧长度字段、多段消息构成、手算例子均在正文。与完成答案一致 |
| Q5：不提供哪些保证 | 第 6 章"ZeroMQ 不做什么" | 通过。send 语义、I/O 线程、slow joiner、无持久化/确认/重投递、线程安全、应用责任清单六项均在正文。与完成答案一致 |

折叠块独占检查：7 个折叠块（5 个代码、2 个补充）全部收起后，逐目标复核——Q1 依赖的三障碍与映射图在正文；Q2 依赖的策略定义与对照表在正文，实测结果已在正文以 diagram 形式给出摘要；Q3 依赖的类型分工、HWM 定义、四组 HWM 对照表在正文；Q4 依赖的字段切分、flags 位表、实测十六进制、手算过程全在正文；Q5 依赖的六项限制全在正文。无目标被折叠块独占。

## 代码运行

全部代码在 `/Users/wendadawen/.workbuddy/binaries/python/envs/default/bin/python`（Python 3.13.12，pyzmq 27.1.0，libzmq 4.3.5）下实际执行。脚本保存于 `/tmp/zmqlab/`。

| 页面位置 | 脚本 | 退出码 | 输出与页面描述一致 |
|---|---|---|---|
| 第 2 章"代码：PUSH 与 PULL 的轮询分发" | p1_push.py | 0 | 一致。`worker0 took [1, 4] / worker1 took [2, 5] / worker2 took [3, 6]` |
| 第 3 章"代码：抓取 ZMTP 握手与帧的原始字节" | p2_wire.py | 0 | 一致。117 字节；greeting 十六进制、signature/version/mechanism/as-server 分段、三个帧的 flags/size/body 逐项与页面预期输出相同 |
| 第 4 章"代码：用手写对端验证过滤发生在发布端" | p4_filter.py | 0 | 一致。连接上只出现 READY 命令帧与两条 `A-match-*`，`B-nomatch` 缺席 |
| 第 4 章 ROUTER 信封（正文 diagram） | p3_router.py | 0 | 一致。`[b'worker-7', b'done: job 3']` 与 `[b'client-1', b'', b'status?']` |
| 第 5 章"代码：PUB 丢弃与 PUSH 阻塞的对照" | p5_hwm.py | 0 | 一致。`PUB accepted 100/100, SUB received 5 -> dropped 95` 与 `PUSH accepted 5/100, PULL received 5 -> lost 0` |
| 第 5 章 HWM 四组对照表（正文表格） | p6_hwm_side.py | 0 | 一致。5/200、200/200、5/200、200/200 |
| 第 2 章轮询跳过满队列（正文 diagram） | p7_skip.py | 0 | 一致。`stuck worker took: [0, 2]` 与 `idle worker took: [1, 3, 4, 5, 6, 7, 8, 9]`。连续运行两次结果相同 |
| 第 6 章"代码：slow joiner 与 connect 先于 bind" | p8_boundary.py | 0 | 一致。`['late-0', 'late-1', 'late-2']` 与 `b'queued-before-bind'` |

另有 p9_req.py（REQ 连续两次 send 报 EFSM，errno=156384763）已运行通过，但该论断（evidence.md 的 C11）未写入正文——大纲未安排 REQ 状态机章节，故不纳入。

页面中的 libzmq 源码片段（第 2 章"补充：轮询跳过满队列"）不是可运行代码，为 `src/lb.cpp` 的节选，标为 `language-text`，未声称可执行。

## 机械检查

```
$ python3 .dojo/scripts/validate.py wiki/zmq/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/zmq/index.html

$ python3 .dojo/scripts/validate.py wiki/zmq/overview.html
validation ok: /Users/wendadawen/code/dojo/wiki/zmq/overview.html
```

补充的机械检查：

- 占位符与组件标记：`【…】` 0 处；`@content` / `@component` / TODO / TBD 均已清除（validate.py 覆盖）。
- id 唯一性与锚点：42 处 id，无重复；无同页锚点指向缺失 id（validate.py 覆盖）。来源章节的 6 个 h3 补加了 id，以便侧边目录生成稳定锚点。
- 标签配对：details 7/7、summary 7/7、table 3/3、thead 3/3、tbody 3/3、tr 19/19、td 72/72、th 12/12、div 24/24、section 2/2、pre 24/24、code 187/187、ul 19/19、li 73/73、p 120/120、blockquote 1/1，全部平衡。
- 本地资源：katex.min.css / katex.min.js / auto-render.min.js / prism 系列均为 `../../libs/` 相对路径，HTTP 200 可达。
- 互链：index.html 有 `href="overview.html"`（导航栏"概览"）；overview.html 有两处 `href="index.html"`（顶部导航"完整说明 →"与文末"完整说明"）。

## 公式渲染与交互

- 起本地静态服务（`python3 -m http.server 8899`），index.html、overview.html、libs/katex.min.js 均返回 200。
- 正文中的行内公式只有 `$2^{63}-1$`，出现 2 处（第 3 章 flags 位表的 LONG 行、来源章节"外部数字与实验条件"）。用 libs 下的 katex.min.js 以 `throwOnError: true` 渲染该式，成功返回 1091 字节 HTML，无语法错误。
- `dojo:summary` 中不含 `$`，不触发 validate.py 的公式定界符检查；`description` 为纯文本，无 `$`。
- 折叠交互：7 个 `<details>` 均使用外壳提供的 `details` / `details.code-details` 类，summary 前缀符合 style-guide 的三种（`代码：` 5 个、`补充：` 2 个），无 `展开：`（本页未安排长手算折叠块）。
- 目录与锚点：外壳脚本按 `body > h2, body > h3` 生成侧边目录；本页 6 个正文 h2 + 1 个来源 h2、24 个 h3 全为 body 直接子元素，未被 `<section>` 包裹（learning-goals 与 misconceptions 两个 section 内的 h2 是组件自带标题，外壳选择器为直接子元素故不进目录，符合模板设计）。
- 代码块复制按钮：5 个 `.code-block` 容器由外壳脚本自动挂载复制按钮。
- 章节折叠按钮与 j/k 快捷键：依赖 `body > h2`，本页结构满足。

## 写作偏差

| 项 | 处理 |
|---|---|
| 大纲第 3 章计划用"数值示例折叠块"承载 `01 03 61 62 63 00 01 64` 的逐字节手算 | 局部修正。该手算仅 4 句，折叠反而打断阅读，改为直接写入正文。折叠块的职责（承载更长手算）在此不成立，未违反"正文必须包含"的要求 |
| 大纲第 5 章计划用"数值示例折叠块（构造示例）"承载四组 HWM 对照 | 局部修正。改为正文中的对照表格，因为这组数据是本章核心结论（丢弃发生在发布端）的直接证据，按 write.md 属"回答学习目标所需的结论"，不应折叠 |
| 大纲第 5 章计划的"补充折叠块：为什么实际容量 ≠ HWM" | 局部修正。改为 callout-yellow 直接放在正文 HWM 定义之后。理由：该限制会改变读者对随后实测数字的解读，属"会改变结论的条件和限制"，按 write.md 必须在正文 |
| evidence.md 的 C11（REQ 的 EFSM 状态机） | 不纳入。大纲未安排 REQ 状态机章节；C12（REQ 空分隔帧）已在第 4 章通过 ROUTER 信封实测落地，足以说明 REQ 的线上格式。C11 保留在 evidence.md 备查 |
| 需要返回规划的偏差 | 无。未发现缺少核心事实、证据或结构的情况；未新增学习目标、未更换贯穿示例、未改变前置知识映射 |
