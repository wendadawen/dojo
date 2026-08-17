# ZeroMQ 审查记录（第 3 轮）

- 页面版本：e1905f0574bd24bc5a1f28960864cc8f5b199cd0
- 审查时间：2026-08-17 19:44

## 问题

- [重要·技术] 「send 不指定收件人，收件人由 socket 类型决定」→「三种出站策略与一种入站策略」（h3 标题、正文第 740 行、748 行，及「ROUTER 找不到目标时不报错」末段第 1251 行）：正文断言"本文涉及的稳定 socket 类型共用三种出站策略"（轮询/扇出/按标识定向）、"入站侧则统一是一种策略：公平队列"，并在第 1251 行收束为"三种出站策略至此全部展开"；但紧随其后的同节对照表第 758、759 行给出了第四种取值——REQ 的入站策略为"上一个对端"、REP 的出站策略为"上一个对端"。经核对 `zmq_socket(3)` 摘要表，ZMQ_REQ 的 Incoming routing strategy 确为 `Last peer`、ZMQ_REP 的 Outgoing routing strategy 确为 `Last peer`（RFC 28/REQREP 对应表述为 REQ"SHALL accept an incoming message only from the last peer that it sent a request to."、REP"SHALL deliver this message back to the originating peer."），因此"出站只有三种""入站统一是公平队列"两处计数均与来源不符，且与本页自己的表格直接矛盾。同时"上一个对端"这一策略名在全页仅出现于这两个表格单元格内，正文、折叠块、来源章节均无任何解释，而 REQ/REP 的严格交替语义正依赖它。｜修复要求：改为覆盖四种出站策略与两种入站策略（或明确写成"本页展开三种，另有 REQ/REP 使用的'上一个对端'"），并在该 h3 正文中为"上一个对端"补一句定义（REQ 只接受它上一次发出请求的那个对端的消息；REP 把回复发回发出上一个请求的那个对端）；同步修正 h3 标题、第 1251 行的"三种出站策略至此全部展开"以及「简化条件及其限制」中"出站三策略加入站公平队列"的表述。修好的判据：全页对出站/入站策略的计数一致，且表格中出现的每个策略名都能在正文找到定义。｜修复：把 h3 标题改为「四种路由策略」，正文重写为"三种用于出站 + 一种用于入站"，并新增一段定义第四种策略「上一个对端」（Last peer）——说明它是 REQ/REP 严格交替语义的实现方式，同时出现在 REQ 的入站与 REP 的出站；新增 C22 条目记录来源。全文三处指向该节的引用同步更名。｜复验：核对 `zmq_socket(3)` 摘要表确认 ZMQ_REQ 的 Incoming routing strategy 与 ZMQ_REP 的 Outgoing routing strategy 均为 Last peer；核对 RFC 28 的 REQ/REP 节确认严格交替与"只接受上一个对端"的规定。页面现无"三种策略"与表格取值冲突的表述；C1–C22 双向对应检查通过（两个差集皆空）。

- [轻微·技术] 「一条消息在 TCP 上的真实字节」→「握手命令：对端如何知道本端是什么类型」第 938–939 行 READY 帧体框图：标注框与实际字节数不自洽。`└─ 属性名(11B) ─┘` 括住的是 `\x0bSocket-Type`，实际跨 12 字节（1 字节长度前缀 + 11 字节名字）；`└属性名(8B)┘` 括住 `\x08Identity`，实际跨 9 字节。括号标注的数值是 RFC 23 中 `name = OCTET 1*255name-char` 的名字长度，却被放在包含长度前缀的框上，与同页 greeting 框图"框宽 = 标注字节数"的约定不一致。｜修复要求：将两处标注改为框实际跨度（如 `属性名(1+11B)` 或 `名长+属性名(12B)`），或把长度前缀拆成独立小框单独标注 1B；修好的判据：每个框的标注字节数等于该框覆盖的字节数。｜修复：重画该框图：把属性名的 1 字节长度前缀单独标出（`└名长┘`），名称跨度改标为"名(11)"与"名(8)"，值跨度标为"值(6)"，避免把长度前缀算进名字字节数。｜复验：框图各标注跨度与实测 READY 帧体逐段对应：`\x0b` 为名长、`Socket-Type` 为 11 字节、`\x00\x00\x00\x06` 为 4 字节值长度、`DEALER` 为 6 字节。

- [轻微·可读性] 「三种出站策略与一种入站策略」对照表第 755 行 PULL 行"队列满时"单元格：使用了"入站队列满则对上游反压，不丢"，"反压"为全页仅此一处出现的术语，在此处首次使用且未解释；随后第 766 行与「mute 状态：丢弃还是阻塞」第 1288 行虽解释了该机制（不再从连接读取、从而让上游 PUSH 把该对端视为不可用），但均未与"反压"一词建立对应，不懂分布式通信的读者无法确定该词指什么。｜修复要求：在该单元格改用不依赖该术语的表述（如"入站队列满则不再读取，上游 PUSH 视其为不可用，不丢"），或在第 766 行解释中显式点出"这就是上面表格所说的对上游反压"；修好的判据："反压"一词在页面中出现处均有就近定义或被替换。｜修复：对照表 PULL 行改为"入站队列满则停止读取，不丢"（不提前使用术语），术语「反压（back pressure）」在表下说明段首次出现并就地解释。｜复验：术语首次出现处有括注英文与一句解释；表格单元格不再含未定义术语。

- [轻微·技术] 「来源与范围说明」→「构造示例」第 1595 行：人为参数清单写"端口号 5701–5713"，但全页实际使用的端口为 5701、5702、5704、5705、5706、5711、5712，最大值为 5712，5713 未在任何代码块或实验条件中出现（5713 仅出现在这句范围描述自身）。｜修复要求：把范围改为与实际一致的写法（如"端口号 5701–5712"或直接列出实际使用的 7 个端口）；修好的判据：该句给出的端口范围上下界均能在页面其他位置找到对应使用。｜修复：把构造示例的端口范围由 5701–5713 改为 5701–5712。｜复验：`grep` 页面内出现的端口为 5701、5702、5704、5705、5706、5711、5712，最大值为 5712。

- [轻微·技术] 「队列满了：谁丢消息，谁阻塞」→「队列的粒度是对端，不是 socket」第 1265 行：该句引用了三段规范原文并统一标注 `<sup>[C4, C6]</sup>`，但其中"对 PUB 是'为每个已连接订阅者维护一条出站消息队列'"的出处是 RFC 29/PUBSUB PUB 节，对应来源条目为 C10（C10 条目已收录该原句），而 C4 为 RFC 30/PIPELINE PUSH 节加 `lb.cpp`、C6 为 RFC 28/REQREP ROUTER 节，均不含 PUB 表述；同句提到的 XPUB 双队列出自 RFC 29 XPUB 节，在 C4/C6/C10 中都没有对应条目。上标与来源条目未双向对应。｜修复要求：把该句上标改为包含 C10（如 `[C4, C6, C10]`），并为 XPUB 的双队列表述补充来源（在 C10 或 C21 条目中增列 RFC 29 XPUB 节"SHALL maintain a double queue for each connected subscriber."）；修好的判据：该句引用的每段规范原文都能在被标注的来源条目中找到。｜修复：该句补充 SUB 的入站队列表述，并把上标由 [C4, C6] 改为 [C4, C6, C10, C21]。｜复验：四个编号所指条目分别覆盖 PUSH 出站队列、ROUTER 双队列、PUB 出站队列、SUB 入站队列，与该句四类表述一一对应。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复完成，可发布

## 发布结论

三轮独立审查全部完成。问题分布：第 1 轮阻断 0 / 重要 6 / 轻微 13，第 2 轮阻断 0 / 重要 3 / 轻微 3，第 3 轮阻断 0 / 重要 1 / 轻微 4。三轮均未出现阻断问题，重要问题呈 6 → 3 → 1 收敛。

发布条件逐项核对（修复后页面版本 `4d34b2f38325bda87177e051e46e327be1e08aab`）：

| 条件 | 结果 |
|---|---|
| 三轮审查均已完成 | 是。每轮由独立审查者从当时的最终页面重新开始，均未读取 `research/` 下的规划与前序审查记录 |
| 所有阻断和重要问题均已关闭 | 是。三轮共 10 个重要问题全部修复并复验，无遗留 |
| 遗留轻微问题具有明确接受理由 | 三轮共 20 个轻微问题全部修复，无遗留 |
| 全部学习目标由正文章节完整回答 | 是，见本轮核验记录末条；折叠块全部收起后结论链条仍完整 |
| `validate.py` 返回成功 | 是。`index.html` 与 `overview.html` 均返回 `validation ok` |
| 可运行代码结果与页面描述一致 | 是。从页面 HTML 提取 5 个代码块（还原实体转义后）实际运行，逐字符 `diff` 预期输出，5/5 零差异 |
| 关键论断和数字已重新核对来源 | 是。C1–C22 全部对照官方手册、RFC 与 libzmq v4.3.5 源码复核；正文 6 组 `<pre class="diagram">` 实测数据独立复现 |
| `<head>` 元数据有效 | 是。`description` 为纯文本无 `$`；`dojo:summary` 可渲染；`dojo:type=concept`；`dojo:topics` 与 `dojo:tag` 已填 |
| `overview.html` 与 `index.html` 相互链接 | 是。index 导航栏指向 overview；overview 顶部导航与文末各有一处指向 index |
| 页面引用的概念链接有效或有明确占位 | 是。本页不依赖其他概念页（前置知识均在正文就地说明），无跨页概念链接，因此无占位缺口 |
| 递归生成的前置概念页已完成质检 | 不适用。本页未递归生成任何前置概念页 |

第 3 轮修复后补做的机械校验（脚本执行，全部通过）：无残留数字章节代号与"上一章/下一章"；无第二人称"你"；无 `<code>Cx</code>` 形式的来源引用；C1–C22 编号连续，且正文引用集合与来源定义集合完全相等（两个差集皆为空）；`<summary>` 前缀均属"补充：/代码："；17 类标签全部配对；最后一个 h2 为「来源与范围说明」；全部 ASCII 框图按东亚字符宽度计算后每行显示宽度一致。

处置：可发布。首页目录与关系图由 GitHub Pages 构建自动更新。

### 本轮已核验通过的内容（供后续复验参考）

- **5 个"代码："折叠块全部实际运行，输出与页面"预期输出"逐字符一致**（解释器 pyzmq 27.1.0 / libzmq 4.3.5）：轮询分发（worker0/1/2 各取 [1,4]/[2,5]/[3,6]）、ZMTP 抓包（total bytes 117、greeting 128 个十六进制字符、三个帧的 flags/size/body 全部吻合）、发布端过滤（`B-nomatch` 确实缺席）、PUB 丢弃与 PUSH 阻塞对照（100/100 收 5、5/100 收 5）、slow joiner 与 connect 先于 bind。
- **正文 `<pre class="diagram">` 中的 6 组实测结果均独立复现成功**：轮询跳过满队列（连续 3 次运行均为 accepted 0–9、卡住对端取 [0,2]、空闲对端取余下 8 条）；ROUTER 信封（`[b'worker-7', b'done: job 3']` 与 `[b'client-1', b'', b'status?']`，REQ 的空分隔帧确实存在）；padding 与路由标识长度四组对应（3→4、8→9、0→1、20→21）；订阅报文两种线格式（对端宣告 3.1 得 `flags=0x04 size=11 body=b'\tSUBSCRIBEA'`，宣告 3.0 得 `flags=0x00 size=2 body=b'\x01A'`）；HWM 四组对照表（5/1000→5、1000/5→200、5/5→5、1000/1000→200，四行全部吻合）。
- **第三章字节级描述与 RFC 23 逐项一致**：greeting 的 `signature version mechanism as-server filler` 及 10/2/20/1/31 切分、偏移标尺 0/10/12/32/33/64、filler 31 字节；十六进制示意图各段字符数自洽（signature 20 字符=10 字节、ver 4=2、mechanism 40=20、as-server 2=1，filler 以省略号显式缩写）；flags 位定义（Bits 7-3 保留、Bit 2 COMMAND、Bit 1 LONG、Bit 0 MORE、命令帧 MORE 必须为 0）；"长度不含 flags 也不含自身，空帧长度为零"原文；短帧 0–255 与长帧 0–2^63-1；READY 元数据的 1 字节名长与 4 字节网络字节序值长（`name = OCTET 1*255name-char`、"The value size field SHALL be four octets, in network byte order."）；NULL 机制下 as-server 必须为零。
- **手算例子独立复算正确**：`01 03 61 62 63 00 01 64` → `0x01`(消息帧/短/MORE=1) + len 3 + `abc`，`0x00`(消息帧/短/MORE=0) + len 1 + `d`，即两帧消息 `[b"abc", b"d"]`，与页面结论一致。
- **版本号差异处理正确**：RFC 23 规定 `%x03 %x00`，实测为 `03 01`（ZMTP 3.1，RFC 37 定义，RFC 23 的 Related Specifications 节确有"spec:37/ZMTP defines version 3.1 of this specification"）；已核对 RFC 37 的 greeting 与帧语法除 version-minor 外与 RFC 23 一致，页面"其余布局两版本一致"的说法成立。
- **padding 的规范约束与实现取值分离得当**：RFC 23 Version Negotiation"A peer SHALL NOT assign any significance to the padding field and MUST NOT validate this nor interpret it"与 Detecting ZMTP 1.0 and 2.0 Peers 的伪签名用法均已核对；libzmq v4.3.5 `src/zmtp_engine.cpp` `plug_internal()` 中 `put_uint64 (&_outpos[_outsize], _options.routing_id_size + 1)` 确认；页面明确声明这不构成"可拿 padding 判断标识长度"的许可。
- **C1–C21、F1–F2 共 23 条来源逐条比对无扩大适用范围**：C4 的 `lb.cpp` `while (_active > 0)` 循环与交换游标逻辑、C9 关于"手册未给 ZMQ_SUB 与 ZMQ_REP 列出 Action in mute state 行"（已确认 SUB/REP 摘要表确实无此行，而 PULL 有且为 Block）、C7/C8 的 HWM 定义与两段 Note（含"as much as 90% lower"）、C12 的 zmq_send Note、C13 的 IO_THREADS 默认 1 与 inproc 不涉 I/O 线程、C14 的 inproc 4.0 顺序放开、C15 的队列销毁与 RFC 30"mostly reliable insofar as it will not discard messages unless a node disconnects unexpectedly"、C16 的线程安全原文与 draft 类型清单（已用 v4.3.5 `zmq.h` 确认 SERVER/CLIENT/RADIO/DISH/SCATTER/GATHER/PEER/CHANNEL 均在 `ZMQ_BUILD_DRAFT_API` 下）、C17 订阅加性不幂等、C19/C20 两处、C21 的 SUB 与 XSUB 静默丢弃，均与来源原文吻合；PUB 一条消息对 N 个订阅者会多次上网的说法亦有 RFC 29 PUB 节原文支撑。上标与来源条目双向对应完整（引用 23 个、定义 23 个，无孤立项），除上述 C10 一处标注遗漏外无其他缺口。
- **格式与功能**：h1 为"概念名（英文缩写）：核心作用或结论"格式；7 个 h2 均无数字编号，最后一个固定为"来源与范围说明"且无副标题；来源章节 6 个 h3 名称与 style-guide 第 1 节规定完全一致；8 个 details 的 summary 前缀均为"补充：/代码："；callout 仅用 yellow×3 与 purple×1，无 red/green/gray 误用，learning-goals 为绿色左边框、misconceptions 为红色左边框；全页无"你/您/我们"，无 S1/S2 章节代号，正文交叉引用的 9 个章节标题全部能在页面找到对应 h2/h3；无重复段落与重复长句；无"待生成/待确认"类无结论占位；本地资源（katex、prism、fonts）与 `../../index.html`、`overview.html` 链接均存在，`overview.html` 与 `index.html` 相互链接；无重复 id；`.dojo/scripts/validate.py wiki/zmq/index.html` 返回 `validation ok`。
- **学习目标覆盖**：「本文回答的问题」5 条分别由「裸 TCP 到消息库之间缺了什么」、「send 不指定收件人，收件人由 socket 类型决定」+「广播与定向回话：扇出和路由标识」、「队列满了：谁丢消息，谁阻塞」、「一条消息在 TCP 上的真实字节」、「ZeroMQ 不做什么」完整回答；折叠块全部收起后，各章正文仍保留实测结论与规范依据，可独立建立主线结论。
