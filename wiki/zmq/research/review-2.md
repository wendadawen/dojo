# ZeroMQ 审查记录（第 2 轮）

- 页面版本：5117caddb64f51271ebff99c287468e03ce91adf
- 审查时间：2026-08-17 18:41

## 问题

- [重要·技术] 「一条消息在 TCP 上的真实字节」→「64 字节的 greeting」，实测 greeting 十六进制图（index.html:906，`ff00000000000000047f 0301 4e554c4c…`）：mechanism 段写成 `4e554c4c` + 20 个零字节 = 48 个十六进制字符 = 24 字节，与同图下方标注的"20 字节"、正文第 911 行"`4e554c4c`（即 `NULL`）加 16 个零，共 20 字节"、以及代码块实测输出 `data[12:32]`（`b'NULL' + 16 个 \x00`）三处均矛盾。RFC 23 Formal Grammar 规定 `mechanism = 20mechanism-char`，本机实测 greeting 该段确为 20 字节。按该图逐段读取会把 as-server 定位到偏移 36 而非 32，与图首行"32 33"的偏移刻度自相矛盾。｜修复要求：将 906 行 mechanism 段改为 `4e554c4c` 加 16 个 `00`（共 40 个十六进制字符），使该段字节数等于 20，并与 906 行各段拼接后满足 10+2+20+1+31=64；改后重新核对 907 行分隔标注与 908 行字节数标注仍逐段对齐。｜修复：mechanism 段改为 `4e554c4c` + 32 个 `0`（共 40 个十六进制字符 = 20 字节），并重新对齐分隔标注与字节数标注行。｜复验：脚本核算：示意图各段字符数为 20/4/40/2，对应 10/2/20/1 字节；把前四段拼接后与实测 greeting 前 33 字节逐字符比对，完全一致（脚本断言 `joined == expected` 为真）。

- [重要·技术] 「队列满了：谁丢消息，谁阻塞」→「mute 状态：丢弃还是阻塞」，index.html:1287"SUB 一侧则根本不发生丢弃，丢弃在发布端完成<sup>[C9]</sup>"，以及「三种出站策略与一种入站策略」对照表 SUB 行（index.html:757）"手册未列该行；丢弃发生在发布端"：该表述扩大了适用范围。本页自己引用的 RFC 29/PUBSUB 在 The SUB Socket Type 的 For processing incoming messages 条目中明确规定"SHALL silently discard messages if the queue for a publisher is full."，即 SUB 在其为某发布端维护的入站队列满时同样静默丢弃。页面把"本页实测场景下丢弃只发生在发布端"写成了"SUB 一侧根本不发生丢弃"这一无条件论断，与来源直接冲突，会让读者得出"SUB 端在任何情况下都不会丢消息"的错误结论。｜修复要求：把 1287 行与 757 行的无条件表述改为带条件的表述——说明 RFC 29 同时规定 SUB 入站队列满时也会静默丢弃，并说明本页四组 HWM 对照表中订阅端 HWM 小却未丢的原因是发布端出站队列先行承接、订阅端队列未持续处于满状态；同时在来源章节为该 SUB 丢弃规则补上可定位条目（RFC 29/PUBSUB，The SUB Socket Type，For processing incoming messages）。｜修复：删除该无条件论断。「mute 状态」节改为说明"接收侧队列满时也会丢消息"并引用 RFC 29 的 SUB 条目；对照表 SUB 行改为"入站队列满则静默丢弃"；表下说明段同步改写；新增 C21 条目。另外把「丢弃发生在发布端队列」一节整体重写为「两侧都会丢，先满的那一侧丢」，明确第 2 行"200/200"不能推广成"订阅端 HWM 小不会丢消息"，并说明该结果源于本实验的构造条件（订阅端在发送结束后才排空）。｜复验：核对 RFC 29/PUBSUB 的 The SUB Socket Type 原文确认"SHALL silently discard messages if the queue for a publisher is full."；页面现有三处提到接收侧丢弃且均带 C21 上标，无处再声称 SUB 侧不丢。

- [重要·技术] 来源编号与论断错配，涉及 index.html:913、1080、1081 三处上标与 C18/C19/C20 三个条目：（1）913 行引用 RFC 23"size 是发送方标识的字节数（0 或更多）加 1，占据 padding 字段"标注为 <sup>[C18]</sup>，1080 行"padding 是 `0000000000000004`，即路由标识 `cli` 的长度加一"同样标注 <sup>[C18]</sup>，但 C18 条目（index.html:1553）的内容是 READY 元数据属性编码与 ZMTP 2.0 变更，不含该伪签名规定；含该规定的是 C19 条目（index.html:1554）。（2）1081 行"RFC 23 规定使用 NULL 安全机制时该字段必须为零"标注 <sup>[C19]</sup>，但该 as-server 规定实际写在 C20 条目（index.html:1555）中，C19 条目不含此内容。结果是 C19 条目在正文中无任何一处按其自身内容被正确引用，读者按上标回查无法定位到支撑原文。｜修复要求：将 913 行与 1080 行的 padding／伪签名上标由 [C18] 改为 [C19]；将 1081 行的 as-server 上标由 [C19] 改为 [C20]；改后逐一核验每个上标编号所指条目确实包含支撑该处论断的原文引文，且 C18–C20 每个条目都至少被一处内容相符的正文引用。｜修复：把 padding 伪签名的两处上标由 [C18] 改为 [C19]；as-server 那处由 [C19] 改为 [C20]；C18 仅保留在 READY 元数据编码与 ZMTP 2.0 变更两处。｜复验：脚本做双向对应检查：正文引用集合与来源定义集合均为 C1–C21，"引用但未定义"与"定义但未引用"两个差集皆为空；逐一核对三处上标所指条目均含支撑该处论断的原文引文。

- [轻微·格式] 「一条消息在 TCP 上的真实字节」→「帧头：1 字节 flags 加长度」，"代码：抓取 ZMTP 握手与帧的原始字节"折叠块内 index.html:1081 与 1082 为两段"简化条件"，其中 1082 行文字是 1081 行的真子串（1081 行仅在句末多出 as_server 取值那一句），构成完整重复段落。｜修复要求：删除 1082 行整段，使该折叠块只保留 1081 行一段"简化条件"。｜修复：删除重复的那一段（不含 as_server 说明的那行），保留完整版本。｜复验：`grep -c` 结果由 6 降为 5，抓包折叠块内只剩一段"简化条件"。

- [轻微·格式] 「64 字节的 greeting」字段框图 index.html:892 行（`│        + %x7F      │    │                       │ (1)                     │`）只有 5 条竖线，而同一框体的 890、891 行各有 6 条：as-server 列与 filler 列之间的分隔竖线缺失，该行框线断裂。｜修复要求：在 892 行 `(1)` 之后补回 as-server 与 filler 之间的分隔竖线，使 890–892 三行竖线数量一致（各 6 条），且该行显示宽度仍与框体其余各行相同。｜修复：整体重画该字段框图：五列宽度改为 21/10/21/10/20，文字重排为每列三行加一行字节数，并用脚本按东亚字符宽度计算逐格补齐。｜复验：脚本校验框体 7 行显示宽度全部为 93（宽度集合长度为 1），三行内容行竖线数均为 6，断言通过。

- [轻微·技术] 「send 不指定收件人，收件人由 socket 类型决定」→「bind 与 connect 不是服务端与客户端」，index.html:865"顺序上，自 libzmq 4.0 起 <code>zmq_bind()</code> 与 <code>zmq_connect()</code> 的调用顺序不再有要求<sup>[C14]</sup>"，紧随其后给出的却是 tcp 传输（`tcp://127.0.0.1:5712`）的实测。来源 `zmq_inproc(7)` 原文为"Since version 4.0 the order of zmq_bind() and zmq_connect() does not matter just like for the tcp transport type."，该 4.0 版本变更针对 inproc 传输，tcp 传输本就不受顺序约束——这一点 C14 条目已注明，但正文未体现，读者会误以为 4.0 之前 tcp 也要求先 bind。｜修复要求：改写 865 行，明确"4.0 起放开顺序限制"针对 inproc 传输，tcp 传输原本即不要求 bind 先于 connect，使正文表述与 C14 条目末句一致。｜修复：改写为："tcp 传输本来就不要求 bind 先于 connect；inproc 传输在 libzmq 4.0 之前要求先 bind，自 4.0 起也放开了这一限制，与 tcp 一致"。｜复验：正文表述现与 C14 条目末句"just like for the tcp transport type"一致，不再把 inproc 的版本变更扩大为通用规则。

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 3
- 处置：修复（全部 6 项已修复并复验；本轮无返回规划项）

## 本轮核验记录（供复验参考，不计入问题）

- 5 个"代码："折叠块全部实际运行（pyzmq 27.1.0 / libzmq 4.3.5），逐字符 diff 与页面"预期输出"完全一致：轮询分发、ZMTP 抓包（117 字节 / 三帧）、发布端过滤、PUB 丢弃与 PUSH 阻塞对照、slow joiner 与 connect 先于 bind。
- 正文 `<pre class="diagram">` 内 6 处实测结果均独立复现且多次运行稳定：PUSH 轮询分发、轮询跳过满队列（`[0,2]` / 余下 8 条）、ROUTER 信封（含 REQ 空分隔帧）、订阅报文两种线格式（3.1 的 `\tSUBSCRIBEA` 与 3.0 的 `\x01A`）、padding 与路由标识长度四组对应（4/9/1/21）、HWM 四组对照表（5/200、200/200、5/200、200/200）。
- 帧头手算例子 `01 03 61 62 63 00 01 64` 独立复算，解析为两帧消息 `[b"abc", b"d"]`，与页面结论一致。
- 第三章字节级描述逐项对照 RFC 23 的 Framing 节、Formal Grammar 节与 The NULL Security Mechanism 节：flags 位定义（bit 7–3 保留 / bit 2 COMMAND / bit 1 LONG / bit 0 MORE）、短帧 0–255 与长帧 2^63−1、"长度不含 flags 也不含自身"、READY 属性名 1 字节长度与属性值 4 字节网络字节序长度、greeting 的 10/2/20/1/31 切分与 64 字节总长——除上述 mechanism 段十六进制串外均一致。
- libzmq v4.3.5 源码核对：`src/lb.cpp` 的 `lb_t::sendpipe` 中 `while (_active > 0)` 循环与 `_active` 归零返回 `EAGAIN` 的摘录准确；`src/zmtp_engine.cpp` 的 `plug_internal()` 中 `_outpos[_outsize++] = UCHAR_MAX;`、`put_uint64 (&_outpos[_outsize], _options.routing_id_size + 1);`、`0x7f` 三处与页面描述一致。
- 手册核对：HWM 定义与"任意单个对端"、默认 1000 条、SNDHWM"低多达 90%"与 RCVHWM"可能更低或更高"两处 Note、`ZMQ_IO_THREADS` 默认 1、inproc 不涉及 I/O 线程、send 成功仅表示已入队、ROUTER 静默丢弃与 `ZMQ_ROUTER_MANDATORY` 下 `EHOSTUNREACH`、非线程安全类型清单、对照表前五列（方向／出入站策略／兼容对端／mute 动作）均与 `zmq_socket(3)`、`zmq_setsockopt(3)`、`zmq_send(3)`、`zmq_ctx_set(3)`、`zmq_inproc(7)` 一致；"共记录 20 种类型"经清点确认。
- 「本文回答的问题」5 条均由正文章节完整回答，逐条对应「裸 TCP 到消息库之间缺了什么」、「send 不指定收件人，收件人由 socket 类型决定」＋「广播与定向回话：扇出和路由标识」、「队列满了：谁丢消息，谁阻塞」、「一条消息在 TCP 上的真实字节」、「ZeroMQ 不做什么」。
- 折叠块全部收起后，各章结论所需的实测数据与论断均留在正文，结论链条完整。
- 格式：h1 为"概念名（英文缩写）：结论"形式；h2/h3 无数字编号；最后一个 h2 为"来源与范围说明"；来源章节 h3 名称符合固定清单；summary 前缀仅"代码："（5）与"补充："（3）；callout 仅用 yellow（3）与 purple（1），无 red/green/gray 误用；正文引用其他章节均用章节标题；无第一人称复数与第二人称；无"待生成／待确认"类占位表述。
- 页面功能：`libs/` 下 katex、auto-render、prism 全部 7 个本地资源存在；`../../index.html` 与 `overview.html` 链接有效，overview 与 index 相互链接；无外部 http 链接；`.dojo/scripts/validate.py wiki/zmq/index.html` 返回 `validation ok`。
