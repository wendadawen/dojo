# ZeroMQ 核心论断与证据

## 来源清单

| 代号 | 来源 | 定位方式 |
|---|---|---|
| S1 | `zmq_socket(3)`，0MQ API 手册 latest | https://libzmq.readthedocs.io/en/latest/zmq_socket.html |
| S2 | `zmq_setsockopt(3)`，0MQ API 手册 latest | https://libzmq.readthedocs.io/en/latest/zmq_setsockopt.html |
| S3 | `zmq_send(3)`，0MQ API 手册 latest | https://libzmq.readthedocs.io/en/latest/zmq_send.html |
| S4 | `zmq_ctx_set(3)`，0MQ API 手册 latest | https://libzmq.readthedocs.io/en/latest/zmq_ctx_set.html |
| S5 | `zmq_inproc(7)`，0MQ API 手册 latest | https://libzmq.readthedocs.io/en/latest/zmq_inproc.html |
| S6 | RFC 23/ZMTP（status: stable），ZeroMQ Message Transport Protocol 3.0 | https://rfc.zeromq.org/spec/23/ |
| S7 | RFC 28/REQREP（status: stable） | https://rfc.zeromq.org/spec/28/ |
| S8 | RFC 29/PUBSUB（status: stable） | https://rfc.zeromq.org/spec/29/ |
| S9 | RFC 30/PIPELINE（status: stable） | https://rfc.zeromq.org/spec/30/ |
| S10 | libzmq v4.3.5 源码 `src/lb.cpp`（负载均衡出站） | https://github.com/zeromq/libzmq/blob/v4.3.5/src/lb.cpp |
| S11 | libzmq v4.3.5 源码 `src/fq.cpp`（公平队列入站） | https://github.com/zeromq/libzmq/blob/v4.3.5/src/fq.cpp |
| S12 | 本机实测，pyzmq 27.1.0 + libzmq 4.3.5，macOS，脚本见"实测记录" | 脚本名 + 实际输出 |

来源优先级说明：定义与语义以 RFC（S6–S9）和官方 API 手册（S1–S5）为准；实现细节以固定 tag 源码（S10、S11）为准；实测（S12）只用于验证前两类结论在具体版本上的实际行为，不单独作为定义依据。

## 核心论断（C）

### C1｜ZeroMQ socket 是消息队列抽象，语义由 socket 类型决定

- 内容：`zmq_socket()` 创建的 socket 不是操作系统 socket，而是一层异步消息队列抽象；具体的消息传递语义由创建时指定的类型决定。
- 来源：S1，"0MQ sockets present an abstraction of an asynchronous *message queue*, with the exact queueing semantics depending on the socket type in use."
- 适用条件：无。
- 置信状态：已确认。

### C2｜ZeroMQ 的传输单位是帧，一条消息由一到多个帧组成且原子交付

- 内容：一条 0MQ 消息由一个或多个消息段（帧）构成；实现保证原子交付，对端要么收到一条消息的全部帧，要么一个都收不到。帧数量除内存外无上限。
- 来源：S3，"A 0MQ message is composed of 1 or more message parts. 0MQ ensures atomic delivery of messages: peers shall receive either all message parts of a message or none at all. The total number of message parts is unlimited except by available memory."；S6 Framing 节，"Messages consist of one or more frames and an implementation SHALL always send and deliver messages atomically, that is, all the frames of a message, or none of them."
- 适用条件：无。
- 置信状态：已确认。

### C3｜PUSH 出站用轮询，PULL 入站用公平队列

- 内容：PUSH 只发不收，向一组匿名 PULL 对端按轮询分发；PULL 只收不发，从所有已连接 PUSH 对端按公平队列取消息。
- 来源：S9，"The PUSH socket type talks to a set of anonymous PULL peers, sending messages using a round-robin algorithm."；"The PULL socket type talks to a set of anonymous PUSH peers, receiving messages using a fair-queuing algorithm."；S1 摘要表 PUSH 出站 Round-robin、PULL 入站 Fair-queued。
- 适用条件：对端可用（即其出站队列未满）。
- 置信状态：已确认，且经 S12 实测（e2_push.py：6 个任务在 3 个 PULL 上得到 1,4 / 2,5 / 3,6）。

### C4｜轮询只在"可用"对端间进行，队列满的对端会被跳过

- 内容：PUSH 只把队列未满的对端视为可用，并只向可用对端轮询；实现层面，`lb_t::sendpipe` 在 `write()` 失败时把该管道移出活跃区间并继续尝试下一个，而不是等待它。
- 来源：S9，"SHALL consider a peer as available only when it has a outgoing queue that is not full."、"SHALL route outgoing messages to available peers using a round-robin strategy."；S10 `lb_t::sendpipe` 第 71–108 行：`while (_active > 0) { if (_pipes[_current]->write(msg_)) break; ... _active--; ... }`。
- 适用条件：非多段消息中途（多段消息中途失败会走 rollback + EAGAIN 分支）。
- 置信状态：已确认，且经 S12 实测（e8_lb_skip.py：慢对端只拿到 0、2 两条，其余 8 条全给了空闲对端）。

### C5｜PUB 出站是扇出，且订阅过滤在发布端执行

- 内容：PUB 向所有已连接订阅者扇出消息；在 ZMTP 之上，过滤发生在发布端——SUB/XSUB 把订阅前缀作为 SUBSCRIBE 消息发给发布端，发布端对消息第一帧起始处做二进制比较后再决定是否发送。
- 来源：S8，"SHALL perform a binary comparison of the subscription against the start of the first frame of the message."；S6，"When using ZMTP, message filtering SHALL happen at the publisher side (the PUB or XPUB socket)."，订阅报文语法 `subscribe = %x00 short-size %d1 subscription`；S1 摘要表 PUB 出站 Fan out。
- 适用条件：使用 ZMTP 的传输（如 tcp）。S8 同时允许"MAY, depending on the transport, send all messages to all subscribers"，即在别的传输上可能不做发布端过滤。
- 置信状态：已确认，且经 S12 实测（e7_filter.py：只订阅前缀 `A` 时，TCP 连接上只出现 `A-match-1`、`A-match-2`，`B-nomatch` 未上线）。

### C6｜ROUTER 用路由标识信封定向投递

- 内容：ROUTER 为每个对端维护一对队列并用唯一二进制 identity 标识；入站时在消息前面加一帧 identity 后交给应用，出站时剥掉第一帧作为目标 identity 查找队列。目标队列不存在或已满时，按配置静默丢弃或返回错误，且不阻塞。
- 来源：S7 ROUTER 节，"SHALL prefix each incoming message with a frame containing the identity of the originating double queue."、"SHALL remove the first frame from each outgoing message and use this as the identity of a double queue."、"SHALL either silently drop the message, or return an error, depending on configuration, if the queue does not exist, or is full."、"SHALL NOT block on sending."；S1 ROUTER 节，"If the peer does not exist anymore, or has never existed, the message shall be silently discarded. However, if 'ZMQ_ROUTER_MANDATORY' socket option is set to '1', the socket shall fail with EHOSTUNREACH in both cases."
- 适用条件：默认配置下静默丢弃；设 `ZMQ_ROUTER_MANDATORY=1` 时改为报错。
- 置信状态：已确认，且经 S12 实测（e4_router.py：DEALER 侧收到 `[b'worker-7', b'hello-from-dealer']`，REQ 侧收到 `[b'client-1', b'', b'hello-from-req']`）。

### C7｜HWM 是每个对端的队列上限，默认 1000 条消息

- 内容：`ZMQ_SNDHWM` / `ZMQ_RCVHWM` 是 0MQ 为"任意单个对端"在内存中排队的最大消息条数的硬上限；单位是消息条数，默认值 1000，0 表示无上限。
- 来源：S2 ZMQ_SNDHWM / ZMQ_RCVHWM 节，"The high water mark is a hard limit on the maximum number of outstanding messages 0MQ shall queue in memory for any single peer that the specified 'socket' is communicating with. A value of zero means no limit."；同节属性表 Option value unit = messages、Default value = 1000。
- 适用条件：见 C8。
- 置信状态：已确认。

### C8｜实际队列容量可能偏离 HWM 设定值

- 内容：0MQ 不保证队列能恰好容纳 HWM 条消息。接收侧"实际上限可能更低或更高，取决于传输"；发送侧"实际上限可能低至设定值的 90% 以下，也可能超过阈值"，TCP 传输是典型例子。
- 来源：S2，ZMQ_RCVHWM 注："0MQ does not guarantee that the socket will be able to queue as many as ZMQ_RCVHWM messages, and the actual limit may be lower or higher, depending on socket transport."；ZMQ_SNDHWM 注："0MQ does not guarantee that the socket will accept as many as ZMQ_SNDHWM messages, and the actual limit may be as much as 90% lower depending on the flow of messages on the socket. The socket may even be able to accept more messages than the ZMQ_SNDHWM threshold."
- 适用条件：无。
- 置信状态：已确认。此论断解释了为何实测条数不必等于设定值，正文中不得把实测条数写成"HWM 的确定行为"。

### C9｜队列满进入 mute 状态；丢弃或阻塞由 socket 类型决定

- 内容：达到 HWM 后 socket 进入 mute 状态。PUB / XPUB / XSUB 丢弃，且 PUB 的 `zmq_send()` 永不阻塞；ROUTER 默认丢弃（设 `ZMQ_ROUTER_MANDATORY=1` 改为阻塞或返回 EAGAIN）；REQ / PUSH / PULL / DEALER / PAIR 阻塞，且明确不丢消息。
- 来源：S1 各类型的 "Action in mute state" 摘要行与正文。PUB："then any messages that would be sent to the *subscriber* in question shall instead be dropped until the mute state ends. The *zmq_send()* function shall never block for this socket type."；PUSH："then any zmq_send operations on the socket shall block until the mute state ends or at least one downstream *node* becomes available for sending; messages are not discarded."；DEALER 同 PUSH 表述；ROUTER："then any messages sent to the socket shall be dropped until the mute state ends. … If 'ZMQ_ROUTER_MANDATORY' is set to '1', the socket shall block or return EAGAIN in both cases."；REQ："The REQ socket shall not discard messages."
- 适用条件：S1 未给 SUB 与 REP 列出 "Action in mute state" 行。SUB 只收不发，丢弃动作发生在 PUB 侧；REP 的相关表述是"If the original requester does not exist any more the reply is silently discarded."与"SHALL not block on sending."（S7）。正文中不得为 SUB / REP 编造 mute 行为。
- 置信状态：已确认。
- 实测对照（S12，e3_hwm.py）：SNDHWM=5 / RCVHWM=5 的 PUB 连发 100 条全部被接受，订阅端只收到 5 条；同样参数的 PUSH 在第 5 条之后返回 EAGAIN，共接受 5 条，PULL 全部收到、零丢失。

### C10｜丢弃发生在发布端队列，接收端 HWM 不足不会导致 PUB 丢消息

- 内容：PUB/SUB 场景下，消息被丢弃的位置是发布端为该订阅者维护的出站队列。仅把订阅端 `ZMQ_RCVHWM` 设小、发布端 HWM 保持默认时，消息不会丢失——它们堆在发布端队列里等订阅端取走。
- 来源：S8 PUB 节，"SHALL maintain a single outgoing message queue for each connected subscriber."、"SHALL silently drop the message if the queue for a subscriber is full."（丢弃条件是发布端队列满）。
- 适用条件：发布端出站队列尚未达到 HWM；订阅端最终会继续读取。
- 置信状态：已确认，且经 S12 实测（e9b.py：SNDHWM=5/RCVHWM=1000 → 收到 5/200；SNDHWM=1000/RCVHWM=5 → 反复排空后收到 200/200；SNDHWM=1000/RCVHWM=1000 → 200/200）。

### C11｜REQ 强制 send-recv 交替，违反时报 EFSM

- 内容：REQ 必须一次发一条、再收一条，严格交替；连续两次 send 会因 socket 状态不符而失败，errno 为 EFSM。
- 来源：S7 REQ 节，"SHALL send and then receive exactly one message at a time."；S3 ERRORS 节，"EFSM: The `zmq_send()` operation cannot be performed on this socket at the moment due to the socket not being in the appropriate state. This error may occur with socket types that switch between several states, such as ZMQ_REP."
- 适用条件：未设置 `ZMQ_REQ_RELAXED` 等放宽选项。
- 置信状态：已确认，且经 S12 实测（e5_req.py：第二次 send 抛错，errno=156384763 = `zmq.EFSM`，描述 "Operation cannot be accomplished in current state"）。

### C12｜REQ 在消息前加一个空分隔帧

- 内容：REQ 发出的消息在线上格式为"一个空帧作为分隔符 + 一到多个数据帧"，空帧由 REQ socket 自己添加，应用不可见。
- 来源：S7 REQ 节，"The request and reply messages SHALL have this format on the wire: A delimiter, consisting of an empty frame, added by the REQ socket. One or more data frames, comprising the message visible to the application."、"SHALL prefix the outgoing message with an empty delimiter frame."
- 适用条件：无。
- 置信状态：已确认，且经 S12 实测（e4_router.py：ROUTER 从 REQ 收到 `[b'client-1', b'', b'hello-from-req']`，中间那个空 bytes 就是分隔帧；同一 ROUTER 从 DEALER 收到的消息没有这一帧）。

### C13｜send 返回成功只表示已入队

- 内容：`zmq_send()` 调用成功不表示消息已发送到网络，只表示消息已在该 socket 上排队、0MQ 已接管这条消息的责任。
- 来源：S3 Note，"A successful invocation of `zmq_send()` does not indicate that the message has been transmitted to the network, only that it has been queued on the 'socket' and 0MQ has assumed responsibility for the message."
- 适用条件：无。
- 置信状态：已确认。

### C14｜消息的实际收发由 context 的 I/O 线程池完成，inproc 不使用 I/O 线程

- 内容：context 持有一个 I/O 线程池处理 I/O 操作，默认 1 个线程；纯 inproc 通信不涉及 I/O 线程，可将线程数设为 0。
- 来源：S4 ZMQ_IO_THREADS 节，"The 'ZMQ_IO_THREADS' argument specifies the size of the 0MQ thread pool to handle I/O operations. If your application is using only the 'inproc' transport for messaging you may set this to zero, otherwise set it to at least one."，Default value = 1；S5 Note，"No I/O threads are involved in passing messages using the 'inproc' transport."
- 适用条件：无。
- 置信状态：已确认。

### C15｜connect 与 bind 的先后顺序无关

- 内容：自 4.0 版本起，inproc 传输上 `zmq_bind()` 与 `zmq_connect()` 的调用顺序不再有要求，与 tcp 一致。
- 来源：S5，"Before version 4.0 the 'name' must have been previously created by assigning it to at least one 'socket' within the same 0MQ 'context' as the 'socket' being connected. Since version 4.0 the order of zmq_bind() and zmq_connect() does not matter just like for the tcp transport type."
- 适用条件：libzmq ≥ 4.0。
- 置信状态：已确认，且经 S12 实测（e6_slowjoiner.py 后半段：PUSH 先 connect 到无人监听的 5607 并 send，PULL 随后 bind，立刻收到该消息）。

### C16｜PUSH/ROUTER/PUB 在对端断开时销毁其队列并丢弃其中消息

- 内容：这些 socket 为每个对端创建队列；当该对端断开时销毁其队列，并丢弃队列中尚未发出的消息。
- 来源：S9 PUSH 节，"SHALL create this queue when a peer connects to it. If this peer disconnects, the PUSH socket SHALL destroy its queue and SHALL discard any messages it contains."；S7 ROUTER 节与 S8 PUB 节有对应表述。
- 适用条件：无。
- 置信状态：已确认。这是 S9 所称"mostly reliable insofar as it will not discard messages unless a node disconnects unexpectedly"的具体来源。

### C17｜常用 socket 类型均非线程安全

- 内容：REQ/REP/DEALER/ROUTER/PUB/SUB/XPUB/XSUB/PUSH/PULL/PAIR 都不是线程安全的；只有 CLIENT、SERVER、DISH、RADIO、SCATTER、GATHER、PEER、CHANNEL（均为 draft）标注为线程安全。
- 来源：S1，"Applications MUST NOT use a *not* thread safe socket from multiple threads under any circumstances. Doing so results in undefined behaviour."，以及各类型的 Thread safety 标注。
- 适用条件：无。
- 置信状态：已确认。

### C18｜订阅是加性的、不幂等

- 内容：订阅可叠加：同时订阅 "A" 和空字符串等价于只订阅空字符串；订阅两次 "A" 计为两次订阅，需要两次退订才能撤销。
- 来源：S6，"Subscriptions SHALL be additive and SHALL NOT be idempotent. That is, subscribing to 'A' and '' is the same as subscribing to '' alone. Subscribing to 'A' and 'A' counts as two subscriptions, and would require two unsubscribe messages to undo."
- 适用条件：ZMTP 传输。
- 置信状态：已确认。

## 核心公式与线格式（F）

### F1｜ZMTP greeting 的 64 字节布局

```
greeting  = signature version mechanism as-server filler
signature = %xFF padding %x7F        ; 10 octets，padding 为 8 octets 且无语义
version   = version-major version-minor
mechanism = 20 mechanism-char        ; 大写字母/数字/-_.+/%x0，右侧补零
as-server = %x00 | %x01              ; 1 octet
filler    = 31 %x00                  ; 31 octets，把 greeting 补到 64
```

- 来源：S6 Formal Grammar 节（ABNF）。
- RFC 23 规定 `version-major = %x03`、`version-minor = %x00`（即 ZMTP 3.0）。
- 版本冲突记录：实测 libzmq 4.3.5 发出的版本字节是 `03 01`，即 ZMTP 3.1，对应 RFC 37/ZMTP（S6 的 Related Specifications 明确列出 "spec:37/ZMTP defines version 3.1 of this specification"）。正文写线格式时必须写 3.1 是实测结果、3.0 是 RFC 23 的规定值，不得把两者混为一谈。
- 置信状态：已确认（布局），版本号差异已定位到 RFC 37，非冲突未解。

### F2｜ZMTP 帧头布局

```
message      = *message-more message-last
message-more = ( %x01 short-size | %x03 long-size ) message-body
message-last = ( %x00 short-size | %x02 long-size ) message-body
short-size   = OCTET        ; 帧体 0–255 octets
long-size    = 8 OCTET      ; 帧体 0–2^63-1 octets
command      = ( %x04 short-size | %x06 long-size ) command-body
```

flags 字节（bit 0 为最低位）：

| 位 | 名称 | 含义 |
|---|---|---|
| 7–3 | 保留 | 必须为 0 |
| 2 | COMMAND | 1 = 命令帧，0 = 消息帧 |
| 1 | LONG | 0 = 长度用 1 字节，1 = 长度用 8 字节网络字节序 |
| 0 | MORE | 1 = 后面还有帧，0 = 本条消息最后一帧；命令帧此位必须为 0 |

- 来源：S6 Framing 节与 Formal Grammar 节。原文："A frame consists of a flags field (1 octet), followed by a size field (one octet or eight octets) and a frame body of size octets. The size does not include the flags field, nor itself, so an empty frame has a size of zero."
- 置信状态：已确认，且经 S12 实测逐字节对照（e1_wire.py，见下）。

## 实测记录（N）

全部实测在本机完成：macOS，pyzmq 27.1.0，libzmq 4.3.5。脚本位于 `/tmp/zmqlab/`，页面中引用的代码为可独立运行的最小版本。

### N1｜ZMTP 线上字节（e1_wire.py）

用一个手写 TCP 对端假装 ROUTER，让真实 DEALER socket 连上来并发送两帧消息 `[b"A", b"BC"]`。抓到的 117 字节：

```
greeting  (64B): ff00000000000000047f 0301 4e554c4c0000...00
  signature    : ff 00000000000004 7f      （padding 非零，符合"无语义"）
  version      : 03 01                     （ZMTP 3.1）
  mechanism    : b'NULL\x00...\x00'（20B）
frame flags=0x04 COMMAND=True  LONG=False MORE=False size=44
  body: b'\x05READY\x0bSocket-Type\x00\x00\x00\x06DEALER\x08Identity\x00\x00\x00\x03cli'
frame flags=0x01 COMMAND=False LONG=False MORE=True  size=1  body=b'A'
frame flags=0x00 COMMAND=False LONG=False MORE=False size=2  body=b'BC'
```

对应结论：F1（greeting 布局）、F2（flags 位含义、MORE 序列）、C2（多段消息 = MORE=1…MORE=0）。

### N2｜PUSH 轮询分发（e2_push.py）

1 个 PUSH bind，3 个 PULL connect，发 6 个任务：

```
worker0 received: ['1', '4']
worker1 received: ['2', '5']
worker2 received: ['3', '6']
```

对应结论：C3。构造条件：3 个 PULL 均已完成连接（sleep 0.5s），任务负载为空，故轮询顺序整齐。

### N3｜PUB 丢弃 vs PUSH 阻塞（e3_hwm.py）

SNDHWM=5、RCVHWM=5，各连发 100 条、接收端不及时读取：

```
PUB : attempted 100, accepted 100
SUB : received 5 of 100  -> dropped 95
PUSH refused (EAGAIN) at message 5
PUSH: attempted 100, accepted 5
PULL: received 5 of accepted 5 -> lost 0
```

对应结论：C9。注意按 C8，"5 条"是本次运行的实际容量，不是 HWM 的保证值。

### N4｜丢弃发生在发布端（e9b.py）

固定发 200 条，改变两侧 HWM，反复排空后统计订阅端收到条数：

```
SNDHWM=5     RCVHWM=1000  -> 5   / 200
SNDHWM=1000  RCVHWM=5     -> 200 / 200
SNDHWM=5     RCVHWM=5     -> 5   / 200
SNDHWM=1000  RCVHWM=1000  -> 200 / 200
```

对应结论：C10。只有发布端 HWM 小时才丢；订阅端 HWM 小只是让消息在发布端多等一会儿。

### N5｜ROUTER 信封与 REQ 空分隔帧（e4_router.py）

一个 ROUTER 同时接受 DEALER（routing id `worker-7`）与 REQ（routing id `client-1`）：

```
ROUTER received: [b'worker-7', b'hello-from-dealer']
ROUTER received: [b'client-1', b'', b'hello-from-req']
```

对应结论：C6（入站加 identity 帧）、C12（REQ 额外加空分隔帧）。

### N6｜REQ 状态机（e5_req.py）

连续两次 send：

```
second send failed: errno=156384763 name=Operation cannot be accomplished in current state zmq.EFSM=156384763
```

对应结论：C11。

### N7｜slow joiner 与 connect 先于 bind（e6_slowjoiner.py）

```
subscriber got: ['late-0', 'late-1', 'late-2']
after late bind, PULL got: b'queued-before-bind'
```

前半段：SUB connect 后立刻发的 3 条 `early-*` 全部消失，等待 0.5s 后发的 3 条 `late-*` 全部收到——订阅生效有延迟，其间发布端没有匹配订阅者可发。对应 C5（发布端过滤）与 scope M2/M5。
后半段：对应 C15。

### N8｜轮询跳过满队列的对端（e8_lb_skip.py）

inproc 传输，PUSH 的 SNDHWM=1，两个 PULL 分别设 RCVHWM=1 与 100，发 10 条：

```
accepted by PUSH: [0,1,2,3,4,5,6,7,8,9]
slow PULL got: [0, 2]
fast PULL got: [1, 3, 4, 5, 6, 7, 8, 9]
```

对应结论：C4。慢对端队列满后被移出活跃集合，后续消息全部走空闲对端；PUSH 本身没有阻塞，因为始终存在可用对端。

### N9｜订阅过滤发生在发布端（e7_filter.py）

手写 TCP 对端假装 SUB，只发送订阅前缀 `A` 的 SUBSCRIBE 报文，真实 PUB 依次发 3 条消息。该 TCP 连接上实际出现的帧：

```
COMMAND size=25 body=b'\x05READY\x0bSocket-Type\x00\x00\x00\x03PUB'
MESSAGE size=9  body=b'A-match-1'
MESSAGE size=9  body=b'A-match-2'
```

`B-nomatch` 从未出现在连接上。对应结论：C5，并直接反驳 scope M5。

## 未纳入正文的存疑项

| 项 | 状态 | 处理 |
|---|---|---|
| S1 摘要表未给 SUB / REP 列出 "Action in mute state" | 证据不足 | 正文不描述 SUB / REP 的 mute 行为，只说明 PUB 侧丢弃与 REP 的"原请求方已断开则静默丢弃回复"（有 S7 出处） |
| S1 中 SCATTER/GATHER 的 "Compatible peer sockets" 写成自身（疑似文档笔误） | 存在冲突 | 不写入正文（draft 类型已排除在范围外） |
| RFC 23 写 `version-minor = %x00`，实测为 `%x01` | 已定位为 RFC 37/ZMTP 3.1 | 正文明确区分"RFC 23 规定值"与"libzmq 4.3.5 实测值" |
