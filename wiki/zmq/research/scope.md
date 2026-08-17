# ZeroMQ 内容范围

## 0. 概念歧义处理

"ZMQ" 在不同语境中可能指：

| 候选含义 | 判断 |
|---|---|
| ZeroMQ / ØMQ / 0MQ 消息库 | 采纳。这是 "ZMQ" 在软件工程语境下的主流含义，官方站点 zeromq.org 与 API 文档均以 `zmq_` 为函数前缀、以 `libzmq` 为核心实现名。 |
| ZMTP（ZeroMQ Message Transport Protocol） | 不作为页面主体。它是 ZeroMQ 在 TCP 上的线协议（RFC 23/ZMTP），属于本页核心机制的一部分，在"消息如何被切成帧"一章中展开，不单列为独立含义。 |
| RabbitMQ / ActiveMQ / Kafka 一类 broker 中间件 | 排除，且作为常见误解处理。ZeroMQ 不是 broker，没有独立服务进程。 |

歧义处理状态：已裁定。采纳"ZeroMQ 消息库"含义，依据是官方文档与 RFC 的命名体系。

## 1. 概念含义

- 中文名称：ZeroMQ
- 英文名称：ZeroMQ（亦写作 ØMQ、0MQ）
- 常见缩写：ZMQ；C API 函数前缀 `zmq_`；参考实现库名 `libzmq`
- 简要定义：ZeroMQ 是一个嵌入在应用进程内的消息传输库。它把"发送与接收离散消息"这件事从字节流层面抽出来，让应用按预设的消息模式（请求-应答、发布-订阅、流水线、独占对）收发完整消息，而不必自己处理消息边界、连接重建和多对端分发。
- 正式定义依据：官方 API 手册对 `zmq_socket` 的描述是"0MQ sockets present an abstraction of an asynchronous *message queue*"，且 socket 类型决定该 socket 的消息传递语义（zmq_socket(3)）；ZMTP 规范开篇说明它解决的问题是"TCP carries a stream of octets with no delimiters, but we want to send and receive discrete messages"（RFC 23/ZMTP, Goals）。
- 本文采用的语境：进程间与跨主机的应用层消息通信。示例统一使用 Python 绑定 pyzmq 27.1.0 + libzmq 4.3.5 在本机运行。

### 包括什么

| 内容 | 为什么属于本概念 |
|---|---|
| socket 类型与消息模式 | socket 类型是 ZeroMQ 的核心抽象：同一套 `zmq_send`/`zmq_recv` 接口，语义完全由类型决定。 |
| 消息帧与多段消息 | ZeroMQ 传输的最小单位是帧，一条消息可含多帧且原子交付，这是"消息"而非"字节流"的直接体现。 |
| 队列与高水位（HWM）行为 | ZeroMQ 在每条连接上维护队列；队列满后是丢弃还是阻塞，由 socket 类型决定，直接决定可靠性边界。 |
| bind / connect 与自动重连 | ZeroMQ 的连接管理不对应"服务端/客户端"角色，两端顺序无关，断线自动重连，这是它与裸 socket 的关键差异。 |
| 传输层（inproc / ipc / tcp） | 同一套 socket 语义可换传输层，是"模式与传输解耦"的具体表现。 |
| 内部 I/O 线程与 context | 消息的实际收发发生在库自己的后台线程，这解释了 `send` 返回成功却未上网络的现象。 |

### 不包括什么

| 内容 | 排除理由 |
|---|---|
| 消息持久化、重投递、事务、确认（ack） | ZeroMQ 不提供这些。PIPELINE 规范只承诺"mostly reliable insofar as it will not discard messages unless a node disconnects unexpectedly"，可靠投递需由应用层协议自行实现。 |
| broker / 服务注册 / 服务发现 | 不在库内。ZeroMQ 提供 XPUB/XSUB 等原始 socket 类型供应用自己搭代理，但库本身不含任何常驻服务。 |
| CurveZMQ 加密与 ZAP 认证细节 | 属于独立的安全机制规范（RFC 25/ZMTP-CURVE、RFC 27/ZAP），不影响本页学习目标的回答。 |
| draft socket 类型（CLIENT/SERVER/RADIO/DISH 等） | 官方文档标记为 draft，接口可能变动；核心语义可由稳定类型完整说明。 |
| 各语言绑定的 API 差异 | 属于工程细节，不改变概念机制。 |

### 相邻概念

| 相邻概念 | 关键区别 | 是否纳入 |
|---|---|---|
| 消息队列中间件（Kafka / RabbitMQ） | 那些系统有独立 broker 进程、消息落盘、消费位点；ZeroMQ 是进程内库，无 broker、无持久化。 | 作为常见误解处理，不展开中间件本身 |
| 裸 TCP socket | TCP 提供字节流，无消息边界、无多对端分发、断线不自恢复；ZeroMQ 在其上补齐这三点。 | 纳入，作为问题背景 |
| gRPC / HTTP RPC | 那些框架绑定了请求-应答语义与序列化格式；ZeroMQ 只管消息搬运，不定义 payload 格式，也不限于请求-应答。 | 仅在适用边界处一句区分 |
| MPI | 面向紧耦合并行计算的集合通信，进程组固定；ZeroMQ 面向松耦合服务，节点可随时加入退出。 | 不展开 |

## 2. 学习目标

### Q1：ZeroMQ 解决了裸 TCP 之上哪三个具体问题？

- 完成答案：TCP 只提供无分隔的字节流，应用需要自己切分消息边界；TCP 连接是一对一的，应用需要自己管理多个对端并决定消息发给谁；TCP 连接断开后需要应用自己重连并处理未发出的数据。ZeroMQ 分别用帧、socket 类型的路由策略、以及自动重连 + 每连接队列解决这三点。
- 为什么是核心目标：不理解它解决什么问题，就无法判断何时该用它、何时裸 socket 或 HTTP 更合适，也无法理解 socket 类型为何存在。
- 依赖内容：TCP 字节流特性（无需概念页，正文一句说明即可）；帧结构（K2）；socket 类型（K1）。

### Q2：socket 类型如何决定一条消息去往哪个对端？

- 完成答案：ZeroMQ 的 `zmq_send` 不指定收件人，收件人由 socket 类型的出站策略决定。PUSH/DEALER/REQ 用轮询（round-robin）在可用对端间逐个分发，一条消息只到一个对端；PUB/XPUB 用扇出（fan out），一条消息复制给所有匹配订阅的对端；ROUTER 从消息第一帧取出路由标识（routing id），定向发给对应队列。入站侧 PULL/DEALER/ROUTER/REP/SUB 都用公平队列（fair-queuing）轮流从各对端读取。
- 为什么是核心目标：这是 ZeroMQ 与所有"显式指定目标地址"的通信方式最根本的差别，也是读别人 ZeroMQ 代码时首先要判断的东西。
- 依赖内容：轮询与公平队列定义（K3）；ROUTER 信封（K4）。

### Q3：队列满时消息会丢还是会阻塞，取决于什么？

- 完成答案：取决于 socket 类型。ZeroMQ 为每个对端单独维护队列，容量由 `ZMQ_SNDHWM` / `ZMQ_RCVHWM` 控制（默认 1000 条消息）。队列满后 socket 进入 mute 状态：PUB / XPUB / XSUB / ROUTER（默认配置）丢弃消息且 `zmq_send` 永不阻塞；PUSH / PULL / REQ / DEALER / PAIR 阻塞或返回 EAGAIN 且不丢消息。所以"用了 ZeroMQ 消息就不会丢"是错的，是否丢消息由所选 socket 类型直接决定。
- 为什么是核心目标：这是生产环境最常踩的坑，也是"ZeroMQ 是否可靠"这个问题的准确答案所在。
- 依赖内容：每连接队列（K5）；HWM 语义（C7、C8）；mute 状态（C9）。

### Q4：一条 ZeroMQ 消息在 TCP 线上长什么样？

- 完成答案：连接建立后先交换 64 字节 greeting（10 字节签名 + 2 字节版本 + 20 字节安全机制名 + 1 字节 as-server + 31 字节填充），再交换 READY 命令帧携带 Socket-Type 等元数据，之后才是应用消息。每个应用帧的结构是 1 字节 flags + 长度字段（短帧 1 字节 / 长帧 8 字节）+ 帧体；flags 的 bit 0 是 MORE（后面还有帧）、bit 1 是 LONG、bit 2 是 COMMAND。多段消息就是一串 MORE=1 的帧后跟一个 MORE=0 的帧，原子交付。
- 为什么是核心目标：不看到线上字节，"消息边界"和"多段消息原子性"始终是抽象说法；抓到字节后 ROUTER 信封、订阅过滤发生在哪一侧等问题都能自己判断。
- 依赖内容：ZMTP 帧格式（F1、C10、C11）；greeting（C12）。

### Q5：ZeroMQ 不提供哪些保证，因此哪些事必须应用自己做？

- 完成答案：不提供持久化、不提供重投递、不提供投递确认、不提供跨重启的消息留存；`zmq_send` 返回成功仅表示消息已进入库的队列并由库接管，不表示已上网络、更不表示对端已收到。PUB/SUB 的订阅生效需要时间（slow joiner），在此之前发出的消息无处可去；非线程安全的 socket 类型不得跨线程共享。因此可靠投递、幂等、超时重试、心跳探活都必须应用层实现。
- 为什么是核心目标：这些边界决定了 ZeroMQ 能不能用在某个场景；不清楚它们会写出"看起来能跑，压力上来就掉数据"的代码。
- 依赖内容：Q3 的丢弃语义；C13（send 语义）；C14（slow joiner）；C15（线程安全）。

## 3. 内容分级

### 核心内容

| 编号 | 内容 | 对应目标 | 必须说明的结论 |
|---|---|---|---|
| K1 | socket 类型是语义载体 | Q2 | 同一套 API，出入站路由策略由类型决定，类型之间有兼容约束 |
| K2 | 帧与多段消息 | Q1、Q4 | 传输单位是帧；一条消息可含多帧；全收或全不收 |
| K3 | 轮询、扇出、公平队列三种策略 | Q2 | 三种策略各自的分发结果，以及"轮询会跳过满队列的对端" |
| K4 | ROUTER 的路由标识信封 | Q2 | 入站加一帧标识，出站剥一帧标识；标识不存在时默认静默丢弃 |
| K5 | 每对端一条队列 | Q3 | 队列是 per-peer 而非 per-socket；HWM 是单个对端的上限 |
| K6 | HWM 与 mute 状态下的丢弃/阻塞分工 | Q3 | 哪些类型丢、哪些类型阻塞，以及默认值 1000 |
| K7 | ZMTP greeting 与帧头字节布局 | Q4 | 64 字节 greeting、flags 三个有效位、短帧/长帧长度字段 |
| K8 | bind/connect 顺序无关与自动重连 | Q1、Q5 | connect 可先于 bind；断线后队列保留并自动重连 |
| K9 | send 的返回语义 | Q5 | 成功只代表入队，不代表上网络或对端已收 |
| K10 | 订阅过滤发生在发布端 | Q2、Q5 | ZMTP 下 SUB 把订阅前缀发给 PUB，PUB 端做二进制前缀比较后再决定是否发送 |

### 辅助内容

| 编号 | 内容 | 服务对象 |
|---|---|---|
| B1 | "ZeroMQ 不是 broker" | 消除把它当 Kafka/RabbitMQ 的误解 |
| B2 | "Zero" 的含义（零 broker、零延迟设计目标） | 澄清名字带来的误解，说明它不代表"零延迟" |
| B3 | I/O 线程与 context | 解释 K9 为什么成立 |
| B4 | inproc / ipc / tcp 传输对比 | 说明模式与传输解耦（K1 的延伸） |
| B5 | slow joiner 现象 | Q5 的具体表现 |

### 扩展内容

| 编号 | 内容 | 纳入? |
|---|---|---|
| E1 | XPUB/XSUB 搭建发布代理 | 排除。属于应用层拓扑设计，不影响学习目标 |
| E2 | CurveZMQ 加密握手 | 排除。属独立安全规范 |
| E3 | 心跳（ZMQ_HEARTBEAT_IVL）与探活 | 仅在 Q5 边界处一句提及存在，不展开参数 |
| E4 | 在推理系统中的实际用法 | 排除。属工程实践，非概念机制 |

## 4. 前置知识映射

本页目标读者不具备该领域背景，需要的前置知识如下：

| 前置知识 | 被哪些目标依赖 | wiki 中是否有概念页 | 处理方式 |
|---|---|---|---|
| TCP 是无分隔的字节流 | Q1、Q4 | 无 | 不递归生成。这不是需要独立概念页的机制，正文用一句话加一个具体例子（两次 send 可能被合并读出）说明即可 |
| 进程 / 线程 / 阻塞调用 | Q3、Q5 | 无 | 不递归生成。属通识，正文首次使用时就地说明 |
| 网络传输路径与通信原语 | 无（不被任何学习目标依赖） | 有：`wiki/gpu-communication/index.html` | 不引用。该页讲的是 GPU 间集合通信与 PCIe/NVLink/RDMA 硬件路径，与 ZeroMQ 的应用层消息语义不在同一层，强行引用会引入无关概念 |

结论：本页不递归生成任何前置概念页。wiki 中现有页面（`gpu-communication`、`moe-serving`、`vllm-cudagraph`、`expertplex` 等）均属 GPU 计算与推理系统主题，没有一个是本页学习目标的前置依赖。按 `guides/concept/plan.md` 第 2.4 节，前置知识只在"读者理解核心机制前必须掌握"时才建立引用，因此本页不制造跨主题链接。

## 5. 明确不展开的内容

| 内容 | 与概念的关系 | 不展开的原因 |
|---|---|---|
| 19 种 socket 类型逐一讲解 | 都是 ZeroMQ 的 socket 类型 | 8 种稳定类型（REQ/REP/DEALER/ROUTER/PUB/SUB/PUSH/PULL）足以覆盖三种核心路由策略与两种 HWM 行为；draft 类型接口未定，逐一列举不增加机制理解 |
| CurveZMQ / ZAP 安全机制 | ZMTP 的可插拔安全机制 | 属于另一组独立规范（RFC 25、RFC 27），不影响任何学习目标的回答 |
| 各语言绑定 API 差异 | 使用层面 | 不改变机制，属工程细节 |
| 高可用模式（Majordomo、Freelance 等） | 基于 DEALER/ROUTER 构建的应用层协议 | 属于用 ZeroMQ 搭出来的东西，不是 ZeroMQ 本身 |
| ZeroMQ 内部 mailbox / signaler 实现 | libzmq 内部实现 | 只影响工程规模与性能，不改变对外语义 |

## 6. 常见误解和适用边界

### 误解

| 编号 | 错误理解 | 正确结论 | 形成原因 | 影响目标 |
|---|---|---|---|---|
| M1 | ZeroMQ 是消息队列中间件，要先部署一个 ZeroMQ 服务 | 它是链接进应用进程的库，没有任何常驻服务进程。所谓"队列"是 socket 在自己进程内存里为每个对端维护的缓冲区 | 名字里有 MQ，与 RabbitMQ / ActiveMQ 形似 | Q1、Q3 |
| M2 | 用了 ZeroMQ 消息就不会丢 | 是否丢消息由 socket 类型决定。PUB 在订阅端队列满时静默丢弃且 send 永不阻塞；PUSH 则阻塞不丢。实测：SNDHWM=5 的 PUB 连续发 200 条，订阅端只收到 5 条 | 把"消息库"等同于"可靠投递" | Q3、Q5 |
| M3 | `zmq_send` 返回成功说明对端收到了 | 官方文档明确："A successful invocation of `zmq_send()` does not indicate that the message has been transmitted to the network, only that it has been queued on the socket and 0MQ has assumed responsibility for the message." | 与阻塞式写 socket 的直觉混淆 | Q5 |
| M4 | bind 的一端是服务端，connect 的一端是客户端，必须先 bind | bind/connect 只决定谁提供端点地址，不决定消息方向也不决定顺序。实测：PUSH 先 connect 到无人监听的地址，消息照样入队；PULL 随后 bind，消息立刻被收到 | 沿用 BSD socket 的角色直觉 | Q1 |
| M5 | SUB 端设置订阅前缀，所以过滤是在订阅端做的、网络上仍然传全量 | ZMTP 规定"message filtering SHALL happen at the publisher side"。实测抓包：SUB 只订阅前缀 `A` 时，`B-nomatch` 这条消息根本没有出现在 TCP 连接上 | 订阅 API 写在 SUB 一侧 | Q2、Q4 |
| M6 | 一个 socket 可以在多个线程里共享使用 | 除 draft 的线程安全类型外，全部常用类型都非线程安全，文档写明"Applications MUST NOT use a *not* thread safe socket from multiple threads under any circumstances. Doing so results in undefined behaviour." | 把它当作普通 fd | Q5 |

### 适用边界

| 维度 | 内容 |
|---|---|
| 解决什么问题 | 在字节流传输之上提供离散消息、多对端分发策略、断线自恢复，让应用不必自己写这三层胶水代码 |
| 不解决什么问题 | 消息持久化、投递确认、重投递、事务、服务发现、消息格式定义 |
| 结论成立需要的条件 | 通信双方都使用 ZMTP 兼容实现（STREAM 类型例外）；socket 类型配对合法；单个 socket 只在一个线程内使用；队列容量足以吸收突发流量 |
| 条件不满足时会发生什么 | socket 类型不配对时可能连上但语义错乱（如 REQ 连 PUB）；跨线程共享 socket 是未定义行为；队列不足时按类型丢弃或阻塞；进程崩溃时队列中未发出的消息随内存一起消失 |
