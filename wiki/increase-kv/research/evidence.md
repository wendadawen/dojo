# 增量 KV 传输——核心论断与证据

本页机制的直接来源是任务发起人 2026-09-03 提供的工程描述（行为级机制定义），无公开论文或文档定位；此类条目在「来源定位」注明「工程描述」，审查时按描述原文核对。背景性概念引用仓库前置页（其来源核对见各页来源章节）。

## C 论断（机制与归因）

| 编号 | 论断 | 来源定位 | 适用条件 | 置信状态 |
|---|---|---|---|---|
| C1 | PD 分离下 prefill 产出的整条 prompt KV 需要传到 D 侧才能继续 decode | 工程描述（「prefill 侧……把整条 prompt 的 KV 全量 RDMA 过来」隐含的前提）；背景佐证：wiki/hetero-pd、wiki/beyond-buzz-disaggregation 页 | PD 分离架构 | 已确认（描述基准） |
| C2 | D 节点自己也运行 prefix cache；请求前 $N$ 个 token 可在 D 侧本地命中，命中部分 KV 已在 D 侧显存 | 工程描述（「decode 节点自己也开了 prefix cache……这部分 KV 已经躺在 D 侧显存里」）；命中机制见 wiki/prefix-caching 页 | D 侧开启 prefix cache | 已确认（描述基准） |
| C3 | 旧行为下 P 侧不掌握 D 侧命中信息，把整条 prompt 的 KV 全量 RDMA 到 D，命中块被覆写一遍；覆写内容与已驻留内容相同，属纯冗余 | 工程描述（「仍然把整条 prompt 的 KV 全量 RDMA 过来，把命中的块又覆写一遍」）；「内容相同」由同一 prompt 前缀与同一模型确定性 prefill 推出（本页组织性论证） | 命中块确为同一前缀的 KV | 已确认（描述基准 + 推理） |
| C4 | D 侧算出 `num_skip_blocks`（命中块数），随 `block_ids` 消息传给 P 侧 | 工程描述（「D 侧算出一个 num_skip_blocks 随 block_ids 消息传给 P 侧」） | 消息通道存在（描述基准） | 已确认（描述基准） |
| C5 | P worker 发送前把 local/remote 两条 block id 列表在同一 skip 偏移处切掉同样长度的头部，然后发送剩余部分 | 工程描述（「P worker 发送前把 local/remote 两条 block id 列表在同一 skip 偏移处切掉同样长度的头部」） | skip 偏移有效（num_skip_blocks ≥ 0 且不超过总块数） | 已确认（描述基准） |
| C6 | 切头的效果是「少发字节且不破坏配对」：传输字节按命中块数线性减少；两条列表在同一偏移等长切头后，剩余部分保持按位配对 | 工程描述（「从而少发字节且不破坏配对」）；线性关系由 C5 直接推出 | — | 已确认（描述基准 + 推理） |
| C7 | local/remote 两条列表按位置配对、对应同一逻辑块序列；只在一条上切或切得不等长会使剩余部分错位、破坏配对 | 由 C5「同一偏移、同样长度」约束反推其必要性（本页组织性论证：若配对无关，同步切头即无必要） | P worker 的发送组织依赖按位配对 | 已确认（由描述约束推出的论证） |
| C8 | block id 与块粒度组织来自分页式 KV 管理（PagedAttention） | wiki/paged-attention 页（该页已核对 vLLM SOSP 2023） | 块式 KV cache 系统 | 已确认（引用链） |
| C9 | 命中判定与块粒度复用机制（radix tree 匹配、前缀逐 token 精确）来自 prefix caching | wiki/prefix-caching 页 | D 侧 prefix cache 开启 | 已确认（引用链） |
| C10 | 本机制只省传输，不省 P 侧 prefill 计算；P 侧是否重算是 P 侧 lookup（P 侧 prefix cache）那条线的事 | 工程描述的全部内容限于传输环节（描述中 P 侧行为仅「发送」），P 侧计算未在描述中出现；区分性陈述为本页组织性论证 | — | 已确认（描述范围 + 推理） |

## F 公式

| 编号 | 公式 | 来源定位 | 置信状态 |
|---|---|---|---|
| F1 | 每 token KV 占用 $m = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot b$ | wiki/kv-cache 页第 3 章（贯穿示例借用） | 已确认（引用链） |
| F2 | 旧传输量 $T_{old} = B_{total} \cdot m \cdot s_{blk}$；新传输量 $T_{new} = (B_{total} - n_{skip}) \cdot m \cdot s_{blk}$；节省率 $= n_{skip}/B_{total}$（$B_{total}$ 为 prompt 总块数，$n_{skip}$ 为命中块数，$s_{blk}$ 为每块 token 数） | 由 C5/C6 与块组织直接推出（本页组织视角） | 已确认（推导式） |

## N 数字

| 编号 | 数字 | 来源定位 | 实验条件 | 置信状态 |
|---|---|---|---|---|
| N1 | 贯穿示例模型：Llama-3.1-8B（GQA：32 层、8 个 KV 头、头维 128、bf16）→ 每 token 128 KB | wiki/kv-cache 页第 3 章（该页已核对的算例：$2\times32\times8\times128\times2=131{,}072$ B = 128 KB/token；Llama-3.1-8B 配置来源核对见该页） | 与前置页同一算例，保证跨页口径一致 | 已确认（引用链） |
| N2 | 贯穿示例场景：prompt 8192 token、每块 16 token（构造参数）、共 512 块；前 4096 token（256 块）在 D 侧命中 → `num_skip_blocks`=256；每块 KV = 16 × 128 KB = 2 MiB；旧传 1 GiB、新传 512 MiB | 构造示例（块大小 16 为构造参数便于手算，不声称任何框架的默认值；其余数字为 2 的幂便于复算） | 构造 | 已确认（构造声明） |
| N3 | （撤销：块大小 16 改为纯构造参数，见 N2；前置页未固定该数值，不引用外部默认值） | — | — | 已撤销 |

## 贯穿示例（构造数据）

- 模型取 kv-cache 页已核对的 Llama-3.1-8B 算例（每 token 128 KB）；prompt 8192 token、块大小 16（构造）、D 侧头部命中 4096 token。
- 全部数字可手算：每块 $16 \times 128$ KB $= 2$ MiB；512 块 $= 1$ GiB；skip 256 块后剩 256 块 $= 512$ MiB。
- 命中率扫描（同一示例改命中块数）：0 / 128 / 256 / 384 / 512 块命中时的传输量表，展示节省率 $= n_{skip}/B_{total}$。
