# 增量 KV 传输——术语表

全文首次出现即按下表登记；同一对象全页一种写法。

## 符号

| 符号 | 首次出现 | 含义 |
|---|---|---|
| $m$ | 第 3 章 | 每 token 的 KV cache 字节占用（沿用 kv-cache 页公式：$m = 2 \cdot L_{\text{layers}} \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot b$，写法与该页一致） |
| $L_{\text{layers}}$ | 第 3 章 | Transformer 层数 |
| $H_{\text{kv}}$ | 第 3 章 | 每层 KV 头数 |
| $d_{\text{head}}$ | 第 3 章 | 每个注意力头的维度 |
| $b$ | 第 3 章 | 每个缓存元素的字节数（bf16 为 2） |
| $s_{blk}$ | 第 3 章 | 每块 token 数（贯穿示例取 16，构造参数） |
| $B_{total}$ | 第 3 章 | prompt 的总块数（贯穿示例 512） |
| $n_{skip}$ | 第 3 章 | 头部连续命中的块数，即 `num_skip_blocks` 的数值（贯穿示例 256） |
| $N$ | 引言 | 请求头部在 D 侧命中的 token 数（叙述用；与 $n_{skip} \cdot s_{blk}$ 对应） |
| $T_{old}$ / $T_{new}$ | 第 3 章 | 旧 / 新机制的传输字节数 |

## 术语

| 术语 | 首次出现 | 定义或含义 |
|---|---|---|
| 增量 KV 传输（increase KV） | 引言 | 本页主题：D 侧前缀命中的块不再由 P 侧重传，P 侧只传 D 侧没有的增量部分 |
| PD 分离 | 引言 | prefill 与 decode 部署到不同节点的推理架构；本页只用其最小事实（P 算 prefill、D 算 decode、KV 需从 P 搬到 D） |
| P 侧 / P worker | 引言 | prefill 节点 / 其上执行 KV 发送的工作进程 |
| D 侧 | 引言 | decode 节点 |
| RDMA | 引言 | 远程直接内存访问：绕过两端 CPU 的网络内存读写，PD 间 KV 传输的载体（本页只取此最小含义） |
| prefix cache | 引言 | 跨请求复用相同前缀 KV 的缓存（机制见 prefix-caching 页） |
| 命中（hit） | 引言 | 请求前缀与缓存内容逐 token 精确匹配（见 prefix-caching 页） |
| block / 块 | 引言 | KV cache 的固定 token 数管理单元（见 paged-attention 页）；贯穿示例每块 16 token（构造） |
| block id | 第 1 章 | 块的编号，P/D 两侧指代同一逻辑块的共同语言 |
| `block_ids` 消息 | 第 2 章 | D 侧发给 P 侧、用于本次传输的消息（描述中的命名），`num_skip_blocks` 随它传递 |
| `num_skip_blocks` | 第 2 章 | D 侧算出的头部连续命中块数；P 侧按它在块序列头部切掉等长一段 |
| skip 偏移 | 第 2 章 | 切头位置：块序列前 $n_{skip}$ 块与剩余部分的分界 |
| local/remote 两条列表 | 第 2 章 | P worker 维护的两条按位置配对的 block id 列表（描述中的命名）；对应同一条 prompt 的块序列，本页只约定其配对行为，取数来源是实现细节 |
| 配对 | 第 2 章 | 两条列表相同位置的表项对应同一逻辑块这一按位关系 |
| 覆写 | 第 1 章 | 传输到达后把 D 侧已驻留的（内容相同的）KV 重写一遍 |
| 增量部分 | 第 2 章 | prompt 中 D 侧未命中、必须传输的后段块 |
| lookup | 第 3 章 | P 侧查自己 prefix cache 以避免重算的机制（省计算线，不属于本页机制） |

## 写法约定

- 中英混排时节点简称固定：P 侧（prefill）、D 侧（decode）；不使用「prefill 节点 / decode 节点」以外的新变体
- 字段与消息名一律行内代码格式：`num_skip_blocks`、`block_ids`
- 「skip」作为动词（跳过传输）与名词（skip 偏移 / skip 数）在句中自明；数值统一写 $n_{skip}$
- 构造示例数字（8192/512/256/16）与引用前置页的数字（128 KB/token、Llama-3.1-8B）不在同一表格列混排
