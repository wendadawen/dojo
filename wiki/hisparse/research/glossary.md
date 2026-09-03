# HiSparse 术语表

登记全文术语、缩写与符号；写作与审查以此为准，保证全文一致。

## 符号

| 符号 | 含义 | 原文定位 |
|---|---|---|
| $L_{\text{ctx}}$ | 请求上下文长度（token 数） | Table 2 |
| $k$ | 稀疏 indexer 每个查询选中的 token 数 | Table 2 |
| $\mathcal{S}_t^{(\ell)}$ | 解码步 $t$、层 $\ell$ 的选择集；$|\mathcal{S}_t^{(\ell)}| = k$ | Table 2 |
| $B$ | 每请求每层 GPU cache 容量（KV 记录槽数）；$B \ge k$，部署时固定 | Table 2 |
| $N_{\text{batch}}$ | 并发解码请求数 | Table 2 |
| $N_{\ell}$ | 稀疏注意力层数 | Table 2 |
| $W_{\text{KV}}$ | 每 token 每层存储的 KV 元素数 | Table 2 |
| $s$ | 每 KV 元素字节数 | Table 2 |

写法约定：全文用 $B$（不写"B 格数"以外的变体）、$k$、$L_{\text{ctx}}$；选择集写 $\mathcal{S}_t^{(\ell)}$；层数写 $\ell$。比例写 $B{=}2k$ 形式。

## 术语与缩写

| 术语 | 首次出现 | 定义/含义 | 处理方式 |
|---|---|---|---|
| KV cache | 第 1 章 | 解码时保留的每 token 每层 key/value 状态 | 链接 wiki/kv-cache/ |
| HBM | 第 1 章 | GPU 高带宽显存（High Bandwidth Memory） | 首次出现给全称 |
| top-$k$ 稀疏注意力 | 第 1 章 | 每步只对 indexer 选出的 $k$ 个位置做注意力 | 链接 wiki/dsa/ |
| indexer（选择器/索引器） | 第 1 章 | 产生选择集 $\mathcal{S}_t^{(\ell)}$ 的组件；DSA 为学习到的 lightning indexer | 链接 wiki/dsa/，正文统一用"indexer" |
| DSA | 第 1 章 | DeepSeek Sparse Attention；token 级、co-trained；GLM-5.1/5.2 使用 | 接口级介绍，Table 1 事实 |
| NSA | 第 1 章 | Native Sparse Attention；block 级、trained | 接口级介绍 |
| Quest | 第 1 章 | training-free 的 page 级选择器（per-page min/max 摘要） | 接口级介绍 |
| 选择集漂移 | 第 1 章 | $\mathcal{S}_t$ 跨步变化、被跳过位置之后可能回来 | 正文定义 |
| 容量墙（capacity wall） | 第 1 章 | KV 驻留需求随 $N_{\text{batch}} \times L_{\text{ctx}}$ 增长先于算力耗尽 | 论文术语，正文定义 |
| admission（准入） | 第 1 章 | 调度器允许请求进入 decode batch | 正文定义 |
| TTFT | 第 1 章 | time to first token，首 token 前时延（含排队） | 首次出现定义 |
| TPOT | 第 1 章 | time per output token，首 token 后每 token 时延 | 首次出现定义 |
| PD-colocated / PD-disaggregated | 第 1 章 | prefill 与 decode 共享 / 分离 GPU 池 | 首次出现一句话定义 |
| prefill / decode | 第 1 章 | 输入的批量处理阶段 / 逐 token 生成阶段 | 链接 wiki/kv-cache/ |
| host KV pool | 第 2 章 | host pinned DRAM 中的完整 KV 权威副本池 | 正文定义 |
| pinned host memory | 第 2 章 | 页锁定主机内存，GPU 可直接对其发起加载 | 首次出现一句解释 |
| GPU cache（设备缓存） | 第 2 章 | 每请求每层 $B$ 格的 HBM 缓存（论文图注称 hot device buffer） | 正文定义；统一写"GPU cache" |
| 页表（page table） | 第 2 章 | 逻辑位置→cache 槽位或 host-only 哨兵的映射 | 链接 wiki/paged-attention/ 类比；正文定义本文语义 |
| LRU | 第 2 章（第 3 章展开） | 最近最少使用替换：淘汰最久未被访问的槽位 | 首次出现一句解释（不建概念页） |
| hit 提升（hit promotion） | 第 3 章 | 同一步内 hit 条目在 recency 序上排到新 fetch 的 miss 之上 | 论文机制，正文定义 |
| Bélady | 第 3 章 | 离线最优替换（淘汰将来最晚再访问者），作命中率上限对照 | 首次出现一句解释（不建概念页） |
| miss / 命中率 | 第 2 章 | 选择的位置不在 GPU cache / 命中比例 | 正文定义 |
| Resolve kernel | 第 4 章 | HiSparse 的融合 miss 解析 CUDA kernel（每稀疏层一次） | 论文机制 |
| top_k_device_locs | 第 4 章 | Resolve 输出的物理设备槽位稠密向量，与选中逻辑位置对齐 | 论文机制（代码符号） |
| GPU-assisted IO | 第 4 章 | GPU 线程直接对 pinned host 内存发向量化非一致加载（借自 Strata） | 链接 wiki/strata/ |
| CUDA graph | 第 4 章 | 录制固定形状 GPU 工作并重放；捕获使 Resolve 无 host 分支 | 链接 wiki/vllm-cudagraph/ |
| 软件管理 TLB 类比 | 第 4 章 | 论文原句：逻辑索引进、物理槽位出，像软件管理的 TLB | 标注为论文所用类比 |
| anchor 层 | 第 5 章 | 运行 top-$k$ indexer 的层 | 论文术语 |
| shared 层 | 第 5 章 | 复用前面 anchor 选择集的层 | 论文术语 |
| IndexCache / IndexShare | 第 5 章 | 跨层选择复用的方法（GLM-5.2 原生 IndexShare 每 4 层一组） | 引文事实 |
| plan-then-IO | 第 5 章 | anchor 记录 miss plan、copy-only kernel 重放到 shared 层的预取方案 | 论文机制 |
| no-IO oracle | 第 5 章 | 跳过全部 host IO 的性能上界配置（输出无效） | 论文机制 |
| write-through | 第 2 章 | 新 token KV 先落 reserved slot，再经 backup stream 写回 host pool | 论文机制 |
| HiCache | 第 2 章或评价章 | SGLang 的分层前缀缓存（跨请求复用）；HiSparse 复用其 host-tier 基础设施 | 链接 wiki/prefix-caching/ |
| SGLang | 第 1 章 | 开源 serving 框架，HiSparse 已并入上游 | 首次出现一句定位 |
| decode-only 速率 | 第 6 章 | 排除 prefill 时间的生成速率，作 PD 分离 decode 池吞吐的 proxy | 论文定义 |
| PCIe Gen5 ×16 / NVLink-C2C | 第 2/6 章 | host-device 互连；约 64 GB/s 每方向 / Grace-Hopper 上的高带宽一致互连 | 链接 wiki/gpu-communication/ |
| LongBenchV2 | 第 3 章 | 长上下文评测集（trace 来源） | 仅作实验条件说明 |
| BF16 / FP8 | 第 6 章 | KV 以 BF16 存储；模型名中的 FP8 指权重精度 | 实验设置说明 |

## 一致性规则

- 中文行文，系统名 HiSparse、kernel 名 Resolve、配置项 top_k_device_locs 保持英文原样（行内代码格式）。
- "选择集"统一此词（不用"选中集合/稀疏集"）；"驻留"统一（不用"常驻/驻留"混用时以"驻留"为主，"常驻"仅用于"常驻 GPU"这类固定搭配）。
- 层编号 $\ell$、步编号 $t$ 全文一致；示例中位置编号用正整数。
- 吞吐单位 tokens/s；时间 ms/s；显存 GB/TB；kernel 时间 μs。
