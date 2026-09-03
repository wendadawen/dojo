# KV cache 布局（NHD 与 HND）：内容范围

## 1. 概念含义

### 1.1 名称与缩写

- 概念名称：KV cache 的显存布局（memory layout），特指 NHD 与 HND 两种排列
- 英文名称：KV-Cache layout (NHD / HND)
- 常见缩写：NHD = (num_tokens, num_heads, head_dim)；HND = (num_heads, num_tokens, head_dim)
- 字母顺序即维度从外到内的物理排列顺序。"HDN"不是标准写法，业界只有 NHD 与 HND 两种命名（FlashInfer 官方文档只定义这两种）。

### 1.2 简要定义

KV cache 每层为每个 token 存 $H_{\mathrm{kv}}$ 个 KV 头、每头 $d$ 维向量。这批数放进显存时，三个维度（token 数 $N$、头数 $H$、头维 $d$）谁排在外层、谁排在内层，就是布局。NHD 让同一 token 的所有头连续；HND 让同一头的整页 token 连续。

### 1.3 正式定义（来源见 evidence.md）

- FlashInfer 官方 kv_layout 教程：NHD 为最后三维 $(\text{seq\_len}, \text{num\_heads}, \text{head\_dim})$；HND 为 $(\text{num\_heads}, \text{seq\_len}, \text{head\_dim})$。
- 分页形式（FreeKV 论文 §4.2）：NHD 为 $(n_{\text{page}}, p, n_{\text{kv}}, d)$；HND 为 $(n_{\text{page}}, n_{\text{kv}}, p, d)$，其中 $p$ 为页大小。
- vLLM main 分支（`vllm/v1/kv_cache_layout.py`，RFC #42082）：逻辑形状恒为 $[L, B, H, N, C]$（层数、块数、头槽、块内状态数、每格字节数），`KVCacheLayout` 枚举值是 stride 置换；兼容别名 NHD→LBNHC、HND→LBHNC。

### 1.4 本文采用的语境

推理引擎的 decode/prefill 执行路径中，分页 KV cache 的显存排布与读写代价。以 FlashInfer 的三维定义为主轴，以 vLLM main 分支的实现为工程落地视角。

### 1.5 包括什么

- 三维 $N/H/D$ 的含义与两种排列的物理后果（谁连续、谁跨步）
- 写入路径为什么 NHD 自然（投影输出形状）
- 读取路径 HND 换来什么（低精度 kernel、页级搬运的连续传输单元）
- 分页形式的两种布局
- vLLM 的实现：KVCacheLayout 枚举与 stride 置换、后端协商、`VLLM_KV_CACHE_LAYOUT`、SM100 上解析为 head-major 的原因
- `as_strided` 双视图的本地验证（同一显存两种布局）

### 1.6 不包括什么

- MLA 的 latent cache 布局：MLA 缓存的是压缩表示，维度结构不同，见已有页面 mla
- FlashInfer 的 page table / indptr 寻址机制：见已有页面 paged-attention
- FreeKV 的 streamed recall 全机制：只引用其布局与传输单元结论
- 各硬件后端的完整选择矩阵（TRT-LLM/XQA/FA2/FA3 的全部判定条件）：只讲与布局直接相关的 SM100 trtllm-gen 路径
- NVFP4 / FP8 量化本身：见相关量化页，本文只用其"每元素字节数变少"这一性质
- 旧版 vLLM（RFC #42082 之前）逐版本的历史演变：只在演进注记中说明旧写法（每层独立 buffer + permute），不做版本考古

### 1.7 相邻概念

- 分页 KV cache（paged attention）：布局是"页内三维怎么排"，分页是"token 到物理块的映射"。分页解决分配碎片，布局决定页内读写模式。两者正交但都在页粒度上讨论。纳入引用（paged-attention 页），不重复讲解。
- GQA/MQA：决定 $H_{\mathrm{kv}}$ 与查询头数的比例，影响"同头整页连续"的收益大小。纳入引用（mqa-gqa 页）。
- 张量 stride（步长）：理解布局切换的数学基础。无已有页面，正文内给最小含义（元素下标每加一，物理位置移动多少）。

## 2. 学习目标

### Q1：KV cache 的三个维度是什么，NHD 与 HND 两种布局分别把谁排在外层？

- 完成答案：读者能说出 $N$（token 数）、$H$（KV 头数）、$d$（每头维度）的来源；能画出/写出同一页数据在两种布局下的物理顺序；能说出"同一 token 的头连续（NHD）"与"同一头的整页 token 连续（HND）"这一根本差异；能写出分页形式 $(n_{\text{page}}, p, H, d)$ 与 $(n_{\text{page}}, H, p, d)$
- 为什么是核心目标：不掌握维度含义与排列，后续所有"谁连续、谁跨步"的讨论都无从谈起
- 依赖内容：KV cache 存什么（kv-cache 页）、GQA 下 $H_{\mathrm{kv}}$ 的含义（mqa-gqa 页）

### Q2：为什么 NHD 是写入路径的自然布局？

- 完成答案：读者能说明注意力投影输出 $xW_K$、$xW_V$ 的形状本来就是 $(N, H_{\mathrm{kv}} \cdot d)$，reshape 成 $(N, H, d)$ 零成本；能解释 decode 每步追加 token 时 NHD 的写入为什么是按头并行的小段连续写；能复算 FlashInfer 的论断"NHD 与投影输出一致、无需转置"
- 为什么是核心目标：解释"为什么默认是 NHD"这一事实，防止把 HND 误当成普遍更优
- 依赖内容：Q1 的维度定义、KV cache 写入时机（prefill 批量写、decode 追加写，kv-cache 页）

### Q3：HND 把什么变连续，这个连续性在哪些场景换来收益？

- 完成答案：读者能计算两种布局下"取一个 KV 头的一整页"的传输单元大小（NHD 为 $d$ 个元素、HND 为 $p \cdot d$ 个元素），能复算 $d=128$、fp16、$p=32$ 时的 256 B 对 8 KB 对比；能说出 HND 有收益的两类场景——低精度（fp8/nvfp4）kernel 的显存访问（trtllm-gen 要求 head-major）与页级大块搬运（offload/换入换出）；能说出 fp16 下两者性能差异不显著（FlashInfer 实测表述）
- 为什么是核心目标：这是 HND 存在的全部理由，也是本文的核心结论
- 依赖内容：Q1/Q2 的结论、每元素字节数与 dtype 的关系

### Q4：vLLM 怎么在同一块显存上支持两种布局，SM100 上为什么解析成 HND？

- 完成答案：读者能解释"逻辑形状固定 $[L,B,H,N,C]$ + stride 置换"如何让同一物理 buffer 表达两种布局（as_strided 机制）；能说出 `KVCacheLayout` 枚举成员与 NHD/HND 别名的对应；能描述后端协商（`supported_kv_cache_layouts` 交集）与 `VLLM_KV_CACHE_LAYOUT` 环境变量的作用；能说明 SM100 trtllm-gen kernel 只接受 head-major 块内布局、因此 Blackwell 上 FlashInfer 声明 (LBHNC, BLHNC)；能解释写入 kernel 如何通过 transpose 适配布局
- 为什么是核心目标：用户要求的深度终点；把前三章的概念落到真实工程代码
- 依赖内容：Q1–Q3 全部结论、stride 概念（正文内最小解释）

## 3. 内容分级

### 核心内容（缺一不可）

| 内容 | 支持的目标 | 必须说明的结论 |
|---|---|---|
| 三维 $N/H/D$ 定义与两种排列 | Q1 | 字母顺序=物理维度顺序；分页形式 |
| NHD 与投影输出一致 | Q2 | $xW_K$ 输出 $(N, H\cdot d)$，NHD 零转置 |
| HND 的传输单元对比（256 B / 8 KB） | Q3 | FreeKV 论文数字及条件 |
| HND 对低精度 kernel 友好 | Q3 | FlashInfer 表述 + trtllm-gen 文档（NHD 触发 transpose+拷贝） |
| fp16 下无显著差异、默认 NHD | Q3 | FlashInfer 表述 + vLLM 默认偏好 LBNHC |
| KVCacheLayout 枚举与 stride 置换 | Q4 | 逻辑形状固定、枚举值=置换、别名映射 |
| 后端协商与环境变量 | Q4 | 交集协商、覆盖语义、解析时机 |
| SM100 强制 head-major | Q4 | trtllm-gen 消费 head-major 块内布局 |
| 写入路径 transpose 适配 | Q4 | HND 视图 transpose(1,2) 后喂 NHD kernel |

### 辅助内容（消除障碍）

| 内容 | 服务的核心内容 |
|---|---|
| stride（步长）最小解释 + as_strided 双视图本地验证 | Q4 的机制理解 |
| FreeKV 混合布局（GPU 用 NHD、CPU 用 HND） | Q3 的应用实例，展示"没有全局最优" |
| vLLM 演进注记：旧版每层独立 buffer + permute 写法 | Q4，帮读者对照旧代码 |
| "HDN 不是标准写法" | Q1，用户实际遇到的拼写困惑 |

### 扩展内容

| 内容 | 纳入/排除 |
|---|---|
| FlashInfer 五字母布局名（LBNHC 等）全部六种 | 排除：只讲 NHD/HND 对应的两种 + 提及其他四种存在 |
| XQA kernel、FA2/FA3 的布局偏好细节 | 排除：与主线无关 |
| 混元 VL 3.0 部署语境 | 排除：内部文档不可引用；通用场景（Blackwell 部署选型）覆盖同一需求 |

## 4. 前置知识映射

| 前置概念 | 被谁依赖 | 页面状态 |
|---|---|---|
| KV cache 是什么、为何存在、prefill 写/decode 追加 | Q1、Q2 | 已有：wiki/kv-cache/index.html |
| 分页 KV cache（页、块表） | Q1（分页形式）、Q3（页级搬运） | 已有：wiki/paged-attention/index.html |
| GQA/MQA（KV 头数与查询头数） | Q1（$H_{\mathrm{kv}}$ 含义） | 已有：wiki/mqa-gqa/index.html |
| 张量 stride/视图 | Q4 | 无页面；正文给最小含义：元素下标每加一，物理地址移动的元素数。不递归生成（as_strided 代码示例即完整解释，独立成页价值不足） |

## 5. 明确不展开的内容

- MLA 压缩缓存布局：属于 mla 页的职责，本文只在"不包括什么"处指路
- 页表/indptr 寻址：paged-attention 页已有，本文只在分页形式处引用
- FreeKV 的异步流式召回与精度实验：论文机制超出布局主线，只引布局结论
- TMA（Tensor Memory Accelerator）硬件细节：只在解释 trtllm-gen 连续性要求时提及"16 字节盒宽"约束，不展开 TMA 架构
- 逐版本 vLLM 演变史：一条演进注记即可，不做考古

## 6. 常见误解与适用边界

### 误解

1. 错误理解："HND 是更新的格式，性能更好，应该默认用 HND。"
   正确结论：fp16 下两者性能差异不显著（FlashInfer 官方表述）；NHD 与投影输出一致、写入零转置，vLLM 默认偏好 LBNHC（NHD）。HND 的收益集中在低精度 kernel 与页级大块搬运两类读路径。
   形成原因：把"Blackwell 上被强制"误推广为"普遍更好"。
   影响：Q3。

2. 错误理解："切换布局要搬运/复制数据。"
   正确结论：布局是同一块显存的不同 stride 视图（as_strided/permute），逻辑数据等价、物理重排只发生一次性的 permute 或发生在写入/读取 kernel 内部，不需要复制整块缓存。
   形成原因：把"布局"误解为"两种独立的数据副本"。
   影响：Q4。

3. 错误理解："HDN 是第三种布局。"
   正确结论：标准命名只有 NHD 和 HND，字母顺序即维度顺序；HDN 是拼写串位。
   形成原因：口口相传或笔记笔误。
   影响：Q1。

4. 错误理解："HND 解决 KV cache 的碎片化问题。"
   正确结论：分配层面的碎片由分页机制解决（paged-attention）；HND 解决的是页内"跨头交错"导致的读取碎片——单头取整页时的传输单元从 $d$ 个元素升到 $p \cdot d$ 个。
   形成原因：把"碎片"一词不加区分地用于分配和访问两个层面。
   影响：Q1、Q3。

### 适用边界

- 概念解决：页内三维排列的读写模式选择。
- 不解决：分配碎片（分页解决）、容量规划（kv-cache 页）、量化精度（量化页）。
- 结论成立条件：传输单元对比基于 fp16、$d=128$、$p=32$ 的 FreeKV 论文设定，换参数数值变但"NHD 单头跨步、HND 单头连续"的方向不变；"fp16 无显著差异"是 FlashInfer 的实践表述，不构成数学保证。
- vLLM 细节基于 2026-09-03 拉取的 main 分支；`[L,B,H,N,C]` 逻辑形状与协商机制是 RFC #42082 引入的现行形态，旧版代码（每层独立 buffer、`(num_blocks, 2, block_size, H, d)` 形状加 permute）机制同源但形状体系不同。
