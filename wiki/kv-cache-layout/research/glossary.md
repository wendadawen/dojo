# 术语表

全文首次出现位置以章节计（写作时以最终页面为准）。同一对象全文只用一种写法。

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| KV cache | 第 1 章 | 每层为每个 token 缓存的 $k/v$ 向量集合；完整讲解见 kv-cache 页 |
| $N$ | 第 1 章 | token 数（序列方向维度大小）；分页语境下指一页内的 token 数，此时与页大小 $p$ 同义，页面统一用 $p$ 指页内、$N$ 指视图逻辑形状的 token 维 |
| $H_{\mathrm{kv}}$ | 第 1 章 | KV 头数；GQA 下少于查询头数（见 mqa-gqa 页）；vLLM 文档中对应 H |
| $d$ | 第 1 章 | 每个头的维度 head_dim；vLLM 中对应 C 的构成部分 |
| $p$ | 第 1 章 | 页大小：一页（块）内的 token 数，vLLM 的 block_size |
| $n_{\text{page}}$ | 第 1 章 | 页（块）总数 |
| 布局（layout） | 第 1 章 | 三维 $N/H/D$ 在物理内存中的排列顺序 |
| NHD | 第 1 章 | (num_tokens, num_heads, head_dim) 排列；同一 token 的所有头连续 |
| HND | 第 1 章 | (num_heads, num_tokens, head_dim) 排列；同一头的整页 token 连续 |
| HDN | 第 1 章 | 非标准拼写，页面仅在第 1 章澄清处出现一次 |
| 页（page）/ 块（block） | 第 1 章 | 分页 KV cache 的分配单位，见 paged-attention 页；vLLM 源码称 block，本文统一称页 |
| $b$ | 第 3 章 | 每元素字节数（fp16 为 2、fp8 为 1） |
| 传输单元 | 第 3 章 | 一次连续内存访问/搬运的最大元素数 |
| offload | 第 3 章 | 把 KV cache 从 GPU 显存移到 CPU 内存（及反向），页级搬运的典型场景 |
| fp8 / nvfp4 | 第 3 章 | 低精度 KV cache 数据类型；只使用"每元素字节数"性质 |
| trtllm-gen | 第 3 章 | TensorRT-LLM 生成并由 FlashInfer 暴露的注意力 kernel 家族，SM100 上使用 |
| TMA | 第 3 章 | Tensor Memory Accelerator，Blackwell 的张量搬运单元；仅提及 16 字节盒宽约束 |
| stride（步长） | 第 4 章 | 张量某维下标每加一，物理位置移动的元素数 |
| 视图（view） | 第 4 章 | 同一物理 storage 按给定形状与步长解释出的张量 |
| `as_strided` | 第 4 章 | PyTorch 按 (形状, 步长) 构造视图的函数 |
| $L$ | 第 4 章 | vLLM 逻辑形状的层数维 |
| $B$ | 第 4 章 | vLLM 逻辑形状的页数维（num_blocks） |
| $C$ | 第 4 章 | 每个 (头槽, 存储状态) 格的字节数；标准注意力下 $=(d + d_v) \cdot b'$（$b'$ 为 dtype 字节数，即 K 与 V 拼接） |
| 存储状态（state） | 第 4 章 | 一格 C 对应的缓存内容；标准注意力一 token 一状态，故 $N=p$ |
| `KVCacheLayout` | 第 4 章 | vLLM 枚举，成员值为逻辑轴到物理顺序的 stride 置换 |
| LBNHC / LBHNC | 第 4 章 | `KVCacheLayout` 成员；分别等价 NHD / HND |
| `VLLM_KV_CACHE_LAYOUT` | 第 4 章 | vLLM 环境变量，显式指定布局 |
| `supported_kv_cache_layouts` | 第 4 章 | 注意力后端声明的偏好布局列表（最前最偏好），或 None 表示无偏好 |
| SM100 / SM90 / SM12x | 第 4 章 | NVIDIA 计算能力代号：SM100/103=Blackwell（B200 等）、SM90=Hopper（H100 等）、SM12x=RTX 50 系 |
| `reshape_and_cache_flash` | 第 4 章 | vLLM CUDA 写入算子，期望 NHD 形状的 K/V 输入 |
| GQA | 第 1 章（引用） | 分组查询注意力，见 mqa-gqa 页 |
| prefill / decode | 第 2 章 | 预填充阶段批量处理提示词 / 逐 token 生成阶段 |
