# 核心论断与证据

编号规则：C 论断 / F 公式 / N 数字。来源拉取/核对日期均为 2026-09-03。

来源缩写：

- [FI-doc] FlashInfer 官方文档 "KV-Cache Layout in FlashInfer"，docs.flashinfer.ai/tutorials/kv_layout.html；GitHub 固定版本 flashinfer-ai/flashinfer commit 3d43dc9dc1a2ae804eaa7e40b4555e471fd03fe3 的 docs/tutorials/kv_layout.rst（两者内容一致，rst 为固定锚点）
- [FI-trtllm-api] FlashInfer 官方 API 文档 flashinfer.prefill.trtllm_batch_context_with_kv_cache，docs.flashinfer.ai/generated/flashinfer.prefill.trtllm_batch_context_with_kv_cache.html
- [FreeKV] 论文 FreeKV, arXiv:2505.13109v2, §4.2 "Hybrid layouts and streamed recall"
- [vllm-main] vLLM main 分支源码，raw.githubusercontent.com 拉取于 2026-09-03（main 是滚动分支，审查时可重新拉取核对；关键文件与函数名如下）
- [vllm-v0.13.0] vLLM tag v0.13.0 源码，raw.githubusercontent.com 拉取于 2026-09-03，用于演进注记（C21）

## C 论断（机制与事实）

| 编号 | 论断 | 来源定位 | 原文片段/关键内容 | 适用条件 | 置信 |
|---|---|---|---|---|---|
| C1 | NHD 布局把最后三维组织为 (seq_len, num_heads, head_dim)，HND 组织为 (num_heads, seq_len, head_dim) | [FI-doc] Layout: NHD/HND 节 | "NHD: the last 3 dimensions are organized as (seq_len, num_heads, head_dim). HND: the last 3 dimensions are organized as (num_heads, seq_len, head_dim)." | 无 | 已确认 |
| C2 | NHD 是自然布局：与 $xW_K$、$xW_V$ 的输出一致，无需转置 | [FI-doc] Layout 节 | "The NHD layout is more natural because it's consistent with the output of xW_k and xW_v without transpose." | 标准注意力投影 | 已确认 |
| C3 | HND 对使用低精度数据类型（如 fp8）的 GPU 实现更友好 | [FI-doc] Layout 节 | "The HND layout is more friendly for GPU implementation when KV-Cache uses low-precision data type (e.g. fp8)." | 低精度 KV cache | 已确认 |
| C4 | fp16 KV cache 下两种布局没有显著性能差异，FlashInfer 优先 NHD | [FI-doc] Layout 节 | "In practice we don't observe significant performance difference between these two layouts on fp16 kV-Cache and we prioritize NHD layout for better readability." | fp16 | 已确认 |
| C5 | 分页形式下 NHD 为 $(n_{\text{page}}, p, n_{\text{kv}}, d)$，HND 为 $(n_{\text{page}}, n_{\text{kv}}, p, d)$ | [FreeKV] §4.2 | "the shapes of NHD and HND layouts are $(n_{\text{page}}, p, n_{\text{kv}}, d)$ and $(n_{\text{page}}, n_{\text{kv}}, p, d)$, respectively, where p is the page size." | 分页 KV cache | 已确认 |
| C6 | NHD 下给定 KV 头一页内 $p$ 个向量的内存不连续，最大传输单元仅 $d$ 个元素；HND 下 $p \cdot d$ 个元素连续 | [FreeKV] §4.2 | "under the NHD layout, for a given KV head, the memory of p = 3 key/value vectors within a page is non-contiguous. When recalling a key/value page, the maximum transfer unit contains only d elements"; "The HND layout ensures that p key/value vectors within a page are contiguous for each KV head, allowing a transfer unit of $p \times d$ elements" | 页级搬运（offload/召回）场景 | 已确认 |
| C7 | FreeKV 采用混合布局：GPU 内存用 NHD（消除解码期逐步转置），CPU 内存用 HND（保证 CPU-GPU 传输连续），只在 offload 页时做一次 NHD-HND 转置 | [FreeKV] §4.2 | "FreeKV employs the NHD layout on GPU to eliminate the need for per-step transposes during decoding, and the HND layout on CPU to ensure contiguous and efficient CPU-GPU data transfers during recall. With the hybrid layouts, the NHD-HND transpose is only required when offloading a KV page" | FreeKV 系统设计 | 已确认 |
| C8 | trtllm-gen 后端的 KV cache 以 HND 为默认；NVFP4 KV cache 下用 NHD 会触发自动 transpose 与 .contiguous() 拷贝（K/V 数据与 block scale 张量都会），产生额外内存分配与拷贝开销 | [FI-trtllm-api] 参数 kv_layout 说明 | "kv_layout (str = \"HND\") – Layout of kv-cache, can be \"HND\" or \"NHD\", default is \"HND\". For the trtllm-gen backend with NVFP4 KV cache, using NHD will trigger an automatic transpose and .contiguous() copy of both the KV data and block scale tensors to convert them to HND layout. This incurs extra memory allocation and data copy overhead. Use HND for better performance." | trtllm-gen 后端 | 已确认 |
| C9 | trtllm-gen kernel 要求 scale 张量最后两维 (page_size, head_dim//16) 连续，因为 kernel 把它们 reshape 成 (16, page_size·head_dim/16/16) 以满足 TMA 16 字节盒宽下限 | [FI-trtllm-api] Contiguity requirements 节 | "The last two dims (page_size, head_dim // 16) must be contiguous (i.e. stride[-1] == 1 and stride[-2] == head_dim // 16). This is because the kernel reshapes them into (16, page_size * head_dim / 16 / 16) to satisfy TMA's 16-byte box width minimum." | trtllm-gen 后端 + NVFP4 | 已确认 |
| C10 | vLLM main 的 KVCacheLayout 枚举：逻辑形状恒为 [L, B, H, N, C]，每个成员的值是逻辑轴到物理顺序的 stride 置换；LBHNC=(0,1,2,3,4)，LBNHC=(0,1,3,2,4) | [vllm-main] vllm/v1/kv_cache_layout.py L11-28 | "The logical shape is always [L, B, H, N, <content>] (RFC #42082). Each member's value is a stride permutation that maps logical axes to physical (memory) order."；"LBHNC = (0, 1, 2, 3, 4)  # [L, B, H, N, C] (identity)"；"LBNHC = (0, 1, 3, 2, 4)  # [L, B, N, H, C]" | vLLM main（RFC #42082 之后） | 已确认 |
| C11 | vLLM 中 NHD/HND 是兼容别名：NHD→LBNHC、HND→LBHNC | [vllm-main] vllm/config/cache.py L19-22 | `_LAYOUT_COMPAT_ALIASES = {"NHD": "LBNHC", "HND": "LBHNC"}` | 同 C10 | 已确认 |
| C12 | vLLM 逻辑形状 [L,B,H,N,C] 中：N 为每块存储状态数（标准注意力下等于块大小），C 为每 (头, 状态) 格的字节数，标准注意力下为 (head_size + head_size_v)×dtype 字节，即 K 与 V 沿 C 拼接 | [vllm-main] vllm/v1/kv_cache_interface.py L259-275、L390-418 | "Return the 4D logical shape (B, H, N, C) where C is in bytes."；"state_content_bytes: C in bytes when packed; None means dense K/V content."；"return (self.head_size + self.head_size_v) * get_dtype_size(self.dtype)" | 同 C10；head_size_v 默认等于 head_size | 已确认 |
| C13 | VLLM_KV_CACHE_LAYOUT 环境变量可取 LBNHC/LBHNC/LHBNC/NHD/HND/BLHNC/BLNHC/BHLNC，默认 None（布局选择留给后端）；envs 注释给出 N=num_states、H=num_heads、C=state_content | [vllm-main] vllm/envs.py L1804-1808 及 L1799-1803 注释 | "VLLM_KV_CACHE_LAYOUT": env_with_choices("VLLM_KV_CACHE_LAYOUT", None, ["LBNHC", "LBHNC", "LHBNC", "NHD", "HND", "BLHNC", "BLNHC", "BHLNC"])；注释 "Where N=num_states, H=num_heads and C=state_content. The default value will leave the layout choice to the backend." | vLLM main | 已确认 |
| C14 | 布局解析在 engine core 跑一次：每个后端声明 supported_kv_cache_layouts（偏好从前往后），取交集；VLLM_KV_CACHE_LAYOUT 显式指定时必须在候选内否则报错；解析结果记入 cache_config.kv_cache_layout 并经 RPC 传给 worker | [vllm-main] vllm/v1/attention/backends/utils.py L240-308 resolve_kv_cache_layout docstring 及 L287-293 | "An explicit VLLM_KV_CACHE_LAYOUT must be one of the candidates or resolution fails, with the legacy NHD/HND names as aliases"；"VLLM_KV_CACHE_LAYOUT={requested} does not satisfy every supported set" | vLLM main | 已确认 |
| C15 | 无后端声明偏好时的默认偏好序为 (LBNHC, LBHNC, BLNHC, BLHNC)，LBNHC（即 NHD）最优先，与 main 默认一致 | [vllm-main] vllm/v1/attention/backends/utils.py L174-180 | "_DEFAULT_LAYOUT_PREFERENCE = (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC, KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)"，注释 "LBNHC (NHD) first to match main's default." | vLLM main | 已确认 |
| C16 | SM100（设备能力 major==10，Blackwell）上 FlashInfer 后端只声明 (LBHNC, BLHNC) 两种 head-major 布局，因为 trtllm-gen kernel 消费 head-major 的块内布局，L/B 的嵌套顺序对 kernel 无关紧要 | [vllm-main] vllm/v1/attention/backends/flashinfer.py L536-543 | `if capability is not None and capability.major == 10: return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)`，注释 "The trtllm-gen kernels consume head-major block interiors; the L/B nesting outside the block is immaterial to them." | SM100 + FlashInfer 后端 | 已确认 |
| C17 | vLLM 把 KVCacheLayout 翻译成 FlashInfer 的布局名传给 kernel：LBNHC→"NHD"，LBHNC/BLHNC/BHLNC→"HND"，BLNHC→"NHD" | [vllm-main] vllm/v1/attention/backends/utils.py L156-171 | `_FLASHINFER_LAYOUT_NAMES = {"LBNHC": "NHD", "LBHNC": "HND", "BLHNC": "HND", "BLNHC": "NHD", "BHLNC": "HND"}`；flashinfer.py 多处调用 get_flashinfer_layout_string(self.kv_cache_layout) | vLLM main + FlashInfer kernel 调用 | 已确认 |
| C18 | vLLM FlashInfer 写路径：per-layer KV cache 视图形状 (B, H, N, 2·hs)（HND 风格，K/V 沿最后一维拼接），先 transpose(1,2) 转成 (B, N, H, hs) 再 split 出 K、V，喂给 reshape_and_cache_flash | [vllm-main] vllm/v1/attention/backends/flashinfer.py L2560-2574 | "# (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))"，`k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)`，`torch.ops._C_cache_ops.reshape_and_cache_flash(...)` | vLLM main FlashInfer wrapper 写路径（非 KV sharing 层） | 已确认 |
| C19 | supports_trtllm_attention：SM90 与 SM12x 只有 XQA decode kernel（无 TRTLLM prefill）；SM100/SM103 两个阶段都有 TRTLLM kernel；需要 NVIDIA artifactory 可达（下载 cubins）；batch-invariant 模式禁用。can_use_trtllm_attention 额外要求查询头数整除 KV 头数 | [vllm-main] vllm/utils/flashinfer.py L481-528 | "SM90 (Hopper) and SM12x support the XQA decode kernel but not TRTLLM prefill. SM100+ supports TRTLLM for both phases."；`return supports_trtllm_attention(is_prefill=is_prefill) and (num_qo_heads % num_kv_heads == 0)` | vLLM main | 已确认 |
| C20 | 同一块物理内存可用 as_strided 构造 NHD 与 HND 两个视图：NHD 视图 stride 为 $(H \cdot d, d, 1)$、HND 视图 stride 为 $(N \cdot d, d, 1)$；两个视图逻辑数据等价、物理排布不同，切换不需要复制 | 本地验证（PyTorch 2.13.0，2026-09-03 运行，脚本与输出存本目录 verify_strided.py / verify_strided.out） | 运行输出：NHD 视图 token0 的所有头为物理前 6 个连续元素；HND 视图 head0 为前 12 个连续元素；stride 分别为 (6,3,1) 与 (12,3,1)（参数 N=4,H=2,d=3） | torch.as_strided 语义 | 已确认 |
| C21 | RFC #42082 之前（v0.13.0 为例），vLLM 的 KV cache 是每层独立的 buffer，形状 (num_blocks, 2, block_size, num_kv_heads, head_size)（"2" 为 K/V）；布局切换由 get_kv_cache_stride_order 给出置换、kv_cache.permute(*stride_order) 应用：NHD 为 (0,1,2,3,4)，HND 为 (0,1,3,2,4)（交换 block_size 与 num_kv_heads 两维） | [vllm-v0.13.0] vllm/v1/attention/backends/flashinfer.py L308-334、L1198-1199（tag v0.13.0 拉取于 2026-09-03） | `return (num_blocks, 2, block_size, num_kv_heads, head_size)`；`elif cache_layout == "HND": stride_order = (0, 1, 3, 2, 4)`；`kv_cache_permute = kv_cache.permute(*stride_order)`；注释 "`stride_order` indicates the permutation that gets us from `get_kv_cache_shape` to the actual memory layout we want." | vLLM v0.13.0（旧形状体系） | 已确认 |

## F 公式

| 编号 | 公式 | 来源 | 说明 |
|---|---|---|---|
| F1 | NHD 页形状 $(n_{\text{page}}, p, H_{\mathrm{kv}}, d)$；HND 页形状 $(n_{\text{page}}, H_{\mathrm{kv}}, p, d)$ | [FreeKV] §4.2（同 C5） | 分页 KV cache 的完整形状，$p$ 为页大小 |
| F2 | NHD 视图 stride（页内三维）：$(H_{\mathrm{kv}} \cdot d, d, 1)$；HND 视图 stride：$(p \cdot d, d, 1)$ | 行主序张量的通用步长规则 + [vllm-main] compute_layout_strides（从最内维累乘）+ 本地验证 C20 | stride 的通用定义，与具体框架无关 |
| F3 | NHD 单头取一页的传输单元 $= d$ 个元素 $= d \cdot b$ 字节；HND $= p \cdot d$ 个元素 $= p \cdot d \cdot b$ 字节（$b$ 为每元素字节数） | [FreeKV] §4.2 的传输单元结论（C6）代数化 | $b$：fp16 为 2、fp8 为 1 |

## N 数字

| 编号 | 数字 | 来源 | 实验条件 |
|---|---|---|---|
| N1 | $d=128$、fp16（$b=2$ 字节）时，NHD 单头取一页的最大传输单元为 256 字节 | [FreeKV] §4.2 | "equivalent to just 256 bytes for d = 128 and Float16 precision" |
| N2 | $p=32$ 时，HND 单头取一页的传输单元为 8 KB | [FreeKV] §4.2 | "or 8KB when p = 32"（承接 $p \times d$ 元素结论） |
| N3 | SM100 上 FlashInfer 支持的 kernel block size 扩展到 128/256/512/1024（其他平台 16/32/64），大页仅在 SM100 + GQA/MQA（num_qo_heads//num_kv_heads > 1）+ trtllm 可用时提供 | [vllm-main] flashinfer.py L421-441 get_supported_kernel_block_sizes | Blackwell trtllm-gen 动态 kernel |
| N4 | 本地验证参数：N=4、H=2、d=3，物理内存 24 个元素；NHD stride (6,3,1)、HND stride (12,3,1) | 本地运行（C20） | torch 2.13.0，CPU |

## 冲突与不足记录

- "HND 为什么对低精度 kernel 友好"的深层硬件原因：FlashInfer 文档只给出结论（C3），trtllm-gen 文档给出 TMA 16 字节盒宽约束（C9）作为部分机制。页面正文把"低精度下每元素字节变少，NHD 的跨步间隔（$H \cdot d \cdot b$ 字节）随之变小、更难凑满宽向量/TMA 盒"这一解释明确标注为推断，不作为来源结论。
- vllm main 是滚动分支：C10-C19 均标注拉取日期，审查时重新拉取核对；若与页面描述不一致以最新源码为准。
