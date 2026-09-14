<!-- review-meta
round: 5
page: wiki/kv-cache-layout/index.html
reviewed_content_sha256: 62f15b9c4d7e0576
-->
# KV cache 布局（NHD/HND）审查记录（第 5 轮）

- 页面版本：ac3f75c98b1a7b0d2bc83820cdc63110ce9a5e10
- 审查时间：2026-09-14 17:00
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题；1. 一页 KV cache 的三个维度（1.1 每个元素存的是什么 / 1.2 两种排法：字母顺序就是物理顺序 / 1.3 分页形式与命名澄清 / 本章问题）；2. 写入路径：NHD 与投影输出一致（2.1 投影输出的形状 / 2.2 decode 每步的追加写 / 本章问题）；3. 读取路径：HND 把单头整页变连续（3.1 attention 按（头，页）读取 / 3.2 低精度 kernel / 3.3 页级搬运与混合布局 / 3.4 什么时候选哪种 / 本章问题）；4. vLLM 实现：逻辑形状固定，stride 置换切换布局（4.1 stride / 4.2 KVCacheLayout 枚举与逻辑形状 / 4.3 布局协商 / 4.4 SM100 / 4.5 写入 kernel 的适配与演进注记 / 本章问题）；来源与范围说明。全部折叠块与图注均逐段通读。

## 来源核对（本轮逐条回源，附原文片段）

- **[C1]** FlashInfer 官方文档 "KV-Cache Layout in FlashInfer"（docs.flashinfer.ai/tutorials/kv_layout.html）："NHD: the last 3 dimensions are organized as (seq_len, num_heads, head_dim)"、"HND: the last 3 dimensions are organized as (num_heads, seq_len, head_dim)"。与 §1.2、[C1] 定义一致。
- **[C2]** 同文档："The NHD layout is more natural because it's consistent with the output of xW_k and xW_v without transpose." 与 §2.1、2.1 节折叠块表述一致。
- **[C3]** 同文档："The HND layout is more friendly for GPU implementation when KV-Cache uses low-precision data type (e.g. fp8)." 与 §3.2 表述一致。
- **[C4]** 同文档："we don't observe significant performance difference between these two layouts on fp16 kV-Cache"、"we prioritize NHD layout for better readability." 与 §3.4、常见误解 callout 一致。
- **[C5/F1/C6/C7/N1/N2]** FreeKV，arXiv:2505.13109v2 §4.2 "Hybrid layouts and streamed recall"（https://arxiv.org/html/2505.13109v2）："the shapes of NHD and HND layouts are (n_page,p,n_kv,d) and (n_page,n_kv,p,d), respectively, where p is the page size"；NHD "the maximum transfer unit contains only d elements"、"256 bytes for d=128 and Float16 precision"；HND "allowing a transfer unit of p×d elements, or 8KB when p=32"；GPU 用 "the NHD layout on GPU to eliminate the need for per-step transposes during decoding"、CPU 用 "the HND layout on CPU to ensure contiguous and efficient CPU-GPU data transfers"、"the NHD-HND transpose is only required when offloading a KV page"。与 §1.3、§3.1、§3.3、N1/N2 逐项一致。
- **[C8/C9]** FlashInfer API 文档 flashinfer.prefill.trtllm_batch_context_with_kv_cache（docs.flashinfer.ai/generated/flashinfer.prefill.trtllm_batch_context_with_kv_cache.html）：kv_layout "default is 'HND'"；NVFP4 下 "using NHD will trigger an automatic transpose and .contiguous() copy of both the KV data and block scale tensors"、"This incurs extra memory allocation and data copy overhead"、"Use HND for better performance"；kv_cache_sf "must be contiguous … stride[-1] == 1 and stride[-2] == head_dim // 16"，理由 "the kernel reshapes them into (16, page_size * head_dim / 16 / 16) to satisfy TMA's 16-byte box width minimum"；kv_cache "head_dim (last dim) must have stride 1. This is a TMA hardware constraint"。与 §3.2 三段表述逐句一致。
- **[C10/C11/C12/C17]** vLLM main（2026-09-03 拉取）源码：vllm/v1/kv_cache_layout.py 的 KVCacheLayout 六成员 `LBHNC=(0,1,2,3,4)`、`LBNHC=(0,1,3,2,4)`、`LHBNC=(0,2,1,3,4)`、`BLHNC=(1,0,2,3,4)`、`BLNHC=(1,0,3,2,4)`、`BHLNC=(1,2,0,3,4)`，docstring "Each member's value is a stride permutation that maps logical axes to physical (memory) order."、逻辑形状 "always [L, B, H, N, C] (RFC #42082)"；vllm/config/cache.py `_LAYOUT_COMPAT_ALIASES = {"NHD": "LBNHC", "HND": "LBHNC"}`；vllm/v1/attention/backends/utils.py `_FLASHINFER_LAYOUT_NAMES = {"LBNHC":"NHD","LBHNC":"HND","BLHNC":"HND","BLNHC":"NHD","BHLNC":"HND"}`；vllm/v1/kv_cache_interface.py AttentionSpec：num_heads 默认 = num_kv_heads、state_content_size_bytes = "(head_size + head_size_v) * dtype_size"、tokens_per_state 默认 1（故 num_states = block_size = p）。与 §4.2、映射表、C12 逐项一致。
- **[C13/C14/C15]** vllm/envs.py `env_with_choices("VLLM_KV_CACHE_LAYOUT", None, ["LBNHC","LBHNC","LHBNC","NHD","HND","BLHNC","BLNHC","BHLNC"])`（默认 None=不设置）；vllm/v1/attention/backends/utils.py resolve_kv_cache_layout docstring "Runs once in the engine core… An explicit VLLM_KV_CACHE_LAYOUT must be one of the candidates or resolution fails… the result is recorded there (see CacheConfig.kv_cache_layout); it reaches workers through the set_kv_cache_layout RPC"；默认偏好序 `_DEFAULT_LAYOUT_PREFERENCE = (LBNHC, LBHNC, BLNHC, BLHNC, BHLNC, LHBNC)`，注释 "LBNHC (NHD) first to match main's default"。与 §4.3 逐步陈述、C13/C14/C15 一致。
- **[C16/C19/N3]** vllm/v1/attention/backends/flashinfer.py supported_kv_cache_layouts：`if capability.major == 10: return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)`，注释 "The trtllm-gen kernels consume head-major block interiors; the L/B nesting outside the block is immaterial to them."；vllm/utils/flashinfer.py supports_trtllm_attention docstring "SM90 (Hopper) and SM12x support the XQA decode kernel but not TRTLLM prefill. SM100+ supports TRTLLM for both phases."、can_use_trtllm_attention 要求 `num_qo_heads % num_kv_heads == 0`；get_supported_kernel_block_sizes：use_large_pages 时返回 `[16,32,64,128,256,512,1024]`，否则 `[16,32,64]`。与 §4.4 的 (LBHNC, BLHNC) 声明、硬件分档、整除条件、页大小陈述一致。
- **[C18]** 同 flashinfer.py 写路径：`# (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))`、`k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)`。与 §4.5 代码节选逐字一致。
- **[C21]** vLLM tag v0.13.0 `vllm/v1/attention/backends/flashinfer.py`：形状 `(num_blocks, 2, block_size, num_kv_heads, head_size)`、`get_kv_cache_stride_order` 中 NHD=`(0,1,2,3,4)`、HND=`(0,1,3,2,4)`、`kv_cache.permute(*stride_order)`。与 §4.5 演进注记一致。
- **[C20/N4] 可运行代码实测**：把 4.1 节代码块原样执行（本机 torch 2.8.0），输出与页面「预期输出」逐字一致——`物理顺序: [0.0 … 15.0]`、`NHD 下 token0 的全部头: [[0.0, 1.0], [2.0, 3.0]]`、`HND 下头0 的整页: [0.0 … 7.0]`、`NHD stride: (4, 2, 1)  HND stride: (8, 2, 1)`。N4 的 stride (4,2,1)/(8,2,1) 复算无误。
- **公式复算**：p·H_kv·d = 16（§1.1）；NHD stride (H_kv·d, d, 1)=(4,2,1)、HND (p·d, d, 1)=(8,2,1)（§4.1）；256 B=128×2 与 8192 B=32×128×2=8 KB，比值为 p=32（§3.1、§3.4）。全部可复算。

## 问题

- [轻微·格式] 4.2 节图 4（`.dg-flow` HTML 结构图，index.html 第 566-599 行）与图注（第 598 行）：图中的维度符号以纯文本写出——「L, B, H, N, C：层数、页数、头槽、页内状态、每格字节」、「物理顺序 L, B, N, H, C」、「物理顺序 L, B, H, N, C」、「N 与 H 两轴交换物理位置」，图注写「L、B、H 的外层顺序变化」——而同一页面正文与表格一律写作 `$L, B, H, N, C$`（第 546 行 `$[L, B, H, N, C]$`、第 559-560 行 `$L, B, N, H, C$`、`$L, B, H, N, C$`）。同一变量在正文（KaTeX 渲染）与图内（纯 ASCII）两种写法，违反 style-guide §11「同一变量在页面中保持同一种写法」。｜引文依据：`<div class="dg-node-note">N 与 H 两轴交换物理位置</div>`（第 574 行）对照正文 `逻辑形状恒为 $[L, B, H, N, C]$`（第 546 行）｜修复要求：将该结构图各节点文本与图注中的 L/B/H/N/C 等数学变量改为 `$L, B, H, N, C$` 等 `$...$` 写法（该图为 HTML div，KaTeX auto-render 会正常渲染）；同一处出现的置换值 `(0,1,3,2,4)` 与正文表格保持一致写法。｜修复：｜复验：

（说明：本节仅列一项，且为格式一致性问题，不涉及事实或来源。其余候选项经核对不成立：正文「下面代码用…」的用法与站内既有页面一致，"两个主角"为可接受的技术比喻，均不构成缺陷。）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布

核对要点回顾：全页 30 处来源/公式/数字引用（C1-C21、F1-F3、N1-N4）全部可定位到来源原文且支持页面表述；引文编号正文与来源说明双向覆盖无缺号；两幅 SVG 图的图注读数与图上刻度逐格核对一致（NHD 图 t2=槽 8-11、h0=槽 0-1/4-5/8-9/12-13；HND 图 h0=槽 0-7、t2=槽 4-5 与 12-13；访问段图 NHD 4 段各 2 元素、HND 1 段 8 元素）；4.1 节可运行代码实测输出与页面「预期输出」逐字一致；summary/description/正文之间的数字（256 B、8 KB、32 倍、p=32、d=128）互相一致；未发现元话语堆叠、会话指代、调试叙事或 AI 拼接腔；`.dojo/scripts/validate.py` 返回 validation ok；引用的 kv-cache、mqa-gqa、paged-attention 三个前置概念页均真实存在；overview.html 与 index.html 双向链接。