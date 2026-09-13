<!-- review-meta
round: 4
page: wiki/kv-cache-layout/index.html
reviewed_content_sha256: c1caf8a3ca15debb
-->
# KV cache 布局（NHD/HND）审查记录（第 4 轮）

- 页面版本：f883ef697c0ffa9c0f929cd4dcb4da624c6949f8
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 已完整阅读章节：核心问题；1. 一页 KV cache 的三个维度（1.1 每个元素存的是什么 / 1.2 两种排法 / 1.3 分页形式与命名澄清 / 本章问题）；2. 写入路径：NHD 与投影输出一致（2.1 / 2.2 / 本章问题）；3. 读取路径：HND 把单头整页变连续（3.1 / 3.2 / 3.3 / 3.4 / 本章问题）；4. vLLM 实现：逻辑形状固定，stride 置换切换布局（4.1 / 4.2 / 4.3 / 4.4 / 4.5 / 本章问题）；来源与范围说明。含全部 details 折叠块与两幅 SVG 图注。
- 机械验证：`python3 .dojo/scripts/validate.py wiki/kv-cache-layout/index.html` → `validation ok`（退出码 0）。`dojo:type=concept`、`dojo:topics`（内存与缓存, 推理系统）、`dojo:tag`（KV cache）均在 AGENTS.md / catalog_builder.py 词表内。

## 来源核对（逐条，含原文片段或关键数值）

- **C1** 定位 FlashInfer 官方 docs/flashinfer-ai/flashinfer commit `3d43dc9` 的 `docs/tutorials/kv_layout.rst`（`https://raw.githubusercontent.com/flashinfer-ai/flashinfer/3d43dc9/docs/tutorials/kv_layout.rst`，HTTP 200；线上 docs.flashinfer.ai/tutorials/kv_layout.html 同文）：`NHD: the last 3 dimensions are organized as (seq_len, num_heads, head_dim)`、`HND: (num_heads, seq_len, head_dim)`。页面第 131 行表述一致。**支持**。
- **C2** 同上：`The NHD layout is more natural because it's consistent with the output of xW_k and xW_v without transpose.` 页面第 290、311 行一致。**支持**。
- **C3** 同上：`The HND layout is more friendly for GPU implementation when KV-Cache uses low-precision data type (e.g. fp8).` 页面第 434 行一致。**支持**。
- **C4** 同上：`In practice we don't observe significant performance difference between these two layouts on fp16 kV-Cache and we prioritize NHD layout for better readability.`（另见同页 `(NHD by default)`）。页面第 451 行一致。**支持**。
- **C5 / F1** FreeKV arXiv:2505.13109v2 §4.2（arxiv.org/html/2505.13109v2）：分页形状 `(n_page, p, n_kv, d)`（NHD）与 `(n_page, n_kv, p, d)`（HND）。页面第 237–238、269、464、685 行一致。**支持**。
- **C6** FreeKV §4.2：`for a given KV head, the memory of p=3 key/value vectors within a page is non-contiguous. The maximum transfer unit contains only d elements`；`The HND layout ensures that p key/value vectors within a page are contiguous for each KV head, allowing a transfer unit of p×d elements`。页面第 329–334、336 行一致。**支持**。
- **C7** FreeKV §4.2：`employs the NHD layout on GPU to eliminate the need for per-step transposes during decoding, and the HND layout on CPU to ensure contiguous and efficient CPU-GPU data transfers during recall`；`the NHD-HND transpose is only required when offloading a KV page`。页面第 442 行一致。**支持**。
- **C8** FlashInfer API 文档 `flashinfer.prefill.trtllm_batch_context_with_kv_cache`（docs.flashinfer.ai/generated/…，HTTP 200）：`kv_layout (str = "HND") … default is "HND"`；`For the trtllm-gen backend with NVFP4 KV cache, using NHD will trigger an automatic transpose and .contiguous() copy of both the KV data and block scale tensors to convert them to HND layout. This incurs extra memory allocation and data copy overhead. Use HND for better performance.` 页面第 434 行一致（范围问题见问题 3）。**支持**。
- **C9** 同上 `kv_cache_sf` 段：`The last two dims (page_size, head_dim // 16) must be contiguous … the kernel reshapes them into (16, page_size * head_dim / 16 / 16) to satisfy TMA's 16-byte box width minimum`；KV 数据段：`The head_dim (last dim) must have stride 1. This is a TMA hardware constraint`。页面第 436 行一致。**支持**。
- **C10** vLLM main `vllm/v1/kv_cache_layout.py`：docstring `The logical shape is always [L, B, H, N, <content>] (RFC #42082). Each member's value is a stride permutation that maps logical axes to physical (memory) order.`；成员 `LBHNC=(0,1,2,3,4) # [L,B,H,N,C] (identity)`、`LBNHC=(0,1,3,2,4) # [L,B,N,H,C]`、`LHBNC=(0,2,1,3,4)`、`BLHNC=(1,0,2,3,4)`、`BLNHC=(1,0,3,2,4)`、`BHLNC=(1,2,0,3,4)`。页面第 546、556、559–560、598 行一致，且"物理第 k 位放置换中第 k 个编号的逻辑轴"的读法与 `# [L, B, N, H, C]` 注释自洽。**支持**。
- **C11** vLLM main `vllm/config/cache.py`：`_LAYOUT_COMPAT_ALIASES = {"NHD": "LBNHC", "HND": "LBHNC"}`；`_layout_from_name` 先查该别名表。页面第 563 行一致。**支持**。
- **C12** vLLM main `vllm/v1/kv_cache_interface.py`：`AttentionSpec.num_heads` 返回 `num_head_slots`（None 时）否则 `num_kv_heads`；`get_num_kernel_states` = block_size / tokens_per_state（`tokens_per_state` 默认 1）即 N = p；`state_content_size_bytes` = `(head_size + head_size_v) * get_dtype_size(dtype)` 即 C 为 K、V 两个 d 维向量拼接字节。页面第 551–553 行一致。**支持**。
- **C13** vLLM main `vllm/envs.py`：`env_with_choices("VLLM_KV_CACHE_LAYOUT", None, ["LBNHC","LBHNC","LHBNC","NHD","HND","BLHNC","BLNHC","BHLNC"])`，默认 None。页面第 620 行取值列表与"默认不设置"一致。**支持**。
- **C14** vLLM main `vllm/v1/attention/backends/utils.py` `resolve_kv_cache_layout`：docstring `Runs once in the engine core … An explicit VLLM_KV_CACHE_LAYOUT must be one of the candidates or resolution fails, with the legacy NHD/HND names as aliases for LBNHC/LBHNC`；`if (requested := envs.VLLM_KV_CACHE_LAYOUT) is not None: layout = _layout_from_name(requested); if layout not in candidates: raise ValueError(...)`；`cache_config.kv_cache_layout = layout.name`；`it reaches workers through the set_kv_cache_layout RPC`。页面第 615–624 行一致。**支持**。
- **C15** 同上：`_DEFAULT_LAYOUT_PREFERENCE = (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC, KVCacheLayout.BLNHC, KVCacheLayout.BLHNC, KVCacheLayout.BHLNC, KVCacheLayout.LHBNC)`，仅在全部后端返回 None 时取用（`or [_DEFAULT_LAYOUT_PREFERENCE]`）。页面第 624 行序列逐一对应。**支持**。
- **C16** vLLM main `vllm/v1/attention/backends/flashinfer.py`：`if capability is not None and capability.major == 10: # The trtllm-gen kernels consume head-major block interiors; the L/B nesting outside the block is immaterial to them. return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)`。页面第 630 行一致。**支持**。
- **C17** 同上 `_FLASHINFER_LAYOUT_NAMES = {"LBNHC":"NHD","LBHNC":"HND","BLHNC":"HND","BLNHC":"NHD","BHLNC":"HND"}`。页面第 611 行映射一致。**支持**。
- **C18** 同上写入路径（flashinfer.py 约 2595–2622 行）：注释 `# (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))`，代码 `k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)`，随后 `torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping, self.cache_dtype, layer._k_scale, layer._v_scale)`。页面第 638–641 行节选与注释为源码原文，一致。**支持**。
- **C19** vLLM main `vllm/utils/flashinfer.py` `supports_trtllm_attention`：`SM90 (Hopper) and SM12x support the XQA decode kernel but not TRTLLM prefill. SM100+ supports TRTLLM for both phases.`；`if not has_nvidia_artifactory(): return False`（注释 `Requires NVIDIA artifactory to be accessible to download cubins`）；`can_use_trtllm_attention` 要求 `num_qo_heads % num_kv_heads == 0`。页面第 628 行一致。**支持**。
- **C20 / N4** 本机复算：在 `/tmp/kvcheck/run.py` 原样执行 4.1 节代码（本机 torch 2.8.0），输出为 `物理顺序: [0.0 … 15.0]`、`NHD 下 token0 的全部头: [[0.0, 1.0], [2.0, 3.0]]`、`HND 下头0 的整页: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]`、`NHD stride: (4, 2, 1)  HND stride: (8, 2, 1)`，另验证 `nhd`/`hnd`/`flat` 三者 storage data_ptr 相同。与页面第 534–537 行"预期输出"逐行一致。页内声明的运行环境为 PyTorch 2.13.0，本机版本不同但输出相同；不作为问题。**支持**。
- **C21** vLLM tag `v0.13.0` `vllm/v1/attention/backends/flashinfer.py`：`get_kv_cache_shape` 返回 `(num_blocks, 2, block_size, num_kv_heads, head_size)`；`get_kv_cache_stride_order` 对 `cache_layout == "NHD"` 返回 `(0,1,2,3,4)`、`"HND"` 返回 `(0,1,3,2,4)`；使用处 `kv_cache_permute = kv_cache.permute(*stride_order)`。页面第 646 行一致（(0,1,3,2,4) 交换的正是 block_size 与 num_kv_heads 两维）。**支持**。
- **F2** 行主序三维 stride $(H_{\mathrm{kv}}d, d, 1)$ / $(pd, d, 1)$ 由 C20 实跑验证，且与 `compute_layout_strides` 的最内维累乘规则一致。**支持**。
- **F3** $d·b$ 对 $p·d·b$：$128×2=256$、$32×128×2=8192=8$ KB，与 N1/N2 一致。**支持**。
- **N1 / N2** FreeKV §4.2：`256 bytes for d=128 and Float16 precision`、`8KB when p=32`。页面第 426–427、488–489、704–705 行一致。**支持**。
- **N3** vLLM main `vllm/v1/attention/backends/flashinfer.py` `get_supported_kernel_block_sizes`：基础返回 `[16, 32, 64]`；`use_large_pages`（`is_device_capability_family(100)` 且 `num_qo_heads // num_kv_heads > 1` 且 `can_use_trtllm_attention(...)`）为真时返回 `[16, 32, 64, 128, 256, 512, 1024]`。页面第 630、706 行"扩展到 128 到 1024（其他平台为 16/32/64）"与之一致。**支持**。
- **页面链接**：`../../wiki/kv-cache/index.html`、`../../wiki/mqa-gqa/index.html`、`../../wiki/paged-attention/index.html`、`overview.html` 均存在；overview.html 反向链接 index.html；无"（待生成）"占位。libs 下 katex/auto-render/prism/dojo-concept.css 均存在。**通过**。
- **C20 的 `research/measured.md`**：`wiki/kv-cache-layout/research/measured.md` 存在（本轮未读取其内容），路径可解析。**通过**。
- **构造示例标记**：$p=4, H_{\mathrm{kv}}=2, d=2$ 的贯穿示例在第 133、712 行均注明"取小值便于手算，不代表工程配置"，4.1 节代码参数同源，未被写成来源事实。**通过**。

## 问题

- [轻微·格式] dojo:summary（第 7 行）、第 81 行、第 252 行：三个维度写作 $N/H/D$（大写 $D$），而正文第 84、91、98、131、133、237、238、280、290 行等同一量一律写 $d$，同一变量全页两种写法｜引文依据：不适用（页内一致性；第 7 行原文"KV cache 布局指 $N/H/D$ 三个维度（token 数、KV 头数、每头维度）…NHD 为 $(p, H_{\mathrm{kv}}, d)$"同句已混用 $D$ 与 $d$）｜修复要求：把三处 $N/H/D$ 改为 $N/H/d$，或将全页统一为同一写法（含 dojo:summary）｜修复：｜复验：
- [轻微·技术] 第 287 行（2.1 节符号表）：$N$ 定义为"本批 token 数"，同句括号却写"decode 每步为 1"；decode 批量执行时一个 batch 由各 running sequence 各 1 个 token 组成，$N$ 等于批内序列数而非 1，定义与括号互相矛盾｜引文依据：不适用（页内推算）｜修复要求：删去"decode 每步为 1"，或改为"prefill 为整段提示词；decode 时一批 $N$ 行对应本步各请求各一个 token"｜修复：｜复验：
- [轻微·技术] 第 452 行（3.4 节第一条）：条目主语为"低精度（fp8/nvfp4）走 trtllm-gen kernel"，随后接"NHD 会触发转置拷贝<sup>[C8]</sup>"；C8 来源只对 NVFP4 KV cache 断言自动 transpose 与 .contiguous() 拷贝，未对 fp8 断言，此处把 NVFP4 条件下的观察写成了 fp8/nvfp4 的论断｜引文依据：FlashInfer API 文档 `trtllm_batch_context_with_kv_cache`：「For the trtllm-gen backend with NVFP4 KV cache, using NHD will trigger an automatic transpose and .contiguous() copy of both the KV data and block scale tensors」｜修复要求：把转置拷贝限定到 nvfp4，例如"低精度（fp8/nvfp4）走 trtllm-gen kernel：HND 是默认且高效路径；其中 NVFP4 KV cache 下 NHD 会触发转置拷贝"｜修复：｜复验：
- [轻微·表述] 第 503 行（第 4 章开头）："本章按 stride 视图、枚举定义、布局协商、SM100、写入适配的顺序拆 vLLM 的答案"属"本章将…"类元话语（列本章写作提纲），不是上一节结论与本章问题的衔接语境｜引文依据：不适用｜修复要求：删去该路线图分句，直接以"引擎必须回答：布局切换会不会搬数据？多种注意力后端各有偏好时听谁的？"两问进入 4.1；如需保留篇幅信息，改写为陈述本章要回答这两个问题的句子，不罗列小节顺序｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布。全部来源论断均定位到原文片段并逐条核对通过（含 FlashInfer 官方文档与固定 commit 3d43dc9、FreeKV arXiv:2505.13109v2 §4.2、FlashInfer API 文档、vLLM main 分支各文件、vLLM v0.13.0 tag）；4.1 节代码本机实跑，输出与页面预期逐行一致；公式与贯穿示例全部可复算（16 = 4×2×2、256 B = 128×2、8 KB = 32×128×2、32× = 8192/256、槽位与 stride 逐项核对）；未发现数字与来源不符、推断被包装成来源结论、页内互相矛盾或算式与结论不符之处。上述 4 项轻微问题不影响核心结论与来源一致性，建议本轮一并修掉后发布。
