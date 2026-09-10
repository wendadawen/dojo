# DeepSeek-V4.1-Flash 审查记录（第 3 轮）

- 页面版本：`index.html` 工作树哈希 `b34f57bbcb4e08a20d538ac808715397711288fe`（1688 行）；`overview.html` 工作树哈希 `9c190d4000863f0ae04b1fa61f3db393e74d52e0`（86 行）
- 审查时间：2026-09-10 20:09 CST
- 审查者：独立审查者（第 3 轮，未参与写作，未参与前两轮审查与修复；本轮仅读取页面、`official/`、`ckpt/`、`research/*.py`、`research/*.out` 与三份规范）
- 已完整阅读章节：核心问题（5 条，含 5 个解答折叠块）→ 常见误解 → 1. 890 字节的账（1.1 / 1.2 / 1.3 / 本章问题，含「补充：压缩发生在 token 方向」「补充：为什么缩放因子按 16 通道」「展开：890 的完整加总与「不共享会是多少」的反事实」）→ 2. prefill 为什么只跑一半（2.1 / 2.2 / 2.3 / 本章问题，含「补充：报告的设计描述与参考实现的分工」）→ 3. 一个 query 看到哪些位置（3.1 / 3.2 / 3.3 / 3.4 / 3.5 / 3.6 / 本章问题，含「展开：可达条目数的枚举核对」「补充：参考实现在组未填满时的一个已实测行为」「代码：可达条目数与 Top-K 选择的语义复算」）→ 4. 8B 与 16B（4.1 / 4.2 / 本章问题，含「补充：超连接模块为什么只占 0.98M」「展开：两个半栈的完整加总」）→ 5. Engram 与 DSpark（5.1 / 5.2 / 本章问题，含「补充：为什么图像 token 被屏蔽」）→ 来源与范围说明（论断与来源（C）/ 公式与来源（F）/ 外部数字与实验条件（N）/ 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）。

---

## 机械验证结果

### 1. validate.py

```
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/deepseek-v4-1/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/deepseek-v4-1/index.html
EXIT=0
```

validate.py 同时覆盖：模板占位符 `【】` 与 `@content/@component/TODO/TBD`（0 处）、重复 id（0 处）、同页锚点缺失（0 处）、断链的本地引用（0 处）、wiki 页五项元数据、`dojo:topics` 词表校验、`description` 纯文本、`dojo:summary` 定界符配对、KaTeX 本地资源与 auto-render 初始化、公式定界符外的数学 Unicode 字符、SVG `<text>` 内公式、`<pre>` 框线图。全部通过。

### 2. 脚本核对项

| 检查项 | 结果 |
|---|---|
| 引用编号双向闭合 | `<sup>[…]</sup>` 共 39 处；定义 C1–C11、F1–F8、N1–N9 共 28 条。**定义但未被引用：0 条；引用但无定义：0 条**；组合引用 2 处（`[C6, F7]`、`[C8, F5]`），格式合法 |
| 相邻双上标 `</sup><sup>` | 0 处 |
| Unicode 数学字符（U+2212 减号） | 全文 0 处 |
| Unicode 数学字符（U+00D7 乘号） | 3 处，全部位于 `<code>compress_ratios = [0,0,2×18,1×20,0×3]</code>` 内，属 style-guide §11 明示的 `<code>` 豁免范围；正文、标题、summary、列表、表格（公式定界符外）无 U+00D7 |
| 其他 Unicode 数学字符（公式定界符外、代码外） | 仅 `→`（U+2192）6 处，位于两个 HTML 结构图的 `.dg-arrow` 流程箭头，非数学符号（与 content-examples.md A5 正例同做法）；无希腊字母、上下标、关系符、根号、求和号 |
| TAB 字符 | 0 处 |
| 残留占位符 | 0 处（`占位` 出现 3 次均为正文实义：`占位置`／`索引为 $-1$ 的占位槽位`／`噪声 token …占位`） |
| h1 数量 | 1 |
| `<head>` 五项元数据 | `description`（纯文本、无 `$`）、`dojo:summary`（仅行内 `$…$`，`$` 计数为偶）、`dojo:type=concept`、`dojo:topics=推理系统,内存与缓存`（AGENTS.md 固定大类内）、`dojo:tag=KV cache 压缩` —— 五项齐备 |
| overview ↔ index 互链 | `index.html` 导航含 `<a class="overview-link" href="overview.html">概览</a>`；`overview.html` 导航含 `<a href="index.html">完整说明 →</a>`，双向成立 |
| 前置概念链接有效 | index.html 24 个本地引用、overview.html 16 个本地引用，逐个 stat 全部存在（16 个前置概念页均有 `index.html` 与 `overview.html`） |
| 问题块 | 页面级「核心问题」5 题、章节级「本章问题」5 节共 11 题，**16 题全部带 `解答：` 折叠块**；`<ol class="chapter-questions">` + `<li>` 内「问题段落 → details」结构一致；5 条核心问题答案末尾分别写明「完整账目见第 1 章／完整推导见第 2 章／完整规则与可运行复算见第 3 章／完整加总见第 4 章／完整机制见第 5 章」 |
| summary 前缀 | 26 个 details：`解答：` 16、`补充：` 6、`展开：` 3、`代码：` 1，无越界前缀 |
| 章节编号 | h2 为 1–5 连续编号 + 不编号的「核心问题／常见误解／来源与范围说明」；h3 章内从 1 连续递增（1.1–1.3、2.1–2.3、3.1–3.6、4.1–4.2、5.1–5.2）；来源章节 h3 用六个固定命名；正文无 `S1/S2/S3` 章节代号 |

### 3. headless Chrome 渲染

```
$ "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --dump-dom \
    "file:///Users/wendadawen/code/dojo/wiki/deepseek-v4-1/index.html"
EXIT=0，dump 440,311 字节
```

- `.katex` 节点：**751**；`katex-error`：**0**；`ParseError` / `KaTeX parse error`：**0**
- 目录自动生成 33 项（h2 8 + h3 25），与正文标题数一致；阅读时间脚本输出「全文 11,686 字 · 约 38 分钟阅读」；代码复制按钮注入 8 个
- 两个结构图为 HTML（`.dg-stack`、`.dg-flow`），无 SVG `<text>`、无框线字符图；折线图内公式写在 `.dg-node-note` 的 `$…$` 中由 KaTeX 渲染

### 4. 可运行代码（§3.6）

把 `<details class="code-details">` 内的 Python 代码块按 HTML 实体反转义后原样落盘实跑：

```
$ /usr/bin/python3 /tmp/block_0.py
EXIT=0
```

与页面 `<p><b>预期输出</b>：</p>` 下的 `language-text` 块**逐字符一致**（`difflib.unified_diff` 为空）：

```
== 1. 可达条目数：公式 vs 枚举 ==
  r in (1,2,4), i in 0..199 全部比对：不符 0 处
  i=4999 r=2: 可达 2500 条（覆盖 token 0..4999）
  i=4999 r=1: 可达 5000 条（覆盖 token 0..4999）
  i=4 r=2: 可达 2 条（覆盖 token 0..3）
== 2. Top-K 选择：不可达条目置 -inf ==
  槽位总数 12，可达 5 条，K=4
  未屏蔽时前 4 名: [0, 2, 6, 7]  <- 含未完成组的槽位
  屏蔽后前 4 名  : [0, 2, 3, 4]
  屏蔽后选中项全在可达范围内: True
```

代码自带的「验证的机制／观察重点／简化条件」三段说明与输出一致，未把示例表述为生产实现。

### 5. 独立复算关键数字

| 数字 | 页面位置 | 独立复算过程 | 存档比对 |
|---|---|---|---|
| 288 | §1.2、§1.3 | $512/2 + 512/16 = 256 + 32 = 288$ | `verify_cache_size.out`：「主 KV 条目 : 512 维 fp4 + 每 16 通道 1 字节 scale = 288 B」 |
| 68 | §1.3 | $128/2 + 128/32 = 64 + 4 = 68$ | 同上：「indexer K : 128 维 fp4 + 每 32 通道 1 字节 scale = 68 B」 |
| 144 | §1.3 | $288 / 2 = 144$ | `verify_cache_size.out`：「层 2- 7 ( 6 层, ratio=2): 主 KV 144.0 B/token + indexer K 34.0 B/token」 |
| 720 / 170 / 890 | §1.3 | $144×3+288=720$；$34×3+68=170$；$720+170=890$ | 同上：「合计: 主 KV 720 + indexer K 170 = 890 B/token」 |
| 8352 | §1.3 展开、常见误解、本章问题 | $18×144 + 20×288 = 2592 + 5760 = 8352$ | 同上：「若无跨层共享 (层 2-39 各自存主 KV): 8352 B/token」；$8352/720=11.6$ |
| 377.31M | §4.1 | $126.62+1.97+35.39+212.34+0.98+0.010 = 377.31$ | `verify_active_params.out`：「层 0: attn 126.62M gate 1.97M 共享专家 35.39M top-6 路由 212.34M … hc 0.98M norm 0.010M」 |
| 385.40M | §4.1 | $134.71+1.97+35.39+212.34+0.98+0.010 = 385.40$ | 同上「层 20: attn 134.71M …」 |
| 7.8929B | §4.2、核心问题 | 用真实张量头重算 encoder 半栈 | `verify_active_params.out`：「编码器半栈 (层 0-19) 每 token 激活: 7.8929B」 |
| 7.5758B | §4.2、核心问题 | 用真实张量头重算 decoder 半栈 | 同上：「解码器半栈 (层 20-39) 每 token 激活: 7.5758B」 |
| 317.10M | §4.2 表、核心问题 | $7.8929 - 7.5758 = 0.3171$ | `verify_halfstack_diff.out`：「编码器半栈 - 解码器半栈 = +317.10M」 |
| 314.65M | §4.2 表、展开、§5.1 | $2×157.327360 = 314.654720$ | `verify_halfstack_diff.out`：「编码器 (层 0-19) engram_dense 合计: 314.65M」（层 1、14 各 157.33M） |
| 128 / 512 / 2048 / 16384 | §3.1、§3.3 | config `window_size=128`、`index_topk=512`、`candidate_topk_blocks=2048`、`candidate_block_size=8`，$2048×8=16384$ | `verify_csa2_modes.out`：「candidate_topk_blocks=2048 × block_size=8 = 16384 个候选位置 (报告: 16384)」；`count_params.out` 形状 `indexer.wk.weight: [128, 512]` |

### 6. 来源逐条核对（check.md §2.2 四步）

**论断与来源（C）**

| 条目 | 页面标注位置 | 实际读到的原文片段／关键数值 | 判定 |
|---|---|---|---|
| C1 | 报告行 314–319 | 行 314–317：「Its language backbone comprises 40 causal Transformer layers, organized into a 20-layer causal encoder followed by a 20-layer decoder. Each layer incorporates both global attention and sliding window attention (SWA), except for the first two layers, which use SWA only.」 | 一致（引文落在 314–317，标注区间完整覆盖） |
| C2 | 报告 §2.3.1 行 490–520；`model.py` L496–503、L653–661、L722–737 | 行 497–498：「Full Mode. The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices.」；`model.py` L498–500 注释全文与页面引文逐字一致；L654–661 `self.is_kv_source` / `self.is_index_source` 与 compressor/indexer 创建；L725–737 `_compress_topk_idxs` | 一致 |
| C3 | 报告 §2.4.4 行 689–691、行 703–704；`model.py` L760、L707、L546/L552 | 行 689–691：「we select E2M1 with one E4M3 scale per 16 channels, following NVFP4 …, but omitting its second-level global scale」；行 703–704：「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」；L759 注释「Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.」＋L760 `fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`；L707 窗口 KV `act_quant(kv, fp8_block_size, …)`；L546/L552 索引器 K/Q 的 `fp4_act_quant` | 一致 |
| C4 | config 四字段；`model.py` L328–365、L344；`engram.py`；`probe_engram_layout.out` | `engram_layer_ids=[1,14]`、`engram_max_ngram_size=4`、`engram_n_heads=8`、`engram_head_dim=256`；L344 `n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads`；L361–362 注释「signed sqrt before the sigmoid, matching the training kernel」＋`torch.copysign(dot.abs()…sqrt(), dot)`；实测输出「layers: (1, 14)」 | 一致 |
| C5 | config 七字段；`model.py` L1100–1156、L1131–1132、L1265–1266；`headers.json` mtp 张量数 | config 七字段取值与页面相同；L1131–1132 与页面引文逐字一致；L1264–1266 注释「the MTP head reads the attention input of its target layers, not their output」＋`main_hiddens.append(h.mean(dim=2))`；实测 `mtp.*` 张量 **2401** 个 | 一致 |
| C6 | 报告 §2.2 行 388–393、行 544–546 | 行 389–392：「For global attention, CED treats the bottom L/2 layers …, the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H_{L/2}, using layer-dependent projection weights」；行 544–546：「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer, i.e. the last layer of the causal encoder.」 | 一致 |
| C7 | 报告 §2.3.2 行 542–570；config 三字段；`model.py` L583–610、L604–605 | 行 569–570：「For example, selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions.」；行 566–567「each block is assigned the maximum index score among its positions」；config 三字段取值一致；L583–610 `select_candidate_blocks`（L599 `amax`、L604–605 钉住最新块） | 一致 |
| C8 | config 五字段；`model.py` L809–827、L821–823、L817 | config 五字段一致；L817 `scores = F.softplus(scores).sqrt()`；L821–823 注释「the bias picks experts but does not scale them: weights come from the raw scores」与页面引文一致；L824–826 `weights /= weights.sum(…) + 1e-20`、`weights *= self.route_scale` | 一致 |
| C9 | config 两字段；`kernel.py` L406–462；`model.py` L948–966；`verify_sinkhorn.py` | `hc_mult=4`、`hc_sinkhorn_iters=20`；kernel L427/L429/L436–460 三套系数与迭代；`verify_sinkhorn.out`：「comb: 最大差 8.941e-08」「行归一化 20 次 (含 1 次 softmax), 列归一化 20 次, 末步 = col」 | 一致 |
| C10 | `model.py` L537–548、L554；config `kv_source_layer_ids`；`debug_consistency7.out` | L537 `if self.owns_k and latent is not None:`（发布在分支内）、L554 `index_k = shared_attn.index_k[:bsz, : end_pos // ratio]`（无条件执行）；config `kv_source_layer_ids=[2,8,14,20]`；实测 step 8/10/12/14 读到自己的 cache? **False** 且首 4 值与 layer4 的 k_cache 相同，step 9/11/13/15 为 **True** 且等于 layer2 自身 k_cache | 一致 |
| C11 | `model.py` L1261–1267；`inference/` 目录检索；报告行 404–410 | L1261 `for i, layer in enumerate(self.layers):` 及 1262–1267 无跳过条件；对 `inference/` 全目录检索 `replay|bounded|prefill the last|half|skip`，命中项仅 `engram.py` 的 `skip_special_tokens` 与 int64 溢出注释，无重放代码；报告行 404–410 描述 SWA replay 与 Decoder SWA Bounded Replay | 一致 |

**公式与来源（F）**

| 条目 | 页面标注位置 | 实际读到的原文片段／关键数值 | 判定 |
|---|---|---|---|
| F1 | `model.py` L458–485、L538–543、L752–757；`verify_compressor_pooling.py` | L464–485 池化：`kv = (kv * score.softmax(dim=2)).sum(dim=2)`、`self.norm(kv.to(dtype))`，L461–462 `ratio == 1` 时 `return self.norm(self.wkv(x))`（无门控）；L538 与 L752 注释「a latent stands for the first token of its group, so group j takes position j * ratio」；`verify_rope_yarn.out` 亦标注「代码依据: Indexer.forward L539-543 / Attention._compress_kv L753-757」；`verify_compressor_pooling.out`：「独立重实现 … 最大差: 0.000e+00」 | 一致 |
| F2 | `model.py` L550–557；`ckpt/debug_consistency4.out` | L556–557 `index_score = torch.einsum("bshd,btd->bsht", q, index_k)`、`(index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)`；`debug_consistency4.out` 逐 query 列出分数原值 | 一致 |
| F3 | `model.py` L563–567；`verify_reach_topk.py` | L564–565 prefill `torch.arange(1, seqlen + 1) // ratio` + `masked_fill_(… >= compress_lens, -torch.inf)`；L567 decode `compress_lens = end_pos // ratio`；脚本输出「r in (1,2,4), i in 0..199 全部比对：不符 0 处」 | 一致 |
| F4 | `model.py` L596–610 | L598–599 `F.pad(…, value=-torch.inf)` + `unflatten(-1, (-1, block_size)).amax(dim=-1)`（块分取最大）；L604–605 钉住最新块并置 `torch.inf`；L607 `scores.topk(min(topk_blocks, num_blocks)…)` | 一致 |
| F5 | `model.py` L811–826；`verify_gate_formula.py` | L811–817 打分分支；L822–826 top-k、归一化、`*= self.route_scale`；`verify_gate_formula.out`：「官方 vs 独立重实现 indices 完全一致: True」「权重最大差: 0.000e+00」「权重行和 = route_scale: [1.5, …]」 | 一致 |
| F6 | `kernel.py` L426–460；`verify_sinkhorn.py` | L427 `pre[i,j] = T.sigmoid(…hc_scale[0]…)+eps`；L429 `post[i,j] = 2*T.sigmoid(…)`（无 eps）；L436–443 softmax+eps；L445–458 行列交替；`verify_sinkhorn.out`：「行归一化 20 次 … 列归一化 20 次, 末步 = col」 | 一致 |
| F7 | 报告行 388–396（页面注明 389–392 为文字、394–396 为 Eq. 1） | 行 394–396 排版为 `C_l = H_{L/2}W_l^{KV}, Z_l = H_{L/2}W_l^{Z}, l > L/2, (1)`；行 397 说明 `C`/`Z` 含义 | 一致（标注细化到「哪几行是公式」这一步，定位精度高于区间本身） |
| F8 | `kernel.py` L310–403；`verify_sparse_attn_window.py` | L355 `T.fill(scores_max, -1e30)`、L364 无效槽位置 `-T.infinity`、L383 `sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`、L385 `acc_o[i,j] /= sum_exp[i]`；`verify_sparse_attn_window.out`：sink 三种取值「最大差 0.00e+00」、「两槽 vs 两槽+一个 -1 槽: 最大差 0.00e+00」、「整行 -1 的输出: [0.0, 0.0, 0.0, 0.0]」 | 一致 |

**外部数字与实验条件（N）**

| 条目 | 页面标注位置 | 实际读到的原文片段／关键数值 | 判定 |
|---|---|---|---|
| N1 | 报告行 18–19、行 133–140；`verify_cache_size.py` | 行 18–19：「reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash」；行 133–140：「552B backbone parameters … activate 8B parameters per token during prefill and 16B during decode … only approximately 1/4 as much runtime KV cache storage and 1/8 as much persistent KV cache storage」；脚本输出「合计: 主 KV 720 + indexer K 170 = 890 B/token … 一致: True」 | 一致 |
| N2 | 报告行 320–322；`verify_active_params.py`、`verify_halfstack_diff.py` | 行 320–321：「has 552B backbone parameters and 196B Engram parameters, activating 8B parameters per token during prefill and 16B during decode」；实测输出 7.8929B / 7.5758B / embed 0.6619B / head 0.6619B / 比值 1.960 / 「最接近的读数: prefill 7.9B / decode 15.5B」 | 一致 |
| N3 | 报告行 320；`count_params.py`；`headers.json` | 行 320 552B/196B；`count_params.out`：「backbone … 551.5662B」「engram: 196,928,504,320 = 196.9285B」；`headers.json` 键数 **96085**；`engram.embed.weight` 形状 `[384006168, 256]`、`[384016682, 256]`，各约 98.31B / 98.31B | 一致 |
| N4 | config 与报告 §2.3.2；`headers.json` | config：`window_size=128`、`index_topk=512`、`candidate_topk_blocks=2048`、`candidate_block_size=8`、`index_n_heads=32`、`index_head_dim=128`、`num_key_value_heads=1`、`head_dim=512`；形状 `indexer.wk.weight: [128, 512]` | 一致 |
| N5 | 报告 §2.4.4 行 691–700；`verify_fp4_error.py` | 行 691–698：「the format supports magnitudes up to 448 × 6 = 2688」「the maximum magnitude observed during training is around 10」「the L2 norm of the 512-channel KV latent is at most approximately √512」；实测输出「整体相对误差 … 0.0951」「FP8 … 0.0265」「比值 = 3.58x」「行 L2 范数 = 22.6274」「余量 … 268.8x (报告观察值)」 | 一致 |
| N6 | `verify_active_params.py` | 见上表 377.31M / 385.40M 行，六项分明细与合计均可复算 | 一致 |
| N7 | `debug_consistency3~7.out` | `debug_consistency3.out`：**layer 2: 4 个位置有差, 首个 pos 10, max 8.850e-03 (pos 14)**；`debug_consistency7.out` 修正后 bf16：layer 2 max **5.005e-03**；修正 + fp32：layer 0–6 max **1.490e-08 – 2.980e-08**；`run_mini.out` 注明「该差异已在 debug_consistency3~7 中定位到根因」 | 数值一致；括注「偶数位置」见问题 3 |
| N8 | `verify_compressor_pooling.py` | 「fp32 层面池化结果最大差 … 4.530e-06 (相对 2.18e-06)」「池化结果 cast 到 bf16 后不一致元素: 5/4096 (1 个 bf16 ulp ≈ 8.122e-03)」 | 一致 |
| N9 | 报告行 35–38、行 139–140 | 行 36–38：「achieves approximately 4-fold and 437-fold reductions in per-token global KV cache size relative to DeepSeek-V4-Flash and DeepSeek-V1, respectively」；行 139–140：「1/4 … runtime KV cache storage and 1/8 … persistent KV cache storage」；页面已标「登记为报告宣称」，`verify_cache_size.out` 同样只登记 | 一致，且已按 §2.2 第 4 步降级标注 |

**构造示例／辅助解释／简化条件**

- 「构造示例」小节把 5 处加总与枚举示例逐一登记，并区分「第 3.6 节的代码输出为实际运行结果」——与实测一致。
- 「辅助解释与类比边界」共登记 4 处（两级筛选、"按 16 通道分组的动机"、"块分取最大值"、"超连接分摊干扰"），每条都写明来源只描述做法、不给理由。第 5.2 节的访存瓶颈归因在正文即标为「以下为辅助解释」，与来源边界（来源只说明 DSpark 用于提高解码效率）一致。
- 「简化条件及其限制」5 条覆盖字节账、激活参数、缩小模型一致性实测、可见性核对范围、服务实现宣称，均写明成立条件与不能推出的结论。

---

## 问题

- [重要·格式] 来源与范围说明／发布条件：check.md §5「递归生成的前置概念页已完成各自质检」无法从仓库存档确认。页面 16 个前置概念全部有效可打开，但其中 2 个（`../rmsnorm/`、`../hyper-connections/`）的 `research/` 下**不存在任何 `review*.md`**，另有 7 个（`mla`、`dsa`、`deepseek-moe`、`speculative-decoding`、`mxfp4-qat`、`residual-connection`、`aux-loss-free-routing`）只记录 2 轮审查。｜引文依据：`ls wiki/rmsnorm/research` → `concept_probes.out, concept_probes.py, draft-check.md, evidence.md, glossary.md, outline.md, rms_page_code.out, rms_page_code.py, scope.md`（无 review 文件）；`wiki/hyper-connections/research` 同样无 review 文件；`wiki/mla/research` 只有 `review.md`＋`review-2.md`。｜修复要求：由编排者确认这 9 个页面是否已在别处完成三轮独立审查；若已完成，将记录补入各自 `research/review-*.md` 或在本轮结论中写明豁免依据，即可关闭；若未完成，须先补齐其审查，本页方可发布。｜修复：｜复验：
- [轻微·格式] §4.2「展开：两个半栈的完整加总」：两处字面算式与所标 4 位小数结果不能完全对上。（i）`$2\times157.33=314.65$M` 的乘积应为 314.66（正确值 314.65 来自未截断的 $2\times157.327360$，页面把「已四舍五入的乘数」与「按原值再取整的积」写在同一式子里）；（ii）`$20\times377.31+32.13+314.65 = 7546.2+32.13+314.65 = 7892.98$M $= 7.8929$B` 中 7892.98M 按 4 位小数是 7.8930B；解码器侧 `$7546.2+8.09+21.60 = 7575.89$M $= 7.5758$B 同理（7575.89M = 7.5759B）。尾差分别约 0.08M 与 0.09M（约 0.001%），来自「基准 377.31M 是六项 2 位小数之和，而 7.8929B/7.5758B 是张量头精确求和后再取 4 位」。同一段的 317.09M/317.10M 已经写明「尾差来自四舍五入」，这两处未写。｜引文依据：页面原文「合计约 20×377.31+32.13+314.65 = 7546.2+32.13+314.65 = 7892.98M = 7.8929B，与实测输出一致」；`verify_active_params.out`「编码器半栈 (层 0-19) 每 token 激活: 7.8929B」；按 `ckpt/headers.json` 精确重算 encoder=7892.88774M（= 7.8929B）、decoder=7575.79462M（= 7.5758B），而页面字面式给 7892.98M / 7575.89M。｜修复要求：在两行合计后各加一句与 317.09/317.10 同形的四舍五入说明，或把「= 7.8929B」改为「≈ 7.8929B」并将 `2×157.33=314.65` 改为 `2×157.33≈314.65`。｜修复：｜复验：
- [轻微·技术] 来源与范围说明 [N7]：括注「（从层 2 起、偶数位置）」不能由所引存档完全核对，且与存档的一处记录相左。｜引文依据：`debug_consistency3.out` [E0]「layer 2: 4 个位置有差, 首个 pos 10, max 8.850e-03 (pos 14)」支持「从层 2 起、首个差异在偶数位置 10」；但同一文件的 topk 选择对比列出「layer 2: 4 个位置选择不同 -> [(10, …), (11, …), (12, …), (14, …)]」，**位置 11 为奇数**。｜修复要求：把括注收窄为「首个差异出现在层 2 的位置 10」，或删除「偶数位置」，保留可核对的数值 8.9e-3。｜修复：｜复验：
- [轻微·格式] h1：`DeepSeek-V4.1-Flash：890 字节的全局 KV 缓存是怎么算出来的` 的冒号右侧是问句，不是 style-guide §1 要求的「核心作用或结论」；同站其余 14 个概念页右侧均为陈述式结论。｜引文依据：不适用（格式项）。｜修复要求：改为陈述式表述，如「DeepSeek-V4.1-Flash：把每 token 全局 KV 缓存压到 890 字节的三项叠加」。｜修复：｜复验：

---

## 结论

- 统计：阻断 **0** / 重要 **1** / 轻微 **3**
- 处置：**修复**（内容是完整的、可发布的；仅剩 1 条发布门控待编排者确认）

### 末轮复核专项（本轮重点 1）

**(a) 来源章节行号是否真的包含所引文字 —— 已关闭。** 逐条打开 `tech_report.txt` 与两份源码，把「标注区间内的原文」逐字抄回（见上表）：C1（314–319 ⊇ 314–317）、C2（490–520 ⊇ 497–498）、C3（689–691、703–704 均精确命中）、C6（388–393 ⊇ 389–392；544–546 精确）、C7（542–570 ⊇ 566–570）、C11（404–410 精确）、F7（明确区分「389–392 文字 / 394–396 公式」，并核对到 Eq.(1) 的实际排版行）、N1（18–19、133–140）、N2/N3（320–322）、N5（691–700）、N9（35–38、139–140）——全部包含所引内容；跨行情形（C6 的 $H_{L/2}$ 公式、N5 的 $448\times6=2688$ 与「around 10」）逐行确认无错位。源码行号同样逐条落到实际语句（L344、L461–485、L498–500、L537–548、L554、L563–567、L583–610、L654–661、L707、L725–737、L759–760、L811–826、L948–966、L1131–1132、L1261–1267；`kernel.py` L310–403、L406–462），未出现「编号存在但内容不符」。

**(b) §4.2「展开：两个半栈的完整加总」的层数与算式 —— 层数自洽，算式有 0.001% 的舍入尾差（问题 2）。** 层数逐项对上：编码器 20 层（0–19），其中 Full 层 2/8/14 三个；解码器 20 层（20–39），其中 Full 层 20 一个、Reindex 层 24/28/32/36 四个——与 `compress_ratios=[0,0,2×18,1×20,0×3]`、`verify_csa2_modes.out` 的层表、`verify_halfstack_diff.out` 的逐层 attn 表完全一致。每层增量也对得上：137.33−126.62=10.71、134.71−126.62=8.09、132.02−126.62=5.40。按字面复算：$7546.2+32.13+314.65=7892.98$M、$7546.2+8.09+21.60=7575.89$M、$7892.98-7575.89=317.09$M，三条内部链条各自成立；但把 7892.98M 写成 7.8929B、7575.89M 写成 7.5758B 时落到第 4 位小数不一致（精确值 7892.88774M / 7575.79462M），已在问题 2 记录并要求补一句四舍五入说明。

### check.md §5 逐项核对

| 发布条件 | 结果 |
|---|---|
| 三轮审查均已完成且均由独立审查者执行 | 本页为第 3 轮，由未参与写作与前两轮修复的审查者执行（满足）；前两轮记录在 `review-1.md`／`review-2.md`，本轮未读取其内容 |
| 每条来源论断都有引文依据记录 | 满足（C1–C11、F1–F8、N1–N9 共 28 条全部逐条抄录原文片段或关键数值） |
| 无法核对的论断已删除或降级 | 满足（N9 三个倍数、§2.2 的 prefill 减半与有界重放、DSpark 访存归因均标为「报告宣称／服务实现／辅助解释」，未作模糊保留） |
| 所有阻断和重要问题均已关闭 | 本页内容层面无阻断、无重要；**唯一 1 条重要项（前置概念页质检状态）属发布门控、待编排者确认** |
| 遗留轻微问题具有明确的接受理由 | 满足（3 条轻微：1 条建议补四舍五入说明、1 条建议收窄括注、1 条 h1 措辞——均不影响结论、公式、数字与主线理解，接受理由见各条） |
| 全部学习目标由正文完整回答 | 满足（5 条核心问题分别由第 1–5 章完整回答，答案末句指明章节） |
| 两级问题块均有解答折叠块 | 满足（核心 5 题 + 本章 11 题 = 16 题，16 个 `解答：` 折叠块，无只问不答） |
| 数学符号全部 LaTeX，结构图为 HTML/内联 SVG | 满足（公式定界符外无任何数学 Unicode 字符；两个结构图为 HTML 并含 `$…$` 标签；`<code>` 内的 `×` 属明示豁免） |
| `validate.py` 返回成功 | 满足（`validation ok`，EXIT=0） |
| 可运行代码结果与页面一致 | 满足（§3.6 代码实跑输出与「预期输出」逐字符一致） |
| 关键论断和数字已重新核对来源 | 满足（288/144/720/170/890/8352、377.31M/385.40M、7.8929B/7.5758B/317.10M/314.65M、128/512/2048/16384 全部独立复算并与 `ckpt/*.out` 比对） |
| `<head>` 五项元数据有效 | 满足（`description` 纯文本、`dojo:summary` 可渲染、`dojo:type=concept`、`dojo:topics` 在固定大类内、`dojo:tag`） |
| `overview.html` 与 `index.html` 相互链接 | 满足 |
| 引用概念链接有效或有占位 | 满足（index 24 个、overview 16 个本地引用全部存在，无占位） |
| 递归生成的前置概念页已完成各自质检 | **无法确认**（见问题 1） |

**发布结果：本页内容层面满足 §5 全部条件，可发布；§5 中「前置概念页质检」一条需编排者确认后，发布条件整体成立。**
