# 滑动窗口注意力审查记录（第 2 轮）

- 页面版本：`4aabe679f1325dad8966409fc46b94afaf47b25c`（`git hash-object wiki/sliding-window-attention/index.html`）
- 审查时间：2026-09-10 16:40
- 审查者：独立子代理（未参与写作与第 1 轮审查）
- 已完整阅读章节：h1 与元信息 → reading-time → blockquote.meta → callout（本页的「窗口」指单层可见距离上限）→ 核心问题（4 条含解答）→ 常见误解 → 1. 窗口把可见集切成「最近 $W$ 个位置」→ 1.1 因果掩码管方向，窗口管距离 → 本章问题（第 1 章）→ 2. 省下的是打分次数与缓存字节 → 2.1 每层打分次数从 $N^2$ 降到 $N\cdot W$ → 2.2 环形缓冲让缓存不随长度增长 → 2.3 窗口 KV 为什么保持 FP8 → 本章问题（第 2 章）→ 3. 层堆叠接力：感受野 $k\times W$ 与它的天花板 → 3.1 信息每层前移 $W$ → 3.2 5120 对 1M → 3.3 只用窗口会崩：注意力汇聚点 → 本章问题（第 3 章）→ 4. 环形缓冲：槽位、覆盖与位置编码 → 4.1 槽位 = 位置对窗口取模（含代码折叠块）→ 4.2 prefill 逐行、decode 整环 → 4.3 位置编码用原序列位置 → 本章问题（第 4 章）→ 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件、构造示例、辅助解释与类比边界、简化条件及其限制）

## 机械验证结果

1. `validate.py`：

```
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/sliding-window-attention/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/sliding-window-attention/index.html
EXIT=0
```

2. 第 4.1 节可运行代码：将 `<pre><code class="language-python">` 内容原样复制到 `/tmp/swa_code.py` 执行，与页面「预期输出」逐字符比对，结果 `EXACT MATCH: True`。实际输出：

```
W = 8  序列长度 = 20  全程一致
最后一步 p = 19
  槽位序列   : [4, 5, 6, 7, 0, 1, 2, 3]
  映射回位置 : [12, 13, 14, 15, 16, 17, 18, 19]
  期望因果窗 : [12, 13, 14, 15, 16, 17, 18, 19]
  位置 % W   : [4, 5, 6, 7, 0, 1, 2, 3]
```

3. 独立复算页面数字（`/usr/bin/python3` 手算，不使用页面脚本）：`128×512×1 B = 65536 B = 64 KiB`；`40×64 KiB = 2560 KiB = 2.50 MiB`；`3×64 KiB = 192 KiB = 0.1875 MiB`，合计 `2.6875 MiB`（页面写「约 2.69 MiB」，一致）；`1 048 576×512×1 B = 512 MiB/层`，`×40 = 20 GiB`；`65536 / 536870912 = 1/8192`；`40×128 = 5120`，`5120/1048576 = 0.488% ≈ 0.49%`；打分次数对照表三行 `4096²=16 777 216 / 4096×128=524 288 / 32×`、`131072²=17 179 869 184 / 131072×128=16 777 216 / 1024×`、`1048576²=1 099 511 627 776 / 1048576×128=134 217 728 / 8192×`，全部与页面表格逐位一致。
4. 复算脚本 `research/verify_swa_accounts.py` 的存档输出 `research/verify_swa_accounts.out` 与页面数字一致（含 `2.50 MiB`、`1/8192`、`5120`、`0.49%`、三行 `32x/1024x/8192x`）。
5. Unicode 数学字符检查：剥离 `$...$`/`$$...$$` 后，正文、标题、summary、表格、callout 中裸 Unicode 数学字符计数为 0（validate.py 同样通过）。
6. 概念页链接：`overview.html` 列出的 5 个前置概念（standard-attention、causal-mask、kv-cache、rope、attention-sink）目录均存在且含 `index.html`；`index.html` 与 `overview.html` 相互链接存在。
7. `dojo:topics = "注意力机制,内存与缓存"` 均在 `catalog_builder.ALLOWED_TOPICS` 词表内。

## 问题

- [重要·技术] `index.html` 全文（除第 3.3 节的「注意力汇聚点」外）：正文大量依赖前置概念「因果掩码」「KV cache」「RoPE」「标准注意力/全注意力」，但页面正文中没有任何一处指向 `../causal-mask/index.html`、`../kv-cache/index.html`、`../rope/index.html`、`../standard-attention/index.html` 的链接；`第 1.1 节` 整节在讲因果掩码、`第 4.3 节` 整节在讲 RoPE，均只出现名词不给出跳转。｜引文依据：不适用（链接检查）｜修复要求：在第 1 章首次出现「因果掩码」「全注意力」、第 2 章首次出现「KV cache」、第 4.3 节首次出现「RoPE」处，各补一个指向对应概念页的 `<a href="...">` 链接（目标页均已存在，不需要占位）；或补一处前置概念列表。｜修复：在 blue callout 内并入四条前置概念链接（standard-attention / causal-mask / kv-cache / rope），并在第 1 章「全注意力」「因果掩码」、第 2 章「KV cache」、第 4.3 节「RoPE」首次出现处各补一个链接；callout 仍为 1 个｜复验：已复跑 validate.py 通过并核对修改位置｜

- [重要·技术] `index.html:1097`（第 4.3 节）：正文以代码格式写 decode 取 `freqs_cis[start_pos]`，但官方参考实现中不存在该表达式；`model.py:767` 始终用切片 `freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]`（`grep -n "freqs_cis\[start_pos" model.py` 的结果为 `767: freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]`、`1039: main_freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]`，无独立下标形态）。代码形态被写成反引号内的字面代码，读者按此检索会找不到。｜引文依据：`official/inference/model.py:767 freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]` ｜修复要求：把 `decode 取 <code>freqs_cis[start_pos]</code>` 改为与源码一致的切片形态，例如「decode 时 `seqlen=1`，同一表达式取到 `freqs_cis[start_pos : start_pos+1]`」；来源说明 [C5] 已用正确切片形态，正文与之统一即可。｜修复：第 4.3 节改为「实现统一取切片 freqs_cis[start_pos : start_pos + seqlen]，prefill 时 seqlen 为本次长度、decode 时 seqlen = 1」，正文不再出现 freqs_cis[start_pos] 独立下标形态｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·技术] `index.html:931`（第 2.3 节）：把窗口 KV 称作最近位置的「原始精度」副本。来源只说明窗口 KV 保留 FP8、全局 KV 用 FP4，并未称窗口 KV 为原始精度（原始精度应为 BF16）。｜引文依据：技术报告 §2.4.4「We retain FP8 for the SWA KV cache due to its sensitivity to quantization. Compared with the FP8 main KV cache in DeepSeek-V4, this format nearly halves the storage footprint」——窗口 KV 是与 DeepSeek-V4 主 KV 同为 FP8 的量化副本，而非原始精度。｜修复要求：将「原始精度」改为「未经 FP4 压缩的较高精度」或「保留 FP8 的精度副本」。｜修复：2.3 节「原始精度」改为「保留 FP8 的较高精度副本（比全局 KV 的 FP4 少一道压缩）」｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·可读性] `index.html:807-837`（第 1 章结构图）与 `index.html:803`：结构图只画 6 个位置（位置 4–9 高亮）并标注「示意 6 个位置」，但同一章正文的构造数字使用 128（位置 4999/5000 的窗口 4872–4999）。读者可能把图上的 6 误当成 $W$。｜引文依据：不适用｜修复要求：在图注或图题中补一句「本图用 6 个位置示意，取值与模型中实际的 $W=128$ 无关」，与文末「构造示例」的说明对应。｜修复：第 1 章图注末尾补「本图用 6 个位置示意，与实际模型中的 W=128 无关」，与文末构造示例说明对应｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·格式] `index.html:881`（第 2 章标题「2. 省下的是打分次数与缓存字节」）：第 2 章含 2.1 打分次数、2.2 缓存字节、2.3 窗口 KV 为何保持 FP8 三节，其中 2.3 讲的是精度取舍而非「省下的量」，与章标题的「省下的是……字节」不完全对应。｜引文依据：不适用｜修复要求：把章标题的覆盖范围写全（如「2. 省下的是打分次数与缓存字节——以及精度取舍」），或把 2.3 的结论句并入章首过渡，使章标题与三节内容一致。｜修复：第 2 章标题改为「2. 省下的是打分次数与缓存字节——以及精度取舍」，与 2.1/2.2/2.3 三节内容一致｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·可读性] `index.html:757-789`（核心问题）、`index.html:791-798`（常见误解）：`query`、`prefill`、`decode`、`KV cache`、`RoPE` 等术语在页面级「核心问题」的解答中首次出现时未作解释，且页面前部没有前置概念列表（前置概念只在 `overview.html` 中列出）。读者若直接进入 `index.html`，需读完整页后回看才能理解首屏术语。｜引文依据：不适用｜修复要求：与第 1 条合并处理——在页面前部补一处前置概念清单（可复用 `overview.html` 第 67–74 行的五条），或把第 1 条要求的概念链接集中放在该处。｜修复：前置概念清单已并入开篇 blue callout（与第 1 条合并修复），直接进入 index.html 的读者在首屏即可看到五个概念页的跳转（含 attention-sink）｜复验：已复跑 validate.py 通过并核对修改位置｜

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复（2 条重要问题必须关闭；4 条轻微问题建议一并处理，其中 1 条可与第 1 条合并修复）
- 已核对的来源论断与引文依据：
  - [C1] / [F1] / [N1]：Mistral 7B arXiv:2310.06825v1 §2「Sliding Window Attention」原文「the hidden state in position i of the layer k, h_i, attends to all hidden states from the previous layer with positions between i−W and i. Recursively, h_i can access tokens from the input layer at a distance of up to W×k tokens」；Figure 1 图注「each token can attend to at most W tokens from the previous layer (here, W=3)」；正文「At each attention layer, information can move forward by W tokens. Hence, after k attention layers, information can move forward by up to k×W tokens.」「using a window size of W=4096, we have a theoretical attention span of approximately 131K tokens」；Table 1「dim 4096 … n_layers 32 … window_size 4096」。页面把 $[i-W+1,i]$ 与论文 $[i-W,i]$ 的差异明确标注，未扩大范围。
  - [C2] / [F2] / [N2]：同篇 §2「Rolling Buffer Cache」原文「The cache has a fixed size of W, and the keys and values for the timestep i are stored in position i mod W of the cache. As a result, when the position i is larger than W, past values in the cache are overwritten, and the size of the cache stops increasing.」「On a sequence length of 32k tokens, this reduces the cache memory usage by 8x」（32 768/4096=8）。页面只引用倍数关系，未引用其 2x 加速比数字。
  - [C3] / [N3] / [N4]：StreamingLLM arXiv:2309.17453v1 §1「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV」；§3.1「Window attention (0+y) has a drastic increase in perplexity.」（属 Table 2 标题句）与「removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function (Equation 1) in attention computation.」；Table 1 标题「Perplexities are measured on the first book (65K tokens) in the PG19 test set」，行值 `0+1024 → 5158.07`、`4+1020 → 5.40`、`4"\n"+1020 → 5.60`；Table 3 行值 `Learnable Sink 1+1023 → 18.01`、`Vanilla 2+1022 → 18.05`。表号（Table 1/2/3）与页面 [N3]/[N4] 的标注一致。
  - [C4] / [N5] / [N6]：`official/inference/config.json`：`window_size=128`、`head_dim=512`、`n_layers=40`、`n_mtp_layers=3`；`official/inference/model.py:700-720` `_window_kv` 内 `act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)` 与注释「The K stays fp8, quantized over the whole post-RoPE vector」；`model.py:777` `kv = torch.cat([kv, compress_kv], dim=1)`；`kernel.py:41-42` `def act_quant_kernel(..., inplace=False): """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16."""`（支持页面在「简化条件」中登记的 2 B/元素与部署口径 1 B/元素之别）；技术报告 §2.4.4「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」；报告 §4.2.1「For the additional branch of sliding window attention, the window size nwin is set to 128.」；报告 §2.1「Each layer incorporates both global attention and sliding window attention (SWA), except for the first two layers, which use SWA only.」与「For sliding window attention (SWA), CED maintains the conventional layer-wise computation across all layers.」（支持「40 层各有一份窗口缓存」「除最前两层外另挂全局 KV」）；报告 §2.4.3「The drafter comprises three Transformer blocks with a sliding attention window of 128 tokens.」（支持「3 个 DSpark 草稿层各注册一份同尺寸缓存」）；`model.py:1100-1101` `class DSparkBlock(Block): """DSpark stage stored under the mtp.* checkpoint namespace."""`、`model.py:1210-1211` `for layer_id in range(args.n_mtp_layers): self.mtp.append(DSparkBlock(args.n_layers + layer_id, args))`；`ckpt/headers.json` 中 `mtp.0/1/2.attn.wkv.weight` 形状均为 `[512, 5120]`、`layers.*.attn.kv_norm.weight` 形状 `[512]`（支持 512 维单 KV 头与草稿层同尺寸）。报告 §4.2.1「We set the number of Transformer layers to 40 … with 20 layers in the encoder and 20 layers in the decoder.」
  - [C5]：`model.py:767` `freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]`（窗口 KV 的 RoPE 用原序列位置）；StreamingLLM §3.2「When determining the relative distance and adding positional information to tokens, StreamingLLM focuses on positions within the cache rather than those in the original text. This distinction is crucial for StreamingLLM's performance.」及示例「if the current cache has tokens [0, 1, 2, 3, 6, 7, 8] and is in the process of decoding the 9th token, the positions assigned are [0, 1, 2, 3, 4, 5, 6, 7], rather than the positions in the original text」。
  - [N7]：`research/verify_swa_accounts.out`「decode 槽位映射回的位置集合 == 期望因果窗口: True (不符 0 处)」（$W=128$、序列长 300）；`ckpt/verify_sparse_attn_window.out`「[4] 窗口环形缓冲 (window_size=8), 序列长度 20 / decode 槽位映射回的位置集合与 prefill 因果窗口一致 (所有 20 步): True」（$W=8$、序列长 20 的官方函数对照）。页面 [N7] 对两种规模分别标注，未把自写复算与官方函数对照混为一谈。
  - [F3]：$n_{\text{score}}(N)=N\times W$ 的算术复算见「机械验证结果」第 3 项，三行均逐位一致；页面在「简化条件」中已声明全注意力按非因果 $N^2$ 计、且「少 8192 倍」仅指打分次数而非端到端延迟。
