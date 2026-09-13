<!-- review-meta
round: 1
page: wiki/deepseek-v4-1/index.html
reviewed_content_sha256: b7e599bb2a229803
-->
# DeepSeek-V4.1-Flash 审查记录（第 1 轮）

- 页面版本：`0796f2784a735cee015aebd9b7dc17d3c4335d61`（`git hash-object wiki/deepseek-v4-1/index.html`；overview.html 为 `9c190d4000863f0ae04b1fa61f3db393e74d52e0`）
- 审查时间：2026-09-10 19:40 CST
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节：核心问题、常见误解、1. 890 字节的账——压缩、共享与 FP4 各贡献多少（1.1 压缩条目、1.2 一条条目 288 字节、1.3 只有四个层生产主 KV、本章问题）、2. prefill 为什么只跑一半——CED 把解码器的全局 KV 前移（2.1 主干被切成编码器与解码器两半、2.2 解码器的全局 KV 由 $H_{L/2}$ 投影而来、2.3 滑动窗口 KV 不能一起省、本章问题）、3. 一个 query 看到哪些位置——窗口、压缩条目与两级筛选（3.1 可见集由两部分拼成、3.2 可达条目数、3.3 两级筛选、3.4 三模式、3.5 选中的槽位怎么参与计算、3.6 可达性与 Top-K 语义的复算、本章问题）、4. 8B 与 16B——激活参数的账怎么算（4.1 单层 377M 的构成、4.2 半栈加总、本章问题）、5. Engram 与 DSpark——两个不改缓存账的模块（5.1 Engram、5.2 DSpark、本章问题）、来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制）；并完整阅读 overview.html 全文。所有 `<details>` 折叠块内容均逐字读过。

## 机械验证结果

1. **哈希与 validate.py**

```
$ git hash-object wiki/deepseek-v4-1/index.html
0796f2784a735cee015aebd9b7dc17d3c4335d61
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/deepseek-v4-1/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/deepseek-v4-1/index.html
exit=0
```

2. **3.6 节代码实跑**：从 `<pre><code class="language-python">` 提取（HTML 实体反转义）后写入临时文件运行，与页面「预期输出」逐行 `diff`，**内容完全一致，唯一差异是页面预期输出块末尾无换行、实际 stdout 末尾有换行**（HTML `<pre>` 提取产生的伪差）。退出码 0，无 stderr。

3. **引用编号双向闭合**（脚本核对 `<sup>[Cx/Fx/Nx]</sup>` 与来源章节 `<p>[Cx/Fx/Nx]` 定义集合）：
   - 正文引用集合：C1–C11、F1–F7、F9、N1–N9；定义集合相同。
   - 正文引用未定义：`[]`；定义未被引用：`[]`。
   - 相邻双上标 `</sup><sup>`：无。

4. **Unicode 数学字符 / TAB / 占位符**：正文（剔除 `<head>`、`script`、`style`、`pre`、`code` 与 `$...$`）残留 `×(U+00D7)` 4 处、`−(U+2212)` 3 处、`↑(U+2191)` 1 处（返回顶部按钮，UI 元素）。TAB 字符 0 个；无「待生成」/TODO/TBD/【】占位符。

5. **headless Chrome 渲染**：

```
$ "/Applications/Google Chrome.app/..." --headless=new --disable-gpu --virtual-time-budget=8000 \
    --dump-dom file:///Users/wendadawen/code/dojo/wiki/deepseek-v4-1/index.html
class="katex* 节点数：718    katex-error：0    <math> 元素：237    DOM 大小：429363 字节
```

   页面公式全部由 KaTeX 渲染，无渲染错误（stderr 的 `CVDisplayLinkCreateWithCGDisplay failed` 为 macOS headless 噪音，与页面无关）。

6. **独立复算与存档核对**（不引用页面自述，自行计算后与 `research/ckpt/*.out` 比对）：

| 页面数字 | 独立复算 | 存档/来源 | 结论 |
|---|---|---|---|
| 288 B/条 | $512/2+512/16=256+32$ | `verify_cache_size.out` 行 3 | 一致 |
| 68 B/条（索引器 K） | $128/2+128/32=64+4$（粒度 32） | `verify_cache_size.out` 行 4「每 32 通道 1 字节 scale = 68 B」 | 数值一致，**页面写法用的是 /16** |
| 144、288、720、170、890 | $288/2=144$；$3\times144+288=720$；$3\times34+68=170$；$720+170=890$ | `verify_cache_size.out` 行 12 | 一致 |
| 8352、11.6 倍 | $18\times144+20\times288=8352$；$8352/720=11.6$ | `verify_cache_size.out` 行 17–18 | 一致 |
| 377.31M / 385.40M | $126.62+1.97+35.39+212.34+0.98+0.010=377.31$；层 20 替换注意力为 134.71 → 385.40 | `verify_active_params.out` 行 3–4 | 一致 |
| 7.8929B / 7.5758B / 317.10M / 314.65M | $20\times377.31+32.13+314.65=7892.99$M；$7546.2+8.09+21.60=7575.89$M；差 317.09M | `verify_active_params.out` 行 6–7、`verify_halfstack_diff.out` 行 54–57 | 一致（尾差四舍五入） |
| 128 / 512 / 2048 / 16384 | `window_size`；`index_topk`；`candidate_topk_blocks`；$2048\times8$ | `config.json` L26/L36/L38–39 | 一致 |
| 9.5% / 2.65% / 3.58× | $9.51/2.65=3.585$ | `verify_fp4_error.out` 行 5/9/10 | 一致 |
| 268× | $448\times6/10=268.8$ | `verify_fp4_error.out` 行 12–14 | 一致 |
| 1.960 | $15.4687/7.8929=1.9598$ | `verify_active_params.out` 行 13 | 一致 |
| 40% | $288/720=40\%$ | — | 一致 |

7. **公式逐符号核对**（对照 `official/inference/model.py`、`kernel.py` 与报告 Eq. 编号）：

| 页内编号 | 页面公式 | 来源位置 | 核对结果 |
|---|---|---|---|
| F1 | $C_j=\mathrm{RMSNorm}(\sum_{t=jr}^{(j+1)r-1}\mathrm{softmax}_{t'}(s_{t'})v_t)$，位置 $j\cdot r$ | `model.py` L464–485；L538–543 注释「a latent stands for the first token of its group, so group j takes position j * ratio」 | 一致：`kv=(kv*score.softmax(dim=2)).sum(dim=2)` → `self.norm(...)`；`freqs_cis[:seqlen-seqlen%ratio:ratio]`。$r=1$ 分支 `return self.norm(self.wkv(x))`（L461–462）与页面「退化」表述一致 |
| F2 | $\sum_h w_{q,h}\mathrm{ReLU}(q_h\cdot k_j)$ | `model.py` L555–557 | 一致：`index_score=(index_score.relu_()*weights.unsqueeze(-1)).sum(dim=2)`，`weights=self.weights_proj(x)*(scale*n_heads**-0.5)` |
| F3 | $n_{\text{reach}}(i)=\lfloor(i+1)/r\rfloor$（prefill）、$\lfloor(s+1)/r\rfloor$（decode） | `model.py` L563–567、L744 | 一致：`arange(1,seqlen+1)//ratio`；decode `end_pos//ratio` |
| F4 | 块分 $\max_{j\in b}I_{q,j}$、钉住最新块、取前 2048 | `model.py` L596–610 | 一致：`unflatten(-1,(-1,block_size)).amax(-1)`、`last=(compress_lens-1)//block_size` 置 `+inf`、`topk(min(2048,num_blocks))` |
| F5 | $\sqrt{\mathrm{softplus}(xW^\top)}$、$\mathrm{topk}_6(\text{score}+b)$、$w=\text{score}_{\text{idx}}/(\sum\text{score}_{\text{idx}}+10^{-20})\times1.5$ | `model.py` L811–826 | 一致：L817 `F.softplus(scores).sqrt()`；L822 `(scores+bias).topk(6)`；L823–826。页面略去 `self.gate_temp` 除法，但 `gate_temp` 默认 1.0（L67）且 `config.json` 未覆盖，不影响 |
| F6 | $\text{pre}=\sigma(ms_0+b)+\epsilon$、$\text{post}=2\sigma(ms_1+b)$、$\text{comb}=\text{Sinkhorn}_{20}(\mathrm{softmax}(ms_2+b)+\epsilon)$ | `kernel.py` L426–460 | 一致：L427/429/430–443；行/列各归一化 20 次、末步为列（`verify_sinkhorn.out` 行 7 复现） |
| F7 | $C_l=H_{L/2}W_l^{KV}$、$Z_l=H_{L/2}W_l^{Z}$（$l>L/2$） | 报告 Eq.(1)，`tech_report.txt` L394–395 | 一致。页面 [F7] 标注「报告行 388–393」（引导句），Eq. 本行在 394–395 |
| F9 | $o=\frac{\sum_{t\in\mathcal T}e^{q\cdot k_t/\sqrt d-m}v_t}{\sum_{t\in\mathcal T}e^{q\cdot k_t/\sqrt d-m}+e^{\text{sink}-m}}$，$m=\max_{t\in\mathcal T}q\cdot k_t/\sqrt d$ | `kernel.py` L310–403 | 一致：`scale=(1.0/d)**0.5`；`reduce_max` 只覆盖选中槽位；`sum_exp[i]+=exp(attn_sink[i]-scores_max[i])`；`-1` 槽位置 $-\infty$、取值 0。`verify_sparse_attn_window.out` 与解析解差 0.00e+00 |

   F 编号缺 [F8]（正文与来源两侧都没有），见问题清单。

8. **前置概念链接**：index.html / overview.html 全部 `../*/index.html` 链接目标在 `wiki/` 下均存在（21 个全部 `OK`）；`overview.html` 与 `index.html` 互相链接；`../../libs/katex.min.css`、`prism-*.css`、`katex.min.js`、`auto-render.min.js`、`prism*.js` 均存在。

9. **`<head>` 元信息**：`description`（纯文本）、`dojo:summary`（含 `$...$`）、`dojo:type=concept`、`dojo:topics=推理系统,内存与缓存`、`dojo:tag` 均存在，validate.py 未拒绝主题词。

## 问题

- [重要·技术] 1.3 节（行 885、行 880 表、行 767 与行 918 核心/章节解答）：索引器 K 的缩放因子粒度写错，且给出假算式「$128/16=4$」（$128/16=8$）。实际上索引器 K 是每 32 通道一个 E8M0 缩放因子，$128/32=4$，才能得到 68 字节。｜引文依据：`official/inference/model.py` L759 注释「Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.」；L546/L552 `fp4_act_quant(k, fp4_block_size, ...)` 与 L28 `fp4_block_size = 32`；存档 `ckpt/verify_cache_size.out` 行 4「indexer K : 128 维 fp4 + 每 32 通道 1 字节 scale = 68 B」。｜修复要求：把两处 `128/16` 改为 `128/32`，并写明索引器缩放因子为每 32 通道一个（E8M0），与 1.2 节主 KV 的每 16 通道 E4M3 分开陈述；改后 `128/2+128/32=64+4=68` 可复算成立。｜修复：两处 `128/16` 均改为 `128/32`；1.3 节正文补「每 32 通道一个 E8M0，粒度与类型都与主 KV 的每 16 通道 E4M3 不同」（依 `model.py` L759 注释），表格行改为「索引器 K（128 维 FP4 + 逐 32 通道缩放因子）」；`128/2+128/32=64+4=68` 现已可复算｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [重要·技术] 3.3 节（行 1069 与行 1076）：同一段内自相矛盾且与来源不符。正文先写「可达条目最多上千条」，随即写搜索范围「从「全部可达条目」缩到 16384 个候选，缩小了约两个数量级」——16384 大于「上千条」，两个数不能同时成立；来源说的是索引器「仍然对全部因果可见位置打分」、候选池把「随上下文线性增长」变成「与上下文长度无关的常数」，在 1M 上下文、$r=2$ 下可达条目为 500000，缩减约 30 倍（约 1.5 个数量级），且短上下文下 16384 候选并不构成任何缩减。｜引文依据：报告 `tech_report.txt` L550–551「the remaining indexers still score the full causally visible context. For extremely long contexts, this cost remains a major computational bottleneck.」；L576–579「For a fixed candidate-pool size, the number of positions scored per query by each subsequent indexer is bounded independently of context length. The first Full Mode layer still scans the entire causally visible range.」；本页 3.6 节实跑输出「i=4999 r=1: 可达 5000 条」。｜修复要求：删去「最多上千条」，改为「可达条目随上下文线性增长（1M 上下文、$r=2$ 时约 50 万条）」，把缩减倍数限定为长上下文条件、改述为「从随上下文线性增长变为固定 16,384」，并注明候选池在短上下文下不起缩减作用。｜修复：删去「最多上千条」，改「可达条目随上下文线性增长：1M 上下文、压缩比 2 时约 50 万条」；缩减改述为「从随上下文线性增长的条目数变为固定 16384 个候选位置，1M 时约 30 倍，短上下文下可达条目少于 16384 故不起缩减作用」；3.3 节本章问题的 summary 与解答同步改述｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [轻微·格式] 页面顶部（行 746 与行 750）：两个完全相同且相邻的 `<h1 class="title">DeepSeek-V4.1-Flash：890 字节的全局 KV 缓存是怎么算出来的</h1>`，渲染后标题出现两次。｜引文依据：不适用｜修复要求：删除行 750 的重复 h1，保留行 746 一处（其后紧跟 `reading-time` 段落）。｜修复：删除 _content_1.html 中重复的 h1（模板已提供该元素），页面 h1 计数为 1｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [轻微·可读性] 3.2 节（行 1043）首次使用「穿越示例」，但全页未定义该词，3.6 节（行 1220）与「简化条件及其限制」（行 1468）继续引用。｜引文依据：不适用｜修复要求：在行 1043 首次出现处写明它指「位置 $i=4999$ 的遍历算例」，或改用自描述名称。｜修复：3.2 节首次出现处补定义：「用本页贯穿的算例核算。该算例追踪位置 $i=4999$ 的 query 在层 2 与层 20 上的开销与可见范围（下称穿越示例）」｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [轻微·技术] 4.2 节差额表（行 1308）行标签方向与符号矛盾：行标签写「解码器多出的注意力参数」，数值却为「+2.43M」，而该表计算的是「编码器半栈 − 解码器半栈」；实际是编码器的注意力参数比解码器多 2.43M（编码器 2564.50M、解码器 2562.07M）。｜引文依据：`ckpt/verify_halfstack_diff.out` 行 49–51「编码器 20 层 attn 合计: 2564.50M／解码器 20 层 attn 合计: 2562.07M／解码器 - 编码器 attn 差: -2.43M」。｜修复要求：把行标签改为「编码器多出的注意力参数」或「注意力项差异（编码器较多）」，符号保持 +2.43M。｜修复：行标签改为「编码器多出的注意力参数」，符号保持 +2.43M（依 `verify_halfstack_diff.out`：编码器 attn 2564.50M、解码器 2562.07M）｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [轻微·格式] 正文残留未渲染的 Unicode 数学运算符（style-guide §11）：行 1310「编码器半栈 − 解码器半栈」「7.8929B − 7.5758B」中的 `−`（U+2212）与行 1407「(4−1)×8」中的 `−` 未包进 `$...$`；同页行 1357 与行 1450 的同类运算已写成 `$(4-1)\times8=24$`、`$2048\times8=16384$`，写法不一致。｜引文依据：不适用｜修复要求：这三处 `−` 改为 LaTeX（如 `$7.8929\text{B}-7.5758\text{B}$`、`$(4-1)\times8$`），与页内既有写法统一；`×`（U+00D7）在散文中按 `.dojo/scripts/validate.py` 的约定可保留。｜修复：三处 U+2212 改为 LaTeX：表内改为「编码器半栈与解码器半栈之差」与 $7.8929\text{B}-7.5758\text{B}$，「(4−1)×8」改为 $(4-1)\times8$；页面 U+2212 计数为 0｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [轻微·格式] 来源与范围说明（行 1436–1444）：公式编号从 [F7] 直接跳到 [F9]，缺 [F8]，正文与来源两侧都没有该项，易被误读为漏项。｜引文依据：不适用｜修复要求：将 [F9] 顺延为 [F8] 并同步正文 3.5 节与来源章节的引用，或保留编号但在来源章节说明 F8 已并入他项。｜修复：[F9] 重编号为 [F8]，正文 3.5 节与来源章节同步；`[F9]` 计数为 0｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [轻微·格式] 报告行号引用偏差，按标注位置定位不到引文：[C3]（行 1426）引「行 698–700」的 SWA FP8 句，实际在 `tech_report.txt` L703–704；同条引「行 691–694」的 E2M1/每 16 通道句，实际在 L689–691。[C6]（行 1429）引「同报告行 537–540」的 Full Mode 句，实际在 L544–546。｜引文依据：`tech_report.txt` L703–704「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」；L689–691「we select E2M1 with one E4M3 scale per 16 channels, following NVFP4 ... but omitting its second-level global scale」；L544–546「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer, i.e. the last layer of the causal encoder.」｜修复要求：把 [C3] 的两处行号更正为 689–691 与 703–704，[C6] 的第二处更正为 544–546。｜修复：[C3] 行号改为 689–691 与 703–704，[C6] 第二处改为 544–546；新增脚本 `verify_line_refs.py` 对全部 21 条行号标注做跨行核对，存档 `ckpt/verify_line_refs.out`｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [轻微·技术] 三处设计动机/效果陈述被写成事实，但来源只描述做法、未给理由，页面也未标注为推断或辅助解释：1.2 节补充（行 864）「整条用一个缩放因子时，某个特别大的通道会迫使其他通道全部落到低量级上」；3.3 节图注（行 1100）「块分用块内最大值而不是平均值，是为了不让一个高分位置被同块的低分位置稀释」；4.1 节补充（行 1296）「它让每层读写残差流的路径不止一条，从而分摊了单一路径上的干扰」。｜引文依据：报告 L689–691 只说选该格式是「to balance accuracy and simplicity」、L566–567 只说「each block is assigned the maximum index score among its positions」、L586–588 只说「maintains n residual streams between adjacent Transformer blocks」，均无上述因果说明。｜修复要求：在来源章节「辅助解释与类比边界」登记这三条为辅助解释（或补来源、或删除）；不改动其数值结论。｜修复：三处动机陈述已登记为辅助解释：在「辅助解释与类比边界」新增一段，逐条说明它们是解释而非来源结论（缩放粒度动机、块分取最大值的理由、超连接的信息流作用），并各挂对应编号｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [轻微·技术] 5.1 节（行 1357）：门控描述漏掉关键一步。实现是「归一化点积取绝对值开方、保留符号后再过 sigmoid」，页面写成「归一化点积再过 sigmoid」。｜引文依据：`model.py` L362「gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))」；L361 注释「signed sqrt before the sigmoid, matching the training kernel」。｜修复要求：把该句改为「归一化点积取绝对值开方、保留原符号后过 sigmoid」，或明确指出此处为简化描述。｜修复：改为「门控值取当前残差流与查表结果的归一化点积，先取绝对值、开平方、再保留原符号，最后过 sigmoid（实现里的「带符号平方根」处理，与训练内核一致）」并补 [C4]（依 `model.py` L361–362）｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

- [轻微·可读性] 5.2 节（行 1395）：「按维度取均值后拼接」未指明在哪一维取均值，读者无法复算；实现是在超连接副本维（`hc_mult`）取均值，再把三层的 $[b,s,d]$ 沿最后一维拼接。｜引文依据：`model.py` L1265–1266「if i in self.target_layer_ids: main_hiddens.append(h.mean(dim=2))」，`h` 形状为 $[b,s,hc,d]$。｜修复要求：写明「按超连接副本维取均值（`mean(dim=2)`），再把层 37、38、39 的结果沿特征维拼接」。｜修复：改为「先在超连接副本维上取均值（该维大小为 4），再把三层结果沿特征维拼接」并补 [C5]（依 `model.py` L1265–1266）｜复验：已重跑 validate.py 通过（validation ok）、headless Chrome 实测 katex=243 且无占位符与标签重叠、引用编号 28 项双向闭合、TAB 与 U+2212 计数均为 0｜

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 9
- 处置：修复

（说明：两条重要问题均不影响 890 字节这一核心结论的最终数值，但分别破坏了该结论的一项推导和两级筛选的规模表述，必须修复；9 条轻微问题不影响主线理解。本轮未发现需要改变内容范围或大纲的问题，故不返回规划。）
