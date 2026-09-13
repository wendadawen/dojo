<!-- review-meta
round: 2
page: wiki/deepseek-v4-1/index.html
reviewed_content_sha256: b7e599bb2a229803
-->
# DeepSeek-V4.1-Flash 审查记录（第 2 轮）

- 页面版本：index.html `4eb379df45290f14968e5e89e00d547ef362d89b`；overview.html `9c190d4000863f0ae04b1fa61f3db393e74d52e0`
- 审查时间：2026-09-10
- 审查者：独立子代理（未参与写作与前序审查，仅使用两份页面、`official/` 与 `ckpt/` 来源、`research/*.py|*.out` 实测产物与三份质检规范）
- 已完整阅读章节（按顺序，含全部折叠块）：核心问题、常见误解、1 890 字节的账（1.1 压缩条目 / 1.2 一条条目 288 字节 / 1.3 只有四个层生产主 KV / 本章问题）、2 prefill 为什么只跑一半（2.1 编码器解码器切分 / 2.2 全局 KV 由 $H_{L/2}$ 投影 / 2.3 滑动窗口 KV 不能一起省 / 本章问题）、3 一个 query 看到哪些位置（3.1 可见集 / 3.2 可达条目数 / 3.3 两级筛选 / 3.4 三模式 / 3.5 选中槽位参与计算 / 3.6 复算代码与预期输出 / 本章问题）、4 8B 与 16B（4.1 单层 377M / 4.2 半栈加总 / 本章问题）、5 Engram 与 DSpark（5.1 / 5.2 / 本章问题）、来源与范围说明（论断与来源（C）/ 公式与来源（F）/ 外部数字与实验条件（N）/ 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）

## 机械验证结果

**通用校验**

| 项目 | 命令/方法 | 结果 |
|---|---|---|
| validate.py（index） | `/usr/bin/python3 .dojo/scripts/validate.py wiki/deepseek-v4-1/index.html` | `validation ok`，exit 0 |
| validate.py（overview） | 同上，传 overview.html | `validation ok`，exit 0 |
| h1 数量 | 正则统计 `<h1` | 1 |
| TAB 字符 | `src.count('\t')` | 0 |
| 引用编号双向闭合 | 正文 `<sup>[…]</sup>` 集合 vs 来源章节 `[Cx]/[Fx]/[Nx]` 定义集合 | 差集 1 = `[]`，差集 2 = `[]`（两侧均为 C1–C11、F1–F8、N1–N9 全集） |
| 相邻双上标 | 正则 `<sup>…</sup><sup>…</sup>` | 0 处 |
| 裸露 `[Cx]/[Fx]/[Nx]`（未包 sup） | 正文段正则 | 0 处 |
| 残留占位符 | 检索 `待生成/待补充/TODO/TBD/FIXME/XXX/占位符/placeholder/lorem/（略）` | 0 |
| 前置概念链接 | 逐一 `ls ../<name>/index.html`（kv-cache、rmsnorm、mla、mxfp4-qat、mixed-precision-quant、residual-connection、sliding-window-attention、rope、dsa、cross-layer-kv-sharing、attention-sink、hyper-connections、deepseek-moe、aux-loss-free-routing；overview.html 另加 ngram、speculative-decoding） | 全部存在 |
| overview ↔ index 互链 | overview.html L38 `href="index.html"`；index.html L727 `href="overview.html"` | 双向存在 |
| 问题块 | 核心问题 `<ol class="chapter-questions">` 1 个（5 问 5 解答）；5 个「本章问题」各 2/2/3/2/2 问，解答数与问题数一一相等 | 无只列问题未作答 |
| summary 前缀 | 26 个 `<summary>`：解答 16、补充 6、展开 3、代码 1 | 均在规范允许的三种前缀 + `解答：` 之内 |

**headless Chrome 渲染**

```
/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --headless=new --disable-gpu \
  --dump-dom file:///Users/wendadawen/code/dojo/wiki/deepseek-v4-1/index.html
```
- `.katex` 节点数 **736**；`katex-display` **8**（与源文件 `$$` 出现 16 次 = 8 组独立公式一致）
- `katex-error` 0；无公式残留 `$` 定界符（DOM 中 4 处 `$…$` 均落在 `<meta name="dojo:summary">` 与 auto-render 的 JS 配置里，非渲染失败）
- `readingTime` 已由脚本填充（DOM 中出现「全文 … 字」）

**3.6 可运行代码实跑**

提取 index.html L1154–1201 的 `language-python` 块（HTML 实体反转义）写到 `/tmp/_r2_code.py` 运行：

```
/usr/bin/python3 /tmp/_r2_code.py
```
- 与页面「预期输出」（L1205–1216）**逐字符一致**（`exp.strip() == got.strip()` → True）
- 与存档 `research/ckpt/verify_reach_topk.out` **逐字符一致**（True）

**数字独立复算（全部与 ckpt/*.out 存档比对通过）**

| 复算对象 | 独立计算 | 存档依据 | 结论 |
|---|---|---|---|
| 主 KV 单条目 288 | $512/2+512/16=256+32=288$ | `verify_cache_size.out`「512 维 fp4 + 每 16 通道 1 字节 scale = 288 B」 | 一致 |
| 索引器 K 单条目 68 | $128/2+128/32=64+4=68$ | 同上「128 维 fp4 + 每 32 通道 1 字节 scale = 68 B」 | 一致 |
| 折合 144 / 34 | $288/2=144$，$68/2=34$ | `verify_cache_size.out` 层 2-7/8-13/14-19 行 | 一致 |
| 主 KV 720 | $3\times144+288=432+288=720$ | 同上「主 KV 720」 | 一致 |
| 索引器 170 / 合计 890 | $3\times34+68=170$；$720+170=890$ | 同上「indexer K 170 = 890 B/token」 | 一致 |
| 反事实 8352 与 11.6 倍（1M 上下文、无共享） | $18\times144+20\times288=2592+5760=8352$；$8352/720=11.60$ | 同上「若无跨层共享 … 8352 B/token」「减少 11.60x」 | 一致 |
| 单层 377.31M | $126.62+1.97+35.39+212.34+0.98+0.010=377.31$ | `verify_active_params.out` 层 0 行 | 一致 |
| 层 20 = 385.40M | $134.71+1.97+35.39+212.34+0.98+0.010=385.40$ | 同上 层 20 行 | 一致 |
| 编码器半栈 7.8929B | $20\times377.31+3\times10.71+2\times157.33=7546.2+32.13+314.65=7892.98$M | `verify_active_params.out`「编码器半栈 7.8929B」；`verify_halfstack_diff.out` | 一致 |
| 解码器半栈 7.5758B | $20\times377.31+8.09+4\times5.40=7546.2+8.09+21.60=7575.89$M | `verify_active_params.out`「解码器半栈 7.5758B」 | 一致 |
| 差额 317.10M | $7.8929-7.5758=0.3171$B；$314.65+2.43+0.02=317.10$ | `verify_halfstack_diff.out`「+317.10M / +314.65M / +2.43M / +0.02M」 | 一致 |
| Engram 稠密投影 314.65M | $2\times157.33=314.66$（页取 314.65，尾差因四舍五入） | `verify_halfstack_diff.out`「层 1: engram_dense 157.33M；层 14: 157.33M；合计 314.65M」 | 一致 |
| 条目数 128 / 512 / 2048 / 16384 | `window_size=128`、`index_topk=512`、`candidate_topk_blocks=2048`、$2048\times8=16384$ | `official/inference/config.json`；`verify_csa2_modes.out`「candidate_topk_blocks=2048 × block_size=8 = 16384」 | 一致 |
| 头部维度 32×128 / 512 | `index_n_heads=32`、`index_head_dim=128`、`head_dim=512` | config.json；`verify_cache_size.out` | 一致 |
| 张量数 96085 / mtp 2401 | `len(headers.json)=96085`；`mtp.0/1/2.*` 键 2401 个 | `ckpt/headers.json` | 一致 |
| 主干 551.57B / Engram 196.93B | `count_params.out`「551.5662B」「196.9285B」，两层表 98305.58M/98308.27M | `ckpt/count_params.out` | 一致 |

**公式逐符号核对（F1–F8，全部与来源一致）**

- **F1**（L820）$C_j=\mathrm{RMSNorm}(\sum_{t=jr}^{(j+1)r-1}\mathrm{softmax}_{t'}(s_{t'})v_t)$ ↔ `model.py` L465 `kv, score = self.wkv(x), self.wgate(x)`；L475 `kv = (kv * score.softmax(dim=2)).sum(dim=2)`；L485 `return self.norm(kv.to(dtype))`。逐通道 softmax 门控、先池化再 RMSNorm，一致。
- **F2**（L1074）$I_{q,j}=\sum_h w_{q,h}\mathrm{ReLU}(q_h\cdot k_j)$ ↔ `model.py` L555–557：`weights = self.weights_proj(x) * (…)`、`einsum("bshd,btd->bsht", q, index_k)`、`(index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)`。ReLU 在加权内、按头求和，一致（常数因子折入 $w_{q,h}$）。
- **F3**（L1032）$n_{\text{reach}}(i)=\lfloor(i+1)/r\rfloor$ ↔ `model.py` L564 `compress_lens = (torch.arange(1, seqlen + 1) // ratio)`（prefill）、L567 `compress_lens = end_pos // ratio`（decode，单 token 即 $\lfloor(s+1)/r\rfloor$）。一致。
- **F4**（L1070）$\text{block}(b)=\max_{j\in b}I_{q,j}$、钉最新块、取前 2048 ↔ `model.py` L598–599 `F.pad(...).unflatten(-1,(-1,block_size)).amax(dim=-1)`；L604–605 `last=...; scores.masked_fill(arange==last, torch.inf)`；L607 `scores.topk(min(topk_blocks,num_blocks))`。一致。
- **F5**（L1260）$\text{score}=\sqrt{\mathrm{softplus}(xW^\top)}$、$\text{idx}=\mathrm{topk}_6(\text{score}+b)$、$w=\text{score}_{\text{idx}}/(\sum_{\text{idx}}\text{score}_{\text{idx}}+10^{-20})\times1.5$ ↔ `model.py` L817 `scores = F.softplus(scores).sqrt()`；L822 `indices = (scores + bias).topk(self.topk,-1)[1]`；L823 `weights = scores.gather(1, indices)`；L825 `weights /= weights.sum(-1,keepdim=True) + 1e-20`；L826 `weights *= self.route_scale`。偏置只进 topk、不进权重，一致。
- **F6**（L1294 说明 + 来源 [F6]）$\text{pre}=\sigma(ms_0+b)+\epsilon$、$\text{post}=2\sigma(ms_1+b)$、$\text{comb}=\text{Sinkhorn}_{20}(\mathrm{softmax}(ms_2+b)+\epsilon)$ ↔ `kernel.py` L427 `pre = sigmoid(mixes*hc_scale[0]+hc_base[j]) + eps`；L429 `post = 2*sigmoid(mixes*hc_scale[1]+hc_base) + 0`（无 eps）；L440–443 softmax 后 `+eps`；L446–458 行列交替归一化。`verify_sinkhorn.out`「行归一化 20 次（含 1 次 softmax），列归一化 20 次，末步 = col」与页面表述一致。
- **F7**（L962）$C_l=H_{L/2}W_l^{KV}$、$Z_l=H_{L/2}W_l^{Z}$（$l>L/2$）↔ `tech_report.txt` L394–396 报告 Eq. (1) 逐符号一致（$W_l^{KV}$、$W_l^{Z}$ 上标均正确）。
- **F8**（L1132）$o=\sum_{t\in\mathcal{T}}e^{q\cdot k_t/\sqrt d-m}v_t/(\sum_{t\in\mathcal{T}}e^{q\cdot k_t/\sqrt d-m}+e^{\text{sink}-m})$，$m=\max_{t\in\mathcal{T}}q\cdot k_t/\sqrt d$ ↔ `kernel.py` L355 `T.fill(scores_max,-1e30)`、L367 `acc_s *= scale`、L369 `reduce_max(..., clear=False)`、L373 `exp(acc_s - scores_max)`、L374–376 累加、L383 `sum_exp += T.exp(attn_sink[i] - scores_max[i])`、L385 `acc_o /= sum_exp`。汇聚点只进分母、只由选中槽位取最大、`-1` 槽位在 L362/L364 置 0 与 `-inf` 不贡献，均与页面 3.5 节三条说明一致；`verify_sparse_attn_window.out` 三组解析解差 0.00e+00。

**行号与引文逐条核对（tech_report.txt）**

| 标注 | 引文关键值 | 实际落点 | 判定 |
|---|---|---|---|
| [C1] 行 314–319 | 「Its language backbone comprises 40 causal Transformer layers, organized into a 20-layer causal encoder followed by a 20-layer decoder. Each layer incorporates both global attention and sliding window attention (SWA), except for the first two layers, which use SWA only.」 | 314–317 | 落在区间内 |
| [C2] §2.3.1 行 490–520 | 「Full Mode. The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices.」 | 497–498 | 落在区间内 |
| [C3] §2.4.4 行 689–691 | 「we select E2M1 with one E4M3 scale per 16 channels, following NVFP4 … but omitting its second-level global scale」 | 689–691 | 精确 |
| [C3] 行 703–704 | 「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」 | 703–704 | 精确 |
| [C6] 行 388–393 | 「For global attention, CED treats the bottom L/2 layers … the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H_{L/2}, using layer-dependent projection weights」 | 389–392 | 落在区间内 |
| [C6] 行 544–546 | 「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer, i.e. the last layer of the causal encoder.」 | 544–546 | 精确 |
| [C7] §2.3.2 行 542–570 | 「For example, selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions.」 | 569–570 | 落在区间内 |
| [C7] 行 566–567（辅助解释处引用） | 「each block is assigned the maximum index score among its positions」 | 566–567 | 精确 |
| [C11] 行 404–410 | 「an SWA replay process」「we introduce Decoder SWA Bounded Replay, which only prefills the last n_win tokens」 | 404、409–410 | 落在区间内 |
| [N1] 行 18–19 | 「reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash」 | 18–19 | 精确 |
| [N1] 行 133–140 | 「552B backbone parameters … 8B parameters per token during prefill and 16B during decode … approximately 1/4 as much runtime KV cache storage」 | 133–137 | 落在区间内 |
| [N2] 行 320–322 | 「activating 8B parameters per token during prefill and 16B during decode」 | 320–321 | 落在区间内 |
| [N3] 行 320 | 「552B backbone parameters and 196B Engram parameters」 | 320 | 精确 |
| [N5] §2.4.4 行 691–700 | 「supports magnitudes up to 448 × 6 = 2688」「the maximum magnitude observed during training is around 10」 | 692、698 | 落在区间内 |
| [N9] 行 35–38 | 「achieves approximately 4-fold and 437-fold reductions in per-token global KV cache size relative to DeepSeek-V4-Flash and DeepSeek-V1」 | 37–38 | 落在区间内（原文为 N-fold，页面写作 1/4、1/437，等价） |
| [N9] 行 139–140 | 「approximately 1/4 as much runtime KV cache storage and 1/8 as much persistent KV cache storage」 | 139–140 | 精确 |

**代码/源码行号核对（model.py / kernel.py）**：[C2] L496–503、L653–661、L722–737；[C3] L760/L707/L546/L552；[C4] L328–365、L344；[C5] L1100–1156、L1131–1132、L1265–1266；[C8] L809–827、L817、L821–823；[C9] kernel.py L406–462、model.py L948–966；[C10] L537–548（L537 `if self.owns_k and latent is not None:`）+ L554（`index_k = shared_attn.index_k[...]` 无条件执行）；[C11] L1261–1267 主层循环无跳过条件、`official/inference/` 全目录 `grep -rin replay` 无命中；[F1] L458–485；[F2] L550–557；[F3] L563–567；[F4] L596–610；[F5] L811–826；[F8] kernel.py L310–403。以上除 [F1] 的组位置子项与 [F7] 的 Eq. 编号外，均在标注区间内（见问题 3、4）。

**CKPT 数字核对**：`verify_compressor_pooling.out`「fp32 层面池化结果最大差 4.530e-06（相对 2.18e-06）」「cast 到 bf16 后不一致元素 5/4096」（页面 1.1 节 4.5e-6 / 2.2e-6 / 5 个 ulp）；`verify_fp4_error.out`「0.0951 / 0.0265 / 3.58x」「448 x 6 = 2688」「余量 268.8x」（页面 1.2 节 9.5% / 2.65% / 3.58 倍 / 268 倍）；`verify_gate_formula.out`「indices 完全一致: True，权重最大差 0.000e+00」（[C8]/[F5]）；`verify_csa2_modes.out`「Full = [2,8,14,20]、Reindex = [24,28,32,36]、Reuse = 30 层、SWA = [0,1]」（[C2]/3.4 节）；`debug_consistency7.out`「偶数步 layer2 读到自己的 cache? False（值等于 layer4 的缓存）、奇数步 True」（[C10]）；`debug_consistency3.out`「layer 2: 首个 pos 10, max 8.850e-03」（[N7] 8.9e-3）。以上全部一致。

**无法独立复算的登记项**：[N9] 的 1/4、1/437、1/8 三个倍数，页面已明确标注为「报告宣称」且未据此比较架构优劣，处理符合 check.md §2.2 第 4 条，本轮不记为问题。

## 问题

- [重要·技术] 4.2 节折叠块「展开：两个半栈的完整加总」（index.html L1320、L1322）：两处「18 个普通层」与紧随其后的加总式矛盾。编码器为层 0–19 共 20 层，其中 Full 层 3 个（2、8、14），普通层应为 17；解码器为层 20–39 共 20 层，其中 Full 层 1 个（20）+ Reindex 层 4 个（24、28、32、36），普通层应为 15。按文中「18 个普通层」复算得 $18\times377.31+32.13+314.65=7138.36$M，而 L1321 自己的算式是 $20\times377.31+32.13+314.65=7892.98$M，同一段落内两数互斥；解码器同理（$18\times377.31+8.09+21.60=6821.27$M vs $7575.89$M）。最终结论 7.8929B / 7.5758B 与实测一致，但读者按字面层数无法复现该展开｜引文依据：`verify_halfstack_diff.out`「层 2/8/14: attn 137.33M」「层 20: attn 134.71M」「层 24/28/32/36: attn 132.02M」；`official/inference/config.json` `kv_source_layers=[2,8,14,20]`、`index_source_layers=[2,8,14,20,24,28,32,36]`｜修复要求：把两处「18 个普通层各 377.31M」改为与算式一致的表述——或改成「20 个层各按 377.31M 起算」，或改成「17 个普通层各 377.31M，另加层 2、8、14 三个 Full 层的 377.31M 基准」并同步解码器为 15 个普通层加 5 个特殊层，改后重算两行合计必须仍得 7892.98M 与 7575.89M｜修复：｜复验：

- [轻微·技术] 4.1 节折叠块「补充：超连接模块为什么只占 0.98M」（index.html L1294）：「输入是 5120 维的隐状态」与源码不符。`model.py` L938–941 定义 `hc_dim = hc_mult * args.dim = 4×5120 = 20480`，L952 `x = x.flatten(2).float()` 后做 `F.linear(x, hc_fn)`，`hc_fn` 形状为 `[mix_hc, hc_dim] = [24, 20480]`，即输入是展平后的 20480 维（4 份残差流），输出才 24 维；0.98M 正是 $2\times24\times20480=983{,}040$。按页面写的 5120 维复算只有 0.25M｜引文依据：`count_params.out`「ok layers.0.hc_attn_fn: [24, 20480] F32」｜修复要求：将「输入是 5120 维的隐状态」改为「输入是展平后的 $4\times5120=20480$ 维残差流」，或删除该分句只保留 0.98M 的数值来源说明｜修复：｜复验：

- [轻微·技术] 3.3 节正文（index.html L1101）：「若按常规打分它会因平均分偏低而落选」引入了不存在的「平均分」路径。实现与报告的块分都是块内**最大值**：`model.py` L597–599 `F.pad(...).unflatten(-1,(-1,block_size)).amax(dim=-1)`；`tech_report.txt` L566–567「each block is assigned the maximum index score among its positions」。最新块落选的机制是「被更早的完整块以更大块分压过」（`model.py` L602–603 注释 "could otherwise be outscored by an older, full block"），不是平均分偏低；文末「辅助解释与类比边界」已登记「块分取最大值」的动机，但未覆盖此处的平均分归因｜引文依据：`model.py` L597–599、L602–603；`tech_report.txt` L566–567｜修复要求：删除「因平均分偏低」，改为「块分可能被更早的完整块压过」，或在文末辅助解释中把这条一并登记为推断｜修复：｜复验：

- [轻微·技术] 5.2 节正文（index.html L1391）：「decode 阶段的瓶颈是每生成一个 token 都要走一遍全栈，而这一步几乎全是访存、算力用不满」把访存瓶颈写成事实，但来源只描述做法与效果：`tech_report.txt` L186–187「introduce the DSpark … speculative decoding architecture to improve decoding efficiency through semi-autoregressive draft generation and confidence-scheduled verification」、L188–191「the single-token Decode FLOPs … remain nearly constant across context lengths」，均未给出「访存瓶颈/算力用不满」的归因。[C5] 的引用范围（config 与 `model.py`）也不含该结论｜引文依据：`tech_report.txt` L186–187、L188–191｜修复要求：将该句降级为辅助解释（「以下为辅助解释：…」）并给出失效边界，或删除「几乎全是访存、算力用不满」只保留报告支持的「提高解码效率」｜修复：｜复验：

- [轻微·格式] 来源章节（index.html L1448）[F7]：「报告行 388–393（报告 Eq. 1）」。行 388–393 只有文字描述，报告 Eq. (1) 的公式排版与编号在行 394–396（`C_l = H_{L/2}W_l^{KV}, Z_l = H_{L/2}W_l^{Z}, l > L/2,   (1)`）。公式符号虽在 L389–392 的文字中已出现，但标注的「Eq. 1」落在区间之外｜引文依据：`tech_report.txt` L394–396｜修复要求：把 [F7] 的行号改为「行 388–396」或「行 389–396（报告 Eq. 1）」｜修复：｜复验：

- [轻微·格式] 来源章节（index.html L1435）[F1]：把「RoPE 位置 $j\cdot r$」与压缩器池化合并标注为 `model.py` L458–485。池化确实在 L458–485（L475 softmax 池化、L485 RMSNorm），但「组 $j$ 取位置 $j\cdot r$」的语义在 L538–543（`Indexer.forward` 内，L538 注释 "group j takes position j * ratio"）与 L752–757，不在标注区间｜引文依据：`model.py` L538–543、L752–757｜修复要求：把 [F1] 拆开，池化标 `L458–485`，组位置标 `L538–543`（或 `L752–757`）｜修复：｜复验：

- [轻微·格式] 公式定界符外出现裸 Unicode 乘号 U+00D7 共 3 处：index.html L1448「索引头 32×128、KV 头 1×512」（同句前半已写 `$2048\times8=16384$`，同一句内 $\times$ 与 `×` 混用）与 L1294「4 份 × 三套系数」。style-guide.md §11 要求「数学变量、…数学运算符和关系符都必须包在 `$...$`」且「同一变量在页面中保持同一种写法」，此处属同一运算两种写法并存。`validate.py` 未拦截该字符｜引文依据：style-guide.md §11；index.html L1448 原文「候选 $2048\times8=16384$、索引头 32×128、KV 头 1×512」｜修复要求：L1448 两处改为 `32$\times$128`、`1$\times$512`（或整体包进行内公式）；L1294 的「4 份 × 三套系数」属中文列举分隔，可保留，或统一写成「4 份（乘）三套系数」以免与数学乘号混淆｜修复：｜复验：

- [轻微·格式] 「主要依据」（index.html L751）未列出 `research/official/config.json`（HuggingFace 版），而页面两处引用的字段名只存在于该文件：[C8]「配置 … `norm_topk_prob=true`」与 [C10]「配置 `kv_source_layer_ids=[2,8,14,20]`」。`official/inference/config.json` 中对应字段为 `n_activated_experts`、`kv_source_layers`，不含 `norm_topk_prob`｜引文依据：`official/config.json` `text_config.norm_topk_prob: true`、`text_config.kv_source_layer_ids: [2,8,14,20]`；`official/inference/config.json` 全文无 `norm_topk_prob`｜修复要求：在「主要依据」中补列 `official/config.json`（并注明与 `inference/config.json` 的关键字段一致），或把 [C8]/[C10] 的字段名改为推理版命名并改引推理版｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 7
- 处置：修复。核心结论与全部关键数字（288 / 144 / 720 / 170 / 890 / 8352、377.31M / 7.8929B / 7.5758B / 317.10M / 314.65M、128 / 512 / 2048 / 16384）经独立复算与 `ckpt/*.out` 存档逐项一致；F1–F8 与 `model.py` / `kernel.py` / 报告 Eq. (1) 逐符号一致；3.6 节代码实跑输出与页面「预期输出」及 `verify_reach_topk.out` 逐字符一致；`validate.py` 对 index.html 与 overview.html 均返回成功；headless Chrome 渲染 736 个 KaTeX 节点、0 渲染错误。无阻断问题，唯一重要问题集中在 4.2 节展开块的层数标注，关闭该条并处理 7 条轻微问题后可进入第 3 轮复核。
