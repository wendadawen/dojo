# 计划审查第 5 轮

第 5 轮源码位置核查通过 276 个节点，有结论的席位 4 个（审查者 3 未计入），有事实错误，不通过。之后范围改过几次——加视觉支路、补薄包装类图、按新条文重做、按任务要求改一批——每次改动都重审了一遍，各成一节。

## 第 5 轮

- 事实错误｜第二行左侧写成生成方式而非形状｜Attention `widx`、Indexer `lens`、NgramHashState `pos`｜model.py:720、563-564；engram.py:169｜改成 `[B, T, 5120] → int32 [B, T, 128]`、`[B, T, 5120] → int64 [T, 1]`、`int64 [B, T] → int64 [B, T]`
- 事实错误｜NgramHashState `upd` 折叠｜`DEAD` 写成压缩词表空位，实为哨兵 -1｜engram.py:138、165-166｜改成 `DEAD = -1`（不参与 n-gram；本路径 engram_mask 为 None，不写 cache）
- 措辞｜Transformer `layers` 源码列｜`model.py:907` 是 `class Block`，调用在 1267｜model.py:1267｜改到 1267
- 措辞｜`flatten`、`head`、`mback`、`prod`｜`flatten(x)` 没写从第 2 维起；「只取最后一个位置」易读成对 weight 切片；`x` 说成拉平前的输入；公式没写 `unsqueeze(2)`｜model.py:952、1010-1012、891-904；engram.py:179｜改成 `flatten(2)`、对输入取末位置、用拉平后的隐状态、补 `unsqueeze(2)`
- 措辞｜量化公式、MoE `counts`、介绍/描述、若干说明｜`max(|·|)` 实为每组 absmax；`counts` 类型写「张量」实为 list；描述写成压缩比 > 0 的层都由索引器选 Top-512；说明写了前提外（ratio=2）的层型｜kernel.py `reduce_absmax`；model.py:894、725-726｜公式写明每组 absmax；第二行写明 Python list；非 index source 层复用已发布的 topk；先写本图路径
- 措辞｜MoE 连线 `shared`、Transformer `layers` 多输出｜`shared` 未点开 `Expert.forward`；末层 `pre_mix` 无张量节点｜model.py:903、1267-1268｜补点开；在 layers 与 col 之间补张量

### 处理

- widx/lens/pos、DEAD、layers 源码列、pre_mix 单独成 `lpm`、flatten(2)、head 取末位置、mback 用拉平后隐状态、prod 补 `unsqueeze(2)`、量化公式改每组 absmax、介绍/描述与层型：均成立，按上表改；不成立的是 `shared` 不点开 `Expert.forward`（那张图是传入 weights 的路由专家）、`counts` 仍写「张量」（计划里没有 list 这一类，第二行已写明是 list）。

## 通过之后的收口

源码位置核查通过 276 个节点，有结论的席位 4 个（审查者 4 未计入），无事实错误，通过。

- 措辞｜MoE 连线 `shared`｜未点开到 `Expert.forward`（路由专家的 `mexp` 有点开）｜plan.md；model.py:903｜补上

### 处理

- 无事实错误。页面核对发现 `acc`、`madd`、`off` 第一行用了源码里没有的名字，改回源码那一行；`shared` 仍不点开（那张图是传入 weights 的路由专家，含 emw；共享专家不传 weights，前提已写明）。

## 加视觉支路后的复审

源码位置核查通过 371 个节点。前提从「不传图像」改成「传一张图」，新增 ViT、vision.Block、vision.Attention、vision.MLP、vision.RMSNorm、Aligner 六张图，Transformer、Block、MoE、Gate、Engram、NgramHashState 六张图随 `image_mask`/`engram_mask` 从 None 变成实际张量而改。

- 事实错误｜Image 支路漏步、漏变换（Engram）｜`g → nadd` 漏一步（token_mask 非 None 时 gate 先在图像位置置 0）；`nadd` 折叠的 `gate` 一符号指两段；`vf` 折叠没写是什么变换｜model.py:362-365
- 事实错误｜漏画、漏连线（Transformer、Aligner、vision、Attention）｜`mrg` 的 `img.types`/`img.start`/`image_start`/`image_end`/`image_newline`、`venc` 的 `n_vit_h`/`n_vit_w`、`eng` 的 `engram_mask`、Aligner `vpad` 的 `n_h`/`n_w`/`r`、vision.RMSNorm `vmw` 的 `self.weight`、Attention `core` 的 `attn_sink` 未画未连；vision.Block `vby → vbad2` 与 `vbm1 → vbad2` 不区分次序｜model.py:1231-1239、1238、1262-1263、780；vision.py:115-117、34、83-84
- 事实错误｜MoE/Gate 画法与用词｜`mfm` 孤立无连线；`gate_op` 第一行仍带 `.flatten()`（与算子 `mmask` 重复画同一调用）；`acc` 折叠 `x[idx]` 详略不一致；`mmask`/`bsel` 线上字与折叠混用压平前后的量｜model.py:892、900、819-820
- 事实错误｜id 统一、算子/张量交替｜`image_mask`（`im`/`bim`/`cmsk`/`nmask`）、`patches`/`vp`、`vcs`/`vbc`、`vlaout`/`vemb`、`vax` 撞 id；`layers → eng`、`layers → avg`、`hin → resid0`、`wkv_q → wchunk` 是算子/张量直连｜plan.md｜统一 id、补中间节点
- 事实错误｜Gate `bsel`/`bvl`、vision.RMSNorm `vrout`、Aligner `vpad` 折叠、ViT `vcs_op` 除数、Transformer `venc`｜`bsel` 实为 `unsqueeze(-1)` 广播、整行同偏置；`bvl` 条件应挂 `vision_enabled`；`vrout` 的「进入 RMSNorm 时 x 的 dtype」在 t=0 不存在；`vpad` 对未指定尺寸的图断言、且与 ceil 差 1；`vcs_op` 除数 2j/64 对不上源码（`rope_dim`=32）；`venc` 挂了两条点开、后覆盖前，ViT 点不进去｜model.py:819-820、139-140、1238、1225-1226；vision.py:117、41、9-10｜按表改
- 措辞｜符号两义、id 不存在、全称、摘要与面板一组｜`base`（widx 折叠 vs 面板）、`head_dim`（视觉 64/主干 512）；`vbc` 引用了不存在的 `vcos`；MTP、RoPE 首次出现未给全称；摘要/描述图像嵌入语序反、写回的是取完词嵌入后的隐状态、`venc` 第二行不是形状说法、Sinkhorn 只 comb、压缩 KV 未区分「只由 4 个 source 层产出」、DSpark 叫法不一、config 表组名无上下文、Engram 层 1/14 无解释、`VL` 未展开、介绍指代不清、六张新图标题前缀不统一、面板「本类」未说清、vision.RMSNorm 面板无 eps｜model.py:419、376、630；vision.py:50、82、100；dataflow.md 等｜逐条改

### 处理

- 事实错误逐条改：Engram 补 `nmask → gmask → gm`、`nadd` 符号统一、`vf` 补公式；Transformer 补 `tt`、`patches → venc`、`vp → venc：n_vit_h、n_vit_w`、`emv → eng`，`im`/`bim` 与 `vlaout`/`vemb` 统一 id；Aligner 补 `vout → vpad` 与 `vrd → vpad`、`vpad` 折叠改条件句、输入 id `vax` → `vout`；vision.Block 两条线标 residual 与上一层输出；vision.RMSNorm 把 patches bf16 写进前提、`vmw` 补 `self.weight`；`venc` 拆 `venc`（self.vision）/`alen`（self.aligner）各挂一个点开；Attention 补 `attn_sink`；MoE `gate_op` 去 `.flatten()`、`mmask`/`acc` 折叠改；Gate `bsel` 改扩维广播、`bvl` 改 `vision_enabled`；ViT `vcs_op` 改 `rope_dim`；算子/张量交替补中间节点。
- 其中 4 条是上一轮遗留（`eng` 无连线、`core` 无节点、算子/张量直连、`patches`/`vp` 与 `vcs`/`vbc` 跨图 id），本轮一并改。
- 不改（写理由）：`vp → vcs_op` 的 `n_vit_h`/`n_vit_w` 不单独成节点（惯例：形参连一条线并写在线上）；六张新图标题前缀只在与已有图撞 tab 名时带 `vision.`；`shared` 不点开、`counts` 写「张量」、三张薄包装类不单独成图——前几轮已裁定。

## 视觉支路改完再复审

源码位置核查通过 378 个节点。5 席从头重审（上一版的改动已在计划里），有事实错误，改完后不通过。

- 事实错误｜Indexer.forward｜层 20 是 candidate source，`select_candidate_blocks` 与 masked_fill 两步无节点、无缓存写入，而 Transformer 的 `shar` 与 `layers` 说明已在说层 20 写、24–36 读；`lat → kw → krope → kq → kwrite` 画成无条件执行，实际只在 `self.owns_k` 的层跑（前提路径上 24/28/32/36 不占 kv source、latent 为 None，整段跳过、只从 `shared_attn.index_k` 读键）｜model.py:569-575、583-610、500、517-548、655、722-737
- 事实错误｜同量多图多 id、漏连线｜Aligner `vemb` 与 Transformer `vemb`、`patches`/`vp` 不一；MoE `mmask` 的输入 `image_mask` 悬空；Aligner `vpad` 的 `r` 未连｜vision.py:119、98、115、117；model.py:1238、892
- 事实错误｜Attention `core`、Transformer `eng`｜`attn_sink`、`engram_mask` 无节点无连线（同上一轮，未改）｜model.py:780、1262-1263
- 措辞｜同上一轮一组 + 写法｜`bsel` 广播、`bvl` 条件、`vrout` dtype、`vby`/`vbad2` 次序、`vpad` 断言、`vf` 变换、`vcs_op` 除数、`base`/`head_dim` 两义、MTP/RoPE 全称、摘要等；MoE `mmask` 与 Gate `bsel` 线上字与折叠用词混着压平前后｜同上

### 处理

- 事实错误：Transformer 补 `emv → eng`、`venc` 拆 `venc`/`alen` 接张量 `vla`；mask 的 id 统一（`bim`/`cmsk`/`nmask` → `im`/`emv`，压平后的仍叫 `mfm`）；`patches`/`vp` 统一成 `patches`、`vlaout` → `vemb`；Aligner 补 `vrd → vpad：r` 与 `vout → vpad：n_vit_h、n_vit_w`；Attention 补 `attn_sink`（fp32 [64]）与连线；MoE 补 `image_mask` 连线、去 `.flatten()`、改 `mmask`/`acc` 折叠；Engram 改 `nadd` 折叠、`vf` 补公式；vision.Block 两条线标 residual 与上一层输出；vision.RMSNorm 把 patches bf16 写进前提；ViT `vcs_op` 改 `rope_dim`、`widx` 的 `base` 改 `idx_base`。
- 审查者 1 的 Indexer 两条（主干）也改：`iscm`/`topk` 间补 `csel`、`cmf` 两算子与 `cmask`/`cmfd` 两张量、`shar` 缓存（层 20 写、24–36 读）；`lat` 第三行写明只有 4 个 kv source 层是实际张量、其余为 None，`lat → kw → krope → kq → kwrite` 跳过。
- 改动后再核 build_page、check_source、site check、`--geometry` 全过（380 节点、18 图）；下一轮 5 席重审。不改（写理由）：同上（`vp → vcs_op` 形参不成节点、视觉图前缀、shared/counts/薄包装类）；另 `PatchEmbed` 不单独成图（只有一次线性投影，画在 ViT 的 `vpe` 上）。
- 仲裁｜审查者 1 第 1 条（Indexer 候选块节点是多画）｜作者核实前提把路径锁在「compress_ratio 为 2」，与 8 个 index source 层的画法不符；审查者给两条互斥改法，作者选「改前提」而非删节点——删了会丢掉上一版判「必须补」的两步。

## 网络中断后的补审

源码位置核查通过 379 个节点。有结论的席位 2 个（节点与形状、读者与措辞），有事实错误，不通过；路径覆盖、连线与结构、公式与面板三席收口前未返回，按 review-plan.md 记「未计入」。5 席本都重派，未到 4 席有结论，到轮数上限后改为把已知问题改完并做最后一批改动核对。

- 事实错误｜MoE.forward `mmask`｜第一行 `image_mask.flatten()` 里被压平的 `image_mask` 无产出节点、无线｜model.py:889、892
- 措辞｜描述、摘要、资料｜「切成 patch 过 32 层…编码」缺连接；裸用「source 层」未区分 kv/index；压缩 KV 那句句式挤压；FP8/FP4 大写、Top-512 大写 T 与全篇不一；「块号」指什么不明｜dataflow.md
- 措辞｜前提｜顿号连接两个分句；层 20 压缩器口径与 Compressor 图不一｜前提
- 措辞｜面板、折叠一组｜`mrg` 的 `img.start`/`numel(types)`/`types` 未交代来源；`IMAGE_START` 与 `image_start` 未注明区别；Attention 面板 `r`/`g` 两义、YaRN 的 `corrected_dim` 未交代；`qb`/`grp` 未说是本卡的量；E8M0/E4M3/E2M1 未解释；「窗口 KV/窗口键」换名；`imask` 公式两处写法不一；Expert `g1`/`w2` 未说 fp4 打包；Aligner `vv`/`vpad`/`vunf` 的 `r` 与 Attention 的 `r` 同形不同义；`wkv`/`w1`/`w2` 跨图同名未注明；config 表 `DSpark` 未交代｜dataflow.md 等

### 处理

- MoE 补 `im` 节点与 `im → mmask` 连线。描述、摘要、资料、前提、面板、折叠逐条改：加逗号、source → kv source、压缩 KV 改「除 128 个位置的滑动窗口外」；FP8/FP4 → 小写、Top-512 → top-512；「块号」改「同一层内同一子模块的编号」、补模型卡一行；顿号改分号、层 20 压缩器写明「不做逐 token 池化、只留 wkv 一路」；`mrg` 注明大写是 token 类型、小写是学到的嵌入，输出低秩投影下标 `r` → `m` 并注明与 compress_ratio 无关，`qb`/`grp` 补单卡 64/8，`wquant` 补 E8M0、`cquant` 补 E4M3/E2M1，`imask` 改与节点一致，Expert/Aligner/Attention 折叠措辞按表改，多图同名权重在各自说明点明模块，config 表 `DSpark` 补说明。
- 收口机械核查：落盘后重跑 build_page（18 图 381 节点）、check_source（计划 379、页面 18 视图 381 节点）、site check、`--geometry` 全过；截图逐张看过；介绍一组改成两句（模型、这次前向），数字出自补进「资料」的 model-card.md。
- 已知不足：三席未返回（这三类覆盖在第 7、8 轮做过）；「最后一批改动也要核对」没做（到收口时无可用的仲裁席位），改动只经作者自查与机械检查。

## 补审之后

源码位置核查通过 379 个节点。有结论的席位 3 个（路径覆盖、连线与结构、公式与面板），只重派这三席（已返回的两席不重派），审收口后的现状；按 review-plan.md 把上一版末尾到这一版结束的全部改动交一个独立子代理核对。

- 事实错误｜Indexer 的 `csel`/`cmask`/`shar`/`cmf`/`cmfd`｜这两段在前提锁定的路径上不执行，画了是多画（`if is_candidate_source` 与 `elif uses_candidates` 互斥，只在层 20 / 24+ 成立；前提写的是「索引器走 compress_ratio 为 2 分支」）｜model.py:569-575；config `candidate_source_layer`=20；前提第 28 行｜删这 5 节点让 `iscm` 直连 `topk`，或改前提明确画成层 20 写候选、24/28/32/36 读候选
- 事实错误｜Transformer `emb`、Engram `nemb`｜`ParallelEmbedding` 与 `ParallelEngramEmbedding` 是源码自定义、各有 `forward` 的类，全页无图，用到它们的节点没点开，反而把类内算式抄进折叠｜model.py:168-178、312-325；plan.md｜各补一张图，折叠改成写调用加点开
- 写法｜Transformer.forward、Gate.forward、Block.forward｜`layers → eng`、`layers → avg` 算子直连算子，循环里那份 h 无张量节点；`bsel → gsel` 选出的 bias 无节点无线；`hin → resid0`/`x4 → resid1` 张量直连张量，`residual = x` 未注明同一份｜model.py:1263、1266、819-822、981、988；plan.md｜给 h 补张量节点（`eh` 不能兼作 Block 的输出）、Gate 补 fp32 [B×T, 384] 张量节点、Block 画成算子节点或并进 `hin`/`x4` 并注明
- 事实错误｜ViT `vcs_op` 折叠与面板、vision.Attention `vcq` 说明｜`inv_freq_j = θ^{-2j/head_dim}` 的除数对不上源码（`rope_dim`=32，写成 2j/64，差一倍，说明自相矛盾）；`vcq` 写「一对相邻的通道配成一个二维向量」，实为每头 64 维前一半配后一半（第 i 维配第 i+32 维）｜vision.py:10、92、100、20-21｜改 `θ^{-2j/rope_dim}`、符号说明同步改；`vcq` 改成前一半配后一半
- 措辞｜折叠抄被调类算式、符号两义、面板介绍｜`head`/`fnorm`/`an`/`fn`/`qr_op`/`wkv_win`/`cnorm`/`kw`/`vnorm`/`vbn1`/`vbn2` 的折叠抄了被调类内部算式（这些类本页各有图，应写调用）；Attention 图 `t`（widx 式指 query 位置、源码名 `end`；core 式指槽位号）且 widx 用了源码没有的名字、Indexer 图 `i`（`red` 式头维、`lens` 式 query 位置）；Transformer 面板「介绍」只写模型是什么、没写这次前向在算什么｜plan.md；model.py:418、557、563-564；kernel.py:358｜折叠改写调用；widx 式用 `end`、core 槽位号换字母、`lens` 的 `i` 换字母并各自点明；面板补这次前向

### 处理

- 事实错误：ViT `vcs_op` 除数改 `rope_dim`；vision.Attention `vcq` 改前一半配后一半；新增 `ParallelEmbedding.forward` 与 `ParallelEngramEmbedding.forward` 两张图，`emb`/`nemb` 折叠改调用加点开；Indexer 改前提（走 compress_ratio > 0 分支 + 8 个 index source 层共用一张图、层 20 写候选、24/28/32/36 读候选、不建键只读 index_k），前提与图一致。写法：Transformer 用 `hl` 节点第三行写明（循环每层返回又喂下一层），连线改 `layers → hl`、`hl → eng`、`hl → avg`；Gate 新增 `bbias`（fp32 [B×T, 384]），连线 `bsel → bbias → gsel`；Block 删 `resid0`/`resid1` 并进 `hin`/`x4` 第三行，连线改 `hin → cpost：residual`、`x4 → cpost2：residual`。措辞：11 处折叠由抄算式改写调用；Attention `widx` 的 `t` → `end`、`core` 槽位号 `t` → `s`；Indexer `lens` 的 `i` → `s` 并注明；Transformer 面板补「介绍｜这次前向」一行。
- 仲裁｜路径覆盖两条｜第 1 条（Indexer 候选块多画）：作者核实前提确实与 8 层画法不符，选「改前提」而非删节点（删了会丢掉上一版判「必须补」的两步）。第 2 条（两个 embedding 类该不该各补图）**必须改**：依据 plan.md 第 78 行（模型自定义、有 forward 的类每个都画一张，用到它的节点都点开）与第 87 行（折叠写算子自己的计算、不抄被调类内部算式），两个类都是源码自定义、各有 forward、本次用到，不属豁免的 PyTorch 自带模块或普通函数；作者的反驳（薄包装类不单独成图、`PatchEmbed` 也这么画、前几轮已裁定）不构成条文依据，不予采纳。
- 最后一批改动核对：独立子代理逐项核过，**(a) 通过、(b) 无**（无凭空内容、漏步、接错线；未违背前几轮裁定——`shared`、`counts`、`PatchEmbed`、`patches → vcs_op` 的 `n_vit_h`/`n_vit_w`、六张视觉图前缀均未动）；三道机械检查全过（build_page 20 图 394 节点、check_source 计划 392 / 页面 394、site check）。已知不足：上一次复审的「节点与形状」「读者与措辞」两席按上节计入、未重派；页面几何与截图由作者在页面级重跑，本记录只覆盖计划与生成。

## 补薄包装类图

源码位置核查通过 409 个节点，有结论的席位 5 个，有事实错误（两处重复点开），改完后按「最后一批改动也要核对」交仲裁核对。plan.md 本轮加「只有一行计算的薄包装类（如 `PatchEmbed`、`ParallelEmbedding`）也要单独画」，据此给 4 个薄包装类补图：`Linear`、`ColumnParallelLinear`、`RowParallelLinear`、`PatchEmbed`（原先画在 ViT 的 `vpe` 上，本轮抽出）。一开始给几个视觉节点加了「点开 → Linear.forward」，作者自查发现 vision.py 用的是 PyTorch 自带 nn.Linear，已先撤 6 条。

- 事实错误｜点开｜Attention `qr_op` 用了 `self.wq_a`（Linear）却漏「点开 → Linear.forward」（同图 `wkv_win` 接齐了）；但 `wkv_win`、Indexer `kw` 各写了两条点开（RMSNorm 与 Linear），生成器一键只存一个目标、后覆盖前，RMSNorm 目标失效而折叠仍写「见点开的那张图」｜model.py:770、640、705、544；build_page.py:251-257｜补 `qr_op` 点开；一个节点只留一条点开
- 事实错误｜点开对不上｜Attention `woa` 是 `self.wo_a.weight.view(...)`，只读权重不调 forward（源码注明用 einsum 代替 Linear）｜model.py:783-787｜删点开或折叠写明只取 weight
- 事实错误｜PatchEmbed `pe_w`｜算子同时用 `proj.weight` 与 `proj.bias`，weight 画成节点、bias 只写折叠｜vision.py:40、43｜补 bias 张量节点或去掉 `pe_w`
- 措辞｜`lin_op`/`cpl_op`/`rpl_op` 说明｜选路枚举漏了 fp32 也走 `F.linear`；`cpl_op` 写了本页不存在的 fp4；把 `y += self.bias` 并进 world_size > 1（实由 `self.bias is not None` 把守）｜model.py:186-207、446、448、514-515、645-649、273-277｜按表逐条改
- 措辞｜三处折叠的「本页每个调用点都没传 bias」、`rpl_y` 与 `rpl_op`｜「本页」字面覆盖整页，但视觉支路多处显式带 bias；`rpl_y` 把 all_reduce 与 bias 两段都归因单卡，折叠说 bias 与 world_size 无关（前后不一致）｜`rpl_y` vs `rpl_op`｜改「主干的调用点都没传 bias」；节点写「all_reduce（单卡不执行）与 bias（本页 `self.bias` 为 None）」
- 措辞｜符号｜Attention `eins` 与面板、`cf` 与面板：同一量折叠写 `o_{b,s,g,r}`、面板写 `o_{b,s,g,m}`；字母 `g` 一会儿压缩分组序号、一会儿 o_groups 组号｜model.py:753-757、539-543｜`eins` 改用 `m`；压缩分组序号 `g` → `j` 并注明
- 措辞｜PatchEmbed `pe_op` 折叠、`lin_w`/`cpl_w`/`rpl_w` 第二列、vision 三图线性投影｜`pe_op` 公式没写 bias、说明却列了 `proj.bias`；三个权重节点第二列固定写 `fp8 [out, in]`，折叠说 dtype 随调用点变；vision.Attention/MLP/Aligner 的线性投影既不点开 `Linear.forward` 也不注明是 nn.Linear｜`pe_op`；`lin_w` 等；Linear 图折叠｜公式写含 bias 或注明；第二列写「（随调用点变，见折叠）」；补点开或注明

### 处理

- 重复点开：`wkv_win`、`kw` 各留原有的 `RMSNorm.forward`（折叠引用的是它），撤本轮新加的 `Linear.forward`；Linear 在 `qb`/`proj`/`g1`/`u1`/`w2`/`nwkv` 等点得到。`woa`：撤点开（只读权重）。`qr_op`：审查者要求补 Linear 点开，但会与同节点 RMSNorm 点开重复，按统一规则保留 RMSNorm、不补 Linear。PatchEmbed 补 `pe_b`（`self.proj.bias`，bf16 [1024]）与连线，`pe_op` 公式改含 bias。
- `lin_op`/`cpl_op`/`rpl_op` 说明按表改、`rpl` 的 bias 段拆出、三处「本页」改「主干」；Attention `eins` 的 `r` → `m`、`cf` 与面板的压缩分组序号 `g` → `j` 并注明；`lin_w`/`cpl_w` 第三行补说明；视觉三图线性投影注明是 nn.Linear；最后一批改动核对指出的两处残留（`lin_w` 第二列、`cpl_op` 末句）已改。改动后重跑 build_page（24 图 411 节点）、check_source（计划 409、页面 24 视图 411 节点）、site check、`--geometry` 全过。
- 仲裁｜第 1 条（`qr_op`/`wkv_win`/`kw` 该不该补 Linear 点开）**不必改**——plan.md 第 78 行只约束到「类」这一级，不要求一个节点挂两条点开，两个类都点得到，`build_page` 的单值赋值是真实引擎约束；第 2 条（`woa` 撤点开）**不必改（作者处理正确）**——依据 model.py:783-787 与 plan.md 第 85 行。
- 最后一批改动核对：独立子代理核过，**通过**；(a) 无凭空内容、漏步、接错线，(b) 无违背（`shared`、`counts`、`patches → vcs_op` 的 `n_vit_h`/`n_vit_w`、六张视觉图前缀均未动；旧的「薄包装类不单独成图」已被新裁定取代；「节点只留一条点开」的规则偏离已列入本版仲裁）；它指出的两处措辞残留已改。

## 按新条文重做

源码位置核查通过 377 个节点，有结论的席位 5 个，有事实错误，改完后四道机械检查全过。plan.md 本轮新加、改写两条，本页据此重做。① 通用基础层不画（第 78 行）：按旧条文补的 9 张图删掉——`ParallelEmbedding`、`Linear`、`ColumnParallelLinear`、`RowParallelLinear`、`PatchEmbed`、`vision.MLP`、`vision.RMSNorm`、`RMSNorm`、`ParallelHead`；保留 `ParallelEngramEmbedding`（有查表 + fp8 反量化，不只是把通用算子套在权重上）。删图后指向它们的点开全删（37 → 14，净删 23），折叠里「见点开的那张图」改成直接写算式；先前仲裁判「必须补这 4 张图」已被新条文取代。② 一个节点一步（第 81 行）：源码一行里套的类调用拆开、一步一节点，普通函数/方法不单独成节点，残差相加把 `+` 单独成节点。③ 折叠「说明」里不写公式（`build_page.py` 机械检查），全页无残留。④ 第一张图面板公式一条拆两行（用 `<br>`），只改 Transformer。拆节点口径：算子第一行写外层那一次类调用、内层实参省略成 `...`，被省略的实参由前置张量节点承接。

- 事实错误｜Compressor.forward｜前提写层 20 `compress_ratio`=1、只留 wkv 一路，但图只画了 ratio > 1 那一支，层 20 的 `if ratio == 1: return self.norm(self.wkv(x))` 无节点无说明｜model.py:462、485｜两支都画进同一张图
- 事实错误｜Transformer `layers`/`hl`/`lpm`｜循环回流没接上（`layers` 的残差输入只连 `hc` 与 `eh`，`lpm` 没连回 `layers`）｜model.py:1261-1267｜加 `hl → layers`、`lpm → layers`
- 事实错误｜拆分开的节点第一行实参｜`qr_op`、`wkv_win`、`kw`、`vlin` 写成层输入 `x`，实为内层调用的输出｜model.py:770、705、544；vision.py:119｜改成写外层调用
- 事实错误｜连线漏｜Transformer `venc`/`alen` 的 `n_vit_h`/`n_vit_w`；Aligner `vv` 的 `n_vit_h`/`n_vit_w`；vision.Attention `vwo` 的 `S`｜model.py:1226；vision.py:116、60｜补线并在线上写明
- 事实错误｜vision.Block `vbc`｜同一对 cos/sin 在 ViT 与 vision.Attention 是 `vcs`，这里换了 id｜vision.py:82｜改用 `vcs`
- 事实错误｜Block `mixes_a`/`mixes_f`、Gate `lg`、Engram `ewm`、NgramHashState `cmap`/`prod`/`mod`/`off`｜第一行当操作数直接出现的权重与缓冲没节点没连线｜model.py:982、989、811；engram.py:356、164、179、183、184｜画成节点并连线
- 写法｜`vview`、`vbad`/`vbad2`、`vlin`｜`vview` 生成式的迭代对象写成 `...`；`vbad`/`vbad2` 第二行把两操作数写成 `× 2`、折叠式子的 `x` 同时指本层输入与注意力输出；`vlin` 第一行用 `...` 代替实参｜vision.py:56、83-84；plan.md｜按表改（`vlin` 保留省略号，见处理）
- 措辞｜`shar` 源码列、同一 id 两图、前提与面板、描述与资料｜`shar` 源码列 `model.py:1166-1180` 是类定义、实例在 1180 建立；Transformer `shar`（整包）vs Indexer `shar`（只 candidates）两图同 id 指不同量；前提把 KV 缓存序列维笼统说成 max_seq_len 4096（窗口 KV 是 window_size 128）、层 20 压缩器口径与 Compressor 图不一、`head` 折叠对整段 h 而面板只取末位置、Attention 与 Indexer 两处 `softmax_scale` 指 512/128、`cf` 与 `fpos` 压缩组序号 j/g 不一、Transformer 面板「图像嵌入」的 img/img.start/types/i 未交代、config 表「Hyper-Connections」只说名字；描述「…24、28、32、36 层各自选」缺宾语、资料「模型卡」的 CSA2/DSpark 只并列堆砌｜model.py:630、555、753-757、539-543、462；plan.md｜逐条改

### 处理

- 删图与去引用（主改）：删 9 张基础层图，删后无指向它们的点开、无「见点开的那张图」残留（`ParallelEngramEmbedding` 保留）；这些节点折叠里的算式改成直接写。一个节点一步（拆分）：Transformer `venc`/`alen` 各成节点接张量 `vla`；Attention 拆 `wqa`+`qra`+`qr_op` 与 `wkv_proj`+`wkva`+`wkv_win`；Compressor 拆 `projk`+`kv0` 与 `projg`+`csc`；Indexer 拆 `wk_p`+`wka`+`kw`；vision.Block 拆 `norm1(x)` → attn / `norm2(x)` → mlp（`+` 由 `vbad`/`vbad2` 单独画）；vision.MLP 的 `vw2` 并入 `vmlp` 折叠；Aligner 拆 `aw1`+`alg`+`vlin`。
- 事实错误：Compressor 补层 20 那一支（`projk1`、`kv1`、`cnorm1`，`ax → projk1 → kv1 → cnorm1 → lat`）；Transformer 补 `hl → layers`、`lpm → layers`；拆分节点第一行实参改写外层调用；`venc`/`alen` 补 `n_vit_h`/`n_vit_w`、`vv` 补 `vout → vv`、`vwo` 补 `vax → vwo`、vision.Block `vbc` → `vcs`；Block 补 `afn`/`asc`/`abs`/`ffn_a`/`fsc`/`fbs`、Gate 补 `gwt`、Engram 补 `eqw`/`ekw`、NgramHashState 补 `tmap`/`mul`/`prm`/`ofs` 及连线（按 plan.md 第 88 行）；`shar` 源码列改 1180。
- 写作、措辞：`vview` 补全；`vbad`/`vbad2` 第二行写开、线上字改「注意力子层的输出」「前馈子层的输出」、折叠左侧写 `x_out`、右侧写 `x`（残差）与 `x_1`/`x_2`；前提区分窗口 KV 的序列维；`head` 折叠补末位置切片；两处 `softmax_scale` 写明取哪个 head_dim；Indexer `fpos` 统一 `j`；面板「图像嵌入」补符号、`mrg` 折叠补 `types` 来源；config 表「Hyper-Connections」补括注；Indexer 缓存 id `shar` → `shar_cand`；描述补宾语；资料模型卡改中文。
- 不改（写理由）：`vlin` 第一行保留 `self.w2(...)`（写成完整表达式会把两次类调用压回一个节点，违反第 81 行；与外层调用 + `...` 口径一致）；算子第一行 `...` 省略内层实参（第 83 行「不用省略号」是对张量第一行说的）。
- 最后一批改动核对：独立子代理核过拆分 8 处、层 20 支路、六组 hc_* 参数节点、NgramHashState 四个 buffer 节点，**通过**；指出并已改两处（非事实错误）——`tmap`/`mul` 源码行号互换（已对调）、本文件早先写「点开全删（27 条）」实为 23 条（已改）。四道机械检查全过（15 视图 379 节点，几何穿框 0）。

## 五名审查者复审后的改动

作者按任务要求做一批改动（7 条事实错误、删噪音节点、讲法、面板瘦身），改完派 5 席审一轮。源码位置核查通过 344 个节点（页面 346，15 图），有事实错误，改完后四道机械检查全过。

- 事实错误｜前提外分支、死端、重复连线、说明不符｜Attention 面板「频率基数」写了前提外的 $r=0$ 分支；Attention `qb` 死端、`core` 的 query 连了未加 RoPE 的 `q`；Indexer `kw` 实参写成 `lat`、`wka` 孤点；Compressor `ax → projk`/`ax → projg` 同一量连两遍；ViT `vcs_op` 说明写「前半段配高、后半段配宽」与源码相邻两维交替不符｜model.py:680-687、771-772、544、464-465；vision.py:11-14；config compress_ratios｜面板只写 compress_rope_theta 一支、`q → core` 改用 RoPE 后的那份、补 `lat → kw` 的 wk 投影、每算子只留一条连线、`vcs_op` 改「相邻两维一对、交替排列，偶维配高、奇维配宽」
- 事实错误｜Expert `ex → u1` 与 `g1`/`u1`/`w2`/`silu` 说明｜进 GEMM 的激活说成 fp4，实为 fp8（`linear()` 对 fp4 权重也走 `act_quant`，输出 fp8）｜model.py:186-197；kernel.py:98-124、486-499｜激活按 32 通道量化成 fp8，fp4 只是权重；四处说明同改
- 事实错误｜ParallelEngramEmbedding `pes`、Engram `ekw`｜`pes` 写「每 32 行一组一个 scale」，实为沿 256 通道维每 32 通道一个；`ekw` 文本写 `self.k_weight`、第一行却是 `self.q_weight`｜model.py:310、27、307、347-348；shapes.json｜`pes` 改「每 32 个通道一组一个（每行 8 个，8 = 256/32）」；`ekw` 第一行改 `k_weight`
- 事实错误｜Transformer `mrg`、`hl → layers`｜写回图像 span 的三个学到的嵌入没节点；`hl → layers` 线上字写「层 2–39（除层 1、14 外）」，层 1 不在 2–39 里｜model.py:1220-1222、1235-1237、1261-1267｜补节点并连线；改成「除层 14 外」
- 事实错误｜MoE `gate_op`、Indexer `iscm → topk`、`cmask`/`shar_cand`｜`gate_op` 第一行漏了压平实参；`iscm → topk` 缺直连、`csel`/`cmf` 串成一条链（实为 if/elif 互斥）；同一个 `shared_attn.candidates` 画成两个节点｜model.py:892、569-575、569-571｜把压平写进实参并在线上注明；补直连、标按层互斥；合并节点
- 事实错误｜Attention `widx`、Indexer `lens` 第二行｜输入写成 [B, T, 5120]，实际只吃标量｜model.py:720、564｜改成标量写法
- 事实错误｜全页符号撞名、Engram 说明、Compressor/Engram 面板｜一批符号在两张图里指两回事（多数为本轮换名新引入）：`Z`/`r`/`c`/`S`/`h`/`K`/`E`/`V`/`P`/`Y`/`I`/`C`/`J`/`Q` 等；Engram 说明 `w_e[i] * exp(s[i])` 缺 `w_e` 是 fp8 表、`s` 上转 fp32 再指数；Compressor 面板 `softmax(g)` 与折叠 `softmax(G_g)` 不一致、Engram 面板门控用 `c` 指 clamp_value（折叠已用 `j`）｜model.py 各处、317-320、475、362；kernel.py:49-54｜符号逐处去重换名；Engram 说明补全；面板与折叠统一
- 措辞｜符号、全称、前提与面板一组｜Indexer `imask` 说「不是同图 red 里的头维 h」而 red 已用 `i`；Indexer `qop`/`wproj` 同一 `H_i` 两个名字；CSA2 首次出现没给全称；`wrope` 的 nope 无解释；全页 `d` 三处三种含义；「这两件事互斥」「二者不是一回事」（前提 29、31 条）指代不清；「图像占位」与其余「图像 span」不一；同一字母 `r` 三种含义；vision.Attention vs Attention 的 `d_h` 指 64/512；Transformer `mrg` 公式用 START/END/NEWLINE 而说明说「IMAGE_START 等」｜dataflow.md；model.py:508-509、555、632、630、952-954、1235-1237；vision.py:50、115-118；model-card.md:49｜逐条改

### 处理

- 事实错误：`tmap` 第二行改 129280、第三行说明 99092 是压缩后 id 的取值上界（`cmap` 同改）；`vcs_op` 说明改相邻两维一对交替；Expert 激活 fp8；`pes` 改每 32 通道一组；`ekw` 第一行改 `k_weight`；新增 `imgp` 节点与 `imgp → mrg` 连线、`mrg` 折叠用完整类型名；Indexer 补 `iscm → topk` 直连、`csel`/`cmf` 说明按层互斥、`cmask` 与 `shar_cand` 合成 `cand`；Gate `gwt`/Engram `ekw` 改 bf16（构造未指定 dtype、取参考默认；`attn_sink`/`bias`/`hc_*` 的 fp32 不动）；`hl → layers` 改「层 2–39（除层 14 外）」；`gate_op` 第一行实参改 `None if image_mask is None else image_mask.flatten()`；`widx`/`imask` 输入列改标量写法，`lens`/`fpos` 节点删除、动作并进 `imask`/`krope`。
- 删噪音节点（逐节点 diff：删 43、新增 6 承接，净减 37：383 → 346）：vision.Attention、Aligner、Attention、Compressor、Indexer、MoE、Engram、NgramHashState 各并入相邻节点；Block 六个 hc_* 参数合成 `afn`/`asc`/`abs` 三个；新增 `imgp`/`q2`/`cand`/`nkey`/`nsplit`/`blocked`；保留不删 `act_quant`/`fp4_act_quant`/`eins`/`silu`/`topk`/`gather`/`Sinkhorn`/`apply_rotary_emb`；删除后上家直连下家、折叠公式并进相邻算子。
- 讲法与面板：`ids` 第三行写明 B（本次 1）、T（本次喂进的 token 数）；`lat`/`pool`/`crope` 写明 `r` 是 compress_ratio；`pool` 等节点的第二行与折叠说明写清动作；全页符号去重；缓存节点第二、三行写明第一维是容量（max_batch_size 默认 4）、窗口 KV 第二维是 128 槽的环、压缩/索引缓存第二维是 max_seq_len/r；`shar` 第三行写明跨层共享；八个 RoPE 公式改「末 $d_r$ 维乘一个旋转因子 $e^{i\varphi}$」；面板「配置」只留决定稀疏/结构的值（已在节点形状里出现的删去）、顶部 config.json 面板压成要点；`linear()` 的激活 fp8/fp4 分块量化写进各用 Linear 的节点折叠说明；两套 RoPE base 的分支写进 `fcall` 第三行与 Attention 面板，面板公式只留 compress_rope_theta 一支。
- 最后一批改动核对：本轮是任务要求改动 + 一轮审 + 改，尚未到轮数上限，故未另派仲裁核对子代理；改后四道机械检查全过（15 视图 346 节点，几何穿框 0）。
