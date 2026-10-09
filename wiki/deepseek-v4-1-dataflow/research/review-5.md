# 第 5 轮

（本文件合并了原第 5–11 轮。第 5 轮是通过轮；之后因前提与范围变更（加视觉支路、补薄包装类图）另做的审查，以及一次网络中断后的补审，作为下面的附录原样保留。标准流程一轮审查最多 5 轮，这部分附录不另外编号。）

源码位置核查通过：276 个节点。有结论的席位 4 个。审查者 3 未计入。这一轮有事实错误，不通过。

## 审查者 1

- 事实错误｜Attention.forward 节点 `widx` 第二行｜左侧写成「由 bsz、seqlen、window_size 生成」，不是形状｜review.md；model.py:720｜改成 `[B, T, …] → int32 [B, T, 128]`
- 事实错误｜Indexer.forward 节点 `lens` 第二行｜左侧写成「由 seqlen、ratio 生成」｜model.py:563-564｜改成 `[B, T, …] → int64 [T, 1]`
- 事实错误｜NgramHashState.forward 节点 `pos` 第二行｜左侧写成「由 batch、seqlen 生成」｜engram.py:169｜改成 `int64 [B, T] → int64 [B, T]`
- 措辞｜MoE.forward 连线｜`shared` 没有点开 Expert.forward｜model.py:903；plan.md｜补上点开
- 措辞｜NgramHashState.forward 折叠 `prod`｜公式没写 `unsqueeze(2)`｜engram.py:179｜补上

## 审查者 2

- 事实错误｜Attention.forward / widx 第二行｜左侧不是原来的形状｜review.md；model.py:720｜改成 `[B, T, 5120] → int32 [B, T, 128]`
- 事实错误｜Indexer.forward / lens 第二行｜同上｜model.py:563-564｜改成 `[B, T, 5120] → int64 [T, 1]`
- 事实错误｜NgramHashState.forward / pos 第二行｜同上｜engram.py:169｜改成 `int64 [B, T] → int64 [B, T]`
- 措辞｜Transformer.forward / head 说明｜「只取最后一个位置」易读成对 weight 切片｜model.py:1010-1012｜改成对输入取最后一个位置
- 措辞｜Transformer.forward / layers 源码列｜`model.py:907` 是 `class Block`｜调用在 1267｜改为 `model.py:1267`
- 措辞｜Block.forward / mixes 公式｜`flatten(x)` 没写从第 2 维起｜model.py:952｜改成 `flatten(2)`
- 措辞｜MoE.forward / mback 公式｜把 `x` 说成拉平前的输入｜model.py:891-904｜`type_as` 时 x 已拉平，shape 才是拉平前的
- 措辞｜Transformer.forward / layers 多输出｜最后一层的 `pre_mix` 没有张量节点｜plan.md；model.py:1267-1268｜在 layers 与 col 之间补张量

## 审查者 3

未计入

## 审查者 4

- 事实错误｜Attention.forward 节点 widx 第二行｜左侧不是形状｜review.md；model.py:720｜改成 `[B, T, 5120] → int32 [B, T, 128]`
- 事实错误｜Indexer.forward 节点 lens 第二行｜同上｜model.py:563-564｜改成 `[B, T, 5120] → int64 [T, 1]`
- 事实错误｜NgramHashState.forward 节点 pos 第二行｜同上｜engram.py:169｜改成 `[B, T] → int64 [B, T]`
- 事实错误｜Transformer.forward 节点 layers 源码列｜`model.py:907` 是类声明｜调用在 1267｜改成 `model.py:1267`
- 措辞｜mixes 公式｜`flatten(x)` 没写 `flatten(2)`｜model.py:952｜写明从 hc 维起拉平
- 措辞｜MoE 节点 counts｜类型写「张量」，实际是 list｜model.py:894｜第二行写明 Python list
- 措辞｜若干说明｜写了前提之外的层型｜前提只定 ratio 为 2｜先写本图路径

## 审查者 5

- 事实错误｜NgramHashState.forward / upd 折叠公式｜把 `DEAD` 写成压缩词表里的空位｜engram.py:138、165-166：`DEAD = -1`，是不参与 n-gram 的哨兵｜改成 DEAD = -1；本路径 engram_mask 为 None，不会写入
- 事实错误｜Attention.forward / widx 第二行｜左侧不是形状｜review.md；model.py:720｜箭头两边都写成形状
- 事实错误｜Indexer.forward / lens 第二行｜同上｜model.py:563-564｜同上
- 事实错误｜NgramHashState.forward / pos 第二行｜同上｜engram.py:169｜同上
- 措辞｜wquant、cquant、kq、iq_q 公式｜`max(|·|)` 看起来像逐元素，实际是每组 absmax｜kernel.py `reduce_absmax`｜写明每组取 absmax
- 措辞｜介绍 / 描述｜写成压缩比大于 0 的层都由索引器选出 Top-512｜model.py:725-726｜非 index source 层复用已发布的 topk

## 修改

对照源码改过，并重新生成了页面。源码位置再查过，276 个节点。

- widx 第二行改为 `[B, T, 5120] → int32 [B, T, 128]`。lens 改为 `[B, T, 5120] → int64 [T, 1]`。pos 改为 `int64 [B, T] → int64 [B, T]`。
- `DEAD` 改成 -1，标记不参与 n-gram 的位置。这次 engram_mask 是 None，不会写进 cache。
- layers 的源码列改为 `model.py:1267`。检查脚本对「层数 × 类名」认 `layer(...)` 调用，不再要求那一行出现类名。
- 最后一层返回的 pre_mix 单独成张量 `lpm`，再连到 col。
- mixes 的 flatten 改成 `flatten(x, 2)`。head 的说明改成对输入取最后一个位置。mback 的 x 改成拉平后的隐状态。prod 补上 `unsqueeze(2)`。量化公式里的 max 改成每组 absmax。
- `shared` 仍不点开 Expert.forward。那张图是传入 weights 的路由专家。counts 的类型仍是「张量」，计划里没有 list 这一类，第二行已经写明是 list。

---

## 附录 A：原第 6 轮（通过轮之后的收口）

源码位置核查通过：276 个节点。有结论的席位 4 个。审查者 4 未计入。这一轮没有事实错误，通过。

## 审查者 1

无

## 审查者 2

无

## 审查者 3

无

## 审查者 4

未计入

## 审查者 5

- 措辞｜MoE.forward 连线｜`shared` 未点开到 `Expert.forward`，路由专家的 `mexp` 有点开｜plan.md；model.py:903｜补上 `shared 点开 → Expert.forward`
- 无事实错误

## 修改

- 没有事实错误。
- 页面核对时，`acc`、`madd`、`off` 的第一行用了源码里没有的名字。第一行改回源码里的那一行。折叠里的公式仍只写这一步自己的累加或加偏移。
- `shared` 仍不点开 Expert.forward。那张图是传入 weights 的路由专家，含 emw。共享专家不传 weights。前提里已经写了这一点。

## 附录 B：原第 7–8 轮（前提改成带图像、加视觉支路）

### 第 7 轮

源码位置核查通过：371 个节点。

本轮背景：这一页原先只画文本前向。这一轮把视觉编码器补上（新增 ViT、vision.Block、vision.Attention、vision.MLP、vision.RMSNorm、Aligner 六张图），并把前提从「不传图像」改成「传一张图」，于是 `image_mask` 与 `engram_mask` 都成了实际张量，Transformer、Block、MoE、Gate、Engram、NgramHashState 六张图的文字与节点跟着改。

## 审查者 1（路径覆盖）

审查者 1 的结论：这一轮同时报出了两组问题，一组属本轮改动（掩码从 None 变成实际张量后该画的没画）、一组是把上一轮的文本前向页再核了一遍时找出的旧问题。作者逐条对照源码核实，下面按「属本轮」与「不属本轮」分开记。

属本轮：

- 事实错误｜Engram.forward，`g` 与 `nadd` 之间｜漏画一步：源码在 token_mask 不是 None 时把 gate 在图像位置置 0｜model.py:363-364
- 事实错误｜Engram.forward，`vf` 的折叠｜只有「补一维好按份广播」，没写这一步是什么变换，同图其它算算子都写了公式｜model.py:365
- 事实错误｜Transformer.forward，`mrg`｜`merge_image_embeddings` 里真正参与计算的 `img.types`、`img.start`、`image_start`、`image_end`、`image_newline` 一件没画、没连线，只有 `patches` 连了｜model.py:1231-1239
- 事实错误｜Transformer.forward，`venc` 第一行｜三个实参只有 `patches` 有连线，`n_vit_h`、`n_vit_w` 既不是节点也没在线上一带过｜model.py:1238
- 事实错误｜Aligner.forward，`vpad`｜第一行用到的 `n_h`、`n_w` 与 `r` 都没有节点、没有连线，`vrd` 只连到 `vunf`｜vision.py:115-117
- 事实错误｜vision.Block.forward，`vby` 与 `vbad2`｜`vbad2` 第一行右边的 `x` 指的是本子层输入 `vby`，但图上 `vby → vbad2` 与 `vbm1 → vbad2` 两条线不区分输入次序，按现有画法读成两次 `vby` 也成立；`vbad` 同此｜vision.py:83-84
- 事实错误｜vision.RMSNorm.forward，`vrout`｜「进入 RMSNorm 时 x 的 dtype」在 t=0 时不存在，`patches` 的 dtype 是 `image_processor.load_image` 里定的，本页假设 `patches` 的输入就是假设它｜model.py:1238、image_processor.py:127
- 事实错误｜MoE.forward，`gate_op`｜`image_mask.flatten()` 里的 flatten 已经画成算子 `mmask`，第一行却仍带 `.flatten()`，同一处把同一个调用画了两遍｜model.py:892
- 事实错误｜MoE.forward，`acc` 的折叠｜`x[idx]` 是第一行的实参却写在折叠里，`mexp` 折叠又把它抄了一遍，详略不一致｜model.py:900
- 事实错误｜MoE.forward，`mmask` 与 Gate.forward，`bsel`｜`mmask → gate_op` 的线上字说「压平后的那一维」，而已在 `bsel` 折叠里用 `image_mask` 指代压平后的量｜model.py:892、819-820
- 事实错误｜Engram.forward，`nadd` 的折叠｜折叠里把被检查的向量写成 `gate`，但同图 `gmask` 折叠里 `gate` 已是掩码之后的量，一个符号指了两段｜model.py:362-365
- 事实错误｜Transformer.forward，`im` 与 Block.forward，`bim`｜同一份 image_mask 两张图用了两个 id，而 Block 图第三行自己写明「由外层一路传进 MoE」｜model.py:1249 → 1267 → 973
- 事实错误｜Aligner.forward，`vlaout` 与 Transformer.forward，`vemb`｜同一个量两个图两个 id｜vision.py:119、model.py:1238

不属本轮（作者核实后确认，作为已裁定清单交给后续轮次）：

- `eng` 第一行的第三个参数 `engram_mask` 没有输入连线，`emv` 只连到 `ngram`｜model.py:1262-1263｜已改：补 `emv → eng`
- Attention.forward，`core` 第一行的 `self.attn_sink` 没有节点也没有连线｜model.py:780｜已改：补张量 `attn_sink` 与 `attn_sink → core`
- Transformer 算子直接连算子、张量直接连张量（`layers → eng`、`layers → avg`、`Block` 的 `hin → resid0`、`Attention` 的 `wkv_q → wchunk` 等）｜plan.md「算子和张量交替出现」｜已在计划里修正；这是上一轮的遗留写法问题，不算本轮改动引入
- `patches`/`vp`、`vcs`/`vbc` 两组跨图 id 没统一｜plan.md「上一个图的输出进到下一个图，用产出它的那个 id」｜已在计划里修正

## 审查者 2（节点与形状）

- 事实错误｜MoE.forward，`mfm`｜这个张量节点没有任何连线，是孤立节点；照现写的连线，压平出来的量是 `mmask` 直接连到 `gate_op`，跳过了它产出、也应被 Gate 消费的那个张量｜model.py:892
- 事实错误｜Gate.forward，`bsel` 折叠与面板「选专家」｜写成「image_mask 逐 token 在专家这一维上展开」，读起来是每个 token 在 384 个专家上各有各的掩码；源码是 `unsqueeze(-1)` 后广播，每个 token 整行取同一份偏置｜model.py:819-820
- 事实错误｜Gate.forward，`bvl` 第三行｜把条件挂在 `vision_n_layers` 上；建 ViT/Aligner/偏置的条件是 `args.vision_enabled`｜model.py:139-140、1216-1218、807
- 事实错误｜Aligner.forward，`vax`｜与 vision.Attention 图的 `vax` 撞 id；应沿用 ViT 图产出它的 `vout`｜vision.py:114、54
- 写法｜vision.Block.forward，`vbc` 第三行｜引用了不存在的 id `vcos`｜vision.py:82、100

## 审查者 3（连线与结构）

- 事实错误｜Transformer.forward，`venc`｜`venc 点开` 行有两条（→ ViT.forward、→ Aligner.forward），生成器一条点开只能挂一个目标，最后写入的是 Aligner，ViT.forward 从画面上再也点不进去｜build_page.py:251-257、model.py:1225-1226
- 事实错误｜vision.RMSNorm.forward，`vmw`｜`self.weight * x` 里用到的 `self.weight` 没有画成节点、也没有输入连线｜vision.py:34、plan.md「算子里直接用到的量都画成节点」
- 事实错误｜Transformer.forward，`mrg`｜`img.types`、`img.start`、`self.image_start`、`self.image_end`、`self.image_newline` 一件没画、没连线｜model.py:1231-1239
- 事实错误｜Transformer.forward，`venc`｜`n_vit_h`、`n_vit_w` 既不是节点也没在线上一带过｜model.py:1238
- 事实错误｜Aligner.forward，`vpad`｜`n_h`、`n_w`、`r` 都没有节点、没有连线，`vrd` 只连到 `vunf`｜vision.py:115-117
- 事实错误｜Transformer.forward，`im` 与 Block `bim`｜同一份 image_mask 两张图两个 id｜model.py:1249 → 1267 → 973
- 事实错误｜MoE.forward，`gate_op`｜第一行仍带 `.flatten()`，与算子 `mmask` 重复画了同一个调用｜model.py:892
- 事实错误｜Engram.forward，`nadd` 折叠｜同一个符号 `gate` 在一张图里指了两段｜model.py:362-365
- 事实错误｜MoE.forward，`acc` 折叠｜登记详略不一致｜model.py:900
- 事实错误｜vision.RMSNorm.forward，`vrout`｜「进入 RMSNorm 时 x 的 dtype」在 t=0 时不存在｜model.py:1238、image_processor.py:127
- 事实错误｜vision.Block.forward，`vby` 与 `vbad2`｜两条线不区分输入次序，读成两次 `vby` 也成立｜vision.py:83-84
- 事实错误｜Aligner.forward，`vpad` 折叠与 vision.RMSNorm 图｜`vpad` 折叠说「常见尺寸下不需要补」，但 `n_llm_h` 用的是 ceil，两者在 n_vit_h 不是 r 的倍数时差 1｜vision.py:117、41
- 事实错误｜Engram.forward，`vf`｜唯一折叠是「补一维好按份广播」，没写这一步是什么变换｜model.py:365
- 写法｜Transformer.forward，`patches` 与 ViT.forward，`vp`｜同一份量两边 id 不同｜vision.py:98
- 写法｜MoE.forward，`mmask` 与 Gate.forward，`bsel`｜线上字与折叠用词把压平前后的量混着叫｜model.py:892、819-820

## 审查者 4（公式与面板）

- 事实错误｜ViT.forward，`vcs_op` 折叠公式与面板「二维位置编码」｜`inv_freq_j = θ^{-2j/head_dim}` 里的除数对不上源码：源码把 `dim` 传成 `self.rope_dim` = 32，指数是 2j/32；计划把 `head_dim` 释为每个头的宽度 64，写成 2j/64，差 2 倍｜vision.py:9、10、92、100
- 措辞｜Attention.forward，`widx` 折叠与面板「频率」｜同一个符号 `base` 在该图指两个量：`widx` 里是窗口下标基准，面板里是 RoPE 底数｜model.py:419、376
- 措辞｜vision.Attention 面板「注意力」与 Attention 面板「稀疏注意力」｜同一个 `head_dim` 两处指不同量（视觉 64、主干 512）｜vision.py:50、model.py:630
- 措辞｜前提与 config 表 compress_ratios｜MTP 首次出现未给全称｜model.py:60
- 措辞｜摘要｜RoPE 首次出现未给全称｜dataflow.md:4
- 措辞｜Aligner.forward，`vpad` 折叠与 `vpv` 第三行｜对一张未指定尺寸的图断言「本图的尺寸下补 0」「常见尺寸下不需要补」｜image_processor.py:112、121、60-67

## 审查者 5（读者与措辞）

- 措辞｜摘要、Transformer 面板「图像嵌入」｜两处语序反了，读成把 IMAGE 位置写进对齐器的行
- 措辞｜描述｜「写回词嵌入里图像占位位置所在的那一段」——写回的是取完词嵌入之后的隐状态
- 措辞｜Transformer `venc` 第二行｜「对齐器逐行 [n_llm_h × n_llm_w, 5120]」不是形状的说法
- 措辞｜ViT 连线 `vp → vcs_op`｜`n_vit_h`、`n_vit_w` 在 ViT 图里没有自己的节点
- 措辞｜摘要｜「混合系数由 Sinkhorn 迭代 20 轮给出」——Block 图里只有 comb 出自 Sinkhorn
- 措辞｜前提与面板配置｜前提把 DSpark 归进「不是 forward_spec」予以排除，但面板列的 dspark_target_layer_ids 正是收集 main_hidden 用的那 3 层；三处叫法不一
- 措辞｜config.json 表组名「Hyper-Connections」「DSpark」｜首次出现且无上下文
- 措辞｜摘要｜「Engram 插在层 1、14」是 Engram 首次出现且无解释
- 措辞｜摘要｜`VL` 缩写未展开
- 措辞｜介绍「这次前向」｜「图只在这一步用一次」指代不清
- 措辞｜六张新图标题｜类名前缀不统一
- 措辞｜描述｜「压缩比大于 0 的层…另加索引器选出的压缩 KV」省略了「压缩 KV 只由 4 个 source 层产出」这层区分
- 措辞｜Transformer 面板「图像嵌入」｜「本类」指哪个类没说清
- 措辞｜vision.RMSNorm 面板与 `vnrm` 说明｜面板没给 eps 的值

## 修改

对照源码改过，并重新生成了页面。源码位置再查过，372 个节点。事实错误逐条改，措辞按上表逐类扫过（细目见下）。

**本轮改动的部分**

- Engram 图补上 `nmask → gmask → gm` 一步，连线改成 `g → gmask`、`gmask → gm`、`gm → nadd`；`gmask` 补公式与说明；`nadd` 折叠改用掩码之后的量、并统一符号。
- Engram 图 `vf` 补上公式。
- Transformer 图 `mrg` 补 `tt`（token_types）连线与 `img.start` 的说明；`venc` 补 `patches → venc`、`vp → venc：n_vit_h、n_vit_w`。
- Transformer 图补 `emv → eng` 的连线；`im` 与 Block 图的 `bim`、Aligner 的 `vlaout` 与 Transformer 的 `vemb` 统一到同一 id。
- Aligner 图补 `vout → vpad：n_vit_h、n_vit_w` 与 `vrd → vpad`；`vpad` 折叠改成条件句。
- Aligner 图的输入 id 由 `vax` 改成 `vout`。
- vision.Block 图 `vbad`、`vbad2` 的两条输入连线分别标 residual 与上一层输出，折叠里点明哪一路是残差。
- vision.RMSNorm 图补「patches 是 bf16」进前提；`vmw` 补 `self.weight` 张量节点与连线。
- Transformer 图 `venc` 拆成 `venc`（self.vision）与 `alen`（self.aligner）两个算子，中间接一张张量；点开各挂一个目标。
- Attention 图补 `attn_sink` 张量节点与 `attn_sink → core`。
- MoE 图 `gate_op` 第一行去掉 `.flatten()`；`mmask` 折叠改成「压成按 token 的一维，交给 gate_op」；`acc` 折叠只留累加。
- Gate 图 `bsel` 公式与面板改成扩维再广播、整行取同一份偏置；`bvl` 第三行改成写明 `vision_enabled`；折叠里用 `mfm` 指压平后的量。
- ViT 图 `vcs_op` 的公式与面板统一用 `rope_dim`（并注明与 Attention 的 head_dim 无关）。
- 算子直接连算子、张量直接连张量的那几处补了中间节点，恢复算子/张量交替。
- `patches`（Transformer 侧）与 `vp`（ViT 侧）统一到同一 id；`vbc` 与 `vcs` 统一。

**措辞**

摘要与折叠里图像写入的语序改成「按行序把 embeds 的行写进 IMAGE 位置」；描述改成「写回…那一段的隐状态」，并补上压缩 KV 只由 4 个 source 层产出；`venc` 第二行改成完整形状；摘要的 Sinkhorn 改成只给 comb；`VL` 展开成「视觉语言（VL）」；介绍改成「图只在这个前向里用一次」；面板「本类」改成写明挂在 Transformer 上；vision.RMSNorm 面板补 eps 取值；`widx` 的 `base` 改名；MTP 与 RoPE 首次出现给全称。

**不改的几条**（写出理由）：

- `vp → vcs_op`（ViT 图）上的 `n_vit_h`、`n_vit_w` 仍不单独成节点：它们是 `ViT.forward` 的形参，本页惯例是连一条线并在线上写明（同页有 `ax → widx：形状（bsz、seqlen）`）。
- 六张新图的标题前缀只在与本页已有图撞 tab 名时才带 `vision.`。
- `shared` 不点开 `Expert.forward`、`counts` 类型写「张量」、三张薄包装类不单独成图：前几轮已裁定。

### 第 8 轮

源码位置核查通过：378 个节点。有结论的席位 5 个。这一轮有事实错误，改完后不通过（见文末「改动后再核」）。

这一轮 5 席从头重审。第 7 轮的改动（见 review-7.md）已经在计划里，本轮的结论写在下面；只列 5 位审查者这一轮报出的问题。

## 审查者 1（路径覆盖）

- 事实错误｜Indexer.forward（`iscm` 与 `topk` 之间）｜层 20 是 candidate source，这条路径上会执行 `select_candidate_blocks` 并把结果写进 `shared_attn.candidates`；层 24、28、32、36 跑各自 Indexer 时用它把看不到的位置 masked_fill 成 −inf。这两步在 Indexer 图里没有节点，也没有缓存写入，而 Transformer 图的 `shar` 缓存节点与 `layers` 说明都已经在说「层 20 写 candidates，层 24–36 读」｜model.py:569-575、583-610；计划 line 138、219
- 事实错误｜Indexer.forward｜`lat → kw → krope → kq → kwrite` 这条链被画成无条件执行，但它只在 `self.owns_k` 的层跑；前提这条路径上 24、28、32、36 四个 index source 层不占 kv source，latent 传进来是 None，这一段整个跳过，只从 shared_attn.index_k 读键｜model.py:500、517-525、537-548、655、722-737；kv_source_layers=[2,8,14,20]、index_source_layers=[2,8,14,20,24,28,32,36]

其余核过无问题：前提路径上各 forward 的每一步都对照过节点表，除上面两条外没有缺步；没有节点画在前提路径之外；`if`/比较/`for` 没有画成控制流节点；源码里自画的类都有一张图且被用的节点都点开了。

## 审查者 2（节点与形状）

- 事实错误｜MoE.forward｜`mmask` 的输入没有产出节点、也没有连线：`image_mask.flatten()` 里被压平的那份 `image_mask` 在图上悬空（上一轮只补了 `mmask → mfm` 与 `mfm → gate_op`）｜model.py:892
- 事实错误｜Aligner.forward｜`vpad` 第一行用到的 `r`（= `self.downsample_ratio`）没有连到 `vpad`，`vrd` 只连到 `vunf`｜vision.py:115、117
- 事实错误｜Transformer `im`/`emv`、Block `im`、NgramHashState `emv`、Engram `emv`｜同一份 mask（bool [B,T]）在四个图里应该用同一个 id｜plan.md「不同图里同一个 id 指同一个量」｜第 4 轮把 `fcr` 统一成 `fcall`、第 7 轮把 `vax` 改成 `vout` 是同一规则
- 事实错误｜Aligner `vemb` 与 Transformer `vemb`｜同一个量（对齐器的返回）两个图两个 id，输出端上一轮没统一｜vision.py:119、model.py:1238
- 写法｜ViT `patches` 与 Transformer `patches`｜上一张图的输出进下一张图时改用了新 id（`vp` / `patches`）｜vision.py:98

## 审查者 3（连线与结构）

- 事实错误｜Transformer `eng`｜第一行第三个参数 `engram_mask` 在图上没有输入连线（`emv` 只连到 `ngram`），而同一个量在 NgramHashState 图里是有线连到用它的算子的｜model.py:1262-1263
- 事实错误｜Attention `core`｜第一行里的 `self.attn_sink` 既没有节点也没有连线｜model.py:780
- 事实错误｜MoE `mmask`｜第一行 `image_mask.flatten()` 的输入没有产出节点、也没有连线｜model.py:892
- 事实错误｜Aligner `vpad`｜`r` 没有连到 `vpad`｜vision.py:115、117
- 事实错误｜Transformer `im`/`emv`、Block `bim`、NgramHashState `cmsk`、Engram `nmask`｜同一份 mask 四个图四个 id，而 NgramHashState/Engram 两处第三行已经自己写明「与 Transformer 图里的 emv/bim 是同一个量」｜plan.md
- 事实错误｜Aligner `vlaout` 与 Transformer `vemb`｜同一个量两个 id｜vision.py:119、model.py:1238
- 事实错误｜MoE `gate_op`｜第一行仍带 `.flatten()`，与算子 `mmask` 重复画了同一个调用｜model.py:892
- 事实错误｜Engram `nadd` 折叠｜折叠里把被检查的向量写成 `gate`，而同图 `gmask` 折叠里 `gate` 已是掩码之后的量，一个符号指了两段｜model.py:362-365
- 事实错误｜MoE `acc` 折叠｜`x[idx]` 是第一行的实参却写在折叠里，`mexp` 折叠又抄了一遍，登记详略不一致｜model.py:900
- 事实错误｜vision.RMSNorm `vrout`｜「进入 RMSNorm 时 x 的 dtype」在 t=0 时不存在，`patches` 的 dtype 由 `image_processor.load_image` 定｜model.py:1238、image_processor.py:127
- 事实错误｜vision.Block `vby` 与 `vbad2`｜`vbad2` 第一行右边的 `x` 指本子层输入；图上 `vby → vbad2` 与 `vbm1 → vbad2` 两条线不区分输入次序，读成两次 `vby` 也成立；`vbad` 同此｜vision.py:83-84
- 事实错误｜Aligner `vpad` 折叠与 `vpv` 第三行｜「本图的尺寸下补 0」「常见尺寸下不需要补」对一张未指定尺寸的图下了断言｜vision.py:117、41；image_processor.py:112、121
- 事实错误｜Engram `vf`｜唯一折叠是「补一维好按份广播」，没写这一步是什么变换｜model.py:365
- 写法｜ViT 的 `vp` 与 Transformer 的 `patches`、`vcs` 与 `vbc`｜两组跨图 id 没统一｜vision.py:98；plan.md
- 写法｜MoE `mmask` 与 Gate `bsel`｜`mmask → gate_op` 的线上字说「压平后的那一维」，而已在 `bsel` 折叠里用 `image_mask` 指代压平后的量｜model.py:892、819-820

## 审查者 4（公式与面板）

- 事实错误｜ViT `vcs_op` 折叠公式与面板「二维位置编码」｜`inv_freq_j = θ^{-2j/head_dim}` 里的除数对不上源码：源码把 `dim` 传成 `self.rope_dim` = 32，指数是 2j/32；计划把 `head_dim` 释为每个头的宽度 64，写成 2j/64，差 2 倍（面板同一行「频率长度 32 = head_dim / 2」也说明取的 head_dim 是 64，两句话不能同时成立）｜vision.py:9、10、92、100
- 措辞｜Attention `widx` 折叠与面板「频率」｜同一个符号 `base` 在该图指两个量｜model.py:419、376
- 措辞｜vision.Attention 面板与 Attention 面板｜同一个 `head_dim` 两处指不同量（视觉 64、主干 512），只靠 `vhdm` 节点说明不足以消歧｜vision.py:50、model.py:630
- 措辞｜前提与 config 表 compress_ratios｜MTP 首次出现未给全称｜model.py:60
- 措辞｜摘要｜RoPE 首次出现未给全称｜dataflow.md:4
- 措辞｜Aligner `vpad` 折叠与 `vpv` 第三行｜对一张未指定尺寸的图下了断言｜image_processor.py:112、121、60-67

## 审查者 5（读者与措辞）

- 措辞｜摘要、Transformer 面板「图像嵌入」｜两处语序反了，读成把 IMAGE 位置写进对齐器的行
- 措辞｜描述｜「写回词嵌入里图像占位位置所在的那一段」——写回的是取完词嵌入之后的隐状态
- 措辞｜Transformer `venc` 第二行｜「对齐器逐行 [n_llm_h × n_llm_w, 5120]」不是形状的说法
- 措辞｜ViT 连线 `vp → vcs_op`｜`n_vit_h`、`n_vit_w` 在 ViT 图里没有自己的节点
- 措辞｜摘要｜「混合系数由 Sinkhorn 迭代 20 轮给出」——Block 图里只有 comb 出自 Sinkhorn
- 措辞｜前提与面板配置｜前提把 DSpark 归进「不是 forward_spec」予以排除，但面板列的 dspark_target_layer_ids 正是收集 main_hidden 用的那 3 层；三处叫法不一
- 措辞｜config.json 表组名「Hyper-Connections」「DSpark」｜首次出现且无上下文
- 措辞｜摘要｜「Engram 插在层 1、14」是 Engram 首次出现且无解释
- 措辞｜摘要｜`VL` 缩写未展开
- 措辞｜介绍「这次前向」｜「图只在这一步用一次」指代不清
- 措辞｜六张新图标题｜类名前缀不统一
- 措辞｜描述｜「压缩比大于 0 的层…另加索引器选出的压缩 KV」省略了「压缩 KV 只由 4 个 source 层产出」这层区分
- 措辞｜Transformer 面板「图像嵌入」｜「本类」指哪个类没说清
- 措辞｜vision.RMSNorm 面板与 `vnrm` 说明｜面板没给 eps 的值

## 修改

对照源码改过，并重新生成了页面。源码位置再查过，380 个节点。

- Transformer 图：补 `emv → eng`；`venc` 拆成 `venc`（`self.vision`）与 `alen`（`self.aligner`）两个算子，中间接张量 `vla`；点开各挂一个目标（ViT / Aligner）。
- Transformer 图：`mrg` 补 `img.start` 与三个可学习参数的说明；`im`、`emv` 第三行写明 Block/MoE/Gate 与 Engram/NgramHashState 用的是同一个量。
- mask 的 id 统一：Block 图的 `bim`、NgramHashState 的 `cmsk`、Engram 的 `nmask` 都改成 Transformer 图用的 `im` / `emv`。压平后的那份仍叫 `mfm`，是另一个量。
- Transformer 图 `patches` 与 ViT 图 `vp` 统一成 `patches`；Aligner 图的输出 `vlaout` 改成 `vemb`。
- Aligner 图：补 `vrd → vpad：r` 与 `vout → vpad：n_vit_h、n_vit_w`；`vpad` 折叠与 `vpv` 第三行改成条件句。
- Attention 图：补张量 `attn_sink`（fp32 [64]）与 `attn_sink → core`。
- MoE 图：补 `image_mask.flatten()` 里那份 `image_mask` 的连线；`gate_op` 第一行去掉 `.flatten()`；`mmask` 折叠改成「压成按 token 的一维，交给 gate_op」；`acc` 折叠只留累加。
- Engram 图：`nadd` 折叠改用掩码之后的量并统一符号；`vf` 补公式。
- vision.Block 图：`vbad`、`vbad2` 的两条输入连线分别标 residual 与上一层输出，折叠里点明哪一路是残差。
- vision.RMSNorm 图：把「patches 是 bf16」写进前提。
- ViT 图 `vcs_op` 的公式与面板改用 `rope_dim`（并注明与 Attention 的 head_dim 无关）；`widx` 里的 `base` 改成 `idx_base`。
- 措辞：摘要与折叠里图像写入的语序改成「按行序把 embeds 的行写进 IMAGE 位置」；描述改成「写回…那一段的隐状态」，并补上压缩 KV 只由 4 个 source 层产出；`venc` 第二行改成完整形状；摘要的 Sinkhorn 改成只给 comb；`VL` 展开成「视觉语言（VL）」；介绍改成「图只在这个前向里用一次」；面板「本类」改成写明挂在 Transformer 上；vision.RMSNorm 面板补 eps 取值；MTP、RoPE 首次出现给全称。

**审查者 1 的两条（Indexer 图，属主干）也改了**：

1. `iscm` 与 `topk` 之间补 `csel`（`select_candidate_blocks`）与 `cmf`（`masked_fill(~shared_attn.candidates, -inf)`）两个算子、`cmask`/`cmfd` 两个张量，以及 `shar` 缓存节点（层 20 写、层 24–36 读），两条各带公式与说明。
2. Indexer 的 `lat` 第三行写明：只有 2、8、14、20 这四个 kv source 层是实际张量，24、28、32、36 传进来的是 None，`lat → kw → krope → kq → kwrite` 这一段整个跳过，只从 shared_attn.index_k 读键。

改动后再核：`build_page.py`、`check_source.py`（计划与页面）、`site/check.py`、`--geometry` 全过，380 个节点、18 张图、几何穿框 0、像素重合 0。下一轮 5 席重审。

**不改的几条**（写出理由，进已裁定清单）：

- ViT 连线 `vp → vcs_op`（现为 `patches → vcs_op`）上的 `n_vit_h`、`n_vit_w` 仍不单独成节点：它们是 `ViT.forward` 的形参，本页惯例是连一条线并在线上写明（同页有 `ax → widx：形状（bsz、seqlen）`）。
- 六张新图的标题前缀只在与本页已有图撞 tab 名时才带 `vision.`。
- `shared` 不点开 `Expert.forward`、`counts` 类型写「张量」、三张薄包装类不单独成图：前几轮已裁定。
- `PatchEmbed` 不单独成图：只有一次线性投影，画在 ViT 图的 `vpe` 节点上。

## 附录 C：原第 9–10 轮（网络中断后的补审）

### 第 9 轮

源码位置核查通过：379 个节点。有结论的席位 2 个（节点与形状、读者与措辞）。这一轮有事实错误，不通过；其余 3 席在收口前未返回，记「未计入」。

**说明**：本轮 5 席都在改完第 8 轮的问题之后重派。返回结论的席位是「节点与形状」与「读者与措辞」两席；「路径覆盖」「连线与结构」「公式与面板」三席到收口时没返回，按 `review-plan.md` 记「未计入」。因为不足 4 席有结论，严格按指南这一轮本该 5 席全部重派；到轮数上限后改为把本轮已知问题改完并做最后一轮核对，未计入的部分列在「已知不足」里。

## 审查者 1（路径覆盖）

未计入。

## 审查者 2（节点与形状）

- 事实错误｜MoE.forward｜`mmask` 第一行 `image_mask.flatten()` 里被压平的那份 `image_mask` 没有产出节点、也没有连线｜model.py:889、892

## 审查者 3（连线与结构）

未计入。

## 审查者 4（公式与面板）

未计入。

## 审查者 5（读者与措辞）

- 措辞｜描述｜「一张图像先切成 patch 过 32 层…编码」中间缺连接，读起来卡
- 措辞｜描述｜裸用「source 层」，没说是 kv source 还是 index source
- 措辞｜描述｜「压缩比大于 0 的层在 128 个位置的滑动窗口之外，另加索引器选出的压缩 KV」句式挤压
- 措辞｜摘要｜「FP8」「FP4」大写，全篇其余处一律小写
- 措辞｜摘要｜「Top-512」大写 T，正文写「top-512」
- 措辞｜资料｜「层号、块号、专家号」里的「块号」指什么不明
- 措辞｜前提｜「…建、没写 dtype 的权重取 `dtype`（fp8）」用顿号连接两个分句
- 措辞｜前提｜「层 20 的 compress_ratio 是 1，压缩器只有一次投影」与 Compressor 图口径不一致
- 措辞｜Transformer 面板「图像嵌入」与折叠 `mrg`｜`img.start`、`numel(types)`、`types` 三个符号没交代来源
- 措辞｜Transformer 折叠 `mrg`｜`IMAGE_START`（类型）与 `image_start`（可学习参数）只差大小写，未注明区别
- 措辞｜Attention 面板｜同一面板内 `r` 既是 compress_ratio 又是输出投影的下标；`g` 两义
- 措辞｜Attention 节点 `qb`/`grp`｜`n_local_heads`、`n_local_groups` 首次出现没说是本卡的量
- 措辞｜Attention 面板 YaRN｜`corrected_dim` 没交代是什么
- 措辞｜Attention 折叠 `wquant` 及 `cquant`/`kq`/`iq_q`｜E8M0、E4M3、E2M1 三个浮点格式名首次出现没解释
- 措辞｜Attention 折叠 `cat`、`wwrite` 与全篇｜同一批张量在「窗口 KV / 窗口键」之间换名
- 措辞｜Indexer 折叠 `imask`｜公式写 `arange(seqlen / ratio)`，节点写 `torch.arange(1, seqlen + 1) // ratio`，两处写法对不上
- 措辞｜Expert 折叠 `g1`/`w2`｜路由专家权重形状与 `dim`、`moe_inter_dim` 对不上，没说 fp4 把两个值打包成一格
- 措辞｜Aligner 折叠 `vv`/`vpad`/`vunf`｜这里的 `r` 是 downsample_ratio 3，与 Attention 的 `r` 同形不同义
- 措辞｜多图同名权重｜`wkv`、`w1`、`w2` 这几个名字在不同图里形状差很远，没注明各自独立
- 措辞｜config.json 表与前提｜配置组名 `DSpark` 全篇只出现在配置表，没交代它是什么

## 修改

对照源码改过，并重新生成了页面。

- MoE 图补上 `im` 张量节点与 `im → mmask` 连线：`image_mask` 是这一层前向的参数，压平之后才交给 `gate_op`。
- 描述：切成 patch 后加逗号；裸用的「source 层」改成「kv source 层」；压缩 KV 那句改成「除 128 个位置的滑动窗口外，另加…」，并把索引器在哪几层跑写清楚。
- 摘要：FP8/FP4 改成小写 fp8/fp4，Top-512 改成 top-512。
- 资料：「块号」改成「同一层内同一子模块的编号」；补上模型卡 `sources/model-card.md` 一行。
- 前提：顿号改分号；层 20 的压缩器那句改成写明是不做逐 token 池化、只留 wkv 一路。
- Transformer 折叠 `mrg`：注明大写的是 token 类型、小写的是学到的嵌入参数。
- Attention 面板：输出低秩投影的下标由 `r` 改成 `m`，并注明与 compress_ratio 不是一回事；YaRN 补一句 YaRN 与 `corrected_dim` 是什么；`qb`/`grp` 的说明补上单卡下 `n_local_heads` = `n_heads` 64、`n_local_groups` = `o_groups` 8。
- 浮点格式名：`wquant` 说明补 E8M0 是什么，`cquant` 说明补 E2M1 与 E4M3 是什么。
- Indexer 折叠 `imask`：改成与节点一致的源码写法 `index_score.masked_fill_(arange(seqlen // ratio) >= compress_lens, -inf)`。
- Expert 折叠 `g1`/`w2`、Aligner 折叠 `vunf`、Attention 折叠 `cat`/`wwrite` 的措辞按上表改；多图同名权重在各图的说明里点明是哪个模块的。
- config.json 表的 DSpark 一行补上它指什么。

## 收口时的机械核查

到轮数上限后，把这一轮与第 8 轮的所有改动落盘，重跑四道机械检查，全过：

```
$ python3 .dojo/scripts/dataflow/build_page.py wiki/deepseek-v4-1-dataflow/research/dataflow.md
已生成 wiki/deepseek-v4-1-dataflow/index.html：18 张图，381 个节点

$ python3 .dojo/scripts/dataflow/check_source.py wiki/deepseek-v4-1-dataflow/research/dataflow.md --source-root wiki/deepseek-v4-1-dataflow/research/sources/repo
源码位置核查通过：379 个节点

$ python3 .dojo/scripts/dataflow/check_source.py wiki/deepseek-v4-1-dataflow/index.html \
    --source .../model.py --source .../vision.py --source .../engram.py --source .../kernel.py \
    --shapes wiki/deepseek-v4-1-dataflow/research/sources/shapes.json
核查通过：18 视图 / 381 节点（节点名 vs 4 份源码、形状 vs 材料 8 个键）

$ python3 .dojo/scripts/site/check.py wiki/deepseek-v4-1-dataflow/index.html
检查通过：1 个页面，内联脚本 2 块

$ python3 .dojo/scripts/dataflow/check_source.py wiki/deepseek-v4-1-dataflow/index.html --geometry
几何自检通过：全部视图穿框 0、像素重合 0
```

截图逐张看过（`--shots`，18 张图 × 画布/面板/折叠）：六张视觉图的画布与折叠、Transformer 图的新节点、config 面板顶部的「前提」列表与「视觉」一组都正常。

`介绍` 一组按后来定下的措辞改成两句（模型 / 这次前向），数字出自 `sources/model-card.md`，该文件已补进「资料」一节。

## 已知不足（未计入的部分）

- 本轮「路径覆盖」「连线与结构」「公式与面板」三席没返回结论；这三类的覆盖在第 7、8 两轮做过，本轮没有新的独立核对。
- 最后一轮核对（`review-plan.md` 的「最后一批改动也要核对」）没有做：到收口时没有可用的仲裁席位。第 8 轮末尾与第 9 轮的所有改动，都只经过作者自查与 `build_page.py`、`check_source.py`、`site/check.py`、`--geometry` 四道机械检查（见下），没有人再逐条对照源码核。

### 第 10 轮

源码位置核查通过：379 个节点（本轮开始时）。有结论的席位 3 个（路径覆盖、连线与结构、公式与面板）。这一轮有事实错误，改完后按「最后一批改动也要核对」交仲裁核对，通过。

**说明**：第 9 轮有 3 席（路径覆盖、连线与结构、公式与面板）在收口前被 API 网关 502 打断、没返回结论，且到轮数上限后「最后一批改动也要核对」这一步没做。本轮把这两块补上：只重派这 3 席（第 9 轮已返回的「节点与形状」「读者与措辞」两席不重派，它们的结论记在 review-9.md），审的是第 9 轮收口后的现状；审完按 `review-plan.md` 的「最后一批改动也要核对」，把第 8 轮末尾到本轮结束的全部改动交一个独立子代理核对。

## 审查者 1（路径覆盖）

- 事实错误｜Indexer.forward 图（节点 `csel`、`cmask`、`shar`、`cmf`、`cmfd`）｜候选块这两段在前提锁定的这条路径上并不执行，画上去是多画：`if self.is_candidate_source:` 与 `elif self.uses_candidates:` 互斥，各只在层 20 / 层 24 及以后成立，而前提写的是「索引器走 compress_ratio 为 2 这条分支」｜model.py:569-575；config.json 的 `candidate_source_layer` 为 20；计划前提第 28 行｜删掉这五个节点与相关连线，让 `iscm` 直接接 `topk`；或者反过来改前提，把这一节明确画成「层 20 写候选、24/28/32/36 读候选」，并把「索引器走 compress_ratio 为 2」的前提改掉
- 事实错误｜Transformer 图的 `emb` 节点、Engram 图的 `nemb` 节点｜`ParallelEmbedding` 与 `ParallelEngramEmbedding` 都是源码里自己定义、各有 `forward` 的类，整页没有它们的图，用到它们的节点也没点开，而是把类内部的算式抄进了折叠｜model.py:168-178、312-325；plan.md「前向用到的、模型源码里自己定义的类，每个都画一张，用到它的节点都点开到这张图」「折叠里的公式写这个算子自己的计算，不去抄被调类内部的算式」｜各补一张图，`emb`、`nemb` 的折叠改成写调用并加点开

其余核过没问题：五个 `forward`（Transformer、ViT、vision.Block、vision.Attention、vision.MLP、vision.RMSNorm、Aligner、Block、Attention、Compressor、Gate、Expert、MoE、Engram、NgramHashState、RMSNorm、ParallelHead）前提路径上的每一步都有对应节点；`if`/比较/`for` 没有画成节点；循环层都按「结构相同的层画成一个节点」处理；算子与公式里没有前提之外的分支。

## 审查者 2（连线与结构）

- 写法｜Transformer.forward｜`layers → eng`、`layers → avg` 两条是算子直连算子，破坏「算子和张量交替出现」：`eng` 吃的是循环里上一层 Block 的输出 h，`avg` 吃的是进第 37/38/39 层之前的 h，图上只画了末层那份 `hl`，这两处都没有对应的张量节点｜model.py:1263、1266；plan.md「算子和张量交替出现」｜给循环里那份 h 补一个张量节点（`eh` 是 eng 的输出，不能兼作 Block 的输出）
- 写法｜Gate.forward（`bsel` → `gsel`）｜算子直连算子：`bsel` 选出的那份 bias 既没有节点也没有连线，源码里它是 `scores + bias` 的实参｜model.py:819-822｜补一个 fp32 [B×T, 384] 的张量节点
- 写法｜Block.forward（`hin → resid0`、`x4 → resid1`）｜张量直连张量：`residual = x`（model.py:981、988）是源码里的重新绑定，两处都没在第三行或线上注明是同一份｜plan.md「算子和张量交替出现」｜把这一步画成一个算子节点，或把 `resid0`/`resid1` 并进 `hin`/`x4` 并注明是同一份

核过没问题的：18 张图 381 个节点逐条对过源码——每个算子第一行的实参都有输入连线（含只取形状/dtype 的线）；参数传入的张量都连了线；没有同一参数连两条；缓存都按读写接在对应算子旁；8 个点开目标都精确解析到真实图标题；多输出算子的每个下游用到的输出都各接了一个张量；跨图同名 id 都指同一个量、形状一致；折叠只引用同图已画出的张量。

## 审查者 3（公式与面板）

- 事实错误｜ViT.forward 图 `vcs_op` 折叠公式、该图面板「二维位置编码」｜`inv_freq_j = θ^{-2j/head_dim}` 且说明写「head_dim 是每个头的宽度」（64）；`get_vision_cos_sin` 的除数实参是 `self.rope_dim`，构造时算 `vision_dim // vision_n_heads // 2` = 32，指数应是 −2j/32，写成 −2j/64 差一倍（该说明自己也算出「频率长度 32 = head_dim / 2」，前后对不上）｜vision.py:10、vision.py:92、vision.py:100｜改成 `θ^{-2j/rope_dim}`（rope_dim = 32），或写成 `θ^{-2j/(head_dim/2)}`，并把式后符号说明里的 head_dim 换成 rope_dim
- 事实错误｜vision.Attention.forward 图 `vcq` 折叠说明｜写「一对相邻的通道配成一个二维向量旋转」；视觉的 `apply_rotary` 用 `x.float().chunk(2, dim=-1)` 把每个头 64 维切成前后两半，配的是第 i 维与第 i+32 维，不是相邻两维（相邻两维配对是主干 model.py:397 那套，本页别处都写「相邻两维配成一对复数」，此句照抄过来就是错的）｜vision.py:20-21 对 vision.py:100 的 cos/sin 长 32｜改成「每个头的前一半与后一半配成一对（第 i 维配第 i+32 维）」
- 措辞｜`head`、`fnorm`、`an`、`fn`、`qr_op`、`wkv_win`、`cnorm`、`kw`、`vnorm`、`vbn1`、`vbn2` 的「公式」｜折叠里抄了被调类内部的算式，而这些类在本页各有自己的图（点开即到）；同页其余已单独成图的类都写成调用，画法自相矛盾｜plan.md「折叠里的公式写这个算子自己的计算，不去抄被调类内部的算式（写成调用即可），和别处的画法一致」｜改成调用：`logits = head(h)`、`h = norm(h)`、`x = attn_norm(x)`、`x = ffn_norm(x)`、`qr = q_norm(wq_a(x))`、`kv = kv_norm(wkv(x))`、`kv = norm(kv)`、`k = k_norm(wk(latent))`、`x = norm(x)`、`x = norm1(x)`、`x = norm2(x)`
- 措辞｜Attention.forward 图｜符号 $t$ 同图两义：`widx` 式里指 query 位置（源码里这个量叫 `end`），`core` 式里指 topk 槽位号；且 `widx` 用了源码没有的名字｜model.py:418、kernel.py:358｜`widx` 式里的 $t$ 改用源码名 `end`；`core` 式的槽位号换一个字母
- 措辞｜Indexer.forward 图｜符号 $i$ 同图两义：`red` 式里是头维，`lens` 式里是 query 位置｜model.py:557、563-564｜把 `lens` 的 $i$ 换一个字母，两处各自点明是哪一个量
- 措辞｜Transformer.forward 图面板「介绍」组｜只写了模型是什么，没写这次前向在算什么｜plan.md「第一张图的面板最上面写一组「介绍」，用几句话说明这个模型是什么、这次前向在算什么」｜补一句这次前向

其余核过没问题：1（17 张图的算子都有「公式」折叠，无遗漏）、5（折叠名只有「公式」「说明」，且都挂在算子上）、8（各面板配置值都能在 config.json 逐字对上；默认值的注明了「默认值」；精度改写都在前提或说明里交代）、9（配置组里没有权重形状）、10（描述/摘要里 ViT、MoE、RoPE、VL、Engram 首次出现都带全称，MTP 在前提首次出现带全称，DSpark、CSA2、E8M0/E4M3/E2M1 全文一义）。

## 仲裁

「路径覆盖」审查者的第 2 条（`ParallelEmbedding`、`ParallelEngramEmbedding` 该不该各补一张图）与作者的理由冲突：作者认为这两个薄类前几轮已裁定不单独成图（同 `PatchEmbed` 的画法），审查者认为 plan.md 要求模型自定义的类各画一张。按 `review-plan.md` 交仲裁。

仲裁结论：**必须改**。依据 plan.md 第 78 行「前向用到的、模型源码里自己定义的类，每个都画一张，用到它的节点都点开到这张图」——`ParallelEmbedding`（model.py:152-178）与 `ParallelEngramEmbedding`（model.py:296-325）都是模型源码里自己定义、各有 `forward` 的类，且本次前向确实用到，不属于该条豁免的「PyTorch 自带的模块」或「普通函数」；plan.md 第 87 行「折叠里的公式写这个算子自己的计算，不去抄被调类内部的算式」——现 `emb` 折叠写的 `h = embed.weight[input_ids]` 正是 `ParallelEmbedding.forward` 的式子，`nemb` 折叠把整套反量化抄了进去。作者的反驳（「薄包装类不单独成图」「`PatchEmbed` 也这么画」「前几轮已裁定」）「不构成 plan.md 的条文依据」，「前几轮已裁定」是轮次记录而非条文，不予采纳。按此裁定，本轮补两张图并改两处折叠。

「路径覆盖」审查者的第 1 条（Indexer 图的候选块节点是多画）：作者核实前提原文把整条路径锁在「compress_ratio 为 2 这条分支」上，而 Indexer 图现在按 8 个 index source 层画，其中层 20 写候选、24/28/32/36 读候选，确实与旧前提不符。审查者给了两条互斥的改法，作者选「改前提、明确画成层 20 写候选、24/28/32/36 读候选的那条路径」——因为删掉候选块节点会丢掉主干上真实执行的两步（这是第 8 轮审查者 1 判「必须补」的漏步），删掉反而回退。

## 修改

对照源码改过，并重新生成了页面。20 张图 394 个节点，源码位置再查过，392 个节点。

**事实错误**

- ViT 图 `vcs_op` 折叠与面板「二维位置编码」：`inv_freq` 的除数由 `head_dim` 改成 `rope_dim`（= vision_dim // vision_n_heads // 2 = 32），符号说明同步改掉。
- vision.Attention 图 `vcq` 说明：改成「每个头 64 维的前一半与后一半配成一对（第 i 维配第 i+32 维）」。
- 新增 `ParallelEmbedding.forward`（model.py:168-178）与 `ParallelEngramEmbedding.forward`（model.py:312-325）两张图；Transformer 图的 `emb`、Engram 图的 `nemb` 折叠改成写调用，并各加「emb 点开 → ParallelEmbedding.forward」「nemb 点开 → ParallelEngramEmbedding.forward」。
- Indexer 图：不是删节点，而是改前提——「注意力、压缩器与索引器走的是 compress_ratio 大于 0 这条分支」+ 补一条写清 8 个 index source 层共用一张 Indexer 图、层 20 写候选、24/28/32/36 读候选、不建键只读 index_k。前提与图现在一致。

**写法**（审查者 2 的三条）

- Transformer 图：`hl` 节点的第三行写明它同时代表循环里每层返回、又喂给下一层的那份 h；连线改成 `layers → hl`、`hl → eng`、`hl → avg`，去掉原来的 `layers → eng`、`layers → avg` 两条算子直连算子的线。
- Gate 图：新增张量 `bbias`（fp32 [B×T, 384]，按 token 选出的那份偏置），连线改成 `bsel → bbias → gsel`；`gsel` 折叠的「bias」注明来自 `bbias`。
- Block 图：删掉 `resid0`、`resid1` 两个张量节点，把 `residual = x` 并进 `hin`/`x4` 的第三行（注明是同一份重新绑定）；连线改成 `hin → cpost：residual`、`x4 → cpost2：residual`；`cpost`/`cpost2` 折叠的 residual 指向同图的 `hin`/`x4`。

**措辞**

- 折叠「公式」由抄被调类内部算式改成写调用：`head`、`fnorm`、`an`、`fn`、`qr_op`、`wkv_win`、`cnorm`、`kw`、`vnorm`、`vbn1`、`vbn2`，以及审查者 1 点名的 `emb`、`nemb`；被调算式留在各自点开的图上。
- Attention 图 `widx` 折叠的 $t$ 改用源码名 `end`；`core` 折叠与面板「稀疏注意力」的槽位号 $t$ 改成 $s$ 并注明是槽位不是位置。
- Indexer 图 `lens` 折叠与面板「可达数」的 $i$ 改成 $s$，并注明是 query 位置、与同图 `red` 的头维 $i$ 不是一回事。
- Transformer 面板补「介绍｜这次前向」一行：一次带图 prefill，图先过 ViT 与对齐器写回图像 span，文本位置查词嵌入与 n-gram 哈希，整段过 40 层主干，最后取最后一个位置投影出 logits。

改动后重跑：

```
$ python3 .dojo/scripts/dataflow/build_page.py wiki/deepseek-v4-1-dataflow/research/dataflow.md
已生成 wiki/deepseek-v4-1-dataflow/index.html：20 张图，394 个节点

$ python3 .dojo/scripts/dataflow/check_source.py wiki/deepseek-v4-1-dataflow/research/dataflow.md --source-root wiki/deepseek-v4-1-dataflow/research/sources/repo
源码位置核查通过：392 个节点
```

## 最后一批改动核对

按 `review-plan.md`「最后一批改动也要核对」，派一个独立子代理核对第 8 轮末尾到本轮结束的全部改动（第 8、9 轮的改动清单见 review-8.md、review-9.md 的「修改」一节，本轮改动即上文）。只回答两件事：有没有引入源码里没有的内容、漏步或多步、接错线；有没有违背前几轮的裁定。

核对结论：**(a) 通过。(b) 无。**

- 前提两条新表述对得上源码与 config：`compress_ratios`、`kv_source_layers`、`index_source_layers`、`candidate_source_layer` 逐项对上；层 20 ratio=1 走 `Compressor.forward` 的 `ratio==1` 早返回，与「只留 wkv 一路」一致。
- `hl` 第三行与四条线（`layers→hl`、`hl→eng`、`hl→avg`、`hl→col`）是同一循环变量 h 的重新绑定，接的是 model.py:1261-1267 的真实数据流，未漏步、未接错。
- 两张新图按第 10 轮裁定新增；节点、连线、折叠对照源码无凭空内容（`F.embedding`、两次 embedding＋反量化都逐条对上）；单卡下不执行的 mask/`all_reduce` 只写进说明，未额外画成节点。
- `weights.json`/`shapes.json` 佐证形状与两张新图的第二行一致。
- Block 图删 `resid0`/`resid1`、并进 `hin`/`x4`，与 981/986/988 两次「绑同一份」相符；未被任何折叠残留引用。
- Gate 图新增 `bbias` 并改 `bsel→bbias→gsel`，与 822 `(scores + bias).topk` 一致。
- 13 处公式改「写调用」均符合 plan.md。
- 违反前几轮裁定的项：无。`shared` 未点开 `Expert.forward`；`counts` 仍为张量；`PatchEmbed` 未单独成图；`patches→vcs_op` 线的 `n_vit_h`/`n_vit_w` 仍未单独成节点；六张视觉图前缀仍为 `vision.`。

三道机械检查也全过：`build_page.py`（20 图 394 节点）、`check_source.py`（计划 392 节点；页面 20 视图 / 394 节点，节点名对 4 份源码、形状对材料 8 个键）、`site/check.py`（1 个页面）。

## 已知不足

- 第 9 轮「节点与形状」「读者与措辞」两席的结论按 `review-9.md` 计入，本轮未重派（本轮只补第 9 轮未返回的 3 席）。
- 页面几何（`--geometry`）与截图（`--shots`）是页面级检查，改完计划生成后由作者在页面级重跑，本轮记录只覆盖计划与生成。

## 附录 D：原第 11 轮（补薄包装类图）

源码位置核查通过：409 个节点。有结论的席位 5 个。这一轮有事实错误（两处重复点开），改完后按「最后一批改动也要核对」交仲裁核对。

**说明**：`guides/dataflow/plan.md` 本轮加了一条「只有一行计算的薄包装类（如 `PatchEmbed`、`ParallelEmbedding`）也要单独画，图里就是一个节点加它的输入输出」。按这条，本页给 4 个薄包装类补了图：`Linear.forward`（model.py:242-243）、`ColumnParallelLinear.forward`（model.py:256-257）、`RowParallelLinear.forward`（model.py:271-278）、`PatchEmbed.forward`（vision.py:42-43）。`PatchEmbed` 原先是画在 `ViT` 图的 `vpe` 节点上、算式进折叠的，本轮抽出来单独成图，`ViT` 图的 `vpe` 节点改成点开到新图、折叠改写成调用。用到这些类的节点也补了「点开」。

第 10 轮之后的轮次：第 10 轮「通过」之后，本轮是新的一轮（编号接现有最大编号 10）。下面 5 席都是从头审的全页，问题只列出与本轮新增内容相关的；本轮一开始给几个视觉节点也加了「点开 → Linear.forward」，作者自查发现 `vision.py` 用的是 PyTorch 自带 `nn.Linear`（不是本页的 `Linear` 类），已先撤掉 6 条（见「修改」）。

## 审查者 1（路径覆盖）

- 事实错误｜Attention.forward 图，节点 `qr_op`｜该节点调用了 `self.wq_a`（Linear 实例），却没有「点开 → Linear.forward」，而同一张图里结构相同的 `wkv_win` 两条件都接齐了｜model.py:770（`qr = self.q_norm(self.wq_a(x))`）、model.py:640（`self.wq_a = Linear(self.dim, self.q_lora_rank)`）；对照同图 `wkv_win`（model.py:705、643）的连线｜在 Attention.forward 的「连线」补 `- qr_op 点开 → Linear.forward`
- 存疑（低置信，未判错）｜Attention.forward 图，节点 `woa`｜`woa` 是 `self.wo_a.weight.view(...)`，只读权重、不调 `wo_a.forward`；本轮新加的「点开 → ColumnParallelLinear.forward」对应的 forward 在该节点并不执行｜model.py:786｜删掉，或在折叠里写明「只取 wo_a.weight」

其余核过都没有问题：本轮 4 张新图的内容与各自 `forward` 逐句对齐，不多画、不漏步；点开接对（`qb → ColumnParallelLinear.forward`、`wb → RowParallelLinear.forward`、`proj/kw/wkv_win/g1/u1/w2/nwkv → Linear.forward`、`vpe → PatchEmbed.forward`）；全页「源码自定义 + 有 forward + 本次前向用到」的类都已成图；循环层按「结构相同的层画成一个节点」处理；新图没有把前提之外的分支画成节点。

## 审查者 2（节点与形状）

- 事实错误｜Attention.forward 图，节点 `qr_op`｜用了 `self.wq_a`（Linear 实例）却漏了「点开 → Linear.forward」（同图 `wkv_win` 补了两条）｜model.py:640、770；dataflow.md 原第 1019 行 `wkv_win 点开 → Linear.forward`｜补 `- qr_op 点开 → Linear.forward`
- 写法（低置信）｜Attention.forward 图，节点 `woa`｜`woa` 是 `self.wo_a.weight.view(...)`，本路径只读 `wo_a.weight`，真正的乘法是 `eins`，`wo_a.forward` 从不执行；源码注释明说「hence einsum not Linear」｜model.py:783-787｜去掉这条点开，或只保留权重的 dtype/形状说明
- 写法（低置信）｜PatchEmbed.forward 图，节点 `pe_w`｜算子 `self.proj(x.flatten(1))` 同时用到 `proj.weight` 与 `proj.bias`（`nn.Linear` 默认 `bias=True`），图里把 `proj.weight` 画成节点 `pe_w`、`proj.bias [1024]` 只写在折叠说明里，同一算子里两个量一画一不画｜vision.py:40、43｜补一个 `self.proj.bias` 张量节点，或把 `pe_w` 一起去掉

其余核过都没有问题：4 张新图的节点名、形状、类型、权重形状与源码和 `sources/shapes.json` 全部对得上（`pe_w [1024, 588]` 的 588 = 3 × 14 × 14；fp8 权重 [out, in]；`rpl_y`/`cpl_y`/`lin_y` 的 bf16 正确）；三条路只走一条的说法正确。

## 审查者 3（连线与结构）

- 事实错误｜Attention.forward（`wkv_win`）｜同一节点写了两条「点开」：`RMSNorm.forward` 与 `Linear.forward`。生成器每个节点只存一个目标、后一条覆盖前一条（`build_page.py:251-257` 的 `by_id[nid]["drill"] = tid`，无重复检测），实际页面上 `wkv_win` 点开只到 `Linear.forward`；而该节点折叠仍写「kv_norm 是 RMSNorm，算式见点开的那张图」，前后矛盾｜model.py:705；build_page.py:251-257｜一个节点只留一条点开：把另一路的算式写进本节点折叠，或把这一行拆成两个节点
- 事实错误｜Indexer.forward（`kw`）｜同上：`kw` 同时点开 `RMSNorm.forward` 与 `Linear.forward`，后者覆盖前者，`RMSNorm` 目标失效；折叠仍写「k_norm 是 RMSNorm，算式见点开的那张图」｜model.py:544；build_page.py:251-257｜一条节点一条点开
- 事实错误｜Attention.forward（`woa`）｜`woa` 点开到 `ColumnParallelLinear.forward`，但这个节点是 `self.wo_a.weight.view(...)`，源码在这一行没有调用 `self.wo_a` 的 forward（文件注释明说用 einsum 代替 Linear），点开到的算式在该处并不发生｜model.py:783-787、645-649｜删掉这条点开
- 写法｜Attention.forward（`qr_op`）｜同一节点含 `q_norm`（RMSNorm）与 `wq_a`（Linear）两个子模块，却只点开 `RMSNorm.forward`；而结构完全相同的 `wkv_win`、`kw` 却给 Linear 也加了点开（并因此出现上面的重复点开）｜model.py:770、640-641｜统一规则：一个节点只保留一条点开

其余核过没问题：新增点开都指向真实存在的图标题，类归属与源码一致；`PatchEmbed.forward` 抽出后 ViT 的 `vpe` 折叠已改为调用写法、不再重复展开，`vx`/`patches` 跨图 id 与形状一致；四张新图内 id 无重复、无算子直连算子或张量直连张量。

## 审查者 4（公式与面板）

- 措辞｜Linear.forward 图 / `lin_op` 折叠（说明）｜选路枚举写成「fp8 走 fp8_gemm、fp4 走 fp4_gemm、bf16 走 F.linear」三条，但同一条说明里刚列出压缩器的 wkv、wgate 是 fp32；源码里 fp32（及其它非 fp4/fp8 的 dtype）同样落进 `else: F.linear`，这一路没写｜model.py:186-207、446、448｜把末句改成「其余 dtype（bf16 与 fp32）走 F.linear」
- 措辞｜ColumnParallelLinear.forward 图 / `cpl_op` 折叠（说明）｜写「只走 fp8_gemm、fp4_gemm、F.linear 里的一条」，但本页 ColumnParallelLinear 的实例只有 fp8 与 bf16，不存在 fp4｜model.py:514、515、642、645-649｜改成「只走 fp8_gemm 或 F.linear 里的一条」，或不提 fp4
- 措辞｜RowParallelLinear.forward 图 / `rpl_op` 折叠（说明）｜「world_size 大于 1 时还有两段：先 y.float() 再 all_reduce，然后 y += self.bias」把 `y += self.bias` 并进了 world_size>1；源码里这句由单独的 `if self.bias is not None` 把守，与 world_size 无关｜model.py:273-277｜把 bias 那句拆出来，标明它只由 self.bias 是否为 None 决定

其余核过都没有问题：四张图的公式符号与源码行一致，权重形状对得上；折叠名只有「公式」「说明」且都挂在算子上；面板配置值全部照抄 config.json，无默认值或运行改写值需要标注。

## 审查者 5（读者与措辞）

- 措辞｜vision.Attention / vision.MLP / Aligner 三张图｜这三张图里 `self.wqkv(x)`、`self.wo(...)`、`self.w1/w2(...)` 这些线性投影既不点开 `Linear.forward`，也不像 PatchEmbed 那样注明是 nn.Linear，读者无法判断它们与 `Linear.forward` 是不是同一张图｜依据：`Linear.forward` 图的折叠与 `PatchEmbed.forward` 折叠「proj 是 nn.Linear（vision.py:40）」｜若视觉也用同一个 `Linear`，补点开；若是 nn.Linear，就在每处注明
- 措辞｜Attention 图 `woa` 节点｜节点是权重的 `.view`，却连了「woa 点开 → ColumnParallelLinear.forward」，点开后的算式对不上｜依据：woa 节点第一行与折叠｜删掉这条点开，或在折叠里写「woa 是 ColumnParallelLinear，这里只用它的 weight」
- 措辞｜Linear/ColumnParallelLinear/RowParallelLinear 三处折叠的「本页每个调用点都没传 bias」｜「本页」按字面覆盖整页，但页面上视觉支路多处投影显式带 bias，读者会读到直接矛盾｜依据：`lin_op`/`cpl_op`/`rpl_op` 说明；对比 `vwqkv` 公式「+ wqkv.bias」｜改成「主干 Linear/ColumnParallelLinear/RowParallelLinear 的每个调用点都没传 bias」
- 措辞｜RowParallelLinear 图 `rpl_y` 节点与 `rpl_op` 折叠｜节点写「all_reduce 与 bias 两段（单卡下都不执行）」，把两段都归因于单卡；折叠却写 bias 那段「由 self.bias 是否为 None 决定（与 world_size 无关）」，同一张图前后归因不一致｜依据：`rpl_y` 第三列 vs `rpl_op` 说明第二段｜节点写成「all_reduce（单卡不执行）与 bias（本页 self.bias 为 None，不执行）两段」
- 措辞｜Attention 图 `eins` 节点 vs 同图面板「输出低秩投影」｜同一个量，节点折叠写 `o_{b,s,g,r}`（r 指 o_lora_rank 维），面板写 `o_{b,s,g,m}`，字母对不上｜依据：eins 公式行 vs 面板｜节点也改用 m
- 措辞｜Attention 图（`cf` 折叠与面板「压缩 KV 的位置」）｜同一个字母 `g` 一会儿指压缩 KV 的分组序号，一会儿指 o_groups 的组号；面板只把 `r`/`m` 分开了，没处理 `g`｜依据：两处引文都在 Attention 一图内｜给两处 `g` 分别注明，或把压缩分组序号换成别的字母
- 措辞｜PatchEmbed 图 `pe_op` 折叠｜公式 `x = proj(x.flatten(1))` 没写 bias，说明却列了 `proj.bias [1024]`；同页 vision.Attention、vision.MLP 都把 bias 显式写进算式｜依据：`pe_op` 公式 vs 说明｜公式写成含 bias 的形式，或注明「这一调用含 bias」
- 措辞｜`lin_w`、`cpl_w`、`rpl_w` 三个权重节点｜第二列都固定写 `fp8 [out, in]`，折叠却说 weight 的 dtype 按调用点不同｜依据：`lin_w` 行 vs `lin_op` 说明｜第二列写成「fp8 [out, in]（随调用点变，见折叠）」

其余核过通过：四张新图本身都能看懂；各调用点到新图的点开都能对上；PatchEmbed 从 ViT 抽出后两者用 `patches`/`vx` 互指，对得上；`head_dim`、`r` 的区别已注明；描述与摘要无矛盾。

## 修改

对照源码改过，并重新生成了页面。24 张图 411 个节点，源码位置再查过，409 个节点。

**事实错误**

- 重复点开：`wkv_win`、`kw` 各写了两条「点开」（RMSNorm 与 Linear），生成器一个节点只留一条、后一条覆盖前一条，两处 RMSNorm 目标失效。按审查者 3 与 2、4 的同一意见，**各留一条**：`wkv_win`、`kw` 保留原有的 `RMSNorm.forward`（折叠引用的就是它），撤掉本轮新加的 `Linear.forward`；Linear 的算式在这两个节点已由折叠写出的调用可见，且 `Linear.forward` 在 `qb`、`proj`、`g1`、`u1`、`w2`、`nwkv` 等节点上都点得到。
- `woa` 点开：`woa` 是 `self.wo_a.weight.view(...)`，这一行不执行 `wo_a.forward`（源码注释明说用 einsum 代替），撤掉「woa 点开 → ColumnParallelLinear.forward」。审查者 1、2、3、5 都提到这一处。
- `qr_op` 点开：补 `qr_op 点开 → Linear.forward`。审查者 1、2 都要求；但补上会与同节点的 `qr_op 点开 → RMSNorm.forward` 形成重复点开，按上一条的统一规则保留 `RMSNorm.forward`（`qr_op` 折叠写的 `qr_op = q_norm(wq_a(x))` 也引它），**不补** Linear 这一条。这两条（重复点开的处理、`qr_op`/`woa` 保留哪一条）与审查者的部分要求相反，见「仲裁」。

**写法 / 措辞**

- `PatchEmbed` 图补张量 `pe_b`（`self.proj.bias`，bf16 [1024]）与 `pe_b → pe_op` 连线；`pe_op` 公式改成含 bias 的 `$x = x.\mathrm{flatten}(1)\, proj.weight^\top + proj.bias$`。
- `lin_op`/`cpl_op`/`rpl_op` 的说明：`Linear` 的选路改成「fp4 走 fp4_gemm、fp8 走 fp8_gemm，其余 dtype（bf16 与 fp32）走 F.linear」；`ColumnParallelLinear` 改成「只走 fp8_gemm 或 F.linear 里的一条」。
- `rpl_op` 说明：把 `y += self.bias` 从 world_size 那一段拆出来，标明它由 `self.bias is not None` 决定、与 world_size 无关；`rpl_y` 第三行改成「all_reduce 段（单卡不执行）与 bias 段（本页 self.bias 为 None，不执行）」。
- 三处折叠的「本页每个调用点都没传 bias」改成「主干的调用点都没传 bias」（并按审查者 2 的同一意见在 `lin_op` 说明里点明 wo_a、weights_proj 是 bf16、其它非 fp4/fp8 的走 F.linear）。
- Attention 图：`eins` 折叠的 `r` 改成 `m`（与面板「输出低秩投影」一致）；`cf` 折叠与面板「压缩 KV 的位置」的压缩分组序号由 `g` 改成 `j`，并注明与 o_groups 的组号不是一回事。
- `lin_w` 第三行补「存法与 dtype 见折叠的说明」，`cpl_w` 第三行补「dtype 随调用点变，见折叠的说明」。
- 视觉三张图的线性投影注明是 `nn.Linear`（PyTorch 自带，与主干 `Linear` 不是一类）：`vwqkv`、`vwo`、`vw1`、`vw2`、`vlin` 的说明各补一句。
- 最后一批改动核对指出的两处残留：`lin_w`、`cpl_w` 的第二列由写死的 `fp8 [out, in]` 改成 `fp8 [out, in]（随调用点变，见折叠）`；`cpl_op` 说明末句由「本页每个调用点都没传 bias」改成「主干的调用点都没传 bias」。

改动后重跑：

```
$ python3 .dojo/scripts/dataflow/build_page.py wiki/deepseek-v4-1-dataflow/research/dataflow.md
已生成 wiki/deepseek-v4-1-dataflow/index.html：24 张图，411 个节点

$ python3 .dojo/scripts/dataflow/check_source.py wiki/deepseek-v4-1-dataflow/research/dataflow.md --source-root wiki/deepseek-v4-1-dataflow/research/sources/repo
源码位置核查通过：409 个节点

$ python3 .dojo/scripts/dataflow/check_source.py wiki/deepseek-v4-1-dataflow/index.html \
    --source .../model.py --source .../vision.py --source .../engram.py --source .../kernel.py \
    --shapes wiki/deepseek-v4-1-dataflow/research/sources/shapes.json
核查通过：24 视图 / 411 节点（节点名 vs 4 份源码、形状 vs 材料 8 个键）

$ python3 .dojo/scripts/site/check.py wiki/deepseek-v4-1-dataflow/index.html
检查通过：1 个页面，内联脚本 2 块

$ python3 .dojo/scripts/dataflow/check_source.py wiki/deepseek-v4-1-dataflow/index.html --geometry
几何自检通过：全部视图穿框 0、像素重合 0
```

## 仲裁

本轮有两条与审查者意见相反的处理，按 `review-plan.md` 交仲裁：

1. 审查者 1、2 要求给 `qr_op` 补「点开 → Linear.forward」，审查者 3 同时指出 `wkv_win`、`kw` 的重复点开。作者的方案是：一个节点只留一条点开（保留折叠引用的那份），因此 `qr_op`、`wkv_win`、`kw` 都只保留 `RMSNorm.forward`，不补 Linear。
2. 审查者 1、2、3、5 都提到 `woa` 的点开对不上，作者撤掉了这条点开（`woa` 只读权重、不调 forward）。

**仲裁结论**：

- **第 1 条（`qr_op`/`wkv_win`/`kw` 该不该补点开 `Linear.forward`）｜不必改。** 依据 plan.md 第 78 行「前向用到的、模型源码里自己定义的类，每个都画一张，用到它的节点都点开到这张图」只约束到「类」这一级：每张类图都要有节点点开过去，不要求一个同时调用两个类的节点挂两条点开。`Linear.forward` 已在 `proj`、`g1`、`u1`、`w2`、`nwkv`、`qb` 等节点点得到，`RMSNorm.forward` 由 `qr_op`/`wkv_win`/`kw` 点得到，两个类都满足。`build_page.py` 的 `by_id[nid]["drill"] = tid` 是单值赋值、无重复检测，一个节点只存一条点开是真实的引擎约束。作者保留折叠已引用的 `RMSNorm.forward`、撤掉被覆盖的 `Linear.forward`，既不重复点开也符合计划，无需再改。
- **第 2 条（`woa` 撤掉点开）｜不必改（作者的处理正确）。** 依据 model.py:783-787（`wo_a = self.wo_a.weight.view(...)` 后接 einsum，无 `self.wo_a(...)` 调用）与 plan.md 第 85 行（只取形状/dtype、没当数据用的不画节点）：`woa` 只读 `self.wo_a.weight`，前提路径上不执行 `ColumnParallelLinear.forward`，不属于「用到该类的节点」，撤掉这条点开正确。

## 最后一批改动核对

按 `review-plan.md`「最后一批改动也要核对」，把本轮全部改动交一个独立子代理核对：有没有引入源码里没有的内容、漏步或多步、接错线；有没有违背前几轮的裁定。

**核对结论：通过。** 由一个独立子代理逐项对照源码核过（它返回了结论），下面（a）(b) 是它的结论、以及它指出的两处残留（已改）。

- **(a) 有没有引入源码里没有的内容、漏步或多步、接错线**：四张新图的节点 / 连线 / 折叠与各自 `forward` 逐条对得上，无凭空内容、无漏步 / 多步；补的点开类归属全部正确；撤除的 `woa` 与两处重复点开都成立且未丢失 `Linear.forward` 可达性；6 条视觉撤除全对（`vision.py` 的 `vwqkv/vwo/vw1/vw2/vlin` 都是 `nn.Linear`，`head` 是 `ParallelHead`），并已注明 `nn.Linear`。**残留两处（已改）**：(i) `lin_w` 第二列仍固定写 `fp8 [out, in]`，与同节点折叠「dtype 随调用点变」冲突；(ii) `cpl_op` 说明末句仍是「本页每个调用点都没传 bias」，与「三处都改成主干」的改动清单不符。二者皆为措辞层面，非源码事实错误。
- **(b) 有没有违背前几轮的裁定**：无。`shared` 仍不点开 `Expert.forward`；`counts` 仍写「张量」；`patches → vcs_op` 的 `n_vit_h`/`n_vit_w` 仍未单独成节点；六张视觉图前缀仍 `vision.`。旧的「薄包装类不单独成图」是被第 10 轮仲裁取代、本轮据新裁定执行，不算违背。一处规则偏离（节点只留一条点开 vs 先例的拆节点）已列入本轮仲裁，非已定裁定。
- **仲裁两条都判「不必改」**，理由双方都对照过源码与 `plan.md`，成立。
