# 第 7 轮

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
