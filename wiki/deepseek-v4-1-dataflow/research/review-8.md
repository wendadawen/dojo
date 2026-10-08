# 第 8 轮

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
