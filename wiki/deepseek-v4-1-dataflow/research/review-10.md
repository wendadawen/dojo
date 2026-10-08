# 第 10 轮

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
