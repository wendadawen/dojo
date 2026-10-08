# 第 9 轮

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
