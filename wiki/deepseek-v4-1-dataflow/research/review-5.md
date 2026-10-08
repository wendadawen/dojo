# 第 5 轮

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
