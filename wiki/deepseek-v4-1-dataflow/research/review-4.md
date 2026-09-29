# 第 4 轮

源码位置核查通过：275 个节点。有结论的席位 5 个。这一轮有事实错误，不通过。

## 审查者 1

- 事实错误｜Attention.forward 折叠 fsl｜公式 `$freqs\_cis = freqs\_cis[...]$` 左右同名，右边是整表 `self.freqs_cis`，左边是切片后的局部量｜model.py:767；review.md「同一个符号对应源码里两个量」｜改成 `$freqs\_cis = self.freqs\_cis[start\_pos : start\_pos + seqlen]$`
- 事实错误｜Engram.forward 面板「写入」｜`$h = h + gate \odot value$` 形状对不上，缺少 unsqueeze，与折叠 nadd 不一致｜model.py:365｜改成与 nadd 一致的广播
- 措辞｜Indexer.forward 折叠 red｜左侧 `index_score` 与右侧 `index_score_i` 同名不同秩｜model.py:556-557｜右侧改用求和前的点积
- 措辞｜Block.forward 折叠 mixes_a / mixes_f 与面板｜把填进 comb 的一段叫做 `logits`，源码无此名，且与词表 logits 混淆｜kernel.py:430-431｜改称 softmax 之前的那一块
- 措辞｜Attention.forward 节点 qr_op、qb、wkv_win、wb 等｜第二行写成 `5120 → 1280`，同图其它算子写全形状｜plan.md｜写成含 `[B, T, …]` 的形状
- 措辞｜Attention.forward / NgramHashState.forward 节点 widx、pos、lens｜第二行左侧不是张量形状｜plan.md｜左侧写成由哪些量生成
- 措辞｜Attention.forward 折叠 cat2 说明｜写加 offset 发生在 `_compress_kv`｜model.py:580｜改成 Indexer.forward 的 shift
- 措辞｜MoE.forward 折叠 mexp 说明｜易读成每个 token 只过 1 个专家｜model.py:895-900｜改成当前专家这一步，一个 token 可出现在多个专家的 top-6 里
- 措辞｜NgramHashState.forward 节点 tk 第三行｜「第 k 列是往前看 k 个位置」在 k=0 时不直观｜engram.py:171-175｜改成第 k 列对应 shift=k

## 审查者 2

无

## 审查者 3

无

## 审查者 4

- 事实错误｜无
- 措辞｜Attention.forward 的 `fcall` 与 Indexer.forward 的 `fcr`｜同一份 `freqs_cis` 用了不同 id｜plan.md；model.py:733-734、767｜统一为 `fcall`
- 措辞｜Attention.forward 缓存 `wcache` 第三行｜写「prefill 整块覆盖」，前提 T≥1024 时是把最后 win 个 token 按 cutoff 写入环缓冲｜model.py:712-715｜改成按环序写入
- 措辞｜Engram.forward 面板「写入」｜省略了 unsqueeze，和折叠不一致｜model.py:365｜与折叠对齐
- 措辞｜MoE `mback`、RMSNorm `rback`、Engram `ncast`、Expert `w2`、Compressor `cnorm`｜用了输入的 dtype 或 shape，却没连回对应张量｜plan.md；model.py:904、293、365、851、485｜补 dtype/shape 连线

## 审查者 5

- 措辞｜Transformer.forward / ngram、eng、layers｜没有折叠「公式」｜model.py:1252、1262-1263、1267｜补对本行调用的公式
- 措辞｜Block.forward / attn、moe_op｜没有折叠「公式」｜model.py:985、992｜补调用公式
- 措辞｜Attention.forward / comp_op、cidx_op｜没有折叠「公式」｜model.py:747、750｜补调用公式
- 措辞｜MoE.forward / gate_op、mexp｜没有折叠「公式」｜model.py:892、900｜补调用公式
- 措辞｜Engram.forward 面板「写入」｜省略份维广播｜model.py:365｜与 nadd 对齐

## 修改

对照源码改过，并重新生成了页面。源码位置再查过，275 个节点。

- fsl 改成 `$freqs\_cis = self.freqs\_cis[start\_pos : start\_pos + seqlen]$`。
- Engram 面板「写入」补上 `unsqueeze(gate, -1)` 和 `value.float().unsqueeze(-2)`。
- Sinkhorn 输入不再叫 logits，改称 u，并写明源码里没有这个名字。
- Indexer 的频率表 id 从 fcr 改成 fcall。
- 补了 dtype 或 shape 连线：ax → cnorm、mx → mback、ex → w2、nx → ncast、rx → rback。
- wcache 的写法改成 prefill 按环写入。cat2、mexp、tk、red 的说法按源码改了。
- 只写了末维的第二行改成带 batch 和时间维的形状。widx、pos、lens 的左边改成「由哪些量生成」。
- 给 ngram、eng、layers、attn、moe_op、comp_op、cidx_op、gate_op、mexp 补了这一行调用的公式。
