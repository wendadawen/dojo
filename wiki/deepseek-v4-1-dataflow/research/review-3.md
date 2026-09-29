# 第 3 轮

源码位置核查通过：275 个节点。有结论的席位 5 个。这一轮有事实错误，不通过。

## 审查者 1

- 事实错误｜Engram.forward 折叠 `nemb` 公式｜公式写成 `values.float().unflatten(-1,(n,32)) ⊙ scales.float()`，缺少 `scales.unsqueeze(-1)`，且符号 `n` 未定义｜`model.py:319`｜改成 `unflatten(-1,(-1,32))` 乘 `scales.float().unsqueeze(-1)`，并注明 32 是 `fp8_block_size`
- 措辞｜Attention 算子 `widx`｜写得出窗口下标公式却未写｜`model.py:417-426`｜补 prefill 下标公式
- 措辞｜NgramHashState 算子 `gath`/`upd`/`rep`/`pos`｜未写公式｜`engram.py:169-174`｜各补一行
- 措辞｜MoE 算子 `mpick`｜`torch.where(indices==i)` 未写公式｜`model.py:899`｜补公式
- 措辞｜MoE 算子 `shared`｜同是 `Expert`，未点开到 Expert.forward｜`model.py:903`；plan.md「用到它的节点都点开」｜加 `shared 点开 → Expert.forward`
- 措辞｜Compressor 张量 `cout` 与 Attention/Indexer 的 `lat`｜同一压缩潜在量跨图 id 不一致｜plan.md；`model.py:485` 与 `747`｜将 `cout` 改为 `lat`
- 措辞｜NgramHashState `hashout` 与 Transformer `hash`｜同是 `engram_hashes`，id 不一致｜`engram.py:184` 与 `model.py:1252`｜统一为 `hash`
- 措辞｜Engram 算子 `nadd` 第一行｜已有 `vf` 做 `unsqueeze`，第一行仍写完整 `value.float().unsqueeze(-2)`｜`model.py:365`｜第一行改成 `h + gate.unsqueeze(-1) * v`
- 措辞｜MoE 张量 `counts`｜用于跳过空专家，却无出边｜`model.py:894-897`｜加 `counts → mpick`
- 措辞｜前提｜未写 `max_seq_len`｜`ModelArgs.max_seq_len=4096`｜前提补默认值 4096

## 审查者 2

- 事实错误｜Attention.forward 连线｜有算子 `woa` 与张量 `wav`，且 `wav → eins`，但缺少 `woa → wav`｜`model.py:786-787`｜补上 `woa → wav`
- 事实错误｜Engram.forward 折叠 `nemb` 公式｜写成 `values.unflatten(...) ⊙ scales.float()`，少了 `unsqueeze(-1)`｜`model.py:319`｜改成 `⊙ scales.float().unsqueeze(-1)`
- 措辞｜MoE.forward 算子 `mpick`｜写得出公式却没写｜`model.py:899`｜补公式
- 措辞｜Attention.forward 算子 `widx`｜写得出公式却没写｜`model.py:417-420`｜补窗口因果下标公式
- 措辞｜NgramHash.forward 算子 `gath` / `upd` / `rep`｜写得出公式却没写｜`engram.py:172-174`｜分别补公式
- 措辞｜Transformer.forward 算子 `eng`｜`eng → layers` 算子直接接算子｜`model.py:1262-1267`｜在 `eng` 后加输出张量再连进 `layers`

## 审查者 3

- 事实错误｜Attention.forward 连线｜没有 `woa → wav`，`woa` 悬空｜model.py:786-787｜补上 `woa → wav`
- 事实错误｜Engram.forward 折叠 nemb 公式｜`scales.float()` 少了 `.unsqueeze(-1)`，`n` 没定义｜model.py:319｜写成 `scales.float().unsqueeze(-1)`，`(n, 32)` 改为 `(-1, 32)`
- 措辞｜Attention.forward 算子 fsl｜没有折叠｜model.py:767｜补切片公式
- 措辞｜Attention.forward 标题行号｜写的是 765-789，图里却含 `_window_kv` / `_compress_kv`｜model.py:700-763｜标题改为含辅助函数，或注明图含这两段
- 措辞｜MoE.forward 张量 counts｜没连到后面｜model.py:894-897｜加 `counts → mpick`
- 措辞｜MoE.forward 算子 shared｜没有点开 Expert｜model.py:903｜补 `shared 点开 → Expert.forward`
- 措辞｜NgramHashState.forward 面板「哈希」「桶」｜读起来像先全部异或再取一次模｜engram.py:181-183｜改成每步异或后立刻 mod
- 措辞｜Attention.forward 算子 qr_op、wkv_win｜点开只到 RMSNorm，第一行还含线性投影｜model.py:770、705｜说明里写明点开只看 norm，或拆开

## 审查者 4

- 事实错误｜Engram.forward 折叠 nemb 公式｜漏了 `unsqueeze(-1)`，符号 `n` 未定义｜model.py:319｜改成 `scales.float().unsqueeze(-1)`，`n` 写成 8
- 事实错误｜Attention.forward 折叠 rope_q / wrope / crope / unrope；Indexer.forward 折叠 krope / qrope｜`view_as_complex` 直接作用在末 64 维上，源码先 `unflatten(-1, (-1, 2))`｜model.py:397｜公式补上 unflatten 与 flatten
- 事实错误｜Transformer.forward 节点 layers 源码列｜写成 `model.py:1204，1261-1267`，1204 是 `__init__`｜model.py:1204 与 1267｜去掉 1204，只留调用 Block 的行
- 措辞｜MoE.forward 连线｜`counts` 未连到 mpick/mexp｜model.py:896-897｜补 `counts → mpick`
- 措辞｜多处算子折叠缺公式｜Attention 的 wwrite / widx / cwrite，Indexer 的 kwrite，NgramHashState 的 wr / pos / blk0 / gath / upd / rep，MoE 的 mpick｜对应源码行｜补上公式
- 措辞｜Transformer.forward 节点 layers 源码范围｜写 1261-1267 会盖住已单独成节点的 eng 与 avg｜model.py:1261-1267｜layers 只标 Block 调用行

## 审查者 5

- 事实错误｜无
- 措辞｜Attention.forward / `widx`｜折叠里没写公式｜`model.py:417-420`｜补 prefill 下标公式
- 措辞｜NgramHashState.forward / `gath`、`upd`、`rep`｜写得出公式却没写｜`engram.py:172-174`｜分别补公式
- 措辞｜Engram.forward / `nemb` 公式｜符号 $n$ 未说明，未写出 `unsqueeze(-1)`｜`model.py:317-320`｜写明块数，并写成 `unsqueeze(-1)`
- 措辞｜Engram.forward / `nadd` 第一行｜与 `vf` 重复写了 `unsqueeze`｜`model.py:365`｜第一行改成 `h + gate.unsqueeze(-1) * v`

## 修改

对照源码改过，并重新生成了页面。源码位置再查过，275 个节点。

- 补上 `woa → wav`。
- `nemb` 的反量化改成 `unflatten(-1, (-1, 32))` 乘 `scales.float().unsqueeze(-1)`。32 是 fp8_block_size。
- rope_q、wrope、crope、unrope、krope、qrope 的公式补上 `unflatten(-1, (-1, 2))` 和后面的 flatten。unrope 仍对频率取共轭。
- `layers` 的源码列改为 `model.py:907，1267`。907 是 `class Block`，1267 是前向里的调用。去掉 `__init__` 的 1204，也不再把 eng、avg 的行算进这个节点。
- `eng` 后面加张量 `eh`，再连进 `layers`。
- `counts → mpick`。压缩器输出的 id 从 `cout` 改成 `lat`。NgramHash 的输出 id 从 `hashout` 改成 `hash`。`nadd` 第一行改成使用 `v`。前提补上 `max_seq_len` 默认值 4096。面板「哈希」「桶」改成每步异或后立刻取模。
- 补了这一轮点名的公式：fsl、widx、wwrite、cwrite、kwrite、wr、pos、blk0、gath、upd、rep、mpick，以及同类的 cf32、nhf、vf、ncast、rf32、rback。
- `shared` 仍不点开 Expert.forward。那张图是传入 weights 的路由专家，含 emw。共享专家不传 weights。前提里已经写了这一点。
