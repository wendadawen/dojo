# 第 2 轮

源码位置核查通过：274 个节点。有结论的席位 4 个。审查者 1 未计入。这一轮有事实错误，不通过。

## 审查者 1

未计入

## 审查者 2

- 事实错误｜MoE.forward 节点 `counts`｜把 `counts` 写成 `张量`、`int64 [384]`，但源码在同一行对 `bincount` 做了 `.tolist()`，`counts` 是 Python list，只供 `if counts[i] == 0: continue` 使用｜`model.py:894`｜改成注明是 list/计数，或画未 `.tolist()` 的 `bincount` 输出并改名；不要标成 `int64` 张量
- 事实错误｜Expert.forward 节点 `emw` / `wts`，以及面板公式「SwiGLU」｜图上无条件画了 `weights * x`，面板写成 `$x = weights \odot \mathrm{silu}(gate) \odot up$`；但共享专家调用是 `self.shared_experts(x)`，不传 `weights`，`if weights is not None` 整段不执行。同一张图又被 `shared 点开 → Expert.forward` 使用｜`model.py:849-851`、`model.py:903`｜按路由专家一条路径画（共享在 MoE 侧注明不进 `emw`），或拆开/标明共享跳过；面板不要无条件带 `weights`
- 事实错误｜Block.forward 折叠 `mixes_a` / `mixes_f`，以及面板「pre、post、comb」｜同一处既写 `$comb=\mathrm{Sinkhorn}(\mathrm{softmax}(\cdots)+hc\_eps)$`（softmax 在 Sinkhorn 外），说明又写「Sinkhorn 先做一次行归一化（softmax 后加 hc_eps）」；与 `kernel.py` 中先对整行 softmax+eps、再列归一化、再交替 19 次的实现也不一致｜`kernel.py:436-458`｜写成对第 `j` 行 logits 做 softmax 后加 `hc_eps`，再列归一化并迭代；或 `comb=\mathrm{Sinkhorn}(logits)` 并在 Sinkhorn 定义里包含首次行 softmax，与说明、源码一致
- 措辞｜Engram.forward 算子 `nemb`、`nwkv`｜查表与线性投影写得出公式却未写｜`model.py:353`｜补 `$e=\mathrm{embed}(hash\_ids)$` 再 flatten，以及 `$kv=e\,wkv.weight^\top$`
- 措辞｜Attention.forward 算子 `cat`、`cat2`、`grp`｜简单 concat/view 可写公式却未写｜`model.py:777-785`｜补对应公式
- 措辞｜Attention.forward 算子 `widx` 第二行｜写成 `T, window_size → int32 [B, T, 128]`，左边不是张量形状｜`model.py:720`｜改成 `— → int32 [B, T, 128]` 或注明由 `bsz/seqlen/window_size` 生成
- 措辞｜Attention.forward 折叠 `widx` 说明｜prefill 下标是完整序列 KV 中的位置，不是 ring cache 槽位；写「窗口槽位」易歧义｜`model.py:716-720`、`model.py:417-420`｜改成 prefill 索引的是本次 `window_kv`（长度为 T）里的因果窗口位置
- 措辞｜Engram.forward 算子 `nwkv` 第一行｜仍写整式 `self.wkv(self.embed(...).flatten(-2))`，与上游 `nemb`→`ev` 重复｜`model.py:353`｜第一行改为 `self.wkv(ev)` 或等价，与连线一致
- 措辞｜MoE.forward 算子 `flat`、`y0`、`mback` 等｜view / zeros / cast 可写短公式却未写｜`model.py:891-904`｜能写的补一行公式

## 审查者 3

- 事实错误｜Block.forward 折叠 mixes_a / mixes_f，及面板「pre、post、comb」｜$comb$ 写成 $\mathrm{Sinkhorn}(\mathrm{softmax}(\cdot)+hc\_eps)$，说明又写 Sinkhorn「先做行归一化（softmax 后加 hc_eps）」；若输入已是 softmax+eps，再按说明会重复行归一化，与「共行 20、列 20」也对不上｜`kernel.py:430-458`：对 logits 先 softmax+eps，再列归一化，再交替行/列各 19 次｜改成：$comb=\mathrm{Sinkhorn}(logits)$，$logits=mixes\cdot hc\_scale_2+hc\_base$；写明 Sinkhorn = 行(softmax+eps)→列→交替各 19 次；面板同步改
- 事实错误｜NgramHashState.forward 折叠 off｜公式写成 $(rolling\bmod primes)+offsets$，把上一行 `mod` 的计算算进本行｜`engram.py:183-184`：`mod` 已做 `% primes`，`off` 只做 `cat(hashes)+offsets`｜改成：$hash=hs+offsets$（或 $hash=\mathrm{cat}(hashes)+offsets$）
- 事实错误｜Expert.forward 算子 emw；MoE 的 shared 点开到同一张图｜图画了无条件的 `weights * x`；共享专家 `shared_experts(x)` 不传 weights，该步不执行｜`model.py:849-850`、`903`｜改成：emw 标明仅路由专家；或 shared 另开/注明跳过 weights 乘法
- 措辞｜Engram.forward 算子 nemb、nwkv、spl、ekey｜写得出查表/投影/切分/unflatten 公式却没写｜`model.py:353-355`；review.md「写得出公式却没写的，记成措辞」｜补上对应公式
- 措辞｜Attention.forward 算子 cat、cat2｜写得出 concat 公式却没写｜`model.py:777-778`｜补上公式
- 措辞｜MoE.forward 算子 bc｜第一行漏了 `.tolist()`，counts 实际是 Python list｜`model.py:894`｜第一行补 `.tolist()`，第三行或说明写明是 list
- 措辞｜Attention.forward cidx_op；Indexer.forward shift｜offset（窗口 KV 长度 T）参与计算，但没有从 wchunk（或等价节点）连到该算子｜`model.py:776`、`580`；plan.md「用到的张量都要连线」｜补 `wchunk → cidx_op：offset`（Indexer 侧标明 offset 来自窗口长度）
- 措辞｜Expert.forward 面板「SwiGLU」｜公式始终含 $weights\odot$，共享专家不适用｜`model.py:849-850`｜注明仅路由专家；共享专家无此项

## 审查者 4

- 事实错误｜NgramHashState.forward 折叠 `off` 公式｜公式写成 `(rolling mod primes) + offsets`，把上一行 `mod` 的计算写进来了｜engram.py:183-184；review.md：公式只含这一行自己的计算｜改成 `$hash = hs + offsets$`，并说明 `hs` 是拼接后的桶下标、`offsets` 是各桶起始行
- 事实错误｜MoE.forward 折叠 `acc` 公式｜公式含 `expert(...)`，与已拆出的 `mexp`/`eout` 重复｜model.py:900；边为 `mexp → eout → acc`｜改成只写累加，如 `$y[idx] = y[idx] + eout$`
- 事实错误｜MoE.forward 折叠 `madd` 公式｜公式含 `shared_experts(x)`，与已拆出的 `shared`/`sv` 重复｜model.py:903；边为 `shared → sv → madd`｜改成只写累加，如 `$y = y + sv$`
- 措辞｜Engram.forward 算子 `nemb`、`nwkv`｜写得出查表与线性投影公式却没写｜model.py:353；review.md：写得出却没写记措辞｜`nemb` 补查表+flatten 公式；`nwkv` 补 `$kv = \mathrm{flatten}(embed)\, wkv.weight^\top$`，第一行改为只写 `self.wkv(...)`，不要再写一遍 embed
- 措辞｜Attention.forward 算子 `cat`、`cat2`、`woa`、`grp`｜拼接/变形写得出公式却没写｜model.py:777-786｜在折叠补对应公式
- 措辞｜MoE.forward 算子 `flat`、`bc`｜`view`/`bincount` 写得出公式却没写｜model.py:891, 894｜在折叠补公式
- 措辞｜Engram.forward 算子 `spl`、`ekey`｜`split`/`unflatten` 写得出公式却没写｜model.py:354-355｜在折叠补公式
- 措辞｜Block.forward 面板 / `mixes_*` 折叠里 `comb` 公式｜写成 `Sinkhorn(softmax(单个 logit)+eps)`，像对标量做 softmax｜kernel.py:430-448 是先填满 `comb` 矩阵再做行 softmax｜改成先写由 `mixes` 填矩阵，再写对该矩阵做 Sinkhorn（首步为行 softmax）
- 措辞｜Expert.forward 节点 `emw` / 面板 SwiGLU｜`shared` 也点开到此图，但共享专家不传 `weights`、不走乘法｜model.py:849-850, 903｜说明里写清「仅路由专家」；面板公式区分有无 `weights`
- 措辞｜Expert.forward 折叠 `g1` 说明｜写「先转 fp32 再夹」，夹实际在 `cl`/`cl2`｜model.py:843-847｜`g1` 说明只保留转 fp32；夹放到 `cl`/`cl2`

## 审查者 5

- 事实错误｜Expert.forward 节点 emw / 连线 hm→emw→w2 / 面板「SwiGLU」｜把 `if weights is not None` 画成必经节点；`shared` 与 `mexp` 都点开同一张图，共享专家调用时 weights 为 None，该步不执行，面板公式却始终含 `weights ⊙`｜model.py:849-850、903；review.md「把 if 画成了节点」｜路由专家保留 emw；共享专家改为 hm→w2，或拆图；面板按两条路径分开写
- 事实错误｜Expert.forward 折叠 w2 公式｜写「转回存储 dtype」，源码是 `dtype = x.dtype` 再 `x.to(dtype)`，回到的是输入激活的 bf16，不是权重 fp4/fp8 存储类型｜model.py:842、851｜改成「转回输入 x 的 dtype（bf16）」
- 事实错误｜MoE.forward 节点 bc、counts｜bc 第一行只有 `bincount`，counts 标成 int64 张量；源码同一行有 `.tolist()`，`counts` 是 Python list｜model.py:894｜bc 补上 `.tolist()`，counts 改为 list 长度 384；或把 tolist 拆成单独一步
- 措辞｜摘要｜「窗口 KV 与压缩 KV 都按 FP8、FP4 精度量化」易读成两者都用两种精度｜摘要原文；Attention 里窗口 fp8、压缩 fp4｜改成「窗口 KV 按 FP8、压缩 KV 按 FP4 量化再反量化」
- 措辞｜Engram.forward 折叠 nemb、nwkv｜查表与线性有公式可写却未写｜model.py:353；review.md「写得出公式却没写的，记成措辞」｜nemb 补嵌入反量化公式，nwkv 补 $kv = \cdot\, wkv.weight^\top$
- 措辞｜Attention.forward 折叠 wquant、cquant；Indexer.forward 折叠 kq、iq_q｜量化/反量化有公式可写却未写｜model.py:707、760、546、552｜补原地 quant→dequant 公式
- 措辞｜Attention.forward 连线；Indexer.forward 连线｜cf、fpos 用了 x 的 seqlen，未按 plan 从 ax 连「形状」线｜model.py:753-757、539-543；plan.md「只用到形状也要连线」｜补 `ax → cf：形状（seqlen）`、`ax → fpos：形状（seqlen）`

## 修改

对照源码改过，并重新生成了页面。源码位置再查过，274 个节点。

- Sinkhorn：`comb` 改成先由 mixes 填 `logits`，再 `$comb = \mathrm{Sinkhorn}(logits)$`。行 softmax 加 hc_eps、列归一化、再交替 19 次，写进说明。mixes_a、mixes_f 和面板一起改。
- `off` 的公式改成 `$hash = hs + offsets$`。面板「桶」把取模和加偏移分成两步。
- `acc` 只写 `$y[idx] = y[idx] + eout$`，`madd` 只写 `$y = y + sv$`。
- `bc` 补上 `.tolist()`。`counts` 改成 list，长度 384。
- Expert.forward 按路由专家画，前提里写明。`emw` 留在这张图。`shared` 不再点开这张图，折叠里写明这次不传 weights、不做这一次乘法。面板 SwiGLU 把乘 weights 写成路由专家才有的一步。
- `w2` 改成转回进入 Expert 时 `x` 的 dtype（bf16）。`g1` 的说明不再把夹取算进这一行。
- 摘要改成窗口 KV 按 FP8、压缩 KV 按 FP4。
- 补上审查里点名、写得出公式的算子：nemb、nwkv、spl、ekey、cat、cat2、grp、woa、flat、y0、mback、bc、wquant、cquant、kq、iq_q。`nwkv` 第一行改成 `self.wkv(ev)`。`widx` 第二行改成由 bsz、seqlen、window_size 生成。`widx` 说明改成这次 window_kv 里的位置，晚于该 query 的写成 −1。补了 `wchunk → cidx_op：长度（offset）`、`ax → cf：形状（seqlen）`、`ax → fpos：形状（seqlen）`。
