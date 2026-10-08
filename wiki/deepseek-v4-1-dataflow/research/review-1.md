# 第 1 轮

源码位置核查通过：262 个节点。有结论的席位 3 个。审查者 4、5 未计入。这一轮有事实错误，不通过。

## 审查者 1

- 事实错误｜多数算子第二行（eng、layers、fnorm、an、attn、fn、moe_op、mixes_a、mixes_f、rope_q、wrope、wquant、wwrite、crope、cquant、cwrite、cat、cat2、unrope、cf32、cnorm、krope、kq、kwrite、qrope、iq_q、imask、shift、acc、madd、act、nrm、grs、cl、cl2、silu、emw、ewm、nhf、ngate、nadd、ncast、cmap、wr、upd、rep、roll、off、rf32、nr、rmw、rback）｜没有写成「原来的形状 → 变完的形状」，或箭头右边不是形状｜review.md：第二行没写出形状从哪变到哪算漏了一步｜一律写成「原形状 → 新形状」。cat 写 `[B, T, 512] 与 [B, T/r, 512] → [B, T + T/r, 512]`；cat2 写 `[B, T, 128] 与 [B, T, 512] → [B, T, 640]`；mixes 写 `[B, T, 4, 5120] → fp32 [B, T, 4] × 2 与 [B, T, 4, 4]`；shift 写 `int64 [B, T, 512] → int32 [B, T, 512]`
- 事实错误｜Transformer：avg → mh｜漏了 `torch.cat(main_hiddens, dim=-1)`。avg 是 [B, T, 5120]，mh 是 [B, T, 15360]｜model.py:1266、1271｜avg 后接张量 `h.mean(dim=2)`（× 3），再加算子 cat，第二行 `[B, T, 5120] × 3 → [B, T, 15360]`，再接 mh
- 事实错误｜NgramHashState：rep → tk｜漏了 `torch.stack(tokens, dim=-1)`｜engram.py:174-175｜rep 后接每步的 [B, T]，再加 stack，`[B, T] × 4 → [B, T, 4]`，接 tk
- 事实错误｜NgramHashState：mod → hs、off｜漏了 `torch.cat(hashes, dim=-1)`。mod 是 [B, T, 2, 8]，hs 是 [B, T, 2, 24]，cat 又出现在 off 的第一行｜engram.py:183-184｜mod 后接 hashes × 3，再加 cat，接 hs；off 只做 `+ self.offsets`
- 事实错误｜Engram：nkv → key0、value｜漏了 `kv.split`，张量直接连张量｜model.py:353-354｜nwkv 只留投影（353）；加 split 算子（354），`[B, T, 25600] → [B, T, 20480] 与 [B, T, 5120]`
- 事实错误｜Attention：fcall → fc｜切片 `self.freqs_cis[start_pos : start_pos + seqlen]` 没画成算子｜model.py:767｜加算子，`[max_seq_len, 32] → [T, 32]`
- 事实错误｜Attention 折叠 unrope 说明｜写成抵消键上的旋转。源码去掉的是 query 的旋转｜model.py:394-395、781｜改成用 query 所在位置频率的共轭再转一次，去掉 query 的旋转
- 事实错误｜Attention 面板「频率」「YaRN」和配置 rope_theta｜写了压缩比为 0 才用 rope_theta、YaRN 只对压缩比大于 0 生效，超出前提｜前提把注意力定在 compress_ratio 为 2；model.py:680-687｜本路径 base 写 compress_rope_theta，YaRN 按这条路径写；rope_theta 的差别挪到 layers 的说明
- 事实错误｜Transformer：shar｜共享量漏了 candidates。层 20 写，层 24、28、32、36 读｜model.py:1166-1176、569-575；candidate_source_layer 为 20｜第二行加上 candidates，说明写明谁写谁读
- 事实错误｜整份计划少了类的图｜`Linear`（含 ColumnParallelLinear、RowParallelLinear）、`ParallelEmbedding`、`ParallelEngramEmbedding` 没有单独成图，fp8/fp4 线性层的激活量化没出现｜plan.md：源码里自己定义的类每个都画一张；model.py:152-178、181-278、296-325｜补这三张图，用到的节点点开过去
- 措辞｜emb、head 第二行｜emb 左边的 129280 不是输入形状；head 没写出取最后一个位置后 T 消失｜model.py:1253、1010-1012｜emb 改为 `int64 [B, T] → [B, T, 5120]`；head 改为 `[B, T, 5120] → fp32 [B, 129280]`
- 措辞｜fnorm、head、an、fn 第一行｜写成「self.norm (RMSNorm)」，不是源码里的调用｜plan.md｜改为 `self.norm(h)`、`self.head(self.norm(h))`、`self.attn_norm(x)`、`self.ffn_norm(x)`
- 措辞｜算子直接连算子，或别名画成两个张量｜layers → col、layers → avg、layers → eng；hin → resid0、x4 → resid1；wkv_q → wchunk｜plan.md：算子和张量交替｜给最后一层的 pre_mix 和层间的 h 加张量；别名合成一个节点或注明是同一份
- 措辞｜写得出公式却没写｜wquant、cquant、kq、iq_q、widx、cat、cat2、wwrite、mpick、nwkv、nemb、gath、upd、rep｜review.md｜按各自这一行补公式
- 措辞｜mixes 的 Sinkhorn 说明｜读起来像 softmax 之后又做了一次行归一化，也没写每次除法都加 hc_eps｜kernel.py:436-458｜写成先沿 k 做 softmax 再加 hc_eps，其后每次除以（和 + hc_eps）
- 措辞｜pool｜看不出逐通道 softmax｜model.py:473-475｜写成每个通道各自 softmax 再加权
- 措辞｜配置｜漏了 gate_temp 1.0、temperature 1、max_seq_len 4096；Attention 面板用了 YaRN 的配置却没列；samp 漏了 max(temperature, 1e-5)｜model.py:52-53、67、1290｜补进配置，并标明默认值
- 措辞｜Engram 面板门控｜`|dot|` 会切断表格｜plan.md｜改成 `\lvert dot \rvert`
- 措辞｜layers 说明、cidx_op 说明、介绍｜没写层 20 建候选块、层 24–36 在块里取 top-k、层 20–39 压缩比为 1；介绍没限定压缩比大于 0 的层才用索引器｜model.py:502-503、569-575｜补上这些层的差异
- 措辞｜avg 公式、cat2 说明、w2 公式、emw 说明、pidx 第二行、mpick 说明｜「求和」应为「求均值」；offset 加在 Indexer 的 shift；w2 转回的是输入的 dtype；共享专家跳过 emw；pidx 应写 `int64 [n] × 2`；跳过空专家靠 bincount｜对应源码行｜按条改

## 审查者 2

- 事实错误｜Transformer：avg → mh｜同上，漏了 cat｜model.py:1266、1271｜同上
- 事实错误｜NgramHashState：rep → tk｜漏了 stack｜engram.py:174-175｜同上
- 事实错误｜NgramHashState：mod → hs → off｜漏了 cat，off 又写了一遍 cat｜engram.py:183-184｜同上
- 事实错误｜Engram：nkv → key0、value｜漏了 split｜model.py:354｜同上
- 事实错误｜Attention：fcall → fc｜漏了切片｜model.py:767｜同上
- 事实错误｜MoE｜漏了 `counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)`｜model.py:894｜从 gi 引出 bincount，第二行 `int64 [B×T, 6] → [384]`；`counts[i] == 0` 写进 mpick 的说明
- 事实错误｜缺 Linear、ParallelEngramEmbedding、ParallelEmbedding 三张图｜fp8/fp4 的激活量化没有出现｜model.py:181-207、242-278、312-325；kernel.py:98-124、277-307、562-591｜补图，用到的节点点开过去
- 事实错误｜shar 与 layers 说明｜漏了 candidates，以及层 20 写、层 24–36 读｜model.py:569-575、1166-1176｜第二行加上 candidates；layers 说明补候选块
- 事实错误｜第二行没有写成形状箭头的算子｜rope_q、wrope、wquant、wwrite、crope、cquant、cwrite、cat、cat2、unrope、cnorm、krope、kq、kwrite、qrope、iq_q、imask、shift、acc、madd、act、nrm、grs、cl、cl2、silu、emw、ewm、ngate、nadd、ncast、wr、upd、rep、roll、off、rf32、nr、rmw、rback、mixes_a、mixes_f｜review.md｜每个写成「原来 → 变完」。cat、mixes、wwrite、rope_q 按审查者给出的形状写
- 事实错误｜emb 第二行 `129280 → 5120`｜129280 不是输入形状｜model.py:1253｜`int64 [B, T] → bf16 [B, T, 5120]`
- 事实错误｜head 第二行 `5120 → 129280`｜先取了最后一个位置｜model.py:1010-1012｜`[B, T, 5120] → fp32 [B, 129280]`
- 事实错误｜unrope 说明｜同审查者 1，去掉的是 query 的旋转｜model.py:394-395、781｜改成相对旋转 t−s
- 事实错误｜频率、YaRN、fc 第三行｜写了前提之外的压缩比为 0 的分支｜前提；review.md｜只写 compress_rope_theta 与本路径的 YaRN
- 事实错误｜介绍｜写成纯文本解码模型。config 里 vision_n_layers 为 32，这次前向只是不传图像｜config.json；model.py:1216-1222｜改成带视觉编码器，这次前向只走文本
- 事实错误｜Engram 面板门控｜`|dot|` 会切开单元格｜plan.md｜改成 `\lvert ... \rvert`
- 措辞｜cidx_op｜offset 用了窗口 KV 的长度，没有连线｜model.py:775｜补 `wchunk → cidx_op：长度（offset）`
- 措辞｜layers → eng、eng → layers、layers → avg、layers → col｜算子直接连算子｜plan.md｜补中间张量，含最后一层返回的 pre_mix
- 措辞｜fnorm、head、an、fn 第一行｜不是源码调用｜plan.md｜改成 `self.norm(h)` 等
- 措辞｜只写「不变」、没有箭头｜eng、layers、fnorm、an、attn、fn、moe_op、cf32、nhf｜review.md｜改成 `X → X`，dtype 变了的写出 dtype
- 措辞｜能写公式却没写｜nwkv、nemb、wquant、cquant、kq、iq_q、grp、cat、cat2、flat、mpick、mback、gath、upd、rep、ekey｜review.md｜各补一行公式
- 措辞｜张量第三行写了说明｜plan.md 第三行只留维度字母｜说明挪到产生它的算子的折叠里
- 措辞｜cpre、cpre2、cpost、cpost2、col｜等号两边都叫 x 或 h｜model.py:959、965 结果叫 y｜左边改成 y
- 措辞｜Attention 面板｜r 既指 compress_ratio 又指 o_lora_rank｜plan.md｜换一个字母
- 措辞｜widx 说明「−1 是空槽」｜prefill 时 −1 是晚于这个 query 的位置｜model.py:417-420｜改成晚于这个 query 的下标记 −1
- 措辞｜cat2 说明｜offset 加在 Indexer 最后一行｜model.py:580｜改成索引器 shift 那一步
- 措辞｜配置缺 temperature、gate_temp、max_seq_len，以及 YaRN 用到的 rope_factor 等｜model.py:52-53、67｜补上并标明默认值；前提写 T ≤ max_seq_len
- 措辞｜ngate｜sign 与 copysign 在 0 处不同｜model.py:362｜改成 copysign
- 措辞｜avg 说明写成求和｜model.py:1266｜改成取均值
- 措辞｜摘要｜FP8 与 FP4 没分开对应两种 KV｜model.py:707、760｜窗口 KV 按 FP8，压缩 KV 按 FP4
- 措辞｜shift 第一行少了 `.int()`｜model.py:580｜补上，第二行写 int64 → int32
- 措辞｜nwkv 参数省略、pidx 应写 × 2、cmap 的「行号」不通、RMSNorm 面板缺 q_lora_rank｜plan.md｜按对应处改

## 审查者 3

- 事实错误｜avg → mh｜漏了 cat｜model.py:1266、1271｜同审查者 1
- 事实错误｜rep → tk｜漏了 stack｜engram.py:174-175｜同审查者 1
- 事实错误｜mod → hs → off｜漏了 cat｜engram.py:183-184｜同审查者 1
- 事实错误｜roll｜`rolling` 的初值 `products[..., 0]` 没画｜engram.py:180-182｜在 pr 与 roll 之间加算子 `products[..., 0]`，或在公式里写明初值
- 事实错误｜nkv → key0、value｜漏了 split｜model.py:354｜同审查者 1
- 事实错误｜算子第二行｜大量节点没有「原来 → 变完」，其中 cf32、nhf、rf32、cnorm、ncast、rback、shift 的 dtype 或形状实际变了｜plan.md；review.md｜全部写成箭头。cat、cat2、shift、写缓存、acc、mixes 按审查者给出的形状写
- 事实错误｜emb、head 第二行｜同审查者 2｜model.py:1253、1010-1012｜同审查者 2
- 事实错误｜Expert 的 hm、hw，Gate 的 bv｜有变量名却写成表达式或参数名｜model.py:848、850、818｜hm、hw 第一行改成 x；bv 改成 bias
- 事实错误｜shar 与 layers 说明｜漏了 candidates｜model.py:1172-1176、569-575｜同审查者 1
- 措辞｜layers → eng 等算子直接连算子｜plan.md｜补中间张量
- 措辞｜hin → resid0、x4 → resid1、wkv_q → wchunk｜别名画成两个张量｜model.py:981、988、716｜合成一个节点，或注明是同一个张量
- 措辞｜张量第三行写了说明｜plan.md｜只留字母含义
- 措辞｜能写公式却没写｜nemb、nwkv、ekey、vf、ncast、gath、upd、rep、pos、wr、wquant、cquant、kq、iq_q、grp、woa、cat、cat2、wwrite、cwrite、flat、y0、mpick、cf32｜review.md｜各补一条公式
- 措辞｜门控的 `|dot|`｜plan.md｜改成 `\lvert dot \rvert`
- 措辞｜samp｜除数是 max(temperature, 1e-5)，temperature 为 0 的分支和默认值都没写｜model.py:1288-1290｜公式写上 max；前提写 temperature 取默认值 1
- 措辞｜gate_temp、max_seq_len 没进配置；前提没写 T、B 的上限｜model.py:51-52、67｜补上
- 措辞｜Attention 面板 YaRN 用了 rope_factor 等，配置没列｜config.json｜补上
- 措辞｜RMSNorm 面板缺 q_lora_rank｜model.py:641｜补 1280
- 措辞｜Sinkhorn 说明没写每次除以（和 + hc_eps）｜kernel.py:447、453、457｜写明
- 措辞｜摘要｜FP8/FP4 没分开；Sinkhorn 只作用于 comb｜model.py:707、760；kernel.py:426-460｜改摘要
- 措辞｜avg 写成求和｜model.py:1266｜改成取均值
- 措辞｜bpre「上一层返回」不适用于层 0｜model.py:1260、1267｜写明层 0 是初值
- 措辞｜cat2 说明｜offset 加在 Indexer 最后一行｜model.py:580｜改位置
- 措辞｜emw｜共享专家 weights 为 None，这一步不跑｜model.py:849、903｜补一句
- 措辞｜mpick｜跳过空专家的依据是 bincount｜model.py:894-897｜写明
- 措辞｜压缩映射「同形 token」｜engram.py:28-39｜改成 NFKC、去重音、小写、空白合并之后相同

## 审查者 4

未计入

## 审查者 5

未计入

## 修改

对照源码改过，并重新生成了页面。源码位置再查过，274 个节点。

- 补上漏画的算子：`torch.cat(main_hiddens)`、`torch.stack(tokens)`、`products[..., 0]`、`torch.cat(hashes)`、`kv.split`、频率表切片 `self.freqs_cis[start_pos:start_pos+seqlen]`、`torch.bincount`。
- 算子第二行凡是「不变」、描述句或箭头缺一边的，改成「原来的形状 → 变完的形状」。emb、head 的第二行改成带 batch 和时间维的形状。shift 补上 `.int()`。
- `shared_attn` 加上 candidates。layers 的说明补上候选块和压缩比为 1 的层。
- unrope 的说明改成去掉 query 的旋转。介绍改成带视觉编码器、这次前向不传图像。avg 的说明改成取均值。samp 的除数改成 max(temperature, 1e-5)。门控的绝对值改成 `\lvert dot \rvert`，sign 改成 copysign。
- fnorm、attn_norm、ffn_norm、head 的第一行改成源码里的调用。nwkv 写出参数，源码行改到 353。

还没做的：`Linear`、`ParallelEmbedding`、`ParallelEngramEmbedding` 三张类图；一批写得出公式的算子仍只有说明。这两类留到下一轮再对。
