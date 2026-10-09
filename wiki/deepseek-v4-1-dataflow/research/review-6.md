# 第 6 轮

源码位置核查通过：377 个节点。有结论的席位 5 个。这一轮有事实错误，改完后重跑四道机械检查全过。

## 说明：这一轮的三件事与一次删图

`guides/dataflow/plan.md` 这一轮新加/改写了两条，本页据此重做：

1. **通用基础层不画**（plan.md 第 78 行）。原先按旧条文（「薄包装类也要单独画」）补进本页的 6 张图按新条文删掉：`ParallelEmbedding.forward`、`Linear.forward`、`ColumnParallelLinear.forward`、`RowParallelLinear.forward`、`PatchEmbed.forward`、`vision.MLP.forward`、`vision.RMSNorm.forward`、`RMSNorm.forward`、`ParallelHead.forward`（共 9 张）。保留 `ParallelEngramEmbedding`——它有查表 + fp8 反量化，不只是把通用算子套在权重上。删图后：指向它们的「点开」全删（37 条 → 14 条，净删 23 条）；折叠里「见点开的那张图」改成直接写算式（`emb`、`fnorm`、`head`、`vpe`、`vnorm`、`an`、`fn`、`qr_op`、`wkv_win`、`cnorm`、`kw`）。第 10/11 轮仲裁判「必须补这 4 张图」在当时是对的，已被新条文取代；本页不再据那两条补图。
2. **一个节点一步**（plan.md 第 81 行）。源码一行里套的类调用拆开，一步一节点；普通函数/方法不单独成节点；残差相加把 `+` 单独成节点。
3. **折叠「说明」里不写公式**（`build_page.py` 的机械检查）。全页扫过，无残留（生成器不再报）。
4. **第一张图面板的公式一条拆两行**（公式一行、解释另起一行，单元格里用 `<br>`）。只改 Transformer 图。

拆节点的写法（本轮定的口径）：算子第一行写**外层那一次类调用**，内层实参省略成 `...`（如 `self.q_norm(...)`、`self.aligner(...)`）；被省略的实参由它前面那张张量节点承接（该张量第一行照源码或照产生它的表达式写）。这样既不把两层调用压在一个节点，也不用源码里不存在的中间变量名（`check_source.py` 的节点名核查会拒掉凭空名字）。`vlin` 同理写 `self.w2(...)`（`w1`/`gelu` 已在上一节点 `aw1`/`alg` 里）。

## 审查者 1（路径覆盖）

- 事实错误｜Compressor.forward 图（`cf32`/`projk`/`projg`/`pool`/`cnorm` 一带）｜前提写明层 20 的 `compress_ratio` 是 1、压缩器只留 `wkv` 一路，但图里只画了 ratio > 1 那一支（`cf32 → projk/projg → pool → cnorm`），层 20 的 `if ratio == 1: return self.norm(self.wkv(x))` 那一支既无节点也无说明｜model.py:462、485；前提「层 20 的 compress_ratio 是 1」｜把层 20 那一支也画进同一张图，让该图两支都可见

## 审查者 2（节点与形状）

- 写法｜Aligner.forward / `vlin`｜第一行用省略号代替实参｜plan.md「没有变量名就写产生它的表达式，不用省略号代替」｜写成完整表达式（见下「不改的几条」，与「一步一节点」冲突，保留省略号写法）
- 写法｜vision.Attention.forward / `vview`｜第一行的生成式把迭代对象写成 `...`｜vision.py:56 `for t in self.wqkv(x).chunk(3, dim=-1)`｜补全
- 写法｜vision.Block.forward / 连线 `vba1 → vbad`、`vbm1 → vbad2`｜线上写「上一层输出」，但它们是本层注意力/前馈子层的输出｜vision.py:83-84｜改成「注意力子层的输出」「前馈子层的输出」
- 写法｜vision.Block.forward / `vbad`、`vbad2`｜第二行把两个操作数写成 `bf16 [S, 1024] × 2`，没分别写两个操作数｜plan.md「算子第二行两边都写上完整的形状」｜写开成「bf16 [S, 1024] 与 bf16 [S, 1024] → …」
- 写法｜Transformer.forward / `shar`｜源码列 `model.py:1166-1180` 是 `SharedAttentionRuntime` 的类定义，实例在 1180 建立｜model.py:1166、1180｜改到 1180

## 审查者 3（连线与结构）

- 事实错误｜Transformer.forward / `layers`、`hl`、`lpm`｜循环回流没接上：`layers` 的残差输入只连了 `hc`（层 0）和 `eh`（层 1、14），`lpm` 也没连回 `layers`，环闭不上｜model.py:1261-1267｜加 `hl → layers`、`lpm → layers`
- 事实错误｜Attention.forward / `qr_op`｜第一行实参写成 `x`（同图 x 是层输入），实际传进去的是 `wq_a` 的输出｜model.py:770｜改成写 `self.q_norm(...)`
- 事实错误｜Attention.forward / `wkv_win`｜同上，实参应是 wkv 的输出｜model.py:705｜`self.kv_norm(...)`
- 事实错误｜Indexer.forward / `kw`｜同上，实参应是 wk 的输出｜model.py:544｜`self.k_norm(...)`
- 事实错误｜Aligner.forward / `vlin`｜同上，实参应是 gelu 之后的量｜vision.py:119｜`self.w2(...)`
- 事实错误｜Transformer.forward / `venc`、`alen`｜第一行的 `n_vit_h`、`n_vit_w` 没有连线｜model.py:1226｜补线并在线上写明
- 事实错误｜Aligner.forward / `vv`｜`x.view(n_h, n_w, -1)` 用到 `n_vit_h`、`n_vit_w`，没有连线｜vision.py:116｜加 `vout → vv：n_vit_h、n_vit_w`
- 事实错误｜vision.Attention.forward / `vwo`｜`reshape(n, -1)` 用的 `n = x.size(0) = S` 没有连线｜vision.py:60｜加 `vax → vwo：形状（S）`
- 事实错误｜vision.Block.forward / `vbc`｜同一对 cos、sin 在 ViT 图与 vision.Attention 图里都是 `vcs`，这里换了 id｜vision.py:82｜改用 `vcs`
- 事实错误｜Block（`mixes_a`、`mixes_f`）、Gate（`lg`）、Engram（`ewm`）、NgramHashState（`cmap`、`prod`、`mod`、`off`）｜第一行里当操作数直接出现的权重与缓冲没有节点、没有连线｜model.py:982、989、811；engram.py:356、164、179、183、184｜画成节点并连线

其余核过没问题：点开目标、多输出算子、缓存读写接法、交替、折叠未重复展开。

## 审查者 4（公式与面板）

- 措辞｜vision.Block 图 / `vbad` 折叠｜式子的 `x` 同时指本层输入与注意力输出｜vision.py:83｜把右边写清楚
- 措辞｜vision.Block 图 / `vbad2` 折叠｜同上｜vision.py:84｜同上

其余核过没问题：公式覆盖、折叠名与挂载、本轮改写的直接式逐式对过源码、权重形状与 shapes.json 一致、面板配置只放 config 值、第一张图面板「公式」组四条已是「公式一行 + 解释一行」。

## 审查者 5（读者与措辞）

- 措辞｜前提「缓存按 max_batch_size 分配」｜把 KV 缓存的序列维笼统说成 max_seq_len 4096，但窗口 KV 的序列维是 window_size（128）｜前提；`wcache` 节点「[max_batch_size, 128, 512]」｜区分频率表/压缩 KV、索引键缓存与窗口 KV
- 措辞｜前提「层 20 的压缩器…」一条 vs Compressor 图｜图里把 projg（wgate）与 pool 当通用步骤，没说层 20 跳过｜model.py:462｜在图里写明（与审查者 1 同一处）
- 措辞｜Transformer 折叠 `head` vs 面板「输出」｜折叠写 $logits = h\, head.weight^\top$（对整段 h），面板写 $h_{:,-1}$，只有面板与 head 形状对得上｜`head` 节点第二行；折叠/面板｜折叠补末位置切片
- 措辞｜Attention 面板「稀疏注意力」 vs Indexer 的 `wproj`/面板「索引打分」｜同名 `softmax_scale` 两处指不同的值（512 与 128）｜model.py:630、555｜两处分别写明各自的 head_dim
- 措辞｜Attention 折叠 `cf` vs Indexer 折叠 `fpos`｜压缩组序号一处叫 j、一处叫 g，而 g 在注意力图里又指 o_groups 组号｜model.py:753-757、539-543｜统一用 j
- 措辞｜Transformer 面板「图像嵌入」（同见 `mrg`）｜公式里的 img、img.start、types、i 都没交代｜面板；`mrg` 折叠｜补符号说明
- 措辞｜config 表「Hyper-Connections」、介绍、`hcexp` 说明｜专有名词首次出现只说名字与配置值，没说它是干什么的｜config 表；介绍｜补括注
- 措辞｜Transformer `shar` vs Indexer `shar`｜同一 id 两图指不同的东西（整包 vs 只 candidates）｜plan.md｜Indexer 侧改 id 或注明
- 措辞｜描述｜「…24、28、32、36 层各自选」缺宾语｜描述｜补宾语
- 措辞｜资料「模型卡」条｜CSA2、DSpark 首次出现只并列堆砌｜资料｜改用中文或就地补一句

## 修改

对照源码改过，重新生成页面。源码位置再查 377 个节点；`build_page.py`、`check_source.py`（计划与页面）、`site/check.py`、`--geometry` 四道机械检查全过（15 视图 / 379 节点，几何穿框 0）。

**删图与去引用（本轮的主改）**

- 删 9 张基础层图（见上「说明」）。删后全页无指向它们的「点开」、无「见点开的那张图」残留（`nemb` 指向的 `ParallelEngramEmbedding` 保留，不在删除之列）。
- 这些节点折叠里的算式改成直接写：`emb`（按行查表）、`fnorm`/`an`/`fn`（RMSNorm 式）、`head`（末位置矩阵乘）、`vpe`（flatten + Linear + bias）、`vnorm`/`vbn1`/`vbn2`（vision RMSNorm）、`vmlp`（SwiGLU）、`qr_op`/`wkv_win`/`kw`/`cnorm`（各自被调的 RMSNorm 式）、`projk`/`projg`/`wk_p`/`aw1`/`vlin`（各自 Linear 式）。

**一个节点一步（拆分）**

- Transformer：`venc`（`self.vision(...)`）与 `alen`（`self.aligner(...)`）各成节点，中间接张量 `vla`（`self.vision(patches, n_vit_h, n_vit_w)`），末端 `vemb` 用完整表达式。
- Attention：`self.q_norm(self.wq_a(x))` 拆 `wqa`（`self.wq_a(x)`）+ 张量 `qra` + `qr_op`（`self.q_norm(...)`）；`self.kv_norm(self.wkv(x))` 拆 `wkv_proj` + 张量 `wkva` + `wkv_win`。
- Compressor：`self.wkv(x), self.wgate(x)` 拆 `projk` + 张量 `kv0` 与 `projg` + 张量 `csc` 两个并列节点。
- Indexer：`self.k_norm(self.wk(latent))` 拆 `wk_p` + 张量 `wka` + `kw`。
- vision.Block：`self.norm1(x)` 输出进 `self.attn(...)`（残差 `+` 由 `vbad` 单独画，公式点明左右两个 x）；`self.norm2(x)` 输出进 `self.mlp(...)`（`+` 由 `vbad2` 单独画）。
- vision.MLP（已删图）的 `vw2`：`self.w2(F.silu(gate) * up)` 的算式并入 `vmlp` 折叠。
- Aligner：`self.w2(F.gelu(self.w1(x)))` 拆 `aw1`（`self.w1(x)`）+ 张量 `alg`（`F.gelu(...)`，普通函数与它作用的值同节点）+ `vlin`（`self.w2(...)`）。

**事实错误**

- Compressor 图补层 20 那一支：新增 `projk1`（`self.wkv(x)`）、`kv1`、`cnorm1`（`self.norm(...)`，即 `if ratio == 1: return self.norm(self.wkv(x))`），连线 `ax → projk1 → kv1 → cnorm1 → lat`；`cf32`、`lat`、`pool` 的说明点明哪些层走哪一支。前提那条已写明层 20 走这一支。
- Transformer 补 `hl → layers`、`lpm → layers`，环闭合（`lpm` 节点本就在，之前缺回连）。
- 拆分节点的第一行实参按审查者 3 改成写外层调用（`self.q_norm(...)` 等），不再出现「实参写成层输入 x」。
- `venc`/`alen` 补 `n_vit_h`、`n_vit_w` 的连线；`vv` 补 `vout → vv：n_vit_h、n_vit_w`；`vwo` 补 `vax → vwo：形状（S）`；vision.Block 的 `vbc` 改成 `vcs`（跨图同 id）。
- Block 补 `afn`/`asc`/`abs`（hc_attn_fn/scale/base）与 `ffn_a`/`fsc`/`fbs`（hc_ffn_*）六个参数节点及连线；Gate 补 `gwt`（self.weight）；Engram 补 `eqw`/`ekw`（q_weight/k_weight）；NgramHashState 补 `tmap`/`mul`/`prm`/`ofs`（token_map/multipliers/primes/offsets）及连线。这些都按 plan.md 第 88 行「算子里直接用到的量都画成节点，不管它是不是权重」。
- `shar` 的源码列改到 `model.py:1180`（实例建立处）。

**写作 / 措辞**

- `vview` 补全迭代对象；`vbad`/`vbad2` 第二行写开两个操作数；`vba1 → vbad`、`vbm1 → vbad2` 的线上字改成「注意力子层的输出」「前馈子层的输出」。
- `vbad`/`vbad2` 折叠公式：左侧写 $x_{out}$，右侧的两个量分别写 $x$（残差）与 $x_1$（norm1 输出）/$x_2$（norm2 输出）。
- 前提「缓存…」一条区分窗口 KV 的序列维；Transformer 折叠 `head` 补末位置切片；Attention 与 Indexer 两处 `softmax_scale` 各写明取的是哪个 head_dim；Indexer `fpos` 的组序号统一成 j；Transformer 面板「图像嵌入」补 img/img.start/types/i 的说明，`mrg` 折叠补 types 的来源；config 表 Hyper-Connections 补括注；Indexer 的缓存 id 由 `shar` 改成 `shar_cand`；描述补「各自用索引器选出…top-512 条压缩 KV」；资料「模型卡」条 CSA2 改写成中文。

**不改的几条**（写出理由）

- `vlin` 第一行保留 `self.w2(...)`，不按审查者 2 写成 `self.w2(F.gelu(self.w1(x)))`：后者把 `w1`、`w2` 两次类调用压回一个节点，违反 plan.md 第 81 行「一行套了几层类调用，拆开、一步一个节点」。`vlin` 与本轮其它拆分节点口径一致（外层调用 + `...`，实参由前置张量节点 `alg` 承接）。
- 算子第一行用 `...` 省略内层实参（`self.q_norm(...)`、`self.aligner(...)`、`self.w2(...)`）：plan.md 第 83 行「不用省略号代替」是对**张量**第一行说的；算子的实参在内层、而内层已经单独成节点，写 `...` 不会丢信息，也不会用源码里没有的名字。

## 最后一批改动核对

按 `review.md`「最后一批改动也要核对」，把本轮全部改动交一个独立子代理核对：有没有引入源码里没有的内容、漏步或多步、接错线；有没有违背前几轮的裁定。

**核对结论：通过。** 它逐项核过拆分的 8 处节点、补的层 20 支路、六组 hc_* 参数节点与 NgramHashState 的四个 buffer 节点，均与源码对得上、无断裂；接错线的为 0；未违背前几轮裁定。

它指出并已改的两处（皆非源码事实错误）：

- `tmap`/`mul` 两个新增张量节点的源码行号互换了：`self.token_map` 在 engram.py:154、`self.multipliers` 在 153，计划里写成 153/154。已对调。
- 本文件早先写「指向它们的『点开』全删（27 条）」，实为删 23 条（37 条 → 14 条）。已改。

`build_page.py`、`check_source.py`（计划与页面）、`site/check.py`、`--geometry` 四道机械检查改后重跑全过（15 视图 / 379 节点，几何穿框 0）。

