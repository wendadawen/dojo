# 第 7 轮

源码位置核查通过：344 个节点（页面 346 个节点，15 张图）。有结论的席位 5 个。本轮有事实错误，改完后重跑四道机械检查全过。

本轮是作者按任务要求做的一批改动（7 条事实错误、删噪音节点、讲法、面板瘦身），改完按 `review-plan.md` 派 5 席审一轮。

## 审查者 1（路径覆盖）

- 事实错误｜Attention.forward 面板「频率基数」｜公式写了前提之外的分支（$r=0$ 那一支用 rope_theta）｜model.py:680-687、config compress_ratios｜面板公式按前提锁死到本路径：只写 compress_ratio 大于 0 时用 compress_rope_theta
- 事实错误｜ViT `vcs_op` 说明｜写成「前半段配高、后半段配宽」，与源码的相邻两维交替不符｜vision.py:11-14｜改成「相邻两维一对、交替排列，偶维配高、奇维配宽」
- 事实错误｜Attention `qb` 是死端、`core` 的 query 连了未加 RoPE 的 `q`｜Query 的 RoPE 输出无出边｜model.py:771-772｜`q → core` 改用 RoPE 后的那份（本轮 `qb`/`rope_q` 结构里已修正）
- 事实错误｜Indexer `kw`/`wka`｜`wka` 是孤点、`kw` 实参写成 lat｜model.py:544｜补 `lat → kw` 的 wk 投影，`kw` 只留归一化（本轮结构）
- 事实错误｜Compressor `ax → projk`/`ax → projg`｜同一量连了两遍｜model.py:464-465｜每个算子只留一条，注写在那条边上
- 其余核过：前提路径上每步都有节点，未发现漏步；该成图的类都成图；前提外分支未成节点

## 审查者 2（节点与形状）

- 事实错误｜Expert 图连线 `ex → u1` 与 `g1`/`u1`/`w2`/`silu` 说明｜进 GEMM 的激活说成 fp4，实为 fp8（`linear()` 对 fp4 权重也走 `act_quant`，输出 fp8）｜model.py:186-197、kernel.py:98-124、486-499｜激活按 32 通道量化成 fp8，fp4 只是权重；四处说明同改
- 事实错误｜ParallelEngramEmbedding `pes`｜写成「每 32 行一组一个 scale」，实为沿 256（通道）那一维、每 32 个通道一个｜model.py:310、27、307；shapes.json [384016682, 8]｜改成「每 32 个通道一组一个 scale（每行 8 个），8 = 256 / 32」
- 事实错误｜Engram `ekw`｜文本写 `self.k_weight`，图上第一行却是 `self.q_weight`｜model.py:347-348｜第一行改成 k_weight（q_weight 在 ewm 说明里交代）
- 事实错误｜Transformer `mrg`｜写回图像 span 的三个学到的嵌入没有节点｜model.py:1220-1222、1235-1237｜补一个节点并连线
- 其余核过：config 表逐项一致；层数与差异说明齐；算子第二行两边形状齐；多返回值节点写全；缓存节点齐

## 审查者 3（连线与结构）

- 事实错误｜Transformer `hl → layers`｜写成「层 2–39（除层 1、14 外）」，层 1 不在 2–39 里｜model.py:1261-1267｜改成「除层 14 外」
- 事实错误｜MoE `gate_op`｜第一行 `self.gate(x, image_mask)` 漏了压平实参｜model.py:892｜把压平写进实参（`image_mask.flatten()`），并在线上注明
- 事实错误｜Indexer `iscm → topk` 缺直连、`csel`/`cmf` 串成一条链｜实为 `if/elif` 互斥｜model.py:569-575｜补 `iscm → topk` 直连；`csel`/`cmf` 标成按层互斥（写进折叠说明）
- 事实错误｜Indexer `cmask` 与 `shar_cand`｜同一个 shared_attn.candidates 画成两个节点｜model.py:569-571｜合并成一个节点
- 事实错误｜Attention `widx`、Indexer `lens`｜第二行输入写成 [B, T, 5120]，实际只吃标量｜model.py:720、564｜改成标量写法
- 其余核过：点开目标正确、一节点一条点开、交替、缓存读写接法、跨图同 id 一致

## 审查者 4（公式与面板）

- 事实错误｜全页符号｜一批符号在两张图里指两回事，其中多数是本轮换名时新引入的撞名：`Z`（Aligner 的 embeds vs 压缩器的 latent）、`r`（compress_ratio / downsample / rstd / Block 归一化因子）、`c`（cutoff / clamp / 份号）、`S`（index_score / scores / span）、`h`（隐状态 / 头编号）、`K`（窗口 KV vs 索引键 vs 压缩器 U）、`E`（eout vs engram_mask）、`V`（共享专家 vs 查表行）、`P`（patches vs primes）、`Y`（累加器 vs 压缩 KV 缓存）、`I`(input_ids) vs `\mathbf{I}`(idxs)、`C`（counts vs window_kv_cache vs compressed）、`J`（attn_comb vs Sinkhorn 前的 u）、`Q`（index_score vs cache）
- 事实错误｜Attention 面板「频率基数」｜写了前提外的 $r=0$ 分支｜model.py:680-687｜同上（审查者 1 同处）
- 措辞｜Indexer `imask` 符号｜「不是同图 red 里的头维 h」，red 已用 i｜dataflow.md｜改成「头维 $i$」
- 措辞｜Indexer `qop`/`wproj`｜同一 `H_i` 两个名字｜model.py:508-509、555｜统一
- 措辞｜Transformer 面板/资料｜CSA2 首次出现没给全称｜model-card.md:49｜补 Compressed Sparse Attention 2
- 事实错误｜Engram 说明 $w_e[i] * exp(s[i])$｜缺少 $w_e$ 是 fp8 表、$s$ 上转 fp32 再指数｜kernel.py:49-54、model.py:317-320｜说明补全
- 事实错误｜Compressor 面板「池化」与折叠 `pool` 符号不一致｜面板用 softmax(g)、折叠用 softmax(G_g)｜model.py:475｜面板改成与折叠一致
- 事实错误｜Engram 面板「门控」｜用 $c$ 指 clamp_value｜model.py:362｜改成 $j$（折叠已用 j）
- 其余核过：公式都有；折叠名与挂载对；权重形状与 shapes.json 一致；面板配置只放 config 值；配置默认值已注明

## 审查者 5（读者与措辞）

- 措辞｜描述｜「图像占位」与其余「图像 span」不一｜描述｜改「图像 span」
- 措辞｜前提第 29 条｜「这两件事互斥」指代不清｜前提｜点明互斥的是「自己建键」与「用候选块过滤」
- 措辞｜前提第 31 条｜「二者不是一回事」指代不清｜前提｜改成「草稿路径与本次前向的 tap 不是一回事」
- 措辞｜Aligner vpad/vunf、Block 面板｜同一字母 $r$ 三种含义｜vision.py:115-118、model.py:952-954｜Aligner 用 k、Block 面板用 n，$r$ 留给 compress_ratio
- 措辞｜Compressor 面板 vs 折叠｜同一量两个符号｜model.py:475｜面板与折叠统一
- 措辞｜Engram 面板 vs 折叠｜clamp 常数两处两个字母｜model.py:362｜统一成 j
- 措辞｜vision.Attention vs Attention｜$d_h$ 在两处指 64 与 512｜vision.py:50、model.py:630｜视觉那份换字
- 措辞｜Attention `wrope`｜nope 首次出现无解释｜model.py:632｜补「不加旋转位置编码的一段」
- 措辞｜Transformer `mrg`｜公式用 START/END/NEWLINE，说明说「IMAGE_START 等」｜model.py:1235-1237｜公式改用完整类型名
- 措辞｜全页 `d`｜三处三种含义（dim / index_head_dim / 每组 4096）｜——｜索引器那份换字

## 修改

对照源码逐条改，重新生成页面。

**事实错误**

- `tmap` 第二行改 129280（tokenizer 词表大小），第三行说明 99092 是压缩后 id 的取值上界；`cmap` 说明同改。
- `vcs_op` 说明改成「相邻两维一对、交替排列，偶维配高坐标、奇维配宽坐标，同一对内两维共用一个逆频率」。
- Expert 的 `ex → u1` 线上字与 `g1`/`u1`/`w2` 说明：激活按 32 通道量化成 fp8，fp4 只是权重。
- `pes` 说明改成「每 32 个通道一组一个 scale（每行 8 个），8 = 256 / 32」。
- `ekw` 第一行改成 `self.k_weight`。
- Transformer 新增节点 `imgp`（self.image_start/end/newline 三个学到的嵌入），补 `imgp → mrg` 连线；`mrg` 折叠改用完整类型名。
- Indexer 补 `iscm → topk` 直连，`csel`/`cmf` 的说明点明按层互斥；`cmask` 与 `shar_cand` 合成一个 `cand` 节点。
- Gate 的 `gwt`、Engram 的 `ekw` 改 bf16（构造时未指定 dtype，取参考实现默认 bf16），并说明 `.float()` 上转；`attn_sink`(639)、`bias`(806)、`hc_*`(940) 的 fp32 不动。
- `hl → layers` 改成「层 2–39（除层 14 外）」。
- `gate_op` 第一行实参改 `None if image_mask is None else image_mask.flatten()`，折叠说明注明压平。
- `widx` / `imask` 的输入列改标量写法（widx 由 [B,T,5120] 改成 int 四元组），`lens`、`fpos` 节点删除，动作并进 `imask`/`krope`。

**删噪音节点**（逐节点 diff：删 43 个、新增 6 个承接节点，净减 37：383 → 346）

- vision.Attention：删 `vsplit`、`vview`、`vqkv2`，并入 `vqkv`（chunk+view）。
- Aligner：删 `vv`、`vv2`，并入 `vpad`（view+permute）。
- Attention：删 `qpre`、`grp`、`og`、`woa`、`fsl`、`cf`、`cfq`，分别并入 `q`/`rope_q`、`eins`、`wav`、`fc`。
- Compressor：删 `cf32`、`cxf`、`cunf`、`cunk`、`cuns`、`cunsc`，并入 `projk`/`projg`/`pool`。
- Indexer：删 `lens`、`lv`、`fpos`、`fq`、`wk_p`、`cmask`、`shar_cand`，并入 `imask`/`krope`/`kw`/`cand`。
- MoE：删 `flat`、`mxf`(保留)、`mmask`、`mfm`、`y0`、`mback`、`y3`，并入 `mxf`/`gate_op`/`y`/`mout`。
- Engram：删 `spl`、`ekey`、`nhf`、`vf`、`ncast`、`eqw`、`v`，并入 `nsplit`/`key`/`nh`/`nadd`/`ewm`。
- NgramHashState：删 `blk0`、`blk`、`pos`，并入 `blocked`/`p`。
- Block：`afn`/`asc`/`abs`/`ffn_a`/`fsc`/`fbs` 合成 `afn`/`asc`/`abs` 三个（各含 attn、ffn 两份）。
- 新增的 6 个承接节点：`imgp`（三个学到的嵌入）、`q2`（RoPE 后的 query）、`cand`（合并后的 candidates）、`nkey`/`nsplit`/`blocked`。
- 保留不删：`act_quant`/`fp4_act_quant`、`eins`、`silu`、`topk`、`gather`、`Sinkhorn`、`apply_rotary_emb`。
- 删除后上家直接连下家；折叠里的公式并进相邻算子。

**讲法**

- `ids` 第三行写明 B（批大小，本次 1）、T（本次喂进的 token 数）；`lat`/`pool`/`crope` 等写明 r 是 compress_ratio。
- `pool` 第一行改中文短语「组内加权池化」，源码表达式放说明栏。
- 全页符号去重：`Z`→`Z`(compressor)/`F`(Aligner)；`r` 保留给 compress_ratio，Aligner 用 k、Engram rstd 用 ν、Block 面板归一化因子用 n；`c`（cutoff）→b、（clamp）→j；`S` 在 Indexer 内→Q、Gate 内 scores→R；`h`(头维)→i；`K`(索引键)→J；`E`(eout)→B、`\mathbf{E}`(engram_mask) 保留但 Engram 图内 token_mask 用 E；`V`(共享专家)→V_s、`V`(查表行)→R_v；`P`(primes)→Π；`Y`(累加器) vs `N`(压缩缓存)；`\mathbf{I}`(idxs)→G；`C` 三分（counts n、window_kv_cache W、compressed 保留）；`O`(offsets)→O_b。
- 缓存节点第二行/第三行写明第一维是容量（max_batch_size 默认 4，不是本次 B），窗口 KV 第二维是 128 槽的环，压缩/索引缓存第二维是 max_seq_len / r。
- 跨层共享：`shar` 第三行写明压缩 KV 与索引键各由 4 个 kv source 层写、其后各层读；candidates 由层 20 写、24/28/32/36 读。
- 八个 RoPE 公式改成「末 $d_r$ 维乘一个旋转因子 $e^{i\varphi}$」的形式，源码写法放符号栏末行。

**面板与补写**

- 面板「配置」组只留决定稀疏/结构的值（window_size、index_topk、compress_ratios、hc_mult、hc_sinkhorn_iters、hc_eps、route_scale、n_* experts、swiglu_limit、score_func、索引器三项、rope 组、视觉结构几项、engram 组、量化 dtype）；已在节点形状里出现的（dim、n_heads、head_dim、q_lora_rank、vocab_size…）删去。顶部 config.json 面板同样压成要点。
- `linear()` 里对激活做的 fp8/fp4 分块量化写进各图用 Linear 的节点折叠说明（`wqa`/`wkv_proj`/`qb`/`wb`/`projk`/`g1`/`u1`/`w2`、`nwkv`）。
- 两套 RoPE base 的分支写进 `fcall` 第三行与 Attention 面板「频率基数」；按审查者 1/4 的意见，面板公式只写本路径的 compress_rope_theta 一支，$r=0$ 一支删除。

**最后一批改动核对**

本轮是任务要求的改动 + 一轮审 + 改，尚未到轮数上限，故未按 `review.md` 的「最后一批改动也要核对」另派仲裁核对子代理。改后四道机械检查（`build_page.py`、`check_source.py` 计划与页面、`site/check.py`、`--geometry`）全过（15 视图 / 346 节点，几何穿框 0、像素重合 0）。
