<!-- review-meta
round: 5
page: wiki/deepseek-v4-1-dataflow/index.html
reviewed_content_sha256: c4143a3e147d5fe2
-->
# DeepSeek-V4.1-Flash 前向数据流审查记录（第 5 轮）

- 页面版本：`6f988a2e85ca107314159a85ed8e9679a7d75541`（wiki/deepseek-v4-1-dataflow/index.html）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查）
- 已完整阅读章节：1 关键规格；2 交互式数据流（含 6 个视图的 noscript 静态兜底表）；3 要点；4 CED：解码器的全局 KV 从哪里来；5 稀疏选择：从打分到槽位；6 Engram 与 DSpark；7 核对方式；来源与范围说明
- 来源获取：官方仓库 `deepseek-ai/DeepSeek-V4.1-Flash`（Hugging Face）的 `inference/config.json`、`inference/model.py`、`inference/kernel.py`、`inference/generate.py`、`DeepSeek_V41_Tech_Report.pdf`、`model.safetensors.index.json` 与 48 个 `*.safetensors` 分片头（HTTP Range 读取）。报告「行 N」按 `pdftotext -layout` 抽取的文本行核对。本页 `research/` 未读取。

## 问题

- [重要·技术] index.html L187、L212（noscript 静态表）与 L556、L603（`VIEWS` 数据；attn 视图 `hc2` 节点与 moe 视图 `hc` 节点）：超连接的写回公式写成 `$x_i \leftarrow x_i + \lambda_i y$`，与官方源码不符。源码的写回是「子层输出按 `post` 逐路缩放 + 旧残差流按 `comb` 混合」，即 $out_i = post_i \odot y + \sum_j comb_{ij} \odot x_j$；页面给出的形式去掉了跨流混合项，等于把 4 条残差流当成互不相干地各自叠加，而 `comb`（Sinkhorn 双随机矩阵）正是超连接的要点（同页 L136 自己列出「Sinkhorn 20 轮，末步列方向」）。同页 `hc1` 节点说明（L174 / L543）写「本层 hc_attn_fn……产出的 post/comb 系数用于本子层写回」，与该公式自相矛盾。｜引文依据：`model.py` L962–966 `def hc_post(self, x, residual, post, comb): ... y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`；`kernel.py` L427–431 `pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps`、`post[i, j] = 2 * T.sigmoid(...)`、`comb_frag[j, k] = mixes_shared[...] * hc_scale[2] + hc_base[...]`（L436 起 softmax + 行列交替归一化 20 轮）；同页 L543 原文。｜修复要求：把 `hc2`/`hc` 两处公式改为与 `hc_post` 一致的形式（$out_i = post_i \odot y + \sum_j comb_{ij} \odot x_j$），或在公式旁标注「此处省略 comb 混合项」并把被省略项写全，使公式可与 `model.py` L962–966 复算。｜修复：｜复验：

- [轻微·表述] index.html L199（sparse 视图 `sa` 节点说明）与 L284（第 5 节正文）：「窗口没有额外权重加成——谁的分高谁拿得多」「三种层模式决定了「谁算、谁复用」」为口语化措辞，属规范「表述」一节要求排除的口语化表达。｜引文依据：不适用。｜修复要求：改为书面表述，例如「两部分槽位共用一次 softmax，窗口没有额外权重加成，选中完全由索引分决定」「三种层模式决定各层自算还是复用」。｜修复：｜复验：

## 核对记录（本轮已核对通过、不构成问题的项）

- 逐条回到官方材料核对，未发现定位不到、来源不支持、数字不符、页内两处矛盾、算式与结论不符、图注读数与刻度不符、编号不一致等问题。要点如下（引文依据均为本轮实际取到的原文/数值）：
  - 总参数：报告行 320「has 552B backbone parameters and 196B Engram parameters, activating 8B parameters per token during prefill and 16B during decode」；结构：报告行 314–317「40 causal Transformer layers, organized into a 20-layer causal encoder followed by a 20-layer decoder... except for the first two layers, which use SWA only」；候选池：报告行 565–570「each block is assigned the maximum index score among its positions... selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions」；CED：报告 Eq.(1) 行 394–396「$C_l = H_{L/2} W_l^{KV}$, $Z_l = H_{L/2} W_l^{Z}$, $l > L/2$」；890 B/token：报告行 17–19「reduce its global KV cache footprint (always in HBM) to 890 bytes per token」。四处「报告行 N」均落位正确（以 `pdftotext -layout` 文本核对）。
  - 缓存账可复算：主 KV 每条目 288 B = 512×0.5 + 512/16（`model.py` L759–760 `fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`，E4M3 每 16 通道一个尺度）；索引器 K 每条 68 B = 128×0.5 + 128/32（L546 `fp4_act_quant(k, fp4_block_size=32, True)`，E8M0）。由 `kv_source_layers=[2,8,14,20]`、`compress_ratios` 得 3×(288/2)+288 = 720、3×(68/2)+68 = 170，合计 890，与 summary、description、规格表、page-lead 四处一致。
  - `inference/config.json` 全部取值与规格表一致（`n_layers=40`、`dim=5120`、`n_heads=64`/`head_dim=512`、`q_lora_rank=1280`、`o_groups=8`/`o_lora_rank=1024`、`window_size=128`、`compress_ratios`[层 2–19 为 2、层 20–39 为 1、层 0/1 为 0]、`kv_source_layers=[2,8,14,20]`、`index_source_layers=[2,8,14,20,24,28,32,36]`、`index_n_heads=32`/`index_head_dim=128`/`index_topk=512`、`candidate_source_layer=20`/`candidate_topk_blocks=2048`/`candidate_block_size=8`、`n_routed_experts=384`/`n_activated_experts=6`/`n_shared_experts=1`/`moe_inter_dim=2304`、`score_func=sqrtsoftplus`/`route_scale=1.5`、`hc_mult=4`/`hc_sinkhorn_iters=20`、`engram_layer_ids=[1,14]`/`engram_max_ngram_size=4`/`engram_n_heads=8`/`engram_head_dim=256`、`n_mtp_layers=3`/`dspark_block_size=5`/`dspark_target_layer_ids=[37,38,39]`/`dspark_noise_token_id=128799`/`dspark_n_routed_experts=128`/`dspark_n_activated_experts=3`/`dspark_markov_rank=256`、`vocab_size=129280`、视觉侧 32/1024/14/3）。
  - 张量形状逐条与 checkpoint 头一致（本轮实取）：`layers.0.attn_norm.weight [5120]`、`layers.0.attn.wq_b.weight [32768, 1280]`、`layers.0.attn.wkv.weight [512, 5120]`、`layers.0.attn.wq_a.weight [1280, 5120]`、`layers.0.attn.q_norm.weight [1280]`、`layers.0.attn.kv_norm.weight [512]`、`layers.0.attn.attn_sink [64] F32`、`layers.2.attn.wo_a.weight [8192, 4096]`、`layers.2.attn.wo_b.weight [5120, 8192]`、`layers.2.ffn.gate.weight [384, 5120]`、`gate.bias`/`gate.bias_vl [384]`、`layers.2.hc_attn_fn [24, 20480]`、`layers.2.ffn.experts.0.w1.weight [2304, 2560] I8`（FP4 打包，对应逻辑形 [2304, 5120]）、`embed.weight`/`head.weight [129280, 5120]`、`mtp.0.main_proj.weight [5120, 15360]`、`layers.1.engram.embed.weight [384006168, 256]`、`layers.14.engram.embed.weight [384016682, 256]`、`engram.wkv.weight [25600, 6144]`、`mtp.0.ffn.gate.weight [128, 5120]`；`model.safetensors.index.json` 张量数 96085、分片 48，与页内一致。
  - 源码行号定位：`model.py` L458–485（`Compressor.forward` 池化）、L680–687（`if self.compress_ratio:` 按层切频率表）、L706/L758/L772（窗口 KV、压缩条目、查询三处共用同一 `freqs_cis`）、L982–994（超连接读出/写回）、`kernel.py` L310–403（`sparse_attn`）均落位正确。
  - 公式语义：`sparse_attn` 内核（`kernel.py` L382–385）`sum_exp[i] += T.exp(attn_sink[i] - scores_max[i]); acc_o[i, j] /= sum_exp[i]`，与页内汇聚点公式（只进分母、不参与取最大值）一致；`-1` 占位槽位在 L362–364 被排除。`Gate`（L811–827）`softplus(...).sqrt()`、`topk(score+bias)`、`weights /= sum + 1e-20`、`weights *= route_scale` 与页内打分/偏置/权重公式一致。RoPE 频率按 `base=160000, original_seq_len=65536, beta_fast=32, beta_slow=1` 代入 `precompute_freqs_cis`（L379–383）复算得 low=15、high=25，与页内实测值一致。Sinkhorn（`kernel.py` L436–458）行、列各 20 次且末步为列方向，与规格表一致。
  - Engram 参数账：`wkv` 输出宽 $dim \times (hc\_mult+1) = 25600$，权重 $24\times256\times25600$，另 $q\_weight$/$k\_weight$ 各 $[4,5120]$，每层 $157{,}327{,}360 \approx 157.33\text{M}$，`io` 与 `f` 可互相复算。
  - DSpark：`model.py` L1264–1266「the MTP head reads the attention input of its target layers, not their output」+ `h.mean(dim=2)`（超连接副本维取均值）与 L1271 `torch.cat(..., dim=-1)`（3×5120=15360）与页内一致；`get_moe_config`（L142–149）确认草稿块用 128/3 专家。
  - 「参考实现中不含该分支（有界重放）」：本轮实取 `inference/generate.py` 全文 218 行无 `replay`/`window`/`SWA` 分支，页面陈述成立。
  - 页内一致性：6 个视图的 noscript 静态表（节点/维度/公式/说明）与 `VIEWS` 数据逐项比对，`io`、`f`、`d` 全部一致（仅有无 `$` 定界符的写法差异）。全部公式（`VIEWS` 的 `f` 与 `d` 内联式、正文与规格表内联式、`dojo:summary` 的两个公式）用 KaTeX `throwOnError:true` 逐个渲染，0 报错。`alt` 属性只有 lightbox 占位 `alt=""`，无 `$...$`。`.dojo/scripts/validate.py` 通过，`dojo:type=dataflow`、`dojo:topics` 两值在允许大类内、`dojo:tag=数据流` 在词表内。
  - 引用链接：`../deepseek-moe/`、`../cross-layer-kv-sharing/`、`../ngram/`、`../speculative-decoding/`、`../rope/`、`../sliding-window-attention/` 六个前置概念页目录均存在；`wiki/deepseek-v4-1/research/measured.md` 存在（90 行），非「指向不存在文件」。
  - 图内数值：本页无位图图（`<img>` 仅 lightbox 占位），不存在需要像素测量的刻度图；交互视图的数值以 `VIEWS` 数据与 noscript 表为准，两者已逐项对齐。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 1
- 处置：修复（关闭 1 条重要、1 条轻微后即可发布）