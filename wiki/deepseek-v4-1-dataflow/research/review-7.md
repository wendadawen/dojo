<!-- review-meta
round: 7
page: wiki/deepseek-v4-1-dataflow/index.html
reviewed_content_sha256: c88c88f1e0453b4d
-->
# DeepSeek-V4.1-Flash 前向数据流审查记录（第 7 轮）

- 页面版本：2fa821c6370ff8fdb1d8543eb736603d2e69f0c2（index.html 工作树哈希；HEAD 23797b2）
- 审查时间：2026-09-14 18:00
- 审查者：编排者派发的独立审查者（未参与写作，未读取本页 research/）
- 已完整阅读章节：头部 meta、1. 关键规格、2. 交互式数据流（含 `<noscript>` 六视图静态表）、3. 要点、4. CED、5. 稀疏选择、6. Engram 与 DSpark、7. 核对方式、来源与范围说明、页内脚本与六个 cytoscape 视图定义
- 核对所用文献表版本：
  - 报告行号取自官方技术报告《DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression》的版面保真文本 `DeepSeek_V41_Tech_Report.pdf`（HF 仓库 `deepseek-ai/DeepSeek-V4.1-Flash`，commit `dba1be0a40aa45a94ad051997016db3960a90277`，取回 etag `9a327deea3393d204f789f261ceb4eb472ea910d`；行号对应 `report_layout.txt`）。逐条比对 行 314–322、行 130–140、行 565–570、行 1094–1102、行 655–675、行 382–410。
  - `inference/config.json`（官方推理配置）、`inference/model.py`、`inference/kernel.py`、`inference/engram.py`（官方参考实现副本）。
  - 真实 checkpoint 张量索引：48 个 safetensors 分片、96085 个张量、metadata.total_size=510286023000（与页面「48 分片 / 96085 张量」一致）。
  - 机械验证：`.dojo/scripts/validate.py wiki/deepseek-v4-1-dataflow/index.html` → `validation ok`。

## 问题

- [重要·技术] dojo:summary：`压缩比 2 的三个 source 层折合 144 B/token、压缩比 1 的层折合 288 B/token，加索引器 K 170 B 得每 token 890 字节`｜引文依据：`config.json` `kv_source_layers=[2,8,14,20]`、`compress_ratios=[0,0,2×18,1×20,0×3]`；报告行 1094–1099「The remaining 18 encoder layers use CSA2 with a compression rate of 𝑚 = 2. These layers are divided into three identically configured groups of six layers. In each group, the first layer operates in Full Mode」；报告行 16–18「reduce its global KV cache footprint ... to 890 bytes per token」。分项按字面读为 144+288+170=602，与句末 890 不符；且「三层的 144」暗示 r=2 只有一份共享缓存，与正文「主 KV 720」（=3×144+288，见 1. 关键规格「每 token 全局 KV 缓存 | 890 B | 实测：主 KV 720 + 索引器 K 170」）矛盾｜修复要求：把该分项写明为「每个 source 层 144 B/token、三层共 432 B/token」，或直接写「主 KV 720 B/token + 索引器 K 170 B/token」，使分项之和等于 890｜修复：｜复验：

- [重要·技术] 视图 6「DSpark」节点 b1/b2/b3 与 `<noscript>` 同名表：三行均标 `-> [1,5,V]`，b1 说明写「一次前向产出 5 个草稿位置（...输出为 5 个位置的词表 logits）」，b2 写「同上」｜引文依据：报告行 664–666「The drafter comprises three Transformer blocks with a sliding attention window of 128 tokens. A single forward pass through these blocks computes base logits for five draft positions in parallel, while a lightweight Markov head models dependencies among the draft tokens.」；`model.py` `forward_spec`（L1275–1282）三块顺序执行后仅调用 `self.mtp[-1].forward_head(...)` 产出 logits。即 3 个块一次前向**共同**产出 5 个位置的 logits，而非每块各产出 5 个位置（后者会读成 3×5）｜修复要求：把三行改为：三块共同产出 5 个草稿位置的 logits（前两块输出隐藏态、不加输出头），只有末块接 Markov 头与置信头并输出 `[1,5,V]`；与本页总览节点「DSpark · 3 个草稿块 -> 5 个草稿位置」一致｜修复：｜复验：

- [轻微·表述] `<head>` `description`：「每 r 个 token 池化一条、同压缩比跨层共享、Top-512 稀疏选择」｜引文依据：报告行 1094–1099 明示编码器 18 个 r=2 层分成「three identically configured groups of six layers」，每组各自有 Full 首层；`config.json` `kv_source_layers=[2,8,14,20]`，故同为 r=2 的层 2–7、8–13、14–19 是三个互不共享的缓存，「同压缩比」并非共享的判据（正文 1. 关键规格「来源 2/8/14/20；分组 2–7、8–13、14–19、20–39」是对的）｜修复要求：改为「按 source 层跨层共享」或删去「同压缩比」限定，避免读者以为每种压缩比只有一份缓存｜修复：｜复验：

- [轻微·格式] 视图 2「注意力内部」节点「索引器打分」的维度串：`[1,T,32,128] x [n,r=128] -> [1,T,n]`（`<noscript>` 同）｜引文依据：`config.json` `index_head_dim=128`；本页其余各处 $r$ 一律指压缩比（如 $n_{\mathrm{reach}}(i)=\lfloor (i+1)/r\rfloor$、组 $j$ 覆盖 $[jr,(j+1)r)$）｜修复要求：把 `[n,r=128]` 改为 `[n,128]`（索引器 K 的形状为 n 个位置 × 头维 128），保持 $r$ 全文单义｜修复：｜复验：

- [轻微·表述] 第 4 节 CED：「窗口 KV 是最近位置的原始精度副本，必须由本层经过完整计算后的隐状态生成」｜引文依据：报告行 703–704「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」；`model.py` L706–707 `apply_rotary_emb(kv[..., -rope_head_dim:], freqs_cis); act_quant(kv, fp8_block_size, ...)`（窗口 KV 经 FP8 量化后存环形缓冲）。窗口 KV 不是「原始精度」（未量化）副本，只是未压缩的逐 token FP8 副本；本页 1. 关键规格、视图 2 均已标 FP8｜修复要求：改为「未压缩的逐 token FP8 副本」，与全文 FP8 表述一致｜修复：｜复验：

- [轻微·技术] 视图 2 节点「稀疏注意力」公式与 5. 稀疏选择：$o=\frac{\sum_t e^{q\cdot k_t/\sqrt{d}-m}v_t}{\sum_t e^{q\cdot k_t/\sqrt{d}-m}+e^{\mathrm{sink}-m}}$，其中 $m$、$d$ 全页未定义｜引文依据：`kernel.py` L310–400 稀疏注意力内核 `scores_max` 取并集分数最大值、`sum_exp[i] += exp(attn_sink[i]-scores_max[i])`（$m$ 即该运行最大值）；`scale=(1.0/d)**0.5`（$d$ 为头维）｜修复要求：在公式后补一行符号说明：$m=\max_t q\cdot k_t/\sqrt{d}$，$d$ 为头维（512），与「softmax 在并集上做一次归一化」对应｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复

（附：本轮已回源核对且与官方材料一致的项，未列为问题——总参数量 552B/196B（报告行 320）、40=20+20 层（报告行 314–317）、8B/16B 激活（报告行 320–322）、890 B/token（报告行 16–19）、MoE 384/top-6/1 共享（报告行 1110–1115）、超连接 hc_mult=4/20 轮、索引器 32×128/Top-512、候选池 2048×8=16384（报告行 570–571）、FP4 E2M1 每 16 通道一个 E4M3 scale（model.py L760）、索引器 K 每 32 通道一个 E8M0（model.py L546）、超连接读出/写回（model.py L982–994、kernel.py L426–460）、Engram 门控 signed-sqrt（model.py L362）、DSpark 噪声 token 128799/目标层 37–39/main_proj [5120,15360]（model.py L1128–1135、L1265–1266）、RoPE 层分支与 base/YaRN（model.py L680–687；base 160000 时 low=15、high=25 复算一致）、各权重形状与 config 字段一致；`<noscript>` 静态表与六个视图在无脚本时可读，图注无刻度读数可核。）