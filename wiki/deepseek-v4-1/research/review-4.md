<!-- review-meta
round: 4
page: wiki/deepseek-v4-1/index.html
reviewed_content_sha256: 5fa31dfbfdc07161
-->
# DeepSeek-V4.1-Flash 审查记录（第 4 轮）

- 页面版本：95e9f556ba56a2b7a3781a1973f24b22571d8155
- 审查时间：2026-09-13 19:37
- 审查者：独立子代理（未参与写作与修复；未读取本页 `research/` 下的规划、修复与前序审查文件）
- 已完整阅读章节：引言 callout、核心问题、常见误解、第 1 章（1.1–1.3 及本章问题）、第 2 章（2.1–2.3 及本章问题）、第 3 章（3.1–3.6 及本章问题）、第 4 章（4.1–4.2 及本章问题）、第 5 章（5.1–5.2 及本章问题）、来源与范围说明

## 问题

- [阻断·来源] 开篇 `blockquote.meta`（第 63 行）与「来源与范围说明」整节：页面把来源指到已从仓库移除的文件路径，按本页自己的回源要求无法在仓库内完成核对｜引文依据：`wiki/deepseek-v4-1/research/measured.md` 的登记表把 `official/tech_report.txt`、`official/inference/model.py`、`official/inference/kernel.py`、`official/inference/config.json`、`ckpt/headers.json`、`ckpt/verify_*.out`、`verify_*.py` 等全部标为「现已从仓库移除（内容不发布，且体积可观）」；`find wiki/deepseek-v4-1/research -type f` 只剩 7 个 md 文件，`research/official/tech_report.txt`、`research/ckpt/headers.json`、`research/verify_cache_size.py`、`research/verify_reach_topk.py` 均不存在；meta 末句「实测脚本与存档见文末」在文末没有对应清单｜修复要求：把 meta 的「主要依据」以及 [C1][C4][C5][C10][F1][F2][F3][F5][N1]–[N8] 中的 `research/…` 引用改为实测记录 `research/measured.md`（写法与 `wiki/deepseek-v4-dataflow/index.html` 一致：「实测脚本与运行输出不随仓库分发，实测清单见 research/measured.md」），官方材料引用改指公开仓库 `deepseek-ai/DeepSeek-V4.1-Flash`（`inference/model.py`、`inference/kernel.py`、`inference/config.json`、`DeepSeek_V41_Tech_Report.pdf`）；同时删除「实测脚本与存档见文末」，或补上与之对应的清单｜修复：｜复验：
- [轻微·表述] 第 67 行（引言 callout）：「本页要做的就是把 890 拆开，逐项对上这三个来源。」属以「本页」为主语的元话语式自我预告｜引文依据：不适用｜修复要求：改为直接陈述下文的组织结构（如「下文按压缩、共享、FP4 三项拆开 890 的构成」），不写「本页要做的就是把…」｜修复：｜复验：
- [轻微·表述] 第 606 行（4.1 补充折叠块末）：「参数量小不等于影响小，这一点在阅读同类模型时值得注意。」面向读者的临场评语｜引文依据：不适用｜修复要求：删去该句，或改为可核对的客观限定（说明超连接参数虽少但每 token 都要读全部残差流），不出现「值得注意」这类评语｜修复：｜复验：
- [轻微·表述] 第 531 行（3.6 代码块「观察重点」段）：「观察重点：看第二段两组结果的差别。」以第二人称祈使句向读者下指令｜引文依据：不适用｜修复要求：改为陈述句，直接写出两段输出的差异（如「第二段两组输出的差别在于：未屏蔽时前 4 名含槽位 6、7，屏蔽后落在 0–4 内」），不使用「看…」｜修复：｜复验：

## 本轮核对说明

- 页面引用的官方材料经公开仓库 `deepseek-ai/DeepSeek-V4.1-Flash` 抓取原文逐条核对，全部命中：`inference/config.json` 与 `config.json` 的 `compress_ratios`（前 2 层 0、其后 18 层 2、再 20 层 1、末 3 层 0，共 43 项）、`kv_source_layer_ids=[2,8,14,20]`、`index_source_layer_ids=[2,8,14,20,24,28,32,36]`、`index_n_heads=32`、`index_head_dim=128`、`index_topk=512`、`window_size=128`、`candidate_topk_blocks=2048`、`candidate_block_size=8`、`hc_mult=4`、`hc_sinkhorn_iters=20`、`n_routed_experts=384`、`n_activated_experts=6`、`score_func="sqrtsoftplus"`、`route_scale=1.5`、`engram_layer_ids=[1,14]`、`engram_n_heads=8`、`engram_head_dim=256`、`engram_max_ngram_size=4`、`n_mtp_layers=3`、`dspark_block_size=5`、`dspark_noise_token_id=128799`、`dspark_target_layer_ids=[37,38,39]`、`dspark_markov_rank=256`、`dspark_n_routed_experts=128`、`dspark_n_activated_experts=3` 均与页面一致。
- 技术报告原文核对命中：行 315「40 causal Transformer layers … 20-layer causal encoder followed by a 20-layer decoder」；行 388–391 的 CED 全局 KV 表述与 Eq.(1)「C_l = H_{L/2}W_l^{KV}, Z_l = H_{L/2}W_l^Z, l > L/2」；行 497 Full Mode 定义；行 569–570「2,048 blocks with 8 positions each yields 16,384 candidate positions」；行 690「one E4M3 scale per 16 channels, following NVFP4 … omitting its second-level global scale」；行 692「448 × 6 = 2688」与行 698「magnitude observed during training is around 10」；行 703「We retain FP8」；行 320「552B backbone parameters and 196B Engram parameters, activating 8B parameters per token during prefill and 16B during decode」；行 37「4-fold and 437-fold」；行 139–140「1/4 as much runtime KV cache storage and 1/8 as much persistent」。段落分组（编码器 3 组×6、解码器 5 组×4，Full=20、Reindex=24/28/32/36）亦与页面 [C2]、3.4 节一致。
- 源码位置逐条核对无误（`inference/model.py`：L498–500 注释、L537 与 L554、L546/L552、L707、L760、L1131–1132、L1265–1266、L817、L821–823；Indexer 的 `wk = Linear(head_dim, index_head_dim)` 证实索引键为单头 128 维，故 68 字节/条目成立；`fp4_block_size = 32`、`scale_fmt = "ue8m0"` 证实索引器缩放因子为每 32 通道 E8M0；Compressor 的 `r=1` 退化、组位置 `j·ratio`、`select_candidate_blocks` 的钉最新块、`sparse_attn_kernel` 把汇聚点加在分母，均与页面一致）。`model.safetensors.index.json` 的张量数 96085 与页面标注完全一致，`mtp.*` 计数 2401 与 [C5] 一致。
- 全部算式可复算且与标注一致：512/2+512/16=288；128/2+128/32=68；3×144+288=720；3×34+68=170；720+170=890；18×144+20×288=8352，8352/720=11.6；377.31=126.62+1.97+35.39+212.34+0.98+0.01；7.8929−7.5758=0.3171B；2×24×20480=983040；157.33×2=314.65；(4−1)×8=24；24×256=6144；500000/16384≈30；9.5/2.65=3.58；2688/10≈268。
- 3.6 节代码在本机执行，输出与页面「预期输出」逐字符一致（不符 0 处；i=4999 两行为 2500 与 5000；屏蔽前 [0,2,6,7]、屏蔽后 [0,2,3,4]）。
- `.dojo/scripts/validate.py wiki/deepseek-v4-1/index.html` 返回 validation ok；页面 17 个前置概念链接（kv-cache、rmsnorm、mla、mxfp4-qat、mixed-precision-quant、sliding-window-attention、rope、dsa、cross-layer-kv-sharing、attention-sink、hyper-connections、deepseek-moe、aux-loss-free-routing、ngram、speculative-decoding、residual-connection、deepseek-v4-1-dataflow）全部存在，overview.html 与 index.html 互链正常。
- 未发现同一页内互相矛盾、算式与结论不符、或把实验条件观察写成无条件论断的情形；报告的 1/4、1/437、1/8 三个倍数与 prefill 半栈跳过均已在页面上登记为报告宣称。

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 3
- 处置：修复（阻断项为来源路径失效，须重指到 `research/measured.md` 与公开官方仓库；三条轻微表述按修复要求改写；本轮未发现事实、公式、数字与来源不符的问题）