<!-- review-meta
round: 4
page: wiki/deepseek-v4-1-dataflow/index.html
reviewed_content_sha256: 0f41a6d38333e71b
-->
# DeepSeek-V4.1-Flash 前向数据流审查记录（第 4 轮）

- 页面版本：8d759862520080123666c8217e7ae5303e818527（`wiki/deepseek-v4-1-dataflow/index.html`，670 行）
- 审查时间：2026-09-14 16:50
- 审查者：独立子代理（未参与写作，未读取本页 `research/`）
- 已完整阅读章节：head（description / dojo:summary / dojo:type / topics / tag）→ 1. 关键规格（全表 20 行）→ 2. 交互式数据流（6 个视图 overview/attn/sparse/moe/engram/dspark 的全部节点、io、f、d 与边、图例、crumb）→ 3. 要点 → 4. CED：解码器的全局 KV 从哪里来 → 5. 稀疏选择：从打分到槽位 → 6. Engram 与 DSpark → 7. 核对方式 → 来源与范围说明（含末段核对说明）；两段内联脚本逐字读并做 `node --check`

## 本轮核对（通过的项）

- 版本判定：`dojo:type=dataflow` → 依 `guides/model-dataflow.md`，并按其「与 concept 页同一套规范」链接到 `guides/concept/check.md`、`guides/concept/style-guide.md`。
- 数字与算式逐项复算全部通过：512/2+512/16=288；128/2+128/32=68；3×144+288=720；3×34+68=170；720+170=890；7.8929+7.5758=15.4687；24×20480=491520；24×256=6144；3×5120=15360；2048×8=16384；8×5120=40960（超连接 2×24×20480=983040）；128×512×1B=64 KiB；65536×16=1M；384006168×256 与 384016682×256 之和 ≈196.6B（≈196B 一致）；64×512=32768；8×1024=8192；(8×1024)×(8×512)=8192×4096；5120×(3×5120)=5120×15360。
- 配置字段名与取值与同源页 `wiki/deepseek-v4-1/index.html`（[C1]–[C11]、[F1]–[F8]、[N1]–[N9] 及其 `research/evidence.md`）逐条对齐，无一处冲突：`compress_ratios=[0,0,2×18,1×20,0×3]`、`kv_source_layers=[2,8,14,20]`、`index_source_layers=[2,8,14,20,24,28,32,36]`、`candidate_source_layer=20`、`candidate_topk_blocks=2048`、`candidate_block_size=8`、`index_n_heads/index_head_dim/index_topk=32/128/512`、`window_size=128`、`hc_mult/hc_sinkhorn_iters=4/20`、`n_routed_experts/n_activated_experts=384/6`、`score_func="sqrtsoftplus"`、`route_scale=1.5`、`engram_layer_ids=[1,14]`、`engram_n_heads/engram_head_dim/engram_max_ngram_size=8/256/4`、`n_mtp_layers=3`、`dspark_block_size=5`、`dspark_target_layer_ids=[37,38,39]`、`dspark_markov_rank=256`、`dspark_noise_token_id=128799`、MoE 中间维 2304。
- 头部三元组：`description` 为纯文本无 `$`；`dojo:summary` 的 `$H_{L/2}$`、`$2048 \times 8 = 16384$` 可被 KaTeX 渲染；`dojo:type=dataflow`；`.dojo/scripts/validate.py wiki/deepseek-v4-1-dataflow/index.html` 返回 `validation ok`。
- 链接与资源：`../deepseek-moe/`、`../cross-layer-kv-sharing/`、`../ngram/`、`../speculative-decoding/`、`../rope/`、`../sliding-window-attention/` 六个站内页均存在；`../../libs/` 下 9 个引用资源（katex.min.css/js、auto-render、prism 两件、cytoscape/dagre 两件、dojo-dataflow.css）全部在位；页面指向的实测清单 `wiki/deepseek-v4-1/research/measured.md` 存在且列有本页脚本（`build_dataflow.py`、`_df_views.js`、`verify_dataflow_shapes.py`），非死路径。
- 机械项：三段内联脚本 `node --check` 全部通过；全文无 `alt` 属性含 `$...$`（唯一 `<img>` 为 lightbox 空 alt）；无 Unicode 裸数学字符出现在公式定界符之外；数据流页不生成 `overview.html`，目录 `wiki/deepseek-v4-1-dataflow/` 只有 `index.html`，符合规范。
- 表述：全文只在「来源与范围说明」末段以「本页」自称（style-guide §12 允许），无「本页将/下面来看/需要注意的是」式元话语，无第一人称复数或第二人称，无会话指代与调试叙事（第 7 节「核对方式」以结论＋脚本口径陈述，未写失败与重试过程）。

## 问题

- [重要·功能] index.html L146–154（`<div class="viz">` 内 `#tabs`/`#crumb`/`#cy`/`#legend` 全为空容器）与 L419–653（`var VIEWS` 与 `loadView`）：第 2 节「交互式数据流」的全部内容——6 个视图的节点、`io` 张量形状、`f` 公式、`d` 说明、边、图例与标签页——只写在页尾内联 `<script>` 的 `VIEWS` 对象里，由 `loadView()` 用 `innerHTML`/`cy.add()` 注入空容器并由 cytoscape 画进 canvas；全文没有 `<noscript>` 或任何 HTML 承载。脚本失效时该节只剩一个空框，而这是页面里唯一逐模块标注输入/输出形状与公式的位置（正文第 3–6 节只给机制结论，规格表只给权重形状）。｜引文依据：规范 `guides/model-dataflow.md`「视图」节原文「脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」与「发布前检查」节原文「交互视图在无脚本时仍可读」；页面 L151 `<div id="cy"></div>`、L148–149 空 `#tabs`/`#crumb`、L152 空 `#legend`、L619–628 由脚本填 `legend`/`tabs`、L652 `loadView('overview')`；同仓同类页均带 HTML 兜底：`wiki/deepseek-v4-dataflow/index.html` L145–259、`wiki/kimi-k3-dataflow/index.html` L118–244、`wiki/hy4-preview-dataflow/index.html`（2 处 `<noscript>`）。｜修复要求：为第 2 节补 `<noscript>` 兜底，把 6 个视图的节点逐项写成 HTML 表（列：节点 / 维度 `io` / 公式 `f` / 说明 `d`），数字必须与 `VIEWS` 数据逐项一致，使无脚本时仍能读到同样的前向路径、形状与公式。｜修复：｜复验：
- [重要·技术] index.html L533（engram 视图 `proj` 节点）与 L198（第 6 节正文）：`proj` 节点在同一行同时给出张量形状与参数量，二者不能同时成立——`io:'[1,T,24,256] -> [1,T,5120]'` 意味着权重为 $24\times256\times5120=31{,}457{,}280\approx31.46$M，而同一节点的 `f:'\approx 157.33\ \mathrm{M}'` 对应的是 $24\times256\times25600+2\times4\times5120=157{,}327{,}360$（即 5 份 5120 输出，另加一个小的门控投影）。读者按 `io` 复算得到的结果与同行标注相差 5 倍；正文 L198「真正计入激活的是那个稠密投影，每层 157.33M」沿用同一数字但同样未给出该投影的输出宽度。｜引文依据：页面 L533 原文 `io:'[1,T,24,256] -> [1,T,5120]', f:'\\approx 157.33\\ \\mathrm{M}'`；同页 L198；复算 $24\times256\times5120=31.46$M、$5\times6144\times5120+2\times4\times5120=157.33$M；同源页 `wiki/deepseek-v4-1/index.html` L617/L632/L694/L718 四处也以「每层 157.33M」为口径，且其 `research/review-5.md` 记录该投影实测构成为 `24×256×25600 + 2×4×5120`。｜修复要求：把 `proj` 的 `io` 改为与参数量一致的形状（写明 $24\times256\to25600$，或「5 份 5120 输出 + 门控」），或在 `d` 字段补一句说明 157.33M 由哪几块构成，使 `io` 与 `157.33 M` 能互相复算。｜修复：｜复验：
- [轻微·表述] index.html L217（「来源与范围说明」末段核对说明）：「本页……把报告的宣称（1/4、1/437、1/8 三个倍数、prefill 减半、有界重放）与服务实现的优化分开陈述」中的三个倍数，在本页任何位置（含 head 的 `description`/`dojo:summary`）都没有出现，也没有对应的登记与成立条件；括注里的「prefill 减半」与「有界重放」确在 L111、L185 出现。该括注指向本页不存在的内容，读者按此检索会落空。｜引文依据：全页检索 `1/4|1/437|1/8|437` 只命中 L217 自身；L121 只写 8B/16B、L185 只写有界重放。｜修复要求：把括注改为本页实际陈述的宣称（prefill 减半、有界重放），或补上 1/4、1/437、1/8 三个倍数的登记与其成立条件。｜修复：｜复验：
- [轻微·表述] index.html L111（page-lead）：「再由索引器从至多 512 个条目里稀疏选择」把选择关系写反——索引器是在全部因果可见的压缩条目中选（1M 上下文、$r=2$ 时约 50 万条；解码器另受候选池 16384 个位置限制）选出至多 512 条，而不是在至多 512 个条目里挑。同页 L132「每 query 保留 Top-512」、L177「由索引器选出的至多 512 个压缩条目」、L188「输出是每个 query 的 Top-512 下标」、L484「1M 上下文、$r=2$ 时约 50 万条」均为正确表述。｜引文依据：L111 原文；L132/L177/L188/L484 同页对照。｜修复要求：改为「再由索引器从可见条目中稀疏选出至多 512 条」一类主语与宾语位置正确的表述。｜修复：｜复验：
- [轻微·格式] index.html L555–557（dspark 视图 `b1`/`b2`/`b3` 的 `io`）：三处均写作 `-> [1,5+V]`，按张量记号读是「最后一维大小为 $5+V$」，不是草稿块的输出形状；草稿块一次前向在 5 个位置上产出词表 logits，形状应为 `[1,5,V]`（$V=$ 词表 129280），同页 `out` 节点也只说「5 个草稿位置」。`5+V` 这一写法在仓库其它页面中未出现，属本页特有记号。｜引文依据：页面 L555–557 原文 `io:'-> [1,5+V]'`；`wiki/deepseek-v4-1/index.html` 记录草稿块输出为 logits $[1,5,V]$、置信度 $[1,5]$。｜修复要求：三处 `io` 改为 `[1,5,V]`，或写明「5 个位置 × 词表 logits」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：修复（事实、公式、数字与来源本轮全部核对通过，无阻断项；两项重要问题分别为交互视图缺无脚本兜底、Engram 稠密投影节点的形状与参数量互不相容，修复后可发布）
