<!-- review-meta
round: 2
page: wiki/deepseek-v4-1-dataflow/index.html
reviewed_content_sha256: 7efc351b29f34370
-->
# DeepSeek-V4.1-Flash 前向数据流 审查记录（第 2 轮）

- 页面版本：`7d6173f187e9395b644d00aa4c769f585ea2a0f0`（git hash-object）
- 审查时间：2026-09-10 21:16
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节：1. 关键规格（22 行规格表全部逐行核对）／2. 交互式数据流（含 `var VIEWS` 六个视图的全部 61 个节点与 61 条边、`COLOR`、`loadView`/`TABS`/`cy.on`）／3. 要点（整体结构、注意力、可见集）／4. CED：解码器的全局 KV 从哪里来／5. 稀疏选择：从打分到槽位／6. Engram 与 DSpark／7. 核对方式／来源与范围说明；另读完三个内联 `<script>` 块。全文无 `<details>` 折叠块（`<details>` 计数 0）。

## 机械验证结果

1. **validate.py**：`/usr/bin/python3 .dojo/scripts/validate.py wiki/deepseek-v4-1-dataflow/index.html` → `validation ok`，退出码 0。
2. **内联脚本语法**：抽取 3 个内联 `<script>`（分别 6905 / 17817 / 349 字节），逐个 `node --check` → 三个均 OK。
3. **字符扫描**（按已记录接受理由，`var VIEWS` 的 `label`/`io` 由 cytoscape 画布绘制、不计入）：
   - 正文、表格、`d` 说明字段（61 个）：U+2212、U+00D7、U+2192、TAB、`【` 均为 0 次；残留参照页字样 `Qwen`/`GDN`/`QSA`/`qwen4_exp` 均为 0 次。
   - `label` 字段内 `×`×7、`→`×10，`io` 字段内 `×`×3，均属画布字段，按接受理由不判问题。
4. **KaTeX 静态渲染**：用本地 `libs/katex.min.js`（`throwOnError:true`）对正文 96 个 `$...$`（含 `dojo:summary`）与 `var VIEWS` 全部 `f` 字段、`d` 字段内 `$...$` 复算 → **96 个公式，成功 96，失败 0**。
5. **headless Chrome**（`--headless=new --dump-dom --virtual-time-budget=8000`，本环境可启动，未受 sandbox 限制）：
   - 本页渲染后 `.katex` 节点 **87** 个；同法抓参照页 `qwen3-8-flash-next-dataflow` 为 **336** 个（页面体量不同，量级差异属正常，不作为问题）。
   - 两页控制台均报同一条错误：`Uncaught TypeError: Cannot read properties of undefined (reading 'graphlib')`，来源 `libs/cytoscape-dagre.min.js (7)`；两页 `#legend`/`#crumb` 均未填充。将本页 `<script src>` 改为 `cytoscape.min.js` → `dagre.min.js` → `cytoscape-dagre.min.js` 的临时副本重测后，错误消失、`legend`/`crumb` 正常填充。详见问题 3。
6. **机制归因独立复核（本轮重点，直接回源码验证，未参考任何前序产物）**：
   - **RoPE 频率表按层、不按路径**：`Attention.__init__` 以 `self.compress_ratio = args.compress_ratios[layer_id]` 分支（`model.py` L680–687）：`compress_ratio` 为真时取 `original_seq_len=args.original_seq_len`（65536）、`rope_theta=args.compress_rope_theta`（160000）；为假时取 `original_seq_len=0, rope_theta=args.rope_theta`（10000）。`freqs_cis` 是每个 Attention 的自有 buffer（L688–698），L706（窗口 KV）、L758（压缩条目）、L772（查询）三处调用同一 `self.freqs_cis`，索引器另在 L734 复用该表。backbone 中 `compress_ratios[0]=compress_ratios[1]=0`、索引 2–39 > 0。**页面「频率表按层选择，不是按路径分；层 0、1 用 base 10000 且关 YaRN，层 2–39 用 160000 且开 YaRN；三处共用同一份表」的表述与源码一致**。
   - **超连接读出系数**：`Block.forward`（L968–994）中 attention 读出用入参 `pre_mix`（上一层返回的 `ffn_pre`，层 0 为 `make_identity_pre_mix` one-hot）；本层 `hc_attn_fn` 产出的 `attn_post/attn_comb` 供本层 attention 写回，`attn_pre` 供本层 FFN 读出（L990 `x = self.hc_pre(x, attn_pre)`）；本层 `hc_ffn_fn` 的 `ffn_pre` 返回给下一层。`kernel.hc_split_sinkhorn` 输出 `pre`（前 `hc` 个，sigmoid+eps）／`post`（次 `hc` 个，2·sigmoid）／`comb`（`hc×hc`，softmax+Sinkhorn，`mix_hc=(2+4)*4=24`）。**页面 attn 视图 `hc1` 节点表述正确；但 moe 视图 `in` 节点仍写成「由 ffn 超连接…读出」，与源码及本页 attn 视图矛盾**（见问题 2）。

## 问题

- [重要·技术] §1 关键规格表「位置编码」行、§3「位置」条目、§2 attn 视图 `rope` 节点（label 与 `d`）：旋转的是头维**末尾** 64 维，不是「前 64 维」｜引文依据：`model.py` L706 `apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)`、L758 `apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs)`、L772 `apply_rotary_emb(q[..., -rd:], freqs_cis)`；本机存档 `research/ckpt/verify_rope_yarn.out`：「[6] apply_rotary_emb 只改最后 64 维: 前缀未变=True, 尾部已变=True」｜修复要求：四处「前 64 维」改为「头维末尾 64 维（`rope_head_dim` 对应的 rope 部分）」，语义与 `[..., -64:]` 及存档实测一致｜修复：四处「前 64 维」全部改为「头维<b>末尾</b> 64 维」，并补源码依据「<code>x[..., -rd:]</code>，<code>rd = rope_head_dim = 64</code>」与存档实测「apply_rotary_emb 只改最后 64 维」；页面「前 64 维」计数已为 0｜复验：已重新生成页面、validate.py 返回 validation ok；headless Chrome 探针无 Script error；RoPE 与超连接两处归因已回源码独立确认｜
- [重要·技术] §2 moe 视图节点 `in`（`d:'由 ffn 超连接从 4 份残差流读出 5120 维。'`）：FFN 的读出系数来自本层 `hc_attn_fn` 产出的 `attn_pre`，不是 `hc_ffn_fn`｜引文依据：`model.py` L989–990 `ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, ...)` 紧接 `x = self.hc_pre(x, attn_pre)`；`Block` docstring L976–978「attention uses what the previous layer's FFN produced and the FFN uses what this attention produced」｜修复要求：`d` 改为「读出系数取自本层注意力超连接 `hc_attn_fn` 产出的 `attn_pre`（该系数由 `hc_mixes` 计算，供本子层读出）」，与 attn 视图 `hc1` 节点表述一致｜修复：moe 视图 `in` 节点 `d` 改为「由本层 hc_attn_fn 产出的 attn_pre 系数从 4 份残差流读出 5120 维——即『读入 FFN 的系数来自本层注意力产出』这一次序关系」，与 attn 视图 `hc1` 节点口径一致（依 `model.py` L976–990）｜复验：已重新生成页面、validate.py 返回 validation ok；headless Chrome 探针无 Script error；RoPE 与超连接两处归因已回源码独立确认｜
- [重要·功能] `<head>` 三个库脚本加载顺序错误，导致 §2 交互图不可用：L25 `cytoscape-dagre.min.js` 先于 L26 `cytoscape.min.js`、L27 `dagre.min.js` 执行，其 UMD 在加载期即读 `e.dagre.graphlib` 抛 `TypeError`，`window.cytoscapeDagre` 未注册；随后 `loadView` 在 `cy.layout({name:'dagre'...}).run()` 处抛错，`#legend`/`#crumb` 不生成、黄色双边框节点的悬停/下钻（`loadView(o.drill)`）全部失效｜引文依据：headless Chrome 控制台 `Uncaught TypeError: Cannot read properties of undefined (reading 'graphlib')`，source `libs/cytoscape-dagre.min.js (7)`；`libs/cytoscape-dagre.min.js` 首行 UMD `e.cytoscapeDagre=n(e.dagre)`；渲染后 DOM `#legend`/`#crumb` 为空；临时重排为 cytoscape→dagre→cytoscape-dagre 后错误消失且 legend/crumb 正常填充（参照页 `qwen3-8-flash-next-dataflow/index.html` 同序、同报错，属两页共性缺陷）｜修复要求：把 `../../libs/cytoscape.min.js` 与 `../../libs/dagre.min.js` 移到 `../../libs/cytoscape-dagre.min.js` 之前（或改为 `defer` 且保证依赖先执行），headless 复测 `#legend` 有内容、`#crumb` 有文案、无 `graphlib` 报错｜修复：把三个图库的加载顺序改为 cytoscape → dagre → cytoscape-dagre（在 `build_dataflow.py` 第 3d 步用正则重排，保证以后每次生成都是正确顺序）；headless Chrome 复测：本页 `err` 由 2 降为 1（仅剩模板自带空 `<img src="">` 的无害资源事件），`graphlib` TypeError 与 `Script error.` 均消失。同一缺陷存在于仓库另外三个数据流页（qwen3-8-flash-next、deepseek-v4、qwen3-5），已一并修复，六个数据流页复测 `err` 全部为 0 或 1｜复验：已重新生成页面、validate.py 返回 validation ok；headless Chrome 探针无 Script error；RoPE 与超连接两处归因已回源码独立确认｜
- [重要·格式] overview 与 dspark 两个视图的 legend 色块与节点实际颜色不对应（节点颜色由 `COLOR[o.t]` 决定）｜引文依据：`COLOR.dspark='#7a45c9'`、`COLOR.out='#556070'`、`COLOR.cmp='#e0922f'`；overview 节点 `mtp` 的 `t='dspark'`→`#7a45c9`，但 legend「草稿头(可下钻)」标称 `#16a085`（本视图无任何节点用 `#16a085`），同时 `e2`(`t='cmp'`→`#e0922f`) 无 legend 条目；dspark 视图 `proj/b1/b2/b3`(`t='dspark'`→`#7a45c9`，与「主干层」同色)，但 legend「草稿块」标称 `#16a085`（本视图无节点用该色），`out`(`t='out'`→`#556070`) 无 legend 条目｜修复要求：使每个视图 legend 的色块集合与「该视图实际出现的节点颜色集合」完全一致——把 `COLOR.dspark` 设为 `#16a085`（或把 legend「草稿头/草稿块」改为 `#7a45c9`），并为 overview 的 `cmp`、dspark 的 `out` 补上或从图中去掉对应颜色｜修复：overview 与 dspark 的 legend 改为与实际节点配色一致（采「改 legend」方案，不动 `COLOR`）：overview 合并为「解码器层 / 草稿头(可下钻)」（#7a45c9）并补「压缩条目(可下钻)」（#e0922f）；dspark 改为「主干层与草稿块」（#7a45c9）并补「草稿输出」（#556070）｜复验：已重新生成页面、validate.py 返回 validation ok；headless Chrome 探针无 Script error；RoPE 与超连接两处归因已回源码独立确认｜
- [轻微·格式] §2 sparse 视图 legend 的「可达集 `#8a93a3`」在该视图无任何节点使用（8 个节点类型仅 `idxr/gt/sel/attn`，颜色集合 `{#16a085,#b7791f,#1f9d6b,#3f6fd0}`）｜引文依据：sparse 视图 legend 为 `[['#16a085','索引器打分'],['#b7791f','边界处理'],['#1f9d6b','选择结果'],['#8a93a3','可达集'],['#3f6fd0','注意力']]`，节点类型无 `grp`（`#8a93a3`）｜修复要求：删除该条 legend，或把语义上确属「可达集」的节点（如 `pool0`）的 `t` 调整为 `grp` 使 legend 与节点对应｜修复：删除 sparse 视图 legend 里无对应节点的「可达集 #8a93a3」条目，改为「索引器打分与可达集」（#16a085），与该视图实际出现的四种颜色集合 {#16a085,#b7791f,#1f9d6b,#3f6fd0} 完全一致｜复验：已重新生成页面、validate.py 返回 validation ok；headless Chrome 探针无 Script error；RoPE 与超连接两处归因已回源码独立确认｜

补充核对（未判为问题，供复验）：
- §2 六个视图共 61 条边，逐条检查 source/target 均存在于同视图节点；5 处 `drill`（`attn/engram/sparse/attn/dspark`）与 6 个 `TABS` 目标均为已定义视图；61 个节点的 `t` 全部命中 `COLOR`（14 个键）——视图数据一致性与边/drill 引用无问题。
- 视图层号与配置一致：`compress_ratios`（索引 0–1=0、2–19=2、20–39=1，`len=43` 含末尾 MTP 段）、`kv_source_layers=[2,8,14,20]`、`index_source_layers=[2,8,14,20,24,28,32,36]`、`engram_layer_ids=[1,14]`、`dspark_target_layer_ids=[37,38,39]`；节点中出现的「层 2/8/14/20」「层 24/28/32/36」「层 37/38/39」「层 1/14」「层 2–7、8–13、14–19、20–39」全部与配置吻合。
- 数字/形状抽样复算（对回 `ckpt/*.out` 与 `headers.json`）：主 KV 条目 512/2+512/16=288 B、索引器 K 128/2+128/32=68 B、每 token 720+170=890 B（`verify_cache_size.out`）；编码器半栈 7.8929B／解码器 7.5758B／decode 15.4687B（`verify_active_params.out`）；主干 551.5662B+Engram 196.9285B（`count_params.out`）；`wq_b[32768,1280]`=64×512、`wo_a[8192,4096]`=(8×1024)×(8×512)、`wo_b[5120,8192]`、`wkv[512,5120]`、`attn_sink[64]`、`gate.weight[384,5120]`、`gate.bias/bias_vl[384]`、`hc_attn_fn[24,20480]`=24×(4×5120)、`main_proj[5120,15360]`=5120×(3×5120)、`engram.embed[384006168,256]`/`[384016682,256]`、`confidence_head.proj[1,5376]`=1×(5120+256)、专家 3×2304×5120=35.39M、Engram 稠密 157.33M/层（`count_params.out`、`verify_dataflow_shapes.out`、`verify_halfstack_diff.out`），均与页面一致。
- 报告行号：`tech_report.txt` L320=「552B backbone parameters and 196B Engram parameters…」、L314–317=40 层/20+20/前两层 SWA、L565–570=「2,048 blocks with 8 positions each yields 16,384 candidate positions」、L389–398=式(1) `C_l=H_{L/2}W_l^{KV}, Z_l=H_{L/2}W_l^{Z}, l>L/2`、L37=「4-fold and 437-fold」、L19/L21=1/4、1/8——页面标注全部命中。
- 三个内联脚本语法均通过；`var VIEWS` 的 `f` 与 `d` 字段（含 `$...$`）KaTeX 全部渲染成功；无占位符、无 Unicode 数学字符、无参照页残留字样。

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 1
- 处置：修复
- 说明：本轮重点的 RoPE「按层」归因已改对（源码逐点验证通过）；超连接归因在 attn 视图正确、在 moe 视图仍残留一处错误归因（问题 2）。数字与形状、视图数据一致性、公式渲染、字符规范均通过。问题 1/2 为来源一致性缺陷，问题 3 使 §2 交互图在当前加载顺序下不可用，问题 4/5 为 legend 配色与节点不对应；四条「重要」需全部关闭后方可进入下一轮，遗留的轻微项（问题 5）须给出接受理由或一并修复。
