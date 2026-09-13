<!-- review-meta
round: 3
page: wiki/deepseek-v4-1-dataflow/index.html
reviewed_content_sha256: 7efc351b29f34370
-->
# DeepSeek-V4.1-Flash 前向数据流 审查记录（第 3 轮）

- 页面版本：`b4cd6adefa5d5c87c9cb8e153bf1327d3c54045f`（`git hash-object wiki/deepseek-v4-1-dataflow/index.html`）
- 审查时间：2026-09-10 21:23 CST
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查与修复）
- 已完整阅读章节：页头元数据与资源引用（L3–30）、模板样式与页面自定义样式（L31–786）、页体（L789–925：标题/导语/元信息、1. 关键规格、2. 交互式数据流、3. 要点、4. CED、5. 稀疏选择、6. Engram 与 DSpark、7. 核对方式、来源与范围说明）、三个内联 `<script>` 块（L927–1122、L1125–1359、L1361–1373）。全文无 `<details>` 元素（`<details>`/`<summary>` 计数均为 0），故“含折叠块”一项在本文档为空集。

## 机械验证结果

1. `validate.py`：`/usr/bin/python3 .dojo/scripts/validate.py wiki/deepseek-v4-1-dataflow/index.html` → `validation ok`，退出码 0。
2. 三个内联脚本语法：抽取正文中非 `src` 的 `<script>` 块（3 块）至 `/tmp/r3/block{1,2,3}.js`，用 `/Users/wendadawen/.workbuddy/binaries/node/versions/24.14.0/bin/node --check` 逐个检查 → 3/3 `syntax OK`。
3. Unicode/残留字符扫描（正文 L789–925 与元数据）：U+2212 `−` 0 处；TAB 0 处；`【` 0 处；`Qwen`/`GDN`/`QSA`/`qwen4_exp` 0 处。U+00D7 `×` 8 行、U+2192 `→` 10 行，全部落在 `var VIEWS` 的 `label` 字段（画布绘制）或 `io` 字段（tooltip 以等宽文本转义输出，KaTeX 不介入），符合已记录的接受理由：正文、表格、`d` 字段中均无数学 Unicode 字符。
4. 渲染后文本复核（headless Chrome dump 剥离 script/style/annotation）：可见文本中残留 `$` 0、反斜杠 0、`\mathrm`/`\frac` 0；`\sqrt` 的 1 次命中位于 KaTeX 的 MathML/annotation 标记内，非可见文本。`katex-error` 0，KaTeX 内联节点 29 个（与正文 29 对 `$...$` 一致），显示公式 2 个。`dojo:summary` 的 4 个 `$` 成对，`description` 无 `$`。
5. 库加载顺序静态核对：`cytoscape.min.js`(L24) → `dagre.min.js`(L25) → `cytoscape-dagre.min.js`(L27)，三者均为 head 内同步脚本按序执行。`dagre.min.js` 的 UMD 写入 `window.dagre`；`cytoscape-dagre.min.js` 的 UMD 为 `e.cytoscapeDagre = n(e.dagre)`，**在加载时即读取 `window.dagre`**，故该顺序是必需的且正确。`L1126` 在解析期调用 `cytoscape.use(window.cytoscapeDagre)` 时三个库均已就位。
6. headless Chrome 实测交互图：
   `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --headless=new --disable-gpu --no-sandbox --virtual-time-budget=20000 --dump-dom file:///Users/wendadawen/code/dojo/wiki/deepseek-v4-1-dataflow/index.html`
   退出码 0，输出 116439 字节。渲染后 DOM 中：`#cy` 内 3 个 `<canvas data-id="layer0-selectbox|layer1-drag|layer2-node">`（cytoscape 已用 dagre 布局完成绘制，容器 701×600）；`#legend` 有 6 条色块（`#1f9d6b/#c0392b/#7a45c9/#e0922f/#b7791f/#8a93a3`）；`#crumb` 内容为 `<b>整体总览</b> — 点击黄框节点下钻到内部`；`#tabs` 6 个标签且 `overview` 为 `on`；`#tocList` 已生成（539 字符）。DOM 文本中无 `Script error`、无 `graphlib`、无 `is not a function`。
7. 全 6 个视图布局等价性复核（用仓库内真实 `libs/dagre.min.js` 0.8.5 在 Node 中按 `cy.layout({name:'dagre',rankDir:'TB',nodeSep:V.nodeSep,rankSep:V.rankSep,edgeSep:10})` 同参跑布局）：`overview/attn/sparse/moe/engram/dspark` 全部 `dagre.layout OK`，无抛错；各视图边端点、`drill` 目标、重复边检查 0 问题，无孤立节点。
8. legend 与节点配色逐视图核对（用页面自身 `COLOR[o.t]` 与 `V.legend`）：6/6 视图的 legend 色块集合 **等于** 该视图实际出现的节点颜色集合，无缺失、无多余、无重复。
   实际集合：overview `#1f9d6b,#c0392b,#7a45c9,#e0922f,#b7791f,#8a93a3`；attn `#8a93a3,#3f6fd0,#b7791f,#16a085,#e0922f`；sparse `#16a085,#b7791f,#1f9d6b,#3f6fd0`；moe `#8a93a3,#b7791f,#3f6fd0,#1f9d6b`；engram `#c0392b,#16a085,#b7791f,#8a93a3`；dspark `#7a45c9,#8a93a3,#b7791f,#556070`。
9. 公式字段校验（用仓库内 `libs/katex.min.js` 以 `throwOnError:true` 渲染全部 `f` 字段与 `d` 字段内 `$...$`）：失败 0。
10. 本地引用与资源：`index.html` 引用的 6 个站内页 + 首页 + 11 个 `libs/*` 全部存在；页面点名的 10 个实测脚本（`count_params.py`、`verify_csa2_modes.py`、`verify_active_params.py`、`verify_cache_size.py`、`verify_rope_yarn.py`、`verify_sparse_attn_window.py`、`verify_gate_formula.py`、`verify_sinkhorn.py`、`verify_halfstack_diff.py`、`probe_engram_layout.py`）与 `ckpt/*.out` 存档均存在。
11. 模板残留：`@content`/`TODO`/`TBD` 0 处，`【…】` 0 处；`本章问题`/`context-box`/`diagram`/`dg-*`/`drop-cap` 等类名仅出现在模板自带 CSS 中（已核对 `.dojo/templates/note/index.html` 同样包含），非残留占位符。

### 本轮重点三处的独立回源结论（均为“已修好”）

- **RoPE 旋转位置**：官方 `model.py` L768 `rd = self.rope_head_dim`，L772 `apply_rotary_emb(q[..., -rd:], freqs_cis)`；L706 窗口 KV、L758 压缩条目同为 `[..., -self.rope_head_dim:]`（尾部切片）。`apply_rotary_emb`（L392–406）对最后一维的相邻元素对做复数旋转。配置 `rope_head_dim: 64`。→ 页面“只旋转头维**末尾** 64 维”“`rope_head_dim=64`”正确；本机 `verify_rope_yarn.out` 亦记录“[6] apply_rotary_emb 只改最后 64 维: 前缀未变=True, 尾部已变=True”。
- **频率表按层还是按路径**：`model.py` L680–687 是 `Attention.__init__` 中按 `self.compress_ratio` 的层级分支（`compress_ratio>0` → `original_seq_len=65536` + `compress_rope_theta=160000` 开 YaRN；`=0` → `0` + `rope_theta=10000` 关 YaRN），L706/L758/L772 三处均使用同一个 `self.freqs_cis`。→ 页面“频率表按**层**选择，不是按路径分，查询/窗口 KV/压缩条目共用同一份表”正确。`verify_rope_yarn.out` 复现：base 160000 时 `low=15 high=25`，`65536×16=1048576`。
- **hc 读出系数来源**：`Block.forward`（L968–994）docstring 明写“attention uses what the previous layer's FFN produced and the FFN uses what this attention produced”。代码上：L982 `attn_pre/attn_post/attn_comb = self.hc_mixes(x, self.hc_attn_fn, …)`（本层 `hc_attn_fn`），L983 `hc_pre(x, pre_mix)` 用的是入参，L990 `hc_pre(x, attn_pre)` 即本层 `hc_attn_fn` 产出的 `attn_pre`，L994 `return x, ffn_pre` 交给下一层；L1260/L1267 传入的 `pre_mix` 由 `make_identity_pre_mix`（L1159–1163，首份 one-hot）初始化。`hc_attn_fn` 形状 `[24, 20480]`（`mix_hc=(2+4)×4=24`，`4×5120=20480`，headers 实测 F32）。→ 页面 attn 视图 `hc1.d`（读出系数来自上一层 FFN、层 0 为 one-hot；本层 `hc_attn_fn` 的 post/comb 用于本子层写回、其 `attn_pre` 供本层 FFN 读出）与 moe 视图 `in.d`（读入 FFN 的系数来自本层注意力产出）均与源码一致。
- **三个库加载顺序**：见机械验证第 5 项，顺序正确；并由第 6、7 项实测证明 dagre 布局未抛错、legend 与 crumb 已生成。

### 数字与形状独立复算（18 项，全部命中）

| # | 页面值 | 独立核对结果 |
|---|---|---|
| 1 | 张量头 96085、分片 48 | `headers.json` 顶层键 96085 个，分片名 `model-000NN-of-00048` |
| 2 | `dim=5120`，`attn_norm.weight [5120]` | headers `layers.0.attn_norm.weight [5120] BF16` |
| 3 | `64×512=32768`，`wq_b.weight [32768,1280]` | headers 同形 F8_E4M3 |
| 4 | `q_lora_rank=1280`，`wq_a [1280,5120]`、`q_norm [1280]` | headers 同形 |
| 5 | `o_groups=8`、`o_lora_rank=1024`，`wo_a [8192,4096]=(8×1024)×(8×512)`、`wo_b [5120,8192]` | headers 同形；报告 L1109–1110 给出 8 组与中间维 1024 |
| 6 | `wkv.weight [512,5120]`、`kv_norm [512]`、`attn_sink [64]` | headers 同形（sink F32） |
| 7 | 主 KV 条目 288 B | 512 维 FP4 = 256 B + 每 16 通道 1 B scale（32 B）= 288 B；`model.py` L759 注释“groups of 16 with E4M3 scales” |
| 8 | 索引器 K 每条 68 B | 128 维 FP4 = 64 B + 每 32 通道 E8M0（4 B）= 68 B；`model.py` L759 注释“indexer uses 32 with E8M0” |
| 9 | 每 token 缓存 890 B = 主 KV 720 + 索引器 K 170 | 主 KV：3 组 r=2 ×144 + r=1 ×288 = 432+288 = 720；索引 K：3×34+68 = 170。与 `verify_cache_size.out`“合计 890”一致 |
| 10 | 每 token 激活 8B/16B；半栈 7.8929B/7.5758B | `verify_active_params.out` 同为 7.8929B / 7.5758B，和 15.4687B；报告 L320–321 宣称 8B/16B |
| 11 | `gate.weight [384,5120]`，`bias`/`bias_vl [384]` | headers 同形（bias 各 F32） |
| 12 | 每路由专家约 35.39M | `2304×5120×3 = 35.389M`；headers `experts.0.w1.weight [2304,2560] I8`（=5120/2，fp4 打包）、`w2 [5120,1152] I8`（=2304/2），确证 I8 打包 |
| 13 | MoE 全量 545.0760B | 独立按 40×(384+1)×35.389M + gate 计得 545.0760B，与 `count_params.out` 逐位一致 |
| 14 | 主干 551.57B / Engram 196.93B | 独立按 headers 逐张量求和得 551.5662B（扣 scale 张量）；Engram = 表 196.6138B + 稠密 2×157.33M = 196.93B |
| 15 | `hc_attn_fn [24,20480]` | headers F32；`(2+4)×4=24`、`4×5120=20480` |
| 16 | Engram 表 `[384006168,256]`、`[384016682,256]`，24 行 = `(4−1)×8` | headers 同形；`probe_engram_layout.out` 记录 24 素数之和分别等于 384006168 / 384016682 |
| 17 | `main_proj.weight [5120,15360]=5120×(3×5120)` | headers 同形；`model.py` L1113 `Linear(dim*len(target_ids), dim)`，`target_layer_ids=[37,38,39]` |
| 18 | `2048×8=16384` 候选池；`Full=2/8/14/20`、`Reindex=24/28/32/36`、Reuse 30 层 | 配置 `candidate_topk_blocks=2048`、`candidate_block_size=8`；报告 L569–570 给出 16384；`verify_csa2_modes.out` 层表与本页完全一致（模式计数 SWA 2 / Full 4 / Reuse 30 / Reindex 4） |

补充对回 `ckpt/*.out` 的数值一致性：Sinkhorn float64 最大差 8.941e-08（页面“8.9e-08”）、行/列各 20 次且末步为列；路由 indices 完全一致、权重最大差 0；稀疏注意力与解析解最大差 0、`-1` 槽位与整行 `-1` 行为一致；`verify_reach_topk.out` 验证可达数公式 `⌊(i+1)/r⌋` 与 `-∞` 屏蔽；`run_mini.out` 的模型为 `n_layers=7`、`dim=64`（`mini_model.py` 的 `mini_args`），与页面“dim 64、7 层”一致；`verify_dataflow_shapes.out` 给出层 0 全部张量与 MTP 张量形状，支持页面点名的形状明细。

## 问题

- [轻微·交互] `index.html` L1133 overview 图例：`['#b7791f','全局 KV(可下钻)']` 与 `['#8a93a3','层段(可下钻)']` 声称可下钻，但这两色对应该视图内唯一的 `gt` 节点 `hmid`（L1140）与两个 `grp` 节点 `e3`（L1139）、`d21`（L1142）均无 `drill` 字段，渲染时不会获得 `node[drill]` 的黄色双边框，点击也无响应｜引文依据：页面 L850“黄色双边框节点可点击下钻”；样式 L1290 `{selector:'node[drill]',…}` 与 L1317 `drill:o.drill||undefined`；headless 渲染 DOM 中 `#legend` 实际文本为“全局 KV(可下钻)”“层段(可下钻)”，而 overview 实际带 `drill` 的节点仅 `e0/e1/e2/d20/mtp`｜修复要求：删除这两条图例文案中的“(可下钻)”（这两类节点在全部 6 个视图均无可下钻目标，不应新增目标视图）｜修复：｜复验：
- [轻微·一致性] `index.html` L1178 attn 视图边 `['kvn','cmp','同层两条并行的 KV 路径']`：把“压缩条目”画成由窗口路径的 `kv_norm` 派生，与同一节点 L1168 的 `d` 字段自相矛盾｜引文依据：`cmp.d`“压缩条目由 Compressor 的独立投影产生（自带 wkv 与 norm），不复用窗口 KV 的 wkv/kv_norm”；`headers.json` 中 `layers.2.attn.compressor.wkv.weight` 与 `layers.2.attn.wkv.weight` 为两个独立张量（均 `[512,5120]`），`model.py` L705 vs L747 两处分别调用 `self.wkv` 与 `self.compressor`｜修复要求：使图与文一致——改从独立来源连边（或在 attn 视图补一个 Compressor 投影节点），去除“压缩条目源自窗口 `kv_norm`”这一上游歧义｜修复：｜复验：
- [轻微·可读性] 缩写 `CSA2` 在 description（L6）、导语（L817）与 overview 节点 `e2.d`（L1138）首次使用处均未给出全称｜引文依据：不适用（对照来源：报告 `tech_report.txt` 小节标题“2.3. Compressed Sparse Attention 2 (CSA2)”，行 416；页面全文不含“Compressed Sparse Attention”字样）｜修复要求：在导语首次出现处补全称，例如“CSA2（Compressed Sparse Attention 2，压缩稀疏注意力）”，全页仅需一次且位于首次使用处｜修复：｜复验：
- [轻微·来源] L877“分组是为了让输出投影不用一次处理 32768 维。”把设计动机写成事实，来源未给出该动机｜引文依据：报告 `tech_report.txt` 行 1109–1110 仅陈述“The number of output projection groups is set to 8, and the dimension of each intermediate attention output is set to 1024.”，未给动机｜修复要求：删除该目的性断言，或改写为明确标注的推断并给出可定位依据｜修复：｜复验：

（说明：`_content_*.html`、`scope.md`、`review-1.md`、`review-2.md` 等前序产物按规范未读取；上述 4 条均为本轮独立读源码与页面后新发现，非引用前轮结论。）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：**可发布**

发布条件逐项核对（note 类页面按 `guides/note.md` 发布前检查执行，`guides/concept/check.md` §5 中仅与 note 相关的条目适用）：

| 条件 | 结果 |
|---|---|
| 三轮审查均由未参与写作的独立审查者执行 | 本轮为第 3 轮独立审查完成（前两轮记录按规范未读取） |
| 元数据齐备（`description` 纯文本、`dojo:summary` 可渲染、`dojo:type=note`、`dojo:topics`、`dojo:tag`） | 通过；`description` 无 `$`，`summary` 4 个 `$` 成对且含 `$H_{L/2}$`、`$2048 \times 8 = 16384$`；`topics=模型结构,内存与缓存` 在 `validate.py` 词表内 |
| 无残留占位符与参照页字样 | 通过：`【…】`、`@content`、`TODO`、`TBD`、`Qwen`、`GDN`、`QSA`、`qwen4_exp`、`参照页` 均 0 处 |
| 公式可渲染 | 通过：正文 29 对 `$...$` 全部渲染（`katex-error` 0），全部 `f`/`d` 公式字段经 KaTeX `throwOnError:true` 校验 0 失败 |
| 链接有效 | 通过：6 个概念页 + 首页 + 11 个本地资源均存在，无同页失效锚点、无重复 id |
| 交互图可用 | 通过：headless Chrome 实测 `#cy` 生成 3 个 canvas，`#legend`/`#crumb` 有内容，无 `Script error`/`graphlib` 报错；另以真实 dagre 0.8.5 复核 6 个视图布局全部成功 |
| 正文无会话指代 | 通过：`我们/你/读者/待补充` 等 0 处 |
| 每条来源论断有可定位依据 | 通过（本轮抽验 18 项数字/形状 + 报告行 314–321、565–570、379–413、536–546、1105–1110 与 `model.py` L458–485/680–687/706/758/772/907–994、`kernel.py` L310–403 全部命中，未发现来源不支持的技术论断；仅 L877 一处动机断言列入轻微问题） |
| `.dojo/scripts/validate.py` 返回成功 | 通过（退出码 0） |
| 可运行代码结果与页面描述一致 | 通过：`run_mini.out`、`verify_*.out` 与页面 §7 陈述逐项对上 |
| `overview.html` 与 `index.html` 相互链接 | 不适用：`dojo:type=note` 页面无 `overview.html`，`guides/note.md` 未要求该项 |
| 页面级「核心问题」与章节「本章问题」折叠块 | 不适用：note 类页面不设问题折叠块，`guides/note.md` 未要求该项（全文 `<details>` 数为 0） |

遗留轻微问题的接受理由：4 条均为图例文案/一处箭头语义/一处缩写与一处动机措辞，均不影响核心结论与主线理解，也不构成来源不支持的技术论断，可在发布后随手修；不阻塞发布。
