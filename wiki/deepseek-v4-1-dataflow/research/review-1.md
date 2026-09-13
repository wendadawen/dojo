<!-- review-meta
round: 1
page: wiki/deepseek-v4-1-dataflow/index.html
reviewed_content_sha256: 7efc351b29f34370
-->
# DeepSeek-V4.1-Flash 前向数据流 审查记录（第 1 轮）

- 页面版本：`defb658da0ad740c7ebb479d8bebe6a85e38127f`（`git hash-object wiki/deepseek-v4-1-dataflow/index.html`）
- 审查时间：2026-09-10 20:41
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节（按顺序）：
  1. `<head>`（`description`/`dojo:summary`/`dojo:type`/`dojo:topics`/`dojo:tag` 五项元数据与本地资源引用）
  2. `1. 关键规格`（21 行规格表逐行）
  3. `2. 交互式数据流`（含三个内联 `<script>` 块：页面 UI 脚本、`var VIEWS` + cytoscape 脚本、站内链接脚本）
  4. `3. 要点`（整体结构 / 注意力 / 可见集三节全部列表项）
  5. `4. CED：解码器的全局 KV 从哪里来`
  6. `5. 稀疏选择：从打分到槽位`
  7. `6. Engram 与 DSpark`
  8. `7. 核对方式`
  9. `来源与范围说明`（含核对说明段）
  10. `var VIEWS` 六个视图的全部节点与边（overview 10/9、attn 15/16、sparse 8/8、moe 9/9、engram 7/7、dspark 12/12）

核对所用来源：`official/inference/config.json`（与 `official/config.json` 的字段逐个比对）、`official/inference/model.py`、`official/inference/kernel.py`、`official/tech_report.txt`、`ckpt/headers.json`（96085 张量）、`research/*.py` 与 `ckpt/*.out`。

## 机械验证结果

1. `.dojo/scripts/validate.py`：
   `validation ok: wiki/deepseek-v4-1-dataflow/index.html`，退出码 0。
2. 三个内联 `<script>` 块抽取并语法检查（`/tmp/df_js_{0,1,2}.js`）：
   `node --check` → 三块均 `OK`。
3. 字符与占位符脚本核对：
   - Unicode 数学字符：`U+2212` ×2、`U+00D7` ×15、`U+2192` ×10、`U+221E` ×1（位置见问题 8）。
   - TAB 字符 0；残留占位符 `【` 0；`{{` 0；`TODO`/`lorem` 0。
   - 残留参照页字样：`Qwen3.8` 0、`Qwen` 0、`GDN` 0、`QSA` 0、`qwen4_exp` 0。
4. 无头 Chrome（`--headless=new --disable-gpu --virtual-time-budget=20000 --dump-dom`）：
   本环境下 Chrome 无法启动 —— 待审页与对照页均 `exit=133`、DOM 0 字节，stderr 为
   `sandbox initialization failed: Operation not permitted` 与
   `FATAL ... GPU process isn't usable. Goodbye.`；加 `--no-sandbox` / `--disable-gpu-sandbox` 后进程被 SIGKILL/SIGTERM 终止。两页失败方式完全相同，因此**不存在本页独有的渲染错误**可判定（所提示的 `Script error.` 属既有行为，本页未复现到任何独有错误）。
   以本地 KaTeX 静态复算替代该项：把自动渲染的等值正文（去 `<script>`/`<style>`/`<pre>`/`<code>`）中全部 `$...$`/`$$...$$` 用 `libs/katex.min.js` 以 `throwOnError:true` 渲染 —— 本页 **25 条全部成功、0 失败**（对照页 112 条、0 失败）。
5. `var VIEWS` 结构核对（Node + 本地 KaTeX）：
   - 61 个节点的 `f` 公式全部以 `throwOnError:true` 渲染成功（0 失败）；`d`/`io` 内无 `$...$` 遗漏。
   - 61 条边全部引用已存在的节点（悬空 0）；`drill` 目标 `attn/engram/sparse/dspark` 均为已定义视图；全部 14 种节点类型 `t` 都能命中 `COLOR` 映射（无回退色）。
6. 资源与链接可达性：`libs/` 下 9 个本地资源（katex、auto-render、cytoscape、cytoscape-dagre、dagre、prism ×2、主题 CSS ×2）全部存在；6 个概念链接（`deepseek-moe`、`cross-layer-kv-sharing`、`ngram`、`speculative-decoding`、`rope`、`sliding-window-attention`）对应 `index.html` 全部存在。

### 抽样回溯（独立复算 / 对回存档，共 16 项）

| # | 页面数字 | 独立复算或存档依据 | 结论 |
|---|---|---|---|
| 1 | 每 token 全局 KV 890 B | 主 KV：ratio2 条目 288 B÷2=144 B/token×3 组 + ratio1 288×1 = **720**；索引器 K：68÷2=34×3 + 68 = **170**；720+170=890（与 `verify_cache_size.out` 一致） | 一致 |
| 2 | 主 KV 条目 288 B | 512 维 FP4 = 512×0.5 = 256 B，加每 16 通道 1 B E4M3 scale = 512/16 = 32 B → 288（`model.py` L760 `fp4_act_quant(latent, 16, ...)`） | 一致 |
| 3 | 索引器 K 条目 68 B | 128×0.5=64 B + 128/32=4 B E8M0 → 68（`model.py` L546 用 `fp4_block_size=32`） | 一致 |
| 4 | `wq_b.weight` [32768,1280] | `headers.json` `layers.0.attn.wq_b.weight` = [32768,1280] F8_E4M3；64×512=32768 | 一致 |
| 5 | `wo_a.weight` [8192,4096]、`wo_b.weight` [5120,8192] | `headers.json` 同名张量逐条命中；(8×1024)×(8×512)=8192×4096 | 一致 |
| 6 | `hc_attn_fn` [24,20480] = 24×(4×5120) | `headers.json` `layers.0.hc_attn_fn` = [24,20480] F32；`kernel.py` L409 `mix_hc=(2+hc)*hc=24` | 一致 |
| 7 | `main_proj.weight` [5120,15360] | `headers.json` `mtp.0.main_proj.weight` = [5120,15360] F8_E4M3；5120×(3×5120)=15360 | 一致 |
| 8 | `attn.attn_sink` [64] | `headers.json` `layers.0.attn.attn_sink` = [64] F32 | 一致 |
| 9 | 候选池 2048 块 × 8 = 16384 | `config.json` `candidate_topk_blocks=2048`/`candidate_block_size=8`；报告 569–570 "selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions" | 一致（行号见问题 4） |
| 10 | Engram 每 token 24 行、6144 元素 | `model.py` L344 `n_hash_cols=(4-1)×8=24`；24×256=6144（`verify_active_params.out` 首行） | 一致 |
| 11 | 单专家约 35.39M | 3×2304×5120 = 35,389,440 | 一致 |
| 12 | Engram 稠密投影每层 157.33M | 24×256×5120×5 = 157,286,400，加 `q_weight`/`k_weight` 2×(4×5120)=40,960 → 157,327,360（`verify_halfstack_diff.out` 157.33M） | 一致 |
| 13 | 窗口 KV 64 KiB/层 | 128×512×1 B = 65,536 B | 一致 |
| 14 | prefill 7.8929B / decode 15.4687B | `verify_active_params.out`：编码器半栈 7.8929B、解码器半栈 7.5758B、decode 15.4687B（+embed+head 16.7925B） | 一致 |
| 15 | 上下文 1M | `original_seq_len` 65536 × `rope_factor` 16 = 1,048,576（`verify_rope_yarn.out` [3]） | 一致 |
| 16 | 压缩比分布与 source/模式层号 | `config.json` `compress_ratios` 索引 2–19 全 2、20–39 全 1、0–1 为 0（尾部 3 个 0 对应 MTP）；`kv_source_layers=[2,8,14,20]`、`index_source_layers=[2,8,14,20,24,28,32,36]`；`verify_csa2_modes.out` 层表 Full=[2,8,14,20]、Reindex=[24,28,32,36]、Reuse 30 层 | 一致 |

结论：核心规格与全部 checkpoint 形状、层号、缓存账均可回溯，未发现数字错误。

## 问题

- [重要·技术] `3. 要点`「注意力」位置条目 + `1. 关键规格`「位置编码」行 + 视图 `attn` 的 `rope` 节点：把频率表说成"窗口路径 base=10000、压缩条目 base=160000"两套并存。｜引文依据：`model.py` L680–687 是按**层**选择 —— `if self.compress_ratio: original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta`（65536 / 160000，YaRN 开），`else: original_seq_len, rope_theta = 0, args.rope_theta`（10000，纯滑动窗口层）；L706 窗口 KV、L758 压缩条目、L772 查询三处都调用同一个 `self.freqs_cis`。`verify_rope_yarn.out` 亦按"全局 KV 层 (ratio>0)：base=160000 YaRN=开 / SWA 层 (ratio=0)：base=10000 YaRN=关"分组，而非"窗口路径/压缩条目"。｜修复要求：把该句与 `rope` 节点改为"层 0、1（`compress_ratio=0`）用 base 10000 且关 YaRN；层 2–39（`compress_ratio>0`）的查询、窗口 KV 与压缩条目共用同一 base 160000 + YaRN 频率表"；`1. 关键规格`「位置编码」行同步改为按层陈述。｜修复：频率表改为按<b>层</b>陈述：关键规格「位置编码」行、要点「注意力·位置」条目、`attn` 视图 `rope` 节点三处均改为「层 0、1（compress_ratio=0）用 base 10000 且关 YaRN；层 2–39（compress_ratio>0）用 base 160000 且开 YaRN，其查询、窗口 KV 与压缩条目共用同一份频率表」，并引 `model.py` L680–687 分支与 L706/L758/L772 三处调用｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [重要·技术] 视图 `attn` 的 `hc1` 节点说明：称注意力读出的系数"由 `hc_attn_fn`（形状 [24, 20480]）从展平后的残差流算出"。｜引文依据：`model.py` L982–994 `Block.forward`：注意力读出用的是调用方传入的 `pre_mix`（`x = self.hc_pre(x, pre_mix)`），而 `hc_attn_fn` 产出的 `attn_pre` 被用于**本层 FFN** 的读出（L990 `x = self.hc_pre(x, attn_pre)`），`attn_post`/`attn_comb` 用于注意力写回（L986）；L910–914 docstring 明写 "attention uses what the previous layer's FFN produced and the FFN uses what this attention produced"，返回值是 `ffn_pre`（L994）。｜修复要求：把该节点说明改为"读出系数由上一层 FFN 的超连接混合提供（层 0 为 one-hot），`hc_attn_fn` 产出的 post/comb 系数用于本子层写回"，并保留 `[24, 20480]` 形状（形状本身与 `headers.json` 一致）。｜修复：`hc1` 节点说明改为「读出系数由上一层 FFN 的超连接混合提供（层 0 为 one-hot）；本层 hc_attn_fn 产出的 post/comb 系数用于本子层写回，其 attn_pre 供本层 FFN 读出」，形状 [24, 20480] 保留（依 `model.py` L982–994）｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [轻微·技术] 视图 `attn` 的 `hc1` 公式 `\tilde{x} = \tfrac{1}{4}\textstyle\sum_i \sigma(\cdot)_i \odot x_i` 多出 $\tfrac{1}{4}$ 平均因子。｜引文依据：`kernel.py` L427 `pre[i,j] = T.sigmoid(mixes_shared[j]*hc_scale[0]+hc_base[j]) + eps`；`model.py` L959 `y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)` —— 无 1/4 归一。｜修复要求：删去 `\tfrac{1}{4}`，系数写作 $\mathrm{sigmoid}(\text{mixes}\cdot\text{scale}+\text{base})+\epsilon$。｜修复：删去 $\tfrac{1}{4}$，公式改为 $\tilde{x} = \sum_i \mathrm{sigmoid}(\mathrm{mixes}\cdot\mathrm{scale}+\mathrm{base})_i \odot x_i$（依 `kernel.py` L427 与 `model.py` L959）｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [轻微·技术] `1. 关键规格`「候选池」行依据列写"报告行 565–567"。｜引文依据：报告 565–567 只讲 Full 层做块级选择、"每个块取块内最大索引分"；所引数字在 569–570：`pool larger than the final Top-K set. For example, selecting 2,048 blocks with 8 positions each / yields 16,384 candidate positions.`｜修复要求：改为"报告行 565–570"。｜修复：「候选池」行依据改为「报告行 565–570」｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [轻微·技术] `来源与范围说明` 表"数据流路径…"行括注"（页面标注的 L 行号即该文件行号）"。｜引文依据：全文仅出现 `报告行 314–317`、`报告行 320`、`报告行 565–567` 三处行号引用（`grep -o "L[0-9]{2,4}"` 无命中），没有任何 `model.py`/`kernel.py` 的 L 行号。｜修复要求：删除该括注，或为模型机制描述补上对应的 `model.py`/`kernel.py` 行号。｜修复：来源表该行括注改为具体机制位置：「压缩器池化见 model.py L458–485，频率表按层分支见 L680–687，超连接读出与写回见 L982–994，稀疏注意力内核见 kernel.py L310–403」｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [轻微·技术] 视图 `attn` 的边 `['kvn','cmp','压缩条目']`：暗示压缩条目由该层 `wkv`/`kv_norm` 产生。｜引文依据：`model.py` L639–644 `wkv`/`kv_norm` 服务于窗口 KV（L705 `kv = self.kv_norm(self.wkv(x))`）；压缩条目来自 `Compressor`，后者自带 `wkv`（L446）与 `norm`（L443），调用见 L747 `latent = self.compressor(x, start_pos)`。｜修复要求：把该边改为从压缩路径（Compressor / 上游条目库）接入，或在 `cmp` 节点说明中注明"压缩条目由 Compressor 的独立投影产生，不复用窗口 KV 的 `wkv`/`kv_norm`"。｜修复：边标签改为「同层两条并行的 KV 路径」，并在 `cmp` 节点说明补「压缩条目由 Compressor 的独立投影产生（自带 wkv 与 norm），不复用窗口 KV 的 wkv/kv_norm」｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [轻微·格式] 视图 `attn` 的 `legend` 含两个相同颜色 `#e0922f` 的条目（'位置与门控' 与 '压缩条目'），`var COLOR` 中 `gt` 与 `cmp` 同为 `#e0922f`；图例出现两个无法区分的同色块，而节点 `rope`/`sink`(gt) 与 `cmp` 恰恰是不同语义。｜引文依据：不适用。｜修复要求：给 `gt` 与 `cmp` 分配不同颜色（或在同一视图内合并为一个图例项），使图例色块与语义一一对应。｜修复：`COLOR` 的 `gt` 改为 #b7791f，`cmp` 保持 #e0922f；attn / moe / engram / sparse / overview / dspark 六个视图的 legend 同步改色，图例色块与语义一一对应｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [轻微·格式] Unicode 数学字符直写：视图 `attn` 的 `sa` 说明中 `−1`、视图 `sparse` 的 `pool0` 说明中 `−∞` 用了 `U+2212`；`1. 关键规格` 表 L837「2048 个块 × 8 位置」、L841「阶数 2/3/4 × 8 头」与 `3. 要点` L873/L875「64 头 × 512 维」直写 `U+00D7`。｜引文依据：`guides/concept/style-guide.md` §11"禁止直接使用 Unicode 数学字符替代"；工具提示由 `renderD` 解析 `$...$`，改为 `$-1$`、`$-\infty$` 即可渲染（本地 KaTeX 复算通过）。｜修复要求：把上述 4 处正文/表格与 2 处说明文字改用 `$...$`（`$\times$`、`$-1$`、`$-\infty$`）；`var VIEWS` 的节点 `label` 由 cytoscape 画布绘制、KaTeX 不介入，其 `×`/`→` 与参照页同用法（参照页 `U+00D7` 69 处、`U+2192` 42 处），保留并在此记录接受理由。｜修复：正文与表格的 U+00D7 全部改为 LaTeX（$64\times512$、$2048\times8$、$64\times512=32768$、65536 乘 rope_factor 等），`d` 说明里的 U+2212/U+221E 改为 $-1$/$-\infty$；`var VIEWS` 的 `label`/`io` 字段保留 `×`/`→`——它们由 cytoscape 画布绘制、KaTeX 不介入，与参照页同用法，此处记录为接受理由。改后页面 U+2212 计数 0｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [轻微·格式] L563 样式表：`h2.collapsed .collapse-btn { transform: rotate(-90deg); }` 与紧随其后的注释 `/* ============ 阅读时间估计 ============ */` 及 `.reading-time` 规则挤在同一行，缺换行。｜引文依据：不适用。｜修复要求：在 `}` 后补换行，使 `.reading-time` 独立成行。｜修复：样式表补换行，`.reading-time` 独立成行｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [轻微·可读性] `1. 关键规格`「每 token 激活」行依据列只列两个半栈读数（7.8929B、7.5758B），未说明 decode 为两半之和。｜引文依据：`verify_active_params.out`："prefill (只跑编码器): 7.8929B (+embed 8.5548B)"、"decode (编码器+解码器): 15.4687B (+embed+head 16.7925B)"、"报告宣称: prefill 8B / decode 16B"。｜修复要求：在该行依据列补一句"decode = 两半之和 15.4687B，8B/16B 为报告宣称的取整值"。｜修复：「每 token 激活」行补「decode 为两半之和 15.4687B，8B 与 16B 是报告宣称的取整值」（依 `verify_active_params.out`）｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜
- [轻微·可读性] 导语"解码器的全局 KV … 由编码器末层隐状态投影而来——这是 prefill 只需跑半栈的原因"把 prefill 减半归因于单一条件。｜引文依据：报告 412–414 的总复杂度式 `O(NL) → O(NL/2 + n_win×L/2) ≈ O(NL/2)`，其中 `n_win×L/2` 一项对应 SWA 有界重放（1011–1018 "Decoder SWA Bounded Replay bounds the decoder forward pass to n_win tokens"）；两个条件共同成立才使解码器前向止于编码器输出。｜修复要求：在导语该句加限定（如"配合解码器 SWA 有界重放，prefill 可止于半栈"），§4 已有的正确区分不变。｜修复：导语改为「配合解码器滑动窗口的有界重放，prefill 阶段因此可以止于半栈」（依报告 412–414 的总复杂度式与 1011–1018 的有界重放说明）｜复验：已重新生成页面并跑 validate.py（validation ok）；TAB 0、占位符 0、U+2212 0、Node --check 三块脚本全 OK、引用与链接未受影响｜

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 9
- 处置：修复

核心结论（CED 半栈 prefill、CSA2 跨层共享与 FP4 主 KV、890 B/token 缓存账、两级稀疏选择的 2048×8 / Top-512、Engram 与 DSpark 的接入位置与形状）与报告、官方配置、参考实现、真实张量头及本机实测一致，未发现需要返回规划的问题。两处"重要"均为机制归因错误（频率表按层而非按路径、超连接读出系数来源），不改变核心结论，但会造成对机制的明显误解，须在下一轮前修复并复验；两处"重要"关闭前不予发布。
